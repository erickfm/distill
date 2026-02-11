"""
Efficient Self-Distillation Fine-Tuning (ESDFT).

Extends SDFT with adaptive screening to skip rollout generation for
examples where student and teacher already agree, cutting wall-clock
cost while preserving learning signal.

Three-phase per-step algorithm:

    1.  Cheap Disagreement Screening — two forward passes (no generation)
        estimate per-example KL(student || teacher) on the teacher's
        demonstration tokens.

    2.  Selective Rollout Generation — only examples with screening KL
        above `screening_threshold` get full SDFT treatment (generate,
        score, backprop).  The rest are skipped.

    3.  Self-Calibrating Audit — a random fraction p of "easy" examples
        still get full rollouts as calibration audits.  The discrepancy
        rate (how often an "easy" example turns out to have high
        on-policy KL) drives an adaptive update to p:

            p ← clip(p + α·(discrepancy − target), p_min, p_max)

        Early in training, p stays large (screening unreliable);
        late in training, p shrinks (screening trustworthy).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional

from src.model.ema import EMAModel
from src.model.kl_gradient import (
    analytic_kl_gradient_loss,
    token_level_kl_loss,
    rao_blackwell_kl_loss,
)
from src.run.config import RunConfig
from src.run.utils import set_seeds, save_checkpoint, log_line
from src.run.logger import get_tqdm_kwargs
from src.run.distributed import is_main_process, barrier, get_raw_model


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _compute_disagreement(
    model: torch.nn.Module,
    ema_teacher: EMAModel,
    batch: dict,
    device: str,
) -> torch.Tensor:
    """
    Cheap disagreement screening (two forward passes, no generation).

    Both student and teacher are evaluated on the teacher's input tokens
    (which contain the demonstration context).  Per-example KL divergence
    measures how much the student's next-token distribution disagrees
    with the teacher's.

    Returns:
        disagreement: (B,) per-example mean KL divergence.
    """
    input_ids = batch["teacher_input_ids"].to(device)
    attention_mask = batch["teacher_attention_mask"].to(device)

    raw_model = get_raw_model(model)

    with torch.no_grad():
        # Student forward pass
        s_out = raw_model.model(input_ids=input_ids, attention_mask=attention_mask)
        s_log_probs = F.log_softmax(s_out.logits[:, :-1, :], dim=-1)

        # Teacher forward pass
        t_out = ema_teacher.model.model(input_ids=input_ids, attention_mask=attention_mask)
        t_log_probs = F.log_softmax(t_out.logits[:, :-1, :], dim=-1)

        # Per-token KL(student || teacher) = Σ_v p_s(v) [log p_s(v) - log p_t(v)]
        s_probs = s_log_probs.exp()
        kl_per_token = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)  # (B, T-1)

        # Mask padding
        mask = attention_mask[:, 1:].float()
        kl_per_token = kl_per_token * mask

        # Per-example mean KL
        disagreement = kl_per_token.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

    return disagreement  # (B,)


def _extract_subbatch(batch: dict, mask: torch.Tensor) -> dict:
    """Extract examples selected by a boolean mask into a new batch dict."""
    # Use CPU indices for indexing — batch tensors may still be on CPU
    indices_cpu = mask.cpu().nonzero(as_tuple=True)[0]
    sub = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            sub[key] = val[indices_cpu]
        elif isinstance(val, list):
            idx_list = indices_cpu.tolist()
            sub[key] = [val[i] for i in idx_list]
        else:
            sub[key] = val
    return sub


def _esdft_generation_step(
    model: torch.nn.Module,
    ema_teacher: EMAModel,
    batch: dict,
    config: RunConfig,
    kl_estimator: str,
    max_gen_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full SDFT step on a (possibly sub-) batch.

    Identical to the standard SDFT step but additionally returns
    per-example KL values (detached) for discrepancy tracking.

    Returns:
        loss:           scalar loss for backprop
        per_example_kl: (B,) detached per-example KL on the rollout
    """
    device = config.device
    raw_model = get_raw_model(model)

    # --- Step 1: Student on-policy rollout ---
    student_input_ids = batch["student_input_ids"].to(device)
    student_attention_mask = batch["student_attention_mask"].to(device)

    with torch.no_grad():
        generated = raw_model.generate(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            max_new_tokens=max_gen_len,
            temperature=1.0,
            do_sample=True,
        )

    gen_attention_mask = torch.ones_like(generated)

    # --- Step 2: Student logits on the rollout ---
    student_outputs = raw_model.model(
        input_ids=generated,
        attention_mask=gen_attention_mask,
    )
    student_logits = student_outputs.logits

    # --- Step 3: Teacher logits on the rollout ---
    teacher_input_ids = batch["teacher_input_ids"].to(device)
    teacher_attention_mask = batch["teacher_attention_mask"].to(device)

    prompt_len = student_input_ids.shape[1]
    gen_response = generated[:, prompt_len:]

    teacher_full = torch.cat([teacher_input_ids, gen_response], dim=1)
    teacher_full_mask = torch.cat(
        [teacher_attention_mask, torch.ones_like(gen_response)], dim=1
    )

    with torch.no_grad():
        teacher_outputs = ema_teacher.model.model(
            input_ids=teacher_full,
            attention_mask=teacher_full_mask,
        )
    teacher_logits = teacher_outputs.logits

    # --- Align logits to response portion ---
    teacher_prompt_len = teacher_input_ids.shape[1]
    student_resp_logits = student_logits[:, prompt_len - 1 : -1, :]
    teacher_resp_logits = teacher_logits[:, teacher_prompt_len - 1 : -1, :]

    min_len = min(student_resp_logits.shape[1], teacher_resp_logits.shape[1])
    student_resp_logits = student_resp_logits[:, :min_len, :]
    teacher_resp_logits = teacher_resp_logits[:, :min_len, :]

    # --- Per-example KL (detached, for discrepancy tracking) ---
    with torch.no_grad():
        s_lp = F.log_softmax(student_resp_logits, dim=-1)
        t_lp = F.log_softmax(teacher_resp_logits, dim=-1)
        s_p = s_lp.exp()
        kl_tok = (s_p * (s_lp - t_lp)).sum(dim=-1)  # (B, T)
        per_example_kl = kl_tok.mean(dim=-1)          # (B,)

    # --- Loss for backprop (selected KL estimator) ---
    if kl_estimator == "analytic":
        loss = analytic_kl_gradient_loss(student_resp_logits, teacher_resp_logits)

    elif kl_estimator == "token":
        response_tokens = gen_response[:, :min_len]
        s_log_probs = F.log_softmax(student_resp_logits, dim=-1)
        t_log_probs = F.log_softmax(teacher_resp_logits, dim=-1)
        s_sampled = s_log_probs.gather(
            dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        t_sampled = t_log_probs.gather(
            dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        loss = token_level_kl_loss(s_sampled, t_sampled)

    elif kl_estimator == "rao_blackwell":
        response_tokens = gen_response[:, :min_len]
        s_log_probs = F.log_softmax(student_resp_logits, dim=-1)
        s_sampled = s_log_probs.gather(
            dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        loss = rao_blackwell_kl_loss(
            student_resp_logits, teacher_resp_logits, s_sampled
        )

    else:
        raise ValueError(f"Unknown KL estimator: {kl_estimator}")

    return loss, per_example_kl


# --------------------------------------------------------------------------- #
# Main trainer                                                                 #
# --------------------------------------------------------------------------- #

def do_esdft(
    model: torch.nn.Module,
    config: RunConfig,
    state: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Train with Efficient Self-Distillation Fine-Tuning.

    Extends SDFT with adaptive screening that skips expensive rollout
    generation for examples where the student already agrees with the
    teacher.  Includes a self-calibrating audit mechanism to verify
    screening reliability.

    Args:
        model:  HFModel wrapping the causal LM (student)
        config: Run configuration (includes ESDFT hyperparams)
        state:  Optional state dict for checkpoint resumption

    Returns:
        (trained_model, state_dict)
    """
    logger = config.logger
    loaders = config.loaders
    acc_steps = config.accumulation_steps
    lr = config.lr
    epochs = config.epochs
    lr_schedule = config.lr_schedule
    kl_estimator = config.kl_estimator
    ema_decay = config.ema_decay
    max_gen_len = config.max_gen_len

    # ESDFT-specific
    screening_threshold = config.esdft_screening_threshold

    if state is None:
        state = {}

    logger.info(
        f"---- Begin ESDFT Training "
        f"(KL: {kl_estimator}, threshold: {screening_threshold}) ----"
    )
    set_seeds(config.seed)

    train_loader = loaders["train"]
    num_batches = len(train_loader)
    total_steps = num_batches * epochs

    # --- EMA teacher ---
    raw_model = get_raw_model(model)
    ema_teacher = EMAModel(raw_model, decay=ema_decay)
    if "ema_state" in state:
        ema_teacher.load_state_dict(state["ema_state"])
    logger.info(f"Initialized EMA teacher (decay={ema_decay})")

    model.train()

    # --- Optimizer ---
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    if "optimizer" in state:
        opt.load_state_dict(state["optimizer"])

    # --- LR schedule (WSD) ---
    cur_lr = lr
    scheduler = None
    if lr_schedule:
        total_opt_steps = total_steps // acc_steps
        warmup_steps = round(0.10 * total_opt_steps)
        stable_end = round(0.90 * total_opt_steps)
        max_lr, min_lr = lr, lr * 0.1

        def lr_lambda(step):
            if step < warmup_steps:
                return 1e-8 / max_lr + (1.0 - 1e-8 / max_lr) * (step / warmup_steps)
            elif step < stable_end:
                return 1.0
            else:
                min_factor = min_lr / max_lr
                progress = (step - stable_end) / max(1, total_opt_steps - stable_end)
                return 1.0 - (1.0 - min_factor) * progress

        scheduler = LambdaLR(opt, lr_lambda)
        if "scheduler_epoch" in state:
            scheduler.last_epoch = state["scheduler_epoch"]

    # --- Audit state (self-calibrating) ---
    audit_state = {
        "fraction": state.get("audit_fraction", config.esdft_audit_init),
        "alpha": config.esdft_audit_alpha,
        "target": config.esdft_discrepancy_target,
        "min": config.esdft_audit_min,
        "max": config.esdft_audit_max,
    }

    # --- Running stats ---
    losses: list[float] = []
    screen_stats = state.get("screen_stats", {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "audited": 0,
        "discrepant": 0,
    })

    pbar = tqdm(total=total_steps, **get_tqdm_kwargs(logger, ncols=160))
    num_steps = 0
    resume_step = state.get("step", -1)

    for epoch_idx in range(epochs):
        for batch in train_loader:
            num_steps += 1
            pbar.update()
            if num_steps <= resume_step:
                continue

            batch_size = batch["student_input_ids"].shape[0]

            # ============================================================
            # Phase 1 — Cheap Disagreement Screening
            # ============================================================
            disagreement = _compute_disagreement(
                model, ema_teacher, batch, config.device
            )  # (B,)

            high_mask = disagreement > screening_threshold
            low_mask = ~high_mask
            n_high = high_mask.sum().item()
            n_low = low_mask.sum().item()

            # ============================================================
            # Phase 2 — Audit Selection
            # ============================================================
            audit_mask = torch.zeros(batch_size, dtype=torch.bool, device=disagreement.device)
            if n_low > 0:
                low_indices = low_mask.nonzero(as_tuple=True)[0]
                n_audit = max(0, int(round(audit_state["fraction"] * n_low)))
                if n_audit > 0:
                    perm = torch.randperm(len(low_indices), device=disagreement.device)[:n_audit]
                    audit_mask[low_indices[perm]] = True

            n_audit_actual = audit_mask.sum().item()

            # Union: high-disagreement examples + audited easy examples
            train_mask = high_mask | audit_mask
            n_train = train_mask.sum().item()

            # Bookkeeping
            screen_stats["total"] += batch_size
            screen_stats["generated"] += n_train
            screen_stats["skipped"] += (batch_size - n_train)
            screen_stats["audited"] += n_audit_actual

            if n_train == 0:
                # Entire batch screened out — no gradient
                pbar.set_description(
                    f"ESDFT | LR: {cur_lr:.2e} | skip: 100% "
                    f"| audit_p: {audit_state['fraction']:.2f}"
                )
                continue

            # ============================================================
            # Phase 3 — Full SDFT on selected examples
            # ============================================================
            sub_batch = _extract_subbatch(batch, train_mask)

            loss, per_example_kl = _esdft_generation_step(
                model=model,
                ema_teacher=ema_teacher,
                batch=sub_batch,
                config=config,
                kl_estimator=kl_estimator,
                max_gen_len=max_gen_len,
            )

            # ============================================================
            # Phase 4 — Discrepancy check on audit samples
            # ============================================================
            if n_audit_actual > 0:
                train_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
                audit_in_sub = [
                    i for i, idx in enumerate(train_indices)
                    if audit_mask[idx].item()
                ]
                if audit_in_sub:
                    audit_kl = per_example_kl[audit_in_sub]
                    discrepant_count = (audit_kl > screening_threshold).sum().item()
                    discrepancy_rate = discrepant_count / len(audit_in_sub)
                    screen_stats["discrepant"] += discrepant_count

                    # Adaptive update: p ← clip(p + α·(d − d_target), p_min, p_max)
                    audit_state["fraction"] = max(
                        audit_state["min"],
                        min(
                            audit_state["max"],
                            audit_state["fraction"]
                            + audit_state["alpha"]
                            * (discrepancy_rate - audit_state["target"]),
                        ),
                    )

            # ============================================================
            # Backward & step
            # ============================================================
            scaled_loss = loss / acc_steps
            scaled_loss.backward()

            loss_val = loss.item()
            losses.append(loss_val)

            skip_frac = 1.0 - n_train / batch_size
            pbar.set_description(
                f"ESDFT | LR: {cur_lr:.2e} | KL: {loss_val:.4f} "
                f"| skip: {skip_frac:.0%} | audit_p: {audit_state['fraction']:.2f}"
            )

            # Periodic logging
            if num_steps % 200 == 0 or num_steps == total_steps:
                avg_loss = np.mean(losses[-200:]) if losses else 0.0
                gen_rate = screen_stats["generated"] / max(1, screen_stats["total"])
                disc_rate = (
                    screen_stats["discrepant"] / max(1, screen_stats["audited"])
                )
                logger.info(
                    f"Step {num_steps}/{total_steps} | "
                    f"LR: {cur_lr:.2e} | KL: {avg_loss:.4f} | "
                    f"Gen: {gen_rate:.1%} | Skip: {1 - gen_rate:.1%} | "
                    f"Audit p: {audit_state['fraction']:.3f} | "
                    f"Discrepancy: {disc_rate:.3f}"
                )

            # Optimizer step (with gradient accumulation)
            if num_steps % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

                ema_teacher.update(get_raw_model(model))

                if scheduler:
                    scheduler.step()
                    cur_lr = scheduler.get_last_lr()[0]

            # Checkpoint
            if num_steps % 5000 == 0 or num_steps == total_steps:
                save_checkpoint(
                    model=model,
                    optimizer=opt,
                    step=num_steps,
                    total_steps=total_steps,
                    res_dir=config.res_dir,
                    prefix="esdft",
                    extra_state={
                        "scheduler_epoch": (
                            scheduler.last_epoch if scheduler else None
                        ),
                        "ema_state": ema_teacher.state_dict(),
                        "audit_fraction": audit_state["fraction"],
                        "screen_stats": screen_stats,
                    },
                    logger=logger,
                )
                barrier()

    pbar.close()

    # --- Final screening report ---
    if screen_stats["total"] > 0:
        gen_rate = screen_stats["generated"] / screen_stats["total"]
        disc_rate = screen_stats["discrepant"] / max(1, screen_stats["audited"])
        logger.info("ESDFT Final Screening Report:")
        logger.info(f"  Total examples seen : {screen_stats['total']}")
        logger.info(f"  Rollouts generated  : {screen_stats['generated']} ({gen_rate:.1%})")
        logger.info(f"  Rollouts skipped    : {screen_stats['skipped']} ({1 - gen_rate:.1%})")
        logger.info(f"  Audit samples       : {screen_stats['audited']}")
        logger.info(f"  Discrepant audits   : {screen_stats['discrepant']} ({disc_rate:.3f})")
        logger.info(f"  Final audit fraction: {audit_state['fraction']:.3f}")

        # Write screening summary to results
        log_line(
            {
                "type": "esdft_screening_summary",
                "stats": screen_stats,
                "final_audit_fraction": audit_state["fraction"],
            },
            config.res_dir / "stats.jsonl",
        )

    state = {
        "optimizer": opt.state_dict(),
        "step": num_steps,
        "total_steps": total_steps,
        "scheduler_epoch": scheduler.last_epoch if scheduler else None,
        "ema_state": ema_teacher.state_dict(),
        "audit_fraction": audit_state["fraction"],
        "screen_stats": screen_stats,
    }

    return model, state

