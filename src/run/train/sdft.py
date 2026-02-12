"""
Self-Distillation Fine-Tuning (SDFT) — the paper's core method.

Algorithm 1 from the paper:
    1. For each (query, demonstration) pair:
       a. Student rollout: sample y ~ π_θ(·|x)
       b. Compute teacher + student log-probs on the sampled tokens
       c. Compute KL gradient estimate
    2. Update student: θ ← θ - η·g
    3. Update teacher EMA: ϕ ← α·θ + (1-α)·ϕ
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
from src.run.utils import set_seeds, save_checkpoint
from src.run.logger import get_tqdm_kwargs
from src.run.distributed import is_main_process, barrier, get_raw_model


def do_sdft(
    model: torch.nn.Module,
    config: RunConfig,
    state: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Train with Self-Distillation Fine-Tuning.

    For each training example:
    1. Generate on-policy rollout from the student (conditioned on query only)
    2. Compute teacher logits (conditioned on query + demonstration via EMA model)
    3. Compute student logits on the same rollout
    4. Minimize reverse KL(student || teacher) using the configured gradient estimator
    5. Update EMA teacher weights

    Args:
        model: HFModel wrapping the causal LM (student)
        config: Run configuration
        state: Optional state for resumption

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

    if state is None:
        state = {}

    logger.info(f"---- Begin SDFT Training (KL estimator: {kl_estimator}) ----")
    set_seeds(config.seed)

    train_loader = loaders["train"]
    num_batches = len(train_loader)
    total_steps = num_batches * epochs

    # Initialize EMA teacher from student weights
    raw_model = get_raw_model(model)
    ema_teacher = EMAModel(raw_model, decay=ema_decay)
    if "ema_state" in state:
        ema_teacher.load_state_dict(state["ema_state"])
    logger.info(f"Initialized EMA teacher (decay={ema_decay})")

    # Enable gradient checkpointing to reduce activation memory
    raw_model.enable_gradient_checkpointing()
    logger.info("Enabled gradient checkpointing")

    model.train()

    # Optimizer
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    if "optimizer" in state:
        opt.load_state_dict(state["optimizer"])

    # Learning rate schedule (WSD)
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

    losses = []
    pbar = tqdm(total=total_steps, **get_tqdm_kwargs(logger, ncols=140))
    num_steps = 0
    resume_step = state.get("step", -1)

    for epoch_idx in range(epochs):
        for batch in train_loader:
            num_steps += 1
            pbar.update()
            if num_steps <= resume_step:
                continue

            loss = _sdft_step(
                model=model,
                ema_teacher=ema_teacher,
                batch=batch,
                config=config,
                kl_estimator=kl_estimator,
                max_gen_len=max_gen_len,
            )

            scaled_loss = loss / acc_steps
            scaled_loss.backward()

            loss_val = loss.item()
            losses.append(loss_val)

            pbar.set_description(
                f"SDFT | LR: {cur_lr:.2e} | KL: {loss_val:.4f}"
            )

            if num_steps % 200 == 0 or num_steps == total_steps:
                avg_loss = np.mean(losses[-200:])
                logger.info(
                    f"Step {num_steps}/{total_steps} | LR: {cur_lr:.2e} | KL: {avg_loss:.4f}"
                )

            if num_steps % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)

                # Update EMA teacher after each optimizer step
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
                    prefix="sdft",
                    extra_state={
                        "scheduler_epoch": scheduler.last_epoch if scheduler else None,
                        "ema_state": ema_teacher.state_dict(),
                    },
                    logger=logger,
                )
                barrier()

    pbar.close()

    state = {
        "optimizer": opt.state_dict(),
        "step": num_steps,
        "total_steps": total_steps,
        "scheduler_epoch": scheduler.last_epoch if scheduler else None,
        "ema_state": ema_teacher.state_dict(),
    }

    return model, state


def _sdft_step(
    model: torch.nn.Module,
    ema_teacher: EMAModel,
    batch: dict,
    config: RunConfig,
    kl_estimator: str,
    max_gen_len: int,
) -> torch.Tensor:
    """
    Execute a single SDFT training step.

    1. Generate on-policy rollout from student
    2. Compute student & teacher logits on the rollout
    3. Compute KL loss
    """
    device = config.device
    raw_model = get_raw_model(model)

    # --- Step 1: Student rollout (on-policy generation) ---
    student_input_ids = batch["student_input_ids"].to(device)
    student_attention_mask = batch["student_attention_mask"].to(device)

    with torch.no_grad():
        # Generate from student (no grad needed for generation)
        generated = raw_model.generate(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            max_new_tokens=max_gen_len,
            temperature=1.0,
            do_sample=True,
        )

    # Free generation KV cache before heavy forward passes
    torch.cuda.empty_cache()

    # Build the full sequence for log-prob computation
    # generated includes prompt + new tokens
    gen_len = generated.shape[1]
    gen_attention_mask = torch.ones_like(generated)

    # --- Step 2: Compute student logits on generated sequence ---
    student_outputs = raw_model.model(
        input_ids=generated,
        attention_mask=gen_attention_mask,
    )
    student_logits = student_outputs.logits  # (B, T, V)

    # --- Step 3: Compute teacher logits on generated sequence ---
    # The teacher sees the demonstration in its context
    teacher_input_ids = batch["teacher_input_ids"].to(device)
    teacher_attention_mask = batch["teacher_attention_mask"].to(device)

    # We need to feed the generated tokens through the teacher, but with
    # the teacher's context (demonstration). We concatenate teacher context
    # with the generated response tokens.
    prompt_len = student_input_ids.shape[1]
    gen_response = generated[:, prompt_len:]  # just the generated part

    # Concatenate teacher prompt + generated response
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
    del teacher_outputs  # free intermediate memory
    torch.cuda.empty_cache()

    # Align logits to the generated response portion
    teacher_prompt_len = teacher_input_ids.shape[1]

    # Student logits over the response
    student_resp_logits = student_logits[:, prompt_len - 1 : -1, :]
    # Teacher logits over the response (offset by teacher prompt length)
    teacher_resp_logits = teacher_logits[:, teacher_prompt_len - 1 : -1, :]

    # Trim to same length
    min_len = min(student_resp_logits.shape[1], teacher_resp_logits.shape[1])
    student_resp_logits = student_resp_logits[:, :min_len, :]
    teacher_resp_logits = teacher_resp_logits[:, :min_len, :]

    # --- Step 4: Compute KL loss ---
    if kl_estimator == "analytic":
        loss = analytic_kl_gradient_loss(student_resp_logits, teacher_resp_logits)

    elif kl_estimator == "token":
        # Get per-token log-probs for the sampled tokens
        response_tokens = gen_response[:, :min_len]
        student_log_probs = F.log_softmax(student_resp_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_resp_logits, dim=-1)
        student_sampled = student_log_probs.gather(
            dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        teacher_sampled = teacher_log_probs.gather(
            dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        loss = token_level_kl_loss(student_sampled, teacher_sampled)

    elif kl_estimator == "rao_blackwell":
        response_tokens = gen_response[:, :min_len]
        student_log_probs = F.log_softmax(student_resp_logits, dim=-1)
        student_sampled = student_log_probs.gather(
            dim=-1, index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)
        loss = rao_blackwell_kl_loss(
            student_resp_logits, teacher_resp_logits, student_sampled
        )

    else:
        raise ValueError(f"Unknown KL estimator: {kl_estimator}")

    return loss

