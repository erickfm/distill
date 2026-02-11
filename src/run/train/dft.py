"""
DFT (Distributional Fine-Tuning) baseline.

Uses importance sampling to treat the offline dataset as on-policy samples.
(Wu et al., 2025b — "On the generalization of SFT: A RL perspective with reward rectification")
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LambdaLR

from src.run.config import RunConfig
from src.run.utils import set_seeds, save_checkpoint
from src.run.logger import get_tqdm_kwargs
from src.run.distributed import is_main_process, barrier, get_raw_model


def do_dft(
    model: torch.nn.Module,
    config: RunConfig,
    state: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Train with DFT — importance-weighted policy gradient from offline data.

    DFT uses importance sampling to correct for the distribution mismatch
    between the expert (offline) policy and the current student policy.

    The loss is:
        L = -E_{y~expert}[ (π_θ(y|x) / π_ref(y|x)) * log π_θ(y|x) ]

    where π_ref is the reference (initial / base) policy.

    Args:
        model: HFModel wrapping the causal LM
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

    if state is None:
        state = {}

    logger.info("---- Begin DFT Training ----")
    set_seeds(config.seed)

    train_loader = loaders["train"]
    num_batches = len(train_loader)
    total_steps = num_batches * epochs

    # Store reference model (frozen copy of initial weights)
    raw_model = get_raw_model(model)
    import copy
    ref_model = copy.deepcopy(raw_model)
    ref_model.eval()
    ref_model.requires_grad_(False)
    logger.info("Created frozen reference model for importance weighting")

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

    # LR schedule
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

    losses = []
    pbar = tqdm(total=total_steps, **get_tqdm_kwargs(logger, ncols=120))
    num_steps = 0
    resume_step = state.get("step", -1)

    for epoch_idx in range(epochs):
        for batch in train_loader:
            num_steps += 1
            pbar.update()
            if num_steps <= resume_step:
                continue

            device = config.device
            input_ids = batch["teacher_input_ids"].to(device)
            attention_mask = batch["teacher_attention_mask"].to(device)

            # Student log-probs
            outputs = raw_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            student_logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            student_token_lp = student_log_probs.gather(
                dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)

            # Reference log-probs
            with torch.no_grad():
                ref_outputs = ref_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                ref_logits = ref_outputs.logits[:, :-1, :]
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_token_lp = ref_log_probs.gather(
                    dim=-1, index=targets.unsqueeze(-1)
                ).squeeze(-1)

            # Importance weight: π_θ / π_ref (in log space, then clamp)
            log_importance = (student_token_lp - ref_token_lp).detach()
            importance = torch.clamp(log_importance.exp(), max=5.0)

            # Weighted policy gradient loss
            mask = attention_mask[:, 1:].float()
            loss = -(importance * student_token_lp * mask).sum() / mask.sum().clamp(min=1)

            scaled_loss = loss / acc_steps
            scaled_loss.backward()

            loss_val = loss.item()
            losses.append(loss_val)

            pbar.set_description(f"DFT | LR: {cur_lr:.2e} | Loss: {loss_val:.4f}")

            if num_steps % 200 == 0 or num_steps == total_steps:
                avg_loss = np.mean(losses[-200:])
                logger.info(f"Step {num_steps}/{total_steps} | LR: {cur_lr:.2e} | Loss: {avg_loss:.4f}")

            if num_steps % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                    cur_lr = scheduler.get_last_lr()[0]

            if num_steps % 5000 == 0 or num_steps == total_steps:
                save_checkpoint(
                    model=model,
                    optimizer=opt,
                    step=num_steps,
                    total_steps=total_steps,
                    res_dir=config.res_dir,
                    prefix="dft",
                    logger=logger,
                )
                barrier()

    pbar.close()

    # Cleanup reference model
    del ref_model
    torch.cuda.empty_cache()

    state = {
        "optimizer": opt.state_dict(),
        "step": num_steps,
        "total_steps": total_steps,
    }

    return model, state

