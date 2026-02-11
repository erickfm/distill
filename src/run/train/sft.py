"""
Supervised Fine-Tuning (SFT) baseline.

Standard off-policy training: minimize cross-entropy on expert demonstrations.
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
from src.run.distributed import is_main_process, barrier


def do_sft(
    model: torch.nn.Module,
    config: RunConfig,
    state: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Train a model with supervised fine-tuning on demonstration data.

    For each (query, demonstration) pair, the model is trained to predict
    the demonstration tokens via cross-entropy loss.

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

    logger.info("---- Begin SFT Training ----")
    set_seeds(config.seed)

    train_loader = loaders["train"]
    num_batches = len(train_loader)
    total_steps = num_batches * epochs

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

    # Learning rate schedule (WSD: 10% warmup, 80% stable, 10% decay)
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
    pbar = tqdm(total=total_steps, **get_tqdm_kwargs(logger, ncols=120))
    num_steps = 0
    resume_step = state.get("step", -1)

    for epoch_idx in range(epochs):
        for batch in train_loader:
            num_steps += 1
            pbar.update()
            if num_steps <= resume_step:
                continue

            # Teacher prompt contains the demonstration; we train on those tokens
            input_ids = batch["teacher_input_ids"].to(config.device)
            attention_mask = batch["teacher_attention_mask"].to(config.device)

            # Forward pass — cross-entropy on next-token prediction
            outputs = model.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=model.tokenizer.pad_token_id,
                reduction="mean",
            )

            scaled_loss = loss / acc_steps
            scaled_loss.backward()

            loss_val = loss.item()
            losses.append(loss_val)

            pbar.set_description(f"SFT | LR: {cur_lr:.2e} | Loss: {loss_val:.4f}")

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

            # Checkpoint
            if num_steps % 5000 == 0 or num_steps == total_steps:
                save_checkpoint(
                    model=model,
                    optimizer=opt,
                    step=num_steps,
                    total_steps=total_steps,
                    res_dir=config.res_dir,
                    prefix="sft",
                    extra_state={
                        "scheduler_epoch": scheduler.last_epoch if scheduler else None,
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
    }

    return model, state

