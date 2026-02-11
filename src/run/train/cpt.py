"""
Continual Pre-Training (CPT) baseline for Knowledge Acquisition.

Trains directly on the text corpus using next-token prediction loss.
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


def do_cpt(
    model: torch.nn.Module,
    config: RunConfig,
    corpus_texts: list[str] | None = None,
    state: dict | None = None,
) -> tuple[torch.nn.Module, dict]:
    """
    Continual Pre-Training on raw text corpus.

    For Knowledge Acquisition, the model is trained on the raw
    Wikipedia article text using standard next-token prediction.

    Args:
        model: HFModel wrapping the causal LM
        config: Run configuration
        corpus_texts: List of raw text articles to train on
        state: Optional state for resumption

    Returns:
        (trained_model, state_dict)
    """
    logger = config.logger
    acc_steps = config.accumulation_steps
    lr = config.lr
    epochs = config.epochs
    lr_schedule = config.lr_schedule

    if state is None:
        state = {}

    if corpus_texts is None or len(corpus_texts) == 0:
        logger.warning("No corpus texts provided for CPT")
        return model, state

    logger.info(f"---- Begin CPT Training ({len(corpus_texts)} articles) ----")
    set_seeds(config.seed)

    raw_model = get_raw_model(model)
    device = config.device

    # Tokenize all corpus texts into chunks
    tokenizer = raw_model.tokenizer
    max_length = config.max_gen_len

    all_chunks = []
    for text in corpus_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Split into max_length chunks
        for i in range(0, len(tokens), max_length):
            chunk = tokens[i : i + max_length]
            if len(chunk) > 10:  # skip very short chunks
                all_chunks.append(torch.tensor(chunk))

    logger.info(f"Created {len(all_chunks)} training chunks")

    total_steps = len(all_chunks) * epochs
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

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

    for epoch_idx in range(epochs):
        # Shuffle chunks each epoch
        import random
        random.shuffle(all_chunks)

        for chunk in all_chunks:
            num_steps += 1
            pbar.update()

            input_ids = chunk.unsqueeze(0).to(device)
            attention_mask = torch.ones_like(input_ids)

            outputs = raw_model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :]
            targets = input_ids[:, 1:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            scaled_loss = loss / acc_steps
            scaled_loss.backward()

            loss_val = loss.item()
            losses.append(loss_val)

            pbar.set_description(f"CPT | LR: {cur_lr:.2e} | Loss: {loss_val:.4f}")

            if num_steps % 200 == 0:
                avg_loss = np.mean(losses[-200:])
                logger.info(f"Step {num_steps}/{total_steps} | Loss: {avg_loss:.4f}")

            if num_steps % acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
                    cur_lr = scheduler.get_last_lr()[0]

    pbar.close()

    state = {
        "optimizer": opt.state_dict(),
        "step": num_steps,
        "total_steps": total_steps,
    }

    return model, state

