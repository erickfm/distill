"""
Utility functions for SDFT experiments.

Mirrors the ICML codebase utility pattern.
"""

import json
import random
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.run.distributed import is_main_process


def get_timestamp() -> str:
    """Get a formatted timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def log_line(msg: dict, log_fp: Path) -> None:
    """Append a JSON line to a log file (main process only)."""
    if is_main_process():
        with open(log_fp, "a") as f:
            f.write(json.dumps(msg, default=str) + "\n")


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    res_dir: Path,
    prefix: str = "checkpoint",
    extra_state: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Save a training checkpoint (main process only)."""
    if not is_main_process():
        return

    from src.run.distributed import get_raw_model

    checkpoint = {
        "model": get_raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "total_steps": total_steps,
    }

    if extra_state is not None:
        checkpoint.update(extra_state)

    checkpoint_dir = res_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{prefix}_step-{step}.pth"

    torch.save(checkpoint, path)
    if logger:
        logger.info(f"Saved checkpoint at step {step} to {path}")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """Load a training checkpoint."""
    from src.run.distributed import get_raw_model

    if logger:
        logger.info(f"Loading checkpoint from: {path}")

    state = torch.load(path, map_location=device or "cpu")
    get_raw_model(model).load_state_dict(state["model"])

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])

    if logger:
        logger.info(
            f"Restored model from step {state.get('step', '?')}/{state.get('total_steps', '?')}"
        )

    return state

