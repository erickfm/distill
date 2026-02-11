"""
Minimal distributed training utilities for torchrun.

Mirrors the ICML codebase pattern — lightweight DDP helpers that
work with torchrun and degrade gracefully to single-GPU.

Usage:
    # Single GPU
    python src/run/main.py

    # Multi-GPU (4 GPUs)
    torchrun --nproc_per_node=4 src/run/main.py
"""

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ============================================================================
# Distributed Setup & Status
# ============================================================================

def setup_distributed() -> None:
    """
    Initialize distributed training.
    Assumes launched with torchrun which sets RANK, WORLD_SIZE, LOCAL_RANK env vars.
    Does nothing if not launched with torchrun (single-GPU mode).
    Idempotent — safe to call multiple times.
    """
    if not is_distributed_launch():
        return

    if dist.is_initialized():
        return

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"Initialized distributed training: {get_world_size()} GPUs")


def is_distributed_launch() -> bool:
    """Check if launched with torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def is_distributed() -> bool:
    """Check if currently running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank. Returns 0 for single-GPU."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes. Returns 1 for single-GPU."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is rank 0 (main process)."""
    return get_rank() == 0


def barrier() -> None:
    """Synchronize all processes. No-op in single-GPU mode."""
    if is_distributed():
        device = torch.cuda.current_device()
        dist.barrier(device_ids=[device])


def cleanup_distributed() -> None:
    """Clean up distributed process group."""
    if is_distributed():
        dist.destroy_process_group()


# ============================================================================
# Model Unwrapping
# ============================================================================

def get_raw_model(model: nn.Module) -> nn.Module:
    """
    Get underlying model, unwrapping DDP and torch.compile if needed.

    Handles wrapping order: DDP(CompiledModel(BaseModel))
    """
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


# ============================================================================
# Collective Operations
# ============================================================================

def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """All-reduce a tensor across processes."""
    if not is_distributed():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if average:
        tensor /= get_world_size()
    return tensor


def broadcast_object(obj, src: int = 0):
    """Broadcast any picklable object from source rank to all ranks."""
    if not is_distributed():
        return obj
    obj_list = [obj if get_rank() == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]

