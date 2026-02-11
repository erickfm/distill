"""
Run configuration for SDFT experiments.

Mirrors the ICML codebase config.py pattern — a RunConfig dataclass
plus a setup() function that initializes everything.
"""

import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

from src.model.config import ModelConfig
from src.run.dataloader import make_loaders
from src.run.utils import ensure_dir, get_timestamp, set_seeds
from src.run.logger import setup_logger
from src.run.distributed import (
    get_world_size,
    is_main_process,
    barrier,
    broadcast_object,
)


# --------------------------------------------------------------------------- #
# Suppress noisy warnings                                                      #
# --------------------------------------------------------------------------- #

warnings.filterwarnings("ignore", message=r".*torch\._dynamo.*recompile_limit.*", category=UserWarning)
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
hf_logging.set_verbosity_error()


# --------------------------------------------------------------------------- #
# RunConfig                                                                    #
# --------------------------------------------------------------------------- #

@dataclass
class RunConfig:
    """Runtime configuration for an SDFT experiment."""

    # Data
    task: str  # "science_qa", "tool_use", "medical", "knowledge"
    data_dir: Path
    loaders: dict  # {"train": DataLoader, "val": DataLoader, "test": DataLoader}

    # Model
    model_config: ModelConfig

    # Training
    method: str  # "sdft", "sft", "dft", "cpt"
    batch_size: int
    epochs: int
    lr: float
    lr_schedule: bool
    accumulation_steps: int

    # SDFT-specific
    ema_decay: float
    kl_estimator: str  # "analytic", "token", "rao_blackwell"
    max_gen_len: int

    # Infrastructure
    res_dir: Path
    device: str
    timestamp: str
    seed: int
    num_gpus: int
    logger: logging.Logger
    do_compile: bool

    # Evaluation
    eval_benchmarks: list  # list of benchmark names for prior capability eval

    # ESDFT-specific (defaults make these optional for non-ESDFT runs)
    esdft_screening_threshold: float = 0.5
    esdft_audit_init: float = 0.3
    esdft_discrepancy_target: float = 0.05
    esdft_audit_alpha: float = 0.1
    esdft_audit_min: float = 0.05
    esdft_audit_max: float = 0.5


def validate_stages(stages: list[dict]) -> None:
    """Validate stage configurations."""
    valid_methods = ["sft", "sdft", "dft", "cpt", "sft_reinvoke", "esdft"]
    for stage in stages:
        method = stage.get("method", "sdft")
        assert method in valid_methods, f"Invalid method: {method}. Must be one of {valid_methods}"


def save_config(
    stages: list[dict],
    model_config: ModelConfig,
    run_config: RunConfig,
) -> None:
    """Save configuration to JSON file (main process only)."""
    if not is_main_process():
        return

    logger = run_config.logger

    config_data = {
        "stages": stages,
        "run": {
            "task": run_config.task,
            "method": run_config.method,
            "seed": run_config.seed,
            "batch_size": run_config.batch_size,
            "epochs": run_config.epochs,
            "lr": run_config.lr,
            "lr_schedule": run_config.lr_schedule,
            "ema_decay": run_config.ema_decay,
            "kl_estimator": run_config.kl_estimator,
            "max_gen_len": run_config.max_gen_len,
            "device": run_config.device,
            "timestamp": run_config.timestamp,
            "num_gpus": run_config.num_gpus,
            "do_compile": run_config.do_compile,
            "accumulation_steps": run_config.accumulation_steps,
        },
        "model": {
            "model_name": model_config.model_name,
            "dtype": model_config.dtype,
            "use_vllm": model_config.use_vllm,
        },
        "esdft": {
            "screening_threshold": run_config.esdft_screening_threshold,
            "audit_init": run_config.esdft_audit_init,
            "discrepancy_target": run_config.esdft_discrepancy_target,
            "audit_alpha": run_config.esdft_audit_alpha,
            "audit_min": run_config.esdft_audit_min,
            "audit_max": run_config.esdft_audit_max,
        },
    }

    out_str = json.dumps(config_data, indent=4, ensure_ascii=False, default=str)
    with open(run_config.res_dir / "config.json", "w") as f:
        f.write(out_str)

    logger.info(f"Saved configuration to {run_config.res_dir}/config.json")
    logger.info(out_str)


def setup(
    # Stage config
    stages: list[dict],

    # Task
    task: str,
    data_dir: str | Path,

    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    dtype: str = "bfloat16",
    use_vllm: bool = True,

    # Training
    method: str = "sdft",
    batch_size: int = 4,
    epochs: int = 2,
    lr: float = 1e-5,
    lr_schedule: bool = True,
    accumulation_steps: int = 1,

    # SDFT-specific
    ema_decay: float = 0.999,
    kl_estimator: str = "analytic",
    max_gen_len: int = 2048,

    # Infrastructure
    do_compile: bool = False,
    seed: int = 42,
    res_dir: str | Path = "",
    log_level: str = "INFO",
    max_examples: Optional[int] = None,
    system_prompt: Optional[str] = None,

    # Evaluation
    eval_benchmarks: Optional[list[str]] = None,

    # ESDFT-specific
    esdft_screening_threshold: float = 0.5,
    esdft_audit_init: float = 0.3,
    esdft_discrepancy_target: float = 0.05,
    esdft_audit_alpha: float = 0.1,
    esdft_audit_min: float = 0.05,
    esdft_audit_max: float = 0.5,

    # Process control
    timestamp: Optional[str] = None,
    process_id: Optional[int] = None,
    do_cleanup_distributed: bool = True,
) -> dict:
    """
    Set up all components for an SDFT experiment.

    Returns:
        Dict with "model_config" and "run_config".
    """
    validate_stages(stages)

    # Seed
    if seed == -1:
        seed = int(time.time()) if is_main_process() else None
        seed = broadcast_object(seed, src=0)
    set_seeds(seed)

    # CUDA
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # Paths
    data_dir = Path(data_dir)
    res_dir = Path(res_dir)
    if timestamp is None:
        timestamp = get_timestamp()

    if is_main_process():
        ensure_dir(res_dir)
    barrier()

    num_gpus = get_world_size()

    # Logger
    log_file = res_dir / "training.log"
    logger = setup_logger(
        name=f"sdft_{timestamp}",
        log_file=log_file,
        level=log_level,
        process_id=process_id,
    )
    logger.info(f"Number of GPUs: {num_gpus}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model config
    model_config = ModelConfig(
        model_name=model_name,
        ema_decay=ema_decay,
        max_gen_len=max_gen_len,
        kl_estimator=kl_estimator,
        dtype=dtype,
        use_vllm=use_vllm,
    )

    # Data loaders
    loaders = make_loaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_length=max_gen_len,
        max_examples=max_examples,
        system_prompt=system_prompt,
        seed=seed,
    )

    for split, loader in loaders.items():
        logger.info(f"Loader {split}: {len(loader)} batches, {len(loader.dataset)} examples")

    # Default eval benchmarks
    if eval_benchmarks is None:
        eval_benchmarks = [
            "hellaswag",
            "truthfulqa_mc2",
            "mmlu",
            "ifeval",
            "winogrande",
            "humaneval",
        ]

    # Run config
    run_config = RunConfig(
        task=task,
        data_dir=data_dir,
        loaders=loaders,
        model_config=model_config,
        method=method,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        lr_schedule=lr_schedule,
        accumulation_steps=accumulation_steps,
        ema_decay=ema_decay,
        kl_estimator=kl_estimator,
        max_gen_len=max_gen_len,
        res_dir=res_dir,
        device=str(device),
        timestamp=timestamp,
        seed=seed,
        num_gpus=num_gpus,
        logger=logger,
        do_compile=do_compile,
        eval_benchmarks=eval_benchmarks,
        esdft_screening_threshold=esdft_screening_threshold,
        esdft_audit_init=esdft_audit_init,
        esdft_discrepancy_target=esdft_discrepancy_target,
        esdft_audit_alpha=esdft_audit_alpha,
        esdft_audit_min=esdft_audit_min,
        esdft_audit_max=esdft_audit_max,
    )

    # Save config
    save_config(stages, model_config, run_config)

    return {
        "model_config": model_config,
        "run_config": run_config,
    }

