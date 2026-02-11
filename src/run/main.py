"""
@date: 2026-02-11

main.py — SDFT reproduction pipeline.

Mirrors the ICML codebase main.py pattern:
  run() → setup() → run_experiments() → run_experiment()

Outputs to src/results/{task}/results_YYYY-MM-DD_HH-MM-SS
"""

import argparse
import json
import os
import torch
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Any, Optional
import warnings

warnings.filterwarnings("ignore", message=".*Online softmax is disabled.*", category=UserWarning)

from src.model.config import ModelConfig
from src.model.hf_model import HFModel

from src.run.train.sft import do_sft
from src.run.train.sdft import do_sdft
from src.run.train.dft import do_dft
from src.run.train.sft_reinvoke import do_sft_reinvoke
from src.run.train.cpt import do_cpt
from src.run.train.esdft import do_esdft

from src.run.config import RunConfig, setup
from src.run.eval import do_eval
from src.run.utils import get_timestamp, load_checkpoint
from src.run.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_raw_model,
)


# --------------------------------------------------------------------------- #
# Training method dispatch                                                     #
# --------------------------------------------------------------------------- #

TRAIN_METHODS: Dict[str, Callable] = {
    "sft": do_sft,
    "sdft": do_sdft,
    "dft": do_dft,
    "sft_reinvoke": do_sft_reinvoke,
    "cpt": do_cpt,
    "esdft": do_esdft,
}


# --------------------------------------------------------------------------- #
# Experiment runner                                                            #
# --------------------------------------------------------------------------- #

def run_experiment(
    stage: dict,
    model: torch.nn.Module,
    config: RunConfig,
    func: Callable,
    func_args: Optional[dict] = None,
    eval_args: Optional[dict] = None,
) -> tuple[torch.nn.Module, dict | None]:
    """
    Run a training stage with optional checkpoint loading and train/eval gating.

    Stage config options:
        checkpoint: Optional[str] — path to load weights from before training
        do_train: bool (default True) — whether to run the training function
        do_eval: bool (default True) — whether to run evaluation after
    """
    logger = config.logger

    if func_args is None:
        func_args = {}
    if eval_args is None:
        eval_args = {}

    # Load checkpoint if specified
    state = func_args.get("state", {})
    checkpoint_path = stage.get("checkpoint")
    if checkpoint_path is not None:
        state = load_checkpoint(checkpoint_path, model, device=config.device, logger=logger)
        func_args["state"] = state
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    if stage.get("do_train", True):
        model, state = func(
            model=model,
            config=config,
            **func_args,
        )

    if stage.get("do_eval", True):
        do_eval(
            stage=stage,
            model=model,
            config=config,
            **eval_args,
        )

    return model, state


def run_experiments(
    stages: list[dict],
    model_config: ModelConfig,
    run_config: RunConfig,
) -> None:
    """
    Execute a sequence of training stages.

    Each stage specifies a method (sft, sdft, dft, etc.) and optional
    configuration overrides.
    """
    logger = run_config.logger
    device = run_config.device

    for stage in stages:
        stage_method = stage.get("method", run_config.method)
        logger.info(f"Stage Start: {json.dumps(stage, default=str, ensure_ascii=False)}")

        # Load model fresh for each stage (or reuse if continuing)
        logger.info(f"Loading model: {model_config.model_name}")
        model = HFModel(
            model_name=model_config.model_name,
            dtype=model_config.dtype,
            device=device,
        )

        # Get training function
        func = TRAIN_METHODS.get(stage_method)
        if func is None:
            raise ValueError(f"Unknown training method: {stage_method}")

        # Build func args from stage config
        func_args = {}
        for key in ["reinvoke_prompts", "reinvoke_steps", "reinvoke_lr", "corpus_texts"]:
            if key in stage:
                func_args[key] = stage[key]

        # Temporarily override config for this stage
        original_method = run_config.method
        run_config.method = stage_method

        if "kl_estimator" in stage:
            run_config.kl_estimator = stage["kl_estimator"]
        if "ema_decay" in stage:
            run_config.ema_decay = stage["ema_decay"]
        if "epochs" in stage:
            run_config.epochs = stage["epochs"]
        if "lr" in stage:
            run_config.lr = stage["lr"]

        # ESDFT overrides
        for esdft_key in [
            "esdft_screening_threshold",
            "esdft_audit_init",
            "esdft_discrepancy_target",
            "esdft_audit_alpha",
            "esdft_audit_min",
            "esdft_audit_max",
        ]:
            if esdft_key in stage:
                setattr(run_config, esdft_key, stage[esdft_key])

        # Run
        model, _ = run_experiment(
            stage=stage,
            model=model,
            config=run_config,
            func=func,
            func_args=func_args,
            eval_args={"log": {
                "method": stage_method,
                "stage_config": stage,
            }},
        )

        # Restore config
        run_config.method = original_method

        # Cleanup
        del model
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Main entry point                                                             #
# --------------------------------------------------------------------------- #

def run(
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
) -> None:
    """
    Main entry point for SDFT experiments.

    Mirrors the ICML codebase run() function.
    """
    setup_distributed()

    try:
        configs = setup(
            stages=stages,
            task=task,
            data_dir=data_dir,
            model_name=model_name,
            dtype=dtype,
            use_vllm=use_vllm,
            method=method,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            lr_schedule=lr_schedule,
            accumulation_steps=accumulation_steps,
            ema_decay=ema_decay,
            kl_estimator=kl_estimator,
            max_gen_len=max_gen_len,
            do_compile=do_compile,
            seed=seed,
            res_dir=res_dir,
            log_level=log_level,
            max_examples=max_examples,
            system_prompt=system_prompt,
            eval_benchmarks=eval_benchmarks,
            esdft_screening_threshold=esdft_screening_threshold,
            esdft_audit_init=esdft_audit_init,
            esdft_discrepancy_target=esdft_discrepancy_target,
            esdft_audit_alpha=esdft_audit_alpha,
            esdft_audit_min=esdft_audit_min,
            esdft_audit_max=esdft_audit_max,
            timestamp=timestamp,
            process_id=process_id,
            do_cleanup_distributed=do_cleanup_distributed,
        )

        logger = configs["run_config"].logger
        run_experiments(stages, **configs)

        logger.info("-" * 40)
        logger.info(f"Finished. See {configs['run_config'].res_dir}")

    finally:
        if do_cleanup_distributed:
            cleanup_distributed()


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    torch.cuda.empty_cache()

    root_dir = Path("src").absolute()
    timestamp = get_timestamp()

    # Default: Science Q&A with SDFT
    defaults = {
        "task": "science_qa",
        "data_dir": root_dir / "data/science_qa",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "bfloat16",
        "use_vllm": False,
        "method": "sdft",
        "batch_size": 4,
        "epochs": 2,
        "lr": 1e-5,
        "lr_schedule": True,
        "accumulation_steps": 4,
        "ema_decay": 0.999,
        "kl_estimator": "analytic",
        "max_gen_len": 2048,
        "do_compile": False,
        "seed": 42,
        "res_dir": root_dir / f"results/science_qa/results_{timestamp}",
        "log_level": "DEBUG",
        "do_cleanup_distributed": True,
        "stages": [
            {"method": "sdft", "do_train": True, "do_eval": True},
        ],
    }

    run(**defaults)

