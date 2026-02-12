"""
Base argument configurations for SDFT experiments.

Mirrors the ICML codebase orchestrate/config.py pattern.
"""

from typing import Dict, Any
from pathlib import Path

root_dir = Path("src").absolute()


# --------------------------------------------------------------------------- #
# Skill Learning base args                                                     #
# --------------------------------------------------------------------------- #

SkillLearningBaseArgs: Dict[str, Any] = {
    # Task
    "task": "",  # set per experiment: "science_qa", "tool_use", "medical"
    "data_dir": None,  # set per experiment

    # Model
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "bfloat16",
    "use_vllm": False,

    # Training
    "method": "sdft",
    "batch_size": 4,
    "epochs": 2,
    "lr": 1e-5,
    "lr_schedule": True,
    "accumulation_steps": 4,

    # SDFT
    "ema_decay": 0.999,
    "kl_estimator": "analytic",
    "max_gen_len": 512,  # Science Q&A answers are short; saves ~4x activation memory

    # Infrastructure
    "do_compile": False,
    "seed": -1,
    "res_dir": None,
    "log_level": "INFO",
    "timestamp": None,
    "max_examples": None,
    "system_prompt": None,
    "do_cleanup_distributed": True,

    # Evaluation — lightweight subset for reasonable runtime
    "eval_benchmarks": [
        "hellaswag",
        "truthfulqa_mc2",
        "winogrande",
    ],

    # Stages
    "stages": [],
}


# --------------------------------------------------------------------------- #
# Knowledge Acquisition base args                                              #
# --------------------------------------------------------------------------- #

KnowledgeBaseArgs: Dict[str, Any] = {
    # Task
    "task": "knowledge",
    "data_dir": root_dir / "data/knowledge",

    # Model
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "dtype": "bfloat16",
    "use_vllm": False,

    # Training
    "method": "sdft",
    "batch_size": 4,
    "epochs": 4,  # paper uses 4 epochs for knowledge
    "lr": 1e-5,
    "lr_schedule": True,
    "accumulation_steps": 4,

    # SDFT
    "ema_decay": 0.999,
    "kl_estimator": "analytic",
    "max_gen_len": 2048,

    # Infrastructure
    "do_compile": False,
    "seed": -1,
    "res_dir": None,
    "log_level": "INFO",
    "timestamp": None,
    "max_examples": None,
    "system_prompt": None,
    "do_cleanup_distributed": True,

    # Evaluation
    "eval_benchmarks": [
        "hellaswag",
        "truthfulqa_mc2",
        "mmlu",
        "ifeval",
        "winogrande",
        "humaneval",
    ],

    # Stages
    "stages": [],
}


# --------------------------------------------------------------------------- #
# Scaling experiment args (Qwen 3B, 7B, 14B)                                   #
# --------------------------------------------------------------------------- #

SCALING_MODELS = {
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
}

