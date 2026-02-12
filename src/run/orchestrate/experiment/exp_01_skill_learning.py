"""
Experiment 01 — Skill Learning:  Baseline  vs  SDFT  vs  Efficient SDFT

Compares three conditions on the Science Q&A task (SciKnowEval Chemistry L-3):

    1. Baseline  — base Qwen-7B-Instruct, no fine-tuning (eval only)
    2. SDFT      — standard self-distillation fine-tuning (paper reproduction)
    3. ESDFT     — efficient SDFT with adaptive disagreement screening

All three are evaluated on:
    - Task accuracy (exact-match on Science Q&A)
    - Prior capabilities (HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval)

Prerequisites:
    Prepare data first:
        python -m src.data.prep_skill_learning --task science_qa

Usage:
    # Run all three conditions in parallel across available GPUs
    python -m src.run.orchestrate.experiment.exp_01_skill_learning

    # Or with explicit GPU count
    python -m src.run.orchestrate.experiment.exp_01_skill_learning --num_gpus 3
"""

import argparse
from copy import deepcopy
from pathlib import Path

from src.run.utils import get_timestamp
from src.run.orchestrate.config import SkillLearningBaseArgs, root_dir
from src.run.orchestrate.parallel import run_parallel


def get_configs(timestamp: str | None = None):
    """
    Build the three experiment configurations.

    Returns:
        configs:  list of config dicts (one per condition)
        res_root: path to the shared results directory
    """
    if timestamp is None:
        timestamp = get_timestamp()

    data_dir = str(root_dir / "data" / "science_qa")
    res_root = str(root_dir / "results" / "exp_01" / f"exp_01_{timestamp}")

    configs = []

    # --------------------------------------------------------------------- #
    # Condition 1 — Baseline (no fine-tuning)                                #
    # --------------------------------------------------------------------- #
    baseline = deepcopy(SkillLearningBaseArgs)
    baseline.update({
        "task": "science_qa",
        "data_dir": data_dir,
        "method": "sft",          # method is irrelevant when training is skipped
        "stages": [
            {"method": "sft", "do_train": False, "do_eval": True},
        ],
    })
    configs.append(baseline)

    # --------------------------------------------------------------------- #
    # Condition 2 — SDFT (paper reproduction)                                #
    # --------------------------------------------------------------------- #
    sdft = deepcopy(SkillLearningBaseArgs)
    sdft.update({
        "task": "science_qa",
        "data_dir": data_dir,
        "method": "sdft",
        "stages": [
            {"method": "sdft", "do_train": True, "do_eval": True},
        ],
    })
    configs.append(sdft)

    # --------------------------------------------------------------------- #
    # Condition 3 — Efficient SDFT                                           #
    # --------------------------------------------------------------------- #
    esdft = deepcopy(SkillLearningBaseArgs)
    esdft.update({
        "task": "science_qa",
        "data_dir": data_dir,
        "method": "esdft",
        "stages": [
            {"method": "esdft", "do_train": True, "do_eval": True},
        ],
        # ESDFT-specific hyperparameters
        "esdft_screening_threshold": 0.01,
        "esdft_warmup_frac": 0.1,
        "esdft_audit_init": 0.3,
        "esdft_discrepancy_target": 0.05,
        "esdft_audit_alpha": 0.1,
        "esdft_audit_min": 0.05,
        "esdft_audit_max": 0.5,
    })
    configs.append(esdft)

    return configs, res_root


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Exp 01: Skill Learning — Baseline vs SDFT vs ESDFT"
    )
    parser.add_argument(
        "--num_gpus", type=int, default=-1,
        help="Number of GPUs to use (-1 = all available)",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = parser.parse_args()

    configs, res_root = get_configs()

    print("=" * 72)
    print("  Experiment 01: Skill Learning Comparison")
    print("=" * 72)
    print(f"  Conditions : Baseline, SDFT, ESDFT")
    print(f"  Task       : Science Q&A (SciKnowEval Chemistry L-3)")
    print(f"  Model      : {configs[0]['model_name']}")
    print(f"  Epochs     : {configs[0]['epochs']}")
    print(f"  Batch size : {configs[0]['batch_size']}")
    print(f"  Output     : {res_root}")
    print("=" * 72)
    print()

    run_parallel(
        configs,
        res_root=res_root,
        log_level=args.log_level,
        num_gpus=args.num_gpus,
    )

