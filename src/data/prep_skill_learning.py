"""
Data preparation for Skill Learning tasks.

Three domains from the paper:
1. Science Q&A — Chemistry L-3 subset of SciKnowEval (Feng et al., 2024)
2. Tool Use — ToolAlpaca (Tang et al., 2023)
3. Medical — HuatuoGPT-o1 pipeline (Chen et al., 2024)

Each dataset is downloaded, split, and saved to src/data/{task}/ as JSONL files.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

from datasets import load_dataset


DATA_DIR = Path(__file__).parent


# --------------------------------------------------------------------------- #
# Science Q&A                                                                  #
# --------------------------------------------------------------------------- #

def prep_science_qa(seed: int = 42, output_dir: Optional[Path] = None) -> None:
    """
    Prepare Science Q&A dataset from SciKnowEval Chemistry L-3.

    The HF dataset has a single 'test' split — we filter to
    domain=Chemistry, level=L3, then split 75/5/20 into
    train/val/test ourselves.
    """
    output_dir = output_dir or DATA_DIR / "science_qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SciKnowEval (full test split)...")
    ds = load_dataset("hicai-zju/sciknoweval", split="test")

    # Filter to Chemistry L3
    data = [
        item for item in ds
        if item["domain"] == "Chemistry"
        and item.get("details", {}).get("level") == "L3"
    ]
    print(f"  Chemistry L3 subset: {len(data)} examples")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(0.75 * n)
    n_val = int(0.05 * n)

    splits = {
        "train": data[:n_train],
        "val": data[n_train : n_train + n_val],
        "test": data[n_train + n_val :],
    }

    for split_name, split_data in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for item in split_data:
                # Build the demonstration from the prompt template + answer
                prompt_template = item.get("prompt", {}).get("default", "")
                question = item["question"]
                answer = item.get("answer", "") or item.get("answerKey", "")

                # For MCQ, format choices into the question text
                choices = item.get("choices", {})
                choice_text = choices.get("text", [])
                choice_label = choices.get("label", [])
                if choice_text:
                    q_with_choices = question + "\n" + "\n".join(
                        f"{label}. {text}"
                        for label, text in zip(choice_label, choice_text)
                    )
                else:
                    q_with_choices = question

                # The demonstration is the full worked solution
                demonstration = f"{prompt_template}\n\n{q_with_choices}\n\nAnswer: {answer}"

                entry = {
                    "query": q_with_choices,
                    "answer": answer if answer else item.get("answerKey", ""),
                    "demonstration": demonstration,
                    "task_type": item.get("type", ""),
                }
                f.write(json.dumps(entry) + "\n")
        print(f"  {split_name}: {len(split_data)} examples -> {out_path}")


# --------------------------------------------------------------------------- #
# Tool Use                                                                     #
# --------------------------------------------------------------------------- #

def prep_tool_use(seed: int = 42, output_dir: Optional[Path] = None) -> None:
    """
    Prepare Tool Use dataset from ToolAlpaca.

    Uses the original train-test split from the dataset.
    """
    output_dir = output_dir or DATA_DIR / "tool_use"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ToolAlpaca...")
    ds = load_dataset("TangQiaoYu/ToolAlpaca")

    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        split_data = list(ds[split_name])

        # Hold out 5% of train for validation
        if split_name == "train":
            random.seed(seed)
            random.shuffle(split_data)
            n_val = max(1, int(0.05 * len(split_data)))
            val_data = split_data[:n_val]
            split_data = split_data[n_val:]

            val_path = output_dir / "val.jsonl"
            with open(val_path, "w") as f:
                for item in val_data:
                    entry = _format_tool_use(item)
                    f.write(json.dumps(entry) + "\n")
            print(f"  val: {len(val_data)} examples -> {val_path}")

        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for item in split_data:
                entry = _format_tool_use(item)
                f.write(json.dumps(entry) + "\n")
        print(f"  {split_name}: {len(split_data)} examples -> {out_path}")


def _format_tool_use(item: dict) -> dict:
    """Format a ToolAlpaca item into standard format."""
    return {
        "query": item.get("instruction", item.get("input", "")),
        "api_spec": item.get("api_spec", item.get("tools", "")),
        "answer": item.get("output", item.get("response", "")),
        "demonstration": item.get("output", item.get("response", "")),
    }


# --------------------------------------------------------------------------- #
# Medical                                                                      #
# --------------------------------------------------------------------------- #

def prep_medical(seed: int = 42, output_dir: Optional[Path] = None) -> None:
    """
    Prepare Medical dataset from HuatuoGPT-o1.

    Training data from stage 1, evaluation from stage 2.
    """
    output_dir = output_dir or DATA_DIR / "medical"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HuatuoGPT-o1...")
    ds = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT", "en")

    for split_name in ["train", "test"]:
        if split_name not in ds:
            continue
        split_data = list(ds[split_name])

        # Hold out 5% of train for validation
        if split_name == "train":
            random.seed(seed)
            random.shuffle(split_data)
            n_val = max(1, int(0.05 * len(split_data)))
            val_data = split_data[:n_val]
            split_data = split_data[n_val:]

            val_path = output_dir / "val.jsonl"
            with open(val_path, "w") as f:
                for item in val_data:
                    entry = _format_medical(item)
                    f.write(json.dumps(entry) + "\n")
            print(f"  val: {len(val_data)} examples -> {val_path}")

        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for item in split_data:
                entry = _format_medical(item)
                f.write(json.dumps(entry) + "\n")
        print(f"  {split_name}: {len(split_data)} examples -> {out_path}")


def _format_medical(item: dict) -> dict:
    """Format a medical item into standard format."""
    return {
        "query": item.get("question", item.get("input", "")),
        "answer": item.get("answer", item.get("output", "")),
        "demonstration": item.get("answer", item.get("output", "")),
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Skill Learning datasets")
    parser.add_argument("--task", type=str, choices=["science_qa", "tool_use", "medical", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = {
        "science_qa": prep_science_qa,
        "tool_use": prep_tool_use,
        "medical": prep_medical,
    }

    if args.task == "all":
        for name, func in tasks.items():
            print(f"\n{'='*60}")
            print(f"Preparing {name}...")
            print(f"{'='*60}")
            func(seed=args.seed)
    else:
        tasks[args.task](seed=args.seed)

    print("\nDone!")

