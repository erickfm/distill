"""
Evaluation for SDFT experiments.

Two evaluation dimensions from the paper:
1. Task accuracy — how well the model learned the new task
2. Prior capabilities — whether general abilities are preserved

Task accuracy:
- Science Q&A: exact match on multiple choice
- Tool Use: exact match on tool call output
- Medical: LLM-as-judge evaluation
- Knowledge: LLM-as-judge evaluation

Prior capabilities (via lm-eval-harness):
- HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval
"""

from __future__ import annotations

import json
import torch
import numpy as np
from typing import Optional, Any
from pathlib import Path
from tqdm.auto import tqdm

from src.run.config import RunConfig
from src.run.utils import log_line
from src.run.logger import get_tqdm_kwargs
from src.run.distributed import get_raw_model, is_main_process, barrier


# --------------------------------------------------------------------------- #
# Task-specific accuracy                                                       #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def eval_task_accuracy(
    model: torch.nn.Module,
    config: RunConfig,
    split: str = "test",
    max_examples: Optional[int] = None,
) -> dict[str, float]:
    """
    Evaluate task-specific accuracy.

    Generates model responses and computes exact match or LLM-judge accuracy.

    Returns:
        Dict with accuracy metrics.
    """
    logger = config.logger
    loader = config.loaders.get(split)
    if loader is None:
        logger.warning(f"No {split} loader available for task accuracy eval")
        return {}

    raw_model = get_raw_model(model)
    device = config.device
    task = config.task

    logger.info(f"---- Evaluating task accuracy ({task}, {split}) ----")

    correct = 0
    total = 0
    results = []

    for batch_idx, batch in enumerate(tqdm(
        loader, **get_tqdm_kwargs(logger, desc=f"Eval {task}", ncols=100)
    )):
        if max_examples and total >= max_examples:
            break

        student_ids = batch["student_input_ids"].to(device)
        student_mask = batch["student_attention_mask"].to(device)

        # Generate
        generated = raw_model.generate(
            input_ids=student_ids,
            attention_mask=student_mask,
            max_new_tokens=config.max_gen_len,
            temperature=0.0,  # greedy for evaluation
            do_sample=False,
        )

        # Decode
        prompt_len = student_ids.shape[1]
        for i, (gen, example) in enumerate(zip(generated, batch["examples"])):
            response_tokens = gen[prompt_len:]
            response = raw_model.tokenizer.decode(response_tokens, skip_special_tokens=True)

            # Check accuracy based on task type
            is_correct = _check_accuracy(task, response, example.answer)
            correct += int(is_correct)
            total += 1

            results.append({
                "query": example.query,
                "answer": example.answer,
                "response": response,
                "correct": is_correct,
            })

    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Task accuracy ({task}, {split}): {accuracy:.4f} ({correct}/{total})")

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results,
    }


def _check_accuracy(task: str, response: str, answer: str) -> bool:
    """
    Check if a response is correct for a given task.

    Science Q&A: exact match on answer choice letter
    Tool Use: exact match on tool call
    Medical / Knowledge: requires LLM judge (simplified here to string match)
    """
    response_clean = response.strip().lower()
    answer_clean = answer.strip().lower()

    if task == "science_qa":
        # Extract the answer choice (A, B, C, D, etc.)
        for char in response_clean:
            if char.isalpha() and char.upper() in "ABCDEFGH":
                return char == answer_clean[0] if answer_clean else False
        return response_clean == answer_clean

    elif task == "tool_use":
        # Check if the tool call matches
        return answer_clean in response_clean

    elif task in ("medical", "knowledge"):
        # Simplified string matching — full reproduction should use LLM judge
        return answer_clean in response_clean or response_clean in answer_clean

    else:
        return response_clean == answer_clean


# --------------------------------------------------------------------------- #
# Prior capabilities evaluation (via lm-eval-harness)                          #
# --------------------------------------------------------------------------- #

def eval_prior_capabilities(
    model: torch.nn.Module,
    config: RunConfig,
    benchmarks: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Evaluate prior capabilities using lm-eval-harness.

    Benchmarks from the paper:
    - HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande, HumanEval

    Returns:
        Dict of {benchmark_name: score}.
    """
    logger = config.logger

    if benchmarks is None:
        benchmarks = config.eval_benchmarks

    logger.info(f"---- Evaluating prior capabilities: {benchmarks} ----")

    try:
        import lm_eval

        raw_model = get_raw_model(model)

        # Create lm-eval model wrapper
        lm = lm_eval.models.huggingface.HFLM(
            pretrained=raw_model.model,
            tokenizer=raw_model.tokenizer,
            batch_size=config.batch_size,
        )

        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=benchmarks,
            batch_size=config.batch_size,
        )

        scores = {}
        for task_name, task_results in results.get("results", {}).items():
            # Extract the primary metric for each benchmark
            if "acc_norm,none" in task_results:
                scores[task_name] = task_results["acc_norm,none"]
            elif "acc,none" in task_results:
                scores[task_name] = task_results["acc,none"]
            elif "exact_match,none" in task_results:
                scores[task_name] = task_results["exact_match,none"]
            else:
                # Take the first available metric
                for k, v in task_results.items():
                    if isinstance(v, (int, float)):
                        scores[task_name] = v
                        break

        # Compute aggregate
        if scores:
            scores["average"] = np.mean(list(scores.values()))

        for name, score in scores.items():
            logger.info(f"  {name}: {score:.4f}")

        return scores

    except ImportError:
        logger.warning("lm-eval not installed. Skipping prior capabilities eval.")
        logger.warning("Install with: pip install lm-eval")
        return {}


# --------------------------------------------------------------------------- #
# Pass@k evaluation                                                           #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def eval_pass_at_k(
    model: torch.nn.Module,
    config: RunConfig,
    k_values: list[int] = [1, 4, 16, 64, 128],
    num_examples: int = 100,
    split: str = "test",
) -> dict[str, float]:
    """
    Evaluate pass@k — whether any of k samples is correct.

    Uses temperature=1.0 with nucleus sampling (top_p=0.95) as per the paper.
    """
    logger = config.logger
    loader = config.loaders.get(split)
    if loader is None:
        return {}

    raw_model = get_raw_model(model)
    device = config.device
    task = config.task
    max_k = max(k_values)

    logger.info(f"---- Evaluating pass@k ({task}, k={k_values}) ----")

    pass_counts = {k: 0 for k in k_values}
    total = 0

    for batch in tqdm(loader, **get_tqdm_kwargs(logger, desc=f"Pass@k {task}", ncols=100)):
        if total >= num_examples:
            break

        for example in batch["examples"]:
            if total >= num_examples:
                break

            enc = raw_model.tokenizer(
                example.query,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_gen_len,
            ).to(device)

            # Generate max_k samples
            any_correct_at = [False] * (max_k + 1)  # index i = any correct in first i

            for sample_idx in range(max_k):
                generated = raw_model.generate(
                    input_ids=enc.input_ids,
                    attention_mask=enc.attention_mask,
                    max_new_tokens=512,
                    temperature=1.0,
                    top_p=0.95,
                    do_sample=True,
                )

                prompt_len = enc.input_ids.shape[1]
                response = raw_model.tokenizer.decode(
                    generated[0, prompt_len:], skip_special_tokens=True
                )

                is_correct = _check_accuracy(task, response, example.answer)
                if is_correct:
                    for j in range(sample_idx + 1, max_k + 1):
                        any_correct_at[j] = True

            for k in k_values:
                if any_correct_at[k]:
                    pass_counts[k] += 1

            total += 1

    results = {}
    for k in k_values:
        results[f"pass@{k}"] = pass_counts[k] / total if total > 0 else 0.0
        logger.info(f"  pass@{k}: {results[f'pass@{k}']:.4f}")

    return results


# --------------------------------------------------------------------------- #
# Main eval dispatcher                                                         #
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def do_eval(
    stage: dict,
    model: torch.nn.Module,
    config: RunConfig,
    log: Optional[dict[str, Any]] = None,
) -> None:
    """
    Run all configured evaluations and log results.

    Mirrors the ICML codebase do_eval pattern.
    """
    logger = config.logger
    res_dir = config.res_dir
    log_fp = res_dir / "stats.jsonl"

    logger.info("---- Begin Evaluation ----")

    # 1. Task accuracy
    task_results = eval_task_accuracy(model, config)

    # 2. Prior capabilities (if benchmarks configured)
    prior_results = {}
    if config.eval_benchmarks:
        prior_results = eval_prior_capabilities(model, config)

    # 3. Log results
    if is_main_process():
        entry = {
            "stage": stage,
            "task": config.task,
            "method": config.method,
            "task_accuracy": task_results.get("accuracy"),
            "task_correct": task_results.get("correct"),
            "task_total": task_results.get("total"),
            "prior_capabilities": prior_results,
        }

        if log:
            entry.update(log)

        # Don't save individual result objects in the JSONL
        entry.pop("results", None)
        if "results" in task_results:
            del task_results["results"]

        log_line(entry, log_fp)

    barrier()

