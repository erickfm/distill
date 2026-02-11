"""
Data loading for SDFT experiments.

Loads JSONL datasets with (query, demonstration, answer) tuples and
provides batched iteration with tokenization.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

@dataclass
class SDFTExample:
    """A single example for SDFT training / evaluation."""
    query: str
    answer: str
    demonstration: str  # full expert demonstration for teacher context
    extra: dict  # any task-specific fields (choices, api_spec, etc.)


class SDFTDataset(Dataset):
    """Dataset that loads JSONL files with SDFT examples."""

    def __init__(
        self,
        data_path: str | Path,
        max_examples: Optional[int] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.examples: List[SDFTExample] = []

        with open(self.data_path, "r") as f:
            for line in f:
                item = json.loads(line.strip())
                example = SDFTExample(
                    query=item.get("query", item.get("question", "")),
                    answer=item.get("answer", item.get("output", "")),
                    demonstration=item.get("demonstration", item.get("answer", "")),
                    extra={
                        k: v
                        for k, v in item.items()
                        if k not in ("query", "question", "answer", "output", "demonstration")
                    },
                )
                self.examples.append(example)

        if max_examples is not None and max_examples < len(self.examples):
            self.examples = self.examples[:max_examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> SDFTExample:
        return self.examples[idx]


# --------------------------------------------------------------------------- #
# Prompt formatting                                                            #
# --------------------------------------------------------------------------- #

# Template for student context (query only)
STUDENT_TEMPLATE = "{query}"

# Template for teacher context (query + demonstration)
TEACHER_TEMPLATE = """{demonstration}

{query}"""


def format_student_prompt(
    query: str,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
) -> str:
    """Format query into a student prompt using the model's chat template."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def format_teacher_prompt(
    query: str,
    demonstration: str,
    tokenizer: AutoTokenizer,
    system_prompt: Optional[str] = None,
) -> str:
    """Format query + demonstration into a teacher prompt using the model's chat template."""
    # Teacher sees the demonstration in context before answering
    teacher_context = f"Here is an expert demonstration for reference:\n\n{demonstration}\n\nNow answer the following:"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": f"{teacher_context}\n\n{query}"})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


# --------------------------------------------------------------------------- #
# Collation                                                                    #
# --------------------------------------------------------------------------- #

class SDFTCollator:
    """
    Collates SDFT examples into batched tensors.

    For SDFT training, each batch element needs:
    - student_input_ids: tokenized student prompt
    - teacher_input_ids: tokenized teacher prompt (with demonstration)
    - answer: the ground-truth answer (for evaluation)
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_prompt = system_prompt

    def __call__(self, examples: List[SDFTExample]) -> dict:
        student_prompts = []
        teacher_prompts = []
        answers = []

        for ex in examples:
            student_prompts.append(
                format_student_prompt(ex.query, self.tokenizer, self.system_prompt)
            )
            teacher_prompts.append(
                format_teacher_prompt(
                    ex.query, ex.demonstration, self.tokenizer, self.system_prompt
                )
            )
            answers.append(ex.answer)

        student_enc = self.tokenizer(
            student_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        teacher_enc = self.tokenizer(
            teacher_prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "student_input_ids": student_enc.input_ids,
            "student_attention_mask": student_enc.attention_mask,
            "teacher_input_ids": teacher_enc.input_ids,
            "teacher_attention_mask": teacher_enc.attention_mask,
            "answers": answers,
            "examples": examples,
        }


# --------------------------------------------------------------------------- #
# DataLoader factory                                                           #
# --------------------------------------------------------------------------- #

def make_loaders(
    data_dir: str | Path,
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_length: int = 2048,
    max_examples: Optional[int] = None,
    num_workers: int = 0,
    system_prompt: Optional[str] = None,
    seed: int = 42,
) -> dict[str, TorchDataLoader]:
    """
    Create data loaders for train / val / test splits.

    Args:
        data_dir: Directory containing {train,val,test}.jsonl
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        max_examples: Cap on number of examples per split (for debugging)
        num_workers: DataLoader workers
        system_prompt: Optional system prompt for formatting
        seed: Random seed for shuffling

    Returns:
        Dict of {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    data_dir = Path(data_dir)
    collator = SDFTCollator(tokenizer, max_length, system_prompt)

    loaders = {}
    for split in ["train", "val", "test"]:
        split_path = data_dir / f"{split}.jsonl"
        if not split_path.exists():
            continue

        dataset = SDFTDataset(split_path, max_examples=max_examples)

        generator = torch.Generator()
        generator.manual_seed(seed)

        loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
            num_workers=num_workers,
            generator=generator if split == "train" else None,
            drop_last=(split == "train"),
        )
        loaders[split] = loader

    return loaders

