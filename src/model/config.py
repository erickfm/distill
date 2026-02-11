"""
Model configuration for SDFT reproduction.

Wraps HuggingFace causal LMs with student/teacher context modes.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the SDFT model setup."""

    # Base model identifier (HuggingFace hub or local path)
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # EMA teacher
    ema_decay: float = 0.999

    # Generation
    max_gen_len: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0

    # KL gradient estimator: "analytic", "token", "rao_blackwell"
    kl_estimator: str = "analytic"

    # Precision
    dtype: str = "bfloat16"

    # Whether to use vLLM for fast generation
    use_vllm: bool = True

