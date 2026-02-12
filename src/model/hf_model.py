"""
HuggingFace model wrapper for SDFT.

Provides a unified interface for loading models, computing log-probs,
and managing the EMA teacher.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModel(nn.Module):
    """
    Wrapper around a HuggingFace causal LM that provides:
    - Forward pass for log-prob computation
    - Greedy / sampled generation
    - EMA teacher weight management
    """

    def __init__(
        self,
        model_name: str,
        dtype: str = "bfloat16",
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        torch_dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Prefer flash_attention_2 if available, fall back to sdpa
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"

        # Load WITHOUT device_map so copy.deepcopy works correctly
        # (device_map adds accelerate dispatch hooks that break on deepcopy)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        if device:
            self.model = self.model.to(device)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model_name = model_name

    # ------------------------------------------------------------------ #
    # Gradient checkpointing                                               #
    # ------------------------------------------------------------------ #

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to trade compute for memory."""
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.model.gradient_checkpointing_disable()

    # ------------------------------------------------------------------ #
    # Forward / log-prob computation                                      #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns log-probs of shape (B, T, V) over the vocabulary.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # (B, T, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs

    def get_token_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get per-token log-probs for the *given* token sequence.
        Returns shape (B, T-1) — log p(token_t | tokens_<t).
        """
        log_probs = self.forward(input_ids, attention_mask)  # (B, T, V)

        # Shift: predict token t from position t-1
        shift_log_probs = log_probs[:, :-1, :]  # (B, T-1, V)
        shift_targets = input_ids[:, 1:]  # (B, T-1)

        # Gather the log-probs of the actual tokens
        token_log_probs = shift_log_probs.gather(
            dim=-1, index=shift_targets.unsqueeze(-1)
        ).squeeze(-1)  # (B, T-1)

        return token_log_probs

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive generation from the student model.
        Returns full sequence (prompt + generation).
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return outputs

    # ------------------------------------------------------------------ #
    # Utility                                                             #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

