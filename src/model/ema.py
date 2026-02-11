"""
Exponential Moving Average (EMA) teacher model for SDFT.

The teacher is a copy of the student whose weights are updated via
EMA: ϕ ← α·θ + (1-α)·ϕ  after each optimizer step.
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
from typing import Optional


class EMAModel:
    """
    Maintains an exponential moving average of model parameters.
    
    Used as the teacher in SDFT: the EMA teacher provides stable
    supervision while tracking the student's learning progress.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
    ) -> None:
        """
        Args:
            model: The student model to track.
            decay: EMA decay rate α. Higher = slower tracking.
                   Paper default: 0.999.
        """
        self.decay = decay

        # Deep copy student weights for the teacher
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.shadow.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Update teacher weights: ϕ ← decay·ϕ + (1-decay)·θ
        
        Note: The paper writes ϕ ← α·θ + (1-α)·ϕ where α is the
        EMA rate. We follow the standard convention where decay
        controls how much of the old weights to keep.
        """
        for ema_param, model_param in zip(
            self.shadow.parameters(), model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )

    def forward(self, *args, **kwargs):
        """Forward pass through the EMA teacher model."""
        return self.shadow(*args, **kwargs)

    def get_token_log_probs(self, *args, **kwargs):
        """Get token log-probs from the EMA teacher."""
        return self.shadow.get_token_log_probs(*args, **kwargs)

    @property
    def model(self) -> nn.Module:
        return self.shadow

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.shadow.load_state_dict(state_dict)

