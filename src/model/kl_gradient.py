"""
KL divergence gradient estimators for SDFT.

Implements three estimators from the paper (Appendix A.1):
1. Token-level (partial) estimator — biased but cheap
2. Full analytic per-token estimator — the paper's default
3. Rao-Blackwellized estimator — unbiased, lower variance, more expensive
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def analytic_kl_gradient_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Full analytic per-token KL gradient estimator (Eq. A.1 in the paper).

    Computes the KL divergence analytically at each timestep by
    marginalizing over the vocabulary:

        g_analytic = Σ_t Σ_v log(π_θ(v|y<t,x) / π(v|y<t,x,c)) · ∇log π_θ(v|y<t,x)

    This is equivalent to the standard KL divergence loss between the
    student and teacher distributions at each token position, which can
    be backpropagated through normally.

    Args:
        student_logits: (B, T, V) raw logits from the student model
        teacher_logits: (B, T, V) raw logits from the teacher model (detached)
        mask: (B, T) optional mask (1 = valid token, 0 = padding)

    Returns:
        Scalar KL divergence loss (mean over valid tokens).
    """
    # Compute log-probs
    student_log_probs = F.log_softmax(student_logits, dim=-1)  # (B, T, V)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)  # (B, T, V)

    # KL(student || teacher) = Σ_v p_student(v) * [log p_student(v) - log p_teacher(v)]
    # = Σ_v exp(log_student) * (log_student - log_teacher)
    kl_per_token = F.kl_div(
        teacher_log_probs,  # input (log-probs)
        student_log_probs,  # target (log-probs)
        log_target=True,
        reduction="none",
    ).sum(dim=-1)  # (B, T)

    if mask is not None:
        kl_per_token = kl_per_token * mask
        return kl_per_token.sum() / mask.sum().clamp(min=1)
    else:
        return kl_per_token.mean()


def token_level_kl_loss(
    student_log_probs_sampled: torch.Tensor,
    teacher_log_probs_sampled: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Token-level (partial) KL gradient estimator.

    A widely used approximation that decomposes the KL into token-level terms:

        g_token = Σ_t log(π_θ(y_t|y<t,x) / π(y_t|y<t,x,c)) · ∇log π_θ(y_t|y<t,x)

    where y_t are the actually sampled tokens. This is biased (ignores
    the effect of early tokens on future distributions).

    Args:
        student_log_probs_sampled: (B, T) log p_student for sampled tokens
        teacher_log_probs_sampled: (B, T) log p_teacher for sampled tokens
        mask: (B, T) optional mask

    Returns:
        Scalar loss for token-level REINFORCE-style gradient.
    """
    # log-ratio: log(π_θ / π_teacher) for the sampled tokens
    log_ratio = student_log_probs_sampled - teacher_log_probs_sampled.detach()

    # REINFORCE-style: log_ratio * ∇ log π_θ
    # Since student_log_probs_sampled already provides the ∇ log π_θ path,
    # we need: loss = Σ_t log_ratio.detach() * student_log_probs_sampled
    loss = log_ratio.detach() * student_log_probs_sampled

    if mask is not None:
        loss = loss * mask
        return -loss.sum() / mask.sum().clamp(min=1)
    else:
        return -loss.mean()


def rao_blackwell_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_log_probs_sampled: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Rao-Blackwellized KL gradient estimator.

    Analytically integrates over next-token distributions while
    retaining Monte-Carlo sampling over prefixes. Unbiased with
    provably lower variance than standard MC estimators.

        g_rb = Σ_t [Σ_v log(π_θ(v)/π(v)) · ∇log π_θ(v) + k_θ(y<t) · Σ_{i<t} ∇log π_θ(y_i)]

    where k_θ(y<t) is the stepwise KL term.

    Args:
        student_logits: (B, T, V) raw logits from student
        teacher_logits: (B, T, V) raw logits from teacher (detached)
        student_log_probs_sampled: (B, T) log p_student for sampled tokens
        mask: (B, T) optional mask

    Returns:
        Scalar loss.
    """
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach(), dim=-1)

    # Analytic KL at each position (same as analytic estimator)
    student_probs = student_log_probs.exp()
    kl_per_token = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1)  # (B, T)

    # Analytic part: standard KL loss (already handled by autograd)
    analytic_loss = analytic_kl_gradient_loss(student_logits, teacher_logits, mask)

    # Correction term: k_θ(y<t) * Σ_{i<t} ∇log π_θ(y_i)
    # This requires REINFORCE over the prefix
    cumsum_log_probs = torch.cumsum(student_log_probs_sampled, dim=-1)  # (B, T)
    # Shift so position t gets Σ_{i<t} log π_θ(y_i)
    prefix_log_probs = torch.zeros_like(cumsum_log_probs)
    prefix_log_probs[:, 1:] = cumsum_log_probs[:, :-1]

    correction = kl_per_token.detach() * prefix_log_probs

    if mask is not None:
        correction = correction * mask
        correction_loss = -correction.sum() / mask.sum().clamp(min=1)
    else:
        correction_loss = -correction.mean()

    return analytic_loss + correction_loss

