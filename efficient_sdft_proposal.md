# Adaptive Screening for Efficient Self-Distillation Fine-Tuning

## Problem

SDFT (Shenfeld et al., 2026) achieves strong continual learning by generating on-policy rollouts and minimizing reverse KL divergence between a student (base model) and teacher (demo-conditioned model). But this costs ~2.5× FLOPs and ~4× wall-clock time versus SFT, because every training example requires full autoregressive generation from the student plus a teacher forward pass on that rollout — even when student and teacher already agree.

Humans don't study this way. You skim what you know and concentrate effort where your predictions fail. The core proposal: make SDFT do the same.

## Method

### Phase 1: Cheap Disagreement Screening

For each example (x, c) in a batch, run two forward passes (no generation):

1. **Teacher forward pass** on the demonstration or a cached teacher response, yielding token-level log-probs log π(y_t | y_<t, x, c).
2. **Student forward pass** on the same token sequence, yielding log π_θ(y_t | y_<t, x).

Compute a per-example disagreement score:

$$\hat{\Delta}(x) = \frac{1}{T} \sum_t \left[ \log \pi_\theta(y_t | y_{<t}, x) - \log \pi(y_t | y_{<t}, x, c) \right]$$

This is cheap — two forward passes with no autoregressive decoding. It approximates the teacher-student disagreement without generating a student rollout.

### Phase 2: Selective Rollout Generation

Partition the batch by disagreement:

- **High disagreement** (|Δ̂| > threshold): full SDFT — generate on-policy rollout from student, compute KL gradient, update weights.
- **Low disagreement** (|Δ̂| ≤ threshold): skip rollout generation. No gradient update for this example this step.

Early in training, most examples exceed the threshold — the model hasn't learned much yet. Late in training, most examples are skipped. Generation cost scales with how much the model still needs to learn, not with dataset size.

### Phase 3: Self-Calibrating Audit

The screening estimate Δ̂ is computed on a fixed sequence (the teacher's output), not on the student's own generation. This is the key weakness: the whole point of on-policy learning is that the student reveals errors when generating its own rollouts that wouldn't appear when evaluated on someone else's output.

To catch these cases, randomly sample a fraction p of "low-disagreement" examples and run full rollouts anyway as calibration audits. Track the **discrepancy rate**: how often a screened-as-easy example turns out to have high on-policy disagreement.

- If discrepancy rate is high → screening is unreliable → increase p (more audits).
- If discrepancy rate is low → screening is trustworthy → decrease p (fewer audits).

Update rule for audit fraction:

$$p_{t+1} = \text{clip}\left(p_t + \alpha \cdot (d_t - d_{\text{target}}),\ p_{\min},\ p_{\max}\right)$$

where d_t is the observed discrepancy rate at step t and d_target is the acceptable miss rate (e.g., 0.05). This is a single hyperparameter (d_target) with a principled meaning: "how often am I willing to miss a case where I actually needed to practice."

The audit fraction p self-calibrates over training. Early on, when the model is changing fast between updates, screening estimates go stale quickly and discrepancy is high, so p stays large. Late in training, the model stabilizes, screening becomes reliable, and p shrinks. No schedule needed.

### Optional: Consolidation Sweeps

Instead of screening and generating within every batch, accumulate a full disagreement map over an epoch (cheap — just forward passes, no generation, no weight updates). Then run a targeted fine-tuning pass using only the high-disagreement subset with full SDFT rollouts.

This amortizes the screening cost across the entire dataset before committing to expensive generation. The disagreement map from epoch N is stale by epoch N+1, but the SDFT paper already shows that a single trajectory per prompt suffices, so the signal doesn't need to be perfectly fresh.

## Expected Efficiency Gains

If X% of examples have low disagreement by mid-training, generation cost drops proportionally. Based on the convergence curves in the SDFT paper (Figure 3), performance on new tasks saturates well before training ends — suggesting that a large fraction of examples become "easy" early. A conservative estimate: 50-70% of rollouts could be skipped in the second half of training, bringing effective wall-clock cost to ~2-2.5× SFT (down from 4×).

The audit mechanism adds negligible cost — it's a small fraction of the already-skipped examples, and it decreases over time.

## Key Risks and Open Questions

**Screening validity.** The off-policy disagreement estimate Δ̂ may systematically miss failure modes that only appear under on-policy generation. The self-calibrating audit mitigates this but cannot eliminate it. If certain error types are invisible to screening (e.g., the student assigns high probability to teacher tokens but would generate something completely different), the audit would need to catch them by chance.

**Threshold sensitivity.** The disagreement threshold separating "screen" from "generate" is a hyperparameter. Setting it too high wastes compute on easy examples. Setting it too low misses learning opportunities. An adaptive threshold (e.g., target a fixed fraction of the batch for full rollouts, decreasing over training) may be more robust than a fixed value.

**Interaction with EMA teacher.** The SDFT paper uses an EMA of student weights as the teacher. If weight updates are sparse (only from high-disagreement examples), the EMA evolves more slowly, which changes the teacher's behavior. This could be beneficial (more stable teacher) or harmful (teacher falls behind the student's actual capabilities). Needs empirical investigation.

## Connection to Human Learning

The screening phase is analogous to skimming — rapid assessment of what you already know. Full rollout generation is analogous to active practice — working through problems yourself to expose hidden gaps. The self-calibrating audit is analogous to periodic self-quizzing on material you think you've mastered. The consolidation sweep variant mirrors spaced review — surveying everything, then targeting weak spots.

The key human insight being borrowed: **generation is expensive and should be reserved for where it matters**. Most of learning is knowing what you don't know.
