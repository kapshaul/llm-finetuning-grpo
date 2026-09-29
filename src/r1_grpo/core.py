"""Group Relative Policy Optimization (GRPO) math.

Implements the objective of DeepSeek-R1 (arXiv:2501.12948v1, Equations 1-3) for a batch
of ``B`` prompts with ``G`` sampled completions each, padded to ``T`` completion tokens.

Conventions that the paper does not pin down and that this module makes explicit:

* Advantages are ``(r_i - mean(r)) / std(r)`` within each group using the *population*
  standard deviation (``correction=0``). ``eps`` only guards the division.
* A group whose rewards are all identical carries no learning signal and receives an
  advantage of exactly zero.
* ``reduction="sequence"`` (default) follows Equations 1-2 literally: the likelihood of a
  completion is the product of its token probabilities, so masked token log-probabilities
  are summed *before* forming the likelihood ratio and the KL estimator
  ``pi_ref/pi - log(pi_ref/pi) - 1``. The clipped surrogate and KL are per completion and
  averaged over the ``B * G`` completions.
* ``reduction="token"`` is a *practical variant* taken from the GRPO formulation in
  DeepSeekMath (arXiv:2402.03300): per-token ratios, clipping and KL, averaged over the
  tokens of each completion and then over completions. It is not the R1 equation.

* The KL estimator ``exp(d) - d - 1`` (``d = log pi_ref - log pi``) is computed as
  ``expm1(d) - d`` so it stays non-negative near ``d = 0``. It is an unbiased estimate of
  ``KL(pi || pi_ref)`` only when the completions are sampled from the current policy
  ``pi`` (the first update epoch on a rollout). On later epochs that reuse old-policy
  rollouts it is evaluated on samples from ``pi_old`` and is a biased (still non-negative)
  penalty, exactly as in the paper's formulation. With ``beta == 0`` the KL term is not
  computed at all, reference log-probabilities are ignored and the ``kl`` metric is 0.
* Nothing is clamped to hide numerical problems: an overflowing likelihood ratio or KL
  term raises ``FloatingPointError`` even when clipping would have produced a finite loss.

Old-policy log-probabilities must come from the frozen rollout policy and stay fixed for
all update epochs on that rollout; reference log-probabilities come from the reference
model frozen at the start of the stage. Both, and the advantages, are detached here.
"""

from __future__ import annotations

import math

import torch

REDUCTIONS = ("sequence", "token")

__all__ = ["REDUCTIONS", "group_advantages", "grpo_loss"]


def _check_positive_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return float(value)


def group_advantages(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return group-normalized advantages for rewards of shape ``[B, G]``.

    Each row is one group of ``G`` completions for the same prompt and is normalized
    independently of every other row.
    """
    if not isinstance(rewards, torch.Tensor):
        raise TypeError("rewards must be a torch.Tensor")
    if rewards.dim() != 2 or rewards.shape[0] == 0 or rewards.shape[1] == 0:
        raise ValueError(f"rewards must have a non-empty shape [B, G], got {tuple(rewards.shape)}")
    eps = _check_positive_float(eps, "eps")
    values = rewards.detach()
    if not torch.is_floating_point(values):
        values = values.to(torch.get_default_dtype())
    if not bool(torch.isfinite(values).all()):
        raise ValueError("rewards contain non-finite values")

    mean = values.mean(dim=1, keepdim=True)
    std = torch.std(values, dim=1, keepdim=True, correction=0)
    advantages = (values - mean) / (std + eps)
    tied = (values == values[:, :1]).all(dim=1, keepdim=True)
    return torch.where(tied, torch.zeros_like(advantages), advantages)


def _completion_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        return mask
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("completion_mask must contain only 0/1 (or boolean) values")
    return mask != 0


def _kl_estimator(log_ref_minus_log_policy: torch.Tensor) -> torch.Tensor:
    """Non-negative estimator ``exp(d) - d - 1`` with ``d = log pi_ref - log pi``.

    Written as ``expm1(d) - d``: ``expm1`` is accurate near zero and ``expm1(d) >= d`` holds
    after rounding, so small ``d`` cannot produce a negative value by cancellation.
    """
    d = log_ref_minus_log_policy
    return torch.expm1(d) - d


def grpo_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.04,
    reduction: str = "sequence",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the GRPO loss (negated objective) and scalar diagnostics.

    ``log_probs``, ``old_log_probs``, ``ref_log_probs`` and ``completion_mask`` have shape
    ``[B, G, T]`` and hold per-token log-probabilities of the sampled completion tokens;
    ``advantages`` has shape ``[B, G]``. Gradients flow only through ``log_probs`` and
    never into padded positions.
    """
    tensors = {
        "log_probs": log_probs,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "completion_mask": completion_mask,
        "advantages": advantages,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if log_probs.dim() != 3 or 0 in log_probs.shape:
        raise ValueError(f"log_probs must have a non-empty shape [B, G, T], got {tuple(log_probs.shape)}")
    if not torch.is_floating_point(log_probs):
        raise TypeError("log_probs must be a floating point tensor")
    for name in ("old_log_probs", "ref_log_probs", "completion_mask"):
        if tensors[name].shape != log_probs.shape:
            raise ValueError(
                f"{name} shape {tuple(tensors[name].shape)} does not match log_probs shape {tuple(log_probs.shape)}"
            )
    if advantages.shape != log_probs.shape[:2]:
        raise ValueError(f"advantages shape {tuple(advantages.shape)} must be [B, G] = {tuple(log_probs.shape[:2])}")
    clip_epsilon = _check_positive_float(clip_epsilon, "clip_epsilon")
    if clip_epsilon >= 1:
        raise ValueError(f"clip_epsilon must be < 1, got {clip_epsilon}")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)):
        raise TypeError(f"beta must be a number, got {type(beta).__name__}")
    if not math.isfinite(beta) or beta < 0:
        raise ValueError(f"beta must be finite and non-negative, got {beta!r}")
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")

    mask = _completion_mask(completion_mask.to(log_probs.device))
    lengths = mask.sum(dim=-1)
    if bool((lengths == 0).any()):
        raise ValueError("every completion must contain at least one unmasked token")

    dtype = log_probs.dtype
    old = old_log_probs.detach().to(dtype)
    ref = ref_log_probs.detach().to(dtype)
    adv = advantages.detach().to(dtype)
    for name, tensor in (("log_probs", log_probs.detach()), ("old_log_probs", old), ("ref_log_probs", ref)):
        if not bool(torch.isfinite(tensor[mask]).all()):
            raise ValueError(f"{name} contains non-finite values at completion positions")
    if not bool(torch.isfinite(adv).all()):
        raise ValueError("advantages contain non-finite values")

    # torch.where (not multiplication) keeps padded NaN/inf out of both values and gradients.
    zero = log_probs.new_zeros(())
    policy = torch.where(mask, log_probs, zero)
    old = torch.where(mask, old, zero)
    ref = torch.where(mask, ref, zero)
    low, high = 1.0 - clip_epsilon, 1.0 + clip_epsilon

    hint = (
        "Check that old/reference log-probabilities match the sampling policy, or use "
        "reduction='token' for long completions."
    )
    if reduction == "sequence":
        ratio = torch.exp(policy.sum(dim=-1) - old.sum(dim=-1))
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, low, high) * adv
        surrogate = torch.minimum(unclipped, clipped)
        kl = _kl_estimator(ref.sum(dim=-1) - policy.sum(dim=-1)) if beta > 0 else None
        per_sequence = surrogate - beta * kl if kl is not None else surrogate
        with torch.no_grad():
            ratios = ratio.detach()
            kl_values = kl.detach() if kl is not None else None
            surrogate_mean = surrogate.detach().mean()
            kl_mean = kl_values.mean() if kl_values is not None else None
            clip_fraction = (clipped < unclipped).to(dtype).mean()
    else:
        adv_tokens = adv.unsqueeze(-1)
        ratio = torch.exp(policy - old)
        unclipped = ratio * adv_tokens
        clipped = torch.clamp(ratio, low, high) * adv_tokens
        surrogate = torch.minimum(unclipped, clipped)
        kl = _kl_estimator(ref - policy) if beta > 0 else None
        per_token = surrogate - beta * kl if kl is not None else surrogate
        token_counts = lengths.to(dtype)
        per_sequence = torch.where(mask, per_token, zero).sum(dim=-1) / token_counts
        with torch.no_grad():
            ratios = ratio.detach()[mask]
            kl_values = kl.detach()[mask] if kl is not None else None
            surrogate_mean = (torch.where(mask, surrogate.detach(), zero).sum(dim=-1) / token_counts).mean()
            kl_mean = (
                (torch.where(mask, kl.detach(), zero).sum(dim=-1) / token_counts).mean() if kl is not None else None
            )
            clip_fraction = ((clipped < unclipped) & mask).sum().to(dtype) / mask.sum().to(dtype)

    # Clipping can turn an infinite ratio into a finite loss; that is still a numerical
    # failure (the update is based on garbage), so check the ratio and KL themselves.
    if not bool(torch.isfinite(ratios).all()):
        raise FloatingPointError(
            f"GRPO likelihood ratio is not finite (max {float(ratios.max())}); "
            f"policy and old-policy log-probabilities diverged. {hint}"
        )
    if kl_values is not None and not bool(torch.isfinite(kl_values).all()):
        raise FloatingPointError(f"GRPO KL term is not finite (max {float(kl_values.max())}). {hint}")
    loss = -per_sequence.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"GRPO loss is not finite ({float(loss.detach())}). {hint}")
    metrics = {
        "loss": float(loss.detach()),
        "surrogate": float(surrogate_mean),
        "kl": float(kl_mean) if kl_mean is not None else 0.0,
        "clip_fraction": float(clip_fraction),
        "ratio_mean": float(ratios.mean()),
        "ratio_min": float(ratios.min()),
        "ratio_max": float(ratios.max()),
        "advantage_mean": float(adv.mean()),
        "completion_tokens_mean": float(lengths.to(dtype).mean()),
    }
    bad = sorted(name for name, value in metrics.items() if not math.isfinite(value))
    if bad:
        raise FloatingPointError(f"GRPO diagnostics are not finite: {', '.join(bad)}. {hint}")
    return loss, metrics
