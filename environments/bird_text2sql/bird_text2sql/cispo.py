from __future__ import annotations

import torch


def _safe_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values[mask].sum() / torch.clamp_min(mask.sum(), 1)


def cispo_loss_fn(
    inputs,
    *,
    clip_low_threshold: float = 0.8,
    clip_high_threshold: float = 1.2,
):
    """Tinker-compatible token-level CISPO for Prime-RL custom losses.

    The ratio is clipped and detached, while gradients continue through the
    target-policy log probability for every selected token. Thresholds are
    absolute ratio bounds, matching the ReViSQL call into Tinker's CISPO loss.
    """

    from prime_rl.trainer.rl.loss import LossOutputs

    if clip_low_threshold < 0 or clip_high_threshold < clip_low_threshold:
        raise ValueError("CISPO requires 0 <= clip_low_threshold <= clip_high_threshold")
    log_ratio = inputs.trainer_logprobs - inputs.inference_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(
        ratio, min=clip_low_threshold, max=clip_high_threshold
    ).detach()
    per_token_loss = -(
        inputs.loss_mask * clipped_ratio * inputs.advantages * inputs.trainer_logprobs
    )
    if inputs.loss_weights is not None:
        per_token_loss = per_token_loss * inputs.loss_weights
    loss = per_token_loss.sum()
    mismatch_kl = ratio - log_ratio - 1
    return LossOutputs(
        loss=loss,
        metrics={
            "cispo/importance_ratio": _safe_mean(ratio.detach(), inputs.loss_mask),
            "cispo/clipped_ratio": _safe_mean(clipped_ratio, inputs.loss_mask),
            "cispo/clipped_low": _safe_mean(ratio < clip_low_threshold, inputs.loss_mask),
            "cispo/clipped_high": _safe_mean(ratio > clip_high_threshold, inputs.loss_mask),
            "cispo/mismatch_kl": _safe_mean(mismatch_kl.detach(), inputs.loss_mask),
        },
    )
