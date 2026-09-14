from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bird_text2sql.cispo import cispo_loss_fn  # noqa: E402

try:
    from prime_rl.trainer.rl.loss import LossInputs
except ImportError:
    pytest.skip("Prime-RL is required for CISPO integration tests", allow_module_level=True)


def make_inputs(target, old, advantages):
    return LossInputs(
        trainer_logprobs=target,
        inference_logprobs=old,
        ref_logprobs=None,
        advantages=advantages,
        loss_mask=torch.ones_like(target, dtype=torch.bool),
    )


def test_cispo_matches_detached_clipped_ratio_formula() -> None:
    target = torch.tensor([0.0, -3.0], requires_grad=True)
    old = torch.tensor([-2.0, 0.0])
    advantages = torch.tensor([2.0, -1.0])
    output = cispo_loss_fn(
        make_inputs(target, old, advantages),
        clip_low_threshold=0.8,
        clip_high_threshold=1.2,
    )
    expected_ratio = torch.exp(target.detach() - old).clamp(0.8, 1.2)
    expected = -(expected_ratio * advantages * target).sum()
    assert torch.allclose(output.loss, expected)

    output.loss.backward()
    assert torch.allclose(target.grad, -(expected_ratio * advantages))
    assert output.metrics["cispo/clipped_high"] == 0.5
    assert output.metrics["cispo/clipped_low"] == 0.5


def test_cispo_respects_mask_and_weights() -> None:
    target = torch.tensor([-1.0, -1.0], requires_grad=True)
    inputs = make_inputs(target, target.detach().clone(), torch.ones(2))
    inputs.loss_mask[1] = False
    inputs.loss_weights = torch.tensor([0.25, 9.0])
    output = cispo_loss_fn(inputs)
    output.loss.backward()
    assert torch.allclose(target.grad, torch.tensor([-0.25, 0.0]))
