import math

import pytest
import torch

from r1_grpo.core import group_advantages, grpo_loss


def _t(values, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=requires_grad)


def _close(actual, expected, atol=1e-7):
    torch.testing.assert_close(
        actual, _t(expected) if not isinstance(expected, torch.Tensor) else expected, atol=atol, rtol=0
    )


# ---------------------------------------------------------------------------
# group_advantages
# ---------------------------------------------------------------------------


def test_group_advantages_binary_rewards():
    _close(group_advantages(_t([[1.0, 0.0, 0.0, 1.0]])), [[1.0, -1.0, -1.0, 1.0]])


def test_group_advantages_uses_population_std():
    # mean 2, population std sqrt(2/3); the sample std (1.0) would give [1, -1, 0].
    scale = math.sqrt(3.0 / 2.0)
    _close(group_advantages(_t([[3.0, 1.0, 2.0]])), [[scale, -scale, 0.0]])


def test_tied_groups_get_exactly_zero_advantage():
    rewards = _t([[0.1, 0.1, 0.1], [1.0, 0.0, 1.0], [7.0, 7.0, 7.0]])
    advantages = group_advantages(rewards)
    assert torch.equal(advantages[0], torch.zeros(3, dtype=torch.float64))
    assert torch.equal(advantages[2], torch.zeros(3, dtype=torch.float64))
    assert advantages[1].abs().sum() > 0
    assert torch.equal(group_advantages(_t([[5.0], [2.0]])), torch.zeros(2, 1, dtype=torch.float64))


def test_groups_are_normalized_independently():
    base = _t([[1.0, 0.0, 0.0, 0.0], [2.0, 4.0, 6.0, 8.0]])
    other = base.clone()
    other[1] = other[1] * 10 + 3
    first = group_advantages(base)
    second = group_advantages(other)
    # Vectorized reductions may round differently across batch layouts, so allow a few ulps
    # (float64), far below any cross-group leakage (which would change values at O(1)).
    torch.testing.assert_close(first[0], second[0], atol=1e-12, rtol=0)
    torch.testing.assert_close(first[0], group_advantages(base[:1])[0], atol=1e-12, rtol=0)
    # The same affine transform inside one group leaves its normalized advantages unchanged.
    torch.testing.assert_close(first[1], second[1], atol=1e-7, rtol=0)


def test_group_advantages_validation():
    with pytest.raises(ValueError):
        group_advantages(_t([1.0, 2.0]))
    with pytest.raises(ValueError):
        group_advantages(_t([[1.0, float("nan")]]))
    with pytest.raises(ValueError):
        group_advantages(_t([[1.0, 2.0]]), eps=0.0)


# ---------------------------------------------------------------------------
# grpo_loss: analytical values
# ---------------------------------------------------------------------------


def _pair(log_probs, advantages, beta=0.0, reduction="sequence", ref=None):
    log_probs = _t(log_probs, requires_grad=True)
    old = torch.zeros_like(log_probs)
    ref = old.clone() if ref is None else _t(ref)
    mask = torch.ones_like(log_probs, dtype=torch.bool)
    loss, metrics = grpo_loss(log_probs, old, ref, mask, _t(advantages), beta=beta, reduction=reduction)
    loss.backward()
    return loss, metrics, log_probs.grad


def test_sequence_clipping_both_signs_is_flat():
    # seq0: ratio 1.5, A=+1 -> clipped to 1.2. seq1: ratio 0.5, A=-1 -> min(-0.5, -0.8) = -0.8.
    loss, metrics, grad = _pair([[[math.log(1.5), 0.0], [math.log(0.5), 0.0]]], [[1.0, -1.0]])
    assert loss.item() == pytest.approx(-(1.2 - 0.8) / 2)
    assert metrics["clip_fraction"] == pytest.approx(1.0)
    assert torch.equal(grad, torch.zeros_like(grad))


def test_sequence_pessimistic_side_is_not_clipped():
    # A=+1 with ratio 0.5 and A=-1 with ratio 1.5 keep the unclipped (smaller) term.
    loss, metrics, grad = _pair([[[math.log(0.5), 0.0], [math.log(1.5), 0.0]]], [[1.0, -1.0]])
    assert loss.item() == pytest.approx(-(0.5 - 1.5) / 2)
    assert metrics["clip_fraction"] == 0.0
    # d(-mean(r_i * A_i)) / d(token log-prob) = -r_i * A_i / N for every token of sequence i.
    _close(grad, [[[-0.25, -0.25], [0.75, 0.75]]])


def test_sequence_inside_trust_region():
    loss, _, grad = _pair([[[math.log(1.1), 0.0], [math.log(0.9), 0.0]]], [[1.0, -1.0]])
    assert loss.item() == pytest.approx(-(1.1 - 0.9) / 2)
    _close(grad, [[[-0.55, -0.55], [0.45, 0.45]]])


def test_sequence_and_token_reductions_differ():
    tokens = [[[math.log(1.5), math.log(1 / 1.5)]]]
    sequence_loss, _, _ = _pair(tokens, [[1.0]], reduction="sequence")
    token_loss, _, _ = _pair(tokens, [[1.0]], reduction="token")
    # Sequence ratio is exactly 1; token ratios are 1.5 -> clipped 1.2 and 2/3 (unclipped).
    assert sequence_loss.item() == pytest.approx(-1.0)
    assert token_loss.item() == pytest.approx(-(1.2 + 2 / 3) / 2)


def test_token_reduction_is_mean_per_sequence_then_mean():
    log_probs = _t([[[math.log(1.1), 0.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    mask = _t([[[1, 0, 0], [1, 1, 1]]])
    loss, _ = grpo_loss(
        log_probs,
        torch.zeros_like(log_probs),
        torch.zeros_like(log_probs),
        mask,
        _t([[1.0, 1.0]]),
        beta=0.0,
        reduction="token",
    )
    # Per-sequence means are 1.1 and 1.0; a flat token mean would give (1.1 + 3) / 4.
    assert loss.item() == pytest.approx(-(1.1 + 1.0) / 2)


def test_sequence_kl_matches_hand_computation():
    d = -0.5  # log pi_ref(o) - log pi(o) = -2.5 - (-2.0)
    loss, metrics, grad = _pair([[[-1.0, -1.0]]], [[0.0]], beta=0.04, ref=[[[-1.25, -1.25]]])
    expected_kl = math.exp(d) - d - 1
    assert metrics["kl"] == pytest.approx(expected_kl)
    assert loss.item() == pytest.approx(0.04 * expected_kl)
    expected_grad = 0.04 * (1 - math.exp(d))
    _close(grad, [[[expected_grad, expected_grad]]])


def test_token_kl_differs_from_sequence_kl():
    d = -0.25
    loss, metrics, _ = _pair([[[-1.0, -1.0]]], [[0.0]], beta=0.04, reduction="token", ref=[[[-1.25, -1.25]]])
    assert metrics["kl"] == pytest.approx(math.exp(d) - d - 1)
    assert loss.item() == pytest.approx(0.04 * (math.exp(d) - d - 1))


def test_kl_is_nonnegative():
    generator = torch.Generator().manual_seed(0)
    log_probs = -torch.rand(2, 3, 5, generator=generator, dtype=torch.float64)
    ref = -torch.rand(2, 3, 5, generator=generator, dtype=torch.float64)
    mask = torch.ones_like(log_probs)
    for reduction in ("sequence", "token"):
        _, metrics = grpo_loss(
            log_probs, log_probs, ref, mask, torch.zeros(2, 3, dtype=torch.float64), reduction=reduction
        )
        assert metrics["kl"] >= 0


def test_zero_beta_ignores_reference():
    log_probs = _t([[[-0.3, -0.7]]])
    mask = torch.ones_like(log_probs)
    losses = [
        grpo_loss(log_probs, log_probs - 0.1, _t([[[ref, ref]]]), mask, _t([[1.0]]), beta=0.0)[0].item()
        for ref in (-0.5, -50.0, -400.0)
    ]
    assert losses[0] == losses[1] == losses[2]


@pytest.mark.parametrize("reduction", ["sequence", "token"])
def test_zero_beta_skips_kl_even_if_it_would_overflow(reduction):
    # d = log pi_ref - log pi = 800 per token: exp(d) overflows even in float64;
    # term must not be computed at all.
    log_probs = _t([[[-800.0, -800.0]]], requires_grad=True)
    ref = torch.zeros_like(log_probs)
    loss, metrics = grpo_loss(
        log_probs, log_probs.detach(), ref, torch.ones_like(ref), _t([[1.0]]), beta=0.0, reduction=reduction
    )
    loss.backward()
    assert metrics["kl"] == 0.0
    assert loss.item() == pytest.approx(-1.0)
    assert bool(torch.isfinite(log_probs.grad).all())
    with pytest.raises(FloatingPointError, match="KL"):
        grpo_loss(log_probs, log_probs.detach(), ref, torch.ones_like(ref), _t([[1.0]]), beta=0.04, reduction=reduction)


@pytest.mark.parametrize("reduction", ["sequence", "token"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_near_zero_kl_is_nonnegative(reduction, dtype):
    # exp(d) - d - 1 computed naively can round below zero for tiny |d|; expm1(d) - d cannot.
    deltas = [s * 10.0**-k for k in range(3, 12) for s in (1.0, -1.0)]
    log_probs = torch.full((1, len(deltas), 1), -1.0, dtype=dtype)
    ref = log_probs + torch.tensor(deltas, dtype=dtype).view(1, -1, 1)
    advantages = torch.zeros(1, len(deltas), dtype=dtype)
    loss, metrics = grpo_loss(
        log_probs, log_probs, ref, torch.ones_like(log_probs), advantages, beta=1.0, reduction=reduction
    )
    assert loss.item() >= 0.0
    assert metrics["kl"] >= 0.0


# ---------------------------------------------------------------------------
# grpo_loss: gradients, masking and validation
# ---------------------------------------------------------------------------


def test_old_reference_and_advantages_receive_no_gradient():
    log_probs = _t([[[-0.2, -0.4], [-0.1, -0.3]]], requires_grad=True)
    old = _t([[[-0.25, -0.35], [-0.1, -0.3]]], requires_grad=True)
    ref = _t([[[-0.3, -0.3], [-0.2, -0.2]]], requires_grad=True)
    advantages = _t([[1.0, -1.0]], requires_grad=True)
    loss, _ = grpo_loss(log_probs, old, ref, torch.ones_like(log_probs), advantages)
    loss.backward()
    assert log_probs.grad is not None
    assert old.grad is None and ref.grad is None and advantages.grad is None


def test_on_policy_gradient_equals_advantage_weighted_score():
    # With old == current (first update epoch) the ratio is 1 and the gradient is -A_i / N per token.
    log_probs = _t([[[-0.2, -0.4], [-0.1, -0.3]]], requires_grad=True)
    loss, _ = grpo_loss(
        log_probs, log_probs.detach(), log_probs.detach(), torch.ones_like(log_probs), _t([[1.0, -1.0]])
    )
    loss.backward()
    _close(log_probs.grad, [[[-0.5, -0.5], [0.5, 0.5]]])


@pytest.mark.parametrize("reduction", ["sequence", "token"])
def test_padding_values_cannot_affect_loss_or_gradients(reduction):
    mask = _t([[[1, 1, 0, 0], [1, 0, 0, 0]]])
    advantages = _t([[1.0, -1.0]])

    def run(pad_value):
        log_probs = _t([[[-0.2, -0.4, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]])
        old = _t([[[-0.3, -0.3, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0]]])
        ref = _t([[[-0.25, -0.5, 0.0, 0.0], [-0.3, 0.0, 0.0, 0.0]]])
        padding = mask == 0
        for tensor in (log_probs, old, ref):
            tensor[padding] = pad_value
        log_probs.requires_grad_(True)
        loss, _ = grpo_loss(log_probs, old, ref, mask, advantages, reduction=reduction)
        loss.backward()
        return loss.detach(), log_probs.grad

    clean_loss, clean_grad = run(0.0)
    for pad_value in (1e6, -1e6, float("nan"), float("inf")):
        loss, grad = run(pad_value)
        torch.testing.assert_close(loss, clean_loss, atol=0, rtol=0)
        torch.testing.assert_close(grad, clean_grad, atol=0, rtol=0)
        assert torch.equal(grad[mask == 0], torch.zeros(5, dtype=torch.float64))


def _valid_inputs():
    log_probs = _t([[[-0.2, -0.4], [-0.1, -0.3]]])
    return [log_probs, log_probs.clone(), log_probs.clone(), torch.ones_like(log_probs), _t([[1.0, -1.0]])]


@pytest.mark.parametrize(
    "position, value",
    [
        (1, _t([[[-0.2], [-0.1]]])),
        (3, torch.ones(1, 2, 3)),
        (4, _t([[1.0, -1.0, 0.0]])),
        (3, _t([[[1, 1], [0, 0]]])),  # empty completion
        (3, _t([[[1, 0.5], [1, 1]]])),  # non-binary mask
        (0, _t([[[float("nan"), -0.4], [-0.1, -0.3]]])),
        (2, _t([[[-0.2, float("inf")], [-0.1, -0.3]]])),
        (4, _t([[float("nan"), 1.0]])),
    ],
)
def test_invalid_inputs_raise(position, value):
    inputs = _valid_inputs()
    inputs[position] = value
    with pytest.raises(ValueError):
        grpo_loss(*inputs)


@pytest.mark.parametrize(
    "kwargs",
    [{"reduction": "mean"}, {"clip_epsilon": 0.0}, {"clip_epsilon": 1.5}, {"beta": -0.1}, {"beta": float("nan")}],
)
def test_invalid_hyperparameters_raise(kwargs):
    with pytest.raises(ValueError):
        grpo_loss(*_valid_inputs(), **kwargs)


def test_overflowing_ratio_fails_instead_of_clamping():
    log_probs = _t([[[400.0, 400.0]]])
    with pytest.raises(FloatingPointError):
        grpo_loss(log_probs, torch.zeros_like(log_probs), log_probs, torch.ones_like(log_probs), _t([[-1.0]]), beta=0.0)


@pytest.mark.parametrize("reduction", ["sequence", "token"])
@pytest.mark.parametrize("advantage", [1.0, 0.0])
def test_nonfinite_ratio_fails_even_when_clipping_gives_finite_loss(reduction, advantage):
    # Token ratios exp(800) and the sequence ratio exp(1600) overflow to inf; with A > 0 the
    # clipped branch alone would give a finite loss of -1.2.
    log_probs = _t([[[800.0, 800.0]]])
    with pytest.raises(FloatingPointError, match="ratio"):
        grpo_loss(
            log_probs,
            torch.zeros_like(log_probs),
            log_probs,
            torch.ones_like(log_probs),
            _t([[advantage]]),
            beta=0.0,
            reduction=reduction,
        )


def test_overflowing_kl_fails():
    log_probs = _t([[[-400.0, -400.0]]])
    with pytest.raises(FloatingPointError):
        grpo_loss(log_probs, log_probs, torch.zeros_like(log_probs), torch.ones_like(log_probs), _t([[0.0]]))


@pytest.mark.parametrize("reduction", ["sequence", "token"])
def test_metrics_are_finite_floats(reduction, recwarn):
    inputs = _valid_inputs()
    inputs[0] = inputs[0].clone().requires_grad_(True)
    _, metrics = grpo_loss(*inputs, reduction=reduction)
    assert set(metrics) >= {"loss", "surrogate", "kl", "clip_fraction", "ratio_mean"}
    assert all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())
    # Diagnostics are computed from detached tensors (no "requires_grad" conversion warnings).
    assert not [w for w in recwarn if "requires_grad" in str(w.message)]
