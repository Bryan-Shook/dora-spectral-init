import io

import pytest
import torch
from torch.func import functional_call
from torch.nn import functional as F

from dora_spectral import ControlledLinear
from dora_spectral.initialization import decompose, initialize_factors


@pytest.fixture
def example():
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(5, 9, generator=generator, dtype=torch.float64)
    bias = torch.randn(5, generator=generator, dtype=torch.float64)
    inputs = torch.randn(4, 9, generator=generator, dtype=torch.float64)
    return weight, bias, inputs


def test_matching_spectra_and_repeatability(example):
    weight, _, _ = example
    svd = decompose(weight, 3)
    b_s, a_s = initialize_factors(svd, "spectral", 0.3, 4)
    b_r, a_r = initialize_factors(svd, "random", 0.3, 4)
    for left, right in ((b_s, b_r), (a_s, a_r), (b_s @ a_s, b_r @ a_r)):
        torch.testing.assert_close(
            torch.linalg.svdvals(left), torch.linalg.svdvals(right), atol=1e-10, rtol=1e-8
        )
    repeated_b, repeated_a = initialize_factors(svd, "random", 0.3, 4)
    assert torch.equal(b_r, repeated_b)
    assert torch.equal(a_r, repeated_a)
    assert not torch.allclose(b_s @ a_s, b_r @ a_r)


@pytest.mark.parametrize("rule", ["additive", "exact", "detached"])
@pytest.mark.parametrize("initialization", ["spectral", "random"])
def test_starting_weight_outputs_and_bias(example, rule, initialization):
    weight, bias, inputs = example
    layer = ControlledLinear(weight, bias, rank=3, rule=rule, initialization=initialization)
    torch.testing.assert_close(layer.effective_weight(), weight, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(layer(inputs), F.linear(inputs, weight, bias))
    torch.testing.assert_close(layer(torch.zeros_like(inputs)), bias.expand(4, -1))


def test_exact_gradient_finite_differences(example):
    weight, bias, inputs = example
    layer = ControlledLinear(weight, bias, rank=2, rule="exact")
    parameters = dict(layer.named_parameters())
    names = tuple(parameters)

    def forward(*values):
        return functional_call(layer, dict(zip(names, values, strict=True)), (inputs,))

    assert torch.autograd.gradcheck(forward, tuple(parameters.values()), atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("rule", ["exact", "detached"])
def test_analytic_chain_rule_at_noninitial_state(example, rule):
    weight, _, _ = example
    layer = ControlledLinear(weight, rank=3, rule=rule, initialization="random")
    with torch.no_grad():
        layer.B.add_(0.03)
        layer.magnitude.mul_(1.2)
    z = layer.direction_weight().detach()
    norms = z.norm(dim=1)
    unit = z / norms[:, None]
    gradient = torch.linspace(-1, 1, weight.numel(), dtype=weight.dtype).reshape_as(weight)
    (layer.effective_weight() * gradient).sum().backward()
    dz = gradient
    if rule == "exact":
        dz = gradient - unit * (gradient * unit).sum(dim=1, keepdim=True)
    dz = (layer.magnitude.detach() / norms)[:, None] * dz
    torch.testing.assert_close(layer.B.grad, dz @ layer.A.detach().T)
    torch.testing.assert_close(layer.A.grad, layer.B.detach().T @ dz)
    torch.testing.assert_close(layer.magnitude.grad, (gradient * unit).sum(dim=1))


def test_detached_matches_fixed_denominator_surrogate(example):
    weight, bias, inputs = example
    layer = ControlledLinear(weight, bias, rank=2, rule="detached")
    norm = layer.direction_weight().detach().norm(dim=1)
    loss = layer(inputs).square().sum()
    actual = torch.autograd.grad(loss, tuple(layer.parameters()))
    z = layer.direction_weight()
    surrogate = F.linear(inputs, (layer.magnitude / norm)[:, None] * z, bias)
    expected = torch.autograd.grad(surrogate.square().sum(), tuple(layer.parameters()))
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)


def test_exact_detached_same_forward_different_gradients(example):
    weight, bias, inputs = example
    exact = ControlledLinear(weight, bias, rank=2, rule="exact")
    detached = ControlledLinear(weight, bias, rank=2, rule="detached")
    torch.testing.assert_close(exact(inputs), detached(inputs))
    exact(inputs).square().sum().backward()
    detached(inputs).square().sum().backward()
    assert not torch.allclose(exact.B.grad, detached.B.grad)
    torch.testing.assert_close(exact.magnitude.grad, detached.magnitude.grad)


@pytest.mark.parametrize("rule", ["additive", "exact", "detached"])
def test_learning_freeze_merge_and_checkpoint(example, rule):
    weight, bias, inputs = example
    layer = ControlledLinear(weight, bias, rank=2, rule=rule, initialization="random")
    buffers = {name: value.clone() for name, value in layer.named_buffers()}
    parameters = {name: value.detach().clone() for name, value in layer.named_parameters()}
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad()
        layer(inputs).square().sum().backward()
        assert layer.A.grad.norm() > 0 and layer.B.grad.norm() > 0
        optimizer.step()
    for name, value in layer.named_buffers():
        assert not value.requires_grad
        assert torch.equal(value, buffers[name])
    for name, value in layer.named_parameters():
        assert not torch.equal(value, parameters[name])
    torch.testing.assert_close(layer(inputs), layer.merged_linear()(inputs))
    stream = io.BytesIO()
    torch.save(layer.state_dict(), stream)
    stream.seek(0)
    restored = ControlledLinear(weight * 2, bias, rank=2, rule=rule, gamma=0.7)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    torch.testing.assert_close(layer(inputs), restored(inputs))


def test_checkpoint_rule_mismatch_rejected(example):
    weight, _, _ = example
    exact = ControlledLinear(weight, rank=2, rule="exact")
    detached = ControlledLinear(weight, rank=2, rule="detached")
    with pytest.raises(ValueError, match="same adapter rule"):
        detached.load_state_dict(exact.state_dict())


@pytest.mark.parametrize("gamma", [0, -1, float("nan"), float("inf")])
def test_invalid_gamma(example, gamma):
    with pytest.raises(ValueError, match="gamma"):
        ControlledLinear(example[0], rank=2, gamma=gamma)


def test_zero_dora_row_rejected():
    with pytest.raises(ValueError, match="row norms"):
        ControlledLinear(torch.zeros(3, 4), rank=2, rule="exact")
