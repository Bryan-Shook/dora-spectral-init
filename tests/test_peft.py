"""Compare PEFT extensions against independent dense math and stock PEFT."""

import copy

import pytest
import torch
from peft.tuners.lora.layer import Linear
from reference import ControlledLinear
from torch import nn

from dora_spectral.model import ControlledVariant, make_adapter


@pytest.mark.parametrize("rule", ["additive", "detached", "exact"])
@pytest.mark.parametrize("initialization", ["spectral", "random"])
def test_controlled_forward_and_gradients_match_reference(rule, initialization):
    torch.manual_seed(21)
    base = nn.Linear(9, 5)
    reference = ControlledLinear(
        base.weight, base.bias, rank=3, rule=rule, initialization=initialization, gamma=0.3, seed=7
    )
    actual = make_adapter(base, rule, rank=3, seed=7)
    a = actual.lora_A["default"].weight
    b = actual.lora_B["default"].weight
    actual.register_buffer("initial_A", reference.A.detach().clone())
    actual.register_buffer("initial_B", reference.B.detach().clone())
    actual.lora_variant["default"] = ControlledVariant(rule)
    with torch.no_grad():
        # Compare after leaving initialization; zero-update tests miss gradient bugs.
        reference.A.add_(torch.randn_like(reference.A) * 0.03)
        reference.B.add_(torch.randn_like(reference.B) * 0.04)
        a.copy_(reference.A)
        b.copy_(reference.B)
        if rule != "additive":
            reference.magnitude.mul_(1.1)
            actual.lora_magnitude_vector["default"].weight.copy_(reference.magnitude)
    x = torch.randn(2, 4, 9, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    left, right = actual(x), reference(y)
    torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
    left.square().sum().backward()
    right.square().sum().backward()
    for first, second in ((a.grad, reference.A.grad), (b.grad, reference.B.grad), (x.grad, y.grad)):
        torch.testing.assert_close(first, second, atol=5e-6, rtol=5e-5)
    if rule != "additive":
        torch.testing.assert_close(
            actual.lora_magnitude_vector["default"].weight.grad,
            reference.magnitude.grad,
            atol=5e-6,
            rtol=5e-5,
        )


@pytest.mark.parametrize("rule", ["additive", "detached"])
def test_ordinary_uses_stock_peft_forward_and_gradients(rule):
    torch.manual_seed(12)
    base = nn.Linear(9, 5)
    actual = make_adapter(copy.deepcopy(base), rule, rank=3, seed=2)
    stock = Linear(
        copy.deepcopy(base),
        "default",
        r=3,
        lora_alpha=3,
        lora_dropout=0,
        use_dora=rule == "detached",
    )
    stock.load_state_dict(actual.state_dict())
    stock.base_layer.requires_grad_(False)
    with torch.no_grad():
        actual.lora_B["default"].weight.normal_(std=0.05)
        stock.lora_B["default"].weight.copy_(actual.lora_B["default"].weight)
    x = torch.randn(2, 9, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    torch.testing.assert_close(actual(x), stock(y), atol=0, rtol=0)
    actual(x).square().sum().backward()
    stock(y).square().sum().backward()
    for (name, first), (_, second) in zip(
        actual.named_parameters(), stock.named_parameters(), strict=True
    ):
        if first.requires_grad:
            torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0, msg=name)
    torch.testing.assert_close(x.grad, y.grad, atol=0, rtol=0)
