"""Dense adapters for checking the math."""

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .initialization import Decomposition, Initialization, decompose, initialize_factors

Rule = Literal["additive", "detached", "exact"]


class ControlledLinear(nn.Module):
    """W0 + (BA - C0), optionally normalized by each output-row norm.

    Uses FP32, or FP64 for gradient checks. Builds full weights; optimize later for Qwen.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Tensor | None = None,
        *,
        rank: int = 16,
        initialization: Initialization = "spectral",
        rule: Rule = "additive",
        gamma: float = 1.0,
        seed: int = 0,
        norm_floor: float = 1e-12,
        decomposition: Decomposition | None = None,
    ) -> None:
        super().__init__()
        if rule not in ("additive", "detached", "exact"):
            raise ValueError(f"unknown rule: {rule}")
        if not math.isfinite(norm_floor) or norm_floor <= 0:
            raise ValueError("norm_floor must be finite and positive")
        if weight.ndim != 2 or not 1 <= rank <= min(weight.shape):
            raise ValueError("invalid weight shape or rank")
        if bias is not None and bias.shape != (weight.shape[0],):
            raise ValueError("bias must have one entry per output row")
        self.rule = rule
        self.norm_floor = norm_floor
        dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
        base = weight.detach().to(dtype).clone()
        self.register_buffer("base_weight", base)
        self.register_buffer("bias", None if bias is None else bias.detach().to(dtype).clone())
        svd = decomposition if decomposition is not None else decompose(base, rank)
        if (
            svd.u.shape != (base.shape[0], rank)
            or svd.vh.shape != (rank, base.shape[1])
            or svd.singular_values.shape != (rank,)
            or svd.u.device != base.device
            or svd.u.dtype != base.dtype
        ):
            raise ValueError("decomposition must match weight shape, rank, device and dtype")
        b, a = initialize_factors(svd, initialization, gamma, seed)
        self.B = nn.Parameter(b)
        self.A = nn.Parameter(a)
        self.register_buffer("initial_product", (b @ a).detach().clone())
        norms = base.norm(dim=1)
        if rule != "additive":
            if (norms <= norm_floor).any():
                raise ValueError("DoRA base row norms must exceed the norm floor")
            self.magnitude = nn.Parameter(norms)
        else:
            self.register_parameter("magnitude", None)

    def direction_weight(self) -> Tensor:
        return self.base_weight + (self.B @ self.A - self.initial_product)

    def effective_weight(self) -> Tensor:
        z = self.direction_weight()
        if self.rule == "additive":
            return z
        norm = z.norm(dim=1).clamp_min(self.norm_floor)
        if self.rule == "detached":
            norm = norm.detach()
        return (self.magnitude / norm)[:, None] * z

    def forward(self, inputs: Tensor) -> Tensor:
        # Bias stays outside the weight normalization.
        return F.linear(inputs.to(self.base_weight.dtype), self.effective_weight(), self.bias)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Merge into a frozen linear layer for evaluation."""
        output, input_ = self.base_weight.shape
        merged = nn.Linear(
            input_,
            output,
            bias=self.bias is not None,
            device=self.base_weight.device,
            dtype=self.base_weight.dtype,
        )
        merged.weight.copy_(self.effective_weight())
        if self.bias is not None:
            merged.bias.copy_(self.bias)
        merged.requires_grad_(False)
        return merged

    def get_extra_state(self) -> dict:
        return {"rule": self.rule, "norm_floor": self.norm_floor}

    def set_extra_state(self, state: dict) -> None:
        if state["rule"] != self.rule:
            raise ValueError("construct the checkpoint with the same adapter rule")
        self.norm_floor = state["norm_floor"]
