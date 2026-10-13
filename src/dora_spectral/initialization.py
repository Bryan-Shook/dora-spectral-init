"""Balanced spectral and spectrum-matched random initialization."""

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Initialization = Literal["spectral", "random"]


@dataclass(frozen=True)
class Decomposition:
    u: Tensor
    singular_values: Tensor
    vh: Tensor


@torch.no_grad()
def decompose(weight: Tensor, rank: int) -> Decomposition:
    """Compute the SVD once and reuse it across conditions."""
    if weight.ndim != 2 or not 1 <= rank <= min(weight.shape):
        raise ValueError("rank must be between 1 and min(weight.shape) for a matrix")
    if weight.dtype not in (torch.float32, torch.float64):
        raise ValueError("decompose in FP32, or FP64 for toy correctness checks")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must be finite")
    u, values, vh = torch.linalg.svd(weight, full_matrices=False)
    return Decomposition(u[:, :rank], values[:rank], vh[:rank])


def _random_basis(rows: int, rank: int, template: Tensor, generator: torch.Generator) -> Tensor:
    noise = torch.randn(
        rows, rank, dtype=template.dtype, device=template.device, generator=generator
    )
    q, r = torch.linalg.qr(noise, mode="reduced")
    # Keep the QR diagonal signs consistent.
    signs = torch.where(r.diagonal() < 0, -1, 1)
    return q * signs


@torch.no_grad()
def initialize_factors(
    decomposition: Decomposition,
    initialization: Initialization,
    gamma: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Return B, A with factor spectra sqrt(gamma * sigma)."""
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    u, values, vh = decomposition.u, decomposition.singular_values, decomposition.vh
    if initialization == "spectral":
        left, right = u, vh.T
    elif initialization == "random":
        generator = torch.Generator(device=u.device).manual_seed(seed)
        left = _random_basis(u.shape[0], values.numel(), u, generator)
        right = _random_basis(vh.shape[1], values.numel(), u, generator)
    else:
        raise ValueError(f"unknown initialization: {initialization}")
    scales = (gamma * values).sqrt()
    return (left * scales).clone(), (scales[:, None] * right.T).clone()
