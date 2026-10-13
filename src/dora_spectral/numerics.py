"""Logit drift measured only at explicitly selected token positions."""

import torch


def logit_errors(left, right, mask=None):
    if left.shape != right.shape:
        raise ValueError("logit tensors must have the same shape")
    if mask is not None:
        if mask.shape != left.shape[:-1] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean and match token positions")
        left, right = left[mask], right[mask]
    left, right = left.float(), right.float()
    if not left.numel():
        raise ValueError("select at least one token position")
    delta = left - right
    return {
        "positions": left.numel() // left.shape[-1],
        "max_error": delta.abs().max().item(),
        "relative_rms": (
            delta.square().mean().sqrt() / right.square().mean().sqrt().clamp_min(1e-12)
        ).item(),
        "next_token_agreement": (left.argmax(-1) == right.argmax(-1)).float().mean().item(),
    }
