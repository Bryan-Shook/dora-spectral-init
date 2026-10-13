"""Qwen integration using the tested dense FP32 reference adapter."""

import hashlib
import uuid
from pathlib import Path

import torch
from torch import nn

from .adapters import ControlledLinear
from .initialization import Decomposition, decompose

CONDITIONS = {
    "A-S": ("additive", "spectral"),
    "A-R": ("additive", "random"),
    "D-S": ("detached", "spectral"),
    "D-R": ("detached", "random"),
    "E-S": ("exact", "spectral"),
    "E-R": ("exact", "random"),
    "L0": ("additive", "standard"),
    "D0": ("detached", "standard"),
}


def layer_seed(seed: int, name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:init:{name}".encode()).digest()[:8], "little") % (
        2**63 - 1
    )


class ModelLinear(ControlledLinear):
    def forward(self, inputs):
        # Preserve FP32 adapter/norm arithmetic even in a BF16 transformer.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            output = super().forward(inputs)
        return output.to(inputs.dtype)


def insert_adapters(
    model,
    condition: str,
    rank: int,
    gamma: float,
    seed: int,
    targets: list[str],
    cache_dir: Path | None = None,
) -> dict:
    rule, initialization = CONDITIONS[condition]
    model.requires_grad_(False)
    names = [
        name
        for name, layer in model.named_modules()
        if isinstance(layer, nn.Linear) and name.rsplit(".", 1)[-1] in targets
    ]
    if not names or set(targets) != {name.rsplit(".", 1)[-1] for name in names}:
        raise ValueError("adapter targets are missing from this model")
    metadata = {}
    for name in names:
        original = model.get_submodule(name)
        weight = original.weight.detach().float()
        weight_hash = hashlib.sha256(weight.cpu().numpy().tobytes()).hexdigest()
        svd = None
        if initialization != "standard":
            cache = None if cache_dir is None else cache_dir / f"{weight_hash}-r{rank}.pt"
            if cache is not None and cache.exists():
                saved = torch.load(cache, map_location=weight.device, weights_only=True)
                svd = Decomposition(**saved)
            else:
                svd = decompose(weight, rank)
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    temp = cache.with_suffix(f".{uuid.uuid4().hex}.tmp")
                    torch.save(
                        {
                            "u": svd.u.cpu(),
                            "singular_values": svd.singular_values.cpu(),
                            "vh": svd.vh.cpu(),
                        },
                        temp,
                    )
                    temp.replace(cache)
        adapted = ModelLinear(
            weight,
            original.bias,
            rank=rank,
            rule=rule,
            initialization=initialization,
            gamma=gamma,
            seed=layer_seed(seed, name),
            decomposition=svd,
        )
        if rule != "additive" and (weight.norm(dim=1) <= adapted.norm_floor).any():
            raise ValueError(f"norm floor active: {name}")
        parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, adapted)
        with torch.no_grad():
            error = (adapted.effective_weight() - weight).norm() / weight.norm()
        if error > 1e-5:
            raise ValueError(f"initial reconstruction failed: {name}")
        metadata[name] = {
            "weight_sha256": weight_hash,
            "shape": list(weight.shape),
            "relative_initial_error": float(error),
            "seed": layer_seed(seed, name),
        }
        if svd is not None:
            with torch.no_grad():
                expected = (gamma * svd.singular_values).sqrt()
                factor_errors = [
                    float(
                        (
                            (torch.linalg.svdvals(factor) - expected).abs()
                            / expected.clamp_min(1e-12)
                        ).max()
                    )
                    for factor in (adapted.B, adapted.A)
                ]
            if max(factor_errors) > 1e-5:
                raise ValueError(f"factor spectrum mismatch: {name}")
            metadata[name]["factor_spectrum_relative_errors"] = factor_errors
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = 0.0
        if hasattr(layer, "attention_dropout"):
            layer.attention_dropout = 0.0
    return {
        "layers": metadata,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "arithmetic": "dense FP32 targets; output cast to transformer dtype",
    }


def adapter_state(model) -> dict:
    return {
        name: layer.state_dict()
        for name, layer in model.named_modules()
        if isinstance(layer, ModelLinear)
    }


def restore_adapters(model, state: dict) -> None:
    names = {name for name, layer in model.named_modules() if isinstance(layer, ModelLinear)}
    if names != set(state):
        raise ValueError("checkpoint adapter targets differ")
    for name in names:
        model.get_submodule(name).load_state_dict(state[name])


def merge_adapters(model):
    """Only call on a separately loaded evaluation model; leaves training states intact."""
    for name, layer in list(model.named_modules()):
        if isinstance(layer, ModelLinear):
            # Keep FP32 target computation identical to the unmerged reference path.
            merged = FP32Linear(layer.merged_linear())
            parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
            setattr(model.get_submodule(parent_name) if parent_name else model, leaf, merged)
    return model


class FP32Linear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, inputs):
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            output = self.linear(inputs.float())
        return output.to(inputs.dtype)


def load_model(root: Path, device: str, precision: str, attention: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if precision == "bf16" and device != "cuda":
        raise ValueError("Use fp32 for CPU checks; BF16 requires CUDA in this runner")
    if device == "cuda" and (
        not torch.cuda.is_available()
        or (precision == "bf16" and not torch.cuda.is_bf16_supported())
    ):
        raise RuntimeError("Requested CUDA/BF16 is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_input_names = ["input_ids", "attention_mask"]
    model = AutoModelForCausalLM.from_pretrained(
        root / "model",
        local_files_only=True,
        dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
        attn_implementation=attention,
    ).to(device)
    model.config.use_cache = False
    return model, tokenizer
