"""PEFT projections with only the controlled-initialization/gradient extensions."""

from pathlib import Path

import torch
from peft.tuners.lora.layer import Linear, LoraVariant
from torch import nn
from torch.nn import functional as F

from .initialization import Decomposition, decompose, initialize_factors

CONDITIONS = {
    "A-S": ("additive", "spectral"),
    "A-R": ("additive", "random"),
    "D-S": ("detached", "spectral"),
    "D-R": ("detached", "random"),
    "E-S": ("exact", "spectral"),
    "E-R": ("exact", "random"),
    "L0": ("additive", "ordinary"),
    "D0": ("detached", "ordinary"),
}


class ControlledVariant(LoraVariant):
    """Single-adapter, zero-dropout experiment variant of PEFT's linear forward.

    PEFT still computes the base projection, casts adapter inputs and casts outputs.
    Frozen initial factors cancel the initial update without rounding a BF16 residual.
    Both frozen-factor forwards stay in the input-gradient graph.
    """

    def __init__(self, rule):
        self.rule = rule

    @staticmethod
    def merge_unsafe(*args, **kwargs):
        raise NotImplementedError("evaluate controlled adapters unmerged")

    @staticmethod
    def unmerge(*args, **kwargs):
        raise NotImplementedError("evaluate controlled adapters unmerged")

    def forward(self, module, *, active_adapter, x, result, **kwargs):
        a = module.lora_A[active_adapter]
        b = module.lora_B[active_adapter]
        delta = b(a(x)) - F.linear(F.linear(x, module.initial_A), module.initial_B)
        if self.rule == "additive":
            return result + delta
        direction = module.base_layer.weight.float() + (
            b.weight @ a.weight - module.initial_B @ module.initial_A
        )
        norm = direction.norm(dim=1).clamp_min(1e-12)
        if self.rule == "detached":
            norm = norm.detach()
        scale = module.lora_magnitude_vector[active_adapter].weight / norm
        base = result if module.base_layer.bias is None else result - module.base_layer.bias
        return result + ((scale - 1) * base + scale * delta)


def make_adapter(layer, rule, rank, seed):
    """Use unmodified PEFT LoRA/DoRA classes and FP32 trainable parameters."""
    devices = [layer.weight.device.index] if layer.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        adapter = Linear(
            layer,
            "default",
            r=rank,
            lora_alpha=rank,
            lora_dropout=0,
            use_dora=rule != "additive",
        )
    adapter.base_layer.requires_grad_(False)
    # Direct layer construction needs the upcast normally performed by PeftModel.
    adapter.lora_A.float()
    adapter.lora_B.float()
    if rule != "additive":
        adapter.lora_magnitude_vector.float()
        with torch.no_grad():
            adapter.lora_magnitude_vector["default"].weight.copy_(layer.weight.float().norm(dim=1))
    return adapter


def attach(
    model, condition, rank, gamma, seed, cache_dir: Path | None = None, targets=("q_proj", "v_proj")
):
    rule, initialization = CONDITIONS[condition]
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
    model.config.attention_dropout = 0.0
    layers = [
        (name, layer)
        for name, layer in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets and isinstance(layer, nn.Linear)
    ]
    if not layers:
        raise ValueError("no target projections found")
    audit = {"condition": condition, "backend": "peft", "layers": []}
    for index, (name, layer) in enumerate(layers):
        adapter = make_adapter(layer, rule, rank, seed + index)
        entry = {"name": name}
        if initialization != "ordinary":
            weight = layer.weight.detach().float()
            path = cache_dir / f"{name}.pt" if cache_dir else None
            if path and path.exists():
                values = torch.load(path, map_location=weight.device, weights_only=True)
                if values["shape"] != list(weight.shape) or values["rank"] != rank:
                    raise ValueError("SVD cache shape/rank mismatch")
                svd = Decomposition(values["u"], values["singular_values"], values["vh"])
            else:
                svd = decompose(weight, rank)
                if path:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    torch.save(
                        {
                            "shape": list(weight.shape),
                            "rank": rank,
                            "u": svd.u.cpu(),
                            "singular_values": svd.singular_values.cpu(),
                            "vh": svd.vh.cpu(),
                        },
                        temporary,
                    )
                    temporary.replace(path)
            b, a = initialize_factors(svd, initialization, gamma, seed + index)
            with torch.no_grad():
                adapter.lora_A["default"].weight.copy_(a)
                adapter.lora_B["default"].weight.copy_(b)
            adapter.register_buffer("initial_A", a.detach().clone())
            adapter.register_buffer("initial_B", b.detach().clone())
            adapter.lora_variant["default"] = ControlledVariant(rule)
            expected = (gamma * svd.singular_values).sort().values
            errors = [
                (
                    (torch.linalg.eigvalsh(gram).clamp_min(0) - expected).norm() / expected.norm()
                ).item()
                for gram in (b.T @ b, a @ a.T)
            ]
            if max(errors) > 1e-5:
                raise RuntimeError(f"factor-spectrum gate failed: {name}")
            entry["factor_spectrum_relative_errors"] = errors
        parent, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(parent), attribute, adapter)
        audit["layers"].append(entry)
    audit["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return audit


def adapter_state(model):
    """Checkpoint FP32 adapters and frozen corrections, excluding the pinned base."""
    return {
        name: {
            key: value
            for key, value in module.state_dict().items()
            if not key.startswith("base_layer.")
        }
        for name, module in model.named_modules()
        if isinstance(module, Linear)
    }


def load_adapter_state(model, state):
    expected = adapter_state(model)
    if state.keys() != expected.keys():
        raise ValueError("checkpoint adapter names differ")
    for name, values in state.items():
        if values.keys() != expected[name].keys():
            raise ValueError(f"checkpoint adapter fields differ: {name}")
        model.get_submodule(name).load_state_dict(values, strict=False)
