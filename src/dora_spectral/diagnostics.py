"""Effective-weight measurements and reversible, shared-state DoRA interventions."""

import copy
import json

import torch
from peft.tuners.lora.layer import Linear
from torch.nn import functional as F

from .model import ControlledVariant, adapter_state, load_adapter_state


def adapter_layers(model, all_layers=False):
    layers = [
        (name, module) for name, module in model.named_modules() if isinstance(module, Linear)
    ]
    if all_layers or len(layers) <= 6:
        return layers
    indices = {0, 1, len(layers) // 2, len(layers) // 2 + 1, len(layers) - 2, len(layers) - 1}
    return [layers[index] for index in sorted(indices)]


@torch.no_grad()
def effective_weight(module):
    weight = module.base_layer.weight.float()
    product = module.lora_B["default"].weight @ module.lora_A["default"].weight
    if hasattr(module, "initial_A"):
        product = product - module.initial_B @ module.initial_A
    direction = weight + module.scaling["default"] * product
    if module.use_dora["default"]:
        magnitude = module.lora_magnitude_vector["default"].weight
        direction = (
            magnitude[:, None] * direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        )
    return direction


@torch.no_grad()
def snapshot(model, all_layers=False):
    return {
        name: effective_weight(module).clone() for name, module in adapter_layers(model, all_layers)
    }


def step_comparison(left, right):
    difference = sum((left[key] - right[key]).double().square().sum() for key in left).sqrt()
    norm_left = sum(value.double().square().sum() for value in left.values()).sqrt()
    norm_right = sum(value.double().square().sum() for value in right.values()).sqrt()
    dot = sum((left[key].double() * right[key].double()).sum() for key in left)
    return {
        "relative_distance": (difference / (norm_left + norm_right + 1e-30)).item(),
        "cosine": (dot / (norm_left * norm_right)).clamp(-1, 1).item()
        if norm_left * norm_right > 0
        else None,
        "exact_norm": norm_left.item(),
        "detached_norm": norm_right.item(),
    }


class MovementRecorder:
    def __init__(self, model, directory, horizon, steps=(0, 1, 10, 50, 100)):
        self.model = model
        self.path = directory / "diagnostics.jsonl"
        self.steps = set(steps) | {horizon}
        self.horizon = horizon
        self.previous = {}
        self.magnitudes = {}

    @torch.no_grad()
    def before(self, completed):
        if completed + 1 in self.steps:
            self.previous = snapshot(self.model, completed + 1 == self.horizon)
            self.magnitudes = {
                name: module.lora_magnitude_vector["default"].weight.clone()
                for name, module in adapter_layers(self.model, completed + 1 == self.horizon)
                if module.use_dora["default"]
            }

    @torch.no_grad()
    def record(self, completed):
        if completed not in self.steps:
            return
        rows = []
        for name, module in adapter_layers(self.model, completed == self.horizon):
            weight = effective_weight(module)
            base = module.base_layer.weight.float()
            row = {
                "layer": name,
                "relative_net_movement": ((weight - base).norm() / base.norm()).item(),
                "mean_row_angle_radians": F.cosine_similarity(weight, base, dim=1)
                .clamp(-1, 1)
                .acos()
                .mean()
                .item(),
            }
            if completed == 0:
                a, b = module.lora_A["default"].weight, module.lora_B["default"].weight
                row["initial_product_norm"] = (b @ a).norm().item()
                row["a_singular_values"] = torch.linalg.svdvals(a).tolist()
                row["b_singular_values"] = torch.linalg.svdvals(b).tolist()
            if name in self.previous:
                previous = self.previous[name]
                step = weight - previous
                unit = F.normalize(previous, dim=1)
                parallel = (step * unit).sum(dim=1, keepdim=True) * unit
                row.update(
                    step_norm=step.norm().item(),
                    parallel_step_norm=parallel.norm().item(),
                    perpendicular_step_norm=(step - parallel).norm().item(),
                )
            if module.use_dora["default"]:
                magnitude = module.lora_magnitude_vector["default"].weight
                row["relative_magnitude_change"] = (
                    (magnitude - base.norm(dim=1)).norm() / base.norm()
                ).item()
                if magnitude.grad is not None:
                    row["magnitude_gradient_norm"] = magnitude.grad.norm().item()
                if name in self.magnitudes:
                    row["magnitude_step_norm"] = (magnitude - self.magnitudes[name]).norm().item()
            rows.append(row)
        with self.path.open("a") as handle:
            handle.write(json.dumps({"step": completed, "layers": rows}) + "\n")
        self.previous = {}
        self.magnitudes = {}


def counterfactual(model, optimizer, batch, loss_function):
    """Compare actual clipped AdamW steps; restore parameters, moments, RNG, gradients and rule."""
    layers = adapter_layers(model, True)
    if not layers or any(
        not isinstance(m.lora_variant.get("default"), ControlledVariant)
        or not m.use_dora["default"]
        for _, m in layers
    ):
        raise ValueError("counterfactuals require controlled DoRA layers")
    saved = copy.deepcopy(adapter_state(model))
    moments = copy.deepcopy(optimizer.state_dict())
    parameters = [p for p in model.parameters() if p.requires_grad]
    gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
    rules = [module.lora_variant["default"].rule for _, module in layers]
    mode, cache = model.training, model.config.use_cache
    before = snapshot(model, True)
    devices = sorted({p.device.index for p in parameters if p.is_cuda})
    steps = {}
    try:
        with torch.random.fork_rng(devices=devices):
            for rule in ("exact", "detached"):
                load_adapter_state(model, saved)
                optimizer.load_state_dict(copy.deepcopy(moments))
                optimizer.zero_grad(set_to_none=True)
                for _, module in layers:
                    module.lora_variant["default"].rule = rule
                model.train()
                model.config.use_cache = False
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                loss = loss_function(logits, batch["labels"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                steps[rule] = {
                    name: value - before[name] for name, value in snapshot(model, True).items()
                }
            load_adapter_state(model, saved)
            result = step_comparison(steps["exact"], steps["detached"])
            result["layers"] = {
                name: step_comparison({name: steps["exact"][name]}, {name: steps["detached"][name]})
                for name in before
            }
            model.eval()
            with torch.no_grad():
                baseline = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                baseline_loss = loss_function(baseline, batch["labels"]).item()
                mask = batch["labels"][:, 1:] != -100
                base_logp = baseline[:, :-1][mask].float().log_softmax(-1)
                result["baseline_answer_nll"] = baseline_loss
                result["probes"] = {}
                common_norm = 0.1 * min(result["exact_norm"], result["detached_norm"])
                for rule in steps:
                    for equal_norm in (False, True):
                        norm = result[f"{rule}_norm"]
                        if norm == 0 or (equal_norm and common_norm == 0):
                            result["probes"][f"{rule}-equal-{equal_norm}"] = {"zero_step": True}
                            continue
                        multiplier = common_norm / norm if equal_norm else 1.0
                        hooks = []
                        try:
                            for name, module in layers:
                                delta = steps[rule][name] * multiplier

                                def perturb(module, inputs, output, delta=delta):
                                    return (output.float() + F.linear(inputs[0].float(), delta)).to(
                                        output.dtype
                                    )

                                hooks.append(module.register_forward_hook(perturb))
                            logits = model(
                                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                            ).logits
                            logp = logits[:, :-1][mask].float().log_softmax(-1)
                            result["probes"][f"{rule}-equal-{equal_norm}"] = {
                                "effective_step_norm": norm * multiplier,
                                "answer_nll_change": loss_function(logits, batch["labels"]).item()
                                - baseline_loss,
                                "mean_answer_token_kl": (base_logp.exp() * (base_logp - logp))
                                .sum(-1)
                                .mean()
                                .item(),
                            }
                        finally:
                            for hook in hooks:
                                hook.remove()
                result["probe_definition"] = (
                    "Local effective-weight interventions via output hooks; equal norms use "
                    "0.1 times "
                    "the smaller step norm. These are not realizable factor-training trajectories."
                )
            return result
    finally:
        load_adapter_state(model, saved)
        optimizer.load_state_dict(moments)
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient
        for (_, module), rule in zip(layers, rules, strict=True):
            module.lora_variant["default"].rule = rule
        model.train(mode)
        model.config.use_cache = cache
