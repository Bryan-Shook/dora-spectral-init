"""H100 correctness, overfit, longest-batch memory, and real-model resume gates."""

import argparse
import contextlib
import gc
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .data import collate
from .diagnostics import adapter_layers, effective_weight
from .experiment import (
    load_prepared,
    record_invocation,
    require_gpu,
    run_metadata,
    write_metadata,
)
from .initialization import decompose, initialize_factors
from .model import CONDITIONS, ControlledVariant, adapter_state, attach, make_adapter
from .numerics import logit_errors
from .training import Budget, answer_nll, evaluate, load_model, train, write_json


@contextlib.contextmanager
def retain_result(directory, result, budget):
    try:
        yield
    finally:
        result["seconds"] = budget.elapsed
        write_json(directory / "validation.json", result)


def gradient_checks(weight):
    """Use a real weight slice, independent dense gradients, and a separate FP64 gradcheck."""
    results = {}
    device = weight.device
    for rule in ("additive", "exact", "detached"):
        with torch.random.fork_rng(devices=[device.index] if weight.is_cuda else []):
            torch.manual_seed(42)
            base = nn.Linear(9, 5, bias=True, device=device)
            with torch.no_grad():
                base.weight.copy_(weight[:5, :9].float())
            actual = make_adapter(base, rule, 3, 7)
            b0, a0 = initialize_factors(decompose(base.weight.detach(), 3), "spectral", 1, 7)
            actual.register_buffer("initial_A", a0.clone())
            actual.register_buffer("initial_B", b0.clone())
            actual.lora_variant["default"] = ControlledVariant(rule)
            with torch.no_grad():
                actual.lora_A["default"].weight.copy_(a0 + torch.randn_like(a0) * 0.03)
                actual.lora_B["default"].weight.copy_(b0 + torch.randn_like(b0) * 0.03)
                if rule != "additive":
                    actual.lora_magnitude_vector["default"].weight.mul_(1.1)
            a = actual.lora_A["default"].weight.detach().clone().requires_grad_()
            b = actual.lora_B["default"].weight.detach().clone().requires_grad_()
            x = torch.randn(2, 4, 9, device=device, requires_grad=True)
            reference_x = x.detach().clone().requires_grad_()
            direction = base.weight.detach() + b @ a - b0 @ a0
            actual_parameters = [actual.lora_A["default"].weight, actual.lora_B["default"].weight]
            reference_parameters = [a, b]
            if rule != "additive":
                magnitude = (
                    actual.lora_magnitude_vector["default"].weight.detach().clone().requires_grad_()
                )
                norm = direction.norm(dim=1, keepdim=True)
                if rule == "detached":
                    norm = norm.detach()
                direction = magnitude[:, None] * direction / norm
                actual_parameters.append(actual.lora_magnitude_vector["default"].weight)
                reference_parameters.append(magnitude)
            left, right = actual(x), F.linear(reference_x, direction, base.bias.detach())
            torch.testing.assert_close(left, right, atol=3e-6, rtol=3e-5)
            left_gradients = torch.autograd.grad(left.square().sum(), [*actual_parameters, x])
            right_gradients = torch.autograd.grad(
                right.square().sum(), [*reference_parameters, reference_x]
            )
            errors = []
            for left_gradient, right_gradient in zip(left_gradients, right_gradients, strict=True):
                torch.testing.assert_close(left_gradient, right_gradient, atol=5e-6, rtol=5e-5)
                errors.append((left_gradient - right_gradient).abs().max().item())
            results[rule] = {"max_gradient_absolute_errors": errors, "passed": True}
    generator = torch.Generator().manual_seed(12)
    base = torch.randn(3, 4, dtype=torch.float64, generator=generator)
    a = torch.randn(2, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    b = torch.randn(3, 2, dtype=torch.float64, generator=generator, requires_grad=True)
    m = torch.rand(3, dtype=torch.float64, generator=generator, requires_grad=True)
    initial = (b @ a).detach()

    def exact(a, b, m):
        direction = base + b @ a - initial
        return m[:, None] * direction / direction.norm(dim=1, keepdim=True)

    results["fp64_exact_gradcheck"] = torch.autograd.gradcheck(
        exact, (a, b, m), atol=1e-6, rtol=1e-4
    )
    return results


def validate(args, config):
    from transformers import AutoTokenizer

    require_gpu()
    prepared, examples = load_prepared(config, args.data)
    config = dict(config, learning_rate=1e-4, gamma=1.0, counterfactuals=False)
    metadata = run_metadata(config, "VALIDATION", 0, prepared)
    write_metadata(args.directory, metadata)
    budget = Budget(args.seconds)
    result = {
        "passed": False,
        "config": metadata["config"],
        "data_sha256": prepared["examples_sha256"],
        "source_sha256": metadata["source_sha256"],
        "versions": metadata["versions"],
        "gates": {},
    }
    cache = args.data / "svd" / prepared["model_revision"] / str(config["rank"])
    with (
        record_invocation(args.directory, args.reports, budget, kind="validation") as invocation,
        retain_result(args.directory, result, budget),
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["model_revision"]
        )
        tokenizer.pad_token = tokenizer.eos_token
        config["pad_id"] = tokenizer.pad_token_id
        probe = collate(examples["train"][:2], config["pad_id"], "cuda")
        mask = probe["attention_mask"].bool().cpu()
        model = load_model(config, prepared, "cuda").eval()
        with torch.no_grad():
            baseline = model(
                input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
            ).logits.cpu()
            repeat = model(
                input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
            ).logits.cpu()
        noise = logit_errors(repeat, baseline, mask)
        criteria = {
            "max_logit_error": max(1e-4, 2 * noise["max_error"]),
            "relative_logit_rms": max(1e-5, 2 * noise["relative_rms"]),
            "effective_weight_relative_error": 1e-5,
            "minimum_token_agreement": noise["next_token_agreement"],
            "overfit_max_final_initial_ratio": 0.2,
            "overfit_max_final_nll": 0.1,
            "minimum_memory_headroom_fraction": 0.1,
        }
        write_json(args.directory / "criteria.json", {"repeat_noise": noise, "criteria": criteria})
        result["baseline_pilot_nll"] = answer_nll(model, examples["pilot"], config, budget)
        baseline_dir = args.directory / "baseline-pilot"
        baseline_dir.mkdir(exist_ok=True)
        result["baseline_generation"] = evaluate(
            model, tokenizer, examples["pilot"][:16], config, baseline_dir, budget
        )
        anchor = next(module for name, module in model.named_modules() if name.endswith("q_proj"))
        result["gradient_checks"] = gradient_checks(anchor.weight.detach())
        del model, anchor
        gc.collect()
        torch.cuda.empty_cache()
        initializations = {}
        for condition in CONDITIONS:
            budget.check(reserve=120)
            model = load_model(config, prepared, "cuda").eval()
            audit = attach(model, condition, config["rank"], 1, 0, cache)
            with torch.no_grad():
                logits = model(
                    input_ids=probe["input_ids"], attention_mask=probe["attention_mask"]
                ).logits.cpu()
                errors = [
                    (
                        (effective_weight(module) - module.base_layer.weight.float()).norm()
                        / module.base_layer.weight.float().norm()
                    ).item()
                    for _, module in adapter_layers(model, True)
                ]
            drift = logit_errors(logits, baseline, mask)
            allowed = all(
                ("lora_A." in name or "lora_B." in name or "lora_magnitude_vector." in name)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
            passed = (
                max(errors) <= criteria["effective_weight_relative_error"]
                and allowed
                and drift["max_error"] <= criteria["max_logit_error"]
                and drift["relative_rms"] <= criteria["relative_logit_rms"]
                and drift["next_token_agreement"] >= criteria["minimum_token_agreement"]
            )
            initializations[condition] = {
                "passed": passed,
                "logits": drift,
                "max_effective_weight_relative_error": max(errors),
                "audit": audit,
            }
            write_json(args.directory / "initializations.json", initializations)
            if not passed:
                raise RuntimeError(f"initialization equality gate failed: {condition}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result["gates"]["initialization_and_gradients"] = True
        model = load_model(config, prepared, "cuda")
        attach(model, "E-S", config["rank"], 1, 0, cache)
        frozen = {
            name: module.base_layer.weight.detach().cpu().clone()
            for name, module in adapter_layers(model, True)
        }
        subset = examples["train"][:32]
        before = answer_nll(model, subset, config, budget)
        train(
            model,
            subset,
            config,
            0,
            args.overfit_steps,
            args.directory / "overfit",
            budget,
            dict(metadata, probe="overfit"),
            overfit=True,
        )
        after = answer_nll(model, subset, config, budget)
        for name, module in adapter_layers(model, True):
            if not torch.equal(frozen[name], module.base_layer.weight.cpu()):
                raise RuntimeError("frozen backbone changed")
        del frozen
        result["overfit"] = {"initial_nll": before, "final_nll": after, "steps": args.overfit_steps}
        result["gates"]["overfit_and_freeze"] = (
            after < criteria["overfit_max_final_nll"]
            and after < before * criteria["overfit_max_final_initial_ratio"]
        )
        if not result["gates"]["overfit_and_freeze"]:
            raise RuntimeError("32-example overfit gate failed")
        longest = sorted(examples["train"], key=lambda row: len(row["input_ids"]), reverse=True)[
            : config["batch_size"]
        ]
        memory = train(
            model,
            longest,
            dict(config, epochs=1),
            0,
            1,
            args.directory / "longest-batch",
            budget,
            dict(metadata, probe="longest-batch"),
        )
        total = torch.cuda.get_device_properties(0).total_memory
        free, _ = torch.cuda.mem_get_info()
        result["memory"] = {
            **memory,
            "device_total_bytes": total,
            "free_after_bytes": free,
            "longest_tokens": max(len(row["input_ids"]) for row in longest),
        }
        result["gates"]["memory"] = (
            memory["peak_reserved_bytes"] <= 0.9 * total and free >= 0.1 * total
        )
        if not result["gates"]["memory"]:
            raise RuntimeError("longest-batch memory headroom gate failed")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        expected_state = None
        for label in ("uninterrupted", "resumed"):
            model = load_model(config, prepared, "cuda")
            attach(model, "E-R", config["rank"], 1, 0, cache)
            directory = args.directory / label
            if label == "resumed":
                train(
                    model,
                    examples["train"],
                    config,
                    0,
                    3,
                    directory,
                    budget,
                    dict(metadata, probe="resume"),
                )
            train(
                model,
                examples["train"],
                config,
                0,
                6,
                directory,
                budget,
                dict(metadata, probe="resume"),
            )
            state = {
                name: {key: value.detach().cpu().clone() for key, value in values.items()}
                for name, values in adapter_state(model).items()
            }
            if expected_state is None:
                expected_state = state
            else:
                for name, values in state.items():
                    for key, value in values.items():
                        torch.testing.assert_close(
                            value, expected_state[name][key], atol=1e-6, rtol=1e-5
                        )
            del model
            gc.collect()
            torch.cuda.empty_cache()
        result["gates"]["checkpoint_resume"] = True
        result["passed"] = all(result["gates"].values())
        result["seconds"] = budget.elapsed
        write_json(args.directory / "validation.json", result)
        invocation["status"] = "complete"
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    parser.add_argument("--data", type=Path, default=Path("runs/data"))
    parser.add_argument("--directory", type=Path, default=Path("runs/validation-h100"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--seconds", type=float, default=14400)
    parser.add_argument("--overfit-steps", type=int, default=100)
    args = parser.parse_args()
    if args.seconds <= 120 or args.overfit_steps < 1:
        parser.error("seconds must exceed 120 and overfit steps must be positive")
    if args.directory.exists():
        parser.error("use a new validation directory; reruns must retain earlier evidence")
    return validate(args, yaml.safe_load(args.config.read_text()))


if __name__ == "__main__":
    raise SystemExit(main())
