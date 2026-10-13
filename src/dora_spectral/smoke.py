"""Try all six conditions on small random tensors."""

import argparse
import json
import platform
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from .adapters import ControlledLinear
from .initialization import decompose


def run(config: dict, device: str) -> dict:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; check allocation and torch build"
        )
    torch.manual_seed(config["seed"])
    weight = torch.randn(config["output_features"], config["input_features"], device=device)
    bias = torch.randn(config["output_features"], device=device)
    inputs = torch.randn(32, config["input_features"], device=device)
    targets = F.linear(inputs, weight + 0.1 * torch.randn_like(weight), bias)
    svd = decompose(weight, config["rank"])
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": device,
        "config": config,
        "conditions": [],
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        result.update(
            gpu=properties.name,
            total_memory_bytes=properties.total_memory,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
        torch.cuda.reset_peak_memory_stats()
    for rule in ("additive", "detached", "exact"):
        for initialization in ("spectral", "random"):
            layer = ControlledLinear(
                weight,
                bias,
                rank=config["rank"],
                gamma=config["gamma"],
                seed=config["seed"],
                rule=rule,
                initialization=initialization,
                decomposition=svd,
            )
            initial_error = (layer(inputs) - F.linear(inputs, weight, bias)).abs().max().item()
            if initial_error > 1e-5:
                raise RuntimeError(f"starting output mismatch: {initial_error}")
            initial_loss = F.mse_loss(layer(inputs), targets).item()
            optimizer = torch.optim.AdamW(
                layer.parameters(), lr=config["learning_rate"], weight_decay=0
            )
            for _ in range(config["steps"]):
                optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(layer(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite smoke-test loss")
                loss.backward()
                optimizer.step()
            final_loss = F.mse_loss(layer(inputs), targets).item()
            if not final_loss < initial_loss:
                raise RuntimeError(f"smoke loss did not decrease: {rule}/{initialization}")
            result["conditions"].append(
                {
                    "rule": rule,
                    "initialization": initialization,
                    "initial_output_max_error": initial_error,
                    "initial_loss": initial_loss,
                    "final_loss": final_loss,
                }
            )
    if device == "cuda":
        torch.cuda.synchronize()
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(yaml.safe_load(args.config.read_text()), args.device)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
