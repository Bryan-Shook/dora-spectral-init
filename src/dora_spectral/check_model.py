"""Measure initial and merged Qwen logits before committing the pilot budget."""

import argparse
from pathlib import Path

import torch

from .data import prompt, read_jsonl
from .experiment import environment, write_json
from .model import CONDITIONS, FP32Linear, insert_adapters, load_model, merge_adapters
from .prepare import verify_assets


def errors(actual, reference):
    difference = actual.float() - reference.float()
    return {
        "max_logit_error": float(difference.abs().max()),
        "relative_rms_logit_error": float(
            difference.square().mean().sqrt()
            / reference.float().square().mean().sqrt().clamp_min(1e-12)
        ),
        "next_token_agreement": float((actual.argmax(-1) == reference.argmax(-1)).float().mean()),
    }


@torch.inference_mode()
def check(root, device, precision, rank, gamma, tolerance=None):
    manifest = verify_assets(root)
    model, tokenizer = load_model(root, device, precision, "sdpa")
    model.eval()
    inputs = tokenizer(
        [prompt(row["question"]) for row in read_jsonl(root / "pilot.jsonl")[:4]],
        padding=True,
        return_tensors="pt",
    ).to(device)
    reference = model(**inputs).logits
    repeat = model(**inputs).logits
    originals = {
        name: layer
        for name, layer in model.named_modules()
        if name.endswith((".q_proj", ".v_proj"))
    }

    def replace(name, layer):
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, layer)

    # Establish the cost in logits of the intended FP32 target arithmetic itself.
    for name, layer in originals.items():
        replace(
            name,
            FP32Linear(
                torch.nn.Linear(
                    layer.in_features, layer.out_features, bias=layer.bias is not None
                ).to(device)
            ),
        )
        converted = model.get_submodule(name).linear
        converted.weight.copy_(layer.weight.float())
        if layer.bias is not None:
            converted.bias.copy_(layer.bias.float())
    arithmetic_reference = model(**inputs).logits
    for name, layer in originals.items():
        replace(name, layer)
    conditions = {}
    for condition in CONDITIONS:
        metadata = insert_adapters(
            model, condition, rank, gamma, 0, ["q_proj", "v_proj"], root / "svd-cache"
        )
        adapted = model(**inputs).logits
        merge_adapters(model)
        merged = model(**inputs).logits
        torch.testing.assert_close(adapted, merged, atol=1e-5, rtol=1e-5)
        conditions[condition] = {
            "against_base": errors(adapted, reference),
            "against_fp32_target_path": errors(adapted, arithmetic_reference),
            "merge": errors(merged, adapted),
            "metadata": metadata,
        }
        for name, layer in originals.items():
            replace(name, layer)
    maximum = max(row["against_base"]["max_logit_error"] for row in conditions.values())
    return {
        "assets": manifest,
        "environment": environment(device),
        "repeat_execution": errors(repeat, reference),
        "fp32_target_arithmetic_baseline": errors(arithmetic_reference, reference),
        "conditions": conditions,
        "max_logit_tolerance": tolerance,
        "accepted": None if tolerance is None else maximum <= tolerance,
        "note": "Prespecify BF16 tolerance using arithmetic/repeat baselines before training.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("data/qwen-gsm8k"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=1)
    parser.add_argument("--max-logit-tolerance", type=float)
    args = parser.parse_args()
    result = check(
        args.assets, args.device, args.precision, args.rank, args.gamma, args.max_logit_tolerance
    )
    write_json(args.output, result)
    print(f"Saved numerical checks to {args.output}; accepted={result['accepted']}")
    if result["accepted"] is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
