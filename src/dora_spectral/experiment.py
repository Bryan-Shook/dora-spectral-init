"""Prepare data or train and score one explicitly selected experiment condition."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import signal
from pathlib import Path

import torch
import yaml

from .data import encode_example, parse_answer, split_examples
from .model import CONDITIONS, attach
from .training import Budget, BudgetExpired, answer_nll, evaluate, load_model, train, write_json


def prepare(config, directory):
    """CPU-only downloads/preprocessing; official test is never loaded."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "prepared.json"
    if protocol_path.exists():
        saved = json.loads(protocol_path.read_text())
        if saved["config"] != config:
            raise ValueError("existing preparation uses a different config; choose a new directory")
        return saved
    model_revision = config["model_revision"]
    dataset_revision = config["dataset_revision"]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=model_revision)
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"hf://datasets/{config['dataset']}@"
            f"{dataset_revision}/main/train-00000-of-00001.parquet"
        },
        split="train",
    )
    partitions = split_examples(list(dataset))
    encoded = {
        key: [encode_example(tokenizer, row) for row in rows] for key, rows in partitions.items()
    }
    excluded = [
        row["id"] for row in encoded["train"] if len(row["input_ids"]) > config["max_length"]
    ]
    encoded["train"] = [
        row for row in encoded["train"] if len(row["input_ids"]) <= config["max_length"]
    ]
    if len(encoded["pilot"]) != config["validation_examples"]:
        raise ValueError("duplicate grouping changed pilot count; revise/freeze config explicitly")
    if len(encoded["train"]) < 32:
        raise ValueError("not enough training examples after length filtering")
    for rows in partitions.values():
        if any(parse_answer(row["answer"]) is None for row in rows):
            raise ValueError("dataset contains an unparseable reference answer")
    encoded_path = directory / "examples.json"
    write_json(encoded_path, encoded)
    saved = {
        "config": config,
        "model_revision": model_revision,
        "dataset_revision": dataset_revision,
        "counts": {key: len(rows) for key, rows in encoded.items()},
        "split_ids": {key: [row["id"] for row in rows] for key, rows in partitions.items()},
        "excluded_training_ids": excluded,
        "lengths": {
            key: {
                "max": max(len(row["input_ids"]) for row in rows),
                "mean": sum(len(row["input_ids"]) for row in rows) / len(rows),
            }
            for key, rows in encoded.items()
        },
        "examples_sha256": hashlib.sha256(encoded_path.read_bytes()).hexdigest(),
    }
    write_json(protocol_path, saved)
    return saved


def run(args, config):
    from transformers import AutoTokenizer

    prepared = json.loads((args.data / "prepared.json").read_text())
    encoded_path = args.data / "examples.json"
    if hashlib.sha256(encoded_path.read_bytes()).hexdigest() != prepared["examples_sha256"]:
        raise ValueError("prepared-data checksum mismatch")
    for key in ("model", "dataset", "max_length"):
        if config[key] != prepared["config"][key]:
            raise ValueError(f"prepared-data configuration mismatch: {key}")
    for key in ("model_revision", "dataset_revision"):
        if config[key] != prepared[key]:
            raise ValueError(f"prepared-data revision mismatch: {key}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("run requires a GPU with native BF16 support")
    if args.directory.resolve() == args.data.resolve():
        raise ValueError("use a separate output directory for each candidate")
    if not math.isfinite(args.lr) or not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("learning rate and scale must be finite and positive")
    if not math.isfinite(args.seconds):
        raise ValueError("time budget must be finite")
    if args.evaluation_count is not None and args.evaluation_count <= 0:
        raise ValueError("evaluation count must be positive")
    config = dict(config, learning_rate=args.lr, gamma=args.gamma)
    examples = json.loads(encoded_path.read_text())
    horizon = math.ceil(len(examples["train"]) / config["batch_size"]) * config["epochs"]
    steps = horizon if args.steps is None else args.steps
    if not 1 <= steps <= horizon or args.seconds <= 120 or args.lr <= 0:
        raise ValueError("invalid steps, time budget, or learning rate")
    metadata = {
        "backend": "peft-controlled-v1",
        "config": config.copy(),
        "condition": args.condition,
        "seed": args.seed,
        "data_sha256": prepared["examples_sha256"],
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "datasets")
        },
    }
    args.directory.mkdir(parents=True, exist_ok=True)
    metadata_path = args.directory / "run.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError("run directory already belongs to a different experiment")
    write_json(metadata_path, metadata)
    budget = Budget(args.seconds)

    def stop(signum, frame):
        # Stop at a safe accumulation boundary, never halfway through optimizer.step().
        budget.cancelled = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["model_revision"]
        )
        tokenizer.pad_token = tokenizer.eos_token
        config["pad_id"] = tokenizer.pad_token_id
        model = load_model(config, prepared, "cuda")
        # Revision/rank cache is shared across candidate output directories.
        cache = args.data / "svd" / prepared["model_revision"] / str(config["rank"])
        audit = attach(model, args.condition, config["rank"], args.gamma, args.seed, cache)
        write_json(args.directory / "initialization.json", audit)
        train(model, examples["train"], config, args.seed, steps, args.directory, budget, metadata)
        evaluation_dir = args.directory / f"{args.split}-step-{steps}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        rows = examples[args.split]
        nll = answer_nll(model, rows, config, budget)
        scored = rows[: args.evaluation_count] if args.evaluation_count else rows
        result = evaluate(model, tokenizer, scored, config, evaluation_dir, budget)
        write_json(evaluation_dir / "metrics.json", dict(result, answer_nll=nll))
        status = "complete"
    except BudgetExpired:
        status = "interrupted"
    finally:
        # Each invocation's cost is retained, including interrupted/resumed jobs.
        with (args.directory / "invocations.jsonl").open("a") as handle:
            handle.write(
                json.dumps({"seconds": budget.elapsed, "status": locals().get("status", "failed")})
                + "\n"
            )
    return 0 if status == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=Path("runs/data"))
    cell = commands.add_parser("run")
    cell.add_argument("--data", type=Path, default=Path("runs/data"))
    cell.add_argument("--directory", type=Path, required=True)
    cell.add_argument("--condition", choices=CONDITIONS, required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--lr", type=float, required=True)
    cell.add_argument("--gamma", type=float, default=1)
    cell.add_argument("--steps", type=int, help="total completed steps; resume to this target")
    cell.add_argument("--seconds", type=float, required=True, help="time cap for this invocation")
    cell.add_argument("--split", choices=("pilot", "development"), default="pilot")
    cell.add_argument("--evaluation-count", type=int, help="score first N saved split IDs")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.command == "prepare":
        prepare(config, args.data)
        return 0
    return run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
