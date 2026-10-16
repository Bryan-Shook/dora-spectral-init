"""Prepare data or train and score one explicitly selected experiment condition."""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import signal
import sys
import time
from pathlib import Path

import torch
import yaml

from .data import encode_example, parse_answer, split_examples
from .model import CONDITIONS, attach
from .reporting import environment, export_run, utc_now
from .streams import STREAM_VERSION, data_order_stream
from .training import Budget, BudgetExpired, answer_nll, evaluate, load_model, train, write_json


def prepare(config, directory):
    """CPU-only downloads/preprocessing; official test is never loaded."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    directory.mkdir(parents=True, exist_ok=True)
    protocol_path = directory / "prepared.json"
    if protocol_path.exists():
        return load_prepared(config, directory)[0]
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


def load_prepared(config, directory):
    prepared = json.loads((directory / "prepared.json").read_text())
    encoded_path = directory / "examples.json"
    if hashlib.sha256(encoded_path.read_bytes()).hexdigest() != prepared["examples_sha256"]:
        raise ValueError("prepared-data checksum mismatch")
    for key in ("model", "dataset", "max_length", "validation_examples"):
        if key not in config and key not in prepared["config"]:
            continue
        if config[key] != prepared["config"][key]:
            raise ValueError(f"prepared-data configuration mismatch: {key}")
    for key in ("model_revision", "dataset_revision"):
        if config[key] != prepared[key]:
            raise ValueError(f"prepared-data revision mismatch: {key}")
    return prepared, json.loads(encoded_path.read_text())


def source_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }


def runtime_versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "datasets")
    }


def run_metadata(config, condition, seed, prepared):
    return {
        "backend": "peft-controlled-v1",
        "config": config.copy(),
        "condition": condition,
        "seed": seed,
        "random_streams": {
            "version": STREAM_VERSION,
            "data_order_seed": data_order_stream(seed),
        },
        "data_sha256": prepared["examples_sha256"],
        "source_sha256": source_hashes(),
        "versions": runtime_versions(),
    }


def write_metadata(directory, metadata):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run.json"
    if path.exists() and json.loads(path.read_text()) != metadata:
        raise ValueError("run directory already belongs to a different experiment")
    write_json(path, metadata)


@contextlib.contextmanager
def record_invocation(directory, reports, budget, **details):
    invocation = {
        "started_at": utc_now(),
        "status": "running",
        "environment": environment(),
        **details,
    }
    write_json(directory / "status.json", invocation)
    try:
        yield invocation
    except BudgetExpired:
        invocation["status"] = "interrupted"
    except BaseException as error:
        invocation.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
        raise
    finally:
        invocation.update(seconds=budget.elapsed, finished_at=utc_now())
        with (directory / "invocations.jsonl").open("a") as handle:
            handle.write(json.dumps(invocation) + "\n")
        write_json(directory / "status.json", invocation)
        try:
            destination = export_run(directory, reports)
            print(f"Experiment evidence: {destination}", flush=True)
        except (OSError, ValueError) as error:
            if invocation["status"] == "failed":
                print(f"Evidence export failed: {error}", file=sys.stderr, flush=True)
            else:
                raise


def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("run requires a GPU with native BF16 support")


@contextlib.contextmanager
def phase_time(invocation, name):
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    began = time.monotonic()
    try:
        yield
    finally:
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        invocation.setdefault("phase_seconds", {})[name] = time.monotonic() - began


def run(args, config):
    from transformers import AutoTokenizer

    prepared, examples = load_prepared(config, args.data)
    require_gpu()
    if args.directory.resolve() == args.data.resolve():
        raise ValueError("use a separate output directory for each candidate")
    if not math.isfinite(args.lr) or not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("learning rate and scale must be finite and positive")
    if not math.isfinite(args.seconds):
        raise ValueError("time budget must be finite")
    if args.evaluation_count is not None and args.evaluation_count <= 0:
        raise ValueError("evaluation count must be positive")
    config = dict(
        config,
        learning_rate=args.lr,
        gamma=args.gamma,
        counterfactuals=getattr(args, "counterfactuals", False),
    )
    horizon = math.ceil(len(examples["train"]) / config["batch_size"]) * config["epochs"]
    steps = horizon if args.steps is None else args.steps
    if not 1 <= steps <= horizon or args.seconds <= 120 or args.lr <= 0:
        raise ValueError("invalid steps, time budget, or learning rate")
    metadata = run_metadata(config, args.condition, args.seed, prepared)
    write_metadata(args.directory, metadata)
    budget = Budget(args.seconds)

    def stop(signum, frame):
        # Stop at a safe accumulation boundary, never halfway through optimizer.step().
        budget.cancelled = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with record_invocation(
        args.directory,
        args.reports,
        budget,
        target_steps=steps,
        split=args.split,
        evaluation_count=args.evaluation_count,
    ) as invocation:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["model_revision"]
        )
        tokenizer.pad_token = tokenizer.eos_token
        config["pad_id"] = tokenizer.pad_token_id
        with phase_time(invocation, "model_loading"):
            model = load_model(config, prepared, "cuda")
        # Revision/rank cache is shared across candidate output directories.
        cache = args.data / "svd" / prepared["model_revision"] / str(config["rank"])
        with phase_time(invocation, "initialization"):
            audit = attach(model, args.condition, config["rank"], args.gamma, args.seed, cache)
        write_json(args.directory / "initialization.json", audit)
        with phase_time(invocation, "training_and_checkpoint_io"):
            train(
                model, examples["train"], config, args.seed, steps, args.directory, budget, metadata
            )
        evaluation_dir = args.directory / f"{args.split}-step-{steps}"
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        rows = examples[args.split]
        with phase_time(invocation, "answer_nll"):
            nll = answer_nll(model, rows, config, budget)
        scored = rows[: args.evaluation_count] if args.evaluation_count else rows
        with phase_time(invocation, "generation"):
            result = evaluate(model, tokenizer, scored, config, evaluation_dir, budget)
        write_json(evaluation_dir / "metrics.json", dict(result, answer_nll=nll))
        invocation["status"] = "complete"
    return 0 if invocation["status"] == "complete" else 1


def prepare_test(config, directory):
    """Called only after study authorization; never filter the official test questions."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    path, metadata_path = directory / "test-examples.json", directory / "test-prepared.json"
    expected = {
        key: config[key] for key in ("model", "model_revision", "dataset", "dataset_revision")
    }
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (
            metadata["revisions"] != expected
            or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]
        ):
            raise ValueError("official-test preparation provenance mismatch")
        return json.loads(path.read_text()), metadata
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    dataset = load_dataset(
        "parquet",
        data_files={
            "test": f"hf://datasets/{config['dataset']}@"
            f"{config['dataset_revision']}/main/test-00000-of-00001.parquet"
        },
        split="test",
    )
    if len(dataset) != 1319:
        raise ValueError("expected all 1,319 GSM8K test questions")
    rows = [encode_example(tokenizer, {**row, "id": index}) for index, row in enumerate(dataset)]
    if any(parse_answer(row["answer"]) is None for row in rows):
        raise ValueError("invalid test reference answer")
    write_json(path, rows)
    metadata = {
        "revisions": expected,
        "count": len(rows),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    write_json(metadata_path, metadata)
    return rows, metadata


def score_test(args, config):
    from transformers import AutoTokenizer

    from .model import load_adapter_state
    from .study import authorize_test, read_study

    manifest = read_study(args.study, check_versions=True)
    authorize_test(args.study, manifest)
    if config != manifest["config"]:
        raise ValueError("official-test shared configuration differs from frozen study")
    prepared, _ = load_prepared(config, args.data)
    require_gpu()
    if args.base:
        condition, seed = "BASE", 0
        checkpoint = None
    else:
        checkpoint = torch.load(
            args.checkpoint / "checkpoint.pt", map_location="cpu", weights_only=False
        )
        saved = checkpoint["metadata"]
        condition, seed = saved["condition"], saved["seed"]
        expected_path = args.study / f"confirmation/{condition}-s{seed}"
        if args.checkpoint.resolve() != expected_path.resolve():
            raise ValueError("use the locked study's confirmation checkpoint directory")
        selected = json.loads((args.study / "selection.json").read_text())["winners"][condition]
        if saved != run_metadata(saved["config"], condition, seed, prepared):
            raise ValueError("checkpoint code, environment, or data no longer matches")
        if seed not in manifest["confirmation_seeds"] or checkpoint["step"] != manifest["horizon"]:
            raise ValueError("official testing requires a completed confirmation checkpoint")
        if any(saved["config"][key] != selected[key] for key in ("learning_rate", "gamma")):
            raise ValueError("checkpoint is not the locked selected configuration")
        config = saved["config"].copy()
    metadata = run_metadata(config, condition, seed, prepared)
    metadata["selection_sha256"] = hashlib.sha256(
        (args.study / "selection.json").read_bytes()
    ).hexdigest()
    if checkpoint is not None:
        metadata["checkpoint_sha256"] = hashlib.sha256(
            (args.checkpoint / "checkpoint.pt").read_bytes()
        ).hexdigest()
    write_metadata(args.directory, metadata)
    budget = Budget(args.seconds)
    with record_invocation(args.directory, args.reports, budget, split="test") as invocation:
        rows, test_metadata = prepare_test(config, args.data)
        write_json(args.directory / "test-prepared.json", test_metadata)
        tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["model_revision"]
        )
        tokenizer.pad_token = tokenizer.eos_token
        config = dict(config, pad_id=tokenizer.pad_token_id)
        with phase_time(invocation, "model_loading"):
            model = load_model(config, prepared, "cuda")
        if checkpoint is not None:
            cache = args.data / "svd" / prepared["model_revision"] / str(config["rank"])
            with phase_time(invocation, "initialization_and_checkpoint_loading"):
                attach(model, condition, config["rank"], config["gamma"], seed, cache)
                load_adapter_state(model, checkpoint["adapters"])
        with phase_time(invocation, "answer_nll"):
            nll = answer_nll(model, rows, config, budget)
        with phase_time(invocation, "generation"):
            result = evaluate(model, tokenizer, rows, config, args.directory, budget)
        write_json(args.directory / "metrics.json", dict(result, answer_nll=nll))
        invocation["status"] = "complete"
    return 0 if invocation["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=Path("runs/data"))
    cell = commands.add_parser("run")
    cell.add_argument("--data", type=Path, default=Path("runs/data"))
    cell.add_argument("--directory", type=Path, required=True)
    cell.add_argument(
        "--reports", type=Path, default=Path("reports"), help="small-result report root"
    )
    cell.add_argument("--condition", choices=CONDITIONS, required=True)
    cell.add_argument("--seed", type=int, required=True)
    cell.add_argument("--lr", type=float, required=True)
    cell.add_argument("--gamma", type=float, default=1)
    cell.add_argument("--steps", type=int, help="total completed steps; resume to this target")
    cell.add_argument("--seconds", type=float, required=True, help="time cap for this invocation")
    cell.add_argument("--split", choices=("pilot", "development"), default="pilot")
    cell.add_argument("--evaluation-count", type=int, help="score first N saved split IDs")
    cell.add_argument(
        "--counterfactuals", action="store_true", help="dedicated shared-state DoRA diagnostics"
    )
    score = commands.add_parser("score-test", help="evaluate locked confirmation checkpoints")
    score.add_argument("--study", type=Path, required=True)
    score.add_argument("--data", type=Path, default=Path("runs/data"))
    score.add_argument("--checkpoint", type=Path)
    score.add_argument("--base", action="store_true")
    score.add_argument("--directory", type=Path, required=True)
    score.add_argument("--reports", type=Path, default=Path("reports"))
    score.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.command == "prepare":
        prepare(config, args.data)
        return 0
    if args.command == "score-test":
        if args.base == (args.checkpoint is not None) or args.seconds <= 120:
            parser.error("choose exactly one of --base / --checkpoint, and seconds > 120")
        return score_test(args, config)
    return run(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
