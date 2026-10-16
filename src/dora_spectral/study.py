"""Staged, resumable fixed-rank study. Official-test access requires locked confirmation."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from .reporting import environment, export_run, utc_now

CONDITIONS = ("A-S", "A-R", "D-S", "D-R", "E-S", "E-R", "L0", "D0")
CONTROLLED = CONDITIONS[:6]
DORA = ("D-S", "D-R", "E-S", "E-R")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("torch", "transformers", "peft", "datasets")
    }


def immutable_json(path, value):
    if path.exists() and read_json(path) != value:
        raise ValueError(f"locked file differs: {path}; use a new study for a changed protocol")
    write_json(path, value)


def read_study(directory, check_source=True, check_versions=False):
    manifest = read_json(directory / "study.json")
    if check_source and manifest["source_sha256"] != sources():
        raise ValueError("study source changed; keep the frozen code or create a new study")
    if check_versions and manifest["versions"] != versions():
        raise ValueError("study package versions changed")
    if read_json(directory / "config.json") != manifest["config"]:
        raise ValueError("study configuration changed")
    if yaml.safe_load((directory / "config.yaml").read_text()) != manifest["config"]:
        raise ValueError("study YAML configuration changed")
    frozen = directory / "frozen.json"
    if frozen.exists() and read_json(frozen)["study_sha256"] != sha(directory / "study.json"):
        raise ValueError("frozen study manifest changed")
    return manifest


def initialize(args):
    config = yaml.safe_load(args.config.read_text())
    prepared = read_json(args.data / "prepared.json")
    if sha(args.data / "examples.json") != prepared["examples_sha256"]:
        raise ValueError("prepared examples checksum mismatch")
    for key in ("model", "dataset", "max_length", "model_revision", "dataset_revision"):
        if config[key] != prepared["config"][key]:
            raise ValueError(f"preparation differs: {key}")
    examples = read_json(args.data / "examples.json")
    if len(examples["pilot"]) != 500 or len(examples["development"]) != 500:
        raise ValueError("the protocol requires 500 pilot and 500 development examples")
    if config["rank"] != 16 or config["epochs"] != 3 or not config.get("diagnostics"):
        raise ValueError("this core protocol requires rank 16, three epochs, and diagnostics")
    seeds = args.seeds
    if len(seeds) not in (3, 5) or len(set(seeds)) != len(seeds) or set(seeds) & {0, 1, 2, 10, 40}:
        raise ValueError("choose three or five distinct fresh confirmation seeds")
    if (
        not 0 <= args.previous_hours < args.budget_hours - args.reserve_hours
        or args.reserve_hours < 10
    ):
        raise ValueError(
            "invalid hours; reserve at least 10 hours and account for previous allocations"
        )
    if (
        not all(
            math.isfinite(value)
            for value in (args.budget_hours, args.reserve_hours, args.previous_hours, args.margin)
        )
        or args.margin <= 0
    ):
        raise ValueError("budget values and the positive practical margin must be finite")
    if args.directory.exists():
        raise ValueError("use a new study directory")
    epoch = math.ceil(len(examples["train"]) / config["batch_size"])
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "config": config,
        "data": str(args.data.resolve()),
        "data_sha256": prepared["examples_sha256"],
        "validation": str(args.validation.resolve()),
        "validation_sha256": sha(args.validation / "validation.json"),
        "source_sha256": sources(),
        "versions": versions(),
        "conditions": list(CONDITIONS),
        "epoch_steps": epoch,
        "horizon": epoch * 3,
        "tuning_seed": 10,
        "confirmation_seeds": seeds,
        "gammas": [0.1, 0.3, 1.0],
        "learning_rates": [2e-5, 1e-4, 5e-4, 2e-3],
        "baseline_learning_rates": [2e-6 * 1000 ** (i / 11) for i in range(12)],
        "practical_margin_pp": args.margin,
        "budget_hours": args.budget_hours,
        "reserve_hours": args.reserve_hours,
        "previous_hours": args.previous_hours,
        "pilot_ids": [row["id"] for row in examples["pilot"]],
        "development_ids": [row["id"] for row in examples["development"]],
        "environment": environment(),
    }
    manifest["baseline_learning_rates"][0] = 2e-6
    manifest["baseline_learning_rates"][-1] = 2e-3
    validation = read_json(args.validation / "validation.json")
    if (
        not validation["passed"]
        or validation["source_sha256"] != manifest["source_sha256"]
        or validation["data_sha256"] != manifest["data_sha256"]
        or validation["versions"] != manifest["versions"]
    ):
        raise ValueError("validation did not pass for these sources and data")
    if any(validation["config"].get(key) != value for key, value in config.items()):
        raise ValueError("shared configuration differs from the validated configuration")
    args.directory.mkdir(parents=True)
    write_json(args.directory / "study.json", manifest)
    write_json(args.directory / "config.json", config)
    (args.directory / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(
        args.validation / "validation.json", args.directory / "validation-evidence.json"
    )
    write_json(args.directory / "run.json", {"kind": "study", **manifest})
    export_run(args.directory)
    return manifest


def job(
    condition,
    seed,
    lr,
    gamma,
    steps,
    count,
    directory,
    split="development",
    start=0,
    kind="train",
    counterfactuals=False,
):
    return {
        "condition": condition,
        "seed": seed,
        "learning_rate": lr,
        "gamma": gamma,
        "steps": steps,
        "count": count,
        "directory": directory,
        "split": split,
        "from_steps": start,
        "kind": kind,
        "counterfactuals": counterfactuals,
    }


def candidate_name(condition, gamma, lr, seed):
    def slug(value):
        return format(value, ".12g").replace(".", "p").replace("-", "m").replace("+", "")

    return f"{condition}-g{slug(gamma)}-lr{slug(lr)}-s{seed}"


def search_jobs(manifest):
    jobs = []
    for condition in CONDITIONS:
        gammas = manifest["gammas"] if condition in CONTROLLED else [1.0]
        rates = (
            manifest["learning_rates"]
            if condition in CONTROLLED
            else manifest["baseline_learning_rates"]
        )
        for gamma in gammas:
            for lr in rates:
                name = candidate_name(condition, gamma, lr, manifest["tuning_seed"])
                jobs.append(
                    job(
                        condition,
                        manifest["tuning_seed"],
                        lr,
                        gamma,
                        manifest["epoch_steps"],
                        256,
                        f"search/{name}",
                    )
                )
    return jobs


def metric_path(directory, specification):
    path = directory / specification["directory"]
    return path / (
        f"{specification['split']}-step-{specification['steps']}/metrics.json"
        if specification["kind"] == "train"
        else "metrics.json"
    )


def complete(directory, manifest, specification, verify=False):
    path = directory / specification["directory"]
    metrics_path = metric_path(directory, specification)
    if not metrics_path.exists() or not (path / "run.json").exists():
        return False
    metadata, metrics = read_json(path / "run.json"), read_json(metrics_path)
    if (
        metadata.get("source_sha256") != manifest["source_sha256"]
        or metadata.get("data_sha256") != manifest["data_sha256"]
        or metadata.get("versions") != manifest["versions"]
    ):
        raise ValueError(f"job provenance mismatch: {path}")
    if (
        metadata["condition"] != specification["condition"]
        or metadata["seed"] != specification["seed"]
    ):
        raise ValueError(f"job identity mismatch: {path}")
    expected_config = (
        manifest["config"]
        if specification["condition"] == "BASE"
        else dict(
            manifest["config"],
            learning_rate=specification["learning_rate"],
            gamma=specification["gamma"],
            counterfactuals=specification["counterfactuals"],
        )
    )
    if metadata["config"] != expected_config:
        raise ValueError(f"job shared configuration mismatch: {path}")
    if specification["condition"] != "BASE" and any(
        metadata["config"][key] != specification[key] for key in ("learning_rate", "gamma")
    ):
        raise ValueError(f"job hyperparameters mismatch: {path}")
    if metrics["count"] != specification["count"]:
        return False
    if not all(math.isfinite(metrics[key]) for key in ("accuracy", "answer_nll", "seconds")):
        raise ValueError(f"nonfinite evaluation: {path}")
    if specification["kind"] == "train":
        if (
            not (path / "checkpoint.pt").exists()
            or read_json(path / "training-summary.json")["completed_steps"] < specification["steps"]
        ):
            return False
    if verify:
        rows = read_json(metrics_path.parent / "predictions.json")
        expected = (
            list(range(1319))
            if specification["split"] == "test"
            else manifest[f"{specification['split']}_ids"][: specification["count"]]
        )
        if [row["id"] for row in rows] != expected or any(
            type(row["correct"]) is not bool for row in rows
        ):
            raise ValueError(f"evaluation IDs/denominator differ: {path}")
        if abs(100 * sum(row["correct"] for row in rows) / len(rows) - metrics["accuracy"]) > 1e-9:
            raise ValueError(f"accuracy does not match saved predictions: {path}")
    return True


def numerical_failure(directory, specification):
    path = directory / specification["directory"] / "status.json"
    if not path.exists():
        return False
    status = read_json(path)
    message = status.get("error", {}).get("message", "").lower()
    return status["status"] == "failed" and (
        "nonfinite training loss" in message or "non-finite" in message
    )


def metrics_for(directory, manifest, jobs):
    result = []
    for specification in jobs:
        if complete(directory, manifest, specification, verify=True):
            result.append(
                {**specification, "metrics": read_json(metric_path(directory, specification))}
            )
        elif not numerical_failure(directory, specification):
            raise ValueError(f"unfinished job: {specification['directory']}")
    return result


def ranking(row):
    return (
        -row["metrics"]["accuracy"],
        row["metrics"]["answer_nll"],
        row["learning_rate"],
        row["gamma"],
    )


def promotions(directory, manifest):
    results = metrics_for(directory, manifest, search_jobs(manifest))
    promoted = []
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        if condition in CONTROLLED:
            chosen = [
                min(group, key=ranking)
                for gamma in manifest["gammas"]
                if (group := [row for row in rows if row["gamma"] == gamma])
            ]
        else:
            chosen = sorted(rows, key=ranking)[:3]
        if not chosen:
            raise ValueError(f"all candidates failed for {condition}")
        for row in chosen:
            promoted.append(
                {
                    **{key: value for key, value in row.items() if key != "metrics"},
                    "steps": manifest["horizon"],
                    "count": 500,
                    "from_steps": manifest["epoch_steps"],
                }
            )
    immutable_json(directory / "promotions.json", promoted)
    return promoted


def selection(directory, manifest):
    if (directory / "test-opened.json").exists():
        raise ValueError("selection is closed after official-test access")
    results = metrics_for(directory, manifest, read_json(directory / "promotions.json"))
    winners = {}
    for condition in CONDITIONS:
        rows = [row for row in results if row["condition"] == condition]
        if not rows:
            raise ValueError(f"no full-run candidate for {condition}")
        winners[condition] = min(rows, key=ranking)
    boundaries = {}
    for condition, row in winners.items():
        rates = (
            manifest["learning_rates"]
            if condition in CONTROLLED
            else manifest["baseline_learning_rates"]
        )
        boundaries[condition] = {
            "lr_boundary": row["learning_rate"] in (min(rates), max(rates)),
            "gamma_boundary": condition in CONTROLLED
            and row["gamma"] in (min(manifest["gammas"]), max(manifest["gammas"])),
        }
    value = {
        "study_sha256": sha(directory / "study.json"),
        "winners": winners,
        "full_run_finalists": results,
        "boundaries": boundaries,
        "scope": "one tuning seed; bounded search; one-epoch screening can discard slow starters",
    }
    immutable_json(directory / "selection.json", value)
    immutable_json(
        directory / "selection-lock.json", {"selection_sha256": sha(directory / "selection.json")}
    )
    return value


def confirmation_jobs(directory, manifest):
    selected = read_json(directory / "selection.json")
    if read_json(directory / "selection-lock.json")["selection_sha256"] != sha(
        directory / "selection.json"
    ):
        raise ValueError("selected settings changed after locking")
    if selected["study_sha256"] != sha(directory / "study.json"):
        raise ValueError("selection manifest changed")
    jobs = []
    for seed in manifest["confirmation_seeds"]:
        for condition in CONDITIONS:
            winner = selected["winners"][condition]
            for epoch, count in ((1, 128), (2, 128), (3, 500)):
                jobs.append(
                    job(
                        condition,
                        seed,
                        winner["learning_rate"],
                        winner["gamma"],
                        epoch * manifest["epoch_steps"],
                        count,
                        f"confirmation/{condition}-s{seed}",
                        start=(epoch - 1) * manifest["epoch_steps"],
                    )
                )
    immutable_json(
        directory / "confirmation-plan.json",
        {"selection_sha256": sha(directory / "selection.json"), "jobs": jobs},
    )
    return jobs


def test_jobs(directory, manifest):
    selected = read_json(directory / "selection.json")["winners"]
    jobs = [job("BASE", 0, 0, 1, 0, 1319, "test/BASE", split="test", kind="test")]
    for seed in manifest["confirmation_seeds"]:
        for condition in CONDITIONS:
            winner = selected[condition]
            jobs.append(
                job(
                    condition,
                    seed,
                    winner["learning_rate"],
                    winner["gamma"],
                    0,
                    1319,
                    f"test/{condition}-s{seed}",
                    split="test",
                    kind="test",
                )
            )
    return jobs


def authorize_test(directory, manifest):
    if not (directory / "frozen.json").exists():
        raise ValueError("freeze the study before testing")
    plan = read_json(directory / "confirmation-plan.json")
    if plan["selection_sha256"] != sha(directory / "selection.json"):
        raise ValueError("selection changed after confirmation started")
    for specification in confirmation_jobs(directory, manifest):
        if not complete(directory, manifest, specification, verify=True):
            raise ValueError(
                "all locked paired confirmation runs must finish before official-test access"
            )
    immutable_json(
        directory / "test-opened.json", {"selection_sha256": sha(directory / "selection.json")}
    )


def diagnostic_jobs(manifest):
    return [
        job(
            condition,
            40,
            1e-4,
            1,
            100,
            16,
            f"diagnostics/{condition}-s40",
            split="pilot",
            counterfactuals=True,
        )
        for condition in ("E-S", "E-R", "D-S", "D-R")
    ]


def ledger(directory):
    path = directory / "ledger.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def spent_hours(directory, manifest):
    validation = read_json(directory / "validation-evidence.json")
    return (
        manifest["previous_hours"]
        + validation["seconds"] / 3600
        + sum(row["seconds"] for row in ledger(directory)) / 3600
    )


def measured_rates(directory, manifest):
    validation = read_json(directory / "validation-evidence.json")
    generation = [
        validation["baseline_generation"]["seconds"] / validation["baseline_generation"]["count"]
    ]
    training, overhead = [], [60.0]
    for path in directory.rglob("training-summary.json"):
        if "diagnostics" in path.relative_to(directory).parts:
            continue
        row = read_json(path)
        training.append(row["seconds"] / row["completed_steps"])
    for path in directory.rglob("metrics.json"):
        row = read_json(path)
        generation.append(row["seconds"] / row["count"])
    charges = {}
    for entry in ledger(directory):
        if "directory" in entry:
            charges.setdefault(entry["directory"], []).append(entry["seconds"])
    for name, candidate_charges in charges.items():
        candidate = directory / name
        summary_path = candidate / "training-summary.json"
        if summary_path.exists():
            training_seconds = read_json(summary_path)["seconds"]
            generation_seconds = sum(
                read_json(path)["seconds"] for path in candidate.rglob("metrics.json")
            )
            overhead.append(
                max(
                    0,
                    (sum(candidate_charges) - training_seconds - generation_seconds)
                    / len(candidate_charges),
                )
            )
    if not training:
        raise ValueError("pilot timings are needed before freezing")
    return {
        "step_seconds": max(training),
        "question_seconds": max(generation),
        "overhead_seconds": max(overhead),
    }


def estimate(specification, rates):
    return (
        (specification["steps"] - specification["from_steps"]) * rates["step_seconds"]
        + specification["count"] * rates["question_seconds"]
        + rates["overhead_seconds"]
    )


def remaining_seconds(directory, manifest, rates):
    outstanding = [
        item
        for item in search_jobs(manifest)
        if not complete(directory, manifest, item) and not numerical_failure(directory, item)
    ]
    if (directory / "promotions.json").exists():
        outstanding += [
            item
            for item in read_json(directory / "promotions.json")
            if not complete(directory, manifest, item) and not numerical_failure(directory, item)
        ]
    else:
        outstanding += [
            job(
                "pending",
                10,
                0,
                0,
                manifest["horizon"],
                500,
                "pending",
                start=manifest["epoch_steps"],
            )
        ] * 24
    if (directory / "selection.json").exists():
        outstanding += [
            item
            for item in confirmation_jobs(directory, manifest)
            if not complete(directory, manifest, item)
        ]
        outstanding += [
            item
            for item in test_jobs(directory, manifest)
            if not complete(directory, manifest, item)
        ]
    else:
        outstanding += [
            job("pending", 0, 0, 0, manifest["epoch_steps"], count, "pending")
            for _ in range(len(manifest["confirmation_seeds"]) * 8)
            for count in (128, 128, 500)
        ]
        outstanding += [job("pending", 0, 0, 0, 0, 1319, "pending", kind="test")] * (
            len(manifest["confirmation_seeds"]) * 8 + 1
        )
    outstanding += [
        item for item in diagnostic_jobs(manifest) if not complete(directory, manifest, item)
    ]
    return sum(estimate(item, rates) for item in outstanding)


def freeze(directory, manifest):
    pilot = read_json(directory / "pilot.json")
    validation_path = directory / "validation-evidence.json"
    if (
        sha(validation_path) != manifest["validation_sha256"]
        or not read_json(validation_path)["passed"]
        or not pilot["passed"]
    ):
        raise ValueError("validation and complete pilot must pass before freezing")
    rates = measured_rates(directory, manifest)
    forecast = remaining_seconds(directory, manifest, rates) / 3600
    spent = spent_hours(directory, manifest)
    if spent + forecast + manifest["reserve_hours"] > manifest["budget_hours"]:
        raise ValueError(
            f"budget gate failed: {spent:.2f} spent + {forecast:.2f} forecast + reserve exceeds cap"
        )
    value = {
        "study_sha256": sha(directory / "study.json"),
        "rates": rates,
        "spent_hours_at_freeze": spent,
        "remaining_forecast_hours": forecast,
        "scope": "core eight-condition study; no optional 200-hour extensions",
    }
    immutable_json(directory / "frozen.json", value)
    return value


def launch(command, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as log:
        with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise


def run_jobs(args, manifest, jobs, phase):
    began = getattr(args, "stage_began", time.monotonic())
    for index, specification in enumerate(jobs):
        if complete(args.directory, manifest, specification):
            continue
        if phase in ("screen", "promote") and numerical_failure(args.directory, specification):
            continue
        remaining = args.wall_hours * 3600 - (time.monotonic() - began) - 120
        seconds = min(args.job_seconds, int(remaining))
        if seconds <= 120:
            raise RuntimeError("allocation time limit reached; repeat the same stage to resume")
        if (args.directory / "frozen.json").exists():
            rates = measured_rates(args.directory, manifest)
            if (
                spent_hours(args.directory, manifest)
                + remaining_seconds(args.directory, manifest, rates) / 3600
                + manifest["reserve_hours"]
                > manifest["budget_hours"]
            ):
                raise RuntimeError(
                    "updated forecast exceeds cap while preserving confirmation/reserve; "
                    "stop and review scope"
                )
        elif (
            spent_hours(args.directory, manifest) + seconds / 3600 + manifest["reserve_hours"]
            > manifest["budget_hours"]
        ):
            raise RuntimeError("pilot allocation would consume the study reserve")
        path = args.directory / specification["directory"]
        command = [
            sys.executable,
            "-m",
            "dora_spectral.experiment",
            "--config",
            str(args.directory / "config.yaml"),
        ]
        if specification["kind"] == "test":
            command += ["score-test", "--study", str(args.directory)]
            if specification["condition"] == "BASE":
                command += ["--base"]
            else:
                command += [
                    "--checkpoint",
                    str(
                        args.directory
                        / f"confirmation/{specification['condition']}-s{specification['seed']}"
                    ),
                ]
        else:
            command += [
                "run",
                "--condition",
                specification["condition"],
                "--seed",
                str(specification["seed"]),
                "--lr",
                repr(specification["learning_rate"]),
                "--gamma",
                repr(specification["gamma"]),
                "--steps",
                str(specification["steps"]),
                "--split",
                specification["split"],
                "--evaluation-count",
                str(specification["count"]),
            ]
            if specification["counterfactuals"]:
                command += ["--counterfactuals"]
        command += ["--data", manifest["data"], "--directory", str(path), "--seconds", str(seconds)]
        print(
            f"{phase}: {index + 1}/{len(jobs)} {specification['directory']} "
            f"-> {specification['steps']} steps",
            flush=True,
        )
        started = time.monotonic()
        exit_code = None
        try:
            exit_code = launch(command, path / "console.log")
        finally:
            with (args.directory / "ledger.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "phase": phase,
                            "directory": specification["directory"],
                            "seconds": time.monotonic() - started,
                            "exit_code": exit_code,
                            "finished_at": utc_now(),
                        }
                    )
                    + "\n"
                )
            if (path / "run.json").exists():
                export_run(path)
        if exit_code != 0 or not complete(args.directory, manifest, specification):
            if phase in ("screen", "promote") and numerical_failure(args.directory, specification):
                continue
            raise RuntimeError(
                f"job unfinished/failed; inspect {path}/console.log, then resume the stage"
            )


def pilot(args, manifest):
    jobs = [
        job(
            condition,
            seed,
            1e-4,
            1,
            100 if condition in DORA else 20,
            16,
            f"pilot/{condition}-s{seed}",
            split="pilot",
        )
        for condition in CONDITIONS
        for seed in ((0, 1, 2) if condition in DORA else (0,))
    ]
    run_jobs(args, manifest, jobs, "pilot")
    durations = {}
    for condition in CONDITIONS:
        rows = [
            read_json(args.directory / item["directory"] / "training-summary.json")
            for item in jobs
            if item["condition"] == condition
        ]
        durations[condition] = sum(row["seconds"] / row["completed_steps"] for row in rows) / len(
            rows
        )
    choice = args.directory / "pilot-choice.json"
    if choice.exists():
        slowest = read_json(choice)["condition"]
    else:
        slowest = max(durations, key=durations.get)
        immutable_json(choice, {"condition": slowest, "seconds_per_step": durations})
    full = job(
        slowest,
        0,
        1e-4,
        1,
        manifest["horizon"],
        500,
        f"pilot/{slowest}-s0",
        split="pilot",
        start=100 if slowest in DORA else 20,
    )
    run_jobs(args, manifest, [full], "pilot-full")
    metrics = read_json(metric_path(args.directory, full))
    validation = read_json(args.directory / "validation-evidence.json")
    passed = (
        metrics["answer_nll"] < validation["baseline_pilot_nll"]
        and 5 <= metrics["accuracy"] <= 90
        and metrics["truncation_rate"] <= 0.01
    )
    write_json(
        args.directory / "pilot.json",
        {
            "passed": passed,
            "full_job": full,
            "metrics": metrics,
            "baseline_nll": validation["baseline_pilot_nll"],
        },
    )
    if not passed:
        raise ValueError("pilot learning/generation gate needs investigation before freeze")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runs/study-main"))
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create the proposed protocol after validation")
    init.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    init.add_argument("--data", type=Path, default=Path("runs/data"))
    init.add_argument("--validation", type=Path, default=Path("runs/validation-h100"))
    init.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    init.add_argument("--budget-hours", type=float, default=100)
    init.add_argument("--reserve-hours", type=float, default=10)
    init.add_argument("--previous-hours", type=float, default=0)
    init.add_argument("--margin", type=float, default=2)
    for name in ("pilot", "screen", "promote", "confirm", "test", "diagnose"):
        stage = commands.add_parser(name)
        stage.add_argument(
            "--wall-hours", type=float, default=4, help="less than remaining GPU allocation"
        )
        stage.add_argument("--job-seconds", type=int, default=3600)
    for name in ("freeze", "select", "status", "analyze"):
        stage = commands.add_parser(name)
        if name == "analyze":
            stage.add_argument("--resamples", type=int, default=10000)
            stage.add_argument("--two-way", action="store_true")
    account = commands.add_parser(
        "account", help="charge additional idle/setup GPU allocation hours"
    )
    account.add_argument("--hours", type=float, required=True)
    account.add_argument("--note", required=True)
    args = parser.parse_args()
    args.stage_began = time.monotonic()
    if hasattr(args, "wall_hours") and (
        not math.isfinite(args.wall_hours) or args.wall_hours <= 0 or args.job_seconds <= 120
    ):
        parser.error("wall hours must be finite/positive and job seconds must exceed 120")
    if args.command == "init":
        initialize(args)
        return 0
    manifest = read_study(
        args.directory,
        check_source=args.command not in ("status", "analyze"),
        check_versions=args.command
        in ("pilot", "screen", "promote", "confirm", "test", "diagnose"),
    )
    try:
        if args.command == "pilot":
            pilot(args, manifest)
        elif args.command == "freeze":
            print(json.dumps(freeze(args.directory, manifest), indent=2))
        elif args.command in ("screen", "promote", "select", "confirm", "test", "diagnose"):
            if not (args.directory / "frozen.json").exists():
                raise ValueError("complete pilot and freeze the study first")
            if args.command == "select":
                selection(args.directory, manifest)
            else:
                if args.command == "screen":
                    jobs = search_jobs(manifest)
                elif args.command == "promote":
                    jobs = promotions(args.directory, manifest)
                elif args.command == "confirm":
                    jobs = confirmation_jobs(args.directory, manifest)
                elif args.command == "diagnose":
                    jobs = diagnostic_jobs(manifest)
                else:
                    authorize_test(args.directory, manifest)
                    immutable_json(
                        args.directory / "test-opened.json",
                        {"selection_sha256": sha(args.directory / "selection.json")},
                    )
                    jobs = test_jobs(args.directory, manifest)
                run_jobs(args, manifest, jobs, args.command)
        elif args.command == "account":
            if not math.isfinite(args.hours) or args.hours < 0:
                raise ValueError("hours must be finite and nonnegative")
            with (args.directory / "ledger.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "phase": "extra-allocation",
                            "seconds": args.hours * 3600,
                            "note": args.note,
                            "finished_at": utc_now(),
                        }
                    )
                    + "\n"
                )
        elif args.command == "status":
            print(
                json.dumps(
                    {
                        "conditions": manifest["conditions"],
                        "spent_accounted_hours": spent_hours(args.directory, manifest),
                        "frozen": (args.directory / "frozen.json").exists(),
                        "selected": (args.directory / "selection.json").exists(),
                        "test_opened": (args.directory / "test-opened.json").exists(),
                    },
                    indent=2,
                )
            )
        elif args.command == "analyze":
            from .analysis import analyze

            analyze(args.directory, manifest, args.resamples, args.two_way)
    finally:
        if not args.directory.resolve().is_relative_to(Path("reports").resolve()):
            export_run(args.directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
