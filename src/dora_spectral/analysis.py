"""Paired-seed inference, paired-item bootstrap, CSV tables, and scientific figures."""

import csv
import importlib.metadata
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from .data import parse_answer
from .reporting import report_directory
from .study import (
    CONDITIONS,
    CONTROLLED,
    complete,
    read_json,
    runtime_accounting,
    runtime_by_stage,
    sha,
    sources,
    test_jobs,
    write_json,
)


def contrast_arrays(correct):
    values = {condition: correct[:, index, :] for index, condition in enumerate(CONDITIONS)}
    additive = values["A-S"] - values["A-R"]
    detached = values["D-S"] - values["D-R"]
    exact = values["E-S"] - values["E-R"]
    return {
        "delta_A": additive,
        "delta_D": detached,
        "J": detached - additive,
        "delta_E": exact,
        "I": exact - detached,
        "exact_vs_additive": exact - additive,
    }


def interval_interpretation(lower, upper, margin):
    if lower > margin or upper < -margin:
        return "practically meaningful interaction"
    if lower > -margin and upper < margin:
        return "practically small in this setting"
    if lower > 0 or upper < 0:
        return "detectable; practical size uncertain"
    return "inconclusive"


def summarize_contrasts(correct, margin=2, resamples=10000, seed=722, two_way=False):
    from scipy.stats import t

    if correct.ndim != 3 or correct.shape[0] < 3 or correct.shape[1] != 8:
        raise ValueError("need at least three complete paired seed blocks and eight conditions")
    if (
        not np.isin(correct, [0, 1]).all()
        or not math.isfinite(margin)
        or margin <= 0
        or resamples < 100
    ):
        raise ValueError("invalid outcomes, margin, or bootstrap count")
    contrasts = contrast_arrays(correct.astype(np.float64))
    rng = np.random.default_rng(seed)
    seed_rng = np.random.default_rng(seed + 1)
    names = list(contrasts)
    per_question = np.stack([contrasts[name].mean(axis=0) for name in names])
    item_samples = np.empty((len(names), resamples))
    two_samples = np.empty_like(item_samples) if two_way else None
    for start in range(0, resamples, 128):
        stop = min(resamples, start + 128)
        questions = rng.integers(0, correct.shape[2], size=(stop - start, correct.shape[2]))
        item_samples[:, start:stop] = per_question[:, questions].mean(axis=-1) * 100
        if two_way:
            for offset, question_ids in enumerate(questions):
                seeds = seed_rng.integers(0, correct.shape[0], size=correct.shape[0])
                for index, name in enumerate(names):
                    two_samples[index, start + offset] = (
                        contrasts[name][seeds][:, question_ids].mean() * 100
                    )
    result = {}
    critical = float(t.ppf(0.975, correct.shape[0] - 1))
    for index, name in enumerate(names):
        seed_values = contrasts[name].mean(axis=1) * 100
        mean, sd = float(seed_values.mean()), float(seed_values.std(ddof=1))
        half_width = critical * sd / math.sqrt(len(seed_values))
        lower, upper = mean - half_width, mean + half_width
        result[name] = {
            "mean_pp": mean,
            "seed_sd_pp": sd,
            "seed_values_pp": seed_values.tolist(),
            "paired_seed_t_95_pp": [lower, upper],
            "paired_question_bootstrap_95_pp": np.quantile(
                item_samples[index], [0.025, 0.975]
            ).tolist(),
            "interpretation": interval_interpretation(lower, upper, margin),
            "primary": name == "J",
        }
        if two_way:
            result[name]["two_way_bootstrap_95_pp"] = np.quantile(
                two_samples[index], [0.025, 0.975]
            ).tolist()
    return result


def validated_predictions(path):
    rows = read_json(path)
    if len(rows) != 1319 or [row["id"] for row in rows] != list(range(1319)):
        raise ValueError(f"need every official-test question in order: {path}")
    for row in rows:
        reference = Fraction(row["reference_answer"])
        parsed = parse_answer(row["text"])
        expected = not row["truncated"] and parsed is not None and parsed == reference
        if type(row["correct"]) is not bool or row["correct"] != expected:
            raise ValueError(
                f"saved correctness does not match strict scoring: {path}, id {row['id']}"
            )
    return rows


def load_confirmed(directory, manifest):
    selected = read_json(directory / "selection.json")
    if selected["study_sha256"] != sha(directory / "study.json"):
        raise ValueError("selection is not bound to this study")
    plan = read_json(directory / "confirmation-plan.json")
    if plan["selection_sha256"] != sha(directory / "selection.json"):
        raise ValueError("selected settings changed after confirmation")
    for specification in test_jobs(directory, manifest):
        if not complete(directory, manifest, specification, verify=True):
            raise ValueError("every locked test result must be complete before analysis")
    baseline = validated_predictions(directory / "test/BASE/predictions.json")
    signature = [(row["id"], row["question_sha256"], row["reference_answer"]) for row in baseline]
    correct = []
    table = []
    for seed in manifest["confirmation_seeds"]:
        block = []
        for condition in CONDITIONS:
            path = directory / f"test/{condition}-s{seed}"
            metadata, metrics = read_json(path / "run.json"), read_json(path / "metrics.json")
            if (
                metadata["source_sha256"] != manifest["source_sha256"]
                or metadata["data_sha256"] != manifest["data_sha256"]
                or metadata["selection_sha256"] != sha(directory / "selection.json")
            ):
                raise ValueError(f"test provenance mismatch: {path}")
            winner = selected["winners"][condition]
            if (
                metadata["condition"] != condition
                or metadata["seed"] != seed
                or any(metadata["config"][key] != winner[key] for key in ("learning_rate", "gamma"))
            ):
                raise ValueError(f"test configuration differs from locked winner: {path}")
            predictions = validated_predictions(path / "predictions.json")
            if [
                (row["id"], row["question_sha256"], row["reference_answer"]) for row in predictions
            ] != signature:
                raise ValueError("question/reference pairing differs across conditions")
            outcomes = np.array([row["correct"] for row in predictions], dtype=float)
            if metrics["count"] != 1319 or abs(outcomes.mean() * 100 - metrics["accuracy"]) > 1e-9:
                raise ValueError(f"metrics disagree with test predictions: {path}")
            block.append(outcomes)
            training = read_json(
                directory / f"confirmation/{condition}-s{seed}/training-summary.json"
            )
            table.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "learning_rate": winner["learning_rate"],
                    "gamma": winner["gamma"],
                    "accuracy": metrics["accuracy"],
                    "answer_nll": metrics["answer_nll"],
                    "training_seconds": training["seconds"],
                    "test_generation_seconds": metrics["seconds"],
                    "peak_reserved_GiB": training["peak_reserved_bytes"] / 2**30,
                }
            )
        correct.append(block)
    return np.asarray(correct), table, selected


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_records(path):
    records = {
        row["step"]: row for row in (json.loads(line) for line in path.read_text().splitlines())
    }
    return [records[step] for step in sorted(records)]


def figures(directory, output, manifest, contrasts, selected):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200}
    )
    output.mkdir(parents=True, exist_ok=True)
    for filename, names in (
        ("primary-interaction", ("delta_A", "delta_D", "J")),
        ("secondary-interaction", ("delta_E", "delta_D", "I")),
    ):
        figure, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        margin = manifest["practical_margin_pp"]
        axis.axvspan(
            -margin, margin, alpha=0.12, color="gray", label=f"±{margin:g} pp practical band"
        )
        axis.axvline(0, color="gray", linewidth=0.8)
        for index, name in enumerate(names):
            row = contrasts[name]
            lower, upper = row["paired_seed_t_95_pp"]
            axis.errorbar(
                row["mean_pp"],
                index,
                xerr=[[row["mean_pp"] - lower], [upper - row["mean_pp"]]],
                fmt="o",
                capsize=4,
            )
        axis.set(
            yticks=range(len(names)),
            yticklabels=names,
            xlabel="Accuracy difference (percentage points)",
            title="Paired-seed means and Student-t 95% intervals",
        )
        axis.legend(loc="best", fontsize=8)
        figure.savefig(output / f"{filename}.png")
        plt.close(figure)
    figure, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
    search_rows = []
    for axis, condition in zip(axes.flat, CONDITIONS, strict=True):
        candidates = []
        for path in (directory / "search").glob(f"{condition}-g*/run.json"):
            metadata = read_json(path)
            metric = path.parent / f"development-step-{manifest['epoch_steps']}/metrics.json"
            if metric.exists():
                metrics = read_json(metric)
                candidates.append(
                    (
                        metadata["config"]["learning_rate"],
                        metadata["config"]["gamma"],
                        metrics["accuracy"],
                    )
                )
                search_rows.append(
                    {
                        "condition": condition,
                        "learning_rate": candidates[-1][0],
                        "gamma": candidates[-1][1],
                        "one_epoch_accuracy": candidates[-1][2],
                    }
                )
        if candidates:
            lr, gamma, accuracy = zip(*sorted(candidates), strict=True)
            if condition in CONTROLLED:
                points = axis.scatter(lr, gamma, c=accuracy, vmin=0, vmax=100, cmap="viridis", s=70)
                axis.set_yscale("log")
                axis.set_ylabel("gamma")
                figure.colorbar(points, ax=axis, label="Accuracy (%)")
                winner = selected["winners"][condition]
                axis.scatter(
                    [winner["learning_rate"]],
                    [winner["gamma"]],
                    marker="*",
                    s=180,
                    facecolors="none",
                    edgecolors="red",
                )
            else:
                axis.plot(lr, accuracy, marker="o")
                axis.set_ylabel("Accuracy (%)")
        axis.set_xscale("log")
        axis.set(title=condition, xlabel="Learning rate")
    figure.suptitle(
        "One-epoch development screening; stars mark selected three-epoch configurations"
    )
    figure.savefig(output / "search-surfaces.png")
    plt.close(figure)
    write_csv(output.parent / "search-surfaces.csv", search_rows)
    for horizontal in ("steps", "training-seconds"):
        figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for condition in CONDITIONS:
            all_x, all_accuracy, all_nll = [], [], []
            for seed in manifest["confirmation_seeds"]:
                root = directory / f"confirmation/{condition}-s{seed}"
                logs = {
                    row["step"]: row
                    for row in (
                        json.loads(line)
                        for line in (root / "training.jsonl").read_text().splitlines()
                    )
                }
                x, accuracy, nll = [], [], []
                for epoch in (1, 2, 3):
                    step = epoch * manifest["epoch_steps"]
                    evaluation = root / f"development-step-{step}"
                    rows = read_json(evaluation / "predictions.json")[:128]
                    x.append(step if horizontal == "steps" else logs[step]["elapsed_seconds"])
                    accuracy.append(100 * sum(row["correct"] for row in rows) / 128)
                    nll.append(read_json(evaluation / "metrics.json")["answer_nll"])
                all_x.append(x)
                all_accuracy.append(accuracy)
                all_nll.append(nll)
            axes[0].plot(
                np.mean(all_x, axis=0), np.mean(all_accuracy, axis=0), marker="o", label=condition
            )
            axes[1].plot(
                np.mean(all_x, axis=0), np.mean(all_nll, axis=0), marker="o", label=condition
            )
        axes[0].set(ylabel="Development accuracy, fixed 128 questions (%)", xlabel=horizontal)
        axes[1].set(ylabel="Answer NLL, all 500 development examples", xlabel=horizontal)
        axes[1].legend(fontsize=8, ncol=2)
        figure.savefig(output / f"learning-curves-{horizontal}.png")
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for condition in CONDITIONS:
        values = {}
        for seed in manifest["confirmation_seeds"]:
            path = directory / f"confirmation/{condition}-s{seed}/diagnostics.jsonl"
            records = diagnostic_records(path)
            common = set.intersection(
                *(set(row["layer"] for row in record["layers"]) for record in records)
            )
            for record in records:
                measurements = [
                    row["relative_net_movement"]
                    for row in record["layers"]
                    if row["layer"] in common
                ]
                values.setdefault(record["step"], []).append(float(np.mean(measurements)))
        axis.plot(
            sorted(values),
            [np.mean(values[step]) for step in sorted(values)],
            marker="o",
            label=condition,
        )
    axis.set(
        xlabel="Optimizer steps",
        ylabel="Mean relative effective-weight movement",
        title="Representative layers; descriptive means across paired seeds",
    )
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output / "effective-movement.png")
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for condition in CONDITIONS:
        times, movement = {}, {}
        for seed in manifest["confirmation_seeds"]:
            root = directory / f"confirmation/{condition}-s{seed}"
            logs = {
                row["step"]: row["elapsed_seconds"]
                for row in (
                    json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()
                )
            }
            records = diagnostic_records(root / "diagnostics.jsonl")
            common = set.intersection(
                *(set(row["layer"] for row in record["layers"]) for record in records)
            )
            for record in records:
                step = record["step"]
                times.setdefault(step, []).append(logs[step] if step else 0)
                movement.setdefault(step, []).append(
                    np.mean(
                        [
                            row["relative_net_movement"]
                            for row in record["layers"]
                            if row["layer"] in common
                        ]
                    )
                )
        axis.plot(
            [np.mean(times[step]) for step in sorted(times)],
            [np.mean(movement[step]) for step in sorted(times)],
            marker="o",
            label=condition,
        )
    axis.set(
        xlabel="Recorded training seconds",
        ylabel="Mean relative effective-weight movement",
        title="Representative layers; descriptive means across paired seeds",
    )
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output / "effective-movement-time.png")
    plt.close(figure)


def failure_records(directory):
    failures = []
    for path in directory.rglob("invocations.jsonl"):
        metadata = read_json(path.parent / "run.json")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") in ("failed", "interrupted"):
                failures.append(
                    {
                        "directory": str(path.parent.relative_to(directory)),
                        "condition": metadata["condition"],
                        "seed": metadata["seed"],
                        "status": row["status"],
                        "error_type": row.get("error", {}).get("type", ""),
                        "error": row.get("error", {}).get("message", ""),
                        "seconds": row.get("seconds", 0),
                    }
                )
    return failures


def analyze(directory, manifest, resamples=10000, two_way=False):
    correct, table, selected = load_confirmed(directory, manifest)
    contrasts = summarize_contrasts(
        correct, manifest["practical_margin_pp"], resamples, two_way=two_way
    )
    user = manifest.get("environment", {}).get("user")
    output = (
        directory / "analysis"
        if directory.resolve().is_relative_to(Path("reports").resolve())
        else report_directory(directory, user=user) / "analysis"
    )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "contrasts": contrasts,
        "confirmation_seeds": manifest["confirmation_seeds"],
        "test_count": 1319,
        "base_reference": read_json(directory / "test/BASE/metrics.json"),
        "practical_margin_pp": manifest["practical_margin_pp"],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": 722,
        "selection_sha256": sha(directory / "selection.json"),
        "analysis_source_sha256": sources(),
        "analysis_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")
        },
        "scope": selected["scope"],
        "search_boundaries": selected["boundaries"],
        "runtime_accounting": runtime_accounting(directory),
        "planning_target_hours": manifest.get("target_hours"),
        "interval_scope": (
            "Seed intervals condition on this test set and selected configurations; "
            "item bootstrap conditions on trained models. Only J is primary."
        ),
    }
    write_json(output / "statistics.json", summary)
    write_csv(output / "seed-results.csv", table)
    failures = failure_records(directory)
    condition_table = []
    for condition in CONDITIONS:
        rows = [row for row in table if row["condition"] == condition]
        accuracies = [row["accuracy"] for row in rows]
        condition_table.append(
            {
                "condition": condition,
                "learning_rate": rows[0]["learning_rate"],
                "gamma": rows[0]["gamma"],
                "accuracy_mean": float(np.mean(accuracies)),
                "accuracy_seed_sd": float(np.std(accuracies, ddof=1)),
                "answer_nll_mean": float(np.mean([row["answer_nll"] for row in rows])),
                "peak_reserved_GiB": max(row["peak_reserved_GiB"] for row in rows),
                "recorded_failed_invocations": sum(
                    row["condition"] == condition and row["status"] == "failed" for row in failures
                ),
                "recorded_interrupted_invocations": sum(
                    row["condition"] == condition and row["status"] == "interrupted"
                    for row in failures
                ),
            }
        )
    write_csv(output / "condition-results.csv", condition_table)
    write_csv(
        output / "contrasts.csv",
        [
            {
                "contrast": name,
                "mean_pp": row["mean_pp"],
                "lower_95_pp": row["paired_seed_t_95_pp"][0],
                "upper_95_pp": row["paired_seed_t_95_pp"][1],
                "interpretation": row["interpretation"],
            }
            for name, row in contrasts.items()
        ],
    )
    write_csv(
        output / "cost-by-stage.csv",
        [
            {"stage": name, "measured_runtime_hours": seconds / 3600}
            for name, seconds in runtime_by_stage(directory).items()
        ],
    )
    components = {}
    for path in directory.rglob("invocations.jsonl"):
        for line in path.read_text().splitlines():
            for name, seconds in json.loads(line).get("phase_seconds", {}).items():
                components[name] = components.get(name, 0) + seconds
    write_csv(
        output / "cost-components.csv",
        [{"component": name, "seconds": seconds} for name, seconds in components.items()],
    )
    write_csv(
        output / "failures.csv",
        failures,
        fields=["directory", "condition", "seed", "status", "error_type", "error", "seconds"],
    )
    diagnostic_rows = []
    for seed in manifest["confirmation_seeds"]:
        for condition in CONDITIONS:
            path = directory / f"confirmation/{condition}-s{seed}/diagnostics.jsonl"
            for line in path.read_text().splitlines():
                record = json.loads(line)
                diagnostic_rows.extend(
                    {
                        "condition": condition,
                        "seed": seed,
                        "step": record["step"],
                        "layer": row["layer"],
                        "measurements_json": json.dumps(row),
                    }
                    for row in record["layers"]
                )
    write_csv(output / "effective-weight-diagnostics.csv", diagnostic_rows)
    counterfactual_rows = []
    for condition in ("E-S", "E-R", "D-S", "D-R"):
        path = directory / f"diagnostics/{condition}-s40/counterfactuals.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                row = json.loads(line)
                counterfactual_rows.append(
                    {
                        "condition": condition,
                        "step": row["step"],
                        "measurements_json": json.dumps(row),
                    }
                )
    write_csv(
        output / "counterfactual-diagnostics.csv",
        counterfactual_rows,
        fields=["condition", "step", "measurements_json"],
    )
    figures(directory, output / "figures", manifest, contrasts, selected)
    print(f"Analysis data and figures: {output}")
    return summary
