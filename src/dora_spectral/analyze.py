"""Paired seed intervals and matched-question bootstrap for frozen test results."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import read_jsonl
from .experiment import write_json
from .model import CONDITIONS


def contrast(values):
    a = values["A-S"] - values["A-R"]
    d = values["D-S"] - values["D-R"]
    e = values["E-S"] - values["E-R"]
    return {"delta_A": a, "delta_D": d, "delta_E": e, "J": d - a, "I": e - d}


def seed_interval(values):
    from scipy.stats import t

    values = np.asarray(values, dtype=float)
    deviation = values.std(ddof=1)
    half = t.ppf(0.975, len(values) - 1) * deviation / np.sqrt(len(values))
    return {
        "seed_values": values.tolist(),
        "mean": float(values.mean()),
        "sd": float(deviation),
        "paired_seed_95_interval": [float(values.mean() - half), float(values.mean() + half)],
    }


def analyze(root: Path, protocol: dict, resamples=10000):
    seeds = protocol["confirmation_seeds"]
    all_values, arrays = {}, {}
    summaries = []
    reference_ids = None
    for condition in CONDITIONS:
        accuracies, questions = [], []
        for seed in seeds:
            directory = root / f"{condition}-seed{seed}" / "test"
            summary = json.loads((directory / "summary.json").read_text())
            expected = {**protocol["selected_configs"][condition], "seed": seed}
            if (
                summary["split"] != "test"
                or summary["config"] != expected
                or summary["asset_hash"] != protocol["asset_hash"]
            ):
                raise ValueError("Result does not match frozen protocol")
            rows = read_jsonl(directory / "predictions.jsonl")
            ids = [row["id"] for row in rows]
            if reference_ids is None:
                reference_ids = ids
            if ids != reference_ids or len(set(ids)) != len(ids):
                raise ValueError("Question IDs must match across all conditions/seeds")
            correct = np.array([row["correct"] for row in rows], dtype=float) * 100
            if len(rows) != summary["count"] or not np.isclose(correct.mean(), summary["accuracy"]):
                raise ValueError("Prediction file and summary disagree")
            accuracies.append(summary["accuracy"])
            questions.append(correct)
            summaries.append(summary)
        all_values[condition] = np.asarray(accuracies)
        arrays[condition] = np.stack(questions)
    result = {
        "conditions": {c: seed_interval(v) for c, v in all_values.items()},
        "contrasts": {name: seed_interval(values) for name, values in contrast(all_values).items()},
        "protocol": protocol,
        "run_summaries": summaries,
        "interpretation": "Seed intervals condition on selected configs and the locked test set.",
    }
    # Match item resampling across all cells and seeds; seeds remain fixed here.
    rng = np.random.default_rng(9821)
    item_contrasts = contrast({c: a.mean(axis=0) for c, a in arrays.items()})
    bootstrap = {name: [] for name in item_contrasts}
    for _ in range(resamples):
        indices = rng.integers(0, len(reference_ids), len(reference_ids))
        for name, values in item_contrasts.items():
            bootstrap[name].append(values[indices].mean())
    result["question_bootstrap"] = {
        name: {"resamples": resamples, "95_interval": np.quantile(values, [0.025, 0.975]).tolist()}
        for name, values in bootstrap.items()
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/confirmation"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, analyze(args.runs, json.loads(args.protocol.read_text())))
    print(f"Saved paired analysis to {args.output}")


if __name__ == "__main__":
    main()
