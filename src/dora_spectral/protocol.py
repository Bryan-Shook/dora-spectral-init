"""Generate equal-budget searches, promotions, and frozen confirmation configs.

Writes job specifications only. Does not submit jobs or spend a GPU allocation.
"""

import argparse
import json
import math
from pathlib import Path

import yaml

from .experiment import RunConfig, config_dict, write_json
from .model import CONDITIONS


def grid(base: RunConfig) -> list[dict]:
    jobs = []
    for condition in CONDITIONS:
        if condition in ("L0", "D0"):
            candidates = [(1.0, 2e-6 * (1000 ** (i / 11))) for i in range(12)]
        else:
            candidates = [
                (gamma, rate) for gamma in (0.1, 0.3, 1.0) for rate in (2e-5, 1e-4, 5e-4, 2e-3)
            ]
        for index, (gamma, rate) in enumerate(candidates):
            config = config_dict(base)
            config.update(
                condition=condition, seed=10, gamma=gamma, learning_rate=rate, train_limit=None
            )
            jobs.append({"name": f"{condition}-{index:02d}", "config": config})
    return jobs


def ranked(summary):
    config = summary["config"]
    return (-summary["accuracy"], summary["answer_nll"], config["learning_rate"], config["gamma"])


def collect_screen(jobs: list[dict], runs: Path) -> list[dict]:
    results = []
    for job in jobs:
        summary = json.loads((runs / job["name"] / "screen" / "summary.json").read_text())
        if summary["config"] != job["config"] or summary["split"] != "dev":
            raise ValueError(f"screen config/split mismatch: {job['name']}")
        if (
            summary["count"] != 256
            or summary["nll_count"] < summary["count"]
            or summary["training_progress"]["epoch"] != 1
        ):
            raise ValueError("Screen each one-epoch checkpoint on 256 dev questions/full dev NLL")
        results.append({**job, "summary": summary})
    if len({r["summary"]["asset_hash"] for r in results}) != 1:
        raise ValueError("search candidates use different assets")
    return results


def promote(results: list[dict]) -> list[dict]:
    winners = []
    for condition in CONDITIONS:
        rows = [r for r in results if r["config"]["condition"] == condition]
        if condition in ("L0", "D0"):
            chosen = sorted(rows, key=lambda row: ranked(row["summary"]))[:3]
        else:
            chosen = [
                min(
                    [r for r in rows if r["config"]["gamma"] == gamma],
                    key=lambda row: ranked(row["summary"]),
                )
                for gamma in (0.1, 0.3, 1.0)
            ]
        winners.extend({"name": row["name"], "config": row["config"]} for row in chosen)
    return winners


def select(promoted: list[dict], runs: Path) -> dict:
    rows = []
    for job in promoted:
        summary = json.loads((runs / job["name"] / "final-dev" / "summary.json").read_text())
        if (
            summary["config"] != job["config"]
            or summary["split"] != "dev"
            or summary["count"] != summary["nll_count"]
            or summary["training_progress"]["epoch"] != job["config"]["epochs"]
        ):
            raise ValueError("Selection requires full-development scores at the fixed final epoch")
        rows.append({**job, "summary": summary})
    if len({r["summary"]["asset_hash"] for r in rows}) != 1:
        raise ValueError("different assets in selection")
    chosen, by_scale = {}, {}
    for condition in CONDITIONS:
        group = [r for r in rows if r["config"]["condition"] == condition]
        winner = min(group, key=lambda row: ranked(row["summary"]))
        chosen[condition] = winner["config"]
        if condition not in ("L0", "D0"):
            by_scale[condition] = {str(r["config"]["gamma"]): r["config"] for r in group}
    return {
        "asset_hash": rows[0]["summary"]["asset_hash"],
        "selected_configs": chosen,
        "fixed_scale_lr_winners": by_scale,
        "development_results": rows,
    }


def write_jobs(output: Path, jobs: list[dict]):
    if output.exists():
        raise FileExistsError("Use a new configuration directory")
    output.mkdir(parents=True)
    for job in jobs:
        (output / f"{job['name']}.yaml").write_text(yaml.safe_dump(job["config"], sort_keys=False))
    write_json(output / "jobs.json", jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("--base", type=Path, default=Path("configs/pilot.yaml"))
    promotion = sub.add_parser("promote")
    selection = sub.add_parser("select")
    for command in (promotion, selection):
        command.add_argument("--jobs", type=Path, required=True)
        command.add_argument("--runs", type=Path, default=Path("runs/search"))
    frozen = sub.add_parser("freeze")
    frozen.add_argument("--selection", type=Path, required=True)
    frozen.add_argument(
        "--margin",
        type=float,
        required=True,
        help="Prespecified practical margin in accuracy percentage points",
    )
    frozen.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    for command in (search, promotion, selection, frozen):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "search":
        base = RunConfig(**yaml.safe_load(args.base.read_text()))
        if base.epochs != 3:
            raise ValueError("The planned search requires a fixed three-epoch schedule")
        write_jobs(args.output, grid(base))
    elif args.command == "promote":
        jobs = json.loads(args.jobs.read_text())
        write_jobs(args.output, promote(collect_screen(jobs, args.runs)))
    elif args.command == "select":
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, select(json.loads(args.jobs.read_text()), args.runs))
    else:
        selected = json.loads(args.selection.read_text())
        if set(selected["selected_configs"]) != set(CONDITIONS):
            raise ValueError("Freeze all eight conditions")
        if not math.isfinite(args.margin) or args.margin <= 0:
            raise ValueError("margin must be finite and positive")
        if len(args.seeds) < 3 or len(set(args.seeds)) != len(args.seeds) or 10 in args.seeds:
            raise ValueError("Use at least three unique fresh confirmation seeds")
        configurations = {}
        jobs = []
        for condition, config in selected["selected_configs"].items():
            config = dict(config)
            config.pop("seed")
            configurations[condition] = config
            for seed in args.seeds:
                candidate = {**config, "seed": seed}
                RunConfig(**candidate)
                jobs.append({"name": f"{condition}-seed{seed}", "config": candidate})
        write_jobs(args.output, jobs)
        write_json(
            args.output / "frozen-protocol.json",
            {
                "asset_hash": selected["asset_hash"],
                "selected_configs": configurations,
                "confirmation_seeds": args.seeds,
                "practical_margin_points": args.margin,
                "parser": "last-#### exact Decimal; malformed/truncated = incorrect",
                "primary": "J",
                "secondary": "I",
            },
        )
    print(f"Saved {args.command} specification to {args.output}")


if __name__ == "__main__":
    main()
