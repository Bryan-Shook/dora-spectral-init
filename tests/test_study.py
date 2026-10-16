import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dora_spectral import study
from dora_spectral.analysis import analyze, summarize_contrasts


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(study, "versions", lambda: {"torch": "fixture"})
    monkeypatch.setattr("dora_spectral.reporting.getpass.getuser", lambda: "tester")
    config = {
        "model": "fixture",
        "model_revision": "model-revision",
        "dataset": "fixture",
        "dataset_revision": "dataset-revision",
        "max_length": 16,
        "rank": 16,
        "epochs": 3,
        "batch_size": 32,
        "diagnostics": True,
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    data = tmp_path / "runs/data"
    examples = {
        "train": [{"id": i, "input_ids": [1, 2]} for i in range(1280)],
        "pilot": [{"id": i} for i in range(1000, 1500)],
        "development": [{"id": i} for i in range(2000, 2500)],
    }
    study.write_json(data / "examples.json", examples)
    checksum = study.sha(data / "examples.json")
    study.write_json(data / "prepared.json", {"config": config, "examples_sha256": checksum})
    validation = tmp_path / "runs/validation"
    study.write_json(
        validation / "validation.json",
        {
            "passed": True,
            "config": config,
            "source_sha256": study.sources(),
            "versions": study.versions(),
            "data_sha256": checksum,
            "seconds": 10,
            "baseline_pilot_nll": 1.0,
            "baseline_generation": {"seconds": 1, "count": 16},
        },
    )
    args = SimpleNamespace(
        directory=tmp_path / "runs/main",
        config=config_path,
        data=data,
        validation=validation,
        seeds=[101, 202, 303],
        target_hours=100,
        margin=2,
        wall_hours=4,
        job_seconds=3600,
    )
    manifest = study.initialize(args)

    def finish(specification, accuracy=None):
        root = args.directory / specification["directory"]
        metadata = {
            "condition": specification["condition"],
            "seed": specification["seed"],
            "config": {
                **config,
                "learning_rate": specification["learning_rate"],
                "gamma": specification["gamma"],
                "counterfactuals": specification["counterfactuals"],
            },
            "source_sha256": manifest["source_sha256"],
            "versions": manifest["versions"],
            "data_sha256": checksum,
        }
        if specification["kind"] == "test":
            metadata["selection_sha256"] = study.sha(args.directory / "selection.json")
            if specification["condition"] == "BASE":
                metadata["config"] = config.copy()
        study.write_json(root / "run.json", metadata)
        ids = (
            list(range(1319))
            if specification["split"] == "test"
            else manifest[f"{specification['split']}_ids"][: specification["count"]]
        )
        rows = [
            {
                "id": i,
                "question_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
                "reference_answer": "1",
                "text": "#### 1" if i % 2 else "#### 0",
                "correct": bool(i % 2),
                "truncated": False,
            }
            for i in ids
        ]
        metric = study.metric_path(args.directory, specification)
        study.write_json(
            metric,
            {
                "count": len(rows),
                "accuracy": 100 * sum(row["correct"] for row in rows) / len(rows)
                if accuracy is None
                else accuracy,
                "answer_nll": 0.5,
                "truncation_rate": 0,
                "seconds": 1,
            },
        )
        study.write_json(metric.parent / "predictions.json", rows)
        if specification["kind"] == "train":
            (root / "checkpoint.pt").write_bytes(
                b"fixture checkpoint; actual training is tested separately"
            )
            duration = (
                specification["steps"]
                * 0.01
                * (study.CONDITIONS.index(specification["condition"]) + 1)
            )
            study.write_json(
                root / "training-summary.json",
                {
                    "completed_steps": specification["steps"],
                    "seconds": duration,
                    "peak_reserved_bytes": 2**30,
                },
            )
            (root / "training.jsonl").write_text(
                "".join(
                    json.dumps({"step": i, "elapsed_seconds": i * 0.01}) + "\n"
                    for i in range(1, specification["steps"] + 1)
                )
            )
            (root / "diagnostics.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "step": i,
                            "layers": [{"layer": "q_proj", "relative_net_movement": i * 0.001}],
                        }
                    )
                    + "\n"
                    for i in (0, 1, manifest["horizon"])
                )
            )
        study.write_json(
            root / "status.json", {"status": "complete", "environment": {"user": "tester"}}
        )
        (root / "invocations.jsonl").write_text('{"phase_seconds": {"initialization": 0.01}}\n')

    calls = []

    def launch(command, output):
        calls.append(command)

        def option(name):
            return command[command.index(name) + 1]

        path = Path(option("--directory"))
        if "score-test" in command:
            if "--base" in command:
                condition, seed, lr, gamma = "BASE", 0, 0, 1
            else:
                saved = study.read_json(Path(option("--checkpoint")) / "run.json")
                condition, seed = saved["condition"], saved["seed"]
                lr, gamma = saved["config"]["learning_rate"], saved["config"]["gamma"]
            specification = study.job(
                condition,
                seed,
                lr,
                gamma,
                0,
                1319,
                str(path.relative_to(args.directory)),
                split="test",
                kind="test",
            )
        else:
            specification = study.job(
                option("--condition"),
                int(option("--seed")),
                float(option("--lr")),
                float(option("--gamma")),
                int(option("--steps")),
                int(option("--evaluation-count")),
                str(path.relative_to(args.directory)),
                split=option("--split"),
                counterfactuals="--counterfactuals" in command,
            )
        finish(specification)
        return 0

    monkeypatch.setattr(study, "launch", launch)
    return args, manifest, finish, calls


def run_selection(args, manifest, finish):
    for specification in study.search_jobs(manifest):
        finish(specification)
    for specification in study.promotions(args.directory, manifest):
        finish(specification)
    return study.selection(args.directory, manifest)


def test_equal_grid_and_tie_breaking(campaign):
    args, manifest, finish, _ = campaign
    jobs = study.search_jobs(manifest)
    assert len(jobs) == 96
    assert all(
        sum(item["condition"] == condition for item in jobs) == 12 for condition in study.CONDITIONS
    )
    selected = run_selection(args, manifest, finish)
    promoted = study.read_json(args.directory / "promotions.json")
    assert len(promoted) == 24
    assert all(
        sum(item["condition"] == condition for item in promoted) == 3
        for condition in study.CONDITIONS
    )
    assert selected["winners"]["A-S"]["gamma"] == 0.1
    assert selected["winners"]["A-S"]["learning_rate"] == 2e-5
    assert selected["winners"]["L0"]["learning_rate"] == 2e-6
    assert len(study.confirmation_jobs(args.directory, manifest)) == 72


def test_validation_must_match_the_study_environment(campaign):
    args, _, _, _ = campaign
    validation = study.read_json(args.validation / "validation.json")
    validation["versions"] = {"torch": "a different runtime"}
    study.write_json(args.validation / "validation.json", validation)
    args.directory = args.directory.parent / "other-study"
    with pytest.raises(ValueError, match="validation did not pass"):
        study.initialize(args)
    assert not args.directory.exists()


def test_validation_can_be_reused_after_orchestration_and_analysis_changes(campaign):
    args, _, _, _ = campaign
    validation = study.read_json(args.validation / "validation.json")
    validation["source_sha256"].update(
        {"study.py": "older driver", "analysis.py": "older analysis"}
    )
    study.write_json(args.validation / "validation.json", validation)
    args.directory = args.directory.parent / "new-study"
    manifest = study.initialize(args)
    assert manifest["source_sha256"] == study.sources()
    assert manifest["validation_sha256"] == study.sha(args.validation / "validation.json")


@pytest.mark.parametrize("module", ["model.py", "training.py", "streams.py", "validation.py"])
def test_validation_cannot_be_reused_after_runtime_changes(campaign, module):
    args, _, _, _ = campaign
    validation = study.read_json(args.validation / "validation.json")
    validation["source_sha256"][module] = "different runtime"
    study.write_json(args.validation / "validation.json", validation)
    args.directory = args.directory.parent / "new-study"
    with pytest.raises(ValueError, match="validation did not pass"):
        study.initialize(args)
    assert not args.directory.exists()


def test_runtime_accounting_is_automatic_and_ignores_unmeasured_manual_entries(campaign):
    args, manifest, _, _ = campaign
    assert manifest["cost_basis"] == "recorded_process_wall_time"
    assert "previous_hours" not in manifest
    rows = [
        {"phase": "screen", "directory": "search/candidate", "seconds": 900, "exit_code": 1},
        {"phase": "screen", "directory": "search/candidate", "seconds": 1800, "exit_code": 0},
        {"phase": "confirm", "directory": "confirmation/replica", "seconds": 300},
        {"phase": "extra-allocation", "seconds": 7200, "note": "old manual estimate"},
    ]
    (args.directory / "ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    assert study.runtime_by_stage(args.directory) == {
        "validation": 10,
        "screen": 2700,
        "confirm": 300,
    }
    accounting = study.runtime_accounting(args.directory)
    assert accounting["measured_runtime_hours"] == pytest.approx(3010 / 3600)
    assert accounting["allocated_gpu_hours"] is None


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf")])
def test_runtime_accounting_rejects_invalid_timers(campaign, seconds):
    args, _, _, _ = campaign
    (args.directory / "ledger.jsonl").write_text(
        json.dumps({"phase": "screen", "directory": "search/a", "seconds": 10})
        + "\n"
        + json.dumps({"phase": "screen", "directory": "search/b", "seconds": seconds})
        + "\n"
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        study.runtime_accounting(args.directory)


def test_incomplete_search_and_changed_selection_are_rejected(campaign):
    args, manifest, finish, _ = campaign
    with pytest.raises(ValueError, match="unfinished"):
        study.promotions(args.directory, manifest)
    run_selection(args, manifest, finish)
    selected = study.read_json(args.directory / "selection.json")
    selected["winners"]["A-S"]["gamma"] = 1
    study.write_json(args.directory / "selection.json", selected)
    with pytest.raises(ValueError, match="changed after locking"):
        study.confirmation_jobs(args.directory, manifest)


def test_complete_staged_workflow_and_analysis(campaign):
    pytest.importorskip("scipy")
    pytest.importorskip("matplotlib")
    args, manifest, _, calls = campaign
    study.pilot(args, manifest)
    assert study.read_json(args.directory / "pilot-choice.json")["condition"] == "D0"
    study.pilot(
        args, manifest
    )  # Resuming a completed pilot must retain the original slowest choice.
    study.freeze(args.directory, manifest)
    study.run_jobs(args, manifest, study.search_jobs(manifest), "screen")
    before = len(calls)
    study.run_jobs(args, manifest, study.search_jobs(manifest), "screen")
    assert len(calls) == before
    study.run_jobs(args, manifest, study.promotions(args.directory, manifest), "promote")
    study.selection(args.directory, manifest)
    study.confirmation_jobs(args.directory, manifest)
    with pytest.raises(ValueError, match="must finish"):
        study.authorize_test(args.directory, manifest)
    study.run_jobs(args, manifest, study.confirmation_jobs(args.directory, manifest), "confirm")
    study.authorize_test(args.directory, manifest)
    study.run_jobs(args, manifest, study.test_jobs(args.directory, manifest), "test")
    study.run_jobs(args, manifest, study.diagnostic_jobs(manifest), "diagnose")
    copied = study.export_run(args.directory)
    assert not list(copied.rglob("*.pt"))
    offline_manifest = study.read_study(copied, check_source=False)
    result = analyze(copied, offline_manifest, resamples=100, two_way=True)
    assert result["test_count"] == 1319
    assert result["contrasts"]["J"]["mean_pp"] == 0
    output = args.directory.parents[1] / "reports/tester/main/analysis"
    assert (output / "statistics.json").exists()
    assert len(list((output / "figures").glob("*.png"))) == 7
    assert (output / "seed-results.csv").exists()
    assert result["runtime_accounting"]["allocated_gpu_hours"] is None
    assert result["runtime_accounting"]["measured_runtime_hours"] > 0
    assert result["planning_target_hours"] == 100
    with (output / "cost-by-stage.csv").open() as handle:
        costs = list(csv.DictReader(handle))
    assert set(costs[0]) == {"stage", "measured_runtime_hours"}
    assert not any(row["stage"] in ("earlier-allocations", "extra-allocation") for row in costs)


def test_planning_target_never_blocks_pilot_freeze_or_training(campaign):
    args, _, _, calls = campaign
    args.directory = args.directory.parent / "beyond-target"
    args.target_hours = 0.0001  # Even the selected validation already exceeds this target.
    manifest = study.initialize(args)
    assert not manifest["runtime_target_enforced"]
    study.pilot(args, manifest)
    frozen = study.freeze(args.directory, manifest)
    assert frozen["target_exceeded"]
    assert frozen["projected_runtime_hours"] > frozen["planning_target_hours"]
    assert not frozen["runtime_target_enforced"]
    candidate = study.search_jobs(manifest)[0]
    before = len(calls)
    study.run_jobs(args, manifest, [candidate], "screen")
    assert len(calls) == before + 1
    command = calls[-1]
    assert int(command[command.index("--steps") + 1]) == manifest["epoch_steps"]
    assert study.complete(args.directory, manifest, candidate)


def test_allocation_time_limit_still_stops_launches(campaign):
    args, manifest, _, calls = campaign
    args.wall_hours = 0.01
    with pytest.raises(RuntimeError, match="allocation time limit reached"):
        study.run_jobs(args, manifest, [study.search_jobs(manifest)[0]], "screen")
    assert not calls


@pytest.mark.parametrize("target", [0, -1, float("nan"), float("inf")])
def test_planning_target_must_be_finite_and_positive(campaign, target):
    args, _, _, _ = campaign
    args.directory = args.directory.parent / "invalid-target"
    args.target_hours = target
    with pytest.raises(ValueError, match="planning target must be finite and positive"):
        study.initialize(args)
    assert not args.directory.exists()


def test_complete_binds_shared_config(campaign):
    args, manifest, finish, _ = campaign
    specification = study.search_jobs(manifest)[0]
    finish(specification)
    path = args.directory / specification["directory"] / "run.json"
    metadata = study.read_json(path)
    metadata["config"]["max_length"] = 32
    study.write_json(path, metadata)
    with pytest.raises(ValueError, match="shared configuration"):
        study.complete(args.directory, manifest, specification)


def test_seed_intervals_use_seed_blocks_and_bootstrap_is_paired():
    pytest.importorskip("scipy")
    import numpy as np

    correct = np.zeros((3, 8, 4))
    correct[:, 2, :] = [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]
    result = summarize_contrasts(correct, resamples=200)
    assert result["J"]["seed_values_pp"] == [25, 50, 75]
    assert abs(result["J"]["paired_seed_t_95_pp"][1] - (50 + 4.3026527297 * 25 / np.sqrt(3))) < 1e-6
    repeated = summarize_contrasts(correct, resamples=200)
    assert result == repeated
    with_two_way = summarize_contrasts(correct, resamples=200, two_way=True)
    assert (
        result["J"]["paired_question_bootstrap_95_pp"]
        == with_two_way["J"]["paired_question_bootstrap_95_pp"]
    )
    with pytest.raises(ValueError, match="three complete"):
        summarize_contrasts(correct[:2], resamples=100)
