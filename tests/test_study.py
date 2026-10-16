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
        budget_hours=100,
        reserve_hours=10,
        previous_hours=0,
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


def test_freeze_preserves_reserve_and_complete_binds_shared_config(campaign):
    args, manifest, finish, _ = campaign
    study.pilot(args, manifest)
    with pytest.raises(ValueError, match="budget gate failed"):
        study.freeze(args.directory, dict(manifest, budget_hours=10.1))
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
