import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dora_spectral.reporting import export_run, export_runs


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("dora_spectral.reporting.getpass.getuser", lambda: "mchoi345")

    def create(name="A-S-s0"):
        directory = tmp_path / "runs" / "h100-check" / name
        write_json(
            directory / "run.json",
            {
                "condition": name.split("-s")[0],
                "seed": 0,
                "config": {"rank": 16, "learning_rate": 1e-4, "gamma": 1},
                "source_sha256": {"model.py": "original-source-hash"},
            },
        )
        (directory / "training.jsonl").write_text('{"step": 100, "answer_nll": 0.5}\n')
        (directory / "invocations.jsonl").write_text('{"status": "complete", "seconds": 12}\n')
        write_json(directory / "pilot-step-100" / "metrics.json", {"count": 32, "accuracy": 50})
        write_json(directory / "pilot-step-100" / "predictions.json", [{"id": 7, "correct": True}])
        (directory / "checkpoint.pt").write_bytes(b"checkpoint stays on cluster")
        (directory / "weights.safetensors").write_bytes(b"model weights")
        return directory

    return create


def test_export_copies_evidence_verbatim_and_retains_raw_files(candidate, tmp_path):
    source = candidate()
    destination = export_run(source)
    assert destination == tmp_path / "reports" / "mchoi345" / "h100-check" / source.name
    for path in source.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            assert (destination / path.relative_to(source)).read_bytes() == path.read_bytes()
    assert (source / "checkpoint.pt").exists()
    assert not (destination / "checkpoint.pt").exists()
    assert not (destination / "weights.safetensors").exists()
    assert not (destination / "README.md").exists()
    assert not (destination / "summary.json").exists()


def test_interrupted_evidence_is_preserved_without_parsing_logs(candidate):
    source = candidate()
    write_json(source / "status.json", {"status": "interrupted", "target_steps": 609})
    with (source / "training.jsonl").open("a") as handle:
        handle.write('{"step":')
    destination = export_run(source)
    assert (destination / "training.jsonl").read_bytes() == (source / "training.jsonl").read_bytes()
    assert json.loads((destination / "status.json").read_text())["status"] == "interrupted"


def test_batch_export_can_be_updated_without_losing_other_runs(candidate):
    first = candidate()
    second = candidate("D-R-s0")
    outputs = export_runs(first.parent)
    assert len(outputs) == 2
    write_json(first / "pilot-step-609" / "metrics.json", {"count": 500, "accuracy": 70})
    destination = export_run(first)
    assert (destination / "pilot-step-100" / "metrics.json").exists()
    assert (destination / "pilot-step-609" / "metrics.json").exists()
    assert (outputs[1] / "run.json").exists()
    assert (second / "checkpoint.pt").exists()


def test_export_refuses_to_mix_different_experiments(candidate):
    source = candidate()
    export_run(source)
    metadata = json.loads((source / "run.json").read_text())
    metadata["config"]["learning_rate"] = 0.01
    write_json(source / "run.json", metadata)
    with pytest.raises(ValueError, match="different experiment"):
        export_run(source)


def test_copy_cli_works_without_model_dependencies(candidate):
    source = candidate()
    package_root = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "dora_spectral.reporting",
            "--runs",
            str(source.parent),
            "--user",
            "mchoi345",
        ],
        env={**os.environ, "PYTHONPATH": str(package_root)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Experiment evidence:" in result.stdout
    assert (source.parents[2] / "reports" / "mchoi345" / "h100-check" / source.name).is_dir()


def test_recorded_execution_user_and_explicit_override(candidate, tmp_path):
    source = candidate()
    write_json(source / "status.json", {"environment": {"user": "teammate"}})
    assert export_run(source) == tmp_path / "reports" / "teammate" / "h100-check" / source.name
    assert export_run(source, user="mchoi345") == (
        tmp_path / "reports" / "mchoi345" / "h100-check" / source.name
    )


def test_execution_user_survives_missing_status_and_partial_invocation(candidate, tmp_path):
    source = candidate()
    (source / "invocations.jsonl").write_text('{"environment": {"user": "teammate"}}\n{"status":')
    assert export_run(source) == tmp_path / "reports" / "teammate" / "h100-check" / source.name


@pytest.mark.parametrize("user", ["", ".", "..", "../teammate", "team/name", "team\\name"])
def test_user_is_a_single_directory_component(candidate, user):
    with pytest.raises(ValueError, match="path component"):
        export_run(candidate(), user=user)


def test_external_run_names_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first = tmp_path / "cluster-a" / "candidate"
    second = tmp_path / "cluster-b" / "candidate"
    for directory in (first, second):
        write_json(directory / "run.json", {"condition": "A-S"})
    assert export_run(first) != export_run(second)


def test_empty_source_and_nested_destination_are_rejected(candidate, tmp_path):
    with pytest.raises(ValueError, match="no run.json"):
        export_runs(tmp_path / "missing")
    source = candidate()
    with pytest.raises(ValueError, match="separate"):
        export_run(source, reports=source)


@pytest.mark.parametrize("outcome", ["complete", "interrupted", "failed"])
def test_runner_automatically_exports_each_outcome(tmp_path, monkeypatch, outcome):
    pytest.importorskip("peft")
    from transformers import AutoTokenizer

    from dora_spectral import experiment

    monkeypatch.chdir(tmp_path)
    config = {
        "model": "test-model",
        "dataset": "test-data",
        "model_revision": "model-revision",
        "dataset_revision": "data-revision",
        "max_length": 16,
        "batch_size": 1,
        "epochs": 3,
        "rank": 2,
    }
    data = tmp_path / "runs" / "data"
    write_json(data / "examples.json", {"train": [{}], "pilot": [{}]})
    write_json(
        data / "prepared.json",
        {
            "config": config,
            "model_revision": config["model_revision"],
            "dataset_revision": config["dataset_revision"],
            "examples_sha256": hashlib.sha256((data / "examples.json").read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(experiment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(experiment.torch.cuda, "is_bf16_supported", lambda **kwargs: True)
    monkeypatch.setattr(experiment.signal, "signal", lambda *args: None)
    monkeypatch.setattr(experiment.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(
        experiment, "environment", lambda: {"git_commit": "run-commit", "user": "mchoi345"}
    )
    monkeypatch.setattr(
        AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(eos_token="eos", pad_token_id=0),
    )
    monkeypatch.setattr(experiment, "load_model", lambda *args: object())
    monkeypatch.setattr(experiment, "attach", lambda *args: {"layers": []})

    def training(*args):
        if outcome == "interrupted":
            raise experiment.BudgetExpired()
        if outcome == "failed":
            raise RuntimeError("training failed")

    monkeypatch.setattr(experiment, "train", training)
    monkeypatch.setattr(experiment, "answer_nll", lambda *args: 0.5)
    monkeypatch.setattr(experiment, "evaluate", lambda *args: {"count": 1, "accuracy": 0})
    directory = tmp_path / "runs" / "h100-check" / "A-S-s0"
    args = SimpleNamespace(
        data=data,
        directory=directory,
        reports=Path("reports"),
        condition="A-S",
        seed=0,
        lr=1e-4,
        gamma=1,
        steps=1,
        seconds=3600,
        split="pilot",
        evaluation_count=1,
    )
    if outcome == "failed":
        with pytest.raises(RuntimeError, match="training failed"):
            experiment.run(args, config)
    else:
        assert experiment.run(args, config) == (0 if outcome == "complete" else 1)
    exported = tmp_path / "reports" / "mchoi345" / "h100-check" / "A-S-s0"
    status = json.loads((exported / "status.json").read_text())
    assert status["status"] == outcome
    assert status["environment"]["git_commit"] == "run-commit"
    assert status["environment"]["user"] == "mchoi345"
    if outcome == "failed":
        assert status["error"]["message"] == "training failed"
    assert (exported / "invocations.jsonl").read_bytes() == (
        directory / "invocations.jsonl"
    ).read_bytes()
