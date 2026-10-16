import importlib.util
import json
import threading
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "study_progress", Path(__file__).parents[1] / "scripts/run_study.py"
)
progress = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(progress)


def command(directory, test=False):
    if test:
        return [
            "python",
            "-m",
            "dora_spectral.experiment",
            "score-test",
            "--directory",
            str(directory),
        ]
    return [
        "python",
        "-m",
        "dora_spectral.experiment",
        "run",
        "--directory",
        str(directory),
        "--steps",
        "609",
        "--split",
        "pilot",
        "--evaluation-count",
        "8",
    ]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def test_progress_shows_evaluation_counts_and_evidence_export_without_changing_data(tmp_path):
    messages = []
    reporter = progress.EvaluationProgress(
        command(tmp_path), tmp_path / "console.log", messages.append
    )
    save(reporter.summary, {"completed_steps": 609})
    reporter.poll()
    assert "answer NLL" in messages[-1]
    for count in (4, 8):
        save(reporter.predictions, [{"id": i} for i in range(count)])
        before = reporter.predictions.read_bytes()
        reporter.poll()
        assert f"{count}/8 answers" in messages[-1]
        assert reporter.predictions.read_bytes() == before
    before = len(messages)
    reporter.poll()
    assert len(messages) == before
    save(reporter.status, {"status": "complete"})
    reporter.poll()
    assert "exporting evidence" in messages[-1]


def test_progress_does_not_mistake_old_predictions_for_current_work(tmp_path):
    messages = []
    predictions = tmp_path / "pilot-step-609/predictions.json"
    save(predictions, [{"id": i} for i in range(8)])
    save(tmp_path / "status.json", {"status": "complete"})
    save(tmp_path / "training-summary.json", {"completed_steps": 609})
    reporter = progress.EvaluationProgress(
        command(tmp_path), tmp_path / "console.log", messages.append
    )
    reporter.poll()
    assert messages == []
    save(predictions, [{"id": 0}])
    reporter.poll()
    assert "1/8 answers" in messages[-1]
    assert not any("8/8" in message for message in messages)


def test_progress_retries_partial_files_and_supports_official_test(tmp_path):
    messages = []
    reporter = progress.EvaluationProgress(
        command(tmp_path, test=True), tmp_path / "console.log", messages.append
    )
    reporter.predictions.write_text('[{"id":')
    reporter.poll()
    assert messages == []
    save(reporter.predictions, [{"id": 0}, {"id": 1}])
    reporter.poll()
    assert "2/1319 answers" in messages[-1]


@pytest.mark.parametrize("result", [0, 1, 7])
def test_launcher_preserves_worker_arguments_return_code_and_final_progress(
    tmp_path, capsys, result
):
    original = command(tmp_path)
    output = tmp_path / "console.log"
    calls = []

    def worker(arguments, log):
        calls.append((arguments, log))
        save(tmp_path / "pilot-step-609/predictions.json", [{"id": i} for i in range(8)])
        save(tmp_path / "status.json", {"status": "complete" if result == 0 else "failed"})
        return result

    assert progress.launch_with_progress(original, output, worker) == result
    assert calls == [(original, output)]
    assert "8/8 answers" in capsys.readouterr().out
    assert "8/8 answers" in output.read_text()


def test_launcher_restores_the_original_launch_function_after_errors(monkeypatch):
    original = progress.study.launch

    def interrupted():
        assert progress.study.launch is not original
        raise KeyboardInterrupt

    monkeypatch.setattr(progress.study, "main", interrupted)
    with pytest.raises(KeyboardInterrupt):
        progress.main()
    assert progress.study.launch is original


@pytest.mark.parametrize("error", [RuntimeError("worker error"), KeyboardInterrupt()])
def test_launcher_preserves_worker_exceptions(tmp_path, error):
    def worker(arguments, output):
        raise error

    with pytest.raises(type(error)) as caught:
        progress.launch_with_progress(command(tmp_path), tmp_path / "console.log", worker)
    assert caught.value is error


def test_progress_is_visible_while_worker_is_still_generating(tmp_path, monkeypatch, capsys):
    observed = threading.Event()
    original_log = progress.EvaluationProgress.log

    def record(self, message):
        original_log(self, message)
        if "4/8 answers" in message:
            observed.set()

    monkeypatch.setattr(progress.EvaluationProgress, "log", record)

    def worker(arguments, output):
        path = tmp_path / "pilot-step-609/predictions.json"
        save(path, [{"id": i} for i in range(4)])
        assert observed.wait(5), "progress did not appear while generation was running"
        save(path, [{"id": i} for i in range(8)])
        save(tmp_path / "status.json", {"status": "complete"})
        return 0

    assert progress.launch_with_progress(command(tmp_path), tmp_path / "console.log", worker) == 0
    text = capsys.readouterr().out
    assert "4/8 answers" in text and "8/8 answers" in text
