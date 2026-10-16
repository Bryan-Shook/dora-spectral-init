"""Run the unchanged study CLI with progress from its saved evaluation files."""

import json
import threading
from pathlib import Path

from dora_spectral import study


class EvaluationProgress:
    def __init__(self, command, output, emit=None):
        def option(name, default=None):
            return command[command.index(name) + 1] if name in command else default

        self.directory = Path(option("--directory"))
        self.output = Path(output)
        self.test = "score-test" in command
        self.steps = int(option("--steps", 0))
        self.total = int(option("--evaluation-count", 1319 if self.test else 0))
        split = option("--split", "pilot")
        evaluation = self.directory if self.test else self.directory / f"{split}-step-{self.steps}"
        self.summary = self.directory / "training-summary.json"
        self.predictions = evaluation / "predictions.json"
        self.status = self.directory / "status.json"
        self.seen = {
            path: self.signature(path) for path in (self.summary, self.predictions, self.status)
        }
        self.emit = emit or self.log
        self.last_count = None
        self.last_status = None

    @staticmethod
    def signature(path):
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size, stat.st_ino
        except OSError:
            return None

    def changed_json(self, path):
        signature = self.signature(path)
        if signature is None or signature == self.seen[path]:
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            return None  # Retry transient/incomplete reads; never interrupt the worker.
        self.seen[path] = signature
        return value

    def log(self, message):
        print(message, flush=True)
        try:
            with self.output.open("a") as handle:
                handle.write(message + "\n")
        except OSError:
            pass  # The original launcher owns log-file errors and worker cleanup.

    def say(self, text):
        self.emit(f"[evaluation] {self.directory.name}: {text}")

    def poll(self, final=False):
        summary = self.changed_json(self.summary)
        completed = summary.get("completed_steps") if isinstance(summary, dict) else None
        if not final and isinstance(completed, int) and completed >= self.steps:
            self.say("training/checkpoint saved; evaluating answer NLL, then generating answers.")
        predictions = self.changed_json(self.predictions)
        if isinstance(predictions, list) and len(predictions) != self.last_count:
            self.last_count = len(predictions)
            if self.total:
                self.say(
                    f"generated {self.last_count}/{self.total} answers "
                    f"({100 * self.last_count / self.total:.1f}%)."
                )
            else:
                self.say(f"generated {self.last_count} answers.")
        status = self.changed_json(self.status)
        if isinstance(status, dict) and status.get("status") != self.last_status:
            self.last_status = status.get("status")
            if self.last_status == "complete":
                self.say("run complete." if final else "evaluation saved; exporting evidence...")
            elif self.last_status in ("failed", "interrupted"):
                self.say(f"run {self.last_status}; retaining partial results.")


def launch_with_progress(command, output, launch):
    """Observe one worker while preserving its original command, exit status, and signals."""
    output.parent.mkdir(parents=True, exist_ok=True)
    progress = EvaluationProgress(command, output)
    progress.say(
        "loading model/checkpoint; answer NLL precedes generation."
        if progress.test
        else f"evaluation will generate {progress.total} answers after training."
    )
    stop = threading.Event()

    def monitor():
        while not stop.wait(1):
            progress.poll()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        return launch(command, output)
    finally:
        stop.set()
        thread.join(timeout=2)
        if not thread.is_alive():
            progress.poll(final=True)


def main():
    original_launch = study.launch

    def monitored(command, output):
        return launch_with_progress(command, output, original_launch)

    study.launch = monitored
    try:
        return study.main()
    finally:
        study.launch = original_launch


if __name__ == "__main__":
    raise SystemExit(main())
