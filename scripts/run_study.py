"""Run the unchanged study CLI with evaluation progress and optional Telegram alerts."""

import argparse
import getpass
import json
import re
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.request import Request, urlopen

from dora_spectral import study

TELEGRAM_CONFIG = Path("configs/telegram.local.json")
NOTIFIED_STAGES = {
    "init",
    "pilot",
    "freeze",
    "diagnose",
    "screen",
    "promote",
    "select",
    "confirm",
    "test",
    "analyze",
}


def telegram_settings():
    """An absent local config disables alerts; invalid credentials never stop a study."""
    try:
        settings = json.loads(TELEGRAM_CONFIG.read_text())
        token, chat_id = settings["bot_token"], settings["chat_id"]
        if not isinstance(token, str) or not re.fullmatch(r"[0-9]+:[A-Za-z0-9_-]+", token):
            raise ValueError("invalid bot token")
        if type(chat_id) not in (str, int) or not str(chat_id).strip():
            raise ValueError("invalid chat ID")
        return token, chat_id
    except FileNotFoundError:
        return None
    except Exception:
        # Exception messages can include credentials or file contents. Never log them.
        print("[telegram] Cannot read valid credentials; alerts disabled.", file=sys.stderr)
        return None


def send_telegram(settings, message):
    """Best-effort delivery with a short timeout; never expose a token in an error log."""
    token, chat_id = settings
    try:
        request = Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=json.dumps({"chat_id": chat_id, "text": message}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=10) as response:
            result = json.load(response)
        if not isinstance(result, dict) or result.get("ok") is not True:
            raise ValueError("Telegram rejected the message")
    except Exception:
        print("[telegram] Alert delivery failed; study result unchanged.", file=sys.stderr)
        return False
    print("[telegram] Alert sent.", flush=True)
    return True


def stage_context(arguments):
    if "--help" in arguments or "-h" in arguments:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--directory", type=Path, default=Path("runs/study-main"))
    parser.add_argument("command", nargs="?")
    args, _ = parser.parse_known_args(arguments)
    return (args.command, args.directory) if args.command in NOTIFIED_STAGES else None


def notify_stage(settings, context, outcome, seconds, detail):
    try:
        stage, directory = context
        message = (
            f"DoRA study: {outcome}\n"
            f"Stage: {stage}\n"
            f"Study: {str(directory)[:1000]}\n"
            f"Worker: {getpass.getuser()}@{socket.gethostname()}\n"
            f"Elapsed: {seconds / 3600:.2f} hours\n"
            f"{detail}"
        )
        send_telegram(settings, message)
    except Exception:
        print("[telegram] Alert delivery failed; study result unchanged.", file=sys.stderr)


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
    if sys.argv[1:] == ["--telegram-test"]:
        settings = telegram_settings()
        if settings is None:
            print("[telegram] Configure configs/telegram.local.json first.", file=sys.stderr)
            return 1
        return 0 if send_telegram(settings, "DoRA study: Telegram test alert.") else 1
    context = stage_context(sys.argv[1:])
    settings = telegram_settings() if context else None
    began = time.monotonic()
    outcome, detail = "stopped", "Inspect the console log before resuming."
    original_launch = study.launch

    def monitored(command, output):
        return launch_with_progress(command, output, original_launch)

    study.launch = monitored
    try:
        result = study.main()
        outcome = "completed" if result in (0, None) else "stopped"
        detail = f"Exit code: {result if result is not None else 0}"
        return result
    except BaseException as error:
        detail = f"Reason: {type(error).__name__}. Inspect the console log before resuming."
        raise
    finally:
        study.launch = original_launch
        if settings:
            notify_stage(settings, context, outcome, time.monotonic() - began, detail)


if __name__ == "__main__":
    raise SystemExit(main())
