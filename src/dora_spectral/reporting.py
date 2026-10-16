"""Copy small experiment evidence into reports without loading models or checkpoints."""

import argparse
import getpass
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EVIDENCE_SUFFIXES = {".json", ".jsonl", ".log", ".xml", ".yaml", ".yml"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def environment():
    """Capture execution-time provenance before a run, including Git state and GPU details."""
    packages = {}
    for name in ("torch", "transformers", "peft", "datasets"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "user": getpass.getuser(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_tracked_changes": _command(["git", "status", "--porcelain", "--untracked-files=no"]),
        "packages": packages,
        "gpus": _command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def report_directory(directory, reports=Path("reports"), user=None):
    """Mirror runs/<campaign>/<candidate> under reports/<user>/<campaign>/<candidate>."""
    user = getpass.getuser() if user is None else user
    if not user or user in {".", ".."} or Path(user).name != user or "\\" in user:
        raise ValueError("report user must be a single, nonempty path component")
    source = directory.resolve()
    try:
        relative = source.relative_to(Path("runs").resolve())
    except ValueError:
        identity = hashlib.sha256(str(source).encode()).hexdigest()[:12]
        relative = Path("external") / f"{source.name}-{identity}"
    return reports.resolve() / user / relative


def _recorded_user(directory):
    for name in ("status.json", "invocations.jsonl", "run.json"):
        path = directory / name
        if not path.exists():
            continue
        text = path.read_text()
        records = [text] if name.endswith(".json") else reversed(text.splitlines())
        for record in records:
            try:
                saved = json.loads(record)
            except ValueError:
                continue
            if isinstance(saved, dict) and isinstance(saved.get("environment"), dict):
                user = saved["environment"].get("user")
                if isinstance(user, str) and user:
                    return user
    return None


def export_run(directory, reports=Path("reports"), user=None):
    """Copy evidence verbatim and retain originals for resume; never copy model weights."""
    source = directory.resolve()
    destination = report_directory(
        source, reports, _recorded_user(source) if user is None else user
    )
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("raw runs and reports must use separate, non-nested directories")
    metadata = json.loads((source / "run.json").read_text())
    existing = destination / "run.json"
    if existing.exists() and json.loads(existing.read_text()) != metadata:
        raise ValueError(f"report destination belongs to a different experiment: {destination}")
    for path in sorted(source.rglob("*")):
        if path.suffix not in EVIDENCE_SUFFIXES or not path.is_file():
            continue
        relative = path.relative_to(source)
        if any((source / parent).is_symlink() for parent in (relative, *relative.parents)):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(path, temporary)
        temporary.replace(target)
    return destination


def export_runs(directory, reports=Path("reports"), user=None):
    candidates = sorted(
        path.parent for path in directory.rglob("run.json") if not path.is_symlink()
    )
    if not candidates:
        raise ValueError(f"no run.json files found under {directory}")
    return [export_run(path, reports, user) for path in candidates]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", type=Path, default=Path("runs"), help="one run or a campaign root"
    )
    parser.add_argument(
        "--reports", type=Path, default=Path("reports"), help="evidence destination"
    )
    parser.add_argument(
        "--user", help="override report user; default: recorded execution user or current OS user"
    )
    args = parser.parse_args()
    for destination in export_runs(args.runs, args.reports, args.user):
        print(f"Experiment evidence: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
