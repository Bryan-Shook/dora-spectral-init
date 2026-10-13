"""Download pinned model/data and persist auditable partitions for offline jobs."""

import argparse
import json
from pathlib import Path

from .data import digest, split_gsm8k, write_jsonl


def prepare(root: Path, model_id: str, model_revision: str, dataset_revision: str) -> dict:
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download

    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Assets already prepared; use a new directory to change revisions")
    api = HfApi()
    model_sha = api.model_info(model_id, revision=model_revision).sha
    dataset_sha = api.dataset_info("openai/gsm8k", revision=dataset_revision).sha
    snapshot_download(
        model_id,
        revision=model_sha,
        local_dir=root / "model",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"],
    )
    dataset = load_dataset(
        "openai/gsm8k", "main", revision=dataset_sha, cache_dir=str(root / "dataset-cache")
    )
    result = split_gsm8k(list(dataset["train"]), list(dataset["test"]))
    files = {}
    for name, rows in result.pop("partitions").items():
        path = root / f"{name}.jsonl"
        write_jsonl(path, rows)
        files[name] = {"count": len(rows), "sha256": digest(path.read_text())}
    manifest = {
        "model_id": model_id,
        "model_revision": model_sha,
        "dataset_id": "openai/gsm8k",
        "dataset_config": "main",
        "dataset_revision": dataset_sha,
        "files": files,
        **result,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_assets(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        if digest((root / f"{name}.jsonl").read_text()) != info["sha256"]:
            raise ValueError(f"partition hash mismatch: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/qwen-gsm8k"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.output, args.model, args.model_revision, args.dataset_revision), indent=2
        )
    )


if __name__ == "__main__":
    main()
