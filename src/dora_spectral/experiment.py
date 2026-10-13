"""Single-device GSM8K training and offline, merged-checkpoint evaluation."""

import argparse
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from .data import answer_number, collate, digest, encode_example, prompt, read_jsonl, write_jsonl
from .model import (
    CONDITIONS,
    ModelLinear,
    adapter_state,
    insert_adapters,
    load_model,
    merge_adapters,
    restore_adapters,
)
from .prepare import verify_assets


@dataclass
class RunConfig:
    condition: str = "D-S"
    seed: int = 0
    rank: int = 16
    gamma: float = 1.0
    learning_rate: float = 1e-4
    epochs: int = 3
    microbatch: int = 8
    accumulation: int = 4
    max_length: int = 1024
    max_new_tokens: int = 512
    warmup_ratio: float = 0.03
    gradient_clip: float = 1.0
    precision: str = "bf16"
    attention: str = "sdpa"
    gradient_checkpointing: bool = False
    checkpoint_every: int = 25
    targets: tuple = ("q_proj", "v_proj")
    train_limit: int | None = None

    def __post_init__(self):
        self.targets = tuple(self.targets)
        if self.condition not in CONDITIONS:
            raise ValueError("unknown condition")
        for name in (
            "rank",
            "epochs",
            "microbatch",
            "accumulation",
            "max_length",
            "max_new_tokens",
            "checkpoint_every",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.train_limit is not None and self.train_limit <= 0:
            raise ValueError("train_limit must be positive")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("targets must be nonempty and unique")
        for name in ("gamma", "learning_rate", "gradient_clip"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("invalid warmup_ratio")
        if self.precision not in ("fp32", "bf16") or self.attention not in ("eager", "sdpa"):
            raise ValueError("unsupported precision/attention backend")
        if self.condition in ("L0", "D0") and self.gamma != 1:
            raise ValueError("ordinary baselines have no gamma sweep; use gamma=1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def config_dict(config):
    return json.loads(json.dumps(asdict(config)))


def loss_sum(model, batch):
    labels = batch["labels"][:, 1:]
    logits = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
    ).logits[:, :-1]
    count = int((labels != -100).sum())
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, count


def epoch_order(size, seed, epoch):
    # Independent of model loading, initialization, dropout, and evaluation RNGs.
    return torch.randperm(
        size, generator=torch.Generator().manual_seed(seed + 1_000_003 * epoch + 17)
    ).tolist()


def lr_multiplier(step, total, warmup):
    if step < warmup:
        return (step + 1) / max(1, warmup)
    fraction = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(fraction, 1)))


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def environment(device):
    import importlib.metadata

    versions = {}
    for package in ("torch", "transformers", "datasets", "huggingface-hub", "peft"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
    }
    if device == "cuda":
        result.update(
            gpu=torch.cuda.get_device_name(),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        result["git_diff"] = subprocess.check_output(["git", "diff"], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        result["git_commit"] = None
    return result


def source_snapshot():
    return {path.name: path.read_text() for path in Path(__file__).parent.glob("*.py")}


def checkpoint(path, model, optimizer, scheduler, progress, config, asset_hash):
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "adapters": adapter_state(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "progress": progress,
            "config": config_dict(config),
            "asset_hash": asset_hash,
            "source_hash": digest(json.dumps(source_snapshot(), sort_keys=True)),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary,
    )
    temporary.replace(path)


def movement(model, all_layers=False):
    rows = []
    adapters = [
        (name, layer) for name, layer in model.named_modules() if isinstance(layer, ModelLinear)
    ]
    selected = adapters if all_layers else adapters[:2] + adapters[-2:]
    with torch.no_grad():
        for name, layer in dict(selected).items():
            effective = layer.effective_weight()
            base = layer.base_weight
            cosine = F.cosine_similarity(effective, base, dim=1).clamp(-1, 1)
            rows.append(
                {
                    "layer": name,
                    "relative_update_norm": float((effective - base).norm() / base.norm()),
                    "mean_row_angle_radians": float(cosine.acos().mean()),
                }
            )
    return rows


def train(
    root: Path,
    output: Path,
    config: RunConfig,
    device="cuda",
    resume=False,
    stop_after_epochs=None,
    stop_after_steps=None,
):
    manifest = verify_assets(root)
    asset_hash = digest(json.dumps(manifest, sort_keys=True))
    output.mkdir(parents=True, exist_ok=True)
    if not resume and any(output.iterdir()):
        raise FileExistsError("Run directory is not empty; use --resume or a new output")
    saved = None
    if resume:
        saved = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config_dict(config) or saved["asset_hash"] != asset_hash:
            raise ValueError("resume config/assets differ from checkpoint")
        if saved["source_hash"] != digest(json.dumps(source_snapshot(), sort_keys=True)):
            raise ValueError("implementation changed since checkpoint; preserve the original code")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    started = time.perf_counter()
    model, tokenizer = load_model(root, device, config.precision, config.attention)
    metadata = insert_adapters(
        model,
        config.condition,
        config.rank,
        config.gamma,
        config.seed,
        list(config.targets),
        root / "svd-cache",
    )
    if (
        manifest["model_id"] == "Qwen/Qwen2.5-1.5B"
        and config.rank == 16
        and set(config.targets) == {"q_proj", "v_proj"}
    ):
        expected_count = 2_179_072 + (
            0 if config.condition.startswith("A") or config.condition == "L0" else 50_176
        )
        if len(metadata["layers"]) != 56 or metadata["trainable_parameters"] != expected_count:
            raise ValueError("Qwen target count or trainable parameter count differs from the plan")
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rows = read_jsonl(root / "train.jsonl")
    if config.train_limit:
        rows = rows[: config.train_limit]
    examples, excluded = [], []
    for row in rows:
        encoded = encode_example(row, tokenizer, config.max_length)
        if encoded is None:
            excluded.append(row["id"])
        else:
            examples.append(encoded)
    if not examples:
        raise ValueError("no training examples fit max_length")
    write_json(
        output / "data.json",
        {
            "training_examples": len(examples),
            "excluded_overlength_ids": excluded,
            "train_limit": config.train_limit,
        },
    )
    effective_batch = config.microbatch * config.accumulation
    steps_per_epoch = math.ceil(len(examples) / effective_batch)
    total_steps = config.epochs * steps_per_epoch
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    warmup = math.ceil(total_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total_steps, warmup)
    )
    progress = {"epoch": 0, "window": 0, "step": 0, "allocated_seconds": 0.0}
    if saved:
        restore_adapters(model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        progress = saved["progress"]
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    info = {
        "config": config_dict(config),
        "assets": manifest,
        "adapter_metadata": metadata,
        "environment": environment(device),
        "schedule_total_steps": total_steps,
    }
    if not resume:
        write_json(output / "run.json", info)
        write_json(output / "source.json", source_snapshot())
    else:
        write_json(output / f"resume-{progress['step']}.json", info)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    stop_epoch = config.epochs if stop_after_epochs is None else stop_after_epochs
    if not 1 <= stop_epoch <= config.epochs:
        raise ValueError("stop-after-epochs must be within the fixed training horizon")
    prior_seconds = progress["allocated_seconds"]
    model.train()
    with (output / "training.jsonl").open("a") as log:
        for epoch in range(progress["epoch"], stop_epoch):
            order = epoch_order(len(examples), config.seed, epoch)
            first_window = progress["window"] if epoch == progress["epoch"] else 0
            for window in range(first_window, steps_per_epoch):
                if stop_after_steps is not None and progress["step"] >= stop_after_steps:
                    break
                group = [
                    examples[i]
                    for i in order[window * effective_batch : (window + 1) * effective_batch]
                ]
                # Normalize once over ALL target tokens in the accumulation window.
                denominator = sum(
                    sum(label != -100 for label in row["labels"][1:]) for row in group
                )
                if denominator == 0:
                    raise ValueError("empty supervised accumulation window")
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for begin in range(0, len(group), config.microbatch):
                    batch = collate(
                        group[begin : begin + config.microbatch], tokenizer.pad_token_id, device
                    )
                    loss, _ = loss_sum(model, batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite training loss")
                    (loss / denominator).backward()
                    total_loss += loss.detach().item()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, config.gradient_clip, error_if_nonfinite=True
                )
                used_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                progress.update(step=progress["step"] + 1, epoch=epoch, window=window + 1)
                if window + 1 == steps_per_epoch:
                    progress.update(epoch=epoch + 1, window=0)
                record = {
                    "step": progress["step"],
                    "epoch": epoch + 1,
                    "answer_nll": total_loss / denominator,
                    "target_tokens": denominator,
                    "active_tokens": sum(len(row["input_ids"]) for row in group),
                    "padded_tokens": sum(
                        math.ceil(len(group[j : j + config.microbatch]))
                        * max(len(r["input_ids"]) for r in group[j : j + config.microbatch])
                        for j in range(0, len(group), config.microbatch)
                    ),
                    "learning_rate": used_lr,
                    "gradient_norm": float(grad_norm),
                    "clipped": float(grad_norm) > config.gradient_clip,
                }
                if progress["step"] in (1, 10, 50, 100, total_steps):
                    record["movement"] = movement(model, progress["step"] == total_steps)
                sync(device)
                record["session_elapsed_seconds"] = time.perf_counter() - started
                if device == "cuda":
                    record.update(
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    )
                log.write(json.dumps(record) + "\n")
                log.flush()
                if progress["step"] % config.checkpoint_every == 0 or progress["window"] == 0:
                    progress["allocated_seconds"] = prior_seconds + time.perf_counter() - started
                    checkpoint(
                        output / "checkpoint.pt",
                        model,
                        optimizer,
                        scheduler,
                        progress,
                        config,
                        asset_hash,
                    )
                if progress["window"] == 0:
                    checkpoint(
                        output / f"epoch-{epoch + 1}.pt",
                        model,
                        optimizer,
                        scheduler,
                        progress,
                        config,
                        asset_hash,
                    )
            if stop_after_steps is not None and progress["step"] >= stop_after_steps:
                break
    sync(device)
    progress["allocated_seconds"] = prior_seconds + time.perf_counter() - started
    checkpoint(output / "checkpoint.pt", model, optimizer, scheduler, progress, config, asset_hash)
    write_json(output / "status.json", {**progress, "complete": progress["step"] == total_steps})
    return progress


@torch.inference_mode()
def answer_nll(model, tokenizer, rows, device, microbatch, max_length):
    encoded = [encode_example(row, tokenizer, max_length) for row in rows]
    if any(row is None for row in encoded):
        raise ValueError("Held-out solution exceeds max_length; increase the common limit")
    total, count = 0.0, 0
    for start in range(0, len(encoded), microbatch):
        loss, tokens = loss_sum(
            model, collate(encoded[start : start + microbatch], tokenizer.pad_token_id, device)
        )
        total += float(loss)
        count += tokens
    return total / count


@torch.inference_mode()
def evaluate(
    root: Path,
    run: Path | None,
    output: Path,
    split="pilot",
    device="cuda",
    limit=None,
    checkpoint_name="checkpoint.pt",
    frozen_protocol: Path | None = None,
    base_config: RunConfig | None = None,
):
    manifest = verify_assets(root)
    if split == "test":
        if limit is not None or frozen_protocol is None:
            raise ValueError("Official test requires --frozen-protocol and no --limit")
        protocol = json.loads(frozen_protocol.read_text())
        if protocol.get("asset_hash") != digest(json.dumps(manifest, sort_keys=True)):
            raise ValueError("Frozen protocol assets do not match")
    saved = (
        None
        if run is None
        else torch.load(run / checkpoint_name, map_location="cpu", weights_only=True)
    )
    if saved is not None:
        config = RunConfig(**saved["config"])
    elif split == "test":
        config = RunConfig(**next(iter(protocol["selected_configs"].values())), seed=0)
    else:
        config = base_config or RunConfig(precision="fp32" if device == "cpu" else "bf16")
    if saved is not None and saved["asset_hash"] != digest(json.dumps(manifest, sort_keys=True)):
        raise ValueError("checkpoint/assets mismatch")
    if split == "test" and saved is not None:
        if config.train_limit or saved["progress"]["epoch"] != config.epochs:
            raise ValueError("Official test requires a full completed confirmation run")
        selected = protocol.get("selected_configs", {}).get(config.condition)
        expected = config_dict(config)
        expected.pop("seed")
        if selected != expected or config.seed not in protocol.get("confirmation_seeds", []):
            raise ValueError("Run is not a prespecified confirmation condition/seed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Evaluation output exists; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, tokenizer = load_model(root, device, config.precision, config.attention)
    merge_error = None
    rows = read_jsonl(root / f"{split}.jsonl")
    nll_rows = rows
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    if saved is not None:
        insert_adapters(
            model,
            config.condition,
            config.rank,
            config.gamma,
            config.seed,
            list(config.targets),
            root / "svd-cache",
        )
        restore_adapters(model, saved["adapters"])
        model.eval()
        probe = tokenizer(prompt(rows[0]["question"]), return_tensors="pt").to(device)
        before = model(**probe).logits
        merge_adapters(model)
        after = model(**probe).logits
        merge_error = float((before - after).abs().max())
        torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
        del before, after
    model.eval()
    model.config.use_cache = True
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # The training length limit must not discard held-out gold answers for NLL.
    nll = answer_nll(
        model, tokenizer, nll_rows, device, config.microbatch, model.config.max_position_embeddings
    )
    predictions = []
    tokenizer.padding_side = "left"
    for start in range(0, len(rows), config.microbatch):
        group = rows[start : start + config.microbatch]
        inputs = tokenizer(
            [prompt(row["question"]) for row in group], padding=True, return_tensors="pt"
        ).to(device)
        generated = model.generate(
            **inputs,
            do_sample=False,
            num_beams=1,
            max_new_tokens=config.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        for row, tokens in zip(group, generated[:, inputs.input_ids.shape[1] :], strict=True):
            ids = tokens.tolist()
            # Reaching the limit without EOS counts as truncated even if #### is present.
            truncated = tokenizer.eos_token_id not in ids and len(ids) >= config.max_new_tokens
            text = tokenizer.decode(ids, skip_special_tokens=True)
            predicted, gold = answer_number(text), answer_number(row["answer"])
            if gold is None:
                raise ValueError(f"Malformed gold answer: {row['id']}")
            predictions.append(
                {
                    "id": row["id"],
                    "text": text,
                    "answer": None if predicted is None else str(predicted),
                    "gold": str(gold),
                    "truncated": truncated,
                    "correct": not truncated and predicted is not None and predicted == gold,
                }
            )
    sync(device)
    write_jsonl(output / "predictions.jsonl", predictions)
    summary = {
        "split": split,
        "count": len(rows),
        "nll_count": len(nll_rows),
        "condition": config.condition if saved else "base",
        "seed": config.seed if saved else None,
        "config": config_dict(config),
        "asset_hash": digest(json.dumps(manifest, sort_keys=True)),
        "accuracy": 100 * sum(p["correct"] for p in predictions) / len(rows),
        "answer_nll": nll,
        "truncated": sum(p["truncated"] for p in predictions),
        "elapsed_seconds": time.perf_counter() - started,
        "merge_max_logit_error": merge_error,
        "environment": environment(device),
    }
    if saved:
        summary["training_progress"] = saved["progress"]
    if device == "cuda":
        summary.update(
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--config", type=Path, default=Path("configs/pilot.yaml"))
    training.add_argument("--resume", action="store_true")
    training.add_argument("--stop-after-epochs", type=int)
    training.add_argument("--stop-after-steps", type=int)
    scoring = sub.add_parser("evaluate")
    scoring.add_argument("--run", type=Path)
    scoring.add_argument("--split", choices=("pilot", "dev", "test"), default="pilot")
    scoring.add_argument("--limit", type=int)
    scoring.add_argument("--checkpoint", default="checkpoint.pt")
    scoring.add_argument("--frozen-protocol", type=Path)
    scoring.add_argument(
        "--base-config", type=Path, help="Shared evaluation settings for an untouched-base pilot"
    )
    for command in (training, scoring):
        command.add_argument("--assets", type=Path, default=Path("data/qwen-gsm8k"))
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.command == "train":
        config = RunConfig(**yaml.safe_load(args.config.read_text()))
        try:
            result = train(
                args.assets,
                args.output,
                config,
                args.device,
                args.resume,
                args.stop_after_epochs,
                args.stop_after_steps,
            )
        except Exception as error:
            if not isinstance(error, FileExistsError):
                write_json(
                    args.output / "failure.json",
                    {"type": type(error).__name__, "message": str(error)},
                )
            raise
    else:
        result = evaluate(
            args.assets,
            args.run,
            args.output,
            args.split,
            args.device,
            args.limit,
            args.checkpoint,
            args.frozen_protocol,
            None
            if args.base_config is None
            else RunConfig(**yaml.safe_load(args.base_config.read_text())),
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
