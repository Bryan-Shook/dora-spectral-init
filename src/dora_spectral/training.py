"""Shared training, resumable adapter checkpoints, and unmerged evaluation."""

import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .data import collate, parse_answer, prompt, score_answer
from .model import adapter_state, load_adapter_state


class BudgetExpired(RuntimeError):
    pass


class Budget:
    def __init__(self, seconds, previous=0):
        self.seconds = seconds
        self.previous = previous
        self.start = time.monotonic()
        self.cancelled = False

    @property
    def elapsed(self):
        return self.previous + time.monotonic() - self.start

    def check(self, reserve=0):
        if self.cancelled or self.elapsed + reserve >= self.seconds:
            raise BudgetExpired("run time budget reached; remaining work is unfinished")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def target_loss(logits, labels):
    """Sum shifted answer-token losses, leaving window normalization to the caller."""
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )


def token_count(batch):
    return int((batch["labels"][:, 1:] != -100).sum())


def epoch_windows(rows, seed, batch_size, epochs):
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(seed + 10000 + epoch).shuffle(order)
        for offset in range(0, len(order), batch_size):
            yield [rows[i] for i in order[offset : offset + batch_size]]


def schedule_multiplier(step, horizon, warmup_fraction):
    warmup = max(1, math.ceil(horizon * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, horizon - warmup)))


def load_model(config, prepared, device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config["model"],
        revision=prepared["model_revision"],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        attn_implementation=config["attention_backend"],
    ).to(device)
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def save_checkpoint(path, model, optimizer, step, metadata):
    adapters = adapter_state(model)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "adapters": adapters,
            "optimizer": optimizer.state_dict(),
            "step": step,
            "metadata": metadata,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
        },
        temporary,
    )
    temporary.replace(path)


def train(
    model, rows, config, seed, steps, directory, budget, metadata, device="cuda", overfit=False
):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config["learning_rate"], betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    horizon = math.ceil(len(rows) / config["batch_size"]) * config["epochs"]
    windows = epoch_windows(rows, seed, config["batch_size"], config["epochs"])
    if overfit:
        windows = (rows for _ in range(steps))
        horizon = steps
    path = directory / "checkpoint.pt"
    directory.mkdir(parents=True, exist_ok=True)
    start_step = 0
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved["metadata"] != metadata:
            raise ValueError("checkpoint protocol mismatch")
        load_adapter_state(model, saved["adapters"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = saved["step"]
        if start_step > steps:
            raise ValueError("checkpoint is beyond the requested training target")
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        random.setstate(saved["python_rng"])
    history = []
    log_path = directory / "training.jsonl"
    previous_seconds = 0
    if start_step and log_path.exists():
        previous_records = [json.loads(line) for line in log_path.read_text().splitlines()]
        previous_seconds = max((row["elapsed_seconds"] for row in previous_records), default=0)
    model.config.use_cache = False
    model.train()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    began = time.monotonic()
    try:
        for step, window in enumerate(windows):
            if step < start_step:
                continue
            if step >= steps:
                break
            budget.check(reserve=60)
            optimizer.zero_grad(set_to_none=True)
            lr = config["learning_rate"] * schedule_multiplier(
                step, horizon, config["warmup_fraction"]
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            total_tokens = sum(sum(token != -100 for token in row["labels"][1:]) for row in window)
            loss_value, active, padded = 0.0, 0, 0
            if not total_tokens:
                raise ValueError("empty answer-token accumulation window")
            for offset in range(0, len(window), config["microbatch_size"]):
                budget.check(reserve=30)
                batch = collate(
                    window[offset : offset + config["microbatch_size"]], config["pad_id"], device
                )
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                loss = target_loss(logits, batch["labels"]) / total_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                loss_value += loss.item()
                active += int(batch["attention_mask"].sum())
                padded += batch["input_ids"].numel()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            record = {
                "step": step + 1,
                "answer_nll": loss_value,
                "lr": lr,
                "gradient_norm": norm.item(),
                "clipped": norm.item() > 1,
                "active_tokens": active,
                "padded_tokens": padded,
                "elapsed_seconds": previous_seconds + time.monotonic() - began,
            }
            history.append(record)
            with (directory / "training.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            write_json(directory / "budget-progress.json", {"spent_seconds": budget.elapsed})
            if (step + 1) % 25 == 0:
                save_checkpoint(path, model, optimizer, step + 1, metadata)
            print(f"{directory.name}: step {step + 1}/{steps}, NLL {loss_value:.4f}", flush=True)
    except BudgetExpired:
        # Discard a partial accumulation window; preserve the last completed update.
        completed = history[-1]["step"] if history else start_step
        save_checkpoint(path, model, optimizer, completed, metadata)
        raise
    completed = history[-1]["step"] if history else start_step
    save_checkpoint(path, model, optimizer, completed, metadata)
    result = {
        "completed_steps": completed,
        "seconds": previous_seconds + time.monotonic() - began,
        "initial_nll": history[0]["answer_nll"] if history else None,
        "final_nll": history[-1]["answer_nll"] if history else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0,
    }
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    # Deduplicate records in case an infrastructure interruption rolled back a checkpoint.
    records = list({row["step"]: row for row in records}.values())
    result.update(
        active_tokens_per_second=sum(row["active_tokens"] for row in records) / result["seconds"],
        padded_tokens_per_second=sum(row["padded_tokens"] for row in records) / result["seconds"],
        clipping_fraction=sum(row["clipped"] for row in records) / len(records),
    )
    write_json(directory / "training-summary.json", result)
    return result


@torch.no_grad()
def answer_nll(model, rows, config, budget, device="cuda"):
    model.eval()
    loss, count = 0.0, 0
    for offset in range(0, len(rows), config["microbatch_size"]):
        budget.check(reserve=30)
        batch = collate(rows[offset : offset + config["microbatch_size"]], config["pad_id"], device)
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        loss += target_loss(logits, batch["labels"]).item()
        count += token_count(batch)
    return loss / count


@torch.no_grad()
def evaluate(model, tokenizer, rows, config, directory, budget, device="cuda"):
    model.eval()
    predictions = []
    began = time.monotonic()
    model.config.use_cache = True
    tokenizer.padding_side = "left"
    for offset in range(0, len(rows), config["evaluation_batch_size"]):
        budget.check(reserve=60)
        examples = rows[offset : offset + config["evaluation_batch_size"]]
        batch = tokenizer(
            [prompt(row["question"]) for row in examples], padding=True, return_tensors="pt"
        ).to(device)
        output = model.generate(
            **batch,
            do_sample=False,
            max_new_tokens=config["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output[:, batch["input_ids"].shape[1] :].tolist()
        for row, tokens in zip(examples, generated, strict=True):
            ended = tokenizer.eos_token_id in tokens
            if ended:
                tokens = tokens[: tokens.index(tokenizer.eos_token_id)]
            truncated = not ended and len(tokens) >= config["max_new_tokens"]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            parsed = parse_answer(text)
            predictions.append(
                {
                    "id": row["id"],
                    "text": text,
                    "parsed": str(parsed) if parsed is not None else None,
                    "correct": score_answer(text, row["answer"], truncated),
                    "truncated": truncated,
                }
            )
        write_json(directory / "predictions.json", predictions)
    result = {
        "count": len(rows),
        "accuracy": 100 * sum(r["correct"] for r in predictions) / len(rows),
        "truncation_rate": sum(r["truncated"] for r in predictions) / len(rows),
        "seconds": time.monotonic() - began,
    }
    write_json(directory / "evaluation.json", result)
    return result
