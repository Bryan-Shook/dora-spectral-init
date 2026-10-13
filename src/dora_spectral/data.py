"""Locked GSM8K partitions, answer-only labels, and exact numeric scoring."""

import hashlib
import json
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def split_gsm8k(train: list[dict], test: list[dict], holdout: int = 500) -> dict:
    """Group normalized duplicates; preserve official test and report overlap removals."""

    def normalized(question):
        return " ".join(question.split()).casefold()

    test_questions = {normalized(row["question"]) for row in test}
    groups = defaultdict(list)
    removed = []
    for index, row in enumerate(train):
        item = {**row, "id": f"train-{index}", "question_hash": digest(row["question"])}
        if normalized(row["question"]) in test_questions:
            removed.append(item["id"])
        else:
            groups[normalized(row["question"])].append(item)
    ordered = sorted(
        groups.values(), key=lambda g: min(digest("dora-gsm8k-v1\n" + row["question"]) for row in g)
    )
    partitions = {"pilot": [], "dev": [], "train": []}
    for group in ordered:
        destination = (
            "pilot"
            if len(partitions["pilot"]) < holdout
            else ("dev" if len(partitions["dev"]) < holdout else "train")
        )
        partitions[destination].extend(group)
    if not partitions["train"]:
        raise ValueError("not enough training examples after holdout/duplicate checks")
    partitions["test"] = [
        {**row, "id": f"test-{i}", "question_hash": digest(row["question"])}
        for i, row in enumerate(test)
    ]
    return {
        "partitions": partitions,
        "duplicate_groups": sum(len(g) > 1 for g in ordered),
        "train_test_overlap_removed": removed,
    }


def prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def encode_example(row: dict, tokenizer, max_length: int) -> dict | None:
    prefix = tokenizer.encode(prompt(row["question"]), add_special_tokens=False)
    solution = re.sub(r"<<.*?>>", "", row["answer"])
    answer = tokenizer.encode(" " + solution, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer must have an EOS token")
    ids = prefix + answer + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        return None  # Caller records exclusions; never truncate away the answer.
    return {
        "input_ids": ids,
        "labels": [-100] * len(prefix) + answer + [tokenizer.eos_token_id],
        "id": row["id"],
    }


def collate(rows: list[dict], pad_id: int, device: str = "cpu") -> dict:
    length = max(len(row["input_ids"]) for row in rows)
    return {
        "input_ids": torch.tensor(
            [row["input_ids"] + [pad_id] * (length - len(row["input_ids"])) for row in rows],
            device=device,
        ),
        "attention_mask": torch.tensor(
            [[1] * len(row["input_ids"]) + [0] * (length - len(row["input_ids"])) for row in rows],
            device=device,
        ),
        "labels": torch.tensor(
            [row["labels"] + [-100] * (length - len(row["labels"])) for row in rows], device=device
        ),
    }


def answer_number(text: str) -> Decimal | None:
    """Strict last-#### parser; trailing prose, fractions, NaN, etc. are malformed."""
    if "####" not in text:
        return None
    value = text.rsplit("####", 1)[1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?", value):
        return None
    try:
        return Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
