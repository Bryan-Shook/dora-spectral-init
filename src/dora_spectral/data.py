"""Deterministic training splits, answer masking, and strict GSM8K scoring."""

import hashlib
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction

import torch


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def split_examples(examples: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for index, example in enumerate(examples):
        row = {"id": index, **example}
        groups[" ".join(row["question"].split()).casefold()].append(row)
    ordered = sorted(
        groups.values(), key=lambda rows: digest("dora-gsm8k-v1\n" + rows[0]["question"])
    )
    result = {"pilot": [], "development": [], "train": []}
    for rows in ordered:
        partition = (
            "pilot"
            if len(result["pilot"]) < 500
            else ("development" if len(result["development"]) < 500 else "train")
        )
        result[partition].extend(rows)
    return result


def prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def encode_example(tokenizer, example: dict) -> dict:
    # Encode the inference prefix separately: every prefix token is masked.
    prefix = tokenizer.encode(prompt(example["question"]), add_special_tokens=False)
    answer = " " + re.sub(r"<<.*?>>", "", example["answer"])
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    return {**example, "input_ids": prefix + target, "labels": [-100] * len(prefix) + target}


def collate(examples: list[dict], pad_id: int, device="cpu") -> dict:
    length = max(len(row["input_ids"]) for row in examples)
    ids, labels, masks = [], [], []
    for row in examples:
        padding = length - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * padding)
        labels.append(row["labels"] + [-100] * padding)
        masks.append([1] * len(row["input_ids"]) + [0] * padding)
    return {
        key: torch.tensor(value, dtype=torch.long, device=device)
        for key, value in (("input_ids", ids), ("labels", labels), ("attention_mask", masks))
    }


def parse_answer(text: str) -> Fraction | None:
    if "####" not in text:
        return None
    value = text.rsplit("####", 1)[1].strip()
    match = re.fullmatch(
        r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?)\s*[.!]?", value
    )
    if not match:
        return None
    try:
        value = match[1].replace(",", "")
        if "/" in value:
            numerator, denominator = value.split("/")
            return Fraction(Decimal(numerator)) / Fraction(Decimal(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def score_answer(text: str, reference: str, truncated: bool = False) -> bool:
    predicted, expected = parse_answer(text), parse_answer(reference)
    return (
        not truncated and predicted is not None and expected is not None and predicted == expected
    )
