import copy
from fractions import Fraction

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dora_spectral.data import collate, encode_example, parse_answer, score_answer, split_examples
from dora_spectral.model import CONDITIONS, adapter_state, attach, load_adapter_state
from dora_spectral.training import Budget, epoch_windows, target_loss, train


class Tokenizer:
    eos_token_id = 999

    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def test_masking_calculator_and_padding():
    row = encode_example(Tokenizer(), {"question": "2 + 2?", "answer": "<<2+2=4>>4\n#### 4"})
    boundary = row["labels"].index(ord(" "))
    assert all(token == -100 for token in row["labels"][:boundary])
    assert row["labels"][-1] == 999
    assert ord("<") not in row["labels"]
    shorter = {"input_ids": [1, 2], "labels": [-100, 2]}
    batch = collate([row, shorter], 999)
    assert batch["attention_mask"][1].sum() == 2
    assert (batch["labels"][1, 2:] == -100).all()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("work\n#### 1,250.00", Fraction(1250)),
        ("#### -0.5", Fraction(-1, 2)),
        ("#### 1/2", Fraction(1, 2)),
        ("#### 1\nmore text", None),
        ("no delimiter 42", None),
        ("#### 1,25", None),
        ("#### 1/0", None),
        ("#### 2\n#### 3", Fraction(3)),
    ],
)
def test_parser(text, expected):
    assert parse_answer(text) == expected
    assert not score_answer(text, "#### 3", truncated=True)


def test_split_repeatability_duplicate_groups_and_sizes():
    examples = [{"question": f"question {i}", "answer": "#### 1"} for i in range(1100)]
    examples.append({"question": " QUESTION   3 ", "answer": "#### 1"})
    split = split_examples(examples)
    assert split == split_examples(examples)
    assert sum(map(len, split.values())) == 1101
    locations = [
        name for name, rows in split.items() if any(row["id"] in (3, 1100) for row in rows)
    ]
    assert len(locations) == 1
    assert len(split["pilot"]) >= 500 and len(split["development"]) >= 500


def test_shifted_token_loss_and_accumulation():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 2, 3], [-100, 1, 2, -100]])
    total = (labels[:, 1:] != -100).sum()
    loss = target_loss(logits, labels) / total
    expected = torch.nn.functional.cross_entropy(
        torch.stack([logits[0, 1], logits[0, 2], logits[1, 0], logits[1, 1]]),
        torch.tensor([2, 3, 1, 2]),
    )
    torch.testing.assert_close(loss, expected)
    full_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    accumulated = sum(target_loss(logits[i : i + 1], labels[i : i + 1]) / total for i in range(2))
    torch.testing.assert_close(torch.autograd.grad(accumulated, logits)[0], full_gradient)


def tiny_model():
    torch.manual_seed(5)
    return Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            attention_dropout=0,
        )
    )


@pytest.mark.parametrize("condition", CONDITIONS)
def test_tiny_transformer_initialization_learning_reload(condition, tmp_path):
    model = tiny_model().eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    initial = model(ids).logits.detach()
    audit = attach(model, condition, rank=2, gamma=1, seed=7, cache_dir=tmp_path)
    torch.testing.assert_close(model(ids).logits, initial, atol=1e-6, rtol=1e-5)
    assert len(audit["layers"]) == 2
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    assert not torch.equal(model(ids).logits, initial)
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert torch.equal(parameter, frozen[name])
    trained = model(ids).logits.detach()
    restored = tiny_model().eval()
    attach(restored, condition, rank=2, gamma=1, seed=7, cache_dir=tmp_path)
    load_adapter_state(restored, adapter_state(model))
    torch.testing.assert_close(restored(ids).logits, trained, atol=0, rtol=0)


def test_resume_matches_uninterrupted_training(tmp_path):
    model = tiny_model()
    attach(model, "D-R", rank=2, gamma=1, seed=2)
    split_model = copy.deepcopy(model)
    rows = [
        {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]},
        {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
    ]
    config = {
        "learning_rate": 0.01,
        "batch_size": 2,
        "microbatch_size": 1,
        "epochs": 3,
        "warmup_fraction": 0.03,
        "pad_id": 0,
    }
    metadata = {"condition": "D-R", "seed": 0}
    train(model, rows, config, 0, 3, tmp_path / "full", Budget(1000), metadata, device="cpu")
    train(split_model, rows, config, 0, 1, tmp_path / "split", Budget(1000), metadata, device="cpu")
    train(split_model, rows, config, 0, 3, tmp_path / "split", Budget(1000), metadata, device="cpu")
    for left, right in zip(model.parameters(), split_model.parameters(), strict=True):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_seed_streams_and_partial_windows():
    rows = list(range(5))
    windows = list(epoch_windows(rows, 0, 2, 2))
    assert [len(row) for row in windows] == [2, 2, 1, 2, 2, 1]
    assert sorted(sum(windows[:3], [])) == rows
    torch.manual_seed(913)
    assert windows == list(epoch_windows(rows, 0, 2, 2))


def test_budget_reserve_and_prior_spend():
    from dora_spectral.training import BudgetExpired

    budget = Budget(100, previous=90)
    with pytest.raises(BudgetExpired):
        budget.check(reserve=11)
    budget.check(reserve=1)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_bf16_initialization_and_generation_reload(condition):
    model = tiny_model().to(torch.bfloat16).eval()
    ids = torch.tensor([[1, 2, 3]])
    initial = model(ids).logits.detach()
    attach(model, condition, rank=2, gamma=1, seed=3)
    torch.testing.assert_close(model(ids).logits, initial, atol=0, rtol=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    logits = model(ids).logits.detach()
    generated = model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0)
    restored = tiny_model().to(torch.bfloat16).eval()
    attach(restored, condition, rank=2, gamma=1, seed=3)
    load_adapter_state(restored, adapter_state(model))
    torch.testing.assert_close(restored(ids).logits, logits, atol=0, rtol=0)
    assert torch.equal(
        restored.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=0), generated
    )
