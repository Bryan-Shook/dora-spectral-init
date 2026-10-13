"""Offline tests: no pretrained downloads or GPU allocation."""

import copy
import json
from decimal import Decimal

import pytest
import torch
from torch.nn import functional as F

from dora_spectral.data import (
    answer_number,
    collate,
    digest,
    encode_example,
    split_gsm8k,
    write_jsonl,
)
from dora_spectral.experiment import RunConfig, evaluate, loss_sum, train
from dora_spectral.model import (
    CONDITIONS,
    adapter_state,
    insert_adapters,
    merge_adapters,
    restore_adapters,
)
from dora_spectral.protocol import grid, promote


class ToyTokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 8 for char in text]


def test_answer_mask_padding_and_no_silent_truncation():
    row = {"id": "x", "question": "q", "answer": "<<1+1=2>>2 #### 2"}
    encoded = encode_example(row, ToyTokenizer(), 100)
    prefix = "Question: q\nAnswer:"
    assert encoded["labels"][: len(prefix)] == [-100] * len(prefix)
    assert encoded["labels"][len(prefix) :] == ToyTokenizer().encode(" 2 #### 2") + [9]
    assert encode_example(row, ToyTokenizer(), 3) is None
    padded = collate([encoded, {"input_ids": [1, 2], "labels": [-100, 2]}], 9)
    assert padded["labels"][1, 2:].eq(-100).all()
    assert padded["attention_mask"][1, 2:].eq(0).all()


@pytest.mark.parametrize(
    "text,value",
    [
        ("reason #### 1,234.00", Decimal(1234)),
        ("#### 2\n#### -0.5", Decimal("-0.5")),
        ("answer 2", None),
        ("#### NaN", None),
        ("#### 1,23", None),
        ("#### 2 apples", None),
    ],
)
def test_exact_parser(text, value):
    assert answer_number(text) == value


def test_split_hash_order_and_duplicate_groups():
    rows = [{"question": str(i), "answer": "#### 1"} for i in range(10)]
    actual = split_gsm8k(rows, [], holdout=2)["partitions"]
    expected = sorted(rows, key=lambda row: digest("dora-gsm8k-v1\n" + row["question"]))
    assert [r["question"] for r in actual["pilot"]] == [r["question"] for r in expected[:2]]
    duplicated = rows + [{"question": " 0 ", "answer": "#### 1"}]
    parts = split_gsm8k(duplicated, [], holdout=2)["partitions"]
    owners = [
        name for name, group in parts.items() if any(r["question"].strip() == "0" for r in group)
    ]
    assert len(owners) == 1
    overlap = split_gsm8k(rows, [{"question": "0", "answer": "#### 1"}], holdout=2)
    assert overlap["train_test_overlap_removed"] == ["train-0"]


def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=2,
        attention_dropout=0,
    )
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config)


@pytest.mark.parametrize("condition", list(CONDITIONS))
def test_tiny_qwen_initial_logits_freeze_merge_and_restore(condition, tmp_path):
    model = tiny_model().eval()
    ids = torch.tensor([[3, 4, 5, 6]])
    reference = model(ids).logits.detach()
    metadata = insert_adapters(model, condition, 2, 1.0, 3, ["q_proj", "v_proj"], tmp_path)
    assert len(metadata["layers"]) == 4
    torch.testing.assert_close(model(ids).logits, reference, atol=1e-6, rtol=1e-5)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == (name.endswith((".A", ".B", ".magnitude")))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    model(ids).logits.square().mean().backward()
    optimizer.step()
    trained = model(ids).logits.detach()
    state = copy.deepcopy(adapter_state(model))
    restored = tiny_model().eval()
    insert_adapters(restored, condition, 2, 1.0, 3, ["q_proj", "v_proj"], tmp_path)
    restore_adapters(restored, state)
    torch.testing.assert_close(restored(ids).logits, trained)
    merge_adapters(restored)
    torch.testing.assert_close(restored(ids).logits, trained)


def test_token_weighted_accumulation_matches_whole_batch():
    model = tiny_model()
    insert_adapters(model, "D-R", 2, 1, 2, ["q_proj", "v_proj"])
    rows = [
        {"input_ids": [3, 4, 5], "labels": [-100, 4, 5]},
        {"input_ids": [3, 4, 5, 6, 7], "labels": [-100, -100, 5, 6, 7]},
    ]
    total, count = loss_sum(model, collate(rows, 2))
    (total / count).backward()
    expected = [p.grad.clone() for p in model.parameters() if p.requires_grad]
    model.zero_grad()
    for row in rows:
        loss, _ = loss_sum(model, collate([row], 2))
        (loss / count).backward()
    for left, parameter in zip(
        expected, [p for p in model.parameters() if p.requires_grad], strict=True
    ):
        torch.testing.assert_close(left, parameter.grad, atol=1e-7, rtol=1e-5)
    # Independently verify shifted cross entropy and the supervised token count.
    batch = collate(rows, 2)
    logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    hand = F.cross_entropy(
        logits[:, :-1].reshape(-1, 32),
        batch["labels"][:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    torch.testing.assert_close(total, hand)
    assert count == 5


@pytest.fixture
def assets(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    root = tmp_path / "assets"
    (root / "model").mkdir(parents=True)
    tokenizer = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "Question": 3,
                ":": 4,
                "Answer": 5,
                "####": 6,
                "1": 7,
                "2": 8,
                "q": 9,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
        bos_token="[BOS]",
        pad_token="[EOS]",
    )
    fast.save_pretrained(root / "model")
    tiny_model().save_pretrained(root / "model")
    files = {}
    for split, size in (("train", 5), ("pilot", 2), ("dev", 2), ("test", 2)):
        rows = [
            {"id": f"{split}-{i}", "question": f"q {i}", "answer": "1 #### 1"} for i in range(size)
        ]
        write_jsonl(root / f"{split}.jsonl", rows)
        files[split] = {"count": size, "sha256": digest((root / f"{split}.jsonl").read_text())}
    (root / "manifest.json").write_text(
        json.dumps({"model_id": "tiny-local", "model_revision": "test", "files": files})
    )
    return root


def test_interrupted_training_resume_matches_uninterrupted_and_evaluates(assets, tmp_path):
    config = RunConfig(
        condition="E-R",
        rank=2,
        seed=4,
        epochs=2,
        microbatch=1,
        accumulation=2,
        precision="fp32",
        attention="eager",
        max_length=64,
        max_new_tokens=3,
        checkpoint_every=1,
    )
    first = tmp_path / "resumed"
    second = tmp_path / "uninterrupted"
    train(assets, first, config, device="cpu", stop_after_steps=2)
    train(assets, first, config, device="cpu", resume=True)
    train(assets, second, config, device="cpu")
    resumed = torch.load(first / "checkpoint.pt", weights_only=True)
    full = torch.load(second / "checkpoint.pt", weights_only=True)
    assert resumed["progress"]["step"] == full["progress"]["step"] == 6
    for name, state in resumed["adapters"].items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, full["adapters"][name][key], atol=0, rtol=0)
    result = evaluate(assets, first, tmp_path / "eval", device="cpu", limit=1)
    assert result["count"] == 1 and result["nll_count"] == 2
    assert result["merge_max_logit_error"] == 0
    with pytest.raises(ValueError, match="Official test"):
        evaluate(assets, first, tmp_path / "bad", split="test", device="cpu")
    with pytest.raises(ValueError, match="resume config"):
        train(
            assets,
            first,
            RunConfig(**{**config.__dict__, "learning_rate": 0.01}),
            device="cpu",
            resume=True,
        )


def test_equal_search_and_per_scale_promotions():
    jobs = grid(RunConfig())
    assert len(jobs) == 96
    assert all(sum(j["config"]["condition"] == c for j in jobs) == 12 for c in CONDITIONS)
    results = [
        {**job, "summary": {"config": job["config"], "accuracy": i, "answer_nll": 1}}
        for i, job in enumerate(jobs)
    ]
    promoted = promote(results)
    assert len(promoted) == 24
    for condition in ("A-S", "A-R", "D-S", "D-R", "E-S", "E-R"):
        assert {
            r["config"]["gamma"] for r in promoted if r["config"]["condition"] == condition
        } == {0.1, 0.3, 1.0}


def test_qwen_numerical_check_and_baseline_evaluation(assets, tmp_path):
    from dora_spectral.check_model import check

    report = check(assets, "cpu", "fp32", rank=2, gamma=1, tolerance=1e-5)
    assert report["accepted"]
    assert set(report["conditions"]) == set(CONDITIONS)
    result = evaluate(
        assets,
        None,
        tmp_path / "base-score",
        device="cpu",
        limit=1,
        base_config=RunConfig(precision="fp32", max_new_tokens=3, microbatch=1),
    )
    assert result["condition"] == "base"


def test_prepare_pins_revisions_and_writes_hashed_local_splits(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import datasets
    import huggingface_hub

    from dora_spectral.prepare import prepare, verify_assets

    calls = {}

    class FakeApi:
        def model_info(self, model, revision):
            return SimpleNamespace(sha="model-immutable-sha")

        def dataset_info(self, dataset, revision):
            return SimpleNamespace(sha="dataset-immutable-sha")

    def snapshot(model, **kwargs):
        calls["model_revision"] = kwargs["revision"]
        kwargs["local_dir"].mkdir(parents=True)

    def load(dataset, config, **kwargs):
        calls["dataset_revision"] = kwargs["revision"]
        return {
            "train": [{"question": str(i), "answer": "#### 1"} for i in range(1002)],
            "test": [{"question": "unseen", "answer": "#### 1"}],
        }

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(datasets, "load_dataset", load)
    manifest = prepare(tmp_path / "assets", "Qwen/Qwen2.5-1.5B", "main", "main")
    assert calls == {
        "model_revision": "model-immutable-sha",
        "dataset_revision": "dataset-immutable-sha",
    }
    assert manifest["files"]["train"]["count"] == 2
    assert verify_assets(tmp_path / "assets") == manifest
    with pytest.raises(FileExistsError):
        prepare(tmp_path / "assets", "model", "main", "main")
    (tmp_path / "assets" / "train.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="partition hash"):
        verify_assets(tmp_path / "assets")


def test_paired_contrasts_and_bootstrap(tmp_path):
    from dora_spectral.analyze import analyze
    from dora_spectral.experiment import config_dict

    seeds = [101, 202, 303]
    selected = {}
    for condition in CONDITIONS:
        config = config_dict(RunConfig(condition=condition))
        config.pop("seed")
        selected[condition] = config
        for seed in seeds:
            directory = tmp_path / f"{condition}-seed{seed}" / "test"
            directory.mkdir(parents=True)
            correct = condition in ("D-S", "E-S")
            predictions = [{"id": str(i), "correct": correct} for i in range(3)]
            write_jsonl(directory / "predictions.jsonl", predictions)
            (directory / "summary.json").write_text(
                json.dumps(
                    {
                        "split": "test",
                        "config": {**config, "seed": seed},
                        "asset_hash": "hash",
                        "accuracy": 100 if correct else 0,
                        "count": 3,
                    }
                )
            )
    protocol = {"selected_configs": selected, "confirmation_seeds": seeds, "asset_hash": "hash"}
    result = analyze(tmp_path, protocol, resamples=20)
    assert result["contrasts"]["J"]["mean"] == 100
    assert result["contrasts"]["I"]["mean"] == 0
    assert result["question_bootstrap"]["J"]["95_interval"] == [100, 100]
