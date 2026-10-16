import copy
import json

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dora_spectral.data import collate
from dora_spectral.diagnostics import (
    MovementRecorder,
    adapter_layers,
    counterfactual,
    effective_weight,
)
from dora_spectral.model import adapter_state, attach
from dora_spectral.training import (
    Budget,
    BudgetExpired,
    target_loss,
    token_count,
    train,
    training_records,
)
from dora_spectral.validation import gradient_checks


def model():
    torch.manual_seed(12)
    result = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    attach(result, "D-R", 2, 1, 4)
    return result


def batch():
    return collate([{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}], 0)


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            assert_nested_equal(first, second)
    else:
        assert left == right


def test_effective_movement_accounts_for_initial_cancellation(tmp_path):
    actual = model()
    recorder = MovementRecorder(actual, tmp_path, 1)
    recorder.record(0)
    recorder.before(0)
    with torch.no_grad():
        for _, module in adapter_layers(actual, True):
            torch.testing.assert_close(effective_weight(module), module.base_layer.weight)
            module.lora_A["default"].weight.add_(0.02)
    recorder.record(1)
    records = [
        json.loads(line) for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()
    ]
    assert all(row["initial_product_norm"] > 0 for row in records[0]["layers"])
    assert all(row["relative_net_movement"] < 1e-6 for row in records[0]["layers"])
    assert all(row["step_norm"] > 0 for row in records[1]["layers"])


def test_frozen_initial_factors_match_parameter_layout_without_aliasing():
    actual = model()
    for _, module in adapter_layers(actual, True):
        for initial, parameter in (
            (module.initial_A, module.lora_A["default"].weight),
            (module.initial_B, module.lora_B["default"].weight),
        ):
            assert initial.stride() == parameter.stride()
            assert initial.data_ptr() != parameter.data_ptr()
            torch.testing.assert_close(initial, parameter, atol=0, rtol=0)


def test_counterfactual_restores_training_state_and_matches_step_norms():
    actual, inputs = model(), batch()
    parameters = [parameter for parameter in actual.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0)

    def loss_function(logits, labels):
        return target_loss(logits, labels) / token_count(inputs)

    loss_function(actual(input_ids=inputs["input_ids"]).logits, inputs["labels"]).backward()
    optimizer.step()
    actual.eval()
    original = copy.deepcopy(adapter_state(actual))
    moments = copy.deepcopy(optimizer.state_dict())
    gradients = [parameter.grad.clone() for parameter in parameters]
    rng = torch.get_rng_state()
    result = counterfactual(actual, optimizer, inputs, loss_function)
    assert_nested_equal(original, adapter_state(actual))
    assert_nested_equal(moments, optimizer.state_dict())
    for parameter, gradient in zip(parameters, gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)
    assert torch.equal(rng, torch.get_rng_state())
    assert not actual.training
    assert all(
        module.lora_variant["default"].rule == "detached"
        for _, module in adapter_layers(actual, True)
    )
    assert result["relative_distance"] > 0
    assert (
        abs(
            result["probes"]["exact-equal-True"]["effective_step_norm"]
            - result["probes"]["detached-equal-True"]["effective_step_norm"]
        )
        < 1e-10
    )


def test_diagnostics_do_not_change_training_and_survive_resume(tmp_path):
    left = model()
    right = copy.deepcopy(left)
    rows = [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}]
    config = {
        "learning_rate": 0.01,
        "batch_size": 1,
        "microbatch_size": 1,
        "epochs": 3,
        "warmup_fraction": 0.03,
        "pad_id": 0,
    }
    train(left, rows, config, 1, 3, tmp_path / "plain", Budget(1000), {}, device="cpu")
    config = dict(config, diagnostics=True, counterfactuals=True)
    train(right, rows, config, 1, 1, tmp_path / "recorded", Budget(1000), {}, device="cpu")
    train(right, rows, config, 1, 3, tmp_path / "recorded", Budget(1000), {}, device="cpu")
    assert_nested_equal(adapter_state(left), adapter_state(right))
    records = [
        json.loads(line)
        for line in (tmp_path / "recorded/diagnostics.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in records] == [0, 1, 3]


def test_independent_validation_gradient_checks():
    torch.manual_seed(2)
    result = gradient_checks(torch.randn(5, 9))
    assert result["fp64_exact_gradcheck"]
    assert all(result[rule]["passed"] for rule in ("additive", "exact", "detached"))


def test_resume_recovers_partial_final_log_and_retains_evidence(tmp_path):
    actual = model()
    rows = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3]}]
    config = {
        "learning_rate": 0.01,
        "batch_size": 1,
        "microbatch_size": 1,
        "epochs": 3,
        "warmup_fraction": 0.03,
        "pad_id": 0,
    }
    train(actual, rows, config, 0, 1, tmp_path, Budget(1000), {}, device="cpu")
    with (tmp_path / "training.jsonl").open("a") as handle:
        handle.write('{"step": 2, "answer_nll":')
    result = train(actual, rows, config, 0, 3, tmp_path, Budget(1000), {}, device="cpu")
    assert result["completed_steps"] == 3
    assert [row["step"] for row in training_records(tmp_path / "training.jsonl")] == [1, 2, 3]
    assert "discarded_partial_line" in (tmp_path / "training-log-recovery.jsonl").read_text()
    (tmp_path / "corrupt.jsonl").write_text('{"step":\n{"step": 2}\n')
    with pytest.raises(ValueError, match="before final line"):
        training_records(tmp_path / "corrupt.jsonl", repair=True)


def test_budget_interruption_at_post_update_probe_resumes_without_repeating_update(
    tmp_path, monkeypatch
):
    import dora_spectral.training as training

    expected = model()
    interrupted = copy.deepcopy(expected)
    rows = [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}]
    config = {
        "learning_rate": 0.01,
        "batch_size": 1,
        "microbatch_size": 1,
        "epochs": 51,
        "warmup_fraction": 0.03,
        "pad_id": 0,
        "counterfactuals": True,
    }
    train(expected, rows, config, 101, 51, tmp_path / "full", Budget(1000), {}, device="cpu")
    original_probe = training.counterfactual

    def expire_at_step_50(*args, **kwargs):
        # Step 0 runs before any moments exist; step 50 has completed real updates.
        optimizer = args[1]
        if optimizer.state:
            assert all(int(value["step"]) == 50 for value in optimizer.state.values())
            raise BudgetExpired("injected post-update diagnostic timeout")
        return original_probe(*args, **kwargs)

    monkeypatch.setattr(training, "counterfactual", expire_at_step_50)
    path = tmp_path / "interrupted"
    with pytest.raises(BudgetExpired):
        train(interrupted, rows, config, 101, 51, path, Budget(1000), {}, device="cpu")
    saved = torch.load(path / "checkpoint.pt", weights_only=False)
    assert saved["step"] == 50
    assert all(int(value["step"]) == 50 for value in saved["optimizer"]["state"].values())
    assert training_records(path / "training.jsonl")[-1]["step"] == 50
    monkeypatch.setattr(training, "counterfactual", original_probe)
    restored = model()
    train(restored, rows, config, 101, 51, path, Budget(1000), {}, device="cpu")
    assert_nested_equal(adapter_state(expected), adapter_state(restored))
    expected_checkpoint = torch.load(tmp_path / "full/checkpoint.pt", weights_only=False)
    actual_checkpoint = torch.load(path / "checkpoint.pt", weights_only=False)
    assert_nested_equal(expected_checkpoint["optimizer"], actual_checkpoint["optimizer"])
    probe_lines = (path / "counterfactuals.jsonl").read_text().splitlines()
    probes = [json.loads(line) for line in probe_lines]
    assert [row["step"] for row in probes] == [0, 50]
