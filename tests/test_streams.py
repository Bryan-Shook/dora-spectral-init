import copy
import random

import pytest
import torch
from test_training import tiny_model

from dora_spectral.experiment import run_metadata
from dora_spectral.model import adapter_state, attach
from dora_spectral.streams import (
    STREAM_VERSION,
    data_order_stream,
    initialization_stream,
    reference_factor_seed,
)
from dora_spectral.training import epoch_windows


@pytest.mark.parametrize(
    "seed,expected",
    [
        (0, (18430430853280097008, 14951198113637959414, 15526567003889769464)),
        (101, (3371932083139265750, 17272009519655946368, 9280325769875697026)),
    ],
)
def test_streams_preserve_the_teammate_v1_seed_mapping(seed, expected):
    name = "model.layers.0.self_attn.q_proj"
    assert STREAM_VERSION == "dora-streams-v1"
    assert (
        initialization_stream(seed, name),
        reference_factor_seed(seed, name),
        data_order_stream(seed),
    ) == expected


def test_streams_distinguish_purpose_matrix_and_seed_block():
    names = [f"model.layers.{layer}.self_attn.{kind}_proj" for layer in range(28) for kind in "qv"]
    seeds = (0, 1, 2, 10, 40, 101, 202, 303)
    derived = [
        function(seed, name)
        for seed in seeds
        for name in names
        for function in (initialization_stream, reference_factor_seed)
    ] + [data_order_stream(seed) for seed in seeds]
    assert len(set(derived)) == len(derived)
    assert all(0 <= value < 2**64 for value in derived)


@pytest.mark.parametrize("condition", ["A-R", "D-R", "E-R", "L0", "D0"])
def test_initialization_is_stable_when_projection_registration_order_changes(condition):
    left = tiny_model()
    right = copy.deepcopy(left)
    attention = right.model.layers[0].self_attn
    attention._modules = dict(reversed(list(attention._modules.items())))
    left_audit = attach(left, condition, 2, 1, 101)
    right_audit = attach(right, condition, 2, 1, 101)
    assert left_audit["stream_version"] == STREAM_VERSION
    assert {row["name"]: row for row in left_audit["layers"]} == {
        row["name"]: row for row in right_audit["layers"]
    }
    first, second = adapter_state(left), adapter_state(right)
    assert first.keys() == second.keys()
    for name in first:
        for key in first[name]:
            torch.testing.assert_close(first[name][key], second[name][key], atol=0, rtol=0)


def test_controlled_arms_share_directions_across_rules_and_gamma():
    base = tiny_model()
    expected = {}
    for condition in ("A-R", "D-R", "E-R"):
        for gamma in (0.1, 0.3, 1.0):
            model = copy.deepcopy(base)
            audit = attach(model, condition, 2, gamma, 101)
            for row in audit["layers"]:
                name = row["name"]
                assert row["initialization_seed"] == initialization_stream(101, name)
                module = model.get_submodule(name)
                for factor in ("lora_A", "lora_B"):
                    value = getattr(module, factor)["default"].weight.detach() / gamma**0.5
                    key = name, factor
                    if key not in expected:
                        expected[key] = value.clone()
                    torch.testing.assert_close(value, expected[key], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_ordinary_references_share_a_and_zero_b(dtype):
    left = tiny_model().to(dtype)
    right = copy.deepcopy(left)
    for model, condition in ((left, "L0"), (right, "D0")):
        attach(model, condition, 2, 1, 101)
    for name, values in adapter_state(left).items():
        other = adapter_state(right)[name]
        torch.testing.assert_close(values["lora_A.default.weight"], other["lora_A.default.weight"])
        assert torch.count_nonzero(values["lora_B.default.weight"]) == 0
        assert torch.count_nonzero(other["lora_B.default.weight"]) == 0


def test_global_rng_consumption_does_not_change_initialization_or_data_order():
    left = tiny_model()
    right = copy.deepcopy(left)
    expected_order = list(epoch_windows(list(range(64)), 101, 8, 3))
    torch.rand(37)
    random.seed(54)
    random.random()
    torch_state, python_state = torch.get_rng_state(), random.getstate()
    attach(left, "A-R", 2, 1, 101)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert python_state == random.getstate()
    torch.rand(93)
    random.random()
    attach(right, "A-R", 2, 1, 101)
    for name, values in adapter_state(left).items():
        for key, value in values.items():
            torch.testing.assert_close(value, adapter_state(right)[name][key], atol=0, rtol=0)
    assert expected_order == list(epoch_windows(list(range(64)), 101, 8, 3))


def test_adjacent_seed_blocks_do_not_reuse_epoch_permutations():
    rows = list(range(64))
    first = list(epoch_windows(rows, 0, 64, 3))
    second = list(epoch_windows(rows, 1, 64, 3))
    assert len({tuple(order) for order in first + second}) == 6
    generator = random.Random(data_order_stream(0))
    expected = []
    for _ in range(3):
        order = rows.copy()
        generator.shuffle(order)
        expected.append(order)
    assert first == expected


def test_run_evidence_records_stream_version_and_shuffle_seed():
    metadata = run_metadata({}, "A-R", 101, {"examples_sha256": "fixture"})
    assert metadata["random_streams"] == {
        "version": STREAM_VERSION,
        "data_order_seed": data_order_stream(101),
    }
    assert "streams.py" in metadata["source_sha256"]
