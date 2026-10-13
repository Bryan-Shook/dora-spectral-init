import pytest
import torch

from dora_spectral.numerics import logit_errors


def test_padding_cannot_inflate_valid_token_drift():
    original = torch.tensor([[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]])
    changed = original.clone()
    changed[0, 2] = torch.tensor([100.0, -100.0])
    valid = torch.tensor([[True, True, False]])
    measured = logit_errors(changed, original, valid)
    assert measured["positions"] == 2
    assert measured["max_error"] == 0
    assert measured["relative_rms"] == 0
    assert measured["next_token_agreement"] == 1
    assert logit_errors(changed, original)["max_error"] == 102


def test_drift_metric_tracks_selected_tokens():
    original = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    changed = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    measured = logit_errors(changed, original, torch.tensor([[True, False]]))
    assert measured["max_error"] == 1
    assert measured["next_token_agreement"] == 0
    assert measured["positions"] == 1


def test_empty_or_mismatched_token_selection_rejected():
    logits = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="at least one"):
        logit_errors(logits, logits, torch.zeros(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="mask"):
        logit_errors(logits, logits, torch.ones(1, 3, dtype=torch.bool))
