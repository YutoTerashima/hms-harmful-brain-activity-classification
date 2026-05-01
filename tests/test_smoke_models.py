from __future__ import annotations

import pytest


def test_synthetic_spectrogram_model_forward():
    torch = pytest.importorskip("torch")
    from hms_solution.datasets import make_synthetic_batch
    from hms_solution.models import ModelSpec, create_model

    x, _ = make_synthetic_batch("spectrogram", batch_size=2)
    model = create_model(ModelSpec(name="spectrogram_cnn", family="spectrogram"))
    with torch.no_grad():
        logits = model(torch.tensor(x))
    assert tuple(logits.shape) == (2, 6)


def test_synthetic_raw_eeg_model_forward():
    torch = pytest.importorskip("torch")
    from hms_solution.datasets import make_synthetic_batch
    from hms_solution.models import ModelSpec, create_model

    x, _ = make_synthetic_batch("raw_eeg", batch_size=2)
    model = create_model(ModelSpec(name="resnet1d_gru", family="raw_eeg"))
    with torch.no_grad():
        logits = model(torch.tensor(x))
    assert tuple(logits.shape) == (2, 6)

