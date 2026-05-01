from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_original_notebooks_are_present_and_parseable():
    nbformat = pytest.importorskip("nbformat")
    expected = [
        "model1/eeg_gen_spec.ipynb",
        "model1/infer.ipynb",
        "model1/train_eff0_stage1.ipynb",
        "model1/train_eff0_stage2.ipynb",
        "model2/hms-resnet1d-gru-infer.ipynb",
        "model2/hms-resnet1d-gru-train.ipynb",
        "model3/hms-eff1-train-stage1.ipynb",
        "model3/hms-eff1-train-stage2.ipynb",
        "model3/hms-infer.ipynb",
    ]
    for relative_path in expected:
        path = REPO_ROOT / "notebooks" / "original" / relative_path
        assert path.exists(), relative_path
        notebook = nbformat.read(path, as_version=4)
        assert notebook.cells, relative_path


def test_configs_load():
    from hms_solution.config import load_config

    for name in [
        "model1_effnet_spectrogram.yaml",
        "model2_resnet1d_gru.yaml",
        "model3_effnet_official_spec.yaml",
        "ensemble.yaml",
    ]:
        config = load_config(REPO_ROOT / "configs" / name)
        assert config["name"]


def test_ensemble_preserves_submission_shape():
    from hms_solution.ensemble import blend_predictions
    from hms_solution.inference import probabilities_to_submission

    eeg_ids = [1, 2, 3]
    first = probabilities_to_submission(
        eeg_ids,
        [
            [0.50, 0.10, 0.10, 0.10, 0.10, 0.10],
            [0.10, 0.50, 0.10, 0.10, 0.10, 0.10],
            [0.10, 0.10, 0.50, 0.10, 0.10, 0.10],
        ],
    )
    second = probabilities_to_submission(
        eeg_ids,
        [
            [0.20, 0.20, 0.20, 0.20, 0.10, 0.10],
            [0.10, 0.20, 0.20, 0.20, 0.20, 0.10],
            [0.10, 0.10, 0.20, 0.20, 0.20, 0.20],
        ],
    )
    blended = blend_predictions([first, second], [0.7, 0.3])
    assert list(blended.columns) == list(first.columns)
    assert blended.shape == (3, 7)
    assert pytest.approx(blended.drop(columns=["eeg_id"]).sum(axis=1).tolist()) == [1.0, 1.0, 1.0]


def test_public_processed_artifacts_are_non_empty():
    class_counts = pd.read_csv(REPO_ROOT / "data" / "processed" / "public_development_class_counts.csv")
    model_summary = pd.read_csv(REPO_ROOT / "reports" / "results" / "model_family_summary.csv")
    assert len(class_counts) == 6
    assert class_counts["expert_segments"].sum() == 106800
    assert set(model_summary["component"]) == {"model1", "model2", "model3", "ensemble"}


def test_readme_references_certificate_and_competition():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "assets/kaggle_certificate.png" in readme
    assert "hms-harmful-brain-activity-classification" in readme
    assert "123rd of 2,767 teams" in readme

