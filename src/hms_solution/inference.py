"""Inference helpers for Kaggle submission files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import HMS_CLASSES, PROBABILITY_COLUMNS
from .metrics import softmax


def probabilities_to_submission(
    eeg_ids: list[int] | np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Create a Kaggle-style submission frame."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(HMS_CLASSES):
        raise ValueError(f"Expected probability shape (n, {len(HMS_CLASSES)}), got {probs.shape}")
    probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-12)
    frame = pd.DataFrame(probs, columns=PROBABILITY_COLUMNS)
    frame.insert(0, "eeg_id", eeg_ids)
    return frame


def logits_to_submission(eeg_ids: list[int] | np.ndarray, logits: np.ndarray) -> pd.DataFrame:
    """Create a Kaggle-style submission frame from logits."""

    return probabilities_to_submission(eeg_ids, softmax(logits, axis=1))


def write_submission(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a submission CSV after validating required columns."""

    missing = [column for column in ["eeg_id", *PROBABILITY_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing submission columns: {missing}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ["eeg_id", *PROBABILITY_COLUMNS]].to_csv(output, index=False)

