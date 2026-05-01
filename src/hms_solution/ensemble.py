"""Submission blending utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import PROBABILITY_COLUMNS


def load_prediction(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [column for column in ["eeg_id", *PROBABILITY_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame.loc[:, ["eeg_id", *PROBABILITY_COLUMNS]].copy()


def blend_predictions(frames: Iterable[pd.DataFrame], weights: Iterable[float]) -> pd.DataFrame:
    """Blend probability submissions with a weighted average."""

    frame_list = list(frames)
    weight_array = np.asarray(list(weights), dtype=np.float64)
    if len(frame_list) != len(weight_array):
        raise ValueError("Number of frames must match number of weights.")
    if not frame_list:
        raise ValueError("At least one prediction frame is required.")
    weight_array = weight_array / weight_array.sum()
    base_ids = frame_list[0]["eeg_id"].tolist()
    blended = np.zeros((len(base_ids), len(PROBABILITY_COLUMNS)), dtype=np.float64)
    for frame, weight in zip(frame_list, weight_array):
        if frame["eeg_id"].tolist() != base_ids:
            raise ValueError("Prediction files must have identical eeg_id ordering.")
        blended += frame[PROBABILITY_COLUMNS].to_numpy(dtype=np.float64) * weight
    blended /= np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)
    out = pd.DataFrame(blended, columns=PROBABILITY_COLUMNS)
    out.insert(0, "eeg_id", base_ids)
    return out

