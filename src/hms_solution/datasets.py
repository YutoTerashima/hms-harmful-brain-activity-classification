"""Dataset and metadata helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import HMS_CLASSES, TARGET_COLUMNS
from .metrics import normalize_rows


def load_train_metadata(path: str | Path) -> pd.DataFrame:
    """Load Kaggle train.csv metadata."""

    frame = pd.read_csv(path)
    missing = [column for column in TARGET_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing target vote columns: {missing}")
    return frame


def add_consensus_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized probability targets and hard consensus class columns."""

    out = frame.copy()
    votes = out[TARGET_COLUMNS].to_numpy(dtype=np.float64)
    probs = normalize_rows(votes)
    for index, class_name in enumerate(HMS_CLASSES):
        out[f"{class_name}_probability"] = probs[:, index]
    out["consensus_label"] = [HMS_CLASSES[index] for index in np.argmax(probs, axis=1)]
    out["vote_entropy"] = -np.sum(np.clip(probs, 1e-12, 1.0) * np.log(probs + 1e-12), axis=1)
    return out


def summarize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize soft votes and hard consensus counts."""

    enriched = add_consensus_columns(frame)
    rows = []
    total_rows = len(enriched)
    vote_totals = enriched[TARGET_COLUMNS].sum()
    consensus_counts = enriched["consensus_label"].value_counts()
    for class_name in HMS_CLASSES:
        vote_column = f"{class_name}_vote"
        rows.append(
            {
                "class_name": class_name,
                "total_votes": float(vote_totals[vote_column]),
                "vote_share": float(vote_totals[vote_column] / max(1.0, vote_totals.sum())),
                "consensus_count": int(consensus_counts.get(class_name, 0)),
                "consensus_share": float(consensus_counts.get(class_name, 0) / max(1, total_rows)),
            }
        )
    return pd.DataFrame(rows)


def make_synthetic_batch(
    model_family: str,
    batch_size: int = 4,
    num_classes: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a tiny deterministic batch for smoke tests."""

    rng = np.random.default_rng(2024)
    if model_family == "raw_eeg":
        x = rng.normal(size=(batch_size, 16, 1000)).astype("float32")
    else:
        x = rng.normal(size=(batch_size, 1, 128, 256)).astype("float32")
    raw_targets = rng.uniform(0.05, 1.0, size=(batch_size, num_classes)).astype("float32")
    y = raw_targets / raw_targets.sum(axis=1, keepdims=True)
    return x, y

