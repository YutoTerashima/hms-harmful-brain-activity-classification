"""Small reporting helpers used by metadata scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PUBLIC_DEVELOPMENT_CLASS_COUNTS = pd.DataFrame(
    [
        {"class_name": "seizure", "expert_segments": 20933},
        {"class_name": "lpd", "expert_segments": 14856},
        {"class_name": "gpd", "expert_segments": 16702},
        {"class_name": "lrda", "expert_segments": 16640},
        {"class_name": "grda", "expert_segments": 18861},
        {"class_name": "other", "expert_segments": 18808},
    ]
)


def write_public_context(out_dir: str | Path) -> Path:
    """Write a small public context table when Kaggle train.csv is unavailable."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = PUBLIC_DEVELOPMENT_CLASS_COUNTS.copy()
    frame["share"] = frame["expert_segments"] / frame["expert_segments"].sum()
    frame["source_note"] = (
        "Published public study summary for the HMS/Kaggle development cohort; "
        "not a replacement for local train.csv-derived statistics."
    )
    output_path = output_dir / "public_development_class_counts.csv"
    frame.to_csv(output_path, index=False)
    return output_path

