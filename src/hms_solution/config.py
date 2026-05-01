"""Configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: dict[str, Any]
    data: dict[str, Any]
    train: dict[str, Any]
    output: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {config_path}")
    return data


def as_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load a config and expose the standard experiment sections."""

    data = load_config(path)
    return ExperimentConfig(
        name=str(data.get("name", Path(path).stem)),
        model=dict(data.get("model", {})),
        data=dict(data.get("data", {})),
        train=dict(data.get("train", {})),
        output=dict(data.get("output", {})),
    )

