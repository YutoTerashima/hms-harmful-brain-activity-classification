"""Metrics for probability-distribution labels."""

from __future__ import annotations

import numpy as np


def normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize row vectors into probability distributions."""

    array = np.asarray(values, dtype=np.float64)
    totals = array.sum(axis=1, keepdims=True)
    totals = np.maximum(totals, eps)
    return array / totals


def kl_divergence(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Row-wise KL divergence between target and predicted distributions."""

    target_prob = np.clip(normalize_rows(target), eps, 1.0)
    pred_prob = np.clip(normalize_rows(prediction), eps, 1.0)
    return np.sum(target_prob * np.log(target_prob / pred_prob), axis=1)


def mean_kl_divergence(target: np.ndarray, prediction: np.ndarray, eps: float = 1e-7) -> float:
    """Mean row-wise KL divergence."""

    return float(np.mean(kl_divergence(target, prediction, eps=eps)))


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable NumPy softmax for inference utilities."""

    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)

