"""EEG preprocessing and spectrogram generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import BIPOLAR_PAIRS


@dataclass(frozen=True)
class SpectrogramConfig:
    sample_rate: int = 200
    n_fft: int = 256
    hop_length: int = 64
    eps: float = 1e-6


def clean_eeg_frame(frame: pd.DataFrame, channels: Iterable[str]) -> pd.DataFrame:
    """Interpolate and fill EEG channel data without changing channel order."""

    selected = frame.loc[:, list(channels)].astype("float32")
    selected = selected.interpolate(limit_direction="both")
    selected = selected.fillna(0.0)
    return selected


def bipolar_montage(frame: pd.DataFrame, pairs: list[tuple[str, str]] | None = None) -> np.ndarray:
    """Create a double-banana-style bipolar montage matrix from an EEG frame."""

    pairs = pairs or BIPOLAR_PAIRS
    missing = sorted({channel for pair in pairs for channel in pair if channel not in frame.columns})
    if missing:
        raise ValueError(f"Missing EEG channels: {missing}")
    clean = clean_eeg_frame(frame, sorted({channel for pair in pairs for channel in pair}))
    signals = [clean[left].to_numpy() - clean[right].to_numpy() for left, right in pairs]
    return np.stack(signals).astype("float32")


def log_spectrogram(signal: np.ndarray, config: SpectrogramConfig | None = None) -> np.ndarray:
    """Convert a 1D signal into a normalized log spectrogram."""

    cfg = config or SpectrogramConfig()
    try:
        from scipy.signal import spectrogram

        _, _, spec = spectrogram(
            signal,
            fs=cfg.sample_rate,
            nperseg=cfg.n_fft,
            noverlap=cfg.n_fft - cfg.hop_length,
            mode="magnitude",
        )
    except Exception:
        windows = []
        for start in range(0, max(1, len(signal) - cfg.n_fft), cfg.hop_length):
            chunk = signal[start : start + cfg.n_fft]
            if len(chunk) < cfg.n_fft:
                chunk = np.pad(chunk, (0, cfg.n_fft - len(chunk)))
            windows.append(np.abs(np.fft.rfft(chunk)))
        spec = np.stack(windows, axis=1) if windows else np.zeros((cfg.n_fft // 2 + 1, 1))
    log_spec = np.log1p(np.asarray(spec, dtype=np.float32) + cfg.eps)
    log_spec -= log_spec.mean()
    log_spec /= log_spec.std() + cfg.eps
    return log_spec.astype("float32")


def eeg_to_multichannel_spectrogram(
    frame: pd.DataFrame,
    config: SpectrogramConfig | None = None,
    pairs: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Create one spectrogram per bipolar EEG channel pair."""

    montage = bipolar_montage(frame, pairs=pairs)
    specs = [log_spectrogram(channel_signal, config=config) for channel_signal in montage]
    min_time = min(spec.shape[1] for spec in specs)
    return np.stack([spec[:, :min_time] for spec in specs]).astype("float32")

