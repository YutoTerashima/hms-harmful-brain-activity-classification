"""PyTorch model definitions used by the reproducible scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    num_classes: int = 6
    in_channels: int = 1


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("PyTorch is required for model construction. Install the gpu extra.") from exc
    return torch, nn


def create_model(spec: ModelSpec):
    """Create a model from a small spec."""

    _, nn = require_torch()

    if spec.family == "raw_eeg":
        return ResNet1DGRU(in_channels=16, num_classes=spec.num_classes)
    if spec.name.startswith("efficientnet"):
        try:
            import timm

            model = timm.create_model(
                spec.name,
                pretrained=False,
                in_chans=spec.in_channels,
                num_classes=spec.num_classes,
            )
            return model
        except Exception:
            return SpectrogramCNN(in_channels=spec.in_channels, num_classes=spec.num_classes)
    if spec.family == "spectrogram":
        return SpectrogramCNN(in_channels=spec.in_channels, num_classes=spec.num_classes)
    raise ValueError(f"Unknown model family: {spec.family}")


class SpectrogramCNN(require_torch()[1].Module):
    """Compact CNN fallback for spectrogram smoke tests and CPU validation."""

    def __init__(self, in_channels: int = 1, num_classes: int = 6):
        torch, nn = require_torch()
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(64, num_classes)
        self._torch = torch

    def forward(self, x):
        features = self.features(x).flatten(1)
        return self.head(features)


class ResidualBlock1D(require_torch()[1].Module):
    """Small residual block for raw EEG smoke and baseline experiments."""

    def __init__(self, channels: int, kernel_size: int = 7):
        _, nn = require_torch()
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class ResNet1DGRU(require_torch()[1].Module):
    """Raw EEG model inspired by the archived ResNet1D-GRU notebook."""

    def __init__(self, in_channels: int = 16, hidden: int = 96, num_classes: int = 6):
        _, nn = require_torch()
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=15, padding=7),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.MaxPool1d(4),
        )
        self.residual = nn.Sequential(ResidualBlock1D(hidden), ResidualBlock1D(hidden))
        self.gru = nn.GRU(hidden, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, num_classes))

    def forward(self, x):
        features = self.residual(self.stem(x))
        sequence = features.transpose(1, 2)
        encoded, _ = self.gru(sequence)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)

