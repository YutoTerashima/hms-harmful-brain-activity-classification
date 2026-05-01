"""Training and smoke-run helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .datasets import make_synthetic_batch
from .models import ModelSpec, create_model, require_torch


def seed_everything(seed: int = 2024) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_loss(logits, targets):
    torch, _ = require_torch()
    log_probs = torch.log_softmax(logits, dim=1)
    return torch.nn.functional.kl_div(log_probs, targets, reduction="batchmean")


def run_synthetic_training(
    model_name: str,
    family: str,
    device: str = "cpu",
    steps: int = 2,
    lr: float = 1e-3,
) -> dict[str, Any]:
    """Run a deterministic synthetic train loop to validate the model path."""

    torch, _ = require_torch()
    seed_everything(2024)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    model = create_model(ModelSpec(name=model_name, family=family)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    x_np, y_np = make_synthetic_batch("raw_eeg" if family == "raw_eeg" else "spectrogram")
    x = torch.tensor(x_np, device=device)
    y = torch.tensor(y_np, device=device)

    losses = []
    start = time.perf_counter()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = kl_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_name": model_name,
        "family": family,
        "device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "steps": steps,
        "final_loss": losses[-1],
        "losses": losses,
        "parameter_count": int(parameter_count),
        "elapsed_seconds": round(elapsed, 4),
    }


def write_json_report(result: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

