from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from hms_solution.config import load_config
from hms_solution.ensemble import blend_predictions, load_prediction
from hms_solution.inference import probabilities_to_submission, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend model submission files.")
    parser.add_argument("--config", default="configs/ensemble.yaml")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _smoke_frames():
    rng = np.random.default_rng(2024)
    eeg_ids = np.arange(100000, 100005)
    frames = []
    for _ in range(3):
        probs = rng.uniform(0.05, 1.0, size=(len(eeg_ids), 6))
        probs /= probs.sum(axis=1, keepdims=True)
        frames.append(probabilities_to_submission(eeg_ids, probs))
    return frames


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    entries = config.get("inputs", [])
    output_path = Path(config.get("output", {}).get("path", "reports/results/ensemble_submission.csv"))

    if args.smoke:
        frames = _smoke_frames()
        weights = [entry.get("weight", 1.0) for entry in entries] or [0.5, 0.3, 0.2]
        output_path = Path("reports/results/smoke_ensemble_submission.csv")
    else:
        frames = [load_prediction(entry["path"]) for entry in entries]
        weights = [entry.get("weight", 1.0) for entry in entries]
    blended = blend_predictions(frames, weights)
    write_submission(blended, output_path)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

