from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import add_src_to_path

add_src_to_path()

from hms_solution.inference import logits_to_submission, write_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and write a Kaggle-style submission.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="reports/results/smoke_submission.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.smoke:
        raise NotImplementedError(
            "Full inference requires trained checkpoints under checkpoints/. Use --smoke to validate output format."
        )
    rng = np.random.default_rng(2024)
    eeg_ids = np.arange(100000, 100005)
    logits = rng.normal(size=(len(eeg_ids), 6))
    submission = logits_to_submission(eeg_ids, logits)
    write_submission(submission, Path(args.out))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

