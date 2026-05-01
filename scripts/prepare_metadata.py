from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from hms_solution.datasets import add_consensus_columns, load_train_metadata, summarize_labels
from hms_solution.reporting import write_public_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare small metadata artifacts from Kaggle train.csv.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--max-manifest-rows", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = data_dir / "train.csv"

    if not train_csv.exists():
        public_path = write_public_context(out_dir)
        print(f"train.csv not found. Wrote public context fallback: {public_path}")
        return 0

    metadata = load_train_metadata(train_csv)
    enriched = add_consensus_columns(metadata)
    summary = summarize_labels(metadata)
    summary.to_csv(out_dir / "label_distribution.csv", index=False)

    manifest_columns = [
        column
        for column in [
            "eeg_id",
            "spectrogram_id",
            "patient_id",
            "consensus_label",
            "vote_entropy",
        ]
        if column in enriched.columns
    ]
    enriched.loc[:, manifest_columns].head(args.max_manifest_rows).to_csv(
        out_dir / "sample_manifest.csv",
        index=False,
    )
    print(f"Wrote {out_dir / 'label_distribution.csv'}")
    print(f"Wrote {out_dir / 'sample_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

