from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from hms_solution.config import as_experiment_config
from hms_solution.training import run_synthetic_training, write_json_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Model 3: official spectrogram EfficientNet model.")
    parser.add_argument("--config", default="configs/model3_effnet_official_spec.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = as_experiment_config(args.config)
    if not args.smoke:
        train_csv = Path(config.data.get("train_csv", "data/raw/train.csv"))
        if not train_csv.exists():
            raise FileNotFoundError(
                f"{train_csv} not found. Run scripts/download_data.py first or use --smoke."
            )
    result = run_synthetic_training(
        model_name=config.model.get("name", "efficientnet_b0"),
        family=config.model.get("family", "spectrogram"),
        device=args.device,
        steps=2 if args.smoke else 10,
        lr=float(config.train.get("learning_rate", 2e-4)),
    )
    output_path = Path("reports/results/model3_smoke_metrics.json" if args.smoke else "reports/results/model3_run_metrics.json")
    write_json_report(result, output_path)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

