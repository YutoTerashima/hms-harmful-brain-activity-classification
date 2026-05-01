from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from hms_solution.constants import COMPETITION_SLUG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Kaggle HMS competition data.")
    parser.add_argument("--competition", default=COMPETITION_SLUG)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--file", default=None, help="Optional single Kaggle file, e.g. train.csv.")
    parser.add_argument("--unzip", action="store_true", default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        print("Kaggle CLI is not installed. Install it with: python -m pip install kaggle")
        print("Then place kaggle.json under ~/.kaggle or set KAGGLE_USERNAME/KAGGLE_KEY.")
        print(f"Target directory prepared: {data_dir}")
        return 2

    command = [
        kaggle,
        "competitions",
        "download",
        "-c",
        args.competition,
        "-p",
        str(data_dir),
    ]
    if args.file:
        command.extend(["-f", args.file])
    if args.unzip:
        command.append("--unzip")
    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

