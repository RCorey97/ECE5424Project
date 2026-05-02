from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import split_logits_for_calibration
from src.pipeline import calibrate_and_evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logits", default=str(ROOT / "results" / "baseline_logits.npz"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--tag", default="truthfulqa")
    p.add_argument("--val-frac", type=float, default=0.5)
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.logits)
    logits, labels = data["logits"], data["labels"]
    print(f"Loaded {len(labels)} predictions from {args.logits}")

    (val_logits, val_labels), (test_logits, test_labels) = split_logits_for_calibration(
        logits, labels, val_frac=args.val_frac, seed=args.seed,
    )
    print(f"Calibration split: {len(val_labels)} val / {len(test_labels)} test")

    report = calibrate_and_evaluate(
        val_logits, val_labels,
        test_logits, test_labels,
        out_dir=args.out_dir,
        n_bins=args.n_bins,
        tag=args.tag,
    )

    print("\n=== Calibration report ===")
    print(json.dumps(report, indent=2))
    print(f"\nArtefacts written to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
