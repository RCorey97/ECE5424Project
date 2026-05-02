from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import accuracy_by_bin, expected_calibration_error


def reliability_diagram(
    confidences: np.ndarray,
    correct: np.ndarray,
    out_path: str | Path,
    n_bins: int = 10,
    title: Optional[str] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = accuracy_by_bin(confidences, correct, n_bins=n_bins)
    centers = np.array([(b.lower + b.upper) / 2 for b in bins])
    accuracies = np.array([b.accuracy if b.count > 0 else 0.0 for b in bins])
    counts = np.array([b.count for b in bins])

    width = 1.0 / n_bins
    ece = expected_calibration_error(confidences, correct, n_bins=n_bins)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax_top.bar(
        centers, accuracies, width=width * 0.9, edgecolor="black",
        color="#4C72B0", alpha=0.85, label="Accuracy",
    )
    ax_top.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax_top.set_ylim(0, 1)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylabel("Accuracy in bin")
    ax_top.set_title(title or f"Reliability diagram (ECE = {ece:.3f})")
    ax_top.legend(loc="upper left")

    ax_bot.bar(
        centers, counts, width=width * 0.9, edgecolor="black",
        color="#55A868", alpha=0.85,
    )
    ax_bot.set_xlabel("Predicted confidence")
    ax_bot.set_ylabel("Count")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def confidence_histogram(
    confidences: np.ndarray,
    out_path: str | Path,
    bins: int = 20,
    title: Optional[str] = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(confidences, bins=bins, range=(0.0, 1.0), edgecolor="black", color="#C44E52")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.set_title(title or "Confidence histogram")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
