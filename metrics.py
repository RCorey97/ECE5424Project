from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BinStats:

    lower: float
    upper: float
    count: int
    accuracy: float
    confidence: float


def expected_calibration_error(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> float:

    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.size
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not np.any(mask):
            continue
        acc_bin = correct[mask].mean()
        conf_bin = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def brier_score(confidences: np.ndarray, correct: np.ndarray) -> float:

    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return 0.0
    return float(np.mean((confidences - correct) ** 2))


def overconfidence_gap(confidences: np.ndarray, correct: np.ndarray) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return 0.0
    return float(confidences.mean() - correct.mean())


def accuracy_by_bin(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> List[BinStats]:
    confidences = np.asarray(confidences, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    stats: List[BinStats] = []
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        count = int(mask.sum())
        if count == 0:
            stats.append(BinStats(lower, upper, 0, float("nan"), float("nan")))
        else:
            stats.append(
                BinStats(
                    lower=lower,
                    upper=upper,
                    count=count,
                    accuracy=float(correct[mask].mean()),
                    confidence=float(confidences[mask].mean()),
                )
            )
    return stats


def summary(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 10,
) -> dict:

    return {
        "n": int(np.asarray(confidences).size),
        "accuracy": float(np.mean(correct)) if len(correct) else 0.0,
        "avg_confidence": float(np.mean(confidences)) if len(confidences) else 0.0,
        "overconfidence_gap": overconfidence_gap(confidences, correct),
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "brier": brier_score(confidences, correct),
    }
