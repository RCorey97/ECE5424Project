from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _nll_from_logits(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    scaled = logits / max(T, 1e-6)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    return float(-log_probs[np.arange(len(labels)), labels].mean())


@dataclass
class TemperatureScaler:

    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if logits.ndim != 2:
            raise ValueError("logits must be 2-D (n_samples, n_classes)")

        result = minimize_scalar(
            lambda T: _nll_from_logits(logits, labels, T),
            bounds=(0.05, 20.0),
            method="bounded",
            options={"xatol": 1e-4},
        )
        self.temperature = float(result.x)
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        return logits / max(self.temperature, 1e-6)

    def transform(self, logits: np.ndarray) -> np.ndarray:

        return _softmax(self.transform_logits(logits), axis=-1)

    def predicted_confidence(self, logits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.transform(logits)
        preds = np.argmax(probs, axis=1)
        return preds, probs[np.arange(len(preds)), preds]


@dataclass
class IsotonicCalibrator:

    iso: IsotonicRegression | None = None

    def fit(self, confidences: np.ndarray, correct: np.ndarray) -> "IsotonicCalibrator":
        confidences = np.asarray(confidences, dtype=float)
        correct = np.asarray(correct, dtype=float)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.iso.fit(confidences, correct)
        return self

    def transform(self, confidences: np.ndarray) -> np.ndarray:
        if self.iso is None:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        return np.asarray(self.iso.transform(np.asarray(confidences, dtype=float)))


    def save(self, path: str | Path) -> Path:
        if self.iso is None:
            raise RuntimeError("Cannot save an unfit IsotonicCalibrator")
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"type": "IsotonicCalibrator", "iso": self.iso}, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "IsotonicCalibrator":
        import joblib

        payload = joblib.load(Path(path))
        if payload.get("type") != "IsotonicCalibrator":
            raise ValueError(f"{path} is not an IsotonicCalibrator dump")
        return cls(iso=payload["iso"])


def split_logits_for_calibration(
    logits: np.ndarray,
    labels: np.ndarray,
    val_frac: float = 0.5,
    seed: int = 0,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    n_val = int(len(idx) * val_frac)
    val_idx, eval_idx = idx[:n_val], idx[n_val:]
    return (
        (logits[val_idx], labels[val_idx]),
        (logits[eval_idx], labels[eval_idx]),
    )
