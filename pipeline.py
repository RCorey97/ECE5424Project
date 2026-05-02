from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from .calibration import IsotonicCalibrator, TemperatureScaler, _softmax
from .data import MCQuestion
from .metrics import summary
from .model import Model
from .plots import confidence_histogram, reliability_diagram


def score_dataset(
    model: Model,
    questions: Sequence[MCQuestion],
    show_progress: bool = True,
    batch_size: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:


    questions = list(questions)
    labels = np.asarray([q.correct_index for q in questions], dtype=int)

    progress = None
    if show_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(total=len(questions), desc="scoring", unit="q")
        except ImportError:
            progress = None

    score_batch = getattr(model, "score_questions", None)
    chunks: List[np.ndarray] = []
    for start in range(0, len(questions), batch_size):
        chunk = questions[start : start + batch_size]
        if score_batch is not None:
            scores = np.asarray(score_batch(chunk))
        else:
            scores = np.stack([model.score_question(q) for q in chunk], axis=0)
        chunks.append(scores)
        if progress is not None:
            progress.update(len(chunk))
    if progress is not None:
        progress.close()

    return np.concatenate(chunks, axis=0), labels


def predictions_dataframe(
    questions: Sequence[MCQuestion],
    logits: np.ndarray,
) -> pd.DataFrame:

    probs = _softmax(logits, axis=-1)
    preds = np.argmax(probs, axis=1)
    confs = probs[np.arange(len(preds)), preds]
    rows = []
    for q, logit_row, prob_row, pred, conf in zip(questions, logits, probs, preds, confs):
        row = {
            "qid": q.qid,
            "category": q.category,
            "question": q.question,
            "correct_index": q.correct_index,
            "predicted_index": int(pred),
            "predicted_answer": q.options[int(pred)],
            "true_answer": q.options[q.correct_index],
            "raw_confidence": float(conf),
            "is_correct": int(pred == q.correct_index),
        }
        for i, (opt, lg, pr) in enumerate(zip(q.options, logit_row, prob_row)):
            row[f"option_{i}"] = opt
            row[f"logit_{i}"] = float(lg)
            row[f"prob_{i}"] = float(pr)
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_and_evaluate(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    out_dir: str | Path,
    n_bins: int = 10,
    tag: str = "model",
) -> dict:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_probs = _softmax(val_logits, axis=-1)
    test_probs = _softmax(test_logits, axis=-1)

    raw_val_preds = np.argmax(val_probs, axis=1)
    raw_val_conf = val_probs[np.arange(len(raw_val_preds)), raw_val_preds]
    raw_val_correct = (raw_val_preds == val_labels).astype(int)

    raw_test_preds = np.argmax(test_probs, axis=1)
    raw_test_conf = test_probs[np.arange(len(raw_test_preds)), raw_test_preds]
    raw_test_correct = (raw_test_preds == test_labels).astype(int)

    temp = TemperatureScaler().fit(val_logits, val_labels)
    ts_val_preds, ts_val_conf = temp.predicted_confidence(val_logits)
    ts_test_preds, ts_test_conf = temp.predicted_confidence(test_logits)
    ts_test_correct = (ts_test_preds == test_labels).astype(int)

    iso = IsotonicCalibrator().fit(ts_val_conf, (ts_val_preds == val_labels).astype(int))
    iso_test_conf = iso.transform(ts_test_conf)

    report = {
        "tag": tag,
        "temperature": temp.temperature,
        "raw": summary(raw_test_conf, raw_test_correct, n_bins=n_bins),
        "temperature_scaled": summary(ts_test_conf, ts_test_correct, n_bins=n_bins),
        "temperature_then_isotonic": summary(iso_test_conf, ts_test_correct, n_bins=n_bins),
    }

    reliability_diagram(
        raw_test_conf, raw_test_correct, out_dir / f"{tag}_reliability_raw.png",
        n_bins=n_bins, title=f"{tag} -- raw (ECE={report['raw']['ece']:.3f})",
    )
    reliability_diagram(
        ts_test_conf, ts_test_correct, out_dir / f"{tag}_reliability_temperature.png",
        n_bins=n_bins,
        title=f"{tag} -- temperature scaled "
              f"(T={temp.temperature:.2f}, ECE={report['temperature_scaled']['ece']:.3f})",
    )
    reliability_diagram(
        iso_test_conf, ts_test_correct, out_dir / f"{tag}_reliability_isotonic.png",
        n_bins=n_bins,
        title=f"{tag} -- temp + isotonic "
              f"(ECE={report['temperature_then_isotonic']['ece']:.3f})",
    )
    confidence_histogram(
        raw_test_conf, out_dir / f"{tag}_confidence_hist_raw.png",
        title=f"{tag} -- raw confidence",
    )
    confidence_histogram(
        iso_test_conf, out_dir / f"{tag}_confidence_hist_calibrated.png",
        title=f"{tag} -- calibrated confidence (temp + isotonic)",
    )

    with open(out_dir / f"{tag}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    eval_df = pd.DataFrame(
        {
            "raw_confidence": raw_test_conf,
            "raw_correct": raw_test_correct,
            "temperature_scaled_confidence": ts_test_conf,
            "calibrated_confidence": iso_test_conf,
            "temperature_scaled_correct": ts_test_correct,
            "predicted_index": ts_test_preds,
            "true_index": test_labels,
        }
    )
    eval_df.to_csv(out_dir / f"{tag}_test_evaluation.csv", index=False)

    return report
