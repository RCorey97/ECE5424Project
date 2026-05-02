from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import IsotonicCalibrator
from src.data import _split_semicolons  
from src.freeform import (
    build_chat_prompt,
    generate_with_confidence,
    judge_correct,
)
from src.metrics import summary
from src.plots import confidence_histogram, reliability_diagram


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(ROOT / "TruthfulQA.csv"))
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--hf-token", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--calibrator-out",
                   default=str(ROOT / "results" / "freeform_calibrator.joblib"))
    p.add_argument("--predictions-out",
                   default=str(ROOT / "results" / "freeform_predictions.csv"))
    p.add_argument("--plots-dir", default=str(ROOT / "results"))
    p.add_argument("--plots-tag", default="freeform")
    p.add_argument("--val-frac", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.model import _load_hf_token

    token = _load_hf_token(args.hf_token)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = (
        torch.bfloat16 if args.dtype == "auto" and device == "cuda"
        else torch.float32 if args.dtype == "auto"
        else getattr(torch, args.dtype)
    )

    print(f"Loading TruthfulQA from {args.csv}")
    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(args.limit)
    print(f"Using {len(df)} questions")

    print(f"Loading model {args.model} on {device} ({torch_dtype})")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=token
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch_dtype, trust_remote_code=True, token=token,
    ).to(device)
    model.eval()

    try:
        from tqdm.auto import tqdm

        iterator = tqdm(df.iterrows(), total=len(df), desc="generating")
    except ImportError:
        iterator = df.iterrows()

    rows: List[dict] = []
    for qid, row in iterator:
        question = str(row["Question"]).strip()
        correct_answers = _split_semicolons(row.get("Correct Answers"))
        incorrect_answers = _split_semicolons(row.get("Incorrect Answers"))
        best_answer = str(row.get("Best Answer", "")).strip()
        if best_answer and best_answer not in correct_answers:
            correct_answers.append(best_answer)
        best_incorrect = str(row.get("Best Incorrect Answer", "")).strip()
        if best_incorrect and best_incorrect not in incorrect_answers:
            incorrect_answers.append(best_incorrect)

        prompt = build_chat_prompt(tokenizer, question)
        result = generate_with_confidence(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
        )
        verdict = judge_correct(result.text, correct_answers, incorrect_answers)
        rows.append({
            "qid": int(qid),
            "category": str(row.get("Category", "")),
            "question": question,
            "answer": result.text,
            "avg_token_prob": result.avg_token_prob,
            "min_token_prob": result.min_token_prob,
            "num_tokens": result.num_tokens,
            "judged_correct": verdict,
        })

    pred_df = pd.DataFrame(rows)
    Path(args.predictions_out).parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.predictions_out, index=False)
    print(f"Saved {len(pred_df)} free-form predictions -> {args.predictions_out}")

    judged = pred_df.dropna(subset=["judged_correct"]).copy()
    print(f"Judge coverage: {len(judged)}/{len(pred_df)} "
          f"({100 * len(judged) / max(len(pred_df), 1):.1f}%)")

    if len(judged) < 40:
        raise SystemExit(
            "Too few judged samples to fit a calibrator -- try --limit larger "
            "(the judge is strict: it abstains when neither pool overlaps "
            "the answer)."
        )

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(judged))
    n_val = int(len(judged) * args.val_frac)
    val_idx, test_idx = order[:n_val], order[n_val:]

    raw_conf = judged["avg_token_prob"].to_numpy(dtype=float)
    correct = judged["judged_correct"].to_numpy(dtype=float)

    val_conf, val_correct = raw_conf[val_idx], correct[val_idx]
    test_conf, test_correct = raw_conf[test_idx], correct[test_idx]

    calibrator = IsotonicCalibrator().fit(val_conf, val_correct)
    cal_test_conf = calibrator.transform(test_conf)

    report = {
        "model": args.model,
        "n_generated": int(len(pred_df)),
        "n_judged": int(len(judged)),
        "raw": summary(test_conf, test_correct),
        "calibrated": summary(cal_test_conf, test_correct),
    }

    calibrator_path = Path(args.calibrator_out)
    calibrator.save(calibrator_path)
    report_path = calibrator_path.with_name(calibrator_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved calibrator -> {calibrator_path}")
    print(f"Saved report     -> {report_path}")

    plots_dir = Path(args.plots_dir)
    tag = args.plots_tag
    reliability_diagram(
        test_conf, test_correct, plots_dir / f"{tag}_reliability_raw.png",
        title=f"Free-form raw avg-token-prob (ECE={report['raw']['ece']:.3f})",
    )
    reliability_diagram(
        cal_test_conf, test_correct, plots_dir / f"{tag}_reliability_calibrated.png",
        title=f"Free-form calibrated (ECE={report['calibrated']['ece']:.3f})",
    )
    confidence_histogram(
        test_conf, plots_dir / f"{tag}_confidence_hist_raw.png",
        title="Free-form raw confidence (avg token prob)",
    )
    confidence_histogram(
        cal_test_conf, plots_dir / f"{tag}_confidence_hist_calibrated.png",
        title="Free-form calibrated confidence",
    )

    print("\n=== Free-form calibration report ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
