from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_truthfulqa
from src.pipeline import predictions_dataframe, score_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=str(ROOT / "TruthfulQA.csv"))
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--num-choices", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of questions for a quick run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mock", action="store_true",
                   help="Use the synthetic MockModel instead of HuggingFace.")
    p.add_argument("--device", default=None,
                   help='"cuda", "cpu" or "auto" (default: cuda if available).')
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"],
                   help='"auto" picks bfloat16 on CUDA, float32 on CPU.')
    p.add_argument("--batch-size", type=int, default=16,
                   help="Prompts per forward pass; raise on H100, lower on small GPUs.")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token (overrides env / HF Token file).")
    p.add_argument("--out", default=str(ROOT / "results" / "baseline_predictions.csv"))
    p.add_argument("--logits-out", default=str(ROOT / "results" / "baseline_logits.npz"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading TruthfulQA from {args.csv} ...")
    questions = load_truthfulqa(
        args.csv, num_choices=args.num_choices, seed=args.seed, limit=args.limit
    )
    print(f"Built {len(questions)} multiple-choice questions "
          f"({args.num_choices} options each)")

    if args.mock:
        from src.model import MockModel

        print("Using MockModel (synthetic miscalibrated logits).")
        model = MockModel(seed=args.seed)
    else:
        from src.model import HFModel

        print(f"Loading HuggingFace model {args.model} (this may take a while)...")
        model = HFModel(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            hf_token=args.hf_token,
        )
        print(f"Model loaded on device={model.device}, dtype={model.dtype}, "
              f"batch_size={model.batch_size}")

    logits, labels = score_dataset(model, questions, batch_size=args.batch_size)
    df = predictions_dataframe(questions, logits)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    logits_out = Path(args.logits_out)
    np.savez(logits_out, logits=logits, labels=labels)

    accuracy = float((df["predicted_index"] == df["correct_index"]).mean())
    avg_conf = float(df["raw_confidence"].mean())
    print(f"\nBaseline accuracy:    {accuracy:.3f}")
    print(f"Average confidence:   {avg_conf:.3f}")
    print(f"Overconfidence gap:   {avg_conf - accuracy:+.3f}")
    print(f"Saved per-question predictions to {out_path}")
    print(f"Saved raw logits to            {logits_out}")


if __name__ == "__main__":
    main()
