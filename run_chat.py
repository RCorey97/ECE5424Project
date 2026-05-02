from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chat import HumilityChatbot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--calibrator",
                   default=str(ROOT / "results" / "freeform_calibrator.joblib"))
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default="auto",
                   choices=["auto", "bfloat16", "float16", "float32"])
    p.add_argument("--hf-token", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--share", action="store_true",
                   help="Expose a public *.gradio.live URL (needed on Colab).")
    p.add_argument("--server-name", default="0.0.0.0")
    p.add_argument("--server-port", type=int, default=7860)
    p.add_argument("--cli", action="store_true",
                   help="Skip Gradio and use a simple REPL loop instead.")
    return p.parse_args()


def format_reply(resp) -> str:

    return (
        f"{resp.phrased_answer}\n\n"
        f"---\n"
        f"**Confidence meter:** `{resp.meter_bar}`  **Level:** {resp.level}\n\n"
        f"*raw signal {resp.raw_confidence*100:.1f}%  ->  "
        f"calibrated {resp.calibrated_confidence*100:.1f}%,  "
        f"generated {resp.num_tokens} tokens*"
    )


def run_cli(bot: HumilityChatbot) -> None:
    print("\nHumility chatbot ready. Type a question ('exit' to quit).\n")
    history = []
    while True:
        try:
            msg = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"exit", "quit"}:
            break
        if not msg:
            continue
        resp = bot.chat(msg, history=history)
        print(f"Assistant > {resp.pretty()}\n")
        history.append((msg, resp.answer))


def run_gradio(bot: HumilityChatbot, args) -> None:
    import gradio as gr

    def respond(user_message: str, history):
        tuples = []
        if history:
            pending_user = None
            for item in history:
                if isinstance(item, dict):
                    if item["role"] == "user":
                        pending_user = item["content"]
                    elif item["role"] == "assistant" and pending_user is not None:
                        tuples.append((pending_user, item["content"]))
                        pending_user = None
                else:
                    tuples.append(tuple(item))
        resp = bot.chat(user_message, history=tuples)
        return format_reply(resp)

    with gr.Blocks(title="Humility-Enhanced LLM") as demo:
        gr.Markdown(
            "# Humility-Enhanced LLM\n\n"
            "Ask a question. The model will answer, display a confidence "
            "meter calibrated on TruthfulQA, and hedge its language when it "
            "is not sure."
        )
        gr.ChatInterface(
            fn=respond,
            type="messages",
            examples=[
                "What happens if you eat watermelon seeds?",
                "Who wrote the play Hamlet?",
                "How many planets are in the solar system?",
                "What's the capital of Australia?",
            ],
        )

    demo.queue().launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


def main() -> None:
    args = parse_args()
    print(f"Loading chatbot (model={args.model})...")
    bot = HumilityChatbot(
        model_name=args.model,
        calibrator_path=args.calibrator,
        device=args.device,
        dtype=args.dtype,
        hf_token=args.hf_token,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(
        f"Model loaded on {bot.device}. "
        f"Calibrator: {'loaded' if bot.calibrator is not None else 'NONE (using raw)'}."
    )

    if args.cli:
        run_cli(bot)
    else:
        run_gradio(bot, args)


if __name__ == "__main__":
    main()
