from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for with as by "
    "from that this those these and or but not it its into over under do "
    "does did has have had s t am pm".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]


def token_f1(pred: str, ref: str) -> float:

    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not pred_toks or not ref_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(ref_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(ref_toks)
    return float(2 * precision * recall / (precision + recall))


def judge_correct(
    pred: str,
    correct_answers: Sequence[str],
    incorrect_answers: Sequence[str],
    margin: float = 0.0,
) -> Optional[int]:

    best_correct = max((token_f1(pred, c) for c in correct_answers), default=0.0)
    best_incorrect = max((token_f1(pred, c) for c in incorrect_answers), default=0.0)
    if best_correct == 0.0 and best_incorrect == 0.0:
        return None
    return int(best_correct > best_incorrect + margin)


@dataclass
class GenerationResult:
    text: str
    avg_token_prob: float  
    min_token_prob: float
    num_tokens: int
    token_log_probs: List[float]


def _strip_trailing_special(
    token_ids: Sequence[int],
    eos_token_ids: Sequence[int],
) -> List[int]:
    out: List[int] = []
    eos_set = set(int(t) for t in eos_token_ids)
    for tok in token_ids:
        if int(tok) in eos_set:
            break
        out.append(int(tok))
    return out


def generate_with_confidence(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 192,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> GenerationResult:

    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    eos_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None:
        eos_ids.append(int(tokenizer.pad_token_id))

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=top_p,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    seq = out.sequences[0, inputs.input_ids.shape[1]:].tolist()
    scores = out.scores  # tuple of length len(seq); each (batch, vocab)

    clean_ids = _strip_trailing_special(seq, eos_ids)
    log_probs: List[float] = []
    for step, tok_id in enumerate(clean_ids):
        step_logits = scores[step][0].float()
        step_log_probs = torch.log_softmax(step_logits, dim=-1)
        log_probs.append(float(step_log_probs[tok_id].item()))

    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
    if not log_probs:
        return GenerationResult(text, 0.0, 0.0, 0, [])

    avg_lp = float(np.mean(log_probs))
    min_lp = float(np.min(log_probs))
    return GenerationResult(
        text=text,
        avg_token_prob=float(np.exp(avg_lp)),
        min_token_prob=float(np.exp(min_lp)),
        num_tokens=len(log_probs),
        token_log_probs=log_probs,
    )


def build_chat_prompt(
    tokenizer,
    user_message: str,
    history: Optional[Sequence[tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
) -> str:

    if system_prompt is None:
        system_prompt = (
            "You are a careful, honest assistant. Answer the user's question "
            "concisely. If you are not sure, say so rather than inventing "
            "details."
        )

    history = list(history or [])
    messages = [{"role": "system", "content": system_prompt}]
    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": user_message})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    rendered = [f"System: {system_prompt}"]
    for user, assistant in history:
        rendered.append(f"User: {user}")
        rendered.append(f"Assistant: {assistant}")
    rendered.append(f"User: {user_message}")
    rendered.append("Assistant:")
    return "\n".join(rendered)
