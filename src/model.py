from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Protocol, Sequence

import numpy as np

from .data import MCQuestion


class Model(Protocol):
    def score_question(self, question: MCQuestion) -> np.ndarray: ...
    def score_questions(self, questions: Sequence[MCQuestion]) -> np.ndarray: ...
    def generate_answer(self, question: MCQuestion, max_new_tokens: int = 64) -> str: ...

def _load_hf_token(explicit: Optional[str] = None) -> Optional[str]:


    if explicit:
        return explicit.strip() or None
    for env_key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(env_key)
        if value:
            return value.strip()

    here = Path(__file__).resolve().parents[1]
    for candidate in (here / "HF Token", here / "HF_TOKEN", here / ".hf_token"):
        if candidate.is_file():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    return line
    return None



class HFModel:
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        device: str | None = None,
        dtype: str = "auto",
        batch_size: int = 8,
        hf_token: str | None = None,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))

        if dtype == "auto":
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)
        self.dtype = torch_dtype

        token = _load_hf_token(hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            token=token,
        ).to(device)
        self.model.eval()

        self._letter_token_cache: dict[int, list[int]] = {}

    def _letter_token_ids(self, n: int) -> List[int]:

        if n in self._letter_token_cache:
            return self._letter_token_cache[n]

        ids: List[int] = []
        for i in range(n):
            letter = chr(ord("A") + i)
            for variant in (f" {letter}", letter):
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(token_ids) == 1:
                    ids.append(token_ids[0])
                    break
            else:  
                token_ids = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
                ids.append(token_ids[-1])
        self._letter_token_cache[n] = ids
        return ids

    def score_question(self, question: MCQuestion) -> np.ndarray:
        return self.score_questions([question])[0]

    def score_questions(self, questions: Sequence[MCQuestion]) -> np.ndarray:
        torch = self._torch
        questions = list(questions)
        if not questions:
            return np.zeros((0, 0), dtype=np.float32)

        n_options = len(questions[0].options)
        if any(len(q.options) != n_options for q in questions):
            return np.stack([self._score_single(q) for q in questions], axis=0)

        letter_ids = self._letter_token_ids(n_options)
        out = np.empty((len(questions), n_options), dtype=np.float32)

        for start in range(0, len(questions), self.batch_size):
            chunk = questions[start : start + self.batch_size]
            prompts = [q.formatted_prompt() for q in chunk]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False,
            ).to(self.device)
            with torch.inference_mode():
                outputs = self.model(**inputs)
            next_logits = outputs.logits[:, -1, :]
            picked = next_logits[:, letter_ids].detach().to(torch.float32).cpu().numpy()
            out[start : start + len(chunk)] = picked

        return out

    def _score_single(self, question: MCQuestion) -> np.ndarray:
        torch = self._torch
        prompt = question.formatted_prompt()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        next_logits = outputs.logits[0, -1, :]
        letter_ids = self._letter_token_ids(len(question.options))
        return next_logits[letter_ids].detach().to(torch.float32).cpu().numpy()


    def generate_answer(self, question: MCQuestion, max_new_tokens: int = 64) -> str:
        torch = self._torch
        prompt = question.formatted_prompt() + " "
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(
            output[0, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        first = text.strip().splitlines()[0] if text.strip() else ""
        return first
