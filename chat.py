from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .calibration import IsotonicCalibrator
from .confidence_meter import MeterReading, make_reading
from .freeform import GenerationResult, build_chat_prompt, generate_with_confidence
from .model import _load_hf_token


@dataclass
class ChatResponse:
    answer: str
    raw_confidence: float
    calibrated_confidence: float
    level: str
    phrased_answer: str
    meter_bar: str
    num_tokens: int

    def pretty(self) -> str:
        return (
            f"{self.phrased_answer}\\n"
            f"  confidence : {self.meter_bar}  -> {self.level}\\n"
            f"  (raw signal {self.raw_confidence*100:.1f}% "
            f"-> calibrated {self.calibrated_confidence*100:.1f}%)"
        )


class HumilityChatbot:

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        calibrator_path: str | Path | None = None,
        device: Optional[str] = None,
        dtype: str = "auto",
        hf_token: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.system_prompt = system_prompt

        if dtype == "auto":
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        token = _load_hf_token(hf_token)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, token=token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            token=token,
        ).to(device)
        self.model.eval()

        self.calibrator: Optional[IsotonicCalibrator] = None
        if calibrator_path is not None:
            path = Path(calibrator_path)
            if path.is_file():
                self.calibrator = IsotonicCalibrator.load(path)
            else:
                print(
                    f"[HumilityChatbot] Calibrator not found at {path}; "
                    f"falling back to raw average-token-probability."
                )

    def _calibrate(self, raw: float) -> float:
        if self.calibrator is None:
            return float(raw)
        return float(self.calibrator.transform([raw])[0])

    def generate(
        self,
        user_message: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> Tuple[GenerationResult, str]:
        prompt = build_chat_prompt(
            self.tokenizer, user_message, history=history,
            system_prompt=self.system_prompt,
        )
        result = generate_with_confidence(
            self.model, self.tokenizer, prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return result, prompt

    def chat(
        self,
        user_message: str,
        history: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> ChatResponse:
        result, _ = self.generate(user_message, history=history)
        text = "\\n".join(
            line for line in result.text.splitlines()
            if "confidence meter" not in line.strip().lower()
        ).strip()
        raw_conf = result.avg_token_prob
        calibrated = self._calibrate(raw_conf)
        reading: MeterReading = make_reading(text or "(no answer)", calibrated)
        return ChatResponse(
            answer=text,
            raw_confidence=float(raw_conf),
            calibrated_confidence=float(calibrated),
            level=reading.level,
            phrased_answer=reading.phrased_answer,
            meter_bar=reading.bar,
            num_tokens=int(result.num_tokens),
        )