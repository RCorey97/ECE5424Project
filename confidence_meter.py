from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


LEVELS: List[Tuple[str, float, float]] = [
    ("Very Low", 0.00, 0.20),
    ("Low", 0.20, 0.40),
    ("Medium", 0.40, 0.60),
    ("High", 0.60, 0.80),
    ("Very High", 0.80, 1.0001),
]


@dataclass
class MeterReading:
    confidence: float
    level: str
    bar: str
    phrased_answer: str


def confidence_to_level(confidence: float) -> str:
    confidence = max(0.0, min(1.0, float(confidence)))
    for label, lower, upper in LEVELS:
        if lower <= confidence < upper:
            return label
    return LEVELS[-1][0]


def confidence_bar(confidence: float, width: int = 20) -> str:

    confidence = max(0.0, min(1.0, float(confidence)))
    filled = int(round(confidence * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {confidence * 100:5.1f}%"


def phrase_answer(answer: str, confidence: float) -> str:

    level = confidence_to_level(confidence)
    answer = answer.strip()
    if level in ("High", "Very High"):
        return answer
    if level == "Medium":
        return f"{answer} (I'm only moderately sure -- you may want to double-check.)"
    return (
        f"I'm not very confident, but my best guess is: {answer}. "
        "Please verify this against a trusted source before relying on it."
    )


def make_reading(answer: str, confidence: float, bar_width: int = 20) -> MeterReading:
    return MeterReading(
        confidence=float(confidence),
        level=confidence_to_level(confidence),
        bar=confidence_bar(confidence, width=bar_width),
        phrased_answer=phrase_answer(answer, confidence),
    )
