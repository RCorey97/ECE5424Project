from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


@dataclass
class MCQuestion:

    qid: int
    category: str
    question: str
    options: List[str]
    correct_index: int
    extra: dict = field(default_factory=dict)

    def letters(self) -> List[str]:
        return [chr(ord("A") + i) for i in range(len(self.options))]

    def formatted_prompt(self) -> str:
        lines = [f"Question: {self.question}"]
        for letter, option in zip(self.letters(), self.options):
            lines.append(f"{letter}. {option}")
        lines.append("Answer:")
        return "\n".join(lines)


def _split_semicolons(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [piece.strip() for piece in str(value).split(";") if piece.strip()]


def load_truthfulqa(
    csv_path: str | Path,
    num_choices: int = 4,
    seed: int = 0,
    limit: Optional[int] = None,
) -> List[MCQuestion]:

    df = pd.read_csv(csv_path)
    rng = random.Random(seed)
    questions: List[MCQuestion] = []

    for qid, row in df.iterrows():
        best_answer = str(row["Best Answer"]).strip()
        if not best_answer:
            continue

        incorrect_pool: List[str] = []
        best_incorrect = str(row.get("Best Incorrect Answer", "")).strip()
        if best_incorrect:
            incorrect_pool.append(best_incorrect)
        for distractor in _split_semicolons(row.get("Incorrect Answers")):
            if distractor and distractor not in incorrect_pool:
                incorrect_pool.append(distractor)

        if len(incorrect_pool) < num_choices - 1:
            continue

        chosen_distractors = incorrect_pool[: num_choices - 1]
        if len(incorrect_pool) > num_choices - 1:
            chosen_distractors = rng.sample(incorrect_pool, num_choices - 1)

        options = [best_answer, *chosen_distractors]
        order = list(range(num_choices))
        rng.shuffle(order)
        shuffled_options = [options[i] for i in order]
        correct_index = order.index(0)

        questions.append(
            MCQuestion(
                qid=int(qid),
                category=str(row.get("Category", "")),
                question=str(row["Question"]).strip(),
                options=shuffled_options,
                correct_index=correct_index,
                extra={"best_answer": best_answer},
            )
        )

        if limit is not None and len(questions) >= limit:
            break

    return questions


def train_val_test_split(
    items: Sequence[MCQuestion],
    val_frac: float = 0.2,
    test_frac: float = 0.4,
    seed: int = 0,
) -> tuple[List[MCQuestion], List[MCQuestion], List[MCQuestion]]:

    rng = random.Random(seed)
    indices = list(range(len(items)))
    rng.shuffle(indices)

    n = len(items)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    n_train = n - n_test - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return (
        [items[i] for i in train_idx],
        [items[i] for i in val_idx],
        [items[i] for i in test_idx],
    )
