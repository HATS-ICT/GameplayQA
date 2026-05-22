"""Multiple-choice question formatting."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class MultipleChoiceQuestion:
    formatted_text: str
    correct_option: str
    correct_answer: str
    option_distractor_types: dict[str, str | None]
    shuffled_order: str


def _rng_for_row(row: dict[str, str], seed: int) -> random.Random:
    key = f"{seed}:{row.get('id', '')}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))


def format_multiple_choice_from_row(row: dict[str, str], seed: int = 13) -> MultipleChoiceQuestion:
    question = row.get("question", "").strip()
    correct = row.get("correct_option", "").strip()
    question_code = row.get("question_code", "")
    rng = _rng_for_row(row, seed)

    if "EXIST" in question_code:
        correct_answer = "True" if correct.lower() == "true" else "False"
        wrong_answer = "False" if correct_answer == "True" else "True"
        options = [(correct_answer, None, "C"), (wrong_answer, "exist_negation", "D1")]
    else:
        options = [(correct, None, "C")]
        for i in range(1, 4):
            distractor = row.get(f"distractor_{i}", "").strip()
            if distractor:
                dtype = row.get(f"distractor_{i}_type", "").strip() or "unknown"
                options.append((distractor, dtype, f"D{i}"))

    rng.shuffle(options)
    lines = [f"Q: {question}", ""]
    correct_letter = "A"
    option_distractor_types: dict[str, str | None] = {}
    order_labels: list[str] = []

    for idx, (text, dtype, label) in enumerate(options):
        letter = chr(ord("A") + idx)
        lines.append(f"{letter}. {text}")
        option_distractor_types[letter] = dtype
        order_labels.append(label)
        if label == "C":
            correct_letter = letter

    return MultipleChoiceQuestion(
        formatted_text="\n".join(lines),
        correct_option=correct_letter,
        correct_answer=correct,
        option_distractor_types=option_distractor_types,
        shuffled_order="-".join(order_labels),
    )


def build_answer_prompt(formatted_question: str, num_videos: int) -> str:
    noun = "video" if num_videos == 1 else "videos"
    verb = "Watch the video" if num_videos == 1 else "Watch all videos"
    return (
        f"{verb} carefully and answer the following multiple choice question.\n"
        f"Answer with ONLY the letter of your choice. Do not include any explanation.\n\n"
        f"{formatted_question}\n\n"
        f"Your answer:"
    ).replace("all videos", f"all the {noun}")


def build_text_only_prompt(formatted_question: str) -> str:
    return (
        "Answer the following multiple choice question.\n"
        "Answer with ONLY the letter of your choice. Do not include any explanation.\n\n"
        f"{formatted_question}\n\n"
        "Your answer:"
    )
