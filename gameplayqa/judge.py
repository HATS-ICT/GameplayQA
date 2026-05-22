"""LLM-as-a-judge for benchmark JSONL output."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .progress import console, make_progress
from .providers import load_dotenv_if_available


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_judged(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {row["question_id"]: row for row in load_jsonl(path)}


def extract_selected_option(result: dict, client: OpenAI, judge_model: str) -> tuple[str, str | None]:
    options = sorted((result.get("option_distractor_types") or {"A": None, "B": None, "C": None, "D": None}).keys())
    prompt = (
        "Extract which option letter the model selected for this multiple choice question.\n"
        f"Available option letters: {', '.join(options)}\n\n"
        f"Question:\n{result.get('question_text', '')}\n\n"
        f"Model response:\n{result.get('model_output', '')}\n\n"
        'Return JSON exactly like {"selected_option":"A"}; use "X" if unclear.'
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": "You extract one selected multiple choice letter. Return compact JSON only."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        selected = str(json.loads(raw).get("selected_option", "X")).upper().strip()
    except json.JSONDecodeError:
        selected = "X"
    if selected not in options and selected != "X":
        selected = next((opt for opt in options if opt in selected), "X")
    return selected, None


def judge_one(result: dict, client: OpenAI, judge_model: str) -> dict:
    judged = dict(result)
    judged["judge_model"] = judge_model
    judged["judge_timestamp"] = datetime.now().isoformat()
    if result.get("error"):
        judged.update({"selected_option": "X", "is_correct": False, "selected_distractor_type": None, "judge_error": "benchmark error"})
        return judged
    try:
        selected, judge_error = extract_selected_option(result, client, judge_model)
        correct = selected == result.get("correct_option")
        distractor_type = None
        if not correct and selected != "X":
            distractor_type = (result.get("option_distractor_types") or {}).get(selected)
        judged.update(
            {
                "selected_option": selected,
                "is_correct": correct,
                "selected_distractor_type": distractor_type,
                "judge_error": judge_error,
            }
        )
    except Exception as exc:
        judged.update({"selected_option": "X", "is_correct": False, "selected_distractor_type": None, "judge_error": str(exc)})
    return judged


def run_judge(input_path: Path, output_path: Path, judge_model: str = "gpt-5-mini", workers: int = 5, limit: int | None = None) -> None:
    load_dotenv_if_available()
    client = OpenAI()
    rows = load_jsonl(input_path)
    if limit:
        rows = rows[:limit]
    existing = existing_judged(output_path)
    todo = [row for row in rows if row["question_id"] not in existing]
    console.print(f"Loaded {len(rows)} results, {len(existing)} already judged, judging {len(todo)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    judged_rows = list(existing.values())
    lock = threading.Lock()
    correct = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(judge_one, row, client, judge_model) for row in todo]
        with make_progress() as progress:
            task = progress.add_task("Judging responses", total=len(todo))
            for future in as_completed(futures):
                judged = future.result()
                with lock:
                    judged_rows.append(judged)
                if judged.get("is_correct"):
                    correct += 1
                if judged.get("judge_error"):
                    console.print(f"[red]judge error[/red] {judged['question_id']}: {judged.get('judge_error')}")
                progress.advance(task)
    judged_rows.sort(key=lambda row: row["question_id"])
    with output_path.open("w", encoding="utf-8") as f:
        for row in judged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    total_correct = sum(1 for row in judged_rows if row.get("is_correct"))
    accuracy = total_correct / len(judged_rows) * 100 if judged_rows else 0
    console.print(f"Done. total={len(judged_rows)} accuracy={accuracy:.2f}% output={output_path}")
