"""Benchmark runner."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .data import cropped_video_paths, read_csv_rows
from .models import InputType, MODEL_REGISTRY, print_model_table
from .progress import console, make_progress
from .providers import get_processor
from .questions import build_answer_prompt, format_multiple_choice_from_row


@dataclass
class BenchmarkResult:
    question_id: str
    question_text: str
    correct_option: str
    correct_answer: str
    option_distractor_types: dict[str, str | None]
    shuffled_order: str
    model_output: str
    model_name: str
    model_id: str
    timestamp: str
    split: str
    error: str | None = None
    frames_used: int | None = None
    videos_used: int | None = None


def load_existing_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line)["question_id"])
                except Exception:
                    continue
    return ids


def process_question(row: dict[str, str], processor, video_dir: Path, model_name: str, model_id: str, split: str, seed: int) -> BenchmarkResult:
    timestamp = datetime.now().isoformat()
    qid = row.get("id", "")
    try:
        mc = format_multiple_choice_from_row(row, seed=seed)
        video_paths, video_indices = cropped_video_paths(row, video_dir)
        missing = [path for path in video_paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing cropped video(s): " + ", ".join(str(path) for path in missing))
        prompt = build_answer_prompt(mc.formatted_text, len(video_paths))
        response = processor.process(video_paths, prompt, video_indices=video_indices)
        output = (response.get("response") or "").strip()
        if not output:
            raise RuntimeError("Model returned an empty response")
        return BenchmarkResult(
            question_id=qid,
            question_text=mc.formatted_text,
            correct_option=mc.correct_option,
            correct_answer=mc.correct_answer,
            option_distractor_types=mc.option_distractor_types,
            shuffled_order=mc.shuffled_order,
            model_output=output,
            model_name=model_name,
            model_id=model_id,
            timestamp=timestamp,
            split=split,
            frames_used=response.get("frames_used"),
            videos_used=response.get("videos_used"),
        )
    except Exception as exc:
        return BenchmarkResult(
            question_id=qid,
            question_text=row.get("question", ""),
            correct_option="",
            correct_answer=row.get("correct_option", ""),
            option_distractor_types={},
            shuffled_order="",
            model_output="",
            model_name=model_name,
            model_id=model_id,
            timestamp=timestamp,
            split=split,
            error=str(exc),
        )


def run_benchmark(
    model_key: str,
    csv_path: Path,
    video_dir: Path,
    output_path: Path,
    split: str,
    frames_dir: Path | None = None,
    workers: int = 5,
    limit: int | None = None,
    resume: bool = False,
    seed: int = 13,
) -> None:
    if model_key not in MODEL_REGISTRY:
        print(f"Unknown model: {model_key}")
        print_model_table()
        return
    model_config = MODEL_REGISTRY[model_key]
    rows = read_csv_rows(csv_path)
    if limit:
        rows = rows[:limit]
    if resume:
        existing = load_existing_ids(output_path)
        rows = [row for row in rows if row.get("id") not in existing]
        console.print(f"Resume enabled: skipping {len(existing)} existing question ids")
    console.print(f"Benchmarking {len(rows)} questions from {csv_path}")
    console.print(f"Model: {model_config.name} ({model_config.api_model_id})")
    console.print(f"Video dir: {video_dir}")
    if model_config.input_type == InputType.VISION_FRAMES:
        console.print(f"Frames dir: {frames_dir}")
    console.print(f"Output: {output_path}")
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor = get_processor(model_config, frames_dir=frames_dir)
    file_lock = threading.Lock()
    mode = "a" if resume and output_path.exists() else "w"
    ok = 0
    errors = 0
    started = time.time()
    with output_path.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_question, row, processor, video_dir, model_config.name, model_config.api_model_id, split, seed)
                for row in rows
            ]
            with make_progress() as progress:
                task = progress.add_task("Benchmarking questions", total=len(rows))
                for future in as_completed(futures):
                    result = future.result()
                    if result.error:
                        errors += 1
                        console.print(f"[red]error[/red] {result.question_id}: {result.error}")
                    else:
                        ok += 1
                        with file_lock:
                            out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                            out.flush()
                    progress.advance(task)
    elapsed = time.time() - started
    console.print(f"Done. successful={ok} errors={errors} elapsed={elapsed:.1f}s output={output_path}")
