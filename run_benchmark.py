"""Benchmark a multimodal model on GameplayQA."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from gameplayqa.benchmark import run_benchmark
from gameplayqa.judge import run_judge
from gameplayqa.models import InputType, MODEL_REGISTRY, print_model_table
from gameplayqa.paths import (
    QUESTION_FRAME_DIR,
    QUESTION_VIDEO_DIR,
    RESULTS_DIR,
    csv_for_split,
    judged_result_path,
    raw_result_path,
    timestamped_result_path,
)


def default_judged_output(input_path: Path, split: str, model: str, explicit_output: bool) -> Path:
    if explicit_output:
        return RESULTS_DIR / "judged" / split / f"{input_path.stem}_judged.jsonl"
    return judged_result_path(split, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", "-m", default="gemini-3-flash", help="Model key")
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split")
    parser.add_argument("--csv", "-c", default=None, help="Override question CSV")
    parser.add_argument("--video-dir", "-v", default=None, help="Directory with cropped videos")
    parser.add_argument("--frames-dir", "-f", default=None, help="Directory with extracted frames")
    parser.add_argument("--output", "-o", default=None, help="Output JSONL path")
    parser.add_argument("--timestamped", action="store_true", help="Append a timestamp to the default output filename")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Parallel benchmark workers")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit questions")
    parser.add_argument("--resume", "-r", nargs="?", const="__default__", default=None, help="Resume an output JSONL; omit the value to resume the default stable output")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic answer shuffle seed")
    parser.add_argument("--judge", action="store_true", help="Run LLM-as-a-judge after the benchmark finishes")
    parser.add_argument("--judge-model", "-j", default="gpt-5-mini", help="OpenAI model used for automatic judging")
    parser.add_argument("--judge-workers", type=int, default=5, help="Parallel workers for automatic judging")
    parser.add_argument("--judge-output", default=None, help="Judged JSONL output path for --judge")
    args = parser.parse_args()

    if args.list:
        print_model_table()
        return

    csv_path = Path(args.csv) if args.csv else csv_for_split(args.split)
    video_dir = Path(args.video_dir) if args.video_dir else QUESTION_VIDEO_DIR / args.split
    frames_dir = Path(args.frames_dir) if args.frames_dir else QUESTION_FRAME_DIR / args.split
    model_config = MODEL_REGISTRY.get(args.model)
    if model_config and model_config.input_type != InputType.VISION_FRAMES:
        frames_dir = None

    explicit_output = args.output is not None
    if args.resume:
        output_path = raw_result_path(args.split, args.model) if args.resume == "__default__" else Path(args.resume)
        resume = True
    else:
        output_path = Path(args.output) if args.output else raw_result_path(args.split, args.model)
        if args.timestamped and not args.output:
            output_path = timestamped_result_path(output_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        resume = False

    run_benchmark(args.model, csv_path, video_dir, output_path, args.split, frames_dir, args.workers, args.limit, resume, args.seed)

    if args.judge:
        if not output_path.exists():
            print(f"Skipping judge step because benchmark output was not created: {output_path}")
            return
        judge_output = Path(args.judge_output) if args.judge_output else default_judged_output(output_path, args.split, args.model, explicit_output or args.timestamped)
        run_judge(output_path, judge_output, args.judge_model, args.judge_workers)


if __name__ == "__main__":
    main()
