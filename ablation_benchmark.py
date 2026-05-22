"""Run GameplayQA ablation benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from gameplayqa.ablation import ABLATION_TYPES, run_ablation
from gameplayqa.models import print_model_table
from gameplayqa.paths import QUESTION_FRAME_DIR, QUESTION_VIDEO_DIR, ablation_result_path, csv_for_split, timestamped_result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List ablation types")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument("--ablation", "-a", choices=ABLATION_TYPES, default="no_video")
    parser.add_argument("--model", "-m", default="gpt-5-mini")
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa")
    parser.add_argument("--csv", "-c", default=None)
    parser.add_argument("--video-dir", "-v", default=None)
    parser.add_argument("--frames-dir", "-f", default=None)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--timestamped", action="store_true", help="Append a timestamp to the default output filename")
    parser.add_argument("--workers", "-w", type=int, default=5)
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--resume", "-r", nargs="?", const="__default__", default=None)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if args.list:
        print("Available ablations: " + ", ".join(ABLATION_TYPES))
        return
    if args.list_models:
        print_model_table()
        return

    csv_path = Path(args.csv) if args.csv else csv_for_split(args.split)
    video_dir = Path(args.video_dir) if args.video_dir else QUESTION_VIDEO_DIR / args.split
    frames_dir = Path(args.frames_dir) if args.frames_dir else QUESTION_FRAME_DIR / args.split
    if args.resume:
        output_path = ablation_result_path(args.split, args.ablation, args.model) if args.resume == "__default__" else Path(args.resume)
        resume = True
    else:
        output_path = Path(args.output) if args.output else ablation_result_path(args.split, args.ablation, args.model)
        if args.timestamped and not args.output:
            output_path = timestamped_result_path(output_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
        resume = False
    run_ablation(args.ablation, args.model, csv_path, video_dir, frames_dir, output_path, args.split, args.workers, args.limit, resume, args.seed)


if __name__ == "__main__":
    main()
