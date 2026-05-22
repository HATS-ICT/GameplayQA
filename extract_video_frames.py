"""Extract reusable frames from cropped GameplayQA question videos."""

from __future__ import annotations

import argparse
from pathlib import Path

from gameplayqa.paths import QUESTION_FRAME_DIR, QUESTION_VIDEO_DIR, csv_for_split
from gameplayqa.video import extract_question_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split")
    parser.add_argument("--csv", type=str, default=None, help="Override question CSV path")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory with cropped question videos")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for extracted frames")
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum frames per cropped video")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Re-extract existing frame folders")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N videos")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else csv_for_split(args.split)
    video_dir = Path(args.video_dir) if args.video_dir else QUESTION_VIDEO_DIR / args.split
    output_dir = Path(args.output_dir) if args.output_dir else QUESTION_FRAME_DIR / args.split
    counts = extract_question_frames(csv_path, video_dir, output_dir, args.max_frames, args.workers, args.force, args.limit)
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
