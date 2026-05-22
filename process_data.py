"""Prepare GameplayQA videos for benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

from gameplayqa.paths import DATA_DIR, QUESTION_FRAME_DIR, QUESTION_VIDEO_DIR, csv_for_split
from gameplayqa.video import crop_questions, extract_question_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split to process")
    parser.add_argument("--csv", type=str, default=None, help="Override question CSV path")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Downloaded dataset directory")
    parser.add_argument("--video-dir", type=str, default=None, help="Output directory for cropped question videos")
    parser.add_argument("--frames-dir", type=str, default=None, help="Output directory for extracted frame cache")
    parser.add_argument("--max-frames", type=int, default=32, help="Maximum frames per cropped video")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for frame extraction")
    parser.add_argument("--force", action="store_true", help="Rebuild both cropped videos and frame cache")
    parser.add_argument("--force-crop", action="store_true", help="Overwrite existing cropped videos")
    parser.add_argument("--force-frames", action="store_true", help="Re-extract existing frame folders")
    parser.add_argument("--skip-crop", action="store_true", help="Skip video cropping")
    parser.add_argument("--skip-frames", action="store_true", help="Skip frame extraction")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else csv_for_split(args.split)
    video_dir = Path(args.video_dir) if args.video_dir else QUESTION_VIDEO_DIR / args.split
    frames_dir = Path(args.frames_dir) if args.frames_dir else QUESTION_FRAME_DIR / args.split

    if not args.skip_crop:
        print("Cropping question videos")
        crop_counts = crop_questions(
            csv_path=csv_path,
            data_dir=Path(args.data_dir),
            output_dir=video_dir,
            force=args.force or args.force_crop,
            limit=args.limit,
        )
        print(f"Crop summary: {crop_counts}")

    if not args.skip_frames:
        print("Extracting frame cache")
        frame_counts = extract_question_frames(
            csv_path=csv_path,
            video_dir=video_dir,
            output_dir=frames_dir,
            max_frames=args.max_frames,
            workers=args.workers,
            force=args.force or args.force_frames,
            limit=args.limit,
        )
        print(f"Frame summary: {frame_counts}")


if __name__ == "__main__":
    main()
