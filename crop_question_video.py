"""Crop GameplayQA question videos from the downloaded dataset."""

from __future__ import annotations

import argparse

from gameplayqa.paths import DATA_DIR, QUESTION_VIDEO_DIR, csv_for_split
from gameplayqa.video import crop_questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split to crop")
    parser.add_argument("--csv", type=str, default=None, help="Override question CSV path")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Downloaded dataset directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for cropped videos")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cropped videos")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N questions")
    args = parser.parse_args()

    from pathlib import Path

    csv_path = Path(args.csv) if args.csv else csv_for_split(args.split)
    output_dir = Path(args.output_dir) if args.output_dir else QUESTION_VIDEO_DIR / args.split
    counts = crop_questions(csv_path, Path(args.data_dir), output_dir, force=args.force, limit=args.limit)
    print(f"Summary: {counts}")


if __name__ == "__main__":
    main()
