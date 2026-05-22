"""Evaluate judged GameplayQA result files."""

from __future__ import annotations

import argparse
from pathlib import Path

from gameplayqa.evaluate import run_evaluation
from gameplayqa.paths import RESULTS_DIR, csv_for_split, judged_result_path


def discover_judged_files(split: str, model: str | None) -> list[Path]:
    if model:
        path = judged_result_path(split, model)
        return [path] if path.exists() else []
    judged_dir = RESULTS_DIR / "judged" / split
    if not judged_dir.exists():
        return []
    return sorted(judged_dir.glob("*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="*", help="Judged JSONL file(s). If omitted, discovers results/judged/<split>/*.jsonl")
    parser.add_argument("--model", "-m", default=None, help="Evaluate the default judged result for one model")
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split metadata")
    parser.add_argument("--csv", action="append", default=None, help="Question CSV metadata path; may be repeated")
    parser.add_argument("--output-json", default=None, help="Evaluation JSON path")
    parser.add_argument("--output-csv", default=None, help="Evaluation CSV path")
    parser.add_argument("--output-md", default=None, help="Evaluation Markdown path")
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv] if args.csv else [csv_for_split(args.split)]
    result_paths = [Path(p) for p in args.results] if args.results else discover_judged_files(args.split, args.model)
    if not result_paths:
        print(f"No judged results found for split={args.split}. Run llm_as_a_judge.py first or pass result files explicitly.")
        return
    output_dir = RESULTS_DIR / "evaluation" / args.split
    run_evaluation(
        result_paths,
        csv_paths,
        Path(args.output_json) if args.output_json else output_dir / "evaluation_results.json",
        Path(args.output_csv) if args.output_csv else output_dir / "evaluation_results.csv",
        Path(args.output_md) if args.output_md else output_dir / "evaluation_results.md",
    )


if __name__ == "__main__":
    main()
