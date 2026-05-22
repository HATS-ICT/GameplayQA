"""Judge GameplayQA benchmark results with an LLM."""

from __future__ import annotations

import argparse
from pathlib import Path

from gameplayqa.judge import run_judge
from gameplayqa.paths import RESULTS_DIR, ablation_result_path, judged_result_path, raw_result_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=None, help="Benchmark JSONL file. If omitted, uses --split/--model")
    parser.add_argument("--model", "-m", default="gemini-3-flash", help="Model key used to locate the default raw result")
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split used to locate default paths")
    parser.add_argument("--ablation", choices=["no_video", "random_single_frame", "shuffled_frames"], default=None, help="Judge an ablation result instead of a normal benchmark result")
    parser.add_argument("--output", "-o", default=None, help="Judged JSONL output path")
    parser.add_argument("--judge-model", "-j", default="gpt-5-mini", help="OpenAI model used as judge")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Parallel workers")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit results")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
    elif args.ablation:
        input_path = ablation_result_path(args.split, args.ablation, args.model)
    else:
        input_path = raw_result_path(args.split, args.model)

    if args.output:
        output_path = Path(args.output)
    elif args.input:
        output_path = RESULTS_DIR / "judged" / args.split / f"{input_path.stem}_judged.jsonl"
    elif args.ablation:
        output_path = judged_result_path(args.split, f"{args.ablation}_{args.model}")
    else:
        output_path = judged_result_path(args.split, args.model)

    run_judge(input_path, output_path, args.judge_model, args.workers, args.limit)


if __name__ == "__main__":
    main()
