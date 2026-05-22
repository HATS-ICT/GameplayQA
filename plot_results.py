"""Generate plots from GameplayQA evaluation JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from gameplayqa.evaluate import plot_results
from gameplayqa.paths import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["qa", "generalization"], default="qa", help="Dataset split for default paths")
    parser.add_argument("--input", "-i", default=None, help="Evaluation JSON path")
    parser.add_argument("--output-dir", "-o", default=None, help="Plot output directory")
    args = parser.parse_args()
    input_path = Path(args.input) if args.input else RESULTS_DIR / "evaluation" / args.split / "evaluation_results.json"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / "plots" / args.split
    plot_results(input_path, output_dir)


if __name__ == "__main__":
    main()
