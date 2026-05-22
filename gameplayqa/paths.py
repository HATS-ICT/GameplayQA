"""Repository path defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
QUESTION_VIDEO_DIR = DATA_DIR / "question_videos"
QUESTION_FRAME_DIR = DATA_DIR / "question_video_frames"

SPLIT_TO_CSV = {
    "qa": DATA_DIR / "qa.csv",
    "generalization": DATA_DIR / "qa_generalization.csv",
}


def csv_for_split(split: str) -> Path:
    try:
        return SPLIT_TO_CSV[split]
    except KeyError as exc:
        valid = ", ".join(SPLIT_TO_CSV)
        raise ValueError(f"Unknown split {split!r}. Expected one of: {valid}") from exc


def raw_result_path(split: str, model: str) -> Path:
    return RESULTS_DIR / "raw" / split / f"{model}.jsonl"


def judged_result_path(split: str, model: str, suffix: str = "_judged") -> Path:
    return RESULTS_DIR / "judged" / split / f"{model}{suffix}.jsonl"


def ablation_result_path(split: str, ablation: str, model: str) -> Path:
    return RESULTS_DIR / "ablation" / split / f"{ablation}_{model}.jsonl"


def timestamped_result_path(base_path: Path, timestamp: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{timestamp}{base_path.suffix}")
