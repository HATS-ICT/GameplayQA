"""Dataset row parsing helpers."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

VIDEO_COLUMNS = [f"video_{i}_file_name" for i in range(1, 6)]


@dataclass(frozen=True)
class VideoRef:
    relative_path: str
    index: int


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_video_indices(value: str, count: int) -> list[int]:
    raw = [part.strip() for part in (value or "").split(",") if part.strip()]
    indices: list[int] = []
    for part in raw:
        try:
            indices.append(int(part))
        except ValueError:
            continue
    if len(indices) != count:
        return list(range(count))
    return indices


def video_refs_from_row(row: dict[str, str]) -> list[VideoRef]:
    paths = [row.get(col, "").strip() for col in VIDEO_COLUMNS if row.get(col, "").strip()]
    indices = parse_video_indices(row.get("video_indices", ""), len(paths))
    return [VideoRef(relative_path=path, index=idx) for path, idx in zip(paths, indices)]


def question_video_stems(row: dict[str, str]) -> list[str]:
    refs = video_refs_from_row(row)
    qid = row.get("id", "")
    if len(refs) <= 1:
        return [qid]
    return [f"{qid}-v{ref.index}" for ref in refs]


def cropped_video_paths(row: dict[str, str], video_dir: Path) -> tuple[list[Path], list[int]]:
    refs = video_refs_from_row(row)
    stems = question_video_stems(row)
    paths = [video_dir / f"{stem}.mp4" for stem in stems]
    return paths, [ref.index for ref in refs]


def source_video_paths(row: dict[str, str], data_dir: Path) -> tuple[list[Path], list[int]]:
    refs = video_refs_from_row(row)
    return [data_dir / ref.relative_path for ref in refs], [ref.index for ref in refs]
