"""Video cropping and frame extraction."""

from __future__ import annotations

import base64
import io
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .data import cropped_video_paths, question_video_stems, read_csv_rows, source_video_paths
from .progress import console, make_progress


@dataclass(frozen=True)
class FrameExtractionConfig:
    max_frames: int = 32
    max_resolution: int = 720
    fps_for_short_videos: float = 1.0
    short_video_threshold: float = 32.0


@dataclass(frozen=True)
class ExtractedFrames:
    frames: list[bytes]
    frame_count: int
    video_duration: float
    original_fps: float
    sample_fps: float
    resolution: tuple[int, int]


def crop_segment(input_path: Path, output_path: Path, start: float, end: float, force: bool = False) -> str:
    if output_path.exists() and not force:
        return "skipped"
    if end <= start:
        raise ValueError(f"Invalid crop range: {start} to {end}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start),
        "-i",
        str(input_path),
        "-t",
        str(end - start),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for cropping but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or str(exc)) from exc
    return "created"


def crop_questions(csv_path: Path, data_dir: Path, output_dir: Path, force: bool = False, limit: int | None = None) -> dict[str, int]:
    rows = read_csv_rows(csv_path)
    if limit:
        rows = rows[:limit]
    counts = {"created": 0, "skipped": 0, "failed": 0}
    work_items = []
    for row in rows:
        sources, _ = source_video_paths(row, data_dir)
        outputs, _ = cropped_video_paths(row, output_dir)
        start = float(row["video_start"])
        end = float(row["video_end"])
        for source, output in zip(sources, outputs):
            work_items.append((row.get("id", ""), source, output, start, end))

    with make_progress() as progress:
        task = progress.add_task("Cropping question videos", total=len(work_items))
        for question_id, source, output, start, end in work_items:
            if not source.exists():
                console.print(f"[red]missing[/red] {question_id}: {source}")
                counts["failed"] += 1
                progress.advance(task)
                continue
            try:
                status = crop_segment(source, output, start, end, force=force)
                counts[status] += 1
            except Exception as exc:
                counts["failed"] += 1
                console.print(f"[red]failed[/red] {output.name}: {exc}")
            progress.advance(task)
    return counts


def _cv2():
    import cv2

    return cv2


def get_video_info(video_path: Path) -> tuple[float, float, int, int]:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        return duration, fps, width, height
    finally:
        cap.release()


def resize_dimensions(width: int, height: int, max_size: int) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        return width, height
    longer = max(width, height)
    if longer <= max_size:
        return width, height
    scale = max_size / longer
    return int(width * scale), int(height * scale)


def extract_frames(video_path: Path, config: FrameExtractionConfig | None = None) -> ExtractedFrames:
    cv2 = _cv2()
    config = config or FrameExtractionConfig()
    duration, original_fps, width, height = get_video_info(video_path)
    new_width, new_height = resize_dimensions(width, height, config.max_resolution)
    if duration < config.short_video_threshold:
        sample_fps = config.fps_for_short_videos
        total_frames = min(int(duration * sample_fps), config.max_frames)
    else:
        total_frames = config.max_frames
        sample_fps = total_frames / duration if duration else 0
    total_frames = max(1, min(total_frames, config.max_frames))
    target_times = [i * duration / total_frames for i in range(total_frames)] if duration else [0]

    cap = cv2.VideoCapture(str(video_path))
    frames: list[bytes] = []
    try:
        for target_time in target_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, target_time * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            if (width, height) != (new_width, new_height):
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buffer = io.BytesIO()
            Image.fromarray(frame_rgb).save(buffer, format="JPEG", quality=85)
            frames.append(buffer.getvalue())
    finally:
        cap.release()
    return ExtractedFrames(frames, len(frames), duration, original_fps, sample_fps, (new_width, new_height))


def frames_to_base64_urls(frames: list[bytes]) -> list[str]:
    return [f"data:image/jpeg;base64,{base64.b64encode(frame).decode('utf-8')}" for frame in frames]


def save_frames(video_path: Path, frames_root: Path, max_frames: int, force: bool = False) -> dict:
    out_dir = frames_root / video_path.stem
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists() and not force:
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        if metadata.get("max_frames") == max_frames:
            return {"status": "skipped", **metadata}
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = extract_frames(video_path, FrameExtractionConfig(max_frames=max_frames))
    for idx, frame in enumerate(extracted.frames):
        (out_dir / f"frame_{idx:02d}.jpg").write_bytes(frame)
    metadata = {
        "question_id": video_path.stem,
        "video_path": str(video_path),
        "frame_count": extracted.frame_count,
        "video_duration": extracted.video_duration,
        "original_fps": extracted.original_fps,
        "sample_fps": extracted.sample_fps,
        "resolution": list(extracted.resolution),
        "max_frames": max_frames,
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return {"status": "extracted", **metadata}


def load_preextracted_frames(video_path: Path, frames_dir: Path, max_frames: int) -> tuple[list[bytes], dict] | None:
    frames_path = frames_dir / video_path.stem
    metadata_path = frames_path / "metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    frames: list[bytes] = []
    for idx in range(min(int(metadata.get("frame_count", 0)), max_frames)):
        frame_path = frames_path / f"frame_{idx:02d}.jpg"
        if frame_path.exists():
            frames.append(frame_path.read_bytes())
    return (frames, metadata) if frames else None


def extract_question_frames(
    csv_path: Path,
    video_dir: Path,
    output_dir: Path,
    max_frames: int = 32,
    workers: int = 4,
    force: bool = False,
    limit: int | None = None,
) -> dict[str, int]:
    rows = read_csv_rows(csv_path)
    stems: list[str] = []
    for row in rows:
        stems.extend(question_video_stems(row))
    if limit:
        stems = stems[:limit]
    videos = [video_dir / f"{stem}.mp4" for stem in stems]
    counts = {"extracted": 0, "skipped": 0, "missing": 0, "error": 0}

    def work(video_path: Path) -> dict:
        if not video_path.exists():
            return {"status": "missing", "question_id": video_path.stem}
        try:
            return save_frames(video_path, output_dir, max_frames, force=force)
        except Exception as exc:
            return {"status": "error", "question_id": video_path.stem, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(work, path) for path in videos]
        with make_progress() as progress:
            task = progress.add_task("Extracting frame cache", total=len(videos))
            for future in as_completed(futures):
                result = future.result()
                status = result["status"]
                counts[status] += 1
                if status in {"missing", "error"}:
                    detail = result.get("error") or "video not found"
                    console.print(f"[red]{status}[/red] {result['question_id']}: {detail}")
                progress.advance(task)
    return counts
