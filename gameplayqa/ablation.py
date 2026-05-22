"""Ablation benchmark runner."""

from __future__ import annotations

import json
import random
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .data import cropped_video_paths, read_csv_rows
from .models import MODEL_REGISTRY, print_model_table
from .progress import console, make_progress
from .providers import OpenAIFrameProcessor, OpenRouterProcessor, get_processor
from .questions import build_answer_prompt, build_text_only_prompt, format_multiple_choice_from_row
from .video import FrameExtractionConfig, extract_frames, frames_to_base64_urls, load_preextracted_frames

ABLATION_TYPES = ["no_video", "random_single_frame", "shuffled_frames"]


@dataclass
class AblationResult:
    question_id: str
    question_text: str
    correct_option: str
    correct_answer: str
    option_distractor_types: dict[str, str | None]
    shuffled_order: str
    model_output: str
    model_name: str
    model_id: str
    ablation_type: str
    timestamp: str
    split: str
    error: str | None = None
    frames_used: int | None = None
    videos_used: int | None = None
    ablation_metadata: dict | None = None


class FrameAblationProcessor:
    def __init__(self, base_processor, frames_dir: Path | None, mode: str, seed: int | None):
        self.base_processor = base_processor
        self.frames_dir = frames_dir
        self.mode = mode
        self.seed = seed

    def process(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None = None) -> dict:
        if isinstance(self.base_processor, OpenAIFrameProcessor):
            return self._process_openai(video_paths, prompt, video_indices)
        if isinstance(self.base_processor, OpenRouterProcessor):
            return self._process_openrouter(video_paths, prompt, video_indices)
        raise ValueError("Frame ablations require an OpenAI or OpenRouter vision-frame model")

    def _frames_for_video(self, video_path: Path, max_frames: int) -> tuple[list[bytes], dict]:
        pre = load_preextracted_frames(video_path, self.frames_dir, max_frames) if self.frames_dir else None
        frames = pre[0] if pre else extract_frames(video_path, FrameExtractionConfig(max_frames=max_frames)).frames
        indices = list(range(len(frames)))
        digest = hashlib.sha256(str(video_path).encode("utf-8")).hexdigest()
        rng = random.Random((self.seed or 0) + int(digest[:12], 16))
        if self.mode == "random_single_frame":
            chosen = rng.choice(indices) if indices else 0
            return [frames[chosen]], {"selected_frame_indices": [chosen]}
        rng.shuffle(indices)
        return [frames[i] for i in indices], {"shuffle_order": indices}

    def _content(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None, openai: bool) -> tuple[list[dict], dict]:
        video_indices = video_indices or list(range(len(video_paths)))
        max_per_video = max(1, self.base_processor.model_config.get_max_frames() // max(1, len(video_paths)))
        content: list[dict] = []
        metadata: dict = {"per_video": []}
        frames_used = 0
        text_type = "input_text" if openai else "text"
        image_type = "input_image" if openai else "image_url"
        for video_path, idx in zip(video_paths, video_indices):
            frames, meta = self._frames_for_video(video_path, max_per_video)
            metadata["per_video"].append(meta)
            if len(video_paths) > 1:
                content.append({"type": text_type, "text": f"Below are frames from Video {idx + 1}:"})
            for url in frames_to_base64_urls(frames):
                if openai:
                    content.append({"type": image_type, "image_url": url})
                else:
                    content.append({"type": image_type, "image_url": {"url": url}})
            frames_used += len(frames)
        content.append({"type": text_type, "text": prompt})
        metadata["frames_used"] = frames_used
        return content, metadata

    def _process_openai(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None) -> dict:
        from openai import OpenAI

        content, metadata = self._content(video_paths, prompt, video_indices, openai=True)
        client = OpenAI(api_key=self.base_processor.api_key)
        response = client.responses.create(model=self.base_processor.model_config.api_model_id, input=[{"role": "user", "content": content}])
        return self.base_processor._result(response.output_text or "", frames_used=metadata["frames_used"], videos_used=len(video_paths), ablation_metadata=metadata)

    def _process_openrouter(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None) -> dict:
        content, metadata = self._content(video_paths, prompt, video_indices, openai=False)
        result = self.base_processor._post(content)
        return {**result, "frames_used": metadata["frames_used"], "videos_used": len(video_paths), "ablation_metadata": metadata}


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ids.add(json.loads(line)["question_id"])
    return ids


def process_question(row: dict[str, str], processor, video_dir: Path, model_name: str, model_id: str, split: str, ablation_type: str, seed: int) -> AblationResult:
    timestamp = datetime.now().isoformat()
    qid = row.get("id", "")
    try:
        mc = format_multiple_choice_from_row(row, seed=seed)
        if ablation_type == "no_video":
            prompt = build_text_only_prompt(mc.formatted_text)
            response = processor.process_text(prompt)
        else:
            paths, indices = cropped_video_paths(row, video_dir)
            missing = [path for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError("Missing cropped video(s): " + ", ".join(str(path) for path in missing))
            prompt = build_answer_prompt(mc.formatted_text, len(paths))
            response = processor.process(paths, prompt, video_indices=indices)
        output = (response.get("response") or "").strip()
        if not output:
            raise RuntimeError("Model returned an empty response")
        return AblationResult(
            question_id=qid,
            question_text=mc.formatted_text,
            correct_option=mc.correct_option,
            correct_answer=mc.correct_answer,
            option_distractor_types=mc.option_distractor_types,
            shuffled_order=mc.shuffled_order,
            model_output=output,
            model_name=model_name,
            model_id=model_id,
            ablation_type=ablation_type,
            timestamp=timestamp,
            split=split,
            frames_used=response.get("frames_used"),
            videos_used=response.get("videos_used"),
            ablation_metadata=response.get("ablation_metadata"),
        )
    except Exception as exc:
        return AblationResult(qid, row.get("question", ""), "", row.get("correct_option", ""), {}, "", "", model_name, model_id, ablation_type, timestamp, split, error=str(exc))


def run_ablation(
    ablation_type: str,
    model_key: str,
    csv_path: Path,
    video_dir: Path,
    frames_dir: Path | None,
    output_path: Path,
    split: str,
    workers: int = 5,
    limit: int | None = None,
    resume: bool = False,
    seed: int = 13,
) -> None:
    if ablation_type not in ABLATION_TYPES:
        print("Available ablations: " + ", ".join(ABLATION_TYPES))
        return
    if model_key not in MODEL_REGISTRY:
        print_model_table()
        return
    cfg = MODEL_REGISTRY[model_key]
    base = get_processor(cfg, frames_dir=frames_dir)
    processor = base if ablation_type == "no_video" else FrameAblationProcessor(base, frames_dir, ablation_type, seed)
    rows = read_csv_rows(csv_path)
    if limit:
        rows = rows[:limit]
    if resume:
        existing = load_existing_ids(output_path)
        rows = [row for row in rows if row.get("id") not in existing]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    ok = 0
    errors = 0
    start = time.time()
    lock = threading.Lock()
    console.print(f"Running ablation={ablation_type} model={cfg.name} questions={len(rows)} output={output_path}")
    with output_path.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_question, row, processor, video_dir, cfg.name, cfg.api_model_id, split, ablation_type, seed) for row in rows]
            with make_progress() as progress:
                task = progress.add_task("Running ablation", total=len(rows))
                for future in as_completed(futures):
                    result = future.result()
                    if result.error:
                        errors += 1
                        console.print(f"[red]error[/red] {result.question_id}: {result.error}")
                    else:
                        ok += 1
                        with lock:
                            out.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
                            out.flush()
                    progress.advance(task)
    console.print(f"Done. successful={ok} errors={errors} elapsed={time.time() - start:.1f}s")
