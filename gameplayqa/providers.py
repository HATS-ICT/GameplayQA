"""Inference provider adapters."""

from __future__ import annotations

import base64
import os
from pathlib import Path

import requests

from .models import InferenceProvider, InputType, ModelConfig
from .video import FrameExtractionConfig, extract_frames, frames_to_base64_urls, load_preextracted_frames


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(".env.local"))
    load_dotenv()


def video_mime_type(video_path: Path) -> str:
    return {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
        ".mkv": "video/x-matroska",
    }.get(video_path.suffix.lower(), "video/mp4")


class VideoProcessor:
    def __init__(self, model_config: ModelConfig, frames_dir: Path | None = None):
        load_dotenv_if_available()
        self.model_config = model_config
        self.frames_dir = frames_dir

    def process(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None = None) -> dict:
        raise NotImplementedError

    def process_text(self, prompt: str) -> dict:
        raise NotImplementedError


class GoogleProcessor(VideoProcessor):
    def __init__(self, model_config: ModelConfig, frames_dir: Path | None = None):
        super().__init__(model_config, frames_dir)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY is not set")

    def process(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None = None) -> dict:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        video_indices = video_indices or list(range(len(video_paths)))
        parts = []
        is_multi = len(video_paths) > 1
        for video_path, idx in zip(video_paths, video_indices):
            if is_multi:
                parts.append(types.Part(text=f"Below is Video {idx + 1}:"))
            parts.append(
                types.Part(
                    inline_data=types.Blob(
                        data=video_path.read_bytes(),
                        mime_type=video_mime_type(video_path),
                    )
                )
            )
        parts.append(types.Part(text=prompt))
        response = client.models.generate_content(
            model=self.model_config.api_model_id,
            contents=types.Content(parts=parts),
        )
        return self._result(response.text or "", videos_used=len(video_paths))

    def process_text(self, prompt: str) -> dict:
        from google import genai

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(model=self.model_config.api_model_id, contents=prompt)
        return self._result(response.text or "", frames_used=0)

    def _result(self, response: str, **extra: object) -> dict:
        return {
            "model": self.model_config.name,
            "model_id": self.model_config.api_model_id,
            "provider": self.model_config.provider.value,
            "response": response,
            **extra,
        }


class OpenAIFrameProcessor(VideoProcessor):
    def __init__(self, model_config: ModelConfig, frames_dir: Path | None = None):
        super().__init__(model_config, frames_dir)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")

    def _load_frames(self, video_path: Path, max_frames: int) -> tuple[list[bytes], dict | None]:
        if self.frames_dir:
            pre = load_preextracted_frames(video_path, self.frames_dir, max_frames)
            if pre:
                return pre
        extracted = extract_frames(video_path, FrameExtractionConfig(max_frames=max_frames))
        return extracted.frames, {"resolution": list(extracted.resolution)}

    def process(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None = None) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        video_indices = video_indices or list(range(len(video_paths)))
        max_per_video = max(1, self.model_config.get_max_frames() // max(1, len(video_paths)))
        content = []
        frames_used = 0
        for video_path, idx in zip(video_paths, video_indices):
            frames, _ = self._load_frames(video_path, max_per_video)
            if len(video_paths) > 1:
                content.append({"type": "input_text", "text": f"Below are frames from Video {idx + 1}:"})
            for url in frames_to_base64_urls(frames):
                content.append({"type": "input_image", "image_url": url})
            frames_used += len(frames)
        content.append({"type": "input_text", "text": prompt})
        response = client.responses.create(model=self.model_config.api_model_id, input=[{"role": "user", "content": content}])
        return self._result(response.output_text or "", frames_used=frames_used, videos_used=len(video_paths))

    def process_text(self, prompt: str) -> dict:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.model_config.api_model_id,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        )
        return self._result(response.output_text or "", frames_used=0, videos_used=0)

    def _result(self, response: str, **extra: object) -> dict:
        return {
            "model": self.model_config.name,
            "model_id": self.model_config.api_model_id,
            "provider": self.model_config.provider.value,
            "response": response,
            **extra,
        }


class OpenRouterProcessor(VideoProcessor):
    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_config: ModelConfig, frames_dir: Path | None = None):
        super().__init__(model_config, frames_dir)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")

    def _post(self, content: list[dict]) -> dict:
        payload: dict = {"model": self.model_config.api_model_id, "messages": [{"role": "user", "content": content}]}
        if self.model_config.provider_only:
            payload["provider"] = {"only": self.model_config.provider_only}
        response = requests.post(
            self.BASE_URL,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
        )
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {response.text[:500]}")
        data = response.json()
        if "error" in data:
            raise RuntimeError(f"OpenRouter API error: {data['error']}")
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return self._result(text)

    def process(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None = None) -> dict:
        if self.model_config.input_type == InputType.VIDEO_NATIVE:
            return self._process_videos(video_paths, prompt, video_indices)
        return self._process_frames(video_paths, prompt, video_indices)

    def _process_videos(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None) -> dict:
        video_indices = video_indices or list(range(len(video_paths)))
        content: list[dict] = []
        for video_path, idx in zip(video_paths, video_indices):
            if len(video_paths) > 1:
                content.append({"type": "text", "text": f"Below is Video {idx + 1}:"})
            data_url = f"data:{video_mime_type(video_path)};base64,{base64.b64encode(video_path.read_bytes()).decode('utf-8')}"
            content.append({"type": "video_url", "video_url": {"url": data_url}})
        content.append({"type": "text", "text": prompt})
        result = self._post(content)
        return {**result, "videos_used": len(video_paths)}

    def _process_frames(self, video_paths: list[Path], prompt: str, video_indices: list[int] | None) -> dict:
        video_indices = video_indices or list(range(len(video_paths)))
        max_per_video = max(1, self.model_config.get_max_frames() // max(1, len(video_paths)))
        content: list[dict] = []
        frames_used = 0
        for video_path, idx in zip(video_paths, video_indices):
            pre = load_preextracted_frames(video_path, self.frames_dir, max_per_video) if self.frames_dir else None
            frames = pre[0] if pre else extract_frames(video_path, FrameExtractionConfig(max_frames=max_per_video)).frames
            if len(video_paths) > 1:
                content.append({"type": "text", "text": f"Below are frames from Video {idx + 1}:"})
            for url in frames_to_base64_urls(frames):
                content.append({"type": "image_url", "image_url": {"url": url}})
            frames_used += len(frames)
        content.append({"type": "text", "text": prompt})
        result = self._post(content)
        return {**result, "frames_used": frames_used, "videos_used": len(video_paths)}

    def process_text(self, prompt: str) -> dict:
        result = self._post([{"type": "text", "text": prompt}])
        return {**result, "frames_used": 0, "videos_used": 0}

    def _result(self, response: str, **extra: object) -> dict:
        return {
            "model": self.model_config.name,
            "model_id": self.model_config.api_model_id,
            "provider": self.model_config.provider.value,
            "response": response,
            **extra,
        }


def get_processor(model_config: ModelConfig, frames_dir: Path | None = None) -> VideoProcessor:
    if model_config.provider == InferenceProvider.GOOGLE:
        return GoogleProcessor(model_config, frames_dir)
    if model_config.provider == InferenceProvider.OPENAI:
        return OpenAIFrameProcessor(model_config, frames_dir)
    if model_config.provider == InferenceProvider.OPENROUTER:
        return OpenRouterProcessor(model_config, frames_dir)
    raise NotImplementedError(f"Unsupported provider: {model_config.provider}")
