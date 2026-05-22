"""Model registry for GameplayQA benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class InferenceProvider(Enum):
    OPENAI = "openai"
    GOOGLE = "google"
    OPENROUTER = "openrouter"


class InputType(Enum):
    VIDEO_NATIVE = "video_native"
    VISION_FRAMES = "vision_frames"


DEFAULT_MAX_FRAMES = 32


@dataclass(frozen=True)
class ModelConfig:
    name: str
    version: str
    provider: InferenceProvider
    input_type: InputType
    api_model_id: str
    max_frames: int | None = None
    provider_only: list[str] | None = None

    def get_max_frames(self) -> int:
        return self.max_frames or DEFAULT_MAX_FRAMES


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gemini-2.5-pro": ModelConfig("Gemini 2.5 Pro", "gemini-2.5-pro", InferenceProvider.GOOGLE, InputType.VIDEO_NATIVE, "gemini-2.5-pro"),
    "gemini-2.5-flash": ModelConfig("Gemini 2.5 Flash", "gemini-2.5-flash", InferenceProvider.GOOGLE, InputType.VIDEO_NATIVE, "gemini-2.5-flash"),
    "gemini-3-pro": ModelConfig("Gemini 3 Pro", "gemini-3-pro", InferenceProvider.GOOGLE, InputType.VIDEO_NATIVE, "gemini-3-pro-preview"),
    "gemini-3-flash": ModelConfig("Gemini 3 Flash", "gemini-3-flash", InferenceProvider.GOOGLE, InputType.VIDEO_NATIVE, "gemini-3-flash-preview"),
    "gpt-5.2": ModelConfig("GPT-5.2", "gpt-5.2", InferenceProvider.OPENAI, InputType.VISION_FRAMES, "gpt-5.2"),
    "gpt-5": ModelConfig("GPT-5", "gpt-5", InferenceProvider.OPENAI, InputType.VISION_FRAMES, "gpt-5"),
    "gpt-5-mini": ModelConfig("GPT-5 Mini", "gpt-5-mini", InferenceProvider.OPENAI, InputType.VISION_FRAMES, "gpt-5-mini"),
    "gpt-5-nano": ModelConfig("GPT-5 Nano", "gpt-5-nano", InferenceProvider.OPENAI, InputType.VISION_FRAMES, "gpt-5-nano"),
    "claude-4.5-opus": ModelConfig("Claude 4.5 Opus", "claude-4.5-opus", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "anthropic/claude-opus-4.5"),
    "claude-4.5-sonnet": ModelConfig("Claude 4.5 Sonnet", "claude-4.5-sonnet", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "anthropic/claude-sonnet-4.5"),
    "claude-4.5-haiku": ModelConfig("Claude 4.5 Haiku", "claude-4.5-haiku", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "anthropic/claude-haiku-4.5"),
    "qwen3-vl-8b": ModelConfig("Qwen3 VL 8B", "qwen3-vl-8b", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "qwen/qwen3-vl-8b-instruct", max_frames=30, provider_only=["alibaba"]),
    "qwen3-vl-30b": ModelConfig("Qwen3 VL 30B", "qwen3-vl-30b", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "qwen/qwen3-vl-30b-a3b-instruct", max_frames=30, provider_only=["fireworks"]),
    "qwen3-vl-235b": ModelConfig("Qwen3 VL 235B", "qwen3-vl-235b", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "qwen/qwen3-vl-235b-a22b-instruct", max_frames=30, provider_only=["alibaba"]),
    "seed-1.6": ModelConfig("Seed 1.6", "seed-1.6", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "bytedance-seed/seed-1.6", max_frames=24),
    "seed-1.6-flash": ModelConfig("Seed 1.6 Flash", "seed-1.6-flash", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "bytedance-seed/seed-1.6-flash", max_frames=24),
    "internvl3-78b": ModelConfig("InternVL3 78B", "internvl3-78b", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "opengvlab/internvl3-78b", max_frames=4),
    "deepseek-v3.2": ModelConfig("DeepSeek V3.2", "deepseek-v3.2", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "deepseek/deepseek-v3.2"),
    "ministral-3b": ModelConfig("Ministral 3B", "ministral-3b-2512", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "mistralai/ministral-3b-2512", max_frames=8),
    "ministral-8b": ModelConfig("Ministral 8B", "ministral-8b-2512", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "mistralai/ministral-8b-2512", max_frames=8),
    "ministral-14b": ModelConfig("Ministral 14B", "ministral-14b-2512", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "mistralai/ministral-14b-2512", max_frames=8),
    "mistral-large": ModelConfig("Mistral Large", "mistral-large-2512", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "mistralai/mistral-large-2512", max_frames=8),
    "gemma-3-27b": ModelConfig("Gemma 3 27B", "gemma-3-27b-it", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "google/gemma-3-27b-it", provider_only=["chutes"]),
    "gemma-3-12b": ModelConfig("Gemma 3 12B", "gemma-3-12b-it", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "google/gemma-3-12b-it", provider_only=["chutes"]),
    "gemma-3-4b": ModelConfig("Gemma 3 4B", "gemma-3-4b-it", InferenceProvider.OPENROUTER, InputType.VISION_FRAMES, "google/gemma-3-4b-it", provider_only=["chutes"]),
}


def print_model_table() -> None:
    print("\nAvailable models")
    print("=" * 104)
    print(f"{'key':<20} {'name':<24} {'provider':<12} {'input':<15} {'max frames':<10}")
    print("-" * 104)
    for key, cfg in MODEL_REGISTRY.items():
        print(f"{key:<20} {cfg.name:<24} {cfg.provider.value:<12} {cfg.input_type.value:<15} {cfg.get_max_frames():<10}")
    print("=" * 104)
