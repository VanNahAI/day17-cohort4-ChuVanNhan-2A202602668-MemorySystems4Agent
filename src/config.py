from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    env_file = root / ".env"
    if env_file.exists():
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError("Install python-dotenv to load .env, or use process environment variables") from exc
        load_dotenv(env_file, override=False)
    provider = normalize_provider(os.getenv("LLM_PROVIDER", "ollama"))
    defaults = {"ollama": "gpt-oss:120b", "openai": "gpt-4o-mini", "custom": "gpt-4o-mini",
                "gemini": "gemini-2.5-flash", "anthropic": "claude-sonnet-4-5", "openrouter": "openai/gpt-oss-120b"}
    model = ProviderConfig(provider, os.getenv("LLM_MODEL", defaults[provider]),
                           float(os.getenv("LLM_TEMPERATURE", "0")),
                           os.getenv(f"{provider.upper()}_API_KEY"),
                           os.getenv(f"{provider.upper()}_BASE_URL"))
    threshold = int(os.getenv("COMPACT_THRESHOLD_TOKENS", "1800"))
    keep = int(os.getenv("COMPACT_KEEP_MESSAGES", "6"))
    if threshold <= 0 or keep < 1:
        raise ValueError("Compact threshold must be positive and keep_messages must be at least 1")
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return LabConfig(root, root / "data", state_dir, threshold, keep, model, model)