from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass
class ProviderConfig:
    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    provider = value.strip().lower()
    provider = {"anthorpic": "anthropic", "google": "gemini", "ollama-cloud": "ollama"}.get(provider, provider)
    if provider not in {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}:
        raise ValueError(f"Unsupported provider: {value}")
    return provider


class OllamaChatModel:
    """Native Ollama chat API; no SDK needed for local or cloud inference."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.base_url = (config.base_url or "https://ollama.com").rstrip("/")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Invalid Ollama base URL")
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            if parsed.scheme != "https":
                raise ValueError("Remote Ollama requires HTTPS")
            if not config.api_key:
                raise ValueError("Set OLLAMA_API_KEY for Ollama Cloud")

    def invoke(self, messages: list[dict[str, str]]):
        payload = json.dumps({"model": self.config.model_name, "messages": messages,
                              "stream": False, "options": {"temperature": self.config.temperature}}).encode()
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = Request(f"{self.base_url}/api/chat", data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
        if result.get("error"):
            raise RuntimeError(f"Ollama request failed: {result['error']}")
        content = result["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Ollama returned empty response content")
        return SimpleNamespace(content=content, usage_metadata={
            "input_tokens": result.get("prompt_eval_count"),
            "output_tokens": result.get("eval_count"),
        })


def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    if not config.model_name.strip():
        raise ValueError("Model name cannot be empty")
    if provider == "ollama":
        return OllamaChatModel(config)
    kwargs = {"model": config.model_name, "temperature": config.temperature}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if provider in {"openai", "custom", "openrouter"}:
        from langchain_openai import ChatOpenAI
        if provider == "custom" and not config.base_url:
            raise ValueError("Set CUSTOM_BASE_URL for custom provider")
        base_url = config.base_url or ("https://openrouter.ai/api/v1" if provider == "openrouter" else None)
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(**kwargs)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(**kwargs)