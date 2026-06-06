"""LLM access — one factory, swappable providers (config: llm.provider)."""
from __future__ import annotations

from ..config import Config
from .providers import (
    AnthropicProvider,
    LLMProvider,
    NullProvider,
    OllamaProvider,
    OpenAIProvider,
)
from .tools import TOOLS


def get_llm(cfg: Config) -> LLMProvider:
    provider = (cfg.get("llm.provider", "none") or "none").lower()
    model = cfg.get("llm.model", "")
    if provider == "ollama":
        return OllamaProvider(cfg.get("llm.host", "http://localhost:11434"), model)
    if provider == "anthropic":
        return AnthropicProvider(model or "claude-sonnet-4-6", Config.env("ANTHROPIC_API_KEY"))
    if provider == "openai":
        return OpenAIProvider(model or "gpt-4o-mini", Config.env("OPENAI_API_KEY"))
    return NullProvider()


__all__ = ["get_llm", "LLMProvider", "TOOLS"]
