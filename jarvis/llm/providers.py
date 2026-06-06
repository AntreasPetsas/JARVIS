"""LLM providers behind one tiny interface.

Every provider speaks plain HTTP via httpx — no vendor SDKs to install. Ollama
runs locally; Anthropic/OpenAI are optional cloud fallbacks keyed from .env.
Each provider can also explain *why* it's unavailable via hint().
"""
from __future__ import annotations

import json
from typing import Protocol

import httpx

# A "chat" turn result, normalised across providers:
#   {"text": str, "tool_calls": [{"id": str, "name": str, "arguments": dict}, ...]}
ChatResult = dict


class LLMProvider(Protocol):
    name: str
    supports_tools: bool

    async def available(self) -> bool: ...

    async def complete(self, system: str, user: str, max_tokens: int = 500) -> str: ...

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   max_tokens: int = 500) -> ChatResult: ...

    async def hint(self) -> str: ...


class NullProvider:
    name = "none"
    supports_tools = False

    async def available(self) -> bool:
        return False

    async def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        return ""

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   max_tokens: int = 500) -> ChatResult:
        return {"text": "", "tool_calls": []}

    async def hint(self) -> str:
        return ("No language model is configured. Set llm.provider to 'ollama' (local) "
                "or 'anthropic'/'openai' in config.yaml.")


class OllamaProvider:
    name = "ollama"
    supports_tools = True

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    async def _status(self) -> tuple[bool, str]:
        """(ready, message). Distinguishes 'not running' from 'model not pulled'."""
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.host}/api/tags")
        except httpx.HTTPError:
            return False, (f"I can't reach Ollama at {self.host}. Install it from "
                           "ollama.com and make sure it's running.")
        if r.status_code != 200:
            return False, f"Ollama responded with status {r.status_code} at {self.host}."
        tags = [m.get("name", "") for m in r.json().get("models", [])]
        base = self.model.split(":")[0]
        if any(t == self.model or t.startswith(base) for t in tags):
            return True, "ready"
        have = ", ".join(tags) if tags else "none yet"
        return False, (f"Ollama is running, but the model '{self.model}' isn't pulled. "
                       f"Run  ollama pull {self.model}  in a terminal. Installed: {have}.")

    async def available(self) -> bool:
        return (await self._status())[0]

    async def hint(self) -> str:
        return (await self._status())[1]

    async def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            return r.json().get("message", {}).get("content", "").strip()

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   max_tokens: int = 500) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        if tools:  # Ollama uses the OpenAI function-tool wire format
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{self.host}/api/chat", json=payload)
            r.raise_for_status()
            msg = r.json().get("message", {}) or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls", []) or []):
            fn = tc.get("function", {}) or {}
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            calls.append({"id": str(i), "name": fn.get("name", ""), "arguments": args or {}})
        return {"text": (msg.get("content") or "").strip(), "tool_calls": calls}


class AnthropicProvider:
    name = "anthropic"
    supports_tools = True

    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self.api_key = api_key

    async def available(self) -> bool:
        return bool(self.api_key)

    async def hint(self) -> str:
        return "Set ANTHROPIC_API_KEY in your .env to use the Claude API."

    async def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            r.raise_for_status()
            blocks = r.json().get("content", [])
            return "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   max_tokens: int = 500) -> ChatResult:
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        system = " ".join(m["content"] for m in messages if m["role"] == "system")
        conv = [{"role": m["role"], "content": m["content"]}
                for m in messages if m["role"] in ("user", "assistant")]
        payload: dict = {"model": self.model, "max_tokens": max_tokens,
                         "system": system, "messages": conv}
        if tools:
            payload["tools"] = [{"name": t["name"], "description": t["description"],
                                 "input_schema": t["parameters"]} for t in tools]
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
            r.raise_for_status()
            blocks = r.json().get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
        calls = [{"id": b.get("id", str(i)), "name": b.get("name", ""), "arguments": b.get("input", {}) or {}}
                 for i, b in enumerate(blocks) if b.get("type") == "tool_use"]
        return {"text": text, "tool_calls": calls}


class OpenAIProvider:
    name = "openai"
    supports_tools = True

    def __init__(self, model: str, api_key: str | None):
        self.model = model
        self.api_key = api_key

    async def available(self) -> bool:
        return bool(self.api_key)

    async def hint(self) -> str:
        return "Set OPENAI_API_KEY in your .env to use the OpenAI API."

    async def complete(self, system: str, user: str, max_tokens: int = 500) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   max_tokens: int = 500) -> ChatResult:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: dict = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                args = {}
            calls.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})
        return {"text": (msg.get("content") or "").strip(), "tool_calls": calls}
