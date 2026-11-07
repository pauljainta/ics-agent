"""Minimal OpenAI-compatible chat client (stdlib only).

Targets a vLLM server's ``/v1/chat/completions`` endpoint (vLLM implements the
OpenAI protocol), so the PLCBench agents can run a larger open model served on a
GPU box (e.g. Qwen2.5-32B-Instruct on CHPC) while the testbed stays local and
reaches it over an SSH tunnel. Dependency-free (``urllib``); the ``openai`` SDK
is not required. Config-driven: ``base_url`` + ``model`` select the deployment,
shared identically by both comparison arms.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

DEFAULT_BASE_URL = "http://localhost:8000/v1"


class OpenAIError(RuntimeError):
    """Raised when the OpenAI-compatible server cannot be reached or errors."""


def openai_chat(
    prompt: str,
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 180.0,
    system: str | None = None,
    request_options: Mapping[str, Any] | None = None,
) -> str:
    """Call /v1/chat/completions (non-streaming) and return the message content.

    A single user message carries the whole prompt (the agents already assemble
    their own SYSTEM/TASK/ENVIRONMENT structure), unless ``system`` is given.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # Provider-specific controls are opt-in so existing vLLM/OpenAI callers
    # retain their exact request shape.
    request_body.update(dict(request_options or {}))
    body = json.dumps(request_body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions", data=body, headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OpenAIError(f"openai-compatible request to {base_url} failed: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise OpenAIError(f"server returned non-JSON: {exc}") from exc
    try:
        content = payload["choices"][0]["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            finish = payload["choices"][0].get("finish_reason")
            raise OpenAIError(f"completion contained no visible content (finish_reason={finish!r})")
        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAIError(f"unexpected completion shape: {payload!r}") from exc


def openai_available(base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0,
                     api_key: str | None = None) -> bool:
    """Best-effort reachability check against /v1/models (for preflight)."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False
