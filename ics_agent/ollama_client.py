"""Minimal Ollama HTTP client (stdlib only).

Lets the PLCBench agents run on a local Ollama server (Metal on macOS), which is
how the model runs on a machine without a CUDA GPU. Deliberately dependency-free
(``urllib``) because neither ``ollama`` nor ``langchain_ollama`` is installed; the
backend is config-driven so the frozen model is a one-line change (model id +
options), and the same deployment is shared by the matched baseline, its
comparator, and the native actor.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_HOST = "http://localhost:11434"


class OllamaError(RuntimeError):
    """Raised when the Ollama server cannot be reached or returns an error."""


def ollama_generate(
    prompt: str,
    *,
    model: str,
    host: str = DEFAULT_HOST,
    temperature: float = 0.0,
    num_predict: int = 512,
    timeout: float = 120.0,
    extra_options: dict[str, Any] | None = None,
) -> str:
    """Call Ollama's /api/generate (non-streaming) and return the completion text.

    Options are pinned for a deterministic, bounded decision: temperature and
    num_predict map to the frozen decoding config and output ceiling. Raises
    OllamaError on transport/HTTP failure so callers can record a backend error
    rather than silently degrade.
    """
    options: dict[str, Any] = {"temperature": temperature, "num_predict": num_predict}
    if extra_options:
        options.update(extra_options)
    body = json.dumps(
        {"model": model, "prompt": prompt, "stream": False, "options": options}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(f"ollama request to {host} failed: {exc}") from exc
    except (ValueError, json.JSONDecodeError) as exc:
        raise OllamaError(f"ollama returned non-JSON: {exc}") from exc
    if "response" not in payload:
        raise OllamaError(f"ollama response missing 'response': {payload!r}")
    return str(payload["response"])


def ollama_available(host: str = DEFAULT_HOST, timeout: float = 2.0) -> bool:
    """Best-effort check that an Ollama server is reachable (for preflight)."""
    try:
        with urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False
