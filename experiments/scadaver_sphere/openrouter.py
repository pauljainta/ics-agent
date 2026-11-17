"""Small OpenRouter Chat Completions client for client-side tool execution."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenRouterError(RuntimeError):
    pass


@dataclass
class OpenRouterClient:
    model: str
    api_key_env: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    request_timeout: float = 240.0
    max_tokens: int = 1536
    retries: int = 1
    seed: int | None = None
    request_records: list[dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.request_records is None:
            self.request_records = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
             max_tokens: int | None = None, call_index: int | None = None,
             mode: str = "investigation",
             remaining_budget: dict[str, int | float] | None = None) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise OpenRouterError(f"required environment variable {self.api_key_env} is not set")
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        request_body = {"model": self.model, "messages": messages, "tools": tools,
                        "tool_choice": "auto", "temperature": 0, "max_tokens": effective_max_tokens,
                        "stream": False}
        if self.seed is not None:
            request_body["seed"] = self.seed
        payload = json.dumps(request_body).encode()
        request = Request(self.base_url, data=payload, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        })
        record_context = {
            "model_call_index": call_index,
            "mode": mode,
            "model": self.model,
            "provider": None,
            "requested_max_tokens": effective_max_tokens,
            "seed": self.seed,
            "remaining_budget_before": remaining_budget,
            "remaining_budget_after": None,
            "termination_reason": None,
        }
        body: dict[str, Any] | None = None
        for attempt in range(self.retries + 1):
            result: Queue[object] = Queue(maxsize=1)
            started = time.monotonic()

            def request_once() -> None:
                try:
                    with urlopen(request, timeout=self.request_timeout) as response:
                        result.put((response.status, json.loads(response.read())))
                except BaseException as exc:
                    result.put(exc)

            threading.Thread(target=request_once, daemon=True).start()
            try:
                outcome = result.get(timeout=self.request_timeout)
            except Exception:
                self.request_records.append({**record_context, "attempt": attempt + 1,
                                             "latency_seconds": round(time.monotonic() - started, 3),
                                             "status": "deadline_exceeded"})
                if attempt == self.retries:
                    raise OpenRouterError("OpenRouter request exceeded the per-request deadline")
                continue
            if isinstance(outcome, BaseException):
                record = {**record_context, "attempt": attempt + 1,
                          "latency_seconds": round(time.monotonic() - started, 3),
                          "status": "transport_error", "error_type": type(outcome).__name__}
                if isinstance(outcome, HTTPError):
                    record["http_status"] = outcome.code
                    self.request_records.append(record)
                    raise OpenRouterError("OpenRouter returned an HTTP error") from outcome
                self.request_records.append(record)
                if attempt == self.retries:
                    raise OpenRouterError("OpenRouter request failed") from outcome
                continue
            status, body = outcome
            choice = body.get("choices", [{}])[0]
            message = choice.get("message") if isinstance(choice, dict) else None
            tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
            completion_details = (body.get("usage") or {}).get("completion_tokens_details") or {}
            provider_seed = body.get("seed")
            if self.seed is None:
                seed_acknowledgement = "not_requested"
            elif provider_seed is None:
                seed_acknowledgement = "not_reported"
            elif provider_seed == self.seed:
                seed_acknowledgement = "acknowledged"
            else:
                seed_acknowledgement = "mismatch"
            record = {
                **record_context,
                "attempt": attempt + 1,
                "latency_seconds": round(time.monotonic() - started, 3),
                "http_status": status,
                "provider": body.get("provider"),
                "model": body.get("model"),
                "response_id": body.get("id"),
                "finish_reason": choice.get("finish_reason"),
                "requested_max_tokens": effective_max_tokens,
                "seed_requested": self.seed,
                "seed_reported_by_provider": provider_seed,
                "seed_acknowledgement": seed_acknowledgement,
                "provider_limit_evidence": {
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": (body.get("usage") or {}).get("completion_tokens"),
                },
                "message_content_present": bool(isinstance(message, dict) and message.get("content")),
                "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
                "reasoning_tokens": completion_details.get("reasoning_tokens"),
                "usage": body.get("usage"),
            }
            self.request_records.append(record)
            break
        if body is None:
            raise OpenRouterError("OpenRouter request failed")
        try:
            response = dict(body["choices"][0]["message"])
            response["_scadaver_response_metadata"] = {
                "finish_reason": choice.get("finish_reason"),
                "requested_max_tokens": effective_max_tokens,
                "model_call_index": call_index,
                "mode": mode,
            }
            return response
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError("OpenRouter returned no assistant message") from exc

    def update_last_request(self, **values: Any) -> None:
        if self.request_records:
            self.request_records[-1].update(values)

    def mark_termination(self, reason: str) -> None:
        self.update_last_request(termination_reason=reason)
