"""N3-only OpenRouter client with reproducible tool-choice controls."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class N3InferenceError(RuntimeError):
    pass


@dataclass
class N3OpenRouterClient:
    model: str
    api_key_env: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    request_timeout: float = 240.0
    max_tokens: int = 1536
    retries: int = 1
    seed: int | None = None
    output_token_fallbacks: tuple[int, ...] = ()
    request_records: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.request_records is None:
            self.request_records = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *,
             max_tokens: int | None = None, call_index: int | None = None,
             mode: str = "investigation",
             remaining_budget: dict[str, int | float] | None = None,
             tool_choice: str | dict[str, Any] = "auto") -> dict[str, Any]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise N3InferenceError(f"required environment variable {self.api_key_env} is not set")
        requested = max_tokens if max_tokens is not None else self.max_tokens
        candidates = (requested,) + tuple(
            value for value in self.output_token_fallbacks if 0 < value < requested)
        context = {
            "model_call_index": call_index, "mode": mode, "model": self.model,
            "provider": None, "requested_max_tokens": requested,
            "requested_tool_choice": tool_choice, "seed": self.seed,
            "remaining_budget_before": remaining_budget,
            "remaining_budget_after": None, "termination_reason": None,
        }
        body: dict[str, Any] | None = None
        effective_requested: int | None = None
        for candidate_index, candidate in enumerate(candidates):
            request_body: dict[str, Any] = {
                "model": self.model, "messages": messages, "tools": tools,
                "tool_choice": tool_choice, "temperature": 0, "max_tokens": candidate,
                "stream": False,
            }
            if self.seed is not None:
                request_body["seed"] = self.seed
            request = Request(self.base_url, data=json.dumps(request_body).encode(), method="POST", headers={
                "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            })
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
                    self.request_records.append({**context, "attempt": attempt + 1,
                                                 "requested_max_tokens": candidate,
                                                 "latency_seconds": round(time.monotonic() - started, 3),
                                                 "status": "deadline_exceeded"})
                    if attempt == self.retries:
                        raise N3InferenceError("OpenRouter request exceeded the per-request deadline")
                    continue
                if isinstance(outcome, BaseException):
                    record = {**context, "attempt": attempt + 1,
                              "requested_max_tokens": candidate,
                              "latency_seconds": round(time.monotonic() - started, 3),
                              "status": "transport_error", "error_type": type(outcome).__name__}
                    if isinstance(outcome, HTTPError):
                        record["http_status"] = outcome.code
                        if outcome.code in {400, 422} and candidate_index + 1 < len(candidates):
                            record["output_limit_fallback_to"] = candidates[candidate_index + 1]
                            self.request_records.append(record)
                            break
                        self.request_records.append(record)
                        raise N3InferenceError("OpenRouter returned an HTTP error") from outcome
                    self.request_records.append(record)
                    if attempt == self.retries:
                        raise N3InferenceError("OpenRouter request failed") from outcome
                    continue
                status, body = outcome
                effective_requested = candidate
                choice = body.get("choices", [{}])[0]
                message = choice.get("message") if isinstance(choice, dict) else None
                tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
                usage = body.get("usage") or {}
                details = usage.get("completion_tokens_details") or {}
                provider_seed = body.get("seed")
                seed_status = (
                    "not_requested" if self.seed is None else "not_reported" if provider_seed is None
                    else "acknowledged" if provider_seed == self.seed else "mismatch"
                )
                self.request_records.append({
                    **context, "attempt": attempt + 1, "requested_max_tokens": candidate,
                    "output_limit_fallback_from": requested if candidate != requested else None,
                    "latency_seconds": round(time.monotonic() - started, 3),
                    "http_status": status, "provider": body.get("provider"), "model": body.get("model"),
                    "response_id": body.get("id"), "finish_reason": choice.get("finish_reason"),
                    "seed_requested": self.seed, "seed_reported_by_provider": provider_seed,
                    "seed_acknowledgement": seed_status,
                    "provider_limit_evidence": {
                        "finish_reason": choice.get("finish_reason"),
                        "completion_tokens": usage.get("completion_tokens"),
                    },
                    "message_content_present": bool(isinstance(message, dict) and message.get("content")),
                    "tool_call_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
                    "reasoning_tokens": details.get("reasoning_tokens"), "usage": usage,
                })
                break
            if body is not None:
                break
        if body is None:
            raise N3InferenceError("OpenRouter request failed")
        try:
            response = dict(body["choices"][0]["message"])
            response["_scadaver_response_metadata"] = {
                "finish_reason": choice.get("finish_reason"),
                "requested_max_tokens": effective_requested, "requested_tool_choice": tool_choice,
                "model_call_index": call_index, "mode": mode,
            }
            return response
        except (KeyError, IndexError, TypeError) as exc:
            raise N3InferenceError("OpenRouter returned no assistant message") from exc

    def update_last_request(self, **values: Any) -> None:
        if self.request_records:
            self.request_records[-1].update(values)

    def mark_termination(self, reason: str) -> None:
        self.update_last_request(termination_reason=reason)
