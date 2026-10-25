"""Capture a health-checked OpenAI-compatible model deployment identity."""

from __future__ import annotations

import hashlib
import json
import platform
import urllib.request
from datetime import datetime, timezone
from typing import Any


def stable_model_listing(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep the serving fields; vLLM regenerates `created` and permission IDs."""
    return sorted(({
        "id": str(item["id"]),
        "root": str(item.get("root") or ""),
        "max_model_len": item.get("max_model_len"),
    } for item in payload.get("data", [])), key=lambda item: item["id"])


def probe_vllm(base_url: str, expected_model: str, *, token: str | None = None,
               timeout: float = 5.0) -> dict[str, Any]:
    """Probe /health and /v1/models before admitting a target-model cell."""
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(base + "/health", headers=headers),
                                timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"vLLM unhealthy: HTTP {response.status}")
    with urllib.request.urlopen(urllib.request.Request(base + "/v1/models", headers=headers),
                                timeout=timeout) as response:
        payload = json.load(response)
    listing = stable_model_listing(payload)
    served = [item["id"] for item in listing]
    if expected_model not in served:
        raise RuntimeError(f"expected model {expected_model!r} not served: {served}")
    stable_bytes = json.dumps(listing, sort_keys=True, separators=(",", ":")).encode()
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": base, "expected_model": expected_model, "served_models": served,
            "stable_model_listing": listing,
            "models_identity_sha256": hashlib.sha256(stable_bytes).hexdigest(),
            "models_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "client_python": platform.python_version(), "status": "healthy"}


def require_target_identity(model: dict[str, Any], deployment: dict[str, Any]) -> None:
    if model.get("role") == "target":
        if deployment.get("status") != "healthy" or model.get("model_id") not in deployment.get("served_models", []):
            raise ValueError("target model requires a matching health-checked deployment")
        for field in ("weight_revision", "vllm_version", "runtime_digest"):
            if not deployment.get(field):
                raise ValueError(f"target deployment missing {field}")


def verify_live_target_deployment(model: dict[str, Any],
                                  frozen: dict[str, Any], live: dict[str, Any]) -> None:
    """Bind a frozen serving-node record to a fresh health-checked model listing."""
    require_target_identity(model, frozen)
    if model.get("role") != "target":
        return
    if live.get("status") != "healthy" or model.get("model_id") not in live.get("served_models", []):
        raise ValueError("target model is not healthy on the live endpoint")
    frozen_hash = frozen.get("models_identity_sha256")
    if not frozen_hash:
        raise ValueError("target deployment lacks a pinned stable model-listing hash")
    if frozen_hash != live.get("models_identity_sha256"):
        raise ValueError("live model listing differs from the frozen deployment")
    model_info = next((item for item in live.get("stable_model_listing", [])
                       if item["id"] == model["model_id"]), None)
    if model_info is None:
        raise ValueError("live model listing lacks the target model details")
    if (frozen.get("max_model_len") is not None
            and frozen["max_model_len"] != model_info["max_model_len"]):
        raise ValueError("served context length differs from the frozen deployment")
    if frozen.get("served_model") and frozen["served_model"] != model["model_id"]:
        raise ValueError("served model name differs from the frozen deployment")
    if (model.get("weights") and frozen.get("model_source")
            and model["weights"] != frozen["model_source"]):
        raise ValueError("model source differs from the experiment descriptor")
    revision = str(frozen["weight_revision"])
    if not model_info["root"].rstrip("/").endswith("/snapshots/" + revision):
        raise ValueError("served model root does not match the frozen weight revision")
