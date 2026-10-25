"""Deployment admission must survive volatile vLLM metadata and reject swaps."""

from __future__ import annotations

import hashlib
import json

import pytest

from ics_agent.plcbench_model_identity import stable_model_listing, verify_live_target_deployment


REVISION = "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c"
ROOT = "/cache/snapshots/" + REVISION


def _identity(root: str = ROOT, *, created: int = 1, permission_id: str = "one"):
    payload = {"data": [{"id": "plcbench", "root": root, "max_model_len": 8192,
                         "created": created, "permission": [{"id": permission_id}]}]}
    listing = stable_model_listing(payload)
    digest = hashlib.sha256(json.dumps(listing, sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"status": "healthy", "served_models": ["plcbench"],
            "stable_model_listing": listing, "models_identity_sha256": digest}


def test_stable_listing_ignores_per_request_vllm_fields():
    assert _identity(created=1, permission_id="one") == _identity(
        created=2, permission_id="two")


def test_target_deployment_rejects_swapped_root_and_wrong_revision():
    model = {"role": "target", "model_id": "plcbench"}
    frozen = {**_identity(), "weight_revision": REVISION,
              "vllm_version": "0.27.0", "runtime_digest": "container-sha"}
    verify_live_target_deployment(model, frozen, _identity(created=2))
    with pytest.raises(ValueError, match="differs"):
        verify_live_target_deployment(model, frozen, _identity(root="/cache/snapshots/other"))
    with pytest.raises(ValueError, match="weight revision"):
        verify_live_target_deployment(model, {**frozen, "weight_revision": "other"},
                                      _identity())
    with pytest.raises(ValueError, match="context length"):
        verify_live_target_deployment(model, {**frozen, "max_model_len": 4096}, _identity())
    with pytest.raises(ValueError, match="pinned stable"):
        verify_live_target_deployment(model, {k: v for k, v in frozen.items()
                                              if k != "models_identity_sha256"}, _identity())
