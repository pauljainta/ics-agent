"""Isolated localhost Mattermost social fixture adapter; no plant controls."""

from __future__ import annotations

import json
import urllib.request
from typing import Any
from urllib.parse import urlsplit

from .plcbench_observations import ExternalObservation, observation


class LocalMattermost:
    def __init__(self, base_url: str, token: str) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Mattermost fixture must be an isolated localhost server")
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _request(self, path: str, *, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(self.base_url + "/api/v4" + path, data=data,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def post_fixture_signal(self, channel_id: str, message: str) -> dict[str, Any]:
        """Post only to a local fixture channel; caller owns the test server."""
        if not channel_id or len(message) > 1000:
            raise ValueError("invalid fixture signal")
        post = self._request("/posts", payload={"channel_id": channel_id, "message": message})
        return {"post_id": post["id"], "channel_id": post["channel_id"],
                "created_at_ms": post["create_at"]}

    def observe_channel(self, channel_id: str) -> list[ExternalObservation]:
        result = self._request(f"/channels/{channel_id}/posts")
        posts = result.get("posts", {})
        return [observation("social", f"mattermost:{channel_id}",
                            {"post_id": post["id"], "user_id": post["user_id"],
                             "message": post["message"]}, timestamp_ms=post["create_at"])
                for post_id in result.get("order", []) if (post := posts.get(post_id))]
