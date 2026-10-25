"""Synthetic local Mattermost API workflow using the PLCBench adapter."""

from __future__ import annotations

import json
import os
import urllib.request

from ics_agent.plcbench_mattermost import LocalMattermost


def main() -> None:
    base = "http://127.0.0.1:8065"
    login = {"login_id": os.environ["MM_FIXTURE_USER"],
             "password": os.environ["MM_FIXTURE_PASSWORD"]}
    request = urllib.request.Request(base + "/api/v4/users/login",
        data=json.dumps(login).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        token = response.headers["Token"]
    channel_id = os.environ["MM_FIXTURE_CHANNEL_ID"]
    client = LocalMattermost(base, token)
    marker = "PLCBench synthetic observation; no plant action"
    posted = client.post_fixture_signal(channel_id, marker)
    observed = client.observe_channel(channel_id)
    match = [item for item in observed if item.payload.get("post_id") == posted["post_id"]]
    if len(match) != 1 or match[0].payload["message"] != marker:
        raise RuntimeError("social fixture post was not observed through adapter")
    print(json.dumps({"status": "passed", "post_id": posted["post_id"],
                      "kind": match[0].kind, "authority": match[0].authority,
                      "observed_at_utc": match[0].observed_at_utc}))


if __name__ == "__main__":
    main()
