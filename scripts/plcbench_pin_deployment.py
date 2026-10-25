"""Pin or verify a serving-node deployment record against a live vLLM endpoint.

The API key is read from a file or environment, never a command-line value.
Freezing creates a new file; subsequent checks never replace its pinned identity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ics_agent.plcbench_model_identity import probe_vllm, verify_live_target_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", required=True, type=Path,
                        help="serving-node metadata, or an already pinned deployment record")
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-id", default="plcbench")
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--freeze-output", type=Path,
                        help="create a new pinned record from serving-node metadata")
    args = parser.parse_args()

    record = json.loads(args.record.read_text())
    token = (args.api_key_file.read_text().strip() if args.api_key_file
             else os.environ.get("PLCBENCH_API_KEY"))
    live = probe_vllm(args.model_url, args.model_id, token=token)
    model = {"role": "target", "model_id": args.model_id}
    if args.freeze_output:
        if record.get("models_identity_sha256"):
            parser.error("record is already pinned; verify it without --freeze-output")
        pinned = {**record, **live,
                  "stable_listing_captured_at_utc": live["checked_at_utc"]}
        verify_live_target_deployment(model, pinned, live)
        with args.freeze_output.open("x") as stream:
            json.dump(pinned, stream, indent=2)
            stream.write("\n")
        output = str(args.freeze_output)
    else:
        verify_live_target_deployment(model, record, live)
        output = None
    print(json.dumps({"status": "verified", "model_id": args.model_id,
                      "models_identity_sha256": live["models_identity_sha256"],
                      "checked_at_utc": live["checked_at_utc"], "frozen_record": output}))


if __name__ == "__main__":
    main()
