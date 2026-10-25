"""Capture exact Qwen/vLLM runtime metadata on the serving node."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-revision", required=True,
                        help="exact Hugging Face commit SHA for cached weights")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.weight_revision) != 40 or any(c not in "0123456789abcdef"
                                               for c in args.weight_revision.lower()):
        parser.error("--weight-revision must be a 40-character commit SHA")
    config = args.model_dir / "config.json"
    if not config.is_file():
        parser.error("model directory lacks config.json")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze", "--all"],
                            check=True, capture_output=True, timeout=30).stdout
    payload = {"weight_revision": args.weight_revision,
               "model_config_sha256": sha(config.read_bytes()),
               "vllm_version": importlib.metadata.version("vllm"),
               "runtime_digest": sha(freeze), "runtime_kind": "python-environment",
               "python_version": platform.python_version(),
               "served_model": "plcbench"}
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "runtime_digest": payload["runtime_digest"]}))


if __name__ == "__main__":
    main()
