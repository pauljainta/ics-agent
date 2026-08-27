#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_REPO_ROOT))

from huggingface_hub import snapshot_download  # noqa: E402

from ics_agent.experiments.model_variant_runner import MODEL_VARIANTS, _model_variant  # noqa: E402
from ics_agent.paths import portable_path  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download local Hugging Face models used by model-variant runners."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=f"Model key or alias. Known: {', '.join(sorted(MODEL_VARIANTS))}",
    )
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Optional local destination. Defaults to the repo-local model path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved download target without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model = _model_variant(args.model)
    local_dir = Path(args.local_dir).expanduser().resolve() if args.local_dir else model.local_dir
    payload = {
        "model_key": model.key,
        "model_label": model.label,
        "model_id": model.model_id,
        "local_dir": portable_path(local_dir),
        "quantization_for_runner": model.quantization,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model.model_id,
            local_dir=local_dir,
        )
        payload["status"] = "downloaded"
    else:
        payload["status"] = "resolved"
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
