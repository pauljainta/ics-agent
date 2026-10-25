#!/usr/bin/env python3
"""Generate a hash-checked JSON bridge map from the canonical YAML map."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    source = args.source.read_bytes()
    data = yaml.safe_load(source)
    if not isinstance(data, dict) or not isinstance(data.get("mappings"), dict):
        parser.error("source has no mappings dictionary")
    data["source_yaml_sha256"] = hashlib.sha256(source).hexdigest()
    out = args.out or args.source.with_suffix(".json")
    out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
