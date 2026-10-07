"""Read-only validation of CTF bindings against a sphere-usecases checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .operator_handoffs import validate_contract_tags
from .paths import REPO_ROOT
from .utils import load_yaml


MAP_DIR = REPO_ROOT / "configs" / "ctf" / "operator_handoffs"


def check(source_root: Path) -> list[dict[str, object]]:
    source_root = source_root.resolve()
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    results = []
    for map_path in sorted(MAP_DIR.glob("*.yaml")):
        mapping = load_yaml(map_path)
        source = mapping["source"]
        contract_path = source_root / source["contract"]
        reference_path = source_root / source["workflow_reference"]
        errors = []
        if revision != source["revision"]:
            errors.append("source revision differs from binding; review before relying on tag validation")
        if not contract_path.is_file():
            errors.append(f"missing contract: {source['contract']}")
        else:
            errors.extend(validate_contract_tags(mapping, load_yaml(contract_path)))
        if not reference_path.is_file():
            errors.append(f"missing workflow reference: {source['workflow_reference']}")
        results.append(
            {"utility": mapping["utility"], "source_revision": revision,
             "binding_revision": source["revision"], "errors": errors}
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sphere_usecases", type=Path)
    args = parser.parse_args()
    results = check(args.sphere_usecases)
    print(json.dumps(results, indent=2))
    return 1 if any(item["errors"] for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
