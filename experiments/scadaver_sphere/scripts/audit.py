"""Record local prerequisites without contacting an experiment target."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("scadaver_audit.json")
    result = {"scadaver_path": shutil.which("scadaver"), "mrg_path": shutil.which("mrg")}
    if result["mrg_path"]:
        completed = subprocess.run(["mrg", "whoami", "--json"], text=True, capture_output=True, check=False, timeout=10)
        result["mrg_whoami_exit"] = completed.returncode
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
