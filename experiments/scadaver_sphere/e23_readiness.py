"""Read-only E2/E3 worker/network preflight; never calls a model."""
from __future__ import annotations
import argparse, json, os
from .e23_runner import preflight, profile
def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--experiment", choices=("e2","e3"), required=True); p.add_argument("--condition", choices=("telemetry","metadata"), default="metadata"); p.add_argument("--worker-url", default=os.environ.get("SCADAVER_WORKER_URL")); a=p.parse_args()
    if not a.worker_url: p.error("--worker-url or SCADAVER_WORKER_URL is required")
    print(json.dumps(preflight(a.worker_url, profile(a.experiment,a.condition)[0]), indent=2, sort_keys=True))
if __name__ == "__main__": main()
