#!/usr/bin/env bash
# Operator-only selector.  The agent worker has no path to this script.
set -euo pipefail

profile=${1:?usage: activate_controller_profile.sh legacy_python_controller|openplc_controller}
case "$profile" in
  legacy_python_controller)
    sudo systemctl stop scadaver-openplc-controller.service || true
    sudo systemctl start scadaver-legacy-controller.service
    ;;
  openplc_controller)
    sudo systemctl stop scadaver-legacy-controller.service
    sudo systemctl start scadaver-openplc-controller.service
    ;;
  *) echo "unknown controller profile: $profile" >&2; exit 2 ;;
esac

active=$(systemctl is-active "scadaver-${profile%%_*}-controller.service" 2>/dev/null || true)
if [[ "$active" != active ]]; then
  echo "requested controller did not become active" >&2
  exit 1
fi
if systemctl is-active --quiet scadaver-legacy-controller.service && systemctl is-active --quiet scadaver-openplc-controller.service; then
  echo "controller conflict detected" >&2
  exit 1
fi
