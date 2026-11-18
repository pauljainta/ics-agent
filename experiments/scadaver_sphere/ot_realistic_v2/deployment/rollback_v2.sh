#!/usr/bin/env bash
set -euo pipefail

# V2-only rollback. It does not connect to or modify the original realization.
role=${1:?usage: rollback_v2.sh controller|simulator|engineering|decoy|access|evaluator}
case "$role" in
  controller)
    sudo systemctl disable --now scadaver-v2-supervisory.service || true
    sudo systemctl stop scadaver-openplc-controller.service || true
    ;;
  simulator)
    sudo systemctl disable --now scadaver-v2-simulator.service || true
    ;;
  engineering)
    for service in hmi opcua s7; do
      sudo systemctl disable --now "scadaver-v2-engineering@${service}.service" || true
    done
    ;;
  decoy)
    sudo systemctl disable --now scadaver-v2-decoy.service || true
    ;;
  access)
    sudo systemctl disable --now scadaver-v2-access-worker.service || true
    ;;
  evaluator)
    sudo systemctl disable --now scadaver-v2-operator-broker.service || true
    ;;
  *) echo "unknown role: $role" >&2; exit 2 ;;
esac
