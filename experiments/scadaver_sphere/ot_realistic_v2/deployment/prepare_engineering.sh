#!/usr/bin/env bash
set -euo pipefail

# Run only on the NEW V2 engineering node. No process-network route is added.
readonly ROOT=/home/researcher/scadaver-lab
readonly UNIT="$ROOT/experiments/scadaver_sphere/ot_realistic_v2/deployment/scadaver-v2-engineering@.service"
test -f "$UNIT"
sudo install -m 0644 "$UNIT" /etc/systemd/system/scadaver-v2-engineering@.service
sudo install -d -o researcher -g researcher -m 0750 /var/lib/scadaver-v2
sudo systemctl daemon-reload
for service in hmi opcua s7; do
  sudo systemctl enable --now "scadaver-v2-engineering@${service}.service"
done
