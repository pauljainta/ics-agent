#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/researcher/scadaver-lab
readonly UNIT="$ROOT/experiments/scadaver_sphere/ot_realistic_v2/deployment/scadaver-v2-decoy.service"
readonly SCADAVER=/home/researcher/.cargo/bin/scadaver
readonly WRAPPER="$ROOT/experiments/scadaver_sphere/node_setup/decoy-service.sh"

test -x "$SCADAVER"
"$SCADAVER" --help >/dev/null
test -f "$UNIT"
test -f "$WRAPPER"
chmod 0755 "$WRAPPER"
sudo install -m 0644 "$UNIT" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now scadaver-v2-decoy.service
