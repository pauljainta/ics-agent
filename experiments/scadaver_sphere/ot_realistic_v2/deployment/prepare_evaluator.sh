#!/usr/bin/env bash
set -euo pipefail

# Run on the private evaluator node. Actions remain disabled after installation.
readonly ROOT=/opt/scadaver-v2
readonly UNIT="$ROOT/experiments/scadaver_sphere/ot_realistic_v2/deployment/scadaver-v2-operator-broker.service"
test -f "$UNIT"
sudo install -d -m 0700 /etc/scadaver-v2 /var/lib/scadaver-v2
if ! sudo test -f /etc/scadaver-v2/operator-broker.env; then
  token=$(openssl rand -hex 32)
  printf 'SCADAVER_V2_ENABLE_ACTIONS=0\nSCADAVER_V2_ACTION_TOKEN=%s\n' "$token" \
    | sudo tee /etc/scadaver-v2/operator-broker.env >/dev/null
  sudo chmod 0600 /etc/scadaver-v2/operator-broker.env
fi
sudo install -m 0644 "$UNIT" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now scadaver-v2-operator-broker.service
echo 'Operator broker installed with state-changing actions disabled.'
