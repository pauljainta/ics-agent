#!/usr/bin/env bash
set -euo pipefail

readonly ROOT=/home/researcher/scadaver-lab
readonly UNIT="$ROOT/experiments/scadaver_sphere/ot_realistic_v2/deployment/scadaver-v2-access-worker.service"
test -f "$UNIT"
python3 -m pip install --user --disable-pip-version-check \
  'asyncua==1.1.5' 'python-snap7==2.0.2'
sudo install -m 0644 "$UNIT" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now scadaver-v2-access-worker.service
