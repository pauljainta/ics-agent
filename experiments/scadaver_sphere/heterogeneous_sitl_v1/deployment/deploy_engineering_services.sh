#!/usr/bin/env bash
# Must be run by an operator on 198.18.4.40 after the bundle is copied there.
set -euo pipefail
root=/home/researcher/scadaver-lab
unit_dir="$root/experiments/scadaver_sphere/heterogeneous_sitl_v1/deployment"

python3 -m pip install --user --upgrade asyncua python-snap7
mkdir -p /home/researcher/scadaver-state
for unit in scadaver-engineering.service scadaver-opcua.service scadaver-s7.service; do
  sudo install -m 0644 "$unit_dir/$unit" "/etc/systemd/system/$unit"
done
sudo systemctl daemon-reload
# Replace only the legacy temporary Python HTTP process; it is not a controller.
pkill -f 'python3 -m http.server 8080 --bind 0.0.0.0' || true
sudo systemctl enable --now scadaver-engineering.service scadaver-opcua.service scadaver-s7.service
