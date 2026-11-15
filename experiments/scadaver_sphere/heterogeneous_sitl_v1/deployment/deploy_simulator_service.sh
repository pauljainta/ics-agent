#!/usr/bin/env bash
# Must be run by an operator on 198.18.4.20 after the bundle is copied there.
set -euo pipefail
root=/home/researcher/scadaver-lab
unit="$root/experiments/scadaver_sphere/heterogeneous_sitl_v1/deployment/scadaver-simulator.service"
sudo install -m 0644 "$unit" /etc/systemd/system/scadaver-simulator.service
sudo systemctl daemon-reload
# Stop only old simulator processes matching this exact program and role.
pkill -f 'plant_runtime.py simulator' || true
sudo systemctl enable --now scadaver-simulator.service
