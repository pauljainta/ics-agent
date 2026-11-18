#!/usr/bin/env bash
set -euo pipefail

# Run only on the NEW V2 controller after copying the repository to /opt/scadaver-v2.
# This script deliberately does not touch the original realization.
readonly ROOT=/opt/scadaver-v2
readonly OPENPLC=/opt/openplc-v3
readonly DEPLOY="$ROOT/experiments/scadaver_sphere/ot_realistic_v2"

test -d "$OPENPLC/webserver"
test -f "$DEPLOY/openplc/water_level_controller_v2.st"
test -f "$DEPLOY/openplc/mbconfig_v2.cfg"

# A dual-homed controller must never become a router.
sudo sysctl -w net.ipv4.ip_forward=0
printf 'net.ipv4.ip_forward=0\n' | sudo tee /etc/sysctl.d/90-scadaver-v2-no-forward.conf >/dev/null
sudo iptables -P FORWARD DROP

sudo install -d -m 0750 /run/scadaver-v2
sudo install -m 0644 "$DEPLOY/deployment/scadaver-v2-supervisory.service" /etc/systemd/system/
sudo install -m 0644 "$DEPLOY/openplc/water_level_controller_v2.st" "$OPENPLC/webserver/st_files/"
sudo install -m 0644 "$DEPLOY/openplc/mbconfig_v2.cfg" "$OPENPLC/webserver/core/mbconfig.cfg"
sudo systemctl daemon-reload
sudo systemctl enable --now scadaver-v2-supervisory.service

cd "$OPENPLC/webserver"
sudo -u openplc ./scripts/compile_program.sh water_level_controller_v2.st

echo 'Prepared V2 controller files. Activate the compiled project through the supported OpenPLC management workflow before validation.'
