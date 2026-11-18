#!/usr/bin/env bash
set -euo pipefail

# Run only on the NEW private simulator node.
readonly ROOT=/home/researcher/scadaver-lab
readonly UNIT="$ROOT/experiments/scadaver_sphere/ot_realistic_v2/deployment/scadaver-v2-simulator.service"
test -f "$UNIT"
sudo install -m 0644 "$UNIT" /etc/systemd/system/

# Defense in depth: only controller and evaluator can access private Modbus.
sudo iptables -C INPUT -p tcp --dport 1503 -s 198.18.6.1 -j ACCEPT 2>/dev/null \
  || sudo iptables -A INPUT -p tcp --dport 1503 -s 198.18.6.1 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 1503 -s 198.18.6.3 -j ACCEPT 2>/dev/null \
  || sudo iptables -A INPUT -p tcp --dport 1503 -s 198.18.6.3 -j ACCEPT
sudo iptables -C INPUT -p tcp --dport 1503 -j DROP 2>/dev/null \
  || sudo iptables -A INPUT -p tcp --dport 1503 -j DROP
sudo systemctl daemon-reload
sudo systemctl enable --now scadaver-v2-simulator.service
