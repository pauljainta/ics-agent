#!/usr/bin/env bash
set -euo pipefail

role="$1"
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git curl build-essential pkg-config libssl-dev
python3 -m pip install --user 'pymodbus==3.6.9'
mkdir -p "$HOME/scadaver-lab"

if [[ "$role" == "access" || "$role" == "decoy" ]]; then
  if [[ ! -x "$HOME/.cargo/bin/scadaver" ]]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    export PATH="$HOME/.cargo/bin:$PATH"
    git clone https://github.com/Whispergate/SCADAVER.git "$HOME/scadaver-src"
    git -C "$HOME/scadaver-src" checkout 7f482a4a1d4c86f9cb766337d946a32637bb6185
    cargo install --path "$HOME/scadaver-src" --locked
  fi
fi

case "$role" in
  controller) nohup python3 "$HOME/scadaver-lab/experiments/scadaver_sphere/node_setup/plant_runtime.py" controller >/tmp/controller.log 2>&1 & ;;
  simulator) nohup python3 "$HOME/scadaver-lab/experiments/scadaver_sphere/node_setup/plant_runtime.py" simulator >/tmp/simulator.log 2>&1 & ;;
  engineering) nohup python3 -m http.server 8080 --bind 0.0.0.0 >/tmp/engineering.log 2>&1 & ;;
  decoy) nohup "$HOME/.cargo/bin/scadaver" run modbus-server --port 1502 --reg-value 42 --coil-value 1 >/tmp/decoy.log 2>&1 & ;;
  access) nohup python3 "$HOME/scadaver-lab/experiments/scadaver_sphere/node_setup/access_worker_service.py" >/tmp/access-worker.log 2>&1 & ;;
  *) echo "unknown role" >&2; exit 2 ;;
esac
