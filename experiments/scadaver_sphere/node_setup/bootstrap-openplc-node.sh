#!/usr/bin/env bash
set -euo pipefail

# Bootstrap for the separate OpenPLC realization only. The legacy bootstrap
# remains untouched and continues to launch the Python controller.
role="$1"
repo_root="$HOME/scadaver-lab"
case "$role" in
  controller)
    sudo apt-get update
    sudo apt-get install -y git sqlite3 build-essential pkg-config libmodbus-dev python3 python3-pip
    sudo bash "$repo_root/experiments/scadaver_sphere/openplc_v3/deployment/deploy_openplc_v3_controller.sh"
    ;;
  *)
    exec "$repo_root/experiments/scadaver_sphere/node_setup/bootstrap-node.sh" "$role"
    ;;
esac
