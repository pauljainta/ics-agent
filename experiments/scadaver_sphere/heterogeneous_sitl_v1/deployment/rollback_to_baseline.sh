#!/usr/bin/env bash
# Restores the pre-upgrade Python controller and leaves all preserved backups intact.
set -euo pipefail
sudo systemctl disable --now scadaver-openplc-controller.service || true
sudo systemctl enable --now scadaver-legacy-controller.service
sudo systemctl is-active --quiet scadaver-legacy-controller.service
