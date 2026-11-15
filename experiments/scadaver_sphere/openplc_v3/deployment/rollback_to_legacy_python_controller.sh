#!/usr/bin/env bash
set -euo pipefail

# Intended only for an OpenPLC-specific realization with the legacy assets
# present. It does not delete OpenPLC; it returns service ownership to the
# original Python controller path recorded by the legacy bootstrap.
systemctl disable --now openplc-controller.service || true
systemctl enable --now scadaver-controller.service
