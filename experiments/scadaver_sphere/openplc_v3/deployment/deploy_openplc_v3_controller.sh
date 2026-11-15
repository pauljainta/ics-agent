#!/usr/bin/env bash
set -euo pipefail

# This script is for a NEW OpenPLC realization/controller node.  It must not be
# applied to the legacy Python-controller realization.  The checked revision is
# recorded here and re-verified before installation.
readonly REPO=https://github.com/thiagoralves/OpenPLC_v3.git
readonly REVISION=b5d41356dab4aeadca0dd7ca64ba542f870b595d
readonly ROOT=/opt/openplc-v3
readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

id -u openplc >/dev/null 2>&1 || useradd --system --home "$ROOT" --shell /usr/sbin/nologin openplc
test ! -e "$ROOT" || { echo "refusing to overwrite existing $ROOT" >&2; exit 1; }
git clone "$REPO" "$ROOT"
git -C "$ROOT" checkout --detach "$REVISION"
test "$(git -C "$ROOT" rev-parse HEAD)" = "$REVISION"
cd "$ROOT"
./install.sh linux

install -D -m 0644 "$HERE/../program/water_level_controller.st" "$ROOT/webserver/st_files/water_level_controller.st"
install -D -m 0644 "$HERE/mbconfig.cfg" "$ROOT/webserver/core/mbconfig.cfg"
cd "$ROOT/webserver"
./scripts/compile_program.sh water_level_controller.st

# Preserve the legacy external Modbus port while enabling the OpenPLC server.
sqlite3 openplc.db "UPDATE Settings SET Value='1502' WHERE Key='Modbus_port';"
install -D -m 0644 "$HERE/openplc-controller.service" /etc/systemd/system/openplc-controller.service
chown -R openplc:openplc "$ROOT"
systemctl daemon-reload
systemctl enable --now openplc-controller.service
