#!/bin/sh
set -eu

# Run only as the experiment administrator on the isolated decoy VM after the
# pinned SCADAver binary has been independently validated with --help.
exec "${SCADAVER_BIN:?set SCADAVER_BIN}" run modbus-server --port 1502 --reg-value 42 --coil-value 1
