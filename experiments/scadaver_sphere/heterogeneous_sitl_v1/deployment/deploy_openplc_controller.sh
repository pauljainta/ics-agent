#!/usr/bin/env bash
# Install the OpenPLC v3 web runtime before replacing the legacy controller.
# Program activation deliberately remains in OpenPLC's supported upload/compile
# workflow; compiling a copied ST file bypasses the Programs database record.
set -euo pipefail
root=/home/researcher/scadaver-lab
bundle="$root/experiments/scadaver_sphere"
runtime=/opt/openplc-v3

if [[ ! -d "$runtime" ]]; then
  sudo git clone https://github.com/thiagoralves/OpenPLC_v3.git "$runtime"
  sudo git -C "$runtime" checkout b5d41356dab4aeadca0dd7ca64ba542f870b595d
  sudo chown -R "$USER":"$USER" "$runtime"
  (cd "$runtime" && yes | sudo ./install.sh linux)
  # The web service runs as this user and must be able to regenerate the
  # compiler-produced glueVars.cpp during each supported program compilation.
  sudo chown -R "$USER":"$USER" "$runtime"
fi

sudo install -m 0644 "$bundle/heterogeneous_sitl_v1/deployment/scadaver-legacy-controller.service" /etc/systemd/system/scadaver-legacy-controller.service
sudo install -m 0644 "$bundle/heterogeneous_sitl_v1/deployment/scadaver-openplc-controller.service" /etc/systemd/system/scadaver-openplc-controller.service
sudo systemctl daemon-reload
sudo systemctl enable --now scadaver-openplc-controller.service

cat <<EOF
OpenPLC web service is available.  Do not stop the legacy controller yet.

Use the supported OpenPLC workflow to activate the IEC program:
  1. Sign in to the local OpenPLC web UI on port 8080.
  2. Upload $bundle/openplc_v3/program/water_level_controller.st.
  3. Add the uploaded file as a Program, select it, and compile it in the UI.
  4. Configure one Modbus TCP slave at 198.18.4.20:1503, unit 1:
       coils read/write: start 0, size 1
       holding-register read: start 0, size 1
  5. Start the PLC from the UI and verify the OpenPLC core process and port 1502.
  6. Only after a passing control-loop trace, stop the legacy controller.

The UI creates the active Programs record and regenerates the I/O glue file.
EOF
