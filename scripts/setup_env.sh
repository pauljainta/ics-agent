#!/usr/bin/env bash
set -euo pipefail

if ! command -v module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  # Makes the `module` command available in non-login shells.
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  if ! module load python/3.10.3; then
    echo "WARNING: could not load module python/3.10.3; using existing python." >&2
  fi
else
  echo "WARNING: module command unavailable; using existing python." >&2
fi

PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "${PYTHON_VERSION}" != "3.10.3" ]]; then
  echo "ERROR: expected python 3.10.3, got ${PYTHON_VERSION}." >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python -m venv .venv
fi

source .venv/bin/activate

VENV_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')"
if [[ "${VENV_VERSION}" != "3.10.3" ]]; then
  echo "ERROR: .venv is not using python 3.10.3 (found ${VENV_VERSION})." >&2
  echo "Delete .venv and rerun scripts/setup_env.sh to recreate it with python/3.10.3." >&2
  exit 1
fi

pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready with Python ${VENV_VERSION}. Reuse this single venv: source .venv/bin/activate"
