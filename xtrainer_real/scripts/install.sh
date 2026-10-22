#!/usr/bin/env bash
# Local venv and pinned official SDK. Never connects to the robot.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
SDK_DIR="$KIT_DIR/.deps/dobot_xtrainer"
SDK_COMMIT="04cbec5f57aee6355b70c3933d84962376029f1d"
if [[ ! -d "$SDK_DIR/.git" ]]; then
  if [[ -e "$SDK_DIR" ]]; then
    echo "Existing non-Git SDK directory: $SDK_DIR; preserve it and choose a clean .deps path." >&2
    exit 2
  fi
  mkdir -p "$KIT_DIR/.deps"
  git clone https://github.com/Dobot-Arm/dobot_xtrainer.git "$SDK_DIR"
  git -C "$SDK_DIR" checkout --detach "$SDK_COMMIT"
fi
if [[ "$(git -C "$SDK_DIR" rev-parse HEAD)" != "$SDK_COMMIT" ]]; then
  echo "SDK revision mismatch. Existing checkout was preserved: $SDK_DIR" >&2
  exit 2
fi
if ! git -C "$SDK_DIR" diff --quiet || ! git -C "$SDK_DIR" diff --cached --quiet; then
  echo "SDK has local edits; installation stopped without overwriting them." >&2
  exit 2
fi
if [[ ! -x "$KIT_DIR/.venv/bin/python" ]]; then
  "${XTRAINER_BOOTSTRAP_PYTHON:-python3}" -m venv "$KIT_DIR/.venv"
fi
"$KIT_DIR/.venv/bin/python" -m pip install -r "$KIT_DIR/requirements.txt"
"$KIT_DIR/.venv/bin/python" -m dsr_xtrainer.cli init
"$KIT_DIR/.venv/bin/python" -c 'from dsr_xtrainer.config import Config; from dsr_xtrainer.hardware import load_sdk; c=Config("'"$KIT_DIR"'/examples/site.demo.json"); load_sdk(c); print("Pinned SDK imports verified; no hardware connection.")'
echo "Installed under $KIT_DIR. Next: ./dry_run.sh and edit local/site.json + local/calibration.json."
