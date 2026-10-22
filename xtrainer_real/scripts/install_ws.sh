#!/usr/bin/env bash
# Installs only inside this kit; no robot connections or SDK source changes.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
SDK_DIR="$KIT_DIR/.deps/amsongbc_dobot_xtrainer"
SDK_COMMIT="40bd10bbdbc1059f192970840975f03a179e1d21"
if [[ ! -d "$SDK_DIR/.git" ]]; then
  if [[ -e "$SDK_DIR" ]]; then
    echo "Preserving existing non-Git path: $SDK_DIR" >&2
    exit 2
  fi
  mkdir -p "$KIT_DIR/.deps"
  git clone --filter=blob:none --no-checkout https://github.com/Amsongbc/dobot_X-trainer.git "$SDK_DIR"
  git -C "$SDK_DIR" sparse-checkout set --no-cone '/README.md' '/LICENSE' '/dobot_control/' '/scripts/' '/dobot_client/' '/experiments/' '/third_party/feetech/' '/requirements.txt'
  git -C "$SDK_DIR" checkout --detach "$SDK_COMMIT"
fi
if [[ "$(git -C "$SDK_DIR" rev-parse HEAD)" != "$SDK_COMMIT" ]]; then
  echo "SDK revision mismatch; existing checkout preserved: $SDK_DIR" >&2
  exit 2
fi
if ! git -C "$SDK_DIR" diff --quiet || ! git -C "$SDK_DIR" diff --cached --quiet; then
  echo "SDK has local source edits; existing checkout preserved." >&2
  exit 2
fi
if [[ ! -x "$KIT_DIR/.venv/bin/python" ]]; then
  "${XTRAINER_BOOTSTRAP_PYTHON:-python3}" -m venv "$KIT_DIR/.venv"
fi
"$KIT_DIR/.venv/bin/python" -m pip install -r "$KIT_DIR/requirements.txt"
"$KIT_DIR/.venv/bin/python" -m ws_bridge.client_adapter --init
"$KIT_DIR/.venv/bin/python" -c 'from ws_bridge.sdk import load_sdk; load_sdk(".deps/amsongbc_dobot_xtrainer"); print("Pinned SDK imports verified; hardware untouched.")'
echo "Next: ./ws_dry_run.sh; then fill local/site.json, calibration.json, transport.json and measured scenes."
