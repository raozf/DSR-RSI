#!/usr/bin/env bash
# Sourced by launchers. Does not change the caller's working directory.
set -euo pipefail
KIT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${XTRAINER_PYTHON:-}" ]]; then
  KIT_PYTHON="$XTRAINER_PYTHON"
elif [[ -x "$KIT_DIR/.venv/bin/python" ]]; then
  KIT_PYTHON="$KIT_DIR/.venv/bin/python"
else
  KIT_PYTHON="python3"
fi
export PYTHONPATH="$KIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
