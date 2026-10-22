#!/usr/bin/env bash
set -euo pipefail
DSR_SOURCE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DSR_PYTHON_BIN="${DSR_PYTHON:-python3.11}"
DSR_ENV_PATH="${DSR_ENV_DIR:-$DSR_SOURCE_ROOT/.venv}"
mkdir -p "$DSR_SOURCE_ROOT/runtime/tmp"
export TMPDIR="$DSR_SOURCE_ROOT/runtime/tmp"
if [[ -n "${DSR_WHEELHOUSE:-}" ]]; then
  "$DSR_PYTHON_BIN" -m venv "$DSR_ENV_PATH"
  "$DSR_ENV_PATH/bin/python" -m pip install --no-index --find-links "$DSR_WHEELHOUSE" -r "$DSR_SOURCE_ROOT/requirements/orchestration.lock"
  "$DSR_ENV_PATH/bin/python" -m pip install --no-index --no-build-isolation --no-deps -e "$DSR_SOURCE_ROOT"
elif [[ "${DSR_OFFLINE:-0}" == "1" ]]; then
  "$DSR_PYTHON_BIN" -m venv --system-site-packages "$DSR_ENV_PATH"
  "$DSR_ENV_PATH/bin/python" -m pip install --no-index --no-build-isolation --no-deps -e "$DSR_SOURCE_ROOT"
else
  "$DSR_PYTHON_BIN" -m venv "$DSR_ENV_PATH"
  "$DSR_ENV_PATH/bin/python" -m pip install -c "$DSR_SOURCE_ROOT/requirements/orchestration.lock" -e "$DSR_SOURCE_ROOT[test]"
fi
# PhysicalRSI is loaded through dsr.paths.bootstrap from the pinned local tree.
# Its distribution is not separately installed with conflicting upstream pins.
"$DSR_ENV_PATH/bin/python" -m pip freeze > "$DSR_SOURCE_ROOT/runtime/environment.lock.txt"
"$DSR_ENV_PATH/bin/dsr" doctor
