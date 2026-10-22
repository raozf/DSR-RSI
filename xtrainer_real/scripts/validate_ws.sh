#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
# The two SDKs have the same Python package name; always test in separate processes.
"$KIT_PYTHON" -m unittest discover -s "$KIT_DIR/ws_bridge/tests" -p 'test*.py' -v
"$KIT_PYTHON" - "$KIT_DIR" <<'PY'
import ast, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for directory in ['ws_bridge']:
    for path in (root/directory).rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
for directory in ['configs']:
    for path in (root/directory).glob('*.json'):
        json.loads(path.read_text())
print('Adapter Python and JSON syntax passed')
PY
for script in "$KIT_DIR"/run_ws*.sh "$KIT_DIR"/ws_dry_run.sh "$KIT_DIR"/scripts/*_ws.sh; do
  bash -n "$script"
done
"$KIT_DIR/ws_dry_run.sh"
