#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$KIT_PYTHON" -m unittest discover -s "$KIT_DIR/tests" -v
"$KIT_PYTHON" - "$KIT_DIR" <<'PY'
import ast, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for directory in ["dsr_xtrainer", "tests"]:
    for path in (root/directory).rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
for path in root.glob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
for directory in ["configs", "examples"]:
    for path in (root/directory).rglob("*.json"):
        json.loads(path.read_text())
print("Python and JSON syntax checks passed")
PY
while IFS= read -r script; do
  bash -n "$script"
done < <(find "$KIT_DIR/scripts" -maxdepth 1 -name '*.sh' -print)
for script in "$KIT_DIR"/*.sh; do bash -n "$script"; done
"$KIT_DIR/dry_run.sh"
