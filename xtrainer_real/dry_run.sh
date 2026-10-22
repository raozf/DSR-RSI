#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
TASK_CHOICE="${1:-all}"
case "$TASK_CHOICE" in
  all) TASKS=(align_blocks cover_blocks) ;;
  align_blocks|cover_blocks) TASKS=("$TASK_CHOICE") ;;
  *) echo "Usage: $0 [all|align_blocks|cover_blocks]" >&2; exit 2 ;;
esac
for TASK_NAME in "${TASKS[@]}"; do
  "$KIT_PYTHON" -m dsr_xtrainer.cli run --mock --config "$KIT_DIR/examples/site.demo.json" --task "$TASK_NAME" --scene "$KIT_DIR/examples/$TASK_NAME.scene.json"
done
