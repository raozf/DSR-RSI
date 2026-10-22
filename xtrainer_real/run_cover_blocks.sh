#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
exec "$KIT_PYTHON" -m dsr_xtrainer.cli run --task cover_blocks "$@"
