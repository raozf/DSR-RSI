#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
exec "$KIT_PYTHON" -m ws_bridge.client_adapter --task cover_blocks "$@"
