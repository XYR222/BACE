#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GPU_COUNT=4
exec bash "${SCRIPT_DIR}/run_alfworld_h100.sh" "$@"
