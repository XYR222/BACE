#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT="${SCRIPT_DIR}/SHA256SUMS"
TEMP_OUTPUT="${OUTPUT}.tmp"

cd "${BUNDLE_ROOT}"
find . -type f \
    -not -path './verl-agent-src/.git/*' \
    -not -path './experiments/*' \
    -not -path './deploy/gpu/SHA256SUMS' \
    -not -path './deploy/gpu/SHA256SUMS.tmp' \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > "${TEMP_OUTPUT}"
mv "${TEMP_OUTPUT}" "${OUTPUT}"
echo "Wrote ${OUTPUT}"
