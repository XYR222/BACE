#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CHECKSUM_FILE="${SCRIPT_DIR}/SHA256SUMS"

required_paths=(
    "verl-agent-src/.git"
    "verl-agent-src/recipe/bace_gigpo"
    "verl-agent-src/tests/bace_gigpo"
    "verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_gpu.sh"
    "BACE-work"
    "BACE-work-2"
    "assets/training_data/alfworld_text_16_64/train.parquet"
    "assets/training_data/alfworld_text_16_64/test.parquet"
)

for path in "${required_paths[@]}"; do
    [[ -e "${BUNDLE_ROOT}/${path}" ]] || { echo "Missing required bundle path: ${path}" >&2; exit 1; }
done

if [[ -f "${BUNDLE_ROOT}/assets/models/Qwen2.5-1.5B-Instruct/model.safetensors" ]]; then
    echo "Optional model asset is present."
else
    echo "Optional model asset is absent; prepare it before a real GPU run."
fi
if [[ -f "${BUNDLE_ROOT}/assets/alfworld/logic/alfred.pddl" ]]; then
    echo "Optional ALFWorld asset is present."
else
    echo "Optional ALFWorld asset is absent; prepare it before a real GPU run."
fi

if [[ -f "${CHECKSUM_FILE}" ]]; then
    (cd "${BUNDLE_ROOT}" && sha256sum --check --quiet "deploy/gpu/SHA256SUMS")
else
    echo "Warning: SHA256SUMS has not been generated; only structural checks were run." >&2
fi

if find "${BUNDLE_ROOT}" -name '*.corrupt.partial' -print -quit | grep -q .; then
    echo "A corrupt partial file is present in the bundle." >&2
    exit 1
fi

echo "BACE migration bundle verification passed."
