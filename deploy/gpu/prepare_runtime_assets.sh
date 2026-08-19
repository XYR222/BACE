#!/usr/bin/env bash
set -euo pipefail

# Prepare only the large, machine-dependent runtime assets. The code package
# and tiny parquet fixtures are already included in work-BACE-3.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
ASSET_ROOT=${ASSET_ROOT:-${BUNDLE_ROOT}/assets}
MODEL_DIR=${MODEL_PATH:-${ASSET_ROOT}/models/Qwen2.5-1.5B-Instruct}
ALFWORLD_DIR=${ALFWORLD_DATA:-${ASSET_ROOT}/alfworld}
SOURCE_ASSET_ROOT=${SOURCE_ASSET_ROOT:-}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct}

mkdir -p "${ASSET_ROOT}/models" "${ASSET_ROOT}"

if [[ -n "${SOURCE_ASSET_ROOT}" ]]; then
    [[ -d "${SOURCE_ASSET_ROOT}" ]] || { echo "SOURCE_ASSET_ROOT does not exist: ${SOURCE_ASSET_ROOT}" >&2; exit 2; }
    if [[ -d "${SOURCE_ASSET_ROOT}/models/Qwen2.5-1.5B-Instruct" ]]; then
        mkdir -p "$(dirname "${MODEL_DIR}")"
        rsync -a "${SOURCE_ASSET_ROOT}/models/Qwen2.5-1.5B-Instruct/" "${MODEL_DIR}/"
    fi
    if [[ -d "${SOURCE_ASSET_ROOT}/alfworld" ]]; then
        mkdir -p "${ALFWORLD_DIR}"
        rsync -a "${SOURCE_ASSET_ROOT}/alfworld/" "${ALFWORLD_DIR}/"
    fi
fi

if [[ "${DOWNLOAD_MODEL:-0}" == 1 && ! -f "${MODEL_DIR}/model.safetensors" ]]; then
    command -v huggingface-cli >/dev/null 2>&1 || {
        echo "huggingface-cli is missing; install huggingface_hub in the active GPU environment." >&2
        exit 2
    }
    mkdir -p "${MODEL_DIR}"
    huggingface-cli download "${MODEL_REPO}" --local-dir "${MODEL_DIR}"
fi

if [[ "${DOWNLOAD_ALFWORLD:-0}" == 1 && ! -f "${ALFWORLD_DIR}/logic/alfred.pddl" ]]; then
    command -v alfworld-download >/dev/null 2>&1 || {
        echo "alfworld-download is missing; install alfworld in the active GPU environment." >&2
        exit 2
    }
    mkdir -p "${ALFWORLD_DIR}"
    ALFWORLD_DATA="${ALFWORLD_DIR}" alfworld-download -f
fi

echo "Runtime asset status:"
printf '  MODEL_PATH=%s\n' "${MODEL_DIR}"
printf '  ALFWORLD_DATA=%s\n' "${ALFWORLD_DIR}"
if [[ -f "${MODEL_DIR}/model.safetensors" && -f "${ALFWORLD_DIR}/logic/alfred.pddl" ]]; then
    echo "Both large runtime assets are ready."
else
    echo "Large runtime assets are still incomplete. Set SOURCE_ASSET_ROOT, DOWNLOAD_MODEL=1, or DOWNLOAD_ALFWORLD=1 and rerun."
    exit 1
fi
