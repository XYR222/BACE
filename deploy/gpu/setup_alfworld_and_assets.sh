#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT="${BUNDLE_ROOT}/verl-agent-src"

SOURCE_ENV=${SOURCE_ENV:-verl-agent}
TARGET_ENV=${TARGET_ENV:-verl-agent-alfworld}
ASSET_ROOT=${ASSET_ROOT:-${BUNDLE_ROOT}/assets}
ALFWORLD_DATA=${ALFWORLD_DATA:-${ASSET_ROOT}/alfworld}
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct}
MODEL_PATH=${MODEL_PATH:-${ASSET_ROOT}/models/Qwen2.5-1.5B-Instruct}

command -v conda >/dev/null 2>&1 || {
    echo "conda was not found in PATH." >&2
    exit 2
}

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${SOURCE_ENV}"; then
    echo "Source environment does not exist: ${SOURCE_ENV}" >&2
    exit 2
fi

if conda env list | awk '{print $1}' | grep -Fxq "${TARGET_ENV}"; then
    echo "Using existing environment: ${TARGET_ENV}"
else
    echo "Cloning ${SOURCE_ENV} to ${TARGET_ENV} ..."
    conda create \
        --name "${TARGET_ENV}" \
        --clone "${SOURCE_ENV}" \
        --channel conda-forge \
        --override-channels \
        --yes
fi

conda activate "${TARGET_ENV}"

echo "Installing ALFWorld dependencies ..."
python -m pip install \
    gymnasium==0.29.1 \
    stable-baselines3==2.6.0 \
    alfworld==0.4.2 \
    huggingface_hub \
    tensorboard

echo "Installing the local verl-agent worktree in editable mode ..."
python -m pip install -e "${REPO_ROOT}"

mkdir -p "${ALFWORLD_DATA}" "${MODEL_PATH}"

if [[ -f "${ALFWORLD_DATA}/logic/alfred.pddl" ]]; then
    echo "ALFWorld data already exists; skipping download."
else
    echo "Downloading ALFWorld data to ${ALFWORLD_DATA} ..."
    ALFWORLD_DATA="${ALFWORLD_DATA}" alfworld-download -f
fi

if [[ -f "${MODEL_PATH}/config.json" ]] && compgen -G "${MODEL_PATH}/*.safetensors" >/dev/null; then
    echo "Model files already exist; skipping download."
else
    echo "Downloading ${MODEL_REPO} to ${MODEL_PATH} ..."
    hf download "${MODEL_REPO}" --local-dir "${MODEL_PATH}"
fi

echo "Validating environment and assets ..."
python - <<PY
from pathlib import Path

import alfworld
import flash_attn
import gymnasium
import stable_baselines3
import torch
import vllm
from transformers import AutoConfig, AutoTokenizer

alfworld_data = Path(r"${ALFWORLD_DATA}")
model_path = Path(r"${MODEL_PATH}")

assert (alfworld_data / "logic" / "alfred.pddl").is_file(), "ALFWorld PDDL is missing"
assert (model_path / "config.json").is_file(), "Model config.json is missing"
assert list(model_path.glob("*.safetensors")), "Model safetensors are missing"

config = AutoConfig.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

print(f"torch={torch.__version__}")
print(f"vllm={vllm.__version__}")
print(f"flash_attn={flash_attn.__version__}")
print(f"gymnasium={gymnasium.__version__}")
print(f"stable_baselines3={stable_baselines3.__version__}")
print(f"model_type={config.model_type}")
print(f"tokenizer_size={len(tokenizer)}")
print(f"ALFWORLD_DATA={alfworld_data}")
print(f"MODEL_PATH={model_path}")
PY

echo
echo "ALFWorld environment and runtime assets are ready."
echo "Activate with: conda activate ${TARGET_ENV}"
