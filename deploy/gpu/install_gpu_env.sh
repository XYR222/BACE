#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUNDLE_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
REPO_ROOT="${BUNDLE_ROOT}/verl-agent-src"
ENV_NAME=${BACE_GPU_ENV_NAME:-bace-gpu}

command -v conda >/dev/null 2>&1 || {
    echo "conda was not found. Install Miniconda/Miniforge, then rerun this script." >&2
    exit 2
}
command -v nvidia-smi >/dev/null 2>&1 || {
    echo "nvidia-smi was not found. Install a host NVIDIA driver first." >&2
    exit 2
}

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
fi

conda run -n "${ENV_NAME}" python -m pip install --upgrade pip wheel ninja packaging
conda run -n "${ENV_NAME}" python -m pip install -r "${SCRIPT_DIR}/requirements-gpu.lock.txt"
MAX_JOBS=${MAX_JOBS:-8} conda run -n "${ENV_NAME}" python -m pip install \
    flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
conda run -n "${ENV_NAME}" python -m pip install -e "${REPO_ROOT}"

echo
echo "Environment installed. Activate and check it with:"
echo "  conda activate ${ENV_NAME}"
echo "  bash ${SCRIPT_DIR}/check_gpu_environment.sh"
