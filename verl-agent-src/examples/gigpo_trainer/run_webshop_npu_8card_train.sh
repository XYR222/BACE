#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:--XX:+UseSerialGC -XX:ActiveProcessorCount=1 -Xms128m -Xmx512m -Xss512k -Djava.awt.headless=true}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
unset VLLM_ATTENTION_BACKEND LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
ulimit -u 65536

CONDA_ENV=${CONDA_ENV:-/opt/dpcvol/datasets/8165423358032568398/verl-agent-webshop}
CONDA_SH=${CONDA_SH:-/home/naie/Asend/miniconda3/etc/profile.d/conda.sh}
[[ -f "${CONDA_SH}" ]] || { echo "Conda activation script not found: ${CONDA_SH}" >&2; exit 1; }
# The packaged OpenJDK activation hook reads these variables under `set -u`.
export JAVA_HOME=${JAVA_HOME:-}
export JAVA_LD_LIBRARY_PATH=${JAVA_LD_LIBRARY_PATH:-}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export JAVA_HOME=${CONDA_PREFIX}
export JVM_PATH=${JVM_PATH:-${JAVA_HOME}/lib/server/libjvm.so}

# Accept the upstream positional engine argument while filtering job-platform-only flags.
ENGINE=${ENGINE:-vllm}
filtered_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_url=*|--data_url=*)
            shift
            ;;
        --checkpoint_url|--data_url)
            shift
            [[ $# -eq 0 ]] || shift
            ;;
        *)
            filtered_args+=("$1")
            shift
            ;;
    esac
done
if [[ ${#filtered_args[@]} -gt 0 && "${filtered_args[0]}" != *=* && "${filtered_args[0]}" != --* ]]; then
    ENGINE=${filtered_args[0]}
    filtered_args=("${filtered_args[@]:1}")
fi
hydra_overrides=("${filtered_args[@]}")

# Training defaults below preserve examples/gigpo_trainer/run_webshop.sh.
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
prepared_val_data_size=$((val_data_size * 2))
group_size=${GROUP_SIZE:-8}
mode=${GIGPO_MODE:-mean_norm}
max_steps=${ENV_MAX_STEPS:-15}
max_prompt_length=${MAX_PROMPT_LENGTH:-4096}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
total_epochs=${TOTAL_EPOCHS:-150}
test_freq=${TEST_FREQ:-5}
val_before_train=${VAL_BEFORE_TRAIN:-true}
# Checkpointing and TensorBoard are operational additions and do not change updates.
save_freq=${SAVE_FREQ:-5}
rollout_tp=${ROLLOUT_TP:-2}
n_gpus_per_node=${N_GPUS_PER_NODE:-8}
export N_GPUS_PER_NODE=${n_gpus_per_node}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-64}
ppo_micro_batch_size=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
log_prob_micro_batch_size=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
vllm_gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION:-0.6}
model_path=${MODEL_PATH:-/opt/dpcvol/datasets/8165423358032568398/model/Qwen2.5-1.5B-Instruct}
logger_backends=${LOGGER_BACKENDS:-"['console','tensorboard']"}
resume_mode=${RESUME_MODE:-disable}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.05}

exp_root=${GIGPO_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
run_name=${GIGPO_RUN_NAME:-gigpo_webshop_qwen2.5_1.5b_npu_8card_original_${run_timestamp}}
data_root=${GIGPO_DATA_ROOT:-${exp_root}/data/${run_name}}
checkpoint_dir=${CHECKPOINT_DIR:-${exp_root}/checkpoints/${run_name}}
rollout_dir=${ROLLOUT_DATA_DIR:-${exp_root}/rollout_trajectories/${run_name}}
validation_dir=${VALIDATION_DATA_DIR:-${exp_root}/validation_trajectories/${run_name}}
tensorboard_dir=${TENSORBOARD_DIR:-${exp_root}/tensorboard/${run_name}}
metadata_dir=${METADATA_DIR:-${exp_root}/run_metadata/${run_name}}
identity_report=${IDENTITY_REPORT:-${exp_root}/environment_validation/${run_name}_identity.json}
log_file=${LOG_FILE:-${exp_root}/logs/${run_name}.log}
service_url=${WEBSHOP_SERVICE_URL:-http://127.0.0.1:3000/ABC}
require_service=${REQUIRE_WEBSHOP_HTTP:-0}
mkdir -p "${data_root}" "${checkpoint_dir}" "${rollout_dir}" "${validation_dir}" \
    "${tensorboard_dir}" "${metadata_dir}" "$(dirname "${identity_report}")" "$(dirname "${log_file}")"
export TENSORBOARD_DIR=${tensorboard_dir}
exec > >(tee -a "${log_file}") 2>&1

start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf 'run_name=%q\nstart_time=%q\ngit_head=%q\n' "${run_name}" "${start_time}" "$(git rev-parse HEAD)"
    printf 'algorithm=gigpo\nenvironment=Webshop\nprotocol=use_small:true,human_goals:false,val:0:500,train:500:len(goals)\n'
    printf 'conda_env=%q\njava_home=%q\njvm_path=%q\n' "${CONDA_ENV}" "${JAVA_HOME}" "${JVM_PATH}"
    printf 'java_tool_options=%q\nomp_num_threads=%q\n' "${JAVA_TOOL_OPTIONS}" "${OMP_NUM_THREADS}"
    printf 'visible_devices=%q\nengine=%q\nmodel_path=%q\nresume_mode=%q\n' \
        "${ASCEND_RT_VISIBLE_DEVICES}" "${ENGINE}" "${model_path}" "${resume_mode}"
    printf 'webshop_service_url=%q\nrequire_webshop_http=%q\n' "${service_url}" "${require_service}"
    printf 'train_data_size=%q\nval_data_size=%q\nprepared_val_data_size=%q\ngroup_size=%q\n' \
        "${train_data_size}" "${val_data_size}" "${prepared_val_data_size}" "${group_size}"
    printf 'mode=%q\nmax_steps=%q\nmax_prompt_length=%q\nmax_response_length=%q\n' \
        "${mode}" "${max_steps}" "${max_prompt_length}" "${max_response_length}"
    printf 'total_epochs=%q\ntest_freq=%q\nsave_freq=%q\nval_before_train=%q\nrollout_tp=%q\nn_gpus_per_node=%q\n' \
        "${total_epochs}" "${test_freq}" "${save_freq}" "${val_before_train}" "${rollout_tp}" "${n_gpus_per_node}"
    printf 'vllm_gpu_memory_utilization=%q\n' "${vllm_gpu_memory_utilization}"
    git status --short
} > "${metadata_dir}/run_metadata.txt"
: > "${metadata_dir}/hydra_overrides.txt"
for override in "${hydra_overrides[@]}"; do
    printf '%q\n' "${override}" >> "${metadata_dir}/hydra_overrides.txt"
done
python -m pip freeze > "${metadata_dir}/pip_freeze.txt"
java -version 2> "${metadata_dir}/java_version.txt"
npu-smi info > "${metadata_dir}/npu_smi_before.txt"
git diff --check > "${metadata_dir}/git_diff_check.txt" || true

finalized=0
finalize_run() {
    local exit_code=$1
    if [[ ${finalized} -eq 1 ]]; then
        return
    fi
    finalized=1
    trap - EXIT INT TERM
    set +e
    npu-smi info > "${metadata_dir}/npu_smi_after.txt"
    printf 'end_time=%s\ntrain_exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" \
        >> "${metadata_dir}/run_metadata.txt"
}
trap 'status=$?; finalize_run "${status}"; exit "${status}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ -d "${model_path}" ]] || { echo "Model path not found: ${model_path}" >&2; exit 1; }
[[ -f "${JVM_PATH}" ]] || { echo "JVM library not found: ${JVM_PATH}" >&2; exit 1; }
http_status=$(curl -sS -o /dev/null -w '%{http_code}' "${service_url}" || true)
printf 'webshop_http_status=%q\n' "${http_status:-unreachable}" >> "${metadata_dir}/run_metadata.txt"
if [[ "${http_status}" != 200 ]]; then
    if [[ "${require_service}" == 1 ]]; then
        echo "WebShop HTTP service is required but unavailable: ${service_url}" >&2
        exit 1
    fi
    echo "Warning: WebShop HTTP service is unavailable; the embedded text environment remains usable." >&2
fi
python - <<'PY'
import os
import torch
import torch_npu

count = torch.npu.device_count()
print(f"torch={torch.__version__} torch_npu={torch_npu.__version__} npu_count={count}")
expected = int(os.environ["N_GPUS_PER_NODE"])
if not torch.npu.is_available() or count != expected:
    raise SystemExit(f"Expected exactly {expected} visible NPUs, found {count}")
PY
python examples/bace_gigpo/validate_webshop_real_env.py \
    --sessions 500 501 502 \
    --service-url "${service_url}" \
    --output "${identity_report}"

# The upstream script deliberately prepares 2 * val_data_size rows.
python -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${data_root}" \
    --train_data_size "${train_data_size}" \
    --val_data_size "${prepared_val_data_size}"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    algorithm.bace.enabled=false \
    data.train_files="${data_root}/text/train.parquet" \
    data.val_files="${data_root}/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size}" \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${vllm_gpu_memory_utilization}" \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_invalid_action_penalty=true \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=false \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode="${mode}" \
    env.env_name=Webshop \
    env.webshop.use_small=true \
    env.webshop.human_goals=false \
    env.webshop.search_backend=ray_shared \
    env.webshop.search_pool_size="${WEBSHOP_SEARCH_POOL_SIZE:-16}" \
    env.webshop.search_actor_num_cpus="${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}" \
    env.webshop.worker_init_batch_size="${WEBSHOP_WORKER_INIT_BATCH_SIZE:-16}" \
    env.webshop.sessions_per_actor="${WEBSHOP_SESSIONS_PER_ACTOR:-8}" \
    env.webshop.diagnostics_dir="${metadata_dir}/webshop_runtime" \
    env.seed=0 \
    env.max_steps="${max_steps}" \
    env.rollout.n="${group_size}" \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.critic_warmup=0 \
    trainer.logger="${logger_backends}" \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name="${run_name}" \
    trainer.device=npu \
    trainer.n_gpus_per_node="${n_gpus_per_node}" \
    trainer.nnodes=1 \
    trainer.save_freq="${save_freq}" \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.val_before_train="${val_before_train}" \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.rollout_data_dir="${rollout_dir}" \
    trainer.validation_data_dir="${validation_dir}" \
    trainer.resume_mode="${resume_mode}" \
    "${hydra_overrides[@]}"
