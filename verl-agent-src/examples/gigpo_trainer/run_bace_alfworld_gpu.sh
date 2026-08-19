#!/usr/bin/env bash
set -euo pipefail

# Generic CUDA launcher for BACE-GiGPO on 1/2/4/8 NVIDIA GPUs.
# Activate the environment created by deploy/gpu/install_gpu_env.sh first.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUNDLE_ROOT=$(cd "${REPO_ROOT}/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${BUNDLE_ROOT}/assets/alfworld}

# Prevent inherited accelerator settings from selecting the wrong runtime or
# conflicting with the CUDA device mask injected by Slurm/Ray.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset ASCEND_RT_VISIBLE_DEVICES VLLM_ASCEND_ENABLE_NZ HCCL_OP_EXPANSION_MODE
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

if [[ -z "${GPU_COUNT:-}" ]]; then
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
        GPU_COUNT=${#visible_gpu_ids[@]}
    elif command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    else
        echo "Cannot determine GPU_COUNT: set GPU_COUNT and CUDA_VISIBLE_DEVICES." >&2
        exit 2
    fi
fi
[[ "${GPU_COUNT}" =~ ^[1-9][0-9]*$ ]] || { echo "GPU_COUNT must be a positive integer." >&2; exit 2; }

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
    export CUDA_VISIBLE_DEVICES
fi
IFS=',' read -r -a visible_gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#visible_gpu_ids[@]} -ne ${GPU_COUNT} ]]; then
    echo "GPU_COUNT=${GPU_COUNT}, but CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}." >&2
    exit 2
fi

ENGINE=${ENGINE:-vllm}
filtered_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint_url=*|--data_url=*) shift ;;
        --checkpoint_url|--data_url)
            shift
            [[ $# -eq 0 ]] || shift
            ;;
        *) filtered_args+=("$1"); shift ;;
    esac
done
if [[ ${#filtered_args[@]} -gt 0 && "${filtered_args[0]}" != *=* && "${filtered_args[0]}" != --* ]]; then
    ENGINE=${filtered_args[0]}
    filtered_args=("${filtered_args[@]:1}")
fi
hydra_overrides=("${filtered_args[@]}")

MODEL_PATH=${MODEL_PATH:-${BUNDLE_ROOT}/assets/models/Qwen2.5-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-${BUNDLE_ROOT}/assets/training_data/alfworld_text_16_64/train.parquet}
VAL_FILE=${VAL_FILE:-${BUNDLE_ROOT}/assets/training_data/alfworld_text_16_64/test.parquet}
BACE_EXP_ROOT=${BACE_EXP_ROOT:-${BUNDLE_ROOT}/experiments}
run_timestamp=$(date -u +%Y%m%dT%H%M%SZ)
BACE_RUN_NAME=${BACE_RUN_NAME:-bace_alfworld_h100_${GPU_COUNT}gpu_exact_${run_timestamp}}

train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-64}
group_size=${GROUP_SIZE:-8}
total_epochs=${BACE_TOTAL_EPOCHS:-150}
max_steps=${ENV_MAX_STEPS:-40}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
rollout_tp=${ROLLOUT_TP:-1}
actor_micro_batch=${ACTOR_MICRO_BATCH:-8}
logprob_micro_batch=${LOGPROB_MICRO_BATCH:-8}
ppo_mini_batch=${PPO_MINI_BATCH:-128}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.60}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-16384}
max_num_seqs=${MAX_NUM_SEQS:-128}
save_freq=${SAVE_FREQ:-30}
test_freq=${TEST_FREQ:-10}
resume_mode=${BACE_RESUME_MODE:-disable}
logger_backends=${LOGGER_BACKENDS:-"['console','tensorboard']"}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}

if (( rollout_tp > GPU_COUNT || GPU_COUNT % rollout_tp != 0 )); then
    echo "ROLLOUT_TP=${rollout_tp} must divide GPU_COUNT=${GPU_COUNT}." >&2
    exit 2
fi
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for required_path in "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}"; do
        [[ -e "${required_path}" ]] || { echo "Required path missing: ${required_path}" >&2; exit 2; }
    done
fi

checkpoint_dir=${CHECKPOINT_DIR:-${BACE_EXP_ROOT}/checkpoints/${BACE_RUN_NAME}}
rollout_data_dir=${ROLLOUT_DATA_DIR:-${BACE_EXP_ROOT}/rollout_trajectories/${BACE_RUN_NAME}}
artifact_dir=${BACE_ARTIFACT_DIR:-${BACE_EXP_ROOT}/bace_artifacts/${BACE_RUN_NAME}}
tensorboard_dir=${TENSORBOARD_DIR:-${BACE_EXP_ROOT}/tensorboard/${BACE_RUN_NAME}}
metadata_dir=${RUN_METADATA_DIR:-${BACE_EXP_ROOT}/run_metadata/${BACE_RUN_NAME}}
anchor_dump_dir=${ASEC_ANCHOR_DUMP_DIR:-${BACE_EXP_ROOT}/anchor_groups/${BACE_RUN_NAME}}
validation_report=${TRACE_VALIDATION_REPORT:-${BACE_EXP_ROOT}/trace_validation/${BACE_RUN_NAME}.json}
log_file=${LOG_FILE:-${BACE_EXP_ROOT}/logs/${BACE_RUN_NAME}.log}
export TENSORBOARD_DIR="${tensorboard_dir}"
export ASEC_ANCHOR_DUMP_DIR="${anchor_dump_dir}"

mkdir -p "${checkpoint_dir}" "${rollout_data_dir}" "${artifact_dir}" \
    "${tensorboard_dir}" "${metadata_dir}" "${anchor_dump_dir}" \
    "$(dirname "${validation_report}")" "$(dirname "${log_file}")"
exec > >(tee -a "${log_file}") 2>&1
exec 19>> "${metadata_dir}/shell_trace.log"
BASH_XTRACEFD=19
PS4='+ ${BASH_SOURCE}:${LINENO}: '

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '{"ok": true, "skipped": "DRY_RUN"}\n' > "${metadata_dir}/preflight.json"
else
    python3 "${BUNDLE_ROOT}/deploy/gpu/check_gpu_environment.py" \
        --expected-gpus "${GPU_COUNT}" --model-path "${MODEL_PATH}" \
        --alfworld-data "${ALFWORLD_DATA}" --output "${metadata_dir}/preflight.json"
fi

{
    git_head=$(git rev-parse HEAD 2>/dev/null || printf 'unavailable')
    printf 'run_name=%q\nstart_time=%q\nrepo_root=%q\nbundle_root=%q\n' \
        "${BACE_RUN_NAME}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${REPO_ROOT}" "${BUNDLE_ROOT}"
    printf 'git_head=%q\nengine=%q\nvisible_devices=%q\ngpu_count=%q\n' \
        "${git_head}" "${ENGINE}" "${CUDA_VISIBLE_DEVICES}" "${GPU_COUNT}"
    printf 'model_path=%q\nalfworld_data=%q\ntrain_file=%q\nval_file=%q\n' \
        "${MODEL_PATH}" "${ALFWORLD_DATA}" "${TRAIN_FILE}" "${VAL_FILE}"
    printf 'variant=batch_erv_exact topology=dynamic generation=staged batching=packed acquisition=batch_erv_exact invalid_action_mode=strict_identity\n'
    env | sort
    git status --short 2>/dev/null || true
} > "${metadata_dir}/run_metadata.txt"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --binary > "${metadata_dir}/working_tree.patch" || true
    git diff --cached --binary > "${metadata_dir}/index.patch" || true
else
    : > "${metadata_dir}/working_tree.patch"
    : > "${metadata_dir}/index.patch"
fi
python3 -m pip freeze > "${metadata_dir}/pip_freeze.txt"
(
    cd "${REPO_ROOT}"
    find . -type f \
        \( -name '*.py' -o -name '*.yaml' -o -name '*.yml' -o -name '*.sh' \) \
        -not -path './.git/*' \
        -not -path './.pytest_cache/*' \
        -not -path './outputs/*' \
        -not -path '*/__pycache__/*' \
        -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "${metadata_dir}/source_manifest.sha256"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    nvidia-smi -q > "${metadata_dir}/nvidia_smi_before.txt"
    nvidia-smi topo -m > "${metadata_dir}/gpu_topology.txt" || true
fi
: > "${metadata_dir}/hydra_overrides.txt"
for override in "${hydra_overrides[@]}"; do printf '%q\n' "${override}" >> "${metadata_dir}/hydra_overrides.txt"; done

monitor_gpu() {
    printf 'timestamp,index,utilization_gpu,memory_used_mib,memory_total_mib,power_w,temperature_c\n'
    while true; do
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
            --format=csv,noheader,nounits || true
        sleep "${GPU_SAMPLE_INTERVAL:-10}"
    done
}
monitor_host_resources() {
    local cgroup_path cgroup_dir pids_current pids_limit memory_current pids_source
    cgroup_path=$(awk -F: '$1 == "0" {print $3}' /proc/self/cgroup 2>/dev/null)
    cgroup_dir=/sys/fs/cgroup${cgroup_path}
    printf 'timestamp,pids_current,pids_limit,memory_current_bytes,pids_source\n'
    while true; do
        pids_current=unavailable
        pids_limit=unavailable
        memory_current=unavailable
        pids_source=unavailable
        if [[ -r "${cgroup_dir}/pids.current" ]]; then
            pids_current=$(<"${cgroup_dir}/pids.current")
            pids_source=pids.current
        elif [[ -r "${cgroup_dir}/cgroup.threads" ]]; then
            # Some Slurm cgroup-v2 deployments account memory for the job but
            # do not enable the pids controller in the leaf cgroup.  Counting
            # thread IDs is the compatible local measure of task pressure.
            pids_current=$(wc -l < "${cgroup_dir}/cgroup.threads")
            pids_current=${pids_current//[[:space:]]/}
            pids_source=cgroup.threads
        fi
        [[ -r "${cgroup_dir}/pids.max" ]] && pids_limit=$(<"${cgroup_dir}/pids.max")
        [[ -r "${cgroup_dir}/memory.current" ]] && memory_current=$(<"${cgroup_dir}/memory.current")
        printf '%s,%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            "${pids_current}" "${pids_limit}" "${memory_current}" "${pids_source}"
        sleep "${GPU_SAMPLE_INTERVAL:-10}"
    done
}
gpu_monitor_pid=""
resource_monitor_pid=""
if [[ "${DRY_RUN:-0}" != 1 ]]; then
    monitor_gpu > "${metadata_dir}/gpu_samples.csv" 2>&1 &
    gpu_monitor_pid=$!
    monitor_host_resources > "${metadata_dir}/host_resource_samples.csv" 2>&1 &
    resource_monitor_pid=$!
fi

run_finalized=0
finalize_run() {
    local exit_code=$1
    if [[ ${run_finalized} -eq 1 ]]; then return; fi
    run_finalized=1
    trap - EXIT INT TERM
    set +e
    if [[ -n "${gpu_monitor_pid}" ]]; then
        kill "${gpu_monitor_pid}" 2>/dev/null
        wait "${gpu_monitor_pid}" 2>/dev/null
    fi
    if [[ -n "${resource_monitor_pid}" ]]; then
        kill "${resource_monitor_pid}" 2>/dev/null
        wait "${resource_monitor_pid}" 2>/dev/null
    fi
    if [[ -d "${artifact_dir}" ]] && find "${artifact_dir}" -name summary.json -print -quit | grep -q .; then
        python3 -m recipe.bace_gigpo.validate_trace "${artifact_dir}" --output "${validation_report}"
        printf 'trace_validation_exit_code=%s\n' "$?" >> "${metadata_dir}/run_metadata.txt"
    elif [[ -d "${rollout_data_dir}/bace_trace" ]] && find "${rollout_data_dir}/bace_trace" -name summary.json -print -quit | grep -q .; then
        python3 -m recipe.bace_gigpo.validate_trace "${rollout_data_dir}/bace_trace" --output "${validation_report}"
        printf 'trace_validation_exit_code=%s\n' "$?" >> "${metadata_dir}/run_metadata.txt"
    else
        printf '{"ok": false, "reason": "no complete trace found"}\n' > "${validation_report}"
    fi
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
        nvidia-smi -q > "${metadata_dir}/nvidia_smi_after.txt" 2>&1
    fi
    printf 'end_time=%s\ntrain_exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${exit_code}" >> "${metadata_dir}/run_metadata.txt"
}
trap 'status=$?; finalize_run "${status}"; exit "${status}"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ppo_command=(python3 -m verl.trainer.main_ppo
    algorithm.adv_estimator=bace_gigpo
    algorithm.bace.enabled=true
    algorithm.bace.variant=batch_erv_exact
    algorithm.bace.topology=dynamic
    algorithm.bace.dynamic_root_generation=staged
    algorithm.bace.staged_root_batching=packed
    algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.total_leaf_budget=8
    algorithm.bace.min_natural_roots=2
    algorithm.bace.competence_threshold=0.5
    algorithm.bace.history_initial_mean=0.10
    algorithm.bace.history_initial_strength=2.0
    algorithm.bace.history_base_alpha=1.0
    algorithm.bace.history_base_beta=1.0
    algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1
    algorithm.bace.history_min_strength=2.0
    algorithm.bace.history_max_strength=8.0
    algorithm.bace.local_prior_strength=2.0
    algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.batch_erv_threshold=0.005
    algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12
    algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity
    algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.artifacts.enabled=true
    algorithm.bace.artifacts.directory="${artifact_dir}"
    algorithm.bace.artifacts.include_token_arrays=true
    algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true
    algorithm.bace.replay.max_origin_retries=1
    algorithm.gigpo.step_advantage_w=1.0
    algorithm.gigpo.mode=mean_std_norm
    algorithm.gamma=0.95
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size="${train_data_size}"
    data.val_batch_size="${val_data_size}"
    data.max_prompt_length=2048
    data.max_response_length="${max_response_length}"
    data.filter_overlong_prompts=true
    data.truncation=error
    data.return_raw_chat=true
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=true
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch}"
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${actor_micro_batch}"
    actor_rollout_ref.actor.use_kl_loss=true
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=false
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${logprob_micro_batch}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp}"
    actor_rollout_ref.rollout.name="${ENGINE}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}"
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}"
    actor_rollout_ref.rollout.max_num_seqs="${max_num_seqs}"
    actor_rollout_ref.rollout.enable_chunked_prefill=false
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.free_cache_engine=false
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=true
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${logprob_micro_batch}"
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.actor.use_invalid_action_penalty=true
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
    algorithm.use_kl_in_reward=false
    env.env_name=alfworld/AlfredTWEnv
    env.seed=0
    env.max_steps="${max_steps}"
    env.rollout.n="${group_size}"
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}"
    trainer.critic_warmup=0
    trainer.logger="${logger_backends}"
    trainer.project_name=verl_agent_alfworld
    trainer.experiment_name="${BACE_RUN_NAME}"
    trainer.device=cuda
    trainer.n_gpus_per_node="${GPU_COUNT}"
    trainer.nnodes=1
    trainer.save_freq="${save_freq}"
    trainer.default_local_dir="${checkpoint_dir}"
    trainer.rollout_data_dir="${rollout_data_dir}"
    trainer.test_freq="${test_freq}"
    trainer.total_epochs="${total_epochs}"
    trainer.val_before_train=false
    trainer.resume_mode="${resume_mode}"
    "${hydra_overrides[@]}")

printf '%q ' "${ppo_command[@]}" > "${metadata_dir}/resolved_command.sh"
printf '\n' >> "${metadata_dir}/resolved_command.sh"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: configuration validated; training was not started."
    exit 0
fi

set +e
"${ppo_command[@]}"
train_status=$?
set -e
exit "${train_status}"
