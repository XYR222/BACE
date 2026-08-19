#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export JAVA_TOOL_OPTIONS=${JAVA_TOOL_OPTIONS:--XX:+UseSerialGC -XX:ActiveProcessorCount=1 -Xms128m -Xmx512m -Xss512k -Djava.awt.headless=true}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
unset VLLM_ATTENTION_BACKEND LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT

CONDA_ENV=${CONDA_ENV:-/opt/dpcvol/datasets/8165423358032568398/verl-agent-webshop}
CONDA_SH=${CONDA_SH:-/home/naie/Asend/miniconda3/etc/profile.d/conda.sh}
export JAVA_LD_LIBRARY_PATH=${JAVA_LD_LIBRARY_PATH:-}
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export JAVA_HOME=${JAVA_HOME:-${CONDA_PREFIX}}
export JVM_PATH=${JVM_PATH:-${JAVA_HOME}/lib/server/libjvm.so}

exp_root=${BACE_EXP_ROOT:-/opt/dpcvol/datasets/8165423358032568398/AESC-exp}
run_name=${BACE_RUN_NAME:-bace_webshop_npu_1card_smoke_20260808}
data_root=${BACE_DATA_ROOT:-${exp_root}/data/${run_name}}
rollout_dir="${exp_root}/rollout_trajectories/${run_name}"
checkpoint_dir="${exp_root}/checkpoints/${run_name}"
metadata_dir="${exp_root}/run_metadata/${run_name}"
validation_report="${exp_root}/trace_validation/${run_name}.json"
identity_report="${exp_root}/trace_validation/${run_name}_real_env_identity.json"
log_file="${exp_root}/logs/${run_name}.log"
mkdir -p "${data_root}" "${rollout_dir}" "${checkpoint_dir}" "${metadata_dir}" \
    "$(dirname "${validation_report}")" "$(dirname "${log_file}")"
exec > >(tee -a "${log_file}") 2>&1

train_data_size=${TRAIN_DATA_SIZE:-1}
val_data_size=${VAL_DATA_SIZE:-2}
leaf_budget=${BACE_TOTAL_LEAF_BUDGET:-3}
pilot_roots=${BACE_PILOT_ROOTS:-2}
branch_count=${BACE_BRANCH_COUNT:-1}
max_steps=${ENV_MAX_STEPS:-2}
model_path=${MODEL_PATH:-/opt/dpcvol/datasets/8165423358032568398/model/Qwen2.5-1.5B-Instruct}

start_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)
{
    printf 'run_name=%q\nstart_time=%q\ngit_head=%q\n' "${run_name}" "${start_time}" "$(git rev-parse HEAD)"
    printf 'protocol=use_small:true,human_goals:false,val:0:500,train:500:len(goals)\n'
    printf 'conda_env=%q\njava_home=%q\njvm_path=%q\n' "${CONDA_ENV}" "${JAVA_HOME}" "${JVM_PATH}"
    printf 'java_tool_options=%q\nomp_num_threads=%q\n' "${JAVA_TOOL_OPTIONS}" "${OMP_NUM_THREADS}"
    printf 'visible_devices=%q\nmodel_path=%q\n' "${ASCEND_RT_VISIBLE_DEVICES}" "${model_path}"
    printf 'train_data_size=%q\nval_data_size=%q\n' "${train_data_size}" "${val_data_size}"
    printf 'leaf_budget=%q\npilot_roots=%q\nbranch_count=%q\nmax_steps=%q\n' \
        "${leaf_budget}" "${pilot_roots}" "${branch_count}" "${max_steps}"
    git status --short
} > "${metadata_dir}/run_metadata.txt"
python -m pip freeze > "${metadata_dir}/pip_freeze.txt"
java -version 2> "${metadata_dir}/java_version.txt"
npu-smi info > "${metadata_dir}/npu_smi_before.txt"

curl -fsS http://127.0.0.1:3000/ABC > /dev/null
python examples/bace_gigpo/validate_webshop_real_env.py \
    --sessions 500 501 502 \
    --output "${identity_report}"

python -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${data_root}" \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}"

set +e
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=bace_gigpo \
    algorithm.bace.enabled=true \
    algorithm.bace.topology=dynamic \
    algorithm.bace.dynamic_root_generation=staged \
    algorithm.bace.total_leaf_budget="${leaf_budget}" \
    algorithm.bace.pilot_roots="${pilot_roots}" \
    algorithm.bace.fixed_branch_count="${branch_count}" \
    algorithm.bace.competence_threshold=0.0 \
    algorithm.bace.invalid_action_mode=strict_identity \
    algorithm.bace.artifacts.enabled=true \
    algorithm.bace.artifacts.include_token_arrays=true \
    data.train_files="${data_root}/text/train.parquet" \
    data.val_files="${data_root}/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length=4096 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=true \
    data.truncation=error \
    data.return_raw_chat=true \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size="${leaf_budget}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.enable_chunked_prefill=false \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.use_invalid_action_penalty=true \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=false \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=mean_norm \
    env.env_name=Webshop \
    env.webshop.use_small=true \
    env.webshop.human_goals=false \
    env.webshop.search_backend=ray_shared \
    env.webshop.search_pool_size="${WEBSHOP_SEARCH_POOL_SIZE:-2}" \
    env.webshop.search_actor_num_cpus="${WEBSHOP_SEARCH_ACTOR_CPUS:-0.1}" \
    env.webshop.worker_init_batch_size="${WEBSHOP_WORKER_INIT_BATCH_SIZE:-2}" \
    env.webshop.sessions_per_actor="${WEBSHOP_SESSIONS_PER_ACTOR:-2}" \
    env.webshop.diagnostics_dir="${metadata_dir}/webshop_runtime" \
    env.seed=0 \
    env.max_steps="${max_steps}" \
    env.rollout.n="${leaf_budget}" \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.logger="['console']" \
    trainer.project_name=verl_agent_webshop \
    trainer.experiment_name="${run_name}" \
    trainer.device=npu \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=false \
    trainer.default_local_dir="${checkpoint_dir}" \
    trainer.rollout_data_dir="${rollout_dir}" \
    "$@"
train_status=$?
set -e

if [[ -d "${rollout_dir}/bace_trace" ]] && find "${rollout_dir}/bace_trace" -name summary.json -print -quit | grep -q .; then
    set +e
    python -m recipe.bace_gigpo.validate_trace "${rollout_dir}/bace_trace" --output "${validation_report}"
    validation_status=$?
    set -e
else
    printf '{"ok": false, "reason": "no complete trace found"}\n' > "${validation_report}"
    validation_status=1
fi
npu-smi info > "${metadata_dir}/npu_smi_after.txt"
printf 'end_time=%s\ntrain_exit_code=%s\ntrace_validation_exit_code=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${train_status}" "${validation_status}" >> "${metadata_dir}/run_metadata.txt"
if [[ "${train_status}" -ne 0 ]]; then
    exit "${train_status}"
fi
exit "${validation_status}"
