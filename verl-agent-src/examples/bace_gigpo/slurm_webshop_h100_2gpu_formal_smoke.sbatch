#!/usr/bin/env bash
# Full 150-step Exact-Batch-ERV BACE WebShop-1k run on 2 x H100.
# It is submitted only with an afterok dependency from the smoke gate.
#SBATCH --job-name=bace_ws_2g_formal
#SBATCH --partition=c23g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
#SBATCH --time=07:00:00
#SBATCH --chdir=/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
#SBATCH --output=/hpcwork/xsz96350/fu_project/work-BACE/experiments/webshop1k-qwen2.5-1.5b-exact/slurm/%x-%j.out
#SBATCH --error=/hpcwork/xsz96350/fu_project/work-BACE/experiments/webshop1k-qwen2.5-1.5b-exact/slurm/%x-%j.err

set -euo pipefail

WORKSPACE_ROOT=/hpcwork/xsz96350/fu_project
REPO_ROOT=${WORKSPACE_ROOT}/work-BACE/verl-agent-src
SCRIPT_DIR=${REPO_ROOT}/examples/bace_gigpo
WEBSHOP_ENV=/hpcwork/rwth2089/xsz96350/work-BACE/verl-agent-webshop
MODEL_PATH=${WORKSPACE_ROOT}/model_down/model/Qwen2.5-1.5B-Instruct
TRAIN_FILE=${WORKSPACE_ROOT}/data/text/train.parquet
VAL_FILE=${WORKSPACE_ROOT}/data/text/test.parquet
EXP_ROOT=${WORKSPACE_ROOT}/work-BACE/experiments/webshop1k-qwen2.5-1.5b-exact
RUN_NAME=${BACE_RUN_NAME:-bace_webshop1k_exact_2gpu_seed0}
INVOCATION_ID=${SLURM_JOB_ID:-dryrun}_2gpu_formal
RUN_ROOT=${EXP_ROOT}/runs/${RUN_NAME}
ARTIFACT_DIR=${RUN_ROOT}/bace_artifacts
ROLLOUT_DIR=${RUN_ROOT}/rollout_trajectories
METADATA_DIR=${RUN_ROOT}/metadata
REPORT_DIR=${RUN_ROOT}/trace_validation
LOG_FILE=${RUN_ROOT}/training.log
CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
TENSORBOARD_DIR=${EXP_ROOT}/tensorboard/${RUN_NAME}
TOTAL_STEPS=${BACE_TOTAL_STEPS:-150}
SAVE_FREQ=${BACE_SAVE_FREQ:-5}
MAX_CHECKPOINTS=${BACE_MAX_CHECKPOINTS:-2}
PYTHON_BIN=${WEBSHOP_ENV}/bin/python
CUMEM_ALLOCATOR_SO=${WEBSHOP_ENV}/lib/python3.10/site-packages/vllm/cumem_allocator.abi3.so

[[ "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]]
[[ "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]]
[[ "${MAX_CHECKPOINTS}" =~ ^[12]$ ]]

module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WEBSHOP_ENV}"

# vLLM 0.8.2 is allowed to choose the compatible H100 attention backend.
# This is an operational backend choice, not a training hyperparameter.
unset VLLM_ATTENTION_BACKEND ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_LOGGING_LEVEL=INFO
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAVA_HOME="${WEBSHOP_ENV}/lib/jvm"
export JVM_PATH="${JAVA_HOME}/lib/server/libjvm.so"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JAVA_TOOL_OPTIONS='-Xms64m -Xmx512m -Xss512k -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Djava.awt.headless=true'
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export RAYON_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export TENSORBOARD_DIR

for required in "${REPO_ROOT}" "${WEBSHOP_ENV}" "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" "${JVM_PATH}" "${CUMEM_ALLOCATOR_SO}"; do
    [[ -e "${required}" ]] || { printf 'Missing required asset: %s\n' "${required}" >&2; exit 2; }
done
[[ -f ${MODEL_PATH}/model.safetensors.index.json || -f ${MODEL_PATH}/model.safetensors ]] || {
    printf 'No supported safetensors weights found in %s\n' "${MODEL_PATH}" >&2
    exit 2
}

mkdir -p "${ARTIFACT_DIR}" "${ROLLOUT_DIR}" "${METADATA_DIR}" "${REPORT_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" \
    "$(dirname "${LOG_FILE}")" "${EXP_ROOT}/slurm"
cd "${REPO_ROOT}"

tracker=${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt
if [[ ${DRY_RUN:-0} != 1 && -f "${tracker}" ]]; then
    # This is deliberately before model startup.  It recognizes checkpoints
    # made by an earlier Slurm process and ensures enough space for the next
    # complete write while keeping a 10-GiB filesystem reserve.
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --recommend --output "${METADATA_DIR}/checkpoint_retention_before.json"
    MAX_CHECKPOINTS=$("${PYTHON_BIN}" - "${METADATA_DIR}/checkpoint_retention_before.json" <<'PY'
import json, sys
value = int(json.load(open(sys.argv[1]))['recommended_keep'])
assert value in (1, 2)
print(value)
PY
)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
        --keep "${MAX_CHECKPOINTS}" --output "${METADATA_DIR}/checkpoint_prune_before.json"
    latest_step=$(<"${tracker}")
    if (( latest_step >= TOTAL_STEPS )); then
        "${PYTHON_BIN}" -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
            --output "${REPORT_DIR}/${INVOCATION_ID}_already_complete.json"
        printf 'Target step %s already complete; nothing to resume.\n' "${TOTAL_STEPS}"
        exit 0
    fi
fi

ppo_command=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo

    # BACE method parameters.
    algorithm.adv_estimator=bace_gigpo
    algorithm.bace.enabled=true
    algorithm.bace.variant=batch_erv_exact
    algorithm.bace.topology=dynamic
    algorithm.bace.dynamic_root_generation=staged
    algorithm.bace.staged_root_batching=packed
    algorithm.bace.branch_execution_mode=selected_worker
    algorithm.bace.branch_pool_mode=main_reuse
    algorithm.bace.root_active_executor=true
    algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.total_leaf_budget=8
    algorithm.bace.min_natural_roots=2
    algorithm.bace.capacity_correction_batch_size=1
    algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold=0.5
    algorithm.bace.history_initial_mean=0.10
    algorithm.bace.history_initial_strength=2.0
    algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1
    algorithm.bace.history_min_strength=2.0
    algorithm.bace.history_max_strength=8.0
    algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_threshold=0.005
    algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12
    algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity
    algorithm.bace.tie_break_identity_mode=stable_v1
    algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.artifacts.enabled=true
    algorithm.bace.artifacts.directory="${ARTIFACT_DIR}"
    algorithm.bace.artifacts.include_token_arrays=true
    algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true
    algorithm.bace.replay.max_origin_retries=1

    # Original AFH GiGPO WebShop training parameters, made explicit.
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=16
    data.val_batch_size=128
    data.max_prompt_length=4096
    data.max_response_length=512
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.ppo_mini_batch_size=64
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=false
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.max_num_batched_tokens=8192
    actor_rollout_ref.rollout.max_num_seqs=1024
    actor_rollout_ref.rollout.enable_chunked_prefill=false
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.free_cache_engine=false
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4
    actor_rollout_ref.rollout.val_kwargs.do_sample=true
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
    actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.actor.use_invalid_action_penalty=true
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
    algorithm.use_kl_in_reward=false
    algorithm.gamma=0.95
    algorithm.gigpo.step_advantage_w=1.0
    algorithm.gigpo.mode=mean_norm

    # WebShop-1k infrastructure; this is not an algorithmic parameter change.
    env.env_name=Webshop
    env.webshop.use_small=true
    env.webshop.human_goals=false
    env.webshop.search_backend=ray_shared
    env.webshop.search_pool_size=16
    env.webshop.search_actor_num_cpus=0.1
    env.webshop.worker_init_batch_size=16
    env.webshop.sessions_per_actor=8
    env.webshop.diagnostics_dir="${METADATA_DIR}/webshop_runtime"
    env.seed=0
    env.max_steps=15
    env.rollout.n=8
    env.resources_per_worker.num_cpus=0.1

    trainer.critic_warmup=0
    "trainer.logger=['console','tensorboard']"
    trainer.project_name=verl_agent_webshop_bace
    trainer.experiment_name="${RUN_NAME}"
    trainer.device=cuda
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    trainer.save_freq="${SAVE_FREQ}"
    trainer.test_freq=5
    trainer.total_epochs="${TOTAL_STEPS}"
    trainer.total_training_steps="${TOTAL_STEPS}"
    trainer.val_before_train=true
    trainer.validate_on_last_step=true
    trainer.default_local_dir="${CHECKPOINT_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DIR}"
    trainer.resume_mode=auto
    trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}"
    trainer.max_global_ckpt_to_keep="${MAX_CHECKPOINTS}"
    ray_init.num_cpus=32
)

ppo_command+=("$@")
printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"
printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
"${PYTHON_BIN}" -m verl.trainer.main_ppo --cfg job --resolve \
    "${ppo_command[@]:3}" > "${METADATA_DIR}/resolved_config.yaml"
(
    find recipe/bace_gigpo verl/trainer agent_system examples/bace_gigpo \
        -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) \
        -print0 | sort -z | xargs -0 sha256sum
) > "${METADATA_DIR}/source_manifest.sha256"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'Dry-run complete: %s\n' "${METADATA_DIR}/resolved_command.sh"
    exit 0
fi

"${PYTHON_BIN}" - <<'PY' > "${METADATA_DIR}/preflight.json"
import importlib, json, os, torch
modules = ['torch', 'vllm', 'ray', 'transformers', 'jnius', 'pyserini']
imports = {}
for name in modules:
    try:
        importlib.import_module(name)
        imports[name] = 'ok'
    except Exception as exc:
        imports[name] = repr(exc)
report = {
    'ok': torch.cuda.is_available() and torch.cuda.device_count() == 2
          and all(value == 'ok' for value in imports.values()),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
    'gpu_count': torch.cuda.device_count(),
    'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    'imports': imports,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report['ok']:
    raise SystemExit(3)
PY
nvidia-smi -q > "${METADATA_DIR}/nvidia_smi_q.txt"
nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"
"${PYTHON_BIN}" -m pip freeze > "${METADATA_DIR}/pip_freeze.txt"
# The WebShop executable-action classifier is part of BACE's strict replay
# semantics.  Run its focused regression suite in the submitted environment,
# so this formal run cannot silently fall back to the former full-match rule.
"${PYTHON_BIN}" -m pytest -q tests/bace_gigpo/test_webshop_replay.py \
    > "${METADATA_DIR}/webshop_action_classification_test.txt"
# vLLM 0.8.2's abi3 allocator must use the running CPython's refcount API.
# Record and enforce the patched artifact used by this run.
sha256sum "${CUMEM_ALLOCATOR_SO}" > "${METADATA_DIR}/vllm_cumem_allocator.sha256"
nm -D "${CUMEM_ALLOCATOR_SO}" | grep -q ' U Py_IncRef$'

# Real branch execution is deterministic here, independently of whether a
# cold Beta(0.2,1.8) PPO step happens to allocate a model-sampled branch.
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_webshop_real_env.py" \
    --sessions 500 501 --output "${REPORT_DIR}/real_env_identity.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_webshop_selected_worker_real_env.py" \
    --session 500 --ray-num-cpus 2 --timeout-seconds 600 \
    --output "${REPORT_DIR}/selected_worker_real_env.json"

exec > >(tee -a "${LOG_FILE}") 2>&1
set +e
"${ppo_command[@]}"
train_status=$?
set -e
printf 'train_exit_code=%s\nend_time=%s\n' "${train_status}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${METADATA_DIR}/completion.txt"
(( train_status == 0 )) || exit "${train_status}"

grep -Eq "training/global_step['\"]?:[[:space:]]*${TOTAL_STEPS}|training/global_step[^0-9]+${TOTAL_STEPS}" "${LOG_FILE}"
! grep -Eqi 'CUDA out of memory|OutOfMemoryError|RayTaskError|NCCL[^[:space:]]* error' "${LOG_FILE}"
"${PYTHON_BIN}" -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" \
    --output "${REPORT_DIR}/${INVOCATION_ID}_final.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" \
    --keep "${MAX_CHECKPOINTS}" --output "${METADATA_DIR}/checkpoint_prune_after.json"
"${PYTHON_BIN}" - "${ARTIFACT_DIR}/step_$(printf '%08d' "${TOTAL_STEPS}")" "${CHECKPOINT_DIR}" "${TOTAL_STEPS}" <<'PY'
import json, sys
from pathlib import Path
step = Path(sys.argv[1])
checkpoint_dir = Path(sys.argv[2])
total_steps = int(sys.argv[3])
summary = json.loads((step / 'summary.json').read_text())
manifest = json.loads((step / 'manifest.json').read_text())
metrics = summary['bace_metrics']
assert summary['status'] == 'complete'
assert manifest['variant'] == 'batch_erv_exact'
assert manifest['staged_root_batching'] == 'packed'
assert manifest['branch_execution_mode'] == 'selected_worker'
assert float(metrics['batch_erv_exact']) == 1.0
assert float(metrics['staged_root_batching_packed']) == 1.0
assert float(metrics['root_active_executor_enabled']) == 1.0
roots = [json.loads(line) for line in (step / 'roots.jsonl').read_text().splitlines() if line]
branch_file = step / 'branches.jsonl'
branches = [json.loads(line) for line in branch_file.read_text().splitlines() if line] if branch_file.exists() else []
assert roots
assert all(int(row['environment_reset_key']) >= 500 for row in roots + branches)
assert (checkpoint_dir / 'latest_checkpointed_iteration.txt').read_text().strip() == str(total_steps)
assert (checkpoint_dir / f'global_step_{total_steps}' / 'bace_collector_state.json').is_file()
print('WebShop Exact BACE formal run passed')
PY

printf 'All WebShop Exact BACE formal-run gates passed.\n'
