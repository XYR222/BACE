#!/usr/bin/env bash
# Independent full WebShop-1k C8/correction=1 run; never calls an ALFWorld script.
#SBATCH --job-name=bace_ws_c8_corr1
#SBATCH --partition=c25g
#SBATCH --account=rwth2089
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=244G
#SBATCH --gres=gpu:2
#SBATCH --time=15:00:00
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
RUN_NAME=${BACE_RUN_NAME:-bace_webshop1k_c8_corr1_2gpu_seed0_20260915}
TOTAL_STEPS=${BACE_TOTAL_STEPS:-150}
SAVE_FREQ=${BACE_SAVE_FREQ:-5}
MAX_CHECKPOINTS=${BACE_MAX_CHECKPOINTS:-2}
RUN_ROOT=${EXP_ROOT}/runs/${RUN_NAME}
ARTIFACT_DIR=${RUN_ROOT}/bace_artifacts
ROLLOUT_DIR=${RUN_ROOT}/rollout_trajectories
METADATA_DIR=${RUN_ROOT}/metadata
REPORT_DIR=${RUN_ROOT}/trace_validation
LOG_FILE=${RUN_ROOT}/training.log
CHECKPOINT_DIR=${EXP_ROOT}/checkpoints/${RUN_NAME}
TENSORBOARD_DIR=${EXP_ROOT}/tensorboard/${RUN_NAME}
PYTHON_BIN=${WEBSHOP_ENV}/bin/python
CUMEM_ALLOCATOR_SO=${WEBSHOP_ENV}/lib/python3.10/site-packages/vllm/cumem_allocator.abi3.so

[[ "${RUN_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]]
[[ "${TOTAL_STEPS}" =~ ^[1-9][0-9]*$ && "${SAVE_FREQ}" =~ ^[1-9][0-9]*$ && "${MAX_CHECKPOINTS}" =~ ^[12]$ ]]
module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate "${WEBSHOP_ENV}"
unset VLLM_ATTENTION_BACKEND ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
unset LOCAL_RANK LOCAL_WORLD_SIZE RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=INFO HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONNOUSERSITE=1 WANDB_MODE=offline TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAVA_HOME="${WEBSHOP_ENV}/lib/jvm" JVM_PATH="${WEBSHOP_ENV}/lib/jvm/lib/server/libjvm.so"
export PATH="${JAVA_HOME}/bin:${PATH}"
export JAVA_TOOL_OPTIONS='-Xms64m -Xmx512m -Xss512k -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Djava.awt.headless=true'
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 RAYON_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export TENSORBOARD_DIR

for required in "${REPO_ROOT}" "${WEBSHOP_ENV}" "${MODEL_PATH}" "${TRAIN_FILE}" "${VAL_FILE}" "${JVM_PATH}" "${CUMEM_ALLOCATOR_SO}"; do
    [[ -e "${required}" ]] || { printf 'Missing required asset: %s\n' "${required}" >&2; exit 2; }
done
[[ -f ${MODEL_PATH}/model.safetensors.index.json || -f ${MODEL_PATH}/model.safetensors ]] || { echo "No safetensors weights" >&2; exit 2; }
mkdir -p "${ARTIFACT_DIR}" "${ROLLOUT_DIR}" "${METADATA_DIR}" "${REPORT_DIR}" "${CHECKPOINT_DIR}" "${TENSORBOARD_DIR}" "$(dirname "${LOG_FILE}")" "${EXP_ROOT}/slurm"
cd "${REPO_ROOT}"

rotate_checkpoints() {
    local phase=$1 recommended
    [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]] || return 0
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" --recommend --output "${METADATA_DIR}/checkpoint_retention_${phase}.json"
    recommended=$("${PYTHON_BIN}" - "${METADATA_DIR}/checkpoint_retention_${phase}.json" <<'PY'
import json, sys
value = int(json.load(open(sys.argv[1]))['recommended_keep'])
assert value in (1, 2)
print(value)
PY
)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prune_checkpoints.py" "${CHECKPOINT_DIR}" --keep "${recommended}" --output "${METADATA_DIR}/checkpoint_prune_${phase}.json"
}

if [[ ${DRY_RUN:-0} != 1 ]]; then
    rotate_checkpoints before
    if [[ -f "${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt" ]] && (( $(<"${CHECKPOINT_DIR}/latest_checkpointed_iteration.txt") >= TOTAL_STEPS )); then
        "${PYTHON_BIN}" -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" --output "${REPORT_DIR}/${SLURM_JOB_ID:-local}_already_complete.json"
        echo "Target already complete; nothing to resume."
        exit 0
    fi
fi

ppo_command=(
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    algorithm.adv_estimator=bace_gigpo algorithm.bace.enabled=true algorithm.bace.variant=batch_erv_exact algorithm.bace.acquisition=batch_erv_exact
    algorithm.bace.topology=dynamic algorithm.bace.dynamic_root_generation=staged algorithm.bace.staged_root_batching=packed
    algorithm.bace.branch_execution_mode=selected_worker algorithm.bace.branch_pool_mode=main_reuse algorithm.bace.root_active_executor=true
    algorithm.bace.total_leaf_budget=8 algorithm.bace.min_natural_roots=2 algorithm.bace.capacity_correction_batch_size=1 algorithm.bace.max_branches_per_anchor=2
    algorithm.bace.competence_threshold=0.5 algorithm.bace.history_initial_mean=0.10 algorithm.bace.history_initial_strength=2.0 algorithm.bace.history_forgetting=0.8
    algorithm.bace.history_transfer_fraction=0.1 algorithm.bace.history_min_strength=2.0 algorithm.bace.history_max_strength=8.0 algorithm.bace.local_prior_strength=2.0
    algorithm.bace.batch_erv_threshold=0.005 algorithm.bace.batch_erv_tie_abs_tolerance=1.0e-12 algorithm.bace.batch_erv_tie_rel_tolerance=1.0e-10
    algorithm.bace.invalid_action_mode=strict_identity algorithm.bace.tie_break_identity_mode=stable_v1 algorithm.bace.local_credit_mode=occurrence
    algorithm.bace.credit_mode=c8_macro_local_strict_ancestor algorithm.bace.tree_credit_mode=current algorithm.bace.macro_normalization_mode=stable_occurrence algorithm.bace.tree_ppo_padding_mode=copy_trainable
    algorithm.bace.artifacts.enabled=true algorithm.bace.artifacts.directory="${ARTIFACT_DIR}" algorithm.bace.artifacts.include_token_arrays=true algorithm.bace.artifacts.fsync=false
    algorithm.bace.replay.compare_action_set=true algorithm.bace.replay.max_origin_retries=1
    data.train_files="${TRAIN_FILE}" data.val_files="${VAL_FILE}" data.train_batch_size=16 data.val_batch_size=128 data.max_prompt_length=4096 data.max_response_length=512 data.filter_overlong_prompts=True data.truncation=error data.return_raw_chat=True
    actor_rollout_ref.model.path="${MODEL_PATH}" actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.param_offload=false actor_rollout_ref.actor.fsdp_config.optimizer_offload=false actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.tensor_model_parallel_size=2
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.gpu_memory_utilization=0.6 actor_rollout_ref.rollout.max_num_batched_tokens=8192 actor_rollout_ref.rollout.max_num_seqs=1024 actor_rollout_ref.rollout.enable_chunked_prefill=false actor_rollout_ref.rollout.enforce_eager=false actor_rollout_ref.rollout.free_cache_engine=false
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 actor_rollout_ref.rollout.val_kwargs.do_sample=true actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.fsdp_config.param_offload=true
    actor_rollout_ref.actor.use_invalid_action_penalty=true actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 algorithm.use_kl_in_reward=false algorithm.gamma=0.95 algorithm.gigpo.step_advantage_w=1.0 algorithm.gigpo.mode=mean_norm
    env.env_name=Webshop env.webshop.use_small=true env.webshop.human_goals=false env.webshop.search_backend=ray_shared env.webshop.search_pool_size=16 env.webshop.search_actor_num_cpus=0.1 env.webshop.worker_init_batch_size=16 env.webshop.sessions_per_actor=8 env.webshop.diagnostics_dir="${METADATA_DIR}/webshop_runtime" env.seed=0 env.max_steps=15 env.rollout.n=8 env.resources_per_worker.num_cpus=0.1
    trainer.critic_warmup=0 "trainer.logger=['console','tensorboard']" trainer.project_name=verl_agent_webshop_bace trainer.experiment_name="${RUN_NAME}" trainer.device=cuda trainer.n_gpus_per_node=2 trainer.nnodes=1 trainer.save_freq="${SAVE_FREQ}" trainer.test_freq=5 trainer.total_epochs="${TOTAL_STEPS}" trainer.total_training_steps="${TOTAL_STEPS}" trainer.val_before_train=true trainer.validate_on_last_step=true trainer.default_local_dir="${CHECKPOINT_DIR}" trainer.rollout_data_dir="${ROLLOUT_DIR}" trainer.resume_mode=auto trainer.max_actor_ckpt_to_keep="${MAX_CHECKPOINTS}" trainer.max_global_ckpt_to_keep="${MAX_CHECKPOINTS}" ray_init.num_cpus=32
)
ppo_command+=("$@")
printf '%q ' "${ppo_command[@]}" > "${METADATA_DIR}/resolved_command.sh"; printf '\n' >> "${METADATA_DIR}/resolved_command.sh"
printf 'method=C8\ncredit_mode=c8_macro_local_strict_ancestor\ntree_credit_mode=current\nmacro_normalization_mode=stable_occurrence\ncapacity_correction_batch_size=1\nscheduling_profile=optimized\nenvironment=WebShop-1k\nseed=0\ntarget_steps=%s\ncheckpoint_policy=save_every_%s_keep_latest_%s\n' "${TOTAL_STEPS}" "${SAVE_FREQ}" "${MAX_CHECKPOINTS}" > "${METADATA_DIR}/run_metadata.txt"
( find recipe/bace_gigpo verl/trainer agent_system examples/bace_gigpo -type f \( -name '*.py' -o -name '*.yaml' -o -name '*.sh' -o -name '*.sbatch' \) -print0 | sort -z | xargs -0 sha256sum ) > "${METADATA_DIR}/source_manifest.sha256"

if [[ ${DRY_RUN:-0} == 1 ]]; then echo "Dry-run complete: ${METADATA_DIR}/resolved_command.sh"; exit 0; fi
"${PYTHON_BIN}" -m verl.trainer.main_ppo --cfg job --resolve "${ppo_command[@]:3}" > "${METADATA_DIR}/resolved_config.yaml"
"${PYTHON_BIN}" - <<'PY' > "${METADATA_DIR}/preflight.json"
import importlib, json, os, torch
imports = {}
for name in ['torch', 'vllm', 'ray', 'transformers', 'jnius', 'pyserini']:
    try: importlib.import_module(name); imports[name] = 'ok'
    except Exception as exc: imports[name] = repr(exc)
report = {'ok': torch.cuda.is_available() and torch.cuda.device_count() == 2 and all(v == 'ok' for v in imports.values()), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_count': torch.cuda.device_count(), 'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], 'imports': imports}
print(json.dumps(report, indent=2, sort_keys=True))
if not report['ok']: raise SystemExit(3)
PY
nvidia-smi -q > "${METADATA_DIR}/nvidia_smi_q.txt"; nvidia-smi topo -m > "${METADATA_DIR}/gpu_topology.txt"; "${PYTHON_BIN}" -m pip freeze > "${METADATA_DIR}/pip_freeze.txt"
"${PYTHON_BIN}" -m pytest -q tests/bace_gigpo/test_webshop_replay.py > "${METADATA_DIR}/webshop_action_classification_test.txt"
sha256sum "${CUMEM_ALLOCATOR_SO}" > "${METADATA_DIR}/vllm_cumem_allocator.sha256"; nm -D "${CUMEM_ALLOCATOR_SO}" | grep -q ' U Py_IncRef$'
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_webshop_real_env.py" --sessions 500 501 --output "${REPORT_DIR}/real_env_identity.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_webshop_selected_worker_real_env.py" --session 500 --ray-num-cpus 2 --timeout-seconds 600 --output "${REPORT_DIR}/selected_worker_real_env.json"

exec > >(tee -a "${LOG_FILE}") 2>&1
set +e; "${ppo_command[@]}"; train_status=$?; set -e
printf 'train_exit_code=%s\nend_time=%s\n' "${train_status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${METADATA_DIR}/completion.txt"
(( train_status == 0 )) || exit "${train_status}"
grep -Eq "training/global_step['\"]?:[[:space:]]*${TOTAL_STEPS}|training/global_step[^0-9]+${TOTAL_STEPS}" "${LOG_FILE}"
! grep -Eqi 'CUDA out of memory|OutOfMemoryError|RayTaskError|NCCL[^[:space:]]* error' "${LOG_FILE}"
"${PYTHON_BIN}" -m recipe.bace_gigpo.validate_trace "${ARTIFACT_DIR}" --output "${REPORT_DIR}/${SLURM_JOB_ID:-local}_final.json"
rotate_checkpoints after
"${PYTHON_BIN}" - "${ARTIFACT_DIR}/step_$(printf '%08d' "${TOTAL_STEPS}")" "${CHECKPOINT_DIR}" "${TOTAL_STEPS}" <<'PY'
import json, sys
from pathlib import Path
step, checkpoints, total = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
summary = json.loads((step / 'summary.json').read_text()); manifest = json.loads((step / 'manifest.json').read_text()); metrics = summary['bace_metrics']
assert summary['status'] == 'complete' and manifest['variant'] == 'batch_erv_exact'
assert manifest['staged_root_batching'] == 'packed' and manifest['branch_execution_mode'] == 'selected_worker'
assert manifest['credit_mode'] == 'c8_macro_local_strict_ancestor'
assert float(metrics['batch_erv_exact']) == 1.0 and float(metrics['staged_root_batching_packed']) == 1.0
roots = [json.loads(line) for line in (step / 'roots.jsonl').read_text().splitlines() if line]
branch_path = step / 'branches.jsonl'; branches = [json.loads(line) for line in branch_path.read_text().splitlines() if line] if branch_path.exists() else []
assert roots and all(int(row['environment_reset_key']) >= 500 for row in roots)
# Branch summaries intentionally lack environment_reset_key; validate the origin link instead.
assert all(row['origin_occurrence_id'] in {root['occurrence_id'] for root in roots} for row in branches)
assert (checkpoints / 'latest_checkpointed_iteration.txt').read_text().strip() == str(total)
assert (checkpoints / f'global_step_{total}' / 'bace_collector_state.json').is_file()
print('WebShop C8 Exact-Batch-ERV formal run passed')
PY
echo 'All WebShop C8 formal-run gates passed.'
