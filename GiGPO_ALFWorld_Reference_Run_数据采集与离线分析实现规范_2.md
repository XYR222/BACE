# GiGPO ALFWorld Reference Run：数据采集、保存与离线分析实现规范

> 日期：2026-08-25  
> 适用基线：`verl-agent` 官方 GiGPO + ALFWorld TextWorld (`alfworld/AlfredTWEnv`)  
> 目标：在**不改变原版 GiGPO 训练语义**的前提下，将一次完整 ALFWorld 训练建设成可长期复用的 Reference Archive，使论文引入分析、anchor 证据诊断、BACE 参数预分析、Exact Batch-ERV 离线模拟、replay/snapshot 回归测试和系统效率比较尽量不需要重复训练。

---

# 0. 结论与总体原则

这次原版 GiGPO ALFWorld 训练应同时承担三种角色：

$$
\boxed{
\text{Baseline Training}
+
\text{Reference Data Acquisition}
+
\text{Engineering Regression Corpus}
}
$$

实现时采用一个**外挂式 `TraceRecorder`**，只观察和记录训练过程，不改变：

- task sampling；
- rollout sampling；
- ALFWorld 环境 transition；
- `anchor_obs` 构造；
- GiGPO step-group grouping；
- episode/step advantage；
- PPO loss；
- optimizer / scheduler；
- validation protocol。

数据设计遵循四条总原则。

## 0.1 保存原始事实，而不是只保存当前想看的指标

推荐数据链：

$$
\boxed{
\text{Immutable Raw Facts}
\rightarrow
\text{Versioned Derived Tables}
\rightarrow
\text{Statistics / Figures}
}
$$

例如 `within-action sign conflict` 当前可以定义为同一 $(z,u)$ 内同时出现正、负 $A^S$；以后阈值可能改成 $\pm\epsilon_A$。因此 raw archive 应保存 $G_i$、$A_i^S$、anchor、action 等事实，而不是只保存 `is_conflict=true`。

## 0.2 Occurrence 是核心数据单位

将每一次 actor 真正做出并提交给环境的 macro-action 定义为一个 occurrence：

$$
o_i=(g,r,t,z_i,a_i,G_i,\ldots).
$$

后续几乎所有分析都可以归结到：

$$
\boxed{(z,u,G,\text{trajectory},t,\text{success})}.
$$

推荐稳定 ID：

```text
{run_id}/{update_id}/{task_instance_id}/{trajectory_id}/{step_idx}
```

例如：

```text
seed0/u0075/task03/root06/step17
```

## 0.3 Raw 与 Derived 分开并版本化

推荐：

```text
raw/                    # 完整训练后冻结，不修改
analysis/derived_v1/    # 当前定义
analysis/derived_v2/    # 将来调整定义后重新生成
```

任何 canonicalization、conflict threshold、anchor eligibility、BERV threshold 或 prior 参数变化，都通过新的 derived version 处理。

## 0.4 在线值和离线重算值同时保留

凡是训练时真正参与 GiGPO/PPO 的量，即便理论上可以重算，也应缓存 online value，用于验证：

$$
\boxed{
\text{Offline Reconstruction}
\approx
\text{Online Training Value}
}
$$

这包括：

- online step-group assignment；
- $A^E$；
- $A^S$；
- final combined advantage；
- chosen-token old logprob；
- group mean/std。

---

# 1. 官方 GiGPO ALFWorld 基线合同

当前 `verl-agent` 官方 `examples/gigpo_trainer/run_alfworld.sh` 的核心设置包括：

```text
train_data_size = 16
val_data_size   = 128
group_size      = 8
mode            = mean_std_norm
algorithm.gamma = 0.95
algorithm.gigpo.step_advantage_w = 1.0
env.env_name    = alfworld/AlfredTWEnv
env.max_steps   = 50
env.rollout.n   = 8
trainer.test_freq = 5
trainer.total_epochs = 150
trainer.val_before_train = True
```

官方 master 当前脚本的 GPU 并行参数属于运行配置；如果本地为 8 卡运行，只应将其视为硬件/并行 override，并完整记录 resolved config。**Reference Run 的科学合同由实际运行时 resolved config 决定，而不是手工抄录的脚本默认值。**

GiGPO 核心实现 `gigpo/core_gigpo.py` 中，step group 默认：

```text
enable_similarity = False
```

即在同一个 episode/task group 内，对 `anchor_obs` 做 exact key-based grouping。Reference Archive 必须保存能离线复原这一行为的原始 `anchor_obs` 与在线 group assignment。

---

# 2. Reference Archive 的六层结构

建议保存六层数据：

| 层级 | 核心对象 | 优先级 | 主要用途 |
|---|---|---:|---|
| Run / Manifest | 配置、commit、seed、硬件 | P0 | 完全复现、公平比较 |
| Update | 每个 PPO update | P0 | learning curve、timing、checkpoint 对齐 |
| Task / Group | 每个 ALFWorld task group | P0 | task-family、group-level 统计 |
| Trajectory | 每条 natural root | P0 | episode outcome、replay、coverage |
| Occurrence | 每个真实环境动作 | **P0 核心** | anchor/action/credit/BERV 分析 |
| Online GiGPO Group | 在线构造 step groups | P0 | online/offline 审计、Figure 1 |

另外独立保存：

- token/logprob shards；
- actor/full checkpoints；
- profiler；
- validation records；
- replay regression corpus。

---

# Part I：数据采集与保存

# 3. Run / Manifest：实验身份必须完整保存

## 3.1 Run identity

保存：

```text
run_id
experiment_name
seed
start_time
end_time
hostname
```

**理由：** 三 seed 汇总、错误排查、所有 shard/checkpoint/figure 对齐都依赖稳定 run identity。

## 3.2 代码与依赖版本

保存：

```text
verl_agent_commit
local_patch_commit
ALFWorld_version_or_commit
TextWorld_version
PyTorch_version
transformers_version
vLLM_version
CUDA_version
NCCL_version
Python_version
```

同时保存：

```text
git diff / instrumentation patch
```

**理由：** anchor grouping、ALFWorld transition、generation、PPO 和 replay 都可能受版本影响。未来 BACE 必须能严格复现这次 baseline。

## 3.3 完整 resolved config

保存训练启动后真正解析出的完整配置，至少包括：

```text
model path
reference model path
data train/val files
train_data_size / val_data_size
group_size
max prompt / response length
rollout temperature/top-p/top-k
env max_steps
env seed
actor lr
PPO clip parameters
KL settings
GiGPO gamma
GiGPO step_advantage_w
GiGPO mode
mini/micro batch sizes
rollout tensor parallelism
GPU count
validation interval
checkpoint interval
total epochs
```

**理由：** 后续 GiGPO 与 BACE 公平比较需要同模型、同环境、同 leaf budget、同 trainer 语义。

## 3.4 硬件与系统

保存：

```text
GPU model / count / memory
CPU model / cores
RAM
storage type
interconnect / network
Ray version
```

**理由：** wall-clock 与 GPU-hour 只有在硬件条件明确时才可解释。

---

# 4. Update 层：每个 Policy Update 必须保存什么

一次 update 定义为：

```text
rollout
-> GiGPO grouping/advantage
-> old/ref logprob
-> actor update
```

## 4.1 训练进度

```text
update_id
epoch_id
global_step
wall_clock_since_start
```

## 4.2 Performance

保存 aggregate 和 task-level raw outcomes：

```text
train_success_rate
train_mean_reward
validation_success_rate
validation_mean_reward
per_task_family_success_rate
```

**理由：** 用于 learning curves、low/mid/high competence stage 选点、task-family controller 模拟和跨 seed 统计。

## 4.3 生成规模

```text
num_tasks
num_terminal_trajectories
num_macro_actions
num_prompt_tokens
num_response_tokens
num_actor_generated_tokens
trajectory_length_mean/std/max
response_length_mean/std/max
```

**理由：** BACE 的 branch suffix 与完整 root 长度不同，equal leaves 并不等于 equal generation compute。

## 4.4 环境交互规模

```text
env_steps_total
env_steps_successful_trajectories
env_steps_failed_trajectories
env_invalid_action_steps
```

**理由：** 用于 interaction efficiency 和 early-stage invalid/loop 诊断。

---

# 5. Task / Group 层

每个训练 task group 至少保存：

```text
run_id
update_id
task_instance_id
task_family
group_uid
game_id
game_file_path
task_description
num_rollouts_expected
num_rollouts_actual
num_success
num_failure
```

若 ALFWorld task-family mapping 已存在，应同时保存原始 family label 和当前派生版本，例如：

```text
task_family_raw
task_family_v0
```

**理由：** 后续 competence controller、family-level success、不同任务类型的 anchor 结构均需要 task identity。

---

# 6. Trajectory 层：每条 Natural Root 必须保存什么

## 6.1 Identity

```text
run_id
update_id
task_instance_id
task_family
trajectory_id
root_slot / group_slot
rollout_seed_if_available
```

## 6.2 Environment identity

```text
game_file_path
game_id
environment_mode
max_steps
```

**理由：** ALFWorld replay 必须能重新加载相同 game。

## 6.3 Episode outcome

```text
terminal_reward
terminal_success
terminal_reason
terminated
truncated
num_steps
reward_sequence_ref
```

**理由：** 这是 trajectory credit、competence history、local posterior 和 outcome decomposition 的 source of truth。

## 6.4 Online episode-level GiGPO quantities

缓存训练时实际使用：

```text
trajectory_group_mean_return
trajectory_group_std_return
A_E_online
```

**理由：** 用于验证离线重算 $A^E$ 与在线训练完全一致。

## 6.5 事件序列引用

trajectory row 不重复存所有长文本，仅保存：

```text
first_occurrence_id
num_occurrences
event_log_ref
```

## 6.6 Token/cost summary

```text
prompt_tokens
response_tokens
actor_generated_tokens
```

**理由：** 支持 trajectory/task 级 generation cost 分析。

---

# 7. Occurrence 层：最核心的数据表

每次 actor 真正做出并送入环境的动作写一条 occurrence。

## 7.1 Identity 与位置

```text
occurrence_id
run_id
update_id
task_instance_id
task_family
trajectory_id
step_idx
```

必须保证全局唯一。

## 7.2 Pre-action observation

必须保存：

```text
anchor_obs_before_action_raw_ref
anchor_obs_serialized
anchor_key_online
```

其中：

- `raw_ref` 指向原始环境 observation 文本/结构；
- `serialized` 保存进入 `to_hashable`/grouping 前的确定性序列化表示；
- `anchor_key_online` 缓存在线 GiGPO 实际 key/group 相关信息。

**理由：** 原版 GiGPO 的 step grouping 依赖 pre-action `anchor_obs` exact grouping。只保存 hash 会丢失未来重建、相似度消融、history-aware audit 和 replay assertion 能力。

## 7.3 Post-action observation

```text
observation_after_action_raw_ref
```

**理由：** replay transition fidelity、invalid action effect 和 action-state change 分析需要它。

## 7.4 Admissible action set

必须保存：

```text
admissible_actions_before_action_ref
```

建议同时保存稳定排序版本：

```text
admissible_actions_sorted_ref
```

**理由：** 后续要判断 executable action、canonical action、invalid ratio、action competition，以及 replay 后 action-set 是否精确一致。

## 7.5 模型完整响应

保存：

```text
raw_model_response_ref
raw_thought_text_ref       # 若 parser 能拆分
raw_action_text
```

**理由：** parser/canonicalization 规则未来可能改变，raw response 必须作为 source of truth。

## 7.6 Parser 输出与环境真实执行动作

保存：

```text
parsed_action
executed_action
execution_success
invalid_action_flag
env_feedback_after_action_ref
```

**理由：** 必须严格区分：

$$
\text{LLM output}
\neq
\text{parsed action}
\neq
\text{environment executed action}.
$$

## 7.7 Canonical action 缓存

可以保存当前实现的：

```text
canonical_action_v0
canonicalization_status
canonicalization_reason
```

但该字段属于 `CACHE/DERIVED`，不能替代 `executed_action + admissible set + raw response`。

## 7.8 Reward / outcome

```text
step_reward
done_after_action
terminal_success_of_trajectory
terminal_return_of_trajectory
```

## 7.9 Return-to-go

缓存在线实际值：

```text
return_to_go_G
gamma_used
```

**理由：** Figure 1、step advantage、within-action variance、mixed local credit 都直接依赖 $G_i$。

## 7.10 在线 GiGPO step group

```text
online_step_group_id
online_step_group_size
step_group_mean_G
step_group_std_G
A_S_online
```

**理由：** 用于 online/offline group reconstruction 与 local advantage invariant。

## 7.11 Final combined advantage

```text
A_combined_online
```

**理由：** 用于 offline loss reconstruction、分析 local term 对最终训练信号的贡献。

## 7.12 Replay / horizon metadata

```text
history_ref
environment_action_prefix_ref
remaining_horizon
```

实际可通过 trajectory event sequence 重建 prefix，避免每行重复复制。

---

# 8. Model Context 与 Conversation History

GiGPO anchor grouping 只按当前 observation，但 LLM 的行为依赖完整 history。因此必须能够离线恢复每个 occurrence 的模型输入。

推荐保存 trajectory event log：

```text
system prompt
initial task/user prompt
environment observation 0
model response 0
environment feedback 0
model response 1
...
```

Occurrence 只保存：

```text
trajectory_id
step_idx
history_prefix_end_offset
```

于是模型 history：

$$
h_{r,t}
$$

可以离线精确恢复。

**理由：** 后续可分析同一 `anchor_obs`、同一 action 的 credit conflict 是否与 arrival history 有关，也为 branch-origin context reuse 提供依据。

---

# 9. Token 与 Log-Probability 数据

## 9.1 P1 强烈推荐保存

```text
response_token_ids
response_attention_mask
action_token_mask / action span
chosen_token_old_logprobs
chosen_token_ref_logprobs     # 若当前 pipeline 已经计算
```

**理由：** 后续可分析 policy support、action confidence、PPO clipping、CoT/action token 语义和 branch-origin response reuse。

## 9.2 大数组独立 shard

Token/logprob 数组不要内嵌进主 Parquet，采用：

```text
array_shard_id
offset
length
```

## 9.3 Full vocabulary logits 的保存策略

全量 logits 体积巨大。Reference Run 主档案以 chosen-token logprob 为主；若将来需要 top-k alternatives，可针对固定 checkpoint 和抽样 anchor 重新 query actor，并将该 probe 作为新的 derived dataset。

---

# 10. Online GiGPO Group 表

即使 occurrence 表足够离线重建 group，也必须保存在线实际 group assignment。

每个 group 建议记录：

```text
run_id
update_id
task_instance_id
online_step_group_id
anchor_key
anchor_obs_ref
group_size
occurrence_ids_ref
trajectory_ids_ref
num_unique_trajectories
returns_G_ref
mean_G
std_G
step_advantages_ref
num_unique_raw_actions
num_unique_executed_actions
num_unique_canonical_actions_v0
```

**理由 1：在线/离线一致性**

要求：

$$
\boxed{
\mathcal I^{offline}(z)=\mathcal I^{online}(z)
}
$$

以及：

$$
\boxed{
A_i^{S,offline}\approx A_i^{S,online}
}
$$

**理由 2：论文 Figure 1**

可直接派生 group size、cross-trajectory recurrence、loop-heavy group、action diversity 与 evidence-quality decomposition。

---

# 11. PPO / Optimization 指标

每 update 至少保存：

```text
actor_loss
policy_loss
KL_loss / ref_KL
total_loss
learning_rate
PPO_clip_fraction
approx_kl / policy_kl
entropy_if_available
grad_norm_before_clip
grad_norm_after_clip
optimizer_step_skipped / overflow flag
```

同时保存 advantage aggregate：

```text
A_E mean/std/min/max/quantiles
A_S mean/std/min/max/quantiles
A_combined mean/std/quantiles
fraction A_S == 0
fraction positive/negative A_S
fraction step groups with zero variance
```

**理由：** 后续 BACE 与 GiGPO 差异需要区分来自 evidence acquisition 还是 optimization instability。

---

# 12. System Timing / Profiling

最终论文需要 GiGPO vs BACE 的 wall-clock、公平 GPU-hour 对比，因此 baseline 这次必须完整 profile。

每 update 分阶段计时：

```text
data/task preparation
rollout generation
environment interaction
GiGPO grouping
return/advantage computation
old-policy logprob
reference-model logprob
actor forward/backward
optimizer step
validation
checkpoint I/O
trace logging I/O
```

重点至少单列：

$$
T_{rollout},
T_{env},
T_{group},
T_{adv},
T_{old},
T_{ref},
T_{update}.
$$

同时记录 Recorder 自身开销：

```text
trace_encode_time
trace_write_time
trace_queue_wait
bytes_written
```

**理由：** instrumentation 不能静默污染 baseline timing；未来两边可使用同一 recorder，也可以明确扣分/解释 recorder overhead。

---

# 13. GPU / CPU / Ray 资源指标

按 update 或固定采样周期记录：

```text
GPU utilization
GPU memory allocated/reserved
peak GPU memory
CPU utilization
RAM usage
rollout worker utilization
```

如方便，附加：

```text
pending generation requests
actual rollout batch size
GPU idle/tail time
Ray task queue depth
```

**理由：** BACE 后续主要系统风险在 root-wave / branch-wave 编排和 tail under-utilization，Reference GiGPO 是必须的利用率基准。

---

# 14. Validation 数据保存

每次 validation 不只保存 aggregate success，还保存 task-level records：

```text
checkpoint/update_id
validation_task_id
task_family
success
reward
num_steps
trajectory_ref
```

**理由：** 可做 paired comparison、task-family breakdown、failure analysis，并用于选择 low/mid/high competence checkpoint。

---

# 15. Checkpoint 策略

## 15.1 Actor-only checkpoint

建议每 5-10 个 policy updates 保存一次，至少覆盖训练早、中、晚期。

保存：

```text
actor weights
tokenizer/config
update_id
```

用途：

- 重新 rollout；
- actor probability probe；
- replay/snapshot test；
- 机制实验起点。

## 15.2 Full resumable checkpoint

按 validation interval 或较稀周期保存：

```text
actor
optimizer
scheduler
AMP scaler if any
global step/update id
Python RNG
NumPy RNG
PyTorch CPU RNG
CUDA RNG states
sampler/dataloader state if applicable
```

**理由：** future topology experiment 要从同一训练状态 clone 多种 topology，仅 actor 权重不足以实现严格公平 continuation。

---

# 16. Replay / Snapshot 所需长期恢复字段

每个 occurrence 必须能够恢复：

```text
task_id
game_file_path
trajectory_id
step_idx
environment action prefix
anchor_obs_before_action
admissible_actions_before_action
model history ref
remaining_horizon
```

长期 source of truth 应为：

$$
\boxed{
\text{same game + deterministic action prefix + recorded context}
}
$$

未来 native `StateID` 可作为运行时优化，但不应成为唯一长期 archive key，因为它依赖具体 simulator registry/context。

---

# 17. 推荐存储目录

```text
gigpo_alfworld_reference/
│
├── manifest/
│   ├── run.yaml
│   ├── versions.json
│   ├── hardware.json
│   ├── git_diff.patch
│   └── schema_version.json
│
├── raw/
│   ├── trajectories/
│   │   ├── update_000.parquet
│   │   └── ...
│   ├── occurrences/
│   │   ├── update_000.parquet
│   │   └── ...
│   ├── online_groups/
│   │   ├── update_000.parquet
│   │   └── ...
│   ├── text_events/
│   │   ├── update_000.jsonl.zst
│   │   └── ...
│   └── token_data/
│       └── ...
│
├── updates/
│   ├── update_metrics.parquet
│   └── profiler.parquet
│
├── validation/
│   └── validation_records.parquet
│
├── checkpoints/
│   ├── actor/
│   └── resumable/
│
├── replay_corpus/
│   ├── dev_cases.jsonl.zst
│   └── stress_cases.jsonl.zst
│
└── analysis/
    ├── derived_v1/
    ├── figures/
    └── reports/
```

格式建议：

- **Parquet**：occurrence/group/update/profiler/validation 等结构化表；
- **JSONL.zst**：长文本、嵌套 event log、replay cases；
- **safetensors / trainer native format**：模型与训练状态。

---

# 18. 推荐核心表 Schema

## 18.1 `trajectories.parquet`

```text
run_id
update_id
task_instance_id
task_family
trajectory_id
game_id
game_file
terminal_reward
terminal_success
num_steps
A_E_online
prompt_tokens
response_tokens
actor_generated_tokens
event_log_ref
```

## 18.2 `occurrences.parquet`

```text
occurrence_id
run_id
update_id
task_instance_id
task_family
trajectory_id
step_idx
anchor_obs_raw_ref
anchor_obs_serialized
anchor_key_online
post_obs_raw_ref
admissible_actions_ref
raw_model_response_ref
raw_action_text
parsed_action
executed_action
invalid_action_flag
canonical_action_v0
step_reward
terminal_success
return_to_go_G
online_step_group_id
online_step_group_size
step_group_mean_G
step_group_std_G
A_S_online
A_combined_online
history_ref
remaining_horizon
response_token_ref
old_logprob_ref
ref_logprob_ref
```

## 18.3 `online_groups.parquet`

```text
run_id
update_id
task_instance_id
group_id
anchor_key
anchor_obs_ref
group_size
num_unique_trajectories
occurrence_ids_ref
mean_G
std_G
num_unique_canonical_actions_v0
```

## 18.4 `update_metrics.parquet`

```text
run_id
update_id
epoch_id
train_success
val_success
num_tasks
num_trajectories
actor_generated_tokens
env_steps
actor_loss
ref_kl
clip_fraction
grad_norm
wall_clock_total
```

---

# 19. 字段分级：SOURCE / CACHE / DERIVED

## 19.1 SOURCE：必须原样保存

训练后无法可靠重建：

```text
raw observation
raw model response
executed action
admissible action set
environment reward
game identity
response token ids
```

## 19.2 CACHE：理论上可重算，但要保留在线实际值

```text
anchor_key_online
online step group assignment
A_E_online
A_S_online
combined advantage
old/ref chosen-token logprob
```

## 19.3 DERIVED：离线生成

```text
conflict flag
within-action variance fraction
mixed outcome category
structural anchor flag
Beta posterior
one-step ERV
Exact Batch-ERV
information capacity
hypothetical Q
DP selected plan
```

---

# 20. TraceRecorder 实现架构

建议引入一个统一 recorder，而不是在多处散落 `json.dump`：

```text
TraceRecorder
    on_run_start(...)
    on_update_start(...)
    on_task_start(...)
    on_trajectory_start(...)
    on_occurrence_pre_step(...)
    on_occurrence_post_step(...)
    on_trajectory_end(...)
    on_groups_built(...)
    on_advantages_computed(...)
    on_update_profile(...)
    on_validation(...)
    on_checkpoint(...)
    on_update_end(...)
    on_run_end(...)
```

设计要求：

1. 不修改训练对象；
2. identity 不依赖 Ray/异步完成顺序；
3. 主键稳定；
4. buffer + batch write；
5. 大字段可按 level 开关；
6. 每 update durable flush；
7. schema version 固定；
8. recorder failure 必须显式报错或降级，不能静默丢数据。

---

# 21. 在官方原版 Pipeline 中的 Hook 位置

## 21.1 Environment step hook

在：

```text
env.step(action)
```

之前记录：

```text
pre-action observation
admissible actions
raw model response
parsed action
history ref
```

执行后记录：

```text
executed action
post observation
step reward
done
environment feedback
```

关键要求：**捕获真正送入环境的 `executed_action`，不能只保存 LLM 文本。**

## 21.2 Trajectory finalization hook

episode 结束后回填：

```text
terminal success/reward
trajectory length
return-to-go for every occurrence
```

## 21.3 GiGPO grouping hook

在原版 `build_step_group(anchor_obs, index, ...)` 完成后保存：

```text
group_id -> occurrence_ids
```

及 online group mean/std 等统计。

## 21.4 Advantage hook

在 `compute_gigpo_outcome_advantage` 得到 episode / step / final advantage 后，将值回填到 trajectory 与 occurrence rows。

## 21.5 PPO hook

记录：

```text
token ids
chosen-token old/ref logprob
loss
KL
clip fraction
grad norm
```

## 21.6 Profiler hook

用显式 timer：

```text
with timer("rollout")
with timer("env")
with timer("grouping")
with timer("advantage")
with timer("old_logprob")
with timer("ref_logprob")
with timer("actor_update")
```

各 phase 直接测量，不使用总时间相减推断。

---

# 22. Recorder 性能设计

## 22.1 GPU critical path 只生成轻量 record

推荐：

```text
training process
  -> enqueue compact record
writer thread/process
  -> serialize / compress / write
```

每 update 结束前强制 flush。

## 22.2 大文本统一 blob store

Occurrence 主表仅存：

```text
text_blob_id / ref
```

Observation、history、raw response 写入压缩 text/event shard，避免 Parquet 重复数 KB 字符串。

## 22.3 Token array 独立保存

Token/logprob shard 独立写，Occurrence 表只保存 ref/offset。

---

# 23. 正式完整训练前：Instrumentation Validation Run

先跑：

```text
2-5 policy updates
```

此阶段目标是验证 recorder，不看最终性能。

## 23.1 Group reconstruction invariant

离线用：

```text
task/group identity + exact anchor_obs
```

重建 GiGPO step groups。

要求：

$$
\boxed{
\mathcal I^{offline}(z)=\mathcal I^{online}(z)
}
$$

逐 group 检查 occurrence IDs。

## 23.2 Return-to-go invariant

从 raw step rewards 和 $\gamma$ 重算：

$$
G_i^{offline}
$$

要求与 online 值在 numerical tolerance 内一致。

## 23.3 Advantage invariant

要求：

$$
A_E^{offline}\approx A_E^{online},
$$

$$
A_S^{offline}\approx A_S^{online},
$$

$$
A_{combined}^{offline}\approx A_{combined}^{online}.
$$

## 23.4 Action audit

随机人工审查：

```text
raw response
parsed action
executed action
admissible set
invalid flag
```

保证字段语义正确。

## 23.5 Prefix replay fidelity

随机 occurrence：

```text
load same game
reset
replay previous executed actions
```

检查：

```text
anchor observation/key
admissible action set
done state
step count
```

## 23.6 Recorder overhead

短 run 比较 recorder on/off：

```text
wall-clock/update
rollout time
actor update time
```

得到 logging overhead baseline。

---

# 24. 完整 Reference Run 的在线执行步骤

## Phase A0：冻结实验合同

固定并保存：

```text
GiGPO/verl-agent commit
ALFWorld version
model
launch script + all overrides
group size
train/val size
max steps
rollout/optimizer/GiGPO hyperparameters
hardware
```

## Phase A1：实现 TraceRecorder

完成：

1. stable identity；
2. trajectory event log；
3. env pre/post hook；
4. GiGPO group hook；
5. advantage hook；
6. PPO/profiler hook；
7. checkpoint metadata；
8. durable writer。

## Phase A2：2-5 update validation run

完成第 23 节所有 invariant tests。

## Phase A3：冻结 schema

例如：

```text
schema_version = gigpo_alfworld_ref_v1
```

之后只允许 backward-compatible 新增字段。

## Phase A4：完整 1-seed Reference Run

要求：

- 每 update flush occurrence/group/update shards；
- 定期 actor checkpoint；
- 定期 resumable checkpoint；
- profiler 全程开启；
- validation task-level records 完整。

## Phase A5：Archive integrity audit

训练完成后自动检查：

```text
row counts
primary key uniqueness
missing-field ratios
trajectory-step continuity
group membership integrity
checksum / shard manifest
```

关键恒等式：

$$
\boxed{
\sum_{trajectory} num\_steps
=
\#occurrences
}
$$

## Phase A6：建立 Replay Regression Corpus

从完整 run 分层抽样历史 occurrence，形成固定 dev/stress corpus。

## Phase A7：补齐另外两个 GiGPO seeds

当 schema 和分析脚本稳定后，用完全相同 instrumentation 补齐 3 seeds 总体数据。

---

# Part II：离线分析

# 25. Offline Analysis 的第一原则

离线分析脚本只读取 immutable raw archive，并输出 versioned derived tables。

每个分析结果必须记录：

```text
analysis_version
analysis_git_commit
analysis_seed
input_run_ids
parameter_config
created_time
```

分析中存在随机 root subsampling/tie-breaking 时，必须保存 selected IDs，避免把分析随机性混入 model seed variance。

---

# 26. Archive Integrity 与 GiGPO 完整重建

离线分析的第一个任务不是画图，而是重建原版 GiGPO。

## 26.1 Reconstruct episode groups

用 task/group identity 重建 trajectory group。

## 26.2 Reconstruct step groups

严格使用官方 exact `anchor_obs` grouping 逻辑。

## 26.3 Recompute returns and advantages

重算：

$$
G_i,
\quad
A^E,
\quad
A^S,
\quad
A^{combined}.
$$

## 26.4 生成 reconstruction report

报告：

```text
number of mismatched groups
number of mismatched occurrences
max |G_offline-G_online|
max |A_E_offline-A_E_online|
max |A_S_offline-A_S_online|
max |A_combined_offline-A_combined_online|
```

完整 Reference Run 的验收要求应接近零 mismatch。

---

# 27. Figure 1(a)：Step-Level Group Size Distribution

对每个 update/window 重建所有 step groups，统计：

$$
|\mathcal I(z)|.
$$

建议重点输出：

```text
group size = 1,2,...
mean/median/max group size
fraction singleton
fraction repeated
large-group tail
```

论文主图按照 early / middle / late training stages 汇总；实际 window 可以在离线阶段调整，因此 raw archive 要每 update 保存。

**动机：** 复现并扩展 GiGPO 的 repeated-state 结构观察。

---

# 28. Figure 1(b)：Evidence Quality Decomposition

对 repeated group：

$$
|\mathcal I(z)|\ge2
$$

计算 local return contrast：

$$
\boxed{
\Delta_G(z)
=
\max_i G_i-
\min_i G_i
}
$$

定义 valid/canonical observed action set：

$$
\mathcal C^{obs}(z)=\{u_i\},
$$

$$
K_z=|\mathcal C^{obs}(z)|.
$$

将 group 分成四类。

## Type I — No local credit

$$
\Delta_G(z)\le\epsilon_G.
$$

## Type II — Credit-active but non-comparative

$$
\Delta_G(z)>\epsilon_G,
\qquad K_z<2.
$$

含义：return 有 variation，但 current action 没形成实际竞争。

## Type III — Action-comparative, no observed within-action conflict

$$
\Delta_G(z)>\epsilon_G,
\qquad K_z\ge2,
$$

且同一 action 未出现相反符号的有效 $A^S$。

## Type IV — Action-comparative with within-action sign conflict

存在 $u$：

$$
\exists i,j\in\mathcal I(z,u):
\quad
A_i^S>\epsilon_A,
\quad
A_j^S<-\epsilon_A.
$$

输出 stacked proportion by training stage。

**动机：** 展示：

$$
\boxed{
\text{Structural recurrence}
\neq
\text{uniformly useful decision evidence}
}
$$

---

# 29. Within-Action Noise / Conflict 分析

## 29.1 Conflict rate

建议至少报告两个 denominator：

### All repeated groups

$$
r_{conflict}^{all}
=
\frac{\#conflict\ repeated\ groups}
{\#repeated\ groups}.
$$

### Groups with repeated same action

令：

$$
\mathcal Z_{repeat-act}
=
\{z:\exists u,n_{z,u}\ge2\}.
$$

则：

$$
\boxed{
r_{conflict}^{cond}
=
\frac{\#\{z:\operatorname{Conflict}(z)=1\}}
{|\mathcal Z_{repeat-act}|}
}
$$

第二个更适合比较训练阶段，因为只有同一动作至少出现两次才有机会观察 conflict。

## 29.2 Within-action variance fraction

总体 variation：

$$
T_z=\sum_i(G_i-\bar G_z)^2.
$$

动作内部 variation：

$$
W_z
=
\sum_u\sum_{i\in\mathcal I(z,u)}
(G_i-\bar G_{z,u})^2.
$$

定义：

$$
\boxed{
\eta_z=
\frac{W_z}{T_z+\epsilon}
}
$$

**动机：** sign conflict 是离散诊断，$\eta_z$ 提供连续 robustness metric。

## 29.3 Action-count imbalance

保存/派生：

```text
n_z,u distribution
max action share
entropy/Gini of action counts
number of singleton actions
```

**动机：** 判断 local evidence 是否被 dominant action 或 rare lucky action 主导。

---

# 30. Mixed Outcome、All-Success、All-Failure

离线分类：

```text
all_success
all_failure
mixed_terminal_outcomes
```

**动机：** 这是非常直观的 supplementary statistic，可支持论文引入讨论。

同时保留与 $\Delta_G$ 的区别：当 $\gamma<1$ 或成功步数不同时，即使 all-success，也可能有不同 return-to-go。

---

# 31. Invalid Action 与 Loop 分析

利用 raw/executed action、admissible set 与 repeated state：

```text
invalid_action_ratio
invalid_action_ratio_by_update
invalid_action_ratio_by_task_family
invalid_action_ratio_inside_large_groups
consecutive_invalid_action_length
same-state repeat count
within-trajectory repeat count
```

对 anchor $z$：

$$
N_{cross}(z)
=
|\{r:\exists t,(r,t)\in\mathcal I(z)\}|,
$$

$$
N_{within}(z)
=
|\mathcal I(z)|-N_{cross}(z),
$$

$$
\boxed{
\rho_{loop}(z)=
\frac{N_{within}(z)}{|\mathcal I(z)|}
}
$$

**动机：** 分辨大 group 来自有意义的状态碰撞、invalid action 后 observation 不变，还是单轨迹 loop。

---

# 32. Root-Count Subsampling：模拟不同 Natural-Root 数量

Reference GiGPO 每 task 有完整 group_size=8 natural roots，因此可以对：

$$
R\in\{2,3,4,5,6,7,8\}
$$

进行离线 subsampling。

每个 $R$ 重新计算：

```text
unique anchor count
repeated anchor count
action-comparative anchor count
structural branch-anchor count
group-size distribution
action evidence counts
```

建议每个 task/update/R 做多次随机子集，例如：

```text
100 random R-subsets
```

并保存：

```text
analysis_seed
subsample_id
selected_root_ids
```

**动机：** 直接回答：如果 BACE 只分配 $R$ 条 natural roots，是否已有足够 local evidence 可进入 refinement。

---

# 33. 离线构造 BACE Local Beta Posterior

对 subsampled root set，在每个 observed $(z,u)$ 上统计：

```text
natural_occurrence_count
natural_success_count
natural_failure_count
```

给定当前 local prior：

$$
\alpha_{z,u}=\alpha_0+S_{z,u},
$$

$$
\beta_{z,u}=\beta_0+F_{z,u}.
$$

派生：

```text
posterior mean
posterior variance
posterior concentration
```

**动机：** 无需真正 branch，即可评估自然 evidence 的局部不确定结构。

---

# 34. Exact One-Step ERV / Exact Batch-ERV 离线分析

对每个 anchor 的 observed actions，计算当前 BACE 版本的 exact value。

## 34.1 One-step ERV

对动作 $u$ 计算：

$$
\operatorname{ERV}(z,u).
$$

## 34.2 Exact Batch-ERV

主配置 $L_{max}=2$ 时枚举：

```text
0 samples
[u]
[u,u]
[u,v]
```

得到：

$$
V_z^{(0)},
\quad
V_z^{(1)},
\quad
V_z^{(2)}.
$$

以及：

$$
\delta_z^{(1)}=V_z^{(1)},
$$

$$
\delta_z^{(2)}=V_z^{(2)}-V_z^{(1)}.
$$

## 34.3 Information capacity

给定 $\tau_{BERV}$：

$$
c(z)
=
\mathbf1[\delta_z^{(1)}\ge\tau_{BERV}]
+
\mathbf1[\delta_z^{(2)}\ge\tau_{BERV}],
$$

$$
\boxed{
C_g=\sum_z c(z)
}
$$

分析：

```text
capacity vs R
capacity vs update
capacity vs task family
capacity vs tau_BERV
fraction tasks with C=0
fraction tasks with C>=1/2/3/...
```

**动机：** 在正式 BACE 训练前估计 capacity correction 的触发频率和合理参数范围。

---

# 35. Exact Global DP 离线模拟

对 hypothetical quota：

$$
Q\in\{1,2,3,4,5,6\},
$$

求：

$$
\max_{m_z\in\{0,1,2\}}
\sum_zV_z^{(m_z)}
$$

subject to：

$$
\sum_zm_z=Q.
$$

保存 derived：

```text
selected anchor ids
selected action multiset
objective value
number of tied optima
per-anchor allocation
```

分析：

- allocation concentration；
- same-action repeated sampling frequency；
- tie frequency；
- nonzero-value quota availability。

---

# 36. Competence Controller 离线模拟

GiGPO reference run 全部是 natural roots，非常适合形成 task-family success history。

按 family/update 统计：

$$
S_{c,k}^{root},
\qquad
F_{c,k}^{root}.
$$

离线扫描：

```text
p0
kappa0
lambda_hist
tau_comp
kappa bounds
```

得到：

$$
q_{c,k}=P(\Phi_{c,k}>\tau_{comp}),
$$

以及：

$$
\bar Q_{c,k}
=
\operatorname{round}[(B-R_{min})q_{c,k}].
$$

必须输出：

```text
q by update/family
planned Q distribution
fraction Q=0/1/.../max
first update with nonzero Q
quota oscillation frequency
family-to-family variance
```

**动机：** 在真实 BACE rollout 前排除 early training 过早 branch、长期不 branch 或 quota 振荡过强的参数。

---

# 37. Planned Quota 与 Offline Capacity 的联合模拟

将第 36 节 competence controller 输出的：

$$
\bar Q_g
$$

与第 34 节从 natural roots 得到的：

$$
C_g(R)
$$

结合，离线模拟 capacity correction：

```text
planned_Q
available_capacity
predicted_actual_Q
predicted_root_count
number_of_root_corrections
```

可生成：

```text
planned Q vs actual feasible Q
correction frequency vs training stage
correction frequency vs task family
```

**动机：** 提前判断 family-level readiness 与 instance-level local evidence 是否匹配。

---

# 38. Replay Regression Corpus

从 Reference Archive 固定抽样 historical occurrences。

建议分层：

```text
early / mid / late
short / medium / long prefix
successful / failed trajectory
small / large step group
loop-heavy / normal
near-invalid-action cases
```

建议：

```text
~1k dev cases
~10k stress cases
```

每个 case 保存：

```text
case_id
game_file
prefix executed actions
expected pre-action observation
expected anchor key
expected admissible action set
expected done
expected step count
remaining horizon
```

## 38.1 Prefix replay test

要求：

$$
anchor\_key_{restore}
=
anchor\_key_{recorded}
$$

和：

$$
\mathcal A_{adm}^{restore}
=
\mathcal A_{adm}^{recorded}.
$$

## 38.2 Snapshot regression

未来 StateID snapshot 与 prefix replay 对同一 case 比较：

```text
observation
anchor key
admissible actions
done
remaining horizon
next-action transition
```

**动机：** 将这次 baseline 直接转化为 snapshot/replay 工程的长期回归测试集。

---

# 39. System Baseline 的离线报告

从 profiler 生成：

## 39.1 每 update phase breakdown

```text
rollout
env
grouping
advantage
old logprob
ref logprob
actor update
validation
I/O
```

## 39.2 效率指标

```text
seconds/update
actor generated tokens/update
env steps/update
GPU-hours/update
performance/token
performance/wall-clock
```

## 39.3 Recorder overhead

```text
trace time / total update time
bytes written/update
writer queue pressure
```

**动机：** 未来 BACE 新增 branch wave 后，可以用完全相同 schema 比较新增开销究竟来自 generation、env restore、Batch-ERV 还是 trainer。

---

# 40. Low / Mid / High Competence Checkpoint 选择

基于 Reference Run 的：

```text
validation performance
natural-root training success
task-family success
```

离线选择代表性 checkpoints。

推荐同时保存两套标记：

```text
iteration-based stage        # 如接近 GiGPO 论文的早/中/晚位置
competence-based stage       # 按实际自然成功率选择
```

**动机：** 未来固定 topology mechanism experiment 可以从同一 resumable checkpoint clone。

---

# 41. 离线分析代码组织

推荐：

```text
analysis/
├── load_reference.py
├── validate_archive.py
├── reconstruct_groups.py
├── recompute_returns.py
├── recompute_advantages.py
├── canonicalize_actions.py
├── analyze_group_sizes.py
├── analyze_evidence_quality.py
├── analyze_within_action_noise.py
├── analyze_invalid_and_loops.py
├── subsample_roots.py
├── build_local_posteriors.py
├── compute_exact_erv.py
├── compute_exact_berv.py
├── simulate_capacity.py
├── solve_batch_dp.py
├── simulate_competence_controller.py
├── simulate_topology_correction.py
├── build_replay_corpus.py
├── analyze_system_profile.py
└── make_figures.py
```

每个分析脚本输出 versioned Parquet，再由 figure script 读取；不要只输出 PNG/PDF。

例如：

```text
analyze_evidence_quality.py
  -> evidence_quality_v1.parquet

make_figures.py
  -> figure1a.pdf
  -> figure1b.pdf
```

---

# 42. 优先生成的离线数据产品

完整第一个 seed 后，优先生成以下结果。

## 42.1 Motivation Audit

- Figure 1(a) group size distribution；
- Figure 1(b) evidence quality decomposition；
- within-action conflict；
- invalid-action / loop supplementary statistics。

## 42.2 BACE Feasibility Report

- repeated/action-comparative anchors vs root count；
- Exact Batch-ERV capacity vs root count；
- capacity vs early/mid/late；
- DP allocation concentration；
- tie frequency。

## 42.3 Competence Controller Report

- $q$ traces；
- planned $Q$ distribution；
- first nonzero branch point；
- quota oscillation；
- predicted capacity corrections。

## 42.4 Replay Report

- prefix replay exact match rate；
- admissible-set exact match rate；
- failure case taxonomy。

## 42.5 System Report

- GiGPO phase timing；
- token/env-step cost；
- GPU utilization；
- logging overhead。

---

# 43. 哪些问题可以完全离线回答

Reference Archive 可以回答：

```text
GiGPO group size dynamics
repeated anchor abundance
action diversity
evidence-quality categories
within-action conflict/noise
invalid/loop contribution
root-count vs anchor availability
natural-evidence Beta posterior
one-step ERV
Exact Batch-ERV
information capacity
global DP plan
competence controller traces
predicted root/branch quota
replay fidelity
system baseline
```

这些问题只依赖原版 natural-root evidence，不需要重新训练 GiGPO。

---

# 44. 哪些问题需要真实 BACE rollout

Reference Archive 的定位是：

$$
\boxed{
\text{where BACE would want to experiment}
}
$$

真实 BACE 才能给出：

$$
\boxed{
\text{what happens after the new experiment}
}
$$

因此以下结果来自后续 BACE run：

```text
actual branch terminal outcome
posterior after real branch
realized ERV/regret reduction
branch suffix new anchor collisions
actual BACE policy improvement
root-wave / branch-wave scheduling cost
snapshot restore in active training
```

---

# 45. 最低不可删减保存集

如果工程周期受限，P0 最低集合如下。

## Run

```text
full resolved config
commit hashes
seed
hardware
```

## Trajectory

```text
task id/family
game file
trajectory id
terminal success/reward
num steps
A_E_online
```

## Occurrence

```text
step index
pre-action observation
admissible actions
raw model response
executed action
post-action observation
step reward
terminal success
return-to-go
online group id
A_S_online
combined advantage
history/replay reference
```

## Group

```text
anchor key
group members
group size
mean/std G
```

## Training/System

```text
generated tokens
env steps
loss/KL/clip fraction
grad norm
phase timing
validation task records
actor/full checkpoints
```

---

# 46. 保存优先级总表

## P0：必须

- raw pre/post observations；
- raw/executed actions；
- admissible actions；
- terminal outcome；
- occurrence identity；
- reward sequence / return-to-go；
- online GiGPO group；
- $A^E,A^S,A$；
- replay metadata；
- generated tokens；
- env steps；
- phase timing；
- actor/full checkpoints；
- config/commit/seed/hardware。

## P1：强烈推荐

- response token IDs；
- action token span；
- chosen-token old/ref logprobs；
- PPO clip fraction；
- KL；
- gradient norms；
- per-task validation；
- GPU utilization/memory；
- raw conversation event log；
- recorder overhead。

## P2：扩展诊断

- sparse diagnostic top-k logits；
- detailed Ray scheduling events；
- per-step GPU timing；
- environment internal symbolic-state dump（若方便且稳定）。

---

# 47. 正式实施顺序

## 数据采集与保存阶段

```text
1. 固定官方 GiGPO ALFWorld commit + launch config
2. 写 schema_version = gigpo_alfworld_ref_v1
3. 实现 TraceRecorder
4. 在 env step / trajectory / grouping / advantage / PPO / profiler 加 hook
5. 运行 2-5 updates instrumentation validation
6. 离线重建 group / G / advantages，与 online 对齐
7. 做 prefix-replay fidelity test
8. 测 recorder overhead
9. 冻结 schema
10. 完整运行第一个 instrumented GiGPO seed
11. 训练结束做 archive integrity audit
12. 构造 replay regression corpus
13. 按相同 schema 补齐另外两个 seeds
```

## 离线分析阶段

```text
1. validate_archive
2. reconstruct_groups / recompute_returns / recompute_advantages
3. Figure 1(a): group-size distribution
4. Figure 1(b): evidence-quality decomposition
5. within-action conflict / variance / invalid / loop analyses
6. R=2..8 root-count subsampling
7. local Beta posterior reconstruction
8. Exact ERV / Batch-ERV / information capacity
9. exact global DP simulation
10. competence-controller simulation
11. planned-Q + capacity-correction joint simulation
12. replay/snapshot regression analysis
13. system profile report
14. choose low/mid/high resumable checkpoints
15. freeze derived tables and generate paper figures
```

---

# 48. 最终验收标准

一次完整 Reference Run 只有同时满足以下条件才算可长期复用。

## 48.1 科学可重算性

从 archive 能离线重建：

$$
\boxed{
\text{GiGPO episode groups + step groups + returns + advantages}
}
$$

并与 online 值在数值容差内一致。

## 48.2 Figure 1 完全离线生成

无需 GPU rollout，即可生成：

- group-size distribution；
- evidence-quality decomposition；
- within-action conflict/variance；
- invalid/loop supplementary statistics。

## 48.3 BACE pre-analysis 可离线完成

能够由 natural roots 计算：

- structural anchors；
- local Beta posterior；
- Exact Batch-ERV；
- information capacity；
- global DP selection；
- competence $q$；
- hypothetical planned/actual $Q$。

## 48.4 Replay 可验证

历史 occurrence 能通过 same-game + action-prefix 恢复，并匹配：

```text
anchor observation/key
admissible actions
done state
remaining horizon
```

## 48.5 系统公平性可审计

能够严格报告：

$$
\boxed{
\text{terminal leaves}
+
\text{generated actor tokens}
+
\text{environment steps}
+
\text{wall-clock}
+
\text{GPU-hours}
}
$$

作为未来 GiGPO vs BACE 的统一效率基线。

---

# 49. 参考实现位置

正式实现建议基于当前实际使用 commit 固定这些入口：

1. `verl-agent` repository  
   <https://github.com/langfengQ/verl-agent>

2. 官方 GiGPO ALFWorld 训练脚本  
   `examples/gigpo_trainer/run_alfworld.sh`  
   <https://github.com/langfengQ/verl-agent/blob/master/examples/gigpo_trainer/run_alfworld.sh>

3. GiGPO 核心 advantage/grouping  
   `gigpo/core_gigpo.py`  
   <https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py>

4. ALFWorld 环境 wrapper  
   `agent_system/environments/env_package/alfworld/envs.py`  
   <https://github.com/langfengQ/verl-agent/blob/master/agent_system/environments/env_package/alfworld/envs.py>

5. PPO 主入口  
   `verl/trainer/main_ppo.py`

6. Ray/PPO trainer  
   `verl/trainer/ppo/ray_trainer.py`

---

# 50. 一句话总结

这次原版 GiGPO ALFWorld 完整训练的核心目标不是“多记一些日志”，而是建立一份**可重算、可回放、可审计、可长期复用的 Reference Archive**：

$$
\boxed{
\text{保存 raw trajectory / occurrence / group / optimizer / system facts，}
}
$$

$$
\boxed{
\text{让论文动机、anchor evidence、BERV capacity、controller topology、replay fidelity 与系统效率都可以离线重新定义和计算。}
}
$$

在工程上，最重要的第一步是先用 2-5 个 updates 验证 `TraceRecorder` 能完整重建原版 GiGPO 的 group、return 和 advantage，再冻结 schema 并开始完整 baseline 训练。
