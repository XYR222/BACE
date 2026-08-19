# BACE-GiGPO 代码实现架构与 Replay 工程规范

> 面向 `verl-agent` 的实现参考文档  
> 版本：2026-08-06  
> 状态：主算法已冻结；本文只规定工程实现，不重新讨论方法选择

---

## 0. 文档目的

本文将 BACE-GiGPO 的最终算法翻译为一套可执行的工程设计，供后续实现者在 `verl-agent` 上开发、调试和复现实验。

本文重点回答：

1. 哪些现有仓库和文件可以直接复用；
2. BACE 应在 `verl-agent` 的哪个层级实现；
3. natural root、anchor、branch、leaf 和训练 occurrence 如何组织；
4. ALFWorld 和 WebShop 的中间状态 replay 如何准确实现；
5. 环境状态、LLM 可见历史和 copied CoT--action 如何同时恢复；
6. root/branch 动态调度如何跨任务批处理；
7. all-leaf advantage 与 unique-segment training 如何落到 `DataProto`；
8. 如何减少 replay、序贯 branch 和树数据带来的额外耗时；
9. 应建立哪些单元测试、集成测试和 profiler；
10. 第一版应按什么顺序实现。

本文采用以下最终算法语义：

- `verl-agent`/GiGPO 是唯一训练代码基座；
- 每个实例固定 terminal-leaf budget $B$；
- pilots 后规划 root/branch quota；
- capacity correction 完成后冻结 natural-root backbone；
- anchor 使用原始 GiGPO 的 exact pre-action observation key；
- candidate actions 只来自 natural roots 中真实执行过的 canonical actions；
- anchor 按实际 ERV-softmax 动作分布下的期望 ERV 选择；
- branch origin 必须是自然执行过选中动作的 concrete occurrence；
- 恢复环境后复制该 occurrence 的完整自然 CoT--action，再由冻结策略生成新 suffix；
- replay prefix 不训练；copied origin 和新 suffix 训练；
- natural roots 与 branch leaves 统一计算 all-leaf trajectory advantage；
- 所有可训练 occurrences 进入一个统一 GiGPO/PPO loss；
- action-aggregated local credit 作为可切换增强项。

---

# 1. 总体实现原则

## 1.1 单一代码基座

BACE 应在 `langfengQ/verl-agent` 当前 GiGPO 代码上新增一个独立 recipe，而不是整体 fork 或替换为其他树方法仓库。

推荐目录：

```text
recipe/bace_gigpo/
├── __init__.py
├── config/
│   ├── bace_alfworld.yaml
│   └── bace_webshop.yaml
├── main_bace_gigpo.py
├── bace_ray_trainer.py
├── bace_rollout_coordinator.py
├── bace_tree.py
├── bace_anchor_index.py
├── bace_posterior.py
├── bace_advantage.py
├── bace_batch_assembler.py
├── replay/
│   ├── base.py
│   ├── alfworld.py
│   ├── webshop.py
│   └── validator.py
├── diagnostics/
│   ├── metrics.py
│   ├── trace_writer.py
│   └── tree_visualizer.py
└── scripts/
    ├── run_alfworld_train.sh
    ├── run_alfworld_eval.sh
    ├── run_webshop_train.sh
    └── run_webshop_eval.sh
```

原则：

- 尽量不修改 `verl/` 核心目录；
- 尽量不修改原始 `gigpo/core_gigpo.py`；
- BACE 的新逻辑集中在 recipe、rollout coordinator、replay adapter 和 advantage assembler；
- 原始 GiGPO 必须可以通过配置继续运行，以便逐阶段比较。

## 1.2 方法层与系统层分离

算法上，同一任务内部的 branches 必须顺序执行：

```text
branch 1 outcome
→ posterior update
→ recompute ERV and U(z)
→ branch 2
```

系统上，不同任务的同一轮 branch 应合成批次执行：

```text
Task 1: next branch
Task 2: next branch
Task 3: next branch
...
→ one replay batch
→ one suffix-generation batch
```

因此核心原则为：

$$
\boxed{
\text{task 内顺序，task 间并行。}
}
$$

## 1.3 统计上的完整 leaf 与训练上的 unique segment 分离

一条 branch leaf 概念上是：

```text
root prefix + copied origin + new suffix
```

但训练时不能把完整 prefix 再复制一次。

必须维护两个视图：

### Leaf view

用于：

- terminal reward；
- all-leaf normalization；
- root/branch lineage；
- 分析和可视化。

### Trainable occurrence view

用于 PPO：

- natural-root responses；
- copied branch-origin response；
- new branch-suffix responses。

机械 replay prefix 只负责恢复环境，不产生训练 occurrence。

---

# 2. 可复用仓库与代码范围（github链接在/home/naie/work/work-BACE/BACE-work/BACE相关代码仓库_GitHub链接.md）

## 2.1 `langfengQ/verl-agent`：唯一代码基座

### 直接复用

| 需求 | 现有代码位置 | 使用方式 |
|---|---|---|
| 多轮 agent rollout | `agent_system/multi_turn_rollout/` | 保留模型生成、环境交互和 batch collector |
| GiGPO anchor grouping | `gigpo/core_gigpo.py` | 复用 `to_hashable` 与 exact observation grouping 语义 |
| GiGPO advantage | `gigpo/core_gigpo.py` | 作为 occurrence-level local-credit 基线 |
| PPO/Ray trainer | 原 GiGPO trainer 与 `verl.trainer.ppo` | 复用 actor、ref、KL、old log-prob、optimizer |
| ALFWorld/WebShop wrapper | `agent_system/environments/env_package/` | 复用环境和 prompt/action parser |
| `DataProto` 管线 | `verl.DataProto` | 承载新增 metadata 与统一训练 batch |
| sequence balancing | trainer 中的 `adjust_batch` 等 | advantage 计算后再做长度平衡 |

`gigpo/core_gigpo.py` 当前的 exact grouping 逻辑是：同一任务组内对 `anchor_obs` 调用 `to_hashable`，相同 key 获得同一 step-group ID。BACE 的 natural-root anchor index 应保持这一语义。

### 需要扩展

- rollout 从“一次生成固定 $n$ 条 roots”扩展为多阶段 coordinator；
- 环境 wrapper 增加 `restart_from_turn`；
- rollout result 增加 tree/replay metadata；
- advantage 计算扩展为 all-leaf + local credit；
- copied origin response 能作为新 occurrence 复用 token IDs 和 old log-probs。

---

## 2.2 `recipe/hgpo`：recipe 与 trainer 扩展模板

### 推荐复用

- recipe 自包含目录结构；
- 独立 `main_*.py`、trainer、config 和运行脚本；
- 在 trainer 中注册新 advantage estimator；
- 在 `compute_advantage()` 之前保留原始 group order；
- advantage 完成后再执行 batch balancing。

`recipe/hgpo/hgpo_ray_trainer.py` 明确要求 group-based advantage 在 `adjust_batch()` 之前计算。这一点对 BACE 更重要，因为 all-leaf、anchor group 和 action group 都依赖完整任务组与 tree metadata。

### 不复用

- history-aware anchor key；
- HGPO 自身的多历史长度聚合公式。

原因：BACE 主版本按 exact current observation grouping，与原始 GiGPO 对齐。

---

## 2.3 `recipe/GraphGPO`：tree/graph 数据组织与诊断

### 推荐借鉴

- state node 和 action edge 的 ID 组织；
- trajectory-to-graph mapping；
- 跨 trajectory 的 state/action aggregation；
- 图结构可视化；
- trainer 中额外 metadata 的传递方式。

BACE 可以复用其“节点—边—轨迹成员关系”的思路，建立：

```text
Task
 ├── RootTrajectory
 │    ├── Occurrence
 │    └── Occurrence
 └── BranchLeaf
      ├── CopiedOriginOccurrence
      └── NewSuffixOccurrences
```

### 不复用

- shortest-path reward；
- reverse Dijkstra；
- GraphGPO 的图式 step-return。

原因：BACE 的局部数据采集由 ERV 控制，优化仍采用最终方案中的 all-leaf GiGPO credit。

---

## 2.4 `yuki-younai/PivoARL`：在线 replay 的主要代码来源

PivoARL 的 replay 不是内存快照，而是：

```text
same game/session
→ restart/reset
→ replay recorded environment actions
→ truncate/copy model memory
→ continue rollout
```

### ALFWorld 可借鉴文件

```text
agent_system/environments/env_package/alfworld/envs.py
agent_system/environments/env_package/alfworld/env_manager.py
scripts/test/legacy/test_alfworld_restart.py
scripts/test/legacy/test_alfworld_manager_restart_from_turn.py
```

关键实现：

- `AlfworldWorker.restart()`；
- `AlfworldWorker.restart_from_turn(target_turn, replay_actions)`；
- `AlfworldEnvs.restart_from_turn(target_turns, replay_actions)`；
- `action_journal_per_env`；
- per-env target turn；
- memory truncate/deepcopy；
- early terminal failure。

### WebShop 可借鉴文件

```text
agent_system/environments/env_package/webshop/envs.py
agent_system/environments/env_package/webshop/env_manager.py
```

关键实现：

- worker 记录 `_last_session_idx`；
- `restart()` 重新 `reset(session=_last_session_idx)`；
- `restart_from_turn()` 重放页面操作；
- 每个 worker 独立选择 replay 长度；
- 恢复 `available_actions`。

### 需要改写

PivoARL 的语义为：

```text
失败轨迹 → reflection → pivotal turn → new retry attempt
```

BACE 的语义为：

```text
natural roots
→ ERV anchor/action selection
→ natural origin occurrence
→ copied natural CoT-action
→ new continuation
```

因此只移植 replay infrastructure，不移植：

- reflection；
- pivotal-turn parser；
- previous-attempt prompt；
- retry-specific reward/credit；
- 新 attempt 的反思注入。

---

## 2.5 `HappynessI/Prefix_GRPO`：replay validation 与数据审计

### 推荐借鉴

```text
scripts/build_data/build_alfworld_prefix_rl_change_top3.py
```

其中值得复用的概念：

- `prefix_actions`；
- environment reset key；
- `expected_cut_observation`；
- `replay_cut_observation`；
- `replay_done`；
- `replay_error`；
- `replay_category`；
- 只允许 validated records 进入训练；
- 同一 root lineage 的多个 cut 可按 prefix 长度排序，增量重放共同前缀。

### 不复用

- teacher-prefix selection；
- entropy-change cut selection；
- historical assistant prefix token training；
- prefix old-log-prob sidecar 的训练目标。

原因：BACE 的 replay prefix 是机械环境恢复，不训练；BACE 只训练 copied branch origin 与新 suffix。

---

## 2.6 其他仓库的定位

| 方法 | 使用定位 | 不作为主代码来源的原因 |
|---|---|---|
| BPO | branching 方法和 matched-compute baseline | 当前缺少可充分审计的 ALFWorld/WebShop 官方实现 |
| 3SPO | variable rollout scheduling 和状态统计参考 | 公开代码中未确认完整中间状态 rewind 路径 |
| TRACE | root/prefix 预算分配实验参考 | 主要是文本/ReAct prefix continuation，不直接解决环境隐状态恢复 |
| Tree-GRPO | tree flattening、parent-child metadata | 主要面向搜索/QA 工具环境，不是 ALFWorld replay adapter |
| EnvRL | 低侵入扩展 GiGPO 的代码 diff 参考 | 不提供我们需要的完整 online branch replay |

---

# 3. BACE 软件架构

## 3.1 顶层组件

```text
BaceRayTrainer
│
├── TrajectoryCollector / ActorRolloutRef
│
├── BaceRolloutCoordinator
│   ├── RootBatchScheduler
│   ├── CapacityCorrectionScheduler
│   ├── BranchFrontierScheduler
│   └── TaskRolloutState[]
│
├── AnchorIndex
│   ├── exact observation grouping
│   ├── observed action grouping
│   └── concrete origin pool
│
├── PosteriorEngine
│   ├── task competence posterior
│   ├── local Beta posterior
│   ├── Bayes regret / ERV
│   ├── ERV-softmax
│   └── anchor utility cache
│
├── ReplayAdapter
│   ├── AlfworldReplayAdapter
│   ├── WebshopReplayAdapter
│   └── ReplayValidator
│
├── BaceTreeStore
│   ├── immutable root event logs
│   ├── branch records
│   ├── leaf records
│   └── trainable occurrences
│
├── BaceAdvantageEngine
│   ├── all-leaf A^E
│   ├── occurrence A^S
│   └── optional action-aggregated A^S
│
└── BaceBatchAssembler
    ├── unique segments
    ├── masks and old log-probs
    └── unified DataProto
```

## 3.2 顶层生命周期

```text
freeze pi_old
│
├── Phase A: pilot roots
├── Phase B: topology planning
├── Phase C: planned roots
├── Phase D: capacity-correction roots
├── freeze root backbone and origin pools
├── Phase E: sequential branch rounds
├── Phase F: all-leaf + local advantages
├── Phase G: unique-segment DataProto
├── one unified PPO update
└── update task-family history from natural roots only
```

---

# 4. 核心数据结构

以下是逻辑 schema，不要求逐字照搬为 Python dataclass，但字段语义必须保留。

## 4.1 RootEventLog

一条 natural root 只保存一份不可变 event log：

```text
RootEventLog
    task_id
    task_family
    episode_group_id
    root_id
    environment_reset_key
    initial_observation
    events[]
    terminal_reward
    won
```

每个 event：

```text
RootEvent
    step_index
    pre_action_observation
    anchor_key
    prompt_messages_or_memory_delta
    raw_model_response
    response_token_ids
    response_attention_mask
    response_loss_mask
    old_log_probs
    parsed_environment_action
    canonical_action
    admissible_or_available_actions
    post_action_observation
    reward
    done
    remaining_horizon
```

### 设计理由

- replay 使用 `parsed_environment_action`；
- 模型训练使用 `raw_model_response` 和 token 数据；
- anchor grouping 使用 `pre_action_observation`；
- branch origin 复用 response token IDs 和 old log-probs；
- 一个 root 的完整历史不为每个 occurrence 重复复制。

## 4.2 AnchorRecord

```text
AnchorRecord
    task_id
    anchor_id
    anchor_key
    occurrence_ids[]
    observed_action_ids[]
    current_utility
    current_regret
    branch_count
    max_branch_count
```

## 4.3 OriginOccurrence

```text
OriginOccurrence
    occurrence_id
    root_id
    step_index
    anchor_id
    action_id
    environment_reset_key
    replay_prefix_length
    remaining_horizon
```

注意：`OriginOccurrence` 只保存索引。具体 prefix、memory、response 和 old log-probs 从对应 `RootEventLog` 中按引用取得。

## 4.4 LocalPosterior

```text
LocalPosterior
    task_id
    anchor_id
    action_id
    alpha
    beta
    natural_success_count
    natural_failure_count
    branch_success_count
    branch_failure_count
    version
```

## 4.5 ReplayRequest

```text
ReplayRequest
    request_id
    task_id
    branch_id
    origin_occurrence_id
    environment_reset_key
    target_turn
    parsed_action_prefix[]
    expected_anchor_key
    expected_observation
    expected_action_set
    selected_canonical_action
    remaining_horizon
```

## 4.6 ReplayResult

```text
ReplayResult
    request_id
    replay_ok
    restored_observation
    restored_anchor_key
    restored_action_set
    restored_done
    error_type
    error_message
    replay_step_count
    replay_latency
```

## 4.7 BranchRecord

```text
BranchRecord
    branch_id
    task_id
    parent_root_id
    origin_occurrence_id
    anchor_id
    action_id
    copied_origin_occurrence_id
    suffix_occurrence_ids[]
    terminal_reward
    won
    replay_result
```

## 4.8 LeafRecord

```text
LeafRecord
    leaf_id
    task_id
    source_type = ROOT | BRANCH
    root_id
    branch_id_or_none
    terminal_reward
    trajectory_advantage
```

LeafRecord 不保存完整 token path。它只表示统计上的完整 root-to-terminal outcome。

## 4.9 TrainableOccurrence

```text
TrainableOccurrence
    occurrence_id
    task_id
    leaf_id
    source_type = ROOT | BRANCH_ORIGIN | BRANCH_SUFFIX
    anchor_id
    action_id
    prompt/reference_to_prompt
    response_token_ids
    old_log_probs
    response_loss_mask
    step_return
    trajectory_advantage
    local_advantage
    final_advantage
```

---

# 5. Rollout Coordinator

## 5.1 TaskRolloutState

每个 task instance 维护独立状态：

```text
TaskRolloutState
    task_id
    competence_prior
    competence_posterior
    planned_root_count
    planned_branch_count
    final_root_count
    final_branch_count
    root_ids[]
    structural_anchor_ids[]
    effective_anchor_ids[]
    local_posterior_table
    branch_count_by_anchor
    phase
```

## 5.2 Phase A：pilots

对 batch 中所有任务同时生成 $N_{\mathrm{pilot}}$ 条 roots。

系统层面将其视为普通 GiGPO rollout batch：

```text
G tasks × N_pilot roots
→ one multi-turn rollout collection
```

pilot 完成后：

- 将其加入 immutable root store；
- 用 terminal outcomes 计算 instance competence posterior；
- task-family prior 在当前 batch 内保持冻结。

## 5.3 Phase B：计划 topology

对每个 task 独立计算：

```text
q_g
planned branch quota Q_bar_g
planned root count R_g = B - Q_bar_g
```

只进行 CPU 数值计算，不调用模型。

## 5.4 Phase C：补齐 planned roots

将所有仍缺少 planned roots 的任务合成一个 batch：

```text
Task 1 needs 4 roots
Task 2 needs 2 roots
Task 3 needs 6 roots
→ concatenate root requests
→ generate together
```

## 5.5 Phase D：capacity correction

每轮执行：

1. 对所有当前任务构造 natural-root anchors；
2. 初始化/更新 local posteriors；
3. 计算初始 ERV、动作分布和 $U_{g,0}(z)$；
4. 检查：
   $$
   L_{\max}|\mathcal Z_g^{\mathrm{eff}}|\ge Q_g;
   $$
5. 收集所有 capacity 不足且 $Q_g>0$ 的 tasks；
6. 每个此类 task 增加一条 root，并减少一个 branch slot；
7. 将这些 extra-root requests 合成一个 batch；
8. 重复检查。

算法上每个 task 一次只转换一个 slot，系统上所有 tasks 一起生成该轮 extra roots。

## 5.6 冻结边界

capacity correction 完成后，必须冻结：

- natural root store；
- root backbone；
- structural/effective anchor set；
- observed action sets；
- origin occurrence pools；
- final branch quota。

branch suffix 新发现的状态可以参与最终 training groups，但不能在同一 rollout batch 中成为新的 branch origin。

## 5.7 Phase E：branch frontier rounds

每一轮：

1. 每个仍有 branch quota 的 task 重算其 active anchors 的 $U_{g,b}(z)$；
2. 选择当前 utility 最大且未达到 $L_{\max}$ 的 anchor；
3. 从该 anchor 的 ERV-softmax 中采样 action；
4. 从真实执行过该 action 的 natural occurrences 中采样 origin；
5. 生成一个 `ReplayRequest`；
6. 将所有 tasks 的 requests 合成 replay batch；
7. replay 成功的 requests 进入 suffix-generation batch；
8. 生成 copied origin 后的新 suffix；
9. 写入 BranchRecord 和 LeafRecord；
10. 更新对应 local posterior；
11. 进入下一轮。

第一版采用 round-synchronous 设计。后续只有 profiling 证明 GPU idle 较高时，才改为异步 frontier queue。

---

# 6. Replay 的三层恢复

BACE replay 不只是恢复环境。它需要同时恢复：

1. environment state；
2. LLM-visible context；
3. branch-origin response identity。

三者必须严格区分。

```text
环境状态：通过 reset + action-prefix replay 得到
模型上下文：从原 root event log 重建
branch origin：复制原 natural response，而不是重新生成
```

---

# 7. ALFWorld Replay 详细设计

## 7.1 采用的恢复方式

ALFWorld TextWorld 主实现采用：

```text
bind same game file
→ reset to initial state
→ replay parsed environment actions before anchor
→ verify restored anchor
```

不要求通用环境 snapshot。

## 7.2 与 PivoARL 的对应关系

PivoARL 的 `AlfworldWorker` 每个 Ray actor 持有一个环境实例；`restart_from_turn()` 先调用 `restart()`，再逐条调用 `step(action)`。不同 worker 可以拥有不同 `target_turn`，vector wrapper 将 time-major action journal 转换为每个 worker 的 prefix。

BACE 应复用这一结构，但将输入改为 `ReplayRequest`，并新增严格验证。

## 7.3 Natural rollout 时必须记录什么

假设 root 中：

```text
step 0: go to kitchen
step 1: open fridge 1
step 2: take apple 1 from fridge 1
step 3: go to diningtable 1
```

若 step 2 前的 observation 是 anchor，则 ReplayRequest 应包含：

```text
environment_reset_key = game.tw-pddl path or stable game ID
target_turn = 2
parsed_action_prefix = [
    "go to kitchen",
    "open fridge 1"
]
expected_anchor_key = hash(observation before step 2)
selected_action = "take apple 1 from fridge 1"
```

注意：`target_turn=2` 表示重放 anchor 前两条动作，不包含 anchor action 本身。

## 7.4 ALFWorld worker 接口

推荐逻辑接口：

```text
restart_from_request(request):
    bind request.environment_reset_key
    reset same game
    for action in request.parsed_action_prefix:
        step(action)
        assert episode has not terminated early
    validate current state
    return ReplayResult
```

### 绑定同一 game

不能依赖环境随机选择下一 game。必须使用原 natural root 的 game file 或稳定 reset key。

### Prefix replay

只使用当时实际执行成功的 parsed canonical actions，不能使用 raw model response。

### Early terminal

若在最后一个 prefix action 之前出现 `done=True`，立即标记：

```text
error_type = EARLY_TERMINATION
replay_ok = false
```

不能继续使用该 branch。

## 7.5 恢复后验证

必须检查：

```text
restored_done == False
hash(restored_observation) == expected_anchor_key
selected_action in restored_admissible_actions
```

推荐 debug/审计模式再检查：

```text
normalized(restored_observation) == normalized(recorded_observation)
restored_admissible_actions == recorded_admissible_actions
```

主路径可用 compact hash，完整文本只在采样日志或失败时写盘。

## 7.6 观察 key 与完整状态

BACE anchor grouping 按原 GiGPO exact pre-action observation key。Replay validation 至少应验证同一个 key。

若 ALFWorld wrapper 还提供 inventory、task state 或 admissible actions，可作为更强的工程断言，但不能改变主 anchor grouping 定义，除非作为单独消融。

## 7.7 模型上下文恢复

假设 GiGPO prompt 使用有限历史 $K$。branch origin 在 root step 2：

```text
original root messages:
  initial task
  obs_0
  response_0
  obs_1
  response_1
  obs_2  <- anchor observation
```

恢复环境后，不应重新从环境 replay 过程中构建新的模型回答。应从 RootEventLog 中重建原 occurrence 的 model-visible context：

```text
memory = rebuild(root_events, end_step=2, history_length=K)
prompt = original prompt for step 2
```

如果 prompt builder 在恢复时需要最新 `admissible_actions`，可以将 replay 后的动作集合填入原模板，但必须断言其与记录值一致。

## 7.8 Copied branch origin

原 step 2 的自然模型 response：

```text
Thought: I should take the apple before leaving the kitchen.
Action: take apple 1 from fridge 1
```

BACE 不重新调用模型，而是直接复制：

- raw response；
- response token IDs；
- response mask；
- old log-probs；
- parsed canonical action。

然后环境执行：

```text
take apple 1 from fridge 1
```

这一步产生一个新的 branch-origin training occurrence，但不是新的 model generation。

## 7.9 Suffix generation

执行 copied action 后得到新 observation，随后：

```text
restored original memory
+ copied origin response
+ new environment observation
→ pi_old generates next response
→ continue to terminal or remaining horizon
```

branch suffix 使用与 roots 完全相同的：

- tokenizer；
- prompt builder；
- model sampling config；
- action parser；
- invalid-action handling；
- max-step rule。

## 7.10 完整 ALFWorld 示例

### Natural roots

Root 1：

```text
0 go to kitchen
1 open fridge 1
2 take apple 1 from fridge 1
3 go to diningtable 1
4 put apple 1 on diningtable 1
reward = 1
```

Root 2：

```text
0 go to kitchen
1 open fridge 1
2 examine fridge 1
3 close fridge 1
reward = 0
```

在 step 2 前两条 root 的 observation 相同，因此形成 anchor $z$。observed actions 为：

```text
take apple 1 from fridge 1
examine fridge 1
```

ERV 选择 `take apple ...`，origin pool 中包含 Root 1 step 2。

### Replay

```text
reset Root 1 的同一 game
step("go to kitchen")
step("open fridge 1")
```

验证当前 observation 与 Root 1 step 2 前一致。

### Copy origin

复制 Root 1 step 2 的 CoT--action，不调用 LLM：

```text
Thought: I should take the apple...
Action: take apple 1 from fridge 1
```

环境再次执行该 action。

### New suffix

旧策略重新生成：

```text
3 go to living room
4 examine sofa 1
5 go to diningtable 1
6 put apple 1 on diningtable 1
reward = 1
```

该 branch 的新证据说明：同一自然 edge 在另一个 continuation 下仍能成功。

### 训练数据

训练：

- Root 1 全部 responses；
- Root 2 全部 responses；
- copied Root 1 step 2 response，使用 branch leaf advantage；
- branch 新 suffix responses。

不训练：

- replay 的 `go to kitchen`；
- replay 的 `open fridge 1`。

---

# 8. WebShop Replay 详细设计

## 8.1 采用的恢复方式

WebShop 使用：

```text
same session/task index
→ reset(session=original_session_idx)
→ replay search/click/buy action prefix
→ verify restored page and available actions
```

PivoARL worker 保存 `_last_session_idx`，`restart()` 重新 reset 到同一 session，再逐条 replay actions。BACE 可直接借鉴该 worker 生命周期。

## 8.2 需要记录

```text
session_idx
page/action prefix
pre-action page observation
available_actions before action
selected canonical action
root memory/prompt
```

## 8.3 示例

Natural root：

```text
0 search[waterproof hiking boots]
1 click[item_173]
2 click[black]
3 click[size 10]
4 buy[now]
```

假设 anchor 是 item page，即 step 2 前。

ReplayRequest：

```text
session_idx = original session
prefix = [
  search[waterproof hiking boots],
  click[item_173]
]
expected page key = hash(item_173 page)
selected action = click[black]
```

恢复：

```text
reset(session=session_idx)
replay search
replay click item
verify page
verify click[black] available
```

然后复制 natural response `click[black]` 并生成新 suffix。

## 8.4 WebShop 特殊风险

- server/session 数据必须固定；
- 搜索排序和商品状态若非确定性，应固定数据版本和随机种子；
- 页面文本可能包含动态字段，应设计稳定 normalization 仅用于工程验证；
- canonical action 应采用结构化 `search[...]`、`click[...]`、`buy[...]`，避免依赖自然语言表述。

---

# 9. Replay Validator

## 9.1 验证级别

### Level 0：运行必要检查

每条 branch 必须通过：

```text
no exception
not prematurely terminal
anchor key match
selected action executable
```

### Level 1：严格文本检查

开发阶段增加：

```text
normalized observation exact match
action-set exact match
post-origin next observation match, when deterministic
```

### Level 2：抽样审计

定期保存：

```text
recorded observation
restored observation
recorded action set
restored action set
full replay prefix
worker ID
latency
```

## 9.2 Replay category

借鉴 Prefix-GRPO，定义：

```text
VALIDATED
ANCHOR_KEY_MISMATCH
ACTION_SET_MISMATCH
SELECTED_ACTION_NOT_EXECUTABLE
EARLY_TERMINATION
RESET_KEY_MISMATCH
PARSER_MISMATCH
ENV_EXCEPTION
```

第一版训练只接受 `VALIDATED`。

## 9.3 失败处理

capacity correction 后 branch quota 已冻结。若运行时出现 replay failure：

- 不临时增加新 root；
- 记录该 branch 为 invalid；
- 可从同一 $(z,u)$ 的其他 natural origin 重试一次；
- 若没有可用 origin，尝试该 task 当前 utility 次高的可执行 branch request；
- 超过有限重试次数后减少该 task 的实际 leaf count，并显式记录。

正常情况下 ALFWorld TextWorld replay failure 应接近零。大量失败应视为实现错误，而不是算法现象。

## 9.4 Origin fallback 顺序

推荐：

```text
1. selected (z,u) 中尚未用过的 natural origin
2. selected (z,u) 中已用过但可恢复的 origin
3. same z 下重新采 action
4. same task 下选择次高 U(z)
5. mark branch failure
```

所有 fallback 必须记录，便于审计实际 acquisition distribution。

---

# 10. Memory 与 Prompt 恢复

## 10.1 不保存 Python 深拷贝快照

PivoARL 为 retry attempt 深拷贝 memory，适合其实现；BACE 更推荐 immutable root event log + index，因为：

- 一个 root 可能产生多个 candidate origins；
- 每个 occurrence 保存完整 memory 会重复大量字符串；
- Ray object store 压力大；
- copied origin 已经绑定原 root，不需要独立 attempt 历史。

## 10.2 Memory rebuild

提供统一接口：

```text
rebuild_memory(root_id, step_index, history_length)
```

若 memory 只保留最近 $K$ 步，只读取：

```text
root.events[max(0, step_index-K):step_index]
```

## 10.3 Prompt identity

branch origin 的 prompt 应与原 natural occurrence 一致。建议在 natural rollout 时保存：

- prompt token IDs，或
- 能确定性重建 prompt 的 messages 与环境字段。

开发测试中应验证：

```text
hash(rebuilt_prompt_tokens) == hash(original_prompt_tokens)
```

若不相等，copied origin 的原 old log-probs 就不再严格对应当前 prompt，必须先修复 prompt reconstruction。

## 10.4 Old log-prob 复用条件

copied origin 可复用原 old log-probs，当且仅当：

```text
same pi_old
same prompt token sequence
same response token sequence
same masking convention
```

一个 rollout batch 内 actor 冻结，因此策略条件满足。其余三项必须用断言保证。

---

# 11. AnchorIndex 与 Origin Pool

## 11.1 Natural roots 完成后建立

输入：全部 natural-root occurrences。

步骤：

1. 按 `(task_id, to_hashable(pre_action_observation))` grouping；
2. 保留 group size 至少 2 的 repeated anchors；
3. 排除初始和终止状态；
4. 按 canonical action 分组；
5. 要求至少两个不同 observed actions；
6. 为每个 `(z,u)` 建立 natural origin occurrence list；
7. 用 natural terminal outcomes 初始化 Beta posterior。

## 11.2 Grouping 与 replay origin 不同

Anchor grouping 可以合并不同 histories，只要 exact observation 相同。

Replay 时必须选择一个 concrete occurrence：

```text
anchor z
  action u
    origin root 1 step 4
    origin root 3 step 6
```

选择 origin 后，环境 prefix、model history 和 copied response 全部绑定该 occurrence，不能在多个 origins 之间拼接。

## 11.3 同动作多个 origin

默认从 `I_obs(z,u)` 均匀采样。为减少重复相关性，可使用：

```text
without replacement until exhausted
→ then reset pool
```

这不改变 action acquisition distribution，只改善 origin 多样性。

---

# 12. Posterior 与 ERV Engine

## 12.1 独立纯 CPU 模块

`bace_posterior.py` 不依赖 Ray、vLLM 或环境，输入仅为：

```text
alpha[actions]
beta[actions]
tau_mu
MC sample count
```

输出：

```text
regret
DeltaERV[action]
mu[action]
U(anchor)
```

## 12.2 Vectorized 计算

对一个 anchor 的 $A$ 个 actions：

```text
alpha: [A]
beta: [A]
samples: [M, A]
```

一次 Beta sampling 和 max reduction，避免 Python action loop 中重复创建对象。

## 12.3 Incremental update

一条 branch 只改变一个 `(z,u)` posterior。因此：

- 未选中 anchors 的 ERV 和 utility 保持缓存；
- 只重算被更新的 anchor；
- 达到 $L_{\max}$ 的 anchor 从 active heap 中移除。

## 12.4 Priority queue

可维护：

```text
(-U, anchor_id, posterior_version)
```

取出时检查 version，过期条目丢弃。

第一版 anchor 数通常较小，线性扫描也可接受；priority queue 是后续优化，不是正确性依赖。

---

# 13. Branch Scheduler 与耗时控制

## 13.1 Round-synchronous 第一版

```text
for branch_round:
    each active task creates at most one request
    batch replay all requests
    bucket valid requests by remaining horizon
    batch generate suffixes
    update each task posterior
```

优点：

- 与当前 `TrajectoryCollector` 的 batch 模式接近；
- 实现和调试较简单；
- 不破坏序贯 ERV 语义。

## 13.2 Remaining-horizon buckets

例如：

```text
short: 1-3 steps
medium: 4-6 steps
long: 7+ steps
```

避免两步 branch 长时间等待十步 branch。

## 13.3 Persistent environment workers

环境 Ray actor 在整个 rollout batch 内常驻。每条 branch 只执行：

```text
bind/reset same task
→ replay
→ continue
```

禁止为每条 branch 重新创建 actor 或重新启动 WebShop server。

## 13.4 何时实现异步 frontier

只有 profiler 显示以下现象时：

- suffix-generation GPU idle 高；
- replay latency 占 rollout wall-clock 很大；
- 后几个 branch rounds active task 数过少；

才实现：

```text
WAIT_REPLAY
READY_TO_GENERATE
GENERATING
DONE
READY_FOR_NEXT_BRANCH
```

的异步状态机。第一版不承担这部分复杂度。

---

# 14. All-Leaf Advantage 与 Unique-Segment Training

## 14.1 Leaf 集合

每个 task 的 leaf set：

```text
all natural roots
+ all successful branch rollouts
```

每个 leaf 只需要 terminal reward 与 lineage ID。

计算：

$$
A_{g,\ell}^{E}
=
\frac{R_{g,\ell}-\mu_g^E}{\sigma_g^E+\epsilon}.
$$

## 14.2 Local occurrence groups

收集所有可训练 occurrences：

- root occurrences；
- copied branch origins；
- branch suffix occurrences。

按 final tree 中的 exact anchor key 建组。Replay prefix 不进入 group。

## 14.3 Occurrence-level 主版本

沿用 GiGPO：

$$
A_i^S
=
\frac{G_i-\mu_z}{\sigma_z+\epsilon}.
$$

## 14.4 Action-aggregated 可选版本

同一 `(anchor_id, action_id)`：

$$
A_{z,u}^{S,\mathrm{act}}
=
\frac{1}{n_{z,u}}
\sum_{i\in I(z,u)}A_i^{S,\mathrm{occ}}.
$$

广播回所有同动作 occurrences。

## 14.5 Final advantage

```text
final_advantage = leaf_advantage + omega * local_advantage
```

## 14.6 Trainable mask

| 数据来源 | 训练 mask |
|---|---:|
| natural root response | 1 |
| copied branch-origin response | 1 |
| new branch suffix response | 1 |
| mechanical replay prefix | 0 / 不进入 batch |

## 14.7 不展开完整 branch path

错误：

```text
root prefix + origin + suffix 1
root prefix + origin + suffix 2
```

全部作为独立 sequence 写入 PPO batch。

正确：

```text
root occurrence records      stored once
copied origin occurrence     one new record per branch
new suffix occurrences       one record per generated response
leaf metadata                stores complete-outcome semantics only
```

---

# 15. DataProto 组装

## 15.1 tensor fields

尽量复用 GiGPO：

```text
input_ids
attention_mask
position_ids
responses
old_log_probs
ref_log_prob
response_mask / loss_mask
token_level_scores
advantages
returns
```

## 15.2 non-tensor metadata

新增：

```text
task_id
episode_group_id
leaf_id
root_id
branch_id
origin_occurrence_id
source_type
anchor_id
action_id
traj_uid
step_index
is_replay_prefix
```

完整 observation、prompt 和树 JSON 不应复制到每个 token。放在 CPU-side trace store，并通过 ID 引用。

## 15.3 计算顺序

```text
complete tree collection
→ build leaf rewards
→ build trainable occurrence groups
→ compute advantages
→ assign tensor advantages
→ flatten DataProto
→ adjust/pad for devices
→ PPO update
```

不能先按 sequence length 重排再计算 group advantage，否则 group/leaf 结构容易被破坏。

---

# 16. 配置设计

建议配置：

```yaml
algorithm:
  adv_estimator: bace_gigpo
  bace:
    total_leaf_budget: 8
    pilot_roots: 2
    competence_threshold: 0.5
    max_branches_per_anchor: 2
    erv_threshold: 0.0
    erv_temperature: 0.05
    erv_mc_samples: 512
    local_prior_strength: 2.0
    local_credit_mode: occurrence   # occurrence | action_mean
    origin_sampling: without_replacement
    replay_origin_retries: 1
    train_copied_origin: true
    all_leaf_weighting: uniform

  gigpo:
    step_advantage_w: 1.0
    mode: mean_std_norm

env:
  replay:
    enabled: true
    strict_anchor_key: true
    strict_action_executable: true
    compare_action_set: true
    compare_full_observation: false
    failure_policy: retry_origin_then_next_candidate
    horizon_buckets: [3, 6]

trainer:
  bace:
    trace_sample_rate: 0.01
    profile_timing: true
    save_replay_failures: true
```

实际默认值需通过小规模实验确定，但配置层级应从一开始固定。

---

# 17. 完整端到端示例

设一个 batch 有 4 个 ALFWorld tasks，每个 $B=8$，pilots 为 2。

## 17.1 Pilots

一次生成：

```text
Task 1: roots 0,1
Task 2: roots 0,1
Task 3: roots 0,1
Task 4: roots 0,1
```

## 17.2 Topology planning

得到：

```text
Task 1: planned 6 roots + 2 branches
Task 2: planned 4 roots + 4 branches
Task 3: planned 8 roots + 0 branches
Task 4: planned 5 roots + 3 branches
```

## 17.3 Planned roots

所有缺少的 roots 合成一次大 batch。

## 17.4 Capacity correction

检查后：

```text
Task 2: insufficient anchor capacity
Task 4: insufficient anchor capacity
```

一次生成：

```text
Task 2: one extra root, Q -= 1
Task 4: one extra root, Q -= 1
```

重新构造 anchors，满足 capacity 后冻结。

## 17.5 Branch round 1

```text
Task 1 selects (z11, a3, origin root 2 step 5)
Task 2 selects (z21, a1, origin root 0 step 3)
Task 4 selects (z42, a2, origin root 4 step 6)
```

生成 3 个 ReplayRequests，环境 workers 并行恢复。

验证成功后，三个 copied origins 被执行，三个 suffix 进入同一或按 horizon 分桶的 rollout batch。

## 17.6 Posterior update

```text
Task 1 branch succeeds
Task 2 branch fails
Task 4 branch succeeds
```

分别更新 local posterior 和 utility。

## 17.7 Branch round 2

每个 task 重新选择其下一 branch。Task 1 可能仍选择同一 anchor，Task 2 可能改选另一个 anchor。

## 17.8 Unified training

最终每个 task 构造 8 个 terminal leaves 或显式记录 replay failure 后的实际 leaf 数。

计算：

- all-leaf $A^E$；
- root/copy/suffix occurrence-level $A^S$；
- unified final advantages；
- unique-segment DataProto；
- 一次 PPO update。

---

# 18. 测试设计

## 18.1 PosteriorEngine 单元测试

- Beta update success/failure；
- ERV 非负（允许 Monte Carlo 小误差后 clip）；
- temperature 极限；
- action permutation invariance；
- cached 与 full recompute 一致；
- action utility 的手算例子。

## 18.2 AnchorIndex 单元测试

- exact observation 相同正确 grouping；
- 不同 task 不混组；
- 同 trajectory 重访保留；
- singleton 排除 branch pool；
- 单 action anchor 排除；
- invalid action 排除；
- `(z,u)` origin pool 正确。

## 18.3 ALFWorld replay 单元测试

对已记录 root 的随机多个 cut：

```text
reset same game
replay prefix
compare anchor key
compare action set
execute recorded origin action
compare next observation when deterministic
```

目标：TextWorld 模式接近 100% 通过。

## 18.4 WebShop replay 单元测试

- same session reset；
- search/click prefix replay；
- page key match；
- available actions match；
- session isolation；
- 并行 workers 不串 session。

## 18.5 Prompt identity 测试

随机 occurrence：

```text
original prompt token IDs == rebuilt prompt token IDs
original response token IDs == copied response token IDs
original old log-probs reusable
```

## 18.6 Tree/advantage 测试

构造小型人工树，验证：

- leaf rewards 正确；
- root prefix 没有重复训练；
- copied origin 获得 branch leaf advantage；
- suffix 获得 branch leaf advantage；
- occurrence/local groups 正确；
- action aggregation 只修改 local term；
- token mask 数量符合预期。

## 18.7 End-to-end smoke test

配置：

```text
1 GPU or smallest available setup
2-4 tasks
B=4
fixed topology 3 roots + 1 random branch
no ERV
```

先验证 branch pipeline，再逐步打开 ERV 和 dynamic topology。

---

# 19. 分阶段实现顺序

## Stage 0：冻结 upstream 与复现 GiGPO

- 固定 commit；
- 记录环境、Python、PyTorch、vLLM、Ray 版本；
- 复现 ALFWorld 与 WebShop baseline；
- 保存吞吐、token、wall-clock 和成功率。

## Stage 1：离线 replay validator

- 只从 GiGPO roots 随机抽 cut；
- 不生成 branch；
- 验证 environment replay 和 prompt rebuild；
- 达到稳定通过率。

## Stage 2：固定 topology + random branch

- 固定 $6+2$ 或更小 smoke config；
- random structural anchor；
- random observed action；
- concrete natural origin；
- copied origin + new suffix；
- unique-segment training。

## Stage 3：all-leaf unified advantage

- 先 occurrence-level local credit；
- 验证 GiGPO baseline 可在 branch tree 上运行；
- 检查 shared-prefix 不重复。

## Stage 4：local posterior + ERV

- 固定 topology；
- 开启 sequential expected-ERV branch allocation；
- 对比 random branch。

## Stage 5：competence topology + capacity correction

- 加入 pilots、task-family prior、dynamic quota；
- 加入 global capacity-correction rounds；
- 记录 topology 分布。

## Stage 6：action aggregation

- 在完整 BACE-Occ 稳定后开启 BACE-Act；
- 避免同时调试 rollout acquisition 与新 credit estimator。

## Stage 7：WebShop 迁移

- 复用统一 ReplayAdapter 接口；
- 只替换 reset key、page/action validation 和环境 wrapper。

---

# 20. Profiler 与性能指标

每次 update 记录：

```text
time/pilot_generation
time/topology_planning
time/planned_root_generation
time/capacity_correction_generation
time/anchor_build
time/posterior_erv
time/replay_reset
time/replay_steps
time/replay_validation
time/branch_generation
time/tree_assembly
time/advantage
time/actor_update
```

资源量：

```text
new_generated_tokens
copied_origin_tokens
trainable_tokens
natural_environment_steps
replay_environment_steps
branch_suffix_steps
number_of_generate_calls
mean_active_sequences_per_generate_call
GPU idle fraction
Ray object store bytes
replay_failure_rate
```

核心效率曲线：

- performance vs terminal leaves；
- performance vs new generated tokens；
- performance vs total environment steps；
- performance vs GPU seconds；
- performance vs wall-clock。

---

# 21. 主要耗时与减时策略

## 21.1 最大风险：多阶段 barrier

减时：

- pilots 跨任务合批；
- planned roots 跨任务合批；
- capacity roots 按 correction round 合批；
- 每个 branch round 每个 active task 最多一个 request；
- suffix 按 remaining horizon 分桶。

## 21.2 Replay CPU/Ray 开销

减时：

- persistent workers；
- parsed action journal；
- compact anchor hash；
- 批量 `restart_from_turn`；
- debug 全量比较与主训练轻量比较分离。

## 21.3 Tree 数据和序列化

减时：

- immutable root logs；
- occurrence 只存 ID；
- DataProto 只放紧凑 metadata；
- 完整字符串写 trace store；
- 不展开 full branch paths。

## 21.4 ERV

减时：

- vectorized Beta sampling；
- incremental anchor recompute；
- 小规模在线 MC；
- 不在 Python 三层循环中逐 sample 计算。

---

# 22. 不采纳方案与原因

## 22.1 通用 snapshot/clone 作为第一版

不采纳。ALFWorld 与 WebShop 已可通过 same-instance reset + prefix replay 实现，通用 snapshot 会增加环境内部状态、RNG 和序列化复杂度。

## 22.2 整体合并 PivoARL

不采纳。其 retry/reflection、attempt memory 和 trainer 语义与 BACE 不同，且可能引入另一套 verl 版本。只移植 replay 相关逻辑。

## 22.3 训练完整 replay prefix

不采纳。会重复训练 shared prefix，使梯度权重随 branch 数增长。BACE 训练 copied origin 和 suffix。

## 22.4 重新让模型生成指定 origin action

不采纳。会引入 rejection sampling 或 forced decoding 与原自然 CoT 不匹配的问题。BACE 复制真实 natural CoT--action edge。

## 22.5 Recursive branch-of-branch

不采纳。会引入多层 origin ownership、树深失控、复杂 behavior distribution 和更难的 replay 管理。第一版只从 frozen natural roots 分支。

## 22.6 先实现完全异步调度

不采纳。第一版同步 frontier 足以验证算法；异步系统只在 profiling 证明必要时实现。

## 22.7 直接复制 Prefix-GRPO 的 prefix token objective

不采纳。其 historical teacher prefix 是训练对象；BACE 的环境 replay prefix 不是新采样数据。

---

# 23. 工程不变量

代码中必须断言：

1. 一个 rollout batch 内 `pi_old` 固定；
2. task-family prior 只由 natural-root outcomes 更新；
3. capacity correction 完成后 root pool 冻结；
4. branch origin 只来自 frozen natural roots；
5. selected action 必须是该 origin 自然执行过的 action；
6. replay prefix 不创建新的 training occurrence；
7. copied origin prompt 与 response 必须和 natural occurrence 完全一致；
8. copied origin old log-probs 只在 prompt/response identity 通过时复用；
9. replay 后 anchor key 必须匹配；
10. selected action 必须在恢复状态可执行；
11. branch suffix 使用同一个 `pi_old`；
12. local posterior 在 actor update 后丢弃；
13. all-leaf statistics 与 unique-segment training 分离；
14. advantage 在 batch length balancing 之前计算；
15. 测试阶段不启用 branching。

---

# 24. 首个可用版本的验收标准

第一版 BACE implementation 只有同时满足以下条件才进入正式实验：

### Baseline

- 同一 commit 下原 GiGPO 可复现；
- 关闭 BACE 时与原 trainer 行为一致。

### Replay

- ALFWorld TextWorld random-cut replay success 接近 100%；
- prompt identity test 通过；
- selected action admissibility test 通过；
- parallel workers 不串 game/session。

### Tree and masks

- replay prefix trainable token 数为 0；
- copied origin token 数与预期一致；
- full leaf reward 与 branch suffix reward 对齐；
- 无 shared-prefix duplication。

### Sequential logic

- branch outcome 更新 posterior；
- 下一轮 ERV/utility 实际变化；
- per-anchor branch count 不超过 $L_{\max}$；
- capacity correction 保持 $R_g+Q_g=B$。

### Training

- unified PPO loss 正常反向传播；
- root、branch origin、branch suffix 均有非零训练样本；
- 无 NaN、group ID mismatch 或 old-log-prob mismatch。

### Profiling

- 输出各阶段 timing；
- 输出 new/copy/trainable token 数；
- 输出 replay steps 与 GPU utilization；
- 能与 equal-leaf GiGPO 对比 wall-clock。

---

# 25. 推荐开发分工

## Environment/Replay

- ALFWorld/WebShop adapter；
- restart/replay；
- state validation；
- replay tests。

## Rollout/Coordinator

- pilots、roots、capacity correction；
- branch frontier；
- Ray batching；
- task state machine。

## Bayesian Controller

- priors/posteriors；
- regret/ERV；
- action distribution；
- utility cache。

## Tree/Credit

- tree schema；
- leaf and occurrence views；
- all-leaf advantage；
- action aggregation；
- DataProto assembly。

## Experiment/Diagnostics

- configuration；
- profiler；
- trace viewer；
- ablation harness；
- baseline reproduction。

---

# 26. 代码审计来源

本文档在 2026-08-06 核对了以下公开代码路径：

## `langfengQ/verl-agent`

```text
gigpo/core_gigpo.py
agent_system/multi_turn_rollout/rollout_loop.py
recipe/hgpo/core_hgpo.py
recipe/hgpo/hgpo_ray_trainer.py
recipe/GraphGPO/
```

## `yuki-younai/PivoARL`

```text
agent_system/environments/env_package/alfworld/envs.py
agent_system/environments/env_package/alfworld/env_manager.py
agent_system/environments/env_package/webshop/envs.py
agent_system/environments/env_package/webshop/env_manager.py
scripts/test/legacy/test_alfworld_restart.py
scripts/test/legacy/test_alfworld_manager_restart_from_turn.py
```

## `HappynessI/Prefix_GRPO`

```text
scripts/build_data/build_alfworld_prefix_rl_change_top3.py
```

外部仓库代码只作为最小模块移植的参考。实际移植时必须记录：

- upstream repository；
- source file；
- source commit；
- license header；
- 修改内容；
- 对应单元测试。

---

# 27. 最终实现路线摘要

```text
Base:
    verl-agent / GiGPO

Recipe and trainer pattern:
    HGPO

Tree metadata and diagnostics:
    GraphGPO

Online environment replay:
    PivoARL restart_from_turn

Replay audit and acceptance:
    Prefix-GRPO validation pattern

BACE-specific additions:
    competence topology
    capacity correction
    expected-ERV anchor utility
    natural-origin selection
    copied CoT-action occurrence
    all-leaf credit
    unique-segment unified PPO
```

一句话概括：

> **在原 GiGPO rollout/trainer 上新增一个多阶段 BACE coordinator；使用 PivoARL 风格的 same-instance reset + action-prefix replay 恢复环境，使用 immutable root logs 重建原 model context，复制真实 natural CoT--action 作为 branch origin，再批量生成新 suffix；最后以完整 leaves 计算统计优势、以 unique trainable segments 进行统一 PPO 更新。**
