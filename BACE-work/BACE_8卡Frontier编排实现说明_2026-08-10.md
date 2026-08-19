# BACE 8卡 Frontier 编排实现说明

日期：2026-08-10

对应设计规范：`BACE_GiGPO_ALFWorld_8卡训练Pipeline与编排规范_2026-08-10.md`

## 1. 可实现性结论

该规范可以基于当前 `verl-agent` 工程实现，而且不需要修改 vLLM、FSDP、old/ref logprob 或 PPO update 主干。实现边界位于：

```text
ALFWorld selected-slot API
        +
per-slot observation/history state
        +
BACE trajectory collector 内的 task-aware frontier orchestrator
```

本次把新路径作为可选模式加入：

```yaml
algorithm.bace.topology: dynamic
algorithm.bace.dynamic_root_generation: staged
algorithm.bace.staged_root_batching: frontier
```

以下原有模式均保留，没有删除或替换：

```text
dynamic_root_generation=preallocated
staged_root_batching=sequential
staged_root_batching=packed
```

## 2. 本次实现内容

### 2.1 ALFWorld raw environment selected-slot API

在 `AlfworldEnvs` 中新增：

```python
reset_selected(worker_indices, game_files=None)
step_selected(worker_indices, actions)
replay_selected(worker_indices, game_files, prefix_actions_per_env)
getobs_selected(worker_indices)
get_admissible_selected(worker_indices)
```

这些接口显式接收稳定 worker id，不依赖 `active_worker_indices`。原有 `reset_subset()`、`step()` 和 `replay()` 继续存在。

### 2.2 ALFWorld manager 的 per-slot 状态

`AlfWorldEnvironmentManager` 新增独立的 `_bace_slots` 状态表。每个 persistent sibling slot 保存：

```text
current raw observation
task description
concrete game file
admissible action set
action/observation history
done state
```

新增 manager 接口：

```python
reset_selected(...)
get_observations_selected(...)
replay_selected(...)
step_selected(...)
```

Replay 会在目标 slot 重建 prefix history，因此恢复后的 prompt 仍可执行 prompt token identity 校验。

### 2.3 Task-local 状态机与 slot 生命周期

新模块 `recipe/bace_gigpo/frontier.py` 包含：

```text
FrontierTaskController
FrontierJob
FrontierProfile
BACEFrontierOrchestrator
```

每个 task 持有固定的 `B` 个 sibling slots。slot 从前向后只分配一次：

```text
pilot root slots
→ planned root slots
→ capacity-correction root slots
→ branch slots
```

已经产生 root leaf 的 slot 不会 reset 后复用为 branch slot。

### 2.4 Mixed MODEL_READY frontier

一次 generation wave 会收集当前所有 model-ready jobs：

```text
仍在运行的 pilot/root
其他 task 的 completion/capacity root
已经完成 Replay 的 branch suffix
```

它们共同调用一次：

```python
actor_rollout_wg.generate_sequences(...)
```

然后只对本 wave 的 slot 调用 `step_selected()`。因此不同 task 可以处于不同 acquisition phase，不再受全局 root/branch round barrier 限制。

### 2.5 同 task branch 严格顺序

每个 task 使用独立的 `ExpectedErvCoordinator`。只有 branch `b` terminal 后才：

```text
update local posterior
→ select branch b+1
→ allocate next unused sibling slot
→ Replay
```

不同 task 在同一 tick 产生的 Replay 请求会批量发给不同 Ray workers；同 task 不会提前创建多个 branch。

### 2.6 Replay 与失败规则

Frontier Replay 使用主 task group 中未使用的 sibling slot，不再创建额外 branch environment pool。

Replay 过程：

```text
bind concrete game
→ mechanical prefix Replay（不调用模型）
→ anchor/action-set/prompt identity 验证
→ 执行 copied branch-origin action
→ transition identity 验证
→ branch suffix 加入 MODEL_READY
```

若允许重试，只在同一 anchor/action 下选择另一个 frozen natural origin。重试耗尽或 transition mismatch 会中止当前 acquisition batch，不会悄悄减少 leaf 数。

### 2.7 Logical leaf 与 physical training segment

artifact 保存 logical branch lineage：

```text
task
branch
origin occurrence
selected action
terminal reward
suffix occurrence count
```

PPO DataProto 只包含：

```text
natural root decisions
branch-origin decision
new branch suffix decisions
```

机械 Replay prefix 不进入 PPO batch。

### 2.8 Profiler 与关键数据保存

新增并写入 trainer metrics/artifact 的统计包括：

```text
generation waves and decisions
generation time
selected environment steps
Replay jobs, prefix steps and time
root/branch terminal event count
MODEL_READY batch-size mean/p10/p50/p90
root/replay queue length
per-task slots, roots, branches, freeze state and consumed-slot count
logical branch lineage
ERV acquisition diagnostics and posterior snapshots
```

已有 `roots`、`anchors`、`topology`、`trainable_occurrences`、token fraction、old-logprob diff 等 artifact 保存逻辑继续使用。

### 2.9 Root/Branch/Mixed 成功率口径

BACE 的算法信号与训练日志统计分开处理：

```text
root_success_rate   = natural root episode 的 terminal reward > 0 的比例
branch_success_rate = Replay branch episode 的 terminal reward > 0 的比例
mixed_success_rate  = root 与 branch episode 合并后的比例
```

算法内部仍使用不可变的 `RootEventLog.won` 作为 root/competence/topology/anchor 证据，
并使用 Replay 完成后的 `terminal_reward > 0` 更新 branch posterior。训练 batch 中的兼容字段
`success_rate` 明确定义为 `mixed_success_rate`，其值从 `episode_rewards` 重算，不从复制的
branch-origin 行继承旧 root 元数据。

TensorBoard 和 per-step `summary.json` 额外保存：

```text
episode/root_success_rate
episode/branch_success_rate
episode/mixed_success_rate
bace/root_success_rate
bace/branch_success_rate
bace/mixed_success_rate
```

`trainable_occurrences.jsonl` 为每条训练 occurrence 保存 `success_scope`（root/branch）和
`episode_success`，便于检查 branch origin 与 root 的最终结果是否发生分歧。

## 3. 强制不变量

实现会在运行时检查：

```text
root_count + branch_count == B
consumed sibling slots == B
one slot contributes at most one terminal leaf
no root after root_backbone_frozen
same-task branch selection is terminal-dependent
Replay failure cannot silently reduce the leaf budget
same task group uses the same concrete game/task identity
```

## 4. 运行脚本

新增：

```text
examples/gigpo_trainer/run_bace_alfworld_npu_8card_frontier.sh
```

默认使用此前成功 GiGPO 8卡脚本的 NPU 内存配置：

```text
TP=1
ppo/logprob/ref micro batch=8
gpu_memory_utilization=0.6
max_num_batched_tokens=16384
max_num_seqs=128
max_steps=40
50 epochs
```

输出统一写入：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp
```

## 5. 当前明确限制

当前实现已经是 per-task mixed frontier，且同 tick 的跨 task Replay 会批量并行发给 Ray workers；但 driver 仍按以下顺序同步等待：

```text
batch Replay complete
→ GPU generation
```

尚未在独立线程中实现 CPU Replay 与正在进行的 GPU generation 的时间重叠。这样做不改变算法语义或训练数据，只影响可进一步优化的 wall-clock。首次真实 ALFWorld/NPU smoke 应先验证 selected-slot、prompt identity、leaf budget 和 DataProto，再打开异步重叠，避免把并发故障与算法正确性混在一起。

此外，本地单元测试不能替代真实 128-slot ALFWorld/NPU smoke。正式 50 epoch 前应运行专用 smoke 脚本；它保持 `TRAIN_DATA_SIZE=16`、`GROUP_SIZE=8`，因此会真正创建 128 个 sibling slots：

```bash
examples/gigpo_trainer/run_bace_alfworld_npu_8card_frontier_smoke.sh
```

smoke 通过后再恢复 `TRAIN_DATA_SIZE=16` 和 `TOTAL_EPOCHS=50`。

## 6. 1 epoch A/B 性能实验

本轮新增的可选配置为：

```text
algorithm.bace.frontier_batch_coalescing.enabled
algorithm.bace.frontier_batch_coalescing.max_batch_size
algorithm.bace.frontier_batch_coalescing.min_batch_size
```

coalescing 只合并已经处于 `MODEL_READY` 的 job。它不会预生成额外 root，也不会改变 `R + Q = B`、Replay identity 或同 task branch 串行约束。若当前 ready job 不足以达到 `min_batch_size`，实现立即降级发送，不人为等待或制造候选。

每个 generation wave 现在额外保存到：

```text
generation_waves.jsonl
```

每条记录包含 batch size、root/branch 数量、task 数量、Replay 后新就绪 job 数、队列前后长度、耗时和 coalescing 状态。这样可以区分“确实合并了 ready job”和“尾部没有可合并 job”两种情况。

A/B 脚本：

```text
examples/gigpo_trainer/run_bace_alfworld_npu_8card_frontier_1epoch_ab.sh
```

基线 arm：

```bash
AESC_RUN_NAME=bace_frontier_alfworld_npu_8card_1epoch_ab_baseline \
BACE_COALESCING_ENABLED=false \
bash examples/gigpo_trainer/run_bace_alfworld_npu_8card_frontier_1epoch_ab.sh
```

coalescing arm：

```bash
AESC_RUN_NAME=bace_frontier_alfworld_npu_8card_1epoch_ab_coalescing \
BACE_COALESCING_ENABLED=true \
BACE_COALESCING_MIN_BATCH_SIZE=32 \
bash examples/gigpo_trainer/run_bace_alfworld_npu_8card_frontier_1epoch_ab.sh
```

两次运行必须使用不同的 `AESC_RUN_NAME`。比较时读取各自的 `summary.json` 和 `generation_waves.jsonl`，至少核对总耗时、generation wave 数、batch size p10/p50/p90、root/branch 账目、Replay 成功率和 artifact 完整性。
