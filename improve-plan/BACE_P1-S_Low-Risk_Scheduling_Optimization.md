# P1-S：BACE 低风险调度优化方案
## ——在不改变当前 Exact Batch-ERV 算法语义的前提下缩短训练时间

**适用基线**：当前 `XYR222/BACE` 主线的 No-Pilot + Exact Batch-ERV + staged/packed roots + selected-worker branch execution。  
**目标硬件**：2×/4× NVIDIA H100。  
**文档定位**：P1-S 只解决执行/调度浪费，不改变 root/branch quota、Exact Batch-ERV、anchor、posterior、advantage 或 PPO 目标。  
**不包含**：task-local frontier、真正异步 multi-turn、Pairwise \(K=2\)、capacity-aware topology。这些留到 P2/P3。

---

# 1. 背景与当前实现

当前 BACE 的一个 PPO update 以 `train_batch_size=16`、`total_leaf_budget=8` 为典型配置，因此每个 update 最终仍然产生：

\[
16\times 8=128
\]

个 terminal leaves。

当前 Exact Batch-ERV 主路径可以抽象为：

```text
planned roots（packed wave）
        ↓
capacity check
        ↓
capacity-correction root wave 1
        ↓
capacity check
        ↓
capacity-correction root wave 2
        ↓
...
        ↓
所有 task 的 root backbone 冻结
        ↓
Exact Batch-ERV 一次性确定全部 branches
        ↓
branch requests 按 replay capacity 切 chunk
        ↓
chunk 1 replay + suffix 完整执行
        ↓
chunk 2 replay + suffix 完整执行
        ↓
...
        ↓
PPO
```

当前代码中与 P1-S 直接相关的实现位置：

- `verl-agent-src/recipe/bace_gigpo/rollout_collector.py`
  - `_collect_staged_dynamic_roots_packed`
  - `_collect_root_wave`
  - `_execute_round`
  - `_execute_chunk_selected`
  - `_collect_suffixes_selected`
- `verl-agent-src/recipe/bace_gigpo/env_factory.py`
  - `make_branch_env`
- `verl-agent-src/agent_system/multi_turn_rollout/rollout_loop.py`
  - `vanilla_multi_turn_loop`
- `verl-agent-src/agent_system/environments/env_manager.py`
  - `reset_selected`
  - `get_observations_selected`
  - `replay_selected`
  - `step_selected`
- `verl-agent-src/agent_system/environments/env_package/alfworld/envs.py`
  - selected-worker 的底层 Ray worker 操作

当前生产脚本中：

```text
data.train_batch_size = 16
env.rollout.n = 8
total_leaf_budget = 8
max_num_seqs = 128
```

因此主 ALFWorld pool 对应：

\[
16\times 8=128
\]

个 sibling workers。

但当前 `make_branch_env()` 又单独创建：

\[
16\times 1=16
\]

个 branch replay workers。

这造成了 P1-S 的第一个主要优化点。

---

# 2. P1-S 的两个核心修改

P1-S 只包含两个修改：

1. **P1-S1：Branch 复用主 128-worker pool，消除 16-worker 人工 chunk bottleneck。**
2. **P1-S2：Natural-root rollout 改为 active-only selected execution，停止给已经 done 的 root 继续生成动作。**

它们的共同原则是：

\[
\boxed{\text{不减少任何有效 rollout，只减少空闲、等待和无效 generation。}}
\]

---

# 3. P1-S1：Branch 复用主 128-worker pool

## 3.1 当前问题

当前 Exact Batch-ERV 已经会在算法层一次性决定一个 task 的全部 branch requests。

但是 `_execute_round()` 会执行：

```python
capacity = int(self.branch_envs.replay_capacity)
chunks = self._chunk_requests(requests, capacity)

for execution_wave, chunk in enumerate(chunks):
    self._execute_chunk(...)
```

而当前专用 branch pool 只有 16 个 workers，因此：

\[
\text{replay capacity}=16.
\]

例如某个后期 update 共规划 60 条 branches：

### 当前物理执行

```text
60 requests
   ↓
chunk 1 = 16
   ↓ 完整 replay + suffix
chunk 2 = 16
   ↓ 完整 replay + suffix
chunk 3 = 16
   ↓ 完整 replay + suffix
chunk 4 = 12
   ↓ 完整 replay + suffix
```

算法虽然是 Full Batch-ERV，但执行层仍然被人为拆成四个串行 cohort。

---

## 3.2 为什么主 128-worker pool 足够

当前：

\[
B=8,\qquad R_{\min}=2.
\]

因此单 task 最大 branch quota 为：

\[
Q_{\max}=6.
\]

16 个 task 的理论最大 branch request 数：

\[
16\times 6=96.
\]

而主 pool 有：

\[
128
\]

个 workers。

所以在当前配置下：

\[
\boxed{Q_{\mathrm{total}}\le 96 < 128.}
\]

也就是说，当前一个 Exact Batch-ERV update 中所有 frozen branch requests 都可以同时放进主 worker pool，不存在必须用 16-worker 专用 pool 的算法约束。

---

## 3.3 推荐方案

### 最终推荐

Exact Batch-ERV 在 root backbone 全部冻结后，直接：

```text
branch_envs = main envs
```

并使用主 pool 的 selected-worker API：

```text
replay_selected
get_observations_selected
step_selected
```

执行 branch。

关键点：

> Root phase 结束后不需要保存每个 root 对应环境的 live state。

因为 branch replay 所需的信息已经保存在 `ReplayRequest` / root logs 中，包括：

- `environment_reset_key`
- concrete action prefix
- prefix observations
- target anchor
- copied raw origin response
- origin occurrence id
- remaining horizon

Branch 本来就通过：

```text
reset same game
→ replay prefix
→ validate anchor
→ execute copied origin
→ fresh suffix
```

恢复环境。

因此物理 worker 可以复用。

---

# 4. P1-S1 的 slot 分配方案

建议每个 PPO update 仍把 128 个物理 slots 固定划为：

```text
Task 0 : slots   0..7
Task 1 : slots   8..15
...
Task 15: slots 120..127
```

Root 阶段仍然使用这些 task-local sibling slots。

进入 branch 阶段后，root live state 已不再有用途，可以重新覆盖这些 slots。

对于 Task \(g\) 有 \(Q_g\) 个 branches：

```text
Task g branch 0 → task-local slot 0
Task g branch 1 → task-local slot 1
...
Task g branch Q_g-1 → task-local slot Q_g-1
```

由于：

\[
Q_g\le B=8,
\]

一定放得下。

这种 task-local 映射比“全局随便挑空槽”更容易：

- debug；
- trace；
- replay identity 检查；
- 复现 worker-to-task mapping；
- 后续升级成 task-local frontier。

---

# 5. P1-S1 的代码修改建议

## 5.1 `main_ppo.py`

当前 Exact/packed 模式会：

```text
make_branch_env(config)
```

创建第二个 pool。

建议增加一个明确配置：

```yaml
algorithm:
  bace:
    replay:
      pool_mode: main
```

支持：

```text
main       # 推荐生产模式
dedicated  # 回归测试 / fallback
```

逻辑：

```python
if bace_config.replay.pool_mode == "main":
    branch_envs = envs
else:
    branch_envs = make_branch_env(config)
```

不要立即删除 dedicated path。

它很适合做 A/B correctness test。

---

## 5.2 `rollout_collector.py`

当前 `_execute_round()` 不应再假设：

```text
branch replay capacity = 16
```

如果 `branch_envs is envs`：

```python
capacity = branch_envs.replay_capacity
```

应自然得到主 pool capacity。

在当前配置：

\[
capacity=128.
\]

于是 60 requests：

```text
len(chunks)=1
```

---

## 5.3 worker slot 映射

不要继续让每个 chunk 都默认：

```text
slot = 0..len(chunk)-1
```

建议为每个 request 显式计算：

```python
worker_slot = (
    request.task_batch_index * total_leaf_budget
    + task_local_branch_index
)
```

并将 `worker_slot` 写入：

- replay trace；
- branch execution trace；
- failure/retry metadata。

这样在 main pool reuse 下仍然能够稳定审计。

---

# 6. P1-S1 的正确性不变量

改动后必须保持：

### 不变量 1：Exact Batch-ERV selection 不变

Scheduler 不能重新选择：

- anchor；
- action；
- origin；
- branch count。

调度只执行 coordinator 已冻结的 requests。

---

### 不变量 2：同一个 request 仍执行相同 replay protocol

仍然必须：

```text
reset concrete game
→ replay exact parsed action prefix
→ validate target anchor
→ validate action set（若开启）
→ execute copied origin response
→ validate transition
→ rollout fresh suffix
```

---

### 不变量 3：root backbone 不因 worker reuse 被改写

一旦 Exact topology finalize：

```text
root_output
root_logs
AnchorIndex / selection evidence
```

必须全部视为不可变 CPU/data objects。

之后覆盖物理 environment slot 不得影响算法证据。

---

### 不变量 4：训练样本语义不变

最终仍然是：

```text
natural roots
+ copied branch origins
+ fresh branch suffixes
```

mechanical replay prefix 不进入训练。

---

# 7. P1-S1 的测试计划

## 7.1 单元测试

至少新增：

### Test A：slot uniqueness

同一物理 wave 中：

```text
所有 branch requests → 不重复 worker slot
```

---

### Test B：task-local range

Task \(g\) 的 branch worker slot 必须满足：

\[
8g\le slot < 8(g+1)
\]

（在 B=8 时）。

---

### Test C：replay identity

`main` pool 与 `dedicated` pool 在 deterministic/mock rollout 下：

- target anchor 相同；
- selected origin 相同；
- replay validation 全通过；
- terminal leaf count 相同；
- trainable occurrence schema 相同。

---

### Test D：最大 quota

构造：

```text
16 tasks × 6 branches = 96
```

确认：

```text
branch_execution_waves = 1
```

且无 slot collision。

---

## 7.2 随机生成说明

真实 vLLM sampling 下，改变 batch size/order 后不应要求 token-level bitwise identical。

正确的验收标准是：

> 算法选择和 rollout 分布语义不变，而不是随机采样输出逐 token 完全一致。

需要做精确一致测试时使用：

- mock generator；
- deterministic decoding；
- 或显式 per-request RNG seed。

---

# 8. P1-S1 的性能指标

新增/保留：

```text
branch_replay_capacity
branch_request_count
branch_execution_waves
branch_execution_max_wave_size
branch_suffix_generation_seconds
branch_suffix_generation_waves
branch_suffix_active_sequences
branch_suffix_active_efficiency
```

最关键的验收指标：

\[
\boxed{\text{branch_execution_waves}}
\]

当前典型：

```text
3~4
```

目标：

```text
1
```

在当前 \(B=8,16\) tasks 配置下，只要 branch requests \(\le128\)，就应为 1。

---

# 9. P1-S2：Root active compaction

## 9.1 当前问题

当前 branch suffix 已经是 active-only：

```python
active_indices = [
    index
    for index, request in enumerate(requests)
    if not is_done[index]
    and suffix_step < request.remaining_horizon
]
```

之后只对 active branches：

```text
preprocess
generate_sequences
step_selected
```

但是 natural roots 仍使用 `vanilla_multi_turn_loop()`。

当前 root loop 每一步先：

```python
active_masks = np.logical_not(is_done)
```

但送入：

```python
actor_rollout_wg.generate_sequences(...)
```

的仍然是完整 batch。

因此 `active_masks` 更多是在**结果保存/奖励统计**时屏蔽已经 done 的样本，而不是在 GPU generation 前做物理 compaction。

---

# 10. 一个直观例子

假设某个 root wave 初始有 80 条 roots。

真实 active 数：

```text
step  1: 80
step 10: 70
step 15: 52
step 20: 31
step 25: 15
step 30:  5
```

理想 GPU submission：

```text
80 → 70 → 52 → 31 → 15 → 5
```

dense root implementation 则更接近：

```text
80 → 80 → 80 → 80 → 80 → 80
```

已经 terminal 的 trajectories 仍然参与：

- prompt preprocessing；
- padding；
- model generation；
- decode；

最后再因为 `active_mask=False` 被忽略。

这属于纯执行浪费。

---

# 11. 推荐的 Root selected executor

不要简单在旧 `vanilla_multi_turn_loop()` 中“筛一下 tensor”就结束。

原因是 ALFWorld selected subset 需要同时维护：

- 物理 raw worker id；
- task identity；
- memory/history；
- admissible actions；
- environment state。

当前代码已经有更适合的 selected-slot API：

```text
reset_selected
get_observations_selected
step_selected
```

并且 `AlfWorldEnvironmentManager._bace_slots` 已经能够为 persistent worker 保存独立：

```text
text_obs
task
gamefile
admissible_actions
history
done
```

因此推荐实现一个统一的：

```text
SelectedSlotRootExecutor
```

或者更通用：

```text
SelectedSlotRolloutExecutor
```

---

# 12. Root active-only 的执行流程

一个 packed root wave 的输入仍然是：

```text
(task_index, root_slot, reset_key)
```

但执行改成：

### Step 0：selected reset

```python
worker_slots = [
    task_idx * budget + root_slot
    ...
]

envs.reset_selected(
    worker_slots,
    reset_keys,
)
```

---

### Step t：只处理 active roots

```python
active = [
    root for root in roots
    if not root.done
    and root.step < max_steps
]
```

然后：

```text
get_observations_selected(active_slots)
        ↓
gen_batch.select_idxs(active_task_indices)
        ↓
preprocess_batch
        ↓
pad
        ↓
generate_sequences
        ↓
unpad
        ↓
step_selected(active_slots)
```

---

### terminal root

一条 root 一旦：

```text
done=True
```

立即从 active set 删除。

它之后：

- 不再 tokenize；
- 不再 pad；
- 不再 model generate；
- 不再 env step。

但已经生成的有效 rows 全部保留。

---

# 13. P1-S2 必须保持的 metadata

Root 输出仍必须提供与当前路径相同的关键字段：

```text
uid
traj_uid
occurrence_id
task_batch_index
step_index
anchor_obs
post_action_observation
admissible_actions
raw_model_response
projected_action
action_identity
is_action_valid
is_action_format_valid
is_action_environment_valid
environment_reset_key
task_description
done
remaining_horizon
rewards
episode_rewards
episode_lengths
success_rate
active_masks
```

尤其必须保持：

\[
\text{occurrence\_id} = \text{traj\_uid:step}
\]

的唯一性语义。

否则会影响：

- `build_root_event_logs`
- `AnchorIndex`
- branch origin identity
- replay
- GiGPO occurrence grouping。

---

# 14. P1-S2 与 packed root batching 的关系

P1-S2 **不改变阶段逻辑**。

仍然是：

```text
planned packed root wave
↓
全部 planned roots 完成
↓
capacity check
```

如果需要 correction：

```text
correction root wave
↓
全部本 wave correction roots 完成
↓
capacity check
```

也就是说 P1-S2 只做：

\[
\boxed{\text{wave 内 active compaction}}
\]

不做：

\[
\boxed{\text{task-local wave overlap}}
\]

后者属于 P2-S Exact Task-Local Frontier。

这样可以保证 P1-S 改动范围可控、容易回归。

---

# 15. P1-S2 的性能指标

建议增加与 branch 对称的 root 指标：

```text
root_generation_waves
root_active_sequences
root_dense_equivalent_sequences
root_inactive_sequences_avoided
root_submitted_sequences
root_padding_sequences
root_active_efficiency
root_selected_seconds
```

定义：

\[
\text{root active efficiency}
=
\frac{
\sum_t N_{\text{active},t}
}{
\sum_t N_{\text{wave-size}}
}.
\]

如果为：

\[
0.70,
\]

说明旧 dense implementation 约有 30% 的 sequence-step 是已经 inactive 的物理占位。

---

# 16. P1-S2 的正确性测试

### Test A：所有 roots 相同长度

如果所有 root 都到同一步终止：

```text
active compaction 与 dense execution 的提交数应一致。
```

---

### Test B：不同终止长度

例如：

```text
root A: 2 steps
root B: 4 steps
root C: 6 steps
```

则 active submission 应为：

\[
3+3+2+2+1+1=12
\]

而不是：

\[
3\times6=18.
\]

---

### Test C：root event log equivalence

deterministic/mock 环境下：

- root count 相同；
- per-root step count 相同；
- won 相同；
- anchors 相同；
- action identity 相同；
- branch candidates 相同。

---

### Test D：capacity correction compatibility

active root executor 必须同时支持：

- planned roots；
- completion roots；
- capacity-correction roots。

不能只优化第一波 planned roots。

---

# 17. P1-S 推荐实现顺序

## Phase S1：先做 main-pool branch reuse

原因：

- 改动范围小；
- bottleneck 明确；
- 很容易通过 `branch_execution_waves` 看到收益；
- 不必立刻改 root collector。

建议先用 5~10 个 training steps 做 A/B。

---

## Phase S2：再做 root selected-active executor

原因：

- 涉及 metadata/history；
- correctness 要比 branch pool reuse 更仔细；
- 但覆盖 planned root + correction root 的大量 generation 时间。

---

## Phase S3：统一 selected executor 抽象

S1/S2 稳定后再整理代码：

```text
SelectedSlotExecutor
├─ RootJob
└─ BranchJob
```

为 P2-S task-local frontier 做准备。

不要一开始就大重构。

---

# 18. 不推荐的实现方式

## 18.1 不推荐：直接把 dedicated branch pool 从 16 扩成 128 作为最终方案

这能快速验证 chunk bottleneck，但会形成：

```text
main pool   = 128 workers
branch pool = 128 workers
```

合计 256 个 ALFWorld Ray actors。

缺点：

- CPU/memory 增加；
- worker initialization 增加；
- Ray task pressure 增加；
- 主 pool 在 branch 阶段完全闲置。

它适合做短期诊断开关，不适合最终生产实现。

---

## 18.2 不推荐：在 P1-S 中直接实现 task-local frontier

这会同时改变：

- root/correction barrier；
- root→branch overlap；
- physical wave composition。

虽然算法语义可以保持不变，但改动范围太大，难以判断 S1/S2 各自收益。

P1-S 的目标是先获得低风险、可归因的 speedup。

---

## 18.3 不推荐：P1-S 同时切 Pairwise \(K=2\)

Pairwise 会改变 acquisition feedback granularity，已经属于算法变化。

P1-S 必须保持当前：

\[
K=Q
\]

Full Batch-ERV。

---

# 19. P1-S 的实验设计

## 19.1 四组最小 A/B

推荐：

```text
S0:
当前 HEAD
dedicated branch pool
dense root
```

```text
S1:
main-pool branch reuse
dense root
```

```text
S2:
dedicated branch pool
active root
```

```text
S3:
main-pool branch reuse
active root
```

这样可以分别估计：

\[
\Delta T_{\text{branch-pool}}
\]

和：

\[
\Delta T_{\text{root-compaction}}.
\]

---

# 20. 应记录的系统指标

每个 step 至少记录：

### 总体

```text
timing_s/step
timing_s/gen
GPU utilization mean/p50/p90
GPU memory
```

### Root

```text
planned_root_generation_seconds
capacity_correction_root_generation_seconds
root_active_sequences
root_dense_equivalent_sequences
root_active_efficiency
root_padding_sequences
```

### Branch

```text
branch_request_count
branch_replay_capacity
branch_execution_waves
branch_suffix_generation_seconds
branch_suffix_generation_waves
branch_suffix_active_efficiency
```

### Environment / CPU

```text
Ray worker count
host pids
CPU utilization
replay seconds
```

---

# 21. P1-S 的推荐验收标准

这些是工程建议，不是已有实验事实。

## 必须满足

1. 所有 trace/replay invariants 通过。
2. terminal leaf budget 仍严格为：
   \[
   R_g+Q_g=8.
   \]
3. Exact Batch-ERV request identity 不被 scheduler 修改。
4. branch replay validation 不下降。
5. deterministic regression 测试通过。

---

## 性能目标

### S1 branch reuse

典型有 branch 的 step：

\[
\boxed{\text{branch execution waves }\rightarrow1}
\]

至少不再由 16-worker pool 强制形成 3~6 个 cohort。

### S2 root compaction

应观察到：

\[
\text{root inactive sequences avoided}>0
\]

且后期 success 较高时收益更加明显。

### P1-S 综合

短 profile 中建议以：

\[
\boxed{step wall-clock 下降至少约 10\%}
\]

作为“值得保留”的经验门槛。

若 branch-heavy 后期 step 可下降 15~20% 更理想。

最终数值必须由 P0/P1-S H100 实测确认。

---

# 22. 公平性说明

Root active compaction 是一种通用 agent rollout 优化，理论上 GiGPO 也可使用。

因此论文最终 wall-clock 对比中：

- BACE 的 `main-pool branch reuse` 是 BACE-specific 优化；
- `root active compaction` 若对 GiGPO 同样适用，应同时给 GiGPO 开启。

否则可能把公共系统优化错误归因给 BACE。

---

# 23. P1-S 完成后的执行图

P1-S 完成后仍然保持阶段结构：

```text
planned roots
    ↓
[active-only selected execution]
    ↓
capacity check
    ↓
correction root wave
    ↓
[active-only selected execution]
    ↓
...
    ↓
freeze all task root backbones
    ↓
Exact Batch-ERV(Q)
    ↓
all frozen branch requests
    ↓
[reuse main 128-worker pool]
    ↓
active-only branch suffix
    ↓
PPO
```

注意：

> 仍然没有 root/branch 混跑，也没有 task-local frontier。

这正是 P1-S 的低风险边界。

---

# 24. P1-S 后续接口

如果 P1-S 验证有效，P2-S 可以自然继续：

```text
当前：
wave 内 active-only

P2-S：
跨 task 不再 barrier
```

即：

\[
\boxed{
\text{P1-S = 去掉 wave 内浪费}
}
\]

\[
\boxed{
\text{P2-S = 去掉 task 间不必要等待}
}
\]

P1-S 不应提前承担 P2-S 的复杂度。

---

# 25. 最终推荐

P1-S 的实施顺序：

\[
\boxed{
\text{S1：branch main-pool reuse}
\rightarrow
\text{S2：root active compaction}
\rightarrow
\text{短 H100 profile}
}
\]

只有在这两项稳定并获得明确收益以后，才进入 Exact Task-Local Frontier。

P1-S 的核心不是“让算法少生成 leaf”，而是：

\[
\boxed{
\text{同样的有效 leaf，用更少的物理 generation wave 和更少的 inactive sequence-step 完成。}
}
