# BACE-GiGPO P1-S 调度性能优化修复版

> 日期：2026-08-27  
> 适用范围：当前 BACE-GiGPO / verl-agent / ALFWorld GPU 生产路径  
> 目标：在**不修改算法定义、不修改 Exact 全局分配、不修改 packed root barrier、不修改 Replay 与 PPO 数据语义**的前提下，修复当前调度实现中的两个主要系统瓶颈。

---

## 0. 文档目的

本文件是在对当前仓库实现和 150-step 正式实验 trace 完成审计后形成的修复版实施说明。

本轮只处理系统调度问题，不改变 BACE-GiGPO 方法本身。核心优化为：

1. **S1：branch 阶段复用 main ALFWorld worker pool**，消除 16-worker dedicated branch pool 导致的串行 capacity chunking；
2. **S2：root 阶段引入 active-root executor**，避免已经终止的 root 继续进入 dense LLM generation。

本文将两项优化明确限定为：

```text
算法语义不变
    ↓
逻辑 rollout / branch request 不变
    ↓
仅修改 physical worker 调度与 model submission
    ↓
减少不必要的串行波次和无效 generation
```

因此，本文件不是新算法方案，也不改变当前论文中的 Exact Batch-ERV / root–branch allocation / replay / advantage / PPO 定义。

---

# 1. 当前生产配置与问题背景

当前正式 ALFWorld 训练配置为：

```text
train_batch_size = 16
total_leaf_budget = 8
min_natural_roots = 2
```

主 ALFWorld worker pool：

\[
16 \times 8 = 128 \text{ workers}
\]

当前 dedicated branch pool：

\[
16 \times 1 = 16 \text{ workers}
\]

生产脚本当前仍采用 Exact / packed / selected-worker 路径。相关实现位置包括：

- `run_bace_alfworld_gpu.sh`：生产调度配置；
- `main_ppo.py`：除 frontier 路径外创建额外 branch environment；
- `env_factory.py`：dedicated branch pool 当前按每个 task 1 worker 构建；
- `rollout_collector.py`：branch request 根据 `replay_capacity` 进行 chunk 执行；
- `rollout_loop.py`：root multi-turn loop 当前为 dense batch generation；
- `base.py`：底层 replay capacity 来源于 physical worker 数量。

本方案保持当前：

```text
Exact global allocation
+ packed staged roots
+ capacity correction
+ frozen root topology
+ replay-based branching
+ unified PPO training
```

不启用完整 frontier scheduling。

---

# 2. 审计结论摘要

## 2.1 S1：16-worker branch pool 是真实生产瓶颈

当前每个 task 的 branch 上限为：

\[
Q_g \le B-R_{\min}=8-2=6.
\]

16 个 task 的理论 batch branch 上限为：

\[
Q^{batch}_{\max}=16\times6=96.
\]

而 main pool 有：

\[
128 \text{ workers}.
\]

因此在当前配置下：

\[
96 < 128.
\]

也就是说，在 root topology 已冻结、branch requests 已经确定后，main pool 从容量上可以一次承载当前 batch 所有 branch requests。

150-step 正式实验实际执行了：

\[
5703 \text{ final branches}.
\]

branch scheduling cohort 数量分布为：

| 每 step 的 branch capacity chunk 数 | step 数 |
|---:|---:|
| 0 | 4 |
| 1 | 37 |
| 2 | 25 |
| 3 | 19 |
| 4 | 47 |
| 5 | 18 |

146 个存在 branch 的 step 中：

- 37 个可以在 dedicated pool 中一轮 capacity chunk 完成；
- 109 个必须拆成至少两个串行 chunk；
- 多 chunk 比例：

\[
\frac{109}{146}\approx74.7\%.
\]

最大实际 branch 数为 76：

\[
\lceil 76/16\rceil=5.
\]

例如 step 150：

```text
63 branches
→ 16 + 16 + 16 + 15
→ 4 个串行 capacity chunks
```

该 step 的 branch suffix generation 约为 177.7 s。

全部 150 step 中，branch suffix generation 累计约：

\[
21815\text{ s}\approx6.06\text{ h}.
\]

因此，S1 不是理论优化，而是正式实验中持续存在的并行度瓶颈。

---

## 2.2 S2：root dense executor 确实继续为 done trajectories 提交生成

当前 `_collect_root_wave()` 仍进入 `vanilla_multi_turn_loop()`。

虽然循环计算：

```python
active_masks = np.logical_not(is_done)
```

但 model generation 前没有将 batch compact 为 active roots。

当前逻辑实质上类似：

```text
logical roots = N
active roots = M < N

仍然：
N prompts
→ N-row tokenize / padding
→ dense generate submission
→ decode
→ 后处理阶段再用 active mask 屏蔽无效结果
```

因此，`active_masks` 当前主要保证 reward / episode length / training data 的语义正确，但没有避免 inactive root 继续占用 model-side rollout submission。

正式 trace 根据 root 的实际 episode length 重建得到：

| Training step | 有效 root sequence-step | 当前 dense sequence-step | inactive 比例 | 考虑 4-card padding 后可减少的提交 |
|---:|---:|---:|---:|---:|
| 1 | 5,932 | 6,400 | 7.3% | 6.1% |
| 50 | 5,310 | 5,950 | 10.8% | 9.6% |
| 100 | 1,597 | 2,556 | 37.5% | 27.9% |
| 150 | 1,002 | 2,402 | 58.3% | 47.2% |

随着模型训练后期成功率提高，部分任务更早结束，因此 dense inactive generation 浪费显著增加。

需要强调：

> 上表是 sequence submission / logical sequence-step 的减少空间，不等价于 wall-clock 加速比例。

实际收益仍由：

- prompt/token 长度；
- vLLM continuous batching；
- GPU 饱和程度；
- padding；
- Ray 调度；
- environment latency；
- decode/prefill 比例；

共同决定。

---

# 3. 总体修复原则

两项优化都必须满足以下不变量。

## 3.1 算法不变量

必须保持：

```text
same logical root count
same final root/branch topology
same Exact Batch-ERV allocation rule
same capacity-correction rule
same frozen root backbone
same legal anchor set
same frozen ReplayRequest semantics
same final leaf budget
same PPO training semantics
```

系统优化只能改变：

```text
physical worker mapping
batch composition
capacity chunking
active request compaction
execution ordering among logically independent requests
```

不能改变：

```text
which evidence is collected
which branch is selected
which logical trajectory owns an occurrence
which rollout contributes to training
```

---

## 3.2 logical identity 与 physical slot 必须解耦

必须明确：

\[
\boxed{\text{logical trajectory identity}\neq\text{physical worker slot}}
\]

logical identity 包括：

- task identity；
- root identity；
- branch identity；
- `uid`；
- `traj_uid`；
- `occurrence_id`；
- `task_batch_index`；
- root event ordering；
- replay origin identity。

physical slot 只是当前一次 simulator/model execution 使用的临时资源。

S1 和 S2 都必须围绕这一原则实现。

---

# 4. S1：复用 main pool 执行 branch

## 4.1 当前问题

当前 dedicated branch pool 容量为：

\[
C_{branch}=16.
\]

当本 step branch request 数为 \(N_{branch}\) 时，当前执行需要：

\[
W_{branch}=\left\lceil\frac{N_{branch}}{16}\right\rceil
\]

个 capacity chunks。

这些 chunks 之间存在人为串行 barrier。

例如：

```text
63 requests
capacity = 16

chunk 1: 16
chunk 2: 16
chunk 3: 16
chunk 4: 15
```

即使 GPU 和 main simulator pool 有更多并行资源，这 63 个逻辑 branch 仍然被 dedicated pool 限制为四组。

---

## 4.2 修复目标

增加可切换的 branch pool mode：

```text
branch_pool_mode = dedicated
branch_pool_mode = main_reuse
```

### dedicated

保持现有实现，作为：

- correctness fallback；
- A/B baseline；
- 回归定位路径。

### main_reuse

root 阶段完成且 branch requests 冻结之后，使用 main ALFWorld physical worker pool 执行 branch replay。

当前 main replay capacity：

\[
C_{main}=128.
\]

当前 branch 理论上限：

\[
N_{branch}\le96.
\]

因此当前生产配置通常可以在一个 **branch scheduling cohort** 中装入全部 requests。

---

## 4.3 Replay 语义为什么允许 main pool reuse

branch execution 不依赖 root worker 的 live runtime state。

`ReplayRequest` 已保存恢复 branch 所需的逻辑证据，包括：

- concrete gamefile / reset key；
- executable action prefix；
- prefix observations；
- target anchor；
- admissible actions；
- copied origin response；
- selected canonical action；
- remaining horizon；
- logical identity metadata。

branch 的执行语义为：

```text
reset same game/session
→ replay action prefix
→ validate restored observation / anchor
→ validate selected action
→ execute copied origin action
→ generate fresh suffix
```

因此在以下对象冻结之后：

```text
root_output
root_logs
AnchorIndex
frozen branch requests
```

root workers 的 live simulator state 已不属于 BACE 的必要算法证据。

所以：

\[
\boxed{\text{使用 main physical pool 做 replay 不改变 branch 的算法定义。}}
\]

---

## 4.4 修正后的 slot 映射规则

### 不推荐

不要把 physical worker stride 定义为：

```python
slot = task_batch_index * total_leaf_budget + task_local_branch_index
```

当前配置下：

```text
total_leaf_budget = 8
main rollout group size = 8
```

二者恰好相等，所以数值上可用，但概念上不应绑定。

`total_leaf_budget` 是算法预算；physical stride 是执行器 worker layout。

### 推荐

定义：

```python
slot = task_batch_index * main_group_size + task_local_slot
```

其中 `main_group_size` 必须来自当前实际 main environment / rollout worker layout，例如 runtime 的 rollout group size，而不是从 BACE leaf budget 推导。

当前可以增加一致性检查：

```python
assert total_leaf_budget == main_group_size
```

但该 assert 只是当前生产配置约束，不应替代 physical slot 定义。

### 必须验证

```text
0 <= task_batch_index < task_batch_size
0 <= task_local_slot < main_group_size
slot < replay_capacity
allocated physical slots are unique within a cohort
```

---

## 4.5 不允许硬编码“永远一波”

当前：

\[
N_{branch}^{max}=96<C_{main}=128.
\]

所以生产配置下通常可以：

```text
all frozen branch requests
→ one capacity cohort
```

但实现必须保留通用 fallback：

```python
if len(requests) <= replay_capacity:
    execute_one_cohort(requests)
else:
    for chunk in safe_chunks(requests, replay_capacity):
        execute_one_cohort(chunk)
```

不得硬编码：

```text
capacity = 128
branch_execution_waves = 1
```

未来如果修改：

- task batch size；
- total leaf budget；
- minimum roots；
- rollout group size；
- worker count；

请求数仍可能重新超过 replay capacity。

---

## 4.6 “one wave” 的术语修正

建议以后不用模糊的：

```text
branch_execution_waves = 1
```

而使用：

```text
branch_capacity_cohorts = 1
```

或：

```text
branch_scheduler_chunks = 1
```

原因是 branch 仍然是 multi-turn agent rollout。

即使 63 branches 一次性装入 main pool，仍然会经历：

```text
63 active branches
→ LLM turn 1
→ env.step
→ remaining active branches
→ LLM turn 2
→ ...
```

S1 消除的是：

```text
16 → 16 → 16 → 15
```

这种人为 capacity chunk 串行，不是把完整 multi-turn branch suffix 压缩成一次 generation。

---

## 4.7 pool ownership / lifecycle

main pool reuse 不能简单实现成：

```python
branch_env = main_env
```

然后沿用 dedicated branch adapter 的生命周期管理。

必须避免：

```text
branch adapter close()
→ 意外 close main pool
→ trainer 后续 root rollout 失效
```

推荐显式记录 ownership：

```text
dedicated branch pool:
    owns_pool = True

main reused pool:
    owns_pool = False
```

或者等价的资源生命周期设计。

必须覆盖：

- construction；
- reset；
- replay；
- exception cleanup；
- close / shutdown；
- next training step reuse。

---

## 4.8 S1 correctness tests

### T-S1-1：dedicated vs main-reuse deterministic equivalence

使用 deterministic/mock policy。

固定：

- same roots；
- same AnchorIndex；
- same frozen branch requests；
- same copied origin actions；
- deterministic suffix outputs。

比较：

```text
ReplayRequest identity
restored anchor identity
selected canonical action
final success/reward
remaining horizon handling
branch logical uid/traj_uid
training mask
final leaf ownership
```

要求 logical outputs 完全一致。

---

### T-S1-2：当前最大理论 96 branches

构造：

```text
16 tasks × 6 branches = 96 requests
main replay capacity = 128
```

要求：

```text
one branch capacity cohort
96 unique logical branch IDs
96 valid physical slot assignments
no slot collision
all replay checks pass
```

---

### T-S1-3：capacity overflow fallback

人工构造：

```text
requests > main replay capacity
```

例如：

```text
129 requests
200 requests
```

要求：

```text
safe chunk fallback
no assert caused by current hardcoded configuration
no dropped request
no duplicated request
no slot collision
```

---

### T-S1-4：pool lifecycle

连续多个 training steps：

```text
root using main pool
→ branch reuse main pool
→ next step root using same main pool
```

要求 main pool 在 branch adapter 生命周期结束后仍正常可用。

---

### T-S1-5：Replay identity

对 branch replay 验证：

```text
same reset/game key
same executable prefix
same restored target observation
same anchor key
same selected action legality
same copied origin response
```

若 replay validation 失败，必须走原有 failure handling，而不能静默接受错位 state。

---

# 5. S2：root active executor

## 5.1 当前问题

当前 packed root wave 保持固定 logical batch shape。

若某一 turn：

```text
N roots total
M roots active
N-M roots done
```

当前 model-side path 仍然接近：

```text
construct N prompts
→ tokenize N rows
→ pad
→ generate_sequences(N rows)
→ decode
→ apply active mask later
```

因此已 done 的 root 仍占用 LLM inference submission。

---

## 5.2 修复目标

保留 packed staged root semantics，但在每个 multi-turn generation turn 前对 active logical roots 做 gather/compact：

```text
logical root table: N rows
        ↓
active indices: M rows
        ↓
gather active root inputs
        ↓
selected physical slots / compact generation
        ↓
generate only M active roots
        ↓
environment execution for active logical roots
        ↓
scatter outputs back to logical root table
```

关键是：

\[
\boxed{\text{packed barrier 保留，但 dense model submission 被 active compaction 替代。}}
\]

---

## 5.3 不允许直接启用完整 frontier scheduler

仓库中 `frontier.py` 已包含 selected-slot / selected-worker generation 可复用机制。

但当前生产 Exact 路径仍要求：

```text
staged_root_batching = packed
```

因此正确做法是：

```text
frontier selected-slot execution primitive
        ↓
抽取 / 复用
        ↓
packed root active executor
```

而不是：

```text
packed root scheduler
→ frontier scheduler
```

后者会改变：

- barrier；
- root/correction timing；
- task interleaving；
- scheduling semantics；

不再属于纯工程优化。

---

## 5.4 S2 必须保持的 logical fields

active compaction 后，physical slot 可以变化，但以下 logical metadata 必须完整保留。

### Identity

- `uid`
- `traj_uid`
- `occurrence_id`
- `task_batch_index`
- root index / local root index
- root event index / ordering

### Environment

- `environment_reset_key`
- gamefile/session identity
- current observation
- action prefix / trajectory state
- admissible action set
- action identity
- remaining horizon

### Episode state

- `is_done`
- reward
- success
- episode length
- terminal reason（若已有）

### Training / BACE evidence

- root event ownership
- root_output row mapping
- root_logs row mapping
- natural occurrence identity
- anchor occurrence identity
- capacity-correction root identity

尤其要覆盖 correction roots。

---

## 5.5 planned roots 与 correction roots 必须使用同一 active executor

S2 不能只优化初始 planned root wave。

必须同时覆盖：

```text
planned roots
capacity-correction roots
```

原因是 correction root 仍然会改变后续算法证据：

```text
RootEventLog
→ AnchorIndex
→ action evidence
→ posterior
→ BERV / capacity
→ final topology
```

如果 correction path 仍使用 dense executor：

- 性能收益不完整；
- 两套 root execution semantics 增加维护复杂度；
- 容易产生 logical mapping 差异。

推荐统一成：

```text
_collect_root_wave(...)
    ↓
shared packed-active executor
```

planned/correction 只在 logical root type 上不同，不在 executor 上分叉。

---

## 5.6 environment-side 浪费的表述边界

目前确定的是：

```text
inactive roots 仍进入 dense model-side generation path
```

因此确定性浪费主要包括：

- prompt construction；
- tokenization；
- padding；
- model submission；
- generation；
- decode；
- dense batch tensor handling。

对于 environment step 的实际额外开销需要更谨慎。

即使上层调用保持 dense shape，底层 manager 可能对 done workers 做：

```text
no-op
mask
cheap skip
```

因此正式表述应为：

> S2 已确认可以减少 inactive model generation；environment-side 的额外收益应通过 profile 单独测量，不应在实现前假定与 model-side reduction 等比例。

---

## 5.7 S2 correctness tests

### T-S2-1：dense vs active deterministic equivalence

使用 deterministic/mock policy。

在同一逻辑 root batch 上分别运行：

```text
old dense executor
new active executor
```

要求最终 logical results 完全一致：

```text
same logical action sequence
same reward/success
same episode length
same root ordering
same root_output ownership
same root_logs ownership
same AnchorIndex inputs
same BACE topology inputs
```

physical worker slot 无需一致。

---

### T-S2-2：mixed done states

人工构造每个 turn active 数量变化：

```text
128 → 101 → 64 → 17 → 3 → 0
```

检查：

- only active roots submitted；
- done roots 不被再次生成；
- logical order/scatter 正确；
- no duplicated actions；
- no missing actions。

---

### T-S2-3：4-card padding / divisibility

对 active count 测试：

```text
1, 2, 3, 4, 5, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127
```

保证底层 rollout engine 所需 padding 不导致：

- logical duplicate；
- fake environment actions；
- fake root events；
- extra training occurrences。

padding 只能存在于 physical/model layer。

---

### T-S2-4：capacity-correction roots

构造会触发 correction root 的 deterministic case。

比较 old/new executor：

```text
same correction count
same correction root identity
same natural evidence
same AnchorIndex
same final branch capacity
same final branch allocation
```

---

### T-S2-5：early-success / long-tail mixture

构造：

```text
部分 root 很早 success
部分 root 长时间继续
```

验证 done trajectories 从后续 model submission 中消失，而长尾 trajectories 正常继续。

---

# 6. 随机 rollout 的 equivalence 标准

S1/S2 会改变：

- batch composition；
- request ordering；
- physical worker mapping；
- vLLM batching timing；
- GPU kernel execution grouping。

因此即使：

```text
same initial checkpoint
same seed
same prompts
```

真实 stochastic rollout 也不应要求逐 token 一致。

## 6.1 correctness equivalence

必须使用 deterministic/mock tests 判断算法语义是否保持。

关注：

```text
logical identities
replay identity
root/branch topology
reward/success
training masks
evidence ownership
```

## 6.2 stochastic performance comparison

真实模型下使用：

```text
same initial checkpoint
same algorithm configuration
same hardware
multiple short repeated profiles
```

比较统计量，而不是比较 token-by-token rollout。

---

# 7. 推荐实现顺序

## Phase 0：冻结 S0 baseline

保存当前正式生产代码路径和 profile。

必须记录：

- commit；
- script/config；
- checkpoint；
- GPU 型号与数量；
- vLLM backend；
- batch size；
- total leaf budget；
- branch pool mode；
- wall-clock breakdown。

---

## Phase 1：实现 S1 main-pool branch reuse

先只修改 branch physical execution，不动 root executor。

新增：

```text
branch_pool_mode = dedicated | main_reuse
```

完成：

- capacity-aware slot assignment；
- ownership handling；
- overflow fallback；
- deterministic replay tests；
- 96-branch test；
- lifecycle test。

---

## Phase 2：H100 S0/S1 短 profile

在相同 checkpoint 上运行：

```text
S0 = current dedicated branch pool
S1 = main pool branch reuse
```

此阶段不要同时引入 S2。

目标是单独回答：

> dedicated branch pool 的实际 wall-clock penalty 是多少？

---

## Phase 3：实现 S2 active-root executor

保持 S1 可关闭。

抽取 selected-slot execution 机制，但保留 packed staged semantics。

统一覆盖：

```text
planned roots
correction roots
```

完成 deterministic dense/active equivalence 测试。

---

## Phase 4：四组独立 profile

定义：

### S0

```text
old root dense executor
+ dedicated branch pool
```

### S1

```text
old root dense executor
+ main branch pool reuse
```

### S2

```text
active root executor
+ dedicated branch pool
```

### S3

```text
active root executor
+ main branch pool reuse
```

必须让 S1 与 S2 可独立开关，否则无法识别单项贡献。

---

# 8. H100 profile 指标

至少记录以下指标。

## 8.1 End-to-end

- total step wall-clock；
- rollout wall-clock；
- training/update wall-clock；
- environment wall-clock；
- branch suffix wall-clock。

## 8.2 S1-specific

- branch request count；
- branch replay capacity；
- branch capacity cohort count；
- requests/cohort；
- generation time/cohort；
- average active branch count per model turn；
- branch GPU utilization；
- branch tokens/s。

## 8.3 S2-specific

- logical root count；
- active root count per turn；
- dense-submission counter（S0/S1）；
- compact-submission counter（S2/S3）；
- inactive rows avoided；
- padded physical rows；
- root tokens/s；
- prefill/decode time；
- GPU utilization。

## 8.4 Correctness / semantics

- root count；
- correction root count；
- final branch count；
- leaf count；
- replay validation failure count；
- invalid slot assignment count；
- duplicate logical ID count；
- final success/reward distribution。

---

# 9. 性能比较方法

定义：

\[
T_0=T(S0),\quad T_1=T(S1),\quad T_2=T(S2),\quad T_3=T(S3).
\]

单独收益：

\[
\text{speedup}_{S1}=\frac{T_0}{T_1},
\]

\[
\text{speedup}_{S2}=\frac{T_0}{T_2},
\]

组合收益：

\[
\text{speedup}_{S3}=\frac{T_0}{T_3}.
\]

还可以比较：

\[
(T_0-T_1)+(T_0-T_2)
\]

与：

\[
T_0-T_3
\]

判断两项优化收益是：

- 近似独立；
- 部分重叠；
- 存在正向协同。

---

# 10. “综合加速 10%–20%”应如何表述

当前证据支持：

```text
S1 是真实且高频的串行 capacity bottleneck
S2 是真实且随训练后期增大的 inactive generation waste
```

但当前证据**不能直接证明**：

```text
end-to-end wall-clock 一定提升 10%–20%
```

原因：

### S1

主要减少：

- capacity chunk barrier；
- small-batch inefficiency；
- under-utilization；
- repeated scheduling overhead。

但不减少真正需要执行的 branch decisions/token。

### S2

确实减少无效 generation submission，但实际 wall-clock 收益仍不是 linear sequence reduction。

因此正式表述应为：

> 10%–20% 是当前基于 trace 和系统结构提出的合理工程目标区间，不是现有实验已经证明的结果。是否达到该范围必须由 H100 S0/S1/S2/S3 A/B profile 验证。

---

# 11. 建议新增的配置项

推荐至少增加：

```text
bace.branch_pool_mode:
    dedicated
    main_reuse

bace.root_active_executor:
    false
    true
```

可增加 debug 配置：

```text
bace.assert_slot_identity = true
bace.assert_replay_identity = true
bace.log_capacity_cohorts = true
bace.log_active_root_counts = true
```

生产前建议默认：

```text
assert_slot_identity = true
assert_replay_identity = true
```

若性能影响极低，可长期保留。

---

# 12. 推荐日志字段

## Branch

```text
branch_requests_total
branch_replay_capacity
branch_capacity_cohorts
branch_requests_per_cohort
branch_pool_mode
branch_replay_validation_failures
branch_slot_collisions
branch_generate_wallclock
branch_env_wallclock
```

## Root

```text
root_logical_count
root_active_count_per_turn
root_inactive_count_per_turn
root_model_rows_submitted
root_padding_rows_submitted
root_rows_avoided
root_generate_wallclock
root_env_wallclock
root_active_executor_enabled
```

## Identity

```text
duplicate_uid_count
duplicate_traj_uid_count
invalid_task_batch_index_count
invalid_physical_slot_count
logical_physical_mapping_errors
```

---

# 13. 验收标准

## 13.1 S1 correctness gate

S1 不得进入正式 profile，除非：

- deterministic dedicated/main-reuse outputs 逻辑一致；
- 96-branch test 通过；
- overflow fallback 通过；
- replay identity 通过；
- no slot collision；
- main pool lifecycle 通过。

## 13.2 S2 correctness gate

S2 不得进入正式 profile，除非：

- dense/active deterministic outputs 逻辑一致；
- mixed done-state test 通过；
- padding/divisibility test 通过；
- correction-root test 通过；
- no UID/traj mapping error；
- final BACE evidence inputs 一致。

## 13.3 Production gate

S3 进入生产前至少满足：

```text
correctness tests all pass
no replay semantic regression
no root/branch count regression
no PPO data ownership regression
H100 wall-clock profile shows meaningful positive gain
```

不应仅凭 theoretical submission reduction 切换生产默认值。

---

# 14. 当前推荐的最终实施方案

按照风险和收益，建议顺序如下。

### P1-S1

首先实现：

```text
main/dedicated 可切换 branch pool
```

采用 main physical pool 时：

```text
freeze root backbone
→ freeze branch requests
→ capacity-aware physical slot assignment
→ replay through main pool
→ safe chunk fallback when requests > capacity
```

并保留 dedicated implementation 不删除。

### P1-S1 tests

优先完成：

1. deterministic equivalence；
2. 96-branch current-max test；
3. >capacity fallback；
4. replay identity；
5. pool lifecycle。

### P1-S1 profile

做 S0/S1 H100 短 profile，得到真实 branch scheduling 收益。

### P1-S2

再实现：

```text
packed root active executor
```

只抽取 frontier 中可复用的 selected-slot execution primitive，不启用完整 frontier scheduling。

planned/correction roots 全覆盖。

### Final profile

运行：

```text
S0
S1
S2
S3
```

再依据真实 wall-clock 决定生产默认配置。

---

# 15. 最终判断

当前审计已经足以确认：

## S1

16-worker dedicated branch pool 是正式实验中的真实串行瓶颈。

当前配置下：

\[
Q^{batch}_{max}=96<128=C_{main},
\]

所以 root topology 冻结后复用 main pool 是合理且语义安全的主要修复方向。

## S2

当前 packed root rollout 的 `active_masks` 没有在 model generation 前进行真正的 active compaction。

训练后期 inactive root 比例显著上升，因此 active-root executor 有明确的优化价值。

## 语义安全

两项优化都可以设计为：

\[
\boxed{\text{只改变 execution scheduling，不改变 BACE algorithm semantics。}}
\]

但必须坚持：

- logical identity 与 physical slot 解耦；
- slot stride 来自实际 main worker layout，而非硬编码 leaf budget；
- capacity overflow 始终保留 fallback；
- main pool reuse 明确处理 ownership/lifecycle；
- S2 覆盖 planned 和 correction roots；
- correctness 通过 deterministic tests 验证；
- stochastic rollout 不要求逐 token 一致；
- 最终性能结论必须以 H100 wall-clock A/B 为准。

因此当前可以停止继续论证“这两个问题是否存在”，进入 S1 实现与专项回归测试阶段。

