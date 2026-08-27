# BACE H100 稳定性作业卡死问题说明

> **状态提示（2026-08-27）：历史事故现场。** 指数爆炸问题已经由 quota-aware Exact DP 修复，之后的主实验已完成 150 step。本文中的作业状态和“正式链尚未提交”不再代表当前进度；请以 [`docs/01_当前实现状态与未完成事项.md`](../docs/01_当前实现状态与未完成事项.md) 为准。

更新时间：2026-08-19 03:59（Europe/Berlin）

## 1. 执行摘要

当前 4×H100 稳定性作业 `3051997` 没有发生 CUDA OOM、磁盘不足或 branch replay 容量溢出，而是卡在 step 3 的 Exact Batch-ERV 全局 branch 分配计算中。

作业已经完成 step 1 和 step 2。step 3 的自然 root rollout、anchor 建立和容量检查也已完成，但在生成 `acquisition_rounds` 和实际 branch replay 请求之前停止前进。GPU 已长期处于 0% 利用率，训练日志超过一小时没有更新，但作业进程和资源监控仍然存活。

根因是 `ExactBatchErvEngine.global_allocations()` 当前先枚举所有 anchor 容量的笛卡尔积，再过滤出 branch 数之和等于 quota 的分配。step 3 的一个任务只有 `branch_quota=1`，实际只需比较 26 个单 branch 方案，但当前实现会尝试遍历约 1.69 万亿个组合，造成事实上无法完成的指数级计算。

正式 step 150 作业链尚未提交，现有 fail-closed 门禁正常发挥了作用。

## 2. 当前作业状态

截至 2026-08-19 03:59：

| 项目 | 状态 |
|---|---|
| Slurm Job ID | `3051997` |
| 作业名 | `bace15b_stability` |
| 节点 | `n25g0003` |
| 资源 | 4×H100、64 CPU |
| 开始时间 | 2026-08-19 02:23:36 |
| Slurm 状态 | `RUNNING` |
| step 1 | 完成，`training/global_step:1` |
| step 2 | 完成，`training/global_step:2` |
| step 3 | root 阶段完成，Exact Batch-ERV acquisition 卡死 |
| 正式链 manifest | 不存在 |
| 正式 checkpoint tracker | 不存在 |

作业仍占用 4 张 H100，但 GPU 只保留约 7–8 GiB 基础显存、利用率为 0%。建议立即停止该作业，避免继续浪费配额：

```bash
scancel 3051997
```

## 3. 关键文件和现场证据

实验的用户可见路径保持不变：

```text
/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/
```

该路径目前是符号链接，实际数据位于项目存储：

```text
/hpcwork/rwth2089/xsz96350/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/
```

相关现场文件：

```text
slurm/bace15b_stability-3051997.out
slurm/bace15b_stability-3051997.err
stability_profile/logs/stability_mb16_job3051997/
  3051997_stability_mb16_job3051997_steps1to3.log
stability_profile/bace_artifacts/stability_mb16_job3051997/
  step_00000001/
  step_00000002/
  step_00000003/
stability_profile/run_metadata/stability_mb16_job3051997/
  3051997_stability_mb16_job3051997_steps1to3/
    host_resource_samples.csv
    gpu_samples.csv
```

### 3.1 step 1 和 step 2 正常完成

step 1 和 step 2 均出现完整训练指标：

```text
training/global_step:1
training/global_step:2
```

step 2 的 BACE 关键指标正常：

```text
bace/requested:4
bace/validated:4
bace/skipped:0
bace/branch_replay_capacity:16
bace/branch_execution_waves:1
bace/branch_execution_max_wave_size:4
bace/staged_root_batching_packed:1
bace/batch_erv_exact:1
```

这说明以下链路在本次作业中已经真实跑通：

- 4 卡模型初始化和 FSDP；
- 128 条 root rollout；
- dynamic/staged/packed root 生成；
- Exact Batch-ERV 在小规模候选上的规划；
- branch replay；
- PPO 更新；
- step artifact 写入。

### 3.2 step 3 停止位置

step 3 已写入：

- `manifest.json`；
- `family_topology_plans.jsonl`；
- `capacity_checks.jsonl`；
- `anchors.jsonl`；
- `roots.jsonl`；
- `leaves.jsonl`；
- `topology.jsonl`。

数量包括：

```text
roots:   124
leaves:  5872
anchors: 401
```

但 step 3 没有写入：

- `acquisition_rounds.jsonl`；
- `branch_execution_waves.jsonl`；
- `branches.jsonl`；
- `family_history_updates.jsonl`；
- `trainable_occurrences.jsonl`；
- `summary.json`。

对应代码顺序表明，程序已经完成 root trace 和 topology trace，随后进入：

```python
self.coordinator.initialize(...)
```

但尚未返回到：

```python
requests = self.coordinator.build_round_requests()
self.artifact_store.append("acquisition_rounds", ...)
```

因此卡点位于 Exact Batch-ERV coordinator 初始化期间，而不是 branch replay 执行期间。

## 4. 根因分析

### 4.1 当前算法

问题代码位于：

```text
verl-agent-src/recipe/bace_gigpo/batch_erv.py
```

`ExactBatchErvEngine.global_allocations()` 的核心逻辑为：

```python
ranges = [range(designs[anchor_id].capacity + 1) for anchor_id in anchor_ids]
for sizes in itertools.product(*ranges):
    if sum(sizes) != branch_quota:
        continue
    ...
```

设：

- anchor 数为 `A`；
- 第 `i` 个 anchor 的容量为 `c_i`；
- 当前任务的 branch quota 为 `Q`。

当前实现遍历的状态数为：

```text
Π_i (c_i + 1)
```

但真正有意义的仅是满足以下约束的状态：

```text
Σ_i k_i = Q,  0 <= k_i <= c_i
```

当 anchor 多而 quota 很小时，两者差距极大。

### 4.2 本次 step 3 的实际规模

step 3 有 4 个任务获得一个 branch quota。根据 `capacity_checks.jsonl`：

| 任务 | quota | anchor 数 | 有容量的 anchor | 当前笛卡尔积状态数 | quota=1 的实际可行分配数 |
|---|---:|---:|---:|---:|---:|
| `6228a346-...` | 1 | 13 | 10 | 39,366 | 10 |
| `b0f2871a-...` | 1 | 16 | 15 | 9,565,938 | 15 |
| `cec9d9f3-...` | 1 | 15 | 10 | 59,049 | 10 |
| `faa5e392-...` | 1 | 27 | 26 | **1,694,577,218,886** | **26** |

最后一个任务导致程序尝试枚举约 1.69 万亿个状态。即使单个状态只花极少时间，也无法在 Slurm 作业时限内完成。

这也是为什么：

- step 1 没有 branch，因此没有触发问题；
- step 2 虽然有 4 个 branch，但候选规模相对较小，仍能完成；
- step 3 的 root rollout 已完成，GPU 随后空闲；
- 没有异常栈，因为程序不是崩溃，而是在合法但不可完成的循环中运行。

## 5. 与其他已知问题的关系

### 5.1 不是 branch replay 池容量问题

之前确认的问题是：branch 请求数可能大于 16 个 replay worker，旧实现会一次性提交全部请求并报错。

该问题已经通过 request chunking 修复，并以真实 ALFWorld replay 验证过：

- 17 个请求被拆成 `[16, 1]`；
- 48 个请求被拆成 `[16, 16, 16]`；
- identity 和顺序保持一致。

本次 step 3 甚至尚未生成 `acquisition_rounds`，因此还没有进入 `_execute_round()`，与 replay 分块无关。

### 5.2 不是 GPU OOM

卡住后四张 GPU 的利用率为 0%，显存占用约 7–8 GiB。日志中没有 CUDA OOM。

训练阶段显存仍然较紧，之前观察到单卡峰值接近 H100 80GB 上限，但这不是当前停顿原因。

### 5.3 不是存储不足

实验已经迁移到 `rwth2089` 项目 HPCWORK，当前约有 2.8 TiB 空余。正式链前和每个正式 segment 前也增加了 64 GiB fail-closed 存储检查。

本次 artifact 均能正常写入，没有 `No space left on device`。

### 5.4 cgroup 线程数很高，但不是本次直接卡点

监控显示作业线程数在模型和 Ray worker 初始化后升至约 27,500。Ray 日志也报告启动了数百个 Python worker。

这个数量值得后续继续治理，但它在 step 1 开始前已经达到高位，且 step 1、step 2 仍能完成。线程数此后基本稳定，没有在 step 3 卡点继续增长。因此目前证据更支持 Exact Batch-ERV 组合枚举是直接原因。

同时需注意：当前 `cgroup.threads` 是任务压力诊断值，12,000 是工程安全阈值，并不是 RWTH 暴露的真实 `pids.max`。

## 6. 影响范围

如果不修复：

- 只要某个 task 的有效 anchor 较多，即使 branch quota 只有 1，也可能卡死；
- 单测中的小型人工数据无法覆盖真实 ALFWorld anchor 规模；
- 3-step 稳定性作业不能可靠完成；
- 正式训练可能在任意后续 step 卡住；
- GPU 会被占用但没有训练吞吐；
- 依赖链不会继续，但会浪费当前 segment 的 GPU 配额。

好的一面是，稳定性门禁在正式链提交前发现了问题。当前不存在错误的 step 150 作业链，也没有产生需要恢复的正式 checkpoint。

## 7. 建议修复方案

### 7.1 保留旧实现

按现有维护要求，不删除原笛卡尔积实现。建议：

- 将现有实现保留为 `_global_allocations_cartesian_reference()`；
- 仅用于小规模单测和结果对照；
- 正式路径改用 quota-aware 动态规划。

### 7.2 使用受 quota 约束的动态规划

目标仍然完全相同：

```text
maximize Σ_i value_i(k_i)
subject to Σ_i k_i = Q
           0 <= k_i <= c_i
```

动态规划状态可以定义为：

```text
DP[i, q] = 使用前 i 个 anchor、总共分配 q 个 branch 时的最优值
```

每个状态只尝试：

```text
k = 0 .. min(c_i, q)
```

基础复杂度约为：

```text
O(A * Q * max_capacity)
```

在本实验中 `Q <= 6`、单 anchor 容量最多为 2，因此与 anchor 数近似线性，而不是指数级。

### 7.3 精确处理 tie 和均匀选择

Exact Batch-ERV 不能为了速度破坏 tie 语义。建议每个 DP 状态保存：

- 最优 value；
- tie-optimal predecessor；
- tie-optimal 路径总数；
- 用于复现的稳定 hash/seed 信息。

最终可通过按路径计数加权的回溯，在所有 tie-optimal allocation 中做严格均匀选择，而不需要把所有 allocation 全部物化到内存。

对于 artifact：

- tie 数量较小时，可继续记录完整 `tie_optimal_global_allocations`；
- tie 数量较大时，应记录压缩 DP/DAG、总 tie count、选中路径和确定性摘要；
- trace validator 应验证选中 allocation 属于最优 tie 集合且重算值一致。

### 7.4 增加运行时防护

即使采用 DP，也建议增加：

- coordinator 初始化耗时指标；
- anchor 数、有效 anchor 数、quota、理论状态数；
- reference 笛卡尔积最大状态阈值；
- 超过阈值时禁止进入 reference 实现；
- 每个 acquisition 阶段的显式开始/完成日志；
- 稳定性脚本的无进展 watchdog。

watchdog 应基于训练日志或 artifact 心跳，而不是仅看进程是否存活。若 GPU 长期为 0%、训练日志和 step summary 均无更新，应主动退出非零状态，让 `afterok` 链保持关闭。

## 8. 专项测试要求

修复后至少增加以下测试：

1. 小规模随机问题：DP 与旧笛卡尔积 reference 的最优值、全部 tie 和选中结果一致。
2. `quota=0`：返回空 allocation。
3. `quota=1`、26 个有效 anchor：只处理线性数量候选，快速完成。
4. `quota=2`、大量 anchor：不构造完整笛卡尔积。
5. 不同 anchor capacity（0/1/2）的混合情况。
6. 容量不足时仍然抛出原有错误。
7. 浮点 tie tolerance 保持现有口径。
8. 固定 seed 和 `policy_update_id` 下选择结果可复现。
9. tie 数量巨大时不物化全部 allocation，但均匀回溯结果合法。
10. artifact/trace validator 能处理压缩后的 tie 诊断。
11. 使用本次 step 3 的容量结构构造回归 fixture，防止再次出现万亿级枚举。

性能验收建议：

```text
A = 100
Q = 6
max_capacity = 2
```

在普通 CPU 上应于秒级内完成，不应创建与 `3^A` 同阶的对象或循环。

## 9. 修复后的执行顺序

1. 停止 `3051997`。
2. 保留本次不完整 step 3 artifact，作为问题复现证据。
3. 将旧全量笛卡尔积实现改名保留。
4. 实现 quota-aware DP 和精确 tie 采样。
5. 运行新增专项测试。
6. 运行完整 `tests/bace_gigpo` 回归。
7. 使用本次 step 3 容量结构做 CPU 性能回归。
8. 重新提交 4×H100 三步稳定性作业。
9. 验收 step 3 出现：

   ```text
   training/global_step:3
   trace_validation ok=true
   artifact summary status=complete
   ```

10. 确认 cgroup task 诊断、存储门禁和 GPU 状态均正常。
11. 只有上述检查全部成功后，才自动提交 step 20 至 step 150 的正式 `afterok` 链。

## 10. 当前结论

整体训练架构、checkpoint 恢复、存储迁移、branch replay 分块和 Slurm `afterok` 链仍然可行。

当前唯一已证实的直接阻塞是 Exact Batch-ERV 全局 allocation 的指数级枚举。该问题必须在再次运行稳定性作业和正式实验之前修复。现有门禁已经成功避免把问题带入正式 step 150 作业链。

## 11. 简要说明
假设有 26 个 anchor，每个 anchor 都有三种选择：
分配 0 个 branch
分配 1 个 branch
分配 2 个 branch
当前代码先把所有可能的搭配全部列出来，所以组合数量约为：
3 × 3 × 3 × ... × 3 = 3^26 ≈ 2.54 万亿
考虑部分 anchor 容量不同，本次实际约为 1.69 万亿种。

当前问题是 Exact Batch-ERV 的分配算法发生组合爆炸。
step 3 中某个任务：
有 26 个可用于 branch 的 anchor；
实际只需要选择 1 个 branch；
理论上只需比较 26 种选择。
但当前代码先枚举每个 anchor 分配 0、1、2 个 branch 的全部组合，再筛选总数等于 1 的组合，因此需要遍历约 1.69 万亿种状态。
