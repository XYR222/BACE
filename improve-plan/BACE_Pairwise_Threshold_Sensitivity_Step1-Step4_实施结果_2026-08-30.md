# BACE Pairwise + Threshold Sensitivity：Step 1–4 实施过程与结果

日期：2026-08-30  
适用对象：BACE / Exact Batch-ERV 主线，Qwen2.5-1.5B ALFWorld 2 卡训练 trace  
基线：Full Exact Batch-ERV，`tau_BERV=0.005`，`L_max=2`，每 task leaf budget `B=8`

## 结论摘要

当前不应立刻用 Pairwise 替换正式 Full Batch-ERV 主方法。

- `tau=0.005` 是唯一满足“保留至少 97% BERV”的保守阈值，仍应作为首个线上 Pairwise 对照。
- `tau=0.0075` 可作为第二个、更激进的候选：离线预期保留约 `94.76%` 的 BERV，但 stopping/fallback 压力明显增加。
- `tau=0.01` 和 `0.015` 在首轮不应进入线上实验：弱边际 branch 被删除过多，预期 fallback roots 明显上升。
- 第一 pair 后重新分配的需求是真实存在的：历史 trace 中约 `45.58%` 的 task 会改变后续 anchor 分配；但在 `tau=0.005` 下，尚无足够证据说明这会改善最终 validation。
- P2 可能通过 lazy capacity 减少 root-side correction barrier，也会增加 branch-side round。是否更快或更好，必须用 H100 smoke 的真实 wall-clock 与 validation 判断。

## 范围

本阶段完成原方案的 Step 1 至 Step 4：

1. 对四个 threshold 作 CPU-only 精确离线敏感性分析；
2. 实现 K=2 Pairwise Exact 核心分配；
3. 实现 P1 `Pairwise-Fixed`；
4. 实现 P2 `Pairwise-Stopping / Lazy Capacity`。

未在本阶段自动提交新的正式训练链。P1/P2 的 2--5 step H100 smoke 是下一项线上验证工作。

## Step 1：离线 Threshold Sensitivity

### 方法

输入为已完成的 150-step BACE 2 卡训练 trace。分析脚本复用生产 `ExactBatchErvEngine` 和 A1 的 Beta-Binomial 后验预测枚举；对每个 `Q>=3` task 和每个候选阈值重新计算：

- 每个 anchor 的 Exact Batch-ERV design 与 threshold-positive capacity；
- `P(C>=2)` 与 `P(C>=Q)`；
- K=2 P2 selection、branch posterior update、`C=1` 单 branch 与 `C=0` one-way fallback；
- 第一 pair 后基于真实 outcome 的 replan 指标；
- 预期执行 branch 数、fallback root 数、pair round 数、BERV retention；
- 按训练阶段、任务 family、初始 Q 的拆分。

这不是“将历史 branch 事后按阈值过滤”。每个阈值均重新求解 Exact allocation，并对后续 outcome 作 posterior-predictive 枚举。

### 覆盖范围

- trace steps：150；
- `Q>=3` task：1,130；
- task-threshold states：4,520；
- 损坏或缺失输入记录：0。

### 总体结果

| `tau_BERV` | 初始 `P(C>=2)` | 初始 `P(C>=Q)` | post-pair shortfall 概率 | 预期执行 branch | 预期 fallback roots | 预期 pair rounds | P2 BERV retention |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0050 | 100.00% | 100.00% | 1.06% | 4.691 | 0.015 | 2.619 | 101.69% |
| 0.0075 | 95.31% | 79.47% | 19.23% | 4.297 | 0.408 | 2.419 | 94.76% |
| 0.0100 | 91.33% | 70.71% | 28.41% | 3.987 | 0.718 | 2.265 | 89.54% |
| 0.0150 | 76.28% | 51.42% | 45.76% | 3.165 | 1.540 | 1.822 | 74.30% |

`tau=0.005` 中 P2 retention 略高于 100% 不表示已证明训练更优：这是离线 posterior-predictive adaptive acquisition value 相对于历史 Full Batch 的 BERV objective；它不是 validation 的无偏估计。

### 阈值选择

机械筛选规则：从已评估阈值中选取最大的、BERV retention 分别不低于 97% 与 90% 的阈值。

- 保守候选：`tau_low*=0.005`；
- 进取候选：`tau_high*=0.0075`；
- 暂不推荐：`0.01`、`0.015`。

### Eager correction 的离线边界

历史 trace 记录的是当时用 `tau=0.005` 实际产生的 natural roots。提高阈值后，有些 task 在真实运行中可能需要生成 trace 里从未生成过的新 root；新 root 的 action、成败及其是否产生新 anchor 都不可从历史中得知。

因此输出中的：

```text
eager_correction_rounds_lower_bound = max(Q - C_tau, 0)
```

是已有 root support 下必须进行的最少 slot conversion 数，而不是虚构的“实际 root wave 次数”。未观测的新 root outcome 被视为 censored。P2 的在线运行会真实生成 packed fallback roots，并记录实际 root waves 与耗时。

### Step 1 产物

目录：`analysis_outputs/Pairwise_Threshold_Sensitivity/`

- `threshold_summary.csv`：全局汇总；
- `threshold_phase_summary.csv`：early/middle/late；
- `threshold_family_summary.csv`：任务 family；
- `threshold_q_summary.csv`：初始 branch quota；
- `threshold_task_audit.parquet`：4,520 条 task-level 审计记录；
- `figures/*.svg`：BERV retention、执行 branch、fallback、stopping pressure、weak mass；
- `online_candidates.json`：候选阈值；
- `report.md`：机器可复核的离线报告。

实现：`analysis/pairwise_threshold_sensitivity.py`。

## Step 2：K=2 Pairwise Exact 核心

在 `recipe/bace_gigpo/coordinator.py` 的 `ExactBatchErvCoordinator` 中加入三种显式模式：

```text
full      原始的一轮冻结 Full Batch（默认）
fixed     P1 Pairwise-Fixed
stopping  P2 Pairwise-Stopping
```

Pairwise 固定 `K=2`，每轮均：

1. 用当前 branch-updated posterior 重建 residual Exact designs；
2. 按已使用 branch 数扣除每个 anchor 的 residual `L_max=2`；
3. 用现有 quota-aware exact DP 在全局范围求解本轮 quota；
4. 通过 `policy_update_id + decision_task_key + round` 保持稳定、可复核的 tie identity；
5. 只在全部 active task 的当前 round request 都生成后交给执行层，因此不会退化为 task-by-task 串行 branch。

默认 `full` 仍保留历史一轮分配语义；为避免旧 checkpoint 被无意破坏，只有非 `full` 模式才写入 Pairwise 参数签名。

## Step 3：P1 Pairwise-Fixed

P1 是最小风险的 feedback acquisition 对照。

保留不变的语义：

- 原有 root-side eager capacity correction；
- root correction 完成后的最终 branch quota `Q`；
- 不发生 threshold stopping；
- 不生成 fallback roots；
- Exact 全局 allocation、packed roots、selected-worker/replay 与 PPO 数据语义。

变化仅为 branch phase：先执行每 task 最多两个 branch；将真实 branch outcome 更新到 acquisition posterior 后，再将仍有 quota 的 task 全局打包重规划。最后一轮可为单 branch。

## Step 4：P2 Pairwise-Stopping / Lazy Capacity

P2 不在 branch 前要求完整 `C>=Q`，而是只要求当前轮有 capacity：

```text
C >= 2：执行 pair
C = 1 ：执行 single branch
C = 0 ：停止 branch；全部剩余 quota 转为 fallback natural roots
```

关键语义：

- 初始仅生成 `R_init=B-Q_plan` 个 natural roots，不做 eager full-capacity correction；
- 每轮重新计算当前 threshold-positive residual capacity；
- fallback 不是 task 一停就立即生成，而是在所有 task branch phase 结束后全局 packed 成一个 root stage；
- fallback 是 one-way：该 task 不会在 fallback 后重新打开 branch；
- 最终逐 task 强制检查 `final_roots + executed_branches = B`；
- family history 仅接收 natural root evidence，branch outcome 不进入 family history；
- trace 中保存 P2 最终 topology 与 fallback 细节。

新增可比较的运行指标：

```text
time/initial_root
time/capacity_correction
time/branch_generation
time/fallback_root
time/acquisition_compute
time/ppo
time/step_total

count/root_generation_waves
count/capacity_correction_waves
count/pairwise_branch_rounds
count/fallback_root_waves

generated_tokens/root
generated_tokens/branch
generated_tokens/total
```

其中 PPO 与完整 step 时间在 trainer 层记录；rollout 内部的 root、branch、fallback 与 acquisition 时间在 collector 层记录。

## 验证

### 单元与回归

新增/覆盖的专项断言包括：

- P1 在高 threshold 下仍按 residual structural `L_max` 完成冻结 Q；
- P1 先执行 2 个 branch，随后正确重规划最后的 remainder；
- P2 的 `C=1` 只发出一个 branch；
- P2 的 `C=0` 将所有余量一次性转为 fallback，且不 reopen；
- fallback 后最终 roots/branches 满足 leaf budget；
- root-wave helper 对未调用完整构造函数的诊断/测试对象保持 `full` 兼容回退。

执行结果：

```text
pytest -q tests/bace_gigpo
148 passed, 0 failures, 0 errors, 0 skipped
```

同时完成 Python 编译和 `git diff --check`。

## 仍待完成的线上验证

以下不是代码正确性缺口，而是不能由历史 trace 代替的运行时实验：

1. P1 `tau=0.005` 运行 2--5 step H100 smoke；
2. P2 `tau=0.005` 运行 2--5 step H100 smoke，检查 fallback/replay/trace 和 budget；
3. 如前两者通过，再运行 P2 `tau=0.0075`；
4. 与同一 step-100 checkpoint、同一 seed、同一 GPU/scheduler 的 Full `tau=0.005` 对照 wall-clock、validation、branch/root 比例、credit 与 trace。

在上述 smoke 前，不应把 P1 或 P2 作为新的正式 150-step 主链，也不应同时引入 action-mean、动态 threshold 或 K sweep 等额外变量。
