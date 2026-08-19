# BACE-GiGPO BatchERV Exact 工程实现说明

日期：2026-08-14

对应方案：`BACE_GiGPO（修改版）_完整方案_最终版_BatchERV_Exact_2026-08-14.md`

对应代码仓库：`/home/naie/work/work-BACE/verl-agent-src`

## 1. 实现结论

修改版 BACE 已作为新的可选 variant 接入现有训练栈：

```text
algorithm.bace.variant=batch_erv_exact
algorithm.bace.acquisition=batch_erv_exact
```

原来的 fixed、random、pilot + sequential ERV、preallocated、staged、packed 和 frontier 路径均保留。默认值仍然是：

```text
algorithm.bace.variant=legacy
```

因此本次实现不会静默改变已有实验，也没有删除旧 controller。

新路径已经实现：

1. 取消独立 Pilot phase；
2. 使用滞后的 task-family natural-root history 规划当前 root/branch quota；
3. 一次生成所有 planned natural roots；
4. 使用 current-instance roots 构造 local Beta posterior；
5. 使用 Beta-Binomial 有限求和精确计算 Batch-ERV；
6. 使用 Batch-ERV marginal capacity 做单向 root-side correction；
7. 精确求解跨 anchor 的全局 branch allocation；
8. 对 local/global 数值平局执行可复现的 seeded uniform tie-breaking；
9. 一次冻结并发出全部 branch requests，branch outcome 不参与本 batch 的后续 acquisition；
10. 复用已有严格 Replay、copied branch origin、fresh suffix、all-leaf GiGPO credit 和 unified PPO 数据路径；
11. 保存 topology、capacity、所有 local plans、全局最优分配、Replay 和 history update 等诊断数据。
12. Exact 与 legacy 共用原有 action identity 规则；主版本为 `strict_identity`，valid 与可解析的 environment-invalid/no-op edge 都可参与 Batch-ERV。

## 2. 新旧逻辑的选择关系

旧逻辑：

```text
variant=legacy
acquisition=erv
```

旧逻辑仍读取：

```text
pilot_roots
erv_mc_samples
erv_temperature
erv_threshold
```

新逻辑：

```text
variant=batch_erv_exact
topology=dynamic
dynamic_root_generation=staged
staged_root_batching=packed
acquisition=batch_erv_exact
```

新逻辑不读取 Pilot outcome 来规划当前 topology，也不使用 Monte Carlo。因此以下旧参数在新路径中是 inert 的：

```text
pilot_roots
erv_mc_samples
erv_temperature
erv_threshold
```

为了避免组合配置产生语义不清的运行，新 variant 会校验必须同时使用 `dynamic + staged + packed + batch_erv_exact`。错误组合会在训练开始时直接报错。

## 3. 实际编排流程

### 3.1 reset-key probe

collector 先 reset 每个 task 的第一个环境槽，只读取：

```text
game file / session key
task family
```

该 probe 不生成 actor response、不形成 rollout、不读取 terminal outcome，也不进入训练。

这样可以在任何 current natural-root outcome 出现前读取 lagged family history。

### 3.2 Lagged family topology

对 family `c`，历史状态为：

```text
A = A0 + decayed_successes
B = B0 + decayed_failures
```

controller concentration 沿用已有 bounded transfer：

```text
kappa = clip(history_transfer_fraction * (A + B), min_strength, max_strength)
```

均值保持 `A / (A + B)`，由该 bounded Beta belief 计算：

```text
q_c = P(phi_c > competence_threshold)
```

然后：

```text
planned_Q = round((B - R_min) * q_c)
planned_R = B - planned_Q
```

此处不读取当前 task 的 root outcome。

### 3.3 Planned-root wave

所有 task 的 `planned_R` 个 roots 按 slot-major 顺序合并后，一次交给现有 batched root rollout。

这里的“一次”指一个完整 root-trajectory collection wave；一条多步轨迹内部仍然需要逐 decision step 调用模型和环境。

### 3.4 Exact capacity correction

每轮 root backbone 上执行：

```text
exact observation anchor grouping
-> valid observed canonical action support
-> current-instance competence posterior
-> anchor-action Beta posterior
-> exact V1 / V2
-> delta1 / delta2
-> information capacity
```

若：

```text
capacity < current_Q
```

则每个 deficient task 只执行一次：

```text
R += 1
Q -= 1
```

多个 deficient tasks 的追加 root 会合并成同一 correction wave。该过程只允许 branch slot 转为 root slot，直到 capacity 足够或 `Q=0`。

最终强制检查：

```text
R + Q == total_leaf_budget
```

### 3.5 Exact Batch-ERV

对一个 anchor 的 unordered action multiset plan，精确计算：

```text
BERV(plan) = E_future[max(updated posterior mean)]
             - max(current posterior mean)
```

对动作 `u` 分配 `n_u` 次实验时，成功次数使用 Beta-Binomial predictive probability：

```text
P(K_u=k) = C(n_u,k) * Beta(alpha_u+k, beta_u+n_u-k)
                         / Beta(alpha_u,beta_u)
```

实现使用 `lgamma` 在 log space 计算概率，然后枚举所有 action success-count combinations。主方法没有 Monte Carlo sampling。

当前 `L_max=2` 时，局部只需枚举：

```text
[u]
[u,u]
[u,v]
```

### 3.6 Local capacity 与全局分配

每个 anchor 保存：

```text
V(0), V(1), V(2)
delta(1) = V(1)
delta(2) = V(2) - V(1)
```

容量按层级阈值判断；阈值等号算有效。然后枚举所有：

```text
m_z in [0, capacity(z)]
sum(m_z) = Q
```

选择使 `sum_z V_z(m_z)` 最大的精确全局 allocation，不使用 greedy top-Q。

### 3.7 Tie-breaking

数值 tie 条件：

```text
abs(V1 - V2) <= abs_tol + rel_tol * max(abs(V1), abs(V2))
```

对 local optimal plan set 和 global optimal allocation set 分别均匀采样。seed 由以下内容稳定组合：

```text
global seed
policy update id
task id
anchor id / allocation level
```

实现使用 SHA-256 构造稳定整数 seed，不依赖 Python process-randomized `hash()`。

### 3.8 一次性 parallel branches

全局 allocation 完成后，所有 `(anchor, action)` 与 concrete natural origin 一次性冻结。coordinator 第一次调用返回所有 task 的全部 requests，第二次调用返回空列表。

因此执行关系是：

```text
joint exact planning
-> one replay/validation batch
-> one batched fresh-suffix collection round
```

Branch outcome 只用于 terminal leaf、最终 GiGPO credit、PPO 和离线诊断，不会更新当前 batch 的 acquisition plan。

## 4. Action 与 Replay 约束

Exact Batch-ERV 不额外强制 valid-only，而是直接服从原有配置：

```text
invalid_action_mode=strict_identity
```

主方案候选集合包含自然 roots 中真实出现且 action body 可稳定解析的两类 strict identities：

```text
valid::<environment action>
invalid::<raw action body>
```

这里必须区分：

```text
format-valid       = action body 可稳定解析
environment-valid  = action 属于当前 admissible set
replay-valid       = 同 prefix + 同 raw response 可复现相同 transition
```

主方案要求 format-valid 和 replay-valid，不要求所有候选都 environment-valid。invalid/no-op edge 可以进入 posterior、Exact Batch-ERV、origin selection 和 branch；它不是被修正成某个 admissible action，而是保留自己的 raw identity，研究“犯下同一个错误后 continuation 是否可恢复成功”。

`invalid_action_mode=valid_only_branch` 和 `single_invalid_bucket` 继续保留为显式消融。它们不会成为 Exact variant 的隐式默认值。

每个 branch origin 仍复用现有 ReplayRequest，包含：

```text
task/game/root/step identity
environment action prefix
pre-action observation/key/action set
original prompt and response tokens
old log probabilities
copied action identity
expected post-action transition
remaining horizon
```

Replay fallback 只能在同一 frozen anchor + action 的 natural origin pool 内切换。Exact plan 若在重试耗尽后仍有任一 branch 失败，会中止当前 update，避免在不满足固定 leaf budget 时静默训练。

## 5. History 更新语义

当前 update 结束时，所有 natural roots 都更新下一批 family history，包括 capacity correction roots。

```text
natural roots -> family history
branches      -> never enter family history
```

history update 在 artifact 中同时保存 before、natural-root success/failure/root IDs 和 after，便于检查 branch outcome 是否被误纳入。

## 6. Artifact 与指标

每个训练 step 的目录仍由现有 `BaceArtifactStore` 创建，默认位于：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/bace_artifacts/<run_name>/step_XXXXXXXX/
```

Exact 路径新增或扩充以下记录：

| 文件/stream | 主要内容 |
|---|---|
| `manifest.json` | variant、budget、threshold、tie tolerances、action identity mode 等完整配置 |
| `family_topology_plans.jsonl` | history snapshot、family prior、readiness、planned R/Q |
| `capacity_checks.jsonl` | 每轮 current posterior、anchor designs、V/delta/capacity、deficient tasks、correction count |
| `acquisition_rounds.jsonl` | 所有 local candidate plans、tie-optimal sets、global allocations、最终 plan |
| `replay_attempts.jsonl` | observation/action-set/transition 校验与 retry lineage |
| `branches.jsonl` | branch origin、action、terminal reward、suffix occurrence IDs |
| `posterior_snapshots.jsonl` | frozen acquisition posterior 与 realized selected-edge outcomes |
| `family_history_updates.jsonl` | history before/after、全部 natural-root evidence、branch excluded 标记 |
| `generation_waves.jsonl` | 原 frontier 路径的 wave 数据；Exact packed 路径的 aggregate wave 指标写入 summary |
| `training_occurrences.jsonl` | root/origin/suffix token、mask、advantage、old-log-prob 对账 |
| `summary.json` | topology、Replay、success、I/O 与 orchestration 汇总 |

新增 summary 指标包括：

```text
batch_erv_exact
planned_root_waves / trajectories / generation_seconds
capacity_correction_root_waves / trajectories / generation_seconds
capacity_planning_seconds
root_generation_waves
planned_branches_mean
final_roots_mean
final_branches_mean
capacity_corrections_mean
information_capacity_mean
family_prior_mean
branch_suffix_waves
branch_suffix_trajectories
branch_suffix_generation_seconds
```

## 7. 参数与当前建议

| 配置键 | 当前主脚本值 | 说明 |
|---|---:|---|
| `total_leaf_budget` | 8 | 每 task 最终 leaves |
| `min_natural_roots` | 2 | 最少 breadth budget |
| `competence_threshold` | 0.5 | refinement-ready threshold |
| `history_base_alpha/beta` | 1/1 | family base prior |
| `history_forgetting` | 0.9 | 位于方案建议 0.8--0.95 内 |
| `history_transfer_fraction` | 0.1 | bounded controller strength scale |
| `history_min/max_strength` | 2/8 | bounded concentration |
| `local_prior_strength` | 2 | anchor-action weak prior |
| `max_branches_per_anchor` | 2 | 精确枚举主配置 |
| `batch_erv_threshold` | 0.01 | 工程起始值，需要小规模 sweep |
| `batch_erv_tie_abs_tolerance` | 1e-12 | exact tie absolute tolerance |
| `batch_erv_tie_rel_tolerance` | 1e-10 | exact tie relative tolerance |
| `invalid_action_mode` | `strict_identity` | 主方案保留 valid 与可解析 invalid/no-op identities；其他模式只用于消融 |
| `local_credit_mode` | occurrence | 对齐 GiGPO 主版本 |

`batch_erv_threshold=0` 也是合法配置，但因为阈值等号算有效，精确为零的 marginal slot 也会进入 capacity。它适合作为 no-information-gate 消融，不建议直接当最终主参数。

建议先 sweep：

```text
0.005, 0.01, 0.015
```

再根据 retained branch ratio、correction count、success 和 wall-clock 决定最终值。

## 8. 代码位置

| 文件 | 作用 |
|---|---|
| `recipe/bace_gigpo/batch_erv.py` | Beta-Binomial exact BERV、local plans、capacity、global allocation、seeded tie |
| `recipe/bace_gigpo/topology.py` | 新的 no-pilot lagged-family planner 与 root-side correction；旧 planner 保留 |
| `recipe/bace_gigpo/coordinator.py` | 新的一次性 Exact coordinator；旧 sequential coordinator 保留 |
| `recipe/bace_gigpo/rollout_collector.py` | variant 路由、packed planned roots、correction waves、单轮 branches、artifact/metrics |
| `recipe/bace_gigpo/root_store.py` | 从 reset key 解析 task family 的公共入口 |
| `verl/trainer/config/ppo_trainer.yaml` | 新 variant 和参数默认值 |
| `tests/bace_gigpo/test_batch_erv_exact.py` | Exact 数值、tie、global allocation、topology、batch requests |
| `tests/bace_gigpo/test_root_wave_batching.py` | 验证无 pilot 且 planned roots 合并为一个 wave |

## 9. 运行脚本

完整 8 卡 150 epoch 脚本：

```text
examples/gigpo_trainer/run_bace_alfworld_npu_8card_150epoch_batch_erv_exact.sh
```

真实 8 卡 1 epoch smoke 脚本：

```text
examples/gigpo_trainer/run_bace_alfworld_npu_8card_1epoch_batch_erv_exact_smoke.sh
```

Smoke 明确使用 `invalid_action_mode=strict_identity`，因此 Exact Batch-ERV 会同时考虑 valid 与可解析的 environment-invalid/no-op natural edges。它关闭 validation 和 checkpoint 保存，但保留完整 rollout、Replay、Batch-ERV 与 training diagnostics artifact。

运行：

```bash
cd /home/naie/work/work-BACE/verl-agent-src
bash examples/gigpo_trainer/run_bace_alfworld_npu_8card_150epoch_batch_erv_exact.sh
```

Smoke 运行：

```bash
cd /home/naie/work/work-BACE/verl-agent-src
bash examples/gigpo_trainer/run_bace_alfworld_npu_8card_1epoch_batch_erv_exact_smoke.sh
```

该脚本复用此前 GiGPO-derived 8 卡运行时配置，并保持：

```text
8 NPU
TP=1
max_steps=40
ppo_mini_batch_size=128
actor PPO micro batch=4
rollout/ref log-prob micro batch=8
gpu_memory_utilization=0.6
max_num_batched_tokens=16384
max_num_seqs=128
test_freq=10
save_freq=30
```

所有 checkpoint、rollout、artifact 和 TensorBoard 输出继续写入 `AESC-exp`。

## 10. 验证结果与边界

已完成：

```text
Python compile: passed
bash -n: passed
git diff --check: passed
tests/bace_gigpo: 59 passed
```

测试覆盖：

1. one-sample exact ERV 闭式值；
2. two-sample Beta-Binomial finite sum；
3. local tie-optimal plan set；
4. global exact allocation 与 seeded reproducibility；
5. 无 pilot、只读取 lagged history 的 topology；
6. one-way capacity correction 与 `R+Q=B`；
7. 全部 branch requests 单轮发出；
8. planned roots 单 wave packed；
9. 所有原 BACE Replay、frontier、advantage、artifact 和 WebShop tests 回归通过。

尚未在本次代码实现过程中宣称完成：

1. 真实 8 卡 NPU 的 1-epoch smoke；
2. Exact 路径的真实 wall-clock、NPU 利用率和峰值显存测量；
3. `batch_erv_threshold` 的实验选择；
4. 中断恢复时 family controller history 的 checkpoint restore。当前 history 会完整写入每步 artifact，但和旧实现一样，训练进程重启后不会自动从 artifact 恢复 controller 内存状态。

正式 150 epoch 前应先运行上述 1 epoch smoke 脚本。

检查首个 step 是否满足：

```text
每 task: final_R + final_Q = 8
无 pilot_root_waves
planned_root_waves = 1
ERV rounds = 1（有 branches 时）
requested = validated
branch outcomes excluded from family history
artifact streams 完整
```
