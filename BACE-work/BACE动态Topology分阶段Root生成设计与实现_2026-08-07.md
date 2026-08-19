# BACE 动态 Topology 分阶段 Root 生成设计与实现

> 日期：2026-08-07  
> 代码基座：`langfengQ/verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`  
> 目标：消除 dynamic topology 的 candidate-root preallocation 计算浪费，同时完整保留原实现作为可选兼容路径

## 1. 问题定义

BACE 对每个 task 使用固定 terminal-leaf budget：

```text
R + Q = B
```

其中：

- `B`：task 的总 terminal-leaf budget；
- `R`：最终进入训练的 natural roots；
- `Q`：最终执行并进入训练的 branch leaves。

动态 topology 必须先观察 `pilot_roots` 条自然轨迹，用 competence posterior 的 readiness 规划 branch quota，再根据 frozen natural anchor support 做 root-side capacity correction。因此最终 `R` 在 pilot 完成前不能确定。

## 2. 原 preallocated 实现

原实现将 upstream 的一次性 group rollout 直接配置成：

```text
env.rollout.n = total_leaf_budget = B
```

执行顺序是：

```text
一次性生成 B 条 candidate natural roots
-> 前 pilot_roots 条计算 readiness
-> 计算计划 branch quota
-> 从 candidate roots 中 reveal R 条
-> 丢弃其余 B-R 条 roots
-> 执行 Q 条 branches
```

它在算法数据层是正确的：

- 只有 reveal 的 roots 进入 anchor、posterior、advantage、history 和 PPO；
- 未 reveal roots 不泄漏 outcome 或状态；
- 最终训练数据满足 `R + Q = B`；
- rollout batch 内策略保持冻结。

但计算层会超支。若 `B=8, R=2, Q=6`，模型仍先完整生成 8 条 natural roots，随后丢弃 6 条，再额外生成 6 条 branches。最终使用 8 条 leaf，却支付了最多 14 条 leaf 的环境与模型 rollout 成本。

## 3. 新 staged 实现目标

新增可选配置：

```yaml
algorithm.bace.dynamic_root_generation: preallocated  # preallocated | staged
```

兼容原则：

1. `preallocated` 完整保留现有代码路径和结果语义；
2. `staged` 是新增路径，不删除、不暗改原逻辑；
3. 两种模式使用同一 competence posterior、capacity rule、coordinator、replay validator、advantage 和 PPO loss；
4. `staged` 只改变 natural roots 在何时生成，不改变哪些 roots 最终被使用；
5. validation/test rollout 不使用 staged branching。

## 4. Staged 算法

### 4.1 Pilot 阶段

对每个 task 每次只生成一条 root，连续生成 `pilot_roots` 个 wave：

```text
wave 0: 每个 task 生成 root 0，并记录 concrete reset key
wave 1: 每个 task 在同一 reset key 上生成 root 1
...
wave P-1: 每个 task 生成最后一条 pilot root
```

ALFWorld reset key 是 game file；WebShop reset key 是 session ID。后续 wave 必须绑定 wave 0 的 exact reset key，保证同 task 的所有 roots 来自同一环境实例定义。

### 4.2 初始配额

只使用 pilot outcomes 计算：

```text
posterior = lagged family prior + pilot evidence
readiness = P(phi > competence_threshold)
planned_Q = round((B - P) * readiness)
target_R = B - planned_Q
```

若 `target_R > P`，只为需要的 task 继续生成 `target_R-P` 条 natural roots。

### 4.3 Root-side capacity correction

在当前已生成 roots 上构造 frozen exact structural anchors，计算：

```text
capacity = L_max * effective_anchor_count
```

若 `capacity < Q`：

```text
R = R + 1
Q = Q - 1
```

然后只为发生 correction 的 task 再生成一条 root，重新构建 anchor support 并检查 capacity。循环直到：

```text
Q = 0 或 capacity >= Q
```

最终断言：

```text
R + Q = B
generated_natural_roots = R
discarded_candidate_roots = 0
```

## 5. Worker 子集执行

自然环境池仍按 `train_batch_size * B` 建立，以保持现有资源拓扑和 preallocated 模式兼容。staged 模式不会驱动全部 workers，而是按以下索引选择本 wave 的 worker：

```text
worker_index = task_batch_index * B + root_slot
```

环境增加 subset reset/step 能力：

- `reset_subset(worker_indices, reset_keys)` 只重置本 wave workers；
- 后续 `step(actions)` 只驱动当前 active worker indices；
- admissible actions、observations、infos 的顺序严格与当前 wave task 顺序一致；
- 未选择 workers 不 reset、不 step，因此不产生模型或环境 rollout 计算。

## 6. 跨 Wave 身份与数据合并

每个 task 在整个 staged round 内获得稳定 `task_id`。每条 root 仍获得独立 `traj_uid`。collector 显式写入原始 `task_batch_index`，避免使用 upstream 的 `i // env.rollout.n` 推断，因为 staged wave 的 batch size 和 task 集合会变化。

所有 wave 输出合并后再构造 immutable root event logs。以下属性必须保持：

- 同一 task 的 roots 共享 task ID 和 exact reset key；
- 每条 root/occurrence ID 唯一；
- mechanical reset/replay 不成为训练 occurrence；
- pilot 和补充 roots 都由同一个冻结 policy 采样；
- competence history 只接收最终 selected natural roots；
- branch support 在 branch phase 开始前冻结。

## 7. 配置与回退

```yaml
algorithm:
  bace:
    topology: dynamic
    dynamic_root_generation: staged
```

行为矩阵：

| topology | root generation | 行为 |
|---|---|---|
| `fixed` | 任意值 | 沿用固定 root 一次性 rollout |
| `dynamic` | `preallocated` | 保留原 candidate-root preallocation |
| `dynamic` | `staged` | pilot + 按需 root + capacity correction |

未知配置值必须启动即报错，不能静默回退。现有 smoke 脚本默认继续使用 `preallocated`；新增 staged smoke 通过显式 override 启用。

## 8. 指标

两种 dynamic 模式都记录：

```text
bace/root_generation_mode
bace/generated_roots_mean
bace/discarded_roots_mean
bace/planned_branches_mean
bace/final_roots_mean
bace/final_branches_mean
```

预期：

```text
preallocated: generated_roots_mean = B, discarded_roots_mean = B-R
staged:       generated_roots_mean = R, discarded_roots_mean = 0
```

由于 trainer metrics 需要数值，`root_generation_mode` 可编码为 `0=preallocated, 1=staged`，日志和文档保留文字解释。

## 9. 验证计划

单元测试必须覆盖：

1. staged pilot 只请求 `P` 条 roots；
2. high readiness 且 capacity 足够时不生成未使用 roots；
3. capacity 不足时逐条增加 root、同步减少 branch；
4. low readiness 时最终生成 `B` 条 roots 和零 branch；
5. 多 task 可在不同 wave 停止生成，task/worker mapping 不串扰；
6. preallocated planner 的现有测试继续通过；
7. 两种模式均满足 `R + Q = B`；
8. staged 的 generated/discarded metrics 与实际一致。

工程检查包括完整 `tests/bace_gigpo`、`compileall`、shell syntax 和 `git diff --check`。真实 NPU smoke 需要确认 staged 日志中：

```text
generated_roots_mean == final_roots_mean
discarded_roots_mean == 0
bace/validated > 0
bace/validated <= bace/requested
training/global_step == 1
```

严格 replay validator 可以拒绝 anchor、action-set 或 prompt identity 不一致的 request，因此 `validated == requested` 是理想观测而不是 staged root 生成正确性的必要条件；拒绝数量必须显式记录，不能把失败 request 放入训练。

## 10. 风险与不变量

- 分 wave 调用 vLLM 不得在 wave 间更新 actor；本实现所有 waves 位于同一个 collector 调用和同一次 PPO update 之前。
- 后续 natural roots 必须绑定 pilot 的 exact reset key，不能重新抽 task。
- task ID 必须跨 wave 稳定，否则 structural anchor 无法跨 roots 聚合。
- capacity correction 必须只观察已经生成的 natural roots，不能使用 branch outcome。
- 如果 replay validation 失败，不能临时把未生成 root 补成 branch 替代品；该行为属于另一层预算恢复策略。
- staged 模式节省的是被丢弃 natural roots 的模型与环境 rollout；环境 actor 进程仍预先创建，因此不减少 worker 常驻内存和 actor 数量。

## 11. 实际实现

实现新增但未替换原路径：

```text
recipe/bace_gigpo/topology.py
recipe/bace_gigpo/rollout_collector.py
agent_system/multi_turn_rollout/rollout_loop.py
agent_system/environments/env_package/alfworld/envs.py
agent_system/environments/env_package/webshop/envs.py
agent_system/environments/env_manager.py
```

具体实现包括：

1. `StagedTaskState` 保存每个 task 的 posterior、readiness、planned Q、当前 R/Q 和 effective anchor count；
2. `initialize_staged()` 只读取 pilot roots，计算初始 `R=B-Q`；
3. `correct_staged_capacity()` 每轮每个 deficient task 最多执行一次 `R+1, Q-1`；
4. `finalize_staged()` 要求实际生成 roots 恰好等于最终 R，并重新断言容量和 `R+Q=B`；
5. collector 为每个 task 建立稳定 task ID，并用 `task_index * B + root_slot` 选择自然 worker；
6. wave 0 记录 exact game file/session，后续 wave 固定绑定该 reset key；
7. 通用 rollout loop 可选接收显式 task IDs 和 task batch indices，默认调用行为不变；
8. ALFWorld 和 WebShop vector env 支持任意有序 worker subset 的 reset/step；
9. 多 wave 合并后显式恢复 episode numeric metadata dtype，避免 object array 进入 trainer metrics；
10. 新增 `run_alfworld_npu_1card_staged_smoke.sh`，原 Stage 5 smoke 默认仍走 `preallocated`。

配置值非法时 collector 直接报错；fixed topology 不进入 staged 调度。

## 12. 测试与真实 NPU 结果

### 12.1 单元与静态检查

最终结果：

```text
python -m pytest -q tests/bace_gigpo
28 passed

python -m compileall
passed

bash -n examples/bace_gigpo/*.sh
passed

git diff --check
passed
```

新增回归覆盖 staged high/low competence、逐条 capacity correction、worker subset 顺序与隔离，以及 multi-wave numeric metadata dtype。

### 12.2 首次真实运行发现的问题

首次 staged NPU 运行已经完成 rollout、advantage、old/ref log-prob、actor update 和 generation dump，但最终 `compute_data_metrics` 失败：

```text
AttributeError: 'float' object has no attribute 'item'
```

原因是多个 wave 的 `DataProto.concat` 将 `episode_rewards` 等字段保留为 object dtype。修复后统一将：

```text
episode_rewards
episode_lengths
tool_callings
*success_rate*
```

恢复为 `float32`，并增加对应回归测试。

### 12.3 成功的真实单卡 NPU smoke

日志：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/logs/bace_staged_dynamic_npu_1card_smoke_fix1.log
```

rollout：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/bace_staged_dynamic_npu_1card_smoke_fix1/1.jsonl
```

配置：

```text
tasks: 2
B: 6
pilot roots: 2
competence threshold: 0
root generation: staged
device: Ascend 910B, 1 card
```

关键结果：

```text
bace/root_generation_mode: 1
bace/generated_roots_mean: 2
bace/discarded_roots_mean: 0
bace/planned_branches_mean: 4
bace/final_roots_mean: 2
bace/final_branches_mean: 4
bace/effective_anchors_mean: 2
bace/competence_readiness_mean: 1
bace/erv_rounds: 4
bace/requested: 8
bace/validated: 5
actor/grad_norm: 3.883
training/global_step: 1
```

这次运行直接证明：每个 task 最终需要 2 条 roots 时，staged 路径只生成 2 条，而不是像 preallocated 路径生成 `B=6` 条后丢弃 4 条；unused natural-root rollout 计算已消除。5 条 branch 通过 replay validation 并进入训练，另外 3 条 request 因严格 replay/prompt identity validation 被拒绝，不影响 root 生成节省结论，但说明 branch validation failure 的预算回补仍是独立的后续问题。

## 13. 最终状态

candidate-root preallocation 不再是强制限制：

```yaml
# 完整保留旧实现
algorithm.bace.dynamic_root_generation: preallocated

# 新的按需实现
algorithm.bace.dynamic_root_generation: staged
```

`staged` 已通过真实单卡 NPU PPO update，满足：

```text
generated natural roots = final R
discarded natural roots = 0
R + Q = B
```

仍然保留全部 `B` 个环境 workers 是兼容 upstream/preallocated 模式的资源设计；它只增加常驻 CPU actor 数，不再产生未使用 root 的模型推理和环境 step 计算。
