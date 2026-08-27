# 当前 BACE 方法规范

> 适用范围：当前 GitHub 主线的 ALFWorld + BACE-GiGPO + Exact Batch-ERV。本文描述“方法实际是什么”，不描述某台集群如何提交作业。若本文与旧设计稿冲突，以当前代码、测试和 resolved command 为准。

## 1. 一句话定义

BACE 在每个任务的固定 terminal-leaf 预算内，先根据滞后的 task-family competence history 决定 natural roots 与 branches 的数量，再从已经完成且冻结的 natural roots 中构造可比较 anchor，用 Exact Batch-ERV 选择最有信息价值的 branch，严格 Replay 到 branch origin，最后把 natural occurrences、复制的 branch origin 和新生成的 branch suffix 合并，用 GiGPO-compatible advantage 做 PPO 更新。

当前推荐主线为：

```text
BACE-GiGPO
  + dynamic topology
  + staged/packed natural roots
  + Exact Batch-ERV
  + selected-worker branch executor
  + strict action identity
  + occurrence local credit
```

## 2. 基本对象

### Task、root 与 occurrence

- 一个 parquet row 对应一个 task slot。
- 一个 natural root 是模型在该 task 上从 reset 开始生成的一条完整环境轨迹。
- occurrence 是一次真正提交给环境的 macro action，以及对应的 prompt、response token、mask、old log-prob、前后 observation、reward 和 validity。
- `task_id` 用于本次训练 batch 的分组；`task_family` 从 ALFWorld reset key 提取，用于跨 step 的 competence history。

### Anchor

anchor 是同一 task 下、多个 natural occurrences 共享的 exact pre-action observation。当前结构条件为：

1. 不使用 step 0；
2. 至少有两个 occurrence；
3. 至少观察到两个可统计 action identity；
4. anchor 只由本 step 已冻结的 natural roots 构造。

代码入口是 `recipe/bace_gigpo/anchor_index.py::AnchorIndex`。

### Branch

branch 不是重新采样一个独立 task。它选择某个自然轨迹 occurrence 作为 origin：

```text
reset 同一 ALFWorld game
  -> 机械 replay origin 之前的 prefix
  -> 校验 pre-action observation/action set
  -> 复制并执行 origin action
  -> 校验 reward/done/post-action observation
  -> 从 origin 之后生成新 suffix
```

机械 replay prefix 只负责恢复环境，不进入训练 batch。复制的 origin occurrence和新 suffix 才是 branch 的可训练部分。

## 3. 每个 task 的预算与 topology

当前正式口径：

```text
B = total_leaf_budget = 8
R_min = min_natural_roots = 2
L_max = max_branches_per_anchor = 2
R + Q = B
```

其中 `R` 是最终 natural root 数，`Q` 是 branch quota。Exact 主线没有读取当前 root outcome 后再决定初始 quota 的 pilot wave；它只使用进入本 step 之前已经存在的 lagged family history。

对 task family `f`，历史计数为衰减 success/failure：

```text
s_f <- forgetting * s_f + current natural successes
f_f <- forgetting * f_f + current natural failures
```

当前冷启动 base prior 为：

```text
mean = 0.10
strength = 2.0
Beta(alpha=0.2, beta=1.8)
forgetting = 0.8
```

由历史获得 family prior 后，计算：

```text
readiness = P(competence > 0.5)
Q_planned = round((B - R_min) * readiness)
R_planned = B - Q_planned
```

这里的 round 在代码中是 `floor(x + 0.5)`。

### Capacity correction

初始 roots 生成后，planner 从真实 anchor 和局部 Batch-ERV marginal value 计算 information capacity。如果 capacity 小于 `Q`：

```text
R <- R + 1
Q <- Q - 1
```

然后只为需要修正的 task 多生成一个 natural root，再重新评估，直到 capacity 足够或 `Q=0`。因此最终仍严格满足 `R+Q=B`，不会为了凑 branch 数使用不存在的信息结构。

相关代码：

- `topology.py::ExactBatchTopologyPlanner`
- `rollout_collector.py::_collect_exact_batch_dynamic_roots_packed`

## 4. 局部 posterior 与 Exact Batch-ERV

对一个 anchor 的每个 action，先用该 task 当前 natural roots 形成的 competence posterior mean 作为局部 prior mean，强度为 `local_prior_strength=2`，再加入该 anchor/action 在 natural roots 中观察到的成功/失败结果。

对一个包含多个 action posterior 的 anchor，大小为 `l` 的 batch plan 是允许重复 action 的多重组合。Exact Batch-ERV 枚举该 plan 的 Beta-Binomial outcome，并计算：

```text
ERV(plan) = E[更新后最大 action posterior mean]
            - 当前最大 action posterior mean
```

对 `l=0..L_max`：

- 保存所有 plan；
- 保存 ERV 最大的全部 tie-optimal local plans；
- 计算从 `l-1` 到 `l` 的 marginal value；
- marginal 不低于 threshold 时，该大小计入 anchor capacity。

当前参数：

```text
L_max = 2
batch_erv_threshold = 0.005
tie_abs_tolerance = 1e-12
tie_rel_tolerance = 1e-10
```

实现位于 `batch_erv.py::ExactBatchErvEngine.design_anchor`。

## 5. 全局 Exact 分配

给定 task 的所有 anchor design 和冻结 branch quota `Q`，目标是：

```text
max  sum_a value_a(l_a)
s.t. sum_a l_a = Q
     0 <= l_a <= capacity_a
```

当前实现使用 quota-aware exact dynamic programming，复杂度为：

```text
O(number_of_anchors * Q * L_max)
```

它不会构造所有 `0/1/2` anchor allocation 的 Cartesian product。DP 保存每个 quota state 的最优值、最优路径总数和 tied predecessors；回溯时按完整路径数加权，因此在所有 tie-optimal 完整 allocation 上均匀采样，而不需要物化巨大 tie set。

`_global_allocations_cartesian_reference` 只用于小规模测试 oracle，不应进入正式训练。

## 6. Tie identity 与随机性

Exact tie 选择由 global seed、policy update id、task decision key、anchor 和选择阶段共同派生稳定 seed。

支持两种 identity：

- `legacy_uuid`：用本次运行的 UUID lineage，兼容旧 checkpoint；
- `stable_v1`：用 task batch index、environment reset key 和 root 内容构造稳定身份，UUID 改变时仍保持相同语义选择。

Hydra 默认保留 `legacy_uuid`，不是因为它更适合新实验，而是为了避免旧 checkpoint 签名突然变化。新的独立 run 推荐显式设置：

```text
algorithm.bace.tie_break_identity_mode=stable_v1
```

不能在同一个 dynamic BACE checkpoint 中途切换 identity mode。

## 7. Replay 与 branch 执行

Coordinator 一次冻结该 task 的全部 Exact requests。执行层可以把 requests 按 Replay pool capacity 分成多个 physical wave，但必须保持：

- 不重新规划；
- 不改变 request/branch 顺序；
- 所有原始 origin 在第一波前统一 reservation；
- retry 不能偷用后续 wave 的 origin；
- Exact 模式每一个冻结 request 都必须被实现，否则 step 失败。

如果某个选定 origin 的 Replay 校验失败，允许在同一冻结 anchor/action 内选择另一个未被 reservation 的 natural origin 重试；该替换会保留 branch identity 和 acquisition allocation，并显式记录 parent request、attempt 和 fallback origin。它不是重新计算 Batch-ERV，也不能换 action 或减少 quota。超过 `max_origin_retries` 后仍失败，Exact step 直接失败。

当前推荐 `selected_worker`：只恢复和推进本 chunk 实际使用的 Replay workers；terminal 或 horizon 耗尽的 suffix slot 会从后续生成 wave 中压缩掉。`legacy_dense` 仍保留作兼容对照，但不是当前推荐执行器。

当前恢复后端是 ALFWorld prefix/fast replay。根目录 StateID 文档描述的 snapshot backend 不在当前代码中。

## 8. Action identity

当前主线使用 `invalid_action_mode=strict_identity`。代码分别记录：

- model response 是否可解析；
- action 是否属于环境 admissible set；
- canonical/action identity；
- Replay transition 是否与 natural origin 一致。

不要把“格式无效”“环境不接受”“Replay 到错状态”合并成同一个 invalid 标志。invalid-action penalty 进入 token reward；Replay identity mismatch 则是执行正确性错误，不是普通负 reward。

## 9. 训练 batch 与 advantage

最终训练 batch 包含三种 source：

- `root`：natural root occurrences；
- `branch_origin`：复制 natural origin 的 response token、loss mask 和 rollout old log-prob；
- `branch_suffix`：origin 之后新生成的 occurrences。

BACE advantage 复用 GiGPO 的 task/trajectory macro credit 和 observation step group：

```text
A_occurrence = A_macro + step_advantage_w * A_local
```

当前：

```text
step_advantage_w = 1.0
mode = mean_std_norm
local_credit_mode = occurrence
gamma = 0.95
```

`action_mean` local credit 仍保留，但不是当前主线。branch outcome 会进入该 branch 的 reward、advantage 和局部诊断；它不会写入下一 step 的 family competence history。family history 只使用 natural root outcome，避免主动采样的 branch 分布污染能力估计。

## 10. PPO 更新与 checkpoint

collector 返回合并 batch 后，trainer 依次执行：

1. step-discounted return；
2. rule reward 和 invalid-action penalty；
3. recomputed actor old log-prob；
4. reference log-prob / KL；
5. BACE-GiGPO advantage；
6. 写入 advantage/log-prob 诊断并 finalize 本 step artifact；
7. actor PPO update；
8. validation 和按频率保存 checkpoint。

因此 `summary.json=status=complete` 表示 collector trace 已走到 diagnostics finalize，并不单独证明随后 actor update、validation 或 checkpoint 全部成功；正式验收还必须结合训练日志、global step 和 checkpoint/validation 证据。

dynamic BACE checkpoint 必须同时包含同一 global step 的：

- actor、optimizer、scheduler 和框架状态；
- `data.pt` dataloader state；
- RNG 状态（由框架 checkpoint 管理）；
- `bace_collector_state.json` family history 与参数签名；
- `latest_checkpointed_iteration.txt`。

tracker 只有在上述状态落盘后才原子更新。非零 step 恢复缺少或不匹配 BACE state 时必须 fail-fast，不能静默重置 history。

## 11. 方法不声称什么

- 150-step 旧结果不等于当前 GitHub HEAD 已做同版本长跑。
- `summary.json=status=complete` 不等于全量 trace 语义和 JSON 完整性已通过。
- branch Replay 不等于可移植 StateID snapshot。
- `frontier` scheduler 不是 Exact Batch-ERV 主线的一部分。
- YAML 里的兼容默认值不是正式实验参数；每次实验以 resolved command 为准。
