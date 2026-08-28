# pick-and-place 初始 2-root 但容量修正严重的原因调查

日期：2026-08-27  
实验：`bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0`  
范围：正式 4×H100，训练 step 1–150  
对象：初始 `R_plan=2` 的 task，重点分析 `pick-and-place`

## 1. 要解释的现象

`pick-and-place` 的初始 2-root task 容量修正率很高：

- 254 个 task 初始选择 `R_plan=2`；
- 181 个发生容量修正，task 修正率 **71.26%**；
- 初始 branch slot 为 1,524 个；
- 825 个 branch slot 被转换为补充 root，slot 修正率 **54.13%**；
- 修正 task 平均从 2 条 root 增加到 6.56 条 root；
- 其中 97 个 task 最终补到 8 条 root。

表面上看，初始选择 2 条 root 意味着系统判断该 family 很强、可以把预算留给 6 条 branch；但随后却发现 branch capacity 不够。这不是矛盾的日志错误，而是初始 quota 和后续 capacity 衡量了不同的东西。

## 2. 最关键的区别：family readiness 不是当前 task capacity

### 2.1 初始 2-root 的真正判定依据

Exact planner 在当前 task 的 root 尚未生成时，读取的是上一 step 的 family-level history：

```text
family posterior
    -> readiness = P(competence > 0.5)
    -> Q_plan = round((8 - 2) * readiness)
    -> R_plan = 8 - Q_plan
```

代码位置：`recipe/bace_gigpo/topology.py:305-324`。

因此，初始 `R_plan=2` 只说明：

> 从过去所有同 family 的 natural-root 成败来看，系统认为这个 family 大概率具备完成能力。

它不表示：

- 当前这个具体 task 已经有两个可用于 branch 的 anchor；
- 当前两条 root 会走出不同前缀；
- 当前 anchor 上存在足够多的不同 action；
- 当前 branch 的边际 ERV 会超过阈值。

当前 task 通常是新 UUID，planner 没有该 task 的专属历史；它只知道 family，不知道这一具体场景的物体、容器、房间布局和模型即将采取的路径。

本 run 的 `family_history_updates.jsonl` 还明确记录了 `branch_outcomes_included=false`。因此这里的 family readiness 主要由历史 natural-root 成败驱动，不是把当前 branch 的结果偷偷混入后得到的高 readiness。

### 2.2 容量修正真正检查的内容

root 生成后，planner 才构建当前 task 的 `AnchorIndex`，计算每个 anchor 的 Exact Batch-ERV design：

```text
A = structural anchor 数
C = 所有 anchor capacity 之和
Q_plan = 初始 branch quota
```

需要满足：

```text
C >= Q
```

而每个 anchor 最多提供 2 个 branch slot，故结构理论上限为 `2A`。代码位置：

- `recipe/bace_gigpo/topology.py:326-377`；
- `recipe/bace_gigpo/batch_erv.py:160-201`。

所以 planner 先根据跨 task 的 family 成功历史提出 branch 需求，再根据当前 task 的局部信息价值确认实际可执行 branch 数。

## 3. 直接证据：修正组的 readiness 更高，但 capacity 更低

在 254 个初始 2-root 的 `pick-and-place` task 中，对比发生修正与没有修正的两组：

| 指标 | 发生修正（181） | 未修正（73） |
|---|---:|---:|
| family readiness 均值 | **0.9858** | 0.9671 |
| family competence mean 均值 | **0.9074** | 0.8474 |
| family posterior strength 均值 | **7.45** | 6.45 |
| structural anchor A 均值 | **2.08** | **7.05** |
| information capacity C 均值 | **1.53** | **10.29** |
| 平均补充 root | **4.56** | 0 |

这组数据直接否定了“修正严重是因为 family 成功率低”的解释：

- 修正组的 readiness 反而更高；
- 真正拉开差距的是当前 root 产生的 anchor 数和局部 capacity；
- 修正组的平均 capacity 只有 1.53，而初始目标是 `Q=6`。

另一个明显现象是：修正组中 70/181（38.7%）的 task 首次 `C=0`；而未修正组的首次 capacity 最低为 6，全部足以支撑 `Q=6`。

## 4. 为什么成功率高反而可能降低 branch ERV

### 4.1 Branch 主要测量“信息不确定性”，不是“能否成功”

Batch-ERV 的作用是判断继续采样某个 action 是否能显著减少 action credit 不确定性。

当模型已经很有能力并且策略变得很确定时，可能发生：

```text
任务成功率高
    -> action posterior 接近确定
    -> 继续采样的预期信息增益变小
    -> marginal ERV < 0.005
    -> anchor capacity 变小或为 0
```

因此：

```text
高 competence ≠ 高 branch information value
```

### 4.2 训练后期策略更集中，root 多样性下降

训练过程中的全局 actor entropy 从 step 1 的约 1.096 降到 step 150 的约 0.577。这个趋势说明策略整体更集中。它不单独证明每个 task 都发生了模式坍缩，但与修正组的 anchor 统计一致：

- 修正组平均只有 2.08 个 structural anchor；
- 未修正组平均有 7.05 个；
- 修正组每个 anchor 的平均 observed-action 数约 3.51，未修正组约 4.47。

也就是说，模型变得更会完成任务的同时，也更倾向于在相同场景中重复相似的动作前缀，减少了可用于 branch 的局部对比结构。

### 4.3 `pick-and-place` 的 family pooling 更容易产生错配

`pick-and-place` family 内包含不同物体、容器和布局。family history 把这些 task 的成功结果合并成一个 posterior，但 local anchor 取决于具体场景：

- 某个场景可能有很多可比较的动作前缀；
- 另一个场景可能两条 root 走完全不同的前缀；
- 还可能两条 root 走相同前缀，但后续 action posterior 已经过于确定。

这些 task 的 family success rate 可以相近，但 local Batch-ERV capacity 完全不同。

## 5. 具体 task 证据

### 5.1 step 136：family readiness 几乎饱和，但局部 capacity 为零

step 136 的 `pick-and-place` family history before update 为：

```text
alpha = 99.3339
beta  = 1.7163
family mean = 0.9830
readiness = 0.9995
```

因此该 step 的许多 pick-and-place task 初始规划为：

```text
R_plan=2, Q_plan=6
```

具体 task `661d99d5`：

```text
首次检查：A=6
anchor capacities=[0,0,0,0,0,0]
C=0
```

这里不是没有 structural anchor，而是 6 个 anchor 的所有边际 ERV 都低于阈值。该 task 最终补到：

```text
R=8, Q=0
```

具体 task `6cb529ad` 的首次状态则是：

```text
A=1
capacity=[2]
C=2
```

即使唯一 anchor 达到单 anchor 最大 capacity 2，也无法支撑 `Q_plan=6`，最终同样需要多轮补 root。

### 5.2 step 81：两种原因同时出现

task `cebb9be2`：

```text
A=1, capacity=[2], C=2
```

这是结构 anchor 上限不足；最终从 2 条 root 补到 8 条 root。

同一个 step 的 task `516881d6`：

```text
A=0, C=0
```

两条初始 root 没有形成任何可比较的双 action 前缀。追加 root 后虽然一度出现 anchor，但 capacity 仍不足，最终补到 6 条 root，保留 2 条 branch。

### 5.3 step 96：anchor 数原则上够，但第二个 branch 的 ERV 不够

task `2a74f9a6`：

```text
R_plan=2, Q_plan=6
A=3
anchor capacities=[1,2,2]
C=5
```

结构上限 `2A=6` 足够，但实际 `C=5`，说明至少一个第二 branch 的边际 ERV 低于 `0.005`。这个 task 最终补到 7 条 root，保留 1 条 branch。

## 6. 修正原因在初始 2-root pick-and-place 组中的比例

对 181 个发生修正的 `pick-and-place` task，首次检查分类为：

| 原因 | 判定 | task 数 | 占修正 task | 补充 root slot |
|---|---|---:|---:|---:|
| 无 structural anchor | `A=0` | 38 | 21.0% | 318 |
| 结构 anchor 上限不足 | `A>0` 且 `2A<6` | 81 | 44.8% | 380 |
| ERV 边际容量不足 | `2A>=6` 且 `C<6` | 62 | 34.3% | 241 |

上表的 task 分类是可靠的；slot 级别建议以逐 task 原始记录为准，因为一个 task 可能经历多轮 capacity 变化，并且最终 correction count 不等于首次原因的单一贡献。整体上可以确认：

- 约 21% 是完全没有可比较 anchor；
- 约 45% 是 structural anchor 理论上限不够；
- 约 34% 是 anchor 数量够，但 ERV 边际不足。

这再次说明主要矛盾是“2 条 root 产生的局部结构/信息不足”，而不是 family readiness 低。

> 注：这里把 `2A<6` 作为结构上限主因。它是必要上限判定，不表示每个 anchor 都一定能贡献 2 个 branch；低 ERV 可能同时存在。

## 7. 为什么后期尤其严重

在 pick-and-place 初始 2-root task 中，按训练阶段统计：

| 训练段 | task 数 | 修正 task | 修正率 | 平均 readiness | 平均 A | 平均 C |
|---|---:|---:|---:|---:|---:|---:|
| 61–70 | 3 | 3 | 100.0% | 0.919 | 3.67 | 3.67 |
| 71–80 | 35 | 15 | 42.9% | 0.954 | 5.06 | 7.23 |
| 81–90 | 27 | 9 | 33.3% | 0.953 | 5.07 | 8.11 |
| 91–100 | 21 | 12 | 57.1% | 0.944 | 4.05 | 5.43 |
| 101–110 | 33 | 23 | 69.7% | 0.983 | 3.12 | 4.73 |
| 111–120 | 44 | 39 | 88.6% | 0.999 | 3.11 | 2.11 |
| 121–130 | 33 | 29 | 87.9% | 0.999 | 2.79 | 1.97 |
| 131–140 | 31 | 28 | 90.3% | 0.999 | 2.55 | 1.90 |
| 141–150 | 27 | 23 | 85.2% | 0.999 | 2.63 | 2.15 |

后期出现了明显的错配：

```text
readiness 接近 1 -> Q_plan 接近 6
但 A 和 C 下降 -> 2 条 root 无法支持 Q=6
```

因此后期修正率不是因为模型突然不会做 pick-and-place，而是因为 family-level readiness 饱和得比 local branch capacity 更快。

## 8. 这是不是算法 bug

需要分成两层判断。

### 8.1 不是正确性 bug

当前实现正确地执行了安全修正：

- 不执行超过真实 capacity 的 branch；
- 每次把一个 branch slot 转成 root slot；
- 最终保持 `R+Q=8`；
- 所有合法 branch 才进入 Replay 和 PPO；
- 没有静默丢弃 root。

因此不能把高修正率解释成 Exact solver 错误。

### 8.2 是规划目标之间的匹配问题

`Q_plan` 主要由 family competence/readiness 驱动，而真实可执行 `Q` 由当前 task 的局部 anchor/ERV 驱动。两者在后期明显分离。

如果 BACE 的设计意图是“competence 越高就越多使用 branch”，当前公式是有意的；但从信息采样角度，branch 更直接对应的是不确定性和 marginal ERV，因此当前 quota 公式存在可优化的规划错配。

## 9. 建议

### 9.1 必须保留

- Exact capacity correction；
- `R+Q=8`；
- selected-worker branch execution；
- packed root generation；
- Replay validation；
- occurrence-level PPO credit。

不能为了降低报表中的修正而强行执行 `Q_plan=6`。

### 9.2 首选：lagged capacity-aware quota

不读取当前 task 的 root outcome，仍然保持 no-pilot/lagged 原则，但使用过去 step 的 family capacity 统计做保守上限：

```text
C_hat = family 的历史 capacity 分位数
Q_plan = min(round(6 * readiness), conservative_capacity(C_hat))
R_plan = 8 - Q_plan
```

对 `pick-and-place`，后期历史 `C` 已经明显偏低，这会让一部分 task 初始多生成 root，减少 correction wave。

### 9.3 同时记录 uncertainty，而不只记录 readiness

可以增加：

- family posterior variance/entropy；
- 首次 `A` 和 `C`；
- 每个 anchor 的 `delta_1/delta_2`；
- action diversity 和 prefix diversity；
- 原因标签 `no_structural_anchor`、`anchor_ceiling`、`erv_below_threshold`。

这样能判断高 readiness 是“能力稳定”还是“仍有可利用的不确定性”。

### 9.4 不建议马上降低 ERV threshold

把 `0.005` 降低会减少修正，但会引入低信息价值 branch，改变 Batch-ERV 的选择标准，不能作为单纯性能修复。应先做离线 trace replay，评估不同 threshold 对 branch 价值、训练时间和 validation 的影响。

### 9.5 不建议直接全局提高 `R_min`

把 `R_min=2` 改成 3 或 4 可能减少 correction，但会对所有 family 强制减少 branch，改变正式算法参数。更合理的是先做 family-specific、lagged、可回退的容量上限。

## 10. 最终判断

`pick-and-place` 初始 2-root 修正严重的根本原因是：

```text
初始 root 数由“family 是否擅长完成任务”决定；
最终 branch 数由“当前 task 是否有足够 action-credit 信息价值”决定。
```

训练后期 `pick-and-place` 的 family readiness 已接近 1，因此大量 task 被安排为 `R=2,Q=6`；但策略更集中、具体 task 差异更大、两条 root 产生的 anchor/ERV 不足，导致约 71% 的这些 task 触发修正。

最准确的表述是：

> 这不是“成功率高却算法失败”，而是“family-level competence 高，却不能保证当前 task 的局部 branch 信息容量高”。当前 correction 机制保证了正确性，但初始 quota 对后期 pick-and-place 的 local capacity 过于乐观。

## 11. 数据来源

- `experiments/alfworld-qwen2.5-1.5b-exact/bace_artifacts/bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0/step_*/family_topology_plans.jsonl`
- `experiments/alfworld-qwen2.5-1.5b-exact/bace_artifacts/bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0/step_*/capacity_checks.jsonl`
- `experiments/alfworld-qwen2.5-1.5b-exact/bace_artifacts/bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0/step_*/anchors.jsonl`
- `experiments/alfworld-qwen2.5-1.5b-exact/analysis/training_curves_1_150/all_training_scalars_1_150.csv`
- `verl-agent-src/recipe/bace_gigpo/topology.py`
- `verl-agent-src/recipe/bace_gigpo/batch_erv.py`
