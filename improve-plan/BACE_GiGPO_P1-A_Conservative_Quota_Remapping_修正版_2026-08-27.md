# BACE-GiGPO P1-A：Conservative Quota Remapping 诊断性消融方案（修正版）

> 日期：2026-08-27  
> 适用范围：当前 No-Pilot + Dynamic Topology + Exact Batch-ERV + Capacity Correction 主线  
> 当前基线：`total_leaf_budget=8`，`min_natural_roots=2`  
> 核心实验：将 readiness-to-branch quota 映射从 `floor(6q + 0.5)` 调整为 `floor(4q + 0.5)`  
> 文档定位：**诊断性算法消融，不是已经证明有效的生产修复。**

---

# 0. 文档目的

P1-A 只回答一个问题：

> **当前 BACE 是否在训练后期因为 readiness 较高而过度规划 branch，导致初始 root evidence 不足、capacity correction 频繁、串行补 root 变多，并且可能损害后期训练数据的 breadth？**

P1-A 不修改：

- Exact Batch-ERV 求解；
- anchor 定义；
- action posterior；
- `batch_erv_threshold`；
- `max_branches_per_anchor`；
- capacity correction 安全机制；
- Replay；
- advantage / PPO；
- P1-S 调度实现。

P1-A 只修改 **readiness 到初始 root/branch quota 的映射**。

---

# 1. 当前生产映射

设：

\[
B=8
\]

为每个 task 的 terminal-leaf budget，当前：

\[
R_{\min}=2.
\]

因此 flexible budget：

\[
M=B-R_{\min}=6.
\]

当前代码在 `ExactBatchTopologyPlanner.initialize()` 中使用：

\[
Q_{\mathrm{plan}}
=
\left\lfloor
M q + 0.5
\right\rfloor,
\]

其中：

\[
q=P(\phi_c>\tau_{\mathrm{comp}})
\]

为 lagged family competence posterior 给出的 readiness。

因此当前映射为：

\[
\boxed{
Q_A(q)=\left\lfloor6q+0.5\right\rfloor
}
\]

并且：

\[
R_A(q)=8-Q_A(q).
\]

最大 topology 为：

```text
2 roots + 6 branches
```

注意：代码实现是 `floor(x + 0.5)`，本文统一使用该形式，而不是模糊写成 Python `round()`。

---

# 2. 当前问题：readiness 与实际 information capacity 不一致

当前 controller 用 family readiness 回答：

> 这个 task family 当前是否已经足够 competent，可以更多做局部 refinement？

但它同时把 readiness 直接映射成 branch 数量：

\[
q \longrightarrow Q_{\mathrm{plan}}.
\]

这隐含了一个较强假设：

> readiness 越高，当前具体 task 的少量 natural roots 中就越可能已经暴露足够多的高价值 branch capacity。

正式 150-step 实验显示该假设在训练后期经常不成立。

历史审计结果：

| 指标 | 结果 |
|---|---:|
| task groups | 2,400 |
| 初始规划 branch | 7,116 |
| 最终执行 branch | 5,703 |
| branch→root 修正 | 1,413 |
| 修正 / planned branch | 19.86% |
| 至少修正一次的 task | 361 |
| 出现修正的 step | 77 / 150 |
| correction root waves | 376 |
| correction root generation | 8,534.4 s ≈ 2.37 h |

后期 mismatch 更明显：

```text
readiness ↑
→ Q_plan ↑
→ initial roots ↓
→ actual anchor / ERV capacity 不一定同步 ↑
→ capacity correction ↑
```

因此 P1-A 不是针对 Exact Batch-ERV 本身，而是针对 **进入 Exact allocation 之前的 quota planning**。

---

# 3. P1-A 的修正：Conservative Quota Remapping

P1-A 将：

\[
R_{\min}:2\rightarrow4.
\]

因此：

\[
M=8-4=4.
\]

新的 quota mapping 为：

\[
\boxed{
Q_C(q)=\left\lfloor4q+0.5\right\rfloor
}
\]

并且：

\[
R_C(q)=8-Q_C(q).
\]

于是：

\[
0\le Q_C\le4,
\qquad
4\le R_C\le8.
\]

最大 topology 变为：

```text
4 roots + 4 branches
```

---

# 4. 重要修正：P1-A 不只是“Qmax 从 6 截断到 4”

这是本版文档必须明确的地方。

P1-A 不是：

\[
Q=\min(4,\lfloor6q+0.5\rfloor).
\]

而是：

\[
Q=\lfloor4q+0.5\rfloor.
\]

因此它同时改变：

1. 最大 branch quota；
2. readiness-to-branch mapping 的整体斜率。

例如：

| readiness \(q\) | 当前 A：\(\lfloor6q+.5\rfloor\) | 只限 cap B：\(\min(4,A)\) | P1-A C：\(\lfloor4q+.5\rfloor\) |
|---:|---:|---:|---:|
| 0.2 | 1 | 1 | 1 |
| 0.4 | 2 | 2 | 2 |
| 0.5 | 3 | 3 | 2 |
| 0.6 | 4 | 4 | 2 |
| 0.7 | 4 | 4 | 3 |
| 0.8 | 5 | 4 | 3 |
| 0.9 | 5 | 4 | 4 |
| 1.0 | 6 | 4 | 4 |

因此第一轮 A/C 实验真正测试的是：

\[
\boxed{
\text{更保守的 quota policy 是否比当前 mapping 更合理}
}
\]

而不是单纯测试：

```text
Qmax=4 是否更好。
```

---

# 5. 推荐的 A/B/C 三种 topology policy

为了后续做因果拆解，定义三组：

## A：Current Baseline

\[
\boxed{
Q_A(q)=\lfloor6q+0.5\rfloor
}
\]

对应当前：

```text
Rmin = 2
Qmax = 6
```

---

## B：Hard-Cap Only（后续消融）

\[
\boxed{
Q_B(q)=\min(4,\lfloor6q+0.5\rfloor)
}
\]

含义：

> 只禁止极端的 5/6 branch，不改变中等 readiness 下的 quota。

---

## C：Conservative Remapping（P1-A 主诊断）

\[
\boxed{
Q_C(q)=\lfloor4q+0.5\rfloor
}
\]

对应：

```text
Rmin = 4
Qmax = 4
```

含义：

> 整个 readiness→refinement 映射都更加保守。

---

# 6. 第一轮资源有限时只做 A/C

第一轮 P1-A 不需要马上做三组。

推荐：

```text
A = current mapping
C = conservative remapping
```

先回答：

> 更保守的 quota policy 是否能改善初始 quota/capacity 匹配、减少串行修正并降低 step wall-clock？

如果 C 明显优于 A，再补 B。

后续解释：

```text
B ≈ C > A
→ 主要问题可能是极端 Q=5/6

C > B ≈ A
→ 整个 readiness mapping 偏激进

C > B > A
→ 极端 cap 和整体 slope 都有问题
```

---

# 7. Capacity correction 必须保留

P1-A 不能删除现有 capacity correction。

新的 initial topology 仍然只是计划值。

例如：

```text
initial:
4 roots + 4 branches
```

若 initial roots 暴露的实际 information capacity 仅支持 2 branches，则仍允许：

```text
4R + 4B
→ 5R + 3B
→ 6R + 2B
```

直到：

\[
C_g\ge Q_g
\]

或：

\[
Q_g=0.
\]

因此：

\[
\boxed{
4+4\text{ 是 initial policy 上界，不是强制最终 topology。}
}
\]

这保证 P1-A 不破坏当前 Exact capacity safety mechanism。

---

# 8. P1-A 可能影响训练时间，也可能影响验证集正确率

P1-A 是算法级改动，不是纯系统优化。

它改变：

```text
natural root 数量
branch 数量
anchor evidence
independent trajectory coverage
shared-prefix local refinement 比例
trainable occurrences
trainable tokens
rollout generated tokens
PPO 数据分布
```

因此它可能产生三类结果。

## 8.1 理想情况

```text
correction ↓
serial depth ↓
wall-clock ↓
late validation ↑ / 不降
```

说明当前 mapping 同时过度增加系统开销和局部 branch 比例。

---

## 8.2 只提升系统效率

```text
wall-clock ↓
validation ≈ baseline
```

仍然是可接受结果，说明更激进的 branch quota 没有带来额外学习收益。

---

## 8.3 速度提升但性能下降

```text
wall-clock ↓
validation ↓
```

说明被减少的 branch evidence 仍具有实质学习价值。

此时不能把 P1-A 当最终修复，应进一步做：

- Hard-cap B；
- capacity-aware mapping；
- 或更精确的 readiness/capacity 分离。

---

# 9. Checkpoint blocker：当前不能直接从 Rmin=2 恢复到 Rmin=4

当前 BACE checkpoint 的 `parameter_signature` 中包含：

```text
min_natural_roots
```

恢复时要求：

```text
checkpoint parameters == current parameter_signature
```

因此：

```text
source checkpoint: Rmin=2
new run: Rmin=4
```

会直接触发：

```text
BACE collector parameter signature does not match the checkpoint
```

所以共同 checkpoint 分叉不能直接执行。

---

# 10. 必须实现显式诊断迁移机制

不得：

- 全面关闭 signature check；
- 删除 `min_natural_roots` 检查；
- 修改原 checkpoint；
- 直接编辑 checkpoint 内保存的 signature。

推荐新增一个**显式、白名单、可审计的 diagnostic migration**。

## 10.1 迁移允许条件

只有当：

\[
\Delta\text{signature}
=
\{\texttt{min\_natural\_roots}:2\rightarrow4\}
\]

时允许继续。

也就是说：

```text
saved.min_natural_roots = 2
current.min_natural_roots = 4
```

且除这一字段之外：

```text
variant
topology
acquisition
total_leaf_budget
thresholds
prior parameters
max_branches_per_anchor
replay config
tie mode
...
```

必须完全相同。

---

## 10.2 必须原样恢复的状态

迁移只允许 quota policy 在恢复后发生变化。

以下状态必须从 source checkpoint 原样恢复：

- actor；
- optimizer；
- scheduler；
- global step；
- dataloader / sampler state（若 checkpoint 已保存）；
- BACE competence history；
- family history statistics；
- RNG state（若框架已保存）。

特别是：

\[
\boxed{
\text{family competence history 不能重置。}
}
\]

否则 A/C 就不是共同状态分叉。

---

## 10.3 必须记录 migration metadata

新 run 单独记录：

```text
source_checkpoint
source_parameter_signature
current_parameter_signature
allowed_diff = min_natural_roots: 2 -> 4
migration_mode = diagnostic_only
source_global_step
source_competence_history_hash / summary
```

原 checkpoint 必须保持只读。

---

# 11. 当前短诊断：从 step145 做 5-step A/C

若正式运行目录当前只保留：

```text
global_step_145
global_step_150
```

则原先设想的 step100→150 分叉无法直接执行。

推荐第一轮：

```text
source: global_step_145

A:
Rmin=2
normal strict restore
→ train to 150

C:
Rmin=4
explicit diagnostic 2→4 migration
→ train to 150
```

共 5 个 update。

当前：

```text
16 task groups / update
```

因此：

\[
5\times16=80
\]

个 task groups。

这个规模适合做：

\[
\boxed{
\text{quota / capacity / correction / wall-clock mechanism diagnostic}
}
\]

不适合做最终学习结论。

---

# 12. 5-step 短诊断能回答什么

可以回答：

1. C 的 `Q_plan` 实际下降多少；
2. initial root 数增加多少；
3. initial information capacity 如何变化；
4. quota overprediction 是否下降；
5. correction serial depth 是否下降；
6. planned-root time 与 correction-root time 如何变化；
7. 最终实际 branch 数是否被过度压低；
8. total step wall-clock 是否下降；
9. generated/trainable tokens 如何变化。

---

# 13. 5-step 短诊断不能回答什么

不能据此下结论：

```text
Rmin4 的最终 validation 更高
Rmin4 学得更快
Rmin4 是最终主算法
```

即使 step150 的 validation 恰好更高，也只能作为 sanity signal。

原因：

- 只有 5 个 actor update；
- stochastic rollout 会快速分叉；
- validation 本身有有限样本噪声；
- 后期性能差异需要完整曲线判断。

---

# 14. Raw correction 下降不能单独作为成功标准

这是本版 P1-A 的另一个关键修正。

C 一开始就请求更少 branch，因此：

```text
Q_plan ↓
```

本身就会让 raw correction count 具有结构性下降倾向。

极端地：

```text
Q_plan = 0
```

会得到：

```text
correction = 0
```

但这显然不能证明 topology 更好。

因此删除原先类似：

```text
correction下降30%~50%即成功
```

的单一验收标准。

---

# 15. P1-A 的核心诊断指标

## 15.1 Initial quota deficit

对 task \(g\)，在任何 correction 之前：

\[
D_g^{(0)}
=
\max(0,Q_{g,\mathrm{plan}}-C_g^{(0)}),
\]

其中：

\[
C_g^{(0)}
\]

是 initial planned roots 完成后得到的 information capacity。

它直接衡量：

> 初始 quota 比真实可支持 capacity 高估多少。

---

## 15.2 Normalized initial quota deficit

推荐：

\[
\boxed{
D_{g,\mathrm{norm}}^{(0)}
=
\frac{
\max(0,Q_{g,\mathrm{plan}}-C_g^{(0)})
}{
\max(Q_{g,\mathrm{plan}},1)
}
}
\]

对于：

\[
Q_{g,\mathrm{plan}}=0
\]

的 task，应额外单独统计占比，不应把该值与正常 quota task 混为一谈。

---

## 15.3 Correction / planned branch

\[
\boxed{
\text{normalized correction rate}
=
\frac{
N_{\mathrm{branch\rightarrow root}}
}{
N_{\mathrm{planned\ branch}}
}
}
\]

该指标比 raw correction count 更公平，但仍需和 initial deficit 联合解释。

---

## 15.4 Correction serial depth

对每个 task 记录：

```text
0 corrections
1 correction
2 corrections
...
```

报告：

- mean depth；
- p50/p90；
- max depth；
- tasks remaining deficient after correction wave 1/2/3/...；
- correction waves / update。

该指标最直接对应当前多轮追加 root 带来的串行 latency。

---

# 16. 系统效率指标必须看总账

P1-A 可能把原来后补的 correction roots 前移成 planned roots。

因此不能只比较：

```text
capacity correction time
```

必须同时报告：

\[
T_{root,total}
=
T_{planned-root}
+
T_{correction-root}.
\]

完整记录：

```text
planned_root_generation_seconds
correction_root_generation_seconds
root_generation_seconds_total
branch_suffix_generation_seconds
replay_seconds
PPO/update_seconds
training_step_wallclock
```

最终最关键的是：

\[
\boxed{
T_{step,total}
}
\]

而不是 correction time 单项。

---

# 17. Leaf budget 相同不等于 compute / training mass 相同

无论 A 还是 C，最终仍满足：

\[
R+Q=8.
\]

但：

```text
完整 natural root
```

和：

```text
深 anchor 的 branch suffix
```

通常长度不同。

因此 P1-A 会改变：

- generated tokens；
- environment decisions；
- trainable macro-action occurrences；
- trainable action tokens；
- root/branch gradient mass；
- PPO batch composition。

必须记录：

\[
N^{root}_{trainable\ occurrences},
\]

\[
N^{branch}_{trainable\ occurrences},
\]

\[
N^{root}_{trainable\ tokens},
\]

\[
N^{branch}_{trainable\ tokens}.
\]

同时记录 rollout compute：

\[
N^{root}_{generated\ tokens},
\qquad
N^{branch}_{generated\ tokens}.
\]

注意：

```text
trainable tokens
```

与：

```text
rollout-generated tokens
```

不是同一指标。

---

# 18. 推荐 readiness-conditioned 分析

由于 P1-A 本质上改变：

\[
q\mapsto Q,
\]

建议将 readiness 分 bin：

```text
[0.0, 0.2)
[0.2, 0.4)
[0.4, 0.6)
[0.6, 0.8)
[0.8, 1.0]
```

每个 bin 报告：

- task count；
- mean readiness；
- mean `Q_plan`；
- mean initial roots；
- mean `C^(0)`；
- mean normalized deficit；
- mean `Q_final`；
- correction depth；
- step/root/branch time contribution。

这样可以回答：

> 当前问题是否主要集中在 readiness 很高的区间？

这也为未来 B vs C 的解释提供直接证据。

---

# 19. 第一阶段：Mechanism Success 标准

step145→150 的 5-step A/C 只判断机制。

推荐认为 C 值得继续，当大多数以下条件同时成立：

1. normalized initial quota deficit 明显下降；
2. correction serial depth 明显下降；
3. correction waves / update 下降；
4. total root generation time 没有因 planned roots 增加而恶化太多；
5. total step wall-clock 下降；
6. final branch count 没有简单坍缩到接近 0；
7. generated/trainable token composition 变化可解释；
8. replay / PPO correctness 无异常。

注意：

\[
\boxed{
\text{raw correction count下降不是独立 success criterion。}
}
\]

---

# 20. 第二阶段：Learning Success 标准

只有完整 0→150 run 才判断学习质量。

重点比较：

## 20.1 Validation

- validation success curve；
- validation AUC；
- step 105–150 mean；
- final validation success；
- per-family late-stage mean；
- 多 seed 方差。

当前已有历史 evidence 显示 BACE 的主要 validation gap 集中在后 50 step，因此：

\[
\boxed{
\text{late-stage validation mean / AUC 比单独 step150 更重要。}
}
\]

---

## 20.2 Training composition

比较：

- root fraction；
- branch fraction；
- trainable root/branch occurrences；
- trainable root/branch tokens；
- generated root/branch tokens；
- branch success rate；
- natural-root success rate。

---

## 20.3 系统效率

比较：

- total wall-clock；
- total generation wall-clock；
- root generation；
- correction generation；
- branch generation；
- PPO/update time。

---

# 21. 推荐实验顺序

## Stage 0：实现 checkpoint migration

只支持：

```text
min_natural_roots: 2 -> 4
```

的 diagnostic migration。

所有其它 signature diff 继续 strict reject。

---

## Stage 1：step145 短 A/C

### A

```text
source = global_step_145
Rmin = 2
Q(q) = floor(6q + 0.5)
```

### C

```text
source = same global_step_145
Rmin = 4
Q(q) = floor(4q + 0.5)
diagnostic migration enabled
```

运行 5 updates。

只做 mechanism/profile 判断。

---

## Stage 2：完整 A/C 150-step

只有 Stage 1 显示 C 的 mechanism 明显更合理后再运行。

```text
A: step0→150, current mapping
C: step0→150, conservative mapping
```

然后判断：

```text
系统效率
+
validation / learning
```

是否同时改善。

---

## Stage 3：如 C 有效，再补 B

```text
B: Q = min(4, floor(6q + 0.5))
```

用于区分：

```text
“Q=5/6 太激进”
vs
“整个 readiness-to-quota mapping 太激进”
```

---

## Stage 4：多 seed

只有完整单 seed A/C/B 信号明确后，再进入多 seed 论文实验。

---

# 22. 实现建议

## 22.1 C 组最小实现

当前 planner 已参数化：

```text
min_natural_roots
```

所以 C 组正常从 step0 开始训练时，只需：

```text
algorithm.bace.min_natural_roots=4
```

即可自然得到：

\[
Q=\lfloor4q+0.5\rfloor.
\]

不需要改 Exact Batch-ERV。

---

## 22.2 B 组需要单独 policy mode

B 不是简单设置 `min_natural_roots=4`。

推荐未来增加：

```text
quota_policy = current
quota_policy = hard_cap_4
quota_policy = conservative_rmin4
```

对应：

```text
current:
Q = floor(6q + 0.5)

hard_cap_4:
Q = min(4, floor(6q + 0.5))

conservative_rmin4:
Q = floor(4q + 0.5)
```

这样避免用配置组合隐式表达不同算法。

---

# 23. 推荐新增日志字段

每 task：

```text
readiness
quota_policy
planned_root_count
planned_branch_count
initial_information_capacity
initial_quota_deficit
normalized_initial_quota_deficit
final_root_count
final_branch_count
correction_count
correction_depth
```

每 step：

```text
planned_branches_total
final_branches_total
corrections_total
correction_rate_normalized
correction_waves
planned_root_generation_seconds
correction_root_generation_seconds
root_generation_seconds_total
branch_generation_seconds
step_wallclock
root_generated_tokens
branch_generated_tokens
root_trainable_occurrences
branch_trainable_occurrences
root_trainable_tokens
branch_trainable_tokens
```

---

# 24. 推荐实验命名

建议：

```text
A:
bace_exact_quota_current_rmin2_seed0

C short diagnostic:
bace_exact_quota_rmin4_from145_diag_seed0

C full:
bace_exact_quota_rmin4_full_seed0

B future:
bace_exact_quota_hardcap4_seed0
```

checkpoint migration run 必须在名字或 metadata 中包含：

```text
from145
migration
```

避免误认为独立从零训练。

---

# 25. 结果解释矩阵

| 结果 | 解释 | 下一步 |
|---|---|---|
| mismatch ↓，wall-clock ↓，validation ↑ | 当前 mapping 过激，C 很有希望 | 补 B + 多 seed |
| mismatch ↓，wall-clock ↓，validation ≈ | 激进 branch 没带来额外学习收益 | C 可作为更高效候选 |
| mismatch ↓，wall-clock ↓，validation ↓ | branch refinement 仍有价值 | 测 B / capacity-aware policy |
| mismatch ↓，但 wall-clock 不降 | root 前移的 token 成本抵消 correction 收益 | 优先配合 P1-S / P2-S |
| raw correction ↓，normalized deficit 不变 | 主要是机械减少 quota | 不能算成功 |
| C 的 final branch 大幅坍缩 | mapping 过于保守 | 测 B 或介于 4q/6q 的 mapping |
| A/C 差异很小 | 当前后期问题可能不主要来自 quota mapping | 转向 lineage / scheduler / other factors |

---

# 26. P1-A 当前不做的事情

P1-A 第一轮不修改：

- family Beta prior；
- competence threshold；
- history forgetting；
- Exact BERV threshold；
- max branches per anchor；
- branch scheduler；
- Pairwise \(K=2\)；
- advantage；
- lineage weighting；
- behavior correction。

原因是：

\[
\boxed{
\text{先单独验证 quota mapping，避免多组件共同变化导致无法归因。}
}
\]

---

# 27. 与 P1-S 的边界

## P1-S

问：

> 相同 rollout 计划，能不能跑得更快？

主要改变：

```text
physical workers
active compaction
capacity chunking
```

原则上不改变训练数据分布。

---

## P1-A

问：

> 当前 initial root/branch quota 本身是否规划得太激进？

会改变：

```text
root/branch composition
anchor evidence
trainable data
validation curve
```

因此 P1-A 不应被描述为纯训练时间优化。

---

# 28. 最终推荐

P1-A 当前正式定义为：

\[
\boxed{
\textbf{Conservative Quota Remapping Diagnostic}
}
\]

即将：

\[
Q_A(q)=\lfloor6q+0.5\rfloor
\]

改为：

\[
Q_C(q)=\lfloor4q+0.5\rfloor.
\]

它要验证的核心不是：

> “4 roots + 4 branches 一定更好。”

而是：

> **当前 family readiness 是否被过度用作 branch 数量信号，从而造成 quota overprediction、串行 capacity correction 和过高的局部 refinement 比例。**

推荐执行顺序：

```text
1. 实现严格白名单 checkpoint migration
2. 从 step145 做 5-step A/C 机制诊断
3. 比较 normalized quota deficit、serial depth、total wall-clock、token composition
4. 若机制信号明确，再做完整 0→150 A/C
5. 只有完整 run 才判断 validation / learning quality
6. 若 C 有效，再补 Hard-Cap B，区分 cap effect 与 full remapping effect
7. 最后再进入多 seed
```

最终只有当：

\[
\text{planning mismatch}\downarrow,
\]

\[
\text{serial correction}\downarrow,
\]

\[
T_{train}\downarrow,
\]

并且完整训练中：

\[
\text{validation AUC / late-stage performance}\ge\text{baseline},
\]

才应考虑把 P1-A 从“诊断性消融”升级为正式 topology controller 改进。

