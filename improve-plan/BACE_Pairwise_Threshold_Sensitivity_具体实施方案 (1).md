# BACE：Pairwise + ERV Threshold 敏感性分析与实现方案
## ——先离线筛阈值，再在线验证两种 Pairwise，并同时评估容量修正与训练速度

**日期**：2026-08-29  
**适用对象**：当前 BACE / Exact Batch-ERV 主线  
**当前基线**：Full Exact Batch-ERV，`tau_BERV = 0.005`，`L_max = 2`，总 leaf budget `B = 8`  
**文档目的**：明确下一步到底做什么、先后顺序是什么、两种 Pairwise 如何实现、threshold 如何筛选、如何公平比较性能与 wall-clock。

---

# 0. 结论先行

当前不应仅根据 A1 中：

- realized allocation replan rate `45.575%`；
- P-Fixed median relative gain `0.695%`；
- `tau=0.005` 下 mean threshold shortfall probability `1.408%`；

就判断 Pairwise “值得”或“不值得”。

A1 已经证明两件事：

1. **feedback 经常改变后续 acquisition plan**；
2. 在当前 `tau=0.005` 和当前 BERV objective 下，Full Batch 的 acquisition value 通常已经接近 Pairwise。

但 A1 **没有回答**：

1. Pairwise 是否能减少当前逐 root 的 capacity-correction barrier；
2. Pairwise 在真实 scheduler 下总 wall-clock 是否更慢；
3. `tau_BERV=0.005` 是否过松；
4. 更高 threshold 下 Pairwise-Stopping 是否会明显增加；
5. 少做弱边际 branches 是否会改善 late-stage validation。

因此下一步采用：

\[
\boxed{
\text{离线 threshold sensitivity}
\rightarrow
\text{实现 Pairwise-Fixed}
\rightarrow
\text{实现 Pairwise-Stopping / lazy capacity}
\rightarrow
\text{小规模 continuation A/B}
}
\]

而不是直接把 Pairwise 升级为主方法，也不是直接放弃。

---

# 1. 当前 A1 给我们的事实基线

A1 在 `2400` 个训练 task 上得到：

\[
Q\ge3:\quad 1130/2400=47.083\%.
\]

其中：

- early：`10.625%`
- middle：`59.750%`
- late：`70.875%`

所以 Pairwise 的适用机会主要集中在中后期。

第一 pair 的真实 outcome 后：

\[
\text{anchor allocation replan rate}=45.575\%,
\]

\[
\text{action multiset change rate}=80.885\%.
\]

说明 feedback 确实经常改变后续实验设计。

但 model-based P-Fixed 的：

\[
\text{mean absolute gain}=0.0022791961,
\]

\[
\text{median relative gain}=0.695\%.
\]

同时，在当前：

\[
\tau_{\mathrm{BERV}}=0.005
\]

下：

\[
\text{mean threshold shortfall probability}=1.408\%.
\]

late 上升到约：

\[
2.188\%.
\]

所以当前结论应是：

\[
\boxed{
\text{Pairwise feedback 是活的，但在 }0.005\text{ 下 stopping pressure 仍较弱。}
}
\]

---

# 2. 为什么仍然值得做 Pairwise

## 2.1 A1 只算 acquisition value，没有算完整 wall-clock

当前 Full Batch 为了正式进入 branch phase，需要先满足完整 quota 的 capacity：

\[
C_g\ge Q_g.
\]

若不满足，当前逻辑：

\[
R_g\leftarrow R_g+1,
\qquad
Q_g\leftarrow Q_g-1,
\]

然后生成一条 root，重新构造 anchor / posterior / capacity，再检查。

这可能产生多轮：

```text
+1 root
→ recompute
→ +1 root
→ recompute
→ +1 root
→ recompute
```

Pairwise 本身虽然会增加 branch-side round：

```text
2B → update → 2B
```

但如果调度设计合理，可以把容量需求从：

\[
C\ge Q
\]

变成当前 round 只需：

\[
C\ge k,
\qquad
k=\min(2,Q_{\rm remain}).
\]

因此：

\[
\boxed{
\text{Pairwise 增加 branch barrier，
但可能减少 pre-branch capacity-correction barrier。}
}
\]

最终训练是否更慢必须看：

\[
T_{\mathrm{rollout}}
=
T_{\mathrm{initial-root}}
+
T_{\mathrm{capacity-correction}}
+
T_{\mathrm{branch-rounds}}
+
T_{\mathrm{fallback-root}}.
\]

不能只看 branch round 数。

---

# 3. 一个必须避免的错误：不要简单“一次补两个 roots”

原 capacity correction 每次只加一条 root 有合理原因。

例如：

\[
Q=5,\quad C=2.
\]

如果直接因为“Pairwise 一次两条”就补两个 roots，可能过度修正。

第一条新增 root 可能已经让：

\[
C:2\rightarrow4,
\]

同时：

\[
Q:5\rightarrow4.
\]

这时已经满足：

\[
C=Q=4.
\]

因此本方案不采用：

```text
capacity 不足 → 固定一次补 2 roots
```

而采用：

\[
\boxed{
\textbf{lazy capacity correction}
}
\]

即：

> 只为“下一 pair”检查容量；确实不足时仍一次只新增一条 root，但所有 task 的 correction roots 在全局层面一起 packed 生成。

这样既避免 over-correction，又减少 task-wise sequential waves。

---

# 4. 两种 Pairwise：本轮都值得做，但职责不同

本轮在线实现明确保留两种版本。

---

# 5. P1：Pairwise-Fixed —— 纯 feedback acquisition 对照

## 5.1 目的

只回答：

\[
\boxed{
\text{Full Batch 一次性冻结后续实验，是否损失了有意义的 feedback value？}
}
\]

P1 是最干净的算法对照。

---

## 5.2 P1 保持不变的部分

P1 **完全保留当前 Full Batch 的 root-side capacity correction**。

也就是说仍然先完成：

\[
C_g\ge Q_g.
\]

随后冻结：

- final natural-root backbone；
- anchor pool；
- candidate action support；
- final quota \(Q_g\)。

然后 branch phase 才改成：

\[
K=2.
\]

---

## 5.3 P1 branch 流程

若：

\[
Q=4,
\]

则：

```text
Frozen root support
↓
solve exact quota=2
↓
execute pair 1
↓
observe outcomes
↓
update Beta posterior
↓
solve exact quota=2 again
↓
execute pair 2
```

最终仍：

\[
Q_{\rm executed}=Q_{\rm frozen}.
\]

P1 中：

- 不 early stop；
- 不 branch→root fallback；
- 不因为 post-pair threshold 下降而减少 quota；
- `L_max=2` 跨 round 累计。

---

## 5.4 P1 为什么仍要实现

它能最干净地区分：

\[
\text{feedback selection effect}
\]

和：

\[
\text{branch quantity / topology effect}.
\]

如果 P1 本身比 Full Batch 更好，就说明：

> 即使最终 branch 数不变，仅让真实 outcome 参与下一 pair 的选择，就有训练收益。

---

# 6. P2：Pairwise-Stopping + Lazy Capacity —— 更接近最终候选

P2 是本轮真正需要重点评估的版本。

它同时做两件事：

1. Pairwise feedback；
2. threshold-aware stopping / lazy capacity。

---

# 7. P2 的核心语义

不再要求在 branch phase 开始前一次性满足：

\[
C\ge Q_{\rm plan}.
\]

只要求当前 pair：

\[
C_{\tau}\ge k,
\qquad
k=\min(2,Q_{\rm remain}).
\]

因此如果：

\[
Q_{\rm plan}=6
\]

而当前：

\[
C_\tau=2,
\]

不再继续为了未来可能的 4 条 branch 提前补 roots。

直接先做当前：

\[
2B.
\]

然后：

\[
\text{outcome}
\rightarrow
\text{posterior update}
\rightarrow
\text{recompute BERV/capacity}.
\]

---

# 8. P2 的状态变量

对每个 task 维护：

\[
R_{\rm init}
=
B-Q_{\rm plan},
\]

\[
b_{\rm done}
=
\text{已经实际执行的 branch 数},
\]

\[
r_{\rm corr}
=
\text{branch phase 中新增的 correction/fallback roots},
\]

当前仍可使用的原计划 refinement slots：

\[
q_{\rm remain}
=
Q_{\rm plan}
-
b_{\rm done}
-
r_{\rm corr}.
\]

始终保持：

\[
R_{\rm init}
+
b_{\rm done}
+
r_{\rm corr}
+
q_{\rm remain}
=
B.
\]

---

# 9. P2 每轮怎么做

定义：

\[
k_{\rm target}
=
\min(2,q_{\rm remain}).
\]

使用当前：

- natural-root support；
- 已更新 Beta posterior；
- residual \(L_{\max}\)；

重新计算：

\[
C_\tau.
\]

---

## 9.1 若 \(C_\tau\ge k_{\rm target}\)

直接求：

\[
\text{Exact quota}=k_{\rm target}
\]

的最优 pair，执行。

---

## 9.2 若 \(0<C_\tau<k_{\rm target}\)

例如：

\[
k_{\rm target}=2,\quad C_\tau=1.
\]

不强行补成 2。

先执行当前唯一一个 threshold-positive branch：

\[
k_{\rm actual}=1.
\]

观察 outcome 后再次判断。

这样避免：

> 明明有一个明确高价值实验，却为了凑 pair 先去生成 root。

---

## 9.3 若 \(C_\tau=0\)

此时进入 stopping 语义。

剩余：

\[
q_{\rm remain}
\]

不再用于 branch。

全部转为 natural roots：

\[
r_{\rm fallback}
=
q_{\rm remain}.
\]

这些 fallback roots **一次性 global packed 生成**。

然后结束当前 task 的 branch phase：

\[
\boxed{
\text{one-way fallback}
}
\]

fallback roots 不重新开启 branch。

---

# 10. P2 与原容量修正的区别

当前 Full Batch：

```text
先为完整 Q 建够 capacity
↓
再开始所有 branches
```

P2：

```text
只为下一 pair 检查 capacity
↓
有就先做
↓
看结果
↓
再决定是否还需要后续 refinement
```

因此 P2 的潜在系统收益来自：

\[
\boxed{
\text{不再为了未来尚未确定需要的 branches 预先进行多轮 root correction。}
}
\]

---

# 11. 关于 threshold：0.005 不能直接当成最终值

A1 的 P-Threshold 结论严格条件化于：

\[
\tau=0.005.
\]

所以：

\[
1.408\%
\]

的 mean shortfall probability 只能说明：

> 在 0.005 这个 threshold 下，post-pair capacity collapse 不常见。

不能推出：

> Pairwise-Stopping 一般不常见。

因为：

\[
C_\tau
=
\#\{\Delta \mathrm{BERV}\ge\tau\}
\]

对 \(\tau\) 很敏感。

---

# 12. 第一阶段：离线 Threshold Sensitivity

正式在线训练之前，先离线计算：

\[
\boxed{
\tau
\in
\{0.005,\ 0.0075,\ 0.010,\ 0.015\}
}
\]

如果计算成本很低，可额外加入：

\[
0.0125.
\]

不要一开始就在线完整训练所有 threshold。

---

# 13. 离线分析必须复用当前 Exact BERV

不能通过：

```text
把历史 selected BERV 简单重新 threshold
```

来完成。

必须在每个 counterfactual threshold 下重新计算：

- effective anchor；
- per-anchor `values_by_size`；
- threshold-positive capacity；
- Pairwise round policy；
- outcome 后 posterior；
- post-pair capacity；
- stopping/fallback。

同 action 重复采样仍必须复用当前生产：

```text
ExactBatchErvEngine._beta_binomial_probability
```

不能退化为独立 Bernoulli 近似。

---

# 14. 离线 Threshold Sensitivity 要输出的 10 个核心指标

对每个：

\[
\tau
\]

分别按：

```text
overall
early / middle / late
task family
Q
```

汇总。

---

## 14.1 初始 Pair Feasibility

\[
\boxed{
P(C_\tau\ge2)
}
\]

尤其是在：

\[
Q\ge3
\]

的 task 上统计。

它回答：

> 如果只要求下一 pair，当前自然 roots 多频繁已经足够直接开始 branch？

---

## 14.2 Full-Batch Eager Capacity Gap

\[
\boxed{
Q-C_\tau
}
\]

统计：

- mean；
- median；
- `C<Q` task ratio。

它反映 current Full Batch 为完整 quota 准备 capacity 的压力。

---

## 14.3 Counterfactual Eager Correction Rounds

按历史 natural-root 物理顺序模拟当前：

```text
+1 root
→ recompute
→ +1 root
```

得到：

\[
N_{\rm correction}^{\rm eager}(\tau).
\]

---

## 14.4 Pairwise Replan Rate

沿用 A1：

\[
P(
\text{post-pair optimal remainder}
\neq
\text{frozen remainder}
).
\]

---

## 14.5 Post-Pair Zero-Capacity Probability

\[
\boxed{
P(C_\tau^{\rm after}=0)
}
\]

这比只看 shortfall 更直接对应 P2 stopping。

---

## 14.6 Post-Pair Shortfall Probability

\[
\boxed{
P(
C_\tau^{\rm after}
<
q_{\rm remain}
)
}
\]

继续保留 A1 原指标。

---

## 14.7 Expected Executed Branch Count

在 P2 policy 下：

\[
\boxed{
E[Q_{\rm executed}\mid\tau]
}
\]

同时报告：

\[
E[Q_{\rm plan}-Q_{\rm executed}].
\]

---

## 14.8 Expected Fallback Root Count

\[
\boxed{
E[R_{\rm fallback}\mid\tau].
}
\]

尤其看 late。

---

## 14.9 Expected Pair Rounds

\[
\boxed{
E[N_{\rm pair-rounds}\mid\tau]
}
\]

用于估计新增 branch-side barrier。

---

## 14.10 Retained BERV Value

以当前主线：

\[
\text{Full Batch},\ \tau=0.005
\]

为 reference。

定义：

\[
\boxed{
\mathrm{Retain}(\tau)
=
\frac{
E[V_{\mathrm{P2},\tau}]
}{
E[V_{\mathrm{Full},0.005}]
}
}
\]

需要同时报告 absolute value，不能只报告 ratio。

---

# 15. 额外建议：Weak-Branch Mass

定义弱边际区间：

\[
0.005
\le
\Delta\mathrm{BERV}
<
\tau.
\]

统计：

\[
\boxed{
\rho_{\rm weak}(\tau)
}
\]

即当前 0.005 主线中有多少 branch opportunity 会被更高 threshold 剔除。

同时计算这些机会占总 BERV value 的比例：

\[
\boxed{
\rho_{\rm weak-value}(\tau).
}
\]

如果看到：

```text
weak branch count 很高
但 weak-value mass 很低
```

就说明当前 0.005 可能确实在消耗大量低边际 refinement。

---

# 16. Threshold 候选怎么选

离线结束后，不直接根据 validation 选，因为还没在线训练。

筛出最多两个在线候选：

\[
\tau_{\rm low}^*,
\qquad
\tau_{\rm high}^*.
\]

推荐工程筛选原则：

---

## Candidate A：保守候选

满足大致：

\[
\mathrm{Retain}(\tau)\ge97\%
\]

同时 late：

\[
E[Q_{\rm executed}]
\]

或 weak-branch count 已有可见下降。

---

## Candidate B：激进候选

满足大致：

\[
\mathrm{Retain}(\tau)\ge90\%-95\%
\]

同时：

- stopping probability 明显提高；
- expected branch count 明显降低；
- late fallback root 增加明显。

---

这些是工程筛选规则，不是理论常数。

如果：

\[
0.0075
\]

和：

\[
0.01
\]

恰好分别落在这两个位置，就在线测这两个。

如果离线结果显示 knee 在别处，就不要机械坚持。

---

# 17. 必须画的 Threshold 曲线

至少画：

### Figure 1

横轴：

\[
\tau
\]

纵轴：

\[
E[Q_{\rm executed}].
\]

---

### Figure 2

横轴：

\[
\tau
\]

纵轴：

\[
\mathrm{Retain}(\tau).
\]

---

### Figure 3

同一张图双轴：

```text
executed branch count
vs
retained BERV value
```

用于找 knee point。

---

### Figure 4

\[
P(C_\tau\ge2)
\]

vs

\[
P(C_\tau\ge Q).
\]

这张图直接显示：

> Pairwise 当前-pair capacity 与 Full-Batch full-quota capacity 的差距。

---

### Figure 5

post-pair：

```text
zero-capacity probability
shortfall probability
```

随 threshold 的变化。

---

### Figure 6

按 early / middle / late 分开的 expected executed branch count。

---

# 18. 离线结果的 Go / Hold 判断

## 情况 A：提高 threshold 几乎不改变 branch 数

例如：

```text
tau=0.005 → 0.01
executed Q 只下降 2%
```

则：

> threshold 不是重要杠杆。

在线只保留：

```text
0.005 baseline
Pairwise-Fixed
Pairwise-Stopping 0.005
```

不做 threshold sweep。

---

## 情况 B：branch 明显减少，value 基本保留

例如：

```text
late executed branches -20%
retained BERV value 97%
```

这是最理想结果。

优先在线测试该 threshold。

---

## 情况 C：branch 减少，但 value 同步大幅下降

例如：

```text
branches -20%
BERV value -18%
```

说明 threshold 只是粗暴删实验。

不应优先。

---

## 情况 D：只在 late 出现 knee

这反而很有研究价值。

第一轮仍先测试固定 threshold，不立刻做 dynamic threshold。

如果固定候选有效，再研究：

```text
stage-aware threshold
```

---

# 19. 第二阶段：代码实现顺序

严格按以下顺序。

---

## Step 1：先完成离线 Threshold Sensitivity

输出：

```text
threshold_summary.csv
threshold_phase_summary.csv
threshold_family_summary.csv
threshold_q_summary.csv
threshold_task_audit.parquet
figures/
```

---

## Step 2：实现 Pairwise 核心 Exact Solver

先只支持：

\[
K=2.
\]

复用现有 Exact machinery。

必须支持：

```text
current posterior
remaining quota
residual Lmax
stable tie identity
```

---

## Step 3：实现 P1 Pairwise-Fixed

这是最小风险版本。

不改：

```text
capacity correction
final Q
fallback semantics
```

先把 feedback replanning 跑通。

---

## Step 4：实现 P2 Pairwise-Stopping / Lazy Capacity

增加：

```text
current-pair capacity
C=0 stopping
C=1 single branch
fallback roots
one-way fallback
```

---

## Step 5：再加入 threshold config

推荐：

```yaml
algorithm:
  bace:
    erv_threshold: 0.005
    pairwise:
      enabled: true
      mode: stopping
      batch_size: 2
```

threshold 必须是单一显式配置源。

---

# 20. P2 推荐伪代码

```text
Input:
    B
    planned branch quota Q_plan
    tau_BERV
    K = 2
    Lmax = 2

R_init = B - Q_plan
generate R_init natural roots

b_done = 0
r_corr = 0

while True:

    q_remain = Q_plan - b_done - r_corr

    if q_remain <= 0:
        break

    k_target = min(K, q_remain)

    rebuild acquisition state from:
        all natural roots
        current branch-updated posteriors
        residual Lmax

    C_tau = current threshold-positive residual capacity

    if C_tau == 0:
        # stopping
        fallback = q_remain
        generate fallback natural roots in packed stage
        r_corr += fallback
        break

    k_actual = min(k_target, C_tau)

    solve Exact global plan with quota=k_actual

    execute current pair/single branch round

    update corresponding Beta posteriors

    update residual Lmax

    b_done += k_actual

final assert:
    R_init + r_corr + b_done == B

compute final advantage/PPO exactly as current mainline
```

---

# 21. 关于 “lazy capacity correction” 的第一版边界

第一版 P2 不做无限：

```text
root → branch → root → branch → root ...
```

当 post-pair：

\[
C_\tau=0
\]

时直接：

\[
\boxed{
\text{剩余 slots 全部 fallback roots，branch phase 结束。}
}
\]

这是最简单且可解释的版本。

如果之后发现：

> 很多 task 在 fallback root 后会重新形成非常高价值 anchor，

再研究 rolling reopen。

当前不做。

---

# 22. Global Round-Wise Packing

Pairwise 绝不能 task-wise 顺序执行。

错误：

```text
task 1 pair 1
task 1 pair 2
task 2 pair 1
task 2 pair 2
...
```

正确：

```text
Global Pair Round 1:
    收集所有 active task 的 1~2 branches
    一起 packed execute

Global Pair Round 2:
    所有 task 更新 posterior 后
    再统一收集下一轮 requests
```

P2 stopping 后的 fallback roots：

```text
不要 task 一停就立即生成
```

而是累计后：

```text
global packed fallback-root stage
```

这样 Pairwise 才有可能在 H100/H20 上保持高利用率。

---

# 23. Runtime 必须新增的统计

每 step 至少记录：

```text
time/initial_root
time/capacity_correction
time/branch_generation
time/fallback_root
time/acquisition_compute
time/ppo
time/step_total
```

以及：

```text
count/root_generation_waves
count/capacity_correction_waves
count/pairwise_branch_rounds
count/fallback_root_waves
```

同时记录：

```text
generated_tokens/root
generated_tokens/branch
generated_tokens/total
```

这样才能验证用户提出的核心 hypothesis：

\[
\boxed{
\Delta T_{\rm branch-round}>0
}
\]

但：

\[
\boxed{
\Delta T_{\rm capacity-correction}<0
}
\]

并最终可能：

\[
\boxed{
\Delta T_{\rm step}\le0.
}
\]

---

# 24. 第三阶段：在线实验矩阵

先不要 from-scratch。

优先从共同：

\[
\boxed{\text{step-100 checkpoint}}
\]

继续到：

\[
150.
\]

原因：

- Pairwise opportunity 主要在 middle/late；
- 当前 late gap 正是我们要解释的；
- 能显著节省计算；
- 同 checkpoint continuation 有更强因果解释。

---

# 25. 最小 5 组在线实验

## C0：当前主线

```text
Full Batch
tau = 0.005
```

---

## C1：Pairwise-Fixed

```text
Pairwise-Fixed
K = 2
tau = 0.005
current eager capacity correction
```

回答：

\[
\boxed{
\text{feedback selection 是否有训练收益？}
}
\]

---

## C2：Pairwise-Stopping / Lazy Capacity

```text
Pairwise-Stopping
K = 2
tau = 0.005
pairwise-aware capacity
```

回答：

\[
\boxed{
\text{lazy capacity + stopping 在当前 threshold 下是否改善速度/效果？}
}
\]

---

## C3：Pairwise-Stopping + 离线候选 threshold

```text
Pairwise-Stopping
K = 2
tau = tau_low*
```

如果离线只有一个明显 knee，就只测一个。

---

## C4：Full Batch + 同 threshold

```text
Full Batch
tau = tau_low*
```

这个 control 非常重要。

它回答：

\[
\boxed{
\text{C3 的变化来自 Pairwise，还是仅仅来自 threshold？}
}
\]

---

# 26. 如果资源允许的第 6 组

若离线同时得到一个更激进的：

\[
\tau_{\rm high}^*
\]

且 retained value 仍可接受，则加：

```text
C5:
Pairwise-Stopping
tau = tau_high*
```

但不建议一开始就把：

```text
0.005 / 0.0075 / 0.01 / 0.015
```

全部在线跑。

---

# 27. 在线比较必须固定的变量

所有 C0--C4：

```text
same checkpoint
same seed
same batch
same rollout.n
same gamma
same PPO
same advantage
same anchor rule
same prior
same Lmax
same scheduler backend
same GPU count
same model
same train/val data
```

当前不要同时改：

```text
action_mean
step_advantage_w
Rmin
prior
branch loss weight
behavior correction
```

---

# 28. 在线必须看的 Performance 指标

```text
validation success
late 101--150 AUC
final 10-step mean
root success
branch success
family success
```

---

# 29. 在线必须看的 Acquisition 指标

```text
planned Q
executed Q
Pairwise round count
replan rate
action-change rate
BERV before/after pair
capacity before/after pair
stopping ratio
fallback root count
weak-branch ratio
```

---

# 30. 在线必须看的 Credit 指标

继续复用 A0：

```text
local coverage
local magnitude share
token-weighted local share
macro/local sign conflict
```

目的是确认：

> threshold / Pairwise 没有通过意外改变 credit statistics 产生新的混淆。

---

# 31. 在线必须看的 System 指标

最重要：

\[
\boxed{
T_{\rm step}
}
\]

其次：

```text
root generation waves
capacity correction waves
branch rounds
fallback waves
GPU utilization
generated tokens
```

最终不能只报告：

> Pairwise 多了一轮 branch。

而要比较：

\[
\boxed{
\text{整个 rollout + PPO step 的 wall-clock}
}
\]

---

# 32. 如何解释 C0--C4

## Case A

\[
C1>C0
\]

而：

\[
C2\approx C1.
\]

说明：

> feedback-aware selection 有价值，但 stopping/lazy capacity 贡献有限。

主方法候选：

\[
\boxed{\text{Pairwise-Fixed}}
\]

---

## Case B

\[
C1\approx C0,
\qquad
C2>C1.
\]

说明：

> Full Batch 选址本身已经够好，真正有收益的是 capacity/stopping 机制。

主方法候选：

\[
\boxed{\text{Pairwise-Stopping}}
\]

---

## Case C

\[
C2>C1>C0.
\]

说明：

> feedback selection 和 stopping/lazy capacity 都有效。

---

## Case D

\[
C3>C2
\]

且：

\[
C4\approx C0.
\]

说明：

> 更高 threshold 主要在 Pairwise-Stopping 中发挥价值。

---

## Case E

\[
C4>C0,
\qquad
C3\approx C4.
\]

说明：

> 主要收益其实来自 threshold，而不是 Pairwise。

这时没必要承担额外 Pairwise barrier。

---

## Case F

\[
C3>C4>C0.
\]

说明：

> threshold 与 Pairwise 存在协同。

这是最值得进一步发展的方法结果。

---

# 33. Runtime 的解释

## 如果 C2/C3 比 C0 更快

检查是不是：

\[
N_{\rm capacity-correction-waves}
\]

明显下降。

若 Pairwise branch rounds 增加，但总：

\[
T_{\rm step}
\]

下降，则证明：

\[
\boxed{
\text{Pairwise-aware capacity 消除了更多 eager correction barrier。}
}
\]

这是重要工程结果。

---

## 如果 C2/C3 更慢但效果更好

计算：

\[
\frac{\Delta \text{validation AUC}}
{\Delta T_{\rm wall-clock}}
\]

再判断是否值得。

---

## 如果效果相同但更快

Pairwise 仍然有价值：

\[
\boxed{\text{作为更高效的 rollout scheduler/controller。}}
\]

---

# 34. 当前实现优先级

现在的推荐顺序更新为：

\[
\boxed{
\textbf{P0：Threshold Offline Sensitivity}
}
\]

然后：

\[
\boxed{
\textbf{P1：Pairwise-Fixed 核心实现}
}
\]

然后：

\[
\boxed{
\textbf{P2：Pairwise-Stopping + Lazy Capacity}
}
\]

然后：

\[
\boxed{
\textbf{C0--C4 Step-100 Continuation}
}
\]

最后根据结果决定是否：

- from-scratch；
- multi-seed；
- 动态 threshold；
- K sweep。

---

# 35. 当前不要做的事情

在这条路线被回答之前，不建议同时启动：

```text
step_advantage_w 0.5/0.75
action_mean
ECPO-style shrinkage
new root value
root-vs-branch utility
K = 1/2/4 sweep
dynamic phase threshold
new competence mapping
```

这些都可以后做，但当前一起做会破坏解释性。

---

# 36. 最终的一页式执行清单

## 离线

- [ ] 复用 A1 exact simulator；
- [ ] 加入 `tau = 0.005/0.0075/0.01/0.015`；
- [ ] 重算每个 threshold 的 effective capacity；
- [ ] 输出 `P(C>=2)`；
- [ ] 输出 `P(C>=Q)`；
- [ ] 输出 eager correction rounds；
- [ ] 输出 replan rate；
- [ ] 输出 zero-capacity probability；
- [ ] 输出 shortfall probability；
- [ ] 输出 expected executed Q；
- [ ] 输出 expected fallback roots；
- [ ] 输出 expected pair rounds；
- [ ] 输出 retained BERV value；
- [ ] 按 phase/family/Q 分解；
- [ ] 选最多两个 online threshold candidates。

## P1

- [ ] K=2 Exact solve；
- [ ] outcome 后 posterior update；
- [ ] 第二 pair 重规划；
- [ ] residual Lmax 跨 round；
- [ ] final Q 保持 frozen；
- [ ] global round-wise packing；
- [ ] unit tests；
- [ ] 2--5 step smoke。

## P2

- [ ] current-pair threshold capacity；
- [ ] `C>=2` 执行 pair；
- [ ] `C=1` 执行 single branch；
- [ ] `C=0` stopping；
- [ ] remaining slots → fallback roots；
- [ ] one-way fallback；
- [ ] global packed fallback roots；
- [ ] budget invariant；
- [ ] family-history update；
- [ ] runtime instrumentation；
- [ ] 2--5 step smoke。

## 在线

- [ ] C0 Full 0.005；
- [ ] C1 Pairwise-Fixed 0.005；
- [ ] C2 Pairwise-Stopping 0.005；
- [ ] C3 Pairwise-Stopping `tau*`；
- [ ] C4 Full `tau*`；
- [ ] 同 step-100 checkpoint；
- [ ] 同 seed；
- [ ] 同 scheduler / GPU；
- [ ] 跑至 step 150；
- [ ] 比 performance；
- [ ] 比 acquisition；
- [ ] 比 credit；
- [ ] 比 wall-clock。

---

# 37. 最终判断

当前最合理的态度不是：

\[
\text{“A1 gain 小，所以 Pairwise 不值得做”}
\]

也不是：

\[
\text{“replan rate 高，所以一定要 Pairwise”}.
\]

而是：

\[
\boxed{
\text{A1 已证明 feedback 有真实作用；
接下来需要把 threshold 与 capacity correction 纳入同一个 Pairwise 实验框架，
再用真实训练效果和完整 step wall-clock 做最终判断。}
}
\]

尤其重要的是：

\[
\boxed{
\tau_{\mathrm{BERV}}=0.005
}
\]

当前只是一个工作值，不应该被当成已经验证的最优值。

下一步首先通过离线 sensitivity 找出：

\[
\boxed{
\text{能明显减少 weak branches、提高 stopping pressure，
同时保留大部分 BERV value 的 threshold knee point。}
}
\]

然后再把这个候选与两种 Pairwise 一起做严格在线 continuation 对照。
