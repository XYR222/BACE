# A1：Full Exact Batch-ERV → Pairwise \(K=2\) 离线反馈价值审计
## ——在真正实现 Pairwise 前，先判断 outcome feedback 是否值得付出额外串行深度

**适用项目**：`XYR222/BACE`  
**当前事实基线**：`handoff-refresh-20260828` 的 Exact Batch-ERV；历史 150-step BACE trace 用于诊断。  
**任务类型**：纯离线分析优先；第一阶段不需要 GPU。  
**优先级**：与 A0 同级。  
**目的**：

> 当前 Full Batch-ERV 在 root backbone 冻结后一次性决定全部 \(Q\) 条 branch。Pairwise \(K=2\) 会每执行两条 branch 后读取真实 outcome、更新 posterior，再规划剩余 branch。本审计在真正实现 \(K=2\) 前，量化这种 outcome feedback 到底会多频繁地改变后续实验，以及期望能增加多少 acquisition value。

---

# 1. 当前 Full Batch-ERV 的事实语义

对 task \(g\)，root/capacity correction 完成后：

1. root backbone 冻结；
2. anchor/action support 冻结；
3. 当前 natural-root posterior 冻结；
4. Exact Batch-ERV 以最终 quota \(Q_g\) 一次求解：
   \[
   \max_{\{l_z\}}
   \sum_z V_z(l_z)
   \]
   subject to：
   \[
   \sum_z l_z=Q_g,
   \qquad
   0\le l_z\le c_z.
   \]
5. 一次性冻结所有 branch requests；
6. physical execution 可以分 wave，但中间不重新规划。

因此 Full Batch 的统计特征是：

\[
\boxed{
B_{1:Q}
=
\pi_{\mathrm{acq}}(D_0)
}
\]

所有 branch 都只看初始 root evidence \(D_0\)。

---

# 2. Pairwise \(K=2\) 想改变什么

Pairwise：

\[
k=\min(2,Q_{\mathrm{remain}})
\]

每轮：

```text
当前 posterior D_r
        ↓
Exact solve k 条 branch
        ↓
执行这 k 条
        ↓
观察真实 outcomes
        ↓
更新对应 (anchor, action) Beta posterior
        ↓
重新设计剩余实验
```

数学上：

\[
(B_1,B_2)
=
\pi_{\mathrm{acq}}(D_0),
\]

观察：

\[
Y_1,Y_2,
\]

更新：

\[
D_1=D_0\cup\{Y_1,Y_2\},
\]

再：

\[
(B_3,B_4)
=
\pi_{\mathrm{acq}}(D_1).
\]

它的潜在收益来自：

\[
\boxed{
\text{真实 outcome 改变了剩余最优实验。}
}
\]

如果 outcome 几乎不改变剩余最优计划，则 Pairwise 只增加同步/串行成本，没有统计收益。

---

# 3. 为什么 A1 必须先离线做

当前 Pairwise 的工程代价包括：

- coordinator 状态机修改；
- branch 分批提交；
- 每 pair 后 posterior 更新；
- Exact solver 重跑；
- remaining quota / per-anchor used slots bookkeeping；
- task-local execution 与 GPU batching 的进一步压力。

所以不能仅凭“adaptive 通常更好”就实现。

先回答：

\[
\boxed{
P(
\text{pair-1 outcome 会改变后续 plan}
)
}
\]

以及：

\[
\boxed{
E[
\text{feedback 带来的额外 acquisition value}
]
}
\]

如果这两个量很小，就不值得立即投入。

---

# 4. 当前 trace 已经提供的关键数据

当前 Exact coordinator 在执行前的 `acquisition_rounds.jsonl` 中记录：

- `branch_quota`;
- `current_instance_prior_mean`;
- 每个 anchor：
  - frozen action posteriors；
  - `values_by_size`;
  - `delta_by_size`;
  - `capacity`;
  - `all_plans_by_size`;
  - tie-optimal local plans；
- 全局 Exact allocation；
- selected local plans；
- 具体 branch requests。

执行结束后的 `posterior_snapshots.jsonl` 记录：

- frozen acquisition posteriors；
- completed branch outcomes：
  - task；
  - anchor；
  - action；
  - success。

因此 A1 原则上不需要重新 rollout。

---

# 5. A1 必须分成两个层次

这是本分析最重要的设计。

不能把“拿 Full Batch 前两条真实结果再重规划”直接当成真正 Pairwise 的无偏 counterfactual。

原因：

真正 Pairwise 第一轮会重新求：

\[
Q=2
\]

的最优 plan。

它未必等于 Full Batch \(Q=4/5/6\) 计划中的“前两条”。

因此 A1 分为：

## A1.1：Realized Frozen-Plan Feedback Audit

使用 Full Batch 已经实际执行的前两条 branch outcome，回答：

> 如果我们至少允许在执行当前 Full Batch 的第一 pair 后重新规划，剩余 frozen plan 会不会改变？

这是**真实 outcome 条件下的反馈敏感性诊断**。

---

## A1.2：Model-Based Exact Pairwise Audit

完全根据 root posterior 和 Beta-Binomial predictive distribution，离线模拟真正的 \(K=2\) policy：

> 如果第一 pair 按 \(Q=2\) Exact planning 选择，并对所有可能 outcomes 做精确枚举，Pairwise 的期望 value 相比 Full Batch 能提升多少？

这是判断 Pairwise 是否值得实现的**主结果**。

A1.2 不依赖“历史 Full Batch 恰好执行了真正 Pairwise 会选择的第一对”，因此比 A1.1 更严格。

---

# 6. 建议新建脚本

```text
analysis/
  a1_pairwise_feedback_audit.py
  a1_trace_reader.py
  a1_pairwise_simulator.py
  a1_pairwise_plots.py
```

入口：

```bash
python analysis/a1_pairwise_feedback_audit.py \
  --artifact-root /path/to/bace_artifacts \
  --output-dir /path/to/a1_output \
  --pair-size 2 \
  --threshold 0.005 \
  --max-branches-per-anchor 2 \
  --phase-bounds 50 100 150
```

必须写出：

```text
resolved_analysis_config.json
data_coverage_report.json
```

---

# 7. Phase 0：Trace 完整性检查

逐 step 查找：

```text
step_xxxxxxxx/
  acquisition_rounds.jsonl
  posterior_snapshots.jsonl
  summary.json
```

对每个 task 检查：

1. `branch_quota` 是否存在；
2. frozen posteriors 是否存在；
3. selected allocation 是否存在；
4. requests 数是否等于 quota；
5. completed outcomes 数是否等于成功执行的 branch 数；
6. 每个 outcome 的 `(task, anchor, action)` 是否可映射回 frozen support；
7. trace 是否 JSON 完整。

输出：

`data_coverage_report.json`

必须包含：

```text
steps_total
steps_usable
tasks_total
tasks_q_ge3
tasks_usable_realized_audit
tasks_usable_model_based
corrupted_records
missing_outcomes
```

历史 trace 已知存在过截断风险，所以任何缺失都必须显式报告。

---

# 8. 分析对象

Pairwise 只有在：

\[
Q\ge3
\]

时才与 Full Batch 有实质差异。

因此主分析样本：

\[
\boxed{
\mathcal G_{\mathrm{A1}}
=
\{g:Q_g\ge3\}
}
\]

另保留：

```text
Q = 0
Q = 1
Q = 2
```

作为“Pairwise 无 adaptivity opportunity”的基数统计。

按训练阶段报告：

```text
early  = 1--50
middle = 51--100
late   = 101--150
```

并按 family 再拆一层。

---

# 9. A1.1：Realized Frozen-Plan Feedback Audit

## 9.1 定义 Full Batch 的物理逻辑顺序

必须从 trace 中读取 `requests` 的实际冻结顺序。

不要自己按 anchor ID 猜。

对于：

\[
Q=5
\]

例如：

```text
b0
b1
b2
b3
b4
```

第一 pair 定义为：

```text
b0, b1
```

剩余：

```text
b2, b3, b4
```

---

## 9.2 读取第一 pair 真实 outcome

从 `completed_branch_outcomes` 获取：

\[
Y_0,Y_1\in\{0,1\}.
\]

然后对对应：

\[
(z,u)
\]

posterior 做 Beta 更新：

success：

\[
(\alpha,\beta)
\to
(\alpha+1,\beta),
\]

failure：

\[
(\alpha,\beta)
\to
(\alpha,\beta+1).
\]

---

## 9.3 跟踪 per-anchor 已消耗 branch slot

当前总约束：

\[
L_{\max}=2.
\]

如果 pair 已在 anchor \(z\) 消耗 \(m_z\) 个 branch：

\[
L_z^{\mathrm{remain}}
=
L_{\max}-m_z.
\]

后续重规划不能再次超过该总上限。

---

## 9.4 重新计算剩余 plan

剩余：

\[
q_{\mathrm{remain}}=Q-2.
\]

在更新后的 posterior 上重新计算所有 anchor Exact design。

这里要输出两种 replan：

### Strict-threshold replan

沿用：

\[
\tau_{\mathrm{BERV}}=0.005.
\]

重新计算 information capacity。

记录：

\[
C_{\mathrm{remain}}.
\]

如果：

\[
C_{\mathrm{remain}}<q_{\mathrm{remain}},
\]

不要强行构造计划。

标记：

```text
ADAPTIVE_CAPACITY_SHORTFALL
```

这是非常重要的结果。

它说明真正 Pairwise 若要中途重规划，就必须额外定义：

- early stop；
- branch→root fallback；
- 或冻结 quota 后不再用 threshold。

A1 不替方法做这个决策，只量化发生频率。

---

### Structural-fixed-Q replan

为了纯粹测“feedback 会不会改变排序”，再做一个诊断版本：

- 保留 frozen root support；
- 保留每 anchor 总 \(L_{\max}\)；
- 不让中途 threshold 直接使 quota 不可行；
- 对所有剩余 structurally legal size 计算 BERV value；
- 固定执行剩余 \(q_{\mathrm{remain}}\) 个 slots。

该版本不是最终算法，只用于隔离：

\[
\boxed{\text{feedback-induced reallocation}}
\]

与：

\[
\boxed{\text{threshold-induced quota shrinkage}}.
\]

---

# 10. A1.1 指标

## 10.1 Allocation replan rate

将剩余 Full Batch allocation 记为：

\[
\mathbf l^{\mathrm{frozen}}.
\]

replan 后：

\[
\mathbf l^{\mathrm{replan}}.
\]

定义：

\[
\boxed{
\rho_{\mathrm{alloc-change}}
=
P[
\mathbf l^{\mathrm{replan}}
\neq
\mathbf l^{\mathrm{frozen}}
]
}
\]

---

## 10.2 Allocation L1 distance

\[
\boxed{
D_{L1}
=
\frac12
\sum_z
|l_z^{\mathrm{replan}}-l_z^{\mathrm{frozen}}|.
}
\]

它近似表示多少个 branch slots 被重新分配到了别的 anchor。

---

## 10.3 Action-plan change rate

即使 anchor allocation 不变，同一 anchor 下 action multiset 也可能改变。

例如：

```text
Full Batch remaining: (a2, a2)
Replan:               (a1, a2)
```

定义：

\[
\rho_{\mathrm{action-change}}.
\]

---

## 10.4 Adaptive capacity shortfall

\[
\boxed{
\rho_{\mathrm{shortfall}}
=
P[
C_{\mathrm{remain}}<q_{\mathrm{remain}}
]
}
\]

同时报告：

\[
q_{\mathrm{remain}}-C_{\mathrm{remain}}.
\]

这个指标直接决定 Pairwise implementation 是否需要重新考虑 frozen quota。

---

## 10.5 Realized replan value

在第一 pair outcome 已知后的 posterior \(D_1\) 上：

\[
V_{\mathrm{replan}}(D_1)
\]

与继续执行原 remaining plan 的：

\[
V_{\mathrm{frozen-rem}}(D_1)
\]

比较：

\[
\boxed{
\Delta V_{\mathrm{realized}}
=
V_{\mathrm{replan}}(D_1)
-
V_{\mathrm{frozen-rem}}(D_1).
}
\]

要求：

\[
\Delta V_{\mathrm{realized}}\ge0
\]

除数值 tie / 约束定义差异外。

---

# 11. A1.1 的局限必须写进报告

A1.1 的第一 pair 来自 Full Batch \(Q\) 计划，而不一定是真正 Pairwise \(Q=2\) 会选择的第一 pair。

因此它只能说明：

\[
\boxed{
\text{当前 frozen Full Batch plan 对真实 outcome 有多敏感}
}
\]

不能直接声称：

\[
\boxed{
\text{真正 K=2 会获得同样收益}.
}
\]

真正的判断依赖 A1.2。

---

# 12. A1.2：Model-Based Exact Pairwise Audit

这是 A1 的核心。

## 12.1 Pairwise policy 定义

对于当前 state：

```text
posterior D
remaining quota q
used slots by anchor m_z
```

令：

\[
k=\min(2,q).
\]

执行：

1. 基于当前 posterior 重建设计；
2. 在剩余 per-anchor slot 约束下求 size \(k\) 的 Exact global plan；
3. 得到 pair plan \(P_k\)；
4. 枚举该 pair 所有可能 branch outcome；
5. 对每种 outcome 更新 posterior；
6. 递归规划剩余 \(q-k\)。

注意：

> A1.2 模拟的是“每轮只对当前这 \(k\) 个 slots 做 myopic Exact Batch-ERV，然后 outcome 后再重规划”的 Pairwise policy，而不是求一个全局最优 POMDP acquisition policy。

这与我们准备实现的 \(K=2\) 逻辑一致。

---

# 13. Outcome 的精确枚举

当前 posterior 为 Beta。

若 pair 是：

```text
(a, b)
```

且两个 action 不同，则 outcome：

\[
(Y_a,Y_b)
\in
\{0,1\}^2.
\]

概率由各自 Beta-Bernoulli predictive 得到。

若 pair 是：

```text
(a, a)
```

则两次采样的 success count：

\[
K\in\{0,1,2\}
\]

应使用 Beta-Binomial：

\[
P(K=k)
=
\binom{2}{k}
\frac{B(\alpha+k,\beta+2-k)}
{B(\alpha,\beta)}.
\]

不要把同一 Beta latent probability 下的两次 posterior-predictive observation 简单当成固定 \(p=\alpha/(\alpha+\beta)\) 的独立 Bernoulli。

当前 `ExactBatchErvEngine` 已经实现 Beta-Binomial finite-sum，应尽量复用同一数学函数，避免两套实现。

---

# 14. Pairwise 递归模拟

定义：

\[
F(D,q,\mathbf m)
\]

为当前 Pairwise policy 从 posterior \(D\)、剩余 quota \(q\)、已消耗 anchor slots \(\mathbf m\) 出发的 expected final information gain。

若：

\[
q=0,
\]

则：

\[
F=0.
\]

否则：

\[
k=\min(2,q).
\]

求当前 size-\(k\) plan：

\[
P_k(D,\mathbf m).
\]

枚举 outcome \(y\)：

\[
p(y\mid D,P_k).
\]

更新：

\[
D'=\operatorname{Update}(D,P_k,y),
\]

\[
\mathbf m'=\mathbf m+\operatorname{Slots}(P_k).
\]

然后：

\[
F(D,q,\mathbf m)
=
\sum_y
p(y)
\left[
\Delta U(D\to D')
+
F(D',q-k,\mathbf m')
\right].
\]

其中：

\[
U(D)
=
\sum_z \max_u \mu_{z,u}(D).
\]

因此：

\[
\Delta U
=
U(D')-U(D).
\]

这样避免不同轮次 BERV 的重复计数。

---

# 15. Full Batch 的对照 value

对同一初始 posterior \(D_0\) 和 quota \(Q\)：

当前 Exact Full Batch 已经有：

\[
V_{\mathrm{Full}}(D_0,Q)
\]

即 frozen joint plan 的 expected improvement。

Pairwise：

\[
V_{\mathrm{Pair}}(D_0,Q)
=
F(D_0,Q,\mathbf 0).
\]

定义：

\[
\boxed{
\Delta V_{\mathrm{feedback}}
=
V_{\mathrm{Pair}}
-
V_{\mathrm{Full}}
}
\]

以及相对收益：

\[
\boxed{
r_{\mathrm{feedback}}
=
\frac{
V_{\mathrm{Pair}}-V_{\mathrm{Full}}
}{
V_{\mathrm{Full}}+\epsilon
}.
}
\]

理论上 adaptive policy 不应比其对应 frozen policy 更差，但由于：

- Pairwise 每轮采用 myopic \(k=2\) solver；
- tie handling；
- threshold/capacity policy；

实际实现可能出现轻微负值。

这些 case 必须保留并分析，不能强行 clamp。

---

# 16. 阈值问题：A1 必须同时报告两套定义

Pairwise 最大的潜在方法问题之一是：

> 第一 pair outcome 之后，原来足够的 information capacity 可能下降。

因此 A1.2 至少做两种模拟。

## Variant P-Fixed

**Fixed quota / structural cap**

- \(Q\) 冻结；
- root backbone 冻结；
- 每 anchor 总 branch 次数 \(\le L_{\max}\)；
- 后续仍执行满 \(Q\)；
- threshold 只用于初始 topology/capacity，不在中途强制停止。

用途：

\[
\boxed{\text{纯粹测 adaptivity / replanning 的价值}}
\]

---

## Variant P-Threshold

**Threshold-aware**

每 pair 后重新计算：

\[
\delta_z^{(m)}
\]

和 information capacity。

如果：

\[
C_{\mathrm{remain}}<q_{\mathrm{remain}},
\]

停止并记录 shortfall。

不要在 A1 中自行假设用 root 补齐。

用途：

\[
\boxed{
\text{暴露真正实现 Pairwise 时必须解决的 quota/fallback 问题}
}
\]

两套结果必须分开报告。

---

# 17. Tie handling

当前 Exact engine 存在 tie-optimal：

- global allocation；
- local plan；
- origin。

A1 只分析 acquisition value，不关心 origin replay。

建议：

## 主分析

使用 trace 中记录的：

```text
decision_task_key
policy update id
global seed
```

尽量复用当前 stable seed 逻辑。

如果历史 run 使用 `legacy_uuid`，不要跨 run 比较 request UUID。

---

## 敏感性分析

对于 tie 很多的 task：

- 报 `optimal_tie_count`；
- 可随机 10 个 tie-optimal plan 重算 A1.2；
- 报 feedback value 的范围。

若 tie-sensitive tasks 很少，可只记录不扩展。

---

# 18. A1 必须输出的 per-task 表

`pairwise_task_audit.parquet`

建议字段：

```text
step
phase
task_id
task_family

Q
num_anchors
initial_information_capacity
full_batch_value

first_pair_anchor_multiset
first_pair_action_multiset

# A1.1
realized_pair_outcome
remaining_q
replan_changed
allocation_l1_distance
action_plan_changed
remaining_capacity_after_realized_pair
adaptive_capacity_shortfall
realized_feedback_gain

# A1.2
pairwise_expected_value_fixed
pairwise_expected_gain_fixed
pairwise_relative_gain_fixed

pairwise_expected_value_threshold
expected_executed_branches_threshold
threshold_shortfall_probability

global_tie_count
notes
```

---

# 19. Phase-level 主指标

按：

```text
Early  : 1--50
Middle : 51--100
Late   : 101--150
```

至少报告：

| phase | tasks | Q>=3 ratio | replan rate | action change | shortfall rate | mean feedback gain | median feedback gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| early | | | | | | | |
| middle | | | | | | | |
| late | | | | | | | |

再按 task family 做同一张表。

---

# 20. 必须生成的图

## Figure A1-1：Pairwise opportunity over training

画：

\[
P(Q\ge3)
\]

随 step。

它说明 Pairwise 在哪个阶段有发挥空间。

---

## Figure A1-2：Realized replanning rate

按 step / rolling window：

\[
\rho_{\mathrm{alloc-change}}.
\]

---

## Figure A1-3：Expected feedback gain

\[
\Delta V_{\mathrm{feedback}}
\]

随 step。

这是判断 Pairwise 值不值得实现的核心图。

---

## Figure A1-4：Relative feedback gain

\[
r_{\mathrm{feedback}}.
\]

避免单纯因为后期 BERV 数值尺度不同产生错觉。

---

## Figure A1-5：Adaptive capacity shortfall

按 early/middle/late：

\[
P(C_{\mathrm{remain}}<q_{\mathrm{remain}}).
\]

如果这个比例很高，说明 Pairwise 不是简单把 Full Batch 拆成两批，而必须重新定义 branch quota 的中途语义。

---

## Figure A1-6：Family breakdown

比较六个 ALFWorld family 的：

```text
replan rate
feedback gain
shortfall rate
```

这可以判断 Pairwise 是否只对少数 family 有价值。

---

# 21. 推荐增加的一个关键分析：第几轮反馈最值钱

对：

\[
Q=5/6
\]

可能有：

```text
pair 1
pair 2
last singleton/pair
```

分别统计：

\[
\Delta V_1,
\Delta V_2,\ldots
\]

如果发现：

```text
第一 pair 后 feedback 很大
第二 pair 后几乎为 0
```

那么未来甚至可以考虑：

```text
2 + remaining full-batch
```

而不是全程 Pairwise。

A1 不直接提出主方法，但应把这个现象统计出来。

---

# 22. 如何判断“Pairwise 值得实现”

推荐以下门槛不是论文定理，而是工程决策标准。

## 强烈值得实现

如果 middle 或整体 \(Q\ge3\) 样本中：

```text
replan rate >= 25--30%
```

且：

```text
median relative feedback gain > 5%
```

或者：

```text
mean feedback gain 明显为正，
并集中在当前 BACE 表现关键阶段
```

则 Pairwise 很值得进入真实实现。

---

## 边缘

如果：

```text
replan rate 10--25%
```

且 feedback gain 较小：

Pairwise 可以作为 ablation，但不应抢占 credit-method 优先级。

---

## 暂缓

如果：

```text
replan rate < 10%
```

且：

\[
\Delta V_{\mathrm{feedback}}\approx0,
\]

则 Full Batch 已经足够接近 Pairwise。

不建议现在花工程时间。

---

# 23. Pairwise 的阶段性判断

现有 topology 已知：

```text
step 1--50   branch/task ≈ 0.69
step 51--100 branch/task ≈ 2.72
step 101--150 branch/task ≈ 3.72
```

因此在 A1 前的合理先验是：

- Early：Pairwise 机会少；
- Middle：最值得关注；
- Late：Q 大，但 posterior 可能更稳定，实际 feedback value 未知。

A1 的任务就是用数据验证或推翻这个先验。

不要把这个先验直接写成结果。

---

# 24. A1 的重要边界：离线 value 不等于最终 policy improvement

即使：

\[
V_{\mathrm{Pair}}>V_{\mathrm{Full}},
\]

只能说明：

> 在当前 Beta posterior + BERV objective 下，outcome-adaptive acquisition 更有效。

不能直接推出：

\[
\text{validation success}_{\mathrm{Pair}}
>
\text{validation success}_{\mathrm{Full}}.
\]

因为当前我们已经怀疑：

\[
\text{acquisition utility}
\neq
\text{optimization utility}.
\]

所以 A1 是：

\[
\boxed{\text{Pairwise 实现优先级筛选}}
\]

而不是最终效果证明。

---

# 25. 如果 A1 支持 Pairwise，下一步怎么做

只有 A1 明显支持后才进入真实实现。

推荐资源：

```text
H100：
实现 / debug / 2--5 step smoke

H20：
30--50 step continuation A/B
```

真实实验至少比较：

```text
K = Q   # Full Batch
K = 2   # Pairwise
```

之后有资源再：

```text
K = 1   # Sequential
```

所有版本必须共享：

- same topology；
- same root budget；
- same threshold；
- same credit estimator；
- same scheduler implementation；
- same checkpoint；
- same H20 count。

这样才能把差异归因给 feedback granularity。

---

# 26. 推荐输出目录

```text
analysis_outputs/
  A1_pairwise_feedback/
    resolved_analysis_config.json
    data_coverage_report.json

    pairwise_task_audit.parquet
    phase_summary.csv
    family_summary.csv
    q_summary.csv

    figures/
      pairwise_opportunity_over_steps.png
      realized_replan_rate.png
      expected_feedback_gain.png
      relative_feedback_gain.png
      adaptive_capacity_shortfall.png
      family_feedback_gain.png

    report.md
```

---

# 27. `report.md` 必须明确回答

1. 有多少 task 的 \(Q\ge3\)？
2. Pairwise 的机会主要出现在 early/middle/late 哪一段？
3. 第一 pair 的真实 outcome 多频繁改变剩余 Full Batch plan？
4. 改的是 anchor allocation 还是 action plan？
5. model-based Pairwise 的 expected feedback gain 多大？
6. relative gain 是否只因为 BERV 数值尺度变化？
7. threshold-aware replan 多频繁出现 capacity shortfall？
8. 哪些 family 最受益？
9. feedback 主要来自第一轮还是后续轮？
10. 是否值得真正实现 \(K=2\)？
11. 如果实现，是否必须同时设计 mid-branch quota/fallback 语义？

---

# 28. A1 的完成标准

满足以下条件才算 A1 完成：

1. 至少覆盖所有可读的 150-step Exact artifacts；
2. 明确报告 trace coverage；
3. A1.1 和 A1.2 不混淆；
4. 所有 \(Q<3\) task 被正确视为无 Pairwise adaptivity opportunity；
5. per-anchor \(L_{\max}=2\) 跨 pair 累计约束正确；
6. 同 action 重复采样使用 Beta-Binomial predictive；
7. Full Batch value 与 trace `global_optimal_value` 能 reproduction；
8. 输出 phase/family breakdown；
9. 输出 threshold shortfall；
10. 给出明确 Go / Hold / No-Go 结论。

---

# 29. 推荐实施顺序

```text
读取 Exact trace
       ↓
reproduce 每个 task 的 Full Batch design/value
       ↓
确认与 trace global_optimal_value 一致
       ↓
A1.1：用真实 Full Batch first-pair outcome 做 replan
       ↓
统计 realized replan/change/shortfall
       ↓
A1.2：实现 K=2 posterior-predictive 精确递归
       ↓
计算 expected feedback gain
       ↓
early/middle/late
       ↓
family breakdown
       ↓
决定是否真正实现 Pairwise
```

---

# 30. 最终目的

A1 最终要回答的不是：

> “Pairwise 听起来是不是比 Full Batch 更聪明？”

而是：

\[
\boxed{
\text{在我们当前真实训练分布和 posterior 上，
Full Batch 因为不读取 branch outcome，到底损失了多少实验价值？}
}
\]

只有这个量足够大，才值得接受 Pairwise 带来的：

- 更多 acquisition rounds；
- 更高串行依赖；
- 更复杂 scheduler；
- 潜在训练时间增长。

如果 A1 发现收益主要集中在 middle stage，也会给后续方法一个重要信号：

\[
\boxed{
\text{acquisition adaptivity 的价值本身可能是训练阶段相关的。}
}
