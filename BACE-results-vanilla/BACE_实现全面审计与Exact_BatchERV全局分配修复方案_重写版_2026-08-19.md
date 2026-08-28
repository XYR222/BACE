# BACE 实现全面审计与 Exact Batch-ERV 全局分配修复方案（重写版）

> **状态提示（2026-08-27）：历史修复记录。** 本文分析的 Cartesian global allocation 已由当前 `quota_aware_exact_dp` 替代。当前方法和实现入口见 [`docs/07_当前BACE方法规范.md`](../docs/07_当前BACE方法规范.md) 与 [`docs/08_代码库阅读路线.md`](../docs/08_代码库阅读路线.md)。

> 审计日期：2026-08-19
> 审计仓库：`XYR222/BACE`
> 审计基准：`main`，调试快照 commit `c9fee60c4ad10f9ed0aaf995f881fe9877af5482`
> 当前主训练路径：`algorithm.bace.variant=batch_erv_exact`
> 当前重点环境：ALFWorld / GiGPO / Qwen2.5-1.5B / 4×H100
> 本文定位：**修复 Exact Batch-ERV global allocation 的求解器实现，不修改 BACE 的方法定义。**

---

# 0. 最终结论

当前 4×H100 稳定性作业暴露的问题可以非常简单地概括为：

> **Batch-ERV 方法本身没有发生组合爆炸；发生组合爆炸的是 global allocation solver 的实现。**

当前方法已经先为每个 anchor 精确计算：

$$
V_z^{(0)},\qquad V_z^{(1)},\qquad V_z^{(2)},
$$

其中：

- $V_z^{(0)}=0$；
- $V_z^{(1)}$：给 anchor $z$ 分配 1 条 branch 时的最优 Exact Batch-ERV；
- $V_z^{(2)}$：给 anchor $z$ 分配 2 条 branches 时的最优 Exact Batch-ERV。

随后 global controller 要解决的只是：

$$
\boxed{
\max_{\{m_z\}}
\sum_z V_z^{(m_z)}
}
$$

subject to：

$$
\boxed{
\sum_z m_z=Q,
\qquad
m_z\in\{0,1,2\},
}
$$

并满足各 anchor 的 information-capacity 约束。

这是一个**小预算、多选项的精确离散资源分配问题**。在当前配置中：

$$
B=8,
\qquad
R_{\min}=2,
\qquad
Q\le 6,
\qquad
L_{\max}=2.
$$

因此它本来应该极其容易求解。

真正的错误是当前正式路径采用了类似：

```text
for allocation in CartesianProduct(anchor_1_choices,
                                    anchor_2_choices,
                                    ...,
                                    anchor_A_choices):
    if sum(allocation) == Q:
        evaluate(allocation)
```

的全笛卡尔积写法。

若每个 anchor 都有：

```text
0 branch
1 branch
2 branches
```

三种选择，则搜索空间近似：

$$
3^A.
$$

本次 step 3 的真实现场中：

- anchor 数约为 27；
- 有 branch capacity 的 anchor 约为 26；
- 当前 task 实际 branch quota 只有：
  $$Q=1;$$
- 但旧 solver 仍然遍历约：
  $$\boxed{1.69\times10^{12}}$$
  级别的组合状态。

所以程序并没有 OOM，也没有 deadlock，而是在 CPU 上执行一个极其巨大的 Python 枚举循环。

因此本次正式修复应定义为：

$$
\boxed{
\text{保留 Exact Batch-ERV}
+
\text{将 global Cartesian solver 替换为 quota-aware exact DP}
}
$$

而不是：

- 改成 greedy；
- 减少 anchor 数；
- 降低 Batch-ERV 精度；
- 退回 sequential ERV；
- 修改 branch quota；
- 修改 Bayesian controller。

此外，本次重写同时修正上一版审计中的一个判断：

> **当前 global/macro advantage 沿用 GiGPO 当前代码实现的 occurrence-weighted normalization，不再视为实现 bug。**

这是一个有意识的 baseline-alignment 选择：主实验保持 GiGPO 下游 credit estimator 和 PPO optimization 尽可能不变，只改变 BACE 的 rollout acquisition。之前设计的 leaf-uniform normalization 保留为重要消融，而不再作为必须修改的主路径。

因此当前真正阻塞正式训练的核心问题只有一个：

$$
\boxed{
\textbf{Exact Batch-ERV global allocation solver 的指数枚举。}
}
$$

---

# 1. 当前仓库整体是否符合 BACE 最新方案

基于当前 `batch_erv_exact` 主路径的检查，整体实现方向是正确的，不需要重构整套框架。

| 模块 | 当前最新方案 | 当前实现判断 |
|---|---|---|
| Pilot | 无独立 Pilot | ✅ 对齐 |
| Family competence | 使用 lagged natural-root history | ✅ 对齐 |
| 冷启动 prior | 低均值弱先验，当前推荐 $p_0=0.10,\kappa_0=2$ | ✅ production launcher 已采用 |
| History update | 全部 natural roots 更新，branch 不更新 | ✅ 对齐 |
| Initial topology | $\bar Q=\operatorname{round}[(B-R_{\min})q_c]$ | ✅ 对齐 |
| Current-instance posterior | roots 后更新，用于 local prior/诊断 | ✅ 对齐 |
| Anchor grouping | GiGPO-compatible exact pre-action observation grouping | ✅ 对齐 |
| 同轨迹 repeated occurrence | 保留 | ✅ 对齐 |
| Candidate action | natural observed decision-edge identity | ✅ 对齐 |
| Local posterior | Beta-Bernoulli | ✅ 对齐 |
| BERV | Beta-Binomial exact finite enumeration | ✅ 对齐 |
| Monte Carlo | 主方法不使用 | ✅ 对齐 |
| $L_{\max}$ | 2 | ✅ 对齐 |
| Capacity | 使用 $V^{(1)},V^{(2)}$ 的 marginal value | ✅ 对齐 |
| Capacity correction | 单向 $Q\leftarrow Q-1,R\leftarrow R+1$ | ✅ 对齐 |
| Root backbone | branch 前冻结 | ✅ 对齐 |
| Global allocation objective | 精确最大化总 Batch-ERV | ✅ 数学目标正确 |
| Global allocation solver | 应为小预算 exact solver | ❌ 当前 Cartesian 实现不可扩展 |
| Tie-breaking | seeded uniform over exact ties | ✅ 规则正确；实现需避免物化全部 ties |
| Replay | reset + prefix replay + strict validation | ✅ 对齐 |
| Branch origin | copied natural CoT-action edge | ✅ 对齐 |
| Mechanical prefix | 不训练 | ✅ 对齐 |
| Branch suffix | frozen $\pi_{old}$ fresh continuation | ✅ 对齐 |
| Branch planning | 一次性联合确定后并行执行 | ✅ 对齐 |
| Local credit | GiGPO occurrence-level 主版本 | ✅ 对齐 |
| Global/macro credit | 保持当前 GiGPO implementation semantics | ✅ **有意保持用于公平比较** |
| Behavior correction | 主方法无额外 behavior IS | ✅ 对齐 |
| PPO | standard $\pi_\theta/\pi_{old}$ | ✅ 对齐 |

因此当前工程状态应概括为：

$$
\boxed{
\text{方法框架正确}
+
\text{global solver 实现错误}
}
$$

而不是“方法需要重新设计”。

---

# 2. 本次 H100 现场说明了什么

本次 4×H100 作业最有价值的地方在于，它已经把故障范围缩得非常小。

## 2.1 已经正常完成的部分

step 1、step 2 已正常完成。

step 3 中，也已经正常完成：

1. natural-root rollout；
2. root outcomes 收集；
3. anchor 构造；
4. observed action support 构造；
5. local Beta posterior；
6. Exact Batch-ERV local design；
7. capacity 检查。

说明以下部分不是当前 blocker：

- LLM rollout；
- GPU generation；
- ALFWorld environment；
- replay；
- Beta posterior；
- local Exact Batch-ERV；
- capacity correction；
- artifact storage。

## 2.2 卡住的位置

程序进入：

```text
Exact Batch-ERV global branch allocation
```

之后：

- Slurm job 仍为 RUNNING；
- Python 进程仍存在；
- 没有 exception；
- GPU 长时间 0% utilization；
- 新的 acquisition / branch artifacts 不再产生。

这正符合一个 CPU-side exponential enumeration 的表现。

因此问题不是：

$$
\text{GPU slow}
$$

而是：

$$
\boxed{
\text{CPU 在枚举不必要的 global allocation states。}
}
$$

---

# 3. 当前 solver 为什么会指数爆炸

假设共有：

$$
A
$$

个 anchors。

对 anchor $z_i$，可分配：

$$
m_i\in\{0,1,2\}.
$$

如果直接先枚举完整 Cartesian product：

$$
(m_1,m_2,\ldots,m_A),
$$

则状态数量为：

$$
3^A.
$$

但我们的真正约束是：

$$
\sum_i m_i=Q.
$$

其中：

$$
Q\le6.
$$

这意味着绝大多数笛卡尔积状态从一开始就不可能成为合法解，却仍然被创建和遍历。

## 3.1 本次 $Q=1$ 的荒谬性

如果：

$$
A=26,
\qquad
Q=1,
$$

问题实际上只有：

```text
给 z1 一条 branch？
给 z2 一条 branch？
...
给 z26 一条 branch？
```

即本质上比较：

$$
V_{z_1}^{(1)},
V_{z_2}^{(1)},
\ldots,
V_{z_{26}}^{(1)}.
$$

理论有效候选约为 26 个。

但旧代码却先构造所有：

$$
0/1/2
$$

组合，再筛选总和为 1 的方案。

因此出现了“本来 26 个候选，代码却遍历万亿级状态”的现象。

---

# 4. 这不是 Batch-ERV 的理论组合爆炸

必须在代码注释、实验说明和论文中把这一点写清楚。

Local Batch-ERV 本身计算的是：

$$
V_z^{(m)}
=
\max_{|\mathcal B_z|=m}
\operatorname{BERV}(z,\mathcal B_z).
$$

当前：

$$
L_{\max}=2.
$$

所以每个 anchor 的 local plan 非常小。

例如 actions：

$$
\{a,b,c\},
$$

只需要检查：

### 一条 branch

$$
[a],\ [b],\ [c].
$$

### 两条 branches

$$
[a,a],\ [b,b],\ [c,c],\ [a,b],\ [a,c],\ [b,c].
$$

其 outcome 也通过 Beta-Binomial 精确有限枚举计算。

真正 global 层只是在已经计算好的：

$$
V_z^{(0)},V_z^{(1)},V_z^{(2)}
$$

之间分配一个非常小的 quota。

因此应明确：

$$
\boxed{
\text{Exact Batch-ERV 没有被证明不可扩展；}
}
$$

$$
\boxed{
\text{不可扩展的是当前 Cartesian global solver。}
}
$$

---

# 5. 正确的 global allocation 问题

设共有 anchors：

$$
z_1,\ldots,z_A.
$$

每个 anchor $i$ 有最大 capacity：

$$
c_i\le L_{\max}=2.
$$

已知：

$$
V_i^{(0)},\ldots,V_i^{(c_i)}.
$$

目标：

$$
\boxed{
\max_{m_1,\ldots,m_A}
\sum_{i=1}^{A}V_i^{(m_i)}
}
$$

subject to：

$$
\boxed{
\sum_{i=1}^{A}m_i=Q,
\qquad
0\le m_i\le c_i.
}
$$

这实际上是一个小容量 multiple-choice knapsack / resource allocation problem。

由于：

$$
Q\le6,
$$

最合适的正式 solver 是：

$$
\boxed{
\textbf{quota-aware exact dynamic programming}.
}
$$

---

# 6. Quota-Aware Exact DP

## 6.1 DP 状态

定义：

$$
D[i,q]
$$

表示：

> 只处理前 $i$ 个 anchors，并且恰好已经使用 $q$ 条 branches 时，可以获得的最大总 Batch-ERV。

其中：

$$
0\le i\le A,
\qquad
0\le q\le Q.
$$

初始化：

$$
D[0,0]=0,
$$

$$
D[0,q>0]= -\infty.
$$

---

## 6.2 状态转移

处理 anchor $i$ 时，可以给它：

$$
m\in\{0,1,\ldots,\min(c_i,q)\}
$$

条 branches。

因此：

$$
\boxed{
D[i,q]
=
\max_{m}
\left
\{
D[i-1,q-m]+V_i^{(m)}
\right\}.
}
$$

最终最优值：

$$
\boxed{
D[A,Q].
}
$$

这与原来 Cartesian exhaustive search 求的是**完全相同的精确最优值**。

---

# 7. 复杂度

每个 DP 状态最多尝试：

$$
L_{\max}+1
$$

个选择。

所以时间复杂度：

$$
\boxed{
O(AQ(L_{\max}+1)).
}
$$

在当前：

$$
L_{\max}=2,
\qquad
Q\le6,
$$

近似就是：

$$
O(AQ).
$$

空间复杂度如果保存完整表：

$$
O(AQ),
$$

如果只求最优值，可滚动为：

$$
O(Q).
$$

由于我们还需要 exact tie sampling 的回溯信息，推荐保留一个很小的 predecessor/count table，空间仍然可以忽略。

---

## 7.1 数值对比

### 旧方法

若：

$$
A=100,
$$

状态约：

$$
3^{100}.
$$

完全不可执行。

### 新方法

若：

$$
A=100,
\qquad
Q=6,
\qquad
L_{\max}=2,
$$

粗略 transition 数：

$$
100\times7\times3
\approx2100.
$$

与一次 LLM rollout 相比几乎可以忽略。

---

# 8. $Q=1$ 时的直观解释

当：

$$
Q=1,
$$

DP 本质上退化为：

$$
\boxed{
\max_i V_i^{(1)}.
}
$$

因此本次 step 3：

$$
A\approx26,
\qquad
Q=1,
$$

只需要比较约 26 个一条-branch local values。

可以选择：

1. 直接让 generic DP 处理；
2. 增加 `Q == 1` fast path。

推荐主逻辑仍然由 generic DP 定义，避免过多 special cases。

fast path 只作为微优化，不应成为算法 correctness 的不同分支。

---

# 9. 为什么 DP 仍然是 Exact Batch-ERV

这一点必须明确。

修改前：

```text
Exact local BERV
+ Cartesian exact global optimizer
```

修改后：

```text
Exact local BERV
+ DP exact global optimizer
```

两者解决的是同一个：

$$
\operatorname*{argmax}_{\sum_i m_i=Q}
\sum_i V_i^{(m_i)}.
$$

所以不是：

```text
旧：exact
新：approximate
```

而是：

```text
旧：低效的 exact solver
新：高效的 exact solver
```

因此方法名：

$$
\boxed{
\text{Exact Batch-ERV}
}
$$

完全不需要修改。

论文中的 Batch-ERV 理论也不需要重新推导。

---

# 10. Exact Tie-Breaking：不能只保存一个最优 predecessor

我们已经规定：

> 如果多个 global allocations 在 tie tolerance 下拥有相同最优 Batch-ERV，不加入 depth/count 等二级 heuristic，而是在所有最优完整 allocations 中均匀随机选择，并使用固定 seed 保证复现。

因此 DP 不仅要知道：

$$
\text{best value},
$$

还必须知道：

$$
\text{有多少条完整最优路径达到该值}.
$$

---

## 10.1 为什么简单“每个并列 predecessor 50/50”不正确

假设某 DP state 有两个并列 predecessor：

```text
A -> 后续对应 100 个完整最优 allocations
B -> 后续对应   1 个完整最优 allocation
```

如果当前简单：

```text
P(A)=1/2
P(B)=1/2
```

那么最终 101 个完整最优 allocations 并不等概率。

真正 uniform over complete allocations 要求：

$$
P(A)=\frac{100}{101},
$$

$$
P(B)=\frac{1}{101}.
$$

因此必须使用：

$$
\boxed{
\text{optimal path counting + count-weighted backtracking}.
}
$$

---

# 11. DP 中的 tie-count 定义

对每个状态 $(i,q)$ 保存：

```text
best_value[i][q]
optimal_count[i][q]
optimal_choices[i][q]
```

其中：

- `best_value`：该状态最优总 Batch-ERV；
- `optimal_count`：达到该最优值的完整前缀 allocation 数；
- `optimal_choices`：当前 anchor 哪些 $m$ 可以达到最优值，以及其 predecessor state。

初始化：

$$
C[0,0]=1.
$$

不可达状态：

$$
C[0,q>0]=0.
$$

对候选 transition：

$$
(i-1,q-m)\rightarrow(i,q),
$$

candidate value：

$$
X_m=D[i-1,q-m]+V_i^{(m)}.
$$

若：

$$
X_m>D[i,q]+\epsilon,
$$

则替换最优值，并：

$$
C[i,q]=C[i-1,q-m].
$$

若 $X_m$ 与当前最优值 tie，则：

$$
\boxed{
C[i,q]
\leftarrow
C[i,q]+C[i-1,q-m].
}
$$

这样最终：

$$
C[A,Q]
$$

就是完整 global tie-optimal allocation 数。

---

# 12. Seeded Uniform Backtracking

从：

$$
(A,Q)
$$

开始回溯。

假设当前 state 有多个 tie-optimal choices：

$$
m_1,\ldots,m_r.
$$

每个 choice 对应 predecessor count：

$$
c_j=C[i-1,q-m_j].
$$

则选择概率：

$$
\boxed{
P(m_j)
=
\frac{c_j}{\sum_k c_k}.
}
$$

选择后继续回溯 predecessor。

这样可以证明：

$$
\boxed{
\text{每一个完整 tie-optimal global allocation 都被等概率采样。}
}
$$

随机 seed 推荐继续使用：

```text
global_seed
+ policy_update_id
+ task_id
+ "global_allocation"
```

形成确定性 seeded RNG。

因此同一：

- checkpoint；
- task；
- update；
- seed；

一定得到同一个 global allocation。

---

# 13. Tie tolerance

继续沿用 Exact BERV 的数值 tie 定义：

$$
|x-y|
\le
\epsilon_{\mathrm{abs}}
+
\epsilon_{\mathrm{rel}}\max(|x|,|y|).
$$

主配置：

$$
\boxed{
\epsilon_{\mathrm{abs}}=10^{-12}
}
$$

$$
\boxed{
\epsilon_{\mathrm{rel}}=10^{-10}.
}
$$

注意：

- 不应重新使用 MC 时代的大 tolerance；
- 不应因为 DP 而改变 tie 定义；
- local plan tie 与 global allocation tie 应使用同样的数值比较工具函数。

---

# 14. 不能物化全部 global ties

即使 solver 改成 DP，还有第二个潜在指数风险：

> **不能把所有 tie-optimal allocations 重新展开成 Python list。**

例如：

$$
A=100,
\qquad
Q=6,
$$

如果 100 个 anchors 的第一条 branch value 完全相同，那么最优方案数量可以达到：

$$
\binom{100}{6}
=
1,192,052,400.
$$

约 11.9 亿个。

所以以下数据结构必须禁止出现在正式路径：

```python
tie_optimal_global_allocations = [
    alloc_1,
    alloc_2,
    ...
]
```

正式路径只保存：

```text
optimal_value
optimal_path_count
compressed predecessor choices
selected_allocation
```

最终只回溯采样一个 allocation。

---

# 15. 推荐正式数据结构

可以为 global solver 定义类似：

```python
@dataclass
class DPState:
    best_value: float
    optimal_count: int
    choices: tuple[DPChoice, ...]

@dataclass
class DPChoice:
    branches_for_current_anchor: int
    prev_quota: int
    prev_count: int
```

最终返回：

```python
@dataclass
class GlobalAllocationResult:
    optimal_value: float
    optimal_count: int
    selected_allocation: dict[str, int]
    solver: str = "quota_aware_exact_dp"
```

不要返回全部 ties。

---

# 16. 推荐正式伪代码

```text
function global_allocation_dp(anchor_designs, Q, rng):

    anchors = deterministic_order(anchor_designs)
    A = len(anchors)

    # dp[i][q] = best value after first i anchors using exactly q branches
    initialize all states as unreachable
    dp[0][0].best_value = 0
    dp[0][0].optimal_count = 1

    for i in 1..A:
        z = anchors[i-1]
        cap = min(anchor_designs[z].capacity, L_max, Q)

        for q in 0..Q:
            candidates = []

            for m in 0..min(cap, q):
                prev = dp[i-1][q-m]
                if prev unreachable:
                    continue

                value = prev.best_value + V_z[m]
                candidates.append((m, q-m, value, prev.optimal_count))

            best = max value among candidates

            tie_choices = all candidates numerically tied with best

            dp[i][q].best_value = best
            dp[i][q].optimal_count = sum(
                prev_count for each tied candidate
            )
            dp[i][q].choices = compressed tied predecessor choices

    if dp[A][Q] unreachable:
        raise InfeasibleAllocation

    # Exact uniform sampling over all globally optimal allocations
    allocation = {}
    i = A
    q = Q

    while i > 0:
        choices = dp[i][q].choices
        sample one choice with probability
            choice.prev_count / sum(prev_count)
        allocation[anchor_i] = choice.m
        q = choice.prev_quota
        i -= 1

    assert sum(allocation.values()) == Q

    return:
        optimal_value = dp[A][Q].best_value
        optimal_count = dp[A][Q].optimal_count
        selected_allocation = allocation
```

---

# 17. Anchor 顺序必须确定性

DP 的最优值与 anchor 顺序无关，但 seeded backtracking 的可复现性依赖 deterministic input ordering。

所以进入 DP 前必须：

```text
anchors = sorted(anchor_ids, key=stable_key)
```

或者使用 repository 中稳定的 anchor key serialization。

不要依赖：

- Python set iteration order；
- 临时 UUID 顺序；
- multiprocessing arrival order。

否则同一个 seed 在不同运行中可能得到不同 tie allocation。

---

# 18. 保留旧 Cartesian solver，但只能作为 reference oracle

旧 solver 不建议直接删除。

它有一个很重要的价值：

> **在很小规模上，它可以作为 DP 的 correctness oracle。**

建议重命名：

```python
_global_allocations_cartesian_reference(...)
```

并明确禁止生产调用。

---

## 18.1 Reference solver hard guard

先计算理论 Cartesian state count：

$$
N_{cart}
=
\prod_i(c_i+1).
$$

若：

$$
N_{cart}>N_{max}^{reference},
$$

直接报错。

推荐：

```text
N_max_reference = 1_000_000
```

异常信息示例：

```text
Cartesian reference solver disabled:
state space = 1.69e12 > 1e6.
Use quota-aware exact DP.
```

这样即使后续代码重构误调用 reference，也不会再合法地“卡死”。

---

# 19. DP 与 brute-force 的 correctness 单测

在小规模：

$$
A\le6,
\qquad
Q\le4,
$$

随机生成：

$$
V_i^{(m)}.
$$

同时运行：

```text
Cartesian reference
DP solver
```

必须检查：

## 19.1 最优值一致

$$
V_{DP}^*\approx V_{cart}^*.
$$

## 19.2 最优 allocation set 的计数一致

$$
N_{tie}^{DP}=N_{tie}^{cart}.
$$

## 19.3 Selected allocation 必须属于 brute-force optimal set

对多个 seeds：

```text
selected_DP(seed) in exact_optimal_set_reference
```

## 19.4 小规模频率测试

若 tie-optimal set 很小，例如 3 个方案，运行大量不同 deterministic seeds，采样频率应近似均匀。

这不是训练测试，而是 solver statistical correctness 测试。

---

# 20. 必须加入本次事故 regression fixture

本次 H100 问题不能只用 synthetic toy test 覆盖。

应该把 step 3 的真实结构抽象成 regression fixture：

```text
num_anchors ≈ 27
num_positive_capacity_anchors ≈ 26
Q = 1
L_max = 2
```

最好保存当时每个 anchor 的：

```text
capacity
V^(0)
V^(1)
V^(2)
```

不需要保存模型 tensor，只保存 solver 输入即可。

测试要求：

1. DP 瞬间完成；
2. `sum(selected_allocation.values()) == 1`；
3. selected anchor 必须属于最大 $V^{(1)}$ tie set；
4. `optimal_count` 与直接扫描最大值的 tie count 一致；
5. 不生成 Cartesian states；
6. solver runtime 应远低于 1 秒。

这个 fixture 是防止本次问题再次出现的最重要 regression test。

---

# 21. 还应加入哪些 solver 测试

## Test 1：$Q=0$

输入任意 anchors：

$$
Q=0.
$$

必须直接返回：

```text
selected_allocation = all zero / empty
optimal_value = 0
optimal_count = 1
```

不进入无意义 DP。

---

## Test 2：$Q=1,A=100$

验证：

- runtime 近似线性；
- 结果等价于扫描所有 $V_i^{(1)}$；
- large tie 不物化。

---

## Test 3：$Q=6,A=100$

验证当前最大正式 quota。

应在毫秒到很小的 CPU 时间内完成。

---

## Test 4：Mixed capacity

例如：

```text
z1 capacity=0
z2 capacity=1
z3 capacity=2
z4 capacity=1
...
```

检查不可用的 $m$ 不会进入 DP transition。

---

## Test 5：Infeasible quota

若：

$$
\sum_i c_i<Q,
$$

理论上正常流程的 capacity correction 应已经避免这一情况。

但 solver 仍应 defensive fail：

```text
InfeasibleGlobalAllocation
```

而不是静默返回不足 quota 的 allocation。

---

## Test 6：Huge tie count

构造：

$$
A=100,
\qquad
Q=6,
$$

且所有一条-branch marginal values 相同。

要求：

- `optimal_count` 可以是大整数；
- 内存不随 tie count 增长；
- 不创建 11.9 亿对象；
- seeded backtracking 返回合法 allocation。

Python `int` 可直接支持任意精度 path count。

---

# 22. Artifact / Diagnostics 也必须同步压缩

当前正式 artifact 不应再记录：

```text
all_tie_optimal_global_allocations
```

建议记录：

```json
{
  "solver": "quota_aware_exact_dp",
  "num_anchors": 26,
  "branch_quota": 1,
  "dp_state_count": 54,
  "optimal_value": 0.0412,
  "global_tie_count": 3,
  "selected_allocation": {
    "anchor_x": 1
  },
  "solver_wall_time_ms": 0.31
}
```

对每个 selected anchor 再单独保存：

```text
selected local optimal plan
local plan tie count
selected actions
selected origin IDs
```

这样 trace 足够审计，但不会重新制造组合爆炸。

---

# 23. Runtime guard

建议为 global allocator 增加至少以下 diagnostics：

```text
num_anchors
total_information_capacity
branch_quota
dp_reachable_state_count
global_optimal_count
solver_wall_time_ms
```

如果：

```text
solver_wall_time_ms > warning_threshold
```

输出 warning。

在我们当前：

$$
Q\le6,
L_{\max}=2
$$

的情况下，正常 DP 不应该成为秒级瓶颈。

因此如果 future run 中该阶段超过例如数百毫秒到 1 秒，应优先视为实现回归。

---

# 24. 是否需要 No-Progress Watchdog

可以增加，但它是 secondary protection，不是本次主修复。

例如 coordinator 在进入 global allocation 前记录：

```text
allocation_start_time
```

若超过一个明显异常的 hard timeout，可打印：

```text
num anchors
Q
capacity distribution
solver name
current DP layer
```

但不要依赖 watchdog 来“解决”指数枚举。

正确措施仍然是 DP。

---

# 25. Advantage Normalization：本次不再作为 Bug 修复

上一版审计把：

```text
当前 GiGPO macro normalization 是 physical-occurrence weighted
```

与：

```text
我们之前方法文档中的 terminal-leaf-uniform normalization
```

之间的不一致列为 P1。

在进一步确认当前 GiGPO 实现本身也使用这一 occurrence-weighted 语义之后，本版调整结论：

$$
\boxed{
\textbf{主实验不修改该行为。}
}
$$

原因是：

> BACE 的核心贡献是 rollout acquisition / topology，而不是重新设计 GiGPO 的 downstream advantage estimator。

为了尽可能公平地与当前可运行 GiGPO baseline 比较，主版本应保持：

```text
same GiGPO advantage implementation
same PPO objective
same gamma
same step_advantage_w
same normalization mode
```

只改变：

```text
how rollout evidence is acquired
```

---

# 26. 当前 macro normalization 的真实语义

设同一 task $g$ 中的训练 decision occurrences：

$$
\mathcal I_g.
$$

每个 occurrence $i$ 属于 terminal trajectory/leaf：

$$
\ell(i),
$$

其 terminal return：

$$
R_{\ell(i)}.
$$

当前 GiGPO-compatible实现相当于对 occurrence rows 统计：

$$
\mu_g^E
=
\frac{1}{|\mathcal I_g|}
\sum_{i\in\mathcal I_g}R_{\ell(i)}.
$$

因此一条拥有更多 physical trainable occurrences 的 trajectory 会对 macro baseline 贡献更多统计权重。

这个语义未必是 tree 数据下唯一最理想的选择，但它是：

$$
\boxed{
\text{当前 GiGPO implementation-aligned choice}.
}
$$

---

# 27. 为什么主版本保持 GiGPO occurrence-weighting 更合适

## 27.1 更干净的控制变量

如果 baseline 用 GiGPO occurrence-weighted，而 BACE 改成 leaf-uniform，那么 BACE 同时修改：

```text
1. rollout acquisition
2. macro advantage normalization
```

性能提升将难以归因。

保持 GiGPO 语义后：

```text
GiGPO baseline:
    natural rollout data
    + GiGPO optimizer

BACE:
    actively acquired rollout data
    + same GiGPO optimizer
```

实验更能回答：

> 主动构造局部信用证据本身是否有效？

---

## 27.2 论文叙事更强

推荐写法：

> BACE modifies the rollout acquisition topology while retaining the downstream GiGPO credit assignment and PPO optimization used by the implementation baseline.

这比：

> BACE 同时修改数据采集和 global advantage weighting

更容易证明核心 novelty。

---

## 27.3 不代表 occurrence weighting 在理论上唯一最优

仍需诚实说明：

BACE branch 改变：

- trajectory length distribution；
- root/branch occurrence counts；
- anchor collision distribution；
- suffix length。

所以“同一个 estimator”并不意味着 effective weighting distribution 与 GiGPO 一样。

我们能保证的是：

$$
\boxed{
\text{same estimator applied to differently acquired evidence}.
}
$$

这正是 BACE 方法应产生的变化。

---

# 28. Leaf-Uniform 如何处理

之前设计的 terminal-leaf-uniform normalization 不应删除。

将其重新定位为：

$$
\boxed{
\textbf{关键 optimizer ablation，而不是主方法必需组件。}
}
$$

建议实验至少比较：

1. **GiGPO-compatible occurrence-weighted macro normalization（主）**；
2. **leaf-uniform macro normalization**；
3. 如有必要，**lineage-balanced normalization**。

如果主配置已经胜过 GiGPO：

> 说明收益来自 BACE acquisition，而不是 advantage normalization 改动。

如果 leaf-uniform 再进一步提升：

> 可作为 orthogonal tree-aware optimization enhancement 单独报告。

因此当前代码**不需要为了修复本次 H100 问题而修改 advantage.py 的这一行为**。

---

# 29. 文档必须同步修正的地方

为了避免“代码正确、文档却说错”，建议更新当前最终方法文档中的以下表述。

## 29.1 删除/修改

不要再把主方法写成：

```text
all terminal leaves are uniformly weighted in the global baseline
```

如果当前主实验使用 GiGPO-compatible implementation semantics。

## 29.2 推荐主表述

写成：

> For the downstream optimization, we retain the macro- and step-level advantage semantics of the GiGPO implementation used as our baseline. BACE changes how rollout evidence is allocated and acquired, while the optimizer is kept fixed for controlled comparison.

## 29.3 Leaf-uniform

移到：

```text
Optimization Ablations
```

中。

这样：

```text
方法文档
代码实现
实验配置
论文比较口径
```

保持一致。

---

# 30. 本次不应该改什么

为了避免修复过程中扩大 scope，本次 solver patch 不应顺便改以下部分。

## 30.1 不改 Exact local BERV

保持：

$$
\operatorname{BERV}
=
E[\max \text{ future posterior mean}]
-
\max \text{ current posterior mean}.
$$

继续 Beta-Binomial exact finite enumeration。

---

## 30.2 不改 capacity 定义

继续使用：

$$
\delta_z^{(1)}=V_z^{(1)},
$$

$$
\delta_z^{(2)}=V_z^{(2)}-V_z^{(1)},
$$

以及：

$$
\tau_{\mathrm{BERV}}
$$

判断有效 information slots。

---

## 30.3 不改 topology controller

继续：

$$
q_c=P(\phi_c>\tau_{comp}),
$$

$$
\bar Q=\operatorname{round}[(B-R_{\min})q_c].
$$

---

## 30.4 不改 cold-start prior

继续当前：

$$
p_0=0.10,
\qquad
\kappa_0=2,
$$

即：

$$
\operatorname{Beta}(0.2,1.8).
$$

---

## 30.5 不改 branch parallelism

仍然：

```text
freeze roots
-> jointly choose all branches
-> parallel branch rollout
```

不退回 sequential posterior-feedback branching。

---

## 30.6 不改 tie objective

仍然：

```text
BERV is the only acquisition objective
```

完全并列时才 seeded-uniform。

不加入：

- anchor depth；
- occurrence count；
- remaining horizon；
- lexicographic score；

作为二级信息价值 criterion。

---

# 31. 推荐代码修改范围

本次 patch 应尽可能小。

主要修改：

```text
verl-agent-src/recipe/bace_gigpo/batch_erv.py
```

如果 diagnostics 逻辑在 coordinator：

```text
verl-agent-src/recipe/bace_gigpo/coordinator.py
```

同步修改。

测试：

```text
verl-agent-src/tests/bace_gigpo/test_batch_erv_exact.py
```

新增 performance / regression test 文件也可以。

Trace schema 如需要记录新字段，再修改：

```text
artifacts / validate_trace
```

但不要借此重构 replay、advantage 或 rollout collector。

---

# 32. 推荐修改后的函数接口

例如正式接口可以变为：

```python
def global_allocation(
    designs,
    branch_quota,
    *,
    seed_context,
) -> GlobalAllocationResult:
    return _global_allocation_exact_dp(...)
```

旧接口：

```python
def _global_allocations_cartesian_reference(...):
    ...
```

只在 tests 中调用。

如果现有上层代码依赖：

```text
(value, ties)
```

建议不要继续让 `ties` 表示“全部 allocation objects”。

可以改为：

```text
optimal_value
optimal_count
selected_allocation
```

一次性解决 tie materialization 风险。

---

# 33. 推荐实现顺序

## Phase 1：先写 reference-equivalence tests

在不删旧代码前，先固定旧 brute-force 的小规模输出。

加入若干 deterministic fixtures。

目标：新 DP 写完后有 oracle 对比。

---

## Phase 2：实现 exact DP value solver

先只保证：

$$
V_{DP}^*=V_{reference}^*.
$$

暂时可选择单一 deterministic predecessor 调试。

---

## Phase 3：加入 exact tie counts

验证：

$$
N_{tie}^{DP}=N_{tie}^{reference}.
$$

---

## Phase 4：加入 count-weighted seeded backtracking

验证 selected allocation：

- 始终是全局最优；
- 同 seed 稳定；
- 多 seed 统计近似均匀。

---

## Phase 5：替换 production path

正式训练只调用：

```text
quota_aware_exact_dp
```

---

## Phase 6：给 reference solver 加 hard guard

防止未来误调用。

---

## Phase 7：压缩 artifacts

禁止保存全部 tie allocations。

---

# 34. 修复后 CPU 测试验收

进入 H100 前，必须满足：

### A. 原 BACE unit tests 全部通过

特别是：

- exact one-step ERV；
- exact two-sample BERV；
- local ties；
- global ties；
- no-pilot topology；
- capacity correction；
- frozen batch branch plan；
- replay identity。

### B. DP vs Cartesian reference 全部一致

至少覆盖：

```text
A=1..6
Q=0..4
capacity=0/1/2 mixed
random values
tied values
```

### C. Step-3 regression fixture

要求：

```text
A≈27
Q=1
```

瞬间完成。

### D. Large synthetic

至少：

```text
A=100
Q=6
```

应快速完成且内存稳定。

---

# 35. 修复后 4×H100 三步稳定性验收

CPU tests 通过后，不要直接 step 150。

先重新执行：

```text
4×H100
3 training steps
```

必须看到 step 3 完整结束。

验收项：

## 35.1 Training progress

必须出现：

```text
training/global_step: 3
```

---

## 35.2 Acquisition artifacts

step 3 必须产生完整：

```text
topology
acquisition/global allocation
selected branches
replay attempts
branch results
trainable occurrences
summary
```

---

## 35.3 Global allocator diagnostics

应看到类似：

```text
solver=quota_aware_exact_dp
num_anchors=26
Q=1
solver_wall_time_ms << rollout time
```

绝不能再出现 allocator 时间接近分钟级。

---

## 35.4 GPU behavior

Global DP 期间 GPU 可以短暂空闲，因为该阶段本来是 CPU controller。

但只应是极短时间。

之后必须进入 branch rollout / training，而不是长期 0%。

---

## 35.5 Trace validator

原有 trace validation 必须继续通过。

特别检查：

- $R+Q=B$；
- selected allocation 总数恰好为 $Q$；
- branch origin 来自合法 natural edge；
- replay validation 通过；
- failed branches 不进入 training；
- copied origin identity/logprobs 正确。

---

# 36. 通过三步后再逐级放大

推荐：

```text
3 steps
-> 20 steps
-> 50 steps
-> 150 steps
```

不要修完 solver 就立刻全量长训。

20-step 阶段重点统计：

```text
anchor count distribution
Q distribution
DP runtime distribution
global tie count distribution
capacity correction frequency
branch rollout wall-clock
GPU utilization
```

如果这些稳定，再开放正式主实验。

---

# 37. 推荐新增的性能日志

为了以后快速定位 controller bottleneck，建议每个 task 记录：

```text
bace/global_alloc/num_anchors
bace/global_alloc/quota
bace/global_alloc/total_capacity
bace/global_alloc/reachable_dp_states
bace/global_alloc/optimal_tie_count
bace/global_alloc/solver_time_ms
```

aggregate 记录：

```text
mean
p50
p95
max
```

如果 future benchmark 的 anchor 数显著增加，也可以直接观察 DP 是否仍处于可忽略成本范围。

---

# 38. 论文中的实现描述建议

论文不需要讲本次 Cartesian bug。

方法/实现部分只需说明：

> Given the exact local Batch-ERV values $V_z^{(m)}$ for allocating $m\in\{0,1,2\}$ branches to each anchor, we solve the global branch-budget allocation exactly with a quota-aware dynamic program. Since the per-task branch quota is at most six in our main setting, the resulting optimization has complexity $O(AQL_{\max})$ and is negligible relative to language-model rollouts.

中文：

> 在得到每个 anchor 分配 $m\in\{0,1,2\}$ 条 branches 时的精确局部 Batch-ERV 后，我们使用 quota-aware 动态规划精确求解全局 branch budget 分配。由于主设置中每个任务的 branch quota 不超过 6，该优化的复杂度为 $O(AQL_{\max})$，相对于语言模型 rollout 的开销可以忽略。

Tie 可以在附录说明：

> Numerically tied global optima are sampled uniformly using exact optimal-path counts and a seeded backtracking procedure.

---

# 39. 对“Exact”的准确表述

修复后我们可以更有底气使用：

$$
\boxed{
\textbf{Exact Batch-ERV}
}
$$

因为有两层 exactness：

## Local exactness

Beta-Binomial finite outcome enumeration：

$$
\operatorname{BERV}(z,\mathcal B)
$$

精确计算。

## Global exactness

DP 精确求：

$$
\operatorname*{argmax}_{\sum_zm_z=Q}
\sum_zV_z^{(m_z)}.
$$

没有 greedy approximation。

所以方法可以描述为：

$$
\boxed{
\text{exact local Bayesian value computation}
+
\text{exact global budget allocation}.
}
$$

---

# 40. 修复后完整主流程

最终训练流程保持不变：

```text
lagged family natural-root history
        ↓
family competence posterior q_c
        ↓
initial root / branch quota
        ↓
parallel natural roots
        ↓
exact GiGPO-compatible anchor grouping
        ↓
local Beta posterior
        ↓
Exact local Batch-ERV
        ↓
information-capacity correction
        ↓
freeze root backbone
        ↓
【quota-aware exact DP global allocation】
        ↓
seeded uniform selection among exact global optima
        ↓
select concrete natural origins
        ↓
parallel replay + branch suffix rollout
        ↓
GiGPO-compatible macro + local credit
        ↓
unified PPO
        ↓
all natural roots update next-batch family history
```

唯一被替换的是：

```text
Cartesian exhaustive global allocator
```

变成：

```text
quota-aware exact DP allocator
```

---

# 41. 最终修复清单

## 必须修复（P0）

- [ ] 将 production `global_allocations()` 从 Cartesian exhaustive enumeration 改为 quota-aware exact DP。
- [ ] DP 只保留 $q\in[0,Q]$ 的可达状态。
- [ ] 保留每个 DP state 的 exact optimal path count。
- [ ] 使用 count-weighted seeded backtracking，保证 uniform over complete global optima。
- [ ] 不再物化全部 tie-optimal allocations。
- [ ] artifact 只记录 tie count + selected allocation。
- [ ] 旧 Cartesian solver 改名为 reference-only。
- [ ] reference solver 加理论状态数 hard guard。
- [ ] 加入 step-3 真实规模 regression fixture。
- [ ] 加入 $A=100,Q=6$ 性能测试。

## 建议同步（P1 工程保障）

- [ ] 记录 global allocator runtime。
- [ ] 记录 DP reachable state count。
- [ ] 记录 global optimal tie count。
- [ ] anchor input order 固定化。
- [ ] 增加 infeasible allocation defensive assertion。
- [ ] 更新 trace validator 检查 selected allocation quota。

## 不修改主路径

- [x] 不改 Exact BERV 公式。
- [x] 不改 Beta-Binomial local computation。
- [x] 不改 no-pilot topology。
- [x] 不改 cold-start `Beta(0.2,1.8)`。
- [x] 不改 Batch-ERV capacity correction。
- [x] 不改 observed-edge replay。
- [x] 不改 branch parallel rollout。
- [x] 不改 standard PPO ratio。
- [x] 不为本次 bug 修改 GiGPO-compatible occurrence-weighted advantage。

---

# 42. 文档修改清单

当前方法文档应同步两项。

## 42.1 Global allocation

把模糊的：

```text
枚举 / DP / 小整数规划均可
```

正式收敛为：

$$
\boxed{
\text{主实现使用 quota-aware exact dynamic programming。}
}
$$

理由不是方法必须 DP，而是：

- exact；
- 简单；
- $Q$ 极小；
- 稳定；
- 避免实现误用 Cartesian product。

## 42.2 Global advantage

把之前的：

```text
leaf-uniform 是主版本
```

改成：

```text
主实验保持 GiGPO implementation-compatible macro normalization；
leaf-uniform 作为 tree-aware optimization ablation。
```

这样当前实现和论文口径一致。

---

# 43. 对当前 BACE 仓库的最终评价

## 43.1 方法层

当前 Exact Batch-ERV 主方案是可行的。

本次事故不构成对以下核心思想的反例：

- competence-guided root/branch allocation；
- active local credit evidence acquisition；
- exact Bayesian BERV；
- batch branch planning；
- observed-edge continuation resampling；
- parallel branch rollout。

---

## 43.2 工程层

核心架构也不需要推翻。

当前最明显的问题是：

> 一个本应按小 quota 求解的 global resource allocation，被实现成了对所有 anchors 选择空间的全 Cartesian product。

这是一个典型的 complexity bug。

---

## 43.3 实验层

修复前不应继续正式长训，因为随着训练推进 repeated anchors 很可能越来越多，旧 solver 会更频繁触发指数爆炸。

修复后，global allocation 复杂度近似线性于 anchor 数：

$$
O(AQ L_{\max}),
$$

而：

$$
Q\le6,
L_{\max}=2.
$$

因此它不应再成为系统瓶颈。

---

# 44. 一句话版本

> **本次问题不是 Exact Batch-ERV 在 anchor 多时不可计算，而是代码错误地先枚举所有 anchor 的 0/1/2 branch 笛卡尔积，再筛选总 quota；正确做法是在不改变任何 Batch-ERV 数学定义的前提下，用 quota-aware exact DP 直接求解同一个全局最优 branch allocation，并通过最优路径计数实现 seeded uniform tie-breaking。**

最终修复可概括为：

$$
\boxed{
\text{Exact local Batch-ERV}
+
\text{quota-aware exact global DP}
+
\text{count-weighted uniform tie backtracking}
}
$$

并且主实验下游 optimizer 继续保持：

$$
\boxed{
\text{GiGPO implementation-compatible credit assignment}
}
$$

以确保与 GiGPO baseline 的比较尽可能干净。
