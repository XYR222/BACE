# A1：GiGPO Action-Mean Local Credit 实验方案

> 目的：在**不引入 BACE rollout topology、branch、ERV 或任何主动采样机制**的前提下，单独判断“按动作聚合 local credit”是否本身就是 GiGPO 的通用改进。
>
> 当前代码基准：`handoff-refresh-20260828`。当前 BACE 路径已实现 `local_credit_mode ∈ {occurrence, action_mean}`；A1 的关键是把同一 local-credit 逻辑最小化地用于纯 GiGPO，而不是为了使用该实现而启用 BACE。

---

## 1. 为什么先做 A1

BACE 当前的主要怀疑之一，是主动 branch 会在同一个 anchor-action 对 `(z,u)` 上产生多次 continuation outcome，而原始 GiGPO 的 occurrence-level local credit 会把这些不同 outcome 分别归因给当前动作。这可能导致同一个动作同时得到较强正、负 local advantage。

但在判断这是 BACE 特有问题之前，必须先回答：

> **即使没有 BACE，只在原版 GiGPO 的自然 rollout 上，把 occurrence-level local credit 改成 action-mean，是否已经能提升效果？**

如果 GiGPO 本身就明显受益，则 action mean 是一个通用 optimizer improvement；如果 GiGPO 基本不变而 BACE 显著受益，则更支持“主动重复采样使 action-consistent credit 变得必要”的 BACE-specific 机制解释。

因此 A1 是 A2 的必要前置实验。

---

## 2. 当前 GiGPO local credit 与 Action Mean 的区别

### 2.1 Occurrence-level GiGPO

对同一 task 下相同 anchor state `z` 的所有 occurrence：

\[
\mathcal I(z)=\{i:s_i=z\}.
\]

每个 occurrence 有 step return / reward-to-go `G_i`。原始 local advantage 为：

\[
A_i^{S,\mathrm{occ}}
=
\frac{G_i-\mu_z}{\sigma_z+\epsilon},
\]

其中：

\[
\mu_z=\frac1{|\mathcal I(z)|}\sum_iG_i.
\]

同一动作的不同 occurrence 可以得到不同 local advantage。

### 2.2 Action-Mean Local Credit

对同一 anchor 内相同 canonical action `u`：

\[
\mathcal I(z,u)=\{i:s_i=z,a_i=u\}.
\]

先计算动作平均 return：

\[
\bar G_{z,u}=\frac1{|\mathcal I(z,u)|}\sum_{i\in\mathcal I(z,u)}G_i.
\]

再使用**与原 GiGPO 相同的 anchor group mean/std**：

\[
A_{z,u}^{S,\mathrm{act}}
=
\frac{\bar G_{z,u}-\mu_z}{\sigma_z+\epsilon}.
\]

所有同动作 occurrence 共享该 local advantage：

\[
A_i^S=A_{z,u}^{S,\mathrm{act}},\quad i\in\mathcal I(z,u).
\]

最终仍保持 GiGPO 结构：

\[
A_i=A_i^{\mathrm{macro}}+\omega A_i^S.
\]

**只聚合 local term；绝不能把 macro/trajectory advantage 一并按动作平均。**

---

## 3. 直观例子

假设同一状态 `z` 出现 4 次：

| occurrence | action | 后续结果 |
|---|---|---:|
| o1 | a1 | 1 |
| o2 | a1 | 0 |
| o3 | a2 | 0 |
| o4 | a2 | 0 |

Occurrence-level 会让 `a1` 的两次 occurrence 得到一正一负 local credit；action mean 则估计：

\[
\bar G(z,a_1)=0.5,\qquad \bar G(z,a_2)=0,
\]

从而两个 `a1` occurrence 共享同一、相对更好的 local credit。

Action mean 的统计含义更接近：

\[
Q^\pi(z,u)-V^\pi(z),
\]

而 occurrence-level 更接近单次 noisy realization：

\[
G_i-V^\pi(z).
\]

---

## 4. A1 要验证的假设

### H1-A：通用改进假设

Action mean 能减少自然 GiGPO 中的 continuation noise，因此即使没有 BACE，也会提高 validation success / AUC。

### H1-B：BACE-specific 假设的反证条件

如果 GiGPO action mean 已经获得很大收益，则后续 BACE action mean 的提升不能被解释为 BACE 特有贡献，只能说 BACE 采用了更合适的 local-credit estimator。

### H1-C：history/context 风险

如果相同 observation + action 在不同 arrival history 下实际不是同一条件决策问题，action mean 可能把有意义的 context-conditioned差异平均掉，因此可能不升反降。

---

## 5. 实验必须保持不变的内容

A1 除 local-credit estimator 外，其他全部与可靠 GiGPO baseline 相同：

- 模型与初始化 checkpoint；
- ALFWorld train/validation 数据；
- 每 task group size / rollout 数；
- rollout temperature；
- max environment steps；
- actor LR、PPO clip、KL coefficient；
- `gamma`；
- `step_advantage_w = 1.0`；
- GiGPO exact observation grouping；
- similarity matching 设置；
- invalid-action reward；
- GPU 数、TP、batch size 和调度参数；
- seed；
- validation cadence。

**不得启用 BACE dynamic topology、ERV、branch replay 或 branch-origin 数据。**

---

## 6. 代码实现要求

### 6.1 当前已有可复用逻辑

当前 `recipe/bace_gigpo/advantage.py` 中的 `_action_mean_local_advantage()` 已实现：

1. 先按 GiGPO `step_group_id` 分组；
2. 在 group 内计算原始 group mean/std；
3. 再按 `action_id` 聚合同动作 occurrence；
4. 用 action mean 相对 group mean 计算 local score；
5. 将该 score 广播回同动作 occurrences。

A1 应复用这一数学逻辑，而不是重新设计另一套 action mean。

### 6.2 推荐实现方式

推荐新增一个**GiGPO-only local credit 开关**，例如：

```text
algorithm.gigpo.local_credit_mode=occurrence|action_mean
```

纯 GiGPO advantage 路径中：

- `occurrence`：保持原 `core_gigpo.step_norm_reward()`；
- `action_mean`：调用与 BACE 相同的 action-mean helper。

### 6.3 Action identity

ALFWorld 首轮测试应使用环境落地后的 deterministic action identity，避免仅按模型表面文本进行模糊聚合。

如果 pure GiGPO 现有数据路径暂时没有 `action_id` 字段，应最小化补充该 metadata；不要为 A1 引入额外 parser、semantic clustering 或 action proposal 模型。

---

## 7. 最小实验矩阵

### G0：GiGPO Occurrence Baseline

```text
method = GiGPO
local_credit_mode = occurrence
step_advantage_w = 1.0
```

### G1：GiGPO Action Mean

```text
method = GiGPO
local_credit_mode = action_mean
step_advantage_w = 1.0
```

第一轮只做这两个版本。

不要同时测试：

- `omega=0.5`；
- 新 KL；
- 新 LR；
- similarity anchor；
- shrinkage/gating；
- uniform-action weighting。

这些都会破坏 A1 的可解释性。

---

## 8. 推荐执行阶段

### 8.1 Smoke / correctness

先跑 2–3 training steps，确认：

- action IDs 与 batch row 一一对应；
- 同 `(step_group, action)` 的 local score 完全相等；
- 不同 action 仍可不同；
- macro advantage 与 G0 完全一致；
- `step_advantage_w=0` 时 G0/G1 总 advantage 完全一致；
- action mean 不改变 rollout 数据本身。

### 8.2 短程趋势实验

建议 30–50 steps，主要看方向：

- validation AUC；
- local advantage variance；
- late/early learning slope。

### 8.3 正式实验

若短程方向正，再做与现有 GiGPO baseline 同长度的完整 run，并至少增加多个 seeds。

---

## 9. 必须记录的指标

### 效果指标

- validation success rate curve；
- final validation success；
- validation AUC；
- early / middle / late 三阶段均值；
- seed mean ± std。

### Credit 机制指标

- `|A_macro|` 均值、std；
- `|A_local|` 均值、std；
- `|omega*A_local| / (|A_macro|+|omega*A_local|)`；
- local advantage sign distribution；
- same-action within-group occurrence local variance；
- action-mean 后该 variance 的下降比例；
- 每 anchor action count 分布；
- singleton-action 比例；
- action imbalance ratio。

### PPO 稳定性

- policy KL；
- clip fraction；
- gradient norm；
- entropy；
- actor loss。

---

## 10. 结果如何解释

### 结果 A：G1 显著优于 G0

结论：action mean 是 GiGPO 通用优化，不是 BACE 独有机制。

后续：A2 仍要做，但论文中应将 action mean 定位为 optimizer choice / adopted estimator，而不是 BACE novelty。

### 结果 B：G1 ≈ G0

这是对 BACE 最有价值的情况之一。

后续如果 A2 中 BACE action mean 明显提升，则支持：

> BACE 的主动重复 `(z,u)` evidence 使 action-consistent credit 比自然 GiGPO 更重要。

### 结果 C：G1 低于 G0

说明简单 action aggregation 可能损失 context/history 信息，或 rare-action mean 方差太大。

此时 A2 仍可做一次，因为 BACE 的重复 branch evidence 可能改变 action count 结构，但不应预设它会改善。

---

## 11. Go / No-Go 标准

A1 的目标不是必须找到提升，而是得到可解释结论。

建议进入 A2 的条件：

- G1 至少没有明显训练异常；
- action ID 逻辑验证通过；
- credit diagnostics 能解释 G0/G1 差异。

即使 G1 不提升，也应继续 A2 的单独测试，因为 A2 的假设是“主动重复证据改变了 estimator 的适用性”。

---

## 12. 与后续实验的依赖关系

A1 完成后：

1. 才做 A2（BACE Action Mean）；
2. T2 的 `step_advantage_w` 调参应在最终 credit mode 基本确定后进行；
3. 不要在 A1 阶段引入 A3 probe-origin，否则会混淆 local estimator 与训练 occurrence 权重。

---

## 13. 一句话执行版

> **保持纯 GiGPO 的 rollout、优化器和所有超参不变，只把 local advantage 从“每个 occurrence 使用自己的 return”改成“同 anchor 同 action 共享 action-mean return 对应的 local advantage”，先判断 action aggregation 是否是 GiGPO 本身的通用收益。**
