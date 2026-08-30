# A2：BACE Action-Mean Local Credit 实验方案

> 目的：在当前 BACE Exact Batch-ERV 主线中，仅将 `local_credit_mode` 从 `occurrence` 改为 `action_mean`，验证主动 branch 产生的重复 `(anchor, action)` evidence 是否需要 action-consistent local credit 才能被稳定吸收。
>
> 前置条件：优先完成 A1（GiGPO Action Mean），用于判断 action mean 是通用 GiGPO 改进，还是对 BACE 的增益更明显。

---

## 1. 当前 BACE 的基线语义

当前正式 BACE advantage：

\[
A_i=A_i^{\mathrm{macro}}+\omega A_i^{\mathrm{local}},
\]

主配置：

```text
algorithm.bace.local_credit_mode=occurrence
algorithm.gigpo.step_advantage_w=1.0
algorithm.gigpo.mode=mean_std_norm
algorithm.gamma=0.95
```

BACE 训练 batch 当前包括：

- `root` / natural occurrence；
- `branch_origin`；
- `branch_suffix`。

这意味着 BACE 相比自然 GiGPO 更容易在同一 `(z,u)` 上主动产生重复 evidence。

---

## 2. 为什么 BACE 比 GiGPO 更可能受益于 Action Mean

BACE acquisition 的统计对象本身就是：

\[
p_{z,u}=P(Y=1\mid z,\operatorname{do}(u),\pi_{old}).
\]

ERV 通过比较不同 action posterior 来决定哪里值得继续实验，因此采集阶段的基本单位是：

\[
(z,u).
\]

但 occurrence local credit 的基本单位仍是：

\[
(z,u,\text{one realized continuation}).
\]

这造成潜在不一致：

```text
Acquisition：多次实验同一个动作，估计动作期望价值
Optimization：每一次实验 outcome 单独被当作当前动作信用
```

Action mean 将优化端也改成：

\[
\widehat Q(z,u)=\text{该动作所有当前 evidence 的平均后续回报},
\]

从而形成：

```text
ERV acquisition unit = (z,u)
Local credit unit     = (z,u)
```

这在 BACE 中具有比自然 GiGPO 更强的统计一致性。

---

## 3. A2 的核心假设

### H2-A：主动重复证据假设

BACE 会主动增加同一 `(z,u)` 的多次 continuation，因此 occurrence-level credit 更容易把 continuation noise 误归因给当前动作；action mean 可降低该噪声。

### H2-B：late-stage degradation 假设

当前 BACE 后期 branch 比例较高。如果 selected anchors 上同一 action 被重复采样，occurrence local gradient 的相互冲突可能随 branch 增多而放大。Action mean 可能主要改善中后期，而非训练最早期。

### H2-C：rare-action 风险

如果某个 action 只有 1 个 observation，简单 action mean 不会自动解决 lucky singleton；因此本实验只验证“简单 action aggregation”，不同时加入 shrinkage / ECPO gate。

---

## 4. 唯一允许的算法改动

Control：

```text
algorithm.bace.local_credit_mode=occurrence
```

Treatment：

```text
algorithm.bace.local_credit_mode=action_mean
```

其他全部保持一致：

- Exact Batch-ERV；
- dynamic topology；
- `B=8`；
- `R_min=2`（除非届时正式基线已经更换；关键是两组相同）；
- `L_max=2`；
- competence threshold；
- local prior strength；
- Batch-ERV threshold；
- root/branch quota；
- branch-origin 是否训练（A2 阶段保持当前行为）；
- `step_advantage_w=1.0`；
- LR/KL/PPO clip；
- P1-S 调度开关；
- seed 与 H100 数量。

**A2 不要同时做 A3。**

---

## 5. 当前代码如何计算 Action Mean

当前 `_action_mean_local_advantage()` 的逻辑为：

1. 使用 GiGPO `build_step_group()` 得到同 observation 的 state group；
2. 在 state group 内计算所有 occurrence 的 `step_rewards` 均值 `mu_z`；
3. 若使用 std normalization，计算同一 state group 的 sample std `sigma_z`；
4. 再按 `action_id` 分组；
5. 对每个动作计算：

\[
\bar G_{z,u}=\operatorname{mean}_{i\in(z,u)}G_i;
\]

6. 得到：

\[
A^S_{z,u}
=
\frac{\bar G_{z,u}-\mu_z}{\sigma_z+\epsilon};
\]

7. 广播给所有同 `(z,u)` occurrences。

因此 A2 不需要新理论 estimator；主要是启用当前已有分支，并做完整验证。

---

## 6. 一个 BACE 特有的例子

自然 roots 在 anchor `z` 上已有：

```text
a1 -> success
a2 -> failure
```

BACE 因 ERV 对 `a1` 再做一个 branch：

```text
a1 -> failure
```

Occurrence credit 可能形成：

```text
a1(root success)  -> local positive
a1(branch fail)   -> local negative
a2(root fail)     -> local negative
```

此时同一个 `a1` 被 policy 同时推高和压低。

Action mean 则先得到：

\[
\bar G(z,a_1)=0.5,\qquad \bar G(z,a_2)=0,
\]

两个 `a1` occurrences 共享同一 local direction。

注意：它们的 **macro advantage 仍可不同**，所以完整 trajectory 成败不会被抹掉。

---

## 7. 实验矩阵

### B0：Current BACE

```text
local_credit_mode=occurrence
step_advantage_w=1.0
```

### B1：BACE Action Mean

```text
local_credit_mode=action_mean
step_advantage_w=1.0
```

如果 A1 已得到 G0/G1，则最终分析至少形成：

| 方法 | Occurrence | Action Mean |
|---|---:|---:|
| GiGPO | G0 | G1 |
| BACE | B0 | B1 |

关键不是只看 B1-B0，还要比较：

\[
(B1-B0)\quad vs\quad(G1-G0).
\]

---

## 8. 最重要的机制指标

### 8.1 同动作冲突率

定义在同 `(z,u)` group 中 occurrence-level local advantage 同时含正和负的比例：

\[
\rho_{conflict}
=
\frac{\#\{(z,u):\exists A_i^S>0,\exists A_j^S<0\}}{\#\{(z,u)\}}.
\]

分别报告：

- natural-only groups；
- 含 branch-origin groups；
- 含 branch-suffix groups；
- early / middle / late training。

### 8.2 Within-action variance

\[
W_{z,u}=\operatorname{Var}(G_i\mid z,u).
\]

看 BACE 是否比 GiGPO 有更高或更频繁的 within-action continuation variance。

### 8.3 Branch-created evidence share

报告 `(z,u)` evidence 中：

- natural occurrence 数；
- branch-origin 数；
- branch suffix collision 数。

这样才能知道 action mean 的收益是否真的来自主动 evidence。

---

## 9. 效果与稳定性指标

必须记录：

- validation curve / AUC / final；
- 1–50、51–100、101–150 等阶段均值；
- natural-root success；
- branch success；
- branch fraction；
- `Q_planned / Q_final`；
- correction rate；
- macro/local advantage magnitude；
- local sign conflict rate；
- policy KL；
- clip fraction；
- gradient norm；
- local gradient contribution（若可记录）。

特别关注：**后期 BACE-GiGPO gap 是否缩小。**

---

## 10. 结果解释

### 情况 1：GiGPO 与 BACE 都明显提升

Action mean 是通用改进；BACE 可以采用，但不作为主要 novelty。

### 情况 2：GiGPO 基本不变，BACE 明显提升

最有研究价值。支持：

> Active branching creates repeated action evidence, for which action-consistent local credit is materially more important than in natural GiGPO rollouts.

### 情况 3：GiGPO 提升，BACE 不提升

说明 BACE 的主要问题可能不是 same-action continuation noise，而是 acquisition weighting、branch-origin multiplicity 或 topology。

优先进入 A3。

### 情况 4：BACE 变差

可能原因：

- anchor state 并不充分 Markov，同 action 不同 history 有真实差异；
- singleton / rare action mean 过于噪声；
- branch-origin 被重复计权的问题仍存在，action mean 只是改变数值而未解决权重。

此时不要立刻叠加 shrinkage，先看 A3。

---

## 11. Checkpoint 与复现实验注意事项

`local_credit_mode` 属于 BACE collector / algorithm semantics。当前仓库的 dynamic BACE checkpoint 对关键参数有签名约束；不要把 `occurrence` checkpoint 直接当作长期 `action_mean` run 的无条件恢复点。

最干净的正式对照是：

- 同一个可复现的 actor initialization；
- 两个独立 run root；
- 独立 checkpoint / artifacts / tensorboard；
- 明确记录 Hydra override。

若仅做短程诊断，需要确认当前 checkpoint loader 是否允许该参数改变；若不允许，应从共同上游 checkpoint 重新启动，而不是关闭签名校验。

---

## 12. A2 结束后的决策

如果 B1 胜出，则后续 A3/T2 以 `action_mean` 为候选主 credit mode；如果 B0 胜出，则 A3 先保持 `occurrence`，不要把 action mean 和 probe-origin 同时切换。

T2（omega）必须在 A2 后做，因为 action mean 会改变 `A_local` 的方差和幅度，最佳 `omega` 可能不同。

---

## 13. 一句话执行版

> **保持当前 BACE Exact Batch-ERV 的 rollout、topology、branch 和 PPO 语义完全不变，只把 local credit 从 occurrence-level 换成同 anchor 同 action 的平均 return credit，并与 A1 的 GiGPO 对照共同判断 action mean 是通用收益还是主动 branching 特别需要的信用吸收方式。**
