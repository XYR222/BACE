# BACE Pairwise 在线验证方案
## ——当前阶段如何判断 Pairwise 是否值得进入主方法

**日期**：2026-08-30  
**适用基线**：BACE / Exact Batch-ERV  
**当前主配置**：

```text
total leaf budget B = 8
tau_BERV = 0.005
L_max = 2
Pairwise K = 2
```

---

# 0. 当前最重要的结论

现在不需要继续纠结：

> “Pairwise 到底值不值得试？”

现有离线结果已经足够支持：

\[
\boxed{\textbf{Pairwise 值得做在线实验。}}
\]

原因只有两个：

1. 第一 pair 获得真实 outcome 后，后续 allocation 在相当多 task 上会变化，说明 feedback 确实会影响 acquisition decision；
2. Pairwise-Stopping / round-wise capacity 可能减少原本 Full Batch 中逐条进行的 capacity correction，因此 Pairwise **不一定会让完整训练更慢**。

但现有结果还不能证明：

\[
\boxed{\textbf{Pairwise 应该替换 Full Batch 成为主方法。}}
\]

最终必须通过真实在线训练比较：

\[
\boxed{
\text{性能}
+
\text{wall-clock}
}
\]

来决定。

---

# 1. 当前不要再同时看太多指标

Pairwise 是否值得进入主方法，第一轮只看四个核心量。

## 1.1 Validation AUC

重点看：

\[
\boxed{
\text{step 100}\rightarrow\text{150 的 validation AUC}
}
\]

而不是只看单个 step 的 validation。

它反映 continuation 阶段整体学习效果。

---

## 1.2 Final Validation

使用：

```text
最后 5 step mean
或
最后 10 step mean
```

避免最后一个 step 的随机波动。

记作：

\[
\boxed{
S_{\mathrm{final}}
}
\]

---

## 1.3 Step Wall-Clock

比较：

\[
\boxed{
T_{\mathrm{step}}
}
\]

即完整 training step 的平均耗时。

不能只比较：

```text
branch generation time
```

因为 P2 可能：

- 增加 Pairwise branch rounds；
- 减少 capacity correction；
- 增加 fallback roots；
- 改变 GPU occupancy。

最终只看整个 step 的实际耗时。

---

## 1.4 Final Root / Branch Topology

记录：

\[
\boxed{
R_{\mathrm{actual}}
\quad/\quad
Q_{\mathrm{actual}}
}
\]

它主要用于解释结果：

> 为什么性能或速度发生变化？

它不是第一层 Go / No-Go 指标。

---

# 2. 其他指标降级为解释性指标

下面这些继续记录，但不要让它们决定 Pairwise 是否值得：

```text
replan rate
action-plan change rate
BERV gain
post-pair capacity
fallback count
correction waves
pairwise rounds
generated tokens
```

它们用于：

\[
\boxed{\text{解释 C0/C1/C2 为什么不同}}
\]

而不是替代真实 validation 和 wall-clock。

---

# 3. 第一轮不要改 Threshold

第一轮所有版本统一使用：

\[
\boxed{\tau_{\mathrm{BERV}}=0.005}
\]

原因：

如果同时比较：

```text
Full Batch @ 0.005
vs
Pairwise @ 0.0075
```

最终即使性能不同，也无法判断差异来自：

- Pairwise feedback；
- Pairwise stopping；
- threshold 改变。

因此第一轮只回答：

\[
\boxed{
\textbf{Pairwise 本身是否有价值？}
}
\]

Threshold 优化放到第二轮。

---

# 4. 三组核心实验

当前只需要三个版本。

---

# 5. C0：Current Full Batch

配置：

```text
Acquisition:
    Full Exact Batch-ERV

tau_BERV:
    0.005

capacity:
    current eager capacity correction

branch:
    all final Q frozen and executed
```

C0 是所有实验的基线。

流程：

```text
planned R/Q
↓
generate roots
↓
capacity correction until C >= Q
↓
freeze root support
↓
one Full Batch Exact allocation
↓
execute all Q branches
↓
PPO
```

---

# 6. C1：Pairwise-Fixed

配置：

```text
Acquisition:
    Pairwise Exact-BERV

K:
    2

tau_BERV:
    0.005

capacity:
    keep current eager capacity correction

branch quota:
    fixed after capacity correction

stopping:
    disabled
```

流程：

```text
planned R/Q
↓
current capacity correction
↓
freeze final root support and Q
↓
solve quota=2
↓
execute first pair
↓
observe outcome
↓
update Beta posterior
↓
solve next quota=2
↓
...
↓
execute exactly Q branches
↓
PPO
```

C1 与 C0 的核心差异只有：

\[
\boxed{
\text{Full Batch}
\rightarrow
\text{Pairwise feedback replanning}
}
\]

因此：

\[
\boxed{
C1-C0
}
\]

主要回答：

> **真实 branch outcome 参与下一轮 acquisition decision，本身有没有训练价值？**

这是当前最干净、最重要的 Pairwise 因果对照。

---

# 7. C2：Pairwise-Stopping

配置：

```text
Acquisition:
    Pairwise Exact-BERV

K:
    2

tau_BERV:
    0.005

capacity:
    round-wise / lazy capacity

stopping:
    enabled

fallback:
    remaining branch slots -> natural roots
```

流程：

```text
initial planned roots
↓
compute current capacity
↓
C >= 2:
    execute pair

C = 1:
    execute one branch

C = 0:
    stop branching
    remaining slots -> fallback roots
↓
update posterior after each round
↓
repeat
↓
PPO
```

C2 不再提前要求：

\[
C\ge Q.
\]

只要求当前 round：

\[
\boxed{
C\ge \min(2,Q_{\mathrm{remain}})
}
\]

或者在 \(C=1\) 时执行 single branch。

因此 C2 同时改变：

1. feedback acquisition；
2. branch stopping；
3. capacity scheduling；
4. 最终 root / branch topology。

所以：

\[
\boxed{
C2-C1
}
\]

主要回答：

> **在 Pairwise 已经存在的前提下，Stopping + Round-wise Capacity 是否进一步改善性能或速度？**

---

# 8. 为什么 C1 必须存在

不能只比较：

```text
C0 Full
vs
C2 Pairwise-Stopping
```

因为如果 C2 变好，我们无法判断：

- 是 Pairwise feedback 更好；
- 还是减少 branch 更好；
- 还是取消 eager capacity correction 更好。

所以必须有：

\[
\boxed{
C0
\rightarrow
C1
\rightarrow
C2
}
\]

这三层机制分解。

---

# 9. 第一阶段：2–5 Step H100 Smoke

正式 continuation 前，先分别跑：

```text
C0
C1
C2
```

每组：

\[
\boxed{2\sim5\text{ steps}}
\]

目的不是判断最终 accuracy，而是验证工程路径。

---

# 10. Smoke 必须确认的内容

## 10.1 C1

检查：

```text
Pair 1 正确生成
Pair 1 outcome 正确更新 posterior
Pair 2 使用 updated posterior
最终 executed Q = frozen Q
没有 fallback roots
L_max 跨 round 累计
```

---

## 10.2 C2

检查：

```text
C >= 2 -> pair
C = 1 -> one branch
C = 0 -> stopping
remaining slots -> fallback roots
fallback 后不 reopen
最终 roots + branches = B
family history 只使用 natural roots
```

---

## 10.3 分布式与调度

必须确认：

```text
不是 task-by-task sequential
而是 global round-wise packed execution
```

例如：

```text
Global Pair Round 1:
    所有 active tasks 的第一 pair 一起执行

Global Pair Round 2:
    更新 posterior 后
    所有剩余 active tasks 再一起执行
```

---

# 11. Smoke 最重要的时间统计

必须记录：

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

---

# 12. Pairwise 是否可能更快，真正看什么

不要用：

```text
Pairwise branch rounds 更多
```

直接推出更慢。

完整时间：

\[
\boxed{
T_{\mathrm{step}}
=
T_{\mathrm{root}}
+
T_{\mathrm{capacity}}
+
T_{\mathrm{branch}}
+
T_{\mathrm{fallback}}
+
T_{\mathrm{PPO}}
}
\]

C2 的核心工程假设是：

\[
\Delta T_{\mathrm{branch}}>0
\]

但：

\[
\Delta T_{\mathrm{capacity}}<0.
\]

如果：

\[
|\Delta T_{\mathrm{capacity}}|
>
\Delta T_{\mathrm{branch}}
+
\Delta T_{\mathrm{fallback}},
\]

那么：

\[
\boxed{
T_{\mathrm{step}}^{C2}
<
T_{\mathrm{step}}^{C0}.
}
\]

所以最终必须实测。

---

# 13. Smoke 后的正式实验

如果三组 smoke 都稳定，则从**同一个 step-100 checkpoint**继续到：

\[
\boxed{150}
\]

比较：

```text
C0 Full @ 0.005
C1 Pairwise-Fixed @ 0.005
C2 Pairwise-Stopping @ 0.005
```

---

# 14. 为什么从 Step 100 开始

原因：

1. Pairwise opportunity 主要出现在 middle / late；
2. 当前 BACE 的主要问题恰好是 late-stage gap；
3. continuation 比重新从 0 跑 150 便宜很多；
4. 同 checkpoint 分叉具有更强因果解释。

必须尽可能恢复：

```text
actor
optimizer
scheduler
dataloader
family competence history
RNG state
```

切换点必须位于完整 PPO update boundary。

---

# 15. 正式实验必须固定的变量

C0 / C1 / C2 全部固定：

```text
same step-100 checkpoint
same seed
same model
same GPU count
same scheduler
same rollout batch
same PPO
same advantage
same local_credit_mode
same tau_BERV = 0.005
same L_max = 2
same competence controller
same task prior
same anchor definition
```

当前不要同时修改：

```text
action_mean
step_advantage_w
Rmin
Beta prior
K
behavior correction
branch loss weight
dynamic threshold
```

---

# 16. Pairwise-Fixed 怎么判断值不值得

只比较：

\[
\boxed{
C1
\quad vs\quad
C0
}
\]

重点看：

```text
Validation AUC
Final validation
Step wall-clock
```

---

# 17. C1 决策表

| C1 相对 C0 | 判断 |
|---|---|
| 更准 + 一样快/更快 | **强烈支持 Pairwise-Fixed** |
| 更准 + 略慢 | **支持，进一步看收益/成本比** |
| 一样准 + 更快 | **支持，属于工程收益** |
| 一样准 + 更慢 | **不支持作为主方法** |
| 更差 | **不支持作为主方法** |

---

# 18. 可接受的时间开销

第一轮可以暂时使用：

\[
\boxed{10\%\sim15\%}
\]

作为经验参考线，而不是理论阈值。

例如：

### 情况 1

```text
validation +2 pp
wall-clock +6%
```

值得继续。

### 情况 2

```text
validation +0.2 pp
wall-clock +20%
```

基本不值得。

最终应该看：

\[
\boxed{
\text{learning improvement per wall-clock}
}
\]

而不是只看 sample efficiency。

---

# 19. Pairwise-Stopping 怎么判断值不值得

只比较：

\[
\boxed{
C2
\quad vs\quad
C1
}
\]

原因：

C1 已经包含 Pairwise feedback。

所以：

\[
C2-C1
\]

更接近：

```text
Stopping
+
Round-wise Capacity
+
Topology fallback
```

的额外收益。

---

# 20. C2 决策表

| C2 相对 C1 | 判断 |
|---|---|
| 更准 + 更快 | **强烈支持 P2** |
| 更准 + 一样快 | **支持 P2** |
| 一样准 + 更快 | **支持 P2，工程价值明确** |
| 略准 + 略慢 | 看收益/成本比 |
| 一样准 + 更慢 | 不支持 |
| 更差 | 不支持当前 P2 语义 |

---

# 21. 最终三组结果怎么解释

## Case A

\[
C1>C0,
\qquad
C2\approx C1.
\]

结论：

\[
\boxed{
\text{Pairwise feedback 有价值，
Stopping / Lazy Capacity 额外价值有限。}
}
\]

主候选：

```text
Pairwise-Fixed
```

---

## Case B

\[
C1\approx C0,
\qquad
C2>C1.
\]

结论：

\[
\boxed{
\text{Full Batch selection 本身问题不大，
主要收益来自 stopping / capacity scheduling。}
}
\]

主候选：

```text
Pairwise-Stopping
```

---

## Case C

\[
C2>C1>C0.
\]

结论：

\[
\boxed{
\text{feedback selection 和 stopping/capacity 都有效。}
}
\]

主候选：

```text
Pairwise-Stopping
```

---

## Case D

\[
C0\approx C1\approx C2
\]

且 Pairwise 更慢：

结论：

\[
\boxed{
\text{Pairwise 不值得进入主方法。}
}
\]

保留 Full Batch。

Pairwise 可作为 ablation。

---

## Case E

\[
C0>C1,C2.
\]

说明：

> Full Batch 的 joint planning 或当前 topology semantics 比 Pairwise 更适合训练。

不再继续为 Pairwise 增加复杂度。

---

# 22. Replan Rate 现在只怎么用

现有：

\[
\sim45.6\%
\]

的 allocation replan rate 只说明：

\[
\boxed{
\text{Pairwise 有足够大的发挥空间。}
}
\]

它支持：

> 值得跑 C1。

它**不意味着**：

> C1 一定比 C0 好。

---

# 23. 离线 BERV Gain 现在只怎么用

现有离线结果显示：

> Pairwise 的 BERV objective gain 通常为正，但幅度不大。

它告诉我们：

\[
\boxed{
\text{不要预期 Pairwise 必然带来巨大提升。}
}
\]

但不能用于直接否掉 Pairwise。

因为：

\[
\text{BERV acquisition value}
\neq
\text{最终 RL validation}.
\]

---

# 24. Stopping Probability 现在只怎么用

在：

\[
\tau=0.005
\]

下，post-pair stopping pressure 较弱。

但 C2 的价值不只来自 stopping。

它还改变：

\[
\boxed{
\text{full-quota eager capacity correction}
}
\]

所以 stopping 频率低并不能直接说明 C2 没价值。

最终仍看：

\[
T_{\mathrm{step}}
\]

和：

\[
\text{validation}.
\]

---

# 25. Threshold 什么时候再研究

只有完成：

```text
C0
C1
C2
```

第一轮以后。

如果：

\[
C2
\]

至少表现出：

- 性能不差；
- 或 wall-clock 有优势；

再增加：

\[
\boxed{
C3=
\text{Pairwise-Stopping},
\quad
\tau=0.0075
}
\]

---

# 26. 为什么第二轮先测 0.0075

当前离线结果说明：

\[
0.0075
\]

已经能够明显：

- 减少 weak branch；
- 增加 stopping/fallback pressure；
- 放大 `C>=2` 与 `C>=Q` 的区别；

但没有像：

\[
0.015
\]

那样明显激进。

所以：

\[
\boxed{
0.0075
}
\]

适合作为第一 higher-threshold probe。

但当前不把它认定为最终最优 threshold。

---

# 27. 如果 C3 有效果，再加 Threshold Control

如果：

\[
C3
\]

优于：

\[
C2,
\]

必须增加：

\[
\boxed{
C4=
\text{Full Batch},
\quad
\tau=0.0075
}
\]

否则无法知道：

\[
C3-C2
\]

的收益来自：

- higher threshold；
- 还是 threshold 与 Pairwise 的协同。

---

# 28. 第一轮实验树

```text
                    C0
             Full Batch @ .005
                    |
          -----------------------
          |                     |
          v                     v
   C1 Pairwise-Fixed     C2 Pairwise-Stopping
        @ .005                  @ .005
          |                     |
          |                     |
          -------- compare ------
                    |
                    v
          Decide Pairwise worth
```

---

# 29. 第二轮实验树

只有第一轮结果支持继续：

```text
C2 Pairwise-Stopping @ .005
            |
            v
C3 Pairwise-Stopping @ .0075
            |
      if C3 improves
            |
            v
C4 Full Batch @ .0075
```

用于拆解：

\[
\text{Pairwise effect}
\]

和：

\[
\text{Threshold effect}.
\]

---

# 30. 当前真正的 Go / No-Go 标准

## Pairwise 进入主候选

满足至少一种：

### Go-1

\[
\boxed{
\text{validation 明显更好，
wall-clock 增幅可接受}
}
\]

### Go-2

\[
\boxed{
\text{validation 基本相同，
wall-clock 明显更快}
}
\]

### Go-3

\[
\boxed{
\text{validation 更好，
wall-clock 同时更快}
}
\]

这是最强结果。

---

## Pairwise 暂不进入主方法

如果：

\[
\boxed{
\text{validation 没有改善，
并且 wall-clock 更慢}
}
\]

则当前 Pairwise 不值得进一步复杂化。

---

# 31. 当前不要做的事情

在 C0/C1/C2 被回答之前，不要同时加入：

```text
tau = 0.01 / 0.015 sweep
K = 1 / 4
dynamic threshold
action_mean
step_advantage_w tuning
Rmin tuning
new root value
new competence mapping
branch loss weighting
```

否则实验再次失去解释性。

---

# 32. 当前执行清单

## Step A：Smoke

- [ ] C0 Full `.005`，2–5 steps；
- [ ] C1 Pairwise-Fixed `.005`，2–5 steps；
- [ ] C2 Pairwise-Stopping `.005`，2–5 steps；
- [ ] 检查 budget invariant；
- [ ] 检查 posterior update；
- [ ] 检查 global packed execution；
- [ ] 检查 fallback；
- [ ] 检查 family history；
- [ ] 检查 checkpoint；
- [ ] 检查 step wall-clock；
- [ ] 检查各阶段时间。

---

## Step B：Continuation

从同一个：

```text
step-100 checkpoint
```

分别运行：

```text
C0 -> step150
C1 -> step150
C2 -> step150
```

---

## Step C：只比较四个核心量

```text
1. validation AUC
2. final validation
3. step wall-clock
4. final root / branch topology
```

---

## Step D：做决定

```text
C1 vs C0
    -> Pairwise feedback 是否值得？

C2 vs C1
    -> Stopping / round-wise capacity 是否值得？
```

---

## Step E：Threshold 第二轮

只有 Pairwise 有继续价值时：

```text
C3 Pairwise-Stopping @ 0.0075
```

如果 C3 更好，再加：

```text
C4 Full Batch @ 0.0075
```

---

# 33. 最终一句话

当前阶段已经不需要继续用更多离线指标证明：

\[
\text{“Pairwise 是否值得试”}.
\]

现有证据已经足够支持：

\[
\boxed{
\textbf{值得试。}
}
\]

现在真正需要回答的是：

\[
\boxed{
\textbf{Pairwise 是否能在真实训练中带来更好的 validation / wall-clock trade-off？}
}
\]

因此当前最优先的实验就是：

\[
\boxed{
\textbf{C0 Full .005}
\quad
vs
\quad
\textbf{C1 Pairwise-Fixed .005}
\quad
vs
\quad
\textbf{C2 Pairwise-Stopping .005}
}
\]

先完成 smoke，再从同一 step-100 checkpoint 运行到 step 150。

在这三组结果出来之前，不再扩展更多 Pairwise 变体，也不提前进行 threshold 大范围 sweep。
