# P1-A：BACE Root–Branch Quota 诊断性修正方案
## ——将 `min_natural_roots` 从 2 调整为 4，限制 `Q_max=4`

**适用基线**：当前 No-Pilot + Exact Batch-ERV topology。  
**核心修改**：

\[
\boxed{R_{\min}:2\rightarrow4}
\]

在总 leaf budget：

\[
B=8
\]

下，对应：

\[
\boxed{Q_{\max}:6\rightarrow4}.
\]

**文档定位**：P1-A 是一个低变量、诊断性的算法实验，用于验证当前后期 quota/capacity mismatch 是否为主要问题。  
**重要说明**：P1-A 不把“4+4”宣称为最终理论设计；如果实验支持假设，再进入 P2-A 的 capacity-aware topology。

---

# 1. 当前问题

当前 Exact topology planner 使用 lagged family competence history。

对于 task family \(c\)，得到 family posterior 后计算：

\[
q_c=P(\phi_c>\tau_{\text{comp}}).
\]

当前：

\[
B=8,\qquad R_{\min}=2.
\]

因此 flexible budget 为：

\[
B-R_{\min}=6.
\]

代码中的初始 branch quota：

\[
Q_{\text{plan}}
=
\left\lfloor 6q_c+0.5\right\rfloor.
\]

然后：

\[
R_{\text{plan}}
=
8-Q_{\text{plan}}.
\]

当前 `ExactBatchTopologyPlanner.initialize()` 实际就是：

```python
flexible_budget = total_budget - min_natural_roots

planned_branches = floor(
    flexible_budget * readiness + 0.5
)

root_count = total_budget - planned_branches
branch_count = planned_branches
```

因此当 readiness 很高时：

\[
q_c\rightarrow1
\]

会直接趋向：

\[
R=2,\qquad Q=6.
\]

---

# 2. 为什么当前数据提示这里存在问题

当前 150-step run 中，后期实际 topology 已经表现出明显的 quota/capacity mismatch。

已有实验统计：

- 初始 planned branch slots：7116；
- 最终执行 branches：5703；
- 有 1413 个原计划 branch slots 被 capacity correction 改回 root；
- overall correction rate 约 19.86%；
- 后期 correction rate 进一步升高；
- 共出现 376 个 capacity-correction root waves；
- capacity-correction root generation 累计约 2.37 小时。

后 50 step 的实际平均 topology 约为：

\[
R_{\text{final}}\approx4.28,
\qquad
Q_{\text{final}}\approx3.72.
\]

也就是说系统经常发生：

```text
初始：
2 roots + 6 branches

        ↓
真实 root evidence 暴露出的 information capacity 不足
        ↓
一次 +1 root / -1 branch
        ↓
重新生成 root
        ↓
重新评估
        ↓
继续 correction
        ↓

最终：
大约 4 roots + 4 branches
```

因此 P1-A 的核心问题是：

> 如果系统最后本来就经常需要 4 条左右 natural roots，是否应该一开始就保守地给到至少 4 roots，而不是先从 2 roots 开始、再用多轮 correction 修回来？

---

# 3. P1-A 的核心假设

P1-A 不试图证明“4 是最优值”。

它只测试以下假设。

## H1：当前 \(R_{\min}=2\) 在高 readiness 阶段过于激进

当模型后期变强：

\[
q_c\uparrow
\]

当前 controller 会：

\[
Q_{\text{plan}}\uparrow
\]

并压缩 natural roots。

但是只有两条 roots 时，当前 task 可能根本没有暴露出足够的：

- repeated anchors；
- mixed-action anchors；
- Exact BERV-positive slots。

因此 readiness 与可实现 branch capacity 脱节。

---

## H2：增加初始 natural roots 可以降低 correction serial depth

如果直接从：

\[
R_{\min}=4
\]

起步，可能减少：

```text
root
→ capacity check
→ +1 root
→ capacity check
→ +1 root
...
```

的多轮串行 correction。

即使最终 root 总数与当前方案相近，**把需要的 roots 更早生成**也能缩短 wall-clock。

---

## H3：更多 natural-root coverage 可能改善后期训练

后 50 step 当前 branch 占比已经接近一半。

将：

\[
Q_{\max}=6
\]

限制到：

\[
Q_{\max}=4
\]

可保证：

\[
R_g\ge4.
\]

这会给每个 task 保留至少一半独立 natural-root leaves。

这可能改善：

- long-horizon coverage；
- 新错误模式覆盖；
- 完整任务分布上的持续训练。

但这是待验证假设，不应提前当作事实。

---

# 4. 修改后的公式

P1-A 设置：

\[
R_{\min}=4.
\]

于是：

\[
B-R_{\min}=4.
\]

新的 planned branch quota：

\[
\boxed{
Q_{\text{plan}}
=
\left\lfloor4q_c+0.5\right\rfloor
}
\]

并且：

\[
\boxed{
R_{\text{plan}}
=
8-Q_{\text{plan}}
}
\]

因此：

\[
0\le Q_{\text{plan}}\le4,
\]

\[
4\le R_{\text{plan}}\le8.
\]

---

# 5. 与当前方案的具体差异

| readiness \(q_c\) | 当前 \(R_{\min}=2\)：planned \(R+Q\) | P1-A \(R_{\min}=4\)：planned \(R+Q\) |
|---:|---:|---:|
| 0.00 | 8 + 0 | 8 + 0 |
| 0.10 | 7 + 1 | 8 + 0 |
| 0.25 | 6 + 2 | 7 + 1 |
| 0.50 | 5 + 3 | 6 + 2 |
| 0.75 | 3 + 5 | 5 + 3 |
| 0.90 | 3 + 5 | 4 + 4 |
| 1.00 | 2 + 6 | 4 + 4 |

最主要的变化发生在：

\[
q_c\text{ 较高}
\]

的训练后期。

这正是当前实验中 capacity correction 最严重的时期。

---

# 6. P1-A 不修改哪些东西

为了保证实验可解释，以下全部保持不变：

## Family competence

不修改：

- history initial mean；
- initial strength；
- forgetting；
- transfer fraction；
- concentration bounds；
- competence threshold。

---

## Current-instance posterior

Root 生成后，仍然使用当前自然 root outcome 更新 local/current posterior。

---

## Anchor

不修改：

- exact pre-action observation grouping；
- repeated occurrence requirement；
- mixed-action requirement；
- invalid action identity；
- origin pool。

---

## Exact Batch-ERV

不修改：

\[
V_z^{(1)},
V_z^{(2)},
\delta_z^{(1)},
\delta_z^{(2)}
\]

的计算。

不修改：

- `batch_erv_threshold=0.005`
- `max_branches_per_anchor=2`
- tie tolerance
- global DP allocation

---

## Capacity correction

仍然保留当前规则：

若：

\[
C_g<Q_g,
\]

则：

\[
R_g\leftarrow R_g+1,
\qquad
Q_g\leftarrow Q_g-1.
\]

也就是说 P1-A 的 `4+4` 不是强制最终固定 topology。

例如：

```text
planned = 4 roots + 4 branches
```

如果实际 information capacity 只有 2：

```text
最终仍可变成：
6 roots + 2 branches
```

因此：

\[
\boxed{4+4 是 initial upper/lower bound，不是最终固定分配。}
\]

---

# 7. 代码修改范围

P1-A 理论上不需要修改 planner 公式。

当前代码已经把：

```text
min_natural_roots
```

参数化。

最小实现只需要把 launcher：

```text
algorithm.bace.min_natural_roots=2
```

改为：

```text
algorithm.bace.min_natural_roots=4
```

建议不要覆盖当前 production script，而是新建实验配置，例如：

```text
run_bace_alfworld_gpu_rmin4.sh
```

或者通过 override：

```bash
algorithm.bace.min_natural_roots=4
```

运行。

---

# 8. 推荐实验命名

为了后续审计，建议名称明确包含 topology：

```text
bace_exact_rmin2_qmax6_seed0
bace_exact_rmin4_qmax4_seed0
```

如果使用 P1-S scheduler：

```text
bace_exact_schedv2_rmin2_qmax6_seed0
bace_exact_schedv2_rmin4_qmax4_seed0
```

不要只写：

```text
new_bace
```

否则后面很难复盘。

---

# 9. P1-A 的最干净实验设计

P1-A 必须和 P1-S 分开归因。

推荐顺序：

## Step A：先固定 scheduler

先选择一个稳定 scheduler，例如 P1-S 完成后的 scheduler。

---

## Step B：只改 `min_natural_roots`

对照：

### Control

\[
R_{\min}=2,\quad Q_{\max}=6
\]

### Treatment

\[
R_{\min}=4,\quad Q_{\max}=4
\]

其余完全相同。

这样：

\[
\Delta\text{performance}
\]

才能归因到 root/branch quota，而不是 scheduler。

---

# 10. 推荐的快速诊断实验：从共同 checkpoint 分叉

由于问题主要出现在后期，不必每次都从 step 0 跑到 step 150 才得到第一轮信号。

如果有可信的共同 checkpoint，例如 step 100：

```text
same actor checkpoint
same optimizer state
same dataloader state
same BACE family history
```

可以分叉：

```text
Branch A:
step 101~150
Rmin=2

Branch B:
step 101~150
Rmin=4
```

这样特别适合回答：

> 高 competence 阶段，Rmin=4 是否减少 correction 并改善 late-stage learning？

注意：

- 两个 run 从相同 checkpoint 开始；
- 随机 rollout 之后会自然分叉；
- 不能把单 seed 差异当最终统计结论；
- 但非常适合做机制诊断。

---

# 11. P1-A 需要重点记录的指标

## 11.1 Topology 指标

每 step：

```text
competence_readiness_mean
planned_branches_mean
final_roots_mean
final_branches_mean
capacity_corrections_mean
information_capacity_mean
effective_anchors_mean
```

额外建议记录：

\[
Q_{\text{plan}}-Q_{\text{final}}
\]

和：

\[
\frac{Q_{\text{plan}}-Q_{\text{final}}}{Q_{\text{plan}}}
\]

作为 quota overprediction 指标。

---

## 11.2 Correction 指标

```text
steps_with_capacity_correction
capacity_correction_root_waves
capacity_correction_root_trajectories
capacity_correction_generation_seconds
```

P1-A 的核心验证之一就是：

\[
\boxed{\text{这些指标是否大幅下降。}}
\]

---

## 11.3 Sampling composition

每个阶段记录：

\[
\text{root fraction}
=
\frac{R}{8},
\]

\[
\text{branch fraction}
=
\frac{Q}{8}.
\]

尤其关注 step 101~150。

---

## 11.4 学习效果

主要看：

```text
val/success_rate
```

而不是只看 mixed training success。

建议按：

```text
step 1~50
step 51~100
step 101~150
```

分阶段比较。

如果从 step100 分叉，则重点看：

```text
step 105~150
```

平均值，而不是只看 step150 单点。

---

## 11.5 Wall-clock

记录：

```text
planned root generation
capacity-correction root generation
branch suffix generation
total generation
total step time
```

---

# 12. P1-A 的主要成功条件

以下是推荐的诊断标准，不是已有事实。

## 条件 1：明显减少 correction

希望：

\[
\text{correction rate}
\]

相对当前降低至少约 30%~50%。

如果几乎不降，说明 `Rmin=2` 并不是 correction 的主要原因。

---

## 条件 2：capacity-correction wall-clock 明显降低

因为 P1-A 的一个重要价值就是：

> 把原本后续一轮一轮补的 root，前移到最初 packed root wave。

所以即使最终 root 总数接近，也应该减少 serial depth。

---

## 条件 3：后期 validation 不应变差

如果：

```text
correction ↓
wall-clock ↓
但 validation 明显 ↓
```

说明减少 branch 可能损害了有用 refinement，P1-A 不适合作为最终方向。

---

## 条件 4：最好同时改善 late-stage validation

最理想：

```text
correction ↓
wall-clock ↓
late validation ↑
```

这会强烈支持：

\[
\boxed{
\text{当前高 readiness 阶段 branch quota 过激}
}
\]

这个解释。

---

# 13. P1-A 的可能结果与解释

## Case A：速度更快、性能更好

这是最支持当前假设的结果。

解释：

```text
Rmin=2
→ 太早压缩 natural roots
→ capacity 不足
→ 多轮 correction
→ branch 比例过高
```

下一步应进入 P2-A：

\[
\text{capacity-aware topology}.
\]

---

## Case B：速度更快、性能相近

仍然是成功结果。

说明至少：

\[
R_{\min}=2
\]

没有带来明显学习收益，却付出了 correction serial cost。

P1-A 可作为更稳健默认值。

---

## Case C：速度更快、性能下降

说明：

- 高 branch quota 虽然系统上昂贵；
- 但这些 branches 可能提供有效学习信号。

这时需要寻找：

\[
\text{更好的 capacity prediction}
\]

而不是简单 cap \(Q\)。

---

## Case D：速度几乎不变、性能相近

说明当前 correction overhead 可能主要来自：

- scheduler；
- root trajectory length；
- H100 utilization；

而不是初始 \(R_{\min}\)。

这会把优先级转回 P1-S/P2-S。

---

## Case E：correction 仍然很多

即使：

\[
R_{\min}=4
\]

仍大量：

```text
4+4 → 5+3 → 6+2
```

则说明问题更深：

\[
\boxed{
\text{readiness→quota 映射本身不对}
}
\]

而不是仅仅 \(R_{\min}\) 太小。

这正是进入 capacity-aware topology 的信号。

---

# 14. P1-A 的速度影响为什么不是简单“root 更多所以更慢”

需要避免一个误解。

P1-A 初始确实会生成更多 roots。

例如高 readiness：

### 当前

```text
2 roots
→ correction
→ correction
→ ...
→ 最终约 4 roots
```

### P1-A

```text
一开始 packed 4 roots
```

如果最终都需要约 4 roots，那么总 root 数并没有显著增加。

区别是：

当前：

\[
2+1+1
\]

分多个依赖 wave。

P1-A：

\[
4
\]

尽量在一个 packed wave。

因此 P1-A 的潜在 speedup 主要来自：

\[
\boxed{\text{减少 serial generation barriers}}
\]

而不是“少生成 root token”。

---

# 15. P1-A 也可能增加总 root token

在中等 readiness 下，需要诚实考虑反方向。

例如：

\[
q_c=0.5.
\]

当前：

\[
R=5,\quad Q=3.
\]

P1-A：

\[
R=6,\quad Q=2.
\]

如果当前 5 roots 已经足够 capacity，那么 P1-A 会额外多一条完整 root。

由于 branch suffix 通常比完整 root 短，这可能增加 generation token 数。

所以：

\[
\boxed{\text{P1-A 不保证所有阶段 wall-clock 都下降。}}
\]

它最有可能受益的是：

- readiness 高；
- 当前 correction 多；
- 当前 initial \(R=2/3\) 经常被修回 \(R=4/5\)；

的后期阶段。

这也是为什么建议重点分析 step 101~150。

---

# 16. P1-A 与 P1-S 的实验隔离

P1-S 与 P1-A 解决不同问题：

## P1-S

问：

> 同样的 rollout 计划，能不能执行得更快？

不改变：

\[
R,Q.
\]

---

## P1-A

问：

> 当前 controller 一开始给出的 \(R,Q\) 是否过激？

改变 topology。

---

因此不能只跑：

```text
旧 scheduler + Rmin2
vs
新 scheduler + Rmin4
```

然后直接下结论。

至少应有：

```text
E0: scheduler_old + Rmin2
E1: scheduler_new + Rmin2
E2: scheduler_new + Rmin4
```

其中：

\[
E1-E0
\]

测 P1-S，

\[
E2-E1
\]

测 P1-A。

---

# 17. P1-A 暂时不做的事情

## 不改 competence prior

暂时不动 Beta family history。

否则无法知道是 prior 还是 root floor 起作用。

---

## 不增加 independent Pilot

当前 No-Pilot 逻辑保持。

P1-A 本质上是：

\[
\text{更多 initial natural roots}
\]

而不是重新引入额外 pilot barrier。

---

## 不让 current root outcome 重新决定 planned \(Q\)

当前 root outcome 仍用于 current posterior / BERV，但 P1-A 不新增新的 quota formula。

这是 P2-A 的研究空间。

---

## 不切 Pairwise \(K=2\)

仍然：

\[
K=Q
\]

Full Batch-ERV。

否则 topology 和 acquisition feedback 同时变化。

---

# 18. 如果 P1-A 成功，P2-A 应该怎么发展

P1-A 成功意味着：

> competence 可以判断“是否成熟到值得 refinement”，但不能直接决定“应该 refinement 多少次”。

下一版可将 controller 拆成：

### Readiness

\[
q_c
=
P(\phi_c>\tau_{\text{comp}})
\]

负责：

> 是否倾向从 breadth 转向 refinement。

---

### Realized capacity

从当前自然 roots 建立：

\[
C_g
=
\sum_z c_g(z)
\]

负责：

> 当前 task 实际支持多少 decision-relevant branch experiments。

---

### 最终 quota

可进一步设计：

\[
Q_g
=
f(q_c,C_g)
\]

最简单可能是：

\[
Q_g
=
\min(
Q_{\text{readiness}},
C_g,
Q_{\max}
).
\]

但这一公式不属于 P1-A。

P1-A 的任务只是先验证：

\[
\boxed{\text{当前 }Q_{\max}=6\text{ 是否过激。}}
\]

---

# 19. 推荐运行顺序

## 第一轮：机制诊断

如果已有可靠 step100 checkpoint：

```text
step100 checkpoint
├─ Rmin2 → continue to 150
└─ Rmin4 → continue to 150
```

重点看：

- correction；
- topology；
- late validation；
- wall-clock。

---

## 第二轮：完整训练

若第一轮信号明确：

```text
seed 0 full 150-step
Rmin2 vs Rmin4
```

---

## 第三轮：多 seed

只有在主假设得到支持后，再投入：

```text
seed 0/1/2...
```

做最终论文证据。

这样可以避免在一个未经验证的参数上直接消耗大量 H100 时间。

---

# 20. 推荐最终验收表

| 指标 | 当前 Rmin2 | Rmin4 | 期望方向 |
|---|---:|---:|---|
| planned branch / task | 高 | 低 | ↓ |
| final branch / task | 当前值 | 待测 | 不预设 |
| correction / task | 高 | 低 | **明显 ↓** |
| correction waves | 376 历史总量 | 待测 | **明显 ↓** |
| correction generation time | 2.37 h 历史 | 待测 | **明显 ↓** |
| root fraction | 较低 | 更高 | ↑ |
| branch fraction | 后期约 46.5% | ≤50% 且通常更低 | 控制 |
| late val success | 当前较弱 | 待测 | ≥ baseline |
| total wall-clock | 当前 | 待测 | ↓ |

历史数字仅作为当前已观察到的诊断背景；新的 H100/P1-S baseline 应以 P0 正在采集的数据为准。

---

# 21. 最终推荐

P1-A 推荐只做一个参数修改：

\[
\boxed{
\texttt{algorithm.bace.min\_natural\_roots}=4
}
\]

在 \(B=8\) 下自然得到：

\[
\boxed{
Q_{\max}=4.
}
\]

它的目的不是宣称：

> BACE 理论上就应该永远 4 roots + 4 branches。

而是验证一个已经由现有训练日志强烈提示的机制：

\[
\boxed{
\text{当前后期从 2 roots 开始规划最多 6 branches，
可能过度依赖 family competence，
低估了真实 branch capacity 所需的 natural-root evidence。}
}
\]

如果 Rmin4 同时带来：

\[
\text{correction}\downarrow,
\qquad
\text{wall-clock}\downarrow,
\qquad
\text{late validation}\uparrow\text{ 或不降},
\]

则应把下一阶段重点放在：

\[
\boxed{
\text{从固定 }R_{\min}\text{ 诊断，升级到正式的 capacity-aware root/branch controller。}
}
