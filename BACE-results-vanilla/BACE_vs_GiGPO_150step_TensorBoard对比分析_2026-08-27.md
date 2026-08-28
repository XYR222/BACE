# BACE 与 GiGPO 150-step TensorBoard 对比分析

日期：2026-08-27

## 1. 结论摘要

本报告比较以下两个完整的 150-step、seed 0 实验：

- GiGPO：`gigpo-alfworld-reference/runs/seed_0_v2`
- BACE：`bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0`

最重要的结论是：**BACE 并不是从训练开始就弱于 GiGPO，而是前 100 step 基本持平并略占优，105--150 step 才出现稳定落后。**

| 指标 | GiGPO | BACE | BACE - GiGPO |
|---|---:|---:|---:|
| step 150 validation success | 89.84% | 87.50% | -2.34 pp |
| 最佳 validation success | 90.63%（step 135） | 87.50%（step 150） | -3.13 pp |
| step 5--50 validation 均值 | 20.55% | 22.27% | +1.72 pp |
| step 55--100 validation 均值 | 57.19% | 57.66% | +0.47 pp |
| step 105--150 validation 均值 | 83.91% | 77.66% | **-6.25 pp** |
| step 105--150 六类任务宏平均 | 82.69% | 76.17% | **-6.52 pp** |

最终 2.34 个百分点对应 128 个 validation episode 中相差 3 个成功样本（115 对 112）。只看单次最终评测，差距不大，也不足以在单 seed 下证明 BACE 的真实上限更低；但连续 10 次后期评测中 GiGPO 每次都高于 BACE，说明 BACE 的后期学习速度或后期数据配置确实值得重点检查。

现有数据最支持的原因是：

1. BACE 的 branch 比例随着 competence 上升持续增加。step 101--150 平均每个 task group 只有 4.28 个 root、3.72 个 branch，即每步约 68 个独立 root 和 60 个共享前缀 branch；GiGPO 始终是 128 个独立 root。
2. BACE 后期 root 成功率已经与 GiGPO 接近，但 branch 明显更难。step 101--150 的 BACE root 成功率为 84.58%，branch 为 66.64%，GiGPO root 为 85.25%。因此 BACE 的策略并非没有学会任务，而是接近一半 PPO 数据被用于较难、相关性更强的局部分支。
3. 当前 branch 数主要由“模型是否足够 competent”控制，而不是由“branch 是否仍具有最大的边际学习价值”控制。后期 readiness 平均升至 0.879，计划 branch 数升至 5.31，但实际结构容量只能支持 3.72，造成大量 capacity correction 和额外 root wave。
4. branch 共享前缀，实际独立信息量小于相同数量的 root；当前 occurrence-level credit 又把这些相关 occurrence 直接送入 PPO，可能使局部反事实信号在后期相对过重。

因此，下一版最值得做的不是推翻 Exact Batch-ERV，而是：**保留 Exact 全局分配和 selected-worker 执行，给动态 branch budget 加上 realized-capacity/ERV 门控与后期上限，保证至少 4 个自然 root，并对 branch lineage 做权重或有效样本量修正。**

## 2. 数据来源与可比性

### 2.1 TensorBoard 数据

GiGPO event：

```text
/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-alfworld-reference/
  runs/seed_0_v2/analysis/tensorboard_v1/
  events.out.tfevents.1787795121.login23-1.hpc.itc.rwth-aachen.de.468542.0
```

BACE event：

```text
/hpcwork/rwth2089/xsz96350/work-BACE/experiments/
  alfworld-qwen2.5-1.5b-exact/tensorboard/
  bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0/
  events.out.tfevents.1787682125.n23g0002.hpc.itc.rwth-aachen.de.427793.0
```

两份 event 都包含 step 1--150 的训练标量，以及 step 5、10、……、150 共 30 个共同 validation 点。GiGPO 原始归档另外记录了 step 0 validation，但本报告只比较双方共有的 step 5--150。

### 2.2 相同的核心训练参数

双方均使用：

- Qwen2.5-1.5B-Instruct，全参数训练；
- 相同 train/test parquet；
- train batch 16、validation batch 128；
- 每个 task group 的 leaf/group budget 为 8，总训练 leaf 为每步 128；
- ALFWorld horizon 50、response 512；
- learning rate `1e-6`、PPO mini-batch 256；
- KL coefficient 0.01、gamma 0.95；
- GiGPO `mean_std_norm`、step advantage weight 1.0；
- validation 每 5 step，temperature 0.4 且 `do_sample=true`；
- environment seed 0。

BACE 额外启用 Exact Batch-ERV、dynamic/staged/packed、selected-worker、occurrence credit、family history 和 branch replay。

### 2.3 每步训练任务族分布完全一致

从 GiGPO 的 150 份 trajectory parquet 和 BACE 的 150 份 `family_topology_plans.jsonl` 逐 step 核对后：**150/150 个 step 的六类任务数量都完全一致。** 2400 个 task group 的总分布如下：

| 任务族 | task group 数 | 比例 |
|---|---:|---:|
| Pick & place | 534 | 22.25% |
| Pick two & place | 546 | 22.75% |
| Clean & place | 415 | 17.29% |
| Cool & place | 350 | 14.58% |
| Heat & place | 338 | 14.08% |
| Look in light | 217 | 9.04% |

因此，BACE 后期稍弱不能归因于训练任务族比例不同。具体 game 和随机 rollout 仍然会因为分支、reset 次数和模型输出不同而分叉，这是强化学习实验的正常现象。

### 2.4 不能直接做纯系统性能归因的差异

| 项目 | GiGPO | BACE |
|---|---:|---:|
| GPU | 2×H100 | 4×H100 |
| rollout TP | 2 | 1 |
| actor/logprob micro batch / GPU | 32 | 16 |
| actor torch compile | true | false |
| `max_num_batched_tokens` | 8192 | 16384 |
| `max_num_seqs` | 1024 | 128 |

这些差异主要影响吞吐和数值执行路径。它们不太可能单独解释 validation 的系统性后期差距，但意味着本报告的运行时间对比不是严格的“同硬件算法 benchmark”。

## 3. Validation 曲线

### 3.1 全部共同评测点

| Step | GiGPO | BACE | 差值（pp） |
|---:|---:|---:|---:|
| 5 | 7.81% | 10.16% | +2.34 |
| 10 | 11.72% | 12.50% | +0.78 |
| 15 | 9.38% | 21.88% | +12.50 |
| 20 | 16.41% | 19.53% | +3.13 |
| 25 | 14.84% | 21.09% | +6.25 |
| 30 | 21.88% | 23.44% | +1.56 |
| 35 | 21.88% | 28.91% | +7.03 |
| 40 | 20.31% | 31.25% | +10.94 |
| 45 | 39.84% | 18.75% | -21.09 |
| 50 | 41.41% | 35.16% | -6.25 |
| 55 | 38.28% | 42.19% | +3.91 |
| 60 | 35.16% | 43.75% | +8.59 |
| 65 | 45.31% | 48.44% | +3.13 |
| 70 | 50.78% | 60.94% | +10.16 |
| 75 | 46.88% | 55.47% | +8.59 |
| 80 | 63.28% | 52.34% | -10.94 |
| 85 | 67.19% | 67.19% | 0.00 |
| 90 | 75.78% | 63.28% | -12.50 |
| 95 | 72.66% | 71.88% | -0.78 |
| 100 | 76.56% | 71.09% | -5.47 |
| 105 | 69.53% | 67.97% | -1.56 |
| 110 | 73.44% | 71.09% | -2.34 |
| 115 | 80.47% | 72.66% | -7.81 |
| 120 | 85.94% | 78.91% | -7.03 |
| 125 | 85.16% | 77.34% | -7.81 |
| 130 | 83.59% | 76.56% | -7.03 |
| 135 | 90.63% | 75.00% | -15.63 |
| 140 | 89.84% | 85.16% | -4.69 |
| 145 | 90.63% | 84.38% | -6.25 |
| 150 | 89.84% | 87.50% | -2.34 |

共同 30 个点中，BACE 胜 13 次、平 1 次、负 16 次。平均差值仅为 -1.35 pp，但分阶段后可以看到明显的时序结构：BACE 的问题集中在后 50 step，而不是冷启动或中期学习失败。

### 3.2 最终任务族表现

| 任务族 | GiGPO step 150 | BACE step 150 | 差值（pp） |
|---|---:|---:|---:|
| Pick & place | 96.77% | 100.00% | +3.23 |
| Clean & place | 84.21% | 81.25% | -2.96 |
| Cool & place | 92.31% | 80.00% | -12.31 |
| Look in light | 83.33% | 81.82% | -1.52 |
| Heat & place | 100.00% | 90.91% | -9.09 |
| Pick two & place | 75.00% | 84.21% | +9.21 |
| 六类宏平均 | 88.60% | 86.36% | -2.24 |

单个 step 的分任务 validation 样本数较小，且两组 evaluation 的任务实例组成并非严格配对，因此不应仅凭最终点断言 BACE 对某一类任务必然更强。更稳健的 step 105--150 十次评测均值如下：

| 任务族 | GiGPO 后期均值 | BACE 后期均值 | 差值（pp） |
|---|---:|---:|---:|
| Pick & place | 95.33% | 87.46% | -7.87 |
| Clean & place | 85.35% | 77.10% | -8.25 |
| Cool & place | 72.59% | 77.72% | +5.13 |
| Look in light | 76.56% | 66.65% | -9.91 |
| Heat & place | 92.14% | 87.90% | -4.24 |
| Pick two & place | 74.18% | 60.19% | -13.99 |

BACE 在 Cool 类上有稳定优势，但在需要较长组合行为的 Pick two、以及 Look/Clean 类上后期均值较低。这与“分支强化局部决策、但减少完整独立轨迹覆盖”的解释相符，不过仍需多 seed 和对照消融才能确认因果。

### 3.3 `val/text/test_score` 的解释限制

step 150 的 `val/text/test_score` 是 GiGPO 6.507、BACE 5.814，后期均值分别是 5.619 和 4.542。但当前 trainer 对展开后的 occurrence 行求 reward 均值，该指标受 episode 长度和 occurrence 数影响，并不是简单的“每个 episode 平均得分”。ALFWorld 的非视觉 reward 本身是 `10 × won`，因此本文把 `val/success_rate` 作为主要效果指标，不把 `test_score` 当作独立正确率。

## 4. 训练数据语义：为什么 BACE 的后期训练指标更低

### 4.1 Root 与 branch 比例随训练阶段变化

| 训练阶段 | 平均 root / task | 平均 branch / task | 每步 root 总数 | 每步 branch 总数 | branch 占比 |
|---|---:|---:|---:|---:|---:|
| step 1--50 | 7.314 | 0.686 | 117.0 | 11.0 | 8.58% |
| step 51--100 | 5.280 | 2.720 | 84.5 | 43.5 | 34.00% |
| step 101--150 | 4.278 | 3.723 | 68.4 | 59.6 | **46.53%** |
| 全程 | 5.624 | 2.376 | 90.0 | 38.0 | 29.70% |

150 step 总计：

- BACE 自然 root leaf：13,497；
- BACE branch leaf：5,703；
- GiGPO 自然 root leaf：19,200；
- BACE 的 5,703 个 branch 请求全部通过 replay validation，`skipped=0`。

两者每步都训练 128 个 leaf，但 BACE 有 29.7% 的 leaf 共享某条 root 的前缀。数量相同不代表统计独立性相同；从完整任务初态出发的独立探索覆盖因此下降。

### 4.2 分阶段成功率

| 阶段 | GiGPO root | BACE root | BACE branch | BACE mixed |
|---|---:|---:|---:|---:|
| step 1--50 | 20.88% | 19.51% | 40.08% | 21.13% |
| step 51--100 | 55.75% | 51.57% | 55.15% | 51.39% |
| step 101--150 | **85.25%** | **84.58%** | **66.64%** | **76.38%** |

后期 BACE root 成功率和 GiGPO 已非常接近，说明模型的完整任务能力并未崩溃。BACE mixed 指标偏低，主要因为 branch 是在选中的 anchor 上尝试替代动作，本来就是更难、更有争议的反事实样本。后期 root/branch 成功率相差约 18.0 pp，并不自动等于 branch 逻辑错误；它表明这些样本和普通 root 不是同分布数据。

这也解释了一个看似矛盾的现象：BACE 的训练 root 指标很好，但 validation 仍稍弱。可能的机制是 branch 梯度占比过高，牺牲了一部分对完整任务分布的持续覆盖与优化。

## 5. 优点与不足

### 5.1 BACE 的优点

1. **早期和中期样本效率有优势。** step 5--50 和 55--100 的平均 validation 分别领先 1.72 pp 和 0.47 pp，说明分支信用并非无效。
2. **最终效果接近 GiGPO。** 最终只少 3/128 个成功 episode，同时 Pick & place、Pick two 的最终点更高。
3. **分支链路执行完整。** 5,703/5,703 个请求通过，未出现 replay skip；Exact、packed、selected-worker 标志均在训练中生效。
4. **family history 只使用自然 root。** branch outcome 不回写 competence history，避免控制器被自身选择偏差污染。
5. **active compaction 确实节省执行。** 全程避免了 145,285 个 inactive dense-equivalent sequence，保留 selected-worker 是正确的。
6. **Exact 全局求解本身很快。** solver 平均约 0.328 ms，不是系统瓶颈，也没有必要为了速度退回近似分配。
7. **可审计性强。** BACE 记录 root、anchor、branch、posterior、history、replay 和 trainable occurrence，远强于只看聚合曲线的黑盒训练。

### 5.2 GiGPO 的优点

1. **后期 validation 更稳定。** step 105--150 的十次评测全部高于 BACE，平均领先 6.25 pp。
2. **独立完整轨迹覆盖更大。** 每步 128 个自然 root，没有共享前缀造成的有效样本量折损。
3. **最终在四类任务上更好。** 尤其是 Heat 和 Cool 的最终点，以及 Pick two 的后期均值。
4. **系统成本更低。** 即便只有 2 张 H100，150 step 仍比 BACE 的 4 卡运行更快。

### 5.3 BACE 当前的主要不足

1. **branch budget 在后期过于激进。** competence readiness 越高，branch 越多；但策略成熟后，额外局部分支的边际价值未必继续上升。
2. **独立 root 覆盖下降。** 后 50 step 约一半 leaf 是共享前缀分支，这可能降低对新初态、新长程路径和完整错误模式的覆盖。
3. **相关样本没有显式降权。** 同一 root/anchor 派生的多个 occurrence 在 PPO 中仍按 occurrence 参与，名义 batch size 128 可能高估有效样本量。
4. **当前 controller 不预测可实现结构容量。** 后期每 task 计划 5.31 个 branch，实际只能得到 3.72 个，说明 readiness 到 quota 的映射与真实 anchor capacity 脱节。
5. **branch suffix 执行仍有 tail inefficiency。** active efficiency 全程平均约 0.413；selected-worker 已避免无效 dense worker，但不同分支长度造成的尾部空槽仍然很多。
6. **单 seed、随机 validation。** `temperature=0.4` 且 sampling 开启，最终 3 个 episode 的差距可能包含较大随机波动。

## 6. 系统效率对比

### 6.1 总耗时

| 指标 | GiGPO（2 GPU） | BACE（4 GPU） | BACE / GiGPO |
|---|---:|---:|---:|
| 150 step 累计 `timing_s/step` | 11.34 h | 17.30 h | 1.53× |
| 进度条 wall time | 11:28:41 | 17:19:30 | 1.51× |
| 平均 step | 272.2 s | 415.1 s | 1.53× |
| 平均 generation | 147.4 s | 329.2 s | 2.23× |
| 平均 actor update | 57.8 s | 36.3 s | 0.63× |
| 估算训练 GPU-hours | 22.68 | 69.18 | **3.05×** |

4 卡使 BACE 的 actor update 更快，但无法抵消 root/branch 多波生成，因此总 wall time 仍更长。从当前实现看，BACE 不能宣称系统效率优于 GiGPO；它的价值必须由更好的学习效果或更强的信用质量来证明，而本次单 seed 结果暂未达到这一点。

### 6.2 BACE generation 时间分解

| 项目 | 累计时间 | 说明 |
|---|---:|---|
| 初始 planned root generation | 4.27 h | 每 step 一次 packed planned wave |
| capacity-correction root generation | 2.37 h | 77 个 step 出现，共 376 个附加 wave |
| branch suffix generation | 6.06 h | 146 个有 branch 的 step |
| mechanical replay validation | 35.6 min | 平均约 14.6 s/branch-active step |
| reset-key probe | 7.8 min | 平均 3.12 s/step |
| Exact capacity/global allocation 计算 | 秒级总量 | 不是瓶颈 |

真正昂贵的不是 Exact solver，也不是 replay 前缀机械恢复，而是：

- 后期 readiness 先计划过多 branch；
- 真实 anchor capacity 不够；
- collector 每轮只把每个 deficient task 增加一个 root，再重新评估；
- 最终产生多次 capacity-correction root wave；
- branch suffix 长短不一，selected-worker wave 出现尾部利用率损失。

## 7. 为什么当前 BACE 稍弱：证据分级

### 7.1 高置信度：后期 branch 比例与性能分化同时发生

step 1--50 的 branch 占比仅 8.6%，BACE validation 略高；step 51--100 branch 占比升到 34.0%，两者仍基本相等；step 101--150 branch 占比达到 46.5%，BACE validation 平均落后 6.25 pp。

在 30 个 validation 点上，BACE final branch quota 与 `BACE - GiGPO` validation 差值的描述性相关系数约为 -0.484。该相关关系受到训练阶段共同趋势的混杂，不能单独证明因果，但与“后期 branch 过量”假设一致。

### 7.2 高置信度：quota 预测与真实结构容量不匹配

| 阶段 | planned branch / task | final branch / task | correction / task |
|---|---:|---:|---:|
| step 1--50 | 0.686 | 0.686 | 0.000 |
| step 51--100 | 2.900 | 2.720 | 0.180 |
| step 101--150 | 5.309 | 3.723 | 1.586 |
| step 126--150 | 5.525 | 3.605 | 1.920 |

后 25 step 中，控制器平均每个 task 先计划 5.53 个 branch，之后又因容量不足改回 1.92 个 root。这个过程既增加 wall time，也说明 controller 把 competence 误当成了可分支容量。

### 7.3 中高置信度：branch 数据与最终 evaluation 分布存在偏移

branch 专门位于有可比 alternative action 的 anchor，目标是估计局部信用；validation 则从完整初态评估整条任务。BACE 训练后期 branch 成功率明显低于 root，表明 branch 在集中采样困难局部决策。这种 hard-example mining 可能提升某些局部能力，却减少了完整 root 的长程覆盖。

### 7.4 中等置信度：occurrence credit 对共享 lineage 的权重偏大

当前 `local_credit_mode=occurrence`。共享 root/anchor 的多个分支和多步 occurrence 会进入同一 PPO batch，但没有按 lineage 计算 effective sample size，也没有保证 root 与 branch 的总 loss 权重固定。随着 branch 比例升高，局部项可能相对 GiGPO 的 macro episode 信号越来越强。

### 7.5 可能是收敛速度，而不一定是最终上限

GiGPO 在 step 135 达到峰值 90.63%，之后基本平台；BACE 的最好点恰好是最后的 step 150（87.50%），曲线仍在上升。因此目前证据更准确的表述是“BACE 在 150-step 预算下后期收敛较慢”，而不是“BACE 的最终可达性能必然更低”。

### 7.6 不能排除随机性

只有一个 seed，validation 又使用随机采样。最终差距只有 3/128 个 episode。后期十个点的一致方向提高了问题存在的可信度，但正式算法结论至少需要 3 个训练 seed，或对同一 checkpoint 做多次固定任务、固定解码种子的 paired evaluation。

## 8. 建议的改进方案

### 8.1 P0：先把比较做成可发表的公平实验

1. 固定一套 validation game IDs，并让 GiGPO/BACE 对每个 checkpoint 使用同一批 game、同一生成 seed；同时保留 stochastic evaluation 作为补充。
2. 至少运行 3 个 seed，报告最终值、后 10 次评测均值、AUC 和置信区间。
3. 同时报告四种横轴：policy step、leaf episode、独立 root episode、GPU-hours。BACE 与 GiGPO 的“相同 step”不等于相同独立探索量或相同计算量。
4. 修复两组归档完整性问题后再做最终结论：BACE trace 的截断 JSONL，GiGPO advantage 后置校验的不一致。

### 8.2 P1：改 branch budget，而不改 Exact 分配核心

推荐把当前 quota：

```text
round((B - R_min) × readiness)
```

改为同时受以下三项约束：

```text
quota = min(
    readiness_quota,
    realized_capacity_lower_bound,
    erv_qualified_candidate_count,
    branch_fraction_cap
)
```

具体建议：

- `R_min` 从 2 提高到 4，或设置 `branch_fraction_cap=0.5`；
- 用各 family 最近若干 step 的实际 `final_branch_count`/capacity 做 EWMA 或较低分位数，防止计划 5.5、实际只能执行 3.6；
- 只给 marginal ERV 高于 threshold 的候选分配 quota，不为满足 readiness 推导出的数量而强行补满；
- 增加 hysteresis，避免 quota 随单步 posterior 抖动；
- 在训练后期对 branch budget 做退火，例如 step 100 后把最大 branch 从 4 缓慢降到 2--3，检验是否恢复独立 root 覆盖。

最小消融矩阵建议：

| 实验 | Root 下限 | Branch 上限 | 其他 |
|---|---:|---:|---|
| 当前 BACE | 2 | 6 | readiness-only quota |
| BACE-R4 | 4 | 4 | 其余不变 |
| BACE-capacity | 2 | 6 | 加 realized-capacity gate |
| BACE-R4-capacity | 4 | 4 | 两者同时启用 |
| BACE-anneal | 2→4 | 6→3 | step 75--125 退火 |

### 8.3 P1：修正相关样本权重

保留 packed root、branch plan 和 PPO 数据语义的前提下，可以增加 lineage-aware loss weight：

- 同一 root 派生的所有 branch 总权重不超过一个独立 root 的若干倍；
- 每个 task group 内先分别归一化 root 与 branch，再用显式系数混合；
- 记录 root/branch 的 macro/local advantage 方差、均值绝对值和 PPO loss 贡献；
- 比较 `local_credit_mode=occurrence` 与已有的 `action_mean`；
- 对 `step_advantage_w` 做 `{0.25, 0.5, 1.0}` 小规模消融，避免 BACE 的局部项因样本相关性而压过 macro 项。

一个稳妥的初始版本是保持 macro advantage 不变，只让 branch local contribution 乘以：

```text
w_lineage = 1 / sqrt(number_of_trainable_occurrences_from_same_anchor)
```

具体形式需要通过 advantage 方差与 gradient norm 审计后确定，不建议未经消融直接用于正式主结果。

### 8.4 P1：让 quota 反映“学习价值”，而不只是 competence

当前 readiness 回答的是“模型是否可能胜任任务”，没有回答“此处继续 branch 是否比新 root 更值得”。可以加入：

- family posterior 的学习进展（近期成功率斜率）；
- local posterior entropy 或 ERV 总量；
- anchor 的有效 alternative-action 数；
- branch 与 root 的近期 advantage 方差/有效梯度贡献；
- 每单位 wall time 或生成 token 的 expected ERV。

当 family 已经高 competence、posterior 很确定且 branch marginal ERV 低时，应把预算退回 root；当某个 family 仍有高不确定的关键 anchor 时，才继续高比例 branch。

### 8.5 P2：降低系统开销

1. **预测 capacity，减少 correction wave。** 用历史 realized capacity gate 是同时改善算法与性能的首选。
2. **批量 reserve root。** 若必须修正，可一次为高风险 deficient task 生成 2--3 个 reserve root，再只把需要的 leaf 放入训练；需要明确记录未训练的探测成本。
3. **branch wave 动态补位。** 当前 worker 槽位在短 branch 结束后会形成尾部空洞。可让未开始的 branch 在槽位释放时进入，而不是等待整 wave 结束。
4. **保留 selected-worker active compaction。** 当前已避免 145,285 个 inactive sequence，不应退回 dense executor。
5. **不要优先优化 Exact solver 或机械 replay。** solver 是毫秒级，replay 全程仅约 35.6 分钟；主要时间在 root correction 和 suffix generation。

## 9. 完整性与解释边界

### 9.1 BACE

- 训练确实达到 `training/global_step:150`；
- step 150 validation 为 87.5%；
- step 150 checkpoint 已保存；
- step 150 `summary.json` 为 `status=complete`；
- Slurm 作业 `3116958` 最终状态为 `FAILED/1:0`，原因发生在训练结束后的 trace validator：某个超大 JSONL 行出现 `JSONDecodeError: Unterminated string`；这不是 PPO 或模型训练失败，但意味着严格 trace 验收尚未完成。

### 9.2 GiGPO

- 150/150 个 update 和 31 个 validation 归档均存在；
- 训练进度达到 100%，wall time 为 11:28:41；
- `final_validation.json` 为 `ok=false`：从 update 74 到 150 的 56 个 update 中共有 2406 条 episode/step advantage 后置重算不一致；
- 这不等于在线 PPO 必然算错，也不否定 TensorBoard success curve，但说明 recorder/checker 与在线 advantage 之间尚未做到 bitwise/容差内一致，严格复现置信度需要修复后确认。

### 9.3 结论强度

可以确认：

- 本次 seed 0、150-step 预算下，GiGPO 最终与后期 validation 略强；
- BACE 前中期不弱，差距主要产生在 branch 比例升高的后期；
- BACE 的主要系统成本来自额外 root wave 和 branch suffix，而不是 Exact solver/replay；
- 训练任务族分布不是差距来源。

暂时不能确认：

- BACE 的真实最终上限低于 GiGPO；
- 某个单独任务族上的差异具有统计显著性；
- 只调整一个参数就一定能消除后期差距。

## 10. 推荐下一步

如果只做一个范围受控的下一轮实验，推荐：

1. 保留 Exact Batch-ERV、packed root、selected-worker、strict identity、occurrence trace 和 family-history checkpoint；
2. 增加 `min_natural_roots=4`；
3. quota 加 realized-capacity gate，最大 branch 不超过 4；
4. 暂不修改 advantage 公式，以便把效果变化归因于采样拓扑；
5. 先跑 30-step A/B smoke，检查 validation、branch/root 数量、capacity correction wave、GPU time；
6. 若后期趋势改善，再跑 3-seed 150-step；
7. 第二阶段才消融 lineage weight、`action_mean` 和 `step_advantage_w`。

这个顺序最容易回答核心问题：**BACE 稍弱究竟是 branch 思想本身的问题，还是当前后期 branch budget 过量造成的问题。** 现有证据更支持后者。
