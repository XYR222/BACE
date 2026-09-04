# C0 `R_min=4` 与旧 2 卡 `R_min=2`：训练数据组成和容量修正对比

日期：2026-09-04  
范围：Qwen2.5-1.5B-Instruct、2×H100、ALFWorld、seed 0、150 个 policy update、Exact Batch-ERV、packed root、selected-worker branch。

## 结论

这次 C0 / `R_min=4` 训练完整且其 BACE 账目自洽：150 个 update 共恰好形成 `150 × 16 × 8 = 19,200` 个最终 leaf，所有 update 的 `batch_erv_exact=1` 和 `staged_root_batching_packed=1`，所有有 branch 的 update 都使用 selected-worker 执行。step 150 artifact 为 `complete`，最终 batch 为 83 root + 45 branch，严格等于 128 个 leaf。

相对旧的 2 卡 `R_min=2` 主 run，新 run 的主要数据组成变化是：

- 最终 root 总数从 **13,269** 增至 **14,907**（+1,638）；branch 从 **5,931** 降至 **4,293**（−1,638）。leaf 总数两者均为 19,200，因此这不是样本数变化，而是 root/branch 配比变化。
- `R_min=4` 在全程平均每个 task 有 6.211 root、1.789 branch；旧 `R_min=2` 为 5.529 root、2.471 branch。
- 后期 capacity 不足时，`R_min=4` 的校正 root 数明显较少：总计 847，对照为 1,772（−52.2%）；发生校正的 update 为 72，对照为 84。
- 直接随 branch 数下降的成本也显著下降：branch replay job 从 5,931 降至 4,293，机械 replay 从 88,562 steps / 34.17 分钟降至 64,892 steps / 13.90 分钟。
- 最终 validation：`R_min=4` 为 **85.94%**，旧 `R_min=2` 为 **86.72%**，差 −0.78 个百分点。单 seed、采样 validation 下，这不足以单独证明性能差异；但它说明较少 branch 并没有在本次 run 中带来更高最终 validation。

这不是严格的“只改 `R_min`”因果消融。两次运行的命令中，除 `R_min=2→4` 和新 run 显式写出 `capacity_correction_batch_size=1`、TensorBoard logger 外，训练主超参数一致；但两份源码 manifest 的 `coordinator.py`、`advantage.py`、environment/rollout 相关文件 SHA-256 有差异，且旧 run 没有记录 correction wave 与总 rollout wall time。这些差异不影响本报告所列 trace 内部计数的真实性，但限制了对 validation 差异的因果归因。

## 数据来源与口径

| 项目 | `R_min=4` C0 | 旧 `R_min=2` |
| --- | --- | --- |
| run | `bace_c0_rmin4_corr1_full150_seed0_20260902` | `bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry` |
| artifact | `.../bace_artifacts/bace_c0_rmin4_corr1_full150_seed0_20260902` | `.../bace_artifacts/bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry` |
| resolved command | run metadata job `3447359` | run metadata job `3233743` |
| 最小自然 root | 4 | 2 |
| 总 leaf budget/task | 8 | 8 |
| max branch/anchor | 2 | 2 |
| BERV threshold | 0.005 | 0.005 |
| history prior / decay | Beta(0.2, 1.8) / 0.8 | Beta(0.2, 1.8) / 0.8 |

下文的“root/branch”指每个 update 最终进入 PPO 的 logical trajectory 数，不是 token 数。`planned` 是首次全局 BERV 分配后的目标；`correction` 是 branch 实际容量不足时需要补入的 root。一次 correction wave 可以并行生成多个 task 的 root，因此“校正 root 数”和“wave 数”不是同一概念。

旧 run 的 trace 没有写入 `count/capacity_correction_waves` 和 `time/rollout_total` 这两个后来加入的 meter；表内以 `—` 表示“未记录”，绝不应读成 0。

## 全程汇总

| 指标 | `R_min=4` | 旧 `R_min=2` | 差异（R4 − R2） |
| --- | ---: | ---: | ---: |
| 最终 root | 14,907 | 13,269 | +1,638 |
| 最终 branch | 4,293 | 5,931 | −1,638 |
| 最终 leaf | 19,200 | 19,200 | 0 |
| 平均 root/task/update | 6.211 | 5.529 | +0.682 |
| 平均 branch/task/update | 1.789 | 2.471 | −0.682 |
| 初始计划 root | 14,060 | 11,497 | +2,563 |
| 初始计划 branch | 5,140 | 7,703 | −2,563 |
| capacity correction root | 847 | 1,772 | −925（−52.2%） |
| 有校正的 update | 72 / 150 | 84 / 150 | −12 |
| 平均 correction/task/update | 5.647 | 11.813 | −6.166 |
| branch replay jobs | 4,293 | 5,931 | −1,638 |
| 机械 replay environment steps | 64,892 | 88,562 | −23,670 |
| 机械 replay wall time | 833.9 s（13.90 min） | 2,050.0 s（34.17 min） | −59.3% |
| branch environment steps | 69,185 | 94,493 | −25,308 |
| root / branch / mixed 训练成功率（pooled） | 49.9% / 63.8% / 53.0% | 49.1% / 65.1% / 54.0% | +0.8 / −1.3 / −1.0 pp |
| step 150 validation success | 85.94% | 86.72% | −0.78 pp |

`R_min=4` 的 capacity correction root 与 `corrections_total` 都为 847，这是因为这条 C0 使用的校正批次大小是 1：每一个缺失 leaf 由一条 fallback/correction root 补齐。它不表示 847 次串行 GPU 调用；同一 wave 中会合并多个 task 的待补 root。

## 分阶段数据组成与校正

每格的 root/branch 是“每 task、每 update”的平均值；校正为“每 update”的全部 16 个 task 合计。初始 deficit 是首次 branch 计划落地前、每 task 平均缺失的 branch 容量。训练成功率是该阶段所有 episode 的 pooled 结果。

| steps | run | root/task | branch/task | correction root/update | correction waves/update | 初始 deficit/task | root / branch / mixed success | replay s/update |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1–25 | R4 | 7.845 | 0.155 | 0.000 | 0.000 | 0.000 | 14.3 / 21.0 / 14.4% | 1.7 |
| 1–25 | R2 | 7.688 | 0.313 | 0.000 | — | 0.000 | 16.4 / 36.8 / 17.2% | 2.7 |
| 26–50 | R4 | 7.345 | 0.655 | 0.000 | 0.000 | 0.000 | 27.9 / 35.5 / 28.5% | 3.8 |
| 26–50 | R2 | 6.888 | 1.113 | 0.000 | — | 0.000 | 24.5 / 43.4 / 27.2% | 5.0 |
| 51–75 | R4 | 6.317 | 1.683 | 0.400 | 0.400 | 0.025 | 49.0 / 57.9 / 50.9% | 6.1 |
| 51–75 | R2 | 5.997 | 2.002 | 1.960 | — | 0.152 | 45.6 / 53.4 / 47.5% | 11.7 |
| 76–100 | R4 | 5.250 | 2.750 | 2.520 | 1.800 | 0.165 | 65.7 / 64.5 / 65.3% | 7.2 |
| 76–100 | R2 | 3.967 | 4.032 | 7.960 | — | 0.492 | 70.8 / 65.6 / 68.2% | 21.8 |
| 101–125 | R4 | 5.242 | 2.757 | 12.600 | 3.560 | 0.735 | 82.3 / 69.4 / 77.8% | 7.6 |
| 101–125 | R2 | 4.037 | 3.962 | 22.280 | — | 1.367 | 87.1 / 71.2 / 79.2% | 21.7 |
| 126–150 | R4 | 5.268 | 2.732 | 18.360 | 3.960 | 1.107 | 87.0 / 70.4 / 81.3% | 7.0 |
| 126–150 | R2 | 4.595 | 3.405 | 38.680 | — | 2.315 | 92.8 / 73.9 / 84.8% | 19.1 |

阶段性图景很清楚：

1. **step 1–50**：两条 run 均几乎没有容量不足。较高的 `R_min` 已经让 R4 的 branch 密度略低，但尚未发生 correction；这部分差异主要来自预算约束，而非 replay/correction。
2. **step 51–75**：模型能力提高后，BERV 逐渐计划更多 branch。R4 首次出现少量 correction（0.4 root/update），R2 已为 1.96。此时 R4 的 branch 是 1.683/task，R2 为 2.002/task。
3. **step 76–100**：差异开始扩大。R2 初始计划约 4.032 branch/task，但真正容量不足平均需补 7.960 条 root/update；R4 的计划与最终分配较温和（2.750 branch/task，补 2.520 root/update）。R4 replay 时间只有 7.2 s/update，R2 为 21.8 s/update。
4. **step 101–150**：两条均进到高 competence 区，容量是主要约束。R2 的 `initial deficit` 上升到 2.315/task，R4 是 1.107/task；R2 最后 25 个 update 每次平均补 38.68 条 root，而 R4 补 18.36 条。R4 的分支数稳定在约 2.7/task，R2 则由 3.96 降至 3.41/task，说明低 root 下限虽然给出了更激进的初始 branch 计划，但后期可实际兑现的 branch 也受容量制约。

## 容量修正的 update 分布

| 指标 | `R_min=4` | 旧 `R_min=2` |
| --- | ---: | ---: |
| 无 correction 的 update | 78 | 66 |
| 有 correction 的 update | 72 | 84 |
| 单 update correction 中位数 | 0 | 4 |
| 单 update 最大 correction root | 33 | 54 |
| branch=0 的 update | 10 | 5 |
| 有 branch 的 update | 140 | 145 |
| 单 update branch 最小–最大 | 0–56 | 0–81 |
| 1–15 branch 的 update | 39 | 34 |
| 16–31 branch 的 update | 21 | 20 |
| 32–47 branch 的 update | 55 | 19 |
| 48–63 branch 的 update | 25 | 42 |
| 64 以上 branch 的 update | 0 | 30 |

这说明 `R_min=4` 不是简单地“每 task 多采两条 root”。其影响更早地改变了可分配的 global branch quota：初始全局计划就少 2,563 条 branch，后续因 branch capacity 不足而需要的 correction 又少 925 条。因而它同时降低了 branch evidence、replay 成本和极端 correction 峰值。

## 关键 update 对照

| step | run | 最终 root | 最终 branch | 初始计划 root / branch | correction root | correction wave | mixed success | replay s |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | R4 | 119 | 9 | 119 / 9 | 0 | 0 | 32.8% | 4.7 |
| 50 | R2 | 116 | 12 | 116 / 12 | 0 | — | 26.6% | 4.5 |
| 75 | R4 | 94 | 34 | 91 / 37 | 3 | 3 | 57.8% | 5.5 |
| 75 | R2 | 77 | 51 | 72 / 56 | 5 | — | 55.5% | 18.4 |
| 100 | R4 | 72 | 56 | 70 / 58 | 2 | 2 | 68.0% | 9.4 |
| 100 | R2 | 47 | 81 | 41 / 87 | 6 | — | 75.0% | 33.1 |
| 125 | R4 | 77 | 51 | 71 / 57 | 6 | 4 | 80.5% | 10.2 |
| 125 | R2 | 62 | 66 | 40 / 88 | 22 | — | 91.4% | 20.8 |
| 150 | R4 | 83 | 45 | 64 / 64 | 19 | 4 | 87.5% | 6.2 |
| 150 | R2 | 82 | 46 | 32 / 96 | 50 | — | 83.6% | 14.5 |

step 150 尤其能说明二者差别：R2 初始计划是 32 root + 96 branch，但可用 branch 容量不足 50 条，最终变成 82 + 46；R4 初始计划是 64 + 64，只需要补 19 条 root，最终为 83 + 45。两者最后 branch 数接近，但 R4 用更少的后续修正到达该终态，且 replay 时间低 8.3 秒。

## Validation 曲线的可见节点

| step | `R_min=4` | 旧 `R_min=2` | R4 − R2 |
| ---: | ---: | ---: | ---: |
| 5 | 3.1% | 11.7% | −8.6 pp |
| 25 | 23.4% | 26.6% | −3.2 pp |
| 50 | 46.9% | 35.2% | +11.7 pp |
| 75 | 43.8% | 57.0% | −13.2 pp |
| 100 | 74.2% | 78.1% | −3.9 pp |
| 125 | 81.2% | 84.4% | −3.2 pp |
| 150 | 85.9% | 86.7% | −0.8 pp |

validation 使用采样生成，且此处只有一个 seed；曲线交叉并不反常。更可靠的结论是：R4 大幅降低了 branch/replay/correction 开销，并维持了相近的最终 validation（−0.78 pp），但目前没有证据证明它提高了效果。若需要判断“减少的 branch evidence 是否值得”，下一步应在同一份冻结源码、相同 validation protocol 下至少运行多个 seed 的 R2/R4 对照，并保留本报告同一套 trace 聚合口径。

## 复现与审计注意事项

- R4 有完整 `time/rollout_total` 计量，150 step 合计约 506.6 分钟（只含 BACE rollout，不等于整 step wall time）；旧 run 未记录该字段，不能比较总 rollout 耗时。
- replay wall time 的绝对值也会受节点负载与实现版本影响；其方向和 job/step 数的下降一致，因此“R4 replay 工作量更低”是强结论，“精确快 59.3%”则应视作本次硬件环境下的观测值。
- branch success 高于 root success 是条件选择效应：branch 只从已出现的 anchor 前缀展开，root 则覆盖所有初始 episode；不应把两者直接解释为“branch policy 更强”。
- 两条 run 的最终 mixed success、branch replay 均来自 trace artifact；旧 run 的 step 150 trace 已在此前重验为 `ok=true`，尽管其原 Slurm 收尾阶段曾因 trace 后处理退出码被标为失败。
