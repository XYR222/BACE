# 离线分析结果索引

## A1 Pairwise feedback audit

[`A1_pairwise_feedback/report.md`](A1_pairwise_feedback/report.md) 覆盖 2,400 个 task，其中 1,130 个 `Q>=3` task。约 45.58% 的 task 在第一 pair 后会改变后续 allocation，但平均预期收益很小；当前工程结论为 **HOLD / ABLATION ONLY**。

## Threshold sensitivity

[`Pairwise_Threshold_Sensitivity/report.md`](Pairwise_Threshold_Sensitivity/report.md) 对 4,520 个 task-threshold state 做精确离线重算：

- `0.005` 是首个保守在线候选，离线 BERV retention 约 101.69%；
- `0.0075` 是更激进的第二候选，retention 约 94.76%，fallback 压力更大；
- `0.01` 和 `0.015` 不进入首轮在线实验。

这里的 retention 是 acquisition objective 的 posterior-predictive 比值，不是 validation success rate，也不证明训练效果提升。下一步仍是同一 step-100 checkpoint 上的 C0 Full、C1 Pairwise-Fixed、C2 Pairwise-Stopping 在线 smoke。

`*.parquet` 是可重新生成的 task-level 中间产物，未纳入 Git；CSV、JSON、PNG/SVG 和 Markdown 摘要用于快速复核。
