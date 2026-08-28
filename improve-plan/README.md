# P1 改进方案与专项验证索引

本目录记录 2026-08-27 至 2026-08-28 的 P1-S 调度优化和 P1-A quota/Rmin 诊断。这里同时包含“设计稿”和“已完成结果”，二者不能混用；项目整体事实仍以 [`docs/01_当前实现状态与未完成事项.md`](../docs/01_当前实现状态与未完成事项.md) 为准。

## 先读结果

1. [`P1-S_调度优化测试结果与正式链准入结论_2026-08-28.md`](P1-S_调度优化测试结果与正式链准入结论_2026-08-28.md)
   - S0–S3 H100 profile、三步 stability、外部 checkpoint 隔离门禁；
   - S3（main pool 复用 + active-root executor）是当前新实验推荐执行配置；
   - 性能数据是代表性工程 profile，不是多次重复后的统计显著性结论。
2. [`P1-A_step148-150专项运行结果与判断_2026-08-28.md`](P1-A_step148-150专项运行结果与判断_2026-08-28.md)
   - 从 step 148 对比 Rmin=2 与受控 Rmin=4 migration；
   - 两条路径均完成 step 149–150、trace `ok=true`；
   - 只覆盖两个训练 step，不能据此把 Rmin=4 升为正式默认。
3. [`P1-A_Rmin4与初始2Root容量修正对比分析_2026-08-28.md`](P1-A_Rmin4与初始2Root容量修正对比分析_2026-08-28.md)
   - 解释 Rmin=4 与原 2-root capacity correction 的关系和观察指标。

## 设计材料

- `BACE_GiGPO_P1-S_调度优化修复版_2026-08-27.md`、`BACE_P1-S_Low-Risk_Scheduling_Optimization.md`：S1/S2/S3 的设计与不变量。
- `BACE_GiGPO_P1-A_Conservative_Quota_Remapping_修正版_2026-08-27.md`：conservative quota remapping 设计，尚不等于主线已采用。
- `BACE_P1-A_Rmin4_Qmax4_Diagnostic_Design.md`：Rmin=4 的诊断方案。

判断实际实现时，以当前 Python 代码、测试、resolved command 和 artifacts 为准。目录中的 RWTH job ID、路径和 H100 数值只描述对应运行，不是跨设备操作合同。
