# P1 改进方案与专项验证索引

本目录记录 P1-S 调度优化、P1-A quota/Rmin 诊断，以及 2026-08-29 至 08-30 的 Pairwise/action-mean 方案。这里同时包含“设计稿”“离线结果”和“在线结果”，三者不能混用；项目整体事实仍以 [`docs/01_当前实现状态与未完成事项.md`](../docs/01_当前实现状态与未完成事项.md) 为准。

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
4. [`BACE_Pairwise_Threshold_Sensitivity_Step1-Step4_实施结果_2026-08-30.md`](BACE_Pairwise_Threshold_Sensitivity_Step1-Step4_实施结果_2026-08-30.md)
   - 完成 Pairwise fixed/stopping 实现和 CPU 专项，离线推荐首测 threshold 0.005；
   - 当前结论为 HOLD / ABLATION ONLY，2–5 step H100 在线 smoke 尚未完成。
5. [`BACE_优化调度版_2卡与4卡启动说明_2026-08-31.md`](BACE_优化调度版_2卡与4卡启动说明_2026-08-31.md)
   - 当前 P1-S `main_reuse + active-root` 的 2/4 卡正式入口、seed 和 checkpoint 策略；
   - 文中的 job ID 截至 2026-08-31 仍是排队状态，不是完成证据。

## 当前在线验证方案（未完成）

- [`BACE_Pairwise_当前在线验证与Go_NoGo方案.md`](BACE_Pairwise_当前在线验证与Go_NoGo方案.md)：从同一只读 step-100 checkpoint 对比 C0 Full、C1 Fixed、C2 Stopping；这是下一步方案，不是结果报告。
- `BACE_A1_GiGPO_ActionMean_Experiment_Plan.md`、`BACE_A2_BACE_ActionMean_Experiment_Plan.md`：action-mean 消融方案；代码入口已存在，但不能据此声称在线实验完成。

## 设计材料

- `BACE_GiGPO_P1-S_调度优化修复版_2026-08-27.md`、`BACE_P1-S_Low-Risk_Scheduling_Optimization.md`：S1/S2/S3 的设计与不变量。
- `BACE_GiGPO_P1-A_Conservative_Quota_Remapping_修正版_2026-08-27.md`：conservative quota remapping 设计，尚不等于主线已采用。
- `BACE_P1-A_Rmin4_Qmax4_Diagnostic_Design.md`：Rmin=4 的诊断方案。
- `BACE_A1_FullBatch_to_Pairwise_Offline_Feedback_Audit_Plan (2).md`、`BACE_Pairwise_Exact_BERV_修改方案与当前实现执行清单_2026-08-29 (1).md`、`BACE_Pairwise_Threshold_Sensitivity_具体实施方案 (1).md`：Pairwise 的离线审计、实现和 threshold 设计背景；实际完成度以上述实施结果和当前代码为准。

判断实际实现时，以当前 Python 代码、测试、resolved command 和 artifacts 为准。目录中的 RWTH job ID、路径和 H100 数值只描述对应运行，不是跨设备操作合同。
