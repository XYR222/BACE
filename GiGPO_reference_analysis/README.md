# GiGPO reference 分析材料

本目录服务于独立 GiGPO reference/baseline 的数据记录、credit decomposition 和分叉实验设计：

- `GiGPO_ALFWorld_Reference_Run_数据记录与离线分析规范_1.md`：应记录的运行数据和分析口径；
- `GiGPO_ALFWorld_Reference_Run_数据采集与离线分析实现规范_2.md`：采集与分析实现方案。
- `A0_credit_decomposition_2gpu/report.md`：2-GPU GiGPO/BACE 的离线 credit 分解；结论不支持立即降低 `step_advantage_w` 或切换 action-mean。
- `a0_credit_decomposition.py`：上述结果的可复核分析脚本。
- `BACE_A0_GiGPO_BACE_Credit_Decomposition_Execution_Plan.md`：A0 计划与口径。
- `bundle_2026-08-29/README.md`：本地分析 bundle 的说明；其中 Slurm 日志由 `.gitignore` 排除。
- `legacy_float32_reproduction/README.md`：用 seed-0 frozen source 复现 seed 1/2 旧 float32 GiGPO 的合同和边界；2026-08-31 作业仅为已提交排队，尚无完成结论。

这些材料不是当前 BACE 方法规范，也不证明新 GiGPO fork/action-mean 作业已经完成。比较 BACE 与 GiGPO 时，还必须核对模型、数据、seed、GPU/并行、训练参数和源码身份。约 897 MiB 的本地 `tar.gz` 是冗余传输包，不进入 GitHub。当前 BACE 入口和进度请从 [`docs/README.md`](../docs/README.md) 开始。
