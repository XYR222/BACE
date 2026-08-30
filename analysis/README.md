# 离线分析脚本

本目录保存可复核的 CPU-only 分析程序，不是在线训练入口：

- `a1_pairwise_feedback_audit.py`：在既有 2-GPU BACE trace 上比较 Full Batch 与 K=2 Pairwise 的重规划行为；
- `pairwise_threshold_sensitivity.py`：对不同 Batch-ERV threshold 重新求解 Exact allocation，并估计 stopping/fallback 压力。

脚本输出位于 [`../analysis_outputs/`](../analysis_outputs/README.md)。其中 task-level parquet 由 `.gitignore` 排除；仓库只保留小型 CSV/JSON、图和报告。离线 posterior-predictive 结果不能替代 H100 在线 validation，也不能证明 Pairwise 优于 Full。
