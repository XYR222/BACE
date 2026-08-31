# GiGPO legacy-float32 多 seed 复现

本目录用于严格复现原 seed-0 GiGPO reference 的旧 float32 normalization，而不是使用当前修复后的 `_stable_group_normalize`：

- `run_legacy_float32_reference.sh` 只从校验过 SHA-256 的 seed-0 frozen source archive 导入训练代码；
- seed 1/2 wrapper 只改变 `REF_SEED`、run ID 和输出目录；
- 普通 checkpoint 每 5 step 保存并最多保留两份；约 19 GiB/份的 recorder milestone 被禁用，以控制存储；
- frozen legacy archive validator 的 float32/float64 mismatch 被保留为审计结果，不作为成功门禁；prefix Replay validator 仍必须通过。

2026-08-31 队列快照：seed 1 job `3351704`、seed 2 job `3351705` 均为 `PENDING (Priority)`。这只证明作业已提交，不代表训练或 validation 已完成。状态变化后应以 `sacct`、run `summary.json`、完整 updates、validation 和 Replay 报告更新事实台账。

这些 wrapper 含 RWTH 账号、路径和 frozen archive checksum，不是跨设备通用入口。frozen source、模型、数据、checkpoint、日志和运行产物均不进入 Git。
