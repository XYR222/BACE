# C1 tree credit，C0-compatible capacity correction

本目录是独立的 2×H100 C1 入口。`run_c1_corr1_2gpu.sh` 直接构造训练命令，不调用其他训练 launcher。

- 信用模式：`credit_mode=null`、`tree_credit_mode=o1_local`
- 容量修正：`capacity_correction_batch_size=1`
- 调度：Exact/staged/packed、selected-worker、`main_reuse`、active-root
- 训练：seed 默认 0，train/val 16/128，B=8，Rmin=2，TP=2，PPO 256/32，150 steps

除 capacity correction 从历史 C1 运行时的 4 改为原始 C0 的 1，以及使用新的 run name 外，其余算法与训练参数保持 C1 设置。旧 correction=4 提交器已移出 `examples/bace_gigpo/`，仅在项目归档目录中保留供历史审计。

```bash
cd work-BACE/verl-agent-src
DRY_RUN=1 BACE_RUN_NAME=dry_c1_corr1 bash examples/bace_gigpo/c1_tree_credit_corr1_2gpu/run_c1_corr1_2gpu.sh
sbatch examples/bace_gigpo/c1_tree_credit_corr1_2gpu/run_c1_corr1_2gpu.sh
```

`submit_seeds_0_1_2.sh` 仅是批量提交辅助文件，不会被训练入口调用。
