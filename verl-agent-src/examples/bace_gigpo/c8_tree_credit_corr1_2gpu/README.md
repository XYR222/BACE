# C8，C0-compatible capacity correction

本目录是独立的 2×H100 C8 入口。信用模式固定为 `c8_macro_local_strict_ancestor`，容量修正固定为 `capacity_correction_batch_size=1`。

除将历史 C8 运行时的 correction batch 从 4 改为原始 C0 的 1，以及使用新的 run name 外，训练、调度、PPO、模型和环境参数保持 C8 正式配置。旧 correction=4 目录已移出 `examples/bace_gigpo/`，仅在项目归档目录中保留供历史审计。

```bash
cd work-BACE/verl-agent-src
DRY_RUN=1 BACE_RUN_NAME=dry_c8_corr1 bash examples/bace_gigpo/c8_tree_credit_corr1_2gpu/run_c8_corr1_2gpu.sh
bash examples/bace_gigpo/c8_tree_credit_corr1_2gpu/submit_seeds_0_1_2.sh
```
