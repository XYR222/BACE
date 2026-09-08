# C4 2×H100

命令从项目父目录开始：`cd work-BACE/verl-agent-src`。脚本从自己的相对位置推导 `work-BACE/`；本地环境、ALFWorld 或 Conda 根目录不同可设置 `VERL_AGENT_ENV`、`ALFWORLD_DATA`、`CONDA_BASE`。

该 `.sh` 可直接由 `bash` 在任意已获得 2 GPU 的 allocation 中运行；`#SBATCH` 只是在 Slurm 中的可选资源注释。其他调度器不应照搬当前分区或账户。

`run_c4_2gpu.sh` 是完整独立 C4 launcher，不调用其他训练入口。C4 固定 `credit_mode=c4_macro_strict_ancestor`：在保持 C0 origin semantics 的条件下，对 macro 使用 strict-ancestor backup；这与 C0.5 的 family-local mean、C7 的 flat-leaf 和 C8 的 macro+local backup 都不同。

固定共同配置为 Exact/staged/packed、selected-worker、`main_reuse + root_active_executor=true`、leaf budget 8、minimum roots 2、competence threshold 0.5、step advantage 1.0、ERV threshold 0.005、batch `16/128`、response 512、TP=2、PPO `256/32`、LR `1e-6`、KL 0.01、gamma 0.95、150 steps。这里不做参数 sweep。

只运行 seed 0、1、2：

```bash
bash examples/bace_gigpo/c4_tree_credit_2gpu/submit_seeds_0_1_2.sh
```

单独 seed 的修改只涉及 `BACE_SEED` 和唯一 `BACE_RUN_NAME`，例如：

```bash
sbatch --export=ALL,BACE_SEED=2,BACE_RUN_NAME=bace_c4_opt_seed2_YYYYMMDD,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2 \
  examples/bace_gigpo/c4_tree_credit_2gpu/run_c4_2gpu.sh
```

相同 run name 只可用于同一 seed 的 checkpoint 恢复。dry-run：`DRY_RUN=1 BACE_SEED=1 BACE_RUN_NAME=dry_c4_s1 bash examples/bace_gigpo/c4_tree_credit_2gpu/run_c4_2gpu.sh`；检查解析命令中的 C4 credit mode、seed、`main_reuse` 和 active executor。
