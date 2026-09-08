# C8 2×H100

命令从项目父目录开始：`cd work-BACE/verl-agent-src`。脚本从自身位置推导 `work-BACE/`；如本地环境布局不同，用 `VERL_AGENT_ENV`、`ALFWORLD_DATA`、`CONDA_BASE` 覆盖默认路径。

该 `.sh` 可直接由 `bash` 在任意已获得 2 GPU 的 allocation 中运行；`#SBATCH` 只在 Slurm 中生效，其他调度器自行包装资源申请。

`run_c8_2gpu.sh` 是完整独立 C8 launcher，不调用统一或其他方法 launcher。它固定 `credit_mode=c8_macro_local_strict_ancestor`：保留 C0 origin semantics，同时执行 C4 macro strict-ancestor 与 local strict-ancestor incremental backup。

固定共同配置为 Exact/staged/packed、selected-worker、`main_reuse + root_active_executor=true`、leaf budget 8、minimum roots 2、competence threshold 0.5、step advantage 1.0、ERV threshold 0.005、batch `16/128`、response 512、TP=2、PPO `256/32`、LR `1e-6`、KL 0.01、gamma 0.95、150 steps。这里不做参数 sweep。

只运行 seed 0、1、2：

```bash
bash examples/bace_gigpo/c8_tree_credit_2gpu/submit_seeds_0_1_2.sh
```

单独运行时只改 seed 和 run name：

```bash
sbatch --export=ALL,BACE_SEED=1,BACE_RUN_NAME=bace_c8_opt_seed1_YYYYMMDD,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2 \
  examples/bace_gigpo/c8_tree_credit_2gpu/run_c8_2gpu.sh
```

相同 run name 仅用于同 seed 自动恢复。dry-run：`DRY_RUN=1 BACE_SEED=1 BACE_RUN_NAME=dry_c8_s1 bash examples/bace_gigpo/c8_tree_credit_2gpu/run_c8_2gpu.sh`；检查解析命令中的 C8 credit mode、seed、`main_reuse` 与 active executor。
