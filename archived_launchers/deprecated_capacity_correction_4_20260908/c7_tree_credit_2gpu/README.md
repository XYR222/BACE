# C7 2×H100

命令从项目父目录开始：`cd work-BACE/verl-agent-src`。脚本从自身位置推导 `work-BACE/`；如本地环境布局不同，用 `VERL_AGENT_ENV`、`ALFWORLD_DATA`、`CONDA_BASE` 覆盖默认路径。

该 `.sh` 可直接由 `bash` 在任意已获得 2 GPU 的 allocation 中运行；`#SBATCH` 只在 Slurm 中生效，其他调度器自行包装资源申请。

`run_c7_2gpu.sh` 是完整独立 C7 launcher，不调用其他训练入口。C7 固定 `credit_mode=c7_flat_leaf_gigpo`，使用 flat-leaf GiGPO training support/credit 语义；它有意不同于保留 C0 origin semantics 的 C0.5、C4 和 C8。

固定共同配置为 Exact/staged/packed、selected-worker、`main_reuse + root_active_executor=true`、leaf budget 8、minimum roots 2、competence threshold 0.5、step advantage 1.0、ERV threshold 0.005、batch `16/128`、response 512、TP=2、PPO `256/32`、LR `1e-6`、KL 0.01、gamma 0.95、150 steps。这里不做参数 sweep。

只运行 seed 0、1、2：

```bash
bash examples/bace_gigpo/c7_tree_credit_2gpu/submit_seeds_0_1_2.sh
```

若单独运行 seed 1，只改 seed 与 run name：

```bash
sbatch --export=ALL,BACE_SEED=1,BACE_RUN_NAME=bace_c7_opt_seed1_YYYYMMDD,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2 \
  examples/bace_gigpo/c7_tree_credit_2gpu/run_c7_2gpu.sh
```

相同 run name 用于同 seed 恢复，不能混用。先验证：`DRY_RUN=1 BACE_SEED=1 BACE_RUN_NAME=dry_c7_s1 bash examples/bace_gigpo/c7_tree_credit_2gpu/run_c7_2gpu.sh`；检查 metadata 中的 C7 credit mode、seed 和调度字段。
