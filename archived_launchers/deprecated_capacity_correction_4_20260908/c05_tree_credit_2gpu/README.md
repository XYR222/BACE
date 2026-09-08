# C0.5 2×H100

命令从项目父目录开始：`cd work-BACE/verl-agent-src`。脚本自身推导 `work-BACE/`，默认将 `model_down/`、`data/`、`verl-agent/` 视为其同级目录；可用 `VERL_AGENT_ENV`、`ALFWORLD_DATA`、`CONDA_BASE` 覆盖本地布局。

`.sh` 可在任何已获得 2 GPU 的 allocation 中直接用 `bash` 启动；`#SBATCH` 行仅在 Slurm 的 `sbatch` 下生效。其他调度器只需提供等价资源后调用这一个脚本。

`run_c05_2gpu.sh` 是完整独立入口：不 source 或 exec 任何其他 launcher。C0.5 固定 `credit_mode=c0_5_origin_family_local_mean`，即保留 origin 训练支持，但使用 family-local mean credit 的 C0.5 规则；它不是 C0、C4 或 C8。

固定共同配置为 Exact/staged/packed、selected-worker、`main_reuse + root_active_executor=true`、leaf budget 8、minimum roots 2、competence threshold 0.5、step advantage weight 1.0、ERV threshold 0.005、train/val `16/128`、response 512、TP=2、PPO `256/32`、LR `1e-6`、KL 0.01、gamma 0.95、150 steps。这里不做超参数 sweep。

只运行 seed 0、1、2。执行下列命令才会提交三项正式实验：

```bash
bash examples/bace_gigpo/c05_tree_credit_2gpu/submit_seeds_0_1_2.sh
```

单独提交/修改 seed 时，只改 `BACE_SEED` 和同一个 seed 对应的唯一 `BACE_RUN_NAME`：

```bash
sbatch --export=ALL,BACE_SEED=1,BACE_RUN_NAME=bace_c05_opt_seed1_YYYYMMDD,TARGET_STEP=150,SAVE_FREQ=5,MAX_CHECKPOINTS=2 \
  examples/bace_gigpo/c05_tree_credit_2gpu/run_c05_2gpu.sh
```

同一 run name 仅用于恢复同一个中断作业；新 seed 必须新 run name。`DRY_RUN=1 BACE_SEED=1 BACE_RUN_NAME=dry_c05_s1 bash examples/bace_gigpo/c05_tree_credit_2gpu/run_c05_2gpu.sh` 只写 metadata，检查 `resolved_command.sh` 中的 C0.5 credit mode、`env.seed` 和两项调度字段。
