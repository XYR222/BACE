# C0 optimized 2×H100

本文的命令都假设从项目父目录运行，并先进入 `work-BACE/verl-agent-src`。启动器从自身位置推导 `work-BACE/`；默认假定 `model_down/`、`data/`、`verl-agent/` 与 `work-BACE/` 同级。若环境或 ALFWorld 位置不同，分别设置 `VERL_AGENT_ENV=...`、`ALFWORLD_DATA=...`、`CONDA_BASE=...` 后再运行。

`run_c0_optimized_2gpu.sh` 是普通 Bash 脚本，开头的 `#SBATCH` 行被 `bash` 忽略。因此在任何已获得 2 GPU 的调度器 allocation 中都可直接 `bash examples/.../run_c0_optimized_2gpu.sh`；只有在 Slurm 上才使用 `sbatch`。其他集群应自行申请资源或把这份 `.sh` 包进本地调度器脚本，不能照搬 `c23g/rwth2089`。

这是独立的 C0（`credit_mode=current`）启动目录；`run_c0_optimized_2gpu.sh` 不 source 或 exec 其他 launcher。调度固定为 `main_reuse + root_active_executor=true`，其余训练参数与此前完成的 C0 2 卡三 seed run 一致：batch `16/128`、response 512、horizon 50、LR `1e-6`、PPO mini/micro `256/32`、TP=2、token/seq `8192/1024`、KL `0.01`、gamma `0.95`、150 steps。

本目录的 C0 基准为：`C0_COMPETENCE_THRESHOLD=0.5`、`C0_STEP_ADVANTAGE_W=1.0`、`C0_BATCH_ERV_THRESHOLD=0.005`、`BACE_SEED=0`，以及固定的 `capacity_correction_batch_size=1`。前者是 family competence readiness 阈值；第二项是步骤级 local advantage 权重；第三项是 Exact Batch-ERV 的最小有效阈值。最后一项不是 sweep 参数：当实际 anchor 容量不足时，每一轮至多把一个仍计划的 branch slot 转为 natural root；这是历史 2 卡优化 C0 的 root-correction 语义。

默认 Slurm 资源也恢复为历史 2 卡 C0 的 `c23g / rwth2082 / 2×H100 / 32 CPU / 244 GiB / 23:30:00`。单独运行一个需要恢复能力的实验时，启动器默认每 5 step 保存并轮转最近 2 份 checkpoint，同时永久保留 step `10,75,100,140`；这些是运维设置，不参与 rollout 或梯度计算。

本目录的 36 组参数 sweep 是相互独立的完整训练。批量提交器为避免 checkpoint 占满共享盘，固定传入 `SAVE_FREQ=-1` 和 `MILESTONE_CHECKPOINT_STEPS=none`，因此这些 sweep 作业不会保存或恢复 checkpoint；中断后只能从 step 0 重跑。

即使不保存 checkpoint，当前完整 C0 run 的 BACE artifact 与 rollout 通常仍约占 `15--17 GiB`。36 组建议至少预留 `650 GiB` 可用空间并留出运行余量；当前目标盘若达不到该门槛，不要一次提交全部实验，应先换到更大的物理输出盘或分批运行并安全归档。W&B、TensorBoard、训练日志和 artifact 均应落在 `EXP_ROOT`，不要物理写入源码目录。

`submit_c0_grid_seed0.sh` 已明确列出完整 full-factorial 网格：competence `{0.4,0.5,0.6,0.7}` × step weight `{1.0,0.8,1.2}` × ERV threshold `{0.005,0.0025,0.0075}`，共 36 个 seed-0 作业。除了这三个轴，所有训练/算法参数均固定为上述历史 C0 值，包括 correction batch=1。它只是提交器，必须由用户显式执行：

```bash
cd work-BACE/verl-agent-src
bash examples/bace_gigpo/c0_optimized_2gpu/submit_c0_grid_seed0.sh
```

可通过 `GRID_RUN_TAG` 设置本批次统一且唯一的后缀；默认值为 `20260916`。不要使用已经存在输出的 tag：

```bash
GRID_RUN_TAG=handoff01 \
  bash examples/bace_gigpo/c0_optimized_2gpu/submit_c0_grid_seed0.sh
```

单独运行一个组合时，传入三个环境变量即可；例如 `0.6/0.8/0.0075`：

```bash
sbatch --partition=c23g --account=rwth2082 --time=23:30:00 \
  --export=ALL,BACE_SEED=0,C0_COMPETENCE_THRESHOLD=0.6,C0_STEP_ADVANTAGE_W=0.8,C0_BATCH_ERV_THRESHOLD=0.0075,BACE_RUN_NAME=bace_c0_opt_seed0_c0p6_w0p8_tau0p0075_YYYYMMDD,TARGET_STEP=150,SAVE_FREQ=-1,MAX_CHECKPOINTS=2,MILESTONE_CHECKPOINT_STEPS=none \
  examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh
```

先做 dry-run：

```bash
DRY_RUN=1 BACE_SEED=0 C0_COMPETENCE_THRESHOLD=0.6 C0_STEP_ADVANTAGE_W=0.8 C0_BATCH_ERV_THRESHOLD=0.0075 BACE_RUN_NAME=dry_c0_grid \
  bash examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh
```

检查 `experiments/alfworld-qwen2.5-1.5b-exact/run_metadata/<run>/.../resolved_command.sh` 中的三项和 `env.seed=0`。同一组合续跑时保持相同 run name；任何不同组合必须使用不同 run name。
