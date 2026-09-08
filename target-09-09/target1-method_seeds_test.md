# BACE C0--C8 二卡多 seed 实验交接说明

## 1. 目的与范围

本说明用于重新运行当前 **capacity-correction=1** 的 BACE C0--C8 ALFWorld 实验。标准实验为每个方法使用 seed `0`、`1`、`2` 各运行一次，每次使用同一节点上的 **2 张 H100**。

不要使用 `archived_launchers/` 中的历史入口；其中保留的是 `capacity_correction_batch_size=4` 的审计副本，不是当前实验配置。

当前新提交必须先通过 1-step H100 smoke，且不会由本说明中的命令自动提交其他作业链。不要复用旧实验的 run name 或 checkpoint。

## 2. 工作区与资产路径

脚本采用相对路径推导，但工作区必须具有以下同级布局：

```text
<workspace>/
├── work-BACE/verl-agent-src/       # 本代码仓库及启动脚本
├── verl-agent/                     # 当前 Python/conda prefix
├── model_down/model/Qwen2.5-1.5B-Instruct/
└── data/text/{train.parquet,test.parquet}
```

当前 RWTH 环境的实际路径如下：

| 用途 | 路径 |
| --- | --- |
| 工作区 | `/hpcwork/xsz96350/fu_project` |
| 代码仓库 | `/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src` |
| Python 环境 | `/hpcwork/xsz96350/fu_project/verl-agent` |
| 模型 | `/hpcwork/xsz96350/fu_project/model_down/model/Qwen2.5-1.5B-Instruct` |
| 训练/验证 parquet | `/hpcwork/xsz96350/fu_project/data/text/train.parquet`、`/hpcwork/xsz96350/fu_project/data/text/test.parquet` |
| ALFWorld 资源 | `/home/xsz96350/.cache/alfworld` |
| 实验输出根 | `/hpcwork/xsz96350/fu_project/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact` |

实验输出根在当前机器上是指向 `/hpcwork/rwth2089/xsz96350/work-BACE/experiments/alfworld-qwen2.5-1.5b-exact` 的符号链接。**不要改成 `$WORK`，不要复制模型、parquet 或 ALFWorld cache。**

进入代码目录：

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
```

## 3. 方法与唯一可用入口

所有入口都是完整的 Bash/Slurm 脚本：它们自行构造 Hydra 命令，不依赖另一个训练 launcher。除 credit mode 外，各方法共享相同的二卡训练设置。

| 方法 | credit mode | 脚本（相对于 `work-BACE/verl-agent-src`） |
| --- | --- | --- |
| C0 | `current` | `examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh` |
| C1 | `o1_local` | `examples/bace_gigpo/c1_tree_credit_corr1_2gpu/run_c1_corr1_2gpu.sh` |
| C2 | `o1_tree_macro` | `examples/bace_gigpo/c2_tree_credit_corr1_2gpu/run_c2_corr1_2gpu.sh` |
| C3 | `o1_full_tree` | `examples/bace_gigpo/c3_tree_credit_corr1_2gpu/run_c3_corr1_2gpu.sh` |
| C0.5 | `c0_5_origin_family_local_mean` | `examples/bace_gigpo/c05_tree_credit_corr1_2gpu/run_c05_corr1_2gpu.sh` |
| C4 | `c4_macro_strict_ancestor` | `examples/bace_gigpo/c4_tree_credit_corr1_2gpu/run_c4_corr1_2gpu.sh` |
| C7 | `c7_flat_leaf_gigpo` | `examples/bace_gigpo/c7_tree_credit_corr1_2gpu/run_c7_corr1_2gpu.sh` |
| C8 | `c8_macro_local_strict_ancestor` | `examples/bace_gigpo/c8_tree_credit_corr1_2gpu/run_c8_corr1_2gpu.sh` |

共同 BACE 调度口径：Exact Batch-ERV、dynamic/staged/packed、`selected_worker`、`main_reuse`、`root_active_executor=true`、`total_leaf_budget=8`、`min_natural_roots=2`、`capacity_correction_batch_size=1`、`max_branches_per_anchor=2`。

`capacity_correction_batch_size=1` 的精确定义是：对某一个容量不足的 task，每轮容量评估最多把 **一个** 尚未执行的 branch slot 转换成 natural root，随后重新评估。多个 task 同时新增的 root 仍会在同一个 packed root wave 中执行；它不表示“一次 GPU wave 只能运行一条 root”。

共同训练口径：Qwen2.5-1.5B-Instruct、TP=2、train/val batch=`16/128`、horizon=`50`、response length=`512`、LR=`1e-6`、PPO mini/micro=`256/32`、KL=`0.01`、gamma=`0.95`、`step_advantage_w=1.0`、`competence_threshold=0.5`、`batch_erv_threshold=0.005`。

## 4. 必做 dry-run

任何正式提交前，以新 run name 运行 dry-run。它不会申请 GPU、不会训练；应检查输出的 `resolved_command.sh` 中的模型、数据、TP=2、credit mode、`capacity_correction_batch_size=1` 和输出路径。

以 C8 seed 1 为例：

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src

DRY_RUN=1 \
BACE_SEED=1 \
BACE_RUN_NAME=bace_c8_corr1_seed1_YYYYMMDD \
bash examples/bace_gigpo/c8_tree_credit_corr1_2gpu/run_c8_corr1_2gpu.sh
```

将上述脚本路径和 run name 换成表中的其他方法即可。dry-run 输出位于：

```text
work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/run_metadata/<BACE_RUN_NAME>/local/resolved_command.sh
```

## 5. Slurm 正式提交模板（每个 seed 一次）

单节点、最多 4 张 H100 时优先使用 `c25g/rwth2089`。脚本本身已声明 `--nodes=1 --gres=gpu:2 --cpus-per-task=32`；命令中无需再次声明 GPU 数。`--time` 由本次实验的时限决定，但项目账户单作业硬上限为 24 小时。

以下命令提交 C8 seed 1；`BACE_RUN_NAME` 必须对每个方法和 seed 唯一：

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src

sbatch \
  --partition=c25g --account=rwth2089 --time=15:00:00 \
  --export=ALL,BACE_SEED=1,BACE_RUN_NAME=bace_c8_corr1_seed1_YYYYMMDD \
  examples/bace_gigpo/c8_tree_credit_corr1_2gpu/run_c8_corr1_2gpu.sh
```

对 seed `0`、`1`、`2` 分别提交三次，仅替换两个位置：

```text
BACE_SEED=<0|1|2>
BACE_RUN_NAME=bace_<method>_corr1_seed<seed>_YYYYMMDD
```

例如 C4 seed 2：

```bash
sbatch --partition=c25g --account=rwth2089 --time=15:00:00 \
  --export=ALL,BACE_SEED=2,BACE_RUN_NAME=bace_c4_corr1_seed2_YYYYMMDD \
  examples/bace_gigpo/c4_tree_credit_corr1_2gpu/run_c4_corr1_2gpu.sh
```

若站点要求使用 c23g，可替换为 `--partition=c23g --account=rwth2082`；这不改变训练参数，但 c23g 当前通常排队更久。不要把同一 run name 投到两个分区。

## 6. Checkpoint、输出与验收

默认正式设置为 `TARGET_STEP=150`、每 5 step 保存、最多保留最近 2 个 checkpoint；脚本会自动使用同 run name 下最新完整 checkpoint 恢复。仅在“确实要从中断处恢复”时复用 run name。

每个 run 的关键目录均在输出根下：

```text
bace_artifacts/<run>/
checkpoints/<run>/
rollout_trajectories/<run>/
tensorboard/<run>/
run_metadata/<run>/<SlurmJobID>/
trace_validation/<run>/
logs/<run>/<SlurmJobID>.log
```

完成后至少检查：

1. Slurm 状态为 `COMPLETED` 且 exit code 为 `0:0`；
2. 训练日志包含最后一步的 `training/global_step`；
3. `trace_validation/<run>/*_final.json` 中 `ok=true`；
4. artifact 的 `summary.json` 为 `status=complete`；
5. 运行目录内的 `resolved_command.sh`、`run_metadata.txt` 与 source manifest 存在。

当前已经提交的 `preformal_smoke_corr1_*` 作业仅为 1-step、无 checkpoint 的验证，不是正式实验；不要把它们的 `TARGET_STEP=1`、`SAVE_FREQ=0` 覆盖写入上述正式命令。

## 7. C0 专属可调参数

C0 基线的标准多-seed 重跑应保持以下默认值，不要设置对应环境变量：

```text
C0_COMPETENCE_THRESHOLD=0.5
C0_STEP_ADVANTAGE_W=1.0
C0_BATCH_ERV_THRESHOLD=0.005
```

若明确进行 C0 参数消融，可在 `sbatch --export=ALL,...` 中额外设置这些变量；每种组合都必须使用不同 run name。C1--C8 的当前独立入口没有把这些参数开放为 sweep 轴，避免不经记录地改变方法定义。

## 8. 当前代码身份与限制

当前 Git 基线提交为 `6d2c0ca21631be6bcec0c2f13d8390694ed570cc`，但工作区包含 BACE 实现和脚本的未提交修改。因此 Git commit 本身不足以复现当前状态；每次运行脚本会写出 source SHA-256 manifest，交接前也应保存整个 `work-BACE/` 目录或单独归档该 manifest。

环境安装细节、已验证版本和从零重建步骤见同目录的 `当前运行环境安装清单.md`。
