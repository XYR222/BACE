# GiGPO ALFWorld：当前复现说明

本文只说明当前的**原版 GiGPO ALFWorld**基线如何复现。它不是 BACE 实验：不使用 BACE collector、branch/replay、Exact Batch-ERV、trace validator 或 BACE checkpoint gate。

## 1. 代码与脚本

以迁入的 veRL-agent 源码为准：

```text
源码根目录
/hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src

原始 GiGPO 训练入口
examples/gigpo_trainer/run_alfworld.sh

一步双卡 H100 烟测
examples/gigpo_trainer/slurm_alfworld_h100_smoke.sbatch

完整双卡 H100、seed 0、无 checkpoint 训练
examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch
```

两个 Slurm wrapper 都直接执行 `run_alfworld.sh vllm`。wrapper 只负责 Slurm 资源、模块环境与本地绝对路径；训练算法仍由原始入口的 `algorithm.adv_estimator=gigpo` 和 `env.env_name=alfworld/AlfredTWEnv` 决定。

不要使用 `examples/bace_gigpo/` 下的任何脚本来复现该实验，也不要把 AFH 仓库作为当前源代码入口。

## 2. 必需资产

| 项目 | 固定位置/版本 |
| --- | --- |
| conda 环境 | `/hpcwork/xsz96350/fu_project/verl-agent` |
| 模型 | `/hpcwork/xsz96350/fu_project/model_down/model/Qwen2.5-1.5B-Instruct` |
| 训练数据 | `/hpcwork/xsz96350/fu_project/data/text/train.parquet` |
| 验证数据 | `/hpcwork/xsz96350/fu_project/data/text/test.parquet` |
| ALFWorld 缓存 | `/home/xsz96350/.cache/alfworld` |
| CUDA | module `CUDA/12.8.0` |
| C++ runtime | module `GCCcore/13.3.0` |
| Python/vLLM | Python 3.12、vLLM 0.11.0 |

进入计算节点后，环境准备必须如下：

```bash
module purge
module load GCCcore/13.3.0
module load CUDA/12.8.0
source /home/xsz96350/miniforge3/etc/profile.d/conda.sh
conda activate /hpcwork/xsz96350/fu_project/verl-agent

unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES VLLM_ATTENTION_BACKEND
export VLLM_USE_FLASHINFER_SAMPLER=0
export ALFWORLD_DATA=/home/xsz96350/.cache/alfworld
```

`GCCcore/13.3.0` 必不可少：TextWorld/Fast Downward 需要 `GLIBCXX_3.4.32`；只激活 conda 时会使用节点旧版 C++ 库并在第一次 ALFWorld reset 失败。不能强制 xFormers；清除 `VLLM_ATTENTION_BACKEND` 后 vLLM 会选择当前 H100 上可用的 FlashAttention 路径。清除 ROCm 变量则避免它们与 Slurm 设置的 CUDA 可见设备冲突。

## 3. 先做一步烟测

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
sbatch examples/gigpo_trainer/slurm_alfworld_h100_smoke.sbatch
```

烟测申请 `c25g/rwth2089` 的单节点 2×H100、32 CPU、1 小时。它仍用原版 GiGPO 数据流，只把工作量缩小为：batch 2、group 2、horizon 2、response 64、每 GPU micro-batch 1，并用 `trainer.total_training_steps=1` 限制为一个 optimizer step。

通过标准：`sacct -j <jobid>` 为 `COMPLETED` 和 `0:0`，stdout 出现 `training/global_step:1`，且无 `GLIBCXX`、ROCm/CUDA visibility、CUDA OOM 或 Ray actor death。已成功的验证作业为 `3672157`，证明当前代码、模型、vLLM、Fast Downward、ALFWorld 和 PPO 链路均可运行。

## 4. 完整 seed 0 训练

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
sbatch examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch
```

当前完整 wrapper 的资源配置：

```text
partition/account: c25g / rwth2089
资源:            1 节点，2×H100，32 CPU，240 GiB RAM
时间:            12:30:00
seed:            0
checkpoint:      不保存，且不恢复旧 checkpoint
```

训练参数保持原始 GiGPO 口径：Qwen2.5-1.5B-Instruct 全参数 FSDP、train/val batch `16/128`、每任务 8 个 rollout、horizon 50、response 512、LR `1e-6`、PPO mini-batch 256、actor/logprob/ref micro-batch per GPU 均为 32、TP=2、rollout memory utilization 0.6、KL=0.01、gamma=0.95、开始前验证一次且每 5 step 验证、共 150 epoch。当前 16 行训练 parquet 与 train batch 16 对应 150 个 optimizer update。

完整 wrapper 只额外固定本地模型/数据、`env.seed=0`、独立 run 名、`trainer.save_freq=-1` 与 `trainer.resume_mode=disable`。因此作业若被取消、节点故障或超时，不能从中间 step 恢复，必须从 seed 0 重跑。

## 5. 为什么申请 240 GiB RAM

第一次完整运行 `3790004` 在 step 65 后失败，Slurm 状态是 `OUT_OF_MEMORY`。这不是 CUDA 显存错误：作业未显式申请内存时只得到 162.5 GiB 主存，Slurm 记录的 `MaxRSS` 为 170.4 GiB，并确认有 `oom_kill event=1`。所以当前脚本只增加 `#SBATCH --mem=240G`，不修改任何 GiGPO 算法或 batch 参数。

当前重提作业号为 `3862542`；该号码仅用于本次追踪，其他使用者运行 `sbatch` 后应记录自己的作业号。

## 6. 日志、监控与成功判据

```bash
# 状态和预计开始时间
squeue -j <jobid>
scontrol show job <jobid>

# 完成状态、退出码、峰值主存
sacct -j <jobid> --format=JobID,State,ExitCode,Start,End,Elapsed,ReqMem,MaxRSS -P

# 实时日志
tail -f /hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/gigpo-alf-s0-<jobid>.out
tail -f /hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/gigpo-alf-s0-<jobid>.err
```

完整 run 的 Slurm 日志在：

```text
/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/slurm/
```

W&B 设为 offline，本地运行记录在：

```text
/hpcwork/xsz96350/fu_project/work-BACE/experiments/gigpo-alfworld-upstream-full/wandb/<jobid>/
```

重点指标是 `training/global_step`、`episode/success_rate`、`val/success_rate`、`timing_s/step`、`perf/max_memory_*`。完整成功的最低条件：作业 `COMPLETED (0:0)`，最后出现 `training/global_step:150`，并完成 step 150 validation。

## 7. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `GLIBCXX_3.4.32 not found` | 未加载新 C++ runtime | `module load GCCcore/13.3.0`，再加载 CUDA |
| `Please don't set ROCR_VISIBLE_DEVICES` | HIP/ROCm 与 CUDA 变量冲突 | 清除 `ROCR_VISIBLE_DEVICES` 和 `HIP_VISIBLE_DEVICES` |
| Slurm `OUT_OF_MEMORY` 且 `oom_kill event` | 节点 RAM cgroup 超限 | 增加 `--mem`；当前基线为 240 GiB |
| `CUDA out of memory` | H100 显存不足 | 先单独确认后再降低执行 micro-batch；不要改 group/horizon 等算法参数 |
| 意外从旧训练继续 | 默认恢复 checkpoint | 保持 `trainer.resume_mode=disable` |

## 8. 边界

本文是 GiGPO 基线复现说明，不产出 BACE trace、family history 或 branch/replay 指标。需要 BACE 时必须走其独立脚本和文档，不能直接把两条训练链的指标混合比较。
