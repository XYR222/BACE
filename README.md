# BACE / Exact Batch-ERV

本分支是 2026-09-16 的 ALFWorld 实验交接版本，面向两类 2×H100、seed 0
的完整对照实验：BACE C0 optimized 参数网格，以及原版 GiGPO 的
step-advantage-weight 对照。

当前交付分支：

```text
handoff-refresh-20260916
```

本分支基于 `handoff-refresh-20260908`。模型、ALFWorld 数据、parquet、Python
环境、checkpoint、日志和 `experiments/` 不在 Git 仓库中。

## 实验范围

本批共 **39 个独立的 150-step 实验**：

| 方法 | 参数 | 数量 |
| --- | --- | ---: |
| BACE C0 optimized | competence 4种 × step weight 3种 × ERV threshold 3种 | 36 |
| 原版 GiGPO | step weight 3种 | 3 |
| 合计 | 全部固定 seed 0 | **39** |

BACE C0 网格：

```text
competence_threshold ∈ {0.4, 0.5, 0.6, 0.7}
step_advantage_w     ∈ {1.0, 0.8, 1.2}
batch_erv_threshold  ∈ {0.005, 0.0025, 0.0075}
seed                 = 0
```

GiGPO 网格：

```text
step_advantage_w ∈ {1.0, 0.8, 1.2}
seed             = 0
```

GPU 资源不足时优先运行 36 个 BACE 实验。不要为了增加并发而修改 batch、
rollout group、TP、horizon、response length 或 micro-batch。

## 必读文档

1. [39组实验的完整交接说明](taget-09-16/BACE_C0参数网格与GiGPO_step-weight实验交接说明.md)
2. [环境安装与交付清单](taget-09-16/当前运行环境安装与交付清单.md)
3. [C0 独立入口说明](verl-agent-src/examples/bace_gigpo/c0_optimized_2gpu/README.md)
4. [GiGPO 对照说明](target-09-09/target2-GiGPO-refer.md)

`target-09-09/` 保留上一阶段的 C0–C8、多 seed 交接资料；本次39组实验应以
`taget-09-16/` 为准。`action_problem/`、`improve-plan/`、历史 launcher 和其他
方法变体不是本批参数网格的启动入口。

## 获取代码

```bash
git clone https://github.com/XYR222/BACE.git work-BACE
cd work-BACE
git checkout handoff-refresh-20260916
cd verl-agent-src
```

安装前先阅读[环境安装清单](taget-09-16/当前运行环境安装与交付清单.md)。当前
已验证环境的核心版本为 Python 3.12、CUDA 12.8、PyTorch 2.8、vLLM 0.11.0、
FlashAttention 2.7.4、Gymnasium 0.29.1、Stable-Baselines3 2.6.0 和
ALFWorld 0.4.2。不要在已验证环境中直接升级这些二进制依赖。

## BACE C0 optimized

唯一训练入口：

```text
verl-agent-src/examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh
```

36组提交器：

```text
verl-agent-src/examples/bace_gigpo/c0_optimized_2gpu/submit_c0_grid_seed0.sh
```

该入口固定使用：

- Exact Batch-ERV、dynamic/staged/packed 和 selected-worker；
- `branch_pool_mode=main_reuse`、`root_active_executor=true`；
- `total_leaf_budget=8`、`min_natural_roots=2`；
- `max_branches_per_anchor=2`；
- `capacity_correction_batch_size=1`；
- occurrence credit、stable tie identity 和 copy-trainable PPO padding；
- Qwen2.5-1.5B-Instruct、2×H100、TP=2；
- train/val batch `16/128`、rollout group `8`；
- horizon `50`、response length `512`；
- LR `1e-6`、PPO mini/micro batch `256/32`；
- KL `0.01`、gamma `0.95`、共150个训练 step。

只有以下三个变量参与本次 BACE sweep：

| 含义 | 环境变量 |
| --- | --- |
| competence threshold | `C0_COMPETENCE_THRESHOLD` |
| step-level advantage weight | `C0_STEP_ADVANTAGE_W` |
| Batch-ERV threshold | `C0_BATCH_ERV_THRESHOLD` |

先做 dry-run：

```bash
cd work-BACE/verl-agent-src

DRY_RUN=1 \
BACE_SEED=0 \
C0_COMPETENCE_THRESHOLD=0.4 \
C0_STEP_ADVANTAGE_W=0.8 \
C0_BATCH_ERV_THRESHOLD=0.0025 \
BACE_RUN_NAME=dry_c0_c0p4_w0p8_tau0p0025 \
SAVE_FREQ=-1 \
MILESTONE_CHECKPOINT_STEPS=none \
bash examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh
```

确认生成的 `resolved_command.sh` 中 seed、三个 sweep 参数、
`capacity_correction_batch_size=1`、TP=2、GPU数=2和训练步数都正确。

提交36组正式作业：

```bash
GRID_RUN_TAG=handoff01 \
  bash examples/bace_gigpo/c0_optimized_2gpu/submit_c0_grid_seed0.sh \
  | tee c0_grid_handoff01_jobs.txt
```

`GRID_RUN_TAG` 每批必须唯一。提交器应恰好输出36行 `job-id run-name`。

## 原版 GiGPO

原始入口：

```text
verl-agent-src/examples/gigpo_trainer/run_alfworld.sh
```

2×H100 wrapper：

```text
verl-agent-src/examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch
```

三组实验分别提交为：

```bash
cd work-BACE/verl-agent-src

sbatch --job-name=gigpo_w1p0 \
  --export=ALL,RUN_SEED=0,GIGPO_STEP_ADVANTAGE_W=1.0 \
  examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch

sbatch --job-name=gigpo_w0p8 \
  --export=ALL,RUN_SEED=0,GIGPO_STEP_ADVANTAGE_W=0.8 \
  examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch

sbatch --job-name=gigpo_w1p2 \
  --export=ALL,RUN_SEED=0,GIGPO_STEP_ADVANTAGE_W=1.2 \
  examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch
```

GiGPO 必须保持 `algorithm.adv_estimator=gigpo`，不得加入任何
`algorithm.bace.*` 参数。除 `algorithm.gigpo.step_advantage_w` 外，其余训练参数
保持原始 GiGPO ALFWorld 入口的解析结果。

非 Slurm 环境应由目标调度系统先分配同节点2张GPU，再按交接文档直接调用相应
训练入口；不要照搬 RWTH 的 partition、account 或绝对路径。

## 外部资产与输出

当前 RWTH 布局为：

```text
/hpcwork/xsz96350/fu_project/
├── work-BACE/verl-agent-src/
├── verl-agent/
├── model_down/model/Qwen2.5-1.5B-Instruct/
└── data/text/{train.parquet,test.parquet}

/home/xsz96350/.cache/alfworld/
```

BACE 输出入口为：

```text
work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/
```

GiGPO 输出入口为：

```text
work-BACE/experiments/gigpo-alfworld-upstream-full/
```

本批 sweep 不保存 checkpoint；作业中断后从 step 0 重跑。BACE artifact、rollout、
TensorBoard、metadata、日志和 trace 仍需保存。36个 BACE run 预计约需587 GiB，
建议在提交前准备至少650 GiB可用空间，并用 `readlink -f`、`df -h` 和 `df -i`
确认实际物理落盘位置。

## 验收与最终交付

每个 run 至少应满足：

- 进程退出码为0；
- 达到 `training/global_step:150`；
- step 150 validation 完成；
- TensorBoard 包含 `val/success_rate` 和 `val/text/test_score`；
- BACE 的最终 trace 为 `ok=true`，step-150 summary 为 `status=complete`；
- 实际 seed、sweep 参数和全部固定参数与 `resolved_command.sh` 一致。

最终结果必须为39个 run 分别报告：

- 脚本 SHA-256、完整 resolved command/config、源码 manifest；
- job id、节点/GPU、起止时间、用时、状态和退出码；
- overall success rate 与 test score；
- look-at-object-in-light、pick-and-place、clean、cool、heat、pick-two 六类
  ALFWorld 任务在 validation step 150 的成功率；
- TensorBoard event、BACE trace/summary，以及实际输出物理路径。

不能用较早 validation 的结果替代 step 150，也不能只交付截图或作业号。完整的
TensorBoard tag、结果表模板和提取示例见[实验交接说明](taget-09-16/BACE_C0参数网格与GiGPO_step-weight实验交接说明.md)。

## 代码结构

| 内容 | 位置 |
| --- | --- |
| BACE rollout 与调度 | `verl-agent-src/recipe/bace_gigpo/rollout_collector.py`、`topology.py` |
| Batch-ERV 与 competence history | `verl-agent-src/recipe/bace_gigpo/batch_erv.py`、`competence.py` |
| Credit/advantage | `verl-agent-src/recipe/bace_gigpo/advantage.py`、`flat_leaf.py` |
| Branch Replay | `verl-agent-src/recipe/bace_gigpo/replay/` |
| Trainer 接入 | `verl-agent-src/verl/trainer/ppo/ray_trainer.py` |
| 主配置 | `verl-agent-src/verl/trainer/config/ppo_trainer.yaml` |
| Trace | `verl-agent-src/recipe/bace_gigpo/artifacts.py`、`validate_trace.py` |

上游项目为 [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent)。本仓库
保存的是研究代码快照，不包含原上游仓库的完整 Git 历史。
