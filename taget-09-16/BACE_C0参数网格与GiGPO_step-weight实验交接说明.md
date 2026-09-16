# BACE C0 参数网格与原版 GiGPO step-weight 实验交接说明

运行环境、安装步骤、GPU smoke 与环境交付要求见[当前运行环境安装与交付清单](./当前运行环境安装与交付清单.md)。

## 1. 任务范围

本批次固定使用 seed `0`，总计 **39 个独立完整实验**：

| 方法 | 参数网格 | 数量 |
| --- | --- | ---: |
| BACE C0 optimized | competence `4` 种 × step weight `3` 种 × ERV threshold `3` 种 | 36 |
| 原版 GiGPO | step weight `3` 种 | 3 |
| 合计 |  | **39** |

BACE C0 的完整笛卡尔积为：

```text
competence_threshold ∈ {0.4, 0.5, 0.6, 0.7}
step_advantage_w     ∈ {1.0, 0.8, 1.2}
batch_erv_threshold  ∈ {0.005, 0.0025, 0.0075}
seed                 = 0
```

GiGPO 只改变：

```text
step_advantage_w ∈ {1.0, 0.8, 1.2}
seed             = 0
```

若 GPU 数量不足，**优先运行 36 个 BACE C0 实验**。GiGPO 的 3 个实验可以等 BACE 获得资源或完成后再提交。不要为了提高并发修改 batch、group、TP、horizon、response length、micro-batch 或其他算法参数。

## 2. 代码与资产

当前 RWTH 目录布局：

```text
/hpcwork/xsz96350/fu_project/
├── work-BACE/verl-agent-src/
├── verl-agent/
├── model_down/model/Qwen2.5-1.5B-Instruct/
└── data/text/{train.parquet,test.parquet}
```

ALFWorld 数据位于：

```text
/home/xsz96350/.cache/alfworld
```

进入代码目录：

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src
```

在其他机器交接时，应保持 `work-BACE/`、Python 环境、模型和 parquet 的相对布局，或者在脚本中显式改为该机器的真实路径。不要复制模型或 ALFWorld 数据到实验输出目录。

## 3. BACE C0：36 组实验

### 3.1 唯一入口

训练入口：

```text
work-BACE/verl-agent-src/examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh
```

36 组批量提交器：

```text
work-BACE/verl-agent-src/examples/bace_gigpo/c0_optimized_2gpu/submit_c0_grid_seed0.sh
```

该 C0 入口是独立脚本，不调用其他 BACE launcher。固定方法口径包括：

- `algorithm.adv_estimator=bace_gigpo`、`credit_mode=current`；
- optimized 调度：`main_reuse + root_active_executor=true`；
- Exact Batch-ERV、dynamic/staged/packed、selected-worker；
- `total_leaf_budget=8`、`min_natural_roots=2`、`max_branches_per_anchor=2`；
- `capacity_correction_batch_size=1`，不是 sweep 轴；
- Qwen2.5-1.5B-Instruct、2×H100、TP=2；
- train/val batch `16/128`、group `8`、horizon `50`、response `512`；
- LR `1e-6`、PPO mini/micro `256/32`、KL `0.01`、gamma `0.95`；
- 150 个 optimizer step，每 5 step validation，并在最后一步 validation。

`capacity_correction_batch_size=1` 的含义是：对一个容量不足的 task，每轮最多将一个尚未执行的 branch slot 转换为 natural root，再重新评估容量。它允许同一个 task 经过多轮依次补多个 root，但不允许一轮同时给该 task 补多个 root。

### 3.2 三个 sweep 变量

环境变量与取值一一对应：

| 实验轴 | 环境变量 | 本批取值 |
| --- | --- | --- |
| competence | `C0_COMPETENCE_THRESHOLD` | `0.4, 0.5, 0.6, 0.7` |
| step weight | `C0_STEP_ADVANTAGE_W` | `1.0, 0.8, 1.2` |
| ERV threshold | `C0_BATCH_ERV_THRESHOLD` | `0.005, 0.0025, 0.0075` |
| seed | `BACE_SEED` | 固定 `0` |

除这三项外不得改变 BACE 方法参数。尤其不要把 `capacity_correction_batch_size` 改回 `4`。

### 3.3 正式提交

当前批量提交器默认使用 `c23g/rwth2082`、2×H100、23.5 小时：

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src

GRID_RUN_TAG=handoff01 \
  bash examples/bace_gigpo/c0_optimized_2gpu/submit_c0_grid_seed0.sh \
  | tee c0_grid_handoff01_jobs.txt
```

脚本应恰好输出 36 行 `job-id run-name`。`GRID_RUN_TAG` 必须对每次提交唯一，防止不同批次共用 artifact 或误恢复。若对方不是 Slurm 环境，不执行提交器；应先由其调度系统申请同节点 2 张 GPU，再逐组合执行 `run_c0_optimized_2gpu.sh`。

当前共享盘不足以为 36 组实验长期保存 checkpoint，因此批量提交器固定：

```text
SAVE_FREQ=-1
MILESTONE_CHECKPOINT_STEPS=none
```

即不保存、不恢复 checkpoint；作业中断后从 step 0 重跑。BACE artifact、TensorBoard 和 trace 仍会保存。

### 3.4 单个组合与 dry-run

以 `competence=0.4`、`step weight=0.8`、`ERV threshold=0.0025` 为例：

```bash
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

检查生成的 `resolved_command.sh` 至少包含：

```text
env.seed=0
algorithm.bace.competence_threshold=0.4
algorithm.gigpo.step_advantage_w=0.8
algorithm.bace.batch_erv_threshold=0.0025
algorithm.bace.capacity_correction_batch_size=1
actor_rollout_ref.rollout.tensor_model_parallel_size=2
trainer.n_gpus_per_node=2
trainer.total_training_steps=150
trainer.save_freq=-1
```

正式提交单个组合：

```bash
sbatch --partition=c23g --account=rwth2082 --time=23:30:00 \
  --job-name=c0_c0p4_w0p8_tau0p0025 \
  --export=ALL,BACE_SEED=0,C0_COMPETENCE_THRESHOLD=0.4,C0_STEP_ADVANTAGE_W=0.8,C0_BATCH_ERV_THRESHOLD=0.0025,BACE_RUN_NAME=bace_c0_opt_seed0_c0p4_w0p8_tau0p0025_handoff01,TARGET_STEP=150,SAVE_FREQ=-1,MAX_CHECKPOINTS=2,MILESTONE_CHECKPOINT_STEPS=none \
  examples/bace_gigpo/c0_optimized_2gpu/run_c0_optimized_2gpu.sh
```

## 4. 原版 GiGPO：3 组实验

### 4.1 方法边界

GiGPO 必须使用：

```text
work-BACE/verl-agent-src/examples/gigpo_trainer/run_alfworld.sh
```

当前 Slurm 完整入口为：

```text
work-BACE/verl-agent-src/examples/gigpo_trainer/slurm_alfworld_h100_full_seed0_no_checkpoint.sbatch
```

该方法使用 `algorithm.adv_estimator=gigpo`，不启用 BACE collector、branch/replay、Exact Batch-ERV、family history 或 BACE trace。除了本批指定的 `algorithm.gigpo.step_advantage_w`，其余参数必须保持原版 GiGPO ALFWorld 配置。

完整 wrapper 接受：

```text
RUN_SEED=0
GIGPO_STEP_ADVANTAGE_W=<1.0|0.8|1.2>
```

它使用 2×H100、TP=2、240 GiB RAM、150 step，记录 TensorBoard，不保存 checkpoint，`resume_mode=disable`。

### 4.2 三条正式命令

```bash
cd /hpcwork/xsz96350/fu_project/work-BACE/verl-agent-src

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

若使用其他调度器，获得同节点 2 张 GPU 后执行原始入口，并把 step-weight override 放在命令最后：

```bash
bash examples/gigpo_trainer/run_alfworld.sh vllm \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  actor_rollout_ref.model.path=/path/to/Qwen2.5-1.5B-Instruct \
  env.seed=0 \
  algorithm.gigpo.step_advantage_w=0.8 \
  trainer.experiment_name=gigpo_seed0_w0p8 \
  "trainer.logger=['console','tensorboard']" \
  trainer.save_freq=-1 \
  trainer.resume_mode=disable
```

## 5. 资源优先级与提交顺序

推荐顺序：

1. 先对 C0 做至少一个 dry-run，并做一个 1-step H100 smoke；
2. 提交 36 个 BACE C0 作业；
3. 根据对方 GPU 配额控制并发，不要修改训练参数来追求吞吐；
4. BACE 已获得足够资源后，再提交 3 个 GiGPO 作业。

若调度器支持依赖，可让 GiGPO 等待 BACE 批次完成；若不支持，人工分两批提交即可。“BACE 优先”指资源提交顺序，不表示改变 loss、采样或训练步数。

## 6. 输出、日志与存储

BACE 输出根：

```text
work-BACE/experiments/alfworld-qwen2.5-1.5b-exact/
```

每个 C0 run 主要保存：

```text
bace_artifacts/<run>/
rollout_trajectories/<run>/
tensorboard/<run>/
run_metadata/<run>/<job-id>/
trace_validation/<run>/
logs/<run>/<job-id>.log
```

GiGPO 输出根：

```text
work-BACE/experiments/gigpo-alfworld-upstream-full/
```

当前 RWTH 上，上述实验目录应物理落在 `/hpcwork/rwth2089/xsz96350/work-BACE/experiments/`，旧路径只作为符号链接或查看入口。提交前必须执行 `readlink -f` 和 `df -h/-i`，确认日志、TensorBoard 和 artifact 没有落到 inode 紧张的源码盘。

最近三个 150-step BACE run 的实际占用为：BACE artifact 约 `13--15 GiB/run`，rollout trajectory 约 `1.1--1.3 GiB/run`。按较高值估计，36 个 C0 run 需要约 `587 GiB`，再计入并发写入、日志和安全余量，提交前应至少准备 `650 GiB` 可用空间。

截至文档编写时，当前 `rwth2089` 目标盘仅约有 `436 GiB` 空余，因此**不能在该盘上直接长期保留全部 36 组输出**。对方运行前必须完成以下之一：

1. 为该批次指定容量足够的新物理输出盘，并让 `work-BACE/experiments/alfworld-qwen2.5-1.5b-exact` 指向它；
2. 分批运行，在校验和备份完成后将前一批完整结果迁移到归档盘，再启动下一批；
3. 若研究目标允许减少 artifact 内容，先单独评审和修改 artifact 策略；这会改变可审计性，不能由执行者自行决定。

仅关闭 checkpoint 不能解决全部存储问题。不要在空间不足时依赖“跑完后再清理”，并发作业可能在训练中途同时耗尽配额。

## 7. 验收标准

每个 BACE C0 run 至少满足：

1. Slurm/调度器状态成功，进程退出码为 0；
2. 最终出现 `training/global_step:150`；
3. step 150 validation 完成；
4. `trace_validation/<run>/*_final.json` 为 `ok=true`；
5. step 150 `summary.json` 为 `status=complete`；
6. TensorBoard 中存在 `val/success_rate` 与 `val/text/test_score`；
7. `resolved_command.sh` 中三项 sweep 参数、seed0 与 `capacity_correction_batch_size=1` 均正确。

每个 GiGPO run 至少满足：

1. 作业状态成功，退出码为 0；
2. 最终出现 `training/global_step:150`；
3. step 150 validation 完成；
4. TensorBoard 中存在 `val/success_rate` 与 `val/text/test_score`；
5. 日志或解析命令证明 `algorithm.adv_estimator=gigpo`、seed0 和目标 step weight 生效；
6. 没有 BACE branch/replay/trace 输出。

## 8. 固定参数合同

最终交付不能只写“三个 sweep 参数”。每个 run 都必须保留完整的 `resolved_command.sh`，并在汇总表中至少列出本节参数。`resolved_command.sh` 是实际运行参数的权威记录，文档表格用于快速审计。

### 8.1 BACE C0 参数

| 类别 | 参数 | 固定值或 sweep 值 |
| --- | --- | --- |
| 方法 | `algorithm.adv_estimator` | `bace_gigpo` |
| 方法 | `algorithm.bace.credit_mode/tree_credit_mode` | `current/current` |
| 拓扑 | variant/topology/acquisition | `batch_erv_exact/dynamic/batch_erv_exact` |
| root | dynamic generation / batching | `staged/packed` |
| branch | execution / pool | `selected_worker/main_reuse` |
| root executor | `root_active_executor` | `true` |
| 预算 | total budget / min roots / max branches per anchor | `8/2/2` |
| 修正 | `capacity_correction_batch_size` | 固定 `1` |
| sweep | `competence_threshold` | `0.4/0.5/0.6/0.7` |
| sweep | `algorithm.gigpo.step_advantage_w` | `1.0/0.8/1.2` |
| sweep | `batch_erv_threshold` | `0.005/0.0025/0.0075` |
| history | initial mean / strength / forgetting | `0.10/2.0/0.8` |
| history | transfer / min / max strength | `0.1/2.0/8.0` |
| posterior | local prior strength | `2.0` |
| action | invalid mode / tie identity | `strict_identity/stable_v1` |
| credit | local credit / macro normalization / PPO padding | `occurrence/stable_occurrence/copy_trainable` |
| seed | `env.seed` | 固定 `0` |

### 8.2 共同训练参数

| 参数 | 值 |
| --- | --- |
| 模型 | Qwen2.5-1.5B-Instruct，全参数 FSDP |
| train/val parquet | `train.parquet` / `test.parquet` |
| train/val batch | `16/128` |
| prompt/response length | `2048/512` |
| ALFWorld horizon / rollout group | `50/8` |
| actor learning rate | `1e-6` |
| PPO mini/micro per GPU | `256/32` |
| rollout/ref log-prob micro per GPU | `32/32` |
| rollout TP / GPU 数 | `2/2` |
| rollout GPU utilization | `0.6` |
| BACE max batched tokens / max seqs | `8192/1024` |
| chunked prefill / eager / free cache | `false/false/false` |
| KL loss coefficient/type | `0.01/low_var_kl` |
| gamma / normalization mode | `0.95/mean_std_norm` |
| invalid-action penalty | enabled，coefficient `0.1` |
| validation | 开始前一次、每 5 step、最后 step |
| 总训练步数 | `150` |
| sweep checkpoint | 不保存，`save_freq=-1`、milestone disabled |

### 8.3 GiGPO 参数边界

GiGPO 使用同一模型、数据、batch、group、horizon、response、LR、PPO、TP、KL、gamma、validation 和 seed0。必须记录：

```text
algorithm.adv_estimator=gigpo
algorithm.gigpo.mode=mean_std_norm
algorithm.gigpo.step_advantage_w=<1.0|0.8|1.2>
trainer.save_freq=-1
trainer.resume_mode=disable
```

GiGPO 不得出现任何 `algorithm.bace.*` 参数。其原始入口没有显式写 BACE 使用的 `max_num_batched_tokens=8192` 或 `max_num_seqs=1024`，因此不要为了“对齐”而额外添加；应保留 GiGPO 配置的实际默认解析值并随最终 resolved config 一起交付。

## 9. 最终交付物

执行者完成实验后必须交付以下内容：

1. 实际使用的四个脚本及其 SHA-256：C0 训练入口、C0 网格提交器、GiGPO 原始入口、GiGPO H100 wrapper；
2. 39 个 run 各自的 `resolved_command.sh` 或完整 resolved Hydra config；
3. 每个 run 的 run name、job id、节点/GPU、开始与结束时间、运行时长、状态和退出码；
4. 三个 sweep 参数、seed，以及本节列出的固定方法和训练参数；
5. BACE 的 source manifest、preflight、最终 trace validation 和 step-150 summary；
6. TensorBoard event 文件和下面规定的最终 validation 指标表；
7. Python `pip freeze`、module/CUDA、GPU 型号和拓扑记录；
8. 实际输出路径及 `readlink -f` 后的物理路径。

不得只交付 TensorBoard 截图、作业号或一句“训练成功”。缺少 resolved command 时，无法证明参数组合没有串线。

## 10. ALFWorld 最终成功率

每个实验必须取 **validation step 150** 的值，报告总体成功率、test score 和六类 ALFWorld 任务成功率：

| 指标列 | TensorBoard tag |
| --- | --- |
| overall success | `val/success_rate` |
| test score | `val/text/test_score` |
| look-at-object-in-light | `val/look_at_obj_in_light_success_rate` |
| pick-and-place | `val/pick_and_place_success_rate` |
| clean-then-place | `val/pick_clean_then_place_in_recep_success_rate` |
| cool-then-place | `val/pick_cool_then_place_in_recep_success_rate` |
| heat-then-place | `val/pick_heat_then_place_in_recep_success_rate` |
| pick-two-and-place | `val/pick_two_obj_and_place_success_rate` |

最终结果表必须有 39 行，一行对应一个独立 run：

| 方法 | competence | step weight | ERV threshold | seed | job id | 状态 | 用时 | overall | score | look | pick/place | clean | cool | heat | pick-two | 输出路径 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BACE C0 | 0.4 | 1.0 | 0.005 | 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| … | … | … | … | 0 |  |  |  |  |  |  |  |  |  |  |  |  |
| GiGPO | — | 1.0 | — | 0 |  |  |  |  |  |  |  |  |  |  |  |  |

成功率建议同时保存 `[0,1]` 原值和百分比显示值，避免四舍五入影响后续统计。如果 step 150 的任一任务 tag 缺失，应标记该 run 的交付不完整，不能用较早 validation 值替代。

可以使用 TensorBoard event accumulator 读取最后一步，但应检查返回事件的 `step == 150`：

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_dir = "/path/to/tensorboard/<run>"
ea = EventAccumulator(event_dir, size_guidance={"scalars": 0})
ea.Reload()

tags = [
    "val/success_rate",
    "val/text/test_score",
    "val/look_at_obj_in_light_success_rate",
    "val/pick_and_place_success_rate",
    "val/pick_clean_then_place_in_recep_success_rate",
    "val/pick_cool_then_place_in_recep_success_rate",
    "val/pick_heat_then_place_in_recep_success_rate",
    "val/pick_two_obj_and_place_success_rate",
]
for tag in tags:
    event = ea.Scalars(tag)[-1]
    assert event.step == 150, (tag, event.step)
    print(tag, event.value)
```

除汇总表外，还应保留每个任务的完整 validation 曲线，以便判断最终值是否是偶然波动。
