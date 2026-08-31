# BACE / Exact Batch-ERV

本仓库是 BACE-GiGPO 在 ALFWorld 上的源码快照。新维护者请先阅读
[`docs/README.md`](docs/README.md)，不要从文件名、旧计划或 Slurm 脚本猜测当前进度。

## 当前结论（2026-08-31）

- 4×H100、Qwen2.5-1.5B-Instruct 的主实验已经真实完成 **150/150 个训练 step**；训练进程退出码为 0，最终 validation success rate 为 **87.50%**。
- 另一条 2×H100 BACE run 也已完成 **150/150**；step 150 validation 为 **86.72%**，该 run 的 Slurm `FAILED` 同样发生在训练后的 trace 处理，重验 step-150 artifact 为 `ok=true`。它与 4 卡 run 的并行条件不同，不能用于严格硬件或算法因果比较。
- 150 个 step 的 `summary.json` 均为 `status=complete`，checkpoint tracker 指向 step 150，并保留 step 145/150。
- Slurm job `3116958` 显示 `FAILED/1:0`，但失败发生在训练完成后的 trace JSONL 解析，不代表训练只跑了一部分。当前 artifacts 的重新校验结论见[当前真实进度](docs/01_当前实现状态与未完成事项.md)。
- 当前工作树在该长跑之后继续加入 quota-aware Exact DP、branch 分块、selected-worker、task rotation、stable tie identity、P1-S 调度、checkpoint 隔离，以及 Pairwise Exact、action-mean 诊断和里程碑 checkpoint；本分支的 CPU 回归基线以本次提交记录为准。
- P1-S 已在 4×H100 上完成 S0–S3 profile、三步稳定性训练和外部 checkpoint 隔离门禁。新实验推荐显式启用 `branch_pool_mode=main_reuse` 与 `root_active_executor=true`；YAML 仍保留兼容默认。
- P1-S 现在已经接入当前 2 卡 optimized profile 和 4 卡 rolling 正式入口；入口支持显式 `BACE_SEED`，并可用 `MILESTONE_CHECKPOINT_STEPS=none` 关闭永久保存点。2/4 卡 seed 1/2 作业已经提交，但截至 2026-08-31 审计时仍在排队，不能记为完成结果。
- `min_natural_roots=4` 目前仅通过 step 148→150 的受控诊断迁移验证，尚未成为正式方法默认；当前规范默认仍为 `R_min=2`。
- Pairwise `fixed/stopping` 已实现并通过 CPU 专项测试，离线审计结论仍为 **HOLD / ABLATION ONLY**；正式默认保持 `pairwise.mode=full` 和 occurrence credit。Pairwise 在线 smoke、同 checkpoint 对照及 action-mean 正式实验不能写成已完成。
- Pairwise C1/C2 新增从 step 0 运行到 150 的 `full` phase；两条 seed-0 作业已提交但仍为 `PENDING`。这是一项待完成的独立正式消融，不改变 Full BACE 仍为当前主方法的事实。
- GiGPO seed 1/2 legacy-float32 复现入口使用校验过的 seed-0 frozen source，以避免当前稳定统计修复改变复现口径；相应作业同样仍在排队。
- `experiments/`、模型、ALFWorld 数据、parquet、checkpoint 和原 Python 环境不会上传到 GitHub，因此 GitHub 只能保存结果摘要，不能单独证明或恢复该次长跑。

正式 run 的源码 metadata 没有记录可用 commit SHA（`git_head=HEAD/unavailable`），而且当前代码在 run 后继续更新。因此应把它理解为“原开发工作树的一次完成实验”，不能声称它与当前 GitHub HEAD 字节级一致。换设备后的第一件事仍应是复现测试和最小 GPU 门禁，而不是复用 RWTH 提交链。

## 必读文档

1. [当前真实进度与未完成事项](docs/01_当前实现状态与未完成事项.md)
2. [文档有效性清单](docs/06_文档有效性清单.md)
3. [新设备安装与首次运行](docs/02_新设备安装与首次运行.md)
4. [代码结构与数据流程](docs/03_代码结构与数据流程.md)
5. [资产与实验结果迁移清单](docs/04_资产与实验结果迁移清单.md)
6. [验证验收与故障排查](docs/05_验证验收与故障排查.md)
7. [当前 BACE 方法规范](docs/07_当前BACE方法规范.md)
8. [代码库阅读路线](docs/08_代码库阅读路线.md)
9. [逻辑模式与兼容边界](docs/09_逻辑模式与兼容边界.md)
10. [运行环境安装清单](docs/10_运行环境安装清单.md)

## 仓库结构

- `verl-agent-src/`：迁入的 verl-agent 源码、BACE 实现、入口和测试。
- `deploy/`：CUDA 环境安装与 GPU/资产 preflight。
- `docs/`：当前维护口径和换设备交接文档。
- `BACE-results-vanilla/`：已完成旧 BACE run 的分析、早期输入和辅助工具；先看目录内 `README.md`。
- `GiGPO_reference_analysis/`：GiGPO reference run 的采集与离线分析规范。
- `analysis/`、`analysis_outputs/`：Pairwise feedback/threshold 的可复核离线分析与小型结果。
- `improve-plan/`：P1-S/P1-A、Pairwise 与 action-mean 的设计、专项结果和准入边界。
- `issues_old/`：已经归档的历史问题调查，不作为当前状态入口。
- `BACE-work-2/`：较新的方法设计背景，不是当前状态台账。
- `BACE-work/`、`issues/`：早期实现和仍保留的问题材料；已归档内容位于 `issues_old/`。
- 根目录旧 H100/A100 和 StateID 文档：历史部署或独立设计资料，使用前先查文档清单。

推荐的新设备入口是：

```text
verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_gpu.sh
```

`examples/bace_gigpo/*.sbatch` 和提交链脚本含 RWTH 账号、分区、路径及存储假设，只保留作历史证据。

## 最小源码验证

准备兼容环境后：

```bash
cd verl-agent-src
pytest -q tests/bace_gigpo tests/trainer/ppo/test_metric_utils.py
```

本分支的准确测试数量和命令写入[当前真实进度](docs/01_当前实现状态与未完成事项.md)。真实训练还需要仓库外的 Qwen 模型、ALFWorld 数据和 parquet，详见交接文档。

原上游项目为 [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent)。迁入代码没有保留原上游 `.git` 历史，本仓库不是一组可直接 rebase 的干净 patch。
