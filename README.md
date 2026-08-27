# BACE / Exact Batch-ERV

本仓库是 BACE-GiGPO 在 ALFWorld 上的源码快照。新维护者请先阅读
[`docs/README.md`](docs/README.md)，不要从文件名、旧计划或 Slurm 脚本猜测当前进度。

## 当前结论（2026-08-26）

- 4×H100、Qwen2.5-1.5B-Instruct 的主实验已经真实完成 **150/150 个训练 step**；训练进程退出码为 0，最终 validation success rate 为 **87.50%**。
- 150 个 step 的 `summary.json` 均为 `status=complete`，checkpoint tracker 指向 step 150，并保留 step 145/150。
- Slurm job `3116958` 显示 `FAILED/1:0`，但失败发生在训练完成后的 trace JSONL 解析，不代表训练只跑了一部分。当前 artifacts 的重新校验结论见[当前真实进度](docs/01_当前实现状态与未完成事项.md)。
- 当前 GitHub 代码还包含在该长跑之后加入的 quota-aware Exact DP、branch 分块、selected-worker、task rotation 和 stable tie identity 等修复；CPU 回归基线为 `126 passed`。
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

## 仓库结构

- `verl-agent-src/`：迁入的 verl-agent 源码、BACE 实现、入口和测试。
- `deploy/`：CUDA 环境安装与 GPU/资产 preflight。
- `docs/`：当前维护口径和换设备交接文档。
- `target/`：已版本化的实验分析、历史问题记录和辅助工具；先看 `target/README.md` 的分级。
- `BACE-work-2/`：较新的方法设计背景，不是当前状态台账。
- `BACE-work/`、`issues/`：早期实现和问题调查记录。
- 根目录旧 H100/A100、StateID 和 GiGPO reference 文档：历史部署或独立设计资料，使用前先查文档清单。

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

交接时当前代码的参考结果是 `126 passed`。真实训练还需要仓库外的 Qwen 模型、ALFWorld 数据和 parquet，详见交接文档。

原上游项目为 [`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent)。迁入代码没有保留原上游 `.git` 历史，本仓库不是一组可直接 rebase 的干净 patch。
