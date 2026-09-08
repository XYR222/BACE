# BACE / Exact Batch-ERV

本仓库包含 BACE-GiGPO 的方法实现、ALFWorld/WebShop 环境适配、训练入口、测试与实验交接资料。当前工作入口统一放在 [`target-09-09/`](target-09-09/)；历史文档和旧脚本仅用于追溯，不应替代该目录中的说明。

## 快速入口

- [`target-09-09/target1-method_seeds_test.md`](target-09-09/target1-method_seeds_test.md)：BACE C0–C8 双卡、多 seed 实验入口与验收方式。
- [`target-09-09/target2-GiGPO-refer.md`](target-09-09/target2-GiGPO-refer.md)：原版 GiGPO ALFWorld 对照实验入口。
- [`target-09-09/当前运行环境安装清单.md`](target-09-09/当前运行环境安装清单.md)：环境版本、外部资产与重建步骤。
- [`verl-agent-src/examples/bace_gigpo/README.md`](verl-agent-src/examples/bace_gigpo/README.md)：各 BACE 方法对应的正式脚本。

## 仓库结构

```text
work-BACE/
├── target-09-09/               # 当前目标、实验入口和环境清单
├── verl-agent-src/             # 训练代码主体（迁入的 verl-agent）
│   ├── recipe/bace_gigpo/      # BACE 调度、Replay、credit、trace 与 checkpoint 逻辑
│   ├── agent_system/           # rollout 和 ALFWorld/WebShop 环境接口
│   ├── examples/bace_gigpo/    # BACE C0–C8 的运行与提交脚本
│   ├── examples/gigpo_trainer/ # 原版 GiGPO 入口与 Slurm wrapper
│   ├── tests/                  # BACE、trainer 和环境专项测试
│   └── tools/                  # 日志与 TensorBoard 辅助工具
├── action_problem/             # 动作同一性与 Search-Augmented QA 设计资料
├── improve-plan/               # 方法变体、调度和实验配置规范
├── BACE-results-vanilla/       # 历史 BACE 实验记录与分析脚本
├── GiGPO_reference_analysis/   # GiGPO reference 数据分析与复现资料
├── analysis/                   # 可复核的离线分析代码
├── analysis_outputs/           # 小型离线分析结果
├── archived_launchers/         # 已停用的历史启动脚本；不可用于新实验
├── docs/                       # 早期代码、部署和交接说明
├── deploy/                     # GPU、依赖和资产预检工具
├── BACE-work/                  # 早期实现与设计记录
├── BACE-work-2/                # 后续方法设计背景资料
└── issues_old/                 # 已归档的问题调查
```

## 核心代码位置

| 内容 | 位置 |
| --- | --- |
| BACE rollout 与动态 topology | `verl-agent-src/recipe/bace_gigpo/rollout_collector.py`、`topology.py` |
| Batch-ERV 与 competence history | `verl-agent-src/recipe/bace_gigpo/batch_erv.py`、`competence.py` |
| C0–C8 credit/advantage | `verl-agent-src/recipe/bace_gigpo/advantage.py`、`flat_leaf.py` |
| Branch Replay | `verl-agent-src/recipe/bace_gigpo/replay/` |
| Trainer 接入 | `verl-agent-src/verl/trainer/ppo/ray_trainer.py` |
| 主配置 | `verl-agent-src/verl/trainer/config/ppo_trainer.yaml` |
| Trace 记录与校验 | `verl-agent-src/recipe/bace_gigpo/artifacts.py`、`validate_trace.py` |
| 方法启动脚本 | `verl-agent-src/examples/bace_gigpo/` |

## 阅读顺序

1. 根据任务阅读 `target-09-09/` 中对应的目标文档。
2. 查看 `verl-agent-src/examples/bace_gigpo/README.md` 和具体运行脚本。
3. 按“collector → topology/replay → advantage → trainer → trace”的顺序阅读实现。
4. 需要设计背景时再查阅 `improve-plan/` 和 `action_problem/`。
5. `archived_launchers/`、`BACE-work*/`、`issues_old/` 只用于历史追溯。

## 仓库外资产

模型、ALFWorld cache、训练 parquet、Python 环境、checkpoint、日志和 `experiments/` 不纳入 Git 仓库。所需路径和版本见 [`target-09-09/当前运行环境安装清单.md`](target-09-09/当前运行环境安装清单.md)。

上游项目：[`langfengQ/verl-agent`](https://github.com/langfengQ/verl-agent)。本仓库保存的是项目工作快照，不包含原上游仓库的完整 Git 历史。
