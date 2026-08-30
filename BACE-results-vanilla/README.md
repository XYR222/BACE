# 旧 BACE 结果、分析与历史材料索引

> 本目录现在纳入 GitHub，但它不是当前方法规范的入口。先读 [`docs/README.md`](../docs/README.md)；本目录用于保存已经复核的实验分析、问题演化记录和辅助工具。文件名日期表示材料形成时间，不表示它仍描述当前代码状态。

## 分级说明

| 级别 | 含义 |
|---|---|
| B 当前分析/工具 | 对当前维护仍有直接价值，但实验结论受数据和 run 条件限制 |
| C 历史问题/部署 | 用于理解修复来源或原集群操作，不是当前状态 |
| D 初始任务输入 | 当时的需求和上下文，已经被当前交接文档取代 |

## B：当前仍有用

| 文件 | 内容与限制 |
|---|---|
| `BACE_vs_GiGPO_150step_TensorBoard对比分析_2026-08-27.md` | 单 seed 的 150-step 效果、root/branch 结构和系统成本对比；两组 GPU/并行配置不同，不能单独作因果或严格性能结论 |
| `BACE_2卡结果与4卡及GiGPO对比_2026-08-29.md` | 2 卡 BACE 已完成 150 step、step150 为 86.72%；记录训练后 trace 尾部失败及与 4 卡/GiGPO 的可比性限制 |
| `BACE_Exact_BatchERV容量修正与Branch规划偏差详细说明_2026-08-26.md` | 对正式 BACE run 的 capacity correction、planned/final branch 偏差和成本审计 |
| `BACE_容量修正逐阶段与具体任务原因审计_2026-08-26.md` | 逐阶段、逐任务族解释 capacity correction；适合后续 controller 改进 |
| `analyze_bace_training_curves.py` | 从 TensorBoard event 和 BACE branch artifacts 导出曲线、CSV 和摘要的只读分析工具；运行需要 `tensorboard`、`pandas`、`numpy`、`matplotlib` |
| `export_2gpu_bace_tensorboard.py` | 导出 2 卡 BACE TensorBoard scalar 的只读工具；不包含原 event/checkpoint |

上面三份实验报告描述的是已经完成的历史 run，不等于当前 GitHub HEAD 已做同版本复跑。原 artifacts 没有随 GitHub 上传。

## C：历史问题和原部署记录

| 文件 | 当前边界 |
|---|---|
| `BACE_实现全面审计与Exact_BatchERV全局分配修复方案_重写版_2026-08-19.md` | 记录旧 Cartesian global allocation 的根因和 quota-aware DP 修复方案；当前 DP 已实现 |
| `当前问题说明-Exact-Batch-ERV指数爆炸.md` | 稳定性作业停在 step 3 时的现场说明；其中“150-step 尚未提交”等状态已经过时 |
| `BACE_GiGPO_Branch_Pipeline_Inactive与DoubleReplay问题说明_2026-08-19.md` | 记录旧 dense executor 的 inactive generation 和 double replay；当前推荐 selected-worker 路径已实现 active compaction，并从已验证 worker 直接执行 origin |
| `BACE_4卡与原版GiGPO参数对比及2卡脚本_2026-08-26.md` | 原 RWTH 4 卡 run、上游 GiGPO 和 2 卡入口的参数对照；不能作为新设备配置 |
| `migrate_verl_agent_env_to_rwth2089.sh` | 原集群环境迁移脚本，含 RWTH 绝对路径并会创建归档、staging 和最终环境；只作部署证据，不应在其他设备直接执行 |

## D：初始任务输入

| 文件 | 当前边界 |
|---|---|
| `情况说明.md` | 仓库刚迁入 H100 环境时的任务描述；路径和“当前目标”已经过时 |
| `进度说明2.md` | 正式实验启动前的计划输入；不代表当前进度 |

当前真实进度统一见 [`docs/01_当前实现状态与未完成事项.md`](../docs/01_当前实现状态与未完成事项.md)，当前方法见 [`docs/07_当前BACE方法规范.md`](../docs/07_当前BACE方法规范.md)，代码选择边界见 [`docs/09_逻辑模式与兼容边界.md`](../docs/09_逻辑模式与兼容边界.md)。

## 不纳入 Git 的内容

`__pycache__/` 和 `*.pyc` 是特定 Python 版本生成的缓存，不能作为源码、结果或可复现证据，因此继续由 `.gitignore` 排除。大型 TensorBoard、checkpoint、parquet 和原始日志不进入 Git。小型、已说明 provenance 的 PNG/CSV/JSON 可放入专门的 `analysis_outputs/`，并由该目录 README 明确它们是离线结果而非在线验收。
