# BACE 当前文档入口

本目录是当前唯一的维护入口。结论优先级为：**当前代码与测试 > 当次 artifacts/metadata > 本目录 > 旧设计和计划文档**。

建议阅读顺序：

1. [当前真实进度与未完成事项](01_当前实现状态与未完成事项.md)：150-step 到底完成了什么、失败在哪里、当前代码又多了哪些修复。
2. [文档有效性清单](06_文档有效性清单.md)：逐类说明哪些文档必读、哪些仅作背景、哪些已过时或与当前代码不一致。
3. [新设备安装与首次运行](02_新设备安装与首次运行.md)：clone、环境、外部资产、dry-run 和最小 smoke。
4. [代码结构与数据流程](03_代码结构与数据流程.md)：natural root、Exact Batch-ERV、branch Replay、advantage、checkpoint 和 trace。
5. [资产与实验结果迁移清单](04_资产与实验结果迁移清单.md)：GitHub 不包含什么，续跑旧实验还需复制什么。
6. [验证验收与故障排查](05_验证验收与故障排查.md)：从单测到长跑的门禁和常见故障。
7. [当前 BACE 方法规范](07_当前BACE方法规范.md)：用当前代码口径解释预算、competence、Exact Batch-ERV、Replay、advantage 和 checkpoint。
8. [代码库阅读路线](08_代码库阅读路线.md)：从 launcher 到 trainer、collector、方法模块和测试的最短阅读路径。
9. [逻辑模式与兼容边界](09_逻辑模式与兼容边界.md)：区分 Exact 主线、legacy、frontier、selected-worker 和兼容默认。

## 可选的实验分析

[`target/` 分析与历史材料索引](../target/README.md) 对公开的实验分析、问题修复记录、初始任务输入和辅助脚本逐项分级。其中 [BACE 与 GiGPO 150-step TensorBoard 对比分析](../target/BACE_vs_GiGPO_150step_TensorBoard对比分析_2026-08-27.md) 是单 seed 经验报告，用于理解后期 branch/root 比例、validation 差异和系统成本，不是方法规范，也不能替代多 seed 消融实验。

## 一分钟结论

- 旧主实验已经完成 150 个训练 step，最终 validation success rate 87.50%。
- 作业的 FAILED 状态来自训练后的 trace validator，而非训练失败；完整解释及重新校验结果在 01 文档。
- 当前 GitHub HEAD 比长跑时的工作树更新，已加入若干稳定性修复，二者不能视为完全同一源码版本。
- 原 Slurm/H100 作业链对换设备没有复用价值；只把它当配置和事故证据。
- 新设备应先复现 `126 passed`、dry-run、一步 smoke 和跨进程恢复，再决定正式运行参数。

所有旧文档均保留，没有删除。读到任何“尚未完成 150 step”“当前阻塞于 step 3”“StateID 已实现”之类说法时，先回到 01 和 06 文档核对。
