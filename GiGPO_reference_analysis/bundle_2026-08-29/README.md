# GiGPO ALFWorld Reference Run：复现与离线分析结果包

本目录汇总本次 GiGPO ALFWorld Reference Run 复现、smoke 验证和离线分析产生的可复核结果。原始结果位置保持不变；此处是一个独立副本。

## 内容

- `reference_archive/`：完整 `seed_0_v2` 的 raw Parquet、integrity、manifest、profiler、segments 和 validation 元数据。
- `analysis/`：最终 `gigpo_alfworld_reference_derived_v2r3_20260828`，包括完整 Markdown 报告、JSON/Parquet 分析表、8 张统计图和 32-case prefix replay 结果。
- `smoke_validation/`：两个 v2 smoke 运行的结果；其中 `gigpo_ref_v2_seed0_smoke_3252713` 为最终通过的 3-step smoke。
- `slurm/`：相关 Slurm stdout/stderr 日志。
- `source/analysis/`：离线分析和图表生成脚本。
- `source/entrypoints/`：Reference Run、archive/replay validator 和相关训练入口。
- `source/specifications/`：实验规范、BACE 方法规范和问题记录。

## 明确排除

为避免复制模型和占用大量空间，以下内容未纳入：

- 所有 `checkpoints/`（包括 actor、optimizer 和其他模型权重）；
- 模型目录、conda/verl-agent 环境；
- ALFWorld 数据集和外部缓存；
- AFH 仓库的其他无关文件。

`raw/` 中的 `game_file` 字段仍保留原始绝对路径，这是审计字段，不代表本包包含 ALFWorld 数据。

## 重要结论

- 最终归档 150 updates、603,786 occurrences、19,200 trajectories，训练自然 root 成功率 53.96%。
- Exact BERV 有效 anchor 中 85.26% 可容纳 2 个 branch；global DP 的 `Q=6` 可行率为 89.96%。
- 滞后 competence controller 计划 7,581 个 branch，capacity correction 后为 6,585 个。
- 32 条 prefix replay 全部通过五项检查。
- 旧 Reference Archive 仍保留 1,051 个 step-advantage occurrence 和 1,366 个 episode-advantage occurrence 的历史数值 mismatch；这是修复前归档的审计发现，不是本次新代码运行失败。敏感性复算显示 evidence 类型分类变化为 0。

推荐首先阅读：

```text
analysis/gigpo_alfworld_reference_derived_v2r3_20260828/GiGPO_ALFWorld_Reference_Run_完整离线分析报告_2026-08-28.md
```

## 校验

外层压缩包为 `GiGPO_reference_analysis_2026-08-29.tar.gz`。解压后应存在本 README、最终分析报告、26 项核心产物和 smoke 的 `summary.json` / `final_validation.json`。
