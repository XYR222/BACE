# A0：2 卡 GiGPO / BACE 信用结构分解

## 结论先行

在可比的 2×H100、Qwen2.5-1.5B、ALFWorld、batch=16、rollout.n=8、gamma=0.95、`mean_std_norm`、`step_advantage_w=1` 条件下，BACE 确实提高了 local credit coverage，但没有出现“local credit 大幅失控”的证据：

- early（1–50）：BACE local magnitude share **41.83%**，GiGPO **41.97%**，几乎相同；
- middle（51–100）：BACE **46.49%**，GiGPO **45.31%**，高 **1.18 个百分点**；
- late（101–150）：BACE **59.16%**，GiGPO **57.45%**，高 **1.71 个百分点**；
- token-weighted share 的差异分别为 **−0.01、+1.23、+1.56 个百分点**；
- BACE 的 macro/local sign conflict 反而低于 GiGPO（late **9.00% vs 10.00%**）。

因此，当前 2 卡历史 run 支持“BACE 后期 local credit 增加，但幅度是渐进的”这一结论，不支持仅凭 A0 结果直接把 `step_advantage_w` 大幅下调，也不支持立即切换到 `action_mean`。

## 数据、运行身份与可比性

| run | 数据来源 | steps | occurrence | GPU/partition | 备注 |
|---|---|---:|---:|---|---|
| GiGPO | `experiments/gigpo-alfworld-reference/runs/seed_0_v2` | 150/150 | 603,786（本表 603,786） | 2×H100, c23g, job 3165280 | raw occurrence/token Parquet |
| BACE | `bace_artifacts/bace_alfworld_qwen2_5_1_5b_exact_2gpu_mem244_retry` | 150/150 | 541,376 | 2×H100, c23g, job 3233743 | Exact/staged/packed/selected-worker |

两者模型、ALFWorld 与 parquet 路径相同，TP=2、train/val batch=16/128、rollout.n=8、response max=512、PPO micro/logprob=32、LR=1e-6、gamma=0.95、invalid penalty=0.1。BACE 额外启用了 Exact Batch-ERV、动态/packed root、branch 执行和 artifact 记录。BACE 命令显式写了 `max_num_batched_tokens=8192,max_num_seqs=1024`；GiGPO 命令省略但解析到同一份 veRL 默认值。BACE 还显式开启了 `use_torch_compile=true`，而归档 GiGPO 命令未显式开启，这是性能/数值环境上的非算法差异，后续严格复现实验需统一。

两个 Slurm 作业的训练数据已落盘并达到 step 150，但 Slurm 最终状态为 `FAILED/1:0`：GiGPO/BACE 都是在训练结束后的记录/清理阶段退出非零（BACE 日志还出现截断 JSONL 的 trace validator 异常）。这不改变本报告对已写入 occurrence 的离线统计，但意味着不能把它们称为“形式上干净退出的同版本复现”。

## Early / middle / late 主指标

| run | phase | occurrences | local coverage | local magnitude share | token local share | sign conflict | mean local group |
|---|---|---:|---:|---:|---:|---:|---:|
| GiGPO | early | 282,841 | 39.51% | 41.97% | 41.71% | 4.31% | 7.77 |
| BACE | early | 272,512 | 42.05% | 41.83% | 41.69% | 3.96% | 7.62 |
| GiGPO | middle | 205,606 | 61.15% | 45.31% | 45.11% | 7.98% | 6.31 |
| BACE | middle | 172,096 | 65.24% | 46.49% | 46.35% | 7.46% | 6.64 |
| GiGPO | late | 115,339 | 70.73% | 57.45% | 57.02% | 10.00% | 5.79 |
| BACE | late | 96,768 | 71.89% | 59.16% | 58.58% | 9.00% | 5.75 |

`local coverage` 是非零 local advantage occurrence 的比例；真正近似 PPO 信号占比的是 absolute-magnitude share，token share 则按真实 response mask 加权。BACE 的 coverage 增幅（+2.54/+4.09/+1.16 pp）明显大于 magnitude 增幅（−0.14/+1.18/+1.71 pp），说明增加的 occurrence 并没有等比例放大 local 信号。

## BACE source decomposition

下表的 local-mass contribution 是该 source 占 BACE 当期全部 local absolute mass 的比例，而非该 source 子集内部重新归一化后的比例。

| phase | root occurrence | branch-origin occurrence | branch-suffix occurrence | root local-mass | branch-origin local-mass | branch-suffix local-mass |
|---|---:|---:|---:|---:|---:|---:|
| early | 95.73% | 0.21% | 4.06% | 92.86% | 0.48% | 6.66% |
| middle | 75.91% | 1.41% | 22.68% | 71.64% | 1.92% | 26.44% |
| late | 55.15% | 3.09% | 41.76% | 55.88% | 3.56% | 40.56% |

Branch suffix 在 late 已占 41.76% occurrence、40.56% local mass，是 BACE 与 GiGPO 的主要结构差异；但它没有造成与 occurrence 比例不成比例的 local mass。branch-origin 本身始终很小（late 仅 3.56% local mass）。

一个重要的解释边界是：本表的 root 行仍使用“全 batch 归一化后”的 root occurrence advantage，再按 source 汇总；它不是重新只用 root 计算一套 advantage。因此 root-only 可作为 source 对照，但不能被误读为一次独立的 root-only 训练。

## Reconstruction 与数据质量

- GiGPO 150/150 个 step、token count 全部可取得；BACE 150/150 个 step、`response_token_count` 全部可取得。
- BACE `occurrence_advantage ≈ macro_advantage + local_advantage`：最大绝对误差 **4.77e−7**，通过浮点误差检查。
- GiGPO 的 archived component 与按 A0 physical-occurrence 口径重建的 component 存在历史归档差异：step 1 的 local max error 0.169，step 150 为 0.711；完整归档审计已记录 **1,366 episode-advantage occurrences / 11 groups**、**1,051 step-advantage occurrences / 139 groups** 的 mismatch，combined advantage 仍在 **2e−6** 内一致。这是旧归档的近零方差/归一化口径差异，不应把旧 component 当作修复后 contract 的精确 ground truth。
- 本次 BACE occurrence 扫描没有发现截断 trainable-occurrence record；但训练日志中的全量 trace validator 曾报告其他 JSONL 的截断，因此这里的 coverage 不能替代完整 trace 验证。

## 对方法选择的回答

1. **GiGPO local credit 是否自然增加？** 是。coverage 与 magnitude 都从 early 到 late 上升，且 late conflict 也上升。
2. **BACE 是否增长更快？** coverage 增长略快；magnitude 仅 middle/late 高约 1–2 pp，属于温和偏移。
3. **差异来自 coverage 还是 magnitude？** 主要先来自 coverage，magnitude 放大较小。
4. **差异主要来自哪类 source？** late 的数量和 local mass 主要由 branch-suffix；branch-origin 贡献小。
5. **后期冲突是否增加？** BACE 从 3.96%→9.00%，但仍低于 GiGPO 的 10.00%，没有 BACE 特有的冲突爆发。
6. **是否支持下调 `step_advantage_w`？** 当前证据不足。若要做，应优先小范围 `1.0→0.75/0.5` 对照，并保持其余配置、seed、source weighting 不变。
7. **是否支持 `action_mean`？** 当前证据不足。先检查 branch anchor 的 action diversity 与 within-action noise；不要仅因 coverage 高就切换。
8. **是否值得 gradient decomposition？** 值得作为下一步诊断，尤其取 late checkpoint 比较 macro-only/local-only gradient norm 与 cosine；它能判断 1–2 pp 的 magnitude 差异是否被 PPO clipping/token weighting 放大。

## 产物

- `occurrence_metrics.parquet`：统一 occurrence 表（1,145,162 行）；
- `step_summary.csv`、`phase_summary.csv`、`source_summary.csv`、`action_structure_summary.csv`；
- `data_coverage_report.json`、`reconstruction_report.json`、`resolved_analysis_config.json`；
- `figures/` 下 5 张 PNG；
- 重跑入口：[`../a0_credit_decomposition.py`](../a0_credit_decomposition.py)。

本报告解释的是历史 2 卡 run 的 credit curriculum，不等同于当前 HEAD 在新 seed 上的因果实验。
