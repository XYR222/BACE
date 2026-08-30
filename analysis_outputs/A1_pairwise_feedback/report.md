# A1 Full Batch → Pairwise K=2 离线反馈审计

## 结论

工程判定：**HOLD / ABLATION ONLY**。判定依据：some adaptive sensitivity exists, but the strong engineering gate is not met。

在 `2400` 个训练 task 中，`Q>=3` 的 task 有 `1130` 个；其中 model-based A1.2 可用 `1130` 个，realized A1.1 可用 `1130` 个。整体 realized allocation replan rate 为 `45.575%`，P-Fixed 相对 Full Batch 的 expected gain 均值为 `0.0022791961`、中位相对收益为 `0.695%`。

这只说明当前 Beta posterior + BERV acquisition objective 下的离线 acquisition value；**不能直接推出 validation success 会提升**。

## 计划中的 11 个问题：直接回答

1. `Q>=3` 共 `1130` / `2400` 个 task（`47.083%`）。
2. 机会比例从 early `10.625%` 增至 middle `59.750%`、late `70.875%`；机会主要在 late，但收益并未随机会同步扩大。
3. 第一对真实 outcome 后，anchor allocation 有 `45.575%` 改变。
4. action multiset 改变率为 `80.885%`，明显高于 anchor allocation 改变率；反馈更多改变局部 action plan，而不只是把 slot 搬到另一 anchor。
5. model-based P-Fixed absolute gain 均值 `0.0022791961`，中位相对 gain `0.695%`。
6. early/middle/late 的中位相对 gain 见下表；它们都远低于 5% 强门槛，因此结论不是由后期 BERV 绝对尺度单独造成。
7. P-Threshold 有 `137` / `1130` 个 task 存在非零 shortfall 概率，平均 shortfall 概率 `1.408%`；历史真实首对 outcome 中 `2` 个 task 发生 shortfall。
8. mean expected gain 最大的 family 是 `pick_and_place`；但 family 间 replan 与 gain 排名不完全一致，说明“改计划”不等于“价值大”。
9. `2 + remaining frozen` 的首轮适应 gain 均值 `0.0018538809`，后续继续 Pairwise 的额外 gain 均值 `0.0004253152`；后续轮有价值，但只占总均值的一部分。
10. 工程结论为 **HOLD / ABLATION ONLY**：适合作为受控 ablation，不足以直接替换 Full Batch 主线。
11. 必须设计 mid-branch quota/fallback 语义：shortfall 概率虽小但非零，不能依赖“通常不会发生”。

## 数据完整性与数学复现

- 扫描 step：`150`；有可用 `Q>=3` task 的 step：`121`。
- 损坏/缺失记录：`0`；缺失 branch outcomes：`0`。
- 全 root-only step 按设计不生成的 acquisition/posterior 流：`10`（不计为损坏）。
- Full Batch `global_optimal_value` 成功复现：`1130`；不匹配：`0`。
- 同 action 重复采样复用生产 `ExactBatchErvEngine._beta_binomial_probability`，没有按固定均值独立 Bernoulli 近似。
- 每个 anchor 的 `L_max=2` 在 pair 之间累计扣减。
- 历史 Full Batch global tie count >1 的 task 有 `588` 个；主分析复用 stable seed 选一个 tie-optimal 路径，不跨 UUID 比较 request identity。

完整明细见 `data_coverage_report.json`。若存在坏记录，它们被显式排除而非静默补齐。

## A1.1：Realized Frozen-Plan Feedback

第一 pair 严格按历史 `requests` 的物理冻结顺序读取，再用 `completed_branch_outcomes` 更新 posterior。该结果衡量当前 frozen Full Batch plan 对真实结果的敏感性，不是 K=2 的无偏反事实。

## A1.2：Model-Based Exact Pairwise

- **P-Fixed**：初始 Q、root support 和每-anchor 总 slot 上限冻结；每轮只求 `min(2,q)`，枚举 outcome 后重规划；中途不再用 threshold 缩 quota。
- **P-Threshold**：每轮 outcome 后重新计算 threshold capacity；若剩余 capacity 小于冻结的剩余 Q，立即停止并记 shortfall，不擅自用 root 补齐。
- `hybrid_2_then_frozen_value` 是“先做一对、看 outcome、随后一次性规划全部剩余”的诊断；`P-Fixed - hybrid` 隔离第二轮及以后继续重规划的额外价值。

## Phase 汇总

| phase   |   tasks_q_ge3 |   q_ge3_ratio |   realized_usable |   replan_rate |   action_change_rate |   realized_shortfall_rate |   mean_realized_feedback_gain |   mean_expected_feedback_gain_fixed |   median_expected_feedback_gain_fixed |   mean_relative_feedback_gain_fixed |   median_relative_feedback_gain_fixed |   mean_threshold_shortfall_probability |   mean_expected_executed_threshold |   mean_later_pairwise_gain_vs_hybrid |
|:--------|--------------:|--------------:|------------------:|--------------:|---------------------:|--------------------------:|------------------------------:|------------------------------------:|--------------------------------------:|------------------------------------:|--------------------------------------:|---------------------------------------:|-----------------------------------:|-------------------------------------:|
| early   |            85 |       0.10625 |                85 |      0.517647 |             0.764706 |                0          |                   0.00165876  |                          0.00337139 |                           0.00160535  |                           0.0274962 |                            0.00752614 |                             0.00035568 |                            4.10517 |                          0.000466016 |
| middle  |           478 |       0.5975  |               478 |      0.481172 |             0.8159   |                0          |                   0.00117389  |                          0.00224447 |                           0.000200115 |                           0.0169219 |                            0.00342869 |                             0.00726311 |                            4.37324 |                          0.000329294 |
| late    |           567 |       0.70875 |               567 |      0.425044 |             0.809524 |                0.00352734 |                   0.000845907 |                          0.00214474 |                           0.000391913 |                           0.0164207 |                            0.00763593 |                             0.0218791  |                            4.99202 |                          0.000500163 |

## Family 汇总

| task_family                    |   tasks_q_ge3 |   q_ge3_ratio |   realized_usable |   replan_rate |   action_change_rate |   realized_shortfall_rate |   mean_realized_feedback_gain |   mean_expected_feedback_gain_fixed |   median_expected_feedback_gain_fixed |   mean_relative_feedback_gain_fixed |   median_relative_feedback_gain_fixed |   mean_threshold_shortfall_probability |   mean_expected_executed_threshold |   mean_later_pairwise_gain_vs_hybrid |
|:-------------------------------|--------------:|--------------:|------------------:|--------------:|---------------------:|--------------------------:|------------------------------:|------------------------------------:|--------------------------------------:|------------------------------------:|--------------------------------------:|---------------------------------------:|-----------------------------------:|-------------------------------------:|
| pick_and_place                 |           255 |      0.477528 |               255 |      0.439216 |             0.752941 |                 0         |                   0.00188198  |                          0.00294786 |                           0.000747621 |                           0.0231716 |                            0.0143473  |                             0.0116079  |                            4.67349 |                          0.000458858 |
| look_at_obj_in_light           |           116 |      0.534562 |               116 |      0.465517 |             0.793103 |                 0         |                   0.000945701 |                          0.0021521  |                           0.000451797 |                           0.0161748 |                            0.00856399 |                             0.0109674  |                            4.6777  |                          0.000679245 |
| pick_heat_then_place_in_recep  |           166 |      0.491124 |               166 |      0.421687 |             0.813253 |                 0         |                   0.000620007 |                          0.00165149 |                           0.000235296 |                           0.0134755 |                            0.00416414 |                             0.0129076  |                            4.67624 |                          0.000230207 |
| pick_clean_then_place_in_recep |           185 |      0.445783 |               185 |      0.389189 |             0.843243 |                 0         |                   0.00073517  |                          0.00245215 |                           0.000458083 |                           0.0172922 |                            0.00843996 |                             0.0257209  |                            4.86927 |                          0.000533278 |
| pick_cool_then_place_in_recep  |           162 |      0.462857 |               162 |      0.438272 |             0.790123 |                 0.0123457 |                   0.000997494 |                          0.00177841 |                           0.000288543 |                           0.0158812 |                            0.0060824  |                             0.0166026  |                            4.45832 |                          0.000249966 |
| pick_two_obj_and_place         |           246 |      0.450549 |               246 |      0.552846 |             0.857724 |                 0         |                   0.000778935 |                          0.0022693  |                           3.3723e-15  |                           0.0160268 |                            2.6412e-14 |                             0.00847382 |                            4.6185  |                          0.000436746 |

## Q 基数（包括无 adaptivity opportunity）

|   Q |   tasks |
|----:|--------:|
|   0 |     845 |
|   1 |     236 |
|   2 |     189 |
|   3 |     150 |
|   4 |     274 |
|   5 |     465 |
|   6 |     241 |

`Q=0/1/2` 只计入基数和 opportunity denominator，不进入 A1 的主效应估计。

## 负收益 case（保留、不 clamp）

|   step | task_family                    |   Q |   full_batch_value |   pairwise_expected_value_fixed |   pairwise_expected_gain_fixed |
|-------:|:-------------------------------|----:|-------------------:|--------------------------------:|-------------------------------:|
|     78 | pick_and_place                 |   5 |          0.143388  |                       0.142172  |                   -0.00121617  |
|     81 | pick_clean_then_place_in_recep |   3 |          0.0717    |                       0.0709872 |                   -0.000712866 |
|    127 | pick_heat_then_place_in_recep  |   6 |          0.0562798 |                       0.0562798 |                   -7.0291e-15  |
|    118 | pick_heat_then_place_in_recep  |   5 |          0.0670928 |                       0.0670928 |                   -5.13478e-15 |
|    148 | pick_two_obj_and_place         |   6 |          0.107995  |                       0.107995  |                   -4.7462e-15  |
|    125 | pick_two_obj_and_place         |   5 |          0.105291  |                       0.105291  |                   -3.73312e-15 |
|    107 | pick_clean_then_place_in_recep |   6 |          0.0683374 |                       0.0683374 |                   -3.66374e-15 |
|    124 | pick_heat_then_place_in_recep  |   6 |          0.0991491 |                       0.0991491 |                   -3.63598e-15 |
|    148 | pick_two_obj_and_place         |   6 |          0.135529  |                       0.135529  |                   -3.44169e-15 |
|    128 | pick_heat_then_place_in_recep  |   6 |          0.0686423 |                       0.0686423 |                   -3.40006e-15 |

轻微或实质负值均被保留，因为 Pairwise 是 myopic K=2 policy，并不等价于全局最优 POMDP acquisition policy；tie choice 也可能影响具体路径。

其中低于 `-1e-10` 的实质负收益 task 有 `2` 个；其余表中负数为浮点量级。

## 第几轮贡献

`round_value_contributions.csv` 给出 Pairwise 各物理 pair round 对最终 expected information gain 的贡献。对 Q=5/6，`first_adaptation_gain_vs_full` 与 `later_pairwise_gain_vs_hybrid` 进一步区分“首对后一次重规划”和“后续继续 Pairwise”的价值。它们是 policy-granularity 分解，不应解释成训练回报的因果分解。

## 是否需要 quota/fallback 语义

P-Threshold 的 phase-level `mean_threshold_shortfall_probability` 直接回答该问题：只要它非零，真实 K=2 实现就不能只是把 Full Batch 拆成多个执行波；必须明确 early stop、branch→root fallback，或固定 Q 后取消中途 threshold 三者之一。本审计没有替主方法偷选其中任何一种。

## 输出

- `pairwise_task_audit.parquet`：每个 Q>=3 task 的 A1.1/A1.2 明细。
- `phase_summary.csv`、`family_summary.csv`、`q_summary.csv`：聚合统计。
- `round_value_contributions.csv`：逐 pair round expected information gain。
- `figures/`：计划要求的 6 张图。
- `resolved_analysis_config.json`：精确输入与方法口径。
