# BACE Invalid Action 三组消融实验报告

日期：2026-08-08（UTC）

## 1. 实验目的

本实验比较 Invalid Action 的三种处理方式，验证当前主方案 `strict_identity` 是否应继续作为默认方案：

1. `strict_identity`：逐个保留 raw invalid identity，允许其进入 ERV 和 targeted branch；
2. `valid_only_branch`：natural rollout 仍保留 invalid，但 branch candidate 排除 invalid；
3. `single_invalid_bucket`：统计上将 invalid 合并为 `invalid::<INVALID_BUCKET>`，Replay 仍使用 concrete origin 的原始 response。

三组使用相同的 Qwen2.5-1.5B-Instruct、单卡 NPU、ALFWorld、batch size 2、seed 0/1、每 seed 2 个训练步、每任务 leaf budget 4 和 staged dynamic topology。已有 strict run 作为主方案基线，本轮只新增运行两种消融；三者不覆盖彼此的 Trace。

本实验不是为了替换主方案，而是回答两个问题：invalid action 是否需要被 targeted branch 探索，以及逐 raw identity 的统计精度是否值得其 action fragmentation 成本。

## 2. 运行入口

矩阵脚本现在支持受校验的环境变量：

```bash
BACE_INVALID_ACTION_MODE=strict_identity
# strict_identity | valid_only_branch | single_invalid_bucket
```

默认值仍为 `strict_identity`，因此未传参时原有主方案行为不变。非法模式会在启动训练前直接退出。

本轮消融命令：

```bash
BACE_RUN_PREFIX=bace_1card_invalid_ablation_valid_only_v1_20260808 \
BACE_INVALID_ACTION_MODE=valid_only_branch \
BACE_SEEDS='0 1' TRAIN_DATA_SIZE=2 BACE_TOTAL_EPOCHS=2 \
BACE_TOTAL_LEAF_BUDGET=4 \
bash examples/bace_gigpo/run_alfworld_npu_1card_trace_matrix.sh

BACE_RUN_PREFIX=bace_1card_invalid_ablation_single_bucket_v1_20260808 \
BACE_INVALID_ACTION_MODE=single_invalid_bucket \
BACE_SEEDS='0 1' TRAIN_DATA_SIZE=2 BACE_TOTAL_EPOCHS=2 \
BACE_TOTAL_LEAF_BUDGET=4 \
bash examples/bace_gigpo/run_alfworld_npu_1card_trace_matrix.sh
```

## 3. Correctness 结果

每个 seed 有 2 步，每步 2 个任务，因此每个 mode 有 4 个 step、16 个 branch 请求。

| mode | requested | validated | skipped | discarded roots | Replay attempts | Replay validated | invalid selected branches |
|---|---:|---:|---:|---:|---:|---:|---:|
| strict_identity | 16 | 16 | 0 | 0 | 32 | 32 | 5/16 (31.25%) |
| valid_only_branch | 16 | 16 | 0 | 0 | 32 | 32 | 0/16 (0%) |
| single_invalid_bucket | 16 | 16 | 0 | 0 | 32 | 32 | 2/16 (12.50%) |

三组的每个 step 都满足每任务 `R=2, Q=2, B=4`。离线 validator 对六个新增 step 全部返回 `ok=true`，无 error、无 warning；每个 step 的 4 个 branch-origin 都通过 token ids、loss mask、action identity 和 rollout old-log-prob 精确复制审计。

因此，两种消融都没有引入 correctness 回归：valid-only 确实排除了 invalid branch，single-bucket 确实改变了统计选择，但 concrete Replay 仍严格验证原始 invalid action 的 transition。

## 4. 关键行为差异

| mode | 平均 effective anchors | invalid branch ratio | 平均 trainable occurrences | 解释 |
|---|---:|---:|---:|---|
| strict_identity | 2.625 | 31.25% | 44.5 | 保留全部 raw invalid 语义，branch 选择空间最大 |
| valid_only_branch | 2.000 | 0% | 47.25 | natural invalid 仍训练，但不贡献 targeted branch |
| single_invalid_bucket | 2.125 | 12.50% | 48.75 | 统计上合并 invalid，仍可能选择 concrete invalid origin |

seed0/step1 是最清晰的配对样本：三组 natural invalid occurrence 都为 `7/26`，但 strict 选择 `2/4` 个 invalid branch，single-bucket 选择 `1/4`，valid-only 选择 `0/4`。这证明消融切换的是 branch 统计语义，而不是 environment projection 或 natural action 生成。

valid-only 的有效 anchor 数在部分 step 降低到 `1.5`，因为 invalid action 被从 candidate set 排除；但在本次配置下仍有足够 capacity 完成 `R+Q=B`。single-bucket 没有简单等价于 strict：同一 natural roots 下它选择的 invalid branch 数不同，说明 bucket 合并确实改变了 posterior/ERV 的采样分布。

## 5. old-log-prob 与训练稳定性观察

四个 step 的 actor grad norm：

| mode | grad norm |
|---|---|
| strict_identity | 3.290, 6.588, 7.899, 2.855 |
| valid_only_branch | 3.045, 2.885, 9.331, 2.212 |
| single_invalid_bucket | 3.764, 1.101, 8.124, 2.306 |

三组的 `training/rollout_probs_diff_mean` 都约为 `0.006`，没有看到消融模式导致的整体概率漂移突变。old-log-prob 的 root、branch-origin、branch-suffix 差异仍存在，符合此前结论：主要问题来自 rollout backend 与 FSDP actor 重算路径的数值/执行差异，而不是 Invalid Action identity 复制错误。

四组 step 的 episode success rate 均为 0。这个实验只有 2 个训练步，不能用来判断哪种模式最终提升策略成功率；当前数据只支持工程正确性和采样分布层面的结论。

## 6. 对主方案的判断

当前仍保留 `strict_identity` 作为主方案，理由如下：

1. 它是唯一不主动丢弃 invalid targeted branch 的模式，保留了模型真实决策的完整反事实信号；
2. 它逐 raw identity 区分不同 invalid action，避免把可能具有不同后续反馈的动作错误聚合；
3. 它在本轮真实 ALFWorld 单卡实验中与其他模式一样满足所有 Replay 和 Trace 不变量；
4. valid-only 虽然统计更简单，但完全失去 invalid branch 的信息；
5. single-bucket 可作为 fragmentation 对照，但它改变了 action-level 统计语义，适合诊断，不应在没有更长训练证据前替代主方案。

这不是说两种消融没有价值：valid-only 能估计 invalid branch 的边际贡献，single-bucket 能估计 raw identity fragmentation 的代价。它们应继续作为对照组，而不是删除。

## 7. 数据位置

严格主方案基线：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/
  bace_1card_multitask_trace_v1_20260807_seed0/bace_trace/
  bace_1card_multitask_trace_v1_20260807_seed1/bace_trace/
```

valid-only：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/
  bace_1card_invalid_ablation_valid_only_v1_20260808_seed0/bace_trace/
  bace_1card_invalid_ablation_valid_only_v1_20260808_seed1/bace_trace/
```

single-bucket：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/
  bace_1card_invalid_ablation_single_bucket_v1_20260808_seed0/bace_trace/
  bace_1card_invalid_ablation_single_bucket_v1_20260808_seed1/bace_trace/
```

对应 validator reports：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/trace_validation/
  bace_1card_invalid_ablation_valid_only_v1_20260808_seed0.json
  bace_1card_invalid_ablation_valid_only_v1_20260808_seed1.json
  bace_1card_invalid_ablation_single_bucket_v1_20260808_seed0.json
  bace_1card_invalid_ablation_single_bucket_v1_20260808_seed1.json
```

每个 step 都保存 roots、leaves、anchors、topology、posterior snapshots、Replay attempts、branches、trainable occurrences、raw response、concrete action identity、token ids、loss mask 和 old-log-prob，可继续做逐 action/anchor/branch 分析。

本轮实验后的工程回归：

```text
tests/bace_gigpo: 40 passed
bash -n examples/bace_gigpo/*.sh: passed
compileall recipe/bace_gigpo tests/bace_gigpo: passed
git diff --check: passed
4 ablation validator reports: ok=true, 0 errors, 0 warnings each
```

## 8. 后续实验建议

下一阶段应保持 `strict_identity` 为主线，同时把两种消融扩大到至少 3-5 个 seed、20-50 个训练步，并固定数据顺序或记录完整 dataset fingerprint。重点比较：success rate、realized regret reduction、invalid branch 后缀收益、posterior entropy、anchor fragmentation、PPO clip fraction、throughput 和 old-log-prob 偏差随训练步的变化。

在更长实验前，不建议仅凭本次短 smoke 的 grad norm 或 token 数改变主方案；当前结果支持“strict 继续作为主方案，另外两组保留为诊断性对照”。
