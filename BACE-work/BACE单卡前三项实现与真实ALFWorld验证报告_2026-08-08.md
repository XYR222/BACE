# BACE 单卡前三项实现与真实 ALFWorld 验证报告

日期：2026-08-08（UTC）

## 1. 本轮目标与结论

本轮按“先单卡”的要求完成以下三项：

1. ALFWorld 多任务、多训练步、多 seed 的真实 NPU correctness smoke；
2. 按 `root`、`branch_origin`、`branch_suffix` 分来源调查 rollout old log-prob 与 actor 重算 old log-prob 的差异；
3. 实现可独立运行的 BACE Trace 离线不变量验证器，并用真实 Trace 完成审计。

最终结果：2 个 seed、每个 seed 2 个训练步、每步 2 个任务，共 4 个训练步全部完成。总计 16 个 branch 请求全部生成成功，32 次 Replay（每个分支包含恢复校验和 origin transition 校验）全部为 `VALIDATED`。离线验证器对两个 run 均给出 `ok: true`，无 error、无 warning；40 个 BACE 测试全部通过。

本轮证明的是 BACE 训练数据构造、Replay、invalid action identity、dynamic topology、old-log-prob 记录和离线审计链路在单卡真实 ALFWorld 环境中能够一致工作。由于只有 2 个训练步，且四步任务成功率均为 0，本轮不能用于判断策略训练收益。

## 2. upstream 基座

2026-08-08 重新执行了 `git fetch origin`。当前可获取的最新远端基座为：

```text
remote: https://github.com/langfengQ/verl-agent.git
origin/master: 20bd331bdbc9026a5668e11362178e10ab7400c8
subject: fix(hgpo): use recipe.hgpo.main_hgpo entrypoint in train scripts (#260)
```

当前 `HEAD` 为 `14e5104a02d53b06cc3dbc3bc69d6707e8c19ade`，其 merge-base 正是上述 `origin/master`，远端相对本地没有新增提交，本地在该基座上领先 3 个已有 NPU smoke 基线提交。本轮实现继续保留 fixed、random、candidate-root preallocation 等原有可选逻辑，没有删除旧路径。

## 3. 实现内容

### 3.1 分来源 old-log-prob 诊断

在 `recipe/bace_gigpo/rollout_collector.py` 的训练诊断保存链路中，为每个可训练 occurrence 保存：

```text
response_token_ids
response_loss_mask
rollout_old_log_probs
recomputed_old_log_probs
rollout_vs_recomputed_old_log_prob_max_abs_diff
rollout_vs_recomputed_old_log_prob_mean_abs_diff
rollout_vs_recomputed_probability_max_abs_diff
```

同时在每步 `summary.json` 中增加三个互斥来源的聚合统计：

```text
diagnostics.old_log_prob_by_source.root
diagnostics.old_log_prob_by_source.branch_origin
diagnostics.old_log_prob_by_source.branch_suffix
```

这样可以区分自然 root token、从自然轨迹复制到 branch 的 origin token，以及 Replay 后新生成的 suffix token，避免只看全局最大值时误判问题来源。

### 3.2 离线 Trace 不变量验证器

新增：

```text
recipe/bace_gigpo/validate_trace.py
```

基本用法：

```bash
python -m recipe.bace_gigpo.validate_trace /path/to/bace_trace
```

可选输出和阈值：

```bash
--output report.json
--max-recomputed-logprob-diff VALUE
--max-recomputed-probability-diff VALUE
```

验证器检查：schema 和 step 一致性、记录数与连续 `record_index`、root/occurrence 唯一性、valid/invalid action identity、每任务 `R + Q = B`、anchor branch 容量、topology 引用、Replay request 与 origin transition、失败分支不得进入训练、suffix 数量、同一 leaf 的优势一致性、ERV 多轮 posterior support 冻结，以及 branch-origin 的 token ids、loss mask、action identity、rollout old log-prob 是否与自然 origin 精确一致。

旧 Trace 若没有 token 数组会给 warning 并跳过复制审计；本轮新 Trace 保存了完整数组，因此 16 个 branch-origin 均实际执行并通过了复制审计。

### 3.3 单卡矩阵脚本

新增：

```text
examples/bace_gigpo/run_alfworld_npu_1card_trace_matrix.sh
```

默认矩阵为 seed `0 1`、每 batch 2 个任务、每 seed 2 个训练步、每任务 leaf budget 为 4，使用 staged dynamic topology 和 `strict_identity` invalid action 模式。每个 seed 完成后自动运行离线验证器。

第一次 seed 0 训练结束后，包装脚本调用了系统 `python`，系统环境缺少 `numpy`，导致训练后的验证入口退出。该问题不影响已经完成的训练或 Trace。脚本现增加 `BACE_PYTHON`，默认显式使用 ALFWorld 训练虚拟环境的解释器，并在运行前检查其可执行性。修正后先补验 seed 0，再单独续跑 seed 1，没有重复 seed 0 的 NPU 训练。

## 4. 实验配置与执行过程

核心配置：

```text
device: NPU 0, single card
model: Qwen2.5-1.5B-Instruct
train_batch_size: 2
steps per seed: 2
env.max_steps: 8
total_leaf_budget B: 4 per task
pilot roots: 2
topology: dynamic
dynamic_root_generation: staged
invalid_action_mode: strict_identity
artifacts.enabled: true
```

实际续跑命令与完整矩阵等价：

```bash
BACE_RUN_PREFIX=bace_1card_multitask_trace_v1_20260807 \
BACE_SEEDS='0 1' \
TRAIN_DATA_SIZE=2 \
BACE_TOTAL_EPOCHS=2 \
BACE_TOTAL_LEAF_BUDGET=4 \
bash examples/bace_gigpo/run_alfworld_npu_1card_trace_matrix.sh
```

覆盖到 4 类任务：`pick_and_place`、`pick_heat_then_place_in_recep`、`pick_cool_then_place_in_recep`、`pick_clean_then_place_in_recep`。

## 5. 四步 correctness 结果

| seed/step | 任务族 | requested/validated | 每任务 R/Q/B | invalid occurrence | invalid branch | Replay | grad norm | step 秒 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0/1 | clean, heat | 4/4 | 2/2/4 | 7/26 (26.92%) | 2/4 (50%) | 8 VALIDATED | 3.290 | 309.982 |
| 0/2 | clean, cool | 4/4 | 2/2/4 | 8/26 (30.77%) | 1/4 (25%) | 8 VALIDATED | 6.588 | 269.737 |
| 1/1 | pick-and-place, clean | 4/4 | 2/2/4 | 10/27 (37.04%) | 1/4 (25%) | 8 VALIDATED | 7.899 | 258.173 |
| 1/2 | clean, cool | 4/4 | 2/2/4 | 9/29 (31.03%) | 1/4 (25%) | 8 VALIDATED | 2.855 | 271.616 |

四步的 `generated_roots_mean=2`、`final_roots_mean=2`、`discarded_roots_mean=0`，说明 staged root generation 确实按最终预算生成，没有回到“先生成 4 个 root 再丢弃 2 个”的计算浪费路径。四步均包含被选中的 invalid identity 分支，因而 Replay 验证覆盖了真实 invalid/no-op action，而不只是合法动作路径。

## 6. old-log-prob 分来源结果

每步数据如下，单元格为 `max abs log-prob diff / mean abs diff / max probability diff`：

| seed/step | root | branch-origin | branch-suffix |
|---|---|---|---|
| 0/1 | 0.370640 / 0.024194 / 0.122437 | 0.340992 / 0.023941 / 0.074812 | 0.412022 / 0.019865 / 0.071662 |
| 0/2 | 0.478439 / 0.025348 / 0.134482 | 0.235894 / 0.024665 / 0.111402 | 0.320922 / 0.022183 / 0.089105 |
| 1/1 | 0.417061 / 0.024265 / 0.075713 | 0.217990 / 0.027509 / 0.075713 | 0.235992 / 0.024081 / 0.085684 |
| 1/2 | 0.579654 / 0.023118 / 0.103542 | 0.202609 / 0.020807 / 0.056395 | 0.339169 / 0.024370 / 0.108571 |

按 token 数加权汇总：

| 来源 | token 数 | weighted mean abs diff | 全局 max abs diff | 全局 max probability diff |
|---|---:|---:|---:|---:|
| root | 12106 | 0.024218 | 0.579654 | 0.134482 |
| branch-origin | 1383 | 0.024392 | 0.340992 | 0.111402 |
| branch-suffix | 3166 | 0.022495 | 0.412022 | 0.108571 |

结论：三类来源的加权 mean abs diff 都在 `0.0225` 到 `0.0244`，root 在 4 步中的 3 步具有最大的单 token 极值。因此差异不是 branch Replay 拼接或 branch-origin 复制独有的问题，更符合 vLLM rollout backend 与 FSDP actor 重算路径之间的数值/执行差异。离线验证器已逐分支证明 branch-origin 的 token、mask 和 rollout old log-prob 与自然 origin 精确相同，排除了“复制时改坏 old log-prob”的假设。

目前没有理论或经验确定的失败阈值，所以本轮只记录分布，不把非零偏差直接判为错误。验证器已经支持实验时传入阈值；后续应结合 PPO ratio、clip fraction、模型规模和更长训练确定告警线。

## 7. Trace 与报告位置

训练日志：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/logs/
  bace_1card_multitask_trace_v1_20260807_seed0.log
  bace_1card_multitask_trace_v1_20260807_seed1.log
```

完整 rollout 与逐步 BACE Trace：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/
  bace_1card_multitask_trace_v1_20260807_seed0/bace_trace/step_00000001
  bace_1card_multitask_trace_v1_20260807_seed0/bace_trace/step_00000002
  bace_1card_multitask_trace_v1_20260807_seed1/bace_trace/step_00000001
  bace_1card_multitask_trace_v1_20260807_seed1/bace_trace/step_00000002
```

离线验证报告：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/trace_validation/
  bace_1card_multitask_trace_v1_20260807_seed0.json
  bace_1card_multitask_trace_v1_20260807_seed1.json
```

每步目录包含 manifest、roots、leaves、anchors、topology、posterior snapshots、acquisition rounds、Replay attempts、branches、trainable occurrences 和 summary。关键 token、mask、old log-prob、action identity、环境 reset key、Replay 结果与拓扑引用均已保存，可支持后续逐 token、逐 action、逐 anchor 和逐分支排查。

## 8. 最终验证

```text
tests/bace_gigpo: 40 passed in 62.99s
compileall recipe/bace_gigpo tests/bace_gigpo: passed
bash -n examples/bace_gigpo/*.sh: passed
git diff --check: passed
seed 0 offline validator: ok=true, 0 errors, 0 warnings
seed 1 offline validator: ok=true, 0 errors, 0 warnings
```

## 9. 剩余风险与下一步

1. 本轮只验证单卡和短 horizon smoke；尚未验证多卡下 batch 重排、跨 rank 聚合和并行 rollout 对 occurrence identity/old-log-prob 的影响。
2. 四步 episode success 均为 0，说明样本长度足以验证工程链路，但不能验证 BACE 的最终训练效果。下一步应先做单卡中等长度小规模训练，观察 success、ratio、clip fraction 和来源偏差随 step 的趋势。
3. old-log-prob 的平均差异在三个来源上稳定存在。后续应做同 token、同权重、同 attention mask 下的 vLLM/FSDP 定点对照，定位 dtype、padding/remove-padding、采样 logprob 提取或模型同步中的具体贡献。
4. Trace 保存的是完整诊断数据，便于复现但磁盘占用较大。长实验前应增加按 step 保留策略或压缩归档，同时不能删减失败分支、Replay 请求和 token 级关键字段。
5. 当前没有自动启用跨 action/anchor fallback；本轮 strict identity Replay 已全部成功，没有证据需要改变 ERV 的采样语义。
