# BACE 完整方案实现审计与实验 Trace 设计（2026-08-07）

## 1. 目的与基线

本文对照《BACE完整方案_最终版_2026-08-05(1).md》和《BACE代码实现架构与Replay工程规范_2026-08-06.md》，记录当前 latest-upstream 基座上的实际实现状态，并定义可用于失败排查、统计分析和可复现实验的 CPU 侧 Trace。

代码基座为 `langfengQ/verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`。原有 `dynamic_root_generation: preallocated`、fixed topology 和 random acquisition 均保留；新增能力通过配置选择，不删除旧路径。

## 2. 完整方案实现审计

| 模块 | 状态 | 当前实现 |
|---|---|---|
| Natural root 与不可变 RootEventLog | 已实现 | 保存 task/root/occurrence、精确 observation/action、token、reward、reset key 和 horizon |
| Exact structural anchor | 已实现 | 仅使用 natural roots 构建，要求重复状态和至少两个 observed actions |
| Fixed topology | 已实现 | 固定 R/Q，支持 random 或 ERV acquisition |
| Dynamic topology | 已实现 | competence history、pilot readiness、容量修正和 `R + Q = B` |
| Staged natural-root generation | 已实现 | pilot 后按需生成 root，消除未选 candidate roots 的 rollout 计算；旧 preallocated 仍为可选路径 |
| Beta posterior / regret / exact ERV | 已实现 | sequential ERV，每轮更新 posterior，support 冻结 |
| Leaf-uniform global credit | 已实现 | 同 leaf 的 occurrences 共享 leaf global credit |
| Local credit | 已实现 | `occurrence` 与 `action_mean` 两种模式 |
| ALFWorld replay | 已实现并做过真实 NPU smoke | reset key、prefix action、observation/action-set、prompt identity 验证 |
| WebShop replay | 工程实现完成 | fake-env 回归通过；旧 gym 依赖与数据服务未就绪，尚未做真实 WebShop/NPU smoke |
| CPU artifact/trace store | 已实现 | per-step 原子 JSON、逐行 flush JSONL、可选 fsync、manifest/summary |
| Replay fallback | 部分实现 | 可配置同一 frozen `(anchor, action)` 内备用 natural origin；不扩展 support |
| Invalid/no-op action identity | 已实现 | 主模式保留严格 raw invalid identity，以 transition consistency 而非 admissible membership 判定 replay |
| 更高层 fallback | 未实现 | alternate action/anchor 会改变 acquisition 语义，当前不自动执行 |
| 完整性能 profiling | 部分实现 | 已保存 replay phase timing/env steps；NPU kernel、通信和显存 profiling 仍需专项实验 |

## 3. Trace 配置

```yaml
algorithm:
  bace:
    artifacts:
      enabled: true
      directory: null
      include_token_arrays: true
      fsync: false
    replay:
      compare_action_set: true
      max_origin_retries: 1
```

`directory: null` 时优先写到 `trainer.rollout_data_dir/bace_trace`，否则写到 `trainer.default_local_dir/bace_trace`。每个训练 step 使用独立目录：

```text
bace_trace/
  step_00000001/
    manifest.json
    roots.jsonl
    leaves.jsonl
    anchors.jsonl
    topology.jsonl
    acquisition_rounds.jsonl
    posterior_snapshots.jsonl
    replay_attempts.jsonl
    branches.jsonl
    trainable_occurrences.jsonl
    summary.json
```

同一步重复运行不会覆盖旧数据，而是创建 `step_00000001_attempt_01`。JSONL 每写一条立即 flush；即使训练在 advantage 或 backward 前失败，已完成的 rollout/replay 证据仍保留。`fsync: true` 可进一步要求落盘，但会增加 I/O 开销。

## 4. 关键记录

### 4.1 Root、leaf 与 topology

`roots.jsonl` 同时记录 selected roots 和 preallocated 模式下被丢弃的 candidate roots，并用 `selected_for_training` 区分。`leaves.jsonl` 保存逐 natural occurrence 信息。`anchors.jsonl` 保存 frozen anchor、observed action 列表和每个 action 的 natural origins。`topology.jsonl` 保存计划、最终 root IDs 和每任务 R/Q。

当 `include_token_arrays: false` 时，root/leaf 中的大型 prompt/response/loss-mask/log-prob 数组省略；其余结构诊断仍保存。

### 4.2 Acquisition 与 posterior

每轮 `acquisition_rounds.jsonl` 保存：

- 全部候选 anchor 的 utility、regret、per-action ERV 和 selection probability；
- 每个 action 的 alpha/beta、natural/branch success/failure evidence；
- 选中的 anchor/action/request；
- 选择前 posterior snapshot。

选择后的 posterior 写入 `posterior_snapshots.jsonl`。日志使用 coordinator 已缓存的 ERV 结果，不重新执行 Monte Carlo evaluation，因此不会推进 RNG 或改变后续选择。

### 4.3 Replay 与 fallback

`replay_attempts.jsonl` 分别保存 `initial_validation_N` 和 `aligned_batch_validation`，包括完整 request/result、prefix length、恢复 observation/action set、失败 category/error 和 batch elapsed time。

ALFWorld environment-invalid action 不再统一判为 `SELECTED_ACTION_NOT_EXECUTABLE`。只有 natural origin 本身为 environment-valid 时才要求恢复后仍属于 admissible set；invalid/no-op origin 提交完整 raw response 后，检查 concrete identity、post observation、immediate reward 和 done 是否复现。相关实现与真实 NPU 证据见《BACE_ALFWorld_Invalid_Action_Identity修正实现记录_2026-08-07.md》。

若初始 origin 失败且 `max_origin_retries > 0`，仅从同一 frozen exact anchor、同一 selected action 的未使用 natural origins 中重建 request。branch ID 保持不变，request ID 更新，coordinator 的 pending selection 同步迁移。若无备用 origin 或重试仍失败，该 branch 不进入训练，Trace 保留实际 leaf-count deficit；当前不会静默切换 action 或 anchor。

### 4.4 训练 occurrence 与 old-log-prob 审计

advantage 计算完成后，`trainable_occurrences.jsonl` 逐 occurrence 保存：

- task/leaf/traj/occurrence ID、source type、anchor/action 和 terminal reward；
- leaf、local、final occurrence advantage；
- masked token advantage mean 和 response token count；
- rollout 与 actor 重算 old-log-prob 的最大绝对差；
- branch-origin copied old-log-prob 专项差异。

`summary.json` 汇总 BACE metrics、各流 record counts、Replay category 分布、phase timing、Replay environment steps、candidate action counts、root/branch occurrence 数和 token 占比、最大 old-log-prob 差异。

## 5. 当前验证与边界

已完成 artifact store 单元测试、origin fallback support 约束测试、现有 BACE 回归、compileall 和 `git diff --check`。此前 staged dynamic topology 已通过真实 ALFWorld 单卡 NPU smoke，并确认 `generated_roots_mean=2`、`discarded_roots_mean=0`、`final_roots_mean=2`、`final_branches_mean=4`。

本轮新增 Trace 与 fallback 已完成真实 ALFWorld 单卡 NPU 单步 smoke：

```text
run: bace_staged_trace_npu_1card_smoke_20260807
log: /opt/dpcvol/datasets/8165423358032568398/AESC-exp/logs/bace_staged_trace_npu_1card_smoke_20260807.log
trace: /opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/bace_staged_trace_npu_1card_smoke_20260807/bace_trace/step_00000001
```

该运行完成 PPO actor update，`actor/grad_norm=6.281`；staged topology 得到 `generated_roots=2`、`discarded_roots=0`、`requested=2`、`validated=1`。Trace 完整生成 11 类文件，summary 记录 22 个 trainable occurrences、16 个 natural leaves、4 次 replay attempts、1 条有效 branch 和 1 条 exhausted branch。Replay 分布为 3 次 `SELECTED_ACTION_NOT_EXECUTABLE` 和 1 次 `VALIDATED`，其中一条 branch 的 alternate-origin retry 成功，证明 parent request/attempt/fallback chain 在真实环境中可用。

真实 WebShop/NPU smoke 仍受旧 gym 运行依赖和 WebShop 数据服务限制。

## 6. 后续实验建议

首次正式实验应保留 `include_token_arrays: true`，并在排查完成后评估文件体积。每个 step 优先检查 `summary.json` 的 validated/requested、Replay category、token fraction 和 old-log-prob diff，再沿 request ID 联查 acquisition、replay 和 branch。若同 `(anchor, action)` 的备用 origin 仍大量失败，应先修复环境确定性或 identity 规则，而不是启用跨 action/anchor fallback，否则会改变 ERV 选择分布和实验解释。
