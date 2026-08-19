# BACE ALFWorld Invalid Action / Replay Action Identity 修正实现记录（2026-08-07）

## 1. 依据与结论

本次实现严格依据《BACE_GiGPO_ALFWorld_Invalid_Action_Replay_Action_Identity_实现规范_2026-08-07.md》。主模式不删除、重采样或语义修正 environment-invalid action，而是把自然出现且 action body 可稳定解析的 invalid/no-op response 作为真实 decision edge 保留，并允许参与 anchor competition、ERV 和 branch replay。

旧实现将 `is_action_valid` 的“格式可解析”误当成“环境可执行”，同时 Replay 对所有 action 统一要求属于 admissible set。这会错误拒绝本来可通过相同 raw response 复现的 invalid/no-op transition。本次已改为严格 action identity 与 transition replay consistency。

## 2. Natural rollout 修正

ALFWorld projection 不再原地修改 tokenizer decode 得到的 response。训练数据现在同时保存：

- `raw_model_response`：完整原始 CoT 与 `<action>...</action>`；
- `projected_action`：实际送入 TextWorld 的 parsed action body；
- `is_action_format_valid`：action body 是否可稳定解析；
- `is_action_environment_valid`：parsed action 是否属于 pre-action admissible set；
- `action_identity`：严格统计身份。

严格身份规则为：

```text
valid::<environment command>
invalid::<raw parsed action body>
```

无法解析 action body 的 response 仍保留在 natural training data 中，但 `action_identity=None`，不进入 BACE posterior/ERV candidate set。valid 与 invalid 即使文本相同也不会聚合；不同 invalid strings 默认不合并。

## 3. Candidate、posterior 与消融

`AnchorIndex` 默认使用 `strict_identity`，允许 valid 与 invalid observed identities 共同形成 effective anchor。Coordinator、dynamic topology capacity correction、posterior、ERV、origin fallback 和 action-mean local credit 均使用同一统计 identity。

配置提供三种模式：

```yaml
algorithm.bace.invalid_action_mode: strict_identity
# strict_identity | valid_only_branch | single_invalid_bucket
```

- `strict_identity`：主版本，逐 raw invalid identity 保留；
- `valid_only_branch`：natural training 保留 invalid，但 branch candidate 排除 invalid；
- `single_invalid_bucket`：仅统计上合并为 `invalid::<INVALID_BUCKET>`，concrete replay 仍复制所选 origin 的原始 response。

## 4. Replay 修正

Pre-action validation 继续检查 task/reset、prefix、anchor、prompt identity 和可选 action-set identity。只有 concrete origin 原本是 environment-valid 时，才要求 parsed action 在恢复后的 admissible set 中；environment-invalid origin 不再因此失败。

恢复 anchor 后，branch 提交 concrete origin 的完整 `raw_model_response`，再验证：

1. 实际解析得到的 concrete action identity 与 origin 一致；
2. post-action observation/feedback 与 natural occurrence 一致；
3. immediate reward 一致；
4. done/termination 一致。

只有 transition 验证通过才开始 fresh suffix rollout。invalid action 再次得到相同 `Nothing happens.` no-op feedback属于有效 replay，不是 failure。统计 action identity 与 concrete replay identity 分离，因此 single-bucket 消融也不会把 bucket 字符串提交给环境。

## 5. Trace 与诊断

Root/leaf、Replay request、branch 和 trainable occurrence 现在保存 raw response、parsed action、strict identity、format-valid 和 environment-valid。`summary.json` 新增：

- `invalid_occurrence_ratio`；
- `invalid_fragmentation_by_anchor`；
- `invalid_branch_ratio`；
- `selected_branch_count` / `invalid_selected_branch_count`；
- pre-anchor 与 origin-transition validation category/timing。

## 6. 验证结果

单元测试覆盖 projection 不改写 raw response、strict candidate 保留 invalid、valid-only/single-bucket 消融、invalid pre-replay 和 transition mismatch，完整 BACE 回归结果：

```text
38 passed
```

真实 ALFWorld 单卡 NPU staged dynamic smoke：

```text
run: bace_invalid_identity_npu_1card_smoke_20260807
requested: 2
validated: 2
ERV rounds: 2
invalid selected branches: 1
actor/grad_norm: 6.630
```

日志与 Trace：

```text
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/logs/bace_invalid_identity_npu_1card_smoke_20260807.log
/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/bace_invalid_identity_npu_1card_smoke_20260807/bace_trace/step_00000001
```

该 step 的 15 个可解析 natural occurrences 中有 7 个 environment-invalid，invalid ratio 为 `0.4667`；effective anchor 包含 4 个 candidate identities，其中 2 个为不同 invalid identities。两条 branch 中一条选择 invalid identity。2 次 pre-anchor validation 和 2 次 origin-transition validation 全部为 `VALIDATED`，证明 invalid/no-op edge 可以按规范端到端 replay 并进入 PPO actor update。

## 7. 保留边界

主版本仍不做 synonym mapping、nearest admissible action、embedding/LLM semantic canonicalization 或 invalid rejection sampling。Transition observation 当前采用 exact normalized object identity；如果未来环境 feedback 存在非语义随机文本，应新增明确、可审计的环境专用 transition signature，而不能用模糊语义匹配静默放宽。
