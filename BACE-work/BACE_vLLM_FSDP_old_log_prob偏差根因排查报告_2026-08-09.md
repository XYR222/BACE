# BACE vLLM/FSDP old-log-prob 偏差根因排查报告

日期：2026-08-09

## 1. 结论

本次观测到的 old-log-prob 偏差不是 BACE、Replay、branch-origin 复制或 Invalid Action 方案造成的。

根因已经定位到以下粒度：**同一份 Qwen2.5-1.5B-Instruct 权重在 Ascend NPU 上，vLLM Ascend BF16 推理路径与 Transformers/SDPA BF16 teacher-forcing 路径使用了不同的算子、归约与低精度累加路径，因此产生稳定的逐 token 数值偏差。**

完全绕开 BACE、Replay、FSDP 分片和训练权重同步后，直接从同一份 safetensors 加载模型，固定 prompt 和生成 token，仍复现：

| 对照 | token 数 | 平均绝对 log-prob 差 | P95 | 最大差 | 平均绝对概率差 |
|---|---:|---:|---:|---:|---:|
| vLLM BF16 vs Transformers BF16 | 64 | 0.023451 | 0.083118 | 0.144597 | 0.006207 |
| vLLM BF16 vs Transformers FP32 | 64 | 0.024406 | 0.098746 | 0.128605 | 0.005417 |
| Transformers BF16 vs FP32 | 64 | 0.023250 | 0.085738 | 0.163131 | 0.006048 |

第一个结果与训练 trace 的 token 加权平均 `0.023905` 几乎一致。这是本次根因判断最关键的证据。

目前还不能严谨地把偏差归因到某一个具体 NPU kernel，例如仅归因于 paged attention、RMSNorm 或 matmul。要达到单 kernel 粒度，需要增加逐层 hidden state/logits 对照。但这不会改变上述工程根因结论。

## 2. 训练 trace 复查

复查 seed 0/1、step 1/2 的四份 strict-identity trace，共 178 个 occurrence、16,655 个有效 response token：

| 指标 | 数值 |
|---|---:|
| 平均绝对 log-prob 差 | 0.023905 |
| 中位数 | 0.004839 |
| P95 | 0.105493 |
| P99 | 0.165385 |
| 最大值 | 0.579654 |
| recomputed - rollout 有符号均值 | -0.001150 |
| 正差比例 | 48.60% |

有符号均值接近 0，正负比例接近各半，不符合固定 token 错位、固定 temperature 缩放或恒定权重漂移的特征。

按数据来源统计：

| source_type | 有效 token 数 | 平均绝对 log-prob 差 | 最大差 |
|---|---:|---:|---:|
| root | 12,106 | 0.024218 | 0.579654 |
| branch_origin | 1,383 | 0.024392 | 0.340992 |
| branch_suffix | 3,166 | 0.022495 | 0.412022 |

三类来源处于同一量级，偏差不是 branch 或 Replay 特有现象。

## 3. 已排除项

### 3.1 branch-origin 复制错误

通过 `branches.jsonl` 的 `origin_occurrence_id` 将 16 条 branch-origin 与对应 natural root 逐一匹配。每一对均满足：

- `response_token_ids` 完全相同；
- `response_loss_mask` 完全相同；
- rollout old-log-prob 逐位完全相同，最大差为 0；
- FSDP recomputed old-log-prob 逐位完全相同，最大差为 0。

因此 BACE 的 origin 复制、batch 合并和恢复顺序没有制造该偏差。

### 3.2 padding、长度或整体错位

误差与 response length 的相关系数为 `-0.0072`，与绝对 token 位置的相关系数为 `-0.0488`，均接近 0。不同长度区间的 occurrence 平均差也稳定在约 `0.022-0.025`。

如果 response slice 或 next-token 标签整体错一位，绝大多数 token 都会出现大幅且结构化的差异；当前分布不符合该模式。

### 3.3 KV-cache 增量解码本身

在 Transformers/SDPA BF16 路径中，对同一固定序列比较：

- 整段 forward，`use_cache=False`；
- 整段 forward，`use_cache=True`；
- prompt prefill 后逐 token KV-cache forward。

52 个 token 的 log-prob 均逐位完全相同，平均差和最大差都是 `0.0`。因此不能把问题简单归结为“生成使用 cache、重算不使用 cache”。差异来自 vLLM Ascend 与 Transformers 的后端实现，而不是 cache API 语义。

### 3.4 stale weight 或 BACE 更新时序

- 偏差在 step 1 已存在，此时尚无前一步 actor update 可造成 stale rollout 权重；
- step 1/2 以及不同 seed、Invalid Action 模式的均值都稳定在约 `0.021-0.025`，没有随更新累积；
- 固定输入实验让两个后端直接从同一模型目录独立加载同一份 safetensors，仍复现同量级偏差。

因此 stale weight 不是当前约 `0.02` 基线偏差的原因。训练内 FSDP 到 vLLM 的同步仍可独立增加权重 hash/assert 作为长期防护，但不是本次现象的主因。

### 3.5 temperature 和采样过滤

训练 rollout temperature 为 `1.0`，actor recompute 同样读取 rollout temperature。训练配置为 `top_k=-1`、`top_p=1`，不存在截断分布导致的 log-prob 口径变化。固定输入实验也使用相同设置。

## 4. 误差结构

误差与 rollout log-prob 的相关系数为 `-0.454`：token 概率越低，log-prob 绝对差通常越大。

| rollout log-prob 区间 | 平均绝对差 |
|---|---:|
| `[-0.01, 0]` | 0.000111 |
| `[-0.1, -0.01)` | 0.003170 |
| `[-0.5, -0.1)` | 0.015628 |
| `[-2, -1)` | 0.045634 |
| `[-5, -2)` | 0.057839 |
| `[-10, -5)` | 0.059837 |

这符合 BF16 下 logits 的小幅变化经过 `log_softmax` 后，在低概率 token 上形成更明显 log-prob 差异的特征。最大 log-prob 差不能单独代表概率分布发生同等幅度变化，应同时查看 probability diff。

## 5. 对训练正确性的影响

当前 PPO 实际使用 FSDP actor 重算的 `old_log_probs`，不是 vLLM 返回的 `rollout_log_probs`。同一次 actor 更新中的 new log-prob 也由 actor/FSDP 路径计算，因此 PPO ratio 的分母口径保持一致。vLLM 值目前保留用于诊断。

所以该偏差：

- 不会破坏 BACE 的 anchor/action/Replay 身份语义；
- 不会把 branch-origin 的 old-log-prob 错配到其他 token；
- 不会直接把 vLLM 数值当作 PPO ratio 分母；
- 会形成轻微的 hybrid-engine on-policy gap：采样来自 vLLM 数值分布，而优化基准来自 FSDP actor 数值分布。

这属于混合 rollout/training 后端的数值一致性问题，不是 BACE 方法逻辑问题。现有选择“PPO 使用 recomputed old-log-prob”是正确且必要的。

## 6. 后续建议

1. 保持当前主训练逻辑：继续使用 FSDP recomputed `old_log_probs` 参与 PPO，不切换为 rollout 值。
2. 增加独立诊断阈值，长期记录 log-prob/probability diff 的 mean、P95、max 和低概率分桶，不因单个低概率 token 的 max 自动终止训练。
3. 如需进一步压缩偏差，做后端消融：rollout dtype、attention backend、vLLM Ascend 版本、prefix/chunked prefill；每次使用同一固定 token fixture。
4. 如需定位到单 kernel，在模型首层、中层、末层及 lm_head 保存 FP32 cast 后的 hidden state/logits，比较第一次出现明显漂移的层。
5. 权重同步增加参数摘要或抽样 tensor checksum，作为与本问题相互独立的防回归检查。

原始关键指标保存在同目录的 `BACE_vLLM_FSDP_old_log_prob偏差诊断数据_2026-08-09.json`。
