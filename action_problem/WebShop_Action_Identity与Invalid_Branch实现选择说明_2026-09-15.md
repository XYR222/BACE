# WebShop Action Identity 与 Invalid Branch 实现选择说明

> 日期：2026-09-15  
> 范围：`work-BACE` 中 WebShop 的 action projection、有效性分类、BACE 候选、replay 与 GiGPO invalid-action penalty  
> 本文记录当前已经确定的实现口径及其理由，不讨论 Search-HotpotQA 的独立适配细节。

## 1. 结论

WebShop 当前采用以下规则：

| 模型行为 | GiGPO 格式惩罚 | 是否进入 PPO | 是否进入 BACE 候选 |
|---|---:|---:|---:|
| 格式合法，环境执行有效操作 | 否 | 是 | 是 |
| 格式合法，环境将其处理为 no-op | 否 | 是 | 是 |
| 格式错误 | 是，系数 `0.1` | 是 | 否 |

核心选择是：

1. **格式合法但环境无效的动作仍是模型真实行为。** WebShop 会确定性地解析并处理它，即使结果是不改变页面的 no-op，因此它可以作为 BACE 的干预动作。
2. **格式惩罚与环境有效性分离。** 保持原版 GiGPO 语义，额外 invalid-action penalty 只惩罚输出协议错误，不惩罚格式正确但当前页面不可执行的动作。
3. **格式错误不进入 BACE acquisition。** 它仍进入 PPO 并接受格式惩罚，但没有稳定的受支持 action identity，不建立 BACE posterior，也不消耗 branch 预算。
4. **动作身份以 WebShop 实际 parser/dispatch 为准。** 不用比真实环境更严格的研究者自定义 parser，也不通过语义模型合并动作。

正式 WebShop 配置继续使用：

```text
algorithm.bace.invalid_action_mode=strict_identity
```

这与此前 ALFWorld 主实验的动作处理原则一致。

## 2. 为什么要区分三种概念

### 2.1 `format_valid`

它描述模型是否遵循输出协议。当前 WebShop 与 ALFWorld 共用的格式条件为：

- 存在完整且非空的 `<action>...</action>`；
- 存在完整的 `<think>...</think>`；
- 回复中不包含中文字符。

该字段写入：

```text
is_action_valid
is_action_format_valid
```

其中 `is_action_valid` 保持上游 GiGPO 的含义，供 invalid-action penalty 使用。

### 2.2 `environment_valid`

它描述投影后的动作是否满足 WebShop 当前状态下的真实 dispatch 条件：

- 非空 `search[...]`：环境调用 search；
- `click[x]`：只有 `x` 是当前页面的真实 clickable，且 `x != "search"` 时才调用 click；
- 其他情况：环境确定性执行 no-op。

该字段写入：

```text
is_action_environment_valid
```

它用于 action identity、replay 验证和诊断，不直接控制 GiGPO 格式惩罚。

### 2.3 BACE candidate eligibility

当前由 `format_valid`、action identity 和 `invalid_action_mode` 共同决定：

- `format_valid=false`：`action_identity=None`，从 BACE candidate set 排除；
- `format_valid=true, environment_valid=true`：以 `valid::...` 进入候选；
- `format_valid=true, environment_valid=false`：在 `strict_identity` 下以独立的 `invalid::...` 进入候选。

因此，`environment_valid=false` **不等于**“不是 BACE action”。这里的 `invalid` 更准确地表示“格式合法但该状态下没有有效 WebShop 操作”。

## 3. Projection 与原始模型回复

模型输出示例：

```text
<think>Open the matching item.</think>
<action>Click[B09ABC1234]</action>
```

WebShop projection：

1. 保留完整 raw response，供 PPO、trace 和 copied-origin replay 使用；
2. 抽取 `<action>` body；
3. 将抽取后的 action 转为小写；
4. 单独返回 `format_valid`。

环境实际接收：

```text
click[b09abc1234]
```

CoT 不属于 action identity。两个回复即使推理文本完全不同，只要环境最终执行同一个 parser-level action，就共享同一个有效动作身份。

实现位置：

- `verl-agent-src/agent_system/environments/env_package/webshop/projection.py`
- `verl-agent-src/agent_system/environments/strict_actions.py`

## 4. 为什么必须对齐真实 WebShop parser

WebShop 原始环境的 `parse_action` 使用：

```python
re.match(r"(.+)\[(.+)\]", action)
```

它不是 `fullmatch`。随后环境将 argument 转为小写，再进行 search/click dispatch。

此前 BACE helper 使用了更严格的 `re.fullmatch` 和 action-pool 近似，因此存在两类不一致：

1. `click[item]trailing text` 被真实环境解析为 `click[item]`，旧 helper 却判为不可执行；
2. action pool 可能包含 `click[search]`，但真实环境明确令它 no-op，旧 helper 却可能判为可执行。

此外，真实环境并不以页面是否显示 search template 作为调用 `browser.search` 的 dispatch 条件。因此 helper 也不应自行增加这一限制。

当前修复后的 helper 复刻原始 parser 与 dispatch：

```text
projected action
  -> upstream-compatible parse_action
  -> lowercase argument
  -> real search/click dispatch conditions
  -> parser-level canonical action
```

我们不修改 WebShop 原始环境 parser，因为修改它会改变 benchmark 和原版 GiGPO 的行为。BACE 只在外层读取并复现其语义。

## 5. Action identity 的具体定义

### 5.1 有效 search

```text
valid::search[actual query]
```

query 使用环境 parser 实际抽取并转小写后的内容，不使用 embedding、LLM 或同义词聚类。

### 5.2 有效 click

```text
valid::click[actual clickable]
```

clickable 必须来自执行前页面的 `text_to_clickable` 集合，并排除环境保留的 `search` label。

### 5.3 格式合法的 environment no-op

```text
invalid::<raw action body>
```

例如：

```text
invalid::click[price]
invalid::click[nonexistent product]
```

这些 action 不合并成统一 `INVALID` bucket。理由是它们是模型作出的不同选择，BACE 应分别估计其局部结果，而不是错误共享 posterior。

对 no-op identity 使用精确 action body 是保守策略：允许一定的 false split，避免把实际上不同的模型行为 false merge。

### 5.4 格式错误

```text
action_identity = None
canonical_action = INVALID（仅 trace fallback）
```

它不进入 BACE action set，但原始 PPO occurrence 仍保留。

## 6. GiGPO penalty 语义

训练器中的 penalty 只读取：

```text
is_action_valid == format_valid
```

当格式错误时，在该 occurrence 的最后一个有效 response token 以及 step reward 上减去 `0.1`。格式合法但 environment no-op 时不会额外减去 `0.1`。

这种选择的理由是保持与原版 GiGPO WebShop 配置一致：

- 格式错误是输出协议问题，使用显式局部 penalty；
- 当前状态下不可执行的合法动作通过环境 reward、最终任务结果和 GiGPO/BACE credit 学习；
- 不把两类失败混成同一种 reward shaping。

换言之：

```text
format penalty != environment failure signal
```

## 7. 为什么 no-op 仍进入 BACE

BACE 研究的是模型在局部状态采取某一具体动作后的 outcome uncertainty。格式合法 no-op 满足：

1. 它由模型真实生成；
2. 环境能够确定性解析；
3. 环境能够执行并返回 observation/reward/done；
4. 可以从同一 concrete occurrence 恢复并重放；
5. 可以检查 transition 是否复现。

因此它仍然是一个可观测 intervention。把它排除会人为缩小模型实际 action support，并可能高估一个 anchor 上有效动作的相对表现。

另一方面，保留 no-op 并不意味着把所有失败视为相同动作。`strict_identity` 为每个精确 no-op action 保持独立身份。

## 8. Anchor、posterior 与 Exact Batch-ERV

对每个格式合法 occurrence，BACE 使用：

```text
(task, anchor, action_identity)
```

建立局部统计。

WebShop 当前仍使用 exact observation anchor，而不是 Search-HotpotQA 的 `SequenceMatcher` similarity anchor。一个 anchor 必须具有足够的自然 occurrence 和至少两个不同动作，才会成为有效的主动分支位置。

在 `strict_identity` 下，候选可同时包括：

```text
valid::click[...]
valid::search[...]
invalid::click[...]
```

Exact Batch-ERV 不因 action 是环境有效操作还是 no-op 而更换目标函数；它根据各自 posterior 和全局预算选择需要补充的证据。

格式错误 occurrence 不参与这一过程。

## 9. Branch origin 与 replay

当 BACE 选择 `(z, u)` 后，只从自然 rollout 中真实执行过同一 `u` 的 concrete occurrences 中选择 origin：

```text
origins_by_action[u]
```

不会把一个 action 从 trajectory A 移植到只拥有相似状态的 trajectory B。这保证 copied response、token IDs、old log-prob 和环境动作来自同一个 occurrence。

WebShop replay 保存并恢复：

- 原始 session ID；
- 分支点之前的 projected action prefix；
- 对应的 pre-action observations；
- task description；
- 分支点 observation；
- 执行前 action set。

对 environment-valid action，replay 前要求该动作在恢复后的状态仍满足真实 dispatch 条件。

对已知 environment no-op：

- 不要求它突然变成 executable；
- 仍要求恢复到相同 anchor/action set；
- 执行后严格比较 action identity、post-action observation、immediate reward 和 done。

因此 no-op 只有在转移可复现时才成为有效 branch evidence。

## 10. 与 ALFWorld 的一致性和差异

总体规则一致：

| 分类 | ALFWorld | WebShop |
|---|---|---|
| 格式合法且环境有效 | `valid::command` | `valid::search/click` |
| 格式合法且环境无效 | 独立 `invalid::command`，可分支 | 独立 `invalid::action`，可分支 |
| 格式错误 | penalty，排除 BACE | penalty，排除 BACE |

差异只在环境有效性的权威来源：

- ALFWorld：执行前的 `admissible_commands`；
- WebShop：原始 `parse_action`、当前页面 `text_to_clickable` 和真实 dispatch 条件。

因此 WebShop 不应复用“字符串是否完整出现在 admissible action pool”这一简化规则来替代真实 parser。

## 11. 已完成验证

2026-09-06 的受控修复后完成：

- WebShop parser/dispatch/identity 专项：`13 passed`；
- invalid identity 与 AnchorIndex 专项：`5 passed`；
- 完整 `tests/bace_gigpo` 回归：`182 passed`；
- `git diff --check`：通过。

专项覆盖：

- projection 不修改 raw response；
- search template 与具体 query 的处理；
- click 的当前页面约束；
- upstream `re.match` trailing-text 行为；
- 无页面 search template 时的真实 search dispatch；
- `click[search]` no-op；
- parser 后 canonical valid identity；
- format-valid environment-invalid 的 strict identities；
- invalid/no-op replay transition 一致性；
- selected-worker session 隔离和恢复。

## 12. 当前证据边界

早期 WebShop 一步 gate 已完成 rollout、PPO 和 trace，但该批次因冷启动 competence 规划了 0 个 branch。它证明了 action 记录和训练主链可运行，**没有单独证明真实环境上的 no-op branch 已被 Exact Batch-ERV 选中并完成 replay**。

因此，在新的 WebShop 正式训练前，仍建议保留一项强制 branch 专项门禁，至少验证：

1. 一个 environment-valid search/click branch；
2. 一个格式合法 environment no-op branch；
3. 两者均通过 selected-worker replay；
4. no-op branch 的 identity、observation、reward 和 done 完整复现；
5. copied origin 不重复进入 PPO trainable support；
6. trace 中 `strict_identity`、Exact Batch-ERV 和 packed/selected-worker 标志正确。

这不是修改方法口径，而是补齐目前真实环境证据的空白。

## 13. 当前正式入口

目前 WebShop C8 两卡入口为：

```text
verl-agent-src/examples/bace_gigpo/c8_tree_credit_corr1_2gpu/run_webshop_c8_corr1_2gpu.sbatch
```

与本文直接相关的正式参数包括：

```text
algorithm.bace.variant=batch_erv_exact
algorithm.bace.branch_execution_mode=selected_worker
algorithm.bace.staged_root_batching=packed
algorithm.bace.invalid_action_mode=strict_identity
algorithm.bace.local_credit_mode=occurrence
actor_rollout_ref.actor.use_invalid_action_penalty=true
actor_rollout_ref.actor.invalid_action_penalty_coef=0.1
```

脚本在训练前运行 WebShop action 分类单测、真实环境 identity 验证和 selected-worker replay 验证，并记录源码 SHA-256 manifest。

## 14. 不采用的方案

### 14.1 不使用 `valid_only_branch`

它会排除所有格式合法 environment no-op，与“no-op 是模型真实、可执行和可复现行为”的方法定义冲突。

### 14.2 不使用 `single_invalid_bucket`

它会将所有 no-op 合并，造成严重 false merge，让不同行为共享 posterior。

### 14.3 不把 environment-invalid 直接加入 GiGPO 格式 penalty

这会改变原版 GiGPO reward 语义，使 WebShop BACE 与基线不再是受控比较。

### 14.4 不修改上游 WebShop parser

BACE 应服从 benchmark 的执行语义，而不是重定义 benchmark。

### 14.5 不按 next observation 合并 action

不同动作都可能产生相同 no-op observation，但它们仍是不同模型选择；按结果合并会把 action identity 和 outcome 混为一谈。

### 14.6 不做语义 action clustering

WebShop 已提供确定性的 parser-level action。embedding 或 LLM 聚类会引入额外模型、阈值和不可复现性，没有必要作为主实现。

## 15. 最终方法表述

WebShop 上的 BACE action 定义可以概括为：

> 模型回复首先按照原版 GiGPO 协议判断格式并抽取 action；随后完全按照 WebShop 原始 parser 和 dispatch 规则确定环境层动作身份。所有格式合法的具体动作，包括环境有效操作和确定性 no-op，均保留为独立 BACE 候选。格式错误 occurrence 仍参与 PPO 并接受原版局部格式惩罚，但不进入 BACE acquisition。branch 必须从自然执行过同一动作的 concrete occurrence 恢复，并通过严格转移一致性验证。

这一选择同时满足：

- 保留模型真实行为支持；
- 对齐 WebShop benchmark；
- 保持 GiGPO penalty 口径；
- 避免 invalid action false merge；
- 保证 posterior、ERV、branch execution 和 replay 使用同一个 action identity。
