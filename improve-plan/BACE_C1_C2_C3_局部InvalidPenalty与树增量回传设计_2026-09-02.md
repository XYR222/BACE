# BACE C1/C2/C3：局部 Invalid Penalty 与树增量回传设计

日期：2026-09-02  
适用代码：`work-BACE/verl-agent-src/recipe/bace_gigpo/advantage.py`  
适用模式：`o1_local`（C1）、`o1_tree_macro`（C2）、`o1_full_tree`（C3）

## 1. 目标

保持现有 GiGPO invalid-action penalty 的原始语义：它惩罚的是**当前 occurrence 执行的错误 action**，而不是一个应沿 rollout tree 传播给祖先的 trajectory-level reward shaping 项。

Tree credit 可以让一个 edge 利用其 branch 的后续结果，但不能让某个较晚分叉点的 invalid penalty 被误算为较早祖先的惩罚。因此，本设计回传的是 branch 相对其 natural origin 的 continuation 变化，而不是未经校正的 branch 总回报。

## 2. 术语与前提

对一个 root 上的 trainable edge `e`、某条 branch 的 natural origin `o`、以及该 branch 的 copied-origin occurrence `b`：

- `e`：当前计算 advantage、实际参与 PPO 的 edge；
- `o`：root 上被选为分叉点的自然 occurrence；
- `b`：branch replay 后的 copied-origin。它与 `o` 在相同 state 执行相同 action；
- `G(x)`：现有 reward pipeline 已计算好的 step return，包含该 occurrence 自己的 invalid-action penalty；
- `M(x)`：在完整 C0 physical batch 上冻结的 stable-occurrence macro score，同样保留现有 penalty 口径；
- `t_e`、`t_b`：`e` 和 `b/o` 的环境 step；
- `gamma`：现有 C3 discount。

严格 replay/identity 是这里的必要条件：`b` 和 `o` 必须确实复制同一 action。若 lineage 不完整、origin 缺失、parent root 不一致或 step 不一致，tree-credit index 会 fail closed，而不会推测对应关系。

## 3. 核心规则：回传增量而非 branch 总值

branch 与 natural origin 共享同一个 action 的 invalid penalty。令该共享局部项为 `p(o)`，可直观写作：

```text
G(o) = continuation(root from o) + p(o)
G(b) = continuation(branch from o) + p(o)
```

则：

```text
Delta_G(b, o) = G(b) - G(o)
                = continuation(branch) - continuation(root)
```

`p(o)` 在差值中消失。对 target `e`，构造的 branch candidate 是 `G(e)` 加上这个 continuation delta；所以 `e` 自己的 penalty 若存在，恰好保留一次。

这不是在平均后再人为减一个 penalty，也不是修改 GiGPO 生成 penalty 的位置；它是在现有 post-penalty 值上作代数差分。

## 4. 三种 tree-credit 模式

| 模式 | local component | macro component | PPO support |
|---|---|---|---|
| C1 `o1_local` | direct branch continuation 的均值 | target 自己的 frozen macro | natural root edges + branch suffix；copied origin 仅 evidence |
| C2 `o1_tree_macro` | 与 C1 完全相同 | 所有 descendant branch 的 macro 增量均值 | 同 C1 |
| C3 `o1_full_tree` | 所有 descendant branch 的 discounted continuation 增量均值 | 与 C2 完全相同 | 同 C1 |

### 4.1 C1：direct local credit

令 `D(e)` 是直接从 `e` 分出的 branches。C1 local candidate 为：

```text
G_direct(e) = mean(
    G(e),
    {G(e) + G(b) - G(o_b) | b in D(e)}
)
```

对于 direct branch，`o_b = e`，所以 branch candidate 化简为 `G(b)`。因此 C1 保持原本“当前 edge 与其直接 continuation 等权”的语义，同时明确不会重复或扩散局部 penalty。

C1 macro 不改变：

```text
M_C1(e) = M(e)
```

### 4.2 C2：descendant macro credit

令 `Desc(e)` 是 root 中从 `e` 或其后续 edge 发出的所有 descendant branches。C2 宏观候选为：

```text
M_C2(e) = mean(
    M(e),
    {M(e) + M(b) - M(o_b) | b in Desc(e)}
)
```

这避免把一个较晚 origin 的 `M(b)` 直接赋给较早 `e`；后者会把 `o_b` 的 occurrence-local 项带回祖先。

C2 local 与 C1 相同，随后仍使用既有的 GiGPO local normalization。

### 4.3 C3：full-tree local credit

C3 的 macro 与 C2 完全相同。其 local return 对较早 ancestor 使用 discount：

```text
G_C3(e) = mean(
    G(e),
    {G(e) + gamma^(t_b - t_e) * [G(b) - G(o_b)] | b in Desc(e)}
)
```

当 `e=o_b` 时，指数为零，公式退化为 direct branch 的 `G(b)`。当 `e` 早于 origin 时，branch 只能贡献相对 continuation 变化；origin 的 invalid penalty 仍然被抵消。

fresh `branch_suffix` 不是 root 祖先的重复 occurrence，因此保留自身原始 `G` 与自身 frozen macro，不做 branch-of-branch 推断。

## 5. 一个具体例子

root 在 step 1 有 edge `e`，在 step 3 的 edge `o` 分出 branch `b`。假设：

```text
G(e) = 5
G(o) = 4          # 包含 step 3 action 的局部 invalid penalty
G(b) = 200        # 同一 step 3 action 的 penalty 也已包含
gamma = 0.5
```

对 `e` 的 branch candidate 为：

```text
5 + 0.5^(3-1) * (200 - 4)
```

而不是直接使用 `200`，也不是把 `o` 的 penalty 再次加到 `e`。因此 target `e` 的 reward/penalty 口径由 `G(e)` 保持，branch 只改变其后的 continuation 判断。

## 6. Physical batch、去重与计量

1. `M` 先在完整 physical batch 上按原 C0 `episode_norm_reward` 计算，确保 C1/C2/C3 不另改 GiGPO normalizer。
2. copied-origin rows 保留为 tree evidence，但在 PPO 前从训练 support 删除；同一 action 不会作为两条 policy occurrence 参与 loss。
3. 为 GPU world-size 整除而生成的 `_adjust_batch_padding` 也不进入 unique support，但它与真实 copied origin 分开计量：
   - `tree_credit_copied_origins_removed`：真实 branch copied-origin 数；
   - `tree_credit_adjustment_padding_removed`：训练前的 synthetic padding 数；
   - `tree_credit_ppo_padding_rows`：去重后仅为数据并行整除而添加的零 loss-mask 行数。
4. C0 `tree_credit_mode=current`、rollout、Replay、Exact Batch-ERV、PPO objective 和 invalid penalty 的生成逻辑均未改动。

## 7. 失败保护与可审计性

实现对 lineage 做严格检查：未知 source type、branch 引用缺失 natural origin、parent root 不一致、origin step 不一致、重复 natural occurrence 或空训练 support 都会直接报错。

每个 trainable edge trace 中保留：原始/direct/descendant `G`、base/descendant macro、C1/C2/C3 component、direct/descendant leaf 集合及模式字段。branch evidence 另写入 `tree_credit_branches.jsonl`。诊断项 `tree_credit_invalid_penalty_is_edge_local=1` 明确标识此版本采用本设计。

## 8. 验证

专项单测 `tests/bace_gigpo/test_tree_credit.py` 新增“较晚 step 的 branch 不将 origin 的局部 penalty 回传给较早 ancestor”案例，同时验证 local 和 macro 两条路径。

本次验证结果：

- tree-credit 专项：12 passed；
- trainer/artifact 集成：19 passed；
- 完整 `tests/bace_gigpo`：174 passed。

唯一的测试期 warning 是 Ray/multiprocess 的资源追踪器退出 warning；pytest 返回成功，不影响上述断言或训练逻辑。
