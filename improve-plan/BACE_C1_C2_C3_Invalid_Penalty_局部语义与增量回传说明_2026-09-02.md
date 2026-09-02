# BACE C1/C2/C3：invalid penalty 的局部语义与增量回传说明

## 目的

本文说明 C1、C2、C3 的 tree credit 在存在 GiGPO invalid-action penalty 时如何处理分支结果。

结论很简单：invalid-action penalty 仍然是“当前动作做错”的局部惩罚。它不应因为一条 branch 在较晚位置被判 invalid，就被当作整条 trajectory 的 reward shaping 回传给更早的祖先动作。

这项设计不改变 invalid penalty 的产生规则，也不改变 rollout、replay、Exact Batch-ERV、PPO loss 或 C0 的计算。它只规定 C1/C2/C3 怎样把 branch 的信息合并到一个已经存在的 root edge 上。

## 涉及的三个位置

对于一条 branch，需要区分三个 edge：

- **target edge**：当前要训练、要接收 tree credit 的 root edge，记为 `e`。
- **natural origin**：root 上实际执行过、随后被选来分支的那个 edge，记为 `o`。
- **branch origin**：replay 恢复到 `o` 之前的 state 后，在 branch 中复制执行的那一个动作，记为 `b`。

严格 replay 使 `o` 与 `b` 处于相同 state，并执行相同 action。因此，若这个 action 是 invalid，`o` 与 `b` 都应有同一个 invalid penalty；若它是 valid，两者都没有这项惩罚。branch 后续轨迹的不同，只来自该动作之后继续生成的内容。

## 为什么不能直接把 branch 的最终结果交给祖先

设想 root 的第 1 步是 `e`，第 3 步是 `o`。从第 3 步分出 branch 后，branch origin 为 `b`。

第 3 步动作本身有一个 invalid penalty。由于 `o` 和 `b` 复制同一动作，两边都含有该 penalty。假设 branch 后续虽然比 root 更成功，但若直接把 branch 的完整 return 当作第 1 步 `e` 的候选 return，那么第 3 步的 penalty 也会被一起带给第 1 步。

这会造成错误的学习信号：第 1 步没有执行第 3 步那个错误 action，却被其惩罚降低；反过来，第 3 步发生的局部正确性也会被错误当成第 1 步自身的属性。

因此，branch 对更早 edge 提供的应当是“从分叉点以后，branch 相比 root 改变了多少”，而不是“branch 的完整轨迹数值”。

## 当前的做法：先取 branch 相对 origin 的增量

每个 edge 的 step return 已经按现有 GiGPO 规则计算，其中可能包含 invalid penalty。对 branch，先比较 `b` 与 `o`：

> branch 增量 = branch origin 的 return − natural origin 的 return

由于 `b` 和 `o` 执行同一个动作，它们共有的 invalid penalty 在这一步相减时抵消。剩下的是 branch continuation 相对 root continuation 的真实变化。

然后把这个增量叠加到 target edge `e` 自身的 return 上。这样：

- `e` 原有的 invalid penalty 会保留一次，因为 `e` 本来就是当前训练的动作；
- `o` 的 invalid penalty 不会泄漏到 `e`；
- branch 后续变好或变差的信息仍会正确传给 `e`。

对更早 ancestor，C3 还会按 branch 与 target 的时间距离进行 gamma 折扣：越晚发生的 branch，对越早 target 的影响越小。

## 一个数值例子

考虑一条 root：

- 第 1 步 edge `e` 是当前训练对象；它本身没有 invalid penalty，其当前 return 为 10。
- 第 3 步 edge `o` 被选择分支；这个动作是 invalid，因此其当前 return 为 4，其中包含局部 penalty。
- replay 后 branch origin `b` 复制第 3 步相同动作，因此也含有相同局部 penalty；branch 后续更好，使其 return 为 7。

branch 相对 origin 的增量是 7 减 4，等于 3。这里第 3 步动作的共同 invalid penalty 已经抵消。

如果第 1 步与第 3 步相隔两步，gamma 为 0.95，则 branch 给第 1 步的候选值为：第 1 步自身的 10，加上经过两步折扣后的增量 3。该候选值略高于 10，表示 branch 后续带来了改进；它不包含第 3 步的 invalid penalty。

最后，算法把 root 原始候选值 10 与这个 branch 候选值做等权平均。第 1 步得到的是“保留第 1 步自身性质，同时吸收 branch 后续改进”的信用，而不是替第 3 步承担错误。

如果当前 target 正好就是分叉点，即 `e=o`，则结果会自然退化为普通 direct branch 比较：root 的原始结果和 branch 的结果直接一起平均。因为此时 invalid penalty 本来就是当前 target 的局部项，保留它符合语义。

## C1、C2、C3 分别如何使用它

| 变体 | local / step credit | macro credit |
| --- | --- | --- |
| C1 | 只对直接从当前 edge 分出的 branch 做等权 continuation 合并。 | 保持当前 edge 的 frozen C0 macro，不做树回传。 |
| C2 | 与 C1 相同，只使用直接 branch 的 local 合并。 | 对所有 descendant branch 使用“target 自身 + branch 相对 origin 的 macro 增量”做等权合并。 |
| C3 | 对所有 descendant branch 使用“target 自身 + 经 gamma 折扣的 branch 相对 origin 增量”做等权合并。 | 与 C2 完全相同的 descendant macro 合并。 |

这里的 macro 也可能含有 occurrence-level invalid penalty，因此 macro 使用同一原则：先做 branch origin 与 natural origin 的差，再把差叠加到 target 自身的 frozen macro。这样 C2/C3 不会把较晚动作的局部 penalty 错投到较早 edge。

## 什么没有被改变

- C0 继续使用原有 GiGPO macro 与 local credit。
- invalid-action 的判定、数值和施加位置保持不变。
- `o` 与 `b` 仍保留在 rollout/replay trace 中作为分支证据。
- copied branch-origin 不进入 PPO 的可训练 edge 支持；每个真实 policy edge 仍只训练一次。
- fresh branch suffix 是新的 edge，保留自己的局部 reward 与 macro，不被强行视为 root edge。

## 可审计性

每个 C1/C2/C3 batch 会记录原始、direct 和 descendant 的 return，以及 base 和 descendant macro。并额外记录该轮使用的是 edge-local invalid penalty 语义。专项测试覆盖“branch 在较晚位置分出，而更早 ancestor 接收 tree credit”的场景，验证较晚 origin 的局部 penalty 不会出现在更早 target 的 credit 中。
