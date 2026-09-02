# BACE C1/C2/C3：invalid-action 局部惩罚与树信用设计说明

## 目的

本说明定义 C1、C2、C3 在 BACE rollout tree 中如何使用 branch 证据，同时保持现有 GiGPO invalid-action penalty 的原本语义。

该惩罚的含义是：模型在某一个具体环境状态下输出了错误或不可执行动作，因此当前这一个动作应得到局部负反馈。它不是一段 trajectory 的全局成败标签，也不应因为后面某一步出错而反向加到前面已经执行过的动作上。

因此，树信用的目标是让 branch 提供“从分叉点开始，换一条 continuation 后结果变好了还是变差了”的信息；而不是把 branch 分叉点自身的动作惩罚作为树奖励传播给所有祖先。

## 相关对象

一条 root rollout 由按环境时间顺序排列的自然 edge 构成。自然 edge 是实际在 root trajectory 上执行并用于训练的动作。

当 BACE 在某个自然 edge 选择 branch 时，系统会从该 edge 对应的环境状态恢复前缀，并复制该 edge 的动作作为 branch 的首个动作。这个复制出来的 edge 称为 branch origin；root 上与它对应的原 edge 称为 natural origin。

branch origin 与 natural origin 有三个关键共同点：它们处于同一环境状态、执行同一个动作、并且在 strict replay 下具有相同的动作有效性。因此，如果该动作是 invalid action，两边带有的是同一份局部惩罚。

branch origin 本身只保留为 branch outcome 的证据，不会作为第二个相同 policy edge 进入 PPO 训练。真正进入训练的是 root 的自然 edge 和 branch 的 fresh suffix edge。

## 为什么不能直接把 branch 的结果平均到祖先

branch 的最终结果中既包含“分叉后 continuation 的变化”，也包含“分叉点这个复制动作的局部得分”。如果把 branch origin 的完整结果直接平均给更早的 root edge，那么分叉点的 invalid-action penalty 会被错误地带给祖先。

这会改变 GiGPO 的行为语义。原本，某个动作 invalid，只应惩罚这个动作对应的 policy occurrence；错误实现会让同一条 root 上所有更早动作也因为未来某个动作 invalid 而受罚。这相当于把局部 action penalty 变成 trajectory reward shaping。

## 当前的基本原则：只回传相对变化

当前实现先比较 branch origin 与 natural origin。两者的差值表示 branch 在复制完相同分叉动作之后，后续 continuation 相比 root continuation 带来的净变化。

由于两个 origin 复制同一动作，它们共有的 invalid-action penalty 会在这个差值中自然抵消。随后，系统把这个“后续净变化”叠加到当前被训练 edge 原有的信用上。

结果是：

- 当前 target edge 原有的 invalid-action penalty 会保留一次；
- branch 分叉点的同一份 invalid-action penalty 不会传给更早祖先；
- 如果 branch 让后续完成结果更好，祖先能获得这部分改善的信用；
- 如果 branch 让后续结果更差，祖先会承担相应的负向变化；
- root 上未参与 branch 的 edge 不会因为树信用而改变其局部惩罚语义。

## C1、C2、C3 分别如何使用该原则

### C1：只改 local credit

C1 只对一个 edge 的直接 branch 使用 tree evidence。对没有直接 branch 的 edge，local credit 保持原样。对有直接 branch 的 edge，系统平均 root 原 continuation 与各个 branch continuation 的结果。

直接 branch 的特殊点是：当前 target edge 就是 natural origin。因此 branch 相对 natural origin 的变化加回 target 后，等价于使用 branch 自己的 continuation 结果。这保持了直观的“同一个决策点的多个 continuation 平均”含义，同时没有重复训练 copied origin。

C1 的 macro credit 保持每个训练 edge 自己在 frozen C0 snapshot 中的值，不做 ancestor macro backup。

### C2：在 C1 的 local credit 上加入 macro tree credit

C2 的 local credit 与 C1 完全相同。差别只在 macro credit。

对于一个 root edge，C2 使用该 edge 的原始 frozen macro，以及每个 descendant branch 相对于其 natural origin 的 macro 变化。这个变化加回当前 root edge 的 macro 后，再与原值平均。

因此，C2 可以把更晚 branch 的整体 outcome 证据传给早期 root edge，但不把 branch origin 那个具体复制动作的局部惩罚带回去。

### C3：同时使用 full-tree local 和 macro credit

C3 的 macro 处理与 C2 相同。local credit 则扩展到全部 descendant branch，而不只是直接 branch。

较早 edge 接收一个较晚 branch 的变化时，变化会按两个 edge 之间的环境时间距离折扣。越远的后续 branch，对早期 edge 的 local credit 影响越小。这里被折扣的是 branch 相对 natural origin 的净 continuation 变化，不是 natural origin 的局部 invalid-action penalty。

branch 的 fresh suffix edge 不会被错误地当作 root 祖先；它继续使用自己的独立 edge 信用。

## 具体例子

假设一条 root rollout 有三个自然动作：第 1 步是打开柜门，第 2 步是拿取物品，第 3 步是把物品放入容器。

系统在第 3 步的位置建立一个 branch。branch 会恢复到第 3 步前的相同状态，并首先复制“把物品放入容器”这个动作。假设这个动作因为容器未正确打开而被判定为 invalid。于是 root 第 3 步和 branch origin 都带有同样的一次 invalid-action penalty。

随后，branch 的后续动作可能找到另一种补救方法，最终完成任务；root 则可能失败。此时 branch 的结果优于 root 第 3 步之后的原 continuation。

对于第 3 步本身，C1 的含义很直接：它会比较 root 的这一 continuation 与 branch 的这一 continuation。第 3 步自己的 invalid penalty 仍然存在，因为第 3 步确实执行了 invalid 动作。

对于第 1 步的“打开柜门”，C3 可以使用这个较晚 branch 的信息。但它只接收“branch 最终比 root 从第 3 步开始的原 continuation 改善了多少”。它不会接收“第 3 步动作 invalid”这份局部惩罚，因为第 1 步并没有执行这个错误动作。

换言之，第 1 步可以因 branch 证明“早期状态仍有可行补救路径”而获得或失去相应 continuation 信用；第 3 步则仍然为自己的错误动作负责。两类信息被刻意分开。

如果 branch origin 和 natural origin 都是 valid action，当前设计仍然成立。此时没有需要抵消的局部惩罚，回传的就是单纯的 continuation 差异；该行为与无 invalid penalty 时的常规树 backup 一致。

## Frozen macro 的说明

macro 是在完整 C0 physical batch 上先计算并冻结的。这样 C1、C2、C3 不会因为移除了 copied origin 或改变了支持集而重算 mean、standard deviation 或零方差处理规则。

由于 production reward 中有 occurrence-level invalid penalty，同一 leaf 的不同 edge 在 frozen macro 中不一定相同。当前设计不强行把它们压缩为一个 leaf scalar；每个 target edge 保留自己的 frozen macro，而 branch 只贡献相对其 natural origin 的 macro 变化。这避免为了树信用而悄悄修改既有 GiGPO penalty 口径。

## 可审计性与验证

trace 会记录每个 trainable unique edge 的原始、direct 和 descendant local credit，以及 base 和 descendant macro credit；branch evidence 单独记录 branch origin、natural origin、parent root、分叉时间和终局结果。

运行时还会单独记录真实 copied origin 被移出 PPO 的数量，以及仅为 GPU world size 整除而产生的 synthetic padding 数量，防止两类非训练行被混淆。

单测包含一个较晚 branch 的例子：它验证较早 ancestor 只得到 branch 相对较晚 natural origin 的净变化，而不会得到较晚 origin 的局部 penalty。完整 BACE 测试集已覆盖该实现与既有 C0、Replay、Batch-ERV、trace 和 trainer 集成逻辑的兼容性。
