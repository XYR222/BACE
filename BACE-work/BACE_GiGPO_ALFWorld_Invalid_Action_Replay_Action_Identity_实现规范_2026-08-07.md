# BACE-GiGPO：ALFWorld Invalid Action、Replay 与 Action Identity 实现规范

> 版本：2026-08-07  
> 适用范围：BACE-GiGPO 在 ALFWorld/TextWorld 中的 natural rollout、anchor/action grouping、ERV branch selection、replay，以及可选的 action-aggregated local credit。  
> 本文只固定算法语义与必要实现原则，不展开具体代码结构。

---

# 0. 最终结论

针对模型在 ALFWorld 中输出当前环境不可执行动作的情况，主版本固定以下规则：

1. **Natural rollout 中原样保留 invalid action。** 不删除、不重采样、不自动改写为 admissible action。
2. **当前不可执行不等于不可 replay。** 如果环境会消费该动作、返回失败/no-op feedback 并继续 episode，则它仍是一条真实 interaction step，也可以被恢复并再次执行。
3. **BACE branch 的对象是自然出现过的 CoT--action decision edge。** 不是 canonicalizer 人为构造出来的环境动作。
4. **Valid action 使用严格的环境动作身份；invalid action 保留其 raw textual identity。**
5. **主版本不使用语义 action canonicalization。** 不根据同义词、embedding、LLM 判断或 nearest-action matching，把 invalid 文本映射成某个 valid action。
6. **Branch replay 必须复制 concrete origin 的原始 CoT + raw action response。**
7. **Replay 的核心检查是 transition 可复现，而不是统一要求 branch action 属于 admissible set。**
8. 若启用 action-aggregated local credit，只允许在严格相同的 action identity 内聚合。

概括为：

$$
\boxed{
\text{模型实际输出什么，就让环境执行什么；BACE 只对真实发生过的 decision edge 做 continuation resampling。}
}
$$

---

# 1. 问题定义

在 ALFWorld 中，模型可能输出一个格式正常、但当前状态下不属于环境可执行动作集合的 action。

例如当前环境给出：

```text
open fridge 1
go to countertop 1
inventory
```

模型却输出：

```text
<think>...</think>
<action>take apple 1 from fridge 1</action>
```

环境可能返回：

```text
Nothing happens.
```

随后 episode 仍未结束，模型继续下一步决策。

因此需要区分：

- **format-valid**：能够稳定从模型 response 中解析出 action 文本；
- **environment-valid**：该 action 在当前状态下确实属于环境承认的可执行动作。

二者不是同一个概念。

本文主要讨论：

> **format-valid，但 environment-invalid 的动作。**

如果 response 连稳定的 action body 都无法解析，则仍可按原 GiGPO/verl-agent 的 natural rollout 规则处理，但不进入 BACE 的 action-level posterior 和 ERV candidate set。

---

# 2. Invalid Action 为什么必须保留

设自然 occurrence $i$ 中模型实际生成并送入环境的动作是 $a_i^{\mathrm{raw}}$。

只要原 rollout pipeline 消费了这一步，就应把它视为策略真实产生的数据：

$$
(h_i,s_i)
\xrightarrow{a_i^{\mathrm{raw}}}
(h_i^{+},s_i^{+}).
$$

即使该动作没有产生预期的环境状态转移，这一步仍可能：

- 消耗 interaction step；
- 产生失败/no-op feedback；
- 改变下一步 LLM history；
- 减少 remaining horizon；
- 影响最终任务成功率。

因此主版本不做：

- 删除 invalid step；
- rejection sampling 直到得到 valid action；
- 自动替换成最接近的 admissible action；
- 在训练统计中假装这一步没有发生。

这保证 natural rollout 仍然代表 actor 的真实行为分布。

---

# 3. Invalid Action 为什么可以 Replay

“当前不能作为有效环境动作执行”和“这个 interaction step 无法复现”是两件不同的事。

设自然 occurrence 为：

$$
e_i=(h_i,c_i,a_i^{\mathrm{raw}}),
$$

自然 continuation 为：

$$
e_i\oplus\xi_i^{\mathrm{root}}\rightarrow Y_i^{\mathrm{root}}.
$$

如果恢复到同一个 pre-action occurrence 后，再次提交完全相同的 $a_i^{\mathrm{raw}}$，环境能够复现同样的 invalid/no-op transition，则可以重新采样新的 suffix：

$$
e_i\oplus\xi_i^{\mathrm{branch}}\rightarrow Y_i^{\mathrm{branch}},
$$

其中：

$$
\xi_i^{\mathrm{branch}}\sim\pi_{\mathrm{old}}.
$$

因此：

$$
\boxed{
\text{environment-invalid}\neq\text{unreplayable}.
}
$$

对于 invalid action，branch 实际研究的是：

> 模型在这个状态犯下同一个错误之后，不同 continuation 下最终能否恢复成功。

这仍然是一个真实的局部信用问题。

---

# 4. Branch 的准确方法语义

允许 invalid action 后，不宜再把所有 branch 都描述为对抽象环境动作的严格 $\operatorname{do}(u)$ intervention。

当前方法更准确的定位是：

$$
\boxed{
\text{ERV-guided resampling of naturally observed CoT--action decision edges.}
}
$$

也就是说，BACE 首先从 natural roots 中识别具有信息价值的 observed action class，然后选择其中一个 concrete natural occurrence，复制其原始 response，并重新采样后续 continuation。

因此 branch 的基础对象始终来源于真实 natural rollout，而不是人为生成的新动作。

---

# 5. Action Identity 的最终规则

## 5.1 Valid Action

记 occurrence $i$ 执行动作前环境给出的 admissible set 为：

$$
\mathcal A_i^{\mathrm{adm}}.
$$

如果模型真正送入环境的 action 能被环境明确认定为某个当前有效 command，则统计身份使用该环境动作身份：

$$
\boxed{
u_i=\operatorname{EnvIdentity}(a_i^{\mathrm{raw}}).}
$$

核心原则是：

$$
\boxed{
\text{valid action identity 由环境证明，而不是由文本语义猜测。}
}
$$

---

## 5.2 Invalid / No-op Action

如果 action 文本可以稳定解析，但当前无法对应到有效环境 command，则保留 raw textual identity：

$$
\boxed{
u_i=\operatorname{RawInvalidIdentity}(a_i^{\mathrm{raw}}).}
$$

主版本最多允许无语义的机械处理，例如去除 action body 首尾空白、保证字符串序列化稳定。

不进行：

- synonym mapping；
- entity guessing；
- LLM/embedding semantic clustering；
- nearest admissible action matching；
- 把所有 invalid strings 无条件合并成一个 `INVALID` 类。

因此，不同 invalid strings 默认视为不同 action identities。

---

# 6. 为什么不采用语义 Canonicalization

语义 canonicalization 最大的问题不是“文本被处理过”，而是它可能让统计动作身份与真实执行动作身份不一致。

假设模型实际输出 $a_{\mathrm{raw}}$，环境没有执行有效动作：

$$
T(s,a_{\mathrm{raw}})=\text{no-op}.
$$

如果 canonicalizer 根据文本语义把它映射成 admissible action $u$，而真实环境中：

$$
T(s,u)\neq T(s,a_{\mathrm{raw}}),
$$

那么把二者聚合为同一个 action class 就会错误地认为：

$$
a_{\mathrm{raw}}\equiv u.
$$

这会直接污染：

$$
p_{z,u},
\quad
\Delta\operatorname{ERV}(z,u),
\quad
U(z),
$$

以及可选的：

$$
A^S(z,u).
$$

对 BACE 而言，这类误差尤其严重，因为 action identity 不只影响 credit aggregation，还影响后续 branch allocation 与 origin selection。

因此主版本采用：

$$
\boxed{
\text{宁可保守拆分，也不通过语义规则猜测两个动作在环境中等价。}
}
$$

这也是我们与更积极的文本 canonicalization 方法保持距离的主要原因。

这里不需要把其他方法直接称为“作弊”。更准确的表述是：我们希望避免潜在的 **benchmark-interface leakage** 与 **action-credit contamination**，并保证 posterior、ERV 与 replay 始终对应真实发生过的环境交互。

---

# 7. Candidate Action Set 与 Effective Anchor

对 repeated anchor $z$，自然 root 中的 occurrence 集合为：

$$
\mathcal I_g^{\mathrm{root}}(z).
$$

主版本候选动作只来自 natural rollout 中真实出现、且能够稳定解析 action body 的 action identities：

$$
\boxed{
\mathcal C_g(z)
=
\left\{
u_i:
 i\in\mathcal I_g^{\mathrm{root}}(z),
\ a_i^{\mathrm{raw}}\text{ 可解析}
\right\}.
}
$$

因此 $\mathcal C_g(z)$ 可以同时包含：

- valid observed actions；
- environment-invalid/no-op observed actions。

不包含：

- 从未在 natural roots 中出现的 admissible actions；
- actor-top 但没有自然执行过的动作；
- 无法稳定解析 action identity 的 response。

当前主方案保留全部 observed action identities，不使用额外 action-count truncation。

对应地，effective anchor 的动作竞争条件改为：

$$
\boxed{
|\mathcal C_g(z)|\ge2.
}
$$

因此下面这种竞争是允许的：

$$
\text{valid action}
\quad\text{vs.}\quad
\text{invalid/no-op action}.
$$

这不意味着把 invalid action 当成可执行环境动作，而是把它视为模型在该状态真实可能采取的另一种行为。

其余 anchor 条件继续沿用当前 BACE-GiGPO 方案，例如 exact GiGPO anchor key、重复 occurrence、排除初始状态、非 terminal、natural-root backbone only、replay 可恢复以及 ERV utility 条件。

---

# 8. Local Posterior 与 ERV 的处理

允许 invalid action 后，局部成功概率不再统一解释为：

$$
P(Y=1\mid z,\operatorname{do}(u),\pi_{\mathrm{old}}).
$$

更准确的工作定义是：

$$
\boxed{
p_{g,z,u}
=
\mathbb E_{i\sim\nu_g(\cdot\mid z,u)}
\left[
P\left(
Y=1
\mid
h_i,c_i,a_i^{\mathrm{raw}},\pi_{\mathrm{old}}
\right)
\right].
}
$$

其中 $u$ 是严格 action identity，而 $\nu_g(\cdot\mid z,u)$ 表示属于该 identity 的 natural occurrences 的经验分布。

于是：

- 对 valid action，$p_{g,z,u}$ 表示执行该真实环境动作后的 continuation success probability；
- 对 invalid action，$p_{g,z,u}$ 表示产生并执行该具体 invalid response 后的平均 recoverability。

Beta posterior 仍保持：

$$
p_{g,z,u}\mid\mathcal D
\sim
\operatorname{Beta}(\alpha_{g,z,u},\beta_{g,z,u}).
$$

Bayes simple regret 仍为：

$$
\mathcal R_g(z)
=
\mathbb E\left[
\max_{u\in\mathcal C_g(z)}p_{g,z,u}
\mid\mathcal D
\right]
-
\max_{u\in\mathcal C_g(z)}
\mathbb E[p_{g,z,u}\mid\mathcal D].
$$

one-sample ERV 仍为：

$$
\Delta\operatorname{ERV}_g(z,u)
=
\mathcal R_g(z)
-
\mathbb E_Y\left[
\mathcal R_g^{+}(z;u,Y)
\right].
$$

因此数学机制本身不需要因为 invalid action 重写；真正变化的是 action identity 和 $p_{g,z,u}$ 的解释。

---

# 9. Branch Selection 与 Replay

对 selected anchor $z$，动作仍按当前 ERV 规则分配。例如主方案采用：

$$
\mu_b(u\mid z)
=
\frac{
\exp\left(\Delta\operatorname{ERV}_{g,b}(z,u)/\tau_\mu\right)
}{
\sum_{v\in\mathcal C_g(z)}
\exp\left(\Delta\operatorname{ERV}_{g,b}(z,v)/\tau_\mu\right)
}.
$$

若 anchor 使用 expected ERV utility：

$$
U_{g,b}(z)
=
\sum_{u\in\mathcal C_g(z)}
\mu_b(u\mid z)
\Delta\operatorname{ERV}_{g,b}(z,u).
$$

选中 $z$ 和 action identity $u$ 后，只能从真实执行过该 identity 的 natural occurrences 中选择 concrete origin：

$$
\boxed{
\mathcal I_g(z,u)
=
\left\{
 i\in\mathcal I_g^{\mathrm{root}}(z):u_i=u
\right\}.
}
$$

再采样：

$$
i_b\sim\operatorname{Uniform}(\mathcal I_g(z,u)).
$$

对 $i_b$ 的 branch replay 采用以下原则：

1. 恢复相同 task/game 与该 occurrence 前的自然 prefix；
2. 确认 pre-action anchor 与原 occurrence 一致；
3. 复制原 occurrence 的完整 CoT + raw action response；
4. 将原 raw action 再次送入环境；
5. 复现原来的 valid transition 或 invalid/no-op transition；
6. 从新的 post-action observation 开始，用冻结的 $\pi_{\mathrm{old}}$ 重新采样 suffix；
7. 用新 terminal outcome 更新对应 $(z,u)$ posterior。

必须区分：

$$
\boxed{
\text{statistical action identity}
\neq
\text{environment replay string}.
}
$$

环境 replay 始终使用 concrete origin 的原始 raw response。

---

# 10. Replay 一致性要求

旧规则中统一要求：

$$
u\in\mathcal A^{\mathrm{adm}}
$$

不再适用于所有 branch，因为 invalid/no-op action 本来就允许不属于 admissible set。

新的核心要求是：

$$
\boxed{
\operatorname{ReplayTransitionMatch}(i)=1.
}
$$

即：

- pre-action anchor 恢复正确；
- 相同 raw action 被再次提交；
- post-action transition 与 natural occurrence 的语义一致；
- immediate reward、termination 和关键 feedback 没有异常不一致。

对于 valid action，可额外检查其恢复后仍保持 valid。

对于 invalid action，则允许它再次得到 invalid/no-op feedback，这本身不是 replay failure。

真正的 replay failure 是：

$$
\text{same prefix + same raw action}
\not\Rightarrow
\text{same transition semantics}.
$$

如果 selected anchor 之前的 natural prefix 已经包含 invalid/no-op actions，也必须按原始顺序 replay，不能只重放有效环境动作。

---

# 11. 与训练优势和 Action Aggregation 的接口

Natural root 中的 invalid/no-op occurrence 仍然是 actor 的真实输出，应按正常 GiGPO/BACE 训练规则保留。

如果选中的 branch origin 是 invalid action，copied origin 仍视为该 branch 的一个真实训练 occurrence，并根据新的 branch leaf outcome获得相应优势。

若启用可选的 action-aggregated local credit，则 group 定义为：

$$
\mathcal I_g(z,u)=\{i:z_i=z,\ u_i=u\}.
$$

因此：

- 相同 valid environment identity 可以聚合；
- 完全相同的 invalid raw identity 可以聚合；
- valid 与 invalid 永远不聚合；
- 两个不同 invalid strings 默认不聚合。

最终仍只聚合 local advantage，不聚合 leaf-level trajectory advantage。

这一规则的目的，是确保 action-level credit 不会因为文本 canonicalization 而混入实际执行语义不同的样本。

---

# 12. 风险、诊断与消融

保留 raw invalid identity 的主要风险是 action fragmentation：一个 anchor 可能出现许多只出现一次的 invalid strings，从而增加候选数量和 posterior uncertainty。

第一版不应立刻引入复杂 semantic canonicalizer，而应先记录：

### Invalid occurrence ratio

$$
r_{\mathrm{invalid}}
=
\frac{N_{\mathrm{environment\text{-}invalid}}}
{N_{\mathrm{parsed\ actions}}}.
$$

### Invalid fragmentation

$$
K_{\mathrm{invalid}}(z)
=
|\{\text{distinct invalid action identities at }z\}|.
$$

### Invalid branch ratio

$$
r_{\mathrm{branch-invalid}}
=
\frac{N_{\mathrm{branches\ selecting\ invalid}}}
{N_{\mathrm{all\ branches}}}.
$$

并比较 valid 与 invalid branches 的 ERV、最终成功率、realized regret reduction 与 replay consistency。

建议保留三个版本：

1. **Main：Strict Identity + Invalid Retained**  
   valid 使用环境 identity；invalid 使用 raw identity；invalid 可进入 ERV 和 branch。

2. **Valid-Only Branch**  
   natural rollout 仍保留 invalid，但 invalid 不进入 branch candidate set。

3. **Single INVALID Bucket**  
   所有 environment-invalid actions 仅在统计上合并为单一 $\texttt{INVALID}$ 类；实际 replay 仍使用 concrete origin 的原始 raw response。

第三种仅用于诊断 fragmentation，不作为主版本。

---

# 13. 对旧版 Anchor 规范的更新

本规范覆盖旧版中以下规则。

### 旧规则 A

> 无法映射到 admissible command 的动作不进入 targeted branch action set。

更新为：

$$
\boxed{
\text{格式可解析、自然出现的 environment-invalid action 可以进入 branch candidate set。}
}
$$

### 旧规则 B

> Effective anchor 必须至少出现两个不同的 valid observed actions。

更新为：

$$
\boxed{
|\mathcal C_g(z)|\ge2,
}
$$

其中 $\mathcal C_g(z)$ 由自然 observed 的严格 action identities 构成，可以同时包含 valid 和 invalid actions。

### 旧规则 C

> Replay 后 selected action 必须属于 admissible set。

更新为：

- valid action：可以要求恢复后仍为 valid；
- invalid action：允许恢复后继续 invalid/no-op；
- 两者统一要求 transition semantics 可复现。

### 旧规则 D

> 可以使用文本 canonicalization 将模型输出映射到环境动作。

更新为：

$$
\boxed{
\text{主版本不使用可能改变真实环境动作身份的语义修正。}
}
$$

---

# 14. 最终实现规则

$$
\boxed{
\begin{aligned}
&\textbf{Natural rollout:}
&&\text{invalid/no-op step 原样保留；}\\
&\textbf{Branch object:}
&&\text{natural observed CoT--action edge；}\\
&\textbf{Valid identity:}
&&u=\operatorname{EnvIdentity}(a);\\
&\textbf{Invalid identity:}
&&u=\operatorname{RawInvalidIdentity}(a^{\mathrm{raw}});\\
&\textbf{Semantic canonicalization:}
&&\text{主版本不做；}\\
&\textbf{Candidate set:}
&&\text{仅 natural observed、可解析的 action identities；}\\
&\textbf{Anchor competition:}
&&|\mathcal C_g(z)|\ge2;\\
&\textbf{Branch origin:}
&&i\sim\operatorname{Uniform}(\mathcal I_g(z,u));\\
&\textbf{Replay response:}
&&\text{复制 concrete origin 的原始 CoT + raw action；}\\
&\textbf{Replay validity:}
&&\text{检查 transition 可复现，而非统一要求 admissible；}\\
&\textbf{Fresh suffix:}
&&\xi\sim\pi_{\mathrm{old}};\\
&\textbf{Posterior:}
&&\text{用新 terminal outcome 更新对应 }(z,u);\\
&\textbf{Action aggregation:}
&&\text{只在严格相同 action identity 内进行。}
\end{aligned}
}
$$

最终原则是：

> **BACE-GiGPO 不替模型修正动作，也不把“当前不可执行”误解为“无法 replay”。模型自然产生、能够被环境消费并稳定复现的 invalid/no-op action，仍然是一条真实 decision edge，可以参与主动信用采集；但其统计身份必须忠实于真实环境交互，不能通过语义 canonicalization 被重新解释成另一个可执行动作。**
