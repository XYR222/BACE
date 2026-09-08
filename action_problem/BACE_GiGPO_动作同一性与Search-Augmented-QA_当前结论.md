# BACE-GiGPO：动作同一性判定与 Search-Augmented QA 适配结论

> 版本：2026-09-05  
> 目的：固定 BACE-GiGPO 在不同 agent benchmark 中对“两个动作是否相同”的定义，并明确 ALFWorld、WebShop、Search-Augmented QA 等环境中的 action canonicalization 方案。  
> 核心原则：**优先使用 environment-grounded deterministic canonicalization；在动作等价判定中宁可保守拆分，也不要错误合并。**

---

# 0. 结论先行

BACE-GiGPO 的局部贝叶斯后验与 ERV 都建立在 anchor-action pair：

\[
(z,u)
\]

之上，因此“两个动作是否相同”不是一个次要工程细节，而是直接决定：

- 哪些自然 rollout evidence 会进入同一个 Beta posterior；
- 哪些 branch outcomes 会共同更新同一个 \(p_{g,z,u}\)；
- ERV 判断的是哪一个局部实验；
- action-aggregated local credit 是否具有正确语义。

当前最终推荐将 canonical action 定义为：

\[
\boxed{
u
=
\operatorname{ExecCanon}_z(a)
}
\]

其中：

- \(a\) 是模型生成的完整 response 或 action text；
- \(z\) 是当前环境状态/anchor；
- \(\operatorname{ExecCanon}_z\) 是**确定性的、环境落地的 action canonicalizer**；
- \(u\) 表示环境实际接收到的可执行动作，而不是自然语言表面形式。

两个动作被视为相同，当且仅当：

\[
\boxed{
a_i\sim_z a_j
\iff
\operatorname{ExecCanon}_z(a_i)
=
\operatorname{ExecCanon}_z(a_j)
\neq
\mathrm{INVALID}.
}
\]

该定义的核心语义是：

> **两个模型输出只有在当前状态下会被环境解释为同一个 executable intervention 时，才属于同一个 canonical action。**

---

# 1. 为什么不能直接比较模型输出文本

LLM agent 的一次 response 通常包含：

```text
<think>
...
</think>
<action>
open fridge 1
</action>
```

因此需要区分至少四层 action identity。

| 层级 | 判定标准 | 示例 | BACE 适用性 |
|---|---|---|---|
| Raw response identity | 整个 response 完全相同 | CoT 也必须相同 | 不适合 |
| Action-text identity | `<action>` body 相同 | 大小写不同可能被拆开 | 可用但偏弱 |
| Executable identity | 环境 parser 后动作相同 | `Open Fridge 1` 与 `open fridge 1` 可相同 | **主推荐** |
| Effect identity | 执行动作后环境结果相同 | 两个不同命令碰巧得到相同 next observation | 不作为主定义 |

BACE 关心的是局部干预：

\[
p_{g,z,u}
=
P(Y=1\mid z,u,\pi_{\mathrm{old}})
\]

因此 \(u\) 最合理的语义应是：

\[
\boxed{
\text{environment-executable action}
}
\]

而不是：

\[
\text{LLM surface text}.
\]

---

# 2. CoT 不属于 Action Identity

假设两个自然 rollout 在相同 anchor 上产生：

```text
Response A:
<think>I should first access the cold storage.</think>
<action>open fridge 1</action>
```

以及：

```text
Response B:
<think>The refrigerator is currently closed.</think>
<action>open fridge 1</action>
```

虽然：

\[
\text{CoT}_A\neq\text{CoT}_B,
\]

但若环境最终执行的 command 都是：

```text
open fridge 1
```

则：

\[
\boxed{
u_A=u_B.
}
\]

因此：

- CoT 可以不同；
- response token sequence 可以不同；
- action body 的表面格式可以不同；
- 只要经过同一 environment parser 后执行的是同一个动作，就应进入同一个 action class。

这与 BACE 当前“对自然 rollout 已经暴露的 executable action 进行重复信用实验”的定位一致。

---

# 3. 不采用“执行后状态相同”作为主动作等价关系

一种更激进的定义是：

\[
u_1\sim u_2
\iff
T(z,u_1)=T(z,u_2).
\]

例如两个不同命令碰巧产生完全相同的下一 observation。

主方法不采用这一规则。

原因是单次观察到：

\[
s'_1=s'_2
\]

并不能保证：

\[
u_1
\equiv
u_2.
\]

两个动作仍可能在以下方面不同：

- hidden environment state；
- step cost；
- history 中记录的 action text；
- future transition distribution；
- stochasticity；
- remaining horizon；
- tool-side metadata。

若要从“相同效果”推导“相同动作”，理论上需要接近：

\[
P(s',r\mid z,u_1)
=
P(s',r\mid z,u_2)
\]

的 action-bisimulation 条件，而不是一次 next-state equality。

因此主方法只合并：

\[
\boxed{
\text{环境本身解析为同一个 executable action 的文本变体。}
}
\]

---

# 4. Action Canonicalization 的总设计原则

## 4.1 Environment grounded

优先使用环境已经存在的：

- admissible command；
- structured action object；
- tool name；
- object ID；
- clickable ID；
- normalized arguments；
- parser output。

而不是另行调用：

- embedding；
- LLM judge；
- semantic similarity model。

---

## 4.2 Deterministic

要求：

\[
\operatorname{ExecCanon}_z(a)
\]

对相同输入始终给出相同结果。

不能让 action identity 本身依赖：

- 随机 embedding clustering；
- LLM semantic judge；
- sampling；
- 当前 batch 内其他样本顺序。

---

## 4.3 Conservative

当前推荐原则为：

\[
\boxed{
\textbf{precision over recall in action equivalence}.
}
\]

即：

> 宁可把两个实际上等价的动作暂时拆成两个 action classes，也不要把两个真正不同的动作错误合并。

原因是：

### False split

如果：

\[
u_i\neq u_j
\]

但它们其实可以视为同一个动作，则主要损失是：

- evidence 无法共享；
- posterior concentration 较慢；
- ERV 可能多做一些重复实验。

### False merge

如果：

\[
u_i=u_j
\]

但实际上是两个不同动作，则会：

- 把不同动作的成功/失败混入同一个 Beta posterior；
- 错误改变 posterior mean；
- 错误改变 Bayes regret；
- 错误改变 ERV；
- 错误决定 branch target；
- 污染 action-level local credit。

因此：

\[
\boxed{
\text{false merge 的理论风险显著高于 false split}.
}
\]

---

# 5. ALFWorld：主版本的动作同一性定义

ALFWorld 是 BACE 最适合的主 benchmark 之一，因为动作空间具有明确的环境可执行语义。

定义当前 anchor 的 admissible action set：

\[
\mathcal A_{\mathrm{adm}}(z).
\]

模型生成 response \(a\) 后，经过确定性 parser：

\[
P_z(a).
\]

定义：

\[
\boxed{
\operatorname{ExecCanon}_z(a)
=
\begin{cases}
u,
&
P_z(a)=u\in\mathcal A_{\mathrm{adm}}(z),
\\
\mathrm{INVALID},
&
\text{otherwise}.
\end{cases}
}
\]

具体规则：

1. 从 `<action>...</action>` 中提取 action body；
2. 做不会改变 executable semantics 的机械规范化；
3. 若与当前 admissible command 精确匹配，则直接采用；
4. 若环境已有确定性 parser 能唯一映射到某条 admissible command，则采用该 command；
5. 若无法唯一映射，则标记为 `INVALID`；
6. `INVALID` 不进入 targeted branch candidate set。

例如：

```text
open fridge 1
Open Fridge 1
open   fridge 1
```

若最终都唯一映射到：

```text
open fridge 1
```

则：

\[
u_1=u_2=u_3.
\]

---

# 6. ALFWorld 中的 Candidate Action

当前主方案只使用自然 root 中已经真实执行过的 canonical actions：

\[
\boxed{
\mathcal C_g(z)
=
\mathcal C^{\mathrm{obs}}_g(z).
}
\]

其中：

\[
\mathcal C^{\mathrm{obs}}_g(z)
=
\left\{
\operatorname{ExecCanon}_z(a_i)
:
i\in\mathcal I_g^{\mathrm{root}}(z),
\operatorname{ExecCanon}_z(a_i)\neq\mathrm{INVALID}
\right\}.
\]

有效 branch anchor 要求：

\[
\boxed{
|\mathcal C_g(z)|\ge2.
}
\]

即自然 roots 已经在同一 anchor 上暴露至少两个不同的有效 executable actions。

该设计意味着 BACE 的主版本不是：

> 主动尝试任意未见动作。

而是：

> 对自然 rollouts 已经暴露出的局部动作竞争进行主动证据补充。

---

# 7. WebShop：必须区分 Click 与 Search

WebShop 的动作不是单一类型，至少要区分：

\[
\operatorname{click}[\cdot]
\]

和：

\[
\operatorname{search}[\cdot].
\]

因此不能简单把整个 action string 当作一个无结构 token。

---

## 7.1 Click Action

推荐表示为：

\[
\boxed{
u
=
(\texttt{click},c)
}
\]

其中 \(c\) 为当前页面上的 canonical clickable identifier，例如：

- product ID；
- option ID；
- button ID；
- 当前环境实际使用的 clickable string。

若当前环境明确返回：

```text
click[buy now]
```

则 canonical action 应来自环境的 clickable representation，而不是模型 action body 的任意表面文本。

---

## 7.2 Search Action

Search 的 query 是开放文本，因此：

\[
u
=
(\texttt{search},q).
\]

主版本推荐：

\[
\boxed{
q
=
\operatorname{NormalizeQuery}(q_{\mathrm{raw}})
}
\]

其中 `NormalizeQuery` 只做机械规范化，例如：

1. Unicode normalization；
2. strip；
3. collapse repeated whitespace；
4. lowercase。

第一版不建议做：

- embedding merge；
- synonym replacement；
- stop-word deletion；
- LLM semantic equivalence；
- query paraphrase clustering。

例如：

```text
search[Waterproof   Hiking Shoes]
```

和：

```text
search[waterproof hiking shoes]
```

可以归为同一个 strict action。

但：

```text
search[waterproof hiking shoes]
```

与：

```text
search[waterproof trekking shoes]
```

即使语义相近，主版本仍保留为不同 action classes。

---

# 8. Search-Augmented QA 环境是什么

以 Search-R1 / verl-agent 当前的 Search 环境为例，基本交互结构为：

```text
Question
    ↓
<think>...</think>
<search>query_1</search>
    ↓
<information>
retrieved passages
</information>
    ↓
<think>...</think>
<search>query_2</search>
    ↓
...
    ↓
<answer>final answer</answer>
```

其 action space 本质是两类高层操作：

\[
\boxed{
\texttt{search(query)}
}
\]

以及：

\[
\boxed{
\texttt{answer(text)}.
}
\]

其中：

- `search` 会调用 retrieval tool；
- query 是自由文本 payload；
- 中间步骤通常没有 reward；
- `<answer>` 或达到最大 turn 后 episode 结束；
- terminal reward 根据最终答案与 ground truth 计算。

因此 Search-Augmented QA 与 ALFWorld 的核心差异是：

\[
\boxed{
\text{action operator 是离散的，但 action payload 是开放自然语言。}
}
\]

---

# 9. Search-Augmented QA 的 Anchor 本身也更弱

在 ALFWorld 中，同一 observation 可以通过 exact matching 形成 anchor。

但 Search-QA 的 retrieval observation 几乎很难自然完全重复。

因此 GiGPO / verl-agent 在 Search benchmark 中会使用 textual similarity grouping，而不是只依赖 exact observation equality。

这意味着 Search-QA 中的 anchor：

\[
z
\]

本身更接近：

\[
\boxed{
\text{approximately equivalent information state}
}
\]

而不是严格环境状态。

因此即使 action identity 设计得很好，Search-QA 的：

\[
p_{z,u}
\]

也比 ALFWorld 中具有更强的近似性。

这也是为什么 Search-QA 更适合作为扩展泛化 benchmark，而不是最主要的理论验证环境。

---

# 10. Search-QA 中最困难的是 Search Query Identity

假设同一个近似 anchor 上出现：

```text
<search>where was Albert Einstein born</search>
```

```text
<search>Albert Einstein birthplace</search>
```

```text
<search>Albert Einstein wife</search>
```

应该如何定义同一动作？

当前考虑三种方案。

---

# 11. Search-QA 方案一：Strict Normalized Query

定义：

\[
\boxed{
u^{\mathrm{strict}}
=
(
\texttt{search},
\operatorname{NormalizeQuery}(q)
)
}
\]

则：

```text
where was Albert Einstein born
```

与：

```text
Albert Einstein birthplace
```

通常仍属于两个不同 actions。

### 优点

- 最确定；
- 不引入额外 semantic model；
- 不会错误合并不同 query；
- 与 BACE 的 Bayesian experiment semantics 最一致。

### 缺点

- 同义 query 无法共享 evidence；
- action space 会更稀疏；
- 很难形成多次相同 query occurrence。

当前建议：

\[
\boxed{
\text{作为 Search-QA 主版本 action identity。}
}
\]

---

# 12. Search-QA 方案二：只按 Verb 聚合

例如 BiPACE 提供 `tag_verb` 风格的 action key：

```text
<search>...</search>
```

全部映射为：

\[
u=\texttt{search},
\]

而：

```text
<answer>...</answer>
```

映射为：

\[
u=\texttt{answer}.
\]

此时：

```text
search[Einstein birthplace]
```

和：

```text
search[Einstein wife]
```

被视为同一个动作。

这对某些 action-conditioned credit estimator 有意义，因为它回答：

> 当前状态下，“继续搜索”还是“直接回答”更好？

但对于 BACE 的 ERV 来说过于粗糙。

因为 BACE 想判断的是：

> 哪一个具体局部实验值得重复验证？

如果所有 query 都混入：

\[
p_{z,\mathrm{search}},
\]

则该 posterior 不再区分：

- 搜索正确实体；
- 搜索错误实体；
- 搜索重复信息；
- 搜索真正能改变答案的信息。

因此：

\[
\boxed{
\text{BACE 不采用 verb-only identity 作为 Search-QA 主 action canonicalization。}
}
\]

---

# 13. Search-QA 方案三：Retrieval-Effect Canonicalization

另一个可考虑的方案是：

\[
\boxed{
u^{\mathrm{effect}}
=
(
\texttt{search},
\operatorname{TopKDocIDs}(q)
)
}
\]

即不看 query 文本，而看检索器真正返回了哪些 document/passages。

如果：

```text
where was Einstein born
```

和：

```text
Einstein birthplace
```

返回相同 top-\(K\) 文档，则可以把它们视为相同 retrieval effect。

### 优点

- environment-effect grounded；
- 能合并大量语义同义 query；
- 与工具执行结果直接相关。

### 缺点

即使 retrieval result 完全一致，LLM 的未来 context 仍可能不同，因为历史中包含原始 query：

```text
<search>query text</search>
```

因此：

\[
h'_1\neq h'_2
\]

即使：

\[
\operatorname{TopKDocIDs}(q_1)
=
\operatorname{TopKDocIDs}(q_2).
\]

所以：

\[
Q(z,u_1)
\]

与：

\[
Q(z,u_2)
\]

仍不一定完全相同。

因此该方案适合：

\[
\boxed{
\text{diagnostic / relaxed canonicalization ablation}
}
\]

而不是最严格的主定义。

---

# 14. Search-QA 推荐采用“双层 Action Key”

为兼顾理论严谨性与环境分析，建议同时保存：

## Strict key

\[
\boxed{
u^{\mathrm{strict}}
=
(
\texttt{search},
\operatorname{NormalizeQuery}(q)
)
}
\]

用于：

- Beta posterior；
- Bayes regret；
- ERV；
- branch selection；
- 主训练统计。

---

## Effect key

\[
\boxed{
u^{\mathrm{effect}}
=
(
\texttt{search},
\operatorname{TopKDocIDs}(q)
)
}
\]

用于：

- diagnostic；
- action equivalence analysis；
- relaxed canonicalization ablation。

重点统计：

\[
P
\left(
u_i^{\mathrm{effect}}
=
u_j^{\mathrm{effect}}
\mid
u_i^{\mathrm{strict}}
\neq
u_j^{\mathrm{strict}}
\right).
\]

若该比例非常高，说明 strict query identity 可能过度拆分，可进一步测试 effect-level merging。

---

# 15. ECPO 对我们的启示

ECPO 的 action-consistent credit 思路与 BACE 的 action posterior 语义非常接近。

核心思想是：

\[
u=\operatorname{can}(a),
\]

然后在相同 anchor state 下按 canonical action 聚合 occurrences。

它强调：

\[
\boxed{
\text{action canonicalization 应尽可能 environment grounded。}
}
\]

对 BACE 来说，这支持以下设计：

- 不按完整 CoT 分动作；
- 不按 raw response identity 分动作；
- 尽量对齐环境实际执行的 action；
- action-level posterior 和 action-level credit 应共享同一 canonicalizer。

但当前不应把我们的实现表述为：

> exact reproduction of the ECPO canonicalizer.

更准确的说法是：

> BACE follows the same environment-grounded action-canonicalization principle, while implementing benchmark-specific deterministic executable-action parsers.

---

# 16. BiPACE 对我们的启示

当前公开 BiPACE 代码中存在多种 action-key 方案，包括：

- `first_n`；
- `action_tag`；
- `tag_verb`。

---

## 16.1 `first_n`

定义大致为：

\[
\kappa(a)
=
\operatorname{Hash}
(
\text{response 前 }N\text{ 个 non-pad tokens}
).
\]

对于 CoT-first response：

```text
<think>...</think>
<action>...</action>
```

该 key 很容易主要由 CoT 决定。

因此不适合 BACE。

---

## 16.2 `action_tag`

从：

```text
<action>...</action>
```

中抽取 body 并 hash。

这比 `first_n` 更合理，能够真正对动作 body 分组。

但如果 action grouping 与 environment projection 使用不同的 normalization，则理论上可能出现：

\[
\text{environment-executed action 相同}
\]

但：

\[
\text{action key 不同}.
\]

因此 BACE 不直接采用 raw `action_tag` hash，而是进一步要求：

\[
\boxed{
\text{action key 直接来自 environment-postprocessed executable action。}
}
\]

---

## 16.3 `tag_verb`

例如：

```text
<search>...</search>
```

全部映射到 `search`。

它对回答：

> 当前应该继续调用搜索工具还是回答？

这一类 coarse action credit 很有价值。

但对于 BACE 的具体 branch experiment 太粗。

因此：

\[
\boxed{
\text{不作为 Search-QA 主 action identity。}
}
\]

---

# 17. 不同 Benchmark 的最终 Canonical Action 建议

| Benchmark | 推荐 canonical action |
|---|---|
| ALFWorld | 环境实际执行的 admissible command |
| ScienceWorld | parsed action template + canonical object IDs |
| WebShop click | `("click", canonical_clickable_id)` |
| WebShop search | `("search", normalized_query)` |
| Search-Augmented QA | `("search", normalized_query)` |
| Search-QA answer | terminal action，一般不作为 branch candidate |
| API/tool environment | `(tool_name, normalized structured arguments)` |
| AppWorld | `(api/function, canonical arguments)` |
| Python/tool math | `(tool_name, normalized program/arguments)`，需要独立设计 |
| plain CoT math | 缺乏可靠 executable macro-action identity，不适合作为 BACE 第一批主 benchmark |

---

# 18. 对 Benchmark 选择的影响

根据 action identity 的可靠程度，可以将 benchmark 分成两档。

## 第一档：Executable action identity 天然清楚

包括：

\[
\boxed{
\text{ALFWorld}
}
\]

以及结构化程度较高的：

- ScienceWorld；
- WebShop click；
- structured API/tool tasks。

这些环境最适合验证 BACE 的核心假设：

\[
\text{same anchor}
+
\text{same executable action}
\rightarrow
\text{repeated local experiment}.
\]

---

## 第二档：Action 带开放自然语言 payload

包括：

\[
\boxed{
\text{WebShop search}
}
\]

和：

\[
\boxed{
\text{Search-Augmented QA}.
}
\]

这里：

\[
u
=
(
\text{operator},
\text{free-form payload}
)
\]

action canonicalization 本身就是一个额外近似问题。

因此论文中的 benchmark 定位建议：

1. **ALFWorld 作为主要机制验证环境**；
2. WebShop 作为重要第二环境；
3. ScienceWorld 作为结构化泛化环境；
4. Search-Augmented QA 可作为进一步泛化实验，而不是第一优先级主实验。

---

# 19. BACE 中的统一 Action-Identity 接口

代码层面建议为所有环境统一提供：

```python
def canonicalize_action(
    state,
    raw_response,
    env_info,
) -> CanonicalAction:
    ...
```

输出至少包含：

```python
CanonicalAction(
    valid: bool,
    action_type: str,
    payload: object,
    key: Hashable,
    executed_form: object,
)
```

建议保证：

```text
canonical_action.key
```

直接作为：

\[
(z,u)
\]

中的 \(u\)。

---

# 20. 统一执行流程

对每一个自然 decision occurrence：

```text
1. 得到当前 anchor z
2. Actor 生成 raw response
3. 环境 projection/parser 解析 response
4. 得到真正 executable action
5. canonicalize_action(...)
6. 若 INVALID：
       记录 invalid action
       不进入 BACE targeted action pool
7. 若 valid：
       得到 canonical key u
8. 环境执行 u
9. 将 (z, u, outcome) 写入局部统计
```

对 branch：

```text
1. ERV 选择 (z*, u*)
2. 从自然 occurrences 中找到真实执行过 u* 的 origin pool
3. 恢复一个具体 natural occurrence
4. 重用该 occurrence 的原始 CoT + action response
5. 环境再次执行对应 executable action u*
6. 由冻结 pi_old 生成新 continuation
7. 根据新 outcome 更新 Beta(z*, u*)
```

这样 action posterior、branch execution 和 local credit 使用的是同一个 action identity。

---

# 21. Action Identity 与 Branch Origin 的关系

当前 BACE 主方案采用：

\[
\mathcal I_g^{\mathrm{obs}}(z,u)
=
\left\{
i\in\mathcal I_g^{\mathrm{root}}(z)
:
u_i=u
\right\}.
\]

当 ERV 选择：

\[
(z^*,u^*)
\]

后，从：

\[
\mathcal I_g^{\mathrm{obs}}(z^*,u^*)
\]

中选择 replay origin。

这进一步要求：

\[
\boxed{
\text{posterior 中的 }u
=
\text{origin occurrence 实际执行过的 executable action}.
}
\]

因此不能让 action canonicalization 只用于统计，而 branch execution 又采用另一套 parser。

---

# 22. Action Identity 与 Action-Aggregated Local Credit 的关系

若使用可选的 action-aggregated local advantage：

\[
\mathcal I_g(z,u)
=
\{
i:
z_i=z,\,
u_i=u
\},
\]

则：

\[
A_{g,z,u}^{S,\mathrm{act}}
=
\frac1{|\mathcal I_g(z,u)|}
\sum_{i\in\mathcal I_g(z,u)}
A_i^{S,\mathrm{occ}}.
\]

因此 action canonicalizer 同时决定：

1. Bayesian acquisition 中谁共享 posterior；
2. optimization 中谁共享 local action credit。

这要求两部分严格复用同一个：

\[
\operatorname{ExecCanon}.
\]

不能出现：

```text
ERV 用 environment action ID
训练时用 raw <action> body
```

这样的不一致。

---

# 23. 当前不建议采用的 Action Identity 方案

主版本不采用：

### 23.1 整个 response hash

因为 CoT 会污染 action identity。

### 23.2 First-N response tokens

同样会主要区分 reasoning text。

### 23.3 LLM 判断“两个动作语义是否一样”

原因：

- 不确定；
- 非确定性；
- 额外成本；
- 难以重复；
- action identity 本身变成一个学习问题。

### 23.4 通用 sentence embedding clustering

可能错误合并具有不同 executable semantics 的动作。

### 23.5 只按 action verb

例如把所有 `search(query)` 都看作同一个动作，对 BACE ERV 过于粗糙。

### 23.6 只按一次 next-state equality

不能保证 action 的完整环境语义等价。

---

# 24. 推荐的 Action Canonicalization 消融

建议至少保留以下消融。

## 24.1 Strict executable identity — Main

\[
u
=
\operatorname{ExecCanon}(a).
\]

---

## 24.2 Raw action-tag identity

只对 `<action>` body 做机械 normalization 后 exact match。

用于验证 environment-grounded canonicalization 是否真正必要。

---

## 24.3 Relaxed semantic/effect identity

在 Search-QA 中可使用：

\[
u^{\mathrm{effect}}
=
(\texttt{search},\operatorname{TopKDocIDs}(q)).
\]

作为放宽 action equivalence 的实验。

---

## 24.4 Verb-only identity

在 Search-QA 中使用：

\[
u\in\{\texttt{search},\texttt{answer}\}.
\]

用于验证过粗聚合是否损害 ERV 的局部辨识能力。

---

# 25. 应记录的诊断指标

## 25.1 Canonicalization quality

```text
raw response count
valid action count
INVALID ratio
unique raw-action count
unique canonical-action count
raw-to-canonical merge ratio
```

---

## 25.2 Action-class structure

```text
number of canonical actions per anchor
same-action occurrence count
singleton-action ratio
action-count imbalance
```

---

## 25.3 Search-QA 特有指标

```text
strict-query unique count
effect-key unique count
strict-different/effect-same pair ratio
retrieval-result overlap
query paraphrase collision rate
```

---

## 25.4 ERV sensitivity

比较不同 action canonicalization 下：

```text
Bayes regret distribution
ERV distribution
selected-action distribution
branch concentration
posterior entropy
realized regret reduction
final task success
```

---

# 26. 论文中的推荐表述

## 中文

> 我们使用环境落地的确定性 action canonicalization 来定义局部动作身份。对于模型生成的 response，我们首先通过环境自身的 projection/parser 得到实际执行的结构化动作，再以该 executable action 作为 Bayesian anchor-action posterior、ERV acquisition 和 action-level credit aggregation 的统一动作键。该设计有意采用保守的等价关系：只有当两个输出在当前状态下被环境解释为同一个可执行干预时才进行合并。对于带开放自然语言参数的工具环境，例如 WebShop search 和 Search-Augmented QA，我们保留 action type 并对 payload 仅进行机械规范化；语义或 retrieval-effect 层面的合并只作为扩展消融。

## 英文

> We define action identity through an environment-grounded deterministic canonicalizer. Each model response is first projected by the environment parser into the executable action actually applied at the current state, and this canonical action is shared by the Bayesian anchor-action posterior, ERV acquisition, and action-level credit aggregation. We intentionally adopt a conservative equivalence relation: two responses are merged only when the environment interprets them as the same executable intervention. For tool environments with open-ended textual arguments, such as WebShop search and search-augmented QA, we retain the action operator and apply only mechanical normalization to the payload; semantic or retrieval-effect-based merging is treated as an ablation rather than part of the main method.

---

# 27. 最终固定结论

当前主方案建议固定以下原则：

\[
\boxed{
u
=
\operatorname{ExecCanon}_z(a)
}
\]

并要求：

\[
\boxed{
a_i\sim_z a_j
\iff
\operatorname{ExecCanon}_z(a_i)
=
\operatorname{ExecCanon}_z(a_j).
}
\]

其中：

1. **ALFWorld**：使用环境实际执行的 admissible command；
2. **ScienceWorld**：使用 parsed action template 与 canonical object IDs；
3. **WebShop click**：使用 click type + canonical clickable；
4. **WebShop search**：使用 search type + mechanically normalized query；
5. **Search-Augmented QA**：主版本使用 search type + normalized query；
6. Search-QA 的 retrieval-effect identity 仅作为 diagnostic / ablation；
7. 不使用 CoT、raw response、first-N tokens、LLM semantic judge 作为主动作键；
8. action posterior、branch origin selection、branch execution 和 action-level local credit 必须共享同一个 canonicalizer；
9. 主原则是：

\[
\boxed{
\textbf{precision over recall in action equivalence}.
}
\]

一句话概括：

> **BACE 不试图判断两个自然语言动作“听起来是否相似”，而是判断它们在当前环境状态下是否对应同一个实际可执行干预。对于结构化环境，这一等价关系可以严格定义；对于开放搜索 query，则采用保守的 operator-plus-normalized-payload 表示，并将更激进的语义/效果合并留给消融。**

---

# 28. 相关代码与论文线索

## GiGPO / verl-agent

- Repository: https://github.com/langfengQ/verl-agent
- GiGPO core: https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py
- Search runner: https://github.com/langfengQ/verl-agent/blob/master/examples/gigpo_trainer/run_search.sh
- Search projection: https://github.com/langfengQ/verl-agent/blob/master/agent_system/environments/env_package/search/projection.py
- Search environment: https://github.com/langfengQ/verl-agent/blob/master/agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/env.py

## BiPACE

- Repository: https://github.com/TianxiangZhao/BiPACE
- Action-conditioned credit implementation: https://github.com/TianxiangZhao/BiPACE/blob/main/gigpo/core_cacb.py
- WebShop projection: https://github.com/TianxiangZhao/BiPACE/blob/main/agent_system/environments/env_package/webshop/projection.py
- ALFWorld projection: https://github.com/TianxiangZhao/BiPACE/blob/main/agent_system/environments/env_package/alfworld/projection.py

## Search-R1

- Repository: https://github.com/PeterGriffinJin/Search-R1

## ECPO

- 论文中采用 environment-grounded canonical action 的 action-consistent credit 设计，可作为 BACE action canonicalization 原则的重要参考。
- 当前 BACE 不依赖其具体实现，而采用 benchmark-specific executable-action parser。
