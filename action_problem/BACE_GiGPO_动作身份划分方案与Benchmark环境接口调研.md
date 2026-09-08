# BACE-GiGPO 动作身份划分方案与 Benchmark 环境接口调研

> 版本：2026-09-05  
> 目的：固定 BACE-GiGPO 在不同 benchmark 上如何判断“两个动作是否属于同一个动作”，并总结各环境能够提供到什么粒度的底层 action identity。  
> 核心原则：**优先把 action identity 后移到环境实际执行/解析的层面，而不是停留在模型输出文本层。**

---

# 0. 结论先行

BACE-GiGPO 的局部 Bayesian posterior、ERV acquisition、branch replay 和可选 action-level local credit 都以：

\[
(z,u)
\]

为基本单位。

因此，动作身份 \(u\) 应尽可能表示：

\[
\boxed{
\text{环境真正执行的 executable intervention}
}
\]

而不是：

\[
\text{LLM 生成的 surface action text}.
\]

主方案统一定义：

\[
\boxed{
u
=
\operatorname{ExecCanon}_z(a)
}
\]

其中：

- \(a\)：模型原始 response；
- \(z\)：当前 anchor/state；
- \(\operatorname{ExecCanon}_z\)：benchmark-specific、deterministic、environment-grounded 的动作 canonicalizer；
- \(u\)：最终用于 posterior、ERV、branch、credit aggregation 的唯一 action key。

两个动作被认为相同，当且仅当：

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

实际实现推荐统一成：

\[
\boxed{
\text{raw response}
\rightarrow
\text{action-tag extraction}
\rightarrow
\text{environment projection/parser}
\rightarrow
\text{executed-action key}.
}
\]

这可以理解为：

> **BiPACE-style action extraction + environment-grounded executable-action identity.**

BiPACE 的 `<action>` / `<search>` 结构化抽取可以直接借鉴，但最终 action key 不停留在 raw action-tag body，而是继续“后移一层”，使用环境真正接受、解析或内部表示的动作。

---

# 1. 为什么 BACE 对 Action Identity 的要求更高

在普通 action-conditioned credit estimator 中，动作分类错误主要影响：

\[
\widehat Q(z,u),
\qquad
A(z,u).
\]

但在 BACE 中，action identity 还决定：

\[
\boxed{
p_{g,z,u}
}
\]

\[
\boxed{
\Delta ERV(z,u)
}
\]

\[
\boxed{
\text{branch target}
}
\]

\[
\boxed{
\text{branch origin pool}
}
\]

以及可选的：

\[
\boxed{
A_{g,z,u}^{S,\mathrm{act}}.
}
\]

因此一个错误 action merge 会直接污染：

\[
\text{posterior}
\rightarrow
\text{Bayes regret}
\rightarrow
\text{ERV}
\rightarrow
\text{branch allocation}.
\]

这要求动作分类尽量建立在真实环境执行语义上。

---

# 2. Action Identity 的四个层次

设模型产生：

```text
<think>
I should open the refrigerator first.
</think>
<action>
Open Fridge 1
</action>
```

至少存在四种动作身份定义。

| 层级 | 判定依据 | 示例 | BACE 推荐 |
|---|---|---|---|
| Raw response identity | 整个 response 完全相同 | CoT 也必须相同 | 否 |
| Action-tag identity | `<action>` body 完全相同 | `"Open Fridge 1"` | 可作为第一层抽取 |
| Executable identity | 环境 parser/projection 后执行动作相同 | `"open fridge 1"` | **主方案** |
| Internal symbolic identity | 环境内部结构化 action object / ID 相同 | `(template_id, obj_ids)` | **若可获得则优先** |

因此推荐优先级：

\[
\boxed{
\text{internal symbolic action}
>
\text{parser-level executable action}
>
\text{raw action-tag body}
>
\text{raw response}.
}
\]

---

# 3. CoT 不属于 Action Identity

假设：

```text
Response A:
<think>I should access the refrigerator.</think>
<action>open fridge 1</action>
```

与：

```text
Response B:
<think>The refrigerator is closed.</think>
<action>open fridge 1</action>
```

有：

\[
\text{CoT}_A\neq\text{CoT}_B,
\]

但若环境最终执行：

```text
open fridge 1
```

则：

\[
\boxed{
u_A=u_B.
}
\]

BACE 的 action posterior 要估计的是：

\[
p_{g,z,u}
=
P(Y=1\mid z,u,\pi_{\mathrm{old}}),
\]

所以 \(u\) 应表示当前实际干预，而不是产生该干预之前的完整自然语言推理文本。

---

# 4. 主方案不做语义 Action Clustering

主版本不使用：

- embedding；
- sentence similarity；
- LLM judge；
- synonym replacement；
- learned action ontology；
- semantic clustering。

即使两个动作“意思很像”，只要环境最终执行的 intervention 不完全相同，就保留成不同 action keys。

主原则：

\[
\boxed{
\textbf{precision over recall in action equivalence}.
}
\]

原因：

## False split

两个其实等价的动作被拆开，主要后果是：

- evidence 无法共享；
- posterior concentration 较慢；
- ERV 可能多补一些样本。

## False merge

两个不同动作被错误合并，会：

- 混合成功/失败 evidence；
- 污染 posterior mean；
- 改变 Bayes simple regret；
- 改变 ERV；
- 导致 branch target 错误；
- 污染 local action credit。

因此：

\[
\boxed{
\text{false merge 比 false split 更危险}.
}
\]

---

# 5. 为什么不直接用“执行后 observation 相同”

一种更激进的方案是：

\[
u_1\sim u_2
\iff
T(z,u_1)=T(z,u_2).
\]

主版本不采用。

一次观察到：

\[
s'_1=s'_2
\]

并不意味着两个动作在以下方面都相同：

- hidden state transition；
- step cost；
- history；
- stochastic transition distribution；
- tool metadata；
- remaining horizon。

真正需要的条件接近：

\[
P(s',r\mid z,u_1)
=
P(s',r\mid z,u_2),
\]

这比一次 next-observation equality 强得多。

所以主版本只认：

\[
\boxed{
\text{环境本身定义的 executable-action identity}.
}
\]

---

# 6. 与 BiPACE 的关系

BiPACE 的动作划分思路可以直接借鉴第一步：

\[
\text{response}
\rightarrow
\text{extract }<action>\text{ body}.
\]

但 BACE 推荐进一步做：

\[
\boxed{
\text{response}
\rightarrow
\text{action-tag extraction}
\rightarrow
\text{environment parser/projection}
\rightarrow
\text{canonical action key}.
}
\]

例如：

```text
<action>Open Fridge 1</action>
```

BiPACE-style action-tag key 可能是：

```text
Open Fridge 1
```

而 BACE 主方案继续后移到：

```text
open fridge 1
```

或者更底层：

```text
TextWorld Action(...)
```

若模型输出已经非常规范，则：

\[
u_{\mathrm{BiPACE}}
=
u_{\mathrm{ExecCanon}}
\]

在绝大多数样本上会完全一致。

因此这种“严格后移”基本是一个低成本改进：

- 正常样本行为与 BiPACE 相同；
- corner case 更严谨；
- posterior/ERV/branch 的 action 语义完全统一。

---

# 7. 统一代码接口

建议所有 benchmark adapter 提供：

```python
def get_executed_action_key(
    state,
    raw_response,
    env_info,
):
    ...
```

返回：

```python
CanonicalAction(
    valid: bool,
    action_type: str,
    payload: object,
    key: Hashable,
    executed_form: object,
)
```

其中：

```text
canonical_action.key
```

直接作为 BACE 中的：

\[
u.
\]

统一要求：

\[
\boxed{
u_{\mathrm{posterior}}
=
u_{\mathrm{ERV}}
=
u_{\mathrm{branch}}
=
u_{\mathrm{credit}}.
}
\]

---

# 8. Benchmark 调研总览

当前重点 benchmark 可以按“action identity 能下沉多深”分级。

| Benchmark | 环境是否有真实 parser/executor | 是否有更底层结构化 action identity | 推荐 BACE key | 等级 |
|---|---:|---:|---|---|
| ALFWorld-TextWorld | 是 | **是，TextWorld `Action` object** | serialized internal Action | **A+** |
| ScienceWorld | 是 | **是，`template_id + obj_ids`** | `(template_id, obj_ids)` | **A+** |
| WebShop | 是 | 部分结构化，但无统一 symbolic ID | `(action_type, actual_arg)` | **A** |
| Search-HotpotQA | 是 | structured tool call | `(SEARCH, q_exec)` | **A** |
| GSM8K / MATH / AIME | 否 | 否 | 无 benchmark-native action key | **C / 不适合直接做** |
| Tool-augmented Math | 取决于 wrapper | 通常可以 | structured tool call | **A/B** |

下面分别展开。

---

# 9. ALFWorld：可以拿到真正的 TextWorld 内部 Action

## 9.1 GiGPO 当前使用的环境

verl-agent 的 GiGPO ALFWorld 训练脚本使用：

```text
env.env_name=alfworld/AlfredTWEnv
```

也就是 **ALFWorld TextWorld 模式**。

当前 wrapper 会请求：

```python
textworld.EnvInfos(
    won=True,
    admissible_commands=True,
    ...
)
```

因此每一步至少可以拿到：

```text
admissible_commands
```

并且 verl-agent 的 `AlfworldEnvs` 会把：

```python
info['admissible_commands']
```

保存在当前 worker 状态中。

---

## 9.2 verl-agent 当前 Projection

当前 GiGPO 的：

```text
agent_system/environments/env_package/alfworld/projection.py
```

主要做：

```text
raw response
→ lowercase
→ extract <action>...</action>
→ strip
```

得到一个文本 action。

因此当前 GiGPO 的 action execution 已经不是完整 response，而是抽取后的 command string。

---

## 9.3 TextWorld 底层还可以继续后移

TextWorld 的 `StateTracking` wrapper 内部会检测本次真正发生的 action：

```python
self._last_action = self._inform7.detect_action(
    i7_event,
    valid_actions
)
```

随后：

```python
self._game_progression.update(self._last_action)
```

也就是说，真正更新游戏状态的是：

\[
\boxed{
\text{TextWorld internal Action object}
}
\]

而不是字符串本身。

TextWorld 还支持：

```python
request_infos.last_action
```

并在内部保存：

```python
state["_last_action"]
```

---

## 9.4 TextWorld Action 的内部结构

TextWorld `Action` 类包含：

- `name`
- `preconditions`
- `postconditions`
- `command_template`
- reverse action metadata

其 equality 基于：

\[
(\text{name},\text{preconditions},\text{postconditions}).
\]

因此从理论上最严格的 ALFWorld action key 可以定义为：

\[
\boxed{
u_{\mathrm{ALF}}
=
\operatorname{StableSerialize}
(
\text{TextWorld Action}
).
}
\]

推荐不要直接依赖 Python 对象 hash，而使用 deterministic serialization。

例如：

```python
(
    action.name,
    sorted(serialized_preconditions),
    sorted(serialized_postconditions),
)
```

---

## 9.5 推荐实现

### 主推荐：Internal Action Key

如果实现方便，在 ALFWorld wrapper 中开启：

```python
last_action=True
```

并导出内部 action 的稳定序列化结果。

定义：

\[
\boxed{
u_{\mathrm{ALF}}
=
(
name,
preconditions,
postconditions
).
}
\]

### 简化实现：Admissible Command

若暂时不修改 TextWorld wrapper，也可以使用：

\[
\boxed{
u_{\mathrm{ALF}}
=
\text{environment-executed admissible command}.
}
\]

对于当前 BACE 也已经足够严格。

---

## 9.6 结论

ALFWorld 不存在必须通过语义模型判断“两个动作是否相同”的问题。

可以做到：

\[
\boxed{
\text{模型文本}
\rightarrow
\text{环境 command}
\rightarrow
\text{TextWorld internal Action}.
}
\]

因此它是 BACE action identity 最干净的 benchmark 之一。

---

# 10. ScienceWorld：天然暴露结构化 Action Template 与 Object IDs

ScienceWorld 的 API 在 action identity 方面非常适合 BACE。

官方接口包括：

```python
get_possible_actions()
```

```python
get_possible_actions_with_IDs()
```

```python
get_valid_action_object_combinations()
```

以及最关键的：

```python
get_valid_action_object_combinations_with_templates()
```

后者返回每个合法 action 的结构：

```text
{
    "action": ...,
    "template_id": ...,
    "obj_ids": [...]
}
```

因此 ScienceWorld 已经直接把 action 分解成：

\[
\boxed{
\text{action template}
+
\text{grounded object IDs}.
}
\]

---

## 10.1 推荐 Action Key

直接定义：

\[
\boxed{
u_{\mathrm{SW}}
=
(
template\_id,
obj\_id^{(1)},
obj\_id^{(2)},
\dots
).
}
\]

例如：

```text
("pour X in Y", object_17538, object_17535)
```

对应一个唯一的 structured action。

---

## 10.2 如何从模型文本映射

ScienceWorld `step()` 最终仍接受字符串：

```python
env.step(input_str)
```

因此流程可以是：

```text
raw response
→ extract <action> body
→ lookup current valid action table
→ unique match
→ (template_id, obj_ids)
```

如果没有唯一匹配：

\[
u=\mathrm{INVALID}.
\]

完全不需要：

- LLM judge；
- embedding；
- semantic parser。

---

## 10.3 当前 BEACON/GiGPO-style Wrapper

当前公开的 ScienceWorld RL wrapper 主要还是：

```text
raw response
→ extract <action> body
→ strip
```

并把：

```python
env.get_valid_action_object_combinations()
```

写入：

```python
info["possible_actions"]
```

但官方 ScienceWorld API 已经暴露更严格的：

```text
template_id + obj_ids
```

所以 BACE 可以比现有 wrapper 再下沉一层。

---

## 10.4 结论

\[
\boxed{
\textbf{ScienceWorld 可以直接构造结构化 environment-native action key。}
}
\]

从 action identity 的可解释性看，它和 ALFWorld 都属于最适合 BACE 的环境。

---

# 11. WebShop：能严格拿到 Parser 后的实际操作

WebShop 没有像 ScienceWorld 那样统一的：

\[
(template\_id,obj\_ids)
\]

接口，但它的 action parser 和真正 browser operation 非常明确。

官方 `WebAgentTextEnv.step(action)` 只接受两类：

```text
search[keywords]
click[value]
```

首先执行：

```python
action_name, action_arg = parse_action(action)
```

然后：

```python
action_arg = action_arg.lower()
```

再进入真正环境逻辑。

---

# 12. WebShop Search

例如：

```text
search[Waterproof Hiking Shoes]
```

parser 后：

```text
action_name = "search"
action_arg  = "waterproof hiking shoes"
```

随后：

```python
browser.search(action_arg)
```

而 `browser.search()` 会进一步将字符串拆成：

```python
keywords = keywords.split(" ")
```

再交给 server。

因此最严格可以定义：

\[
\boxed{
u_{\mathrm{WebShopSearch}}
=
(
\mathrm{SEARCH},
tuple(keywords)
).
}
\]

例如：

```python
(
    "SEARCH",
    ("waterproof", "hiking", "shoes")
)
```

这就是 WebShop server 实际接收到的搜索输入。

---

# 13. WebShop Click

当前页面会构建：

```python
text_to_clickable
```

其中包括：

- buttons；
- product links；
- buying options；
- navigation controls。

有效 click 必须满足：

```python
action_arg in self.text_to_clickable.keys()
```

然后：

```python
browser.click(action_arg, self.text_to_clickable)
```

因此最自然的 key 是：

\[
\boxed{
u_{\mathrm{WebShopClick}}
=
(
\mathrm{CLICK},
clickable\_name
).
}
\]

其中：

```text
clickable_name
```

就是环境当前页面真正用于执行点击的 key。

---

## 13.1 是否需要更底层 DOM identity

通常不需要。

虽然理论上可以进一步记录：

- product ASIN；
- option field；
- button role；
- URL transition；

但 WebShop 当前 parser-level：

\[
(\text{CLICK},clickable\_name)
\]

已经对应实际 browser operation。

对于 BACE，这已经足够严格。

---

## 13.2 结论

WebShop 属于：

\[
\boxed{
\text{strict parser-level executable identity}
}
\]

而不是：

\[
\text{internal symbolic action-object identity}.
\]

但这对 BACE 完全足够。

---

# 14. Search-HotpotQA：开放 Query 仍然可以拿到严格 Tool-Call Identity

Search-Augmented QA 最大的误解是：

> query 是开放自然语言，所以必须用 embedding/LLM 判断两个 query 是否同一动作。

实际上不需要。

---

## 14.1 verl-agent Search Projection

GiGPO / verl-agent 的：

```text
agent_system/environments/env_package/search/projection.py
```

会：

1. 提取第一个完整 `<search>...</search>`；
2. 若无 search，则提取 `<answer>...</answer>`；
3. 对 payload `.strip()`；
4. 规范 tag；
5. 检查多 search / 多 answer / search+answer 共存等 invalid 情况。

因此已经得到确定性的 structured action block。

---

## 14.2 SearchEnv 真正解析并执行 Tool Call

`SearchEnv` 会解析：

```python
query = self._parse_action(action)
```

然后调用：

```python
_execute_tool(
    "SearchToolGroup",
    "search",
    query
)
```

并在 metadata 中记录：

```text
tool_group
tool_name
tool_input
```

因此可以直接得到：

\[
\boxed{
u
=
(
\texttt{SearchToolGroup},
\texttt{search},
q
).
}
\]

---

## 14.3 还能继续后移到真正 HTTP Request

`SearchToolGroup.search(query)` 会进一步执行：

```python
query = query.strip()
```

然后构造：

```python
payload = {
    "query": query,
    "topk": topk,
    "return_scores": ...
}
```

并发送给 retrieval server。

所以 BACE 可以直接把：

\[
\boxed{
q_{\mathrm{exec}}
=
\text{真正发送到 retrieval API 的 query}
}
\]

作为 action payload。

最终：

\[
\boxed{
u_{\mathrm{SearchQA}}
=
(
\mathrm{SEARCH},
q_{\mathrm{exec}}
).
}
\]

---

# 15. Search-QA 中不用做 Semantic Query Clustering

例如：

```text
<search>The Godfather director</search>
```

与：

```text
<search>who directed The Godfather</search>
```

即使语义高度接近，也保持：

\[
u_1\neq u_2.
\]

原因：

BACE 本身可以对 singleton / low-count action 主动补 evidence：

\[
p_{z,u}
\rightarrow
\Delta ERV(z,u)
\rightarrow
\text{targeted resampling}.
\]

因此：

\[
\boxed{
\text{query singleton 是 BACE 要解决的问题，}
}
\]

而不是：

\[
\boxed{
\text{必须先通过语义聚类消掉的问题。}
}
\]

---

# 16. Search-QA 的 ANSWER 与 INVALID

定义：

\[
u\in
\{
(\mathrm{SEARCH},q),
(\mathrm{ANSWER},y),
\mathrm{INVALID}
\}.
\]

主 acquisition 只允许：

\[
\boxed{
(\mathrm{SEARCH},q)
}
\]

进入 branch candidate set。

原因：

- `ANSWER(y)` 是 terminal action；
- deterministic verifier 下重复相同 answer 不产生新的 continuation evidence；
- `INVALID` 没有稳定 intended tool-call identity。

但 ANSWER/INVALID 仍可以保留用于：

- trajectory training；
- GiGPO local groups；
- action diagnostics。

---

# 17. Search-QA Replay 推荐

为了让相同：

\[
u=(SEARCH,q)
\]

在 branch replay 中保持真正相同的 environment intervention，推荐：

\[
\boxed{
\text{fixed local retriever}
+
\text{fixed corpus/index}
+
\text{query-result cache}.
}
\]

Cache key 可定义为：

```python
(
    retriever_version,
    corpus_version,
    topk,
    q_exec,
)
```

这样相同 query 在 branch 中得到同一 retrieval result，branch outcome variation 主要来自：

\[
\pi_{\mathrm{old}}
\]

的后续 continuation，而不是检索服务漂移。

---

# 18. 普通数学 Benchmark：真正缺少 Benchmark-Native Action Identity

对于：

- GSM8K；
- MATH；
- MATH-500；
- AIME；

benchmark 本身通常只提供：

```text
question
answer / solution
metadata
```

而没有：

\[
s_t
\rightarrow
a_t
\rightarrow
s_{t+1}
\]

这样的 environment transition API。

例如模型生成：

```text
factor the quadratic
```

与：

```text
rewrite as (x-2)(x-3)
```

benchmark 本身不会告诉我们：

> 这两个 reasoning steps 是否属于同一个环境 action。

因此 plain math 中不存在 benchmark-native：

- action parser；
- action ID；
- executable command；
- structured tool call。

如果要人为分类，只能引入：

- LLM semantic judge；
- reasoning-step classifier；
- embedding；
- learned ontology；
- manually designed reasoning taxonomy。

这些都不符合当前 BACE 主线的“environment-grounded deterministic action identity”。

---

# 19. Tool-Augmented Math 是例外

如果数学环境允许：

- Python；
- calculator；
- symbolic CAS；
- Lean；
- theorem-proving tactics；
- structured function calling；

则可以重新获得：

\[
\boxed{
\text{environment-executable action identity}.
}
\]

例如：

\[
u=
(
\mathrm{PYTHON},
code
)
\]

或者：

\[
u=
(
\mathrm{LEAN\_TACTIC},
tactic,
arguments
).
\]

这时数学 benchmark 可以重新适配 BACE。

关键不是“是不是数学题”，而是：

\[
\boxed{
\text{是否存在 deterministic executable-action interface}.
}
\]

---

# 20. Benchmark 适用条件的最终修正

因此 BACE 的适用条件不应该写成：

> 环境必须拥有有限离散 action space。

而应该写成：

\[
\boxed{
\text{环境必须提供稳定、确定性的 executable-action identity。}
}
\]

这个 identity 可以是：

### ALFWorld

\[
\text{TextWorld internal Action}
\]

### ScienceWorld

\[
(template\_id,obj\_ids)
\]

### WebShop

\[
(action\_type,actual\_argument)
\]

### Search-HotpotQA

\[
(tool\_name,actual\_tool\_input)
\]

因此 BACE 可以支持：

\[
\boxed{
\text{开放但具有 deterministic parser 的参数化动作空间。}
}
\]

---

# 21. 最终推荐 Action Key

| Environment | 推荐主 key |
|---|---|
| ALFWorld-TextWorld | `("TW_ACTION", name, preconditions, postconditions)` |
| ALFWorld 简化实现 | `("ALF_COMMAND", executable_admissible_command)` |
| ScienceWorld | `("SW", template_id, obj_id_1, obj_id_2, ...)` |
| WebShop Search | `("SEARCH", tuple(actual_keywords))` |
| WebShop Click | `("CLICK", actual_clickable_name)` |
| Search-HotpotQA | `("SEARCH", exact_executed_query)` |
| Search Answer | `("ANSWER", exact_parsed_answer)`，不进入 branch acquisition |
| Invalid action | `("INVALID",)` |
| Plain Math | 无推荐 benchmark-native key |
| Tool-Math | `(tool_name, normalized structured args)` |

---

# 22. 与 BACE Posterior 的统一

所有 benchmark 最终都使用：

\[
\mathcal I_g(z,u)
=
\left\{
i:
z_i=z,
u_i=u
\right\}.
\]

局部 posterior：

\[
p_{g,z,u}
\sim
Beta(\alpha_{g,z,u},\beta_{g,z,u}).
\]

自然 occurrence 或 branch outcome：

\[
Y\in\{0,1\}
\]

更新同一个：

\[
(z,u).
\]

这样：

\[
\boxed{
\text{acquisition unit}
=
\text{branch unit}
=
\text{credit aggregation unit}.
}
\]

---

# 23. 与 Branch Origin 的统一

如果 ERV 选择：

\[
(z^*,u^*)
\]

主方案从：

\[
\boxed{
\mathcal I_{\mathrm{obs}}(z^*,u^*)
=
\{
i:
\text{natural occurrence }i
\text{ 在 }z^*
\text{ 实际执行过 }u^*
\}
}
\]

中选 concrete origin。

这要求：

\[
u
\]

必须严格对应该 occurrence 实际执行过的 environment action。

因此 action canonicalizer 不能只是一个“统计分组工具”，它还必须与真实 branch execution 对齐。

---

# 24. 与 Action-Aggregated Local Credit 的统一

若启用：

\[
A_{g,z,u}^{S,\mathrm{act}}
\]

则也直接按同一：

\[
u=\operatorname{ExecCanon}(a)
\]

聚合。

即：

\[
\boxed{
u_{\mathrm{Bayes}}
=
u_{\mathrm{ERV}}
=
u_{\mathrm{branch}}
=
u_{\mathrm{local\ credit}}.
}
\]

这是当前方案最重要的实现一致性要求之一。

---

# 25. 推荐的实现优先级

## P0：先实现统一 Adapter

为每个环境暴露：

```python
get_executed_action_key(...)
```

并记录：

```text
raw_response
projected_action
environment_action
canonical_action_key
valid
```

---

## P1：ALFWorld

先用：

```text
executable admissible command
```

即可。

如果工程上容易，再升级到：

```text
TextWorld internal Action serialization
```

这不会改变方法逻辑，只是让 key 更严格。

---

## P1：ScienceWorld

优先直接使用：

```text
template_id + obj_ids
```

不要只停留在 raw action string。

---

## P1：WebShop

直接记录 parser 后：

```text
SEARCH + actual keywords
CLICK + actual clickable
```

无需额外 semantic normalization。

---

## P1：Search-HotpotQA

记录真正发送给 retrieval server 的：

```text
SEARCH + q_exec
```

并配套 fixed retriever/cache。

---

# 26. 推荐的 Action-Identity Audit

在正式训练前，建议做一个小规模 audit。

对每个 benchmark 记录：

```text
# raw action-tag classes
# projected action classes
# executed action classes
raw→projected merge ratio
projected→executed merge ratio
invalid ratio
singleton-action ratio
actions per anchor
```

特别是比较：

\[
u_{\mathrm{tag}}
\]

和：

\[
u_{\mathrm{exec}}.
\]

定义：

\[
\mathrm{Agreement}
=
\frac{
\#\{i:u_i^{tag}=u_i^{exec}\}
}{
N
}.
\]

如果：

\[
\mathrm{Agreement}\approx 1,
\]

则说明严格后移不会明显改变正常数据，只会修复 parser corner cases。

---

# 27. 推荐的 Ablation

主版本：

\[
\boxed{
\text{Executed-action identity}
}
\]

可做两个轻量消融。

## A. Raw Action-Tag Identity

\[
u=\text{exact extracted action body}.
\]

用于验证继续下沉到环境层是否必要。

## B. Relaxed Semantic / Effect Identity

只在 Search-QA 等开放 payload 环境小规模测试：

- retrieval-equivalent query；
- embedding query cluster。

不建议作为主方法。

---

# 28. 对论文叙事的影响

这个设计使 BACE 的适用范围可以更准确地描述为：

> BACE does not require a finite enumerated action space. It requires a deterministic environment interface that maps model outputs to stable executable-action identities.

也就是：

\[
\boxed{
\text{finite action space 不是必要条件，}
}
\]

真正要求的是：

\[
\boxed{
\text{stable executable-action identity}.
}
\]

因此：

- ALFWorld；
- ScienceWorld；
- WebShop；
- Search-Augmented QA；

都满足主方法要求。

普通 MATH/AIME 不满足，不是因为它们“是数学题”，而是因为 plain CoT reasoning 没有独立环境 action interface。

---

# 29. 最终固定方案

当前 BACE-GiGPO 动作身份方案建议正式冻结为：

\[
\boxed{
u
=
\operatorname{ExecCanon}_z(a)
}
\]

其中：

\[
\operatorname{ExecCanon}
=
\text{action-tag extraction}
+
\text{environment parser/projection}
+
\text{最深可获得的 stable action representation}.
\]

按 benchmark：

\[
\boxed{
\begin{aligned}
\text{ALFWorld: }&
u=\text{TextWorld Action / executable command};
\\[1mm]
\text{ScienceWorld: }&
u=(template\_id,obj\_ids);
\\[1mm]
\text{WebShop: }&
u=(action\_type,actual\_environment\ argument);
\\[1mm]
\text{SearchQA: }&
u=(SEARCH,q_{\mathrm{exec}});
\\[1mm]
\text{Plain Math: }&
\text{无 benchmark-native executable action identity}.
\end{aligned}
}
\]

总体设计原则：

\[
\boxed{
\textbf{Extract like BiPACE, identify at the environment layer.}
}
\]

一句话概括：

> **BACE 不自行学习“两个自然语言动作是否语义相同”，而是尽量直接读取环境真正执行的动作身份。对于 ALFWorld 和 ScienceWorld，可以下沉到内部结构化 action；对于 WebShop 和 Search-QA，可以下沉到 parser/tool-call 层；只有没有独立环境 action interface 的 plain math benchmark 才真正需要额外的动作语义建模。**

---

# 30. 主要代码依据

## ALFWorld / TextWorld

### verl-agent

- `examples/gigpo_trainer/run_alfworld.sh`
- `agent_system/environments/env_package/alfworld/projection.py`
- `agent_system/environments/env_package/alfworld/envs.py`

Repository:

```text
https://github.com/langfengQ/verl-agent
```

### ALFWorld

- `alfworld/agents/environment/alfred_tw_env.py`

Repository:

```text
https://github.com/alfworld/alfworld
```

### TextWorld

- `textworld/envs/wrappers/tw_inform7.py`
- `textworld/logic/__init__.py`

Repository:

```text
https://github.com/microsoft/TextWorld
```

关键接口：

```text
admissible_commands
last_action
_game_progression.valid_actions
TextWorld Action(name, preconditions, postconditions)
```

---

## ScienceWorld

Repository:

```text
https://github.com/allenai/ScienceWorld
```

关键接口：

```python
get_possible_actions()
get_possible_actions_with_IDs()
get_valid_action_object_combinations()
get_valid_action_object_combinations_with_templates()
```

其中后者暴露：

```text
action
template_id
obj_ids
```

BEACON adapter:

```text
https://github.com/ZJU-REAL/BEACON
```

相关文件：

- `agent_system/environments/env_package/sciworld/projection.py`
- `agent_system/environments/env_package/sciworld/envs.py`

---

## WebShop

Repository:

```text
https://github.com/princeton-nlp/WebShop
```

关键文件：

- `web_agent_site/envs/web_agent_text_env.py`
- `web_agent_site/engine/engine.py`

关键接口：

```python
parse_action(action)
browser.search(...)
browser.click(...)
text_to_clickable
```

verl-agent adapter:

- `agent_system/environments/env_package/webshop/projection.py`

---

## Search-HotpotQA / Search-Augmented QA

verl-agent:

```text
https://github.com/langfengQ/verl-agent
```

关键文件：

- `agent_system/environments/env_package/search/projection.py`
- `agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/env.py`
- `agent_system/environments/env_package/search/third_party/skyrl_gym/tools/search.py`

关键执行链：

```text
raw response
→ search_projection
→ SearchEnv._parse_action
→ SearchToolGroup.search
→ HTTP payload {"query": q_exec, ...}
```

因此：

\[
u=(SEARCH,q_{\mathrm{exec}})
\]

具有明确 environment-executed semantics。

---

## Plain Math

典型 benchmark：

- GSM8K
- MATH
- MATH-500
- AIME

这些 benchmark 本身没有独立：

```text
state
action
transition
```

接口，因此 plain CoT step 没有 benchmark-native action identity。

只有在额外接入：

- Python；
- calculator；
- symbolic solver；
- theorem prover；
- structured tool API；

后，才重新获得 environment-grounded executable action key。
