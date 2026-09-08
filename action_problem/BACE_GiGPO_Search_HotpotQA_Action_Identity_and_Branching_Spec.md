# BACE-GiGPO 在 Search-HotpotQA 上的动作身份识别与动作分组方案

> **文档定位**：Search-HotpotQA / Search-Augmented QA 环境适配规范  
> **适用方法**：BACE-GiGPO  
> **版本日期**：2026-09-05  
> **核心目标**：在不引入额外语义分类模型的前提下，为 Search-HotpotQA 定义稳定、可复现、environment-grounded 的 action identity，使 observed-action posterior、ERV 主动分支、action-level credit 与 replay 都具有清晰语义。

---

# 0. 结论先行

对 Search-HotpotQA，**主版本不应该用 LLM、embedding 或语义相似度去判断两个 query 是否“意思相同”**。

最干净的定义是：

\[
\boxed{
u=\operatorname{CanonicalToolCall}(a)
}
\]

其中模型原始输出 \(a\) 先经过 GiGPO / verl-agent Search 环境本身的确定性 parser / projection，得到环境真正执行的工具调用。

对于搜索动作：

\[
\boxed{
u=(\mathrm{SEARCH},q_{\mathrm{exec}})
}
\]

其中 \(q_{\mathrm{exec}}\) 是环境实际传给 retrieval server 的 query 字符串。

对于回答动作：

\[
\boxed{
u=(\mathrm{ANSWER},y_{\mathrm{exec}})
}
\]

其中 \(y_{\mathrm{exec}}\) 是环境实际解析出的最终答案字符串。

无效输出：

\[
\boxed{
u=\mathrm{INVALID}
}
\]

主 BACE acquisition 中只允许：

\[
\boxed{
\mathcal C^{\mathrm{branch}}(z)
=
\{(\mathrm{SEARCH},q)\}
}
\]

也就是说：

- `SEARCH(query)` 是可主动重复的局部实验；
- `ANSWER(answer)` 是终止动作，不作为 branch action；
- `INVALID` 不进入 branch candidate set。

最重要的一点是：

> **BACE 不要求一个 query 在 natural roots 中已经重复多次。**  
> 只要同一 anchor 中自然暴露了多个不同的 observed search actions，即使每个 query 都只出现一次，也可以为每个 \((z,u)\) 建立弱 Beta posterior，再用 ERV 决定下一条 branch 应主动重复验证哪个 query。

因此 Search-HotpotQA 的开放文本 query 并不会破坏 BACE 的核心机制。

主版本建议：

\[
\boxed{
\text{GiGPO similarity anchor}
+
\text{exact executed-query identity}
+
\text{observed SEARCH actions only}
+
\text{cache-backed deterministic replay}
}
\]

这既最大程度对齐原始 GiGPO Search 实现，也避免把“动作语义聚类”本身变成一个额外学习模块。

---

# 1. 为什么需要单独处理 Search-HotpotQA

在 ALFWorld 中，动作来自有限的 admissible commands：

```text
go to fridge 1
open fridge 1
take apple 1 from fridge 1
```

因此两个模型输出是否执行同一种动作，可以由环境直接映射到同一 admissible command。

WebShop 也类似：

```text
search[waterproof hiking shoes]
click[item_123]
click[buy now]
```

最终动作可以映射到确定的页面 action。

Search-HotpotQA 看起来更困难，因为模型生成的是开放 query：

```text
<search>Who directed The Godfather?</search>
```

另一个 rollout 可能生成：

```text
<search>The Godfather director</search>
```

从人的语义看它们高度相近，但字符串不同。

如果我们要求先解决：

\[
\text{“两个自然语言 query 是否语义等价？”}
\]

再运行 BACE，就会引入新的：

- semantic encoder；
- similarity threshold；
- LLM judge；
- clustering algorithm；
- cluster drift；
- non-transitive equivalence；

从而使方法复杂化。

实际检查 GiGPO / verl-agent 的 Search 环境后可以发现，这个问题没有必要被提升到“语义等价”层面。

---

# 2. GiGPO Search 环境真正执行的动作是什么

## 2.1 模型每一步的输出协议

verl-agent 的 Search prompt 要求每一步先生成：

```text
<think> ... </think>
```

然后二选一：

```text
<search> query </search>
```

或：

```text
<answer> final answer </answer>
```

也就是说，一次 macro-action 包含：

\[
\text{CoT}
+
\text{structured action block}.
\]

BACE 的 action identity 不应该使用整段 CoT，而应该使用 parser 后真正进入环境的 action block。

---

## 2.2 `search_projection` 已经完成第一层 canonicalization

GiGPO / verl-agent 中的 `search_projection` 会：

1. 找到第一个完整的 `<search>...</search>`；
2. 若没有 search，则找第一个 `<answer>...</answer>`；
3. 对 tag 内部内容执行 `.strip()`；
4. 将 tag 规范成固定格式；
5. 多个 search tag、多个 answer tag、search 与 answer 同时出现时标记为 invalid；
6. 没有完整合法 action block 时标记 invalid。

因此下面两段原始模型输出：

```text
<think>...</think>
<SEARCH>   The Godfather director   </SEARCH>
```

和

```text
<think>another reasoning...</think>
<search>The Godfather director</search>
extra hallucinated text
```

在合法投影情况下，都可以得到标准化的环境动作：

```text
<search>The Godfather director</search>
```

这里已经消除了：

- CoT 差异；
- tag 大小写；
- tag 外多余文本；
- 首尾空格；

这些 surface variation。

---

## 2.3 SearchEnv 再把 action block 解析成 tool input

环境内部对于：

```text
<search>The Godfather director</search>
```

解析：

\[
q_{\mathrm{exec}}
=
\texttt{"The Godfather director"}.
\]

随后调用固定工具：

```text
tool_group = SearchToolGroup
tool_name  = search
tool_input = query
```

因此从环境执行语义上，真正的动作是：

\[
\boxed{
u=(\texttt{SearchToolGroup},\texttt{search},q_{\mathrm{exec}})
}
\]

对于 BACE 可以简写为：

\[
\boxed{
u=(\mathrm{SEARCH},q_{\mathrm{exec}}).
}
\]

这与 ALFWorld 的：

\[
u=\text{environment-executed admissible command}
\]

是同一层级的定义。

---

# 3. ECPO 给我们的直接证据与文献边界

ECPO 的核心方法明确要求先把模型产生的文本动作映射为 canonical action：

\[
u_{i,t}=\operatorname{can}(a_{i,t}),
\]

然后在同一 anchor 内按 \(u\) 聚合 return。

ECPO 对 canonicalizer 的一般定义是：

> deterministic and environment-grounded，消除表面文本变化，但保持 executable action identity。

ECPO 还明确在 Search-Augmented QA 上训练 NQ + HotpotQA，并测试：

- NQ；
- TriviaQA；
- PopQA；
- HotpotQA；
- 2WikiMultiHopQA；
- MuSiQue；
- Bamboogle。

这说明 **action-level credit calibration 在 Search-QA 上至少是经验可行的**。

但需要保持文献表述准确：

> ECPO 附录对 ALFWorld / WebShop 的 canonicalization 写得很具体：normalize 后匹配环境 admissible actions。  
> 对 Search-QA，论文给出了 `<search>` / `<answer>` 协议和实验结果，但没有单独公开“query-specific canonicalizer”的完整实现细节。

因此本文档不声称：

> “ECPO 已明确按 exact parsed query string 分组。”

我们的主定义来自更直接、可审计的依据：

\[
\boxed{
\text{GiGPO / verl-agent SearchEnv 实际执行的 tool call}
}
\]

也就是说，我们自己的 Search-HotpotQA action identity 会比 ECPO 论文附录中公开的 Search-specific 细节更明确。

---

# 4. 主版本 Action Identity 的正式定义

## 4.1 一级分类：action type

Search-HotpotQA 中定义：

\[
\operatorname{type}(u)
\in
\{
\mathrm{SEARCH},
\mathrm{ANSWER},
\mathrm{INVALID}
\}.
\]

### SEARCH

```text
<search> query </search>
```

### ANSWER

```text
<answer> answer </answer>
```

### INVALID

包括：

- 缺少完整 search / answer block；
- 同时出现 search 和 answer；
- 出现多个 search；
- 出现多个 answer；
- parser 无法得到有效 action。

---

## 4.2 二级身份：payload-specific executable action

不能只把所有 `SEARCH` 当成同一个动作。

下面两个动作：

```text
<search>The Godfather director</search>
```

```text
<search>Francis Ford Coppola nationality</search>
```

显然对应不同的信息获取决策。

因此 fine-grained action key 为：

\[
\boxed{
k_{\mathrm{act}}(u)
=
(\operatorname{type}(u),\operatorname{payload}(u)).
}
\]

具体：

\[
k_{\mathrm{act}}
=
(\mathrm{SEARCH},q_{\mathrm{exec}})
\]

或：

\[
k_{\mathrm{act}}
=
(\mathrm{ANSWER},y_{\mathrm{exec}}).
\]

`INVALID` 使用固定 singleton key：

\[
k_{\mathrm{act}}=\mathrm{INVALID}.
\]

---

# 5. Query 到底应该规范化到什么程度

这是最需要谨慎的地方。

## 5.1 主版本：只使用环境已经执行的确定性 normalization

主版本建议：

\[
\boxed{
q_{\mathrm{key}}
=
q_{\mathrm{exec}}
}
\]

其中 \(q_{\mathrm{exec}}\) 就是 verl-agent parser / SearchToolGroup 最终实际发送给 retrieval server 的 query。

环境已经做了：

- action block extraction；
- tag standardization；
- leading/trailing whitespace stripping。

主版本**不再额外进行**：

- lowercasing；
- stemming；
- stopword removal；
- punctuation deletion；
- entity linking；
- LLM rewriting；
- embedding clustering；
- semantic similarity matching。

原因是：

\[
\boxed{
\text{action identity 应在执行前定义，而不是根据我们认为“语义相似”事后合并。}
}
\]

---

## 5.2 为什么不建议主版本 lower-case / 删除标点

虽然绝大多数 dense retriever 对大小写并不敏感，但从严格方法定义看：

```text
US
```

与：

```text
us
```

不是环境代码显式保证等价的两个 tool inputs。

同理：

```text
Who directed "Crash"?
```

与：

```text
Who directed Crash?
```

很可能检索结果相近，但不是环境接口定义的完全相同 executable action。

因此如果主版本自行合并，就会把：

\[
\text{environment-grounded identity}
\]

改成：

\[
\text{researcher-defined heuristic identity}.
\]

不值得。

---

# 6. 为什么不需要语义 Query Clustering

假设同一个 anchor 下 5 条 natural roots 给出：

```text
q1 = "The Godfather director"
q2 = "who directed The Godfather"
q3 = "Francis Ford Coppola nationality"
q4 = "The Godfather director"
q5 = "Francis Ford Coppola citizenship"
```

精确动作类为：

\[
u_1=(SEARCH,q_1),
\]

\[
u_2=(SEARCH,q_2),
\]

\[
u_3=(SEARCH,q_3),
\]

\[
u_4=(SEARCH,q_5).
\]

其中：

\[
n(u_1)=2,
\quad
n(u_2)=n(u_3)=n(u_4)=1.
\]

ECPO 这类被动动作聚合方法会担心 singleton action evidence 过弱。

但 BACE 的关键能力恰恰是：

\[
\boxed{
\text{主动对 singleton observed action 再采一条 branch。}
}
\]

例如：

\[
p_{z,u_3}\sim Beta(\alpha,\beta)
\]

只有一次 natural evidence，也仍然可以计算：

\[
\Delta ERV(z,u_3).
\]

如果它最值得确认，BACE 就主动重复：

```text
SEARCH["Francis Ford Coppola nationality"]
```

得到第二个 outcome。

因此：

\[
\boxed{
\text{query singleton 是 BACE 要解决的问题，而不是必须提前通过语义聚类消掉的问题。}
}
\]

这是 Search-HotpotQA 适配 BACE 的核心逻辑。

---

# 7. Anchor 如何与 GiGPO Search 保持一致

动作 identity 可以使用 exact executed query，但 anchor 仍应与原始 GiGPO Search 设置对齐。

## 7.1 GiGPO Search 并不是 exact observation grouping

verl-agent 的 `run_search.sh` 明确设置：

```text
enable_similarity=True
similarity_thresh=0.9
```

GiGPO 的相似度函数使用字符串 `SequenceMatcher`。

因此 Search-HotpotQA 的主 BACE adapter 应定义：

\[
\boxed{
z=\operatorname{GiGPOCluster}_{0.9}(o_t)
}
\]

而不是自行换成新的 state representation。

---

## 7.2 SearchEnvironmentManager 的 anchor 是什么

在 SearchEnvironmentManager 中：

- reset 时：
  \[
  anchor = question;
  \]
- search 后：
  \[
  anchor = retrieved\ observation.
  \]

也就是说，GiGPO 的 Search anchor 主要由当前返回的 information text 形成。

完整 interaction history 会进入下一步模型 prompt，但**不进入 GiGPO anchor key 本身**。

因此两个 occurrences 可能：

- 当前 retrieval observation 很相似；
- arrival search histories 不同；

仍被归入同一个 GiGPO anchor。

这与 ALFWorld 中“相同 observation 不要求完整历史相同”的 GiGPO 设计原则是一致的，只是在 Search 环境里使用了 similarity grouping。

---

## 7.3 BACE 中必须保留 concrete occurrence

虽然多个 histories 被聚合成 anchor \(z\)，不能把它们丢掉。

对每个 occurrence \(i\) 仍保存：

```text
question_id
trajectory_id
step_index
raw_anchor_obs
GiGPO_cluster_id
full_search_history
retrieved_information_history
raw_model_response
projected_action
action_key
terminal_outcome
```

BACE 分支时真正恢复的是：

\[
\boxed{
(z,i)
}
\]

中的 concrete occurrence \(i\)，而不是一个抽象 cluster representative。

---

# 8. Search-HotpotQA 中的局部成功概率

对于一个 GiGPO similarity anchor \(z\) 和 observed search query \(u\)，定义：

\[
\boxed{
p_{g,z,u}
=
P
\left(
Y=1
\mid
g,z,\operatorname{do}(u),\pi_{\mathrm{old}}
\right)
}
\]

其中：

- \(g\)：当前 HotpotQA question；
- \(z\)：GiGPO similarity anchor；
- \(u=(SEARCH,q)\)；
- \(Y=1\)：最终回答正确；
- continuation 由冻结的 \(\pi_{\mathrm{old}}\) 完成。

由于 anchor cluster 可能包含多个 arrival histories，更准确地写：

\[
p_{g,z,u}
=
\mathbb E_{h\sim\nu_g(\cdot\mid z)}
\left[
P(Y=1\mid h,\operatorname{do}(u),\pi_{\mathrm{old}})
\right].
\]

因此 branch replay 时，应对 cluster 内 concrete origins 做均衡抽样，而不是永远从同一 history 开始。

---

# 9. Search-HotpotQA 的 Candidate Action Set

## 9.1 主版本：Observed SEARCH Actions Only

对 anchor \(z\)：

\[
\boxed{
\mathcal C_{\mathrm{search}}(z)
=
\left\{
(\mathrm{SEARCH},q):
\text{该 query 在 natural roots 的 }z\text{ 中真实执行过}
\right\}.
}
\]

这与当前 BACE 的 observed-action philosophy 对齐：

> 先让 natural roots 暴露局部行为候选，再用 branch 主动补充证据；不额外凭空生成未见 query。

---

## 9.2 Effective branch anchor 的严格条件

Search-HotpotQA 主版本建议：

\[
\boxed{
|\mathcal C_{\mathrm{search}}(z)|\ge 2.
}
\]

即至少自然出现两个不同的有效 search query。

此外要求：

- \(z\) 不是初始 question state；
- 当前 episode 未终止；
- 仍有至少一个 search turn 可用；
- concrete origin 可恢复；
- retrieval replay 验证通过；
- 至少一个 query 的 ERV 高于阈值；
- 当前 anchor 未超过 \(L_{\max}\) branch capacity。

---

# 10. ANSWER 动作怎么处理

这是 Search-QA 特有的问题。

## 10.1 主版本推荐：参与训练，但不参与主动 Branch Acquisition

`ANSWER(y)` 一旦执行即终止。

如果 verifier 是确定性的，那么重复执行完全相同的：

```text
<answer>Paris</answer>
```

不会产生新的 continuation evidence。

因此：

\[
\boxed{
ANSWER(y)\notin\mathcal C_{\mathrm{branch}}(z).
}
\]

但 ANSWER occurrence 仍然：

- 属于真实 actor decision；
- 进入最终 trajectory advantage；
- 进入原始 GiGPO step-level group；
- 如果启用 action-aggregated credit，可按 exact answer string 聚合。

这样主 BACE acquisition 专门负责：

\[
\boxed{
\text{“下一次应该验证哪个检索决策？”}
}
\]

而不是重复一个 deterministic terminal answer。

---

## 10.2 可选扩展：Search-vs-Stop Mixed-Arm ERV

如果后续希望让 BACE 显式研究：

\[
\text{继续搜索}
\quad\text{vs.}\quad
\text{现在回答}
\]

可以把 observed `ANSWER(y)` 作为一个固定-value arm。

对于 deterministic verifier：

\[
p_{z,ANSWER(y)}
\in\{0,1\}
\]

视为已知常数，不使用 Beta posterior。

Search arms 仍为：

\[
p_{z,SEARCH(q)}
\sim Beta(\alpha,\beta).
\]

ERV 计算时：

- answer arm 可以成为 incumbent；
- answer arm 自身 acquisition value = 0；
- 只有 search arm 可以被主动采样。

这是更完整但略复杂的扩展，不建议首版主实现先加入。

---

# 11. INVALID 动作怎么处理

GiGPO Search parser 已经给出 deterministic invalid criteria。

BACE 主版本：

\[
\boxed{
INVALID\notin\mathcal C_{\mathrm{branch}}(z).
}
\]

理由：

- 无法定义稳定、可重复的 intended tool call；
- replay invalid output 不提供有意义的 action-credit evidence；
- GiGPO 已经可以通过 invalid-action penalty 学习格式约束。

实现中仍保留：

```text
invalid_action_count
invalid_action_rate
```

作为诊断。

如果后续做 action-level advantage aggregation，可以按照 ECPO 的原则：

\[
INVALID
\]

只和其他 INVALID occurrence 聚合，永不与合法 query 合并。

---

# 12. Candidate 数过多怎么办

Search query 是开放空间，同一 anchor 下可能出现多个不同 query。

仍然沿用 BACE 的统一 candidate cap：

\[
|\mathcal C(z)|\le K_{\max}.
\]

推荐：

\[
K_{\max}=4.
\]

Search adapter 不应该发明一套完全新的 selection objective。

建议优先级：

1. natural occurrence count 更高的 query；
2. 当前局部 posterior / ERV 更可能成为 incumbent 或 challenger 的 query；
3. 若仍并列，使用 natural rollout 中记录的 action log-prob；
4. 最后按 exact action key 做 deterministic tie-break。

最关键的是：

> candidate truncation 发生在 exact action key 形成之后，不能通过“先语义聚类再减少 action 数”替代。

---

# 13. Branch Origin 怎么选择

假设 anchor cluster：

\[
z=\{i_1,i_2,i_3\}
\]

包含三个 concrete occurrences。

选中的动作：

\[
u^*=(SEARCH,q^*).
\]

Search query 对任意非终止 Search state 都是 syntactically executable，因此可以在所有满足剩余 turn 约束的 origins 中执行。

主版本建议：

\[
\boxed{
i^*
\sim
\operatorname{Uniform}
\left(
\mathcal I_{\mathrm{eligible}}(z)
\right).
}
\]

若同一 \((z,u)\) 被重复 branch，多次 branch 优先对不同 origins 做 round-robin / without-replacement，再开始重复 origin。

目的：

- 避免一个特定 history 垄断 posterior；
- 让 \(p_{g,z,u}\) 更接近对 anchor arrival-history distribution 的平均；
- 与当前 BACE 的 “uniform concrete origin” 思路一致。

---

# 14. Search Environment 的 Replay 应该怎么做

## 14.1 为什么 Search replay 比物理环境简单

Search state 主要由：

\[
\text{question}
+
\text{past search queries}
+
\text{retrieved information}
\]

构成。

没有 ALFWorld 那种复杂的 object/inventory symbolic world state。

因此恢复一个 concrete origin，只需要重建同一段 interaction history。

---

## 14.2 主实验必须使用固定 Local Retriever

Search-R1 支持：

- local BM25；
- local E5 flat index；
- local E5 ANN / HNSW；
- online Google / SerpAPI 等。

为了 BACE replay，推荐：

\[
\boxed{
\text{固定 Wikipedia corpus + 固定 index + local E5-flat / BM25}
}
\]

不推荐主实验使用 online search engine。

原因：

- online ranking 会随时间变化；
- snippets 会更新；
- API personalization / localization 可能改变结果；
- replay 后 observation 无法稳定验证。

ANN/HNSW 也比 exact flat retrieval 更容易出现细小 nondeterminism，因此如果资源允许，首版优先 E5-flat。

---

## 14.3 推荐加入 Query-Result Cache

即使使用 local retriever，也建议在 acquisition batch 内建立：

\[
\boxed{
\mathrm{CacheKey}
=
(
retriever\_version,
corpus\_version,
topk,
q_{\mathrm{exec}}
)
}
\]

缓存：

```text
query
retrieved_doc_ids
retrieved_text
```

自然 root 第一次执行 query 时写入 cache。

branch replay 时：

- 相同 historical query 优先读取 cache；
- 不重新请求 live retriever；
- selected branch action \(q^*\) 如果已经出现过，也可以用相同 deterministic retrieval result；
- branch 的随机性只来自后续 \(\pi_{\mathrm{old}}\) continuation，而不是搜索服务漂移。

这样：

\[
\boxed{
\text{branch outcome variation}
}
\]

更能解释成：

\[
\boxed{
\text{continuation-policy variation}
}
\]

而不是：

\[
\text{retriever server noise}.
\]

---

# 15. Replay Verification

对 origin \(i^*\) 重放 prefix 后，至少检查：

### 1. 相同 question

\[
question_{\mathrm{replay}}=question_{\mathrm{recorded}}.
\]

### 2. 相同 turn index

\[
t_{\mathrm{replay}}=t_{\mathrm{recorded}}.
\]

### 3. 相同 historical canonical queries

\[
q_{1:t-1}^{\mathrm{replay}}
=
q_{1:t-1}^{\mathrm{recorded}}.
\]

### 4. 相同 retrieved outputs

强验证：

\[
Hash(info_{\mathrm{replay}})
=
Hash(info_{\mathrm{recorded}}).
\]

### 5. 同一 GiGPO anchor cluster

至少满足：

\[
Sim(
o_{\mathrm{replay}},
o_{\mathrm{anchor\ rep}}
)
\ge 0.9.
\]

主工程建议使用第 4 条的 exact hash 作为 concrete-origin replay assertion；第 5 条只是算法层面的 cluster membership。

如果 replay verification 失败：

```text
abort branch
record replay_failure
fallback / reschedule branch slot according to global BACE rule
```

---

# 16. 局部 Beta Posterior 在 Search-HotpotQA 中怎么更新

对于：

\[
u=(SEARCH,q)
\]

定义 weak prior：

\[
p_{g,z,u}
\sim
Beta(\alpha_0,\beta_0)
\]

或继续使用 BACE 当前统一的 instance-conditioned weak prior。

每次 natural occurrence：

```text
anchor z
action SEARCH(q)
...
final answer correct
```

则：

\[
\alpha_{g,z,u}\leftarrow\alpha_{g,z,u}+1.
\]

失败：

\[
\beta_{g,z,u}\leftarrow\beta_{g,z,u}+1.
\]

branch 重复同一 query 后同样更新。

因此 posterior 的统计单位是：

\[
\boxed{
\text{anchor cluster + exact executed query}
}
\]

不是：

\[
\text{query semantic cluster}.
\]

---

# 17. ERV 如何选择 Search Query

对每个候选 query：

\[
u_j=(SEARCH,q_j)
\]

计算：

\[
\Delta ERV(z,u_j).
\]

例如：

\[
p_{z,u_1}\sim Beta(2,2),
\]

\[
p_{z,u_2}\sim Beta(2,1),
\]

\[
p_{z,u_3}\sim Beta(1,2).
\]

即使：

\[
n(z,u_2)=1,
\]

仍可计算：

\[
\Delta ERV(z,u_2).
\]

若其 ERV 最大，则下一条 branch 直接执行：

```text
<search>q2</search>
```

再让冻结 \(\pi_{\mathrm{old}}\) 根据返回信息自由继续。

因此 Search-HotpotQA 中 BACE 的实验含义非常清楚：

\[
\boxed{
\text{“在相似的信息状态下，再验证一次已经自然提出过的检索决策。”}
}
\]

---

# 18. 一个完整的 HotpotQA 例子

问题：

```text
What nationality is the director of the film in which actor X appeared?
```

五条 natural roots 在某个 GiGPO similarity anchor \(z\) 汇合。

当前 information 大致都在讨论某部电影。

它们下一步自然动作：

```text
Root 1:
<search>The Godfather director</search>
final answer correct

Root 2:
<search>who directed The Godfather</search>
final answer wrong

Root 3:
<search>Francis Ford Coppola nationality</search>
final answer correct

Root 4:
<search>The Godfather director</search>
final answer wrong

Root 5:
<answer>American</answer>
final answer wrong
```

canonical action classes：

\[
u_1=
(SEARCH,\text{"The Godfather director"})
\]

count = 2；

\[
u_2=
(SEARCH,\text{"who directed The Godfather"})
\]

count = 1；

\[
u_3=
(SEARCH,\text{"Francis Ford Coppola nationality"})
\]

count = 1；

\[
u_4=
(ANSWER,\text{"American"})
\]

count = 1。

主 BACE branch set：

\[
\mathcal C^{branch}(z)
=
\{u_1,u_2,u_3\}.
\]

注意：

\[
u_1\neq u_2
\]

即使它们语义非常接近。

自然 evidence：

\[
u_1:[1,0],
\]

\[
u_2:[0],
\]

\[
u_3:[1].
\]

BACE 不会因为 \(u_2,u_3\) 是 singleton 就丢掉它们。

使用弱 prior 后计算：

\[
\Delta ERV(z,u_1),
\quad
\Delta ERV(z,u_2),
\quad
\Delta ERV(z,u_3).
\]

假设：

\[
\Delta ERV(z,u_3)
\]

最大。

从 anchor cluster 中均匀选一个 concrete origin，恢复其原始 search history，然后直接执行：

```text
<search>Francis Ford Coppola nationality</search>
```

取得相同 retrieval result 后，由 \(\pi_{\mathrm{old}}\) 重新生成剩余 reasoning / search / answer。

新的 terminal outcome 再更新：

\[
p_{z,u_3}.
\]

这就是一条完整 BACE search branch。

---

# 19. 为什么不按 Retrieval Result 来定义主 Action Identity

一个很诱人的方案是：

> 两个 query 如果返回相同 top-\(k\) documents，就认为是同一个动作。

例如：

```text
"The Godfather director"
```

和：

```text
"who directed The Godfather"
```

都返回同一组三篇文章，于是合并。

不建议作为主版本，原因有四个。

## 19.1 Action identity 被结果定义

同一个 query 在不同 retriever / index 版本下可能得到不同结果。

于是：

\[
\operatorname{can}(u)
\]

不再只由动作决定。

## 19.2 相似检索结果不代表相同决策意图

两个 query 可能偶然命中同一批文章，但关注点不同。

## 19.3 Top-k set equivalence 不稳定

需要继续定义：

- ordered list 是否必须相同？
- Jaccard threshold 是多少？
- passage-level 还是 document-level？
- score 是否考虑？

又引入额外超参数。

## 19.4 BACE 本身不需要靠它解决 singleton

因为 BACE 可以主动 resample exact query。

因此 retrieval-equivalence 适合：

\[
\boxed{
\text{diagnostic / ablation}
}
\]

而非主 action identity。

---

# 20. 为什么不按 Query Embedding / LLM Semantic Judge 聚类

同样不建议主版本采用：

\[
Sim_{\mathrm{emb}}(q_i,q_j)>\tau.
\]

主要问题：

1. similarity threshold 新增超参数；
2. 聚类结果依赖 encoder；
3. 语义等价不等于检索执行等价；
4. 可能错误合并不同实体或关系；
5. cluster membership 可随模型 / encoder 版本变化；
6. 给审稿人留下“性能来自额外 semantic model”的疑问；
7. 破坏 environment-grounded action identity 的简单性。

BACE 的论文主线应该是：

\[
\boxed{
\text{active credit experiment design}
}
\]

而不是：

\[
\text{natural-language action ontology learning}.
\]

---

# 21. 建议做的 Canonicalization 消融

虽然主版本使用 exact executed query，仍建议做一个小规模动作身份消融。

## A. Exact Executed Query — 主版本

\[
u=(SEARCH,q_{\mathrm{exec}})
\]

只使用环境 parser 的 normalization。

## B. Surface-Normalized Query

额外：

- Unicode normalize；
- lower-case；
- collapse internal whitespace；
- trim simple terminal punctuation。

用于测量轻量文本 normalization 是否明显减少 action fragmentation。

## C. Retrieval-Equivalent Query

如果 ordered top-\(k\) document IDs 完全相同则合并。

仅用于诊断。

## D. Semantic Query Cluster

embedding / LLM similarity。

只建议非常小规模验证，不作为必要实验。

如果 A 已经表现稳定，则没有理由升级到更复杂版本。

---

# 22. 必须先做的 Action-Identity 统计诊断

在正式大规模训练前，从若干 actor checkpoints 抽取 rollout groups，至少报告：

## 22.1 Parser 层

```text
valid SEARCH ratio
valid ANSWER ratio
INVALID ratio
multi-tag invalid ratio
empty-query ratio
```

## 22.2 Exact query identity

```text
# unique search queries per anchor
# repeated exact queries per anchor
query singleton ratio
max query occurrence count
action-count imbalance
```

## 22.3 Anchor × Action 可用性

```text
# repeated GiGPO anchors
# anchors with >=2 observed SEARCH actions
# effective BACE search anchors
effective-anchor / repeated-anchor ratio
```

## 22.4 语义 fragmentation 仅作诊断

统计 exact-different queries 中：

- retrieval top-k 完全相同的比例；
- top-k Jaccard 很高的比例；
- query embedding 很近的比例。

这能回答：

> exact action identity 是否把“实际上近似同一个检索行为”拆得过细？

但该统计不自动意味着应该合并。

---

# 23. 一个关键预期：Exact Query Singleton 并不是失败信号

对 ECPO：

\[
n_{z,u}=1
\]

意味着 action mean 很不可靠，因此需要 shrinkage。

对 BACE：

\[
n_{z,u}=1
\]

反而往往是高价值 acquisition candidate。

所以评估 Search-HotpotQA 适配性时，不应该使用：

\[
\boxed{
\text{“exact query repetition 必须很高”}
}
\]

作为硬条件。

真正需要的是：

\[
\boxed{
\text{同一 anchor 中存在多个 observed executable search decisions。}
}
\]

即：

\[
|\mathcal C_{\mathrm{search}}(z)|\ge2.
\]

BACE 的作用正是从这些稀疏自然 evidence 中决定：

> 哪个 singleton / low-count query 最值得主动补样本。

---

# 24. Search-HotpotQA 的训练 occurrence 怎么进入 GiGPO

本文档只改变 Search adapter 的 state/action interpretation，不改变 BACE 的整体训练原则。

branch 完成后：

### 计入真实 occurrences

- natural root 中所有真实生成的 search / answer actions；
- branch origin 被主动执行的 selected search action；
- branch suffix 新生成的 search / answer actions。

### 不计入

- 为恢复历史而 replay 的旧 search prefix；
- cache 中机械返回的 historical retrieval observations；
- 没有重新进行 actor decision 的 prefix。

因此继续满足 BACE 的：

\[
\boxed{
\text{replay prefix 不重复产生训练梯度。}
}
\]

---

# 25. 如果启用按动作聚合的 Local Advantage

未来若使用 action-aggregated anchor advantage：

\[
\overline G_{z,u}
=
\frac{1}{n_{z,u}}
\sum_{i\in\mathcal I(z,u)}G_i,
\]

Search-HotpotQA 中直接按本文 exact action key 分组：

\[
\mathcal I(z,u)
=
\left\{
i:
z_i=z,
\operatorname{CanonicalToolCall}(a_i)=u
\right\}.
\]

即：

```text
SEARCH["The Godfather director"]
```

的所有 occurrences 共享 local action credit。

而：

```text
SEARCH["who directed The Godfather"]
```

仍是另一个 action。

这与采集阶段的：

\[
p_{z,u}
\]

保持完全一致：

\[
\boxed{
\text{acquisition unit}
=
\text{credit aggregation unit}
=
\text{exact executable tool call}.
}
\]

---

# 26. 推荐的数据结构

```python
SearchOccurrence:
    task_id
    question
    trajectory_id
    step_index

    raw_anchor_obs
    anchor_cluster_id
    anchor_cluster_rep

    history_queries
    history_information

    raw_model_response
    projected_action
    action_valid

    action_type          # SEARCH / ANSWER / INVALID
    action_payload       # exact parsed query or answer
    canonical_action_key

    retrieval_doc_ids
    retrieval_text_hash

    terminal_reward
    terminal_success
```

canonical key：

```python
("SEARCH", exact_parsed_query)
("ANSWER", exact_parsed_answer)
("INVALID",)
```

---

# 27. 推荐的 Canonicalizer 伪代码

```python
def canonicalize_search_action(raw_response):
    projected, valid = search_projection([raw_response])
    projected = projected[0]
    valid = bool(valid[0])

    if not valid:
        return ("INVALID",)

    if projected.startswith("<search>"):
        q = extract_search_payload(projected)
        # Do not apply semantic normalization.
        # q is the exact post-projection / executable query.
        return ("SEARCH", q)

    if projected.startswith("<answer>"):
        y = extract_answer_payload(projected)
        return ("ANSWER", y)

    return ("INVALID",)
```

branchable：

```python
def is_branchable(action_key):
    return action_key[0] == "SEARCH"
```

---

# 28. 推荐的 Search-BACE Anchor 构造伪代码

```text
Input:
    natural roots for one HotpotQA question
    GiGPO similarity threshold = 0.9

1. Use official GiGPO Search anchor grouping:
       cluster current retrieval observations with similarity >= 0.9

2. For each anchor cluster z:
       collect all concrete natural occurrences I_root(z)

3. Canonicalize every natural action with SearchEnv parser

4. Build:
       C_search(z) = unique valid observed SEARCH action keys

5. Structural validity:
       repeated anchor
       non-initial
       non-terminal
       remaining search turn > 0
       |C_search(z)| >= 2
       replay metadata complete

6. Initialize Beta posterior for each (z, SEARCH(query))

7. Compute ERV

8. Apply normal BACE capacity correction / branch allocation
```

---

# 29. 推荐的 Search Branch 伪代码

```text
Given selected (z*, SEARCH(q*)):

1. Sample a concrete eligible origin i* from I_root(z*)
   uniformly / origin-balanced.

2. Reset the same HotpotQA task.

3. Restore the exact historical interaction:
       question
       past SEARCH actions
       cached retrieval results

4. Verify:
       same task
       same step
       same history
       same concrete pre-action observation hash
       still belongs to GiGPO anchor z*

5. Directly execute exact q*.

6. Return the cached / deterministic local-retriever result for q*.

7. Let frozen pi_old generate the remaining suffix.

8. Observe final binary correctness Y.

9. Update Beta posterior for (z*, q*).

10. Recompute ERV before allocating the next branch.

11. Only the selected origin action and newly generated suffix
    enter branch training; restored prefix does not.
```

---

# 30. Search-HotpotQA 与 ALFWorld 的统一解释

ALFWorld：

\[
\text{raw LLM action}
\rightarrow
\text{environment parser}
\rightarrow
\text{admissible executable command}.
\]

Search-HotpotQA：

\[
\text{raw LLM action}
\rightarrow
\text{search projection}
\rightarrow
\text{structured tool call}.
\]

因此 BACE 真正需要的适用条件不应写成：

> 环境必须拥有有限离散 action space。

而应写成：

\[
\boxed{
\text{环境必须提供稳定、确定性的 executable-action identity。}
}
\]

这个 identity 可以是：

- ALFWorld command；
- WebShop page action；
- ScienceWorld valid command；
- Search-HotpotQA parameterized tool call：

\[
(\mathrm{SEARCH},q).
\]

所以 BACE 可以支持**开放参数化动作空间**，只要 action parser 是确定性的。

---

# 31. 主要风险

## 31.1 GiGPO Search anchor 本身是近似状态

0.9 SequenceMatcher clustering 可能把不同 history / information states 合并。

这不是 action identity 的问题，而是 GiGPO Search state abstraction 的继承限制。

主版本为公平性继续对齐 GiGPO。

建议记录：

```text
within-anchor history diversity
retrieval-doc overlap
anchor-cluster diameter
```

并可做 history-aware anchor 消融。

---

## 31.2 Query space 可能高度碎片化

如果每个 anchor 暴露很多 singleton queries：

- Beta evidence 初始很弱；
- \(K_{\max}\) truncation 更频繁。

但这正是 ERV active replication 的适用场景。

真正的问题不是 singleton，而是：

\[
|\mathcal C_{\mathrm{search}}(z)|<2
\]

导致没有可比较的自然局部动作。

---

## 31.3 Retriever nondeterminism 会污染 posterior

如果相同 query 在 replay 时返回不同 documents，branch outcome variation 同时包含：

\[
\text{query effect}
+
\text{retriever variation}
+
\text{policy continuation variation}.
\]

因此必须优先 local fixed retriever + cache。

---

## 31.4 语义近义 query 被拆开

这是 exact identity 的代价。

但相对于错误合并动作，宁可保守拆分。

因为 BACE 可以主动补 evidence，而错误合并会直接污染：

\[
p_{z,u}.
\]

---

# 32. 必须记录的实验诊断指标

建议正式实验至少记录：

### Anchor

```text
unique search anchors
repeated search anchors
effective BACE search anchors
anchor group-size distribution
similarity-cluster diameter
```

### Action identity

```text
unique exact SEARCH actions per anchor
SEARCH singleton ratio
repeated exact-query ratio
ANSWER ratio
INVALID ratio
candidate truncation ratio
```

### Acquisition

```text
selected query ERV
query evidence count before branch
query evidence count after branch
same-query repeated branch frequency
origin diversity per (z,u)
realized regret reduction
```

### Replay

```text
history replay success
retrieval output exact-match rate
cache hit rate
replay latency
retriever API failure rate
```

### Outcome

```text
final HotpotQA success
action-ranking accuracy
posterior calibration
branch-vs-root marginal gain
```

---

# 33. 推荐消融

主论文不需要把所有消融都做大规模，最关键的是：

## 33.1 Action identity

\[
\text{Exact Executed Query}
\]

vs.

\[
\text{Surface Normalized Query}.
\]

Retrieval-equivalent / semantic grouping只做小规模诊断即可。

## 33.2 Anchor identity

\[
\text{GiGPO Similarity 0.9}
\]

vs.

\[
\text{Exact Observation}.
\]

如果资源允许再加入：

\[
\text{History-aware Search State}.
\]

## 33.3 Branchable action

主版本：

\[
SEARCH\text{-only}
\]

扩展：

\[
SEARCH + deterministic\ ANSWER\ arm.
\]

## 33.4 Retriever

优先验证：

\[
E5\text{-Flat + Cache}
\]

vs.

\[
E5\text{-Flat without Cache}.
\]

用于确认 replay determinism 的重要性。

---

# 34. 推荐先做的小规模可行性实验

在完整训练前，先从 3–5 个 actor checkpoints 收集自然 Search-HotpotQA rollouts。

每个 checkpoint 统计：

1. GiGPO similarity anchors 数；
2. repeated anchors 数；
3. 每个 anchor 的 distinct exact search-query 数；
4. 满足：
   \[
   |\mathcal C_{\mathrm{search}}(z)|\ge2
   \]
   的比例；
5. singleton query 比例；
6. query occurrence count 分布；
7. exact query vs surface-normalized query 的 action-class 数量差；
8. exact query vs retrieval-equivalent query 的 action-class 数量差；
9. replay exact reconstruction success rate；
10. 同一 exact query 多次 continuation 的 outcome variance。

最关键的判据不是 exact query 重复率，而是：

\[
\boxed{
\text{是否有足够多 repeated anchors 暴露至少两个不同的 observed SEARCH actions。}
}
\]

如果这个比例合理，Search-HotpotQA 就非常适合 BACE。

---

# 35. 最终固定建议

Search-HotpotQA 主版本建议正式冻结为：

\[
\boxed{
\begin{aligned}
&\textbf{State: } &&
\text{沿用 GiGPO Search 的 observation similarity anchor，}\tau=0.9;\\[2mm]
&\textbf{Action type: } &&
SEARCH,\ ANSWER,\ INVALID;\\[2mm]
&\textbf{Search action identity: } &&
(\mathrm{SEARCH},q_{\mathrm{exec}});\\[2mm]
&\textbf{Query normalization: } &&
\text{只使用环境 parser 已经执行的确定性规范化};\\[2mm]
&\textbf{Candidate actions: } &&
\text{natural roots 中 observed valid SEARCH actions only};\\[2mm]
&\textbf{Effective anchor: } &&
|\mathcal C_{\mathrm{search}}(z)|\ge2;\\[2mm]
&\textbf{ANSWER: } &&
\text{参与训练，不作为主动 branch action};\\[2mm]
&\textbf{INVALID: } &&
\text{不进入 branch candidate set};\\[2mm]
&\textbf{Origin: } &&
\text{在 anchor 的 eligible concrete occurrences 中均衡抽样};\\[2mm]
&\textbf{Replay: } &&
\text{固定 local retriever + query-result cache};\\[2mm]
&\textbf{Posterior: } &&
p_{g,z,q}=P(Y=1\mid z,do(SEARCH(q)),\pi_{\mathrm{old}});\\[2mm]
&\textbf{Acquisition: } &&
\text{按原 BACE sequential ERV 规则分配 branch};\\[2mm]
&\textbf{Training: } &&
\text{只训练真实重新决策的 origin + suffix，不训练 replay prefix}.
\end{aligned}
}
\]

一句话概括：

> **在 Search-HotpotQA 中，我们不试图判断两个自然语言 query 是否“语义相同”，而是把环境实际执行的 structured search call 作为 canonical action；BACE 再通过主动重复 observed exact queries 来解决小样本 action credit，而不是通过语义聚类提前抹平 query 差异。**

这个方案对当前 BACE 的改动非常小，却把方法适用范围从有限离散 action space 扩展到了：

\[
\boxed{
\text{开放但具有 deterministic tool-call identity 的参数化动作空间。}
}
\]

---

# 36. 与当前 BACE 主方案的接口

本文只定义 Search-HotpotQA 的 environment adapter，不改变以下 BACE 主机制：

1. task-family / instance competence posterior；
2. root–branch topology controller；
3. root-side capacity correction；
4. one-layer branch；
5. observed-action local posterior；
6. Bayes simple regret；
7. sequential ERV；
8. frozen \(\pi_{\mathrm{old}}\) acquisition batch；
9. replay prefix 不重复训练；
10. 最终 GiGPO-style / action-consistent credit。

因此实现时应把本文内容封装为：

```text
SearchStateAdapter
SearchActionCanonicalizer
SearchReplayManager
```

而不是修改 BACE 的 ERV 核心公式。

---

# 37. 参考资料与代码依据

## BACE 当前方案

- `BACE_GiGPO_整体方案大纲.md`
- `BACE_GiGPO_组件之Anchor_Selection_Spec.md`
- `BACE_GiGPO_组件之Root_Branch_Allocation_Design.md`
- `BACE_GiGPO_组件之Unified_Leaf_Normalization_and_Action_Aggregated_Anchor_Credit_Comparison.md`

## GiGPO / verl-agent

- Feng et al., **Group-in-Group Policy Optimization for LLM Agent Training**, NeurIPS 2025.  
  https://arxiv.org/abs/2505.10978

- Official verl-agent repository:  
  https://github.com/langfengQ/verl-agent

- Search action projection:  
  `agent_system/environments/env_package/search/projection.py`

- Search environment:  
  `agent_system/environments/env_package/search/third_party/skyrl_gym/envs/search/env.py`

- Search environment manager:  
  `agent_system/environments/env_manager.py`

- GiGPO similarity grouping:  
  `gigpo/core_gigpo.py`

- Search GiGPO training config:  
  `examples/gigpo_trainer/run_search.sh`

## ECPO

- Li et al., **When Denser Credit Is Not Enough: Evidence-Calibrated Policy Optimization for Long-Horizon LLM Agent Training**, 2026.  
  https://arxiv.org/abs/2606.05885

关键证据：

- ECPO 对 canonical action 做 environment-grounded deterministic grouping；
- ECPO 在 NQ + HotpotQA 上训练 Search-Augmented QA；
- Search-QA 结果证明 action-level evidence calibration 可以扩展到开放检索决策；
- 但论文没有公开足够细的 Search-specific query canonicalizer 实现，因此本文不把未公开细节当成既定事实。

## Search-R1

- Jin et al., **Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning**, 2025.
- Official repository:  
  https://github.com/PeterGriffinJin/Search-R1

推荐的本地检索配置：

- fixed Wikipedia corpus；
- E5 flat index；
- top-k retrieval；
- local retrieval server。

---

# 38. 最终研究判断

Search-HotpotQA **适合 BACE**，而且它实际上可以成为一个很有价值的 benchmark，因为它证明：

\[
\boxed{
\text{BACE 不要求有限枚举的环境动作。}
}
\]

BACE 真正要求的是：

\[
\boxed{
\text{模型输出能被环境确定性解析为具有稳定身份的 executable action。}
}
\]

在 Search-HotpotQA 中，该动作就是：

\[
\boxed{
(\mathrm{SEARCH},q_{\mathrm{exec}}).
}
\]

这使我们能够在不引入额外语义模型的情况下，完整定义：

\[
(z,u)
\rightarrow
Beta\ posterior
\rightarrow
Bayes\ regret
\rightarrow
ERV
\rightarrow
targeted\ branch.
\]

因此建议将 Search-HotpotQA 正式纳入 BACE 的候选主 benchmark，并先完成第 34 节的小规模 action-identity / anchor-availability audit，再决定是否进入完整端到端训练。
