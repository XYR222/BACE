# BACE-GiGPO：C0.5 / C4 / C7 / C8 Tree-Credit 实现规范

> **版本**：2026-09-06  
> **适用代码基线**：当前 BACE `tree-version` / C0 production 语义  
> **目的**：在现有 C0 / C1 / C2 / C3 实验结果基础上，定义四个新的、语义互相区分且可直接实现的 optimizer 变体：C0.5、C4、C7、C8。  
> **原则**：四个版本都不改变 BACE 的 rollout acquisition、root/branch allocation、ERV、replay 验证、posterior 更新、PPO ratio、optimizer 与 scheduler；它们只改变 **已采集 rollout tree 如何形成 GiGPO advantage 与训练 occurrences**。

---

# 0. 结论先行

四个版本分别回答四个不同问题：

| 版本 | 核心问题 | Selected origin | Strict ancestor macro | Strict ancestor local | Replay prefix 是否复制训练 |
|---|---|---|---|---|---|
| **C0.5** | 同一 selected decision 的 continuation-specific local credit 是否需要保留？ | 保留 C0 occurrences，但同 replay-family 的 **local advantage 后平均** | 否 | 否 | 否 |
| **C4** | 在保留 C0 occurrence semantics 后，trajectory-level tree backup 是否有益？ | 完全 C0 | **是** | 否 | 否 |
| **C8** | 在 C4 上，strict-prefix local return backup 是否还有收益？ | 完全 C0 | **是** | **是** | 否 |
| **C7** | 是否根本不需要特殊 tree credit，而应把每个 terminal leaf 展平成完整 rollout？ | 作为完整 leaf 的一部分重复出现 | vanilla GiGPO | vanilla GiGPO | **是** |

四个版本的核心关系：

\[
\boxed{
C0.5 = C0 + \text{origin-family local-advantage smoothing}
}
\]

\[
\boxed{
C4 = C0 + \text{strict-ancestor macro incremental backup}
}
\]

\[
\boxed{
C8 = C4 + \text{strict-ancestor local-}G\text{ incremental backup}
}
\]

\[
\boxed{
C7 = \text{flatten every terminal leaf into a full pseudo-rollout}
      + \text{vanilla production GiGPO}
}
\]

其中 C7 是一个不同哲学的极端版本，不是 C4/C8 的递进版本。

---

# 1. 必须先固定的 C0 语义

本文所有“保持 C0”均指当前 production C0 的实际行为。

## 1.1 三类主要 physical occurrences

### Natural root occurrence

自然 root 中由冻结 actor 真实生成的动作 occurrence。

正常参与：

- macro advantage；
- local / step advantage；
- PPO loss。

---

### Copied branch-origin occurrence

BACE 选择 natural root 上某个 concrete origin 后：

1. replay 到 origin 之前的状态；
2. 复制执行 natural origin 的同一个动作；
3. 从该动作之后生成新的 branch continuation。

记 natural origin 为：

\[
o,
\]

对应 copied branch-origin 为：

\[
b.
\]

严格 replay 要求：

\[
state_b = state_o,
\]

\[
action_b = action_o,
\]

并且：

\[
validity_b = validity_o.
\]

当前 C0 将 copied branch-origin 作为一个独立 physical occurrence 参与：

- local anchor group；
- macro；
- PPO。

因此 C0 保留了 selected decision 的 occurrence / gradient multiplicity。

---

### Fresh branch suffix occurrence

branch origin 之后真正新生成的动作。

它们是新的 policy decisions，始终正常参与 GiGPO 与 PPO。

---

## 1.2 Mechanical replay prefix

origin 之前只是为了恢复环境而 replay 的历史 prefix：

```text
e0 -> e1 -> ... -> e_{o-1}
```

在 C0 / C0.5 / C4 / C8 中：

\[
\boxed{
\text{不新增 physical training occurrence}
}
\]

即：

- 不重新产生 PPO row；
- 不重新产生 old logprob；
- 不因为 branch 数增加而复制训练。

C7 是唯一故意取消这一规则的版本。

---

## 1.3 C0 advantage

对 physical occurrence \(i\)：

\[
A_i^{C0}
=
A_i^{E,C0}
+
\omega A_i^{S,C0}.
\]

其中：

- \(A_i^{E,C0}\)：当前 production GiGPO-compatible stable macro；
- \(A_i^{S,C0}\)：当前 GiGPO anchor-level local advantage；
- \(\omega=\texttt{step\_advantage\_w}\)。

本文不改变 C0 的 invalid penalty、normalization、old logprob、ratio 或 loss 定义。

---

# 2. 统一符号

对一个 natural root：

```text
e0 -> e1 -> ... -> o -> ... -> root terminal
                    \
                     b -> branch suffix -> branch terminal
```

定义：

- \(e\)：当前讨论的 natural root edge；
- \(o_b\)：branch \(b\) 对应的 natural origin；
- \(b\)：copied branch-origin physical occurrence；
- \(t_e\)：natural edge \(e\) 的 step index；
- \(t_{o_b}\)：branch origin 的 natural step index；
- \(M_i\equiv A_i^{E,C0}\)：C0 frozen macro scalar；
- \(G_i\)：当前 production invalid penalty 已经按 C0 语义施加后的 step return，且尚未进行 local anchor normalization；
- \(A_i^{S,C0}\)：用 \(G_i\) 在 C0 anchor group 中归一化得到的 local advantage。

---

## 2.1 Strict ancestor

branch \(b\) 的 strict natural ancestor set：

\[
Anc^{<}(b)
=
\left\{
e:
root(e)=root(o_b),
\quad
t_e<t_{o_b}
\right\}.
\]

注意是：

\[
\boxed{t_e<t_{o_b}}
\]

而不是：

\[
t_e\le t_{o_b}.
\]

因此 selected origin 自己：

\[
o_b
\]

**不属于 strict ancestor**。

这是 C4 / C8 保留 C0 origin semantics 的关键。

---

## 2.2 某 edge 的 descendant branches

定义：

\[
B^{<}(e)
=
\left\{
b:
e\in Anc^{<}(b)
\right\}.
\]

只有 natural root backbone 上、位于 \(e\) 之后的 branch origins 才属于：

\[
B^{<}(e).
\]

相同 anchor key、相同 action string，但位于其他 root 的 occurrence 不属于 descendant。

---

# 3. Invalid penalty 的统一约束

当前 production 中 explicit invalid penalty 是 edge-local correction，而不是普通 reward shaping。

因此：

\[
G_o
=
G_o^{raw}-p_o,
\]

\[
G_b
=
G_b^{raw}-p_o,
\]

因为 natural origin 与 copied origin 是同一个 state-action decision，其 invalid penalty 相同。

所以：

\[
\boxed{
G_b-G_o
=
G_b^{raw}-G_o^{raw}
}
\]

origin-local invalid penalty自动抵消。

macro 同理。当前 frozen C0 macro 在同一 task 内使用共同的 affine stable normalizer，因此：

\[
\boxed{
M_b-M_o
}
\]

保留 branch-vs-natural continuation 差异，并消去二者共同的 origin-local correction。

这正是 C4/C8 使用 incremental backup 的基础。

---

# 4. C0.5：Origin-Family Local Advantage Smoothing

## 4.1 研究问题

C0 比 C1 更好可能有两个原因：

1. copied branch-origin 的 occurrence / gradient multiplicity 有价值；
2. 同一个 selected decision 的不同 continuations 各自拥有不同 local advantage 也有价值。

C1 同时删除了这两者，因此无法拆开。

C0.5 专门测试：

\[
\boxed{
\text{保留 C0 multiplicity，
但消除同一 replay-family 内 continuation-specific local disagreement}
}
\]

---

## 4.2 Replay family 的定义

对一个被 branch 的 concrete natural origin \(o\)，定义：

\[
\boxed{
\mathcal F(o)
=
\{o\}
\cup
\{b:\operatorname{natural\_origin}(b)=o\}.
}
\]

例如同一个 natural origin 被 branch 两次：

```text
o
├── branch-origin b1
└── branch-origin b2
```

则：

\[
\mathcal F(o)=\{o,b_1,b_2\}.
\]

### 非常重要

C0.5 的 family key 是：

\[
\boxed{
\text{concrete natural origin occurrence id}
}
\]

不是：

\[
(z,a).
\]

如果另一条 natural root 在相同 anchor 也执行相同 action：

```text
o2: same z, same a
```

但：

\[
o_2\neq o,
\]

则 \(o_2\) 不进入 \(\mathcal F(o)\)。

因此 C0.5 **不是全局 action aggregation**。

---

## 4.3 C0.5 主版本：先算 C0 local advantage，再 family-average

第一步完全运行 C0：

\[
A_i^{S,C0}
=
GiGPONorm_z(G_i).
\]

然后只对 replay family 做：

\[
\boxed{
\bar A_{\mathcal F(o)}^{S}
=
\frac{
1
}{
|\mathcal F(o)|
}
\sum_{j\in\mathcal F(o)}
A_j^{S,C0}.
}
\]

对 family 内所有 physical occurrences：

\[
\boxed{
A_j^{S,C0.5}
=
\bar A_{\mathcal F(o)}^{S},
\qquad
j\in\mathcal F(o).
}
\]

其他 occurrence：

\[
\boxed{
A_j^{S,C0.5}
=
A_j^{S,C0}.
}
\]

macro 完全不改：

\[
\boxed{
A_j^{E,C0.5}
=
A_j^{E,C0}.
}
\]

最终：

\[
\boxed{
A_j^{C0.5}
=
A_j^{E,C0}
+
\omega A_j^{S,C0.5}.
}
\]

---

## 4.4 为什么主版本选择“advantage 后平均”

另一个可能方案是：

1. 先平均 family 的 \(G\)；
2. 把平均 \(G\) 广播回 family；
3. 再重新计算整个 anchor 的 local normalization。

但这样会改变 anchor 的：

\[
\sigma_z
\]

甚至改变所有不属于该 replay family 的 occurrence local advantage。

因此它同时测试：

- replay-family smoothing；
- anchor variance compression；
- local scale rescaling。

不够干净。

C0.5 主版本采用：

\[
\boxed{
\text{C0 normalization}
\rightarrow
\text{family local-advantage average}
}
\]

这样：

- C0 anchor baseline / scale保持原样；
- 非 family occurrences 完全不变；
- occurrence 数不变；
- PPO support 不变；
- macro 不变。

如果以后需要，可额外实现：

```text
c0_5_g_before_norm
```

作为 diagnostic，但它不是本文 C0.5 主版本。

---

## 4.5 C0.5 的一个例子

anchor \(z\) 中有：

```text
o : action a1, natural selected origin
b : action a1, copied branch-origin
x : action a2, other natural occurrence
```

C0 local normalization 后假设：

\[
A_o^{S,C0}=-1.0,
\]

\[
A_b^{S,C0}=+1.0,
\]

\[
A_x^{S,C0}=0.
\]

则：

\[
\mathcal F(o)=\{o,b\},
\]

\[
\bar A_{\mathcal F(o)}^S
=
\frac{-1+1}{2}=0.
\]

C0.5：

\[
A_o^{S,C0.5}=0,
\]

\[
A_b^{S,C0.5}=0,
\]

\[
A_x^{S,C0.5}=0.
\]

但 macro 仍各自保留：

```text
o : root leaf macro
b : branch leaf macro
x : its own root macro
```

所以 root/branch outcome 差异不会被抹掉，只是不再把：

> “同一个 selected action 因后续 continuation 不同而一个得到正 local credit、一个得到负 local credit”

解释成当前动作本身的差异。

---

## 4.6 C0.5 的重要性质

对 family：

\[
\mathcal F(o)
\]

有：

\[
\sum_{j\in\mathcal F(o)}
A_j^{S,C0.5}
=
\sum_{j\in\mathcal F(o)}
A_j^{S,C0}.
\]

所以 replay-family 的 **signed local advantage mass** 保持不变。

C0.5 改变的是 family 内部 credit disagreement，而不是删掉 occurrence multiplicity。

---

## 4.7 C0.5 必须满足的 invariants

- [ ] physical occurrence count 与 C0 完全一致；
- [ ] copied branch-origin 仍进入 PPO；
- [ ] macro 每一行与 C0 完全一致；
- [ ] non-family local advantage 与 C0 完全一致；
- [ ] family 内所有 local advantage 完全相同；
- [ ] family local-advantage sum 保持不变；
- [ ] 不按 `(anchor, action)` 跨 natural origins 聚合；
- [ ] branch suffix 不自动进入 origin family。

---

# 5. C4：C0 + Strict-Ancestor Macro Backup

## 5.1 研究问题

C2 比 C1 更好，说明 prefix macro backup 可能有益。

但 C2 建立在 C1 的 origin-collapse 上，无法判断：

> macro backup 本身有益，

还是：

> macro backup 只是补偿 C1 删除 copied-origin training 后的损失。

C4 因此定义为：

\[
\boxed{
C4
=
C0
+
\text{strict-ancestor macro incremental backup}
}
\]

即保留 C0 的所有 occurrence 与 local credit，只增加一项 trajectory-level tree backup。

---

## 5.2 Selected origin 完全保持 C0

对于 natural origin \(o\)：

- natural \(o\) 保留自己的 \(M_o,G_o,A_o^S\)；
- copied branch-origin \(b\) 保留自己的 \(M_b,G_b,A_b^S\)；
- 两者都是 C0 physical occurrences；
- 两者都参与 local anchor group；
- 两者都进入 PPO；
- 不做 C1 的 \(G\) averaging；
- 不做 origin collapse。

因此：

\[
\boxed{
C4\text{ 不修改 selected origin。}
}
\]

---

## 5.3 Macro branch increment

对每条 branch \(b\)，natural origin 为 \(o_b\)。

定义：

\[
\boxed{
\Delta M_b
=
M_b-M_{o_b}.
}
\]

其中：

\[
M_i=A_i^{E,C0}
\]

是当前 frozen C0 macro scalar。

由于 branch-origin 与 natural origin 是同一个 action，二者共同的 origin-local invalid penalty在差值中抵消。

---

## 5.4 Strict ancestor 的 branch candidate

对于：

\[
e\in Anc^{<}(b),
\]

构造：

\[
\boxed{
M_{e\leftarrow b}^{cand}
=
M_e+\Delta M_b.
}
\]

注意 macro 不使用 \(\gamma\)。

原因：

\[
M
\]

是 trajectory-level credit，不是 step return-to-go。

---

## 5.5 多 branch 的 macro 合并

对 strict ancestor \(e\)，若：

\[
B^{<}(e)=\{b_1,\ldots,b_K\},
\]

则：

\[
\boxed{
M_e^{C4}
=
\frac{
M_e
+
\sum_{k=1}^K
M_{e\leftarrow b_k}^{cand}
}{
1+K
}.
}
\]

等价于：

\[
\boxed{
M_e^{C4}
=
M_e
+
\frac{
\sum_{k=1}^K\Delta M_{b_k}
}{
1+K
}.
}
\]

分母必须是：

\[
\boxed{1+K}
\]

因为 original root continuation 本身也是一个 candidate。

---

## 5.6 非 strict ancestor 全部保持 C0

如果 occurrence \(i\)：

- 是 selected natural origin；
- 是 copied branch-origin；
- 是 branch suffix；
- 是 branch origin 之后的 natural suffix；
- 没有 descendant branch；
- 属于另一条 root；

则：

\[
\boxed{
A_i^{E,C4}
=
A_i^{E,C0}.
}
\]

C4 local 对所有 occurrences 完全保持：

\[
\boxed{
A_i^{S,C4}
=
A_i^{S,C0}.
}
\]

最终：

\[
\boxed{
A_i^{C4}
=
A_i^{E,C4}
+
\omega A_i^{S,C0}.
}
\]

---

## 5.7 C4 的一个例子

natural root：

```text
e0 -> e1 -> o -> root terminal
            \
             b -> branch terminal
```

假设 frozen C0 macro：

\[
M_{e0}=0.40,
\]

\[
M_{e1}=0.50,
\]

\[
M_o=0.60,
\]

\[
M_b=-0.40.
\]

则：

\[
\Delta M_b
=
-0.40-0.60
=
-1.00.
\]

对 \(e_1\)：

\[
M_{e1\leftarrow b}^{cand}
=
0.50-1.00
=
-0.50.
\]

所以：

\[
M_{e1}^{C4}
=
\frac{0.50+(-0.50)}{2}
=
0.
\]

对 \(e_0\)：

\[
M_{e0\leftarrow b}^{cand}
=
0.40-1.00
=
-0.60,
\]

\[
M_{e0}^{C4}
=
\frac{0.40-0.60}{2}
=
-0.10.
\]

但是 selected origin：

\[
M_o^{C4}=0.60,
\]

copied branch-origin：

\[
M_b^{C4}=-0.40.
\]

它们仍是两个 C0 occurrences。

local advantage全部保持 C0。

---

## 5.8 C4 的实现重点

C4 可以复用当前 tree-version 的：

- natural-origin mapping；
- root id；
- step index；
- descendant branch indexing；
- C2 的 macro incremental formula。

但必须删除 C2 继承自 C1 的部分：

- **不能删除 copied branch-origin PPO rows**；
- **不能 unique-edge collapse**；
- **不能 direct-\(G\) mean**；
- **不能重新构造 unique support local group**。

最简单的实现思路：

```text
1. 完整计算 C0
2. 保留 C0 physical batch
3. 构建 tree index
4. 只 override strict-ancestor macro
5. local = C0 local
6. final = macro_C4 + omega * local_C0
```

---

## 5.9 C4 invariants

- [ ] training rows 与 C0 完全一致；
- [ ] local advantage逐 row 与 C0完全一致；
- [ ] selected origin macro与 C0一致；
- [ ] copied branch-origin macro与 C0一致；
- [ ] branch suffix macro与 C0一致；
- [ ] 只有 strict natural ancestors 的 macro可改变；
- [ ] macro backup 不乘 \(\gamma\)；
- [ ] 多 branch denominator = `1 + num_descendant_branches`；
- [ ] 没有 descendant branch 时严格退化为 C0。

---

# 6. C8：C0 Origin Semantics + Macro & Local Strict-Ancestor Backup

## 6.1 研究问题

已有：

\[
C3<C2
\]

说明 ancestor local-\(G\) backup 在 C1 origin-collapse 语义下表现较差。

但这不能完全回答：

> local backup 本身有害，

还是：

> local backup 与 C1 origin collapse 组合后有害。

C8 因此定义为：

\[
\boxed{
C8
=
C0\text{ origin semantics}
+
C4\text{ macro backup}
+
\text{strict-ancestor local-}G\text{ backup}
}
\]

---

## 6.2 Selected origin 仍完全保持 C0

selected natural origin \(o\) 与 copied branch-origin \(b\)：

- 都保留为 C0 physical occurrences；
- 不做 \(G_o,G_b\) direct mean；
- 不 collapse；
- 都参与 C0-support anchor normalization；
- 都进入 PPO。

因此：

\[
\boxed{
\text{C8 的 origin multiplicity 与 C0完全一致。}
}
\]

---

## 6.3 Macro 与 C4 完全一致

\[
\boxed{
A_i^{E,C8}
=
A_i^{E,C4}.
}
\]

即：

\[
\Delta M_b=M_b-M_{o_b},
\]

strict ancestors 使用：

\[
M_{e\leftarrow b}^{cand}
=
M_e+\Delta M_b,
\]

然后：

\[
M_e^{C8}
=
\frac{
M_e+\sum_bM_{e\leftarrow b}^{cand}
}{
1+|B^{<}(e)|
}.
\]

C8 不增加新的 macro 规则。

---

## 6.4 Local branch increment

对 branch \(b\)，定义：

\[
\boxed{
\Delta G_b
=
G_b-G_{o_b}.
}
\]

由于 origin 与 copied origin 使用同一个 state-action：

\[
p_b=p_{o_b},
\]

所以共同的 origin-local invalid penalty自动抵消。

---

## 6.5 Strict ancestor 的 counterfactual local candidate

对于：

\[
e\in Anc^{<}(b),
\]

定义距离：

\[
d(e,o_b)
=
t_{o_b}-t_e.
\]

构造：

\[
\boxed{
G_{e\leftarrow b}^{cand}
=
G_e
+
\gamma^{d(e,o_b)}
\Delta G_b.
}
\]

该式表示：

> 保留 target edge \(e\) 自身及其到 origin 之前的 natural prefix，只把 origin 之后的 natural continuation 替换成 branch continuation。

---

## 6.6 多 descendant branches 的 local 合并

若：

\[
B^{<}(e)=\{b_1,\ldots,b_K\},
\]

则：

\[
\boxed{
\widetilde G_e^{C8}
=
\frac{
G_e
+
\sum_{k=1}^K
G_{e\leftarrow b_k}^{cand}
}{
1+K
}.
}
\]

等价：

\[
\boxed{
\widetilde G_e^{C8}
=
G_e
+
\frac{
\sum_{k=1}^K
\gamma^{d(e,o_{b_k})}
\Delta G_{b_k}
}{
1+K
}.
}
\]

selected origin不在 strict ancestor set，所以不执行这个平均。

---

## 6.7 非 strict ancestors 的 \(G\) 不 override

如果 occurrence \(i\) 不是 strict natural ancestor：

\[
\boxed{
\widetilde G_i^{C8}=G_i.
}
\]

特别是：

\[
\widetilde G_o^{C8}=G_o,
\]

\[
\widetilde G_b^{C8}=G_b.
\]

---

## 6.8 然后必须重新跑 GiGPO local normalization

这是 C8 最容易实现错的地方。

C8 不是：

> 直接给 strict ancestor 的最终 local advantage加一个 correction。

正确流程：

\[
\boxed{
G^{C0}
\rightarrow
\text{strict-ancestor }G\text{ override}
\rightarrow
\text{在 C0 physical occurrence support 上重新 GiGPO local normalization}
}
\]

即对每个 anchor \(z\)：

\[
A_i^{S,C8}
=
GiGPONorm_z
(
\widetilde G_i^{C8}
).
\]

anchor group 的成员集合与 C0 完全相同：

- natural root occurrences；
- copied branch-origin occurrences；
- fresh branch suffix occurrences；

都保留。

只是某些 strict natural ancestor rows 的 \(G\) 被 override。

---

## 6.9 一个重要的间接影响

由于 local normalization 是 group-relative：

\[
\mu_z,\sigma_z
\]

会随某个 strict ancestor 的 \(G\) 改变。

因此：

> 即使 selected origin、branch-origin 或其他 occurrence 的 \(G\) 没有显式 override，只要它们与某个被 override 的 strict ancestor 位于同一个 anchor group，它们的最终 \(A^S\) 也可能间接变化。

这属于 C8 正确的 GiGPO group effect，不是 bug。

所以不能写成：

\[
A_o^{S,C8}=A_o^{S,C0}
\]

作为绝对 invariant。

正确说法是：

\[
\boxed{
\text{origin row 不做显式 }G\text{ override；local advantage 是否变化由其 anchor group statistics决定。}
}
\]

---

## 6.10 C8 的一个例子

沿用：

```text
e0 -> e1 -> o -> root
            \
             b -> branch
```

设：

\[
\gamma=0.95,
\]

C0 penalized step returns：

\[
G_{e0}=0.90,
\]

\[
G_{e1}=0.80,
\]

\[
G_o=0.60,
\]

\[
G_b=0.20.
\]

则：

\[
\Delta G_b
=
0.20-0.60
=
-0.40.
\]

### 对 \(e_1\)

距离：

\[
d(e_1,o)=1.
\]

branch candidate：

\[
G_{e1\leftarrow b}^{cand}
=
0.80
+
0.95(-0.40)
=
0.42.
\]

所以：

\[
\widetilde G_{e1}^{C8}
=
\frac{0.80+0.42}{2}
=
0.61.
\]

### 对 \(e_0\)

距离：

\[
d(e_0,o)=2.
\]

\[
G_{e0\leftarrow b}^{cand}
=
0.90
+
0.95^2(-0.40)
=
0.539.
\]

所以：

\[
\widetilde G_{e0}^{C8}
=
\frac{0.90+0.539}{2}
=
0.7195.
\]

### 对 selected origin

不做平均：

\[
\widetilde G_o^{C8}=0.60,
\]

\[
\widetilde G_b^{C8}=0.20.
\]

然后所有 physical occurrences按 C0 anchor support重新计算 local advantages。

macro 则完全按 C4处理。

---

## 6.11 C8 实现重点

可以复用当前 C3 的：

\[
G_e+\gamma^d(G_b-G_o)
\]

逻辑，但必须改两个核心条件：

### 当前 C3

- 基于 C1 unique support；
- selected origin被 collapse / direct mean；
- copied origin不进 PPO。

### C8

- 基于完整 C0 physical support；
- selected origin保持 C0；
- copied origin仍进 PPO；
- 只给 `step_index < origin_step_index` 的 strict natural ancestors override \(G\)。

伪代码：

```python
c0 = compute_c0(full_physical_batch)

g_override = c0.step_returns.copy()

for edge in natural_root_edges:
    branches = strict_descendant_branches(edge)
    if not branches:
        continue

    candidates = [c0.G[edge]]

    for b in branches:
        o = natural_origin[b]
        delta = c0.G[b] - c0.G[o]
        distance = step_index[o] - step_index[edge]

        candidate = (
            c0.G[edge]
            + gamma ** distance * delta
        )
        candidates.append(candidate)

    g_override[edge] = mean(candidates)

local_c8 = gigpo_local_norm(
    full_c0_physical_support,
    g_override,
)

macro_c8 = macro_c4

final = macro_c8 + omega * local_c8
```

---

## 6.12 C8 invariants

- [ ] training rows与 C0完全一致；
- [ ] copied branch-origin仍进入 PPO；
- [ ] macro逐 row与 C4完全一致；
- [ ] selected origin \(G\) 不做 explicit override；
- [ ] copied branch-origin \(G\) 不做 explicit override；
- [ ] 只对 strict natural ancestors做 \(G\) backup；
- [ ] local distance factor为 \(\gamma^{origin\_step-target\_step}\)；
- [ ] denominator = `1 + num_strict_descendant_branches`；
- [ ] local normalization support与 C0完全一致；
- [ ] 无 strict descendant时退化为 C4/C0 local；
- [ ] 不能把 C3 的 unique-edge keep-index逻辑带入 C8。

---

# 7. C7：Flat-Leaf GiGPO

## 7.1 研究问题

C0 / C0.5 / C4 / C8 都保留了：

\[
\boxed{
\text{shared replay prefix 不重复训练}
}
\]

这个 tree-aware 约束。

C7 刻意测试完全相反的极端：

> 不再把 rollout tree 当成需要特殊 credit assignment 的树；每个 terminal leaf都还原成一条从任务初始状态到 terminal 的完整 pseudo-rollout，然后直接运行 production GiGPO advantage 与 PPO loss。

定义：

\[
\boxed{
C7
=
\text{Leaf Expansion}
+
\text{Vanilla GiGPO Objective}
}
\]

---

## 7.2 “取消 branch”指训练视角，而不是 acquisition 视角

BACE acquisition阶段仍然需要：

- root / branch allocation；
- anchor selection；
- ERV；
- replay；
- branch continuation。

否则无法生成 branch leaf。

“取消 branch”只指：

\[
\boxed{
\text{进入 advantage / loss 后，不再区分 root row、branch-origin、replay prefix等 tree-specific credit角色。}
}
\]

训练输入只是一组完整 trajectories。

---

## 7.3 Full pseudo-rollout reconstruction

假设 natural root：

```text
e0 -> e1 -> o -> r3 -> root terminal
```

从 \(o\) 产生 branch：

```text
replay e0 -> e1
execute copied o
b3 -> b4 -> branch terminal
```

C7 构造两条完整训练 trajectories：

### Root leaf

```text
T_root:
e0 -> e1 -> o -> r3 -> root terminal
```

### Branch leaf

```text
T_branch:
e0' -> e1' -> o' -> b3 -> b4 -> branch terminal
```

其中：

\[
e0'=e0,
\qquad
e1'=e1,
\qquad
o'=o
\]

在 token/action 内容上复制，但在 C7 training batch 中它们是：

\[
\boxed{
\text{branch full trajectory 的独立 physical occurrences}
}
\]

---

## 7.4 每个 terminal leaf 对应一条完整 trajectory

若同一 natural root 有两条 branches：

```text
             B1
            /
P -> o -> R
            \
             B2
```

则 C7 产生：

```text
T0 = P + o + R
T1 = P' + o' + B1
T2 = P'' + o'' + B2
```

prefix：

\[
P
\]

在 PPO 中出现三次。

因此：

\[
\boxed{
\text{prefix training multiplicity}
=
\text{descendant terminal-leaf multiplicity}
}
\]

这是 C7 的核心，而不是实现 bug。

---

# 8. C7 的 Macro

C7 不使用：

- C0 frozen macro；
- C4 macro backup；
- C2 incremental macro；
- tree descendant mean。

而是对 flatten 后的 full trajectories直接调用当前 production GiGPO macro path。

即每一条完整 pseudo-rollout：

\[
\tau_\ell
\]

有自己的 terminal outcome：

\[
R_\ell.
\]

然后按照当前 GiGPO-compatible production normalization重新计算：

\[
A_{\ell}^{E,C7}.
\]

### 非常重要

C7 **不冻结 C0 macro statistics**。

因为 C7 的目的就是：

\[
\boxed{
\text{把 flatten 后的数据真正当成一批普通 GiGPO trajectories}
}
\]

所以它的 macro group statistics自然由 flat-leaf batch重新决定。

---

# 9. C7 的 Local Return

每条 pseudo-rollout必须重新沿自己的完整 path计算 step return-to-go。

对于 branch full trajectory：

```text
e0' -> e1' -> o' -> b3 -> b4 -> branch terminal
```

必须计算：

\[
G_{e0'}^{branch},
\]

\[
G_{e1'}^{branch},
\]

\[
G_{o'}^{branch},
\]

等等。

不能简单复制：

\[
G_{e0}^{root},
G_{e1}^{root}.
\]

否则就不是“branch full rollout”。

---

## 9.1 Invalid penalty

C7 完全使用 vanilla production GiGPO invalid semantics。

因此如果某个 copied prefix action invalid：

- root trajectory中有一次该 penalty；
- branch trajectory的 duplicated prefix中也有一次；
- 两条 branch trajectories就会重复两次。

这是 C7 有意测试的：

\[
\boxed{
\text{full trajectory multiplicity}
}
\]

不使用 C4/C8 的 origin-delta cancellation。

---

# 10. C7 的 Anchor Groups

flatten 后，每一条 pseudo-rollout中的每个 step都作为 normal GiGPO occurrence。

因此：

```text
T_root  : e0, e1, o, ...
T_branch: e0', e1', o', ...
```

如果：

\[
state(e0)=state(e0'),
\]

那么它们都会进入同一个 GiGPO anchor group。

所以 C7 会自然产生：

\[
\boxed{
\text{shared-prefix duplicated local occurrences}
}
\]

这正是 C7 与 C8 的根本区别：

### C8

multiple descendant evidence：

\[
\rightarrow
\text{一个 original prefix occurrence 的 }G\text{ backup}
\]

### C7

multiple descendant leaves：

\[
\rightarrow
\text{多个 prefix physical occurrences}
\]

---

# 11. C7 的 PPO / old logprob

对 duplicated prefix：

- token ids复制 natural root；
- context/history复制对应 prefix；
- old logprob复制原 natural occurrence；
- PPO ratio继续使用：
  \[
  \pi_\theta/\pi_{\mathrm{old}}.
  \]

对 fresh suffix：

- 使用实际生成时的 token；
- 使用实际 old logprob。

C7 不新增 behavior correction。

但论文/实验解释必须诚实：

\[
\boxed{
\text{C7 objective形式是 vanilla GiGPO，
但数据不是从初始状态 i.i.d. 自然采样出的 vanilla GiGPO trajectories。}
}
\]

它是：

\[
\boxed{
\text{BACE acquisition-induced, leaf-expanded GiGPO objective}.
}
\]

branching 越多的 lineage 会拥有更大的训练权重。

---

# 12. C7 的一个例子

假设：

```text
root:
e0 -> e1 -> o -> success

branch:
replay e0 -> e1
copied o
q1 -> failure
```

C7 训练时构造：

### Trajectory A

```text
e0_A -> e1_A -> o_A -> success
```

### Trajectory B

```text
e0_B -> e1_B -> o_B -> q1_B -> failure
```

然后：

1. 给 A 计算完整 root return；
2. 给 B 按 failure leaf重新计算完整 return；
3. 把 A/B 当作两条普通 GiGPO trajectories；
4. trajectory-level advantage按 production GiGPO重新算；
5. local anchor groups中：
   - \(e0_A,e0_B\) 都参与；
   - \(e1_A,e1_B\) 都参与；
   - \(o_A,o_B\) 都参与；
6. 两条 trajectory的全部 trainable response tokens都进入 PPO。

所以 shared prefix被训练两次。

如果 branch continuation更好/更差，它会以完整第二条 rollout的形式影响 prefix，而不是先压成 mean backup。

---

# 13. C7 的推荐实现方式

不要尝试在现有 C0 batch上“简单把 replay mask打开”。

推荐单独实现：

```python
def expand_tree_to_full_leaf_trajectories(tree):
    ...
```

输出一个与 vanilla GiGPO rollout batch contract兼容的 flat batch。

高层流程：

```text
BACE acquisition
    ↓
root + branch tree
    ↓
reconstruct every terminal leaf as full trajectory
    ↓
assign new traj_uid per terminal leaf
    ↓
copy prefix tokens / contexts / old logprobs
    ↓
attach branch suffix
    ↓
recompute full-path rewards and step returns
    ↓
run production GiGPO advantage
    ↓
run production PPO loss
```

C7 尽量不要调用 C1/C2/C3/C4/C8 tree-credit函数。

这样才能保证：

\[
\boxed{
\text{C7 真正测试 flat-leaf vanilla objective}
}
\]

而不是某个混合版本。

---

# 14. C7 invariants

- [ ] 每个 terminal leaf恰好生成一条 full pseudo-rollout；
- [ ] root terminal leaf保留原 natural full rollout；
- [ ] branch leaf从初始状态到 branch terminal完整重建；
- [ ] replay/shared prefix在每个 branch leaf中复制成 trainable occurrences；
- [ ] 每个 flat leaf有独立 `traj_uid`；
- [ ] copied prefix old logprob来自 source natural occurrence；
- [ ] fresh suffix old logprob来自实际 branch generation；
- [ ] branch full trajectory的 prefix \(G\) 必须按 branch leaf重新计算；
- [ ] invalid penalty按 production GiGPO对每个 flat physical occurrence正常施加；
- [ ] macro/local normalization从 flat batch重新计算；
- [ ] 不使用 C0 frozen macro；
- [ ] 不使用 incremental backup；
- [ ] 不使用 unique-edge collapse；
- [ ] PPO/loss函数尽可能直接调用 vanilla production GiGPO path。

---

# 15. 四个版本放在同一个例子中

考虑：

```text
e0 -> e1 -> o -> R
            \
             b -> B
```

其中：

- \(o\)：selected natural origin；
- \(b\)：copied branch-origin；
- \(R\)：natural continuation；
- \(B\)：branch continuation。

---

## C0.5

训练 rows：

```text
e0, e1, o, b, branch suffix...
```

与 C0完全相同。

只对：

\[
\mathcal F(o)=\{o,b\}
\]

把 C0 local advantages：

\[
A_o^S,A_b^S
\]

平均后广播。

\(e0,e1\) 不接收 branch backup。

---

## C4

训练 rows仍与 C0完全相同。

\(o,b\) 完全保持 C0。

对：

\[
e0,e1
\]

只修改 macro：

\[
M_e
\rightarrow
mean(
M_e,\,
M_e+(M_b-M_o)
).
\]

local全部 C0。

---

## C8

训练 rows仍与 C0完全相同。

\(o,b\) 不做 direct collapse。

对 \(e0,e1\)：

### Macro

与 C4一致。

### Local

构造：

\[
G_e^{branch-candidate}
=
G_e+
\gamma^{t_o-t_e}(G_b-G_o),
\]

再：

\[
\widetilde G_e
=
mean(
G_e,
G_e^{branch-candidate}
).
\]

随后在原 C0 physical anchor support上重新算 local advantage。

---

## C7

构造：

```text
T_root   = e0 -> e1 -> o -> R
T_branch = e0' -> e1' -> o' -> B
```

全部作为普通 GiGPO trajectories。

没有：

- origin family；
- strict ancestor；
- backup；
- tree-aware unique support。

直接重新算完整 GiGPO advantage和 PPO loss。

---

# 16. 四个版本的直接对比

| 维度 | C0.5 | C4 | C8 | C7 |
|---|---|---|---|---|
| C0 physical support | 保留 | 保留 | 保留 | **不保留，改成 full-leaf flat support** |
| copied origin训练 | 是 | 是 | 是 | 是，作为完整 leaf中的 occurrence |
| selected origin direct \(G\) mean | 否 | 否 | 否 | 不适用 |
| origin-family local smoothing | **是** | 否 | 否 | 否 |
| strict ancestor macro backup | 否 | **是** | **是** | 不适用 |
| strict ancestor local backup | 否 | 否 | **是** | 不适用 |
| replay prefix增加 train row | 否 | 否 | 否 | **是** |
| local normalization support | C0 support，先 C0 norm后 post-process | C0 | C0，部分 \(G\) override后重算 | flat-leaf support |
| macro normalization | C0 | frozen C0 macro + incremental backup | 与 C4相同 | flat batch重新 vanilla GiGPO |
| gamma用于 tree backup | 否 | 否 | **仅 local** | vanilla return recursion |
| shared-prefix gradient multiplicity | C0 | C0 | C0 | **随 leaf数增加** |

---

# 17. 推荐配置接口

建议不要继续沿用容易混淆的 `C1/C2/C3` 内部 bool组合，而增加明确 mode：

```yaml
algorithm:
  bace:
    credit_mode: current
```

允许：

```text
current
c0_5_origin_family_local_mean
c4_macro_strict_ancestor
c7_flat_leaf_gigpo
c8_macro_local_strict_ancestor
```

---

# 18. 推荐代码结构

```python
class BaceCreditMode(str, Enum):
    CURRENT = "current"

    C0_5_ORIGIN_FAMILY_LOCAL_MEAN = (
        "c0_5_origin_family_local_mean"
    )

    C4_MACRO_STRICT_ANCESTOR = (
        "c4_macro_strict_ancestor"
    )

    C7_FLAT_LEAF_GIGPO = (
        "c7_flat_leaf_gigpo"
    )

    C8_MACRO_LOCAL_STRICT_ANCESTOR = (
        "c8_macro_local_strict_ancestor"
    )
```

---

# 19. 推荐高层调度逻辑

```python
if mode == CURRENT:
    return compute_c0(batch)

if mode == C0_5:
    c0 = compute_c0(batch)
    return postprocess_origin_family_local_adv(c0)

if mode == C4:
    c0 = compute_c0(batch)
    tree = build_tree_index(batch)
    macro = strict_ancestor_macro_backup(
        c0_macro=c0.macro,
        tree=tree,
    )
    return combine(
        macro=macro,
        local=c0.local,
    )

if mode == C8:
    c0 = compute_c0(batch)
    tree = build_tree_index(batch)

    macro = strict_ancestor_macro_backup(
        c0_macro=c0.macro,
        tree=tree,
    )

    g_override = strict_ancestor_local_g_backup(
        c0_g=c0.step_returns,
        tree=tree,
        gamma=gamma,
    )

    local = compute_gigpo_local(
        physical_support=c0.physical_support,
        g_override=g_override,
    )

    return combine(
        macro=macro,
        local=local,
    )

if mode == C7:
    flat_batch = expand_tree_to_full_leaf_trajectories(
        collected_tree=batch
    )

    return compute_vanilla_production_gigpo(
        flat_batch
    )
```

---

# 20. 必须共享、不允许顺便修改的组件

除 C7 明确改变 training support 外，C0.5/C4/C8 必须保持：

- root/branch topology；
- ERV；
- posterior；
- branch quota；
- natural origin selection；
- replay validation；
- branch suffix generation；
- invalid-action判定；
- invalid penalty数值；
- PPO ratio；
- old logprob；
- PPO clipping；
- KL；
- optimizer；
- learning rate；
- `step_advantage_w`；
- batch size与 rollout budget。

这样四个实验比较的才是 optimizer / credit semantics，而不是 acquisition变化。

---

# 21. 必须增加的专项测试

## 21.1 C0.5

### Test A

一个 family：

\[
A_o^S=-1,
\quad
A_b^S=+1.
\]

输出：

\[
0,0.
\]

### Test B

另一条 same `(z,a)` natural origin不属于 family，必须保持原值。

### Test C

macro逐 row与 C0完全相同。

### Test D

family local sum before/after一致。

---

## 21.2 C4

### Test A：direct origin不修改

若 branch origin为 \(o\)，验证：

\[
M_o^{C4}=M_o^{C0}.
\]

### Test B：one strict ancestor

验证：

\[
M_e^{C4}
=
\frac{
M_e+
(M_e+M_b-M_o)
}{2}.
\]

### Test C：two branches

验证 denominator为：

\[
3.
\]

### Test D

local逐 row与 C0完全相同。

### Test E

macro不使用 gamma。

---

## 21.3 C8

### Test A：origin不 override \(G\)

\[
G_o^{override}=G_o.
\]

### Test B：one strict ancestor

验证：

\[
G_e^{override}
=
\frac{
G_e+
G_e+\gamma^d(G_b-G_o)
}{2}.
\]

### Test C：later branch

验证 gamma exponent严格为：

\[
origin\_step-target\_step.
\]

### Test D：invalid cancellation

natural origin和 copied origin有同一 local penalty时：

\[
G_b-G_o
\]

不含该共同 penalty。

### Test E

C8 macro逐 row与 C4一致。

### Test F

local normalization使用完整 C0 physical support，而不是 C1 unique support。

---

## 21.4 C7

### Test A：一 root + 一 branch

最终 flat trajectories 数：

\[
2.
\]

### Test B：一 root + 两 branches

最终 full trajectories：

\[
3.
\]

shared prefix trainable occurrence count也应变为 3 倍。

### Test C

branch pseudo-rollout 的 prefix \(G\) 必须使用 branch terminal continuation重新计算。

### Test D

macro/local不读取 C0 frozen advantage。

### Test E

flatten 后直接进入 production GiGPO path。

### Test F

每个 leaf独立 `traj_uid`，但 duplicated prefix old logprob与 source natural row一致。

---

# 22. 推荐 Trace 字段

## 所有模式

```text
credit_mode
task_id
root_id
branch_id
source_type
occurrence_id
natural_origin_occurrence_id
step_index
anchor_id
action_id
```

## C0.5

```text
origin_family_id
family_size
local_c0
local_family_mean
local_c0_5
```

## C4

```text
macro_c0
strict_descendant_branch_ids
macro_branch_delta
macro_candidates
macro_c4
```

## C8

```text
g_c0
strict_descendant_branch_ids
branch_origin_g
natural_origin_g
delta_g
distance
gamma_discount
g_candidates
g_c8_override
local_c8
macro_c4
```

## C7

```text
leaf_id
flat_traj_uid
source_root_id
source_branch_id
copied_prefix_length
is_flattened_prefix_copy
full_leaf_terminal_reward
full_leaf_step_return
```

---

# 23. 实验解释矩阵

已有：

\[
C0>C1,C2,C3
\]

且：

\[
C2>C1,
\qquad
C3\text{ 最差}.
\]

新的四个版本分别提供以下因果判断。

---

## 23.1 C0.5 vs C0

回答：

\[
\boxed{
\text{continuation-specific local credit 是否有价值？}
}
\]

### 若

\[
C0.5\approx C0>C1
\]

说明：

> C0 的主要优势来自 occurrence / gradient multiplicity，而不是同一 replay-family 内不同 local credit。

### 若

\[
C0.5>C0
\]

说明：

> multiplicity 有益，但 continuation-specific local disagreement 有害；C0.5可能成为最佳主候选。

### 若

\[
C0>C0.5
\]

说明：

> continuation-specific local credit本身也提供有效训练信息。

---

# 24. C4 vs C0

回答：

\[
\boxed{
\text{strict-prefix macro tree backup 是否有独立收益？}
}
\]

### 若

\[
C4>C0
\]

说明：

> 保留 C0 origin multiplicity后，branch trajectory outcome继续向 shared prefix传播 macro credit是有效的。

### 若

\[
C4\approx C0
\]

说明：

> C2>C1主要可能是 macro backup在补偿 C1 origin collapse。

### 若

\[
C4<C0
\]

说明：

> upstream macro backup本身也可能不必要，C0已是更合适的 credit estimator。

---

# 25. C8 vs C4

回答：

\[
\boxed{
\text{保留 C0 origin multiplicity后，strict-prefix local-}G\text{ backup 是否有价值？}
}
\]

### 若

\[
C8>C4
\]

说明：

> 原来 C3的劣势至少部分来自 C1 origin-collapse；local tree evidence本身仍可能有用。

### 若

\[
C8<C4
\]

则可以更有把握地认为：

\[
\boxed{
\text{branch outcome适合回传 macro trajectory credit，
但不适合广泛改写 earlier anchor local credit。}
}
\]

---

# 26. C7 vs C0/C8

回答：

\[
\boxed{
\text{branch evidence应该 aggregate 到 tree edge，
还是保留为完整 leaf training multiplicity？}
}
\]

### 若

\[
C7>C0
\]

说明：

> BACE acquisition产生的 lineage weighting本身可能是有效 curriculum；无需复杂 tree-credit estimator。

### 若

\[
C7<C0
\]

说明：

> shared-prefix amplification确实有害，不能简单把每个 branch leaf当成独立 full rollout训练。

### 若

\[
C7>C8
\]

说明：

> averaging / backup可能丢掉了有价值的 per-leaf training multiplicity。

### 若

\[
C8>C7
\]

说明：

> tree-aware evidence aggregation优于完整 prefix duplication。

---

# 27. 推荐实验优先级

如果资源有限，建议：

\[
\boxed{
C0.5
\rightarrow
C4
\rightarrow
C7
\rightarrow
C8
}
\]

理由：

1. **C0.5** 最直接解释 C0 为什么优于 C1；
2. **C4** 当前先验最有希望，因为 C2 比 C1略好；
3. **C7** 能检查是否整个 tree optimizer 被设计得过于复杂；
4. **C8** 用于重新确认 local prefix backup是否确实有害。

如果可以并行，则直接同配置跑：

```text
C0
C0.5
C4
C7
C8
```

并尽量共享：

- 相同 model checkpoint；
- 相同 seed集合；
- 相同 rollout budget；
- 相同 acquisition参数；
- 相同硬件布局；
- 相同 eval protocol。

---

# 28. 最终一句话定义

## C0.5

> **完全保留 C0 的 physical occurrences、macro、normalization 与 PPO support；先按 C0 计算 occurrence-level local advantages，再仅在“同一个 concrete natural origin + 其 copied branch origins”的 replay family 内平均 local advantage并广播回原 occurrences，从而保留 selected-decision multiplicity但减少 continuation-specific local-credit冲突。**

---

## C4

> **完全保留 C0 的 occurrence 与 local-credit语义，只对 branch origin 之前的 strict natural ancestors使用 `target C0 macro + (branch-origin macro − natural-origin macro)` 构造 descendant macro candidates并等权平均；selected origin、copied branch-origin与 branch suffix继续使用 C0 macro。**

---

## C8

> **完整继承 C4，同时对 strict natural ancestors使用 `G_target + gamma^distance × (G_branch-origin − G_natural-origin)` 构造 descendant local-return candidates并与 original return等权平均，再在完整 C0 physical occurrence support上重新计算 GiGPO local advantage；selected origin与 copied origin不做 explicit G collapse。**

---

## C7

> **保留 BACE acquisition，但在 optimization阶段将每个 terminal leaf还原为一条从任务初始状态到 terminal 的完整 pseudo-rollout，包括复制 replay/shared prefix；随后完全按照 production GiGPO的 trajectory/local advantage与 PPO loss处理这些 flat trajectories，使 shared-prefix训练权重自然随 descendant leaf multiplicity增长。**

---

# 29. 最终实现边界

这四个版本不应再混合成更多中间组合，第一轮实验先固定：

\[
\boxed{
C0,\ C0.5,\ C4,\ C7,\ C8
}
\]

分别回答：

\[
\boxed{
\text{origin local smoothing}
}
\]

\[
\boxed{
\text{prefix macro backup}
}
\]

\[
\boxed{
\text{full-leaf multiplicity}
}
\]

\[
\boxed{
\text{prefix local backup}
}
\]

四个机制问题。

这样结果无论哪一个版本最好，都能给出清楚的算法解释，而不会退化成组合式调参。
