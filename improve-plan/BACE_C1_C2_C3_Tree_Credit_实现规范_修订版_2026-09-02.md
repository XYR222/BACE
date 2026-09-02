# BACE-GiGPO：C1 / C2 / C3 Tree-Credit 优势计算实现规范（修订版）

> **版本**：2026-09-02  
> **用途**：用于当前 `XYR222/BACE` 实际代码中实现和核对 C1 / C2 / C3 三个 Tree-Credit 版本  
> **核心原则**：只修改 branch evidence 如何进入 advantage / PPO credit，不修改 rollout acquisition、Batch-ERV、root/branch topology、PPO ratio、token mask、optimizer 和 scheduler  
> **重要修订**：明确区分“方法层面的 leaf-uniform macro 语义”与“当前代码采用的 GiGPO-compatible occurrence-weighted 稳定化实现”

---

# 0. 最重要的口径：方法语义与代码实现必须分开

当前 BACE 的 trajectory-level / macro credit 在方法设计上仍然是：

\[
\boxed{
\text{terminal leaf 是 trajectory-level outcome 的概念单位}
}
\]

也就是说，root leaf 与 branch leaf 都代表一条完整的 root-to-terminal outcome：

\[
\mathcal L_g
=
\mathcal L_g^{root}
\cup
\mathcal L_g^{branch}.
\]

从方法叙事上，我们仍然把 macro credit 理解为：

\[
\boxed{
\text{leaf-level trajectory credit}
}
\]

而不是认为“长 leaf 应该因为 occurrence 更多而拥有更高理论权重”。

但是，**当前实际代码并没有严格使用 pure leaf-uniform normalization**。

为了：

- 保持 GiGPO 训练统计的数值稳定性；
- 保持与原 GiGPO / 当前 BACE baseline 的实现兼容；
- 避免突然改变 macro signal 的尺度；
- 保持已有实验结果的可比性；

当前代码采用：

\[
\boxed{
\text{GiGPO-compatible occurrence-weighted stabilized macro normalization}
}
\]

因此必须明确：

\[
\boxed{
\text{Conceptual objective / credit semantics}
\neq
\text{exact numerical normalization implementation}
}
\]

更准确地说：

\[
\boxed{
\text{leaf-level macro semantics}
\xrightarrow{\text{GiGPO-compatible stabilization}}
\text{occurrence-weighted numerical estimator}
}
\]

这不是说二者数学等价。

而是说：

> **我们在方法层面仍以 leaf 作为 macro outcome 的概念单位；当前代码为了训练稳定性，使用 occurrence-weighted 的 GiGPO-compatible 实现近似这一语义。**

因此本文中的 C0 / C1 / C2 / C3：

\[
\boxed{
\textbf{全部固定使用当前稳定化 macro normalization}
}
\]

以保证实验只比较：

\[
\boxed{
\text{tree credit assignment}
}
\]

而不是把：

\[
\text{tree credit}
+
\text{macro normalization scheme}
\]

同时改变。

如果未来要比较：

\[
\text{strict leaf-uniform}
\quad\text{vs}\quad
\text{current stabilized implementation},
\]

应作为一个**独立 normalization ablation**，不能混入 C1 / C2 / C3。

---

# 1. 四个版本的总体关系

对照基线：

\[
\boxed{
C0=\text{Current BACE}
}
\]

三个新版本：

\[
\boxed{
C1=\text{O1-Local}
}
\]

\[
\boxed{
C2=\text{O1-Tree-Macro}
}
\]

\[
\boxed{
C3=\text{O1-Full-Tree}
}
\]

形成严格递进：

\[
\boxed{
C1:
\text{只解决 copied branch-origin 的重复训练，并在 direct continuation 上平均 }G
}
\]

\[
\boxed{
C2:
C1
+
\text{将 descendant leaf 的 macro trajectory credit backup 到 shared natural prefix}
}
\]

\[
\boxed{
C3:
C2
+
\text{将 descendant discounted return }G
\text{ 也 backup 到 shared natural prefix}
}
\]

对应三个假设：

\[
\boxed{
H_1:
\text{copied-origin over-counting}
}
\]

\[
\boxed{
H_2:
\text{shared-prefix macro under-credit}
}
\]

\[
\boxed{
H_3:
\text{shared-prefix local-}G\text{ under-credit}
}
\]

实现时必须满足：

\[
\boxed{
C1\rightarrow C2
\text{ 只增加 macro descendant backup}
}
\]

\[
\boxed{
C2\rightarrow C3
\text{ 只增加 prefix }G\text{ descendant backup}
}
\]

不能顺便改变：

- macro normalization；
- action aggregation；
- `step_advantage_w`；
- branch loss weight；
- descendant weight；
- PPO ratio；
- token mask；
- ERV；
- branch scheduler；
- optimizer。

---

# 2. C0：当前实际代码的准确基线

---

## 2.1 当前 rollout tree 中的三类 trainable occurrence

当前 C0 的 trainable set 可写为：

\[
\mathcal D_g^{C0}
=
\mathcal D_g^{root}
\cup
\mathcal D_g^{branch\_origin}
\cup
\mathcal D_g^{branch\_suffix}.
\]

其中：

### Natural root occurrence

由：

\[
\pi_{\mathrm{old}}
\]

自然产生。

正常训练。

---

### Copied branch origin

branch 从 natural root 某个 concrete occurrence 发起时：

- 恢复环境；
- replay 已有 prefix；
- 复制原 origin 的 CoT / action；
- 执行相同动作；
- 重新生成 continuation。

因此 copied origin 不是第二次新的 action sample。

它的统计语义是：

\[
\Delta n^{policy}=0,
\]

\[
\Delta n^{evaluation}=1.
\]

但是当前 C0 又令：

\[
\Delta n^{train}=1.
\]

即：

\[
\boxed{
\text{同一个 concrete policy edge 被作为一个新的 branch-origin training occurrence}
}
\]

---

### Fresh branch suffix

branch origin 之后由冻结 policy 真实生成的新动作。

它们是新的 policy samples：

\[
\Delta n^{policy}=1,
\]

因此正常进入 PPO。

---

## 2.2 Replay prefix

Mechanical replay prefix：

- 不进入 PPO；
- 不形成新的 GiGPO occurrence；
- 不重新计算 old logprob；
- 不因为 descendant leaf 数量增加而重复训练。

这一点 C0/C1/C2/C3 全部保持不变。

---

# 3. Macro 的两层定义

这是本修订版最重要的地方。

---

## 3.1 概念层：Leaf-level macro semantics

每一个 terminal leaf：

\[
\ell\in\mathcal L_g
\]

有自己的 trajectory outcome：

\[
R_\ell.
\]

概念上，macro credit 的对象是：

\[
\boxed{
\ell
}
\]

而不是：

\[
\boxed{
\text{leaf 中包含多少 occurrence}
}
\]

因此论文方法语义仍然可以理解为：

\[
\text{root/branch terminal leaves}
\rightarrow
\text{trajectory-level relative credit}.
\]

---

## 3.2 实现层：Stable GiGPO-compatible normalization

当前代码为了稳定性，并未使用严格：

\[
\mu_{leaf}
=
\frac{1}{|\mathcal L_g|}
\sum_{\ell\in\mathcal L_g}
R_\ell
\]

作为唯一 macro normalization 实现。

而是沿用 GiGPO-compatible 的 occurrence-weighted统计路径。

可抽象为：

\[
\mu_g^{E,stable}
=
\frac{
\sum_{\ell\in\mathcal L_g}
w_\ell^{occ}R_\ell
}{
\sum_{\ell\in\mathcal L_g}
w_\ell^{occ}
},
\]

其中：

\[
w_\ell^{occ}
\]

由当前代码实际纳入 macro 统计的 physical occurrences 数量诱导。

相应：

\[
A_\ell^{E,stable}
=
\operatorname{StableNorm}
(
R_\ell;
\{R_{\ell'}\},
\{w_{\ell'}^{occ}\}
).
\]

这里：

\[
\boxed{
A_\ell^{E,stable}
}
\]

仍然是一个 leaf-associated macro scalar，

只是其 mean/std 使用 occurrence-weighted 的稳定化实现。

---

## 3.3 本文不重新实现当前 stable normalizer

实现 C1/C2/C3 时：

\[
\boxed{
\text{必须直接调用当前 production macro normalization path}
}
\]

不要手写：

- leaf-uniform mean/std；
- occurrence-weighted mean/std；
- 自己的 sample std；
- 自己的 zero-variance logic。

本文只规定：

> 如何把当前代码已经算出来的 stable macro leaf signal 重新分配给 unique tree edges。

---

# 4. Base Macro Snapshot

为保证 C1/C2/C3 的差异只来自 tree credit，需要在所有变体中先计算完全相同的 C0 stable macro snapshot。

---

## 4.1 计算时机

先按照当前 C0 physical structure 完整构造：

\[
\mathcal D_g^{C0}.
\]

即包括：

- natural root；
- copied branch origin；
- fresh branch suffix。

然后调用当前实际代码：

```python
compute_current_stable_macro(...)
```

得到：

\[
\boxed{
A_\ell^{E,base}
}
\]

其中：

\[
A_\ell^{E,base}
\equiv
A_\ell^{E,stable,C0}.
\]

---

## 4.2 为什么 C1 删除 copied origin 后仍不重算 macro baseline

因为如果 C1 同时：

1. copied origin 不训练；
2. macro normalization support 也从 C0 physical occurrences 改成 unique edges；

那么 C1 相比 C0 同时改变：

\[
\text{origin training multiplicity}
\]

和：

\[
\text{macro normalization weighting}.
\]

这样无法判断 improvement 来自哪一项。

所以本规范规定：

\[
\boxed{
C1/C2/C3 都冻结同一个 C0 stable macro snapshot
}
\]

即使 C1 后 copied origin 不进入最终 PPO。

这不是因为 occurrence-weighted 是方法理论目标，

而是因为：

\[
\boxed{
\text{它是当前稳定训练实现，必须作为 controlled implementation baseline 固定}
}
\]

---

## 4.3 Future ablation

如果以后要测试：

\[
\text{strict leaf-uniform macro}
\]

则另开：

```text
macro_normalization_mode:
    stable_occurrence
    strict_leaf_uniform
```

但此实验必须与：

```text
tree_credit_mode
```

正交。

---

# 5. 数据对象：Anchor / Action / Occurrence / Unique Edge

---

## 5.1 Anchor

继续完全沿用当前 GiGPO anchor key：

\[
z
=
\operatorname{AnchorKey}
(
o^{pre}
).
\]

C1/C2/C3 不改变：

- exact matching；
- loop occurrence；
- history inclusion；
- state canonicalization。

---

## 5.2 Canonical Action

继续使用当前：

\[
u
=
\operatorname{ActionIdentity}(a).
\]

但 C1/C2/C3：

\[
\boxed{
\text{不是 action-mean}
}
\]

所以相同：

\[
(z,u)
\]

不代表要 collapse。

---

## 5.3 Physical Occurrence

每个 physical occurrence 具有：

```text
occurrence_id
task_id
root_id / branch_id
step_index
anchor_id
action_id
source_type
G
leaf_id
tokens
old_logprob
```

---

## 5.4 Unique Policy Edge

C1/C2/C3 的基本训练对象是：

\[
\boxed{
\text{concrete policy edge}
}
\]

而不是：

\[
(z,u).
\]

最安全的 identity：

\[
\boxed{
edge\_id
=
natural\ occurrence\_id
}
\]

对于 fresh suffix：

\[
edge\_id
=
suffix\ occurrence\_id.
\]

概念上：

\[
e
=
(
\text{task},
\text{root},
\text{step occurrence},
h,
z,
u
).
\]

---

## 5.5 Copied origin

若 branch \(b\) 来自 natural occurrence：

\[
e
\]

则：

\[
\operatorname{origin}(b)=e.
\]

copied branch origin：

\[
e_b^*
\]

在 policy-edge identity 上视为：

\[
\boxed{
e_b^*\equiv e.
}
\]

原因：

- actor-visible context 相同；
- CoT/action 相同；
- action 并未重新 sample；
- 只是 continuation outcome 不同。

---

## 5.6 相同 `(z,u)` 的不同 natural occurrences不能 collapse

例如：

```text
root1 step7:  z, a1
root2 step9:  z, a1
```

即使：

\[
z_1=z_2,
\qquad
u_1=u_2=a_1,
\]

只要：

\[
occurrence\_id_1
\neq
occurrence\_id_2,
\]

就是：

\[
e_1\neq e_2.
\]

C1/C2/C3 都独立估计：

\[
G_{e_1},
\qquad
G_{e_2}.
\]

之后它们仍可进入同一个 GiGPO anchor group。

---

# 6. TreeCreditIndex

建议新增：

```python
TreeCreditIndex
```

至少保存：

```text
branch_id -> origin_occurrence_id
branch_id -> parent_root_id
branch_id -> origin_step_index
branch_id -> leaf_id

natural_occurrence_id -> root_id
natural_occurrence_id -> step_index
natural_occurrence_id -> root_leaf_id

direct_branch_leaves_by_origin[occurrence_id]
descendant_leaves_by_natural_edge[occurrence_id]

leaf_id -> base_stable_macro_advantage
```

以及：

```text
edge_leaf_return[(edge_id, leaf_id)]
```

或 lazy getter：

```python
get_return_from_edge_to_leaf(edge_id, leaf_id)
```

---

# 7. Direct Set 与 Descendant Set

这两个集合不能混淆。

---

## 7.1 Direct Continuation Set

对 natural edge \(e\)：

\[
\boxed{
Direct(e)
=
\{\ell_{root(e)}\}
\cup
\{
\ell_b:
origin(b)=e
\}.
}
\]

只包含：

- 原 natural continuation；
- **直接从 e 发起**的 branches。

例如：

```text
e
├── natural -> L0
├── branch -> L1
└── branch -> L2
```

则：

\[
Direct(e)=\{L0,L1,L2\}.
\]

C1 使用这个集合。

---

## 7.2 Full Descendant Set

定义：

\[
\boxed{
Desc(e)
=
\{
\ell:
\text{完整 root-to-leaf conceptual path 经过 }e
\}.
}
\]

例如：

```text
e1
 |
e2
 |
e3 ------ B1
 |
e4
 |
e5 ------ B2
 |
R
```

则：

\[
Desc(e1)=\{R,B1,B2\},
\]

\[
Desc(e3)=\{R,B1,B2\},
\]

但：

\[
Desc(e4)=\{R,B2\}.
\]

B1 在 e3 已分叉，不经过 e4。

C2/C3 使用：

\[
Desc(e).
\]

---

## 7.3 不能按 anchor key 推 descendant

另一条 root 即使也有：

\[
z
\]

也不是当前 root edge 的 descendant。

必须基于：

```text
root_id
step_index
branch origin
```

做 ancestry。

---

## 7.4 同 root loop

若：

```text
step3: z
step8: z
```

它们是两个不同 edge：

\[
e_3,
e_8.
\]

branch from step8：

- 是 e3 的 descendant；
- 也是 e8 的 descendant；
- direct origin 只有 e8。

branch from step3：

- 是 e3 的 descendant；
- 不是 natural e8 的 descendant。

---

# 8. \(G_{e,\ell}\) 的定义

C1/C3 使用：

\[
\boxed{
G_{e,\ell}
}
\]

表示：

> 从 unique edge \(e\) 开始，沿 terminal leaf \(\ell\) 的 discounted return-to-go。

必须保持当前 GiGPO：

\[
\gamma
\]

和 reward preprocessing。

---

## 8.1 Natural leaf

若：

\[
\ell=\ell_{root(e)},
\]

直接使用当前：

\[
G_e^{root}.
\]

---

## 8.2 Direct branch

若：

\[
origin(b)=e,
\]

则：

\[
G_{e,\ell_b}
\]

就是当前 branch-origin 记录对应的 return-to-go。

即使 branch-origin 在 C1/C2/C3 不再训练，

它的：

\[
G
\]

仍必须保留为 evaluation evidence。

---

## 8.3 Later branch descendant

若：

- e 位于 natural root step \(t\)；
- branch origin 位于同 root step \(t_b\ge t\)；

则：

\[
\boxed{
G_{e,\ell_b}
=
\sum_{j=t}^{t_b-1}
\gamma^{j-t}r_j^{root}
+
\gamma^{t_b-t}
G_{origin(b),\ell_b}.
}
\]

当：

\[
t_b=t,
\]

自动退化成 direct branch。

---

# 9. C1：O1-Local

---

## 9.1 目标

C1 只解决：

\[
\boxed{
\text{copied-origin over-counting}
}
\]

将：

\[
\Delta n^{policy}=0,
\quad
\Delta n^{eval}=1,
\quad
\Delta n^{train}=1
\]

改成：

\[
\boxed{
\Delta n^{policy}=0,
\quad
\Delta n^{eval}=1,
\quad
\Delta n^{train}=0.
}
\]

---

## 9.2 C1 training support

定义：

\[
\boxed{
\mathcal D_g^{U}
=
\mathcal D_g^{root}
\cup
\mathcal D_g^{branch\_suffix}.
}
\]

不包含：

\[
\mathcal D_g^{branch\_origin}.
\]

copied origin：

```text
ppo_eligible = false
local_group_eligible = false
train_mask = 0
```

但必须保留：

```text
origin_occurrence_id
leaf_id
G
terminal outcome
posterior evidence
```

---

## 9.3 C1 direct \(G\) backup

对任意 natural edge \(e\)：

\[
\boxed{
\widetilde G_e^{C1}
=
\frac{
1
}{
|Direct(e)|
}
\sum_{\ell\in Direct(e)}
G_{e,\ell}.
}
\]

若无 direct branch：

\[
\widetilde G_e^{C1}
=
G_e^{root}.
\]

若 branch 两次：

\[
\widetilde G_e^{C1}
=
\frac{
G_e^{root}
+
G_{e,b1}
+
G_{e,b2}
}{3}.
\]

---

## 9.4 Fresh suffix

fresh suffix 是新的 policy draw：

\[
\boxed{
\widetilde G_s^{C1}
=
G_s.
}
\]

---

## 9.5 C1 local group

用 unique support：

\[
\mathcal D_g^{U}
\]

重新构造：

\[
\mathcal I_g^{C1}(z)
=
\{
e\in\mathcal D_g^U:
z_e=z
\}.
\]

其中：

### natural edge

使用：

\[
\widetilde G_e^{C1}.
\]

### fresh suffix

使用：

\[
G_s.
\]

然后调用**当前 production GiGPO local normalization**：

\[
\boxed{
A_e^{S,C1}
=
GiGPONorm_z
(
\widetilde G_e^{C1}
).
}
\]

保持：

```text
mean_std_norm
sample/population std convention
epsilon
zero-variance handling
group-size handling
```

不变。

---

## 9.6 C1 中其他同 anchor edges 也可能变化

因为 copied origin 从 group 删除：

\[
\mu_z,\sigma_z
\]

会变化。

因此同一个 anchor 下的其他 edges：

\[
A^S
\]

可能一起变化。

这是 C1 的正确统计后果。

---

## 9.7 C1 macro

C1 不做 tree macro backup。

natural edge：

\[
\boxed{
A_e^{E,C1}
=
A_{\ell_{root(e)}}^{E,base}.
}
\]

fresh suffix：

\[
\boxed{
A_s^{E,C1}
=
A_{\ell_{branch(s)}}^{E,base}.
}
\]

注意：

> 这里的 \(A^{E,base}\) 是当前代码的 stable GiGPO-compatible leaf-associated macro signal。

不是重新计算的 strict leaf-uniform macro。

---

## 9.8 Copied branch-origin macro 怎么办

C0 中 copied origin有自己的 branch leaf macro signal。

C1 中 copied origin不训练，因此这份独立 training row 被删除。

branch leaf 的 macro information：

- 不再以 copied-origin 独立 PPO row 的方式训练；
- 仍然作用于 branch suffix；
- 在 C2 中进一步 backup 到 shared prefix。

因此 C1 是一个故意的机制诊断：

> 如果单纯删除 origin duplicate 已经改善，说明 over-counting 本身存在问题；如果 C1 变差而 C2恢复，则说明 macro tree backup 是必要的。

---

## 9.9 C1 final advantage

\[
\boxed{
A_e^{C1}
=
A_e^{E,C1}
+
\omega A_e^{S,C1}.
}
\]

---

# 10. C2：O1-Tree-Macro

---

## 10.1 C2 与 C1 的唯一差异

C2 保持：

\[
\boxed{
A_e^{S,C2}
=
A_e^{S,C1}.
}
\]

C2 只增加：

\[
\boxed{
\text{descendant macro backup}
}
\]

---

## 10.2 C2 的 macro backup

对任意 natural unique edge \(e\)：

\[
\boxed{
A_e^{E,C2}
=
\frac{
1
}{
|Desc(e)|
}
\sum_{\ell\in Desc(e)}
A_\ell^{E,base}.
}
\]

其中：

\[
A_\ell^{E,base}
\]

是由当前 stable macro implementation 已计算出的 leaf-associated macro signal。

---

## 10.3 为什么这里直接平均 leaf-associated stable macro

因为 C2 要做的是：

\[
\boxed{
\text{credit propagation}
}
\]

不是：

\[
\boxed{
\text{重新定义 macro normalization}
}
\]

所以我们在 frozen base macro 上做：

\[
\text{descendant mean operator}.
\]

---

## 10.4 Descendant weighting

第一版统一：

\[
w_\ell=1.
\]

不加入：

- lineage balance；
- branch/root special weight；
- BERV weight；
- depth weight；
- importance correction。

如果 later 需要，这些全部单独做 ablation。

---

## 10.5 Fresh suffix

主方法没有 branch-of-branch。

因此 fresh suffix edge通常：

\[
|Desc(s)|=1.
\]

所以：

\[
\boxed{
A_s^{E,C2}
=
A_{\ell(s)}^{E,base}.
}
\]

---

## 10.6 Selected origin

selected natural origin同样属于 natural edge。

所以：

\[
\boxed{
A_{origin}^{E,C2}
=
\operatorname{mean}_{\ell\in Desc(origin)}
A_\ell^{E,base}.
}
\]

这会把：

- original root leaf；
- direct branch leaves；
- later descendant branch leaves；

的 macro outcome evidence共同作用到这个 unique edge。

---

## 10.7 C2 final advantage

\[
\boxed{
A_e^{C2}
=
A_e^{E,C2}
+
\omega A_e^{S,C1}.
}
\]

---

## 10.8 C2 检验的假设

C2回答：

> 在长程 ALFWorld 中，branch leaf 已经提供了关于 shared prefix 后续成功/失败的新证据；如果 shared prefix 仍只使用 original root leaf 的 macro signal，是否造成 prefix macro under-credit？

---

# 11. C3：O1-Full-Tree

---

## 11.1 C3 与 C2 的唯一差异

C3 macro 必须严格保持：

\[
\boxed{
A_e^{E,C3}
=
A_e^{E,C2}.
}
\]

C3 只把 local return estimator从：

\[
Direct(e)
\]

扩展为：

\[
Desc(e).
\]

---

## 11.2 C3 full descendant \(G\) backup

对任意 natural edge \(e\)：

\[
\boxed{
\widetilde G_e^{C3}
=
\frac{
1
}{
|Desc(e)|
}
\sum_{\ell\in Desc(e)}
G_{e,\ell}.
}
\]

若：

\[
|Desc(e)|=1,
\]

则：

\[
\widetilde G_e^{C3}
=
G_e^{root}.
\]

---

## 11.3 Selected origin 与 C1 的关系

若 selected origin后面没有 later branch point：

\[
Desc(e)
=
Direct(e),
\]

则：

\[
\widetilde G_e^{C3}
=
\widetilde G_e^{C1}.
\]

如果 later还有 branch：

\[
Desc(e)
\supset
Direct(e),
\]

C3会继续吸收 later branch evidence。

---

## 11.4 Fresh suffix

无 branch-of-branch 时：

\[
\boxed{
\widetilde G_s^{C3}=G_s.
}
\]

---

## 11.5 C3 local group

继续使用：

\[
\mathcal D_g^U.
\]

natural edge使用：

\[
\widetilde G_e^{C3}.
\]

fresh suffix使用：

\[
G_s.
\]

然后重新调用当前 GiGPO local normalizer：

\[
\boxed{
A_e^{S,C3}
=
GiGPONorm_z(
\widetilde G_e^{C3}
).
}
\]

---

## 11.6 不能平均已经 normalize 的 \(A^S\)

禁止：

\[
A_e^{S,C3}
=
\operatorname{mean}_{\ell\in Desc(e)}
A_{e,\ell}^{S}.
\]

正确流程：

\[
\boxed{
G
\rightarrow
\text{descendant mean}
\rightarrow
\text{unique-edge anchor group}
\rightarrow
\text{重新 normalization}
}
\]

因为：

\[
A^S
\]

是 group-relative quantity，

而 group 的：

\[
\mu_z,\sigma_z
\]

在 edge collapse / \(G\) backup 后都可能变化。

---

## 11.7 C3 final advantage

\[
\boxed{
A_e^{C3}
=
A_e^{E,C2}
+
\omega A_e^{S,C3}.
}
\]

---

# 12. C0 / C1 / C2 / C3 总表

| 维度 | C0 Current | C1 O1-Local | C2 O1-Tree-Macro | C3 O1-Full-Tree |
|---|---|---|---|---|
| 方法层 macro 语义 | leaf-level | leaf-level | leaf-level | leaf-level |
| 实际 macro normalization | current stable occurrence-weighted | **固定 C0 stable** | **固定 C0 stable + backup** | **与 C2 相同** |
| copied origin 是否 PPO 训练 | 是 | 否 | 否 | 否 |
| copied origin 是否保留为 evidence | 是 | 是 | 是 | 是 |
| mechanical replay prefix 是否训练 | 否 | 否 | 否 | 否 |
| fresh suffix 是否训练 | 是 | 是 | 是 | 是 |
| selected origin direct \(G\) mean | 否 | 是 | 是 | 被 full descendant \(G\) 包含 |
| prefix macro descendant backup | 否 | 否 | 是 | 是 |
| prefix \(G\) descendant backup | 否 | 否 | 否 | 是 |
| local group统计单位 | physical occurrence | unique edge | unique edge | unique edge |
| action mean | 当前配置 | 不新增 | 不新增 | 不新增 |
| controller / ERV | current | current | current | current |
| PPO ratio | current | current | current | current |
| token mask contract | current | current | current | current |

---

# 13. 一个完整例子：一个 branch

假设：

```text
e1 -> e2 -> e3 -> e4(selected) -> root suffix -> failure
                              \
                               branch suffix -> success
```

设当前 stable macro snapshot：

\[
A_R^{E,base}=-1,
\]

\[
A_B^{E,base}=+1.
\]

selected origin：

\[
G_{e4,R}=0,
\qquad
G_{e4,B}=1.
\]

所以：

\[
\widetilde G_{e4}^{direct}=0.5.
\]

---

## 13.1 C1

copied branch origin不训练。

selected natural edge：

\[
G:0\rightarrow0.5.
\]

local重新normalize。

macro仍：

\[
A_{e4}^{E,C1}
=
A_R^{E,base}
=
-1.
\]

prefix：

\[
e1,e2,e3
\]

不因 branch descendant直接改变。

---

## 13.2 C2

local与 C1完全一致。

对：

\[
e1,e2,e3,e4
\]

都有：

\[
Desc(e)=\{R,B\}.
\]

因此：

\[
A_e^{E,C2}
=
\frac{-1+1}{2}
=
0.
\]

branch success开始回传到 shared prefix macro。

---

## 13.3 C3

macro与 C2完全一致：

\[
0.
\]

并且：

\[
\widetilde G_{e1},
\widetilde G_{e2},
\widetilde G_{e3},
\widetilde G_{e4}
\]

都利用：

\[
\{R,B\}
\]

descendant returns。

然后各自重新进入对应 anchor group计算 local advantage。

---

# 14. 两个 branch point：Direct 与 Descendant

假设：

```text
e1
 |
e2
 |
e3 ----- B1
 |
e4
 |
e5 ----- B2
 |
R
```

---

## C1 对 e3

\[
Direct(e3)=\{R,B1\}.
\]

因此：

\[
\widetilde G_{e3}^{C1}
=
mean(G_{e3,R},G_{e3,B1}).
\]

B2 不属于 direct continuation。

---

## C2 对 e3

\[
Desc(e3)=\{R,B1,B2\}.
\]

因此：

\[
A_{e3}^{E,C2}
=
mean(
A_R^{E,base},
A_{B1}^{E,base},
A_{B2}^{E,base}
).
\]

---

## C3 对 e3

\[
\widetilde G_{e3}^{C3}
=
mean(
G_{e3,R},
G_{e3,B1},
G_{e3,B2}
).
\]

---

## 对 e4

B1 已在 e3 分叉。

所以：

\[
Desc(e4)=\{R,B2\}.
\]

这要求实现使用真实 tree ancestry。

---

# 15. 相同 anchor/action，不同 concrete origins

假设：

```text
o1: root1 step5, z, a1
o2: root2 step8, z, a1
o3: root3 step6, z, a2
```

从 o1 branch 两次：

```text
b1.origin=o1
b2.origin=o1
```

从 o2 branch一次：

```text
b3.origin=o2
```

C1：

\[
\widetilde G_{o1}
=
mean(G_{o1}^{root},G_{b1},G_{b2}),
\]

\[
\widetilde G_{o2}
=
mean(G_{o2}^{root},G_{b3}).
\]

不能写成一个统一：

\[
mean(G_{o1},G_{b1},G_{b2},G_{o2},G_{b3}).
\]

后者是 action aggregation，不是 O1。

---

# 16. 严格的计算顺序

---

## Step 1：先构造 C0 physical records

保留逻辑记录：

```text
natural root
copied branch origin
fresh branch suffix
```

---

## Step 2：计算 C0 stable macro snapshot

调用当前 production：

```python
base_macro_by_leaf = compute_current_stable_macro(...)
```

从这一刻起：

\[
\boxed{
\text{C1/C2/C3 都不重新计算 macro global mean/std}
}
\]

---

## Step 3：构建 TreeCreditIndex

生成：

```text
Direct(e)
Desc(e)
edge->leaf return
```

---

## Step 4：构建 unique training support

C1/C2/C3：

```text
keep natural root occurrences
keep fresh branch suffix occurrences
drop copied branch origin from PPO/local groups
drop replay prefix
```

---

## Step 5：variant-specific \(G\)

### C1

```text
selected natural edge:
    Direct(e) G mean

other natural edge:
    original G

fresh suffix:
    original G
```

### C2

和 C1完全相同。

### C3

```text
every natural edge:
    Desc(e) G mean

fresh suffix:
    original G
```

---

## Step 6：重新计算 local

调用当前 production local normalizer：

```python
local_adv = compute_current_gigpo_local(
    unique_support,
    g_override=variant_g,
)
```

---

## Step 7：variant-specific macro

### C1

```text
natural edge:
    base macro of its original root leaf

fresh suffix:
    base macro of its branch leaf
```

### C2/C3

```text
natural edge:
    mean(base macro of descendant leaves)

fresh suffix:
    base macro of own branch leaf
```

---

## Step 8：final advantage

\[
A
=
A^E
+
\omega A^S.
\]

保持当前：

\[
\omega=step\_advantage\_w.
\]

---

## Step 9：当前 PPO

保持：

\[
\rho
=
\frac{\pi_\theta}{\pi_{old}}.
\]

保持：

- clip；
- KL；
- entropy；
- old logprob；
- token aggregation；
- optimizer；
- scheduler。

---

# 17. PPO Training Mask

| Segment | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| natural root trainable tokens | 1 | 1 | 1 | 1 |
| copied branch-origin tokens | 1 | **0** | **0** | **0** |
| fresh branch suffix tokens | 1 | 1 | 1 | 1 |
| mechanical replay prefix | 0 | 0 | 0 | 0 |
| prompt/history tokens | current | current | current | current |
| padding | 0 | 0 | 0 | 0 |

---

# 18. Old Logprob / Ratio

natural unique edge：

\[
\log\pi_{old}
\]

沿用 natural occurrence已有值。

fresh suffix：

沿用真实采样时 old logprob。

copied origin：

C1/C2/C3 不进入 PPO。

但可保留 old logprob 用于：

- trace；
- C0 regression；
- diagnostics。

---

# 19. Controller / Posterior 完全不改

即使 copied origin不再训练：

\[
\boxed{
\text{branch outcome 仍然是有效 Bayesian evidence}
}
\]

因此：

- selected pair posterior update；
- task-family history规则；
- BERV；
- capacity；
- branch quota；
- origin selection；

全部不变。

credit mode只能在 rollout collection完成后作用。

---

# 20. Branch origin 直接 terminal

若：

```text
replay -> copied origin -> terminal
```

没有 fresh suffix。

C1：

branch outcome仍进入：

\[
\widetilde G_{origin}^{direct}.
\]

C2：

branch leaf macro仍进入 shared ancestor：

\[
A^{E,desc}.
\]

C3：

branch outcome还进入 ancestor：

\[
\widetilde G^{desc}.
\]

不能因为没有 fresh suffix token就丢掉 leaf evidence。

---

# 21. Fresh suffix 到达已有 Anchor

fresh suffix edge是真实新 policy draw。

即使：

\[
z_s=z_{natural},
\]

甚至：

\[
u_s=u_{natural},
\]

仍然：

\[
edge_s\neq edge_{natural}.
\]

因此：

- fresh suffix保留独立 \(G\)；
- 不与 natural edge做 O1 collapse；
- 但会进入同一 GiGPO local group。

---

# 22. 推荐配置

建议：

```yaml
algorithm:
  bace:
    tree_credit_mode: current
```

允许：

```text
current
o1_local
o1_tree_macro
o1_full_tree
```

另加独立 normalization 轴：

```yaml
algorithm:
  bace:
    macro_normalization_mode: stable_occurrence
```

当前 C0/C1/C2/C3 全部固定：

```text
stable_occurrence
```

未来才可做：

```text
strict_leaf_uniform
```

ablation。

这样明确区分：

\[
\boxed{
\text{tree credit mode}
}
\]

和：

\[
\boxed{
\text{macro normalization implementation}
}
\]

---

# 23. 推荐代码结构

```python
class TreeCreditMode(str, Enum):
    CURRENT = "current"
    O1_LOCAL = "o1_local"
    O1_TREE_MACRO = "o1_tree_macro"
    O1_FULL_TREE = "o1_full_tree"
```

```python
class MacroNormalizationMode(str, Enum):
    STABLE_OCCURRENCE = "stable_occurrence"
    STRICT_LEAF_UNIFORM = "strict_leaf_uniform"
```

第一轮实验要求：

```text
macro_normalization_mode = stable_occurrence
```

不变。

---

# 24. 推荐高层伪代码

```python
def compute_bace_advantages(
    physical_records,
    branch_records,
    config,
):
    # --------------------------------------------------
    # 0. Current stabilized macro snapshot
    #    Conceptually leaf-level; numerically current
    #    GiGPO-compatible occurrence-weighted estimator.
    # --------------------------------------------------
    base_macro_by_leaf = compute_current_stable_macro(
        physical_records
    )

    if config.tree_credit_mode == "current":
        return compute_current_bace_advantage(
            physical_records,
            base_macro_by_leaf,
        )

    # --------------------------------------------------
    # 1. Tree index
    # --------------------------------------------------
    tree = build_tree_credit_index(
        physical_records,
        branch_records,
        base_macro_by_leaf,
    )

    # --------------------------------------------------
    # 2. Unique training support
    # --------------------------------------------------
    unique_records = build_unique_training_occurrences(
        physical_records,
        exclude_copied_branch_origin=True,
    )

    # --------------------------------------------------
    # 3. G transform
    # --------------------------------------------------
    if mode in {"o1_local", "o1_tree_macro"}:
        g_value = compute_direct_backed_returns(
            unique_records,
            tree,
        )

    elif mode == "o1_full_tree":
        g_value = compute_descendant_backed_returns(
            unique_records,
            tree,
            gamma=current_gamma,
        )

    # --------------------------------------------------
    # 4. Current GiGPO local normalization
    # --------------------------------------------------
    local_adv = compute_current_gigpo_local_from_override_g(
        unique_records,
        g_value,
    )

    # --------------------------------------------------
    # 5. Macro transform
    # --------------------------------------------------
    if mode == "o1_local":
        macro_adv = assign_base_macro_to_unique_records(
            unique_records,
            base_macro_by_leaf,
        )

    elif mode in {"o1_tree_macro", "o1_full_tree"}:
        macro_adv = compute_descendant_backed_macro(
            unique_records,
            tree,
            base_macro_by_leaf,
            weights="uniform",
        )

    # --------------------------------------------------
    # 6. Final
    # --------------------------------------------------
    for edge in unique_records:
        edge.advantage = (
            macro_adv[edge.id]
            + step_advantage_w * local_adv[edge.id]
        )

    return unique_records
```

---

# 25. Regression Invariants

---

## 25.1 C0 完全一致

```text
tree_credit_mode=current
macro_normalization_mode=stable_occurrence
```

必须与当前 production：

- macro；
- local；
- final advantage；
- occurrence count；
- token mask；
- loss；

逐项一致。

---

## 25.2 C1/C2/C3 base macro snapshot 完全一致

\[
\boxed{
A_\ell^{E,base,C1}
=
A_\ell^{E,base,C2}
=
A_\ell^{E,base,C3}
=
A_\ell^{E,base,C0}
}
\]

---

## 25.3 C1/C2/C3 eval count 不变

\[
N_{eval}^{C1}
=
N_{eval}^{C2}
=
N_{eval}^{C3}
=
N_{eval}^{C0}.
\]

---

## 25.4 Train occurrence count

\[
\boxed{
N_{train}^{C1}
=
N_{train}^{C2}
=
N_{train}^{C3}
=
N_{train}^{C0}
-
N_{branch\_origin}.
}
\]

---

## 25.5 C1 与 C2 local 必须完全相同

\[
\boxed{
A^{S,C1}
=
A^{S,C2}.
}
\]

---

## 25.6 C2 与 C3 macro 必须完全相同

\[
\boxed{
A^{E,C2}
=
A^{E,C3}.
}
\]

---

## 25.7 No descendant

若：

\[
|Desc(e)|=1,
\]

则：

\[
A_e^{E,C2}
=
A_{\ell(e)}^{E,base},
\]

\[
\widetilde G_e^{C3}
=
G_e^{root}.
\]

---

## 25.8 No direct branch

若：

\[
|Direct(e)|=1,
\]

则：

\[
\widetilde G_e^{C1}
=
G_e^{root}.
\]

---

## 25.9 Controller artifacts一致

同一个 frozen collected tree：

```text
roots
branches
selected anchors
selected actions
origin_occurrence_id
branch outcomes
posterior updates
BERV
```

C0/C1/C2/C3必须完全一致。

---

# 26. 单元测试

---

## Test 1：单 direct branch

验证：

\[
\widetilde G_e^{C1}
=
(G_R+G_B)/2.
\]

copied origin不进入 C1-C3 PPO。

---

## Test 2：同 origin 两 branches

\[
Direct(e)=\{R,B1,B2\}.
\]

训练 edge count仍为1。

---

## Test 3：same `(z,a)` different origins

必须保持：

\[
o1\neq o2.
\]

---

## Test 4：later branch descendant

验证：

\[
Direct(e3)=\{R,B1\}
\]

而：

\[
Desc(e3)=\{R,B1,B2\}.
\]

---

## Test 5：earlier branch不属于 later edge

验证：

\[
Desc(e4)=\{R,B2\}.
\]

---

## Test 6：same-root loop

验证 concrete occurrence ancestry。

---

## Test 7：another root same anchor

不能成为当前 edge descendant。

---

## Test 8：\(\gamma<1\)

验证：

\[
G_{e,\ell_b}
=
\sum_{j=t}^{t_b-1}
\gamma^{j-t}r_j
+
\gamma^{t_b-t}G_{origin,\ell_b}.
\]

---

## Test 9：origin direct terminal

无 suffix仍保留 tree evidence。

---

## Test 10：fresh suffix anchor collision

fresh suffix仍是独立 edge。

---

## Test 11：local zero variance

调用当前 zero-variance handling。

---

## Test 12：stable macro snapshot不漂移

C1/C2/C3 base macro必须与 C0一致。

---

## Test 13：C1/C2 local equality

逐 edge比较。

---

## Test 14：C2/C3 macro equality

逐 edge比较。

---

## Test 15：token mask

```text
copied origin trainable = 0
replay prefix trainable = 0
natural/fresh suffix unchanged
```

---

## Test 16：strict leaf-uniform未被意外启用

C1/C2/C3测试中：

```text
macro_normalization_mode
```

必须固定：

```text
stable_occurrence
```

---

# 27. 推荐 trace 字段

每个 natural unique edge：

```text
edge_id
root_id
step_index
anchor_id
action_id

direct_leaf_ids
descendant_leaf_ids

num_direct_continuations
num_descendant_leaves

g_original
g_direct_mean
g_descendant_mean

macro_base_stable
macro_descendant_mean

local_current
local_c1
local_c3

final_current
final_c1
final_c2
final_c3
```

branch：

```text
branch_id
origin_occurrence_id
parent_root_id
origin_step_index
leaf_id
origin_g_branch
terminal_reward
```

同时记录：

```text
macro_normalization_mode
tree_credit_mode
```

避免实验结果无法追溯。

---

# 28. 必须记录的诊断指标

---

## 28.1 Origin multiplicity

\[
m_e^{direct}
=
|Direct(e)|.
\]

---

## 28.2 Descendant multiplicity

\[
d_e
=
|Desc(e)|.
\]

---

## 28.3 Advantage sign flip

比较：

\[
sign(A^{C0})
\]

与：

\[
sign(A^{C1}),
sign(A^{C2}),
sign(A^{C3}).
\]

---

## 28.4 Gradient-mass proxy

C0：

\[
M_e^{C0}
=
\sum_{j\in physical\ copies(e)}
|A_j^{C0}|n_{token,j}.
\]

C1-C3：

\[
M_e^{Ck}
=
|A_e^{Ck}|n_{token,e}.
\]

---

## 28.5 Prefix credit utilization

统计一条 branch outcome 实际影响多少 unique edges 的：

- macro；
- local。

---

## 28.6 Stable-vs-leaf-uniform diagnostic

虽然第一轮不改 normalization，但建议离线额外记录：

\[
A_\ell^{E,stable}
\]

与 hypothetical：

\[
A_\ell^{E,strict-leaf}
\]

之间的：

- correlation；
- sign disagreement；
- scale ratio。

只做分析，不进入 C1/C2/C3 training。

这样可判断 stable approximation 偏离 conceptual leaf-uniform多少。

---

# 29. 三个版本的主要风险

---

## C1

优点：

- 消除 copied-origin duplicate gradient；
- 降低 direct continuation noise；
- 与 “one policy draw, multiple evaluations” 统计语义一致。

风险：

- copied branch-origin 的独立 macro training被移除；
- branch trajectory evidence可能利用不足。

---

## C2

优点：

- C1基础上恢复 branch leaf 对 shared prefix 的 trajectory-level credit；
- 直接针对 ALFWorld long-horizon prefix under-credit。

风险：

- BERV-selected branch evidence向 upstream传播；
- 存在 backward acquisition bias。

---

## C3

优点：

- 最大程度利用 tree descendant evidence；
- macro与 local 都能利用 branch outcome；
- 最完整的 full-tree Monte-Carlo backup。

风险：

- acquisition bias传播最强；
- upstream local action credit可能被过度改写；
- macro/local同时吸收同一 descendant evidence，可能增加 selected evidence影响强度。

---

# 30. 在线实验矩阵

第一轮：

| Run | tree_credit_mode | macro_normalization_mode |
|---|---|---|
| C0 | `current` | `stable_occurrence` |
| C1 | `o1_local` | `stable_occurrence` |
| C2 | `o1_tree_macro` | `stable_occurrence` |
| C3 | `o1_full_tree` | `stable_occurrence` |

**禁止第一轮把 C2/C3 同时切到 strict leaf-uniform。**

---

# 31. 结果解释

---

## Case A

\[
C1>C0,
\quad
C2\approx C1,
\quad
C3\approx C2.
\]

说明主要问题：

\[
\boxed{
\text{copied-origin over-counting}.
}
\]

---

## Case B

\[
C1\approx C0,
\quad
C2>C1.
\]

说明：

\[
\boxed{
\text{shared-prefix macro under-credit}
}
\]

是主要问题。

---

## Case C

\[
C2\approx C1,
\quad
C3>C2.
\]

说明：

\[
\boxed{
\text{shared-prefix local-}G\text{ backup}
}
\]

才真正重要。

---

## Case D

\[
C3>C2>C1>C0.
\]

说明当前 BACE 对 tree evidence 的利用逐层不足。

---

## Case E

\[
C2>C1>C0,
\quad
C3<C2.
\]

说明：

> branch outcome适合向 shared prefix传播 trajectory-level macro credit，但不适合无条件改写所有 upstream local action credit。

这时 C2 是最自然候选。

---

## Case F

\[
C1<C0,
\quad
C2>C0.
\]

说明：

> copied-origin extra training 可能确实贡献了一部分强 branch signal；但如果进行 unique-edge 去重，需要 C2 的 descendant macro backup补回 tree trajectory evidence。

---

# 32. 推荐实现顺序

---

## Stage 0：抽离 stable macro snapshot

先把当前 macro path封装成：

```python
compute_current_stable_macro(...)
```

C0 regression完全一致。

---

## Stage 1：实现 TreeCreditIndex

不改 advantage。

---

## Stage 2：实现 C1

只做：

```text
copied origin -> evidence only
Direct(e) G mean
unique-edge local regroup
macro = frozen stable base
```

---

## Stage 3：实现 C2

只增加：

```text
Desc(e)
base stable macro descendant mean
```

检查：

\[
A^{S,C2}=A^{S,C1}.
\]

---

## Stage 4：实现 C3

只增加：

```text
G_{e,leaf}
Desc(e) G mean
local regroup
```

检查：

\[
A^{E,C3}=A^{E,C2}.
\]

---

## Stage 5：历史 trace离线重算

比较：

- sign flip；
- magnitude；
- descendant count；
- gradient mass；
- prefix credit utilization；
- stable macro vs hypothetical strict leaf-uniform。

---

## Stage 6：checkpoint continuation

同一 late BACE checkpoint：

```text
C0
C1
C2
C3
```

只改变：

```text
tree_credit_mode
```

---

# 33. 实现 Checklist

## 共用

- [ ] `tree_credit_mode` enum；
- [ ] `macro_normalization_mode` enum；
- [ ] 当前 stable macro snapshot；
- [ ] TreeCreditIndex；
- [ ] unique concrete edge identity；
- [ ] Direct(e)；
- [ ] Desc(e)；
- [ ] branch-origin evidence sidecar；
- [ ] current GiGPO local normalizer支持 `G_override`。

## C1

- [ ] copied origin PPO mask=0；
- [ ] copied origin local group eligible=false；
- [ ] direct \(G\) backup；
- [ ] same `(z,u)` different origins不collapse；
- [ ] unique local regroup；
- [ ] macro使用 frozen stable base。

## C2

- [ ] natural edge descendant macro mean；
- [ ] fresh suffix macro unchanged；
- [ ] uniform descendant weights；
- [ ] C1/C2 local完全一致。

## C3

- [ ] generic \(G_{e,\ell}\) reconstruction；
- [ ] full descendant \(G\) mean；
- [ ] all natural edges可做 backup；
- [ ] fresh suffix \(G\) unchanged；
- [ ] C2/C3 macro完全一致。

## Regression

- [ ] C0 current production exact match；
- [ ] stable macro snapshot不漂移；
- [ ] strict leaf-uniform未意外启用；
- [ ] eval count unchanged；
- [ ] controller artifacts unchanged；
- [ ] topology unchanged；
- [ ] old logprob unchanged；
- [ ] PPO ratio unchanged；
- [ ] token mask unchanged；
- [ ] posterior update unchanged。

---

# 34. 三个版本的一句话正式定义

### C1 — O1-Local

> **保持当前 GiGPO-compatible stable macro normalization 不变；将 copied branch origin 从独立 PPO occurrence 改成 continuation evidence，同一个 concrete natural edge 的 original continuation 与所有 direct branch continuations 先平均 discounted return \(G\)，该 natural edge只训练一次，并以 unique-edge support重新计算 GiGPO local advantage。**

---

### C2 — O1-Tree-Macro

> **完整继承 C1；对每个 natural unique edge，将所有实际经过该 edge 的 terminal descendant leaves 的 frozen stable leaf-associated macro advantages 等权平均并赋给该 edge，从而让 branch outcome 的 trajectory-level information回传 shared prefix；local credit与 C1完全相同。**

---

### C3 — O1-Full-Tree

> **完整继承 C2；进一步对每个 natural unique edge计算所有 descendant leaves 从该 edge 开始的 discounted return-to-go，在 \(G\) 层等权平均后，以 unique-edge support重新构造 GiGPO anchor groups 与 local advantages；macro与 C2完全相同。**

---

# 35. 最容易犯的七个错误

1. **把 current occurrence-weighted stable macro 解释成方法理论目标。**  
   错。方法层面仍是 leaf-level semantics；occurrence weighting是当前稳定化实现。

2. **把 unique edge 写成 `(anchor_id, action_id)`。**  
   错。必须是 concrete occurrence identity。

3. **C1 把 later descendant branch也算入 Direct(e)。**  
   错。C1只看直接从 e发出的 branches。

4. **C2 重新计算 strict leaf-uniform macro mean/std。**  
   错。C2只对 frozen stable macro leaf signal做 descendant backup。

5. **C3 平均已经 normalize 的 local advantage。**  
   错。必须先平均 \(G\)，再重新做 local normalization。

6. **用相同 anchor key推断 descendant。**  
   错。必须用 root-step ancestry。

7. **copied origin不训练后就不更新 posterior。**  
   错。它仍然是有效 branch evaluation evidence。

---

# 36. 当前推荐解释

当前最清楚的论文与实现口径是：

\[
\boxed{
\text{Conceptually leaf-uniform trajectory credit}
}
\]

但：

\[
\boxed{
\text{Numerically retain GiGPO-compatible occurrence-weighted stabilization}
}
\]

在这个固定稳定实现上，再比较：

\[
\boxed{
C0\rightarrow C1\rightarrow C2\rightarrow C3
}
\]

三层 tree-credit mechanism。

因此第一阶段真正要回答的是：

\[
\boxed{
\text{branch evidence 应该如何在 rollout tree 中分配给 policy edges？}
}
\]

而不是同时回答：

\[
\boxed{
\text{macro normalization 是否应该严格 leaf-uniform？}
}
\]

后者应在 tree-credit winner确定后，作为单独 normalization ablation继续研究。
