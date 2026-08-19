# BACE-GiGPO：优势、Loss 与 Branch 分布更新实现规范

> 版本：2026-08-06  
> 适用范围：当前最新完整方案的代码实现、单元测试与实验记录  
> 主版本：**All-Leaf Unified Advantage + Copied Branch-Origin Training Occurrence + Unified PPO Loss**  
> 可选版本：**Unique-Edge Descendant Backup**

> **文档优先级**：当既有总体大纲或组件说明在 advantage、loss mask、copied origin、replay prefix 或 branch posterior 更新范围上存在旧表述时，以本文为当前实现规范。本文不改变 task prior、root--branch capacity correction、anchor structural validity 等其他已固定组件。

---

## 0. 文档目的

本文专门固定以下容易混淆的实现语义：

1. natural root、branch leaf、replay prefix、copied branch origin 和 fresh branch suffix 分别是什么；
2. 哪些对象是 terminal leaf，哪些对象是 trainable occurrence；
3. 每个 occurrence 使用哪一条 leaf 的 trajectory-level advantage；
4. replay prefix 中经过的已有 anchor 是否新增 action occurrence；
5. selected anchor 的 copied action 是否作为新 occurrence 并进入 loss；
6. branch suffix 中到达已有 anchor 或形成新 anchor 时如何处理；
7. 哪些 branch evidence 更新局部 Beta posterior、ERV action distribution 与 anchor utility；
8. 哪些集合在 branch phase 冻结，哪些统计量按 branch 顺序动态更新；
9. 统一 PPO loss 的 token mask、old log-probability 与 ratio 如何构造；
10. 如何实现“不新增 action occurrence，而把 descendant branch 的回报或优势平均到唯一自然 edge”的可选方案。

本文的目标是让代码中的每一个样本都能回答以下问题：

```text
它来自哪里？
它属于哪条 terminal leaf？
它是否是 policy draw？
它是否是 continuation evaluation？
它是否进入 posterior？
它是否进入 local anchor group？
它是否进入 PPO loss？
它使用哪个 advantage？
```

---

# 1. 总体设计结论

当前主版本采用：

$$
\boxed{
\text{all terminal leaves unified normalization}
+
\text{copied selected-origin occurrence}
+
\text{fresh suffix occurrences}
+
\text{one unified PPO loss}
}
$$

核心规则如下。

### 规则 1：所有 terminal leaves 统一计算全局轨迹优势

同一任务实例内的 natural root leaves 与 branch leaves 共同组成：

$$
\mathcal L_g
=
\mathcal L_g^{\mathrm{root}}
\cup
\mathcal L_g^{\mathrm{branch}}.
$$

每条 leaf 使用自己的终局回报，但共享同一个全局均值和标准差。

### 规则 2：replay prefix 只恢复环境

selected anchor 之前的动作重放：

- 不新增 policy occurrence；
- 不新增 training occurrence；
- 不进入 local anchor group；
- 不进入 PPO loss；
- 不更新对应 action posterior；
- 不增加 action count。

### 规则 3：selected anchor 的 copied action 是新的训练 occurrence

selected anchor 处复制原自然 CoT--action，并从该 edge 之后重新采样 continuation。因此它：

- 不是新的 natural policy draw；
- 是新的 direct continuation evaluation；
- 增加 selected pair 的 posterior evidence；
- 作为新的 branch-origin training occurrence；
- 使用对应 branch leaf 的 advantage；
- 进入统一 PPO loss。

### 规则 4：fresh branch suffix 是新的真实策略决策序列

branch boundary 之后由冻结策略新生成的每个动作：

- 是新的 policy occurrence；
- 是新的 training occurrence；
- 可以加入已有 training anchor；
- 可以形成新的 training anchor；
- 可以扩展最终 training action set；
- 但不能扩展当前 batch 已冻结的 branch-selection pool。

### 规则 5：branch controller 的结构池冻结，分布参数动态更新

branch phase 开始后冻结：

$$
\mathcal Z_g^{\mathrm{select}},
\qquad
\mathcal C_g^{\mathrm{select}}(z),
\qquad
\mathcal O_g^{\mathrm{root}}(z,u).
$$

每条 branch 后动态更新：

$$
(\alpha_{g,z,u},\beta_{g,z,u}),
\quad
\Delta\mathrm{ERV}_{g,b}(z,u),
\quad
\mu_{g,b}(u\mid z),
\quad
U_{g,b}(z).
$$

### 规则 6：可选方案不复制 shared edge 的 loss

可选 `Unique-Edge Descendant Backup` 中：

- selected copied origin 不作为额外 PPO occurrence；
- replay prefix 也不新增 occurrence；
- 对每个唯一自然 edge，聚合其所有 descendant leaves 的回报或优势；
- 该唯一 edge 只进入一次 loss。

---

# 2. 统一符号与数据对象

## 2.1 Natural root

任务实例 $g$ 的第 $r$ 条自然 root：

$$
\tau_{g,r}^{\mathrm{root}}
=
(e_{g,r,1},e_{g,r,2},\ldots,e_{g,r,T_r}),
$$

其中每个决策 edge 为：

$$
e_{g,r,t}
=
(h_{g,r,t},c_{g,r,t},u_{g,r,t},z_{g,r,t}).
$$

符号含义：

- $h$：模型在当前动作前的实际输入上下文；
- $c$：该 macro-action 对应的 CoT / reasoning response；
- $u$：canonical environment action；
- $z$：动作执行前的 canonical anchor key。

自然 root 的 terminal return：

$$
R_{g,r}^{\mathrm{root}}.
$$

---

## 2.2 Selected anchor 与 concrete origin

branch controller 选择：

$$
z_b^*\in\mathcal Z_g^{\mathrm{select}},
$$

随后从动作分布中选择：

$$
u_b^*\sim\mu_{g,b}(\cdot\mid z_b^*).
$$

在冻结的自然 origin pool 中，选择一个确实执行过 $u_b^*$ 的 concrete natural occurrence：

$$
i_b^*
\sim
\operatorname{Uniform}
\left(
\mathcal O_g^{\mathrm{root}}(z_b^*,u_b^*)
\right).
$$

该 natural occurrence 保存：

$$
e_b^*
=
(h_b^*,c_b^*,u_b^*,z_b^*).
$$

---

## 2.3 Replay prefix

从任务初始状态恢复到 selected anchor 所重放的环境动作序列：

$$
P_b
=
(a_1,\ldots,a_{t_b-1}).
$$

它只用于：

```text
load same task/game
reset environment
replay saved environment actions
verify restored anchor state
```

它不是一组新的模型生成记录。

---

## 2.4 Copied branch origin

恢复到 $z_b^*$ 后，复制 concrete origin 的完整 response：

$$
(c_b^*,u_b^*).
$$

然后在环境中执行 $u_b^*$，并从动作后 observation 开始由冻结策略生成 fresh continuation。

定义 copied branch-origin occurrence：

$$
o_b^{\mathrm{origin}}
=
(g,b,z_b^*,h_b^*,c_b^*,u_b^*,\ell_b).
$$

它与原 natural occurrence 具有相同的：

$$
(h,c,u),
$$

但属于新的 branch leaf $\ell_b$，并具有新的 continuation return。

---

## 2.5 Fresh branch suffix

执行 copied origin 后，由冻结策略生成：

$$
\xi_b
=
(e_{b,1}^{\mathrm{suffix}},\ldots,e_{b,T_b}^{\mathrm{suffix}}).
$$

suffix 中每一个 edge 都是新的 policy draw：

$$
e_{b,t}^{\mathrm{suffix}}
=
(h_{b,t},c_{b,t},u_{b,t},z_{b,t}).
$$

branch terminal return：

$$
R_{g,b}^{\mathrm{branch}}.
$$

---

## 2.6 概念上的完整 branch leaf

完整 branch leaf 为：

$$
\tau_{g,b}^{\mathrm{leaf}}
=
P_b
\oplus
(c_b^*,u_b^*)
\oplus
\xi_b.
$$

它用于：

- terminal leaf 计数；
- terminal return；
- 全 leaf trajectory normalization；
- 轨迹可视化和 tree bookkeeping。

但其 replay prefix 不自动进入训练 buffer。

---

# 3. 三类 occurrence 计数

为了避免把“动作出现”混为一个概念，代码必须维护三种计数。

## 3.1 Policy-draw count

$$
n_{g,z,u}^{\mathrm{policy}}
=
\text{actor 在状态 }z\text{ 自然生成动作 }u\text{ 的次数}.
$$

计入：

- natural root actions；
- fresh branch suffix actions。

不计入：

- replay prefix actions；
- copied selected-origin action。

---

## 3.2 Continuation-evaluation count

$$
n_{g,z,u}^{\mathrm{eval}}
=
\text{该 state--action edge 获得 terminal outcome evaluation 的次数}.
$$

计入：

- natural root occurrence；
- selected direct-resample occurrence；
- eligible fresh suffix occurrence。

selected copied origin 会增加：

$$
n_{g,z_b^*,u_b^*}^{\mathrm{eval}}
\leftarrow
n_{g,z_b^*,u_b^*}^{\mathrm{eval}}+1.
$$

---

## 3.3 Training-occurrence count

$$
n_{g,z,u}^{\mathrm{train}}
=
\text{进入最终 advantage group 与 PPO buffer 的 occurrence 数}.
$$

主版本计入：

- natural root occurrences；
- copied branch-origin occurrences；
- fresh branch suffix occurrences。

不计入 replay prefix occurrences。

因此 selected copied origin 满足：

$$
\boxed{
\Delta n^{\mathrm{policy}}=0,
\qquad
\Delta n^{\mathrm{eval}}=1,
\qquad
\Delta n^{\mathrm{train}}=1.
}
$$

---

# 4. Terminal leaf 集合与全局轨迹优势

## 4.1 Leaf 集合

对任务实例 $g$：

$$
\mathcal L_g
=
\mathcal L_g^{\mathrm{root}}
\cup
\mathcal L_g^{\mathrm{branch}}.
$$

满足：

$$
|\mathcal L_g|
=
N_g^{\mathrm{root}}
+
N_g^{\mathrm{branch}}.
$$

一条 natural root 即一条 root leaf；每完成一条 branch，即新增一条 branch leaf。

---

## 4.2 Leaf-uniform trajectory normalization

全局均值：

$$
\mu_g^E
=
\frac{1}{|\mathcal L_g|}
\sum_{\ell\in\mathcal L_g}R_{g,\ell}.
$$

全局标准差：

$$
\sigma_g^E
=
\operatorname{Std}
\left(
\{R_{g,\ell}:\ell\in\mathcal L_g\}
\right).
$$

每条 leaf 的轨迹级优势：

$$
\boxed{
A_{g,\ell}^{E}
=
\frac{R_{g,\ell}-\mu_g^E}
{\sigma_g^E+\epsilon_{\mathrm{norm}}}.
}
$$

若：

$$
\sigma_g^E=0,
$$

则：

$$
A_{g,\ell}^{E}=0,
\qquad
\forall\ell\in\mathcal L_g.
$$

---

## 4.3 “统一归一化”不等于“共享 prefix 取 descendant 平均”

设一条 natural root 和两条 branch leaves：

$$
\ell_0:P\oplus e^*\oplus\xi_0\rightarrow R_0,
$$

$$
\ell_1:P\oplus e^*\oplus\xi_1\rightarrow R_1,
$$

$$
\ell_2:P\oplus e^*\oplus\xi_2\rightarrow R_2.
$$

主版本分别得到：

$$
A_0^E,
\qquad
A_1^E,
\qquad
A_2^E.
$$

“统一”的含义是三条 leaves 共用：

$$
\mu_g^E,
\qquad
\sigma_g^E.
$$

主版本中 natural prefix $P$ 仍属于 $\ell_0$ 的自然 occurrence，只使用：

$$
A_0^E.
$$

它不使用：

$$
\frac{A_0^E+A_1^E+A_2^E}{3}.
$$

后者属于第 13 节的可选 descendant-backup 方案。

---

# 5. 主版本的 trainable occurrence map

定义最终训练集合：

$$
\boxed{
\mathcal D_g
=
\mathcal D_g^{\mathrm{root}}
\cup
\mathcal D_g^{\mathrm{branch\text{-}origin}}
\cup
\mathcal D_g^{\mathrm{branch\text{-}suffix}}.
}
$$

## 5.1 Natural root occurrences

每个 natural edge 保留一次：

$$
o_j^{\mathrm{root}}
=
(g,\ell_r,z_j,h_j,c_j,u_j,G_j,\texttt{source=root}).
$$

其 leaf 映射：

$$
\ell(j)=\ell_r.
$$

---

## 5.2 Copied branch-origin occurrences

每条 branch 新增一个：

$$
o_b^{\mathrm{origin}}
=
(g,\ell_b,z_b^*,h_b^*,c_b^*,u_b^*,G_b,\texttt{source=branch\_origin}).
$$

其 leaf 映射：

$$
\ell(o_b^{\mathrm{origin}})=\ell_b.
$$

在 terminal binary reward 且 $\gamma=1$ 时：

$$
G_b=R_{g,\ell_b}.
$$

---

## 5.3 Fresh suffix occurrences

每个 suffix edge形成：

$$
o_{b,t}^{\mathrm{suffix}}
=
(g,\ell_b,z_{b,t},h_{b,t},c_{b,t},u_{b,t},G_{b,t},\texttt{source=branch\_suffix}).
$$

它同样属于 branch leaf：

$$
\ell(o_{b,t}^{\mathrm{suffix}})=\ell_b.
$$

---

## 5.4 Replay prefix 不进入 occurrence map

replay prefix 只能存为环境恢复日志：

```text
source = replay_prefix
train_mask = 0
posterior_eligible = false
local_group_eligible = false
```

不得插入：

$$
\mathcal D_g,
\qquad
\mathcal I_g(z),
\qquad
\mathcal I_g(z,u).
$$

---

# 6. Local anchor advantage

## 6.1 Main：occurrence-level GiGPO local advantage

对最终 trainable occurrence map 按状态分组：

$$
\mathcal I_g(z)
=
\{j\in\mathcal D_g:z_j=z\}.
$$

状态均值：

$$
\mu_{g,z}^{S}
=
\frac{1}{|\mathcal I_g(z)|}
\sum_{j\in\mathcal I_g(z)}G_j.
$$

状态标准差：

$$
\sigma_{g,z}^{S}
=
\operatorname{Std}
\left(
\{G_j:j\in\mathcal I_g(z)\}
\right).
$$

occurrence-level local advantage：

$$
\boxed{
A_j^{S,\mathrm{occ}}
=
\frac{G_j-\mu_{g,z_j}^{S}}
{\sigma_{g,z_j}^{S}+\epsilon_{\mathrm{norm}}}.
}
$$

若：

$$
|\mathcal I_g(z)|<2
\quad\text{或}\quad
\sigma_{g,z}^{S}=0,
$$

则：

$$
A_j^{S,\mathrm{occ}}=0.
$$

---

## 6.2 Optional：action-aggregated local advantage

对 canonical action 分组：

$$
\mathcal I_g(z,u)
=
\{j\in\mathcal I_g(z):u_j=u\}.
$$

动作均值：

$$
\overline G_{g,z,u}
=
\frac{1}{|\mathcal I_g(z,u)|}
\sum_{j\in\mathcal I_g(z,u)}G_j.
$$

动作级 local advantage：

$$
\boxed{
A_{g,z,u}^{S,\mathrm{act}}
=
\frac{\overline G_{g,z,u}-\mu_{g,z}^{S}}
{\sigma_{g,z}^{S}+\epsilon_{\mathrm{norm}}}.
}
$$

然后广播给该动作的所有 training occurrences：

$$
A_j^S=A_{g,z_j,u_j}^{S,\mathrm{act}}.
$$

注意：该版本只聚合 local term，不聚合每条 leaf 的全局 trajectory advantage。

---

# 7. 最终 advantage 赋值

对任意 trainable occurrence $j$：

$$
\boxed{
A_j
=
A_{g,\ell(j)}^{E}
+
\omega A_j^{S}.
}
$$

其中 $A_j^S$ 可取：

- 主版本：$A_j^{S,\mathrm{occ}}$；
- 可选动作聚合：$A_{g,z_j,u_j}^{S,\mathrm{act}}$。

---

## 7.1 Natural root prefix

natural root 中选中 anchor 之前的真实 occurrence：

$$
j\in P_r
$$

使用：

$$
A_j
=
A_{g,\ell_r}^{E}
+
\omega A_j^S.
$$

即使用原 natural root leaf 的轨迹优势。

branch descendants 只通过共同的：

$$
\mu_g^E,
\quad
\sigma_g^E
$$

间接影响该轨迹优势，不直接替换其 numerator。

---

## 7.2 Natural selected-origin occurrence

原 natural edge $e^*$ 属于 root leaf $\ell_r$：

$$
A_{e^*}^{\mathrm{natural}}
=
A_{g,\ell_r}^{E}
+
\omega A_{e^*}^{S}.
$$

---

## 7.3 Copied branch-origin occurrence

第 $b$ 条 branch 的 copied origin 使用 branch leaf：

$$
\boxed{
A_{b}^{\mathrm{origin}}
=
A_{g,\ell_b}^{E}
+
\omega A_{b}^{S}.
}
$$

即使其 $(h,c,u)$ 与自然 occurrence 完全相同，也必须使用自己的 branch return 和 branch leaf advantage。

---

## 7.4 Fresh suffix occurrence

branch suffix 中每个 occurrence：

$$
A_{b,t}^{\mathrm{suffix}}
=
A_{g,\ell_b}^{E}
+
\omega A_{b,t}^{S}.
$$

若当前位置没有形成有效 local group：

$$
A_{b,t}^{S}=0,
$$

则退化为：

$$
A_{b,t}^{\mathrm{suffix}}
=
A_{g,\ell_b}^{E}.
$$

---

# 8. 三类 anchor 位置的主版本处理

考虑：

$$
z_1\xrightarrow{a_1}
z_2\xrightarrow{a_2}
z^*\xrightarrow{u^*}
\text{fresh suffix}.
$$

## 8.1 Selected anchor 之前：$z_1,z_2$

replay 时虽然环境再次执行 $a_1,a_2$，但：

$$
\boxed{
\Delta n^{\mathrm{policy}}=0,
\quad
\Delta n^{\mathrm{eval}}=0,
\quad
\Delta n^{\mathrm{train}}=0.
}
$$

因此：

- 不增加动作次数；
- 不新增 local occurrence；
- 不将 branch return 加入其动作均值；
- 不更新其 Beta posterior；
- 不为 replay response 计算 PPO loss。

若 $z_1$ 原来有：

$$
(a_1,R_0),
\quad
(a_2,R_3),
$$

branch 得到 $R_b$ 后，主版本 local group 仍为：

$$
\mathcal I_g(z_1)
=
\{(a_1,R_0),(a_2,R_3)\}.
$$

而不是：

$$
\{(a_1,R_0),(a_1,R_b),(a_2,R_3)\}.
$$

---

## 8.2 Selected anchor：$z^*$

copied selected action $u^*$：

$$
\boxed{
\Delta n^{\mathrm{policy}}=0,
\quad
\Delta n^{\mathrm{eval}}=1,
\quad
\Delta n^{\mathrm{train}}=1.
}
$$

它：

- 更新 selected pair posterior；
- 进入 final training anchor group；
- 计算 branch-specific $A^E$ 与 $A^S$；
- 进入统一 PPO loss。

若 natural anchor 原来为：

$$
[a_1,a_2],
$$

选择 $a_1$ branch 一次后，training group 为：

$$
[a_1^{\mathrm{natural}},
 a_1^{\mathrm{direct\text{-}resample}},
 a_2^{\mathrm{natural}}].
$$

canonical action set仍为：

$$
\{a_1,a_2\},
$$

没有新增第三种动作类型。

---

## 8.3 Selected anchor 之后：fresh suffix

fresh suffix 中的每个 actor decision：

$$
\boxed{
\Delta n^{\mathrm{policy}}=1,
\quad
\Delta n^{\mathrm{eval}}=1,
\quad
\Delta n^{\mathrm{train}}=1.
}
$$

它可以：

1. 加入已有 training anchor；
2. 与 root occurrence 形成 root--branch collision；
3. 与其他 branch 形成 branch--branch collision；
4. 在同一状态执行新 canonical action，扩展最终 training action set；
5. 单独出现时仍进入 PPO，只是 local advantage 为零。

但它不能在当前 batch 中：

- 新增 selectable anchor；
- 新增 selectable action；
- 成为新的 branch origin；
- 触发 branch-of-branch。


# 9. Branch acquisition posterior 与分布更新

## 9.1 冻结的 controller support

完成 root-side capacity correction 后，正式进入 branch phase，并冻结：

### Selectable anchor set

$$
\mathcal Z_g^{\mathrm{select}}
=
\mathcal Z_g^{\mathrm{eff}}.
$$

### Selectable action set

$$
\mathcal C_g^{\mathrm{select}}(z),
\qquad
z\in\mathcal Z_g^{\mathrm{select}}.
$$

候选动作只来自 frozen natural-root observed actions。

### Concrete natural-origin pool

$$
\mathcal O_g^{\mathrm{root}}(z,u)
=
\{i:\text{natural root occurrence }i\text{ 在 }z\text{ 执行 }u\}.
$$

branch phase 中不把 branch-generated occurrence 加入该 origin pool。

### Per-anchor branch capacity

$$
n_g^{\mathrm{br}}(z)<L_{\max}.
$$

---

## 9.2 Dynamic local Beta posterior

对每个 frozen pair：

$$
(z,u),
\qquad
z\in\mathcal Z_g^{\mathrm{select}},
\quad
u\in\mathcal C_g^{\mathrm{select}}(z),
$$

维护：

$$
p_{g,z,u}\mid\mathcal D_b
\sim
\operatorname{Beta}
(\alpha_{g,z,u}^{(b)},
 \beta_{g,z,u}^{(b)}).
$$

当一个 eligible outcome 为 $Y\in\{0,1\}$ 时：

$$
\alpha_{g,z,u}
\leftarrow
\alpha_{g,z,u}+Y,
$$

$$
\beta_{g,z,u}
\leftarrow
\beta_{g,z,u}+1-Y.
$$

---

## 9.3 哪些 branch evidence 更新 posterior

### A. Selected direct-resample pair：必须更新

当前 branch 选择：

$$
(z_b^*,u_b^*).
$$

branch terminal outcome 为 $Y_b$，则必定更新：

$$
(\alpha_{g,z_b^*,u_b^*},
 \beta_{g,z_b^*,u_b^*}).
$$

这是该 branch 的主要 acquisition target。

---

### B. Fresh suffix 中匹配 frozen existing pair：更新

若 fresh suffix 真正到达：

$$
w\in\mathcal Z_g^{\mathrm{select}},
$$

并由 actor 新生成：

$$
v\in\mathcal C_g^{\mathrm{select}}(w),
$$

则该真实 state--action occurrence 可以用同一 branch terminal outcome更新：

$$
(\alpha_{g,w,v},\beta_{g,w,v}).
$$

理由：

- 该动作由 actor 在 fresh suffix 中真实生成；
- 从该 edge 之后 continuation 是 fresh 的；
- frozen policy 在整个 acquisition batch 内不变；
- 它属于已经定义好的 frozen posterior support。

为避免 loop 重复导致同一 leaf 过度收缩，主实现规定：

$$
\boxed{
\text{一条 branch leaf 对同一个 frozen }(w,v)
\text{ 最多贡献一次 posterior update。}
}
$$

最终 training occurrence map仍可保留全部真实 loop occurrences；该去重只作用于 acquisition posterior。

---

### C. Fresh suffix 的新 action：不更新当前 controller posterior

若：

$$
v\notin\mathcal C_g^{\mathrm{select}}(w),
$$

则：

- 进入最终 training action group；
- 进入 local advantage 与 PPO loss；
- 不扩展当前 branch candidate support；
- 不在当前 batch 初始化新的 ERV candidate posterior。

---

### D. Fresh suffix 的新 anchor：不加入当前 controller

若：

$$
w\notin\mathcal Z_g^{\mathrm{select}},
$$

即使它在最终 tree 中形成 repeated training anchor，也只用于：

- final local advantage；
- training diagnostics；
- PPO update。

不用于当前 batch 的后续 branch selection。

---

### E. Mechanical replay prefix：不更新

对 selected anchor 之前重放的任何：

$$
(z,u),
$$

均不更新 posterior，因为从该 edge 后仍固定了一段中间路径，它不是该 pair 的 direct fresh continuation sample。

---

## 9.4 ERV 更新

对每个 frozen selectable pair，在 posterior 更新后重新计算：

$$
\Delta\operatorname{ERV}_{g,b}(z,u).
$$

其定义为：

$$
\Delta\operatorname{ERV}_{g,b}(z,u)
=
\mathcal R_{g,b}(z)
-
\mathbb E_Y
\left[
\mathcal R_{g,b}^{+}(z;u,Y)
\right].
$$

每条 branch 完成后，所有 posterior 被修改的 anchors 必须重新计算 ERV；工程上也可以直接批量重算全部 frozen selectable anchors，避免遗漏依赖。

---

## 9.5 Action acquisition distribution 更新

对固定 anchor $z$，主版本使用：

$$
\boxed{
\mu_{g,b}(u\mid z)
=
\frac{
\exp
\left(
\Delta\operatorname{ERV}_{g,b}(z,u)/\tau_\mu
\right)
}{
\sum_{v\in\mathcal C_g^{\mathrm{select}}(z)}
\exp
\left(
\Delta\operatorname{ERV}_{g,b}(z,v)/\tau_\mu
\right)
}.
}
$$

候选集合冻结，但其概率随 posterior 顺序更新。

如果实现中保留 frequency base measure，可写为可选形式：

$$
\mu_{g,b}^{\mathrm{freq}}(u\mid z)
\propto
q_g^{\mathrm{freq}}(u\mid z)
\exp
\left(
\Delta\operatorname{ERV}_{g,b}(z,u)/\tau_\mu
\right).
$$

该版本必须单独标记，不与纯 ERV-softmax 主版本混用。

---

## 9.6 Anchor utility 更新

定义：

$$
\boxed{
U_{g,b}(z)
=
\sum_{u\in\mathcal C_g^{\mathrm{select}}(z)}
\mu_{g,b}(u\mid z)
\Delta\operatorname{ERV}_{g,b}(z,u).
}
$$

下一条 branch 选择：

$$
z_b^*
=
\arg\max_{
\substack{
z\in\mathcal Z_g^{\mathrm{select}},\\
n_g^{\mathrm{br}}(z)<L_{\max}
}
}
U_{g,b}(z).
$$

然后：

$$
u_b^*
\sim
\mu_{g,b}(\cdot\mid z_b^*).
$$

branch 完成后：

$$
n_g^{\mathrm{br}}(z_b^*)
\leftarrow
n_g^{\mathrm{br}}(z_b^*)+1.
$$

---

## 9.7 Branch 后哪些“分布”会变化

| 对象 | Branch 后是否更新 | 更新来源 |
|---|---:|---|
| Task-family competence prior | 否 | 只由滞后 natural root history 更新 |
| Current instance competence posterior | 否 | root/branch topology 已冻结 |
| Selectable anchor set | 否 | root phase 后冻结 |
| Selectable action set | 否 | root phase 后冻结 |
| Natural origin pool | 否 | 只包含 natural occurrences |
| Selected pair Beta posterior | 是 | 当前 branch outcome |
| Frozen existing pair posterior in fresh suffix | 是 | eligible fresh suffix outcome |
| New suffix action posterior | 否 | 仅进入 training group |
| New suffix anchor posterior | 否 | 仅进入 training group |
| ERV scores | 是 | posterior 更新后重算 |
| Action distribution $\mu(u\mid z)$ | 是 | ERV 更新后重算 |
| Anchor utility $U(z)$ | 是 | $\mu$ 与 ERV 更新后重算 |
| All-leaf trajectory statistics | 最终统一计算 | 所有 branches 完成后 |
| Final training anchor groups | 最终重建 | roots + copied origins + fresh suffixes |

---

# 10. PPO ratio、token mask 与统一 loss

## 10.1 Old log-probability 来源

### Natural root

直接使用 rollout 时保存的：

$$
\log\pi_{\mathrm{old}}(y_{j,k}\mid h_{j,k}).
$$

### Fresh suffix

直接使用 suffix generation 时保存的 old log-probability。

### Copied branch origin

由于：

- exact history 相同；
- exact CoT--action response 相同；
- batch 内 $\pi_{\mathrm{old}}$ 冻结；

所以复用 concrete natural origin 保存的 old log-probability：

$$
\log\pi_{\mathrm{old}}^{\mathrm{copied}}
=
\log\pi_{\mathrm{old}}^{\mathrm{natural}}.
$$

不得把环境 action string 的频率概率当作 token-level PPO denominator。

---

## 10.2 Standard PPO ratio

对每个 trainable response token：

$$
\boxed{
\rho_{j,k}(\theta)
=
\frac{
\pi_\theta(y_{j,k}\mid h_{j,k})
}{
\pi_{\mathrm{old}}(y_{j,k}\mid h_{j,k})
}.
}
$$

主版本不额外乘：

$$
\frac{\pi_{\mathrm{old}}(u\mid z)}{\mu_{g,b}(u\mid z)}.
$$

因此 branch objective 被明确解释为 acquisition-weighted objective，而不是对 natural rollout distribution 的完全无偏重建。

---

## 10.3 Token mask

| Token / segment | Mask |
|---|---:|
| Natural root prompt/history | $0$ |
| Natural root response CoT/action tokens | $1$ |
| Replay prefix prompt/response/environment history | $0$ |
| Copied branch-origin response CoT/action tokens | $1$ |
| Fresh branch suffix prompt/history | $0$ |
| Fresh branch suffix response CoT/action tokens | $1$ |
| Padding | $0$ |

如果后续实验选择只训练 action tokens，应作为独立 mask 消融，不与主版本默认逻辑混写。

---

## 10.4 PPO clipped surrogate

$$
\ell_{j,k}^{\mathrm{clip}}(\theta)
=
\min
\left(
\rho_{j,k}(\theta)A_j,
\operatorname{clip}
(\rho_{j,k}(\theta),1-\epsilon,1+\epsilon)A_j
\right).
$$

---

## 10.5 Unified token-mean PPO loss

主版本将全部 trainable occurrences 统一计算：

$$
\boxed{
\mathcal L_{\mathrm{PPO}}
=
-
\frac{
\sum_{j\in\mathcal D}
\sum_k
m_{j,k}
\ell_{j,k}^{\mathrm{clip}}(\theta)
}{
\sum_{j\in\mathcal D}
\sum_km_{j,k}
}.
}
$$

其中：

$$
\mathcal D
=
\bigcup_g\mathcal D_g.
$$

如果使用 reference KL：

$$
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+
\beta_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
$$

主版本不再把 root loss 与 branch loss 写成两个独立加权项；branch 权重由实际 trainable tokens 与主动 acquisition 自然体现。

---

## 10.6 Copied origin 的梯度语义

若同一个 exact response edge被一条 natural root 和 $K$ 条 branches 使用，则主版本有：

$$
A_e^{(0)},A_e^{(1)},\ldots,A_e^{(K)}.
$$

忽略 clipping，其合计梯度约为：

$$
\sum_{b=0}^{K}
A_e^{(b)}
\nabla_\theta
\log\pi_\theta(c_e,u_e\mid h_e).
$$

它既利用多个 continuation outcomes，也使该 selected edge 获得更高 acquisition-weighted gradient mass。

该效果是主版本有意接受的，但必须通过：

- $L_{\max}$；
- PPO clipping；
- reference KL；
- branch-origin gradient diagnostics；

控制其强度。

---

# 11. 一个完整数值例子

## 11.1 Tree 结构

假设一条 natural root：

$$
z_1\xrightarrow{a_1}
z_2\xrightarrow{a_2}
z^*\xrightarrow{u_1}
\xi_0
\rightarrow R_0=0.
$$

另一个 natural occurrence 在 $z^*$ 执行：

$$
u_2
\rightarrow R_3=0.
$$

对 $(z^*,u_1)$ 进行两次 branch：

$$
u_1\rightarrow\xi_1\rightarrow R_1=1,
$$

$$
u_1\rightarrow\xi_2\rightarrow R_2=1.
$$

终局 leaves 为：

$$
\{R_0,R_1,R_2,R_3\}
=
\{0,1,1,0\}.
$$

于是：

$$
\mu^E=0.5,
\qquad
\sigma^E=0.5,
$$

$$
A^E
=
[-1,+1,+1,-1].
$$

---

## 11.2 Prefix $z_1$ 的主版本处理

假设 $z_1$ 原 natural group 为：

$$
(a_1,R_0=0),
\qquad
(a_3,R_4=1).
$$

两条 branches 的 replay 都经过 $z_1,a_1$，但主版本不增加 occurrence，因此：

$$
\mathcal I(z_1)
=
\{(a_1,0),(a_3,1)\}.
$$

不是：

$$
\{(a_1,0),(a_1,1),(a_1,1),(a_3,1)\}.
$$

natural $a_1$ occurrence 的全局项仍为：

$$
A_0^E=-1.
$$

---

## 11.3 Selected anchor $z^*$ 的主版本处理

最终 training group：

| Occurrence | Action | Return | Leaf advantage |
|---|---|---:|---:|
| Natural origin | $u_1$ | $0$ | $-1$ |
| Branch origin 1 | $u_1$ | $1$ | $+1$ |
| Branch origin 2 | $u_1$ | $1$ | $+1$ |
| Other natural origin | $u_2$ | $0$ | $-1$ |

occurrence-level版本分别计算各自 local advantage。

若使用 action-aggregated版本：

$$
\overline G(z^*,u_1)
=
\frac{0+1+1}{3}
=
\frac{2}{3},
$$

$$
\overline G(z^*,u_2)=0.
$$

三个 $u_1$ occurrences 共享同一个 local term，但其全局项仍分别为：

$$
-1,+1,+1.
$$

---

## 11.4 Posterior 更新

两条 selected branches 使：

$$
\alpha_{z^*,u_1}
\leftarrow
\alpha_{z^*,u_1}+2,
$$

$$
\beta_{z^*,u_1}
\text{ unchanged}.
$$

如果第一条 branch suffix 还真实到达 frozen anchor $w$，并执行 frozen candidate $v$，则同一 branch outcome还可以对：

$$
(\alpha_{w,v},\beta_{w,v})
$$

贡献一次 eligible posterior update。

但 replay prefix 中的：

$$
(z_1,a_1),
\quad
(z_2,a_2)
$$

不更新 posterior。

---

# 12. 主版本的实现伪代码

```text
Input:
    frozen old policy pi_old
    natural root rollouts
    frozen selectable anchors Z_select
    frozen candidate actions C_select(z)
    frozen natural origin pools O_root(z,u)
    branch quota Q_g
    per-anchor cap L_max

# A. Initialize controller posterior from natural evidence
for each frozen pair (z,u):
    initialize Beta(alpha[z,u], beta[z,u])

# B. Sequential branch acquisition
for b in 1 ... Q_g:

    for z in Z_select with branch_count[z] < L_max:
        compute DeltaERV[z,u] for all u in C_select(z)
        mu_action[u|z] <- softmax(DeltaERV[z,u] / tau_mu)
        U[z] <- sum_u mu_action[u|z] * DeltaERV[z,u]

    z_star <- argmax_z U[z]
    u_star <- sample(mu_action[.|z_star])
    origin <- uniform_sample(O_root[z_star,u_star])

    env <- reset_same_task(origin.task)
    replay origin.environment_action_prefix
    assert restored_state_key == z_star
    assert u_star is executable

    copy origin.COT_action_response
    execute u_star
    generate fresh suffix with pi_old
    observe terminal outcome Y_b

    create branch leaf ell_b
    store one copied branch-origin training occurrence
    store all fresh suffix training occurrences
    store replay prefix only as recovery metadata with train_mask=0

    # Mandatory selected-pair posterior update
    update_beta(z_star, u_star, Y_b)

    # Opportunistic updates from fresh suffix
    seen_pairs <- empty set
    for fresh suffix occurrence (w,v):
        if w in Z_select and v in C_select(w):
            if (w,v) not in seen_pairs:
                update_beta(w, v, Y_b)
                add (w,v) to seen_pairs

    branch_count[z_star] += 1

# C. Final advantage construction
L_g <- all natural root leaves + all branch leaves
compute unified leaf A_E[ell]

D_g <- natural occurrences
       + copied branch-origin occurrences
       + fresh suffix occurrences

build I_g(z) only from D_g
compute occurrence-level or action-aggregated local A_S

for j in D_g:
    A[j] <- A_E[leaf_of(j)] + omega * A_S[j]

# D. PPO buffer
for j in D_g:
    use response-token mask
    use saved old log-probabilities
    copied origin reuses natural origin old log-probabilities

compute one unified token-mean PPO loss
```

---

# 13. 可选方案：Unique-Edge Descendant Backup

## 13.1 动机

主版本将 selected copied origin 作为额外 training occurrence，因此 selected edge 的 PPO weight 会随 branch 数增加。

可选方案希望实现：

$$
\boxed{
\text{多次 branch 只改善同一 edge 的价值估计，
不增加其直接 PPO occurrence 数。}
}
$$

同时，对 selected anchor 之前的共享 prefix，也允许 descendant branch outcomes 影响其信用，但不把机械 replay 伪装成新的动作采样。

---

## 13.2 Unique natural edge

对每个 natural root 中的唯一决策 edge：

$$
e=(h_e,c_e,u_e,z_e),
$$

定义以该 edge 为祖先的 terminal leaves：

$$
\operatorname{Desc}(e)
=
\{\ell:e\text{ 位于 }\ell\text{ 的概念完整路径上}\}.
$$

例如：

$$
\operatorname{Desc}(e)
=
\{\ell_0,\ell_1,\ell_2\}.
$$

其中 $\ell_0$ 是原 natural root leaf，$\ell_1,\ell_2$ 是后代 branches。

---

## 13.3 Descendant return backup

最简单的等权形式：

$$
\boxed{
\widetilde G_e
=
\frac{1}{|\operatorname{Desc}(e)|}
\sum_{\ell\in\operatorname{Desc}(e)}
G_{e,\ell}.
}
$$

在 terminal binary reward 且 $\gamma=1$ 时：

$$
G_{e,\ell}=R_{\ell}.
$$

因此：

$$
\widetilde G_e
=
\frac{1}{|\operatorname{Desc}(e)|}
\sum_{\ell\in\operatorname{Desc}(e)}R_{\ell}.
$$

若 original root 与两条 branch returns 为：

$$
[0,1,1],
$$

则：

$$
\widetilde G_e=\frac{2}{3}.
$$

---

## 13.4 Descendant trajectory-advantage backup

也可以直接聚合全 leaf trajectory advantages：

$$
\boxed{
\widetilde A_e^E
=
\frac{
\sum_{\ell\in\operatorname{Desc}(e)}
\eta_{e,\ell}A_{g,\ell}^{E}
}{
\sum_{\ell\in\operatorname{Desc}(e)}
\eta_{e,\ell}
}.
}
$$

等权时：

$$
\eta_{e,\ell}=1.
$$

因为所有 $A_{g,\ell}^{E}$ 使用相同的 $\mu_g^E$ 和 $\sigma_g^E$，等权平均 advantage 等价于对 descendant return mean 做同一标准化：

$$
\widetilde A_e^E
=
\frac{
\widetilde R_e-\mu_g^E
}{
\sigma_g^E+\epsilon
},
$$

其中：

$$
\widetilde R_e
=
\frac{1}{|\operatorname{Desc}(e)|}
\sum_{\ell\in\operatorname{Desc}(e)}R_{\ell}.
$$

---

## 13.5 Optional variant 的 occurrence 规则

在该方案中：

### Natural edge

每个 unique natural edge只保留一个 training occurrence。

### Copied selected origin

不新增 PPO occurrence：

$$
\Delta n_{z^*,u^*}^{\mathrm{train}}=0.
$$

但仍增加 continuation evaluation 与 posterior evidence：

$$
\Delta n_{z^*,u^*}^{\mathrm{eval}}=1.
$$

### Replay prefix

同样不新增 occurrence。

### Fresh suffix

仍然是新的 policy occurrences，正常进入 local groups和 PPO loss。

---

## 13.6 Optional variant 的 local advantage

用 unique edges 构造：

$$
\mathcal E_g(z)
=
\{e:\text{unique natural or fresh policy edge at }z\}.
$$

对 shared natural edge 使用 descendant-backed return：

$$
\widetilde G_e.
$$

状态均值：

$$
\widetilde\mu_{g,z}^{S}
=
\frac{1}{|\mathcal E_g(z)|}
\sum_{e\in\mathcal E_g(z)}
\widetilde G_e.
$$

局部优势：

$$
\boxed{
\widetilde A_e^S
=
\frac{
\widetilde G_e-\widetilde\mu_{g,z}^{S}
}{
\widetilde\sigma_{g,z}^{S}+\epsilon
}.
}
$$

最终唯一 edge 的优势：

$$
\boxed{
\widetilde A_e
=
\widetilde A_e^E
+
\omega\widetilde A_e^S.
}
$$

该 edge 只进入一次 PPO loss。

---

## 13.7 Distance-weighted ancestor backup

越早的 ancestor 与 selected branch point 距离越远，branch outcome 越是 path-conditioned。可使用：

$$
\lambda_e
=
\lambda_{\mathrm{up}}^{d(e,z^*)},
\qquad
0\le\lambda_{\mathrm{up}}\le1.
$$

若 edge $e$ 的 original root return 为 $G_e^{\mathrm{root}}$，共有 $K$ 条 descendant branches，则：

$$
\boxed{
\widetilde G_e
=
\frac{
G_e^{\mathrm{root}}
+
\lambda_e
\sum_{b=1}^{K}G_{e,b}^{\mathrm{branch}}
}{
1+\lambda_e K
}.
}
$$

边界：

- $\lambda_{\mathrm{up}}=0$：退化为主版本对 pre-anchor prefix 的处理；
- $\lambda_{\mathrm{up}}=1$：所有 descendants 等权 full backup；
- selected edge 的 $d=0$，branch evidence 权重最大。

---

## 13.8 Optional variant 不改变哪些 controller 规则

即使采用 descendant backup，以下规则不变：

- task prior 不由 branches 更新；
- branch-selection anchor/action support冻结；
- selected pair posterior仍更新；
- eligible fresh suffix frozen pairs仍可更新；
- pre-anchor replay不直接更新 Beta posterior；
- branch-of-branch仍禁止；
- terminal leaves仍统一计算全局 statistics。

---

## 13.9 主版本与可选版本对比

| 维度 | 主版本：Copied Origin Occurrence | 可选：Unique-Edge Descendant Backup |
|---|---|---|
| Selected copied origin 是否新增 PPO occurrence | 是 | 否 |
| Selected pair posterior 是否更新 | 是 | 是 |
| Branch evidence 是否直接增加 selected edge gradient mass | 是 | 否 |
| Shared prefix 是否新增 occurrence | 否 | 否 |
| Shared prefix 是否吸收 descendant outcome | 仅通过全局 baseline间接影响 | 是，显式 backup |
| 同一 edge 的 branch returns 如何利用 | 每个 branch 单独 advantage | 聚合为 mean/weighted mean |
| 与原 GiGPO occurrence map 的接近程度 | 更接近 | 改为 tree backup |
| Acquisition weighting 强度 | 较强 | 较弱、更稳健 |
| 实现复杂度 | 较低 | 较高 |
| 推荐角色 | 主方法 | 关键消融/增强 |

---

# 14. 实现数据结构建议

## 14.1 LeafRecord

```python
LeafRecord:
    task_id
    leaf_id
    leaf_type              # root | branch
    root_lineage_id
    selected_anchor_key    # branch only
    selected_action_key    # branch only
    terminal_return
    terminal_success
```

## 14.2 OccurrenceRecord

```python
OccurrenceRecord:
    task_id
    occurrence_id
    leaf_id
    source                  # root | branch_origin | branch_suffix
    state_key
    canonical_action
    prompt_tokens
    response_tokens
    response_mask
    old_log_probs
    return_to_go
    is_policy_draw
    is_direct_resample
    is_trainable
    posterior_eligible
    natural_edge_id         # links copied origin to natural edge
```

## 14.3 ReplayStepRecord

```python
ReplayStepRecord:
    task_id
    branch_id
    prefix_step
    state_key
    environment_action
    train_mask = 0
    posterior_eligible = False
    local_group_eligible = False
```

## 14.4 ControllerPairRecord

```python
ControllerPairRecord:
    task_id
    state_key
    canonical_action
    alpha
    beta
    delta_erv
    action_probability
    natural_origin_ids
    posterior_update_count
```

---

# 15. 必须实现的断言与单元测试

## 15.1 Leaf 数量断言

$$
|\mathcal L_g|
=
N_g^{\mathrm{root}}
+
N_g^{\mathrm{branch}}.
$$

每条成功完成的 branch 必须恰好创建一条 leaf。

---

## 15.2 Copied origin 数量断言

主版本：

$$
|\mathcal D_g^{\mathrm{branch\text{-}origin}}|
=
N_g^{\mathrm{branch}}.
$$

可选 unique-edge版本：

$$
|\mathcal D_g^{\mathrm{branch\text{-}origin}}|=0.
$$

---

## 15.3 Replay prefix mask 断言

对所有 replay steps：

```text
train_mask == 0
posterior_eligible == false
local_group_eligible == false
```

---

## 15.4 Old log-probability 一致性

对 copied origin：

$$
\operatorname{tokens}_{\mathrm{copied}}
=
\operatorname{tokens}_{\mathrm{natural}},
$$

$$
\log\pi_{\mathrm{old}}^{\mathrm{copied}}
=
\log\pi_{\mathrm{old}}^{\mathrm{natural}}.
$$

若不一致，branch sample 必须丢弃。

---

## 15.5 Leaf mapping 断言

- natural occurrence 映射到 natural leaf；
- copied origin 映射到 branch leaf；
- fresh suffix 映射到同一 branch leaf；
- replay prefix 不存在 leaf-based training mapping。

---

## 15.6 Posterior eligibility 断言

一条 branch 中：

- selected pair 必须更新一次；
- fresh suffix frozen pair每个 pair最多更新一次；
- replay prefix pair更新零次；
- new anchor/action更新零次。

---

## 15.7 Candidate support 冻结断言

branch phase 前后：

$$
\mathcal Z_g^{\mathrm{select,after}}
=
\mathcal Z_g^{\mathrm{select,before}},
$$

$$
\mathcal C_g^{\mathrm{select,after}}(z)
=
\mathcal C_g^{\mathrm{select,before}}(z).
$$

只有 posterior、ERV、$\mu$ 和 $U$ 允许变化。

---

## 15.8 Advantage 断言

对任意 copied origin $j_b$：

$$
A_j^E
=
A_{g,\ell_b}^{E},
$$

不得错误使用 origin natural root 的：

$$
A_{g,\ell_r}^{E}.
$$

---

# 16. 训练日志与诊断指标

至少记录：

## 16.1 Occurrence 统计

```text
num_natural_policy_occurrences
num_copied_origin_occurrences
num_fresh_suffix_occurrences
num_replay_prefix_steps
num_trainable_occurrences
policy_count / eval_count / train_count per selected pair
```

## 16.2 Advantage 统计

```text
leaf reward mean/std
A_E distribution by root/branch source
A_S distribution by source
final advantage distribution
selected-origin advantage sign agreement across descendants
```

## 16.3 Loss 统计

```text
root token fraction
branch-origin token fraction
branch-suffix token fraction
gradient norm by source
PPO clipping fraction by source
KL by source
```

## 16.4 Controller 更新统计

```text
selected-pair posterior updates
opportunistic fresh-suffix posterior updates
ERV before/after branch
mu(u|z) before/after branch
U(z) before/after branch
branch concentration per anchor
```

## 16.5 Prefix 诊断

```text
replay prefix train-mask violations
prefix gradient mass vs descendant count
number of ancestors affected in descendant-backup variant
```

---

# 17. 推荐消融矩阵

## 17.1 Origin handling

1. **Main：Copied origin occurrence**  
   copied origin 计算 branch-specific advantage并进入 loss。

2. **Probe-only**  
   copied origin不进入 loss；只更新 posterior，suffix正常训练。

3. **Unique-edge target backup**  
   selected edge不新增 occurrence，平均 original + direct branch returns，只训练一次。

4. **Full ancestor descendant backup**  
   selected edge和全部 prefix unique edges均聚合 descendant outcomes。

---

## 17.2 Local credit

1. Occurrence-level GiGPO；
2. Simple action aggregation；
3. Action aggregation + shrinkage。

---

## 17.3 Global leaf weighting

1. Leaf-uniform；
2. Lineage-balanced。

---

## 17.4 Posterior update scope

1. 只更新 selected pair；
2. selected pair + eligible frozen existing pairs in fresh suffix（主版本）；
3. 动态扩展 action/anchor support（不建议主用）。

---

# 18. 最终固定实现规则

主版本最终固定为：

$$
\boxed{
\begin{aligned}
&\text{所有 root 与 branch terminal leaves统一计算 }A^E;\\
&\text{natural occurrences使用其 natural leaf }A^E;\\
&\text{copied branch origin使用其 branch leaf }A^E;\\
&\text{fresh suffix使用其 branch leaf }A^E;\\
&\text{replay prefix不新增 occurrence，也不进入 loss};\\
&\text{copied origin增加 eval count 与 train count，但不增加 policy count};\\
&\text{final local groups由 root + copied origin + fresh suffix 构造};\\
&\text{主版本采用 occurrence-level }A^S;\\
&\text{action aggregation作为可选优化};\\
&\text{所有 trainable occurrences进入一个 unified token-mean PPO loss};\\
&\text{selected pair posterior必更新};\\
&\text{fresh suffix匹配 frozen existing pair且满足资格时必须更新 posterior};\\
&\text{new suffix anchors/actions只进入训练，不扩展当前 controller};\\
&\text{每条 branch后重算 ERV、action distribution与 anchor utility}.
\end{aligned}
}
$$

可选 `Unique-Edge Descendant Backup` 固定为：

$$
\boxed{
\begin{aligned}
&\text{不新增 copied-origin PPO occurrence};\\
&\text{不新增 replay-prefix occurrence};\\
&\text{对 unique natural edge聚合全部 descendant returns/advantages};\\
&\text{每个 unique edge只进入一次 loss};\\
&\text{fresh suffix仍作为新的真实 occurrences训练};\\
&\text{posterior 与 ERV controller更新规则保持不变}.
\end{aligned}
}
$$

---

# 19. 一句话实现语义

> **主版本将直接 branch 视为对 selected natural edge 的一次额外 continuation evaluation，并使用该 branch leaf 的信用再次训练 copied response；机械 replay prefix不形成新训练 occurrence。可选 descendant-backup版本则不复制 shared edge 的 loss，而把其所有后代 leaves 的回报或优势聚合到唯一自然 edge 上。**

