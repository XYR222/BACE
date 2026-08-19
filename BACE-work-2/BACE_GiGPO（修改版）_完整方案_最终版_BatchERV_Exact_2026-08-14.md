# BACE-GiGPO：基于层次化贝叶斯主动信用探索的并行 Batch-ERV Rollout 拓扑

> **Bayesian Active Credit Exploration with Competence-Guided Rollout Topology and Exact Batch Expected Regret Value for GiGPO**
>
> 最终方案版本：2026-08-14
>
> 本版本统一纳入以下最终修改：**取消独立 Pilot 阶段；全部 natural roots 更新下一批 task-family prior；以 task-family competence belief 直接规划 root/branch budget；保留 root-side capacity correction；使用 exact Batch-ERV 一次性联合选择全部 branches；Batch-ERV 使用 Beta-Binomial 有限求和精确计算，不再使用 Monte Carlo；相同或数值近似相同的 ERV/BERV 采用 seeded uniform tie-breaking；所有 branches 在 root backbone 冻结后并行 rollout。**

---

# 0. 文档目的与最终方案摘要

BACE-GiGPO 的目标不是替换 GiGPO 的核心信用分配机制，而是解决 GiGPO 的一个上游问题：

> **GiGPO 能利用已经出现的局部对比证据，但无法决定这些证据应该在哪里、以什么动作、投入多少 rollout budget 主动获得。**

因此，BACE-GiGPO 将 rollout 看作有限预算下的实验，并将每个任务实例的 rollout budget 分成两类实验：

- **breadth experiments**：从初始状态独立采样 natural roots，扩大状态覆盖并发现可比较的局部决策；
- **refinement experiments**：从 natural-root backbone 上已经自然暴露的关键 decision edges 重新采样 continuation，以提高局部动作信用的辨识度。

最终方法由四层组成：

$$
\boxed{
\text{历史 natural-root competence}
\rightarrow
\text{root/branch topology planning}
}
$$

$$
\boxed{
\text{natural-root evidence}
\rightarrow
\text{anchor--action Beta posterior}
\rightarrow
\text{Exact Batch-ERV}
}
$$

$$
\boxed{
\text{joint branch experiment design}
\rightarrow
\text{parallel replay + rollout}
}
$$

$$
\boxed{
\text{all-leaf global credit}
+
\text{GiGPO local credit}
\rightarrow
\text{unified PPO update}
}
$$

当前主版本固定以下设计。

1. 每个任务实例使用固定 terminal-leaf budget $B$，主配置 $B=8$。
2. **不存在独立 Pilot rollout phase。**
3. 当前 batch 开始时，使用上一批以前积累的 task-family natural-root history 构造 competence belief。
4. 由 task-family competence probability $q_{c,k}$ 直接规划当前实例初始 root/branch quota。
5. 当前实例计划中的 natural roots 一次性生成；必要时执行 root-side capacity correction。
6. 当前 batch 中实际生成的**全部 natural roots**，包括 capacity correction 新增 roots，都作为下一批 task-family prior 的自然 Bernoulli evidence；branch outcomes 永不更新 task-family prior。
7. Anchor 与 GiGPO 对齐：基于 pre-action observation exact key 分组；保留同一 trajectory 内重复 occurrence；不要求完整历史相同。
8. Branch 候选动作只来自该 anchor 在 natural roots 中**真实自然执行过、格式可解析且具有稳定 strict action identity 的 decision edges**；主版本同时保留 valid 与 environment-invalid/no-op identities。
9. 每个 $(z,u)$ 建立 Beta posterior，建模当前冻结策略下该自然 decision edge 的 terminal success probability。
10. Branch acquisition 使用 Bayes simple regret 对应的 **Batch Expected Regret Value（Batch-ERV / batch EVSI）**。
11. Batch-ERV 使用 Beta-Binomial predictive distribution 与有限 outcome enumeration **精确计算**，主方法不使用 Monte Carlo。
12. 主配置每个 anchor 最多分配 $L_{\max}=2$ 条 branches，因此局部 batch plan 可精确枚举。
13. 所有 branch anchor/action/origin 在正式 branch rollout 前一次性确定；所有 branches 随后并行 rollout。
14. 若多个最优动作计划、anchor 计划或全局 allocation 的 BERV 相同，则在数值 tie tolerance 下构造最优集合，并用固定随机种子**均匀随机**打破平局。
15. Branch 从 natural-root occurrence 恢复：reset 同一环境实例、机械 replay prefix、复制对应 natural CoT-action edge，再由冻结的 $\pi_{\mathrm{old}}$ 重新采样 suffix。
16. Mechanical replay prefix 不产生新训练 occurrence、不重复进入 PPO loss。
17. Copied branch-origin CoT-action 是 branch 上的新训练 occurrence，接收该 branch leaf 的全局与局部 credit。
18. Global trajectory credit 在同一任务实例的全部 root/branch terminal leaves 上统一计算。
19. 主版本局部 credit 保持 GiGPO occurrence-level estimator；按动作聚合版本作为重要增强/消融。
20. Root、branch-origin 和 branch-suffix 的所有可训练 tokens 使用同一个 standard old-policy PPO ratio $\pi_\theta/\pi_{\mathrm{old}}$；主方法不加入额外 behavior importance correction。
21. 测试阶段删除所有 prior、posterior、Batch-ERV、branch controller 与 replay，只运行训练后的 actor。

一句话概括：

> **BACE-GiGPO 将 rollout 看作预算受限的实验：历史 natural-root competence 决定 breadth 与 refinement 的预算比例，当前 roots 建立自然局部证据，Exact Batch-ERV 一次性设计最能降低动作选择遗憾的一组 branch experiments，并通过并行 continuation resampling 主动改善 GiGPO 的局部信用证据。**

---

# 1. 研究动机

## 1.1 GiGPO 已经解决“如何用局部证据”，但没有解决“如何主动获得局部证据”

GiGPO 对同一任务实例采样一组 trajectories，并将优势拆成：

$$
A_{i,t}=A_i^E+\omega A_{i,t}^S.
$$

其中：

- $A_i^E$ 衡量完整 trajectory 在同任务 rollout group 中的相对质量；
- $A_{i,t}^S$ 衡量在同一 anchor/state 下不同 occurrences 的相对 return-to-go。

关键问题在于，GiGPO 的 state group 是由自然 rollouts **被动碰撞**产生的。

如果一个关键状态只出现一次，或者虽然重复出现但局部动作证据极不平衡，那么 GiGPO 本身不会主动追加样本。

因此可以把原问题写成：

$$
\boxed{
\text{GiGPO：给定 comparison group，怎样做 credit assignment？}
}
$$

而 BACE-GiGPO 解决：

$$
\boxed{
\text{BACE：有限 rollout budget 应该怎样主动构造更有价值的 comparison groups？}
}
$$

---

## 1.2 Breadth 与 refinement 的预算价值随 competence 变化

策略较弱时，主要瓶颈通常是：

- 找不到成功轨迹；
- 状态覆盖不足；
- 关键状态尚未重复出现；
- 局部比较组尚未形成。

因此自然 roots 的边际价值高。

策略较强时，大量完整 rollouts 会重复通过已经稳定掌握的 prefix，真正不确定的部分可能只集中在少数中后段决策。

此时从关键 anchor 重新采样 continuation 更节省探索预算。

因此我们把 rollout topology 看成 competence-dependent allocation：

$$
\boxed{
\text{low competence}\Rightarrow\text{more breadth}
}
$$

$$
\boxed{
\text{high competence}\Rightarrow\text{more refinement}
}
$$

当前版本为了避免 Pilot 带来的额外全局 rollout barrier，不再对每个实例先跑少量 pilots，而是使用**滞后的 task-family competence belief**规划当前 topology，再由当前实例实际形成的 anchor information capacity 做保守修正。

---

## 1.3 为什么不是 entropy，而是 decision-relevant information value

一般不确定性和真正有训练价值的信用不确定性不同。

考虑一个动作 posterior 方差很高，但即使再观察一次成功或失败，它仍然不可能超过当前 incumbent。此时它虽然“不确定”，但新的样本不会改变当前动作选择。

BACE-GiGPO 更关心：

> **新增一组 branch outcomes 后，我们依据 posterior mean 选择错误动作所造成的 Bayes decision regret，预计会减少多少？**

因此 acquisition score 不是 policy entropy，也不是单纯 posterior variance，而是：

$$
\boxed{
\text{expected reduction in Bayes simple regret}
}
$$

即 ERV / EVSI。

在当前并行版本中，我们进一步从 one-sample ERV 扩展为：

$$
\boxed{
\text{Batch-ERV：一组同时执行的 branch experiments 的联合 expected regret reduction}
}
$$

---

## 1.4 为什么从 sequential ERV 改成 Batch-ERV

Sequential ERV 在统计上更自适应：

$$
e_1\rightarrow Y_1\rightarrow e_2\rightarrow Y_2\rightarrow\cdots
$$

但对分布式 LLM rollout 系统非常不友好，因为每个 branch outcome 都形成新的同步依赖。

当前版本改成：

$$
\boxed{
(e_1,e_2,\ldots,e_Q)
\text{ 在 branch rollout 前联合确定}
}
$$

随后：

$$
\boxed{
Q\text{ 条 branches 一次并行 rollout}
}
$$

这样牺牲 outcome-adaptive acquisition，换取显著更好的 GPU batching 与 wall-clock efficiency。

关键是我们不采用简单的 one-step ERV top-$Q$，而是直接优化**联合 Batch-ERV**，因此能够考虑重复采样同一动作的边际价值递减以及不同动作实验之间的联合价值。

---

# 2. 问题设定与符号

## 2.1 Task、family 与 policy update

具体任务实例记为：

$$
g\in\mathcal G,
$$

其任务类型为：

$$
c_g\in\mathcal C.
$$

在第 $k$ 个 policy update 开始时冻结：

$$
\pi_{\mathrm{old}}\leftarrow\pi_{\theta_k}.
$$

整个 rollout/acquisition batch 中都使用同一个 $\pi_{\mathrm{old}}$。

这是必要条件，因为局部参数：

$$
p_{g,z,u}
$$

表示的是**固定 continuation policy 下**的 success probability。

Actor 更新后，该局部 posterior 即失效，不跨 policy update 缓存。

---

## 2.2 固定 terminal-leaf budget

每个任务实例固定：

$$
B_g=B.
$$

最终满足：

$$
R_g+Q_g=B,
$$

其中：

- $R_g$：最终 natural-root 数；
- $Q_g$：最终 branch 数。

主配置：

$$
\boxed{B=8.}
$$

同时设置最小 natural-root budget：

$$
\boxed{R_{\min}=2.}
$$

固定的是 terminal leaves，而不是 generated tokens。实验必须额外报告：

- actor generated tokens；
- environment steps；
- replay steps；
- wall-clock；
- GPU time；
- peak memory；
- throughput。

---

## 2.3 Natural root

Natural root 是从任务初始状态开始，由冻结 actor 自然完成的完整轨迹：

$$
\tau_{g,r}^{\mathrm{root}}
=
(s_0,a_0,s_1,a_1,\ldots,s_T).
$$

其 terminal outcome：

$$
Y_{g,r}\in\{0,1\}.
$$

主方法的 task-family competence 与 local Beta posterior 都以 binary terminal success 为主要建模对象。

---

## 2.4 Anchor occurrence

对 natural root 中第 $t$ 个动作执行前的 observation 记为：

$$
o_{g,r,t}^{\mathrm{pre}}.
$$

按 GiGPO 的 exact grouping 规则构造 hashable key：

$$
z_{g,r,t}=\kappa(o_{g,r,t}^{\mathrm{pre}}).
$$

定义 natural-root occurrence set：

$$
\mathcal I_g^{\mathrm{root}}(z)
=
\{(r,t):z_{g,r,t}=z\}.
$$

主版本：

- 保留不同 roots 间的重复 occurrence；
- 保留同一 root 不同时间步的循环重访；
- 不要求完整 history 一致；
- 使用 exact pre-action observation key；
- 初始状态不作为 branch anchor；
- terminal state 不作为 branch anchor。

---

## 2.5 Observed decision edge

对自然 occurrence $(r,t)$，模型真实生成的 CoT-action response 记为：

$$
e_{g,r,t}=(h_{g,r,t},c_{g,r,t},u_{g,r,t}),
$$

其中：

- $h$：该 occurrence 对应的模型输入上下文；
- $c$：自然生成的 reasoning / CoT 部分；
- $u$：最终环境动作。

BACE 的 branch 不从未见动作中进行任意干预，而是对已经自然出现的 decision edges 进行 resampling。

因此，对 anchor $z$ 的候选动作集合定义为：

$$
\boxed{
\mathcal C_g(z)=\mathcal C_g^{\mathrm{obs}}(z)
}
$$

即 natural roots 中在 $z$ 上实际执行过、格式可解析且具有稳定 strict action identity 的 decision edges。这里的候选集合同时允许：

- `valid::<environment action>`；
- `invalid::<raw action body>`。

因此“environment-invalid”不等于“不能进入 Batch-ERV”，更不等于“无法 Replay”。无法稳定解析 action body 的 response 才不进入 action-level posterior 与 acquisition support。

---

# 3. 方法总览

一次 policy update 中，对任务实例 $g$ 的流程如下。

## Stage A：Lagged competence planning

由上一批以前的 natural-root history 构造 task-family competence belief：

$$
H_{c,k-1}
\rightarrow
q_{c,k}.
$$

再得到：

$$
\bar Q_g,
\qquad
R_g^{(0)}=B-\bar Q_g.
$$

## Stage B：Natural-root evidence acquisition

一次性生成：

$$
R_g^{(0)}
$$

条 natural roots。

根据这些 roots：

1. 构造 repeated / structural anchors；
2. 构造 observed action sets；
3. 构造 current-instance competence posterior；
4. 构造 anchor-action Beta posteriors；
5. 计算 exact Batch-ERV 与信息容量。

若当前 root backbone 无法支持计划 branch budget，则：

$$
Q_g\leftarrow Q_g-1,
\qquad
R_g\leftarrow R_g+1,
$$

追加 natural root 并重新计算，直到 capacity 足够或 $Q_g=0$。

## Stage C：Exact Batch-ERV experiment design

冻结最终 natural-root backbone 后，对每个 anchor 枚举 $0/1/2$ branch plans，计算 exact Batch-ERV：

$$
V_g^{(m)}(z).
$$

联合求解：

$$
\max_{\{m_z\}}
\sum_zV_g^{(m_z)}(z),
$$

使：

$$
\sum_zm_z=Q_g,
\qquad
m_z\in\{0,1,2\}.
$$

由每个最优 local plan 得到具体 $(z,u)$ branch experiments。

## Stage D：Parallel branch rollout

所有 branch anchor、action、origin 一次性确定后：

$$
\boxed{
\text{all branches run in parallel}
}
$$

Branch outcome 不再用于决定本 batch 中“下一条 branch”。

## Stage E：Unified GiGPO optimization

所有 root 与 branch leaves 完成后：

- all-leaf global normalization；
- final occurrence grouping；
- local GiGPO credit；
- unified PPO objective；
- 更新 actor；
- 更新 task-family natural-root history，供下一批使用。

---

# 4. Task-Family Competence Prior：取消 Pilot 后的定义

## 4.1 建模对象

对 task family $c$，定义冻结策略的 natural-root competence：

$$
\theta_{c,k}
=
P(Y=1\mid c,\pi_{\mathrm{old},k}).
$$

维护 Beta history：

$$
\theta_{c,k}
\sim
\operatorname{Beta}(A_{c,k},B_{c,k}).
$$

这里的 evidence **只来自 natural roots**。

Branch outcomes 不进入该统计，因为 branches 是主动选择的条件 continuation，不能代表从任务初始状态自然 rollout 的成功率。

---

## 4.2 历史更新：使用全部 natural roots

当前 batch 完成后，对 family $c$ 收集：

$$
S_{c,k}^{\mathrm{root}}
=
\sum_{g:c_g=c}\sum_{r=1}^{R_g}Y_{g,r},
$$

$$
F_{c,k}^{\mathrm{root}}
=
\sum_{g:c_g=c}R_g-S_{c,k}^{\mathrm{root}}.
$$

这里的 $R_g$ 包含：

- 初始计划 natural roots；
- capacity correction 新增 natural roots。

使用指数遗忘：

$$
\widetilde S_{c,k}
=
\lambda_{\mathrm{hist}}\widetilde S_{c,k-1}
+
S_{c,k}^{\mathrm{root}},
$$

$$
\widetilde F_{c,k}
=
\lambda_{\mathrm{hist}}\widetilde F_{c,k-1}
+
F_{c,k}^{\mathrm{root}}.
$$

因此：

$$
A_{c,k}=A_0+\widetilde S_{c,k},
$$

$$
B_{c,k}=B_0+\widetilde F_{c,k}.
$$

推荐：

$$
A_0=B_0=1,
\qquad
\lambda_{\mathrm{hist}}\in[0.8,0.95].
$$

### 统计语义

取消 pilot 后，不再追求“每个实例固定贡献相同数量的 Bernoulli samples”。

当前主版本的估计对象明确为：

$$
\boxed{
\text{controller 实际采集到的 natural-root level success probability}
}
$$

因此每条 natural root 是一个等权 Bernoulli evidence。

需要承认：root-heavy 的困难实例会对 family history 提供更多 evidence，这可能形成保守反馈。该问题通过 per-instance-normalized history 作为消融检验，而不是在主算法中加入额外 reweighting。

---

## 4.3 当前 batch 使用滞后 prior

当前第 $k$ 批 rollout 的 topology 只能读取：

$$
H_{c,k-1}.
$$

即：

$$
\boxed{
\text{Batch }k-1\text{ natural roots}
\rightarrow
\text{Batch }k\text{ topology}
}
$$

当前 batch 的 root outcomes 不重新修改已经执行过的初始 root quota；它们主要用于：

- 当前实例 local credit posterior；
- capacity correction；
- 下一 batch 的 family competence history。

这避免重新引入“先 root、更新 topology、再补 root”的额外全局阶段。

---

## 4.4 Bounded competence belief

历史 Beta mean：

$$
\mu_{c,k}
=
\frac{A_{c,k-1}}{A_{c,k-1}+B_{c,k-1}}.
$$

为了避免长期 history 使 controller 过度确定，使用 bounded concentration：

$$
\kappa_{c,k}^{T}
=
\operatorname{clip}
\left(
\tau_T(A_{c,k-1}+B_{c,k-1}),
\kappa_T^{\min},
\kappa_T^{\max}
\right).
$$

定义 controller competence belief：

$$
\boxed{
\phi_{c,k}
\sim
\operatorname{Beta}
\left(
\kappa_{c,k}^{T}\mu_{c,k},
\kappa_{c,k}^{T}(1-\mu_{c,k})
\right).
}
$$

推荐初始范围：

$$
\tau_T\approx0.1,
\qquad
\kappa_T^{\min}=2,
\qquad
\kappa_T^{\max}\in\{4,8\}.
$$

---

## 4.5 Refinement readiness probability

设 competence threshold：

$$
\tau_{\mathrm{comp}}\in(0,1).
$$

定义：

$$
\boxed{
q_{c,k}
=
P(\phi_{c,k}>\tau_{\mathrm{comp}}).
}
$$

推荐初始：

$$
\tau_{\mathrm{comp}}=0.5.
$$

$q_{c,k}$ 的语义不再是“当前实例通过 pilots 证明自己已进入 refinement regime”，而是：

> **当前策略在该 task family 上处于 refinement-ready regime 的 posterior probability。**

这使 topology 成为：

$$
\boxed{
\text{family competence-guided}
}
$$

而当前实例的信息通过后续 root evidence 与 capacity correction 进入。

---

# 5. Root–Branch Topology Planning

## 5.1 初始 branch quota

在固定 $B$ 和 $R_{\min}$ 下：

$$
M=B-R_{\min}.
$$

规划：

$$
\boxed{
\bar Q_g
=
\operatorname{round}(Mq_{c_g,k}).
}
$$

再裁剪：

$$
0\le\bar Q_g\le B-R_{\min}.
$$

初始 natural roots：

$$
\boxed{
R_g^{(0)}=B-\bar Q_g.
}
$$

例如 $B=8,R_{\min}=2$：

| $q_{c,k}$ | 典型初始 topology |
|---:|---:|
| $0.10$ | $7R+1B$ |
| $0.35$ | $6R+2B$ |
| $0.50$ | $5R+3B$ |
| $0.70$ | $4R+4B$ |
| $0.90$ | $3R+5B$ |

这里的 $\bar Q_g$ 是**计划 refinement budget**，不是无条件必须执行的 branch 数。

---

## 5.2 线性 quota 的 Bayes decision 解释

定义 family-level refinement regime：

$$
Z_c
=
\mathbf1[\phi_c>\tau_{\mathrm{comp}}].
$$

理想 branchable slots：

$$
Q^*=MZ_c.
$$

若使用平方 mismatch loss：

$$
\ell(q,Z_c)=(q-MZ_c)^2,
$$

则 posterior expected loss 的连续最优解为：

$$
q^*=Mq_{c,k}.
$$

因此：

$$
\bar Q_g=\operatorname{round}(Mq_{c,k})
$$

是该 surrogate 下的整数 Bayes action。

这不是最终 task return 意义上的全局最优保证，而是一个简单、可解释、无额外 schedule shape 参数的 topology decision rule。

---

# 6. Natural-Root Backbone 与 Anchor 构造

## 6.1 一次性生成 planned roots

对同一批 tasks，把所有计划 natural roots 合并进入第一轮 rollout wave。

例如有 $16$ 个 tasks，各自计划 $5R+3B$，则第一轮约为：

$$
16\times5=80
$$

条 natural roots 并行执行，而不是先为每个任务运行 pilots 再进入第二轮 roots。

---

## 6.2 Repeated anchor

Repeated anchor 集：

$$
\mathcal Z_g^{\mathrm{rep}}
=
\left\{
z:
|\mathcal I_g^{\mathrm{root}}(z)|\ge2
\right\}.
$$

Occurrence 可以来自：

- 不同 natural roots；
- 同一 natural root 的循环重访；
- 两者组合。

与 GiGPO 一致，不按 trajectory 去重。

---

## 6.3 Structural branch anchor

定义：

$$
\operatorname{StructValid}_g(z)=1
$$

当且仅当满足：

1. $|\mathcal I_g^{\mathrm{root}}(z)|\ge2$；
2. exact GiGPO pre-action observation key 相同；
3. $z$ 不是初始状态；
4. $z$ 非 terminal；
5. replay metadata 完整；
6. natural roots 中至少存在两个不同、格式可解析的 strict observed action identities；允许 valid 与 invalid/no-op identity 形成竞争；
7. branch origin 必须来自 natural-root backbone；
8. 主版本只允许一层 branching。

定义结构 anchor 集：

$$
\boxed{
\mathcal Z_g^{\mathrm{struct}}
=
\{z:\operatorname{StructValid}_g(z)=1\}.
}
$$

---

## 6.4 Observed action set

对 anchor $z$：

$$
\mathcal C_g^{\mathrm{obs}}(z)
=
\left\{
\operatorname{can}(a_{g,r,t})
:
(r,t)\in\mathcal I_g^{\mathrm{root}}(z),
\operatorname{can}(a_{g,r,t})\neq\mathrm{INVALID}
\right\}.
$$

其中 `can` 表示严格 action identity，而不是把文本语义修正成 admissible action：

$$
\operatorname{can}(a)=
\begin{cases}
\texttt{valid::<environment action>}, & \text{environment-valid},\\
\texttt{invalid::<raw action body>}, & \text{format-valid but environment-invalid},\\
\mathrm{INVALID}, & \text{action body 无法稳定解析}.
\end{cases}
$$

主版本使用 `strict_identity`，因此前两类都进入候选集合。`valid_only_branch` 与 `single_invalid_bucket` 仅作为显式消融，不是 Exact Batch-ERV 的默认行为。对 invalid/no-op branch，Replay 成功标准是同一 raw response 的 transition identity、observation、reward 与 done 可复现，而不是要求它属于 admissible set。

主版本：

$$
\boxed{
\mathcal C_g(z)=\mathcal C_g^{\mathrm{obs}}(z).
}
$$

不额外加入：

- actor-top unseen actions；
- 全部 admissible actions；
- LLM proposal actions。

这样 branch 只对自然策略已经真实执行过的 decision edges 做 continuation resampling。

---

## 6.5 Action-specific natural origin pool

对 $(z,u)$ 定义：

$$
\boxed{
\mathcal I_g^{\mathrm{obs}}(z,u)
=
\left\{
(r,t)\in\mathcal I_g^{\mathrm{root}}(z):
\operatorname{can}(a_{g,r,t})=u
\right\}.
}
$$

选中 branch experiment $(z,u)$ 后，只从该集合选择 concrete origin。

这保证被 replay 的自然 occurrence **原本就实际执行过动作 $u$**。

因此无需估计：

$$
\pi_{\mathrm{old}}(u\mid z)
$$

在不同 CoT/history 上的复杂边际概率，也无需构造 action-only prompt。

---

# 7. Current-Instance Competence 与 Local Beta Posterior

## 7.1 Roots 完成后的实例 posterior

虽然当前实例 posterior 不再决定最初 topology，但它仍可作为 local posterior 的弱层次化中心。

设最终当前 natural roots 中成功数：

$$
S_g^{\mathrm{root}},
$$

失败数：

$$
F_g^{\mathrm{root}}=R_g-S_g^{\mathrm{root}}.
$$

定义：

$$
\phi_g\mid\mathcal D_g^{\mathrm{root}}
\sim
\operatorname{Beta}
\left(
\kappa_{c,k}^{T}\mu_{c,k}+S_g^{\mathrm{root}},
\kappa_{c,k}^{T}(1-\mu_{c,k})+F_g^{\mathrm{root}}
\right).
$$

其 posterior mean：

$$
\bar\phi_g
=
\frac{
\kappa_{c,k}^{T}\mu_{c,k}+S_g^{\mathrm{root}}
}{
\kappa_{c,k}^{T}+R_g
}.
$$

该量用于：

- local Beta prior center；
- current-instance competence diagnostics；
- family-prior calibration analysis。

它**不再回头修改已经执行的 initial root quota**。

---

## 7.2 Anchor-action 成功概率的精确定义

对自然 observed edge $(z,u)$，定义：

$$
\boxed{
p_{g,z,u}
=
\mathbb E_{i\sim\nu_g(\cdot\mid z,u)}
\left[
P(Y=1\mid h_i,c_i,u,\pi_{\mathrm{old}})
\right],
}
$$

其中：

$$
\nu_g(i\mid z,u)
$$

是支持动作 $u$ 的 natural occurrences 上的经验 origin distribution。

这一定义与实际 branch 操作一致：

> 我们不是在抽象状态 $z$ 下对任意 unseen action 做 intervention，而是在自然策略已经执行过的 CoT-action edges 上重新采样 continuation。

---

## 7.3 弱局部 Beta prior

对每个 $(z,u)$：

$$
p_{g,z,u}
\sim
\operatorname{Beta}
\left(
\kappa_A\bar\phi_g,
\kappa_A(1-\bar\phi_g)
\right).
$$

主配置：

$$
\boxed{\kappa_A=2.}
$$

设 natural-root evidence 中该 pair 的成功/失败次数为：

$$
s_{g,z,u},
\qquad
f_{g,z,u}.
$$

则 posterior：

$$
\boxed{
\alpha_{g,z,u}
=
\kappa_A\bar\phi_g+s_{g,z,u},
}
$$

$$
\boxed{
\beta_{g,z,u}
=
\kappa_A(1-\bar\phi_g)+f_{g,z,u}.
}
$$

posterior mean：

$$
m_{g,z,u}
=
\frac{\alpha_{g,z,u}}
{\alpha_{g,z,u}+\beta_{g,z,u}}.
$$

---

# 8. Bayes Simple Regret 与 Exact ERV

## 8.1 Bayes simple regret

固定 anchor $z$，简写：

$$
p_u=p_{g,z,u},
\qquad
m_u=E[p_u\mid\mathcal D].
$$

当前 posterior-mean decision：

$$
\hat u
=
\arg\max_um_u.
$$

定义 Bayes simple regret：

$$
\boxed{
\mathcal R_g(z)
=
E[\max_up_u\mid\mathcal D]
-
\max_um_u.
}
$$

它等价于该局部动作选择问题的 EVPI：

$$
\mathcal R_g(z)
=
E[p_{u^*}-p_{\hat u}\mid\mathcal D].
$$

---

## 8.2 ERV 的关键化简：主 controller 不需要计算 $E[\max p_u]$

设未来实验得到 observations $\mathbf Y$。

Batch-ERV 定义：

$$
\operatorname{BERV}(\mathcal B)
=
\mathcal R(\mathcal D)
-
E_{\mathbf Y}
[\mathcal R(\mathcal D,\mathbf Y)].
$$

展开：

$$
\begin{aligned}
\operatorname{BERV}(\mathcal B)
={}&
E[\max_up_u\mid\mathcal D]
-
\max_um_u\\
&-
E_{\mathbf Y}
\left[
E[\max_up_u\mid\mathcal D,\mathbf Y]
-
\max_um_u^{\mathbf Y}
\right].
\end{aligned}
$$

根据 tower property：

$$
E_{\mathbf Y}
E[\max_up_u\mid\mathcal D,\mathbf Y]
=
E[\max_up_u\mid\mathcal D].
$$

因此完全抵消，得到：

$$
\boxed{
\operatorname{BERV}(\mathcal B)
=
E_{\mathbf Y}
\left[
\max_um_u^{\mathbf Y}
\right]
-
\max_um_u.
}
$$

这是本版本的关键简化。

主 acquisition controller 根本不需要数值估计 $E[\max p_u]$，因此：

$$
\boxed{
\text{ERV / Batch-ERV 主计算不需要 Monte Carlo。}
}
$$

---

## 8.3 One-sample ERV 的精确闭式

考虑只对动作 $u$ 获取一个新 outcome。

令：

$$
\alpha=\alpha_u,
\qquad
\beta=\beta_u,
\qquad
N=\alpha+\beta,
$$

$$
m=\frac\alpha N.
$$

成功后 posterior mean：

$$
m^+
=
\frac{\alpha+1}{N+1}.
$$

失败后：

$$
m^-
=
\frac\alpha{N+1}.
$$

定义其他动作中当前最大的 posterior mean：

$$
c
=
\max_{v\neq u}m_v.
$$

则：

$$
\boxed{
\operatorname{ERV}(z,u)
=
m\max(m^+,c)
+(1-m)\max(m^-,c)
-
\max(m,c).
}
$$

进一步，因为：

$$
m^-<m<m^+,
$$

可写成：

$$
\boxed{
\operatorname{ERV}(z,u)
=
\begin{cases}
0, & c\le m^-,\\[2mm]
(1-m)(c-m^-), & m^-<c\le m,\\[2mm]
m(m^+-c), & m<c<m^+,\\[2mm]
0, & c\ge m^+.
\end{cases}
}
$$

这说明：

- 若一次失败也不会让 incumbent 失去第一名，ERV 为 $0$；
- 若一次成功也不可能让 challenger 追上第一名，ERV 为 $0$；
- 只有新 evidence 可能改变 posterior-mean 最优动作身份时，one-sample ERV 才为正。

---

# 9. Exact Batch-ERV

## 9.1 Batch plan

对 anchor $z$，定义 local branch plan：

$$
\mathcal B_z.
$$

对每个动作 $u$，令：

$$
n_u(\mathcal B_z)
$$

表示该 batch plan 中动作 $u$ 被重新采样多少次。

主配置：

$$
\boxed{
|\mathcal B_z|\le L_{\max}=2.
}
$$

因此可能计划包括：

- $[u]$；
- $[u,u]$；
- $[u,v]$，$u\neq v$。

---

## 9.2 Beta-Binomial predictive probability

若：

$$
p_u\sim\operatorname{Beta}(\alpha_u,\beta_u),
$$

未来对动作 $u$ 采样 $n_u$ 次，其中成功次数：

$$
K_u=k_u,
$$

则 predictive distribution 为：

$$
\boxed{
P(K_u=k_u)
=
\binom{n_u}{k_u}
\frac{
B(\alpha_u+k_u,\beta_u+n_u-k_u)
}{
B(\alpha_u,\beta_u)
}.
}
$$

更新后的 posterior mean：

$$
\boxed{
m_u^+(k_u)
=
\frac{\alpha_u+k_u}
{\alpha_u+\beta_u+n_u}.
}
$$

对于没有在计划中采样的动作：

$$
n_u=0,
\qquad
k_u=0,
\qquad
m_u^+(0)=m_u.
$$

---

## 9.3 Exact Batch-ERV 有限求和公式

设 action set 有 $K$ 个动作，plan 对应 count vector：

$$
\mathbf n=(n_1,\ldots,n_K).
$$

则：

$$
\boxed{
\begin{aligned}
\operatorname{BERV}_g(z,\mathbf n)
={}&
\sum_{k_1=0}^{n_1}
\cdots
\sum_{k_K=0}^{n_K}
\left[
\prod_{u=1}^{K}
P(K_u=k_u)
\right]\\
&\cdot
\max_v
\frac{\alpha_v+k_v}
{\alpha_v+\beta_v+n_v}
-
\max_v
\frac{\alpha_v}
{\alpha_v+\beta_v}.
\end{aligned}
}
$$

这是确定性有限求和，不是估计量。

因此主方法中不存在：

$$
M_{\mathrm{MC}}.
$$

也不存在 MC seed 对 topology 的影响。

---

## 9.4 $L_{\max}=2$ 时计算极小

假设：

$$
\mathcal C(z)=\{a,b,c\}.
$$

一条 branch 只需评估：

$$
[a],\ [b],\ [c].
$$

两条 branches 只需评估：

$$
[a,a],\ [b,b],\ [c,c],\ [a,b],\ [a,c],\ [b,c].
$$

对 $[u,u]$，只需枚举：

$$
K_u\in\{0,1,2\}.
$$

对 $[u,v]$，只需枚举四种 binary combinations。

相较一次 LLM rollout，该 CPU 开销可忽略。

---

## 9.5 两次同动作采样的显式公式

若：

$$
p_u\sim\operatorname{Beta}(\alpha,\beta),
\qquad
N=\alpha+\beta,
$$

两次采样的成功次数概率：

$$
P(K=0)
=
\frac{\beta(\beta+1)}{N(N+1)},
$$

$$
P(K=1)
=
\frac{2\alpha\beta}{N(N+1)},
$$

$$
P(K=2)
=
\frac{\alpha(\alpha+1)}{N(N+1)}.
$$

对应 posterior means：

$$
m_0=\frac\alpha{N+2},
\qquad
m_1=\frac{\alpha+1}{N+2},
\qquad
m_2=\frac{\alpha+2}{N+2}.
$$

设其他动作最佳 mean：

$$
c=\max_{v\neq u}m_v.
$$

则：

$$
\boxed{
\begin{aligned}
\operatorname{BERV}([u,u])
={}&
P_0\max(c,m_0)
+P_1\max(c,m_1)\\
&+P_2\max(c,m_2)
-\max(c,m).
\end{aligned}
}
$$

---

# 10. Local Batch Value 与 Information Capacity

## 10.1 每个 anchor 的最优 $m$-branch value

定义：

$$
V_g^{(0)}(z)=0.
$$

对于 $m\in\{1,2\}$：

$$
\boxed{
V_g^{(m)}(z)
=
\max_{|\mathcal B_z|=m}
\operatorname{BERV}_g(z,\mathcal B_z).
}
$$

同时保存达到最大值的最优 plan 集：

$$
\mathfrak B_g^{*(m)}(z).
$$

---

## 10.2 边际信息价值

定义：

$$
\delta_g^{(1)}(z)
=
V_g^{(1)}(z),
$$

$$
\delta_g^{(2)}(z)
=
V_g^{(2)}(z)-V_g^{(1)}(z).
$$

直观上：

- $\delta^{(1)}$：给该 anchor 第一条 branch 的 expected decision value；
- $\delta^{(2)}$：在已经允许一条 branch 后，再增加第二条的联合边际价值。

---

## 10.3 Information capacity

设置信息价值阈值：

$$
\tau_{\mathrm{BERV}}\ge0.
$$

主配置 $L_{\max}=2$ 下采用层级式 marginal capacity：

$$
\boxed{
c_g(z)
=
\begin{cases}
0, & \delta_g^{(1)}(z)<\tau_{\mathrm{BERV}},\\[1mm]
1, & \delta_g^{(1)}(z)\ge\tau_{\mathrm{BERV}}
\ \land\ \delta_g^{(2)}(z)<\tau_{\mathrm{BERV}},\\[1mm]
2, & \delta_g^{(1)}(z)\ge\tau_{\mathrm{BERV}}
\ \land\ \delta_g^{(2)}(z)\ge\tau_{\mathrm{BERV}}.
\end{cases}
}
$$

也就是说，第二个 branch slot 只有在第一 slot 已经具有非平凡信息价值时才被计入容量；这避免出现“第二条 slot 有效、第一条 slot 无效”的不可执行容量定义。

总 capacity：

$$
\boxed{
C_g
=
\sum_{z\in\mathcal Z_g^{\mathrm{struct}}}c_g(z).
}
$$

与旧的：

$$
L_{\max}|\mathcal Z_g^{\mathrm{eff}}|
$$

相比，这一 capacity 不再假设每个 anchor 天然都值得两条 branches，而是依据 Batch-ERV 的实际边际价值计数。

---

# 11. Root-Side Capacity Correction

## 11.1 初始化

令：

$$
Q_g^{(0)}=\bar Q_g,
$$

$$
R_g^{(0)}=B-\bar Q_g.
$$

首先一次性生成 $R_g^{(0)}$ 条 natural roots。

随后：

1. 重建 anchor groups；
2. 更新 current-instance posterior；
3. 更新 local Beta posteriors；
4. 精确计算 $V^{(1)},V^{(2)}$；
5. 计算 $C_g$。

---

## 11.2 Capacity 不足

如果：

$$
C_g<Q_g,
$$

则将一个 planned branch slot 转为 natural root：

$$
\boxed{
R_g\leftarrow R_g+1,
\qquad
Q_g\leftarrow Q_g-1.
}
$$

生成一条新的 natural root，然后重新计算全部相关结构。

重复直到：

$$
\boxed{C_g\ge Q_g}
$$

或：

$$
Q_g=0.
$$

---

## 11.3 为什么仍然保留 capacity correction

取消 pilots 后，initial topology 完全来自 family competence history，因此可能：

- family competence 高，但当前实例异常困难；
- 当前 few roots 无法形成足够有效 anchors；
- 计划的 branch budget 缺乏自然证据支撑。

Capacity correction 提供：

$$
\boxed{
\text{family-level planning}
+
\text{instance-level structural correction}.
}
$$

它只允许：

$$
\text{branch slot}\rightarrow\text{root slot},
$$

而不会撤销已经生成的 roots，因此是一个保守、单向的 correction。

---

## 11.4 终止性

每次 correction：

$$
Q_g\leftarrow Q_g-1.
$$

所以最迟在：

$$
Q_g=0
$$

时结束。

因此一定满足：

$$
R_g+Q_g=B.
$$

---

# 12. Freeze Root Backbone

当：

$$
C_g\ge Q_g
$$

后，冻结：

1. 全部 natural roots；
2. exact anchor groups；
3. structural anchor pool；
4. observed action sets；
5. $(z,u)$ natural origin pools；
6. local Beta posterior parameters；
7. final branch budget $Q_g$。

之后：

- 不再新增 root；
- 不再增加 branch origin pool；
- branch suffix 新状态不会成为当前 iteration 的新 branch origin；
- branch outcomes 不重新分配当前 branch budget。

---

# 13. Global Exact Batch-ERV Allocation

## 13.1 局部 plan

对每个 anchor $z$ 和：

$$
m\in\{0,1,2\},
$$

计算：

$$
V_g^{(m)}(z)
$$

并记录对应的最优 local branch plan：

$$
\mathcal B_{g,z}^{*(m)}.
$$

---

## 13.2 全局预算优化

对任务实例 $g$，求：

$$
\boxed{
\max_{\{m_z\}}
\sum_{z\in\mathcal Z_g^{\mathrm{struct}}}
V_g^{(m_z)}(z)
}
$$

subject to：

$$
m_z\in\{0,1,2\},
$$

$$
\sum_zm_z=Q_g,
$$

以及：

$$
\boxed{m_z\le c_g(z).}
$$

因此 global optimizer 只会使用已经通过 Batch-ERV marginal information gate 的 branch slots。

由于：

$$
B=8,
\qquad
Q_g\le6,
\qquad
L_{\max}=2,
$$

这是一个极小的离散预算分配问题，可用：

- 枚举；
- 小型动态规划；
- 极小整数规划；

精确求解。

不需要近似 greedy allocator。

---

## 13.3 为什么不是 one-step ERV top-$Q$

若当前：

$$
\operatorname{ERV}(z_1,a)=0.10,
$$

$$
\operatorname{ERV}(z_2,b)=0.08,
$$

naive top-$Q$ 可能给 $z_1$ 重复两条 branches。

但第二条对 $z_1$ 的边际 value 可能只有：

$$
0.02.
$$

这时联合计划：

$$
[z_1:a,\ z_2:b]
$$

优于：

$$
[z_1:a,\ z_1:a].
$$

Batch-ERV 直接比较完整 future outcome distribution，因此能够自然反映这一点。

---

# 14. ERV/BERV 相同值的处理

## 14.1 相同值不是算法错误

若两个实验 posterior 结构对称，可能有：

$$
\operatorname{ERV}(z,a)
=
\operatorname{ERV}(z,b).
$$

这表示：

> 在当前 Bayesian decision objective 下，两项实验具有完全相同的 expected information value。

因此不应人为加入额外 score 去“制造差异”。

---

## 14.2 数值 tie 定义

浮点计算下定义：

$$
\boxed{
|V_1-V_2|
\le
\epsilon_{\mathrm{abs}}
+
\epsilon_{\mathrm{rel}}
\max(|V_1|,|V_2|)
}
$$

则视为 tie。

建议：

$$
\epsilon_{\mathrm{abs}}=10^{-12},
\qquad
\epsilon_{\mathrm{rel}}=10^{-10}.
$$

因为 Exact Batch-ERV 不再包含 Monte Carlo noise，不需要设置大的 tie tolerance。

---

## 14.3 Local plan tie

定义：

$$
V_{\max}^{(m)}(z)
=
\max_{|\mathcal B|=m}V(z,\mathcal B).
$$

tie-optimal plan 集：

$$
\boxed{
\mathfrak B^{*(m)}(z)
=
\left\{
\mathcal B:
V_{\max}^{(m)}(z)-V(z,\mathcal B)
\le\epsilon_{\mathrm{tie}}
\right\}.
}
$$

若集合中超过一个 plan：

$$
\boxed{
\mathcal B^*
\sim
\operatorname{Uniform}(\mathfrak B^{*(m)}(z)).
}
$$

---

## 14.4 Global allocation tie

若多个：

$$
\{m_z\}
$$

具有相同最优总 Batch-ERV，则定义 global optimal allocation set：

$$
\mathfrak M_g^*.
$$

从中：

$$
\boxed{
\mathbf m_g^*
\sim
\operatorname{Uniform}(\mathfrak M_g^*).
}
$$

主实现使用 quota-aware exact dynamic programming 精确求解。每个 DP state
保存最优值和最优路径数，并按 predecessor path count 做 seeded backtracking，
因此无需物化 $\mathfrak M_g^*$ 也能保证对完整最优 allocations 均匀采样。
旧 Cartesian solver 仅作为小规模测试 oracle。

---

## 14.5 Seeded uniform tie-breaking

为了可复现，随机数 seed 由以下信息确定性组合：

```text
global_seed
policy_update_id
task_id
anchor_key / allocation_level
```

因此：

- 不引入 count/depth/horizon 等额外 acquisition heuristic；
- 同一 seed 与同一训练状态下完全可复现。

---

## 14.6 Threshold equality

阈值采用闭区间：

$$
\boxed{
\delta\ge\tau_{\mathrm{BERV}}
\Rightarrow
\text{valid information slot}.
}
$$

若多个 slots 恰好都等于 threshold，它们同时有效；是否被最终采用由 global Batch-ERV budget allocation 决定。

---

# 15. Branch Origin Selection 与 Replay

## 15.1 Origin selection

当 global plan 中选中 experiment：

$$
(z,u),
$$

从：

$$
\mathcal I_g^{\mathrm{obs}}(z,u)
$$

均匀选择 concrete natural occurrence：

$$
\boxed{
i_b
\sim
\operatorname{Uniform}
(\mathcal I_g^{\mathrm{obs}}(z,u)).
}
$$

若同一个 $(z,u)$ 计划两次 branches，则为两条 branches 分别进行 seeded uniform origin sampling。

---

## 15.2 ALFWorld replay

对选中的 natural occurrence，保存：

```text
task_id
game_file
root_id
step_index
environment action prefix
pre-action observation
original model context/history
original CoT-action response
```

执行 branch：

```text
1. 创建/复用同一 ALFWorld TextWorld game。
2. reset()。
3. 机械执行 anchor 之前的环境 action prefix。
4. 检查 restored pre-action observation key == recorded anchor key。
5. 检查 nonterminal。
6. 恢复与原 natural occurrence 对齐的 GiGPO memory / prompt state。
7. 复制该 natural occurrence 原本生成的 CoT-action response。
8. 再次执行其中的同一个 action u。
9. 从新的 observation 开始，用冻结 pi_old 生成 fresh continuation suffix。
10. 直到 terminal / horizon。
```

---

## 15.3 Replay prefix 与 branch origin 的区别

### Mechanical replay prefix

Anchor 之前的 action prefix：

- 只恢复环境；
- 不作为新的 decision occurrences；
- training mask 为 $0$；
- 不重复进入 PPO。

### Copied branch origin

Anchor 处复制的自然 CoT-action response：

- 是 branch 的第一个训练 decision occurrence；
- 继承 branch leaf outcome；
- 进入最终 state group；
- 进入 PPO loss；
- 因为 response 来自冻结 $\pi_{\mathrm{old}}$ 的自然 occurrence，其 old token log-prob 可以直接复用。

---

# 16. Parallel Branch Rollout

所有 branch experiments：

$$
\mathcal E_g^{\mathrm{branch}}
=
\{(z_b,u_b,i_b)\}_{b=1}^{Q_g}
$$

在正式 rollout 前全部确定。

不同 tasks 的所有 branches 可以合并：

$$
\bigcup_g\mathcal E_g^{\mathrm{branch}}.
$$

然后一次性进入第二个主要 LLM rollout wave。

因此在没有额外 capacity root wave 的理想情况下，主要生成流程约为：

```text
Wave 1: all planned natural roots
Wave 2: all planned branches
```

这正是 Batch-ERV 版本相较 sequential ERV 的主要系统收益。

---

# 17. Branch Outcome 的作用

Branch outcomes 不再用于：

```text
branch 1 outcome
-> posterior update
-> recompute ERV
-> choose branch 2
```

因为 branch plan 已经联合确定。

Branch outcomes 现在用于：

1. 形成 terminal leaves；
2. 增加 selected $(z,u)$ 的真实 continuation evidence；
3. 构建最终 GiGPO state groups；
4. 计算 local credit；
5. policy optimization；
6. 诊断 predicted BERV 与 realized evidence improvement 的一致性。

局部 Beta posterior 可以在 branch 后离线更新用于 diagnostics，但不是当前 batch 的 acquisition feedback loop。

---

# 18. 最终 Tree、Leaf 与 Training Occurrence

## 18.1 Terminal leaf set

对任务实例 $g$：

$$
\boxed{
\mathcal L_g
=
\mathcal L_g^{\mathrm{root}}
\cup
\mathcal L_g^{\mathrm{branch}}.
}
$$

满足：

$$
|\mathcal L_g|=B.
$$

每条 branch leaf 概念上对应：

$$
\tau_{g,\ell}^{\mathrm{full}}
=
P_{g,r\rightarrow z}
\oplus
\xi_{g,z,b}.
$$

但 full path 只用于解释 terminal leaf 和统计，不等于完整 path 都要重复训练。

---

## 18.2 Training occurrence set

训练数据包括：

$$
\boxed{
\mathcal D_g
=
\mathcal D_g^{\mathrm{root}}
\cup
\mathcal D_g^{\mathrm{branch-origin}}
\cup
\mathcal D_g^{\mathrm{branch-suffix}}.
}
$$

不包括：

$$
\boxed{
\mathcal D_g^{\mathrm{mechanical\ replay\ prefix}}.
}
$$

这样避免 shared prefix 因 descendant branch 数量被重复放大。

---

# 19. GiGPO-Compatible Global Trajectory Advantage

## 19.1 Occurrence-weighted normalization（主配置）

为保持与当前 GiGPO implementation baseline 的下游 credit estimator 一致，
主配置对所有真实 trainable decision occurrences 统计 global baseline。设
$\mathcal I_g$ 为 task $g$ 的 occurrence rows，$\ell(i)$ 是 occurrence $i$
所属 terminal trajectory：

$$
\mu_g^E
=
\frac1{|\mathcal I_g|}
\sum_{i\in\mathcal I_g}R_{g,\ell(i)},
$$

$$
\sigma_g^E
=
\operatorname{Std}
\left(
\{R_{g,\ell(i)}\}_{i\in\mathcal I_g}
\right).
$$

定义：

$$
\boxed{
A_{g,i}^E
=
\frac{R_{g,\ell(i)}-\mu_g^E}
{\sigma_g^E+\epsilon_{\mathrm{norm}}}.
}
$$

若：

$$
\sigma_g^E=0,
$$

则：

$$
A_{g,i}^E=0.
$$

这是一项有意的 baseline-alignment 选择：BACE 主实验改变 rollout evidence 的
acquisition/topology，但保持 GiGPO macro/local credit 和 PPO optimizer 语义。
terminal-leaf-uniform normalization 保留为 tree-aware optimization ablation，
不作为主方法必需组件。

---

## 19.2 Shared prefix 不重复训练

虽然 branch leaf 的 full path 在全局 reward statistics 中有完整 initial-to-terminal 语义，但机械 replay prefix 不重复进入训练。

因此：

- root 中原始 prefix occurrence 只训练一次；
- branch copied origin 和 fresh suffix 作为 branch 新 evidence 训练；
- descendant branch 数不会把同一 mechanical prefix 梯度放大 $K$ 倍。

主版本采用 GiGPO-compatible occurrence-weighted global normalization；
leaf-uniform 与 lineage-balanced normalization 作为消融。

---

# 20. Local GiGPO Credit

## 20.1 Final occurrence groups

所有 root 和 branch 完成后，对**真实训练 decision occurrences**重新分组。

定义：

$$
\mathcal I_g(z)
=
\{i:\kappa(o_i^{\mathrm{pre}})=z,\ i\text{ trainable}\}.
$$

可包含：

- natural root occurrence；
- copied branch-origin occurrence；
- branch fresh suffix occurrence；
- root--branch collisions；
- branch--branch collisions。

Mechanical replay prefix 不进入。

---

## 20.2 Occurrence-level local advantage：主版本

对 occurrence $i\in\mathcal I_g(z)$，定义 return-to-go：

$$
G_i.
$$

在 binary terminal reward、$\gamma=1$ 时：

$$
G_i=R_{\mathrm{leaf}(i)}\in\{0,1\}.
$$

状态组：

$$
\mu_{g,z}^S
=
\frac1{|\mathcal I_g(z)|}
\sum_{i\in\mathcal I_g(z)}G_i,
$$

$$
\sigma_{g,z}^S
=
\operatorname{Std}
\{G_i:i\in\mathcal I_g(z)\}.
$$

主版本局部优势：

$$
\boxed{
A_i^{S,\mathrm{occ}}
=
\frac{G_i-\mu_{g,z}^S}
{\sigma_{g,z}^S+\epsilon_{\mathrm{norm}}}.
}
$$

若 group size $<2$ 或 $\sigma=0$：

$$
A_i^{S,\mathrm{occ}}=0.
$$

这一版本最严格对齐原始 GiGPO。

---

## 20.3 Action-aggregated local credit：增强/消融

定义：

$$
\mathcal I_g(z,u)
=
\{i\in\mathcal I_g(z):\operatorname{can}(a_i)=u\}.
$$

动作平均 return：

$$
\bar G_{g,z,u}
=
\frac1{|\mathcal I_g(z,u)|}
\sum_{i\in\mathcal I_g(z,u)}G_i.
$$

Action-aggregated local advantage：

$$
\boxed{
A_{g,z,u}^{S,\mathrm{act}}
=
\frac{\bar G_{g,z,u}-\mu_{g,z}^S}
{\sigma_{g,z}^S+\epsilon_{\mathrm{norm}}}.
}
$$

并赋给所有同 $(z,u)$ occurrences。

例如某动作两次 occurrence local raw advantages 为：

$$
+1,-1,
$$

则动作聚合后：

$$
0,0.
$$

该版本更契合采集阶段的 $(z,u)$ posterior，但由于与已有 action-consistent credit 方法较接近，主论文应把其定位为可组合增强，而不是 BACE 的主要 novelty。

---

# 21. Final Advantage

主版本 occurrence-level：

$$
\boxed{
A_i
=
A_{g,\mathrm{leaf}(i)}^E
+
\omega A_i^{S,\mathrm{occ}}.
}
$$

Action-aggregated 版本：

$$
\boxed{
A_i
=
A_{g,\mathrm{leaf}(i)}^E
+
\omega A_{g,z_i,u_i}^{S,\mathrm{act}}.
}
$$

只聚合 local term，不聚合 global leaf advantage。

因此两个相同动作但最终 leaf outcome 不同的 occurrences，仍然可以拥有不同 global trajectory credit。

---

# 22. Unified PPO Objective

## 22.1 Standard old-policy ratio

对任意训练 token：

$$
r_{i,k}(\theta)
=
\frac{
\pi_\theta(y_{i,k}\mid h_{i,k})
}{
\pi_{\mathrm{old}}(y_{i,k}\mid h_{i,k})
}.
$$

PPO clipped surrogate：

$$
\ell_{\mathrm{PPO}}(r,A)
=
\min
\left(
rA,
\operatorname{clip}(r,1-\epsilon,1+\epsilon)A
\right).
$$

---

## 22.2 不使用额外 behavior correction

Branch acquisition 是主动选择：

$$
(z,u)\sim\text{Batch-ERV controller}.
$$

主版本不乘：

$$
\frac{\pi_{\mathrm{old}}}{\mu_{\mathrm{acq}}}.
$$

原因是当前 branch data 被明确解释为：

$$
\boxed{
\text{acquisition-weighted training evidence}
}
$$

而不是试图无偏重建 vanilla on-policy GiGPO trajectory distribution。

同时：

- anchor selection shift 本身无法通过 action-only importance ratio 完整校正；
- Batch-ERV 现在是组合实验选择，而不是简单可写成单步 behavior distribution 的随机策略；
- 额外 importance correction 会增加方差并削弱主动采集到的 rare but useful evidence。

---

## 22.3 单一 unified loss

定义全部 trainable tokens 的集合：

$$
\mathcal D
=
\mathcal D_{\mathrm{root}}
\cup
\mathcal D_{\mathrm{branch-origin}}
\cup
\mathcal D_{\mathrm{branch-suffix}}.
$$

统一 PPO objective：

$$
\boxed{
\mathcal L_{\mathrm{PPO}}
=
-
\operatorname{Agg}_{i,k\in\mathcal D}
\ell_{\mathrm{PPO}}
(r_{i,k},A_i).
}
$$

其中 `Agg` 保持与实际 GiGPO/verl-agent 配置一致的 response/token mask 与 loss aggregation 语义。

总目标：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+
\beta_{\mathrm{KL}}\mathcal L_{\mathrm{KL}}.
}
$$

主版本不再把 root 和 branch 写成两套独立 loss，也不引入 $\lambda_{\mathrm{branch}}$。

Branch 的训练影响由 controller 实际产生的 trainable occurrences 数量自然体现。

---

# 23. Exact Batch-ERV 的理论性质

## 23.1 Non-negativity

由：

$$
\operatorname{BERV}(\mathcal B)
=
E_{\mathbf Y}
\left[
\max_uE[p_u\mid\mathcal D,\mathbf Y]
\right]
-
\max_uE[p_u\mid\mathcal D],
$$

因为获得信息后决策者总可以忽略信息并维持旧决策，所以：

$$
\boxed{
\operatorname{BERV}(\mathcal B)\ge0.
}
$$

它就是该 batch experiment 对 posterior-mean action decision 的 expected value of sample information。

---

## 23.2 Batch vs Sequential

令：

$$
V_{\mathrm{batch}}^*(Q)
$$

为在观察任何新 outcome 前一次性设计 $Q$ 个 experiments 的最优 expected regret reduction。

令：

$$
V_{\mathrm{seq}}^*(Q)
$$

为每观察一个 outcome 后允许重新选择下一 experiment 的最优 sequential value。

Sequential policy 可以模拟任意 batch plan，因此：

$$
\boxed{
V_{\mathrm{batch}}^*(Q)
\le
V_{\mathrm{seq}}^*(Q).
}
$$

这准确表达本方法的折中：

$$
\boxed{
\text{牺牲统计 adaptivity，换取系统 parallelism。}
}
$$

---

## 23.3 Exact Batch-ERV vs naive one-step top-$Q$

设 naive planner 直接按初始 one-step ERV 选 $Q$ 个 experiments，得到 plan：

$$
\mathcal B_{\mathrm{naive}}.
$$

只要它满足相同 budget/capacity constraints，它就是 batch feasible plans 中的一个候选。

所以最优 Batch-ERV：

$$
\boxed{
V_{\mathrm{batch}}^*(Q)
\ge
V(\mathcal B_{\mathrm{naive}}).
}
$$

因此：

$$
\boxed{
\text{naive parallel}
\le
\text{optimal Batch-ERV}
\le
\text{optimal sequential ERV}.
}
$$

---

## 23.4 One-step ERV 为 $0$ 的决策解释

对动作 $u$：

$$
\operatorname{ERV}(z,u)=0
$$

并不意味着 posterior variance 为 $0$。

它表示：

> 一次额外 outcome 无论成功或失败，都不会改变 posterior-mean 最优动作对应的 expected decision value。

因此 ERV 是 decision-relevant uncertainty，而不是 generic uncertainty。

---

## 23.5 与局部策略梯度方向的联系

在二动作 / top-two subspace 中，设：

$$
m_a\ge m_b.
$$

局部 logit difference 为：

$$
d=\ell_a-\ell_b.
$$

真实局部策略梯度：

$$
\frac{\partial J_z}{\partial d}
=
\pi_a\pi_b(p_a-p_b).
$$

错误梯度方向的 posterior risk 可写为：

$$
\mathcal R_{\mathrm{grad-sign}}(z)
=
\pi_a\pi_b\mathcal R(z).
$$

在 batch 内 $\pi_a,\pi_b$ 固定，因此 Batch-ERV 对 Bayes simple regret 的 expected reduction 同时对应 top-two 局部错误梯度方向风险的比例降低。

这构成 acquisition 与 GiGPO optimization 的理论桥梁。

---

# 24. 为什么精确公式优于 Monte Carlo

旧实现思路需要：

```text
sample Beta values
-> estimate E[max p]
-> hypothetical posterior
-> estimate E[max p] again
-> subtract regrets
```

当前版本直接：

```text
Beta posterior parameters
-> Beta-Binomial predictive probabilities
-> exact future posterior means
-> finite enumeration
-> exact Batch-ERV
```

收益包括：

1. 无 MC sampling noise；
2. 无 $M_{\mathrm{MC}}$ 超参数；
3. capacity threshold 不再受 MC jitter 影响；
4. 同一 posterior 必然得到同一 BERV；
5. tie 的含义清晰；
6. 更容易单元测试；
7. 理论表达更直接；
8. 计算开销相对 LLM rollout 可忽略。

若为了理论诊断确实需要绝对 Bayes simple regret：

$$
E[\max_up_u]
=
\int_0^1
\left[
1-\prod_u I_x(\alpha_u,\beta_u)
\right]dx,
$$

可使用确定性一维 quadrature，而不必 MC。

主 controller 不需要这一积分。

---

# 25. 完整算法伪代码

```text
Input:
    task batch G
    total leaf budget B
    minimum roots R_min
    family histories H_{c,k-1}
    competence threshold tau_comp
    local prior strength kappa_A
    per-anchor branch cap L_max = 2
    information threshold tau_BERV
    frozen actor pi_old
    global random seed

# --------------------------------------------------
# Phase 0: Build lagged family competence beliefs
# --------------------------------------------------
for each task family c:
    compute historical Beta mean mu_c
    compute bounded concentration kappa_c^T
    phi_c ~ Beta(kappa_c^T * mu_c,
                 kappa_c^T * (1 - mu_c))
    q_c = P(phi_c > tau_comp)

# --------------------------------------------------
# Phase 1: Plan initial root/branch topology
# --------------------------------------------------
for each task g:
    Q_bar[g] = round((B - R_min) * q_{c_g})
    Q[g] = clip(Q_bar[g], 0, B - R_min)
    R[g] = B - Q[g]

# --------------------------------------------------
# Phase 2: Natural-root rollout wave
# --------------------------------------------------
run all planned natural roots across tasks in parallel

# --------------------------------------------------
# Phase 3: Root-side capacity correction
# --------------------------------------------------
for each task g:
    while Q[g] > 0:
        build exact GiGPO anchor groups from all current natural roots
        build structural anchors
        build observed action sets and origin pools

        compute current-instance competence posterior
        initialize/update local Beta posteriors

        for each structural anchor z:
            enumerate all 1-branch plans
            compute exact BERV via Beta-Binomial enumeration
            obtain V^(1)(z)

            enumerate all 2-branch plans
            compute exact BERV
            obtain V^(2)(z)

            delta1(z) = V^(1)(z)
            delta2(z) = V^(2)(z) - V^(1)(z)
            capacity(z)
                = 1[delta1 >= tau_BERV]
                + 1[delta2 >= tau_BERV]

        total_capacity = sum_z capacity(z)

        if total_capacity >= Q[g]:
            break

        Q[g] <- Q[g] - 1
        R[g] <- R[g] + 1
        generate one additional natural root for g

    freeze final natural-root backbone and local candidate structures

# --------------------------------------------------
# Phase 4: Exact global Batch-ERV allocation
# --------------------------------------------------
for each task g with Q[g] > 0:
    for each structural anchor z:
        cache V^(0)(z), V^(1)(z), V^(2)(z)
        cache all tie-optimal local plans

    solve exactly with quota-aware DP:
        maximize sum_z V^(m_z)(z)
        subject to sum_z m_z = Q[g]
                   m_z in {0,1,2}
                   information-capacity constraints

    if multiple global optimum allocations tie:
        count optimal DP paths
        sample uniformly by count-weighted deterministic seeded backtracking

    for every anchor z with m_z > 0:
        if multiple local plans tie:
            sample uniformly using deterministic seeded RNG

        expand selected local plan into concrete (z,u) experiments

        for each selected (z,u):
            sample origin uniformly from natural I_obs(z,u)

# --------------------------------------------------
# Phase 5: Parallel branch rollout wave
# --------------------------------------------------
collect all branch experiment records across tasks

for each branch in parallel:
    reset same environment instance
    mechanically replay prefix before anchor
    verify restored anchor
    restore original model context/memory
    copy original natural CoT-action response at origin
    execute same action u
    sample fresh suffix with frozen pi_old
    finish to terminal/horizon

# --------------------------------------------------
# Phase 6: Build final tree statistics
# --------------------------------------------------
for each task g:
    construct all terminal trajectories: roots + branches
    compute GiGPO-compatible occurrence-weighted global A^E

    build final exact state groups from trainable occurrences only:
        natural root occurrences
        copied branch-origin occurrences
        fresh branch suffix occurrences

    exclude mechanical replay prefix

    compute occurrence-level GiGPO A^S
    optionally compute action-aggregated A^S

    final advantage:
        A = A^E + omega * A^S

# --------------------------------------------------
# Phase 7: Unified PPO
# --------------------------------------------------
train all root / branch-origin / branch-suffix tokens
with standard PPO ratio pi_theta / pi_old
and the same GiGPO masking/aggregation convention

update actor

# --------------------------------------------------
# Phase 8: Update lagged family history
# --------------------------------------------------
use ALL natural roots from this batch,
including capacity-correction roots,
to update task-family natural-root history

do NOT use branch outcomes

next policy update repeats the process
```

---

# 26. 完整 ALFWorld 数值例子

下面用一个从 prior 到 PPO 的完整例子说明方法。

## 26.1 任务设定

假设任务实例 $g$ 属于：

```text
pick_and_place
```

总 leaf budget：

$$
B=8,
$$

最少 roots：

$$
R_{\min}=2,
$$

每 anchor 最大 branches：

$$
L_{\max}=2.
$$

设上一批以前该 family 的 bounded competence belief 为：

$$
\phi_c\sim\operatorname{Beta}(6,4).
$$

并设：

$$
\tau_{\mathrm{comp}}=0.5.
$$

假设计算得到：

$$
q_c=P(\phi_c>0.5)=0.746\approx0.75.
$$

于是：

$$
M=B-R_{\min}=6,
$$

$$
\bar Q
=
\operatorname{round}(6\times0.73)
=4.
$$

初始计划：

$$
\boxed{4R+4B.}
$$

这里没有 pilot wave，四条 natural roots 直接同时 rollout。

---

## 26.2 四条 natural roots

假设 outcomes：

| Root | Terminal reward |
|---|---:|
| $r_1$ | $1$ |
| $r_2$ | $0$ |
| $r_3$ | $1$ |
| $r_4$ | $0$ |

当前 natural-root empirical success：

$$
\hat p_g=0.5.
$$

它用于 current-instance posterior 与后续 local prior；但不会回头重新计算 initial $4R+4B$ quota。

---

## 26.3 从 roots 构造 anchors

假设 exact pre-action observation grouping 后形成两个 repeated states：

### Anchor $z_1$

natural occurrences：

| Occurrence | Canonical action | Final reward |
|---|---|---:|
| $r_1,t_4$ | `go to countertop 1` | $1$ |
| $r_2,t_5$ | `go to countertop 1` | $0$ |
| $r_3,t_4$ | `open cabinet 1` | $1$ |

因此：

$$
\mathcal C(z_1)=\{a,b\},
$$

其中：

$$
a=\texttt{go to countertop 1},
$$

$$
b=\texttt{open cabinet 1}.
$$

### Anchor $z_2$

| Occurrence | Canonical action | Final reward |
|---|---|---:|
| $r_2,t_8$ | `take apple 1 from table 1` | $0$ |
| $r_4,t_7$ | `go to fridge 1` | $0$ |

两个动作均有效，但目前自然 evidence 全失败。

---

## 26.4 Current-instance posterior

假设 family transferred prior：

$$
\phi_g\sim\operatorname{Beta}(3,2).
$$

四条 roots 中：

$$
S_g=2,
\qquad
F_g=2.
$$

所以：

$$
\phi_g\mid D_g^{root}
\sim
\operatorname{Beta}(5,4).
$$

posterior mean：

$$
\bar\phi_g=\frac59\approx0.556.
$$

取：

$$
\kappa_A=2.
$$

每个动作局部 prior center：

$$
\alpha_0\approx1.111,
\qquad
\beta_0\approx0.889.
$$

---

## 26.5 Anchor $z_1$ 的 local posteriors

动作 $a$：两次自然 evidence，$1$ success、$1$ failure：

$$
p_a
\sim
\operatorname{Beta}(2.111,1.889).
$$

所以：

$$
m_a\approx0.528.
$$

动作 $b$：一次 success：

$$
p_b
\sim
\operatorname{Beta}(2.111,0.889).
$$

所以：

$$
m_b\approx0.704.
$$

当前 posterior-mean incumbent 是 $b$。

---

## 26.6 Exact one-step ERV 示例

考虑对 $a$ 再采一次。

当前：

$$
\alpha_a=2.111,
\qquad
\beta_a=1.889,
\qquad
N_a=4.
$$

成功后的 mean：

$$
m_a^+
=
\frac{3.111}{5}
\approx0.622.
$$

失败后的 mean：

$$
m_a^-
=
\frac{2.111}{5}
\approx0.422.
$$

其他动作最佳 mean：

$$
c=m_b\approx0.704.
$$

由于：

$$
c\ge m_a^+,
$$

一次采样 $a$ 无论成功失败都无法使其 posterior mean 超过 $b$。

因此：

$$
\boxed{
ERV(z_1,a)=0.
}
$$

这说明：虽然 $a$ posterior 仍有 uncertainty，但一次额外 evidence 对当前动作选择没有 immediate decision value。

---

## 26.7 Batch-ERV 能看到 one-step ERV 看不到的联合价值

假设对 $a$ 连续采两次时，如果两次都成功，其 posterior mean 可以超过 $b$。

那么可能出现：

$$
ERV([a])=0,
$$

但：

$$
BERV([a,a])>0.
$$

这就是 Batch-ERV 相对 naive one-step ERV 排序的重要区别：

> 两项联合实验可能有价值，即使其中单项实验不足以立刻改变 incumbent。

---

## 26.8 假设计算得到 local Batch values

为了完整展示 allocation，假设 Exact Batch-ERV 计算得到：

| Anchor | $V^{(1)}$ | 最优 1-branch plan | $V^{(2)}$ | 最优 2-branch plan |
|---|---:|---|---:|---|
| $z_1$ | $0.045$ | $[b]$ | $0.071$ | $[a,b]$ |
| $z_2$ | $0.018$ | $[c]$ | $0.025$ | $[c,d]$ |

于是：

$$
\delta_{z_1}^{(1)}=0.045,
$$

$$
\delta_{z_1}^{(2)}=0.071-0.045=0.026.
$$

$$
\delta_{z_2}^{(1)}=0.018,
$$

$$
\delta_{z_2}^{(2)}=0.025-0.018=0.007.
$$

假设：

$$
\tau_{\mathrm{BERV}}=0.015.
$$

则：

$$
c(z_1)=2,
$$

$$
c(z_2)=1.
$$

总 capacity：

$$
C_g=3.
$$

但当前计划：

$$
Q_g=4.
$$

因为：

$$
3<4,
$$

需要 capacity correction。

---

## 26.9 Capacity correction

执行：

$$
R_g=5,
$$

$$
Q_g=3.
$$

再生成一条 natural root $r_5$。

假设：

$$
Y_{r_5}=1.
$$

它形成一个新的 structural anchor $z_3$，并提供两个 observed actions 的竞争证据。

重新计算得到：

| Anchor | $V^{(1)}$ | $V^{(2)}$ |
|---|---:|---:|
| $z_1$ | $0.041$ | $0.066$ |
| $z_2$ | $0.016$ | $0.023$ |
| $z_3$ | $0.034$ | $0.050$ |

假设 threshold 后：

$$
c(z_1)=2,
\qquad
c(z_2)=1,
\qquad
c(z_3)=2.
$$

所以：

$$
C_g=5\ge3.
$$

停止 root correction。

最终 topology：

$$
\boxed{5R+3B.}
$$

---

## 26.10 Global Batch-ERV allocation

现在需要从三个 anchors 中联合分配三条 branches。

可能 allocations 包括：

$$
(2,1,0),
$$

$$
(2,0,1),
$$

$$
(1,1,1),
$$

$$
(1,0,2),
$$

等等。

其总 value：

### Allocation A

$$
(z_1:2,z_2:1,z_3:0)
$$

$$
V_A=0.066+0.016=0.082.
$$

### Allocation B

$$
(z_1:2,z_2:0,z_3:1)
$$

$$
V_B=0.066+0.034=0.100.
$$

### Allocation C

$$
(z_1:1,z_2:1,z_3:1)
$$

$$
V_C=0.041+0.016+0.034=0.091.
$$

因此：

$$
\boxed{
\mathbf m^*=(2,0,1).
}
$$

即：

- $z_1$ 分两条 branches；
- $z_3$ 分一条 branch；
- $z_2$ 不分。

假设 $z_1$ 的最优 two-branch plan：

$$
[a,b],
$$

$z_3$ 的最优 one-branch plan：

$$
[e].
$$

最终三个 experiments：

$$
\boxed{
(z_1,a),
(z_1,b),
(z_3,e).
}
$$

全部在 branch rollout 前确定。

---

## 26.11 Tie 示例

假设 $z_1$ 其实有：

$$
BERV([a,b])=0.06600000000000,
$$

$$
BERV([a,c])=0.06600000000001.
$$

根据 tie tolerance，两者视为相同。

则：

$$
\mathfrak B_{z_1}^{*(2)}
=
\{[a,b],[a,c]\}.
$$

利用 seeded RNG：

$$
\mathcal B_{z_1}^{*(2)}
\sim
\operatorname{Uniform}
\left(\{[a,b],[a,c]\}\right).
$$

如果本次 seed 选择 $[a,b]$，则整个 run 可完全复现；换不同训练 seed 可以无偏地探索对称最优解。

---

## 26.12 Origin selection

假设 $(z_1,a)$ 有两个自然支持 occurrences：

$$
\mathcal I^{obs}(z_1,a)
=
\{(r_1,t_4),(r_2,t_5)\}.
$$

均匀抽取：

$$
i_1\sim\operatorname{Uniform}\{(r_1,t_4),(r_2,t_5)\}.
$$

假设本次选择：

$$
(r_2,t_5).
$$

然后：

- replay $r_2$ 在 $t_5$ 之前的环境 prefix；
- 恢复原模型 context；
- 复制该 occurrence 原始 CoT-action；
- 再次执行 $a$；
- fresh rollout suffix。

其他两个 branches 同理。

---

## 26.13 三条 branches 并行 rollout

三个 branches 一次并行执行，假设 terminal outcomes：

| Branch | Experiment | Reward |
|---|---|---:|
| $b_1$ | $(z_1,a)$ | $1$ |
| $b_2$ | $(z_1,b)$ | $0$ |
| $b_3$ | $(z_3,e)$ | $1$ |

最终 $8$ 个 terminal leaves：

### Roots

$$
[1,0,1,0,1]
$$

### Branches

$$
[1,0,1]
$$

总 reward vector：

$$
[1,0,1,0,1,1,0,1].
$$

---

## 26.14 All-leaf global advantage

成功数 $5$，失败数 $3$：

$$
\mu_g^E=\frac58=0.625.
$$

若用 population std：

$$
\sigma_g^E
=
\sqrt{0.625(1-0.625)}
\approx0.484.
$$

所以成功 leaf：

$$
A_{\mathrm{succ}}^E
\approx
\frac{1-0.625}{0.484}
\approx0.775.
$$

失败 leaf：

$$
A_{\mathrm{fail}}^E
\approx
\frac{0-0.625}{0.484}
\approx-1.291.
$$

具体实现应使用与 GiGPO 当前代码一致的 std convention。

---

## 26.15 $z_1$ 的 local occurrence advantage

假设最终在 $z_1$ 的 trainable occurrences 共五个：

| Occurrence | Source | Action | Leaf reward |
|---|---|---|---:|
| $o_1$ | root $r_1$ | $a$ | $1$ |
| $o_2$ | root $r_2$ | $a$ | $0$ |
| $o_3$ | root $r_3$ | $b$ | $1$ |
| $o_4$ | copied branch origin $b_1$ | $a$ | $1$ |
| $o_5$ | copied branch origin $b_2$ | $b$ | $0$ |

则：

$$
G=[1,0,1,1,0].
$$

状态均值：

$$
\mu_{z_1}^S=0.6.
$$

成功 occurrence 获得正 local advantage，失败 occurrence 获得负 local advantage。

最终每个 occurrence：

$$
A_i
=
A_{\mathrm{leaf}(i)}^E
+
\omega A_i^S.
$$

注意 copied branch origin $o_4,o_5$ 是 branch 新 occurrence；它们不是 mechanical prefix，因此进入 loss。

---

## 26.16 PPO update

训练 buffer 包含：

```text
5 natural roots 的全部正常 actor responses
+ 3 copied branch-origin responses
+ 3 条 branch 的 fresh suffix responses
```

不包含：

```text
3 条 branch 在 anchor 之前用于环境恢复的 mechanical replay prefix
```

所有训练 token 使用：

$$
r=\frac{\pi_\theta}{\pi_{\mathrm{old}}}.
$$

使用统一：

$$
\mathcal L_{PPO}+\beta_{KL}\mathcal L_{KL}.
$$

---

## 26.17 更新下一批 family history

当前实例最终有 $5$ 条 natural roots：

$$
[1,0,1,0,1].
$$

因此对 task-family history 贡献：

$$
S^{root}=3,
\qquad
F^{root}=2.
$$

三个 branch outcomes：

$$
[1,0,1]
$$

**不进入 family prior。**

下一批再读取更新后的 history 规划新的 topology。

至此，一个完整 BACE-GiGPO update 完成。

---

# 27. 工程编排与时间开销

## 27.1 原 GiGPO

对每个 task 的 $B=8$ 条完整 rollouts，可以一次并行生成。

理想串行 generation depth 约为一个完整 horizon：

$$
H.
$$

---

## 27.2 旧 Pilot + Sequential Branch 版本

需要：

```text
pilot wave
-> additional-root wave
-> branch 1
-> branch 2
-> ...
```

wall-clock 会受到多个全局 barrier 与小 batch generation 的明显影响。

---

## 27.3 当前 Batch-ERV 版本

无 capacity correction 时主要是：

```text
Wave 1: all roots
Wave 2: all branches
```

若平均 branch anchor 位于 trajectory depth $d$，branch fresh suffix 长度约：

$$
H-d.
$$

粗略串行生成深度：

$$
T_{BACE}
\approx
H+(H-d).
$$

相对 GiGPO：

$$
\frac{T_{BACE}}{T_{GiGPO}}
\approx
2-\frac dH.
$$

例如：

- $d/H=0.25$：约 $1.75\times$ generation depth；
- $d/H=0.50$：约 $1.50\times$；
- $d/H=0.70$：约 $1.30\times$。

实际 wall-clock 还取决于 batch size、vLLM/verl rollout throughput 和 capacity correction 次数。

当前合理工程目标是：

$$
\boxed{
\text{约 }1.3\times\sim1.8\times\text{ GiGPO wall-clock}
}
$$

而不是假设必然 $2\times$ 或 $3\times$。

该数字必须由目标 8-GPU ALFWorld 配置真实 profiling 验证。

---

# 28. 与 Sequential ERV 的关系

Sequential ERV 仍然是一个重要 oracle-style / ablation baseline。

比较：

### Sequential

```text
branch 1
-> observe Y1
-> update posterior
-> branch 2
-> observe Y2
-> ...
```

优势：统计 adaptivity 更强。

缺点：多轮 LLM synchronization。

### Batch-ERV（主版本）

```text
jointly design Q experiments
-> run all Q branches in parallel
```

优势：

- GPU batching 好；
- wall-clock 低；
- controller 更简单；
- exact combinatorial value 仍考虑 batch 内重复 sampling 的联合收益。

缺点：

- 无法根据 branch 1 的实际 outcome 改写 branch 2。

因此实验应报告：

$$
\text{statistical value}
\quad\text{vs}\quad
\text{system latency}.
$$

---

# 29. Benchmark 设计

## 29.1 ALFWorld：主 benchmark

ALFWorld 适合作为第一主实验，因为：

- 多步文本交互；
- terminal success reward 清楚；
- 同一 task 上可以采多条 rollouts；
- repeated observations 可以形成 GiGPO anchors；
- TextWorld 环境支持 reset + deterministic prefix replay 的工程实现；
- 适合验证 branch replay、local credit 与 wall-clock。

Task-family prior 可按 ALFWorld task type 建立，例如：

- pick-and-place；
- clean-and-place；
- heat-and-place；
- cool-and-place；
- examine；
- pick-two-object。

具体名称与数据划分在实现时与 verl-agent/GiGPO 当前配置保持一致。

---

## 29.2 WebShop：第二 benchmark

WebShop 用于验证：

- 方法不依赖 ALFWorld 的特定 symbolic dynamics；
- session/state replay 在网页交互式 agent 中是否可行；
- Batch-ERV 是否仍能找到有价值的 local decision edges。

Task-family 可以按现有训练数据可稳定获得的 query/product-category metadata 定义；若没有可靠 family label，则可使用更粗粒度的 global family prior，并把 family granularity 作为消融。

---

# 30. 主要实验问题

## 30.1 主性能

与 GiGPO 比较：

- success rate；
- reward；
- learning curve；
- sample efficiency；
- generated tokens；
- environment steps；
- wall-clock；
- GPU hours。

---

## 30.2 Topology 是否有效

记录：

- $q_{c,k}$；
- $\bar Q_g$；
- final $Q_g$；
- final $R_g$；
- capacity correction 次数；
- branch quota retention ratio；
- task-family topology distribution；
- root success rate calibration。

---

## 30.3 Batch-ERV 是否真的比简单 branching 好

比较：

1. Random branches；
2. Entropy-based branch selection；
3. Posterior-variance branch selection；
4. One-step ERV top-$Q$；
5. Exact Batch-ERV；
6. Sequential ERV。

关键指标：

- final success；
- action-ranking accuracy；
- mixed-outcome anchor frequency；
- selected anchor 的 realized evidence improvement；
- branch redundancy；
- branch wall-clock。

---

# 31. 必须做的消融

## 31.1 Topology controller

比较：

- Dynamic family-prior topology（主）；
- fixed $8R+0B$；
- fixed $6R+2B$；
- fixed $4R+4B$；
- fixed $2R+6B$；
- previous pilot-based instance controller。

用于回答：

> 去掉 pilot 的 family-guided topology 是否足以获得收益，以及省下的 wall-clock 是否值得。

---

## 31.2 Family-history weighting

比较：

- all-natural-roots pooling（主）；
- per-instance normalized history；
- fixed weak prior；
- global prior without task family。

重点检查 root-heavy hard instances 是否形成过度保守反馈。

---

## 31.3 Capacity

比较：

- Exact Batch-ERV marginal capacity（主）；
- fixed $L_{\max}|Z|$ capacity；
- no capacity correction；
- one-anchor-one-branch。

---

## 31.4 Acquisition

比较：

- Exact Batch-ERV（主）；
- one-step ERV top-$Q$；
- sequential ERV；
- random；
- entropy；
- variance；
- raw Bayes regret。

---

## 31.5 Exact computation validation

不是主性能消融，而是正确性验证：

- exact finite-sum BERV；
- very-high-sample Monte Carlo approximation。

两者应在误差范围内一致。

主算法始终使用 exact 版本。

---

## 31.6 Tie-breaking

主版本：

$$
\boxed{
\text{seeded uniform over tie-optimal set}
}
$$

可诊断比较：

- deterministic lexicographic；
- lower evidence count first；
- deeper/shallower anchor first。

这些不作为主方法，因为会引入第二 acquisition objective。

---

## 31.7 Local credit

比较：

- occurrence-level GiGPO（主）；
- simple action aggregation；
- action shrinkage / calibrated variant。

---

## 31.8 Global normalization

比较：

- all-leaf uniform（主）；
- root/branch separate；
- lineage-balanced。

---

## 31.9 Replay/prefix

比较或诊断：

- unique-segment training（主）；
- flat full-path duplication；
- descendant-mean prefix backup。

重点记录 prefix gradient mass 与 descendant count 的关系。

---

# 32. 诊断指标

至少记录以下日志。

## 32.1 Prior / topology

```text
family posterior mean
family controller concentration
q_c
planned Q_bar
final Q
final R
root correction count
all-root family history counts
```

## 32.2 Anchor

```text
unique anchor count
repeated anchor count
structural anchor count
anchor occurrence count
distinct observed-action count
within-trajectory repeat ratio
```

## 32.3 Exact BERV

```text
V^(1)(z)
V^(2)(z)
delta^(1)(z)
delta^(2)(z)
selected local plans
global optimal allocation value
number of tie-optimal local plans
number of tie-optimal global allocations
```

## 32.4 Replay

```text
replay success rate
observation-key mismatch
terminal mismatch
origin action mismatch
replay environment steps
replay latency
```

## 32.5 Optimization

```text
leaf reward distribution
A^E distribution
A^S distribution
branch-origin advantage distribution
PPO clipping fraction
policy KL
gradient norm
root/branch trainable token ratio
```

## 32.6 Systems

```text
root rollout wall-clock
capacity-correction wall-clock
branch rollout wall-clock
PPO wall-clock
GPU utilization
generated token throughput
number of rollout waves
```

---

# 33. 方法的核心创新定位

当前版本不应把 novelty 写成“首次做 branching”或“首次动态分配 rollout”。

更准确的核心贡献是：

## 33.1 Competence-guided breadth/refinement allocation

使用滞后的 natural-root competence belief，在固定 terminal-leaf budget 中规划 breadth 与 refinement 比例，并通过当前实例的 Batch-ERV information capacity 做保守可行性修正。

---

## 33.2 Decision-theoretic active credit acquisition

采集目标不是 generic uncertainty，而是：

$$
\boxed{
\text{expected reduction in Bayes action-selection regret}.
}
$$

这直接关注“新的证据是否可能改变动作选择”。

---

## 33.3 Exact parallel Batch experimental design

不是 sequential greedy branching，也不是 one-step score top-$Q$。

主方法使用：

$$
\boxed{
\text{Beta-Binomial exact Batch-EVSI}
}
$$

联合设计整个 parallel branch batch。

这同时具有：

- 明确 decision-theoretic objective；
- 可精确计算；
- GPU-friendly parallel rollout；
- 对 repeated action sampling 的边际价值建模。

---

## 33.4 Observed-edge continuation resampling

BACE 不任意强制 unseen actions，而是对 natural policy 已经真实生成过的 CoT-action edges 进行 continuation resampling。

因此方法更接近：

$$
\boxed{
\text{active construction of local comparison evidence}
}
$$

而不是通用 tree search。

---

## 33.5 Acquisition 与 GiGPO credit 的闭环

采集阶段以 $(z,u)$ credit uncertainty 为对象；训练阶段把新增 branches 重新放回 GiGPO 的：

$$
\text{global leaf quality}
+
\text{local state credit}
$$

体系中。

因此 active acquisition 与 local credit assignment 指向同一个 decision structure。

---

# 34. 已知局限

## 34.1 Family-level topology 无法完全识别当前实例难度

取消 pilot 后，initial topology 不再 instance-posterior-adaptive。

如果 family 通常简单但某个实例异常困难，controller 可能一开始 branch 过多；capacity correction 可以向 roots 方向修正。

反过来，如果 family 通常困难但某个实例异常简单，initial roots 已经生成后无法回收为 branches。

因此当前版本具有保守不对称性。

---

## 34.2 All-root history 的 controller feedback

困难实例可能因 capacity correction 产生更多 roots，而这些 roots 又向 family history 贡献更多 failures。

这可能形成：

$$
\text{hard instance}
\rightarrow
\text{more roots}
\rightarrow
\text{more failure evidence}
\rightarrow
\text{future fewer branches}.
$$

主版本将其解释为 root-level competence evidence；per-instance normalization 必须作为重要消融。

---

## 34.3 Beta-Bernoulli 是工作模型

同一个 $(z,u)$ 的不同 origins：

- history 不同；
- latent state 可能不完全相同；
- continuation randomness 不同。

因此 conditional exchangeability 只是工作近似。

需要通过：

- history-aware ablation；
- origin-level diagnostics；
- within-action variance；

检查模型失配。

---

## 34.4 Batch 牺牲 outcome adaptivity

Batch-ERV 是并行系统与统计效率的折中，并不声称优于最优 sequential experimental design。

---

## 34.5 Acquisition-weighted objective 不等于 vanilla on-policy GiGPO

Branch locations 是主动选择的，因此最终训练 distribution 被有意改变。

论文应使用：

> **tree-induced, acquisition-weighted GiGPO training**

而不是声称整个 estimator 是 initial-state vanilla policy objective 的完全无偏估计。

---

# 35. 推荐主超参数

| 参数 | 含义 | 主建议 |
|---|---|---:|
| $B$ | terminal-leaf budget | $8$ |
| $R_{\min}$ | 最少 natural roots | $2$ |
| $\tau_{\mathrm{comp}}$ | refinement regime threshold | $0.5$ |
| $\lambda_{\mathrm{hist}}$ | family history exponential decay | $0.8$--$0.95$ |
| $A_0,B_0$ | family Beta base prior | $1,1$ |
| $\tau_T$ | history-to-controller strength scale | $0.1$ 起步 |
| $\kappa_T^{\min}$ | controller concentration 下限 | $2$ |
| $\kappa_T^{\max}$ | controller concentration 上限 | $4$ 或 $8$ |
| $\kappa_A$ | local action prior strength | $2$ |
| $L_{\max}$ | 每 anchor 最大 branches | $2$ |
| $\tau_{\mathrm{BERV}}$ | marginal information threshold | 通过小规模 sweep 定 |
| $\epsilon_{\mathrm{abs}}$ | BERV tie abs tolerance | $10^{-12}$ |
| $\epsilon_{\mathrm{rel}}$ | BERV tie rel tolerance | $10^{-10}$ |
| $\omega$ | GiGPO local advantage 权重 | 与 GiGPO 主配置对齐/小规模 sweep |
| $\epsilon_{\mathrm{PPO}}$ | PPO clip | 与 GiGPO 对齐 |
| $\beta_{KL}$ | reference KL | 与 GiGPO 对齐 |

明确删除：

$$
\boxed{N_{\mathrm{pilot}}}
$$

和：

$$
\boxed{M_{\mathrm{MC}}.}
$$

---

# 36. 论文式方法概述

可以将主方法概括为：

> We treat agent rollouts as budgeted experiments for constructing local credit evidence. At each policy update, a lagged task-family Beta competence belief allocates the fixed terminal-leaf budget between independent natural roots and local refinement branches. Natural roots expose repeated GiGPO states and naturally executed decision edges. For every eligible state-action edge, we maintain a weakly regularized Beta posterior over terminal success under the frozen policy. We then formulate branch acquisition as a parallel Bayesian experimental-design problem: the value of a branch batch is the expected reduction in Bayes simple regret after observing its outcomes. Under the Beta-Bernoulli model, this Batch-ERV admits an exact finite-sum computation using Beta-Binomial predictive probabilities, eliminating Monte Carlo estimation. We jointly allocate the branch budget across anchors and actions, uniformly breaking exact numerical ties with a seeded random rule, and execute all selected branches in parallel by replaying the corresponding natural prefixes and resampling continuation from the observed CoT-action edges. The resulting root and branch leaves are optimized with a unified GiGPO-style global-plus-local credit objective while excluding mechanical replay prefixes from training.

中文：

> 我们将 agent rollout 视为用于主动构造局部信用证据的有限预算实验。每次策略更新时，滞后的 task-family Beta competence belief 在固定 terminal-leaf budget 中规划独立 natural roots 与局部 refinement branches 的比例。Natural roots 暴露重复的 GiGPO 状态以及策略自然执行过的 decision edges；对每个符合条件的状态—动作边，我们在冻结策略下维护一个弱正则化的 terminal-success Beta posterior。随后，我们把 branch acquisition 表述为并行贝叶斯实验设计问题：一组 branch experiments 的价值定义为观察这些 outcomes 后 Bayes simple regret 的期望降低。在 Beta-Bernoulli 模型下，该 Batch-ERV 可通过 Beta-Binomial predictive probability 和有限 outcome enumeration 精确计算，从而完全去除 Monte Carlo。我们联合分配全部 branch budget，在数值平局时使用 seeded uniform tie-breaking，并通过 prefix replay 与 observed CoT-action edge continuation resampling 一次性并行执行所有 branches。最终 root 与 branch leaves 使用统一的 GiGPO 式全局—局部信用目标训练，同时机械 replay prefix 不重复进入优化。

---

# 37. 一页式最终规则

## Topology

$$
q_{c,k}
=
P(\phi_{c,k}>\tau_{\mathrm{comp}}),
$$

$$
\bar Q_g
=
\operatorname{round}
\left[(B-R_{\min})q_{c_g,k}\right],
$$

$$
R_g^{(0)}=B-\bar Q_g.
$$

无 Pilot。

## Prior update

当前 batch 的全部 natural roots：

$$
\boxed{
\text{all natural roots}
\rightarrow
\text{next-batch family history}
}
$$

Branches：

$$
\boxed{
\text{never update family prior}.
}
$$

## Anchor

$$
\boxed{
|\mathcal I_g^{root}(z)|\ge2,
\quad
|\mathcal C_g^{obs}(z)|\ge2,
\quad
z\neq s_1,
\quad
z\text{ nonterminal},
\quad
z\text{ replayable}.
}
$$

## Local posterior

$$
p_{g,z,u}\sim\operatorname{Beta}(\alpha_{g,z,u},\beta_{g,z,u}).
$$

## Exact Batch-ERV

$$
\boxed{
BERV(\mathcal B)
=
E_{\mathbf Y}
\left[
\max_u m_u^{\mathbf Y}
\right]
-
\max_um_u.
}
$$

使用 Beta-Binomial predictive probability 精确有限求和。

## Capacity

$$
\delta^{(1)}(z)=V^{(1)}(z),
$$

$$
\delta^{(2)}(z)=V^{(2)}(z)-V^{(1)}(z),
$$

$$
C_g
=
\sum_z c_g(z),
$$

其中：

$$
 c_g(z)=0,1,2
$$

按 $\delta^{(1)}$ 与 $\delta^{(2)}$ 是否依次超过 $\tau_{BERV}$ 层级确定。

若：

$$
C_g<Q_g,
$$

则：

$$
Q_g\leftarrow Q_g-1,
\qquad
R_g\leftarrow R_g+1.
$$

## Global batch allocation

$$
\boxed{
\max_{\{m_z\}}
\sum_zV_z^{(m_z)}
\quad
\text{s.t.}
\quad
\sum_zm_z=Q_g,
\quad
m_z\in\{0,1,2\}.
}
$$

## Tie

对所有数值 tie-optimal plans：

$$
\boxed{
\text{seeded uniform tie-breaking}.
}
$$

不增加额外 acquisition heuristic。

## Branch execution

$$
\boxed{
\text{reset + mechanical prefix replay}
\rightarrow
\text{copy natural CoT-action edge}
\rightarrow
\text{fresh suffix rollout}.
}
$$

所有 branches 一次性并行。

## Training

$$
A_i
=
A_{leaf(i)}^E
+
\omega A_i^S,
$$

$$
r=\frac{\pi_\theta}{\pi_{old}},
$$

$$
\boxed{
\mathcal L
=
\mathcal L_{PPO}^{unified}
+
\beta_{KL}\mathcal L_{KL}.
}
$$

Mechanical replay prefix mask $=0$。

---

# 38. 最终结论

当前 BACE-GiGPO 已经从原来的多阶段 sequential controller 简化为一个更适合真实 LLM-RL 系统的两层主动实验框架：

$$
\boxed{
\text{历史 competence}
\rightarrow
\text{规划 breadth/refinement 预算}
}
$$

以及：

$$
\boxed{
\text{current natural-root evidence}
\rightarrow
\text{Exact Batch-ERV}
\rightarrow
\text{parallel local credit experiments}.
}
$$

取消 Pilot phase 去掉了第一处全局 rollout barrier；Batch-ERV 去掉了 branch-by-branch 的 sequential barrier；Exact Beta-Binomial computation 又去掉了 Monte Carlo noise 与 $M_{MC}$ 超参数。

最终方法的核心不再是“做一棵更复杂的树”，而是：

$$
\boxed{
\textbf{在固定 rollout budget 下，主动设计最有决策价值的局部信用实验。}
}
$$

其方法身份可以概括为：

> **Competence-guided topology planning + exact parallel Bayes-regret-reduction experiment design + observed-edge continuation resampling + unified GiGPO credit learning.**
