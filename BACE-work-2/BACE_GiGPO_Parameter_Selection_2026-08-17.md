# BACE-GiGPO 参数选择与推荐配置

> 版本：2026-08-17  
> 适用方案：**无独立 Pilot + natural-root history competence + root-side capacity correction + Exact Batch-ERV + batch-parallel branches + unified GiGPO/PPO**  
> 目的：用于第一版代码实现、参数冻结与后续 sensitivity/ablation 设计。

---

## 0. 参数选择原则

当前版本应尽量把参数分为三类：

1. **方法定义参数：固定，不应为了结果反复调节**；
2. **BACE 核心超参数：少量做 sensitivity**；
3. **GiGPO/PPO 基础训练参数：直接继承现有 verl-agent GiGPO 8 卡脚本，不为 BACE 单独重调**。

第一版最需要重点确认的参数只有：

$$
\boxed{
\tau_{\mathrm{BERV}},
\tau_{\mathrm{comp}},
L_{\max},
\lambda_{\mathrm{hist}},
\kappa_T^{\max}
}
$$

其中 $\tau_{\mathrm{BERV}}$ 建议先通过 controller profiling 冻结，其余参数可先按本文推荐值运行。

---

# 1. 主配置总表

| 模块 | 参数 | 简要含义 | 第一版推荐值 | 建议搜索/消融 | 状态 |
|---|---|---|---:|---|---|
| Rollout | $B$ | 每个 task 的 terminal-leaf 总预算 | **8** | $4,8,16$ 仅预算实验 | 固定主配置 |
| Rollout | $R_{\min}$ | 每个 task 至少保留的 natural roots | **2** | $2,3,4$ | sensitivity |
| Family prior | $A_0,B_0$ | task-family 冷启动 Beta base prior | **1,1** | 一般不调 | 固定 |
| Family history | $\lambda_{\mathrm{hist}}$ | 历史 natural-root 统计遗忘系数 | **0.8** | $0.6,0.8,0.9$ | 核心 sensitivity |
| Family transfer | $\tau_T$ | raw history concentration 的传递比例 | **0.1** | $0.05,0.1,0.2$ | 次级 sensitivity |
| Family transfer | $\kappa_T^{\min}$ | competence belief concentration 下限 | **2** | $1,2$ | 基本固定 |
| Family transfer | $\kappa_T^{\max}$ | competence belief concentration 上限 | **8** | $4,8$ | 核心 sensitivity |
| Topology | $\tau_{\mathrm{comp}}$ | competence 超过何值才偏向 refinement | **0.5** | $0.3,0.5,0.7$ | 核心 sensitivity |
| Local posterior | $\kappa_A$ | anchor-action Beta prior strength | **2** | $1,2,4$ | sensitivity |
| Batch-ERV | $L_{\max}$ | 单个 anchor 一批最多分配几条 branches | **2** | $1,2,3$ | 核心 sensitivity |
| Batch-ERV | $\tau_{\mathrm{BERV}}$ | marginal BERV 低于该值则不计为有效 capacity | **0.005 起步** | $0,.0025,.005,.01,.02$ | **暂不冻结** |
| Exact evaluator | method | BERV 计算方式 | **Exact Beta-Binomial** | MC 仅做数值核验 | 固定 |
| Tie | $\epsilon_{\mathrm{abs}}$ | BERV 绝对并列容差 | $10^{-12}$ | 不调 | 固定 |
| Tie | $\epsilon_{\mathrm{rel}}$ | BERV 相对并列容差 | $10^{-10}$ | 不调 | 固定 |
| Tie | rule | 数值并列时的选择方式 | **seeded uniform** | deterministic key 仅调试 | 固定 |
| Credit | $\omega$ | GiGPO local advantage 权重 | **1.0** | $0.5,0.8,1.0$ | 尽量继承 GiGPO |
| Credit | $\gamma$ | return-to-go discount | **继承 GiGPO；当前可用 0.95** | 不为 BACE 单独调 | baseline 参数 |
| Numerics | $\epsilon_{\mathrm{norm}}$ | advantage 标准化稳定项 | $10^{-6}$ | 不调 | 固定 |
| Training | behavior correction | 是否乘额外 $\pi_{\rm old}/\mu$ | **False** | corrected 版仅消融 | 固定主方案 |
| Training | separate branch loss | root/branch 是否分开 loss | **False** | 分开版本仅消融 | 固定主方案 |
| Credit | macro weighting | global trajectory return 的统计权重 | **GiGPO-compatible occurrence-weighted** | leaf-uniform、lineage-balanced | 主配置/消融 |
| Credit | local credit | 局部 GiGPO credit | **occurrence-level** | action-aggregated | 主配置/增强消融 |

---

# 2. Rollout 与 topology 参数

## 2.1 Terminal-leaf budget：$B=8$

定义：

$$
R_g+Q_g=B.
$$

推荐：

$$
\boxed{B=8}
$$

**含义：** 每个 task 最终保留 8 个 terminal leaves，不论它们来自 natural roots 还是 branches。

**理由：** 当前实验基于 GiGPO 的 group size 8，固定 $B=8$ 能让 BACE 与 GiGPO 在 leaf budget 上直接比较。预算 scaling 可单独测试 $B\in\{4,8,16\}$，但不应用更大的 $B$ 作为主方法去比较 GiGPO-8。

---

## 2.2 最少 natural roots：$R_{\min}=2$

推荐：

$$
\boxed{R_{\min}=2}
$$

初始计划 branch quota：

$$
\bar Q_g=\operatorname{round}[(B-R_{\min})q_{c,k}],
$$

初始 roots：

$$
R_g^{(0)}=B-\bar Q_g.
$$

**理由：** 至少保留两条 natural roots，避免完全依赖单条轨迹中的偶然 loop，同时仍允许高 competence task 最多获得较大的 refinement budget。$R_{\min}=4$ 会明显压缩 branch 空间，因此不建议作为主值。

---

# 3. Task-family competence 参数

当前版本**不再运行独立 Pilot phase**。当前 batch 的 root/branch 计划由滞后的 task-family natural-root history 直接决定；当前 batch 完成后，再用本批次全部 natural roots 更新下一批 history。Capacity correction 新增的 roots 也计入 history，branches 不计入。

## 3.1 冷启动 prior：$A_0=B_0=1$

推荐：

$$
\boxed{A_0=B_0=1}
$$

对应无历史时的均匀 Beta prior：

$$
\operatorname{Beta}(1,1).
$$

**理由：** 冷启动阶段不人为假设某个 ALFWorld task family 更容易或更困难，参数最少且容易复现。

---

## 3.2 历史遗忘：$\lambda_{\mathrm{hist}}=0.8$

自然 root 历史统计：

$$
\widetilde S_{c,k}=\lambda_{\mathrm{hist}}\widetilde S_{c,k-1}+S_{c,k}^{\mathrm{root}},
$$

$$
\widetilde F_{c,k}=\lambda_{\mathrm{hist}}\widetilde F_{c,k-1}+F_{c,k}^{\mathrm{root}}.
$$

推荐：

$$
\boxed{\lambda_{\mathrm{hist}}=0.8}
$$

建议测试：

$$
\{0.6,0.8,0.9\}.
$$

**理由：** actor 持续变化，旧策略下很久以前的成功率不能长期决定当前 topology。$0.8$ 使 controller 主要关注最近若干轮更新，在响应速度与稳定性之间折中。

---

## 3.3 History transfer：$\tau_T=0.1$

先计算 family mean：

$$
\mu_{c,k}=\frac{A_{c,k}}{A_{c,k}+B_{c,k}},
$$

raw concentration：

$$
K_{c,k}=A_{c,k}+B_{c,k}.
$$

再定义传递到当前 controller 的 concentration：

$$
\kappa_{c,k}^{T}
=
\operatorname{clip}
(\tau_TK_{c,k},\kappa_T^{\min},\kappa_T^{\max}).
$$

推荐：

$$
\boxed{\tau_T=0.1}
$$

**理由：** 历史 observation count 不能被直接当作当前 actor 的等量证据。只传递约 10% 的 raw concentration，可以避免旧策略历史过度支配当前 topology。

---

## 3.4 Concentration bounds：$\kappa_T^{\min}=2,\ \kappa_T^{\max}=8$

推荐：

$$
\boxed{\kappa_T^{\min}=2,\qquad \kappa_T^{\max}=8}
$$

**下限理由：** 冷启动时保持一个稳定但不强的 Beta belief。

**上限理由：** 无 Pilot 后，family competence 直接控制当前 topology；若 concentration 无限累积，$q$ 很容易饱和到 0 或 1，使 topology 长时间锁死。建议重点比较：

$$
\boxed{\kappa_T^{\max}\in\{4,8\}}.
$$

若观察到某些 family 长期固定在极端 root/branch 比例，优先降低 $\kappa_T^{\max}$。

---

## 3.5 Competence threshold：$\tau_{\mathrm{comp}}=0.5$

定义 controller belief：

$$
\phi_{c,k}
\sim
\operatorname{Beta}
(\kappa_{c,k}^{T}\mu_{c,k},\kappa_{c,k}^{T}(1-\mu_{c,k})).
$$

refinement readiness：

$$
q_{c,k}=P(\phi_{c,k}>\tau_{\mathrm{comp}}).
$$

推荐：

$$
\boxed{\tau_{\mathrm{comp}}=0.5}
$$

建议测试：

$$
\{0.3,0.5,0.7\}.
$$

**理由：** $0.5$ 是最中性的 competence 分界。这里使用的是 posterior tail probability，而不是简单判断均值是否超过 0.5，因此 posterior uncertainty 也会自然进入 topology decision。

---

# 4. Anchor-action Beta posterior

## 4.1 局部 prior strength：$\kappa_A=2$

对每个 observed anchor-action pair：

$$
p_{g,z,u}\sim
\operatorname{Beta}
(\kappa_A\bar\phi_g,\kappa_A(1-\bar\phi_g)).
$$

在无 Pilot 主版本中，$\bar\phi_g$ 可用当前冻结的 family competence mean 作为弱中心；自然 root evidence 随后直接更新该 posterior。

推荐：

$$
\boxed{\kappa_A=2}
$$

建议测试：

$$
\{1,2,4\}.
$$

**理由：** instance/family competence 只应该给局部动作一个很弱的 shrinkage，避免单个成功/失败 observation 直接把 action credit 推到极端，同时不能压过真正的 $(z,u)$ evidence。

---

# 5. Exact Batch-ERV 参数

## 5.1 每 anchor 最大 branch 数：$L_{\max}=2$

推荐：

$$
\boxed{L_{\max}=2}
$$

对每个 anchor $z$，计算：

$$
V_z^{(0)}=0,\qquad V_z^{(1)},\qquad V_z^{(2)},
$$

并定义 marginal Batch-ERV：

$$
\delta_z^{(1)}=V_z^{(1)},
$$

$$
\delta_z^{(2)}=V_z^{(2)}-V_z^{(1)}.
$$

**理由：** $L_{\max}=2$ 允许对真正关键的 anchor 做第二次局部辨识，同时限制单一 anchor 垄断预算；计算和解释也保持简单。建议消融：

$$
L_{\max}\in\{1,2,3\}.
$$

---

## 5.2 Marginal BERV threshold：$\tau_{\mathrm{BERV}}$

定义 anchor 的有效 branch capacity：

$$
c_g(z)
=
\sum_{r=1}^{L_{\max}}
\mathbf 1[\delta_z^{(r)}\ge\tau_{\mathrm{BERV}}].
$$

总 capacity：

$$
C_g=\sum_z c_g(z).
$$

若：

$$
C_g<Q_g,
$$

则把一个 branch slot 转成新的 natural root：

$$
R_g\leftarrow R_g+1,\qquad Q_g\leftarrow Q_g-1,
$$

重新构造 anchor/posterior/Batch-ERV，直到 capacity 足够或 $Q_g=0$。

### 第一版推荐

$$
\boxed{\tau_{\mathrm{BERV}}=0.005\ \text{作为起步值}}
$$

### 推荐 sweep

$$
\boxed{
\tau_{\mathrm{BERV}}
\in
\{0,0.0025,0.005,0.01,0.02\}
}
$$

**理由：** 当前 Exact Batch-ERV 已无 Monte Carlo 抖动，因此旧 one-sample ERV 阈值不应机械继承。阈值只负责过滤“理论上为正但实际价值极小”的 branch slot。$0.005$ 可理解为要求一条额外 branch 具有至少约 0.5 个百分点量级的 expected decision-value improvement。

### 选择时主要看

不要优先按最终 test reward 反向选择阈值，而应先看 controller 行为：

- branch quota retention ratio；
- capacity correction 次数；
- 最终 $R/Q$ topology 分布；
- selected marginal BERV 分布；
- 第二个 branch slot $\delta^{(2)}$ 的利用率；
- average branches per task。

如果几乎所有 task 都执行满 branch quota，阈值可能过低；如果绝大多数 task 最终退化为 roots-only，阈值可能过高。

---

## 5.3 Exact evaluator

主版本：

```text
berv_method = exact_beta_binomial
```

**理由：** 当前 branch outcome 为 Bernoulli，Beta posterior 下给定分配次数后的成功数服从 Beta-Binomial predictive distribution，可以有限枚举精确计算 batch experiment 的 expected value，不需要 Monte Carlo 参数 $M_{\mathrm{MC}}$。

Monte Carlo 只可作为开发阶段数值核验，不进入正式方法。

---

# 6. Tie handling

## 6.1 数值 tie tolerance

推荐：

$$
\boxed{\epsilon_{\mathrm{abs}}=10^{-12}}
$$

$$
\boxed{\epsilon_{\mathrm{rel}}=10^{-10}}
$$

若两个 value 满足：

$$
|V_1-V_2|
\le
\epsilon_{\mathrm{abs}}
+
\epsilon_{\mathrm{rel}}\max(|V_1|,|V_2|),
$$

则视为并列。

**理由：** Exact BERV 没有 MC sampling noise，因此 tolerance 应只处理 floating-point 误差，不应设得过大。

## 6.2 Tie-breaking

主版本：

```text
tie_break = seeded_uniform
```

**理由：** 若两个 branch plans 在 BERV objective 下等价，再用 depth、frequency 或 horizon 打破并列，相当于偷偷加入第二个 acquisition objective。seeded uniform 更干净且可复现。

---

# 7. Anchor 结构参数：建议直接固定

| 规则 | 主配置 | 理由 |
|---|---|---|
| `min_anchor_occurrences` | **2** | repeated anchor 的最低定义 |
| `min_observed_actions` | **2** | 必须存在真实动作竞争 |
| `anchor_match` | **exact GiGPO pre-action observation** | 与 GiGPO 主实现对齐 |
| 同轨迹循环 occurrence | **保留** | 与 GiGPO grouping 对齐 |
| 初始状态 $s_1$ | **不作为 branch origin** | 否则接近定向 root |
| terminal state | **排除** | 无有效 continuation |
| candidate actions | **observed valid canonical actions only** | 当前方法是对自然暴露 action edges 做主动复采样 |
| candidate cap | **None** | $B=8,L_{\max}=2$ 下通常无需额外截断 |
| invalid/unparseable action | **不进入 targeted branch candidate set** | 无可靠可执行 edge |
| branch depth | **1** | 当前不做 branch-of-branch |
| concrete origin | **在执行过该 action 的 natural occurrences 中均匀采样** | 避免固定依赖某条 arrival history |
| remaining horizon | **继承 origin 剩余 horizon** | 不增加额外 depth/horizon 超参数 |

### 关于旧版 $K_{\max}=4$

早期 anchor 规范曾建议 $K_{\max}=4$。当前主版本候选动作已经收缩为 **natural roots 实际执行过的 valid actions**，且每个 task 只有 $B=8$ 的 evidence，因此第一版可不再设置 action cap。若后续某 benchmark 的 observed-action classes 明显变多，可重新启用：

$$
K_{\max}=4
$$

作为工程保护，但不建议在 ALFWorld 主实验中增加不必要的截断规则。

---

# 8. Advantage / optimization 参数

## 8.1 Local advantage weight：$\omega=1.0$

最终 GiGPO-style advantage：

$$
A_j=A_{\ell(j)}^E+\omega A_j^S.
$$

推荐：

$$
\boxed{\omega=1.0}
$$

**理由：** 主实验应尽量保持 GiGPO optimizer 不变，使提升主要归因于 BACE acquisition。若需要 sensitivity，可测试 $\{0.5,0.8,1.0\}$。

---

## 8.2 Global macro weighting

主版本：

```text
advantage_semantics = gigpo_macro
macro_weighting = occurrence
```

**理由：** 主实验保持当前 GiGPO implementation 的 occurrence-weighted macro
normalization，使 BACE 与 baseline 的主要控制变量是 rollout evidence 的主动采集。
`leaf-uniform` 作为 tree-aware optimization 消融保留；`lineage-balanced` 继续用于
检查某条 root lineage 因 branch 较多而获得过大统计权重的问题。

---

## 8.3 Local credit mode

第一版建议：

```text
local_credit_mode = occurrence
```

**理由：** 保留原始 GiGPO 的 occurrence-level local credit，可以更清楚地验证改进是否来自 BACE 的 active acquisition。

增强/消融版本：

```text
local_credit_mode = action_aggregated
```

该版本把同一 anchor、同一 canonical action 的 local advantage 先平均再广播，更契合动作级 posterior，但与已有 action-aggregation 方法更接近，因此不建议在第一版主实验中同时引入过多变化。

---

## 8.4 Replay prefix 与 branch training mask

主版本固定：

```text
train_natural_root = true
train_copied_branch_origin = true
train_branch_suffix = true
train_mechanical_replay_prefix = false
```

**理由：** replay prefix 只是环境恢复，不是新的 actor decision，不能因 descendant branches 数量而被重复训练。Copied branch origin 与新 suffix 则属于当前 branch leaf 的训练 occurrence。

---

# 9. GiGPO/PPO 基础参数：原则上不重新调

以下参数应以当前使用的 **verl-agent GiGPO ALFWorld 8 卡脚本**为 source of truth：

- actor learning rate；
- PPO clip range；
- KL coefficient；
- PPO minibatch / microbatch；
- max prompt / response length；
- rollout temperature；
- max environment steps；
- optimizer；
- rollout engine / tensor parallel 设置。

第一版 BACE 不应为了取得更好结果单独改变这些值。

若当前复现配置确实使用：

$$
\text{actor LR}=10^{-6},\qquad
\beta_{\mathrm{KL}}=0.01,\qquad
\gamma=0.95,
$$

则直接保持；最终以你们实际 GiGPO 8 卡 baseline script 的值为准。

---

# 10. 当前主方案中应删除的旧参数

以下参数属于之前的 Pilot / sequential ERV / ERV-softmax 版本，**不要继续出现在当前主配置中**：

| 旧参数 | 当前处理 | 原因 |
|---|---|---|
| $N_{\mathrm{pilot}}$ | **删除** | 已取消独立 Pilot phase |
| $M_{\mathrm{MC}}$ | **删除** | Exact Beta-Binomial BERV 取代 MC |
| $\tau_\mu$ | **删除** | 不再用 ERV-softmax 顺序采 branch action |
| ERV-softmax behavior distribution | **删除** | 当前由 joint Batch-ERV 直接规划整批 branch experiments |
| behavior weight $\pi_{\rm old}/\mu_{\rm br}$ | **删除** | 主方案不做额外 behavior correction |
| $w_{\min},w_{\max}$ | **删除** | 无 behavior weight clipping |
| $\lambda_{\mathrm{branch}}$ | **删除** | root / branch 进入 unified PPO objective |
| $K_{\max}$ | **ALFWorld 主版删除** | observed valid action set 已很小；必要时仅作工程保护 |
| $H_{\min}$ | **删除** | branch 继承 concrete origin remaining horizon |
| sequential posterior update rounds | **删除** | branch plan 在 rollout 前一次性确定，之后并行生成 |

---

# 11. 第一版代码建议直接冻结的配置

```yaml
bace:
  # rollout topology
  leaf_budget: 8
  min_natural_roots: 2
  competence_threshold: 0.5

  # family competence
  family_prior_alpha0: 1.0
  family_prior_beta0: 1.0
  history_decay: 0.8
  transfer_ratio: 0.1
  transfer_strength_min: 2.0
  transfer_strength_max: 8.0
  family_history_weighting: all_natural_roots
  include_capacity_roots_in_history: true
  include_branches_in_history: false

  # local beta posterior
  anchor_action_prior_strength: 2.0
  reset_local_posterior_after_actor_update: true

  # exact Batch-ERV
  berv_method: exact_beta_binomial
  berv_threshold: 0.005
  max_branches_per_anchor: 2

  # ties
  tie_abs_tol: 1.0e-12
  tie_rel_tol: 1.0e-10
  tie_break: seeded_uniform

  # anchors
  anchor_match: exact_gigpo
  min_anchor_occurrences: 2
  min_observed_actions: 2
  exclude_initial_state: true
  require_nonterminal: true
  candidate_action_source: observed_valid
  candidate_action_cap: null
  origin_sampling: uniform
  max_branch_depth: 1
  inherit_remaining_horizon: true

  # training data
  train_branch_origin: true
  train_branch_suffix: true
  train_replay_prefix: false

  # credit
  global_credit: all_leaf
  leaf_weighting: uniform
  local_credit_mode: occurrence
  local_advantage_weight: 1.0
  norm_eps: 1.0e-6

  # optimization semantics
  behavior_correction: false
  separate_branch_loss: false
```

PPO/LR/KL/minibatch/microbatch 等继续放在原 GiGPO 配置中，不复制到 BACE namespace。

---

# 12. 推荐的参数确定顺序

如果现在开始跑小规模实验，建议不要同时 grid-search 所有参数，而按以下顺序确定：

### Step 1：先固定方法结构

直接使用：

$$
B=8,\quad
R_{\min}=2,\quad
L_{\max}=2,\quad
\kappa_A=2,
$$

$$
A_0=B_0=1,\quad
\tau_T=0.1,\quad
\kappa_T^{\min}=2,\quad
\kappa_T^{\max}=8,
$$

$$
\tau_{\mathrm{comp}}=0.5,\quad
\lambda_{\mathrm{hist}}=0.8.
$$

### Step 2：只扫 $\tau_{\mathrm{BERV}}$

建议：

$$
\{0,0.0025,0.005,0.01,0.02\}.
$$

根据 controller 行为而非最终 test reward 先确定合理量级。

### Step 3：检查 topology sensitivity

依次检查：

$$
\tau_{\mathrm{comp}}\in\{0.3,0.5,0.7\},
$$

$$
L_{\max}\in\{1,2,3\},
$$

$$
\lambda_{\mathrm{hist}}\in\{0.6,0.8,0.9\}.
$$

### Step 4：检查 prior 是否过度确定

比较：

$$
\kappa_T^{\max}\in\{4,8\}.
$$

如果 topology 容易锁死，用 4；若历史 competence 太抖，用 8。

### Step 5：最后再看局部 posterior 和 optimizer sensitivity

$$
\kappa_A\in\{1,2,4\},
$$

必要时再测试：

$$
\omega\in\{0.5,0.8,1.0\}.
$$

不要在 controller 尚未稳定前优先调 PPO/LR/KL。

---

# 13. 建议重点记录的诊断量

为了真正判断参数是否合理，训练日志至少记录：

### Family/topology

- $\mu_{c,k}$；
- $\kappa_{c,k}^{T}$；
- $q_{c,k}$；
- planned branch quota $\bar Q_g$；
- capacity correction 后的 $Q_g$；
- 最终 $R_g/Q_g$ 分布；
- branch quota retention ratio $Q_g/\bar Q_g$。

### Anchor/Batch-ERV

- repeated / structural anchors 数量；
- 每个 anchor observed action 数；
- $V_z^{(1)},V_z^{(2)}$；
- $\delta_z^{(1)},\delta_z^{(2)}$；
- selected marginal BERV；
- 每个 anchor 的 branch 数；
- 第二个 branch slot 的使用比例；
- tie frequency。

### Replay/system

- replay success rate；
- replay env steps；
- branch remaining horizon；
- root/branch generated tokens；
- wall-clock；
- GPU utilization。

这些诊断量比单看最终 reward 更能判断参数是否真的实现了预期的 breadth–refinement 调度。

---

# 14. 当前推荐结论

第一版主配置建议直接采用：

$$
\boxed{
B=8,
R_{\min}=2,
L_{\max}=2,
\tau_{\mathrm{comp}}=0.5
}
$$

$$
\boxed{
A_0=B_0=1,
\lambda_{\mathrm{hist}}=0.8,
\tau_T=0.1,
\kappa_T^{\min}=2,
\kappa_T^{\max}=8
}
$$

$$
\boxed{
\kappa_A=2,
\tau_{\mathrm{BERV}}=0.005\ \text{（起步值）}
}
$$

$$
\boxed{
\omega=1,
\epsilon_{\mathrm{norm}}=10^{-6}
}
$$

其中真正需要先通过小规模 profiling 再最终冻结的是：

$$
\boxed{\tau_{\mathrm{BERV}}}.
$$

其余核心 sensitivity 建议按：

$$
\tau_{\mathrm{comp}}
\rightarrow
L_{\max}
\rightarrow
\lambda_{\mathrm{hist}}
\rightarrow
\kappa_T^{\max}
\rightarrow
\kappa_A
$$

的顺序检查。

这样可以把 BACE 真正需要调的自由度控制在很少的几个参数内，同时让 GiGPO/PPO 基础优化设置保持不变，便于公平比较和论文解释。
