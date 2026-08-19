# BACE-GiGPO 实验参数选择与推荐配置

> 版本：2026-08-11  
> 用途：ALFWorld 主实验实现、配置文件编写、超参数消融与实验记录  
> 状态：按当前已冻结方案整理

---

# 1. 当前冻结结论

本版本严格采用以下已确认设置：

- GiGPO 局部优势权重：
  $$
  \boxed{\omega=\texttt{step\_advantage\_w}=1.0}
  $$
- ERV 采用 Monte Carlo 方法估计，不使用闭式 one-step ERV 实现；
- effective-anchor 阈值：
  $$
  \boxed{\tau_{\mathrm{ERV}}=0.01}
  $$
- task-family 历史采用指数遗忘：
  $$
  \boxed{\lambda_{\mathrm{hist}}=0.8}
  $$
- 不采用“仅保留最近两个 policy-update iterations”的截断历史方案；
- 其余参数沿用此前推荐的主配置。

本文件区分三类参数：

1. **主实验固定参数**：第一轮正式实验直接固定，不建议搜索；
2. **BACE 核心超参数**：建议做小范围 sensitivity；
3. **继承 GiGPO 参数**：与 GiGPO baseline 保持一致，避免将 baseline tuning 与方法收益混在一起。

---

# 2. 总体推荐配置表

| 类别 | 参数 | 推荐主值 | 建议搜索/消融 | 作用 |
|---|---|---:|---|---|
| Rollout budget | $B$ | **8** | $\{4,8,16\}$ 仅预算实验 | 每个任务实例最终 terminal-leaf budget |
| Pilot | $N_{\mathrm{pilot}}$ | **2** | $\{2,3,4\}$ | topology 决策前自然 root 数 |
| Anchor capacity | $L_{\max}$ | **2** | $\{1,2,3,\infty\}$ | 单个 anchor 当前 batch 最多 branch 数 |
| Competence | $\tau_{\mathrm{comp}}$ | **0.5** | $\{0.3,0.5,0.7\}$ | 判断当前实例是否进入 refinement regime |
| History prior | $A_0,B_0$ | **1,1** | 固定 | task-family Beta 初始平滑 |
| History decay | $\lambda_{\mathrm{hist}}$ | **0.8** | $\{0.6,0.8,0.9\}$ | 跨 epoch/task-update 历史指数遗忘 |
| Prior transfer | $\tau_T$ | **0.1** | $\{0.05,0.1,0.2\}$ | 将历史 concentration 转为 instance prior strength |
| Prior strength floor | $\kappa_T^{\min}$ | **2** | $\{1,2\}$ | task-to-instance prior strength 下限 |
| Prior strength cap | $\kappa_T^{\max}$ | **8** | $\{4,8\}$ | 防止历史 prior 过强 |
| Local Beta prior | $\kappa_A$ | **2** | $\{1,2,4\}$ | anchor-action posterior 初始 pseudo-count |
| ERV MC | $M_{\mathrm{MC}}$ | **512** | $\{256,512,1024\}$ | Bayes regret / ERV Monte Carlo 样本数 |
| ERV filter | $\tau_{\mathrm{ERV}}$ | **0.01** | $\{0,0.005,0.01,0.02\}$ | 过滤低信息价值 anchor |
| ERV action temp. | $\tau_\mu$ | **0.02** | $\{0.01,0.02,0.05\}$ | ERV-softmax 动作分布温度 |
| GiGPO local weight | $\omega$ | **1.0** | $\{0.5,0.8,1.0\}$ | local state advantage 权重 |
| Discount | $\gamma$ | **0.95** | 原则上固定 | state return-to-go discount |
| Norm epsilon | $\epsilon_{\mathrm{norm}}$ | **$10^{-6}$** | 固定 | advantage normalization 数值稳定 |
| KL | $\beta_{\mathrm{KL}}$ | **0.01** | $\{0.005,0.01,0.02\}$ 仅稳定性分析 | reference KL regularization |
| Actor LR | learning rate | **$10^{-6}$** | 与 GiGPO 一致 | actor optimization |
| Loss aggregation | `loss_agg_mode` | **`token-mean`** | `seq-mean-token-mean` 消融 | 统一 PPO loss 聚合 |
| Branch depth | branch depth | **1** | recursive 仅消融 | 当前 batch 只做一层 branch |
| Candidate actions | action set | **全部 observed valid actions** | unseen-actions 仅消融 | ERV 候选动作集合 |
| Origin sampling | origin rule | **uniform** | 可做 count-weighted 消融 | 从真实执行过选中动作的 occurrences 中选择 origin |

---

# 3. Rollout 与拓扑参数

## 3.1 总 leaf budget $B$

推荐：

$$
\boxed{B=8}
$$

含义：每个具体 task instance 在当前 rollout group 最终保留的 terminal leaves 总数：

$$
R_g+Q_g=B.
$$

其中：

- $R_g$：natural roots；
- $Q_g$：branches。

### 选择理由

1. 与当前 GiGPO 的 group size 8 对齐，最有利于公平比较；
2. BACE 只是改变这 8 个 rollout slots 的 topology，而不首先扩大样本预算；
3. $B=8$ 足以形成多个有意义 topology，例如：
   $$
   8+0,\quad 7+1,\quad 6+2,\quad 5+3,\quad 4+4,\dots
   $$
4. 如果第一版直接提高到 16，很难判断收益来自 BACE 还是额外 rollout budget。

### 实验建议

主结果固定 $B=8$。额外做 compute-scaling 时再测试：

$$
B\in\{4,8,16\}.
$$

---

## 3.2 Pilot 数 $N_{\mathrm{pilot}}$

推荐：

$$
\boxed{N_{\mathrm{pilot}}=2}
$$

当前实例 pilot success 数：

$$
S_g^{\mathrm{pilot}}\in\{0,1,2\}.
$$

### 选择理由

- 两条 pilot 是能同时获得实例信息、又不大量占用 $B=8$ 的最小可用设置；
- 当前 task-family prior 会对两条 pilot 进行 shrinkage，因此不会直接使用极端的 $\{0,0.5,1\}$ empirical rate；
- $N_{\mathrm{pilot}}=4$ 会直接消耗一半 leaf budget，降低动态 topology 的可调空间。

建议只做：

$$
N_{\mathrm{pilot}}\in\{2,3,4\}
$$

的独立敏感性实验，不与其他参数做笛卡尔积。

---

## 3.3 单 Anchor 最大 branch capacity $L_{\max}$

推荐：

$$
\boxed{L_{\max}=2}
$$

capacity condition：

$$
Q_g\le L_{\max}|\mathcal Z_g^{\mathrm{eff}}|.
$$

### 选择理由

- $L_{\max}=1$ 强制一 anchor 一 branch，可能过早停止对真正关键 anchor 的重复验证；
- $L_{\max}=\infty$ 容易让一个偶然形成的 anchor 消耗整个 branch budget；
- $L_{\max}=2$ 允许第一次 branch 后根据新 outcome 更新 posterior，再做一次额外 verification，同时限制过度集中。

建议消融：

$$
L_{\max}\in\{1,2,3,\infty\}.
$$

---

# 4. Task-Family Competence Prior 参数

## 4.1 基础 Beta prior：$A_0=B_0=1$

推荐：

$$
\boxed{A_0=B_0=1}
$$

历史统计：

$$
A_{c,e}=A_0+\widetilde S_{c,e},
$$

$$
B_{c,e}=B_0+\widetilde F_{c,e}.
$$

### 选择理由

Beta$(1,1)$ 是最简单的均匀初始 prior：

$$
E[\theta]=0.5.
$$

在没有可靠历史时，它不会预先假定某类 ALFWorld task 容易或困难。

---

## 4.2 历史指数遗忘系数 $\lambda_{\mathrm{hist}}$

当前冻结：

$$
\boxed{\lambda_{\mathrm{hist}}=0.8}
$$

更新：

$$
\widetilde S_{c,e}
=
0.8\widetilde S_{c,e-1}
+
S_{c,e}^{\mathrm{root}},
$$

$$
\widetilde F_{c,e}
=
0.8\widetilde F_{c,e-1}
+
F_{c,e}^{\mathrm{root}}.
$$

### 统计含义

第 $d$ 个历史 epoch/update 的相对权重为：

$$
0.8^d.
$$

例如：

| 历史距离 | 相对权重 |
|---:|---:|
| 上一轮 | $1$ |
| 2 轮前 | $0.8$ |
| 3 轮前 | $0.64$ |
| 4 轮前 | $0.512$ |
| 5 轮前 | $0.4096$ |
| 10 轮前 | $0.134$ |

其半衰期约为：

$$
h_{1/2}
=
\frac{\log 0.5}{\log 0.8}
\approx3.11.
$$

常用的 effective-memory 近似为：

$$
W_{\mathrm{eff}}
\approx
\frac{1}{1-0.8}
=5.
$$

因此 $\lambda_{\mathrm{hist}}=0.8$ 的语义是：

> 历史不会被硬截断，但 prior 的主要质量集中在最近约 3--5 个 policy-update epochs；更早数据仍有影响，但指数衰减。

### 为什么选择 0.8

我们希望同时满足两个目标：

1. task-family history 能稳定只有两条 pilots 的实例 posterior；
2. actor 更新后 competence 会变化，因此旧 policy 的成功率不能长期支配当前 topology。

相比：

- $\lambda=0.9$：历史过长，约相当于 10 轮 effective window；
- $\lambda=0.95$：历史惯性更强；
- $\lambda=0.5$：几乎只保留最近少量信息；

$0.8$ 是较均衡的折中。

### 建议消融

$$
\lambda_{\mathrm{hist}}
\in
\{0.6,0.8,0.9\}.
$$

但主实验固定为 0.8。

---

## 4.3 Task-to-instance transfer 比例 $\tau_T$

推荐：

$$
\boxed{\tau_T=0.1}
$$

定义历史 concentration：

$$
K_{c,e}=A_{c,e}+B_{c,e}.
$$

instance prior strength：

$$
\kappa_{c,e}^{T}
=
\operatorname{clip}
\left(
0.1K_{c,e},
2,
8
\right).
$$

### 选择理由

历史的全部 pseudo-count 不应原封不动传递给一个具体实例，否则当前的两条 pilots 很难改变 posterior。

$\tau_T=0.1$ 表示只将历史 concentration 的约 10% 转化为实例 prior strength，再通过 $[2,8]$ clipping 控制极值。

---

## 4.4 Prior strength 下限 $\kappa_T^{\min}$

推荐：

$$
\boxed{\kappa_T^{\min}=2}
$$

### 理由

- 当前实例本身只有 2 条 pilots；
- prior 最低强度设为 2，相当于提供约两条 pseudo-observations；
- 能稳定 $0/2$、$1/2$、$2/2$ 的离散 pilot observation，又不至于完全压过当前实例。

---

## 4.5 Prior strength 上限 $\kappa_T^{\max}$

推荐：

$$
\boxed{\kappa_T^{\max}=8}
$$

### 理由

task-family history 再丰富，也不能无限提高对单个实例的控制力。8 作为上限允许历史产生明显影响，但仍保留 current-pilot correction 的可能性。

建议消融：

$$
\kappa_T^{\max}\in\{4,8\}.
$$

---

# 5. Instance Competence 参数

## 5.1 Competence threshold $\tau_{\mathrm{comp}}$

推荐：

$$
\boxed{\tau_{\mathrm{comp}}=0.5}
$$

计算：

$$
q_g
=
P
\left(
\phi_g>0.5
\mid
\mathcal D_g^{\mathrm{pilot}}
\right).
$$

计划 branch quota：

$$
\bar Q_g
=
\operatorname{round}
\left[
(B-N_{\mathrm{pilot}})q_g
\right].
$$

在 $B=8,N_{\mathrm{pilot}}=2$ 时：

$$
\bar Q_g=\operatorname{round}(6q_g).
$$

### 选择理由

$0.5$ 表示：

> 只有当 posterior 认为当前实例 natural-root success probability 更可能高于 50% 时，才系统性向 refinement 倾斜。

这是最中性的 competence boundary，解释最简单。

建议 sensitivity：

$$
\tau_{\mathrm{comp}}\in\{0.3,0.5,0.7\}.
$$

---

# 6. Anchor--Action Beta Posterior 参数

## 6.1 局部 prior strength $\kappa_A$

推荐：

$$
\boxed{\kappa_A=2}
$$

局部 prior：

$$
p_{g,z,u}
\sim
\operatorname{Beta}
\left(
2\bar\phi_g,
2(1-\bar\phi_g)
\right).
$$

### 选择理由

- instance competence 只适合作为局部 continuation success 的弱 center；
- $\kappa_A=2$ 防止仅 1 个 action occurrence 时 posterior 直接变成近似 0/1；
- 同时自然 evidence 与 branch outcomes 可以很快改变该 posterior。

建议消融：

$$
\kappa_A\in\{1,2,4\}.
$$

---

# 7. ERV 参数

## 7.1 ERV 仍采用 Monte Carlo 估计

当前实验版本明确保留 Monte Carlo 计算。

对每个 anchor-action posterior：

$$
p_{g,z,u}
\sim
\operatorname{Beta}(\alpha_{g,z,u},\beta_{g,z,u}).
$$

Monte Carlo 估计：

$$
\widehat{E[\max p]}
=
\frac{1}{M_{\mathrm{MC}}}
\sum_{m=1}^{M_{\mathrm{MC}}}
\max_u p_{g,z,u}^{(m)}.
$$

Bayes regret：

$$
\widehat{\mathcal R}_g(z)
=
\widehat{E[\max p]}
-
\max_u m_{g,z,u}.
$$

对下一条 $(z,u)$ branch 的成功与失败 posterior 分别估计后，得到：

$$
\widehat{\Delta\operatorname{ERV}}_g(z,u)
=
\widehat{\mathcal R}_g(z)
-
\left[
 m_{g,z,u}\widehat{\mathcal R}_g^+(z;u,1)
 +(1-m_{g,z,u})\widehat{\mathcal R}_g^+(z;u,0)
\right].
$$

---

## 7.2 Monte Carlo 样本数 $M_{\mathrm{MC}}$

推荐：

$$
\boxed{M_{\mathrm{MC}}=512}
$$

### 选择理由

候选动作数通常很小，而 Beta sampling 是纯 CPU/NumPy/PyTorch 向量运算，512 个 samples 与一次 LLM rollout 相比成本很低。

- 256：速度更快，但小 ERV 差异排序更容易抖动；
- 512：主推荐；
- 1024：更稳定，但通常收益有限。

建议测试：

$$
M_{\mathrm{MC}}\in\{256,512,1024\}.
$$

主实验使用固定 512，并固定 controller RNG seed，以保证可复现性。

### 实现建议

对同一个 $(z,u)$，计算 current/success/failure regret 时应尽量使用 **common random numbers**：

- 对未更新动作复用同一批 Beta base random samples；
- 只改变被更新 action 的 Beta transformation。

这样能明显降低：

$$
\widehat{\mathcal R}(z)
-
\widehat{\mathcal R}^+(z;u,Y)
$$

的 Monte Carlo difference noise。

---

## 7.3 ERV effective-anchor 阈值 $\tau_{\mathrm{ERV}}$

当前冻结：

$$
\boxed{\tau_{\mathrm{ERV}}=0.01}
$$

anchor action distribution：

$$
\mu_{g,b}(u\mid z)
=
\operatorname{Softmax}
\left(
\frac{\widehat{\Delta\operatorname{ERV}}_{g,b}(z,u)}{\tau_\mu}
\right).
$$

anchor utility：

$$
U_{g,b}(z)
=
\sum_u
\mu_{g,b}(u\mid z)
\widehat{\Delta\operatorname{ERV}}_{g,b}(z,u).
$$

capacity correction 前定义：

$$
\boxed{
\mathcal Z_g^{\mathrm{eff}}
=
\left\{
z\in\mathcal Z_g^{\mathrm{struct}}:
U_{g,0}(z)\ge0.01
\right\}.
}
$$

### 为什么选 0.01

阈值必须明显高于“只要 ERV 为正就保留”的极宽松规则。

$0.01$ 的目标是过滤：

- posterior 有轻微不确定性，但单条 branch 几乎不能改变局部决策的 anchor；
- ERV 仅由小样本/MC fluctuation 产生的极小 utility；
- 会占用 capacity、却很难形成有效 refinement 的状态。

它仍然不会过度严格到只保留极少数 anchors。

### 建议消融

$$
\boxed{
\tau_{\mathrm{ERV}}
\in
\{0,0.005,0.01,0.02\}
}
$$

解释：

- $0$：只排除严格零信息 anchor；
- $0.005$：宽松 filtering；
- $0.01$：主版本；
- $0.02$：强 filtering。

### 实验期必须记录

每轮至少记录：

```text
U(z) distribution
positive-ERV anchor fraction
effective-anchor fraction
number of structural anchors
number of effective anchors
capacity = L_max * |Z_eff|
root-to-branch correction count
selected-anchor U(z)
```

如果实际训练长期出现：

- 几乎无 effective anchor：说明 0.01 可能过高；
- 几乎所有 structural anchors 都通过：说明 0.01 filtering 不充分。

但主实验开始前应先固定 0.01，避免根据最终 test performance 反向调阈值。

---

## 7.4 ERV-softmax 温度 $\tau_\mu$

推荐：

$$
\boxed{\tau_\mu=0.02}
$$

动作分布：

$$
\mu(u\mid z)
=
\frac{
\exp(\Delta ERV(z,u)/0.02)
}{
\sum_v\exp(\Delta ERV(z,v)/0.02)
}.
$$

### 选择理由

我们希望：

- 明显偏向更高 ERV 动作；
- 又不完全退化为 deterministic argmax；
- 保留 posterior 估计误差下的 exploration robustness。

建议：

$$
\tau_\mu\in\{0.01,0.02,0.05\}.
$$

其中：

- 0.01：更接近 greedy ERV；
- 0.02：主推荐；
- 0.05：更加平滑。

---

# 8. Anchor Structural Parameters

这些参数属于方法定义，原则上不应在主实验中调。

## 8.1 Minimum repeated occurrences

$$
\boxed{|\mathcal I_g^{\mathrm{root}}(z)|\ge2}
$$

至少重复出现两次才进入 branch anchor 检查。

---

## 8.2 Minimum observed actions

$$
\boxed{|\mathcal C_g(z)|\ge2}
$$

只有存在两个不同、有效、自然执行过的动作时，才存在局部动作竞争。

---

## 8.3 Candidate set

推荐：

$$
\boxed{\mathcal C_g(z)=\mathcal C_{g,z}^{\mathrm{obs}}}
$$

保留全部自然 observed valid actions。

当前版本：

```text
candidate_cap = None
```

不设置 $K_{\max}$。

---

## 8.4 Origin sampling

对选中的 $(z,u)$：

$$
\mathcal I_g^{\mathrm{obs}}(z,u)
=
\{i:u_i=u\}.
$$

推荐：

$$
\boxed{
i^*\sim
\operatorname{Uniform}
(\mathcal I_g^{\mathrm{obs}}(z,u))
}
$$

不额外偏向某一 arrival history。

---

## 8.5 Branch depth

$$
\boxed{D_{\mathrm{branch}}=1}
$$

branch suffix 中的新 anchor 可以用于最终 GiGPO credit，但当前 rollout batch 不再继续 branch-of-branch。

---

# 9. Advantage 参数

## 9.1 GiGPO `step_advantage_w`

当前确认：

$$
\boxed{\omega=1.0}
$$

最终 advantage：

$$
A_j
=
A_{g,\ell(j)}^E
+
A_j^S.
$$

### 选择理由

我们的第一版目标是尽可能保持 GiGPO optimizer 本身不变，将性能差异归因于：

- competence-adaptive topology；
- ERV-guided acquisition；
- branch evidence reuse。

因此直接沿用 GiGPO ALFWorld 配置中的：

```text
algorithm.gigpo.step_advantage_w=1.0
```

是最干净的选择。

建议后续 sensitivity：

$$
\omega\in\{0.5,0.8,1.0\}.
$$

但主实验保持 1.0。

---

## 9.2 Discount factor $\gamma$

推荐：

$$
\boxed{\gamma=0.95}
$$

保持 GiGPO ALFWorld baseline。

注意：

- Beta posterior 使用 terminal binary outcome；
- GiGPO local advantage 的 return-to-go 仍按 $\gamma=0.95$ 计算。

两者承担不同统计角色，不需要强行设成一致。

---

## 9.3 Advantage normalization epsilon

推荐：

$$
\boxed{\epsilon_{\mathrm{norm}}=10^{-6}}
$$

纯数值稳定参数，不做调优。

---

## 9.4 Local credit 主版本

第一阶段主实验建议：

```text
local_credit_mode = occurrence
```

即保持原 GiGPO occurrence-level local advantage。

可选增强：

```text
local_credit_mode = action_aggregated
```

动作聚合只改变局部 $A^S$，不改变每条 leaf 的 $A^E$。

这样可以先证明 BACE acquisition 本身有效，再分析 action aggregation 是否进一步提升。

---

# 10. PPO / GiGPO Optimizer 参数

这部分尽可能与 GiGPO baseline 完全一致。

## 10.1 Actor learning rate

$$
\boxed{\mathrm{LR}=10^{-6}}
$$

不为 BACE 重新调学习率。

---

## 10.2 Reference KL coefficient

推荐：

$$
\boxed{\beta_{\mathrm{KL}}=0.01}
$$

建议只在训练明显不稳定时做：

$$
\{0.005,0.01,0.02\}
$$

的稳定性检查。

---

## 10.3 PPO clip

推荐：

```text
inherit exactly from GiGPO baseline
```

不要因为 BACE 引入新的 branch 数据就首先重新调 PPO clip，否则会破坏归因。

---

## 10.4 Loss aggregation

推荐：

```text
loss_agg_mode = token-mean
```

root、copied branch origin 和 branch suffix 统一进入同一个 PPO objective。

当前主方法没有：

```text
lambda_branch
```

这种独立 branch loss coefficient。

---

# 11. ALFWorld 工程参数

第一版建议完整继承当前 GiGPO ALFWorld 配置。

| 参数 | 推荐值 |
|---|---:|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| train data size | 16 |
| validation data size | 128 |
| group / leaf budget | 8 |
| max prompt length | 2048 |
| max response length | 512 |
| actor LR | $1\times10^{-6}$ |
| max environment steps | 50 |
| $\gamma$ | 0.95 |
| step advantage weight $\omega$ | 1.0 |
| reference KL coefficient | 0.01 |
| invalid-action penalty | 0.1 |
| validation temperature | 0.4 |
| training epochs | 150 |
| rollout engine | 与 baseline 相同 |
| PPO minibatch/microbatch | 与 8 卡 GiGPO baseline 相同 |

---

# 12. 推荐的第一版 YAML/配置逻辑

可将 BACE 新参数组织为：

```yaml
algorithm:
  adv_estimator: gigpo
  gamma: 0.95

  gigpo:
    step_advantage_w: 1.0
    mode: mean_std_norm

  bace:
    # rollout topology
    leaf_budget: 8
    pilot_roots: 2
    max_branches_per_anchor: 2
    competence_threshold: 0.5

    # task-family prior
    family_prior_alpha0: 1.0
    family_prior_beta0: 1.0
    history_decay: 0.8
    transfer_ratio: 0.1
    transfer_strength_min: 2.0
    transfer_strength_max: 8.0

    # local beta posterior
    anchor_prior_strength: 2.0

    # ERV
    erv_method: monte_carlo
    erv_mc_samples: 512
    erv_threshold: 0.01
    erv_action_temperature: 0.02

    # anchor structure
    min_anchor_occurrences: 2
    min_observed_actions: 2
    candidate_action_source: observed
    candidate_action_cap: null
    max_branch_depth: 1
    origin_sampling: uniform

    # branch training
    train_branch_origin: true
    train_branch_suffix: true
    train_replay_prefix: false
    reset_local_posterior_after_update: true

    # local credit
    local_credit_mode: occurrence
```

PPO/KL 等继续放在原 verl-agent/GiGPO 配置下，不建议复制到 BACE namespace 中。

---

# 13. 参数调优优先级

第一轮不要对所有参数做 grid search。

推荐优先级：

$$
\boxed{
\tau_\mu
>
\tau_{\mathrm{ERV}}
>
\tau_{\mathrm{comp}}
>
\lambda_{\mathrm{hist}}
>
\omega
>
\kappa_A
>
M_{\mathrm{MC}}
}
$$

但由于当前已经冻结：

$$
\tau_{\mathrm{ERV}}=0.01,
\qquad
\lambda_{\mathrm{hist}}=0.8,
\qquad
\omega=1.0,
$$

建议实际实施顺序如下。

## Phase 1：代码正确性与 controller diagnostic

固定全部参数，重点验证：

```text
replay correctness
anchor count
observed action count
Beta posterior update
ERV reproducibility
U(z) distribution
capacity correction
root/branch topology distribution
```

## Phase 2：只调 $\tau_\mu$

$$
\tau_\mu\in\{0.01,0.02,0.05\}.
$$

确定动作 acquisition 的集中程度。

## Phase 3：ERV threshold 消融

主值固定：

$$
0.01.
$$

另外运行：

$$
\{0,0.005,0.02\}.
$$

验证“过滤低价值 anchor”是否必要。

## Phase 4：Topology sensitivity

$$
\tau_{\mathrm{comp}}\in\{0.3,0.5,0.7\},
$$

$$
L_{\max}\in\{1,2,3\}.
$$

## Phase 5：History sensitivity

主值：

$$
\lambda_{\mathrm{hist}}=0.8.
$$

对照：

$$
\lambda_{\mathrm{hist}}\in\{0.6,0.9\}.
$$

检查 prior 太短或太长是否影响 topology stability。

## Phase 6：Optimizer-side enhancement

完成 acquisition 主实验后，再测试：

```text
occurrence-level local credit
action-aggregated local credit
```

避免 acquisition 与 optimizer 改动同时发生，导致归因不清。

---

# 14. 必须记录的参数诊断指标

## 14.1 Task prior

```text
family prior mean mu_c
family concentration K_c
transferred strength kappa_T
instance pilot outcomes
instance posterior mean
q_g = P(phi_g > tau_comp)
planned branch quota Q_bar
```

需要额外记录不同历史 age 的累计权重，确认 $\lambda=0.8$ 没有让 prior 过度滞后。

## 14.2 Anchor / ERV

```text
number of repeated anchors
number of structural anchors
number of effective anchors
U(z) distribution
DeltaERV(z,u) distribution
ERV Monte-Carlo standard error
selected anchor utility
selected action ERV
```

## 14.3 Capacity

```text
planned root count
planned branch count
number of capacity-correction roots
actual root count
actual branch count
branch quota retention ratio
branches per anchor
```

## 14.4 Training

```text
root trainable tokens
branch-origin trainable tokens
branch-suffix trainable tokens
PPO clipping fraction
policy KL
advantage mean/std
root vs branch gradient contribution
```

## 14.5 System cost

```text
natural rollout tokens
branch suffix tokens
replay environment steps
GPU generation time
CPU ERV time
wall-clock per update
peak GPU memory
```

---

# 15. 关键参数消融矩阵

不建议一次运行完整笛卡尔积。推荐以下单因素消融。

| 实验 | 主值 | 对照值 | 主要回答的问题 |
|---|---:|---|---|
| ERV threshold | **0.01** | 0 / 0.005 / 0.02 | 是否需要过滤低价值 anchors |
| ERV temperature | **0.02** | 0.01 / 0.05 | acquisition 应多贪心 |
| History decay | **0.8** | 0.6 / 0.9 | history memory 应多长 |
| Max branches/anchor | **2** | 1 / 3 / $\infty$ | breadth-depth 集中程度 |
| Competence threshold | **0.5** | 0.3 / 0.7 | topology readiness threshold |
| Local prior strength | **2** | 1 / 4 | local Bayesian shrinkage 强度 |
| ERV MC samples | **512** | 256 / 1024 | MC 成本与稳定性 |
| Step advantage weight | **1.0** | 0.5 / 0.8 | local credit 权重 sensitivity |
| Local credit mode | occurrence | action-aggregated | action-consistent credit 是否额外有效 |

---

# 16. 当前主实验参数清单

第一版 ALFWorld BACE-GiGPO 主实验可以直接冻结为：

```text
# GiGPO / PPO
step_advantage_w = 1.0
gamma = 0.95
actor_lr = 1e-6
kl_coef = 0.01
loss_agg_mode = token-mean

# rollout topology
B = 8
N_pilot = 2
L_max = 2
tau_comp = 0.5

# task-family history
A0 = 1
B0 = 1
lambda_hist = 0.8
tau_T = 0.1
kappa_T_min = 2
kappa_T_max = 8

# local posterior
kappa_A = 2

# ERV
ERV_method = Monte-Carlo
M_MC = 512
tau_ERV = 0.01
tau_mu = 0.02

# anchor
min_anchor_occurrences = 2
min_observed_actions = 2
candidate_actions = all_observed_valid_actions
candidate_cap = None
origin_sampling = uniform
branch_depth = 1

# branch data
train_copied_branch_origin = True
train_branch_suffix = True
train_replay_prefix = False
reset_anchor_action_posterior_after_actor_update = True

# credit
local_credit_mode = occurrence
epsilon_norm = 1e-6
leaf_weighting = uniform
```

---

# 17. 实验时的决策原则

## 17.1 与 GiGPO 公平比较

所有不属于 BACE 的 optimizer 参数尽量完全继承 GiGPO：

- model；
- learning rate；
- PPO clipping；
- KL coefficient；
- response length；
- environment horizon；
- group/leaf budget；
- train batch；
- validation protocol。

主对比首先保证：

$$
\boxed{
\text{GiGPO: 8 natural rollouts}
\quad\text{vs.}\quad
\text{BACE: 8 terminal leaves with adaptive root/branch topology}
}
$$

并额外报告真实：

- replay steps；
- generated tokens；
- wall-clock；
- GPU hours。

## 17.2 不通过大量调参制造收益

第一轮正式结果应采用本文件的固定主值。只有：

- $\tau_\mu$；
- $\tau_{\mathrm{ERV}}$；
- $L_{\max}$；
- $\tau_{\mathrm{comp}}$；
- $\lambda_{\mathrm{hist}}$；

作为 BACE-specific sensitivity。

其余参数优先固定。

## 17.3 先看机制指标，再看最终成功率

尤其是：

$$
\tau_{\mathrm{ERV}}=0.01
$$

是否合理，不能只看最终 success rate；还必须看：

- effective-anchor fraction；
- average $U(z)$；
- branch quota retention；
- selected-anchor ERV；
- realized posterior/regret reduction；
- root/branch topology distribution。

这样才能判断是 threshold 本身合适，还是某个参数碰巧让最终 reward 更高。

---

# 18. 最终推荐摘要

当前第一版主实验的关键参数为：

$$
\boxed{
B=8,
\quad
N_{\mathrm{pilot}}=2,
\quad
L_{\max}=2
}
$$

$$
\boxed{
\tau_{\mathrm{comp}}=0.5,
\quad
\lambda_{\mathrm{hist}}=0.8,
\quad
\tau_T=0.1,
\quad
\kappa_T\in[2,8]
}
$$

$$
\boxed{
\kappa_A=2,
\quad
M_{\mathrm{MC}}=512,
\quad
\tau_{\mathrm{ERV}}=0.01,
\quad
\tau_\mu=0.02
}
$$

$$
\boxed{
\omega=1.0,
\quad
\gamma=0.95,
\quad
\beta_{\mathrm{KL}}=0.01
}
$$

对应的方法逻辑是：

> **利用带指数遗忘的 task-family history 稳定当前实例的两条 pilot；用 competence posterior 决定计划 refinement 预算；通过 anchor capacity 保证自然 evidence 广度；采用 Monte Carlo ERV 与 $U(z)\ge0.01$ 过滤并选择真正有信息价值的 anchor；在 anchor 内用 $\tau_\mu=0.02$ 的 ERV-softmax 分配 branch action；最后保持 GiGPO 的 $\omega=1.0$ 与统一 PPO optimizer，使实验提升尽可能归因于 BACE 的主动 rollout acquisition。**

