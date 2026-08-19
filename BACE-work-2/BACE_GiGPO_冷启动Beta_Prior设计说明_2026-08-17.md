# BACE-GiGPO 冷启动 Beta Prior 设计说明

> 版本：2026-08-17  
> 适用方案：取消 Pilot、使用历史 natural-root competence 直接规划 Root/Branch topology 的 Exact Batch-ERV 版本

---

# 1. 问题背景

在当前 BACE-GiGPO 中，我们已经取消独立 Pilot phase。每个 policy update 开始时，系统直接根据 task-family 的历史 natural-root competence belief 计算 refinement readiness：

$$
q_{c,k}=P(\phi_{c,k}>\tau_{\mathrm{comp}}),
$$

再用：

$$
\bar Q_g
=
\operatorname{round}\left[(B-R_{\min})q_{c_g,k}\right]
$$

规划当前任务实例的初始 branch 数，并令：

$$
R_g^{(0)}=B-\bar Q_g.
$$

主配置中：

$$
B=8,
\qquad
R_{\min}=2,
\qquad
\tau_{\mathrm{comp}}=0.5.
$$

因此，冷启动 task-family Beta prior 会直接决定训练最初若干 update 是否会触发 branch。

原先若使用：

$$
\operatorname{Beta}(1,1),
$$

则其均值为：

$$
\mu_0=0.5.
$$

同时：

$$
P(\phi>0.5)=0.5.
$$

于是：

$$
\bar Q
=
\operatorname{round}(6\times0.5)
=3,
$$

即一开始就会计划：

$$
\boxed{5R+3B}.
$$

这与我们的训练经验和方法动机不一致：在 ALFWorld 等 benchmark 上，初始 actor 在最早若干 epoch/update 中通常成功率很低，此时更合理的行为应是优先进行 breadth exploration，而不是立即投入较多局部 refinement budget。

因此，当前方案不应再使用对称无信息先验：

$$
\boxed{\operatorname{Beta}(1,1)}.
$$

---

# 2. 设计原则

新的冷启动 prior 应满足以下四个要求。

## 2.1 初始成功率应明显低于 0.5

我们已经知道初始 actor 在目标 benchmark 上并不是“成功/失败各半”的模型，因此 prior mean 不应设为：

$$
0.5.
$$

更合理的是显式编码一个较低的初始 natural-root success expectation。

---

## 2.2 Prior 应偏低，但不能过强

不能简单使用：

$$
\operatorname{Beta}(1,9),
$$

虽然其均值也是：

$$
0.1,
$$

但 concentration 为：

$$
10,
$$

等价于人为加入约 10 个 pseudo-observations，会使后续真实成功 evidence 很难改变 competence belief。

我们希望表达的是：

> 初始模型大概率很弱，但这是一个弱先验，而不是一个高度确定的结论。

因此应该同时控制：

- 初始均值；
- 初始 prior strength。

---

## 2.3 不额外引入固定 warmup epoch

不推荐额外规定：

```text
前 1--2 个 epoch 禁止 branch
```

或：

```yaml
disable_branch_first_n_epochs: 2
```

因为这种规则：

1. 引入额外 schedule 超参数；
2. 与 benchmark、模型大小和训练速度绑定；
3. 破坏 BACE “由 competence 自动决定 breadth/refinement” 的核心逻辑。

更干净的做法是：

$$
\boxed{
\text{早期不 branch，应由 competence prior 自然产生，而不是人工禁止。}
}
$$

---

## 2.4 真实 natural-root evidence 必须能逐渐推翻冷启动 prior

训练开始后，所有当前 batch 中实际生成的 natural roots——包括 capacity correction 新增 roots——都会进入下一批 task-family history。

因此冷启动 prior 只负责最初阶段，之后 topology 应越来越由真实 natural-root evidence 决定。

Branch outcomes 仍然不进入 family competence prior。

---

# 3. 推荐参数化方式

与其直接把：

$$
(A_0,B_0)
$$

作为主要人工参数，更推荐使用更直观的“均值—强度”参数化。

定义：

$$
p_0
=
\text{初始预期 natural-root success rate},
$$

$$
\kappa_0
=
\text{初始 prior strength}.
$$

再令：

$$
\boxed{
A_0=\kappa_0p_0,
\qquad
B_0=\kappa_0(1-p_0).
}
$$

这样：

$$
\mathbb E[\theta_{c,0}]=p_0,
$$

并且：

$$
A_0+B_0=\kappa_0.
$$

这一参数化更容易解释，也更方便在不同 benchmark 上迁移。

---

# 4. 当前推荐主配置

推荐：

$$
\boxed{p_0=0.10}
$$

以及：

$$
\boxed{\kappa_0=2}.
$$

因此：

$$
A_0=2\times0.1=0.2,
$$

$$
B_0=2\times0.9=1.8.
$$

即：

$$
\boxed{
\theta_{c,0}
\sim
\operatorname{Beta}(0.2,1.8).
}
$$

其语义是：

> 在训练开始时，我们预期初始 actor 的 natural-root success rate 大约为 10%，但这一判断只具有约 2 个 pseudo-observations 的强度。

这同时满足“保守冷启动”和“可快速被真实 evidence 推翻”两个目标。

---

# 5. 为什么该 prior 能自然关闭早期 branch

在当前 controller 中，冷启动 competence belief 近似为：

$$
\phi_{c,0}
\sim
\operatorname{Beta}(0.2,1.8).
$$

取：

$$
\tau_{\mathrm{comp}}=0.5,
$$

则：

$$
q_0
=
P(\phi_{c,0}>0.5)
\approx
0.0521.
$$

因为：

$$
B-R_{\min}=6,
$$

所以：

$$
6q_0
\approx
0.312.
$$

最终：

$$
\bar Q
=
\operatorname{round}(0.312)
=0.
$$

因此初始 topology 自动为：

$$
\boxed{8R+0B}.
$$

这正好符合训练早期的预期：

$$
\boxed{
\text{先扩大 breadth，积累 natural-root 成功证据；}
}
$$

$$
\boxed{
\text{只有 competence 提升后，才逐渐启用 refinement branches。}
}
$$

---

# 6. 不同初始均值的比较

固定：

$$
\kappa_0=2,
\qquad
\tau_{\mathrm{comp}}=0.5,
\qquad
B=8,
\qquad
R_{\min}=2.
$$

比较：

$$
p_0\in\{0.05,0.10,0.15\}.
$$

结果如下。

| $p_0$ | Beta prior | $q_0=P(\phi>0.5)$ | $6q_0$ | 初始 branch quota |
|---:|---:|---:|---:|---:|
| $0.05$ | $\operatorname{Beta}(0.1,1.9)$ | $\approx0.0226$ | $0.136$ | $0$ |
| **$0.10$** | **$\operatorname{Beta}(0.2,1.8)$** | **$\approx0.0521$** | **$0.312$** | **$0$** |
| $0.15$ | $\operatorname{Beta}(0.3,1.7)$ | $\approx0.0885$ | $0.531$ | $1$ |

由当前 quota rounding 规则可知，若希望冷启动明确得到：

$$
Q=0,
$$

则初始均值大约需要满足：

$$
\boxed{p_0<0.1435.}
$$

因此：

$$
\boxed{p_0=0.10}
$$

是一个合理且留有安全余量的主起点。

---

# 7. 为什么不通过修改 $\tau_{\mathrm{comp}}$ 解决

另一种方法是保留：

$$
\operatorname{Beta}(1,1)
$$

但把：

$$
\tau_{\mathrm{comp}}
$$

从 $0.5$ 提高到，例如 $0.7$ 或 $0.8$。

不推荐这样做。

原因是两者表达的是不同含义。

## Prior mean 表达

$$
\boxed{
\text{我们相信初始 actor 的 natural-root competence 大概是多少。}
}
$$

## Competence threshold 表达

$$
\boxed{
\text{多高的 competence 才被认为进入 refinement-ready regime。}
}
$$

我们当前真正需要修正的是前者，而不是后者。

如果 benchmark 上初始成功率大约只有 5%--10%，却仍然使用均值 0.5 的 prior，只是把 threshold 提高，本质上是在用错误 prior 配合人为 threshold 抵消错误。

因此推荐保持：

$$
\boxed{\tau_{\mathrm{comp}}=0.5}
$$

并修正冷启动 prior。

---

# 8. 为什么不使用强低成功率先验

例如：

$$
\operatorname{Beta}(1,9)
$$

也满足：

$$
\mathbb E[\phi]=0.1.
$$

但其 strength：

$$
\kappa=10.
$$

如果模型在训练中迅速改善，真实 natural-root successes 必须积累较多才能推翻该 prior。

这可能导致：

$$
\text{early low prior}
\rightarrow
\text{long-term low }q_c
\rightarrow
\text{branch 开启过晚}.
$$

相比之下：

$$
\operatorname{Beta}(0.2,1.8)
$$

同样均值为 0.1，但：

$$
\kappa=2.
$$

它表达的是更合理的：

> “我们有理由认为模型一开始较弱，但实际 natural-root evidence 可以很快改变这个判断。”

---

# 9. 与历史遗忘机制的配合

当前 family history 使用指数遗忘：

$$
\widetilde S_{c,k}
=
\lambda_{\mathrm{hist}}
\widetilde S_{c,k-1}
+
S_{c,k}^{\mathrm{root}},
$$

$$
\widetilde F_{c,k}
=
\lambda_{\mathrm{hist}}
\widetilde F_{c,k-1}
+
F_{c,k}^{\mathrm{root}}.
$$

推荐仍保持：

$$
\boxed{\lambda_{\mathrm{hist}}=0.8.}
$$

原因是取消 Pilot 后，训练早期可能以：

$$
8R+0B
$$

开始，单个 task 会一次贡献较多 failure evidence。

如果历史永不衰减，早期失败可能长期压低 competence posterior，使 controller 过久停留在 root-heavy regime。

指数遗忘使：

$$
\boxed{
\text{旧 actor 的失败证据随着策略改进逐渐失效。}
}
$$

---

# 10. 与 concentration clipping 的配合

当前 controller concentration：

$$
\kappa_{c,k}^{T}
=
\operatorname{clip}
\left(
\tau_TK_{c,k},
\kappa_T^{\min},
\kappa_T^{\max}
\right).
$$

推荐继续使用：

$$
\tau_T=0.1,
$$

$$
\kappa_T^{\min}=2,
$$

$$
\kappa_T^{\max}=8.
$$

其中：

- $\kappa_T^{\min}=2$ 与冷启动 $\kappa_0=2$ 对齐；
- $\kappa_T^{\max}=8$ 防止大量早期 failure evidence 使 competence belief 变成几乎不可逆的高度确定低成功率 posterior。

因此，低均值 cold-start prior 必须与：

$$
\boxed{
\text{history decay + bounded concentration}
}
$$

一起使用。

---

# 11. 新的冷启动行为

使用：

$$
p_0=0.10,
\qquad
\kappa_0=2,
$$

后，预期 topology 演化应呈现：

```text
训练初期：
低 competence
-> q_c 很低
-> 8R + 0B

开始出现稳定成功：
competence posterior 上升
-> 7R + 1B / 6R + 2B

模型进一步成熟：
q_c 持续提高
-> 5R + 3B / 4R + 4B

高 competence：
更多 refinement budget
-> 更高比例 Batch-ERV branches
```

也就是说：

$$
\boxed{
\text{breadth}\rightarrow\text{refinement}
}
$$

不再依赖固定 epoch schedule，而由自然 success evidence 自动触发。

---

# 12. 推荐的参数表修改

当前参数表建议正式改成：

| 参数 | 原设置 | 新推荐 | 说明 |
|---|---:|---:|---|
| $p_0$ | 未显式定义 | **0.10** | 初始预期 natural-root success rate |
| $\kappa_0$ | 未显式定义 | **2** | 冷启动 prior strength |
| $A_0$ | $1$ | **$0.2$** | $A_0=\kappa_0p_0$ |
| $B_0$ | $1$ | **$1.8$** | $B_0=\kappa_0(1-p_0)$ |
| $\tau_{\mathrm{comp}}$ | $0.5$ | **$0.5$** | 不因 cold-start 问题修改 threshold |
| $\lambda_{\mathrm{hist}}$ | $0.8$ | **$0.8$** | 允许旧失败 evidence 衰减 |
| $\tau_T$ | $0.1$ | **$0.1$** | history strength transfer ratio |
| $\kappa_T^{\min}$ | $2$ | **$2$** | 与 cold-start strength 对齐 |
| $\kappa_T^{\max}$ | $8$ | **$8$** | 防止历史置信度锁死 |

---

# 13. 建议的配置写法

不建议在配置中仅暴露：

```yaml
alpha0: 0.2
beta0: 1.8
```

更推荐：

```yaml
family_competence_prior:
  initial_mean: 0.10
  initial_strength: 2.0

  history_decay: 0.8
  transfer_ratio: 0.1
  concentration_min: 2.0
  concentration_max: 8.0
```

程序内部再计算：

```text
alpha0 = initial_mean * initial_strength
beta0  = (1 - initial_mean) * initial_strength
```

这样更容易解释，也方便不同 benchmark 根据 initial actor competence 调整冷启动 belief。

---

# 14. 建议的敏感性实验

冷启动 prior 不需要做大规模 grid search。

建议只比较：

$$
\boxed{
p_0\in\{0.05,0.10,0.15\}
}
$$

固定：

$$
\kappa_0=2.
$$

主要观察：

1. 前若干 updates 的 branch activation time；
2. 第一次出现 $Q>0$ 的 training step；
3. $q_c$ calibration；
4. topology distribution；
5. early success rate；
6. 是否长期停留在 all-root regime；
7. 是否过早触发 branch；
8. 最终 success 与 wall-clock。

如果 $p_0=0.10$ 能实现：

- 初始若干 updates 基本无 branch；
- 随模型成功率提高自然开启 branch；
- 不发生长期 branch starvation；

则应直接固定，不继续细调。

---

# 15. 不推荐的替代方案

## 15.1 Beta(1,1)

不推荐。

原因：

$$
q_0=0.5,
$$

会在没有成功 evidence 时就规划中等规模 branch budget。

---

## 15.2 Beta(1,9)

不推荐作为主版本。

原因：均值合理，但 strength 太大，容易使 branch 开启过晚。

---

## 15.3 固定前两个 epoch 禁止 branch

不推荐作为主版本。

原因：额外 schedule heuristic，削弱 competence-adaptive topology 的方法逻辑。

---

## 15.4 提高 $\tau_{\mathrm{comp}}$ 来压制 early branch

不推荐作为 cold-start 修复方式。

原因：该参数应表示 refinement regime threshold，而不是用来补偿不合理的初始 competence prior。

---

# 16. 最终推荐

当前 BACE-GiGPO 冷启动 task-family competence prior 建议从：

$$
\boxed{
\operatorname{Beta}(1,1)
}
$$

修改为：

$$
\boxed{
p_0=0.10,
\qquad
\kappa_0=2
}
$$

即：

$$
\boxed{
\operatorname{Beta}(0.2,1.8).
}
$$

它的核心作用不是人为规定“训练前两轮不能 branch”，而是把合理的 benchmark cold-start knowledge 编码为一个**低均值、弱强度的 competence prior**。

这样：

$$
\boxed{
\text{初始失败占主导}
\Rightarrow
\text{低 }q_c
\Rightarrow
\text{自然进入 all-root breadth regime}
}
$$

而随着真实 natural-root success evidence 增加：

$$
\boxed{
\text{competence posterior 上升}
\Rightarrow
\text{branch budget 自动开启并逐步增加}.
}
$$

因此，该修改与当前取消 Pilot 后的 BACE-GiGPO 逻辑完全一致，也比固定 warmup schedule 更符合方法本身的 Bayesian competence-guided rollout topology 叙事。
