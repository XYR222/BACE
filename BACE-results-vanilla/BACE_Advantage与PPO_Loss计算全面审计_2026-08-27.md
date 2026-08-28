# BACE Advantage 与 PPO Loss 计算全面审计

日期：2026-08-27  
审计对象：`work-BACE/verl-agent-src` 当前正式训练代码，以及已完成的 150-step BACE 实验  
正式实验：`bace_alfworld_qwen2_5_1_5b_exact_rotation_fix_seed0`

---

## 1. 结论摘要

### 1.1 总体结论

当前正式 BACE 主线的 advantage 和 actor loss **连接正确，公式实现与当前 GiGPO baseline 语义一致，没有发现会使本次 150-step 在线训练结果失效的阻塞性计算错误**。

当前真实计算可以概括为：

\[
\boxed{
A_i^{\text{BACE}}
=
A_i^{\text{macro}}
+
\lambda_{\text{step}} A_i^{\text{local}}
}
\]

其中正式配置为：

\[
\lambda_{\text{step}}=1,
\qquad
\text{normalization}=\texttt{mean\_std\_norm},
\qquad
\text{local credit}=\texttt{occurrence}.
\]

这个 occurrence-level advantage 被复制到该 occurrence 的全部有效 response token，然后进入 clipped PPO、entropy bonus 和独立 KL loss：

\[
\boxed{
L_{\text{actor}}
=
L_{\text{PPO}}
-0.001H
+0.01L_{\text{KL-k3}}
}
\]

正式训练没有 critic，因此没有 value loss；代码中的 `returns` 只是与 `advantages` 相同的兼容字段，不参与 critic 拟合。

### 1.2 正确性判断

| 项目 | 判断 |
|---|---|
| ALFWorld terminal reward 写入 token | 正确 |
| `gamma=0.95` 的 step reward-to-go | 正确，且只用于 local 路径 |
| invalid-action penalty | 正确接入 macro 和当前 occurrence 的 local 路径 |
| macro advantage | 与当前 GiGPO implementation 一致 |
| local advantage | 与 GiGPO step normalization 一致 |
| BACE occurrence credit | 符合正式配置和既定主口径 |
| root/branch 混合训练 | metadata、trajectory 和 anchor 分组链路正确 |
| PPO clipping / dual clipping | 实现正确 |
| entropy 符号 | 正确，最小化 loss 时减去 entropy bonus |
| KL 符号与位置 | 正确；KL 不进入 reward，而作为 actor loss 正则项 |
| response mask | 正式配置 `multi_turn=false` 下前后一致 |
| 数值 trace | 代表 step 中 combined 与 macro+local 一致到 float32 误差 |

### 1.3 不是 bug、但必须明确的设计语义

当前实现并不是“每条 terminal leaf 等权”的 tree-aware optimizer，而是保持 GiGPO 的下游计算不变：

1. macro normalization 是 **physical occurrence weighted**；
2. PPO loss 是 **response-token weighted**；
3. 同一 root 派生出的 branch origin 会作为新的训练 occurrence 再次进入 loss；
4. invalid penalty 同时影响 macro 和 local 两条归一化信用链；
5. branch 增多后，相关 occurrence 的有效权重会随 lineage 大小增长。

这些都符合当前“BACE 只改变 rollout acquisition，优化器尽量与 GiGPO 对齐”的正式实验口径，但它们是后续消融和改进的重点，不应被误写成 leaf-uniform 或 lineage-uniform。

---

## 2. 审计范围与证据

本次逐段检查了：

- `agent_system/reward_manager/episode.py`：episode reward 写入方式；
- `gigpo/core_gigpo.py`：discounted return、macro/local normalization；
- `recipe/bace_gigpo/advantage.py`：BACE advantage 组合；
- `verl/trainer/ppo/ray_trainer.py`：reward、penalty、advantage 的调用顺序；
- `verl/trainer/ppo/core_algos.py`：PPO、dual clip、KL 与 loss aggregation；
- `verl/workers/actor/dp_actor.py`：actor 的最终总 loss 与反向传播；
- BACE 正式实验的 Hydra 解析配置、step trace 和训练日志；
- BACE advantage 专项单测及 trainer integration 单测；
- 上游/基线目录 `AFH/examples/gigpo_trainer/run_alfworld.sh` 和 `AFH/gigpo/core_gigpo.py`。

专项测试结果：

```text
11 passed, 1 warning in 207.14s
```

warning 来自 Ray state API 的弃用提示；pytest 退出后的 `multiprocess.resource_tracker` 析构提示与 advantage/loss 数值无关。

---

## 3. 正式实验的实际参数

以下不是配置文件默认值推测，而是从正式作业 `3116958` 的运行时解析配置核对得到：

| 参数 | 正式值 |
|---|---:|
| `algorithm.adv_estimator` | `bace_gigpo` |
| `algorithm.gamma` | `0.95` |
| `algorithm.gigpo.mode` | `mean_std_norm` |
| `algorithm.gigpo.compute_mean_std_cross_steps` | `true` |
| `algorithm.gigpo.step_advantage_w` | `1.0` |
| `algorithm.gigpo.enable_similarity` | `false` |
| `algorithm.bace.local_credit_mode` | `occurrence` |
| `use_invalid_action_penalty` | `true` |
| invalid penalty | `0.1` |
| `algorithm.use_kl_in_reward` | `false` |
| actor `use_kl_loss` | `true` |
| actor KL coefficient | `0.01` |
| actor KL type | `low_var_kl` / k3 |
| PPO policy loss | `vanilla` |
| clip low / high | `0.2 / 0.2` |
| dual-clip constant | `3.0` |
| entropy coefficient | `0.001` |
| loss aggregation | `token-mean` |
| PPO epochs | `1` |
| PPO mini-batch | `256` |
| micro-batch / GPU | `16` |
| gradient clipping | `1.0` |
| multi-turn framework flag | `false` |
| critic | 不启用 |

上述 advantage、invalid penalty、KL、gamma、clip 和 entropy 参数与原版 GiGPO ALFWorld 脚本保持一致；BACE 的差异主要来自采样拓扑和训练 occurrence 的构成。

---

## 4. 一个“训练 occurrence”是什么

BACE 最终送入 PPO 的基本样本不是整条 episode，而是一次模型决策：

```text
pre-action observation
    -> model response tokens
    -> projected ALFWorld action
    -> environment reward / next observation
```

每一行至少带有：

- `uid`：task group ID；
- `traj_uid`：root 或 branch trajectory ID；
- `anchor_obs`：动作前 observation；
- `rewards`：该环境 step 的即时奖励；
- `episode_rewards`：该 trajectory 的最终总回报；
- `is_action_valid`：动作是否有效；
- `source_type`：`root`、`branch_origin` 或 `branch_suffix`。

自然 root 的每个 env step 是一个 occurrence。一个 branch 则包含：

1. 从自然 root 复制的 `branch_origin` occurrence；
2. 从该分支点继续生成的若干 `branch_suffix` occurrences。

因此 branch origin 的 response token 会再次训练，但它拥有新的 `traj_uid` 和该 branch 的 terminal outcome。这不是 accidental duplicate，而是 BACE 用不同 continuation outcome 给同一决策 occurrence 提供额外信用证据的核心机制。

---

## 5. Reward 的完整计算

### 5.1 ALFWorld 环境奖励

文本 ALFWorld 当前使用：

\[
r_t^{\text{env}}=10\cdot \mathbf 1(\text{won}).
\]

所以通常：

- 未完成任务的中间 step：`0`；
- 完成任务的 terminal step：`10`；
- 失败或 horizon 结束：总回报 `0`。

### 5.2 Episode score 写入 token

对 occurrence \(i\)，reward manager 读取其所属 trajectory 的 `episode_rewards`，并只写入该 response 的最后一个有效 token：

\[
s_{i,k}^{\text{episode}}
=
\begin{cases}
R_{\tau(i)}, & k=\text{last valid response token},\\
0, & \text{otherwise}.
\end{cases}
\]

同一 trajectory 的每个 occurrence 都携带同一个 terminal episode return。因此，一条成功 trajectory 中的每个动作 response 都得到 terminal score 10，而不是只有最后一步得到 10。

### 5.3 Invalid-action penalty

若 occurrence \(i\) 的动作无效，代码在其最后一个 response token 上减去：

\[
p_i=0.1\cdot \mathbf 1(\text{invalid}_i).
\]

于是 macro 路径使用的 token score 为：

\[
\tilde s_{i,k}
=
s_{i,k}^{\text{episode}}
-p_i\mathbf 1(k=\text{last}).
\]

即 occurrence scalar score：

\[
S_i=\sum_k\tilde s_{i,k}
=R_{\tau(i)}-p_i.
\]

### 5.4 Step discounted return

在 reward manager 计算 episode token score之前，代码先按 `traj_uid` 对真实环境即时奖励反向递推：

\[
G_t=r_t^{\text{env}}+\gamma G_{t+1},
\qquad \gamma=0.95.
\]

随后，再对当前无效 occurrence 的 step return 点式减去 0.1：

\[
\tilde G_t=G_t-0.1\mathbf 1(\text{invalid}_t).
\]

注意调用顺序带来的精确语义：invalid penalty 不会重新进入 discounted-return 递推，所以未来 occurrence 的 invalid penalty 不会折扣传播到更早 occurrence。它是对“当前错误动作”的局部惩罚。这与当前 GiGPO 代码一致，也符合把无效动作惩罚视为 action-local penalty 的解释。

### 5.5 KL 不进入 reward

正式配置为：

```text
algorithm.use_kl_in_reward=false
```

所以：

\[
\text{token_level_rewards}=\text{token_level_scores}.
\]

KL 不改变 macro/local advantage；它在 actor loss 最后单独加入，避免 reward KL 和 loss KL 同时启用造成双重惩罚。

---

## 6. Macro advantage 的完整计算

设 \(i\) 是一个训练 occurrence，\(g(i)\) 是其 task group，\(\tau(i)\) 是 trajectory。

首先：

\[
S_i=\sum_k \tilde s_{i,k}.
\]

正式配置 `compute_mean_std_cross_steps=true`，因此 task group 内统计的是全部 physical occurrences：

\[
\mathcal I_g=\{i:g(i)=g\}.
\]

均值：

\[
\mu_g^{E}
=
\frac{1}{|\mathcal I_g|}
\sum_{i\in\mathcal I_g}S_i.
\]

代码使用 PyTorch 默认的 sample standard deviation：

\[
\sigma_g^{E}
=
\sqrt{
\frac{1}{|\mathcal I_g|-1}
\sum_{i\in\mathcal I_g}(S_i-\mu_g^{E})^2
}.
\]

正式 `mean_std_norm` 下：

\[
A_i^{\text{macro}}
=
\frac{S_i-\mu_g^{E}}
{\sigma_g^{E}+10^{-6}}.
\]

该 scalar 被复制到 occurrence 的全部有效 response tokens。

### 6.1 occurrence-weighted 的含义

假设同一 task 中：

- trajectory A 有 20 个 occurrence；
- trajectory B 有 5 个 occurrence。

那么 A 的 terminal outcome 会在 macro 均值和方差中出现 20 次，B 出现 5 次。当前不是“一条 trajectory 一票”，而是“一个 physical occurrence 一票”。

这是 GiGPO 当前实现采用的语义，也是 BACE 主实验为了控制变量有意保留的语义。它不等价于 leaf-uniform，但不能在本次主实验中单独视作 BACE 实现 bug。

### 6.2 `compute_mean_std_cross_steps=false` 的区别

关闭该开关时，代码在构造 task group 统计量时只保留每个 `(task_id, traj_uid)` 的第一次出现；之后仍将所得 task-level mean/std 应用到该 trajectory 的所有 occurrences。

正式实验没有使用这一模式。

### 6.3 singleton 边界

上游代码对 macro group 只有一个样本的特殊处理是：

```text
mean = 0
std = 1
```

因此 singleton macro advantage 会接近原始 score，而不是通常相对归一化直觉下的 0。这是一个潜在边界缺陷，但正式配置每个 task 的 leaf budget 为 8，并且每条 leaf 又包含多个 occurrences，不会进入 singleton 路径。

---

## 7. Local advantage 的完整计算

### 7.1 Anchor group

当前 `enable_similarity=false`，因此 local group 由以下二元组严格决定：

\[
h(i)=(\text{task\_id}_i,\text{exact anchor\_obs}_i).
\]

也就是说：

- 相同文本 observation、但不同 task，不会混组；
- 同一 task 中完全相同的 pre-action observation 会进入同组；
- 自然 root 和 branch occurrence 可以进入同一个 local group；
- 相似但不完全相同的 observation 不会合并。

这与 strict identity / exact anchor 的正式口径一致。

### 7.2 Occurrence credit

正式配置为 `local_credit_mode=occurrence`。对 anchor group \(h\)：

\[
\mu_h^{S}
=
\frac{1}{|\mathcal I_h|}
\sum_{i\in\mathcal I_h}\tilde G_i,
\]

\[
\sigma_h^{S}
=
\sqrt{
\frac{1}{|\mathcal I_h|-1}
\sum_{i\in\mathcal I_h}(\tilde G_i-\mu_h^{S})^2
},
\]

\[
A_i^{\text{local}}
=
\frac{\tilde G_i-\mu_h^{S}}
{\sigma_h^{S}+10^{-6}}.
\]

若 local group 只有一个 occurrence，代码令 mean 等于它自身、std 为 1，所以 local advantage 为 0。这是合理的：没有对照动作/continuation 时，不产生相对局部信用。

### 7.3 为什么不是 action mean

仓库保留了 `local_credit_mode=action_mean` 兼容分支。该模式先对同一 anchor 下同一 action identity 的多个 outcomes 求均值，再与 anchor group mean 比较，使同一 action 的 occurrences 获得相同 local advantage。

正式配置明确选择 `occurrence`，所以即使两个 occurrences 的 canonical action 相同，只要 continuation return 不同，它们仍可获得不同 local credit。这符合当前“occurrence credit”主口径，也与 GiGPO 的 step normalization 完全一致。

---

## 8. 最终 BACE advantage

正式配置：

\[
\lambda_{\text{step}}=1.
\]

所以：

\[
A_i
=
A_i^{\text{macro}}+A_i^{\text{local}}.
\]

对 response token \(k\)：

\[
A_{i,k}=A_i\,m_{i,k},
\]

其中 \(m_{i,k}\) 是 response mask。padding token 的 advantage 为 0，有效 response token 的 advantage 在同一个 occurrence 内保持常数。

函数返回：

```python
return token_scores, token_scores, components
```

即：

\[
\texttt{advantages}=\texttt{returns}=A.
\]

这不是 GAE 意义上的 value target。因为 `bace_gigpo` 不启用 critic，`returns` 只是 veRL 统一接口和日志所需的兼容字段，不会产生 value loss，也不会造成同一信号训练两次。

---

## 9. PPO loss 的完整计算

对每个有效 response token，令：

\[
\Delta_{i,k}
=
\log\pi_\theta(a_{i,k}|x_i)
-
\log\pi_{\text{old}}(a_{i,k}|x_i),
\]

\[
\rho_{i,k}=e^{\Delta_{i,k}}.
\]

### 9.1 标准 PPO clip

正式 clip range 为：

\[
\epsilon_{\text{low}}=\epsilon_{\text{high}}=0.2.
\]

代码先计算：

\[
\ell_1=-A\rho,
\]

\[
\ell_2=-A\operatorname{clip}(\rho,0.8,1.2),
\]

\[
\ell_{\text{clip}}=\max(\ell_1,\ell_2).
\]

这是把 PPO 最大化目标写成最小化 loss 后的标准形式，符号正确。

### 9.2 Dual clip

对 negative advantage，代码额外使用 \(c=3\)：

\[
\ell_3=-3A.
\]

最终 token policy loss：

\[
\ell_{\text{PPO}}
=
\begin{cases}
\min(\ell_3,\ell_{\text{clip}}), & A<0,\\
\ell_{\text{clip}}, & A\ge 0.
\end{cases}
\]

它限制 negative-advantage token 在 ratio 极端变大时产生过大的正 loss，dual-clip 的条件和方向正确。

### 9.3 Token-mean aggregation

正式 `loss_agg_mode=token-mean`：

\[
L_{\text{PPO}}
=
\frac{
\sum_{i,k}m_{i,k}\ell_{i,k}
}{
\sum_{i,k}m_{i,k}
}.
\]

因此 response token 更多的 occurrence 对梯度贡献更大。虽然每个 occurrence 内的 advantage 是常数，但 occurrence 的总 loss 权重与其有效 response 长度近似成正比。

### 9.4 Entropy bonus

代码计算有效 response token 的平均 entropy：

\[
H=\operatorname{tokenmean}(H_{i,k}),
\]

并在最小化目标中使用：

\[
-0.001H.
\]

符号正确：entropy 越大，总 loss 越小，从而鼓励探索。

### 9.5 Low-variance KL loss

令：

\[
d=\log\pi_{\text{ref}}-\log\pi_\theta.
\]

`low_var_kl` / k3 逐 token 计算：

\[
K_{i,k}=e^d-d-1,
\]

并裁剪到 `[-10,10]`，再做 token mean：

\[
L_{\text{KL}}=\operatorname{tokenmean}(K_{i,k}).
\]

最终 actor loss：

\[
\boxed{
L_{\text{actor}}
=L_{\text{PPO}}
-0.001H
+0.01L_{\text{KL}}
}
\]

KL 系数为正，方向正确；由于 reward KL 已关闭，不存在重复 KL 正则。

### 9.6 Mini-batch 和 gradient accumulation

Advantage 在 driver 上对完整训练 batch 先统一计算，之后 actor 才按 PPO mini-batch/micro-batch 切分。因此 mean/std 不会被每张 GPU 或每个 micro-batch 各自重新计算。

非 dynamic batch 模式下，每个 micro-batch loss 除以 gradient accumulation 次数后再 backward，缩放方式正确。最后执行 gradient norm clip 1.0。

---

## 10. 完整数值示例

下面构造同一 task 的两条 trajectory，并假定各自有两个 occurrences：

| trajectory | env rewards | terminal return | outcome |
|---|---|---:|---|
| `success` | `[0, 10]` | 10 | 成功 |
| `failure` | `[0, 0]` | 0 | 失败 |

为便于演示，假设两条 trajectory 的第 1 个 occurrences 共享一个 anchor，第 2 个也共享另一个 anchor，且动作均有效。

### 10.1 Step returns

\[
\gamma=0.95.
\]

成功轨迹：

\[
G_2=10,
\qquad
G_1=0+0.95\times 10=9.5.
\]

失败轨迹：

\[
G_2=0,
\qquad
G_1=0.
\]

因此四个 occurrence 的 step returns 为：

```text
[9.5, 10, 0, 0]
```

### 10.2 Macro advantage

由于每个 occurrence 都携带所属 trajectory 的 terminal return，macro scores 为：

```text
[10, 10, 0, 0]
```

均值：

\[
\mu^E=5.
\]

sample std：

\[
\sigma^E
=\sqrt{\frac{(10-5)^2+(10-5)^2+(0-5)^2+(0-5)^2}{3}}
=\sqrt{\frac{100}{3}}
\approx5.7735.
\]

因此：

\[
A_{\text{success}}^{\text{macro}}\approx+0.8660,
\]

\[
A_{\text{failure}}^{\text{macro}}\approx-0.8660.
\]

### 10.3 Local advantage

第一个 anchor group 的 returns 为 `[9.5, 0]`：

\[
\mu_1^S=4.75,
\qquad
\sigma_1^S\approx6.7175.
\]

所以 local advantages 约为：

```text
[+0.7071, -0.7071]
```

第二个 anchor group 的 returns 为 `[10, 0]`，也得到：

```text
[+0.7071, -0.7071]
```

### 10.4 Combined advantage

\[
A^{\text{success}}
=0.8660+0.7071
\approx1.5731,
\]

\[
A^{\text{failure}}
=-0.8660-0.7071
\approx-1.5731.
\]

如果某个 success response 有 80 个有效 tokens，这 80 个 tokens 都获得 `+1.5731`；padding token 获得 0。

### 10.5 Invalid action 如何改变示例

假设 failure trajectory 的第一个动作无效，则该 occurrence：

```text
macro score: 0 -> -0.1
local step return: 0 -> -0.1
```

然后 macro 和 local 分别重新做组内标准化。不能简单说最终 combined advantage 固定减去 0.2，因为两个 `-0.1` 分别经过不同 group 的 mean/std normalization；它们的最终影响取决于 task group 和 anchor group 的其余样本。

### 10.6 PPO clipping 示例

取正 advantage：

\[
A=1.5731,
\qquad \rho=1.3.
\]

则：

\[
\ell_1=-1.5731\times1.3=-2.0450,
\]

\[
\ell_2=-1.5731\times1.2=-1.8877,
\]

\[
\ell_{\text{PPO}}=\max(-2.0450,-1.8877)=-1.8877.
\]

即 ratio 超过 1.2 后，正 advantage 的收益被 clip。

再取负 advantage：

\[
A=-1.5731,
\qquad\rho=4.
\]

标准 clipped loss 为：

\[
\ell_{\text{clip}}=6.2924.
\]

dual-clip 上限：

\[
\ell_3=-3A=4.7193.
\]

所以：

\[
\ell_{\text{PPO}}=\min(6.2924,4.7193)=4.7193.
\]

### 10.7 Token-mean 长度权重示例

假设成功两个 occurrences 一共 4 个有效 response tokens，失败两个 occurrences 一共 2 个 tokens；在 actor update 刚开始、\(\rho=1\) 时：

\[
L_{\text{PPO}}
=\frac{4(-1.5731)+2(+1.5731)}{6}
\approx-0.5244.
\]

尽管 occurrence-level advantages 的正负和为 0，token-level loss 并不为 0，因为正 advantage occurrences 的 response 更长。这正是 `token-mean` 的真实权重语义。

### 10.8 KL 示例

若某 token：

\[
\log\pi_\theta=-1.0,
\qquad
\log\pi_{\text{ref}}=-1.1,
\]

则：

\[
d=-0.1,
\]

\[
K=e^{-0.1}+0.1-1\approx0.00484.
\]

该 token 对最终总 loss 的 KL 加项约为：

\[
0.01\times0.00484=0.0000484.
\]

---

## 11. 代码与原版 GiGPO 的一致性

### 11.1 Occurrence 模式数值一致

`compute_bace_gigpo_advantage(..., local_credit_mode="occurrence")` 的核心是：

```text
core_gigpo.episode_norm_reward(...)
+ step_advantage_w * core_gigpo.step_norm_reward(...)
```

这与 `core_gigpo.compute_gigpo_outcome_advantage(...)` 的 macro/local 组合完全相同。BACE 只额外：

- 输出 macro/local/combined 三个诊断分量；
- 保留 `action_mean` 作为非正式主线的可选扩展；
- 对 shape、metadata 和空 response 做显式检查。

专项测试逐项验证了：

- `mean_norm` 和 `mean_std_norm`；
- `compute_mean_std_cross_steps=true/false`；
- occurrence BACE 与 GiGPO 输出相同；
- sample std 行为；
- invalid penalty 能进入 macro；
- action-mean 与 occurrence 模式确实不同；
- trainer 能保存 macro/local/combined diagnostics。

### 11.2 原版训练参数一致

原版 `AFH/examples/gigpo_trainer/run_alfworld.sh` 同样使用：

- gamma 0.95；
- step advantage weight 1.0；
- `mean_std_norm`；
- invalid penalty 0.1；
- actor KL loss 0.01 / `low_var_kl`；
- reward KL 关闭。

正式 BACE 的 optimizer 主干没有另换一套 advantage/loss。

---

## 12. 正式 trace 的数值核验

对 step 1、50、100、150 的 `trainable_occurrences.jsonl` 逐 occurrence 核对：

\[
\text{combined}
\stackrel{?}{=}
\text{macro}+\text{local},
\]

以及：

\[
\text{masked token advantage mean}
\stackrel{?}{=}
\text{combined}.
\]

结果：

| step | occurrences | root occ. | branch occ. | root tokens | branch tokens | max component error | max token-mean error | advantage min/max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,952 | 5,952 | 0 | 563,070 | 0 | `4.77e-7` | `2.86e-6` | `[-6.063, 8.971]` |
| 50 | 5,440 | 5,344 | 96 | 432,046 | 8,102 | `5.96e-7` | `3.34e-6` | `[-21.496, 11.458]` |
| 100 | 2,880 | 1,600 | 1,280 | 117,602 | 92,464 | `4.77e-7` | `2.86e-6` | `[-2.878, 10.678]` |
| 150 | 1,792 | 1,025 | 767 | 77,882 | 58,011 | `2.38e-7` | `1.43e-6` | `[-4.716, 6.198]` |

这些误差处于 float32 运算和 JSON 序列化的正常舍入范围，说明正式在线 batch 中没有发现：

- combined 分量漏加或重复加；
- branch occurrence 未获得 advantage；
- padding token 污染 occurrence mean；
- trace 中记录的 advantage 与实际 token tensor 明显不一致。

step 50 出现最小 advantage `-21.496`，说明某些大 anchor/task group 的 macro+local 组合有较强负异常值。训练日志未显示相应的数值崩溃，且 gradient clipping 为 1.0，但以后应增加 advantage 分位数和按 root/branch 分解的监控。

### 12.1 Trace validator 的存储读取问题

代表 step 50、100、150 使用现有 `validate_trace.py` 重跑均为：

```text
ok=true
errors=0
```

step 1 仍能复现正式作业结束时类似的 JSON 读取问题：Python `Path.read_text()` 对位于 rwth2089 实验文件系统上的几个超大 JSONL 偶发/重复得到短内容，从而报 `Unterminated string`；与此同时 `stat`/`wc` 显示完整文件大小，`jq` 可以完整解析原文件全部记录。

因此最新证据更像是超大单行 JSONL 在当前存储/读取路径上的 partial-read 或 reader robustness 问题，而不是 advantage 数学计算错误。它影响严格 artifact 验收可靠性，但不反推在线 PPO 训练计算错误。建议另行把 validator 改为逐行流式读取并带短读重试/文件大小复核。

---

## 13. 风险与改进点

### 13.1 P0：未发现阻塞性 advantage/loss bug

没有发现需要否定当前 checkpoint 或立即重跑 150-step 的 P0 问题。

### 13.2 P1：token-mean 带来的长度权重

这是最值得做消融的 loss 语义。

当前每个 occurrence 的 advantage 虽然是 scalar，但 PPO 按全部 response token 求平均。因此：

\[
\text{occurrence effective weight}
\propto
\text{response token count}.
\]

BACE 会改变 root/branch 的长度分布，所以即使使用与 GiGPO 相同的 `token-mean`，两种方法的有效 sample weighting 也可能不同。

这不是代码 bug，因为它与原版 veRL/GiGPO 配置一致；但若目标是“每个 action occurrence 等权”，更自然的候选是：

```text
seq-mean-token-mean
```

不建议直接改正式主结果，应作为控制消融。

### 13.3 P1：occurrence-weighted macro 与 BACE 树结构耦合

更长的 trajectory、重复 branch origin 和更长 suffix 会在 macro baseline 中占更多票。BACE 越偏向 branch，macro 统计分布越受 tree topology 影响。

当前保留它是为了与 GiGPO implementation 对齐；后续可比较：

1. occurrence-weighted（当前主线）；
2. trajectory/leaf-uniform；
3. lineage-balanced。

### 13.4 P1：相关 lineage 的重复权重

一个高价值 anchor 若生成多个 branches，被复制的 origin response 会多次进入 PPO。这正是获取反事实 continuation 信号的方式，但这些 occurrences 并不统计独立。

当前 loss 不限制同一 root/anchor 派生样本的总权重。后期 branch 比例上升时，local signal 可能压过独立 root 的 macro coverage。

建议先记录而不是立即修改：

- root/branch 各自的 token 数和 loss 占比；
- 每个 lineage 的 occurrence/token 总权重；
- macro/local 的均值、std、P1/P50/P99；
- root/branch 分解后的 `|A|` 与 clipped loss；
- 每个 anchor 的有效样本数和 effective sample size。

### 13.5 P1：invalid penalty 的双通道影响

无效动作的 `-0.1` 同时进入：

- macro score；
- 当前 occurrence 的 local step return。

最终不是固定 `-0.2`，而是分别经过 task 和 anchor normalization。该设计与原版 GiGPO 一致，并且能同时惩罚 episode-level 和 action-local 质量；但在 anchor 方差很小时，local 标准化可能放大很小的 raw penalty。

建议监控 invalid occurrence 的 macro/local 分布，并把以下作为消融，而不是直接改主线：

- macro+local 双通道（当前）；
- 只进入 local；
- penalty 进入 reward-to-go 后重新 discount。

### 13.6 P2：singleton macro 边界

macro group 只有一个统计样本时，当前上游实现返回接近 raw score，而不是 0。正式 B=8 不触发，但未来 B=1 smoke 或异常过滤后可能触发。

建议新增单测并决定明确语义：

- 若坚持相对 advantage，singleton 应为 0；
- 若要保留 raw REINFORCE 信号，应在文档和函数名中明确。

### 13.7 P2：`returns` 命名容易误解

`returns=advantages` 对无 critic 的 GRPO/GiGPO 风格训练是常见接口兼容做法，但日志中的 `critic/returns/*` 并不是环境 discounted return。建议将可观测指标改名或补充 `bace/step_discounted_return/*`，避免分析训练曲线时混淆。

### 13.8 P2：trace validator 应流式读取

现有 validator 对整个 JSONL 使用 `read_text().splitlines()`，而部分单行文件达到数十 MiB。建议：

- 使用 `for line in file` 流式解析；
- 对 apparent EOF/short read 做 stat-size 复核与有限重试；
- artifact writer 使用 atomic finalize marker；
- 避免把完整 roots/topology 序列化成一个超大单行对象。

这属于归档完整性修复，不应与在线 advantage 公式修改捆绑。

---

## 14. 推荐的后续验证矩阵

为了不破坏已经完成的主实验控制变量，建议按以下顺序做：

### 14.1 只增加诊断，不改训练语义

1. 每 step 记录 root/branch 的 occurrence 数和 token 数；
2. 记录 macro/local/combined 的分位数和非有限值计数；
3. 记录 root/branch 各自对 unclipped/clipped PPO loss 的贡献；
4. 记录按 lineage 聚合后的有效权重；
5. 对 `|A|>10` 的 occurrence 保存 task、anchor group size、macro/local 分解。

### 14.2 小规模 optimizer 消融

| 维度 | 当前 | 建议对照 |
|---|---|---|
| loss aggregation | `token-mean` | `seq-mean-token-mean` |
| macro weighting | occurrence | leaf-uniform |
| local credit | occurrence | action-mean |
| step weight | 1.0 | 0.25、0.5 |
| lineage weight | 无 | `1/sqrt(n_lineage)` 等 |
| invalid penalty | macro+local | local-only |

这些消融一次只改一个维度，否则无法判断收益来自 BACE acquisition 还是 optimizer weighting。

### 14.3 建议补充的测试

现有 advantage 单测已经覆盖主要公式。还应补：

- singleton macro 明确语义；
- empty response 在 invalid penalty 前显式拒绝；
- BACE advantage + PPO total loss 的端到端手算 fixture；
- 不同 response 长度下 `token-mean` 与 `seq-mean-token-mean` 的预期差异；
- root/branch duplicated-origin 的 lineage 权重守恒测试；
- invalid penalty 是否应向更早 step discount 的明确契约测试；
- trace streaming reader 的超大行与短读恢复测试。

---

## 15. 最终判断

### 可以确认

1. 当前 BACE 在线训练实际使用的是 GiGPO-compatible macro + local occurrence advantage；
2. `gamma=0.95` 确实生效，但生效在 local discounted-return 路径，不作用于 terminal macro score；
3. invalid penalty 正确进入当前设计的两条信用链；
4. combined advantage 正确复制到有效 response tokens；
5. PPO clip、dual clip、entropy 和 KL 的符号、mask 与调用位置正确；
6. 正式 trace 的代表 step 数值满足 `combined=macro+local`；
7. 现有 150-step checkpoint 不需要因为 advantage/loss 计算而判废。

### 必须保留的解释边界

1. “与 GiGPO 使用相同 estimator”不等于 BACE 与 GiGPO 拥有相同的 effective sample weights；
2. 当前不是 leaf-uniform，也不是 lineage-uniform；
3. response 更长的 occurrence 权重更大；
4. 多 branch 的共享 origin 会被多次训练；
5. trace JSON 读取问题影响严格归档验收，但不是在线 PPO 数学错误。

### 建议

当前正式主结果应保持现有 advantage/loss 口径，以维持与 GiGPO 的控制变量一致性；下一步优先增加 root/branch、lineage、token-length 和 advantage outlier 的诊断，再用小规模消融判断是否需要引入 sequence-balanced 或 lineage-balanced loss。不要在没有消融证据的情况下直接改掉 occurrence credit 或 token-mean，然后把结果仍称作同一个主配置。
