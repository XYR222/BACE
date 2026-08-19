# BACE 与 GiGPO 无 Branch 优势等价性与当前差异说明

日期：2026-08-17  
范围：ALFWorld、GiGPO、BACE legacy、BACE BatchERV Exact  
目的：明确轨迹级 GRPO 形优势与 anchor 步骤级优势的定义、当前实现差异，以及“无 branch 时 BACE 应完全退化为 GiGPO”所要求的工程契约。

---

## 1. 核心结论

### 1.1 术语上并不存在四个独立优势项

GiGPO 的 `macro advantage` 就是轨迹/episode 层面的全局优势；BACE 将同一层级称为 `leaf advantage` 或 `global advantage`。

GiGPO 的 `micro advantage` 就是 anchor/step 层面的局部优势；BACE 将同一层级称为 `local advantage`。

| 语义层级 | GiGPO 名称 | BACE 名称 | 最终作用 |
|---|---|---|---|
| 轨迹终局层 | macro / episode advantage | global / leaf advantage | 判断完整轨迹相对同组轨迹的好坏 |
| anchor 步骤层 | micro / step advantage | local advantage | 判断相同状态下某个 occurrence 的相对好坏 |
| 合成层 | `macro + omega * micro` | `leaf + omega * local` | 广播到 response 的 trainable tokens |

因此，本文比较的是两个对应层级：

```text
BACE leaf-global <-> GiGPO episode-macro
BACE anchor-local <-> GiGPO step-micro
```

### 1.2 用户预期应成为正式 reduction invariant

对于任一任务实例 `g`，若没有 branch：

```text
Q_g = 0
```

且 BACE 与 GiGPO 使用完全相同的 natural-root rollout batch，则应要求：

```text
A_BACE_macro(i) == A_GiGPO_macro(i)
A_BACE_micro(i) == A_GiGPO_micro(i)
A_BACE_final(i, token) == A_GiGPO_final(i, token)
```

这个预期是合理的。BACE 的新增贡献应来自 branch topology、Replay 和 ERV acquisition；当这些新增机制没有实际产生 branch 时，BACE 不应悄悄改变基础 GiGPO 优势。

### 1.3 当前代码不满足完全一致

当前实现只做到“结构对应”，没有做到“逐 occurrence、逐 token 数值一致”。主要差异为：

1. BACE macro 按 unique terminal leaves 统计；GiGPO 当前代码默认按 physical action occurrences 统计。
2. BACE macro 使用原始 `episode_rewards`；GiGPO macro 使用包含 Invalid Action penalty 的 `token_level_rewards`。
3. BACE 使用总体标准差；GiGPO 使用 PyTorch 默认的样本标准差。
4. BACE micro 没有真正接收 `algorithm.gigpo.mode`，始终执行 mean-std normalization。
5. BatchERV Exact 只改变采集与 topology，仍使用同一个 BACE advantage calculator，因此不会自动消除上述差异。

---

## 2. 比较对象和数据单位

必须先区分三个统计单位，否则很容易把“同样是 GRPO 形公式”误解为“结果一定相同”。

### 2.1 Terminal leaf

一条从任务初始状态到终局的完整结果：

```text
leaf = complete trajectory outcome
```

无 branch 时，一条 natural root 对应一条 leaf。

有 branch 时：

```text
leaf set = natural-root leaves + branch leaves
```

### 2.2 Trajectory

无 branch 时，trajectory 与 natural leaf 一一对应。

有 branch 时，一条 branch leaf 在统计意义上是一条完整 trajectory，但训练 buffer 不会再次训练 anchor 之前的 Replay prefix。

### 2.3 Action occurrence

一条 trajectory 通常包含多个模型 response：

```text
trajectory tau_1 -> occurrence 1, occurrence 2, ..., occurrence T_1
trajectory tau_2 -> occurrence 1, occurrence 2, ..., occurrence T_2
```

每个 occurrence 对应一次模型生成的 reasoning + action response，也是 PPO loss 的基本样本。

因此：

```text
2 trajectories != 2 occurrences
```

如果两条轨迹长度分别为 1 和 3，则会产生 4 个 occurrences。

---

## 3. GiGPO 轨迹级 macro advantage

## 3.1 概念公式

GiGPO 的轨迹层保持 GRPO 式组内相对优势。对同一任务的轨迹回报：

$$
R_{g,1}, R_{g,2}, \ldots, R_{g,K},
$$

标准 mean-std 形式为：

$$
\mu_g=\frac{1}{K}\sum_{k=1}^{K}R_{g,k},
$$

$$
\sigma_g=\operatorname{Std}(R_{g,1},\ldots,R_{g,K}),
$$

$$
A_{g,k}^{\mathrm{macro}}
=
\frac{R_{g,k}-\mu_g}{\sigma_g+\epsilon}.
$$

该轨迹级优势随后广播给轨迹中的各个 action occurrences 及其有效 response tokens。

如果使用 `mean_norm`，则不除以标准差：

$$
A_{g,k}^{\mathrm{macro}}=R_{g,k}-\mu_g.
$$

## 3.2 当前 GiGPO 代码实际行为

实际入口为：

```text
gigpo/core_gigpo.py::compute_gigpo_outcome_advantage
```

其 macro 输入为：

```python
episode_advantages = episode_norm_reward(
    token_level_rewards,
    response_mask,
    index,
    traj_index,
    ...,
)
```

`episode_norm_reward()` 先对每条 physical batch row 求：

```python
scores = token_level_rewards.sum(dim=-1)
```

然后默认使用：

```python
compute_mean_std_cross_steps=True
```

在这个默认值下，`seen_pairs` 不会排除同一 `(task, trajectory)` 的后续 occurrences。因此 macro 的均值和标准差实际基于：

$$
\{r_i: i\in\text{physical action occurrences of task }g\},
$$

而不是严格基于：

$$
\{R_{g,k}: k\in\text{unique trajectories of task }g\}.
$$

这带来两个效果：

1. 较长 trajectory 的 terminal score 会在 macro 统计中出现更多次。
2. 每个 occurrence 自己的 Invalid Action penalty 也会进入 score。

所以当前 GiGPO macro 更准确的描述是：

```text
occurrence-weighted episode macro
```

而不是严格的 uniform-trajectory GRPO。

## 3.3 GiGPO macro 的 reward 输入

训练顺序为：

```text
episode reward -> token_level_scores
Invalid Action penalty -> token_level_scores
token_level_rewards = token_level_scores
GiGPO macro normalization
```

在当前配置 `algorithm.use_kl_in_reward=False` 下：

```python
token_level_rewards = token_level_scores
```

因此 GiGPO macro score 可以写成简化形式：

$$
s_i
=
R_{\tau(i)}
-
\lambda_{\mathrm{invalid}}I_i^{\mathrm{format-invalid}}.
$$

其中同一成功 trajectory 的多个 occurrences 都带有相同 terminal reward，但每个 occurrence 的格式 invalid 指示可以不同。

---

## 4. BACE 轨迹级 leaf-global advantage

## 4.1 方案公式

BACE 最终方案定义同一任务的 terminal leaf set：

$$
\mathcal L_g
=
\mathcal L_g^{\mathrm{root}}
\cup
\mathcal L_g^{\mathrm{branch}}.
$$

在该集合上计算：

$$
\mu_g^E
=
\frac1{|\mathcal L_g|}
\sum_{\ell\in\mathcal L_g}R_{g,\ell},
$$

$$
\sigma_g^E
=
\operatorname{Std}
\left(
\{R_{g,\ell}\}_{\ell\in\mathcal L_g}
\right),
$$

$$
A_{g,\ell}^{E}
=
\frac{R_{g,\ell}-\mu_g^E}
{\sigma_g^E+\epsilon}.
$$

这仍然是 GRPO 形的组内相对优势，但采样单位被明确固定为：

```text
one terminal leaf, one vote
```

## 4.2 当前 BACE 代码实际行为

当前实现位于：

```text
recipe/bace_gigpo/advantage.py::compute_bace_gigpo_advantage
```

代码先按以下 key 去重：

```python
key = (task_id, leaf_id)
```

再对 unique leaf rewards 做 normalization：

```python
rewards = [leaf_scores[key] for key in keys]
normalized = _normalized(rewards, epsilon)
```

最后将同一个 leaf advantage 广播回属于该 leaf 的所有 trainable occurrences。

这种 leaf-uniform 设计具有明确动机：

1. branch copied origin 与 suffix 不应因为 occurrence 数量更多而重复改变全局均值；
2. 长 trajectory 不应仅因为包含更多 response 而获得更高全局统计权重；
3. 每个 terminal leaf 在 root/branch 混合树中保持相同的 global vote。

## 4.3 BACE global 的 reward 输入

BACE calculator 接收：

```python
leaf_rewards=data.non_tensor_batch["episode_rewards"]
```

它不使用 `token_level_rewards` 计算 global term。

因此 Invalid Action penalty 的当前流向是：

| 优势分量 | 是否包含格式 Invalid Action penalty |
|---|---:|
| GiGPO macro | 是 |
| GiGPO micro | 是 |
| BACE leaf-global | 否 |
| BACE anchor-local | 是 |

这也是无 branch 时两者无法完全相同的原因之一。

---

## 5. GiGPO anchor 级 micro advantage

## 5.1 Anchor group

GiGPO 在相同 task 内按 observation 建立 step group。

关闭 similarity 时，group key 等价于：

```text
(task_id, exact observation)
```

对应代码：

```python
clusters[to_hashable(obs)].append(index)
```

启用 similarity 时，GiGPO 还可以按文本相似度合并 observation；当前 ALFWorld 主脚本默认：

```text
algorithm.gigpo.enable_similarity=False
```

所以当前对照使用 exact observation grouping。

## 5.2 Step return-to-go

GiGPO 与 BACE 都先调用：

```python
compute_step_discounted_returns(batch, gamma)
```

在当前脚本中：

```text
gamma = 0.95
```

对 occurrence `i`：

$$
G_i
=
\sum_{t=t_i}^{T_i}\gamma^{t-t_i}r_t.
$$

Invalid Action penalty 随后也会从对应 occurrence 的 `step_rewards` 中扣除。

## 5.3 GiGPO micro normalization

对于同一个 anchor group `z`：

$$
\mu_{g,z}^S
=
\frac1{|\mathcal I_g(z)|}
\sum_{i\in\mathcal I_g(z)}G_i,
$$

$$
A_i^{\mathrm{micro}}
=
\frac{G_i-\mu_{g,z}^S}
{\sigma_{g,z}^S+\epsilon}.
$$

当前 GiGPO 代码使用：

```python
torch.std(...)
```

即 PyTorch 默认样本标准差。

若 group 只有一个 occurrence，GiGPO 将均值设为该 occurrence 自身，所以 micro advantage 为 0。

---

## 6. BACE anchor 级 local advantage

## 6.1 Group membership

BACE 主版本 `local_credit_mode=occurrence` 使用：

```python
rows_by_anchor[(task_id, to_hashable(observation))]
```

当满足以下条件时，group key 与 GiGPO 一致：

```text
GiGPO enable_similarity=False
BACE local_credit_mode=occurrence
相同 task_id
相同 exact anchor observation
相同 physical training rows
```

无 branch 且未发生其他过滤时，BACE 与 GiGPO 应具有相同的 anchor group membership。

有 branch 时，BACE group 还包含：

- natural-root occurrences；
- copied branch-origin occurrence；
- branch suffix occurrences；
- suffix 与其他路径碰撞到相同 observation 的 occurrences。

Replay prefix 不进入 trainable local group。

## 6.2 当前 normalization 差异

BACE 通过 `_normalized()` 计算：

```python
std = values.std(unbiased=False)
```

即总体标准差。

所以即便 group membership 和 `step_rewards` 完全相同，BACE 与 GiGPO 仍有尺度差异：

```text
BACE: population std
GiGPO: sample std
```

对 group size 为 `n` 的非退化 group，两者尺度比例约为：

$$
\frac{|A_{\mathrm{BACE}}|}{|A_{\mathrm{GiGPO}}|}
=
\sqrt{\frac{n}{n-1}}.
$$

group 越小，差异越明显。

## 6.3 `algorithm.gigpo.mode` 没有控制 BACE

GiGPO 支持：

```text
mean_norm
mean_std_norm
```

但 trainer 的 BACE 分支调用 `compute_bace_gigpo_advantage()` 时，没有传入 `gigpo_mode`。

BACE `_normalized()` 始终除以标准差。因此：

```text
algorithm.gigpo.mode=mean_std_norm -> BACE 恰好选择同类形式，但标准差定义仍不同
algorithm.gigpo.mode=mean_norm     -> GiGPO 不除 std，BACE 仍除 std
```

这说明当前配置项在 BACE 路径上不是一个真正生效的共享控制参数。

---

## 7. 当前差异总表

| 比较项 | GiGPO 当前实现 | BACE 当前实现 | 无 branch 是否相同 |
|---|---|---|---:|
| 轨迹层名称 | macro / episode | global / leaf | 语义对应 |
| macro 统计单位 | physical occurrences | unique terminal leaves | 否 |
| 长 trajectory 权重 | occurrence 更多，统计权重更高 | 每个 leaf 一票 | 否 |
| macro reward | `token_level_rewards` | `episode_rewards` | 否 |
| macro Invalid penalty | 包含 | 不包含 | 否 |
| macro normalization mode | `mean_norm` 或 `mean_std_norm` | 固定 mean-std | 不一定 |
| macro std | 样本标准差 | 总体标准差 | 否 |
| anchor group key | task + exact obs（主配置） | task + exact obs | 是 |
| step return-to-go | `step_rewards`, gamma=0.95 | 同一 `step_rewards` | 是 |
| micro Invalid penalty | 包含 | 包含 | 是 |
| micro std | 样本标准差 | 总体标准差 | 否 |
| singleton micro | 0 | 0 | 是 |
| 最终组合 | macro + omega * micro | leaf + omega * local | 结构相同，数值不同 |
| response mask | `response_mask` | `response_mask` | 是 |

---

## 8. 例子一：无 Branch、无 Invalid，但轨迹长度不同

这是隔离 occurrence weighting 的最小例子。

同一 task 有两条 natural trajectories，没有 branch：

| Trajectory | 长度 | Terminal reward | Occurrences |
|---|---:|---:|---|
| `tau_fail` | 1 | 0 | `f1` |
| `tau_success` | 3 | 1 | `s1, s2, s3` |

## 8.1 BACE leaf-global

按 unique leaves：

$$
[R_{\mathrm{fail}},R_{\mathrm{success}}]=[0,1].
$$

总体均值与总体标准差：

$$
\mu=0.5,
\qquad
\sigma_{\mathrm{pop}}=0.5.
$$

因此：

$$
A_{\mathrm{fail}}=-1,
\qquad
A_{\mathrm{success}}=1.
$$

广播回 occurrences：

```text
f1 -> -1.0
s1 -> +1.0
s2 -> +1.0
s3 -> +1.0
```

## 8.2 当前 GiGPO macro

默认 cross-step 统计看到的 physical scores 为：

$$
[0,1,1,1].
$$

样本均值和样本标准差：

$$
\mu=0.75,
\qquad
\sigma_{\mathrm{sample}}=0.5.
$$

因此：

```text
f1 -> (0 - 0.75) / 0.5 = -1.5
s1 -> (1 - 0.75) / 0.5 = +0.5
s2 -> +0.5
s3 -> +0.5
```

实际调用当前两个 calculator 的结果正是：

```text
BACE global: [-1.0, 1.0, 1.0, 1.0]
GiGPO macro: [-1.5, 0.5, 0.5, 0.5]
```

这个例子完全没有 branch，也没有 Invalid Action，但两者已经不一致。

原因不是 GRPO 公式形状不同，而是：

```text
统计样本不同 + 标准差定义不同
```

## 8.3 按用户预期的正确退化

无 branch 时，BACE 应复用与 GiGPO 完全相同的 macro primitive、reward 和 normalization。若基准定义选择 current GiGPO implementation，则 BACE 应同样输出：

```text
[-1.5, 0.5, 0.5, 0.5]
```

若我们认为 GiGPO 本身应使用严格 uniform-trajectory GRPO，则应先修正或明确 GiGPO baseline，再让两边共同输出同一套 leaf-uniform 值。不能让两边分别采用不同定义后仍声称“无 branch 完全一致”。

---

## 9. 例子二：无 Branch、全部失败，但出现格式 Invalid

同一 task 有两条自然轨迹，均失败：

```text
episode rewards = [0, 0]
```

其中一条 occurrence 格式合法，另一条格式 Invalid，penalty 为 `0.1`。

## 9.1 BACE global

BACE global 只读取：

```text
episode_rewards = [0, 0]
```

方差为 0：

```text
BACE global = [0, 0]
```

## 9.2 GiGPO macro

GiGPO macro 读取 penalty 后的 scores：

```text
[0, -0.1]
```

使用样本标准差：

$$
\mu=-0.05,
\qquad
\sigma_{\mathrm{sample}}\approx0.07071.
$$

所以：

```text
GiGPO macro approximately [+0.7071, -0.7071]
```

这解释了为什么训练早期 terminal success 全部为 0 时，GiGPO 仍可能有 macro signal，而 BACE leaf-global 全部为 0。

该差异同样与 branch 无关。

---

## 10. 例子三：无 Branch、相同 Anchor 的 micro/local

假设两个 natural trajectories 都访问相同 anchor `z`，对应 return-to-go 为：

$$
G=[0,1].
$$

## 10.1 BACE local

总体标准差：

$$
\sigma_{\mathrm{pop}}=0.5.
$$

因此：

```text
BACE local = [-1.0, +1.0]
```

## 10.2 GiGPO micro

样本标准差：

$$
\sigma_{\mathrm{sample}}=\sqrt{0.5}\approx0.7071.
$$

因此：

```text
GiGPO micro = [-0.7071, +0.7071]
```

二者具有相同的：

- anchor group；
- 正负方向；
- 零/非零结构；
- return-to-go。

但数值尺度不同，所以最终 PPO gradient 仍不完全相同。

若使用 `mean_norm`，GiGPO 应输出：

```text
[-0.5, +0.5]
```

当前 BACE 仍输出 `[-1,+1]`，因为它没有接收该 mode。

---

## 11. 有 Branch 时允许出现的 BACE 扩展

“无 branch 完全一致”不意味着有 branch 时仍必须逐项等同于普通 GiGPO。

当 `Q_g > 0` 时，BACE 引入了普通 GiGPO 不存在的数据：

1. branch terminal leaves；
2. copied branch-origin occurrences；
3. fresh suffix occurrences；
4. Replay prefix mask；
5. branch lineage；
6. ERV acquisition-induced sampling distribution。

因此有 branch 时，BACE 可以定义 tree-aware 扩展：

$$
\mathcal L_g
=
\mathcal L_g^{\mathrm{root}}
\cup
\mathcal L_g^{\mathrm{branch}}.
$$

合理的不变量是：

```text
每个 terminal leaf 在 macro/global 统计中权重明确且可审计；
Replay prefix 不重复进入 PPO；
copied origin 使用 branch leaf return；
local/micro 在最终 trainable occurrence map 上计算；
当 Q_g -> 0 时，严格退化回 GiGPO。
```

最后一条正是当前缺失的 reduction invariant。

---

## 12. BatchERV Exact 的实际情况

## 12.1 Exact 只改变采集，不改变 advantage calculator

Exact 脚本设置：

```text
algorithm.adv_estimator=bace_gigpo
algorithm.bace.variant=batch_erv_exact
algorithm.bace.acquisition=batch_erv_exact
```

`variant=batch_erv_exact` 只选择：

- `ExactBatchTopologyPlanner`；
- staged packed root generation；
- `ExactBatchErvCoordinator`；
- frozen joint branch plan；
- strict Replay realization。

进入 advantage 阶段后，仍调用：

```text
compute_bace_gigpo_advantage()
```

该函数没有 `variant` 参数，也不知道当前是 legacy 还是 BatchERV Exact。

因此：

```text
BatchERV Exact 不会自动获得 GiGPO-compatible macro/micro。
```

## 12.2 Exact 第一步当前不是 root-only

Exact 使用 lagged family history，在生成当前 roots 前确定初始 branch quota。

空 history 下：

```text
prior = Beta(1,1)
competence threshold = 0.5
readiness = P(p > 0.5) = 0.5
flexible budget = 8 - 2 = 6
planned branches = round(6 * 0.5) = 3
```

所以 step 1 默认 topology 为：

```text
5 roots + 3 branches per task
```

16 个 task 即：

```text
80 roots + 48 planned branches
```

现有 Exact smoke 的 manifest 与 topology artifact 已确认该结果。

这说明当前 Exact 默认配置本身也不满足“训练初期几乎无 branch”的预期。

## 12.3 现有 Exact smoke 未完成端到端验证

现有 artifact 包含：

- `manifest.json`；
- roots；
- leaves；
- anchors；
- capacity checks；
- topology。

但缺少：

- `branches.jsonl`；
- `trainable_occurrences.jsonl`；
- `summary.json`。

因此它没有提供一次完整 branch assembly、advantage calculation 和 PPO update 的运行时证据。

---

## 13. 应正式采用的无 Branch 等价性契约

建议将以下契约写入实现规范和自动化测试。

### 13.1 输入条件

对同一个 frozen policy 和同一 natural-root batch：

```text
branch_count = 0
same task IDs
same trajectory IDs
same occurrence rows and order
same response masks
same episode rewards
same invalid-action metadata
same step rewards
same gamma
same omega
same normalization mode
same anchor grouping mode
```

### 13.2 输出条件

逐 occurrence 要求：

$$
A_{i,\mathrm{BACE}}^{\mathrm{macro}}
=
A_{i,\mathrm{GiGPO}}^{\mathrm{macro}},
$$

$$
A_{i,\mathrm{BACE}}^{\mathrm{micro}}
=
A_{i,\mathrm{GiGPO}}^{\mathrm{micro}}.
$$

逐 token 要求：

$$
A_{i,t,\mathrm{BACE}}^{\mathrm{final}}
=
A_{i,t,\mathrm{GiGPO}}^{\mathrm{final}}
$$

对所有 `response_mask[i,t] = 1` 的 token 成立。

不仅要比较符号和零比例，还应比较：

```text
max_abs_diff <= 1e-6
mean_abs_diff <= 1e-7
identical zero mask
identical positive/negative mask
```

### 13.3 不能只测试等长轨迹

至少覆盖：

1. 不同 trajectory lengths；
2. 全失败；
3. 成败混合；
4. 有格式 Invalid penalty；
5. anchor singleton；
6. anchor group size 2；
7. 多个 repeated anchors；
8. `mean_norm`；
9. `mean_std_norm`。

不同长度用例最重要，因为它能直接发现 occurrence weighting 与 leaf weighting 的差异。

---

## 14. 后续实现决策

要实现用户要求，必须先确定“完全一致”以哪个 GiGPO 定义为准。

### 14.1 方案 A：以当前成功运行的 GiGPO 代码为准

无 branch 时直接复用：

```text
core_gigpo.compute_gigpo_outcome_advantage()
```

优点：

- 与现有 GiGPO baseline 真正逐 token 可比；
- 最容易建立 reduction test；
- 不会因为重复实现出现 std/mode 漂移。

代价：

- 接受当前 GiGPO occurrence-weighted macro；
- 接受 Invalid Action penalty 进入 macro；
- 有 branch 时仍需定义如何切换到 tree-aware leaf macro。

### 14.2 方案 B：以严格 uniform-trajectory GRPO 为准

先统一修正 GiGPO 和 BACE，使 macro 都按 unique trajectory/leaf 计算，再进行对照实验。

优点：

- 数学语义更接近标准 GRPO；
- 避免 trajectory length 改变 macro group 权重；
- 与 BACE leaf-uniform tree extension 更自然。

代价：

- 新 GiGPO baseline 不再是此前成功脚本的原始行为；
- 必须重新训练 GiGPO 对照；
- 旧实验不能直接作为严格 baseline。

### 14.3 推荐顺序

为了先恢复可解释的实验对照，推荐：

1. 将方案 A 作为 `gigpo_compatible` 主兼容模式。
2. 保留当前 `leaf_uniform` 逻辑作为可选模式，不删除旧逻辑。
3. 添加无 branch reduction test，要求逐 token 完全一致。
4. 再将 uniform-trajectory 版本作为独立、显式命名的实验变体。
5. BatchERV Exact 先增加 root-only warm start 或 branch warmup，再做 equivalence smoke。

这样可以分别回答两个问题：

```text
BACE 的 topology/acquisition 是否有效？
leaf-uniform macro 是否优于当前 GiGPO macro？
```

而不会把二者混在一次实验中。

---

## 15. 对当前实验结论的影响

### 可以继续成立

1. legacy BACE 训练中存在较高 zero-advantage 比例。
2. 该现象不能只由 branch 数量解释。
3. BACE leaf-global 与当前 GiGPO macro 的统计口径不同。
4. BatchERV Exact 的 topology 结果不能从 legacy run 直接推断。

### 需要重新表述

不能简单说：

```text
BACE 没有遵循 GRPO，GiGPO 遵循 GRPO。
```

更准确的是：

```text
两者都使用 GRPO 形的组内相对优势，
但统计单位、reward 输入、标准差和 mode 控制不同，
所以无 branch 时并不数值等价。
```

### 需要新增验证

在继续长程训练前，至少需要：

```text
root-only GiGPO vs BACE exact-equivalence unit test
root-only 1-step artifact comparison
BatchERV Exact complete one-update smoke
macro/micro/final advantage separate trace
```

---

## 16. 代码与文档证据

- GiGPO macro/micro calculator：`verl-agent-src/gigpo/core_gigpo.py`
- BACE leaf/local calculator：`verl-agent-src/recipe/bace_gigpo/advantage.py`
- GiGPO/BACE shared step returns：`verl-agent-src/verl/trainer/ppo/ray_trainer.py`
- Invalid Action penalty 注入：`verl-agent-src/verl/trainer/ppo/ray_trainer.py`
- Episode reward placement：`verl-agent-src/agent_system/reward_manager/episode.py`
- BatchERV Exact topology：`verl-agent-src/recipe/bace_gigpo/topology.py`
- BatchERV Exact acquisition：`verl-agent-src/recipe/bace_gigpo/coordinator.py`
- BatchERV Exact collector routing：`verl-agent-src/recipe/bace_gigpo/rollout_collector.py`
- BatchERV Exact 训练脚本：`verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_npu_8card_150epoch_batch_erv_exact.sh`
- BACE 最终方案优势定义：`BACE-work/BACE完整方案_最终版_2026-08-05(1).md`
- 优势与 branch lineage 规范：`BACE-work/BACE优势_Loss与Branch分布更新实现规范.md`

---

## 17. 最终判断

用户提出的预期应正式成立：

> 当一个任务没有产生 branch，且输入 natural-root batch 与 GiGPO 相同，BACE 的 macro、micro 和最终 token advantage 应与 GiGPO 完全一致。

当前代码没有满足该要求。

当前 BACE 的 `leaf-uniform global + occurrence local` 是一个有明确动机的 tree-aware estimator，但它从 root-only 情况开始就已经改变了当前 GiGPO baseline 的 advantage。BatchERV Exact 只替换 topology/acquisition，并未解决这一点。

因此，下一步不是直接继续比较长程曲线，而是先把无 branch 等价性写成测试并实现一个显式 `gigpo_compatible` 模式，同时保留当前 `leaf_uniform` 模式用于后续消融。
