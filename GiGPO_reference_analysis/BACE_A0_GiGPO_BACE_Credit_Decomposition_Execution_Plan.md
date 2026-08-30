# A0：GiGPO / BACE 训练阶段信用结构分解分析
## ——分析 trajectory-level advantage 与 step-level/local advantage 在训练过程中的实际占比与作用

**适用项目**：`XYR222/BACE`  
**当前事实基线**：`handoff-refresh-20260828` 分支；历史 150-step GiGPO/BACE run 用于诊断，不等同于当前 HEAD 的同版本长跑。  
**任务类型**：离线分析优先；第一阶段原则上不需要 GPU。  
**优先级**：最高。  
**目的**：在修改 `step_advantage_w`、`local_credit_mode`、branch weighting 等训练机制前，先定量回答：

> 原版 GiGPO 在训练早期、中期、后期分别有多少训练信号来自 trajectory-level credit，多少来自 repeated-state step/local credit？BACE 是否随着 branch 比例增加，显著改变了这种 global/local credit balance？

---

# 1. 为什么必须先做 A0

GiGPO 和当前 BACE 的最终优势形式相同：

\[
A_i = A_i^{\mathrm{macro}} + \omega A_i^{\mathrm{local}},
\]

正式配置均为：

\[
\omega=1.
\]

但“参数相同”不意味着“实际信用占比相同”。

实际 local signal 的训练影响还取决于：

- repeated-state group 出现频率；
- 每个 group 的大小；
- local advantage 的绝对幅度；
- macro/local 的符号关系；
- 每个 occurrence 的 response token 数；
- BACE 中 root / branch-origin / branch-suffix 的组成；
- branch multiplicity 对同一 lineage 的重复训练权重。

因此要区分：

\[
\boxed{\text{配置权重 }\omega}
\]

与：

\[
\boxed{\text{真实 local credit share}}.
\]

A0 的目标不是证明某个改法更好，而是先建立一张可信的 **credit curriculum**：

```text
training step
   ↓
macro credit share
local credit coverage
local magnitude share
macro/local agreement
BACE source composition
```

---

# 2. 当前代码语义必须严格复现

当前 GiGPO / BACE 的分析应以正式训练语义为准，不自行改成 leaf-uniform 或 action-mean。

## 2.1 Macro advantage

当前 `compute_mean_std_cross_steps=true`。

对 task group \(g\) 内的 physical occurrences \(i\)：

\[
S_i
=
R_{\tau(i)}
-
0.1\cdot \mathbf 1(\text{invalid}_i),
\]

其中 \(R_{\tau(i)}\) 是该 occurrence 所属 trajectory 的 terminal episode reward。

然后按同一 task 的 **physical occurrence** 计算：

\[
\mu_g^E
=
\frac{1}{|\mathcal I_g|}
\sum_{i\in\mathcal I_g}S_i,
\]

\[
\sigma_g^E
=
\operatorname{Std}_{\mathrm{sample}}
\{S_i:i\in\mathcal I_g\},
\]

正式 `mean_std_norm`：

\[
\boxed{
A_i^{\mathrm{macro}}
=
\frac{S_i-\mu_g^E}
{\sigma_g^E+10^{-6}}
}
\]

若组内方差数值上为 0，则按当前稳定实现输出 0。

---

## 2.2 Local / step advantage

先按每条 `traj_uid` 的真实环境 reward 反向计算：

\[
G_t
=
r_t+\gamma G_{t+1},
\qquad
\gamma=0.95.
\]

当前 occurrence 无效时：

\[
\widetilde G_t
=
G_t
-
0.1\cdot\mathbf 1(\text{invalid}_t).
\]

注意：invalid penalty 是当前 occurrence 的点式惩罚，不重新向更早 step 折扣传播。

当前 exact-observation local group：

\[
h(i)
=
(\text{task\_id}_i,\text{exact anchor\_obs}_i).
\]

对组 \(h\)：

\[
\boxed{
A_i^{\mathrm{local}}
=
\frac{\widetilde G_i-\mu_h}
{\sigma_h+10^{-6}}
}
\]

正式主线为：

```text
local_credit_mode=occurrence
```

因此本 A0 第一阶段必须分析 occurrence credit，不先切 `action_mean`。

---

# 3. A0 的核心问题

至少回答以下六个问题。

## Q1：GiGPO 的 local credit coverage 是否随训练上升？

定义：

\[
\rho_{\mathrm{coverage}}
=
\frac{
\#\{i:|A_i^{\mathrm{local}}|>\epsilon_A\}
}{
\#\{i\}
}.
\]

建议：

\[
\epsilon_A=10^{-8}.
\]

该指标回答：

> 训练 batch 中有多少 occurrence 实际得到了非零 local credit？

---

## Q2：GiGPO 的 local credit magnitude share 如何变化？

定义 occurrence-uniform：

\[
M_E
=
\sum_i |A_i^{\mathrm{macro}}|,
\]

\[
M_S
=
\sum_i |\omega A_i^{\mathrm{local}}|,
\]

\[
\boxed{
\rho_{\mathrm{mag}}
=
\frac{M_S}{M_E+M_S+\epsilon}.
}
\]

这个指标比 coverage 更重要。

即使 local coverage 很高，如果 local advantage 很小，真实影响仍可能有限。

---

## Q3：按 PPO 实际 token weighting 后，local share 是否更高？

当前 PPO 是 response-token weighted。

令有效 response token 数：

\[
L_i
=
\sum_k \text{response\_mask}_{i,k}.
\]

计算：

\[
M_E^{\mathrm{token}}
=
\sum_i L_i|A_i^{\mathrm{macro}}|,
\]

\[
M_S^{\mathrm{token}}
=
\sum_i L_i|\omega A_i^{\mathrm{local}}|,
\]

\[
\boxed{
\rho_{\mathrm{token}}
=
\frac{
M_S^{\mathrm{token}}
}{
M_E^{\mathrm{token}}+M_S^{\mathrm{token}}+\epsilon
}.
}
\]

这是 A0 的主指标之一。

如果无法从历史 artifact 取得准确 `response_mask/token_count`，必须：

1. 把 occurrence-uniform 结果作为主离线结果；
2. 明确标记 token-weighted 指标缺失；
3. 不允许用“平均 response 长度”静默代替。

---

## Q4：Macro 和 Local 经常同向还是冲突？

对同时非零的 occurrence：

### Sign agreement

\[
\rho_{\mathrm{agree}}
=
P[
\operatorname{sign}(A^E)
=
\operatorname{sign}(A^S)
].
\]

### Sign conflict

\[
\rho_{\mathrm{conflict}}
=
P[
A^E A^S<0
].
\]

同时报告：

\[
\operatorname{Pearson}(A^E,A^S),
\]

\[
\operatorname{Spearman}(A^E,A^S).
\]

解释：

- 高 agreement：local 多数是在增强 trajectory signal；
- 高 conflict：local 更多是在修正 trajectory credit；
- BACE 后期 conflict 上升且 local magnitude 同时上升：需要高度警惕 local signal overshoot。

---

## Q5：BACE 与 GiGPO 的 credit curriculum 是否不同？

按相同训练 step 统计：

\[
\rho_{\mathrm{coverage}}^{\mathrm{GiGPO}}
\quad vs\quad
\rho_{\mathrm{coverage}}^{\mathrm{BACE}},
\]

\[
\rho_{\mathrm{mag}}^{\mathrm{GiGPO}}
\quad vs\quad
\rho_{\mathrm{mag}}^{\mathrm{BACE}},
\]

\[
\rho_{\mathrm{token}}^{\mathrm{GiGPO}}
\quad vs\quad
\rho_{\mathrm{token}}^{\mathrm{BACE}}.
\]

重点分析：

```text
early  = step 1--50
middle = step 51--100
late   = step 101--150
```

不要只报告全程均值。

---

## Q6：BACE 的 local credit 主要由哪类 source 贡献？

BACE 分：

```text
root
branch_origin
branch_suffix
```

分别统计：

- occurrence 数；
- token 数；
- local coverage；
- mean / median \(|A_{\mathrm{macro}}|\)；
- mean / median \(|A_{\mathrm{local}}|\)；
- local magnitude mass；
- macro/local conflict；
- 总 advantage magnitude。

该表用于回答：

> BACE 后期 local signal 的增加，是 natural roots 自然形成的，还是主要来自主动 branch？

---

# 4. 输入数据要求

## 4.1 每个 occurrence 最少需要

建议统一整理为一个中间表，每行一个真实训练 occurrence：

| 字段 | 必须 | 用途 |
|---|---:|---|
| `update_step` | 是 | 训练阶段 |
| `task_id` / `uid` | 是 | macro/local 分组 |
| `traj_uid` | 是 | discounted return |
| `step_index` | 是 | trajectory 内排序 |
| `anchor_obs` | 是 | local group |
| `rewards` | 是 | \(G_t\) |
| `episode_rewards` | 是 | macro score |
| `is_action_valid` | 是 | invalid penalty |
| `response_token_count` | 推荐 | token-weighted share |
| `source_type` | BACE 必须 | root/branch 分解 |
| `action_identity` | 推荐 | 后续 action-level 对照 |
| `task_family` | 推荐 | family breakdown |

---

# 5. 数据来源与优先顺序

## 5.1 GiGPO

优先使用历史 150-step GiGPO 保存的逐-step trajectory parquet。

已有比较分析确认历史 GiGPO run 存在 150 份 trajectory parquet，因此 A0 应优先从这些数据重建，而不是重新训练。

若某字段在 parquet 中缺失：

1. 先检查训练日志 / TensorBoard / rollout artifacts；
2. 仍缺失则把缺口写入 `data_coverage_report.json`；
3. 不允许用 BACE 字段推测 GiGPO 字段。

---

## 5.2 BACE

优先使用历史 150-step BACE：

- trajectory/occurrence 数据；
- per-step BACE artifact；
- training diagnostics。

注意：

> 历史 150-step run 与当前 `handoff-refresh-20260828` HEAD 不完全相同，因此 A0 的结论应写成“解释历史 150-step gap 的诊断结果”，不能写成“当前 HEAD 已验证”。

---

# 6. 建议新建脚本

推荐：

```text
analysis/
  a0_credit_decomposition.py
  a0_credit_io.py
  a0_credit_plots.py
```

主入口：

```bash
python analysis/a0_credit_decomposition.py \
  --gigpo-run /path/to/gigpo_run \
  --bace-run /path/to/bace_run \
  --output-dir /path/to/a0_output \
  --gamma 0.95 \
  --invalid-penalty 0.1 \
  --epsilon 1e-6 \
  --phase-bounds 50 100 150
```

脚本必须自动写出：

```text
resolved_analysis_config.json
data_coverage_report.json
```

---

# 7. 具体执行流程

## Phase 0：数据盘点

对每个 run：

1. 找出全部 step；
2. 检查 step 是否覆盖 1--150；
3. 检查每个 step 是否能读；
4. 检查必需字段；
5. 检查每条 trajectory 内 `step_index` 是否唯一有序；
6. 检查 `episode_rewards` 在同一个 `traj_uid` 内是否一致；
7. 检查 `source_type` 是否只在 BACE 出现；
8. 输出缺失列表。

### 输出

`data_coverage_report.json`

示例：

```json
{
  "gigpo": {
    "steps_found": 150,
    "steps_usable": 150,
    "missing_fields": [],
    "token_count_available": true
  },
  "bace": {
    "steps_found": 150,
    "steps_usable": 149,
    "corrupted_steps": [136],
    "missing_fields": []
  }
}
```

若存在截断 JSONL：

- 跳过损坏 record；
- 记录损坏率；
- 不静默修复；
- 若 step 仍可从 parquet 重建，则优先使用 parquet。

---

## Phase 1：统一成 occurrence table

建议 DataFrame schema：

```text
update_step
run_type               # gigpo / bace
task_id
task_family
traj_uid
step_index
anchor_obs_hash
source_type             # gigpo 可统一写 "root"
env_reward
episode_reward
is_action_valid
response_token_count
action_identity
```

`anchor_obs` 如果文本很长，可保留 hash，但必须保证：

```text
hash 仅用于存储/分组；
exact equality 的来源仍是原 observation。
```

---

## Phase 2：重建 discounted step return

对每个：

```text
(update_step, traj_uid)
```

按 `step_index` 排序。

反向：

```python
running = 0.0
for row in reversed(traj):
    running = row.env_reward + gamma * running
    row.step_return = running
```

再做：

```python
row.local_score = (
    row.step_return
    - invalid_penalty * (not row.is_action_valid)
)
```

---

## Phase 3：重建 macro score

```python
row.macro_score = (
    row.episode_reward
    - invalid_penalty * (not row.is_action_valid)
)
```

不要把未来 invalid penalty 加进 macro/local 的 discounted chain。

---

## Phase 4：重建 macro advantage

分组：

```text
(update_step, task_id)
```

对所有 physical occurrences 做 sample std：

```python
mean = scores.mean()
std = scores.std(ddof=1)
```

如果：

```text
std <= 1e-12
```

则本组 macro advantage 全 0。

否则：

```python
macro_adv = (macro_score - mean) / (std + 1e-6)
```

这一步必须 reproduction-check。

---

## Phase 5：重建 local advantage

分组：

```text
(update_step, task_id, exact_anchor_obs)
```

仍使用 occurrence-level `local_score`。

若：

```text
group_size == 1
```

或：

```text
std <= 1e-12
```

则：

```text
local_adv = 0
```

否则：

```python
local_adv = (local_score - mean) / (std + 1e-6)
```

---

## Phase 6：一致性检查

若历史 BACE diagnostics 保存了 final advantage/component：

验证：

\[
A_{\mathrm{reconstructed}}
=
A_{\mathrm{macro}}+A_{\mathrm{local}}
\]

与保存结果的误差。

推荐门槛：

```text
max_abs_error < 1e-5
```

若历史 run 使用旧数值实现，仅在 near-zero variance group 存在差异：

- 单独列出这些 groups；
- 不允许整体忽略 discrepancy。

---

# 8. 每 step 必须计算的指标

至少输出：

```text
n_occurrences
n_trajectories
n_local_groups
mean_local_group_size
p50_local_group_size
p90_local_group_size
local_group_size_ge2_ratio
local_coverage
mean_abs_macro
mean_abs_local
median_abs_macro
median_abs_local
macro_abs_mass
local_abs_mass
local_magnitude_share
macro_local_sign_agreement
macro_local_sign_conflict
macro_local_pearson
macro_local_spearman
```

如果 token 数可得：

```text
token_macro_abs_mass
token_local_abs_mass
token_local_magnitude_share
```

BACE 再加：

```text
root_occurrence_ratio
branch_origin_occurrence_ratio
branch_suffix_occurrence_ratio

root_local_abs_mass
branch_origin_local_abs_mass
branch_suffix_local_abs_mass
```

---

# 9. 必须生成的 phase-level 汇总

按：

```text
Early : 1--50
Middle: 51--100
Late  : 101--150
```

输出：

`a0_phase_summary.csv`

建议字段：

| run | phase | local coverage | local magnitude share | token local share | sign conflict | mean group size |
|---|---|---:|---:|---:|---:|---:|
| GiGPO | early | | | | | |
| GiGPO | middle | | | | | |
| GiGPO | late | | | | | |
| BACE | early | | | | | |
| BACE | middle | | | | | |
| BACE | late | | | | | |

---

# 10. 必须生成的图

至少 5 张。

## Figure A0-1：Local coverage over training

横轴：

```text
training step
```

纵轴：

\[
\rho_{\mathrm{coverage}}.
\]

两条线：

```text
GiGPO
BACE
```

建议同时保存 raw 和 5-step rolling mean。

---

## Figure A0-2：Local magnitude share

纵轴：

\[
\rho_{\mathrm{mag}}.
\]

这是最重要的图之一。

---

## Figure A0-3：Token-weighted local share

如果 token 数可得：

\[
\rho_{\mathrm{token}}.
\]

如果不可得，不生成伪造图。

---

## Figure A0-4：Macro/local sign conflict

观察后期 BACE 是否显著高于 GiGPO。

---

## Figure A0-5：BACE source decomposition

按 early/middle/late 比较：

```text
root
branch_origin
branch_suffix
```

对 local absolute mass 的贡献。

---

# 11. 推荐增加的两个辅助分析

## 11.1 Distinct-action structure

对 local group 再统计：

```text
distinct_action_count
```

分成：

```text
1 action
2 actions
>=3 actions
```

因为 GiGPO local group 并不要求两个不同 action 才存在，而当前 BACE branch anchor 要求至少两个 observed actions。

该分析可以回答：

> GiGPO 原版的 step-level credit 中，有多少其实来自“同状态同动作重复”，而不是 action competition？

这会直接影响我们是否应该继续把 BACE branch anchor 强制限定为 mixed-action。

---

## 11.2 Natural-only BACE credit

为了区分：

```text
BACE actor 本身形成的 natural local structure
```

和：

```text
主动 branch 后产生的 local structure
```

建议在 BACE 上额外离线计算一次：

```text
只保留 source_type=root
```

得到：

\[
\rho_{\mathrm{local}}^{\mathrm{BACE-root-only}}.
\]

然后比较：

```text
GiGPO
BACE root-only
BACE all data
```

这是非常有价值的三方比较。

---

# 12. A0.2：可选 GPU Gradient Decomposition

A0 第一阶段完成后，如果发现：

```text
BACE late local magnitude share 明显高
```

建议再做真实 gradient diagnostic。

不需要全 150 step。

选：

```text
early checkpoint : ~step 30
middle checkpoint: ~step 80
late checkpoint  : ~step 130/150
```

每阶段取 3--5 个固定 batch。

---

## 方法

固定同一 batch 和 old policy。

分别构造：

### Macro-only

\[
A=A^{\mathrm{macro}}.
\]

得到：

\[
g_E=\nabla_\theta L_{\mathrm{PPO}}(A^E).
\]

### Local-only

\[
A=\omega A^{\mathrm{local}}.
\]

得到：

\[
g_S=\nabla_\theta L_{\mathrm{PPO}}(\omega A^S).
\]

记录：

\[
\|g_E\|_2,
\qquad
\|g_S\|_2,
\]

\[
\boxed{
\rho_{\mathrm{grad}}
=
\frac{\|g_S\|}
{\|g_E\|+\|g_S\|}
}
\]

及：

\[
\boxed{
\cos(g_E,g_S)
=
\frac{g_E^\top g_S}
{\|g_E\|\|g_S\|}
}.
\]

---

## 实现建议

不要长期修改 trainer。

新建诊断入口：

```text
analysis/a0_gradient_decomposition.py
```

要求：

- `optimizer.zero_grad()`；
- macro-only backward；
- 收集梯度向量统计；
- 清梯度；
- local-only backward；
- 不执行 optimizer.step；
- 不改变 checkpoint；
- 固定 RNG；
- 输出参数级和全局 norm。

第一版可只统计：

```text
global grad norm
selected transformer block grad norm
lm_head grad norm
```

避免物化整个模型的巨大扁平梯度向量。

计算 cosine 时可流式累加：

\[
g_E^\top g_S,\quad
\|g_E\|^2,\quad
\|g_S\|^2.
\]

---

# 13. 如何解释结果

## Case A：BACE late local share 显著高于 GiGPO

例如：

```text
GiGPO late local magnitude share = 0.38
BACE late                         = 0.65
```

并且 BACE late validation 更弱。

强烈支持：

\[
\boxed{\text{local-credit overshoot hypothesis}}
\]

下一批 GPU 实验优先：

```text
step_advantage_w
branch weighting
branch-origin mask
```

---

## Case B：Coverage 高，但 magnitude share 不高

说明 BACE 只是让更多 occurrence 获得 local credit，但幅度有限。

不应该仅凭 coverage 下调 \(\omega\)。

---

## Case C：Magnitude share 类似，但 conflict 明显更高

说明问题更可能是：

\[
\boxed{\text{local credit 质量 / 统计单位}}
\]

优先：

```text
occurrence vs action_mean
anchor semantics
continuation noise
```

---

## Case D：GiGPO 自身 late local share 也非常高

这意味着：

> “后期 local credit 多”本身不是问题。

要检查 BACE 的：

```text
branch correlation
source weighting
action-level consistency
```

而不是简单压低所有 local signal。

---

## Case E：BACE root-only 与 GiGPO 很接近，但 BACE all 明显偏离

这是最有价值的一种结果。

它直接说明偏移主要由：

\[
\boxed{\text{主动 branch 数据进入 optimizer 的方式}}
\]

产生，而不是 actor 自然状态分布本身不同。

---

# 14. A0 的成功标准

A0 不是以“找到一个显著性 p-value”为完成标准。

完成标准是：

1. 150-step 两条曲线都有可信 reconstruction；
2. early/middle/late 六行主汇总完整；
3. GiGPO/BACE local coverage 有可比定义；
4. local magnitude share 有可比定义；
5. token-weighted share 能算则算，不能算明确标记；
6. BACE root/branch-origin/branch-suffix 能拆分；
7. 至少完成一次 root-only BACE 对照；
8. 输出一份清晰结论：后续应优先改
   - weight，
   - credit estimator，
   - 还是无需改 local credit。

---

# 15. 推荐输出目录

```text
analysis_outputs/
  A0_credit_decomposition/
    resolved_analysis_config.json
    data_coverage_report.json
    occurrence_metrics.parquet
    step_summary.csv
    phase_summary.csv
    source_summary.csv
    action_structure_summary.csv

    figures/
      local_coverage_over_steps.png
      local_magnitude_share_over_steps.png
      token_local_share_over_steps.png
      macro_local_conflict_over_steps.png
      bace_source_local_mass.png

    report.md
```

---

# 16. `report.md` 必须回答的问题

最终报告控制在 2--5 页，但必须明确回答：

1. GiGPO local credit 是否随训练阶段自然增加？
2. BACE 的 local credit 是否比 GiGPO 增长更快？
3. 差异来自 coverage，还是 magnitude？
4. 差异主要来自 root、branch-origin 还是 branch-suffix？
5. macro/local 后期冲突是否增加？
6. BACE root-only 是否与 GiGPO 接近？
7. 是否有证据支持下调 `step_advantage_w`？
8. 是否有证据支持 `action_mean`？
9. 是否值得进一步做 gradient decomposition？

---

# 17. A0 的资源安排

## 第一阶段

```text
CPU / 普通服务器
```

即可。

这是最优先完成的部分。

## 第二阶段（可选 gradient）

优先：

```text
1--2 张 H20
```

即可做诊断，不应占排队 H100。

只有如果 H20 软件栈不方便做模型恢复，才放到 H100。

---

# 18. 一句话执行顺序

```text
盘点历史 150-step 数据
        ↓
重建 occurrence table
        ↓
严格复现 macro/local advantage
        ↓
做 reconstruction check
        ↓
逐 step 计算 coverage / magnitude / token share / conflict
        ↓
做 GiGPO vs BACE
        ↓
做 BACE root-only vs all
        ↓
按 early/middle/late 汇总
        ↓
根据结果决定是否做 gradient decomposition
        ↓
再决定 ω / action_mean / branch weighting 实验
```

---

# 19. 最终目的

A0 最终不是为了画几张曲线，而是回答：

\[
\boxed{
\text{BACE 是否改变了 GiGPO 原本随训练自然形成的 global/local credit curriculum？}
}
\]

如果答案是“是，而且这种偏移主要在后期由 branch 造成”，那么后续方法改动就有明确证据基础。

如果答案是“没有”，就应把研究重点从 local-credit proportion 转移到 acquisition quality、branch correlation 或其他训练语义上。
