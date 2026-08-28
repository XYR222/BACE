# GiGPO ALFWorld Reference Run：数据记录、指标保存与离线分析实现规范

> 用途：为 BACE-GiGPO 项目建立一次可长期复用的原版 GiGPO ALFWorld Reference Run 数据资产。  
> 目标：在不改变 GiGPO 训练语义的前提下，完整保存足够的原始数据，使后续论文动机分析、anchor 证据分析、BACE 参数预筛选、Exact Batch-ERV 离线模拟、replay/snapshot 回归测试、系统开销比较与机制实验尽可能不依赖重复训练。

---

# 1. 为什么现在要做一套高可观测 GiGPO Reference Run

当前项目后续需要反复回答几类问题：

1. 原版 GiGPO 在 ALFWorld 中到底形成了多少 repeated states；
2. repeated groups 中有多少真正包含有用的局部 decision evidence；
3. 同一 anchor 中是否存在大量 single-action variation、within-action sign conflict、mixed-outcome evidence；
4. 如果 BACE 在某个训练阶段只生成 $R=2,3,\ldots,8$ 条 natural roots，可以形成多少 structural anchors 与可用的 Batch-ERV capacity；
5. competence controller 在不同历史成功率、prior 和阈值下会如何分配 root/branch quota；
6. ALFWorld prefix replay / snapshot 是否能精确恢复历史自然状态；
7. GiGPO 与 BACE 在相同硬件、相同 terminal-leaf budget 下的 token、环境交互与 wall-clock 开销如何比较；
8. 后续机制实验需要从哪些 low/mid/high competence checkpoint 重新启动。

这些问题中的大部分并不需要重新训练模型，而只需要：

$$
\boxed{
\text{完整 natural rollout evidence}
+
\text{完整 occurrence-level credit records}
+
\text{可恢复的环境与模型上下文}
}
$$

因此，本次完整 GiGPO 训练应视为：

$$
\boxed{
\text{Baseline Training}
+
\text{Reference Data Acquisition}
+
\text{Engineering Regression Corpus}
}
$$

而不仅仅是得到一条 success-rate curve。

---

# 2. 总体设计原则

## 2.1 原始事实优先于当前指标

保存策略遵循：

$$
\boxed{
\text{Raw source of truth}
\rightarrow
\text{Versioned derived tables}
\rightarrow
\text{Figures / statistics}
}
$$

例如，我们现在可能定义 within-action sign conflict 为：

$$
\exists i,j\in\mathcal I(z,u):
\quad
A_i^S>0,
\qquad
A_j^S<0.
$$

未来可能加入容差：

$$
A_i^S>\epsilon_A,
\qquad
A_j^S<-\epsilon_A.
$$

因此应该保存：

- 每个 occurrence 的 $G_i$；
- 每个 occurrence 的原始 $A_i^S$；
- anchor key；
- canonical action；

而不是只保存一个 `conflict=true/false`。

**结论：所有会随分析定义、阈值、posterior 或论文叙事变化的量，都应尽量离线派生。**

---

## 2.2 不改变原版 GiGPO 的训练语义

本次 reference run 的 instrumentation 只负责：

- 观察；
- 记录；
- 校验；
- profile。

它不能改变：

- task sampling；
- group size；
- rollout sampling；
- `anchor_obs` grouping；
- GiGPO advantage；
- PPO loss；
- optimizer；
- scheduler；
- validation。

应满足：

$$
\boxed{
\text{Instrumented GiGPO}
\equiv
\text{Original GiGPO}
}
$$

在相同 seed 下，除极小计时扰动外，训练数据和优化结果应一致。

---

## 2.3 Occurrence 是核心数据单位

将每一次真实 agent 决策定义为一个 occurrence：

$$
o_i=(g,r,t,z_i,a_i,G_i,\ldots).
$$

唯一标识推荐：

```text
{run_id}/{update_id}/{task_instance_id}/{trajectory_id}/{step_idx}
```

例如：

```text
seed0/u0075/task03/root06/step17
```

几乎所有后续分析最终都可以归结为：

$$
\boxed{
(z,u,G,\text{trajectory},t,\text{success})
}
$$

这一层。

---

## 2.4 Raw 与 Derived 必须分离

推荐：

```text
raw/
    immutable

derived_v1/
    evidence_quality_v1

derived_v2/
    evidence_quality_v2
```

Raw 数据一经 reference run 完成后不可修改。

任何：

- canonicalization 更新；
- conflict threshold 更新；
- BERV threshold 更新；
- anchor eligibility 定义更新；
- prior 参数更新；

都只生成新的 derived 版本。

---

# 3. 最终数据层级

本次 reference archive 建议保存六层。

| 层级 | 核心对象 | 是否必须 | 主要用途 |
|---|---|---:|---|
| Run / Manifest | 代码、配置、环境、硬件 | 必须 | 完全复现与公平比较 |
| Update | 每个 policy update 的整体状态 | 必须 | learning curve、系统 profile、checkpoint 对齐 |
| Task / Group | 每个 ALFWorld task group | 必须 | success、task-family、group-level分析 |
| Trajectory | 每条 natural root rollout | 必须 | 完整 episode 结果、replay、coverage |
| Occurrence | 每次真实环境动作 | **核心必须** | anchor、action、credit、history、offline BERV |
| GiGPO Group | 在线实际构造的 step group | 必须 | offline/online 一致性、Figure 1、credit audit |

同时保存 checkpoint 与 profiler 数据。

---

# 4. Run / Manifest 层：必须保存的实验身份信息

这部分数据量极小，但没有它就无法形成严格 baseline。

## 4.1 Run identity

保存：

```text
run_id
experiment_name
seed
start_time
end_time
hostname
```

### 理由

后续三 seed 汇总、artifact 对齐、错误排查都依赖稳定的 run identity。

---

## 4.2 Git 与代码版本

保存：

```text
verl_agent_commit
GiGPO/local_patch_commit
ALFWorld_commit
TextWorld_version
fast_downward_version
PyTorch_version
transformers_version
vLLM_version
CUDA_version
NCCL_version
Python_version
```

同时保存：

```text
git diff
```

或 instrumentation patch hash。

### 理由

ALFWorld replay、anchor grouping、generation scheduler、PPO ratio 等都可能受到代码版本影响。后续 BACE 必须尽量使用相同 baseline commit 进行公平对照。

---

## 4.3 完整训练配置

保存完整 resolved config，而不是只保存启动命令。

至少包括：

```text
model path
reference model path
dataset path / split
tasks per update
group size
max turns / max env steps
max prompt tokens
max response tokens
rollout temperature
top-p / top-k
actor lr
PPO clip
KL coefficient
GiGPO omega
gamma
advantage normalization epsilon
batch size / mini-batch / micro-batch
rollout tensor parallel size
GPU count
validation interval
checkpoint interval
```

### 理由

后续主表需要确保 GiGPO 与 BACE：

$$
\boxed{
\text{same budget + same actor + same trainer + same hardware}
}
$$

只记录 wandb summary 不足以保证这一点。

---

## 4.4 硬件信息

保存：

```text
GPU model
GPU count
GPU memory
CPU model / cores
RAM
storage type
network / interconnect
```

### 理由

wall-clock 与 GPU-hour 只能在硬件信息完整时具有比较意义。

---

# 5. Update 层：每次 Policy Update 必须保存的指标

定义 update $k$ 为一次完整：

```text
rollout -> grouping/advantage -> old/ref logprob -> actor update
```

## 5.1 基本训练进度

保存：

```text
update_id
epoch_id
global_step
wall_clock_since_start
```

### 理由

用于所有训练曲线的共同横轴，并与 checkpoint 对齐。

---

## 5.2 Performance 指标

保存：

```text
train_success_rate
train_mean_reward
validation_success_rate
validation_mean_reward
per_task_family_success_rate
```

同时保存原始 task outcomes，而不仅是均值。

### 理由

后续需要：

- performance vs update；
- competence-stage 划分；
- task-family competence history；
- low/mid/high checkpoint 选择；
- seed uncertainty。

---

## 5.3 生成规模

保存：

```text
num_tasks
num_terminal_trajectories
num_macro_actions
num_actor_generated_tokens
num_prompt_tokens
num_response_tokens
mean_response_length
trajectory_length_mean/std/max
```

### 理由

BACE 的 branch suffix 与完整 root 长度不同，因此 terminal leaves 相同并不代表生成 compute 相同。

最终论文至少需要：

$$
\text{Performance vs Generated Tokens}.
$$

---

## 5.4 环境交互规模

保存：

```text
env_steps_total
env_steps_successful_trajectories
env_steps_failed_trajectories
env_invalid_action_steps
```

### 理由

形成 interaction efficiency：

$$
\boxed{
\text{performance / environment steps}
}
$$

并帮助分析早期模型失败是否由 invalid actions、loops 或长无效轨迹造成。

---

# 6. Trajectory 层：每条 Natural Root 必须保存什么

每个 task 在原版 GiGPO 下有完整 natural roots。

推荐每条轨迹保存一条 `trajectory_record`。

## 6.1 Identity

```text
run_id
update_id
task_instance_id
task_family
trajectory_id / root_id
group_slot
seed / rollout seed
```

---

## 6.2 Environment identity

```text
game_file_path
game_id
problem/task description
environment mode
max_steps
```

### 理由

ALFWorld prefix replay 必须重新加载同一个 game。

---

## 6.3 完整 Episode outcome

```text
terminal_reward
terminal_success
terminal_reason
terminated
truncated
num_steps
```

若环境提供 richer reward，也保存原始 reward sequence。

### 理由

这是：

- GiGPO trajectory-level credit；
- local Beta posterior natural evidence；
- competence history；
- Figure 1 outcome decomposition；

的 source of truth。

---

## 6.4 完整事件序列引用

不要在 trajectory row 中重复保存大文本；保存：

```text
first_occurrence_id
num_occurrences
event_log_path / offset
```

完整 step data 由 occurrence/event log 承担。

---

## 6.5 Trajectory-level GiGPO 量

保存训练时实际使用的：

```text
trajectory_group_mean_return
trajectory_group_std_return
trajectory_advantage_A_E
```

### 理由

虽然可以从 rewards 离线重算，但必须保留 online-used value 用于验证：

$$
A^{E}_{offline}=A^{E}_{online}.
$$

---

## 6.6 生成 token 统计

保存：

```text
prompt_token_count
response_token_count
actor_generated_token_count
```

### 理由

用于按 trajectory 和 task 分析 generation cost。

---

# 7. Occurrence 层：整个 Reference Archive 最核心的数据

每个 actor 真正产生并提交给环境的动作都生成一条 occurrence record。

## 7.1 Identity 与位置

必须保存：

```text
occurrence_id
run_id
update_id
task_instance_id
task_family
trajectory_id
step_idx
```

推荐保证 `occurrence_id` 全局唯一。

---

## 7.2 Pre-action observation：必须保存原始值

保存：

```text
anchor_obs_before_action_raw
anchor_obs_serialized
anchor_key_online
```

其中：

- `raw`：环境原始 observation；
- `serialized`：训练代码实际传入 hashable/grouping 逻辑的确定性表示；
- `anchor_key_online`：GiGPO 在线实际使用的 key。

### 理由

原版 GiGPO 的核心 step grouping 是按 pre-action observation exact grouping。

保存 raw observation，而不仅是 hash，是为了：

1. 离线重建 exact grouping；
2. 检查 serialization 是否改变语义；
3. 将来做 similarity / history-aware anchor 消融；
4. replay restore 后做 exact assertion。

---

## 7.3 Post-action observation

保存：

```text
observation_after_action_raw
```

### 理由

用于：

- replay correctness；
- invalid action effect；
- branch 起点执行后续状态对齐；
- 分析动作是否真正改变环境。

---

## 7.4 Admissible action set

必须保存：

```text
admissible_actions_before_action
```

建议使用稳定排序后的 list，同时保留原始顺序如环境有语义。

### 理由

后续需要离线判断：

- 模型动作是否 executable；
- canonical action identity；
- invalid action ratio；
- state 恢复后 action set 是否一致；
- observed action competition；
- WebShop/ALFWorld action-space 分析。

这是一个高价值字段，不能依赖训练后重新访问环境来恢复。

---

## 7.5 模型完整响应

保存：

```text
raw_model_response
raw_thought_text
raw_action_text
```

如果当前 parser 能明确分离 CoT 和 action，则保存解析结果；仍需保留完整 raw response。

### 理由

未来可能更新：

- action parser；
- canonicalization；
- invalid-action 定义；
- action identity；

如果只保存 parser 当前输出，就无法重新审计。

---

## 7.6 环境实际执行动作

保存：

```text
parsed_action
executed_action
execution_success
invalid_action_flag
env_feedback_after_action
```

### 理由

要严格区分：

$$
\text{LLM textual output}
\neq
\text{parser result}
\neq
\text{environment-executed action}.
$$

后续 anchor-action evidence 应以 environment-grounded action identity 为基础。

---

## 7.7 Canonical action

建议 reference run 同时保存当前版本：

```text
canonical_action_v0
canonicalization_status
canonicalization_reason
```

但 `canonical_action_v0` 应视为 **derived-but-cached**，不是不可替换 source of truth。

### 理由

后续 BACE 的 local posterior 和 evidence decomposition 都要按 $(z,u)$ 聚合。

同时保留 raw/executed action，使 canonicalization 规则未来可以重新执行。

---

## 7.8 当前步 reward 与 terminal outcome

保存：

```text
step_reward
done_after_action
terminal_success_of_trajectory
terminal_return_of_trajectory
```

### 理由

对于 terminal sparse reward 环境，当前动作的后续局部 evidence 最终来自 trajectory outcome。

---

## 7.9 Return-to-go

保存在线实际计算值：

```text
return_to_go_G
```

并保存：

```text
gamma_used
```

### 理由

GiGPO occurrence-level local credit 直接依赖：

$$
G_i.
$$

未来 Figure 1 中的 `No local credit`、within-action variance、mixed credit 都基于这个量。

同时保留 reward sequence，可验证离线重算：

$$
G_{offline}=G_{online}.
$$

---

## 7.10 GiGPO step-group identity

保存：

```text
online_step_group_id
online_step_group_size
```

### 理由

可以检查 offline exact regrouping 是否与在线 GiGPO 完全一致。

---

## 7.11 GiGPO local group statistics

保存在线实际使用：

```text
step_group_mean_G
step_group_std_G
step_advantage_A_S
```

### 理由

后续 evidence-quality audit 依赖 occurrence-level local advantage。

例如 within-action sign conflict：

$$
\exists u,i,j:
A_i^S>\epsilon_A,
\quad
A_j^S<-\epsilon_A.
$$

同时这也是 instrumentation correctness 的核心验收量。

---

## 7.12 Final combined advantage

保存：

```text
combined_advantage
```

即当前原版 GiGPO 实际训练 action/token 所使用的最终 scalar advantage。

### 理由

用于：

- offline loss reconstruction；
- 分析 local term 对 total gradient 的影响；
- 比较 occurrence-level 与未来 action-aggregated credit。

---

# 8. Model Context 与 History 如何保存

GiGPO 的 anchor grouping 只依赖当前 observation，但 LLM 决策依赖完整 conversation/history。

后续我们很可能需要分析：

> 相同 exact anchor observation + 相同 canonical action，却产生不同 local credit，是否与 arrival history 有关？

因此 history 必须可恢复。

## 8.1 推荐方式：事件日志 + 索引

不要为每个 step 重复存完整 history。

每条 trajectory 保存事件序列：

```text
initial system/user prompt
environment observation 0
model response 0
environment feedback 0
model response 1
...
```

occurrence 只保存：

```text
trajectory_id
step_idx
history_prefix_end_offset
```

于是：

$$
h_{r,t}
$$

可离线精确重建。

### 理由

这种方式同时满足：

- replay branch origin 的模型上下文恢复；
- history-aware grouping 消融；
- 减少大文本重复存储。

---

# 9. Token 与 Log-Probability 数据

## 9.1 必须/强烈建议保存

```text
response_token_ids
response_attention_mask
action_token_mask / response_action_span
chosen_token_old_logprobs
```

如果当前 pipeline 已经计算 reference logprob，可缓存：

```text
chosen_token_ref_logprobs
```

### 理由

后续可离线研究：

- action support；
- response probability；
- PPO ratio distribution；
- clipping；
- policy confidence；
- 同一 anchor 下 action probability 与 outcome 的关系；
- BACE branch origin response reuse 所需 token metadata。

---

## 9.2 不建议保存完整 vocabulary logits

完整 logits：

$$
T\times|\mathcal V|
$$

空间成本极高。

当前研究目标通常只需要：

$$
\boxed{
\text{chosen-token logprob}
}
$$

因此默认不保存全 vocabulary logits。

如未来确需 top-k policy alternatives，可在特定 checkpoint 对抽样 anchors 重新 query actor，而不是全训练全量保存。

---

# 10. Replay / Snapshot 所需的环境恢复字段

虽然原版 GiGPO 本身不 branching，本次 archive 应直接为未来 BACE branch restoration 服务。

每个 occurrence 至少能够恢复：

```text
task_id
game_file_path
trajectory_id
step_idx
environment_action_prefix
anchor_obs_before_action
admissible_actions_before_action
model_history_ref
remaining_horizon
```

其中 `environment_action_prefix` 可以不在每一 row 完整复制，而通过 trajectory event sequence 前缀重建。

## 10.1 不长期保存 native StateID 作为唯一恢复依据

若未来实现 TextWorld/Fast-Downward StateID snapshot，StateID 只在当前 registry/context 中有效。

长期 reference archive 的恢复 source of truth 应仍是：

$$
\boxed{
\text{same game + deterministic action prefix}
}
$$

StateID 可作为运行时优化字段，但不能替代 portable replay metadata。

---

# 11. GiGPO Group 层：必须保存在线实际 Step Group

虽然理论上 occurrence 表足以离线重建 GiGPO groups，仍建议保存训练时 online group table。

每个 group 记录：

```text
run_id
update_id
task_instance_id
online_step_group_id
anchor_key
anchor_obs_ref
group_size
occurrence_ids[]
trajectory_ids[]
num_unique_trajectories
returns_G[]
mean_G
std_G
step_advantages[]
```

可额外缓存：

```text
num_unique_raw_actions
num_unique_executed_actions
num_unique_canonical_actions_v0
```

### 理由

主要用于两个目的。

### 11.1 在线/离线一致性

必须能验证：

$$
\boxed{
\mathcal I^{offline}(z)=\mathcal I^{online}(z)
}
$$

以及：

$$
\boxed{
A^{S,offline}_i=A^{S,online}_i.
}
$$

### 11.2 Figure 1 和机制诊断

可以直接派生：

- group size distribution；
- repeated anchor count；
- cross-trajectory / within-trajectory recurrence；
- evidence-quality decomposition。

---

# 12. Anchor 形成相关必须可离线计算的指标

以下指标**建议离线计算，不需要在线写死**，前提是 raw occurrence/group 数据完整。

## 12.1 Unique anchor count

$$
N_{unique}.
$$

作用：状态覆盖规模。

---

## 12.2 Repeated anchor count

$$
N_{rep}
=
\left|
\{z:|\mathcal I(z)|\ge2\}
\right|.
$$

作用：自然可形成局部 comparison structure 的数量。

---

## 12.3 Group-size distribution

统计：

$$
|\mathcal I(z)|.
$$

作用：复现/扩展 GiGPO 的 step-level group distribution，并作为论文 Figure 1(a)。

---

## 12.4 Cross-trajectory recurrence

定义：

$$
N_{cross}(z)
=
\left|
\{r:\exists t,(r,t)\in\mathcal I(z)\}
\right|.
$$

---

## 12.5 Within-trajectory repeat count

$$
N_{within}(z)
=
|\mathcal I(z)|-N_{cross}(z).
$$

以及：

$$
\rho_{loop}(z)
=
\frac{N_{within}(z)}{|\mathcal I(z)|}.
$$

### 理由

原版 GiGPO 保留同一 trajectory 内 repeated visits。该指标用于判断大 group 是否主要由 loop 贡献。

---

# 13. Figure 1：Evidence Quality Decomposition 所需指标

这是当前论文最重要的 motivation analysis 之一。

对 repeated group $z$：

$$
|\mathcal I(z)|\ge2.
$$

## 13.1 Local return contrast

定义：

$$
\boxed{
\Delta_G(z)
=
\max_{i\in\mathcal I(z)}G_i
-
\min_{i\in\mathcal I(z)}G_i.
}
$$

用于判断该 group 是否实际产生局部 return variation。

---

## 13.2 Observed action diversity

定义：

$$
\mathcal C^{obs}(z)
=
\{u_i:i\in\mathcal I(z),u_i\neq INVALID\}
$$

以及：

$$
K_z=|\mathcal C^{obs}(z)|.
$$

作用：区分 return variation 是 current-action comparison，还是相同 current action 后 continuation divergence。

---

## 13.3 Evidence categories

### Type I — No local credit

$$
\Delta_G(z)\le\epsilon_G.
$$

### Type II — Credit-active but non-comparative

$$
\Delta_G(z)>\epsilon_G,
\qquad
K_z<2.
$$

### Type III — Action-comparative without observed conflict

$$
\Delta_G(z)>\epsilon_G,
\qquad
K_z\ge2,
$$

且未检测到 within-action sign conflict。

### Type IV — Action-comparative with within-action sign conflict

存在 $u$：

$$
\exists i,j\in\mathcal I(z,u):
\quad
A_i^S>\epsilon_A,
\qquad
A_j^S<-\epsilon_A.
$$

### 理由

该分解用于展示：

$$
\boxed{
\text{structural recurrence}
\neq
\text{uniformly useful decision evidence}
}
$$

---

# 14. Within-Action Noise / Conflict 相关指标

## 14.1 Within-action sign conflict ratio

按 group：

$$
\operatorname{Conflict}(z)
=
\mathbf1[
\exists u,i,j:
A_i^S>\epsilon_A,
A_j^S<-\epsilon_A
].
$$

报告：

$$
\frac{\#\text{conflict groups}}{\#\text{repeated groups}}.
$$

---

## 14.2 Within-action variance fraction

定义总体 variation：

$$
T_z
=
\sum_i(G_i-\bar G_z)^2.
$$

同动作内部 variation：

$$
W_z
=
\sum_u\sum_{i\in\mathcal I(z,u)}
(G_i-\bar G_{z,u})^2.
$$

定义：

$$
\boxed{
\eta_z
=
\frac{W_z}{T_z+\epsilon}.
}
$$

### 理由

用于量化一个 anchor 的 return variation 中，有多少不能由 current-action identity 区分，而发生在同一动作内部。

---

## 14.3 Action-count imbalance

对 anchor $z$：

```text
action_counts = {u: n_z,u}
```

可派生：

$$
\max_u n_{z,u}/|\mathcal I(z)|
$$

或 entropy / Gini。

### 理由

用于分析 rare lucky action、dominant action 和自然证据不平衡。

---

# 15. Mixed Outcome 与成功/失败结构

建议离线统计：

## 15.1 Mixed terminal outcome group

$$
\exists i,j\in\mathcal I(z):
Y_i=1,
Y_j=0.
$$

---

## 15.2 All-success / all-failure

```text
all_success
all_failure
mixed
```

### 理由

这是直观的 supplementary diagnostic，但应与 $\Delta_G$ 区分，因为在 $\gamma<1$ 或成功步数不同的情况下，all-success 也可能具有不同 return-to-go。

---

# 16. Invalid Action 与 Loop 指标

保存 raw action、executed action、admissible set 后，可离线计算：

```text
invalid_action_ratio
invalid_action_ratio_by_update
invalid_action_ratio_by_task_family
invalid_action_ratio_inside_large_anchor_groups
consecutive_invalid_action_length
loop_visit_count
```

### 理由

GiGPO 早期可能产生大量重复状态；需要区分：

- 有意义的状态重访；
- 无效动作导致 observation 不变；
- loop 行为导致 group size 膨胀。

这对 Figure 1 的解释非常重要。

---

# 17. 仅用 GiGPO Natural Roots 就能离线做的 BACE 可行性分析

Reference run 每个 task 有完整自然 group，因此可以从 $8$ roots 中抽取前 $R$ 条或随机子集：

$$
R\in\{2,3,4,5,6,7,8\}.
$$

## 17.1 Root-count subsampling

对每个 $R$ 重新构造：

```text
unique anchors
repeated anchors
action-comparative anchors
structural branch anchors
```

### 理由

直接回答：

> 如果 BACE 只给当前 task 这么多 natural roots，它是否已经拥有足够 local evidence 可以 branch？

---

## 17.2 Passive repeated-state probability empirical curve

估计：

$$
\hat P_R(N_z\ge2).
$$

并观察 repeated-anchor 数量随 $R$ 的增长。

### 理由

与论文 appendix 的 passive-collision analysis 对应。

---

# 18. 离线构造 BACE Local Beta Posterior

当前 BACE 主方案只使用 natural observed actions 构建 local posterior。

对 subsampled root set，针对：

$$
(z,u)
$$

统计：

```text
natural_occurrence_count
natural_success_count
natural_failure_count
```

结合指定 local prior：

$$
\alpha_{z,u}
=
\alpha_{0}+S_{z,u},
$$

$$
\beta_{z,u}
=
\beta_{0}+F_{z,u}.
$$

可离线计算：

```text
posterior_mean
posterior_variance
```

### 理由

无需真实 branch 就可预先分析当前自然证据的不确定结构。

---

# 19. Exact One-Step ERV / Exact Batch-ERV 离线统计

当前主方法采用 Exact Batch-ERV，因此 reference data 应支持完全离线计算。

## 19.1 One-step ERV

对 action $u$：

$$
m_u=\frac{\alpha_u}{\alpha_u+\beta_u}.
$$

根据 exact closed form 计算：

$$
\operatorname{ERV}(z,u).
$$

---

## 19.2 Batch-ERV

对最多：

$$
L_{max}=2
$$

的 local batch plan，枚举：

```text
0 branches
[u]
[u,u]
[u,v]
```

并通过 Beta-Binomial finite sum 计算：

$$
V_z^{(0)},
V_z^{(1)},
V_z^{(2)}.
$$

随后：

$$
\delta_z^{(1)}=V_z^{(1)},
$$

$$
\delta_z^{(2)}=V_z^{(2)}-V_z^{(1)}.
$$

---

## 19.3 Information capacity

对指定 $\tau_{BERV}$：

$$
c(z)
=
\mathbf1[\delta_z^{(1)}\ge\tau_{BERV}]
+
\mathbf1[\delta_z^{(2)}\ge\tau_{BERV}].
$$

$$
\boxed{
C_g=\sum_z c(z).
}
$$

### 理由

这可以在正式 BACE 训练前回答：

- $R=2$ 时平均 capacity 多大；
- $R=4$ 时 capacity 多大；
- early/mid/late 如何变化；
- $\tau_{BERV}$ 选多少才不会长期把 branch capacity 压成 0；
- capacity correction 会有多频繁。

---

# 20. Exact Global DP 离线模拟

给定 hypothetical branch quota：

$$
Q\in\{1,2,3,4,5,6\},
$$

解：

$$
\max_{m_z\in\{0,1,2\}}
\sum_zV_z^{(m_z)}
$$

s.t.

$$
\sum_zm_z=Q.
$$

保存 derived：

```text
selected_anchor_ids
selected_action_multiset
objective_value
number_of_tied_optima
```

### 理由

可以预先观察 Batch-ERV 是否：

- 过度集中；
- 经常选同一动作两次；
- 经常发生 ties；
- 有足够多非零 value experiments。

---

# 21. Competence Controller 可离线模拟的指标

GiGPO reference run 本身全部是 natural roots，因此非常适合构造 competence history。

按 task family $c$ 与 update $k$ 保存 natural-root：

$$
S_{c,k}^{root},
\qquad
F_{c,k}^{root}.
$$

之后离线测试：

```text
p0
kappa0
lambda_hist
tau_comp
kappa_T bounds
```

得到：

$$
q_{c,k}=P(\Phi_{c,k}>\tau_{comp}).
$$

以及：

$$
\bar Q_{c,k}
=
\operatorname{round}[(B-R_{min})q_{c,k}].
$$

## 21.1 必须派生的 controller diagnostics

```text
q by update / family
planned Q distribution
fraction Q=0/1/.../6
first update with nonzero Q
quota oscillation frequency
family-to-family variance
```

### 理由

可以在没有 BACE rollout 的情况下先排除明显不合理 controller 参数。

---

# 22. Replay Regression Corpus

Reference run 完成后，从 raw occurrences 中固定抽样一组 replay cases。

建议分层抽样：

```text
early/mid/late checkpoint
short/mid/long prefix
success/failure trajectory
small/large anchor group
valid/invalid-action-nearby cases
```

推荐至少：

```text
1k cases for development
10k cases for production stress test
```

每个 case 保存：

```text
case_id
game_file
prefix actions
expected anchor_obs
expected anchor_key
expected admissible action set
expected done flag
expected step count
```

## 22.1 Prefix replay 验收

恢复后验证：

$$
anchor\_key_{restore}=anchor\_key_{recorded}.
$$

以及：

$$
\mathcal A_{adm}^{restore}
=
\mathcal A_{adm}^{recorded}.
$$

## 22.2 Snapshot 验收

未来 StateID snapshot 与 prefix replay 同时恢复同一个 reference state，比较：

```text
observation
anchor key
admissible actions
done
remaining horizon
next-action transition
```

### 理由

这套固定 corpus 以后可用于任何 replay/snapshot 优化的回归测试。

---

# 23. PPO / Optimization 指标

本次 reference run 也应保存基本优化统计，以便未来排查 BACE 的训练稳定性差异。

至少保存：

```text
actor_loss
policy_loss
KL_loss / ref_KL
total_loss
learning_rate
PPO_clip_fraction
approx_kl / policy_kl
entropy if available
grad_norm_before_clip
grad_norm_after_clip
optimizer_step_skipped / overflow flag
```

### 理由

未来如果 BACE success 提升或下降，需要区分是：

- rollout evidence 变化；
- gradient magnitude 变化；
- clipping 变化；
- KL drift；
- optimization instability。

---

# 24. Advantage 分布指标

建议每 update 保存聚合统计，同时 raw occurrence 中保留逐样本值。

```text
A_E mean/std/min/max/quantiles
A_S mean/std/min/max/quantiles
combined advantage mean/std/quantiles
fraction A_S == 0
fraction positive/negative A_S
fraction step groups with zero variance
```

### 理由

论文机制问题包括：

- 有多少 anchors 真正产生 nonzero local credit；
- 不同训练阶段 local signal 是否增强或衰减；
- branch 未来是否改善 GiGPO local signal。

---

# 25. System Timing / Profiling：必须一次性采完整

GiGPO 与 BACE 最终需要 wall-clock comparison，因此 reference run 不能只保存总时长。

每个 update 分 phase 记录：

```text
data/task preparation
rollout generation
environment interaction
GiGPO grouping
return/advantage computation
old-policy logprob
reference-model logprob
actor forward/backward
optimizer step
validation
checkpoint I/O
trace logging I/O
```

其中建议至少重点单列：

$$
\boxed{
T_{rollout},
T_{group},
T_{adv},
T_{old},
T_{ref},
T_{update}
}
$$

## 25.1 Trace logging overhead

为了证明 instrumentation 没有严重污染 baseline timing，同时记录：

```text
trace_encode_time
trace_write_time
trace_queue_wait
bytes_written
```

### 理由

未来真正做 GiGPO vs BACE wall-clock 比较时，可：

1. 两边使用相同 recorder；或
2. 从 reference timing 中单独知道 instrumentation overhead。

---

# 26. GPU 与系统资源指标

按 update 或固定采样频率记录：

```text
GPU utilization
GPU memory allocated
GPU memory reserved
peak GPU memory
CPU utilization
RAM usage
rollout worker utilization
```

如果 Ray/vLLM 可方便暴露，也记录：

```text
pending generation requests
batch sizes
GPU idle/tail time
```

### 理由

BACE 的主要系统风险不是 Batch-ERV CPU 计算，而是 root/branch generation wave 和 tail under-utilization。Reference GiGPO 的利用率是必要基准。

---

# 27. Checkpoint 保存策略

需要区分 actor-only checkpoint 与 resumable checkpoint。

## 27.1 Actor checkpoint

建议相对密集，例如：

```text
every 5-10 policy updates
```

至少确保 early/mid/late 关键阶段附近存在 checkpoint。

保存：

```text
actor weights
tokenizer/config
policy update id
```

### 用途

- 重新 rollout；
- 模型概率 probe；
- replay/snapshot test；
- 选择机制实验起点。

---

## 27.2 Full resumable checkpoint

可较稀，例如 validation interval。

保存：

```text
actor
optimizer
scheduler
AMP scaler if any
global step
policy update id
Python RNG
NumPy RNG
PyTorch CPU RNG
CUDA RNG states
sampler/dataloader state if applicable
```

### 理由

Topology sanity experiment 需要从同一训练状态 clone：

$$
8R+0B,
\quad
5R+3B,
\quad
2R+6B.
$$

只有 actor 权重不足以实现严格的 continuation fairness。

---

# 28. Validation 数据也必须保留到 task level

不要只保存 aggregate validation success。

建议每次 validation 保存：

```text
checkpoint/update_id
validation_task_id
task_family
success
reward
num_steps
raw trajectory reference
```

### 理由

后续可以：

- task-family breakdown；
- matched checkpoint comparison；
- failure-case analysis；
- 确定 competence stages；
- 计算 paired statistical tests。

---

# 29. 推荐存储格式

## 29.1 目录结构

推荐：

```text
gigpo_alfworld_reference/
│
├── manifest/
│   ├── run.yaml
│   ├── versions.json
│   ├── hardware.json
│   ├── git_diff.patch
│   └── schema_version.json
│
├── raw/
│   ├── trajectories/
│   │   ├── update_000.jsonl.zst
│   │   └── ...
│   ├── occurrences/
│   │   ├── update_000.parquet
│   │   └── ...
│   ├── online_groups/
│   │   ├── update_000.parquet
│   │   └── ...
│   └── token_data/
│       └── ...
│
├── updates/
│   ├── update_metrics.parquet
│   └── profiler.parquet
│
├── validation/
│   └── validation_records.parquet
│
├── checkpoints/
│   ├── actor/
│   └── resumable/
│
├── replay_corpus/
│   ├── dev_cases.jsonl.zst
│   └── stress_cases.jsonl.zst
│
└── analysis/
    ├── derived_v1/
    ├── figures/
    └── reports/
```

---

## 29.2 格式选择

### Parquet

适合：

- occurrence table；
- group table；
- update metrics；
- profiler；
- validation records。

优点：

- columnar；
- 可快速过滤 update/task/family；
- 适合 pandas/polars/duckdb。

### JSONL.zst

适合：

- 复杂嵌套 trajectory event logs；
- 长文本；
- replay cases。

### safetensors / native checkpoint

用于 model weights 与训练状态。

---

# 30. 推荐核心表 Schema

## 30.1 `trajectories.parquet`

核心字段：

```text
run_id
update_id
task_instance_id
task_family
trajectory_id
game_id
game_file
terminal_reward
terminal_success
num_steps
A_E_online
prompt_tokens
response_tokens
actor_generated_tokens
```

---

## 30.2 `occurrences.parquet`

核心字段：

```text
occurrence_id
run_id
update_id
task_instance_id
task_family
trajectory_id
step_idx
anchor_obs_raw_ref
anchor_obs_serialized
anchor_key_online
post_obs_raw_ref
admissible_actions_ref
raw_model_response_ref
raw_action_text
parsed_action
executed_action
invalid_action_flag
canonical_action_v0
step_reward
terminal_success
return_to_go_G
online_step_group_id
online_step_group_size
step_group_mean_G
step_group_std_G
A_S_online
A_combined_online
history_ref
remaining_horizon
response_token_ref
old_logprob_ref
```

---

## 30.3 `online_groups.parquet`

```text
run_id
update_id
task_instance_id
group_id
anchor_key
anchor_obs_ref
group_size
num_unique_trajectories
occurrence_ids_ref
mean_G
std_G
num_unique_canonical_actions_v0
```

---

## 30.4 `update_metrics.parquet`

```text
run_id
update_id
epoch_id
train_success
val_success
num_tasks
num_trajectories
actor_generated_tokens
env_steps
actor_loss
ref_kl
clip_fraction
grad_norm
wall_clock_total
```

---

# 31. Source-of-Truth 与 Cached/Derived 字段分级

建议每个 schema 字段标注：

```text
SOURCE
CACHE
DERIVED
```

## 31.1 SOURCE

训练结束后无法可靠从其他数据恢复，必须原样保存，例如：

```text
raw observation
raw model response
executed action
admissible action set
environment reward
game identity
response token ids
```

## 31.2 CACHE

理论上可重算，但在线实际值值得保存用于审计，例如：

```text
anchor_key_online
A_E_online
A_S_online
combined advantage
online group assignment
old logprob
```

## 31.3 DERIVED

应由 offline analysis 计算：

```text
conflict flag
within-action variance fraction
BERV
structural anchor flag
capacity
hypothetical Q
```

---

# 32. 实现架构：TraceRecorder

建议在原 GiGPO pipeline 中加入一个统一 recorder，而不是在多个文件散落 `json.dump`。

抽象接口：

```text
TraceRecorder
    on_run_start(...)
    on_update_start(...)
    on_task_start(...)
    on_trajectory_start(...)
    on_occurrence(...)
    on_trajectory_end(...)
    on_groups_built(...)
    on_advantages_computed(...)
    on_update_profile(...)
    on_validation(...)
    on_checkpoint(...)
    on_update_end(...)
    on_run_end(...)
```

## 32.1 设计要求

Recorder 必须：

1. 不修改训练对象；
2. 不依赖异步 completion 顺序生成 identity；
3. 使用稳定 primary key；
4. 支持 buffer + batch write；
5. 允许关闭大型可选字段；
6. 每 update 完成 durable flush；
7. 写 schema version。

---

# 33. 具体代码 Hook 位置

## 33.1 Environment step hook

在调用环境：

```text
env.step(action)
```

之前保存：

```text
pre_action observation
admissible actions
raw model response
parsed/executed action
history reference
```

执行后补充：

```text
post observation
step reward
done
environment feedback
```

### 关键点

必须捕获真正送入环境的 `executed_action`，不能只捕获 LLM 文本。

---

## 33.2 Trajectory finalization hook

episode 完成后补充每个 occurrence：

```text
terminal success
terminal return
return-to-go
trajectory length
```

---

## 33.3 GiGPO grouping hook

在原版 GiGPO `anchor_obs` grouping 完成后记录：

```text
group_id -> occurrence_ids
```

及 online group statistics。

---

## 33.4 Advantage hook

在 $A^E$、$A^S$ 与最终 advantage 计算完成后，将值回填到 trajectory / occurrence rows。

---

## 33.5 PPO hook

保存：

- token ids；
- chosen-token old logprob；
- ref logprob；
- clip fraction；
- loss；
- KL；
- gradient norm。

---

## 33.6 Profiler hook

对各大 phase 使用明确 timer context：

```text
with timer("rollout")
with timer("grouping")
with timer("advantage")
with timer("old_logprob")
with timer("ref_logprob")
with timer("actor_update")
```

不要通过总 wall-clock 相减推断。

---

# 34. 避免 Recorder 本身成为性能瓶颈

## 34.1 不在 GPU critical path 做大量序列化

推荐：

```text
training process
  -> enqueue compact record
background writer process/thread
  -> serialize/compress/write
```

但必须保证 update 结束时 flush 完成。

## 34.2 大文本使用 reference

`occurrences.parquet` 中不重复存数 KB history。

保存：

```text
text_blob_id
```

大文本集中写 compressed blob/jsonl。

## 34.3 Token arrays 分离

token/logprob 大数组可写独立 shard；主表保存 offset/reference。

---

# 35. 正式完整训练前：Instrumentation Validation Run

在完整训练前先跑：

```text
2-5 policy updates
```

目标不是性能，而是验证数据可重建性。

## 35.1 Group reconstruction invariant

离线根据：

```text
task_instance_id + anchor_obs serialized
```

重建 groups。

要求：

$$
\boxed{
\mathcal I^{offline}(z)=\mathcal I^{online}(z)
}
$$

逐 group 比较 occurrence IDs。

---

## 35.2 Return-to-go invariant

从 step rewards 重算：

$$
G_i^{offline}
$$

要求与 online 相等到 numerical tolerance。

---

## 35.3 Advantage invariant

要求：

$$
A_E^{offline}\approx A_E^{online},
$$

$$
A_S^{offline}\approx A_S^{online},
$$

$$
A_{combined}^{offline}\approx A_{combined}^{online}.
$$

---

## 35.4 Action audit

随机人工检查：

```text
raw response
parsed action
executed action
admissible set
invalid flag
```

确认字段语义没有错位。

---

## 35.5 Replay validation

随机抽若干 occurrence：

```text
load same game
reset
replay prefix
```

验证：

```text
anchor key
observation
admissible set
done
```

与 archive 一致。

---

## 35.6 Logging overhead

比较 instrumented 与 recorder-disabled 的短 run：

```text
wall_clock / update
rollout time
actor update time
```

确认 recorder overhead 可接受且可量化。

---

# 36. 完整 Reference Run 的推荐执行步骤

## Phase 0：冻结实验合同

固定：

```text
GiGPO commit
ALFWorld version
model
8-GPU script
group size B=8
tasks/update
max turns
all optimizer/rollout hyperparameters
```

生成 `run.yaml` 和 manifest。

---

## Phase 1：实现 TraceRecorder

完成：

1. occurrence identity；
2. trajectory event log；
3. env pre/post step hook；
4. GiGPO group hook；
5. advantage hook；
6. PPO/profiler hook；
7. checkpoint metadata；
8. durable writer。

---

## Phase 2：Instrumentation Validation Run

运行 2-5 updates，完成第 35 节所有 invariant tests。

只有全部通过，才冻结 schema。

---

## Phase 3：冻结 schema version

例如：

```text
schema_version = gigpo_ref_v1
```

完整 run 中不再随意改字段语义。

如果需要加字段，只能 backward-compatible 添加。

---

## Phase 4：完整 1-seed Reference Run

第一条完整 run 重点是：

- 训练完整；
- raw archive 完整；
- checkpoint 完整；
- profiler 完整。

训练中每 update：

```text
flush metadata / occurrence/group tables
```

每 checkpoint interval：

```text
save actor/full checkpoint
```

---

## Phase 5：Archive Integrity Audit

完整训练结束后执行：

```text
row counts
primary key uniqueness
missing-field ratios
trajectory-step continuity
group membership integrity
checksum / shard manifest
```

核心一致性：

$$
\sum_{trajectories} num\_steps
=
\#occurrences.
$$

每个 online group 的 occurrence ID 必须存在且属于同一 task group。

---

## Phase 6：第一轮 Offline Analysis

优先执行：

### A. Figure 1

- group-size distribution；
- evidence-quality decomposition；
- within-action conflict；
- invalid-action / loop supplementary statistics。

### B. Root-count subsampling

$$
R=2\ldots8.
$$

统计 repeated/action-comparative/structural anchors。

### C. Exact Batch-ERV capacity

对多个：

$$
\tau_{BERV}
$$

profiling：

$$
C_g(R,k).
$$

### D. Competence controller simulation

测试：

```text
p0
kappa0
lambda_hist
tau_comp
```

得到 hypothetical $Q$ traces。

---

## Phase 7：建立 Replay Regression Corpus

从完整 run 固定抽样 historical occurrences，形成：

```text
dev corpus
stress corpus
```

以后所有 snapshot/replay 改动均跑同一 corpus。

---

## Phase 8：补齐最终 3 Seeds

当 recorder schema 和分析代码稳定后，新增另外两个 GiGPO seeds。

三次 run 必须：

- 相同 schema；
- 相同 instrumentation；
- 相同 profiler；
- 相同 checkpoint policy。

最终 Figure 1 与 main baseline 可报告跨 seed uncertainty。

---

# 37. 离线分析代码组织建议

推荐：

```text
analysis/
├── load_reference.py
├── validate_archive.py
├── reconstruct_groups.py
├── recompute_advantages.py
├── canonicalize_actions.py
├── analyze_group_sizes.py
├── analyze_evidence_quality.py
├── analyze_within_action_noise.py
├── subsample_roots.py
├── simulate_competence_controller.py
├── compute_exact_berv.py
├── simulate_capacity.py
├── solve_batch_dp.py
├── build_replay_corpus.py
└── make_figures.py
```

每个 script 输出 versioned parquet，而不是直接只输出 PNG。

例如：

```text
analyze_evidence_quality.py
    -> evidence_quality_v1.parquet
    -> Figure1b.pdf/png
```

---

# 38. Offline Analysis 的随机性管理

Root subsampling、tie-breaking 等分析具有随机性时，必须显式保存：

```text
analysis_seed
subsample_id
selected_root_ids
```

例如对每个 task/update/R：

```text
100 random R-subsets
```

估计 capacity distribution。

### 理由

不要让 offline subsampling 的偶然性被误认为模型 seed variance。

---

# 39. 推荐 early / middle / late 分析窗口

为了与 GiGPO-style training dynamics 对齐，可重点分析：

```text
early
middle
late
```

并在具体关键 update 周围使用小窗口聚合，例如：

$$
k-2,\ldots,k+2.
$$

最终实际 checkpoint 位置应根据 reference run 的训练进度与现有实验协议确定。

### 保存要求

因为分析阶段可能改变窗口，所以 raw data 必须每 update 保存，而不是只保存三次 summary。

---

# 40. 可以从同一 Archive 支持的后续论文图表

至少包括：

## Figure 1(a)

GiGPO step-level group-size distribution。

## Figure 1(b)

Evidence-quality decomposition。

## Appendix

- within-action variance fraction；
- mixed outcome ratio；
- invalid action ratio；
- loop-heavy anchor ratio；
- action-count imbalance；
- group statistics by task family。

## BACE preliminary diagnostics

- capacity vs root count；
- capacity vs training stage；
- hypothetical planned $Q$ vs capacity-corrected $Q$；
- Batch-ERV selected plan distribution；
- tie frequency。

---

# 41. 哪些东西 Reference Run 无法离线回答

需要明确边界。

GiGPO archive 可以告诉我们：

$$
\boxed{
\text{where BACE would want to experiment}
}
$$

但无法告诉我们：

$$
\boxed{
\text{the outcome of a new counterfactual branch}
}
$$

因此以下必须在真实 BACE rollout 中获取：

- branch actual terminal outcome；
- branch 后 posterior update；
- actual Batch-ERV realized gain；
- branch suffix 新 anchor collision；
- BACE downstream policy improvement；
- root/branch generation scheduling cost。

Reference archive 用于**预分析与机制基线**，不能替代 BACE 主实验。

---

# 42. 必须避免的数据设计错误

## 42.1 只存 aggregate metrics

只存：

```text
success rate
mean group size
loss
```

会失去几乎全部后续机制分析能力。

---

## 42.2 只存 anchor hash

必须同时保存原始 pre-action observation。

---

## 42.3 只存 parsed action

必须保存 raw response + executed action + admissible set。

---

## 42.4 不保存同 trajectory repeated occurrences

原版 GiGPO 的 grouping 允许同轨迹重复状态，因此不能提前去重。

---

## 42.5 将当前分析定义写死进 raw 数据

例如只存：

```text
is_conflict
is_effective_anchor
```

而不存 $G_i,A_i^S,u_i$。

---

## 42.6 每个 occurrence 复制完整 history

空间浪费严重，应使用 event log + prefix reference。

---

## 42.7 保存 native snapshot ID 而不保存 portable replay metadata

StateID / process pointer 等运行时对象不能作为长期恢复 source of truth。

---

# 43. 最低必须保存集

如果工程时间有限，最低不可删减项为：

## Run

```text
full resolved config
commit hashes
seed
hardware
```

## Trajectory

```text
task id / family
game file
trajectory id
terminal success/reward
num steps
A_E
```

## Occurrence

```text
step idx
pre-action observation
admissible actions
raw model response
executed action
post-action observation
step reward
terminal success
return-to-go
online group id
A_S
combined advantage
history/replay reference
```

## Group

```text
anchor key
group members
group size
mean/std return
```

## Training/System

```text
generated tokens
env steps
loss/KL/clip fraction
grad norm
phase timing
validation records
checkpoints
```

这一最低集合已经可以支撑当前最关键论文分析。

---

# 44. 推荐完整版保存集的优先级

## P0：必须

- raw observations；
- raw/executed actions；
- admissible actions；
- terminal outcomes；
- occurrence identity；
- return-to-go；
- online groups；
- $A^E,A^S,A$；
- replay metadata；
- generated tokens；
- env steps；
- phase timing；
- actor/full checkpoints；
- config/commit/seed。

## P1：强烈推荐

- token IDs；
- chosen-token old/ref logprobs；
- PPO clip fraction；
- KL；
- gradient norms；
- per-task-family validation；
- GPU utilization/memory；
- recorder overhead；
- raw conversation event log。

## P2：可选

- full top-k logits for sparse diagnostic anchors；
- detailed CPU/Ray scheduling events；
- per-step GPU timing；
- environment internal symbolic state dump。

---

# 45. 最终验收标准

Reference run 只有满足以下条件才算合格。

## 45.1 科学可重算性

从 archive 可以离线重建：

$$
\boxed{
\text{GiGPO trajectory groups + step groups + advantages}
}
$$

并与 online 完全一致到数值容差。

---

## 45.2 Figure 1 可离线生成

无需 GPU rollout，可以从 raw archive 生成：

- group-size distribution；
- evidence-quality decomposition；
- conflict / variance / invalid-action supplementary statistics。

---

## 45.3 BACE controller 可离线预模拟

可以从 natural roots 计算：

- competence history；
- hypothetical $q$ 和 $Q$；
- structural anchor pool；
- local Beta posterior；
- Exact Batch-ERV；
- information capacity；
- global DP selection。

---

## 45.4 Replay 可验证

随机 historical occurrence 可以通过同 game + action prefix 恢复，并匹配：

```text
anchor observation/key
admissible actions
done state
```

---

## 45.5 系统公平性可审计

能够严格报告：

$$
\boxed{
\text{terminal leaves}
+
\text{generated actor tokens}
+
\text{environment steps}
+
\text{wall-clock}
+
\text{GPU-hours}
}
$$

作为未来 GiGPO vs BACE 的公平效率基线。

---

# 46. 推荐实际执行顺序

最终建议按下面顺序实现：

```text
1. 固定原版 GiGPO 8-GPU ALFWorld baseline commit/config

2. 设计 schema_version = gigpo_ref_v1

3. 实现 TraceRecorder
   ├─ env pre/post step
   ├─ trajectory finalization
   ├─ online step grouping
   ├─ advantage
   ├─ PPO/logprob
   ├─ profiler
   └─ checkpoint metadata

4. 跑 2-5 updates instrumentation validation

5. 离线重建 group / G / advantage 并与 online 比较

6. 随机 occurrence 做 prefix-replay fidelity test

7. 验证 logging overhead

8. 冻结 schema

9. 完整跑第一个 instrumented GiGPO seed

10. 完成 archive integrity audit

11. 立即做第一批离线分析
    ├─ Figure 1
    ├─ root-count subsampling
    ├─ exact Batch-ERV capacity
    └─ competence-controller simulation

12. 从 archive 构造 replay regression corpus

13. 根据离线结果冻结 BACE 关键实现/参数范围

14. 补齐另外两个 GiGPO seeds

15. 正式进入 BACE 主实验
```

---

# 47. 一句话总结

这次原版 GiGPO ALFWorld 训练应被建设成一份长期可复用的 **Reference Archive**：

$$
\boxed{
\text{保存 raw trajectory / occurrence / group / optimization / system facts，}
}
$$

$$
\boxed{
\text{让 anchor evidence、Figure 1、BERV capacity、controller topology、replay fidelity 与系统比较全部可以离线重算。}
}
$$

真正最重要的不是多保存几个 summary metric，而是保证以后能够从不可变原始数据重新定义和重算我们关心的所有统计量。这样一次完整 GiGPO baseline 才能同时承担：

- 论文 baseline；
- motivation data；
- BACE feasibility analysis；
- replay/snapshot regression；
- parameter pre-screening；
- system-efficiency reference。
