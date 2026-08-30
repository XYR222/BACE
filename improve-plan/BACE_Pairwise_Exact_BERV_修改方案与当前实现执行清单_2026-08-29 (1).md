# BACE：Pairwise Exact-BERV 修改方案与当前实现执行清单
## ——从 Full Batch Open-Loop Acquisition 改为 Feedback-Aware Pairwise Acquisition

**日期**：2026-08-29  
**适用仓库**：`XYR222/BACE`  
**当前事实基线**：`handoff-refresh-20260828`  
**适用环境**：ALFWorld + Qwen2.5-1.5B + BACE-GiGPO + Exact Batch-ERV  
**本文定位**：下一阶段实现规范。只修改 **branch acquisition 的反馈粒度**，不同时修改 credit estimator、PPO、competence、anchor 定义或普通超参数。

---

# 0. 最终结论

当前 BACE 的 Exact Batch-ERV 是 **open-loop** 的：

\[
D_0
\rightarrow
\text{一次性规划全部 }Q\text{ 条 branches}
\rightarrow
\text{全部执行}
\rightarrow
\text{得到 outcomes}
\]

其中所有 branches 都仅依据 branch phase 开始时的 posterior：

\[
D_0.
\]

我们下一步研究的修改是：

\[
\boxed{
\textbf{Pairwise Exact-BERV}
}
\]

即：

\[
D_0
\rightarrow
\text{Exact 规划最多 2 条}
\rightarrow
\text{执行}
\rightarrow
\text{观察真实 outcomes}
\rightarrow
D_1
\rightarrow
\text{重新规划下一 pair}
\]

核心变化只有：

\[
\boxed{
\text{branch outcome 真正进入本 task 后续 branch acquisition decision}
}
\]

当前必须区分两个版本。

---

## P1：Pairwise-Fixed

- 当前 competence/topology 逻辑不改；
- 当前 root-side capacity correction 不改；
- capacity correction 后的最终 branch quota \(Q\) 冻结；
- root backbone / AnchorIndex / action support 冻结；
- 每轮最多执行 \(K=2\) 条 branch；
- 每轮执行后更新对应 \((z,u)\) Beta posterior；
- 下一轮重新计算 Exact BERV 和 Exact global allocation；
- 最终**必须执行满原来的 \(Q\)**；
- 不生成 fallback roots。

它只回答：

\[
\boxed{
\text{Full Batch 缺少 intermediate outcome feedback 是否造成 acquisition 损失？}
}
\]

---

## P2：Pairwise-Stopping

在 P1 的基础上再加入：

- 每轮 outcome 后重新应用当前 BERV threshold；
- 重新计算剩余 information capacity；
- 如果只剩 1 个 BERV-positive slot，则下一轮只执行 1 条；
- 如果已经没有 BERV-positive slot，则 branch phase 提前结束；
- 未使用 branch slots 统一转成 natural roots；
- fallback root 完成后**不重新打开 branch phase**。

它回答：

\[
\boxed{
\text{真实 branch outcomes 是否能告诉我们 refinement 已经足够，可以提前停止？}
}
\]

---

# 1. 当前为什么要做这个修改

## 1.1 A0 已经削弱了 “local credit 失控” 假设

当前 A0 的 2×H100 历史 run 表明：

- GiGPO late local magnitude share：约 \(57.45\%\)；
- BACE late：约 \(59.16\%\)；
- 差异只有约 \(1.71\) 个百分点；
- token-weighted 差异同样很小；
- BACE macro/local sign conflict 并没有高于 GiGPO。

因此当前没有足够证据支持：

\[
\text{“BACE 后期主要因为 local advantage 总量失控而变差”}.
\]

所以现阶段不应首先大改：

```text
step_advantage_w
local_credit_mode
branch weighting
PPO
```

---

## 1.2 当前 Full Batch 的真正局限是 open-loop

假设最终：

\[
Q=4.
\]

当前 Full Batch 一次求：

\[
(B_1,B_2,B_3,B_4)
=
\pi_{\mathrm{Full}}(D_0,Q=4).
\]

然后全部执行。

即使：

\[
B_1,B_2
\]

的实际结果已经使某个 anchor 的 posterior 明显收敛，后两条 branch 仍按旧 posterior \(D_0\) 下的计划执行。

Pairwise 改成：

\[
(B_1,B_2)=\pi_{K=2}(D_0),
\]

观察：

\[
Y_1,Y_2,
\]

更新：

\[
D_1=D_0\cup\{Y_1,Y_2\},
\]

再求：

\[
(B_3,B_4)=\pi_{K=2}(D_1).
\]

因此第二 pair 可以：

- 更换 anchor；
- 更换 action；
- 改变 global allocation；
- P2 中甚至停止执行。

---

# 2. 当前明确不做什么

本修改期间，以下内容全部冻结。

```text
Rmin
competence threshold
BERV threshold
Lmax
Beta task prior
Beta local prior
forgetting
anchor exact-match rule
observed-edge action identity
branch replay 语义
local_credit_mode
step_advantage_w
macro/local advantage
PPO hyperparameters
ratio / loss 语义
scheduler S3
```

也就是说，本轮实验**不能同时加入**：

- `action_mean`；
- \(\omega=0.5/0.75\)；
- 新 branch loss weight；
- 新 anchor rule；
- 新 root value；
- 新 root/branch utility；
- root/branch 交替式 adaptive controller。

否则无法判断 Pairwise 本身是否有效。

---

# 3. 当前明确不采用的另一个 Pairwise 版本

当前**不采用**以下 rolling root/branch pair 设计：

```text
initial roots
    ↓
BB or RR
    ↓
重建 root pool
    ↓
BB or RR
    ↓
...
```

即暂时不做：

\[
\text{Branch Pair}
\quad vs\quad
\text{Root Pair}
\]

逐轮二选一，也不在 root pair 后重新开放新的 branch origins。

原因：

1. 它同时改变 topology allocation；
2. 它会让 natural-root support 在 branch phase 中继续扩展；
3. 它不再只是 Full Batch vs Pairwise 的干净比较；
4. 当前 A0 并没有证明需要这种更大幅度修改；
5. 当前先验证 intermediate branch feedback 是否有价值更合理。

所以当前第一阶段始终采用：

\[
\boxed{
\text{现有 topology/capacity correction}
\rightarrow
\text{冻结 root support}
\rightarrow
\text{Pairwise branch acquisition}
}
\]

---

# 4. 当前 Full Batch 基线流程

总 leaf budget：

\[
B=8.
\]

当前 competence/history 首先得到计划 root/branch topology：

\[
R+Q=8.
\]

例如：

\[
R=4,\quad Q=4.
\]

当前 Exact 主线：

```text
1. 生成 R 条 natural roots

2. 构造 AnchorIndex

3. 初始化每个 (anchor, action) 的 Beta posterior

4. 计算每个 anchor:
       V(0), V(1), V(2)
       delta(1), delta(2)
       capacity

5. 若 total capacity < Q:
       R <- R + 1
       Q <- Q - 1
       新增一条 root
       重建相关结构
       重复检查

6. capacity 足够后：
       冻结最终 roots / anchor pool / action support

7. 用 quota-aware Exact DP
       一次性求 quota=Q 的 global allocation

8. 一次性冻结全部 ReplayRequests

9. 执行全部 branches

10. 进入 advantage / PPO
```

Pairwise 从 **第 7 步** 开始改。

第 1--6 步第一阶段完全不动。

---

# 5. P1：Pairwise-Fixed 完整算法

设 capacity correction 后：

\[
Q_{\mathrm{fixed}}=4.
\]

初始化：

\[
q_{\mathrm{remaining}}=4.
\]

以及：

\[
m_z=0
\]

表示 anchor \(z\) 已执行多少条 branch。

---

## 5.1 Round 1

定义：

\[
k=\min(2,q_{\mathrm{remaining}})=2.
\]

重新求一个真正的：

\[
\boxed{\text{Exact }k=2\text{ global design}}
\]

注意：

> 不能先求 quota=4 的 plan，再从里面随便取“前两条”。

一般有：

\[
\arg\max V^{(2)}
\neq
\text{quota=4 optimal plan 的任意前两条}.
\]

所以每轮都必须独立求当前 quota \(k\) 的 Exact allocation。

---

## 5.2 执行 Round 1

例如选择：

```text
B1 = (z1, a2)
B2 = (z2, a4)
```

两个 branch 并行执行。

得到：

```text
B1 -> failure
B2 -> success
```

更新：

\[
a_2:\operatorname{Beta}(\alpha,\beta)
\rightarrow
\operatorname{Beta}(\alpha,\beta+1),
\]

\[
a_4:\operatorname{Beta}(\alpha,\beta)
\rightarrow
\operatorname{Beta}(\alpha+1,\beta).
\]

同时更新：

\[
m_{z_1}\leftarrow m_{z_1}+1,
\]

\[
m_{z_2}\leftarrow m_{z_2}+1.
\]

然后：

\[
q_{\mathrm{remaining}}=2.
\]

---

## 5.3 Round 2

重新使用更新后的：

\[
D_1
\]

计算：

- 每个 anchor 的 Exact BERV；
- `values_by_size`；
- local plans；
- global allocation；
- residual per-anchor capacity。

再求：

\[
k=2
\]

的最优 Exact plan。

因此 Round 2 可以和原 Full Batch 的 frozen remainder 完全不同。

---

## 5.4 P1 的关键约束：一定执行满 \(Q\)

P1 是纯 feedback-selection 对照。

所以一旦进入 branch phase：

\[
Q_{\mathrm{fixed}}
\]

不再减少。

即使 Round 1 之后：

\[
\delta<\tau_{\mathrm{BERV}}
\]

也**不能**因此少执行 branch。

否则就同时改变了：

```text
branch selection
+
branch quantity
```

P1 的中途选择只受：

\[
\boxed{\text{residual structural support + per-anchor Lmax}}
\]

约束。

因此：

\[
\boxed{
Q_{\mathrm{executed}}=Q_{\mathrm{fixed}}
}
\]

必须是 P1 的 invariant。

---

# 6. P2：Pairwise-Stopping 完整算法

P2 前半部分和 P1 相同。

不同点是：每轮 outcome 后重新应用：

\[
\tau_{\mathrm{BERV}}=0.005
\]

以及当前 information capacity。

---

## 6.1 本轮目标数

\[
k_{\mathrm{target}}
=
\min(2,q_{\mathrm{remaining}}).
\]

更新 posterior 后重新计算：

\[
C_{\mathrm{remain}}
=
\text{当前剩余 BERV-positive capacity}.
\]

本轮实际执行：

\[
\boxed{
k_{\mathrm{actual}}
=
\min(k_{\mathrm{target}},C_{\mathrm{remain}})
}
\]

---

## 6.2 三种情况

### Case A：容量仍足够

若：

\[
C_{\mathrm{remain}}\ge2,
\]

则继续执行 2 branches。

---

### Case B：只剩 1 个高价值 slot

若：

\[
C_{\mathrm{remain}}=1,
\]

则本轮只执行 1 条。

执行后再次更新 posterior，并继续判断。

不能因为本轮不足 2 条就直接终止整个 phase。

---

### Case C：已经没有 BERV-positive slot

若：

\[
C_{\mathrm{remain}}=0,
\]

则：

```text
stop branch phase
```

令：

\[
Q_{\mathrm{unused}}
=
q_{\mathrm{remaining}}.
\]

后续统一补：

\[
Q_{\mathrm{unused}}
\]

条 natural roots。

---

# 7. P2 的 fallback 规则

P2 必须采用：

\[
\boxed{\text{one-way fallback}}
\]

例如：

```text
4 initial roots
↓
2 branches
↓
BERV capacity = 0
↓
2 fallback roots
↓
PPO
```

不能：

```text
4R
↓
2B
↓
2R
↓
fallback roots 形成新高-BERV anchor
↓
重新 branch
```

也就是说，一旦 branch phase 停止：

\[
\boxed{
\text{本 task 当前 rollout group 不重新开启 branch phase}
}
\]

原因是第一阶段只研究：

\[
\text{branch outcome feedback}
\]

而不是 root/branch rolling allocation。

---

# 8. 一个完整例子

总预算：

\[
B=8.
\]

capacity correction 后：

\[
R=4,\qquad Q=4.
\]

Natural roots 形成：

\[
z_1,z_2.
\]

假设：

\[
a_1\sim Beta(5,2),
\qquad
a_2\sim Beta(3,2),
\]

\[
a_3\sim Beta(4,2),
\qquad
a_4\sim Beta(3,3).
\]

---

## Full Batch

quota=4 的 Exact solver 一次决定：

```text
z1 -> [a2, a2]
z2 -> [a4, a4]
```

全部执行。

---

## P1 Round 1

重新解 quota=2：

```text
B1 = (z1,a2)
B2 = (z2,a4)
```

结果：

```text
B1 = failure
B2 = success
```

更新：

\[
a_2:Beta(3,2)\to Beta(3,3),
\]

\[
a_4:Beta(3,3)\to Beta(4,3).
\]

Round 2 重新求 quota=2。

可能得到：

```text
B3 = (z1,a1)
B4 = (z2,a4)
```

最终仍：

\[
4R+4B.
\]

收益只来自：

\[
\boxed{\text{后两条 branch 根据前两条真实结果重选}}
\]

---

## P2 同样的 Round 1

如果 Round 1 后重新计算得到：

\[
C_{\mathrm{remain}}=0,
\]

则直接停止。

剩余：

\[
2
\]

个 slots 转为 natural roots。

最终：

\[
\boxed{6R+2B}.
\]

如果：

\[
C_{\mathrm{remain}}=1,
\]

则先再做 1 branch，之后可能得到：

\[
\boxed{5R+3B}.
\]

所以 P2 的最终 branch 数可以小于最初 quota。

---

# 9. Frozen Support：第一版必须严格保持

Pairwise 第一版必须冻结 acquisition support。

branch phase 开始后：

```text
AnchorIndex 不新增 anchor
candidate action set 不新增 action
branch suffix 不形成新的 branch origin
branch suffix 中的新 action 不加入 acquisition support
```

branch outcome 只允许更新：

\[
\boxed{
\text{已有 }(z,u)\text{ Beta posterior}
}
\]

原因：

如果 outcome 后同时扩展 support：

\[
\text{feedback}
+
\text{new anchor}
+
\text{new action}
\]

三件事会同时改变，实验无法解释。

---

# 10. \(L_{\max}=2\) 的处理

当前：

\[
L_{\max}=2.
\]

Pairwise 中它必须是**跨 round 累计**的。

定义：

\[
m_z=\text{已执行 branch 数}.
\]

剩余：

\[
L_z^{\mathrm{remain}}
=
L_{\max}-m_z.
\]

例如 Round 1：

```text
z1 used 1
z2 used 1
```

则 Round 2：

```text
z1 residual cap = 1
z2 residual cap = 1
```

绝不能每轮重新恢复成 2。

---

# 11. Exact BERV 数学不改

以下全部不改：

- Beta-Bernoulli posterior；
- Beta-Binomial exact finite enumeration；
- local `values_by_size`；
- local optimal plans；
- Exact global objective；
- quota-aware exact DP；
- stable tie handling。

Pairwise 只是把原来的：

\[
Q
\]

改成每轮：

\[
k=\min(2,q_{\mathrm{remaining}})
\]

重新调用同一个 Exact machinery。

---

# 12. 推荐代码结构

不要直接破坏：

```python
ExactBatchErvCoordinator
```

建议新增：

```text
ExactBatchErvCoordinator
    └── current Full Batch baseline

PairwiseExactErvCoordinator
    ├── mode="fixed"
    └── mode="stopping"
```

这样 current baseline 永远可回退。

---

# 13. 推荐配置

```yaml
algorithm:
  bace:
    pairwise:
      enabled: false
      mode: fixed
      batch_size: 2
```

---

## Current Full Batch

```yaml
pairwise:
  enabled: false
```

---

## P1

```yaml
pairwise:
  enabled: true
  mode: fixed
  batch_size: 2
```

---

## P2

```yaml
pairwise:
  enabled: true
  mode: stopping
  batch_size: 2
```

第一版只允许：

\[
K=2.
\]

不要现在同时 sweep：

```text
K=1 / 2 / 4
```

---

# 14. 当前仓库建议修改文件

主要：

```text
verl-agent-src/recipe/bace_gigpo/coordinator.py
verl-agent-src/recipe/bace_gigpo/rollout_collector.py
verl-agent-src/recipe/bace_gigpo/batch_erv.py
verl-agent-src/recipe/bace_gigpo/types.py
tests/bace_gigpo/
```

原则上不修改：

```text
advantage.py
anchor_index.py
posterior.py
gigpo/core_gigpo.py
PPO trainer / loss
```

---

# 15. `PairwiseTaskState`

建议：

```python
@dataclass
class PairwiseTaskState:
    task_id: str

    planned_quota: int
    remaining_quota: int

    posteriors: dict
    used_slots_by_anchor: dict[str, int]

    completed_branch_count: int = 0
    round_index: int = 0

    stopped: bool = False
    stop_reason: str | None = None

    fallback_root_count: int = 0
```

还应保留：

```text
initial/frozen anchor support
decision_task_key
policy_update_id
```

用于稳定 tie identity。

---

# 16. Coordinator 推荐接口

初始化：

```python
coordinator.initialize(
    roots=...,
    branch_quota_by_task=...,
    prior_mean_by_task=...,
)
```

初始化时：

1. 构造 frozen `AnchorIndex`；
2. 初始化局部 posterior；
3. 保存 `planned_quota`；
4. 设置：
   ```text
   remaining_quota = planned_quota
   ```
5. 不一次性生成所有 branch requests。

---

每轮：

```python
requests = coordinator.build_next_round_requests()
```

执行以后：

```python
coordinator.update_from_branch(
    request,
    success=...,
)
```

然后：

```python
coordinator.finalize_round()
```

终止判断：

```python
coordinator.has_pending_acquisition()
```

P2 额外读取：

```python
coordinator.fallback_root_count_by_task
```

---

# 17. `batch_erv.py` 需要支持 residual cap

Pairwise Round 2 时需要考虑：

\[
L_z^{\mathrm{remain}}.
\]

推荐：

```python
residual_cap = (
    lmax
    - used_slots_by_anchor.get(anchor_id, 0)
)
```

当前 anchor 的设计只允许：

```text
size <= residual_cap
```

---

## P1

初始 threshold 用于当前 topology/capacity correction。

进入 branch phase 后：

```text
不再用 threshold 减少剩余 quota
```

只在 residual structural support 中选择 value 最大的 Exact plan。

---

## P2

每轮重新使用：

\[
\tau_{\mathrm{BERV}}.
\]

因此：

```text
size <= residual structural cap
size <= current BERV-positive capacity
```

---

# 18. 多 task 必须 Global Round-Wise Packing

这是工程实现里最重要的性能约束。

绝不能：

```text
task1 round1
task1 round2
task2 round1
task2 round2
...
```

正确方式：

---

## Global Round 1

对所有 active tasks：

```python
all_requests = []

for task in active_tasks:
    all_requests.extend(
        coordinator.build_task_round_requests(task)
    )
```

把所有 task 的第一 pair requests 合并。

然后使用当前 S3：

```text
main_reuse
selected_worker
root_active_executor=true
```

统一执行。

---

## Global Round 2

所有 task 已经分别更新 posterior。

有些：

- quota 已耗尽；
- P2 已 stop；
- 只剩 1；
- 仍需要 2。

再统一构造下一轮所有 requests。

所以 Pairwise 增加的是：

\[
\boxed{\text{全局 acquisition rounds}}
\]

而不是：

\[
\boxed{\text{task 数}\times acquisition rounds}.
\]

---

# 19. `rollout_collector.py` 推荐流程

伪代码：

```python
coordinator.initialize(...)

while coordinator.has_pending_acquisition():

    requests = coordinator.build_next_round_requests()

    if not requests:
        break

    valid, origin_output, suffix_output, rewards = \
        self._execute_branch_round(requests)

    for request, reward in zip(valid, rewards):
        coordinator.update_from_branch(
            request,
            success=(reward > 0),
        )

    coordinator.finalize_round()
```

P2 全部 branch rounds 完成后：

```python
fallback_counts = (
    coordinator.fallback_root_count_by_task
)
```

然后统一 packed 生成 fallback roots。

---

# 20. P2 fallback roots 的调度方式

不要 task 一 stop 就立刻小 batch 生成 root。

例如：

```text
task A -> 2 fallback roots
task B -> 0
task C -> 1
task D -> 2
```

先累计。

所有 Pairwise branch rounds 结束后：

```text
Fallback Root Stage
```

统一 pack：

```text
A×2
C×1
D×2
```

执行。

这样避免大量极小 root generation waves。

---

# 21. Fallback root 的训练语义

fallback root 必须是普通 natural root：

```text
source_type = root
```

必须：

- 从 initial state 正常 rollout；
- 使用同一个 frozen \(\pi_{\mathrm{old}}\)；
- 正常分配 `traj_uid`；
- 正常进入 final occurrence table；
- 正常进入 macro advantage；
- 正常进入 local grouping；
- 正常进入 PPO；
- 正常计入 family natural-root history。

但是按照 one-way fallback：

```text
不再作为当前 step 的新 branch support
```

---

# 22. Family competence history

P2 中最终 natural roots：

\[
\text{initial roots}
+
\text{fallback roots}.
\]

因此当 step 完成后更新 family history 时，必须使用：

\[
\boxed{\text{最终全部 natural-root outcomes}}
\]

不能只用 branch phase 前的 initial roots。

这是 P2 必须新增的 regression test。

---

# 23. Tie-breaking 与可复现性

Pairwise 每轮都重新调用 Exact solver。

必须继续使用：

```text
tie_break_identity_mode=stable_v1
```

每轮 identity 至少包含：

```text
policy_update_id
decision_task_key
pairwise_round_index
purpose
anchor_id（local tie 时）
```

例如：

```python
choose_uniform(
    candidates,
    policy_update_id,
    decision_task_key,
    pairwise_round_index,
    "global_allocation",
)
```

不要把：

```text
random UUID
worker arrival order
request UUID
```

作为 decision identity。

---

# 24. Artifacts 必须新增 Round 概念

当前已有：

```text
acquisition_rounds.jsonl
posterior_snapshots.jsonl
```

Pairwise 每 round 至少记录：

```json
{
  "policy_update_id": 101,
  "task_id": "...",
  "round_index": 0,
  "mode": "fixed",

  "planned_quota": 4,
  "remaining_quota_before": 4,

  "k_target": 2,
  "k_actual": 2,

  "used_slots_by_anchor_before": {},

  "posterior_before": {},
  "selected_allocation": {},
  "selected_local_plans": {},
  "requests": []
}
```

执行后：

```json
{
  "policy_update_id": 101,
  "task_id": "...",
  "round_index": 0,

  "completed_outcomes": [],
  "posterior_after": {},

  "used_slots_by_anchor_after": {},

  "remaining_quota_after": 2,

  "stopped": false,
  "stop_reason": null
}
```

P2 stop：

```json
{
  "stopped": true,
  "stop_reason": "NO_BERV_POSITIVE_CAPACITY",

  "planned_quota": 4,
  "executed_branch_count": 2,
  "unused_branch_slots": 2,
  "fallback_root_count": 2
}
```

---

# 25. TensorBoard / summary 指标

至少增加：

```text
bace/pairwise/rounds_mean
bace/pairwise/rounds_max

bace/pairwise/planned_branch_count
bace/pairwise/executed_branch_count

bace/pairwise/fallback_root_count
bace/pairwise/early_stop_task_ratio

bace/pairwise/replan_task_ratio

bace/pairwise/round1_branch_count
bace/pairwise/round2_branch_count

bace/pairwise/post_round_capacity_mean
```

P1 invariant：

```text
planned_branch_count == executed_branch_count
fallback_root_count == 0
```

---

# 26. `replan_task_ratio`

建议 Pairwise 运行时额外保留一个 diagnostic：

初始化 branch phase 时仍计算：

```text
counterfactual frozen Full Batch plan
```

但只用于比较，不执行。

Round 1 outcome 后，新的 pairwise remainder 与 frozen remainder 比较。

若变化：

```text
replanned = true
```

统计：

\[
\boxed{
\rho_{\mathrm{replan}}
=
P(\text{feedback changes remaining plan})
}
\]

这样真实训练可以验证 A1 offline 的判断。

---

# 27. 必须的 Unit Tests

## Test 1：Pairwise disabled

```text
pairwise.enabled=false
```

必须完全走当前 Full Batch 路径。

---

## Test 2：Q=0

不产生 branch。

---

## Test 3：Q=1

\[
k=1.
\]

在同一 Exact objective / tie semantics 下，应与 Full Batch 等价。

---

## Test 4：Q=2

只有一轮，没有 replan opportunity。

P1 与 Full Batch 应等价。

---

## Test 5：Q=4 且 feedback 改变第二轮

构造 synthetic Beta，使：

```text
Round 1 outcome 后
Round 2 allocation != frozen Full Batch remainder
```

验证 posterior feedback 被真正使用。

---

## Test 6：P1 执行满 quota

即使 Round 1 后所有 updated marginal：

\[
<0.005,
\]

仍满足：

```text
executed == planned
fallback == 0
```

---

## Test 7：P2 zero-capacity stop

Round 1 后：

\[
C=0.
\]

验证：

```text
executed < planned
fallback = planned - executed
```

---

## Test 8：P2 capacity=1

```text
k_target = 2
C = 1
```

必须：

```text
k_actual = 1
```

执行后继续重新判断。

---

## Test 9：Lmax 跨轮累计

同 anchor 的累计：

\[
m_z\le L_{\max}
\]

在所有 rounds 后都成立。

---

## Test 10：Frozen Support

branch suffix 出现新 observation/action：

```text
下一轮 acquisition support 不新增
```

---

## Test 11：One-Way Fallback

P2 stop 后补 root，即使新 root 形成高-BERV anchor：

```text
不重新进入 branch phase
```

---

## Test 12：Budget invariant

每个 task：

\[
\boxed{
R_{\mathrm{final}}
+
Q_{\mathrm{executed}}
=
B
}
\]

P1：

\[
R_{\mathrm{final}}=R_{\mathrm{initial}}.
\]

P2：

\[
R_{\mathrm{final}}
=
R_{\mathrm{initial}}
+
Q_{\mathrm{unused}}.
\]

---

## Test 13：Family history

P2 fallback roots 必须进入最终 natural-root family history。

---

## Test 14：Determinism

相同：

```text
seed
policy_update_id
decision_task_key
posterior
outcomes
```

必须得到相同 pairwise plans。

---

# 28. CPU regression

代码完成后先跑：

```bash
pytest -q tests/bace_gigpo tests/trainer/ppo/test_metric_utils.py
```

要求：

1. 当前所有 Full Batch 测试继续通过；
2. Pairwise 新测试全部通过；
3. `pairwise.enabled=false` 的 baseline 行为不发生 drift。

---

# 29. 真实 smoke test

先做：

\[
2\text{--}5
\]

个 training steps。

固定当前 S3：

```text
main_reuse
root_active_executor=true
selected_worker
stable_v1
```

逐项检查：

1. Round 1 requests 正确；
2. branch outcomes 正确更新 Beta posterior；
3. Round 2 的 plan 使用 updated posterior；
4. P1 executed = planned；
5. P2 stop/fallback 正确；
6. final source types 正确；
7. advantage reconstruction 仍正确；
8. PPO 正常；
9. checkpoint 正常；
10. artifact validator 正常；
11. walltime 没有出现 task-wise 串行化。

---

# 30. A1 与实现之间的顺序

当前推荐：

\[
\boxed{
\textbf{先完成 A1 Offline Audit，再决定是否投入 P1/P2 正式实现。}
}
\]

A1 至少输出：

### Pairwise opportunity

\[
P(Q\ge3).
\]

### Replanning rate

\[
P(
\text{第一 pair 后剩余 optimal allocation 改变}
).
\]

### Feedback gain

\[
\Delta V_{\mathrm{feedback}}.
\]

### Capacity shortfall

\[
P(
C_{\mathrm{remain}}<Q_{\mathrm{remain}}
).
\]

---

# 31. A1 → 实现决策

## P1 Go

若 \(Q\ge3\) 的 task 中：

```text
replan rate >= 25%
```

并且 feedback gain 不是数值上接近 0：

\[
\rightarrow
\text{优先实现 P1}.
\]

这个 25% 只是工程筛选参考，不是理论常数。

---

## P2 Go

若第一 pair 后：

```text
adaptive capacity shortfall
或 zero-capacity
```

较常见，例如：

\[
\ge 15\%-20\%,
\]

则 P2 值得实现。

---

## Hold

如果：

```text
replan 10--25%
```

且 feedback gain 很小：

先只做短程 P1 continuation。

---

## No-Go

如果：

```text
replan < 10%
feedback gain ≈ 0
capacity shortfall ≈ 0
```

则不要继续增加 Pairwise 工程复杂度。

---

# 32. 当前推荐开发顺序

严格按：

```text
Step 1
完成 A1 offline

Step 2
若 A1 支持 P1：
实现 PairwiseExactErvCoordinator(mode=fixed)

Step 3
完成 P1 unit tests

Step 4
P1 2--5 step smoke

Step 5
若 A1 同时支持 stopping：
实现 mode=stopping

Step 6
完成 fallback-root / family-history tests

Step 7
P2 2--5 step smoke

Step 8
H20 上从共同 step-100 checkpoint 做 continuation

Step 9
比较 C0/C1/C2

Step 10
只对 winner 做 from-scratch / multi-seed
```

---

# 33. H20 第一轮正式实验

不建议直接从 0 跑 150。

优先从同一：

\[
\boxed{\text{step 100 checkpoint}}
\]

分叉到 step 150。

必须恢复：

```text
actor
optimizer
scheduler
dataloader
family competence history
RNG
```

且切换点必须在完整 PPO update boundary。

---

## C0：Current Full Batch

```yaml
pairwise:
  enabled: false
```

---

## C1：P1 Pairwise-Fixed

```yaml
pairwise:
  enabled: true
  mode: fixed
  batch_size: 2
```

---

## C2：P2 Pairwise-Stopping

```yaml
pairwise:
  enabled: true
  mode: stopping
  batch_size: 2
```

其余配置完全相同。

---

# 34. 为什么必须同时有 C1

只比较：

```text
C0 vs C2
```

如果 C2 变好，无法知道：

- 是 branch feedback 让选址更好；
- 还是 branch 数减少、root 数增加。

所以必须有：

\[
\boxed{
C0\rightarrow C1\rightarrow C2
}
\]

做机制分解。

---

# 35. 结果解释

## Case A

\[
C1>C0,
\qquad
C2\approx C1.
\]

说明：

\[
\boxed{\text{主要收益来自 feedback-aware selection}}
\]

主方法优先 P1。

---

## Case B

\[
C1\approx C0,
\qquad
C2>C1.
\]

说明：

\[
\boxed{\text{主要问题是 refinement over-consumption}}
\]

主方法考虑 P2。

---

## Case C

\[
C2>C1>C0.
\]

说明：

- feedback selection 有价值；
- feedback stopping 还有额外价值。

P2 可作为最终主候选。

---

## Case D

\[
C0\approx C1\approx C2.
\]

Pairwise 不是 late gap 的主要来源。

停止继续堆 Pairwise 复杂度，回到：

```text
lineage / effective sample size
within-action noise
action-level credit
breadth/refinement allocation
```

---

## Case E

\[
C1<C0.
\]

说明当前 Full Batch 的 joint planning 本身可能比 myopic \(K=2\) 更有价值。

Pairwise 不应因为“更 adaptive”就默认保留。

---

# 36. 正式实验必须同时看的指标

## Performance

```text
validation success
family validation success
root success
branch success
AUC
```

---

## Acquisition

```text
pairwise round count
replan task ratio
BERV before/after each round
capacity before/after each round
planned vs executed branches
```

---

## Topology

P1：

```text
R/Q 必须与 C0 相同
```

P2：

```text
planned Q
executed Q
fallback root count
early-stop ratio
```

---

## Credit

继续复用 A0：

```text
local coverage
local magnitude share
token-weighted local share
macro/local sign conflict
```

Pairwise 不应该在无意中大幅改变下游 credit semantics。

---

## System

```text
root generation seconds
branch generation seconds
acquisition computation seconds
pairwise rounds
PPO seconds
step walltime
GPU utilization
generated tokens
```

---

# 37. Runtime 控制

Pairwise 最大工程风险是额外 barrier。

但不允许实现成 task-wise sequential。

必须：

\[
\boxed{
\text{global round-wise packed execution}
}
\]

对于所有 task 的 Round 1：

```text
全部 pair requests 一起执行
```

Round 1 完成后再统一进入 Round 2。

在当前 \(Q\le4\) 的主要后期形态下，通常最多增加到约 2 个 branch rounds。

P2 若提前停止较多，也可能因为实际 branch 变少抵消部分 barrier 成本。

最终 runtime 必须实测，不能提前假设。

---

# 38. 当前实施优先级

当前顺序固定为：

\[
\boxed{
A1
>
P1\text{ implementation}
>
P1\text{ smoke}
>
P2\text{ implementation if justified}
>
C0/C1/C2\text{ H20 continuation}
}
\]

当前**不建议**优先：

```text
step_advantage_w sweep
action_mean
Rmin sweep
BERV threshold sweep
Pairwise K sweep
root-value design
rolling RR/BB controller
```

这些都应放在 Pairwise 机制问题被回答之后。

---

# 39. 最小实现版本

如果希望先最快验证 P1，最小改动只有四件事：

1. 新增 `PairwiseExactErvCoordinator(mode=fixed)`；
2. `batch_erv.py` 支持 residual per-anchor cap；
3. `rollout_collector.py` 改成 global round-wise branch loop；
4. 增加 round artifacts 和核心 tests。

不改：

```text
topology planner
capacity correction
anchor_index
advantage
PPO
fallback roots
```

因为 P1 没有 fallback。

这应当是第一阶段最低风险实现。

---

# 40. 一页式伪代码

## P1

```text
# Existing topology phase
R, Q <- current competence + capacity correction
roots <- generate R natural roots

# Freeze support
anchor_index <- build from roots
posterior <- initialize
used[z] <- 0
remaining <- Q

while remaining > 0:

    k <- min(2, remaining)

    designs <- recompute Exact BERV using current posterior
    designs <- restrict by residual Lmax:
               Lmax - used[z]

    # FIXED mode:
    # do not re-apply threshold to reduce remaining quota
    requests <- exact_global_plan(designs, quota=k)

    execute all requests in this GLOBAL round

    for request outcome:
        update corresponding Beta posterior
        used[request.anchor] += 1

    remaining -= len(requests)

PPO(current unchanged)
```

---

## P2

```text
# Same existing topology phase
R, Q_plan <- current competence + capacity correction
roots <- generate R natural roots

freeze support

posterior <- initialize
used[z] <- 0
remaining <- Q_plan
fallback <- 0

while remaining > 0:

    k_target <- min(2, remaining)

    designs <- recompute Exact BERV
    designs <- restrict by residual Lmax
    C <- current threshold-positive residual capacity

    k_actual <- min(k_target, C)

    if k_actual == 0:
        fallback += remaining
        break

    requests <- exact_global_plan(
        threshold-valid designs,
        quota=k_actual
    )

    execute current GLOBAL round

    for request outcome:
        update posterior
        used[anchor] += 1

    remaining -= len(requests)

# one-way fallback
generate `fallback` packed natural roots

# do NOT reopen branch phase

update final natural-root family history
PPO(current unchanged)
```

---

# 41. 最终实施 Checklist

## A1 前

- [ ] 不改 Pairwise 代码；
- [ ] 完成 A1 realized replan audit；
- [ ] 完成 model-based Pairwise expected feedback audit；
- [ ] 输出 replan rate；
- [ ] 输出 feedback gain；
- [ ] 输出 capacity shortfall。

## P1 开发

- [ ] 新增 config；
- [ ] 新增 `PairwiseTaskState`；
- [ ] 新增 `PairwiseExactErvCoordinator`；
- [ ] 支持 residual \(L_{\max}\)；
- [ ] 实现 global round-wise packed loop；
- [ ] 加 stable round tie identity；
- [ ] 保持 current coordinator 完全不变；
- [ ] 加 P1 tests；
- [ ] 加 round artifacts；
- [ ] 2--5 step smoke。

## P2 开发

- [ ] 仅 A1 支持时进行；
- [ ] 每轮重算 threshold capacity；
- [ ] 支持 `k_actual=0/1/2`；
- [ ] 记录 unused quota；
- [ ] packed fallback roots；
- [ ] one-way fallback；
- [ ] fallback roots 计入 family history；
- [ ] P2 tests；
- [ ] 2--5 step smoke。

## H20 实验

- [ ] C0 current；
- [ ] C1 P1；
- [ ] C2 P2；
- [ ] 同 step-100 checkpoint；
- [ ] 同 seed；
- [ ] 同 scheduler；
- [ ] 同 H20 数；
- [ ] 同 PPO/advantage/threshold；
- [ ] 跑到 step 150；
- [ ] 比较 accuracy + acquisition + runtime。

---

# 42. 最终原则

这次修改必须保持一句话：

\[
\boxed{
\textbf{我们现在不是重新设计 BACE 的 root/branch controller，}
}
\]

\[
\boxed{
\textbf{而是在保持现有 topology 和 optimizer 不变的前提下，}
}
\]

\[
\boxed{
\textbf{测试 Full Batch 的 open-loop acquisition 是否应被 feedback-aware Pairwise acquisition 替代。}
}
\]

第一优先级是：

\[
\boxed{\text{P1 Pairwise-Fixed}}
\]

因为它是最干净的机制实验。

只有 A1 与 P1 结果证明：

\[
\text{branch outcome 后经常出现真实 capacity collapse / refinement redundancy}
\]

时，才把：

\[
\boxed{\text{P2 Pairwise-Stopping}}
\]

升级为主候选。

如果 P1/P2 均无收益，则停止继续修改 Pairwise，把研究资源转移到其它更可能解释 late gap 的方向。
