# BACE-GiGPO Branch Rollout Pipeline：Inactive Branch 与 Double Replay 问题说明

> **状态提示（2026-08-27）：历史问题说明。** 本文描述的是旧 dense 执行路径。当前推荐的 `selected_worker` 已实现 active suffix compaction，并在已验证的 selected worker 上直接执行 origin transition；`legacy_dense` 仍保留作兼容对照。当前边界见 [`docs/09_逻辑模式与兼容边界.md`](../docs/09_逻辑模式与兼容边界.md)。

> 版本：2026-08-19
> 适用范围：`XYR222/BACE` 当前 Exact Batch-ERV / ALFWorld branch rollout pipeline
> 目的：说明并规范两个已经确认存在的工程问题：
>
> 1. **inactive branch 仍参与 LLM generation 与 environment step**；
> 2. **一个 branch prefix 在正常路径下通常会 mechanical replay 两次**。
>
> 两个问题都不会改变 BACE 的 Exact Batch-ERV 数学定义，也不要求修改 root/branch topology、anchor、posterior 或 acquisition objective。它们主要属于 **branch execution pipeline 的系统效率与计量问题**。

---

# 1. 总结结论

当前 branch pipeline 的核心统计语义总体正确，但执行层采用了较为“dense”的批处理方式：

- 一个 physical branch wave 内，即使某些 branch 已经 terminal 或耗尽 remaining horizon，它们仍暂时保留在 batch 中；
- branch replay validation 与正式 branch-origin execution 是两个独立阶段，因此 happy path 下相同 prefix 会先 replay 一次用于验证，再 replay 一次用于正式执行。

因此当前实现存在：

$$
\boxed{
\text{logical branch activity}
\neq
\text{physical generation / env-step activity}
}
$$

以及：

$$
\boxed{
\text{restore-and-validate}
\;
\text{与}
\;
\text{restore-and-execute}
\text{被分成两个完整 replay 阶段}
}
$$

准确评级如下。

| 问题 | 是否确认存在 | 是否直接污染 PPO 数据 | 主要影响 | 推荐优先级 |
|---|---|---|---|---|
| inactive branch 仍 generate | 是 | 通常不会 | GPU generation 浪费 | P1 |
| inactive branch 仍 env.step | 是 | 通常不会 | CPU/env 浪费，terminal-after-done 风险 | P1 |
| happy path prefix replay 两次 | 是 | 不会 | mechanical replay 开销增加 | P1 |
| retry / filtering 后再次 replay | 是 | 当前有工程必要性 | 用于 request-worker realignment | 保留或重构 |
| 第二次 replay 成本未完整统计 | 是 | 不会 | 论文成本指标可能低估 | P1，建议优先修 |

---

# 2. 问题一：Inactive Branch 仍然参与 Generation 与 Environment Step

## 2.1 当前逻辑

在 branch origin 执行完成后，每条 branch 进入 fresh suffix rollout。

设 physical wave 中有 $N$ 条 branch，在 suffix step $t$ 时定义：

$$
\text{active}_i^{(t)}
=
\mathbf 1[\neg done_i]
\cdot
\mathbf 1[t < H_i^{\mathrm{remain}}].
$$

当前代码会构造：

```text
active_masks = (~is_done) & horizon_active
```

并在所有 branch 都 inactive 时退出：

```text
if not active_masks.any():
    break
```

这部分逻辑本身没有问题。

真正的问题是：`active_masks` 目前主要用于**事后统计和训练数据过滤**，而不是用于**事前压缩 LLM generation batch 与 environment step batch**。

因此，只要一个 physical wave 中还有至少一条 branch active，代码仍然会对整个 `requests` batch 执行：

```text
preprocess_batch(...)
→ generate_sequences(...)
→ decode(...)
→ branch_envs.step(text_actions)
```

即使其中某些 branch 已经：

- terminal；或
- remaining horizon 已耗尽。

---

# 3. 一个直观例子

假设一个 physical wave 有四条 branch：

```text
B1
B2
B3
B4
```

执行若干 suffix steps 后：

```text
B1: active
B2: terminal
B3: active
B4: active
```

于是：

$$
active\_masks=[1,0,1,1].
$$

理想执行应该是：

```text
只构造 B1 / B3 / B4 的 prompt
        ↓
只生成 3 个 responses
        ↓
只 step B1 / B3 / B4 对应的 env workers
```

但是当前 dense 实现更接近：

```text
构造 B1 / B2 / B3 / B4 的 prompt
        ↓
生成 4 个 responses
        ↓
step 4 个 env workers
        ↓
最后通过 active_masks 丢掉 B2 的训练 row
```

因此：

$$
\boxed{
B2\text{ 虽然逻辑上已经结束，物理上仍可能继续 generate 和 step。}
}
$$

---

# 4. 为什么它通常不会直接污染训练数据

当前 suffix rows 最后会根据 `active_masks` 过滤。

逻辑上类似：

```text
for row in rows:
    if row["active_masks"]:
        effective_rows.append(row)
```

因此 inactive branch 额外生成的 response 一般不会进入最终 PPO training batch。

也就是说，这个问题通常不是：

```text
错误 suffix token 被拿去训练
```

而是：

```text
错误地为已经不需要继续执行的 slot 消耗了 generation / env-step 资源
```

所以当前应将其分类为：

$$
\boxed{\text{rollout scheduler inefficiency}}
$$

而不是主要的 statistical correctness bug。

---

# 5. Inactive Branch 的两种来源

## 5.1 已经 terminal

例如：

$$
done_i=True.
$$

此时该 branch 已经得到 terminal leaf outcome，不应该再产生 suffix token 或环境 transition。

但如果 wave 中还有其他 active branches，当前 dense batch 仍可能继续对该 worker 调用 generation 和 `env.step()`。

这种行为至少产生额外开销，并依赖底层环境对 terminal-after-done step 的容忍。

当前不能简单断言 ALFWorld 一定会因此报错，但系统没有必要依赖这种行为。

---

## 5.2 Remaining horizon 已耗尽

另一种情况是：

$$
done_i=False,
$$

但是：

$$
t\ge H_i^{\mathrm{remain}}.
$$

此时按照 BACE branch 定义，该 branch 的合法 suffix budget 已经结束。

它同样应该被从 generation / environment execution 中移除。

当前实现虽然不会把超 horizon 的 row 加入训练，但仍可能生成并 step，因此同样造成无效计算。

---

# 6. Inactive Branch 的真实性能影响

假设一个 16-branch wave 的 suffix 长度差异很大：

```text
15, 14, 13, 12,
10, 10, 8, 8,
6, 5, 5, 4,
3, 2, 1, 1
```

理想 generation work 更接近：

$$
\sum_i H_i^{\mathrm{suffix}}.
$$

dense 执行则更接近：

$$
N\cdot \max_i H_i^{\mathrm{suffix}},
$$

虽然真实 token cost 还取决于每次 response 长度、padding、vLLM scheduling 等，因此不能简单把两者当成严格 FLOPs 比值，但可以确定：

> **当前会把一部分已经 inactive 的 sequences 继续提交给 rollout engine。**

当 branch suffix 长度方差较大时，这一浪费可能明显。

---

# 7. 问题一的推荐修复：Active-Set Compaction

推荐把 suffix rollout 从固定 dense batch 改成：

$$
\boxed{
\text{persistent worker slots}
+
\text{active-set compaction}
}
$$

当前环境层已经具备类似接口：

```text
get_observations_selected(worker_indices)
replay_selected(worker_indices, requests)
step_selected(worker_indices, text_actions)
```

因此不需要重写整个 ALFWorld 环境。

---

## 7.1 推荐执行方式

physical wave 起始时：

```text
worker 0 → branch B0
worker 1 → branch B1
...
worker 15 → branch B15
```

第一个 suffix step：

```text
active workers = [0,1,...,15]
```

若若干 branch terminal：

```text
worker 2 done
worker 7 done
worker 11 done
```

下一步改为：

```text
active workers = [0,1,3,4,5,6,8,9,10,12,13,14,15]
```

然后只对这些 worker：

```text
构造 prompt
→ LLM generate
→ step_selected(active_worker_indices, actions)
```

这样：

$$
\boxed{
\text{logical active set}
=
\text{physical generation set}
=
\text{physical env-step set}
}
$$

---

# 8. Active-Set Compaction 的注意事项

实现时必须保留 logical branch identity。

不能因为 batch 被压缩，就让：

```text
row index
```

取代：

```text
branch_id / worker_slot_id
```

作为身份。

建议每条 active row 显式维护：

```text
branch_id
request_id
task_batch_index
worker_slot_id
suffix_step
remaining_horizon
```

这样即使 active batch 大小从：

```text
16 → 11 → 7 → 3
```

也不会破坏 branch lineage。

---

# 9. 问题二：一个 Branch Prefix 正常会 Replay 两次

这个问题也已经确认存在。

当前 branch execution 把：

1. restore-and-validate；
2. restore-and-execute；

设计成两个独立阶段。

因此 happy path 下同一个 mechanical prefix 一般会真正执行两次。

---

# 10. 第一次 Replay：Restore + Validate

进入一个 physical chunk 后，首先调用：

```text
ReplayAdapter.replay_and_validate(active_requests)
```

其内部会调用：

```text
manager.replay(requests)
```

这个 `replay()` 不是 metadata 模拟，而是真的执行：

```text
reset same game
→ replay prefix actions
→ restore target anchor
```

随后检查：

- restored observation；
- admissible action set；
- done；
- prompt token identity；
- 以及相关 replay invariant。

因此，第一次 replay 完成后，如果 request valid，则 worker 已经真实停在目标 branch anchor。

---

# 11. 目标 origin action 并没有在第一次 replay 中执行

假设 natural root 中目标 branch edge 是：

$$
(s_5,a_5).
$$

ReplayRequest 的 mechanical prefix 只包含：

$$
a_0,a_1,a_2,a_3,a_4.
$$

因此第一次 replay 后 worker 停在：

$$
s_5,
$$

并没有执行：

$$
a_5.
$$

这点非常重要。

所以 double replay 并不意味着 copied origin action 被执行两次。

---

# 12. 第二次 Replay：Restore + Execute

第一次 replay validation 完成后，当前 `_execute_chunk()` 还会进入 origin execution 阶段。

该阶段逻辑类似：

```text
def execute_origins(origin_requests):
    branch_envs.replay(origin_requests)
    branch_envs.step(copied_origin_responses)
```

因此正式执行 copied origin action 前，又会：

```text
reset same game
→ replay same mechanical prefix
→ return to same anchor
```

一次。

所以正常 happy path 为：

```text
第一次：
reset
→ prefix replay
→ anchor validation

第二次：
reset
→ same prefix replay
→ copied origin action
→ fresh suffix
```

因此：

$$
\boxed{
\text{happy path 下同一 mechanical prefix 至少执行两次。}
}
$$

---

# 13. 一个具体例子

Natural root：

```text
s0 --a0--> s1
s1 --a1--> s2
s2 --a2--> s3
s3 --a3--> s4
s4 --a4--> s5
s5 --a5--> s6
```

选择：

$$
(s_5,a_5)
$$

进行 branch。

第一次 replay：

```text
reset
→ a0
→ a1
→ a2
→ a3
→ a4
→ 到达 s5
→ 验证 s5 / action set / prompt identity
```

验证成功后，当前代码又做：

```text
reset
→ a0
→ a1
→ a2
→ a3
→ a4
→ 再次到达 s5
→ 执行 copied a5
→ fresh suffix rollout
```

所以：

$$
[a_0,\ldots,a_4]
$$

确实 mechanical replay 两遍。

---

# 14. Double Replay 不会导致 Prefix 被训练两次

需要特别区分：

$$
\text{environment replay}
\neq
\text{training occurrence}.
$$

两次 replay 都只是环境恢复。

它们不会产生新的：

- prompt-response PPO rows；
- response tokens；
- old-policy generation samples；
- advantage entries。

最终进入训练的仍然只有：

```text
natural-root occurrences
+
copied branch-origin occurrence
+
fresh branch-suffix occurrences
```

mechanical replay prefix 不加入 training batch。

因此：

$$
\boxed{
\text{double replay 不会造成 shared-prefix gradient duplication。}
}
$$

其核心问题是环境执行与 wall-clock 开销。

---

# 15. 为什么第二次 Replay 不是“完全没用”

这里不能简单地把第二次 `replay()` 删除。

它在当前实现中承担一个重要作用：

$$
\boxed{
\text{request filtering / retry 后的 worker-state realignment}
}
$$

---

## 15.1 例子：validation 后某个 request 失败

初始 worker layout：

```text
worker 0 → R1
worker 1 → R2
worker 2 → R3
worker 3 → R4
```

第一次 replay 后：

```text
R1 ✓
R2 ✗
R3 ✓
R4 ✓
```

最终有效 request set 是：

```text
R1 R3 R4
```

但是它们实际驻留的 worker 是：

```text
worker 0 → R1
worker 2 → R3
worker 3 → R4
```

如果此时普通 `step()` 只传三条 action，并默认作用于：

```text
worker 0
worker 1
worker 2
```

那么 request 与 worker state 会错位。

当前代码通过重新 replay：

```text
R1 R3 R4
```

把它们重新压紧到：

```text
worker 0 → R1
worker 1 → R3
worker 2 → R4
```

然后再统一执行 copied origin action。

因此第二次 replay 当前兼有：

$$
\boxed{
\text{state restoration}
+
\text{worker compaction / alignment}
}
$$

的作用。

---

# 16. 为什么 Happy Path 的第二次 Replay 又是冗余的

如果第一次 replay validation：

```text
16 / 16 全部成功
```

那么 worker layout 本来就是：

```text
worker 0 → R0
worker 1 → R1
...
worker 15 → R15
```

此时不存在 filtering 或 alignment 问题。

理论上可以直接：

```text
step copied origin actions
```

而无需再次：

```text
reset + replay prefix
```

因此：

$$
\boxed{
\text{happy path 的第二次 replay 是可以消除的。}
}
$$

---

# 17. Retry 路径为什么可能 Replay 更多次

如果某个 request validation 失败并换 alternate origin：

```text
original validation replay
→ failed
→ alternate origin replay
→ validate
```

之后为了构造最终 compact valid request batch，还可能再次 replay。

因此更准确的描述不是：

```text
每条 branch replay 恰好两次
```

而是：

$$
\boxed{
\text{happy path 通常两次；retry/filtering path 可能更多次。}
}
$$

---

# 18. Double Replay 的性能代价

设 branch anchor depth 为：

$$
d.
$$

理想 mechanical restore cost：

$$
d.
$$

当前 happy path mechanical restore cost 近似：

$$
2d.
$$

因此仅 mechanical prefix replay 部分的环境步数接近翻倍。

不过必须强调：

- mechanical replay 不需要 LLM generation；
- 它通常远便宜于 fresh suffix token generation；
- 但如果 anchor 很深、branch 数较多，仍会形成明显额外 CPU/environment wall-clock。

例如：

$$
d=20,
$$

且一批有：

$$
48\text{ branches},
$$

happy-path 额外 mechanical steps 近似：

$$
48\times20=960
$$

步。

因此在正式 wall-clock comparison 中不能忽略。

---

# 19. 当前还有一个确定存在的计量问题

当前 origin execution phase 的 timing 大致是：

```text
branch_envs.replay(origin_requests)
started = time.monotonic()
branch_envs.step(...)
```

也就是说：

$$
\boxed{
\text{第二次 prefix replay 发生在计时器启动之前。}
}
$$

所以这部分 replay wall-clock 没有被完整计入对应 phase timing。

同时 origin transition trace 通常只按：

```text
prefix_lengths = 1
```

记录 copied origin transition，而不是重新 replay 的完整 prefix 长度。

因此目前：

$$
\boxed{
\text{replay diagnostics 会低估真实 mechanical replay cost。}
}
$$

这不影响训练结果，但会影响后续论文中的：

- replay environment steps；
- branch overhead；
- rollout wall-clock；
- 相对 GiGPO 成本分析。

因此该计量问题建议在任何正式 efficiency experiment 前修复。

---

# 20. Double Replay 的最小风险修复方案

不建议第一步就全面重构 persistent worker pipeline。

先做一个低风险修复。

## 20.1 先把成本统计准确

至少显式记录：

```text
validation_replay_seconds
validation_replay_env_steps
execution_restore_replay_seconds
execution_restore_replay_env_steps
origin_transition_seconds
suffix_generation_seconds
```

从而使：

$$
\text{total replay env steps}
$$

等于真实执行的所有 mechanical actions，而不是理论 logical prefix 长度。

---

## 20.2 Happy Path Fast Path

如果第一次 replay 后：

```text
所有 requests validation 通过
```

且：

```text
request order == worker order
```

则直接：

```text
step copied origin actions
```

跳过第二次 replay。

伪逻辑：

```text
replay_and_validate(requests)

if all_valid_and_aligned:
    execute_origin_on_current_worker_states()
else:
    replay(final_valid_requests)
    execute_origin()
```

这可以先消除最常见路径上的重复 replay。

---

# 21. 更推荐的最终方案：Persistent Branch Worker Slots

最终更干净的架构是：

$$
\boxed{
\text{persistent branch worker slots}
+
\text{selected replay / selected step}
}
$$

每个 physical wave 分配固定 slots：

```text
slot 0 → branch B0
slot 1 → branch B1
...
```

第一次 replay validation 成功后：

```text
slot state 保留
```

如果某一个 request 失败：

```text
只 reset / replay 该 slot
```

而不是重新 replay 全部成功 slots。

如果一个 branch terminal：

```text
下一轮 suffix 不再对该 slot generate / step
```

这样可以同时解决：

1. inactive branch dense execution；
2. happy-path double replay；
3. retry 时整批 realignment；
4. suffix active-set compaction。

---

# 22. 推荐的最终 Branch Execution Pipeline

建议最终数据流变为：

```text
Frozen Exact Batch-ERV branch plan
        ↓
按 replay capacity 切 physical waves
        ↓
给每条 branch 分配 persistent worker slot
        ↓
selected replay prefix
        ↓
validate anchor / prompt / action support
        ↓
失败 slot：仅该 slot retry alternate origin
        ↓
成功 slot保持当前 restored state
        ↓
selected step copied origin action
        ↓
validate origin transition
        ↓
fresh suffix rollout
        ↓
每一步只 compact active slots
        ↓
terminal / horizon-expired slots立即退出 generation 和 env step
        ↓
merge branch-origin + branch-suffix training occurrences
```

这样在统计意义上仍然是：

$$
\boxed{
\text{one frozen Batch-ERV plan}
}
$$

而在工程上变成：

$$
\boxed{
\text{capacity-bounded, state-preserving, active-set branch execution}
}
$$

---

# 23. 推荐修改优先级

## P1-A：先修成本计量

必须保证任何正式 efficiency comparison 前：

- 第一次 replay prefix steps 被统计；
- 第二次 replay prefix steps 被统计；
- retry replay steps 被统计；
- wall-clock 也对应真实执行。

这是最小风险、收益明确的修改。

---

## P1-B：修 suffix active-set compaction

让 inactive branch 不再：

- 调用 LLM generation；
- 调用 environment step。

这是最可能直接改善 wall-clock/GPU utilization 的修改。

---

## P1-C：加入 happy-path no-second-replay fast path

当所有 request 第一次 replay 均通过且 worker alignment 未改变时：

```text
validate
→ direct origin step
```

不再重复 restore。

---

## P1-D：进一步改为 persistent selected-worker pipeline

将 retry、filtering、suffix execution 全部建立在：

```text
replay_selected
step_selected
```

之上。

这是最终推荐架构，但改动比前几项更大，应在充分测试后采用。

---

# 24. 必须增加的测试

## 24.1 Inactive Branch Generation Test

构造：

```text
4 branches
step 1 后 branch 1 terminal
```

要求下一步：

```text
generate batch size == 3
```

而不是：

```text
4
```

---

## 24.2 Horizon Expiry Test

构造：

```text
B1 remaining_horizon = 1
B2 remaining_horizon = 4
```

要求从 suffix step 2 开始：

```text
B1 不再进入 generation
B1 不再进入 env.step
```

---

## 24.3 Terminal Worker No-Step Test

使用 mock environment 规定：

```text
terminal 后再次 step 直接 raise
```

如果 active compaction 正确，则测试应通过。

---

## 24.4 Happy-Path Replay Count Test

构造所有 requests 都第一次 replay 成功。

目标修复后：

```text
每条 prefix mechanical replay count = 1
```

而不是：

```text
2
```

---

## 24.5 Retry Isolated Replay Test

例如：

```text
R1 success
R2 fail → alternate origin
R3 success
```

要求 persistent-slot 版本：

```text
只重新 replay R2 对应 slot
```

R1、R3 不被 reset。

---

## 24.6 Branch Identity Preservation Test

active compaction 前后必须满足：

```text
branch_id
request_id
origin_occurrence_id
terminal_reward
suffix occurrences
```

严格保持一一对应。

---

## 24.7 Training Equivalence Test

在 deterministic/mock environment 中比较：

```text
旧 dense + double-replay pipeline
vs
新 active + persistent pipeline
```

要求在相同 frozen branch plan 和随机 seed 下：

- branch origin response tokens 一致；
- suffix generated tokens 一致；
- terminal rewards 一致；
- training occurrence IDs 一致；
- advantage input metadata 一致。

只有执行成本发生变化。

---

# 25. 建议新增的运行时指标

正式训练建议记录：

```text
branch_execution_waves
branch_replay_capacity
branch_execution_max_wave_size
branch_validation_replay_steps
branch_execution_restore_replay_steps
branch_retry_replay_steps
branch_total_mechanical_replay_steps
branch_validation_replay_seconds
branch_execution_restore_replay_seconds
branch_suffix_generated_sequences
branch_suffix_active_sequences
branch_suffix_inactive_sequences_avoided
branch_suffix_generation_seconds
branch_env_steps
```

尤其建议报告：

$$
\text{suffix active efficiency}
=
\frac{\text{active generated sequences}}
{\text{dense-equivalent generated sequences}}.
$$

以及：

$$
\text{replay duplication ratio}
=
\frac{\text{actual mechanical replay steps}}
{\text{minimum logical prefix replay steps}}.
$$

优化前 happy path 该 ratio 可能接近：

$$
2.
$$

优化后应接近：

$$
1.
$$

---

# 26. 对 BACE 方法定义的影响

两个修复都不改变：

- task-family competence prior；
- $R/Q$ topology planning；
- root-side capacity correction；
- anchor definition；
- local Beta posterior；
- Exact Batch-ERV；
- global branch allocation；
- seeded tie-breaking；
- observed-edge continuation resampling；
- branch-origin training semantics；
- GiGPO-compatible advantage；
- unified PPO objective。

因此论文方法不需要重新推导。

应该将这些修改描述为：

$$
\boxed{
\text{branch execution scheduler optimization}
}
$$

而不是新的 BACE 算法组件。

---

# 27. 最终推荐

当前两个问题都已经确认存在。

最稳妥的实施顺序是：

```text
1. 先修 replay cost accounting
        ↓
2. suffix active-set compaction
        ↓
3. happy-path 跳过第二次 prefix replay
        ↓
4. persistent selected-worker pipeline
        ↓
5. 完整 regression + 4×H100 stability test
```

核心原则是：

> **只优化 physical execution，不改变 frozen logical branch plan。**

即始终保持：

$$
\boxed{
\text{Exact Batch-ERV 决定“做哪些实验”}
}
$$

而新的执行层只负责：

$$
\boxed{
\text{用尽可能少的无效 generation 和重复 replay 把这些实验执行出来。}
}
$$

一句话总结：

> **问题一是 inactive branch 没有从物理 rollout batch 中及时退出；问题二是 restore-and-validate 与 restore-and-execute 分离导致 happy path 重复恢复 prefix。两者都主要是系统效率问题，不改变 BACE 的统计逻辑；最干净的长期解决方案是 persistent worker slots + selected replay/step + active-set compaction。**
