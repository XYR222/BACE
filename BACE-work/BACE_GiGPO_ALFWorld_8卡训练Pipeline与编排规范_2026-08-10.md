# BACE-GiGPO：基于 verl-agent ALFWorld 8 卡训练的 Pipeline 与编排规范

> 版本：2026-08-10  
> 定位：**训练流水线 / 系统编排设计文档**  
> 代码基座：`langfengQ/verl-agent`，ALFWorld `AlfredTWEnv`，GiGPO trainer  
> 目标硬件：**单机 8 GPU**  
> 目标：在尽量保留原 GiGPO/verl-agent trainer、GPU worker、PPO update 和 ALFWorld group-env 结构的前提下，引入 BACE 的动态 root/branch rollout，而不把系统改造成一个全新的 RL 框架。

---

# 0. 文档范围

本文只回答一个问题：

> **BACE 在 verl-agent 的 ALFWorld GiGPO 训练框架中，应该怎样组织一次完整的训练 iteration，怎样使用 8 张 GPU、128 个 ALFWorld group slots、Ray workers 和 vLLM rollout worker，怎样让动态 roots、capacity correction、replay 和 sequential branches 尽可能高效地运行？**

本文**不重新定义 BACE 的方法本身**。以下内容均视为已经由方法文档确定：

- task-family prior 与 instance posterior 如何计算；
- root/branch quota 如何产生；
- effective anchor 如何定义；
- ERV 如何计算；
- branch action 如何选择；
- advantage 与 loss 的最终数学形式；
- branch-origin 的训练 mask；
- shared prefix 如何参与或不参与具体 credit 统计。

本文只规定这些模块在工程上：

1. **什么时候执行；**
2. **由谁执行；**
3. **哪些任务可以同时执行；**
4. **哪些步骤必须等待前一个结果；**
5. **哪些数据必须保存；**
6. **什么时候进入 old/ref log-prob 与 PPO update；**
7. **怎样避免 8 卡 GPU 因 sequential branching 长时间空闲；**
8. **怎样记录 wall-clock 和开销，保证之后能与 GiGPO 公平比较。**

---

# 1. 设计结论先行

BACE 在 ALFWorld 8 卡上的主 pipeline 推荐采用：

```text
verl-agent 原 8-GPU hybrid actor/rollout/ref resource pool
                         │
                         ▼
              BACE rollout orchestrator
                         │
          ┌──────────────┴──────────────┐
          │                             │
   16 个 task controller          128 个 ALFWorld slots
   每个 task 独立状态机           16 groups × 8 slots
          │                             │
          └──────────────┬──────────────┘
                         ▼
              Global Ready Frontier
       roots 与 branch suffix 的模型决策
              跨 task 动态合批
                         │
                         ▼
                actor_rollout_wg
                    8 GPUs
                         │
                         ▼
          rollout 完成 / tree finalize
                         │
                         ▼
         unique-segment training batch
                         │
                         ▼
       old logprob → ref logprob → PPO
                    8 GPUs
```

最核心的编排原则是：

$$
\boxed{\text{同一个 task 内保持 BACE 的顺序依赖，task 之间尽可能并行。}}
$$

进一步说：

- **不能**把一个 task 的第 2 条 branch 在第 1 条 branch outcome 出来之前启动；
- **可以**让 task A 做第 2 条 branch 时，task B 做第 1 条 branch、task C 做 capacity extra root、task D 仍在跑普通 root；
- 这些当前需要模型决策的状态都送进同一个 GPU generation frontier；
- replay 本身在 CPU/Ray ALFWorld worker 上执行，不占用 LLM generation GPU；
- actor 在整个 acquisition batch 内冻结，所有 task 完成 8 个 terminal leaves 后才进行一次 PPO update。

这比“全 batch 分成 Pilot Round → Root Round → Capacity Round → Branch Round 1 → Branch Round 2”的严格全局 barrier 更高效，同时**不改变 BACE 每个 task 的算法语义**。

---

# 2. 以哪一个 verl-agent GiGPO 配置为基线

## 2.1 上游 ALFWorld GiGPO 脚本中的关键参数

当前 `examples/gigpo_trainer/run_alfworld.sh` 的核心设置包括：

```text
train_data_size = 16
val_data_size   = 128
group_size      = 8

model           = Qwen/Qwen2.5-1.5B-Instruct
max_prompt_len  = 2048
max_response_len= 512
max_env_steps   = 50

env             = alfworld/AlfredTWEnv
env.rollout.n   = 8
history_length  = 2        # 来自 trainer 默认配置

rollout engine  = vLLM
rollout TP      = 2
gpu_memory_utilization = 0.6
free_cache_engine = False

ppo_mini_batch_size          = 256
ppo_micro_batch_size_per_gpu = 32
old/ref logprob micro batch  = 32 per GPU
```

本文将这些参数视为 BACE 首轮实验应尽量保持不变的 baseline contract。

## 2.2 8 卡版本的唯一硬件级改动

截至 2026-08-10，上游 master 的 `run_alfworld.sh` 当前显式写的是：

```text
trainer.n_gpus_per_node=2
```

而 verl-agent 的通用 trainer 配置默认：

```text
trainer.n_gpus_per_node=8
```

本文按实验需求固定为：

```text
trainer.n_gpus_per_node=8
trainer.nnodes=1
```

如果本地 checkout 的 GiGPO 脚本本身已经是 8 卡版本，则应以**本地用于 baseline 的固定 commit**为准，并保证 GiGPO 和 BACE 使用同一 commit、同一环境版本、同一 8 卡机器。

## 2.3 为什么 BACE 不应重新切分 GPU 角色

verl-agent 当前采用 hybrid engine。ActorRollout、Reference Policy 等角色映射到同一个 `global_pool`，trainer 使用 colocated worker classes 在同一个 GPU resource pool 上分阶段工作。

因此首版 BACE **不要**设计成：

```text
GPU0-3 永久用于 rollout
GPU4-7 永久用于 PPO training
```

推荐继续使用：

```text
8 GPUs
  │
  ├─ rollout phase：共同服务 actor inference / vLLM
  ├─ logprob phase：共同计算 old/ref logprob
  └─ update phase：共同进行 actor backward/update
```

这样做的好处是：

1. 与 GiGPO baseline 的资源拓扑一致；
2. 不需要新增另一套模型副本；
3. 避免为了 branch orchestration 引入额外 GPU memory；
4. wall-clock 对比更容易解释；
5. 最大程度复用 `RayPPOTrainer`、`ActorRolloutRefWorker` 和 vLLM sharding manager。

---

# 3. ALFWorld 现有 16 × 8 Group-Environment 结构如何被 BACE 利用

## 3.1 原 GiGPO 的环境矩阵

ALFWorld 环境构造当前使用：

```text
env_num = data.train_batch_size = 16
group_n = env.rollout.n = 8
```

`AlfworldEnvs` 内部创建：

$$
16\times8=128
$$

个 Ray `AlfworldWorker`，每个 worker 持有一个独立 ALFWorld environment instance。

可以把它理解为：

```text
Task 0 : slot 0  slot 1  ... slot 7
Task 1 : slot 0  slot 1  ... slot 7
...
Task 15: slot 0  slot 1  ... slot 7
```

原 GiGPO 中，每个 task 的 8 个 slots 全部用于 8 条自然 root trajectories。

## 3.2 BACE 不改变“每 task 8 个 leaf slots”这一外层结构

BACE 推荐仍然保留：

$$
B=8
$$

个 terminal-leaf slots。

区别只在于一个 slot 的角色从固定的 `ROOT` 变成动态的：

```text
UNUSED
  ├─> PILOT_ROOT
  ├─> NATURAL_ROOT
  ├─> CAPACITY_EXTRA_ROOT
  └─> BRANCH_LEAF
```

因此：

```text
GiGPO:
Task g = [R, R, R, R, R, R, R, R]

BACE example:
Task g = [R, R, R, R, R, B, B, B]
```

关键不变量是：

```text
每个 slot 在一个 PPO iteration 内最多贡献一个 terminal leaf。
```

这使得：

- total terminal-leaf budget 与 GiGPO 完全一致；
- 环境 worker 数量不需要增加；
- 不需要为了 branch 额外创建第 129、130... 个 ALFWorld worker；
- capacity correction 只是把一个尚未消耗的 branch slot 改成 root slot；
- branch replay 使用本 task 尚未消耗的 sibling slot。

## 3.3 为什么“固定 8 个 sibling slots”比动态创建 replay env 更合适

如果一个 task 最终是：

```text
5 roots + 3 branches
```

推荐的 slot 使用方式是：

```text
slot 0: root 0
slot 1: root 1
slot 2: root 2
slot 3: root 3
slot 4: root 4
slot 5: replay prefix -> branch 0
slot 6: replay prefix -> branch 1
slot 7: replay prefix -> branch 2
```

而不是：

```text
先把 8 个环境全部跑成 root
然后杀掉其中 3 个
重新创建 3 个 env
再 reset/replay branch
```

前者更干净，因为每一个 branch slot 从本 iteration 开始时就属于该 task group，并保持在初始状态等待调度。

---

# 4. BACE 需要增加的系统组件

在现有 verl-agent 中，建议只增加五个主要组件。

## 4.1 `BACEOrchestrator`

职责：

- 负责一次 PPO iteration 内全部 16 个 task 的 acquisition 调度；
- 维护每个 task 的状态机；
- 维护 128 个环境 slots 的生命周期；
- 维护 root queue、replay queue 和 model-ready frontier；
- 在所有 task 完成 8 leaves 后构建最终训练 batch。

它替代的主要是当前：

```text
traj_collector.multi_turn_loop(...)
```

这一整段 rollout collection 逻辑。

它**不负责**：

- FSDP；
- vLLM engine 初始化；
- PPO backward；
- reference model；
- optimizer；
- checkpoint model weights。

这些继续交给 verl-agent。

---

## 4.2 `BACETaskController`

每个训练 task 一个 controller，共 16 个。

它只管理本 task 的状态：

```text
PILOT
  ↓
PLAN
  ↓
ROOT_FILL
  ↓
CAPACITY_CHECK
  ├─ capacity不足 → ROOT_CORRECT → CAPACITY_CHECK
  └─ capacity足够 → FREEZE_ROOT_BACKBONE
                         ↓
                    BRANCH_READY
                         ↓
                    BRANCH_RUNNING
                         ↓
                    UPDATE_LOCAL_STATS
                         ↓
              ┌──────────┴──────────┐
              │ remaining branch > 0│
              │                     │
              └────> BRANCH_READY   │
                                    │
                          remaining=0
                                    ↓
                                 FINALIZE
                                    ↓
                                   DONE
```

最重要的规则：

> **一个 task 一旦进入 `FREEZE_ROOT_BACKBONE`，该 task 后面不再生成 root。**

但另一个 task 此时仍可以继续做 root correction。

---

## 4.3 `BACEAlfWorldSlotManager`

这是对当前 `AlfworldEnvs + AlfWorldEnvironmentManager` 的最小扩展。

现有 `AlfworldEnvs.step(actions)` 要求一次传入全部 worker 的 action，并会 step 所有 workers。BACE 不适合这种接口，因为：

- 某些 slots 正在跑 root；
- 某些 slots 还未使用；
- 某些 slots 正在 replay；
- 某些 slots 已 terminal；
- 不能为了 replay slot 5 而把 slot 0~4、6~7 也向前 step 一次。

因此必须支持至少：

```text
step_selected(slot_ids, actions)
```

以及推荐支持：

```text
get_obs_selected(slot_ids)
get_admissible_selected(slot_ids)
```

必要时再支持：

```text
reset_selected(slot_ids)
```

但主流程中，branch 优先使用**尚未消费、仍处于当前 task 初始状态的 sibling slot**，因此同一 iteration 中通常不需要 reset branch slot。

---

## 4.4 `ReplayExecutor`

职责：

```text
ReplayJob
   ↓
取一个未使用 sibling slot
   ↓
按记录的环境 action prefix 机械 step
   ↓
重建 SimpleMemory
   ↓
恢复 anchor 前 observation
   ↓
一致性验证
   ↓
执行 branch-origin action
   ↓
生成新的 current observation
   ↓
送入 MODEL_READY frontier
```

它运行在 CPU / Ray ALFWorld worker 侧，不调用 LLM generation。

---

## 4.5 `TreeBatchAssembler`

职责：

- 保存 logical full leaves；
- 保存 unique trainable segments；
- 维护 root / branch lineage；
- 构造最终 `DataProto`；
- 构造 BACE advantage 计算需要的 sidecar metadata；
- 确保 replay prefix 不重复进入 training buffer。

---

# 5. 128 个环境 Slot 的生命周期

建议每个 slot 有一个明确状态：

```text
INITIALIZED
UNUSED
ROOT_ACTIVE
ROOT_DONE
REPLAYING
BRANCH_ORIGIN_EXECUTED
BRANCH_ACTIVE
BRANCH_DONE
FAILED
```

一个正常 slot 生命周期只有两类。

## 5.1 Root slot

```text
INITIALIZED
   ↓
UNUSED
   ↓ scheduler assigns root
ROOT_ACTIVE
   ↓ multi-turn actor/env loop
ROOT_DONE
```

## 5.2 Branch slot

```text
INITIALIZED
   ↓
UNUSED
   ↓ controller creates ReplayJob
REPLAYING
   ↓ mechanical prefix actions
BRANCH_ORIGIN_EXECUTED
   ↓ actor resumes normal decision
BRANCH_ACTIVE
   ↓ suffix rollout
BRANCH_DONE
```

## 5.3 绝对不能发生的生命周期

```text
ROOT_DONE
   ↓ reset
REPLAYING
   ↓ branch
```

不把已经贡献过 root leaf 的 slot 再复用成 branch leaf。

原因：

1. 一个 slot 对应一个 leaf budget，最容易审计；
2. 避免 reset 后是否仍是相同 game 的隐式依赖；
3. 避免同一 worker 在一个 iteration 中贡献多个 terminal outcomes；
4. 计数和公平比较更简单。

---

# 6. 一次 PPO Iteration 的完整外层结构

推荐的一次 BACE iteration 为：

```text
┌──────────────────────────────────────────────────────────┐
│ 0. Load 16 task groups                                  │
│    freeze current actor / prior snapshot                 │
├──────────────────────────────────────────────────────────┤
│ 1. Reset all 16×8 ALFWorld sibling slots                │
│    establish task-group identity                         │
├──────────────────────────────────────────────────────────┤
│ 2. Launch 2 pilot roots per task                         │
│    32 leaves in acquisition                              │
├──────────────────────────────────────────────────────────┤
│ 3. Per-task planning                                     │
│    decide initial root demand / planned branch quota     │
├──────────────────────────────────────────────────────────┤
│ 4. Dynamic acquisition                                   │
│    root fill + capacity correction + branch execution    │
│    task-local state machine, cross-task scheduling       │
├──────────────────────────────────────────────────────────┤
│ 5. Wait until every task has exactly 8 terminal leaves   │
├──────────────────────────────────────────────────────────┤
│ 6. Tree / unique-segment finalize                        │
├──────────────────────────────────────────────────────────┤
│ 7. Build response/loss masks and reward tensors          │
├──────────────────────────────────────────────────────────┤
│ 8. Compute BACE group/tree statistics                    │
├──────────────────────────────────────────────────────────┤
│ 9. Balance flat training batch across 8 GPU ranks        │
├──────────────────────────────────────────────────────────┤
│10. Recompute old logprob                                 │
├──────────────────────────────────────────────────────────┤
│11. Compute reference logprob                             │
├──────────────────────────────────────────────────────────┤
│12. Actor PPO update                                      │
├──────────────────────────────────────────────────────────┤
│13. Commit task-family prior statistics                   │
├──────────────────────────────────────────────────────────┤
│14. Optional validation / checkpoint                      │
└──────────────────────────────────────────────────────────┘
```

整个步骤 1–11 中使用同一个 frozen behavior actor。

**直到全部 acquisition 完成之前，不更新 actor。**

---

# 7. Phase 0：Iteration 初始化

一次训练 step 从 dataloader 取：

```text
16 task descriptors
```

每个 task 创建：

```text
TaskPlan[g]
SlotPool[g][0..7]
RootStore[g]
AnchorStore[g]
BranchStore[g]
TreeSidecar[g]
```

同时：

1. 冻结当前 task-family prior snapshot；
2. 本 iteration 中 prior 不再改变；
3. 当前 actor 视为 rollout behavior policy；
4. 给所有 root/branch leaf 创建统一的 `task_uid`；
5. 暂不创建最终 `traj_uid`，直到 slot 被实际分配。

推荐的 task-level metadata：

```text
task_uid
task_family
batch_index
planned_branch_quota
current_branch_quota
root_count
branch_count
root_backbone_frozen
branch_remaining
controller_state
```

---

# 8. Phase 1：一次性 Reset 128 个 sibling slots

延续现有 GiGPO：

```text
16 task groups × 8 sibling env workers
```

都执行一次 initial reset。

需要保存每个 slot 的：

```text
gamefile
initial raw observation
initial admissible actions
task description
environment worker id
slot id
```

并进行 group consistency assertion：

```text
同一 task group 的 8 个 sibling slots
必须对应同一 task / same initial anchor semantics
```

如果这一断言失败，说明 group-env 本身不满足 GiGPO/BACE 的实验前提，应立即终止该 batch，而不是在 controller 中补救。

---

# 9. Phase 2：Pilot roots

每个 task 选定两个 slots：

```text
slot 0
slot 1
```

作为 pilot roots。

因此第一阶段共有：

$$
16\times2=32
$$

个 active root trajectories。

## 9.1 GPU 侧运行方式

不要逐 task 执行：

```text
T0 root0
T0 root1
等结束
T1 root0
T1 root1
...
```

而是：

```text
T0:s0, T0:s1,
T1:s0, T1:s1,
...
T15:s0, T15:s1
```

全部进入同一个 model-decision frontier。

每个 agent turn：

```text
收集当前仍 active 的 slots
        ↓
构造当前 ALFWorld model input
        ↓
pad / shard 到 actor_rollout_wg
        ↓
vLLM generate_sequences
        ↓
decode actions
        ↓
仅 step 对应 active slots
        ↓
新的 observation 再进入 frontier
```

这一部分尽量复用 verl-agent 当前 multi-turn rollout 的：

- prompt preprocessing；
- `actor_rollout_wg.generate_sequences`；
- decoding；
- loss mask / step record schema；
- active mask；
- terminal handling。

变化主要是 environment step 从固定全向量 step 改为 selected-slot step。

---

# 10. Phase 3：Pilot 完成后的 per-task Planning

一个 task 的两个 pilots 一完成，不必等待所有 task 的 pilots 都完成后才计算 planning。

例如：

```text
T3 两条 pilots 已完成
T7 还有一条 pilot 未完成
```

则 CPU controller 可以立即：

```text
T3:
  read frozen prior
  read pilot outcomes
  compute planned root / branch topology
  enqueue remaining planned roots
```

这种 CPU planning 与其他 task 的 GPU rollout 可以重叠。

因为 planning 成本很小，它不应形成全局 barrier。

注意：

> task-level planning 可以异步发生，但 actor 仍然不更新；所有 tasks 共享同一个 rollout policy version。

---

# 11. Phase 4：Initial Root Fill

假设某 task 计划：

```text
5 roots + 3 branches
```

其 pilot 已经贡献 2 roots，因此还需要：

```text
3 planned natural roots
```

controller 从当前 task 的 unused slots 中选择：

```text
slot 2
slot 3
slot 4
```

加入 `ROOT_READY` queue。

所有 tasks 的 root jobs 混合起来，送入同一个 GPU frontier。

例如：

```text
T0 needs 6 more roots
T1 needs 4 more roots
T2 needs 2 more roots
T3 needs 0 more roots
...
```

形成：

```text
ROOT_READY = [
  T0:s2, T0:s3, ...,
  T1:s2, T1:s3, ...,
  T2:s2, ...
]
```

不要为不同 topology 分开调用 vLLM。

---

# 12. Phase 5：Capacity Check 与 Extra Root Correction

当一个 task 的 initial planned roots 全部完成后，controller 构建当前 root-derived anchor/capacity information。

如果当前 capacity 不足，则：

```text
current branch quota -= 1
consume one previously reserved branch slot as extra root
```

例如：

```text
initial: 4 roots + 4 branches
capacity insufficient

convert slot 4:
5 roots + 3 branches
```

## 12.1 不使用全 batch barrier

不推荐：

```text
等16个task全部 initial roots结束
→ 全体 capacity check
→ 所有不足task一起 extra root
→ 等全部结束
→ 再一起 check
```

推荐：

```text
T0 root-ready  -> check -> frozen -> can branch
T1 root-ready  -> insufficient -> enqueue extra root
T2 still running original roots
T3 root-ready  -> frozen -> can branch
```

这样可以让已经满足 capacity 的 task 尽早开始 branch，而不会被最慢 task 阻塞。

## 12.2 为什么这不违反“先 roots 后 branch”

BACE 的要求是：

> **同一个 task 内，正式 branch 前必须完成并冻结该 task 的 root backbone。**

并不要求：

> 所有 16 个 tasks 必须在同一个时钟时刻一起冻结 root backbone。

因此跨 task pipeline overlap 是纯系统优化，不改变方法。

---

# 13. Phase 6：冻结单个 Task 的 Root Backbone

当 task $g$ 满足 capacity 后，执行一次不可逆状态转换：

```text
ROOT_FILL / ROOT_CORRECT
          ↓
     FROZEN_ROOT
```

冻结内容包括：

```text
all root trajectories
root occurrence set
replay origin universe
branch-eligible anchor pool
candidate action pool
root prefix journals
planned/final branch quota
```

冻结之后：

```text
该 task 不再生成新 root
该 task 不再增加新的 branch origin
branch suffix 新发现的 state 不作为本 iteration 新 branch origin
```

但 branch outcomes 可以更新本次 sequential branch selection 所需的 local statistics。

---

# 14. Phase 7：Branch 调度的核心——Task 内顺序，Task 间并行

这是整套 pipeline 最重要的部分。

对每个 `FROZEN_ROOT` task：

1. CPU controller 根据当前 BACE local state 选择下一条 branch request；
2. 选择一个 replay origin；
3. 分配一个该 task 的 unused slot；
4. 创建 `ReplayJob`；
5. ReplayJob 完成后，branch suffix 进入 model frontier；
6. branch 到 terminal；
7. outcome 返回 controller；
8. controller 更新本 task local posterior/statistics；
9. **此时**才允许产生该 task 的下一条 branch request。

形式为：

```text
Task A:
B1 -> outcome -> B2 -> outcome -> B3

Task B:
B1 -> outcome -> B2

Task C:
B1 -> outcome -> B2 -> outcome -> B3 -> outcome -> B4
```

但是全局执行可以是：

```text
time ─────────────────────────────────────────────>

Task A:  A-B1 -------- A-B2 -------- A-B3
Task B:    B-B1 ------------ B-B2
Task C: C-B1 ----- C-B2 --------- C-B3 ---- C-B4
Task D: extra-root ---- D-B1 -------- D-B2

GPU:   [A/B/C root+branch ready states continuously batched together]
```

不是：

```text
先完整做完 Task A 所有 branches
再 Task B
再 Task C
```

---

# 15. Replay 的详细执行流水线

## 15.1 `ReplayJob` 必须包含什么

至少保存：

```text
task_uid
branch_uid
slot_id
origin_root_uid
origin_step_idx
gamefile
env_action_prefix
recorded_anchor_obs
recorded_admissible_actions
selected_branch_action
origin_payload / token metadata
remaining_horizon
```

其中 `env_action_prefix` 是**环境已经实际执行成功的动作序列**，不是模型原始自由文本。

## 15.2 为什么优先使用 unused sibling slot，而不是 reset 一个 done slot

因为该 sibling slot：

- 已经属于同一个 task group；
- 从 iteration 开始就初始化在同一 group 的初始任务；
- 尚未贡献 terminal leaf；
- replay 后自然成为一个新的 branch leaf。

这样 branch restore 可以写成：

```text
slot at task initial state
    ↓
execute prefix action 1
    ↓
execute prefix action 2
    ↓
...
    ↓
reach recorded anchor
```

而不依赖“episode done 后 reset 是否会回到完全同一个 game”。

## 15.3 Replay 不调用 LLM

机械 replay 阶段：

```text
for action in env_action_prefix:
    env.step(action)
```

不执行：

```text
actor.generate(action)
```

因此 replay cost 主要是：

- ALFWorld TextWorld step；
- Ray RPC；
- Python environment logic；
- memory reconstruction。

不产生新的 decode tokens。

## 15.4 Replay 同时必须重建 SimpleMemory

verl-agent 的 ALFWorld actor input 不是只包含 raw current observation。

`AlfWorldEnvironmentManager` 会使用：

```text
current observation
recent history
admissible actions
step count / task prompt
```

并且默认 `history_length=2`。

因此 replay 不能只恢复 TextWorld simulator state，还必须恢复 actor 在该时刻应看到的 memory state。

推荐 replay 时每一步同时做：

```text
1. raw env step(executed_action)
2. record previous raw observation
3. update slot-local SimpleMemory
4. update admissible-action set
5. update current raw observation
```

到达 anchor 后，使用与 natural rollout 相同的 `build_text_obs()` 逻辑生成后续模型输入。

## 15.5 Replay verification

到达 target step 后验证：

```text
restored_anchor_obs == recorded_anchor_obs
selected_action is executable under restored environment semantics
restored state is nonterminal
```

建议同时记录：

```text
admissible-set exact match
```

但是否把 admissible-set mismatch 设成 hard failure，依赖最终 anchor/action规范；pipeline 至少必须统计。

## 15.6 Replay 失败如何处理

Replay failure 不应静默继续。由于主 pipeline 在 branch phase 前已经冻结该 task 的 root backbone，因此**不推荐在正式 branch phase 中把失败 branch 再转换成新 natural root**，否则会破坏“同 task 先冻结 roots、再 branches”的流水线不变量。

推荐处理顺序：

```text
ReplayJob -> verification failure
        ↓
record exact failure reason
        ↓
允许一次确定性的 replay retry / worker-level recovery
        ↓
仍失败 -> abort current acquisition batch
```

如果实现了能够保证恢复到同一 game 的 worker re-initialization，也可以在“不产生新 leaf、不改变 root pool”的前提下重建当前 branch slot 后重试。

在 ALFWorld TextWorld 主实验中，replay failure 应被视为工程异常。若失败率明显非零，应优先修复环境、slot mapping、action journal 或 memory reconstruction，而不是修改 BACE 的 root/branch quota。

---

# 16. Branch-Origin 执行与 Suffix 接入

Replay 到 anchor 后：

```text
anchor restored
    ↓
execute finalized BACE branch-origin action
    ↓
environment returns next observation
    ↓
build normal model-visible input
    ↓
branch suffix joins MODEL_READY frontier
```

branch-origin 是否包含 copied CoT、哪些 tokens 进入 loss、哪些 tokens mask 掉，属于方法层；pipeline 只要求：

```text
BranchOriginRecord
  ├─ model-visible token payload
  ├─ environment action
  ├─ train mask
  ├─ old-policy metadata
  └─ branch behavior metadata
```

因此以后即使 branch-origin loss 规则变化，也不需要改 rollout scheduler。

---

# 17. 全局 `MODEL_READY` Frontier

这是 8 卡利用率的关键。

任何时刻，只要某个 slot 需要 actor 做一次正常决策，就进入：

```text
MODEL_READY
```

来源可以是：

```text
pilot root current state
normal root current state
capacity extra root current state
branch suffix current state
```

它们对 actor_rollout_wg 来说，本质都是：

```text
current model input -> generate one macro-action response
```

因此可以放在同一个 generation batch 中。

例如某时刻：

```text
T0 root step 7
T1 branch-1 suffix step 3
T2 extra-root step 5
T5 branch-2 suffix step 1
T9 root step 11
...
```

可以一起送给 vLLM。

不要人为维护：

```text
root vLLM queue
branch vLLM queue
```

并分别调用模型。

应该只有一个：

```text
MODEL_READY frontier
```

---

# 18. 为什么推荐“混合 Frontier”，而不是严格 Branch Rounds

如果使用严格 rounds：

```text
Branch Round 1: 12 tasks
Branch Round 2: 12 tasks
Branch Round 3: 7 tasks
Branch Round 4: 2 tasks
```

后期 GPU batch 会快速缩小。

而 task-local dynamic frontier 可以在一些 task branch 的同时，吸收：

- 其他 task 尚未完成的 initial roots；
- capacity extra roots；
- 不同 branch depth 的其他 task suffix steps。

这不会消除最后的 tail，但能显著减小中间阶段的空闲。

推荐优先级：

```text
语义正确性 > 跨 task mixed batching > speculative branch
```

第一版不使用 stale posterior speculative branch。

---

# 19. Generation Scheduler 的推荐工作方式

## 19.1 Driver 维护四个队列

```text
ROOT_READY
REPLAY_READY
MODEL_READY
FINISHED_EVENTS
```

### `ROOT_READY`

尚未开始、可以从 task initial state直接运行的 root slots。

### `REPLAY_READY`

已经选定 origin/action、等待 CPU replay 的 branch slots。

### `MODEL_READY`

已经有合法 model-visible observation、等待 actor 生成下一步的 slots。

### `FINISHED_EVENTS`

刚刚 terminal 的 root / branch，等待对应 TaskController 消费 outcome。

## 19.2 CPU 与 GPU 可以重叠

理想状态：

```text
CPU:
replay A, replay B, capacity check C, ERV update D

GPU:
generate current actions for E,F,G,H,...
```

不应该：

```text
暂停全部 GPU
等待一个 task 的 ERV 计算
再生成下一条 branch
```

## 19.3 一个 scheduling tick

可以抽象成：

```text
1. Consume FINISHED_EVENTS
2. Advance per-task controllers
3. Create new ROOT_READY / REPLAY_READY jobs
4. Dispatch replay jobs to CPU Ray env workers
5. Move completed replay jobs to MODEL_READY
6. Collect normal active root/branch slots into MODEL_READY
7. Form one actor generation batch
8. generate_sequences()
9. Step only the affected environment slots
10. Return terminal jobs to FINISHED_EVENTS
11. Repeat
```

---

# 20. 与当前 verl-agent multi-turn loop 的接口关系

当前 GiGPO 核心 collector 大致是：

```text
reset envs
for step in 1..max_steps:
    build observations for active envs
    actor_rollout_wg.generate_sequences
    decode actions
    envs.step
    save records
    stop when all done
```

BACE 不应重写 actor rollout protocol，而应该扩展成：

```text
initialize slot matrix
while not all task budgets consumed:
    scheduler determines active decision slots
    build observations for MODEL_READY slots
    actor_rollout_wg.generate_sequences
    selectively step those slots
    update job/task states
```

换言之，最应该复用的是：

```text
preprocess -> generate_sequences -> decode -> record
```

最需要替换的是：

```text
固定 env vector 同步 step
```

变成：

```text
slot-aware selected step + task-state scheduler
```

---

# 21. 推荐的数据对象

## 21.1 `TaskPlan`

```text
task_uid
task_family
pilot_slot_ids
pilot_outcomes
planned_root_count
planned_branch_count
final_root_count
final_branch_count
root_backbone_frozen
controller_state
```

## 21.2 `LeafSlot`

```text
task_uid
local_slot_id
global_worker_id
role
status
leaf_uid
root_uid / branch_uid
```

## 21.3 `RootTrajectory`

```text
leaf_uid
root_uid
task_uid
slot_id
gamefile
terminal_reward
step_records[]
env_action_journal[]
anchor_obs_journal[]
admissible_action_journal[]
```

## 21.4 `ReplayOrigin`

```text
origin_root_uid
origin_step_idx
gamefile
env_action_prefix
recorded_anchor_obs
recorded_admissible_actions
model/memory reconstruction metadata
```

## 21.5 `BranchTrajectory`

```text
leaf_uid
branch_uid
task_uid
slot_id
origin
selected_action
origin_record
new_suffix_records[]
terminal_reward
```

## 21.6 `DecisionRecord`

每一个真实模型训练决策的统一记录：

```text
task_uid
leaf_uid
source_type       # root / branch_origin / branch_suffix
state_key
action
prompt_tokens
response_tokens
loss_mask
anchor_obs
is_action_valid
step_reward
behavior_metadata
```

## 21.7 `TreeSidecar`

保存不能简单塞进 flat DataProto 的结构关系：

```text
leaf -> root lineage
leaf -> selected anchor
branch -> replay origin
logical full leaf path
unique training occurrences
terminal reward map
state occurrence map
```

---

# 22. Logical Full Leaf 与 Physical Training Segment 必须分开

BACE 一定要维护两种表示。

## 22.1 Logical leaf

用于：

- terminal reward statistics；
- all-leaf comparison；
- tree visualization；
- lineage analysis；
- training anchor grouping 所需的逻辑路径信息。

表示为：

```text
root prefix + branch new segment
```

## 22.2 Physical training segment

用于：

- old logprob；
- ref logprob；
- PPO actor loss；
- trainable token count。

表示为：

```text
natural root:
    all natural decision records once

branch:
    branch-origin record according to configured train mask
    + newly generated suffix

NOT:
    replayed prefix
```

这条原则是整个工程正确性的关键。

---

# 23. 为什么不能把完整 Branch Leaf 直接复制成普通 GiGPO trajectory

例子：

```text
Root R:
s0 --a0--> s1 --a1--> s2 --a2--> s3 --a3--> terminal

从 s2 分两个 branches:
B1: s2 --b1--> ...
B2: s2 --b2--> ...
```

如果训练 buffer 展平为：

```text
R : s0 a0 s1 a1 s2 a2 ...
B1: s0 a0 s1 a1 s2 b1 ...
B2: s0 a0 s1 a1 s2 b2 ...
```

则：

```text
s0-a0
s1-a1
```

被 backward 三次。

但模型只自然采样过一次这段 prefix，另外两次只是环境 replay。

因此 BACE assembler 必须生成：

```text
Train segment 1: R entire natural root
Train segment 2: B1 unique branch segment
Train segment 3: B2 unique branch segment
```

logical leaf reconstruction只存在 sidecar，不直接等于 PPO batch。

---

# 24. Rollout 完成后的 Batch Finalize 顺序

当 16 个 tasks 都满足：

```text
root_count + branch_count = 8
```

才退出 acquisition loop。

随后推荐：

```text
A. freeze TreeSidecar
B. create flat DecisionRecord list
C. tokenize/pack into DataProto-compatible batch
D. compute response mask / loss mask
E. attach terminal & step rewards
F. compute all BACE group/tree-sensitive statistics
G. write scalar advantages back to flat records
H. only then rebalance batch across GPU ranks
I. old logprob
J. ref logprob
K. actor update
```

---

# 25. 为什么 BACE 的 group/tree statistics 建议在 `_balance_batch` 之前完成

verl-agent 原 trainer 可以在生成后按 sequence length 重新平衡 batch，以减少不同 DP rank 的 token imbalance。

BACE 有额外的：

```text
task group
leaf lineage
root/branch relationships
state occurrence groups
```

如果先随机/按长度重排，再依靠 Python object sidecar 跨索引追踪，会增加大量 bookkeeping 风险。

因此推荐：

```text
先完成所有依赖 task/tree identity 的统计
    ↓
把最终 advantages / weights 附着到每条 flat record
    ↓
之后 batch 可以自由 reorder/balance
```

一旦每条 record 已经携带自己的训练标量与 masks，后面的 old/ref logprob 和 PPO 不再关心树结构。

---

# 26. Old Logprob 阶段

继续使用原 actor rollout worker 的 log-prob recomputation。

但是输入只应该是：

```text
unique trainable segments
```

而不是：

```text
logical full branch leaves with duplicated replay prefixes
```

这样 BACE 可能比 naive tree flattening 显著减少：

```text
old-logprob token compute
```

同时必须保证：

- branch-origin 需要训练的 tokens 有 old logprob；
- branch-origin 不训练的 tokens 可以保留上下文，但 loss mask 为 0；
- mechanical replay prefix 不应为了 old logprob 被重复 materialize 成 training response。

---

# 27. Reference Logprob 阶段

继续沿用 GiGPO：

```text
ref_policy_wg.compute_ref_log_prob(...)
```

同样只在最终 physical training batch 上执行。

BACE pipeline 不为 replay prefix 增加额外 reference forward。

---

# 28. PPO Actor Update 阶段

在此阶段 BACE 已经被完全“压平”。

actor update 接收到的是：

```text
input ids
response ids
attention mask
loss mask
old log probs
reference log probs
advantages
optional branch weights / metadata
```

此时 trainer 不需要知道：

- 这个 branch 从哪个 root replay；
- capacity correction 做了几轮；
- 这个 task 原计划多少 branches；
- ERV 之前是多少。

因此推荐把 BACE 的复杂性限制在：

```text
rollout acquisition + CPU-side advantage assembly
```

而不是侵入 FSDP PPO inner loop。

---

# 29. Task-Family Prior 的更新时机

当前 batch 中：

```text
prior snapshot is READ-ONLY
```

全部 rollout 完成后，把本 batch 允许进入 task-family history 的 root statistics 送入：

```text
PriorAccumulator
```

推荐在当前 PPO step 完成后执行一次：

```text
commit_prior_update()
```

于是：

```text
Batch k:
uses prior snapshot k

Batch k end:
commit history

Batch k+1:
uses updated prior snapshot k+1
```

不要在 task T0 pilot 完成后就修改全局 prior，让同 batch 的 T15 使用不同 prior 版本。

---

# 30. 8 卡 GPU 上的实际资源时间线

推荐理解成“同一 8 卡池的时间复用”。

```text
Time ─────────────────────────────────────────────────────────────>

8 GPU pool:
[ rollout generation ]
                     [ old logprob ]
                                  [ ref logprob ]
                                               [ actor update ]
                                                          [ next rollout ]

CPU/Ray env:
[reset][env steps][replay][env steps][tree cpu work]................
```

不需要长期保留一张 GPU 专门做 ERV。

BACE controller 计算应尽量在 CPU 上完成，并与 GPU rollout overlap。

---

# 31. Rollout Tensor Parallel = 2 在 8 卡设置下的含义

GiGPO 脚本设置：

```text
actor_rollout_ref.rollout.tensor_model_parallel_size=2
```

对 BACE 的原则不是重新设计 model parallelism，而是：

> **保持 GiGPO 使用的 rollout TP 设置，首先只改变 sequence/job scheduling。**

在 8 卡上会形成多个并行 rollout execution partitions；具体 vLLM/FSDP mesh 应以当前 verl-agent commit 的实际日志为准，不建议在 BACE controller 中硬编码“GPU0-1 是 replica0”之类的拓扑。

controller 只调用：

```text
actor_rollout_wg.generate_sequences(batch)
```

由现有 worker group 处理 sharding。

---

# 32. 一个完整的 16-task 数值例子

设：

```text
train_batch_size = 16
B = 8 leaves / task
pilot = 2 roots / task
```

总 terminal-leaf budget：

$$
16\times8=128.
$$

## 32.1 Pilot

首先：

```text
16 × 2 = 32 pilot roots
```

占用每个 task 的 slots 0、1。

## 32.2 Planning 结果示例

假设 16 个 task 被分成四组：

```text
T0-T3   : planned 8R + 0B
T4-T7   : planned 6R + 2B
T8-T11  : planned 5R + 3B
T12-T15 : planned 4R + 4B
```

计划 branches：

```text
4×0 + 4×2 + 4×3 + 4×4
= 36
```

计划 roots：

```text
128 - 36 = 92
```

pilot 已经有 32 roots，因此 initial root fill 还需要：

```text
92 - 32 = 60 roots
```

这些 60 条不是分四组跑，而是全部加入 root frontier。

## 32.3 Capacity correction

假设 initial roots 完成后：

```text
T9  : capacity不足
T12 : capacity不足
T13 : capacity不足
```

各自把一个 branch slot 转为 extra root：

```text
+3 roots
-3 branches
```

最终：

```text
95 roots + 33 branches = 128 leaves
```

注意这里不需要给额外 roots 创建第 129 个 env；只是把本 task 一个 `UNUSED/branch-reserved` slot 改成 `ROOT_ACTIVE`。

## 32.4 Task-local freeze

假设：

```text
T4 root backbone 很早完成且 capacity 足够
T13 需要 extra root
```

则：

```text
T4 可以立即冻结 root pool并开始 branch 1
```

同时：

```text
T13 extra root 正在 GPU frontier 中正常 rollout
```

二者的 next-action generation 可以进入同一 batch。

## 32.5 Branch dependency

最终 branches 假设为：

```text
T4-T7:   2 each = 8
T8,T10,T11: 3 each
T9:      2
T12,T13: 3 each
T14,T15: 4 each
```

总数：

```text
8 + 9 + 2 + 6 + 8 = 33
```

若严格按 branch round 看，大约会出现：

```text
Round 1: 12 active tasks
Round 2: 12 active tasks
Round 3: 7 active tasks
Round 4: 2 active tasks
```

但实际推荐 scheduler 不人为等这些 rounds 全部同步。

例如：

```text
T14 branch1 terminal
→ CPU update local state
→ replay branch2
→ branch2 suffix可尽快重新进入 frontier
```

不必等待：

```text
T15 branch1
T12 branch1
T8 branch1
```

全部一起 terminal。

## 32.6 最终训练数据

虽然 logical terminal leaves 是：

```text
128
```

physical training data 不是简单 128 条完整复制轨迹。

例如：

```text
Root R:
step 1
step 2
step 3
step 4
step 5

Branch from step 3:
replay step1
replay step2
branch-origin
new step4'
new step5'
```

PPO 数据只有：

```text
R: step1..step5       # 一次
B: branch-origin + new suffix
```

replay step1、step2 不第二次进入训练。

---

# 33. 一个单 Task 的详细执行例子

设 Task A 有 8 个 sibling slots：

```text
A0 A1 A2 A3 A4 A5 A6 A7
```

## 33.1 Pilots

```text
A0 -> pilot root -> success
A1 -> pilot root -> failure
```

controller 计划：

```text
5 roots + 3 branches
```

## 33.2 Planned roots

```text
A2 -> root
A3 -> root
A4 -> root
```

于是 root pool：

```text
A0,A1,A2,A3,A4
```

unused：

```text
A5,A6,A7
```

## 33.3 Capacity check

capacity 足够，冻结 root pool。

从此：

```text
A0-A4 永远不会再改变为 branch slots
A5-A7 永远不会再改变为 roots
```

## 33.4 Branch 1

controller 选择：

```text
origin = root A2, step 6
branch slot = A5
```

A5 原本仍在 task 初始状态。

CPU replay：

```text
A5:
a1
-> a2
-> a3
-> a4
-> a5
```

到达 A2 的 step 6 pre-action anchor。

验证通过，执行 selected branch-origin action。

环境进入新状态 `s6'`。

`A5@s6'` 进入 GPU `MODEL_READY`。

后续由 frozen actor 正常 rollout 到 terminal。

## 33.5 Branch 2 不能提前创建

在 A5 terminal 前：

```text
A6 仍保持 UNUSED
```

因为 branch2 的选择依赖 branch1 outcome。

A5 terminal 后：

```text
controller consumes outcome
updates local stats
selects branch2
assigns A6
```

A6 开始 replay。

## 33.6 Branch 3

同理使用 A7。

最终：

```text
A0-A4 = 5 natural roots
A5-A7 = 3 branches
```

刚好 8 terminal leaves。

---

# 34. Replay 与 GPU Generation 的并行例子

某时刻系统状态：

```text
Task A: A6 正在 replay
Task B: B3 root 需要第 7 步 action
Task C: C5 branch suffix 需要第 2 步 action
Task D: D7 extra root 需要第 10 步 action
Task E: E4 刚 terminal，CPU 正在更新 controller
```

同时执行：

```text
CPU/Ray:
A6 replay action prefix
E4 update controller

GPU:
B3, C5, D7 ... together generate actions
```

A6 replay 完成后，不需要等下一整个 phase：

```text
A6 -> MODEL_READY
```

下一个 generation batch 与其他 ready states 一起进入 GPU。

这就是 BACE 推荐的 pipeline overlap。

---

# 35. Horizon Bucketing

不同 jobs 的剩余 horizon 不同：

```text
root from initial state: potentially 50 steps
late branch: for example only 8 steps left
middle branch: for example 25 steps
```

如果固定把很长和很短的 branch 做成一个严格同步 trajectory batch，短序列完成后会大量 idle。

第一版可以采用简单 bucketing：

```text
remaining horizon 1-8
remaining horizon 9-16
remaining horizon 17-32
remaining horizon 33-50
```

但注意：

- bucketing 只影响 batch formation；
- 不改变 branch selection；
- 不因为 bucket 不同而等待太久。

推荐使用：

```text
soft bucketing
```

即优先选择相近 horizon，但 frontier 数量不足时允许跨 bucket 合批。

---

# 36. 为什么不建议第一版做 Speculative Branching

可以想象为了填满 GPU，在 branch1 outcome 出来前先猜 branch2。

但这会改变 exact BACE：

```text
branch2 selection no longer conditions on branch1 evidence
```

因此第一版不做：

```text
stale-ERV speculative branches
rollback
branch cancellation
```

先实现 exact sequential semantics。

如果实测显示 branch tail 是主要 wall-clock 瓶颈，再单独做一个 systems ablation。

---

# 37. 为什么不建议第一版做 KV Snapshot / Prefix Cache Restore

BACE 的第一版 branch restore 解决的是：

```text
environment state
+ actor-visible recent memory
```

不是直接保存 vLLM 内部 KV cache。

KV-level restore 涉及：

- vLLM engine 生命周期；
- policy weight update 后 cache 失效；
- distributed TP cache ownership；
- prompt identity；
- multiple branch descendants。

因此首版：

```text
environment prefix replay + normal current-state model prompt
```

足够。

在确认 prefill 是主要瓶颈后，再研究 prefix/KV reuse。

---

# 38. Validation Pipeline

validation 完全不运行 BACE acquisition controller。

保持现有 verl-agent validation：

```text
val tasks
→ normal actor-environment multi-turn rollout
→ success rate
```

不进行：

```text
pilot
branch
ERV
capacity correction
replay
```

因此 validation 评价的是训练后 actor 本身，而不是带搜索的 inference policy。

`val_kwargs` 应与 GiGPO baseline 保持一致。

---

# 39. Checkpoint 与 Resume

除原 verl-agent actor checkpoint 外，BACE 需要额外保存：

```text
task-family prior/history state
prior version / update counter
BACE config version
```

不建议第一版持久化：

```text
batch-local anchor posterior
in-flight ReplayJob
partially finished branch tree
```

如果训练在一个 PPO iteration 中途 crash：

```text
重新开始该 rollout batch
```

比恢复半棵 active tree 更简单、风险更小。

---

# 40. 必须加入的 Profiler

为了之后与 GiGPO 做严格开销比较，BACE 不能只记录：

```text
step_time
```

至少记录以下不重叠 critical-path timers。

## 40.1 Rollout acquisition

```text
time/pilot_rollout
time/initial_root_rollout
time/capacity_extra_root_rollout
time/branch_suffix_rollout
```

## 40.2 CPU controller

```text
time/task_prior_plan
time/anchor_build
time/capacity_check
time/erv_compute
time/controller_total
```

## 40.3 Replay

```text
time/replay_wait
time/replay_execute
time/replay_verify
time/branch_origin_execute
```

## 40.4 Finalization

```text
time/tree_finalize
time/data_pack
time/bace_advantage
```

## 40.5 verl standard GPU phases

```text
time/old_logprob
time/ref_logprob
time/update_actor
time/validation
```

## 40.6 整体

```text
time/acquisition_total
time/train_step_total
```

不要嵌套计时后再把父 timer 和所有子 timer 简单求和，否则会 double count。

---

# 41. 必须记录的资源计数器

## 41.1 Leaf / topology

```text
count/terminal_leaves
count/natural_roots
count/branches
count/pilot_roots
count/capacity_extra_roots
count/root_to_branch_conversions
```

## 41.2 Branch scheduling

```text
count/branch_jobs
count/branch_round_equivalent
count/max_branches_per_task
count/replay_failures
count/replay_steps
```

## 41.3 Model compute

```text
tokens/prefill
tokens/new_decode
tokens/trainable
```

必须区分：

```text
new decode tokens
```

与：

```text
copied / injected origin tokens
```

## 41.4 Environment

```text
env/new_agent_steps
env/mechanical_replay_steps
env/invalid_action_steps
```

## 41.5 Frontier occupancy

```text
frontier/model_ready_batch_size_mean
frontier/model_ready_batch_size_p10
frontier/model_ready_batch_size_p50
frontier/model_ready_batch_size_p90
frontier/replay_queue_length
frontier/root_queue_length
```

这组指标非常重要，因为它直接告诉我们 sequential BACE 是否让 GPU batch 过小。

---

# 42. 8 卡 GPU 利用率应如何诊断

如果 BACE 比 GiGPO 慢，按以下顺序检查。

## 第一层：Replay 是否贵

看：

```text
time/replay_execute / time/acquisition_total
```

如果只有几个百分点，则 replay 不是主要问题。

## 第二层：Branch batch 是否太小

看：

```text
MODEL_READY batch size distribution
```

若经常只有 1-4 个 slots，说明 sequential tail 是瓶颈。

## 第三层：vLLM decode throughput

看：

```text
new decode tokens / branch_suffix_rollout second
```

与 GiGPO root decode throughput 比较。

## 第四层：old/ref/update 是否因为 tree 数据膨胀

如果 unique-segment training 正确，BACE 不应因为 shared prefix duplication 让 old/ref 和 actor update 成比例暴涨。

如果暴涨，优先检查 buffer flattening，而不是优化 replay。

---

# 43. 与 GiGPO 公平比较时固定什么

BACE vs GiGPO 主 wall-clock 实验固定：

```text
same machine
same 8 GPUs
same CPU allocation
same model checkpoint
same verl-agent commit
same ALFWorld version / game data
same train_data_size = 16
same val_data_size = 128
same terminal leaves per task = 8
same max_steps = 50
same vLLM engine
same rollout TP = 2
same GPU memory utilization
same actor micro batch
same old/ref logprob micro batch
same validation frequency
same validation sampling settings
```

只改变：

```text
GiGPO collector:
8 natural roots

BACE collector:
dynamic roots + branches, total 8 leaves
```

---

# 44. 推荐的 Runtime 对比表

最终论文建议直接从 profiler 导出：

| Metric | GiGPO 8-GPU | BACE 8-GPU |
|---|---:|---:|
| Terminal leaves / task | 8 | 8 |
| Natural roots / task | 8 | dynamic |
| Branches / task | 0 | dynamic |
| New decode tokens / step |  |  |
| Prefill tokens / step |  |  |
| Trainable tokens / step |  |  |
| New env steps / step |  |  |
| Replay env steps / step | 0 |  |
| Pilot/root rollout time |  |  |
| Branch rollout time | 0 |  |
| Replay time | 0 |  |
| Controller CPU time | ~0 |  |
| Old logprob time |  |  |
| Ref logprob time |  |  |
| Actor update time |  |  |
| Total sec/update |  |  |
| Mean GPU utilization |  |  |
| Mean model frontier batch size |  |  |
| Peak GPU memory |  |  |
| GPU-hours to target success |  |  |

---

# 45. 推荐的代码侵入边界

## 45.1 尽量不修改

```text
verl/trainer/main_ppo.py resource pool logic
ActorRolloutRefWorker
vLLM rollout engine
FSDP actor update
reference model
optimizer
checkpoint actor weights
validation inference
```

## 45.2 主要新增/扩展

```text
BACEOrchestrator
BACETaskController
BACEAlfWorldSlotManager
ReplayExecutor
TreeBatchAssembler
BACE advantage adapter
BACE profiler
```

## 45.3 必须修改或扩展的现有接口

```text
ALFWorld env selected-slot stepping
per-slot memory update
trajectory collector dispatch
final DataProto metadata
trainer acquisition entrypoint
```

这比直接改 `verl/workers/` 或 PPO inner loop 风险低得多。

---

# 46. 推荐的 Trainer 接入点

现有 trainer 的逻辑可以抽象为：

```text
batch from dataloader
        ↓
traj_collector.multi_turn_loop(...)
        ↓
adjust / masks / rewards
        ↓
old logprob
        ↓
ref logprob
        ↓
advantage
        ↓
actor update
```

BACE 推荐改成：

```text
batch from dataloader
        ↓
bace_orchestrator.collect(...)
        ↓
BACECollectionResult:
    flat_train_data
    tree_sidecar
    profiler_stats
        ↓
adjust / masks / rewards
        ↓
BACE group/tree statistics + attach advantages
        ↓
balance batch
        ↓
old logprob
        ↓
ref logprob
        ↓
actor update
```

其中：

```text
bace_orchestrator.collect(...)
```

内部仍反复调用现有：

```text
actor_rollout_wg.generate_sequences(...)
```

所以不是再造推理系统。

---

# 47. 推荐的高层伪代码

```text
function BACE_PPO_STEP(train_batch):

    prior_snapshot = freeze_task_prior()

    slot_matrix = env_manager.reset_all_groups()
    controllers = build_task_controllers(train_batch, prior_snapshot)

    # -------------------------------------------------
    # 1. schedule two pilot roots for every task
    # -------------------------------------------------
    for task in controllers:
        allocate_pilot_slots(task, n=2)

    while not all_tasks_done(controllers):

        # ---------------------------------------------
        # 2. consume terminal events
        # ---------------------------------------------
        finished_events = scheduler.pop_finished_events()

        for event in finished_events:
            task = controllers[event.task_uid]
            task.consume_leaf_result(event)

            if task.just_finished_pilots():
                task.plan_topology(prior_snapshot)
                scheduler.enqueue_required_roots(task)

            if task.just_finished_required_roots():
                task.check_capacity()

                if task.needs_extra_root():
                    task.convert_one_branch_slot_to_root()
                    scheduler.enqueue_one_root(task)

                elif task.can_freeze_root_backbone():
                    task.freeze_root_backbone()
                    scheduler.enqueue_next_branch_replay(task)

            if event.is_branch_terminal:
                task.update_local_branch_statistics(event)

                if task.has_remaining_branch_budget():
                    scheduler.enqueue_next_branch_replay(task)
                else:
                    task.finalize()

        # ---------------------------------------------
        # 3. CPU replay pipeline
        # ---------------------------------------------
        replay_jobs = scheduler.dispatch_replay_jobs()
        replay_executor.run_selected(replay_jobs)

        completed_replays = replay_executor.poll_completed()
        for job in completed_replays:
            scheduler.move_branch_suffix_to_model_ready(job)

        # ---------------------------------------------
        # 4. actor generation pipeline
        # ---------------------------------------------
        model_jobs = scheduler.build_model_frontier_batch()

        if model_jobs not empty:
            model_inputs = preprocess(model_jobs)
            model_outputs = actor_rollout_wg.generate_sequences(model_inputs)
            actions = decode(model_outputs)

            env_outputs = env_manager.step_selected(model_jobs.slot_ids, actions)
            recorder.append(model_jobs, model_outputs, env_outputs)

            scheduler.advance_jobs(env_outputs)

    # -------------------------------------------------
    # 5. all 16 tasks now own exactly 8 leaves
    # -------------------------------------------------
    collection = tree_batch_assembler.finalize(controllers, recorder)

    flat_batch = collection.unique_trainable_data
    tree_sidecar = collection.tree_sidecar

    flat_batch = adjust_batch(flat_batch)
    flat_batch = build_masks_and_rewards(flat_batch)

    flat_batch = compute_bace_statistics_and_advantage(
        flat_batch,
        tree_sidecar,
    )

    flat_batch = balance_batch(flat_batch)

    flat_batch.old_log_prob = actor_rollout_wg.compute_log_prob(flat_batch)
    flat_batch.ref_log_prob = ref_policy_wg.compute_ref_log_prob(flat_batch)

    actor_metrics = actor_rollout_wg.update_actor(flat_batch)

    commit_task_prior_history(collection.allowed_root_statistics)

    return actor_metrics, collection.profiler_metrics
```

这段伪代码的核心不是函数名，而是依赖关系：

```text
同 task 的 branch selection 依赖前一 branch terminal event；
GPU generation 则只依赖某个 slot 当前是否 MODEL_READY。
```

---

# 48. MVP 实现与最终推荐实现的区别

如果第一次实现动态 mixed frontier 过于困难，可以先实现一个语义完全正确的 MVP：

```text
Pilot global batch
→ Planned Root global batch
→ Capacity global rounds
→ Branch round 1
→ Branch round 2
→ ...
```

它适合用于：

- 验证 replay 正确性；
- 验证 branch data/masks；
- 验证 advantage；
- 跑小规模 correctness experiments。

但正式效率实验推荐升级到本文主设计：

```text
per-task state machine
+
cross-task mixed frontier
+
CPU replay / GPU generation overlap
```

因为全局 round barrier 会人为放大 BACE 的 wall-clock 开销。

---

# 49. 单元测试与 Pipeline Invariants

正式大规模训练前必须通过以下 invariants。

## 49.1 Budget invariant

对每个 task：

```text
root_count + branch_count == 8
```

且：

```text
number of consumed slots == 8
```

## 49.2 Slot uniqueness

```text
one slot -> at most one terminal leaf
```

## 49.3 Root freeze invariant

一旦：

```text
root_backbone_frozen == True
```

则：

```text
no later root is generated for this task
```

## 49.4 Sequential branch invariant

对任一 task：

```text
branch b+1 selection_time > branch b terminal_time
```

## 49.5 Replay invariant

```text
mechanical replay steps do not trigger actor generation
mechanical replay steps do not enter train loss
```

## 49.6 Training duplication invariant

一个自然 prefix decision occurrence 在 physical training buffer 中最多出现一次。

## 49.7 Policy version invariant

同一 acquisition batch 中：

```text
all natural roots
all branch origins
all branch suffixes
```

属于同一个 frozen old-policy version。

## 49.8 Prior version invariant

同一 16-task training batch 使用同一个 frozen task-prior snapshot。

---

# 50. 失败恢复规则

## 50.1 环境 replay mismatch

```text
mark branch invalid
record worker/task/origin
follow algorithm-level fallback
```

不能直接把 mismatch branch 当正常样本训练。

## 50.2 Ray worker crash

若一个 ALFWorld slot worker crash：

- 第一版建议 abort 当前 PPO acquisition batch；
- 重建所有 128 env slots；
- 从该 dataloader batch 重新 collect。

不要尝试只恢复半棵 tree，除非后续专门实现 batch-level durable checkpoint。

## 50.3 GPU OOM

首先降低：

```text
frontier generation microbatch / dynamic batch size
```

不要第一反应减少 terminal-leaf budget，因为这会改变算法对比。

## 50.4 Tail underutilization

如果 branch 最后只剩 1-2 tasks：

- 正确完成；
- 记录 utilization；
- 不为了“填满 GPU”违反 sequential branch dependency。

---

# 51. 我认为最合适的第一版正式 Pipeline

最终推荐如下。

## GPU

```text
single node / 8 GPUs
same global hybrid actor-rollout-ref pool as GiGPO
vLLM rollout TP=2
```

## Environment

```text
16 task groups
8 persistent sibling ALFWorld workers per task
128 workers total
one worker slot = one terminal leaf budget
```

## Rollout

```text
2 pilots per task
→ per-task topology planning
→ dynamic root fill
→ per-task capacity correction
→ per-task root freeze
→ exact sequential branches
```

## Parallelism

```text
within task:
    respect branch dependency

across tasks:
    no unnecessary global barriers

model generation:
    one mixed MODEL_READY frontier

replay:
    CPU/Ray parallel and overlapped with GPU generation
```

## Data

```text
logical full leaves for statistics
physical unique segments for training
no replay-prefix duplication
```

## Optimization

```text
after all 128 leaves complete:
    finalize tree
    compute tree/group-sensitive statistics
    flatten/balance
    old logprob
    ref logprob
    PPO update
```

## Evaluation

```text
normal actor-only ALFWorld validation
no BACE test-time branching
```

---

# 52. 为什么这套设计最适合当前 BACE

这套 pipeline 有五个主要优点。

## 52.1 与 GiGPO baseline 的外层预算完全兼容

```text
16 tasks × 8 leaves
```

没有为了 branch 隐式增加 terminal samples。

## 52.2 最大程度复用 verl-agent

GPU trainer、actor rollout worker、reference model 和 PPO inner loop 基本不动。

## 52.3 顺序依赖只限制必要的地方

只有：

```text
same-task next branch selection
```

必须等待。

其他地方都允许 overlap。

## 52.4 Replay 不额外占用 GPU generation

ALFWorld prefix restore 主要使用 CPU/Ray env workers。

## 52.5 成本可以被严格审计

每个 leaf slot、replay step、decode token 和 trainable token 都有明确来源，方便最终进行：

```text
GiGPO vs BACE
same leaf budget
same hardware
same trainer
```

的效率比较。

---

# 53. 代码基线参考

以下链接用于固定实现依据；正式实验应记录实际使用的 commit hash。

1. verl-agent repository  
   <https://github.com/langfengQ/verl-agent>

2. GiGPO ALFWorld training script  
   <https://github.com/langfengQ/verl-agent/blob/master/examples/gigpo_trainer/run_alfworld.sh>

3. ALFWorld environment worker implementation  
   <https://github.com/langfengQ/verl-agent/blob/master/agent_system/environments/env_package/alfworld/envs.py>

4. Environment manager  
   <https://github.com/langfengQ/verl-agent/blob/master/agent_system/environments/env_manager.py>

5. PPO/Ray trainer  
   <https://github.com/langfengQ/verl-agent/blob/master/verl/trainer/ppo/ray_trainer.py>

6. Main PPO entrypoint  
   <https://github.com/langfengQ/verl-agent/blob/master/verl/trainer/main_ppo.py>

---

# 54. 一页式最终 Pipeline

```text
                         ONE PPO ITERATION
================================================================================

Input: 16 ALFWorld task groups
       8 sibling env slots / task
       8 GPUs
       frozen actor + frozen task-prior snapshot

                              │
                              ▼
                    RESET 16 × 8 ENV SLOTS
                              │
                              ▼
                    2 PILOT ROOTS / TASK
                              │
                    mixed actor frontier
                              │
                              ▼
                     PER-TASK PLANNING
                              │
                              ▼
             ┌─────────────────────────────────┐
             │   PER-TASK ACQUISITION STATE    │
             │                                 │
             │ planned roots                   │
             │      ↓                          │
             │ capacity check                  │
             │      ├─ fail -> +1 extra root  │
             │      └─ pass -> freeze roots   │
             │                       ↓         │
             │                 select branch 1 │
             │                       ↓         │
             │                  replay prefix │
             │                       ↓         │
             │                  branch suffix │
             │                       ↓         │
             │                    outcome      │
             │                       ↓         │
             │                 update local   │
             │                       ↓         │
             │                 select branch 2 │
             │                       ...       │
             └─────────────────────────────────┘
                              │
                  task-local dependencies
                              │
             but cross-task jobs run together
                              │
                              ▼
            GLOBAL MODEL_READY FRONTIER ON 8 GPUs
        roots + extra roots + branch suffixes mixed
                              │
                              ▼
                    ALL TASKS REACH 8 LEAVES
                              │
                              ▼
                   FREEZE LOGICAL TREES
                              │
                              ▼
       LOGICAL FULL LEAVES       UNIQUE TRAIN SEGMENTS
         for statistics      ->   for PPO compute
                              │
                              ▼
                    BUILD MASKS / REWARDS
                              │
                              ▼
             BACE GROUP/TREE-SENSITIVE CREDIT
                              │
                              ▼
                    BALANCE FLAT BATCH
                              │
                              ▼
                       OLD LOGPROB
                              │
                              ▼
                       REF LOGPROB
                              │
                              ▼
                      PPO ACTOR UPDATE
                              │
                              ▼
                   COMMIT PRIOR HISTORY
                              │
                              ▼
                 VALIDATION / NEXT ITERATION

================================================================================
```

最终一句话：

> **BACE 在 verl-agent ALFWorld 上不应实现成“先树搜索、再另起一套 trainer”，而应实现成一个替代原 GiGPO trajectory collector 的 task-aware rollout orchestrator：保留 16×8 group-env 与 8-GPU hybrid training 主干，把每个 task 的 8 个 sibling slots 动态解释为 root 或 branch leaves；同 task 保持 exact sequential acquisition，跨 task 通过 mixed frontier 并行，并在所有 leaves 完成后还原成 unique-segment DataProto 进入原有 old/ref-logprob 与 PPO update。**
