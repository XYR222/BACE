# BACE 完整编排 Pipeline：旧版与当前版详细对照

日期：2026-08-10
代码基座：langfengQ/verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8 upstream master，加本地 NPU/ALFWorld 基线提交
实现目录：/home/naie/work/work-BACE/verl-agent-src
当前训练脚本：examples/bace_gigpo/run_alfworld_npu_8card_50epoch.sh

本文用于逐阶段排查 BACE 的正确性、性能和数据来源。“旧版”指原来的 staged sequential 编排；“当前版”指新增的 staged packed 编排；preallocated 路径仍作为第三种兼容模式保留。

## 1. 总览

一次 PPO step 的完整路径：

~~~text
训练 batch
  -> natural root rollout
  -> dynamic topology planning
  -> frozen anchor/index
  -> ERV branch acquisition
  -> strict Replay
  -> branch origin transition
  -> branch suffix rollout
  -> root/branch batch assembly
  -> reward and log-prob recomputation
  -> BACE advantage
  -> artifact trace
  -> actor update
~~~

三种 root 模式：

| 模式 | natural root 行为 | 是否生成后丢弃 root | 是否保留 |
|---|---|---:|---|
| dynamic + preallocated | 一次先生成 B 条 candidate roots，再由 planner 选择 R 条 | 是 | 原路径 |
| dynamic + staged + sequential | pilot、completion、correction 分成多个窄 wave | 否 | 原 staged 路径 |
| dynamic + staged + packed | reset key probe 后，把同一阶段已知缺口合并成宽 wave | 否 | 当前优化路径 |

三种模式共用 competence posterior、capacity rule、AnchorIndex、ERV、Replay validator、advantage 和 PPO update；packed 只改变 natural-root 的调度、合并和日志解析。

### 用户理解的 group 视角

更直观地说，一个训练 batch 包含 16 个逻辑 task row：

~~~text
batch = [task_0, task_1, ..., task_15]
每个 task row = 一个 group
每个 group 内部再决定 root 数 R 和 branch 数 Q
~~~

理想的 group pipeline 是：

~~~text
对每个 group 并行生成 pilot roots
-> 根据该 group 的 pilot outcome 计算 posterior
-> 该 group 决定补多少 natural roots、多少 branches
-> 同一阶段的 root requests 并行
-> 同一 ERV round 的 branch requests 并行
-> branch outcome 回写该 group 的 ERV posterior
-> 进入下一 ERV round
~~~

当前 packed 实现正是把这些 group 操作在全局 batch 中按 slot 打包：

~~~text
一次 packed wave = 所有 group 的同一个 root slot
例如：[task0/root0, task1/root0, ..., task15/root0,
      task0/root1, task1/root1, ..., task15/root1]
~~~

这和“每个 group 内部并行 pilot”是等价的，只是底层 vLLM 调用采用全局 batch，而不是为每个 group 单独调用模型。

需要区分“逻辑 task 固定”和“环境实例固定”：当前训练 parquet 的 row 固定了 prompt/task index，但 ALFWorld 的 concrete gamefile 是 reset 时由环境产生的，并没有作为 parquet row 的 env_kwargs 传入。当前 ALFWorld worker 虽按 group 使用相同 seed offset，通常会得到同类/相同实例，但严格身份仍以 reset 返回的 extra.gamefile 为准。因此 packed 路径先做 reset-key probe，再让同一 group 的所有 root slot 绑定同一个 gamefile。若未来数据行显式携带 gamefile 并由 env reset 使用，就可以去掉 probe。

## 2. 代码入口和运行时对象

### 2.1 Shell/Hydra

examples/bace_gigpo/run_alfworld_npu_8card_50epoch.sh 负责：

~~~text
设置 NPU/Conda/HF offline 环境
  -> prepare train/test parquet
  -> verl.trainer.main_ppo
  -> 保存 logs/checkpoints/tensorboard/rollout_trajectories/run_metadata
~~~

当前方法配置：

~~~text
algorithm.adv_estimator=bace_gigpo
algorithm.bace.enabled=true
algorithm.bace.topology=dynamic
algorithm.bace.dynamic_root_generation=staged
algorithm.bace.staged_root_batching=packed
algorithm.bace.total_leaf_budget=4
algorithm.bace.pilot_roots=2
env.rollout.n=4
env.max_steps=40
~~~

显存/并行配置沿用此前成功的 8 卡脚本：

~~~text
tensor_model_parallel_size=1
ppo_micro_batch_size_per_gpu=8
rollout.log_prob_micro_batch_size_per_gpu=8
ref.log_prob_micro_batch_size_per_gpu=8
gpu_memory_utilization=0.6
max_num_batched_tokens=16384
max_num_seqs=128
gradient_checkpointing=true
~~~

配置定义：verl/trainer/config/ppo_trainer.yaml 的 algorithm.bace。

### 2.2 main_ppo.py 初始化

main_ppo.py 检查 dynamic topology 时 env.rollout.n 等于 total_leaf_budget，fixed topology 时等于 fixed_root_count，然后创建：

1. 主 natural environment pool envs；
2. 独立 branch replay pool branch_envs；
3. BaceTrajectoryCollector；
4. RayPPOTrainer。

natural pool 和 branch pool 必须分离；branch replay/step 不能污染 natural worker 状态。

### 2.3 两类环境池

ALFWorld 典型拓扑：

~~~text
natural pool:
  env_num = train_data_size
  group_n = total_leaf_budget
  worker_count = train_data_size * total_leaf_budget

branch pool:
  env_num = train_data_size
  group_n = 1
  用于 Replay 和 branch origin/suffix
~~~

staged 仍预创建完整 natural worker pool，但每个 wave 只 reset/step 被选择的 worker：

~~~text
worker_index = task_batch_index * B + root_slot
~~~

ALFWorld reset key 是 extra.gamefile；WebShop reset key 是 session_idx。

## 3. 全局数据身份和不变量

| 字段 | 作用 | 生命周期 |
|---|---|---|
| task_batch_index | 当前训练 batch 中的 task 行号 | root、branch、artifact |
| uid / task_id | 同一逻辑 task 的稳定身份 | 一个 BACE step 内跨所有 root wave |
| traj_uid | 一条 natural trajectory 的唯一身份 | root rollout 到 root event |
| root_id | natural root 的 immutable identity | planner、anchor、Replay |
| occurrence_id | 一次 action occurrence 的唯一 ID | leaves、Replay origin、advantage |
| environment_reset_key | concrete ALFWorld game/session | natural root 和 branch Replay |
| anchor_obs / anchor_key | action 前结构状态 | AnchorIndex、ERV、Replay |
| action_identity | valid/invalid/unparsed action identity | AnchorIndex、Replay、artifact |
| leaf_id | PPO occurrence 所属 terminal leaf | advantage、validator |
| source_type | root、branch_origin、branch_suffix | advantage、诊断 |

必须满足：

~~~text
同 task 的 root 使用同一 concrete reset key
root occurrence_id 全局唯一
branch origin 必须来自 frozen natural occurrence
branch origin response/token/log-prob 必须复制 natural origin
Replay 失败 request 不得进入训练 batch
每个 task: final_root_count + final_branch_count = total_leaf_budget
~~~

## 4. 旧版 Pipeline：staged sequential

### 4.1 Pilot root waves

旧 collector 对每个 pilot slot 调用一次 collect(task_indices)：

~~~text
wave 0: 每个 task -> worker(task * B + 0)
wave 1: 每个 task -> worker(task * B + 1)
~~~

第二个 wave 使用 wave 0 记录的 exact reset key，保证同 task roots 来自同一 concrete game/session。

每个 wave 内部调用 vanilla_multi_turn_loop：

~~~text
reset subset
for step in 0 .. env.max_steps-1:
    preprocess observation + prompt
    actor_rollout_wg.generate_sequences
    tokenizer decode
    environment.step
    记录 action/anchor/reward/done/reset key
停止：所有 worker done 或达到 max_steps
~~~

输出是 flattened DataProto，一行通常对应一次 action occurrence。

### 4.2 Posterior 和初始 quota

旧版先执行：

~~~text
root_batches -> _concat_batches -> build_root_event_logs
~~~

再由 DynamicTopologyPlanner.initialize_staged 计算：

~~~text
family prior = CompetenceHistory.prior(task_family)
posterior = prior + pilot won outcomes
readiness = P(posterior competence > competence_threshold)
planned_Q = round((B - pilot_roots) * readiness)
target_R = B - planned_Q
~~~

CompetenceHistory 只用最终 natural root outcome 更新，并带 forgetting、transfer fraction 和 strength bound；branch outcome 不回写 family competence history。

### 4.3 Completion 和 capacity correction

若 target_R > pilot_roots，旧版每轮只补一个 slot：

~~~text
while some task still needs root:
    needed = tasks whose next slot is needed
    collect(needed)
    root_output = concat(all root batches again)
~~~

随后检查 structural anchor capacity：

~~~text
capacity = max_branches_per_anchor * effective_anchor_count
if capacity < branch_count:
    root_count += 1
    branch_count -= 1
    collect(one corrective slot per deficient task)
    concat/rebuild logs again
~~~

算法满足 R + Q = B，但旧编排会产生多个约 16 条输入的小 batch，并反复复制完整 DataProto、重建全部 root logs。

## 5. 当前 Pipeline：staged packed

实现：recipe/bace_gigpo/rollout_collector.py 的 _collect_staged_dynamic_roots_packed。

### 5.1 Reset-key probe

先只 reset 每个 task 的 slot 0 worker，读取 concrete reset key，不调用模型：

~~~text
[task0 slot0, task1 slot0, ..., taskN-1 slot0]
~~~

直接合并两个 pilot 而不先冻结 key 不安全，因为 slot 0/slot 1 可能抽到不同 game/session。probe 后所有 slot 绑定同 task key。

### 5.2 Packed pilot wave

_pending_root_slots 以 slot-major 顺序构造：

~~~text
(task0, slot0), (task1, slot0), ...
(task0, slot1), (task1, slot1), ...
~~~

对于 train_data_size=16、pilot_roots=2，一次是 32 条 task-slot。每个 slot 显式携带 worker_index、stable uid、task_batch_index 和 task-specific reset_key。

因此不再依赖 i // env.rollout.n 推断 task identity，policy 仍在同一次 PPO update 前保持 frozen。

### 5.3 Packed completion

pilot 完成后计算每 task target root count，再一次性打包已知缺口：

~~~text
for slot in range(max(target_R)):
    for task in tasks:
        if generated[task] <= slot < target_R[task]:
            append(task, slot)
~~~

如果每个 task 从 2 条 pilot 补到 4 条，completion 是一个约 32 条输入的 wave，而不是两个约 16 条 wave。

### 5.4 Packed capacity correction

capacity correction 保留依赖边界：每轮每个 deficient task 最多增加一个 root；同一轮不同 task 的 correction slots 可以合并。新 root 的真实 anchor support 可能改变下一轮是否需要 correction，因此不能无条件预生成所有 correction roots。

### 5.5 Root 合并和日志

packed 同时维护 root_batches 和新增 root_logs。每个 wave 只解析新增 output，topology 确定后才执行一次 _concat_batches。最终断言：

~~~text
generated_roots == final_roots
discarded_roots == 0
R + Q == B
~~~

## 6. Root Event、Action Identity 和 Anchor

build_root_event_logs 将 flattened DataProto 按 traj_uid 聚合：

~~~text
row -> step_index/occurrence_id
    -> pre/post observation
    -> prompt/response tokens
    -> raw response/projected action
    -> admissible action set
    -> action identity
    -> reward/done/remaining horizon
    -> RootEvent
~~~

action identity 独立保存格式和环境有效性：

~~~text
format invalid       -> unparsed / INVALID
format valid + env valid   -> valid::<canonical action>
format valid + env invalid -> invalid::<canonical action>
~~~

AnchorIndex 按 (task_id, hash(anchor_obs)) 聚合 structural anchor，保存 origins_by_action、observed_action_ids、observation、admissible action set 和 natural origin references。默认 strict_identity 不自动跨 anchor/action fallback。

## 7. Dynamic topology

DynamicTopologyPlanner 的 staged 状态：

~~~text
posterior
readiness
planned_branch_count
current root_count
current branch_count
effective_anchor_count
~~~

规划只读 natural roots：

~~~text
posterior -> readiness -> initial R/Q
anchor support -> capacity correction
finalize -> immutable TopologyPlan
~~~

最终硬约束：

~~~text
R + Q = B
Q <= max_branches_per_anchor * effective_anchor_count
~~~

finalize_staged 会拒绝实际 generated root 数不等于最终 target；history 更新只接收最终 selected natural roots。

## 8. ERV、Replay 和 branch

### 8.1 Coordinator 初始化

ExpectedErvCoordinator.initialize 在 frozen natural roots 上创建 AnchorIndex、初始化 Beta local posterior、筛选 effective anchors，并使用 topology 给出的每 task branch quota。没有可用 anchor 时记录 NO_EFFECTIVE_ANCHOR，不强行制造 branch。

### 8.2 ERV round

每个 task 每轮最多创建一个 request：

~~~text
过滤达到 max_branches_per_anchor 的 anchor
计算候选 ERV utility/regret
选择 utility 最高 anchor
按 posterior probability 选择 action identity
从 frozen natural origins 选择 origin
构造 ReplayRequest
~~~

request 固定保存 action prefix、prefix observations、reset key、expected anchor/action set、copied response tokens/loss mask/old log-probs、prompt tokens、post-action observation/reward/done 和 remaining horizon。

### 8.3 Strict Replay

branch pool 先恢复 reset key 和 action prefix，检查 anchor observation、anchor key 和 action set。若失败，可在同 anchor/action 下尝试 alternate frozen origin；仍失败则放弃 request。通过 initial validation 后执行复制的原始 action，再验证 action identity、post-action observation、reward、done。只有两阶段都通过的 request 才能进入 suffix 和训练 batch。

### 8.4 Branch suffix 和 posterior update

通过 origin transition 后生成剩余 horizon。suffix occurrence 写入自己的 occurrence_id 和 lineage，但使用同一 branch leaf。terminal reward 返回后更新 selected anchor/action posterior，suffix 中同 frozen support 的 anchor/action 也可同步更新，然后继续下一 ERV round。ERV round 之间有依赖，不能无条件全部并行。

## 9. PPO batch、advantage 和 update

最终 batch：

~~~text
[root_output]
+ [branch_origin_output per valid request]
+ [branch_suffix_output]
-> _concat_batches
~~~

每行设置 source_type、leaf_id、traj_uid、occurrence_id、task_batch_index。

Ray trainer 随后执行 reward、invalid-action penalty、KL、old/ref log-prob、advantage 和 actor update。

BACE advantage：

~~~text
leaf_credit = task 内 terminal leaf reward 的归一化结果
local_credit = 同 anchor occurrence reward 的归一化结果
occurrence_credit = leaf_credit + step_advantage_w * local_credit
token_advantage = occurrence_credit broadcast 到 response tokens
~~~

默认 local_credit_mode=occurrence。同一 leaf 的 occurrence 共享 leaf credit；branch origin token/log-prob 来自 natural origin。

## 10. Artifact 和反查路径

每个 step 目录：

~~~text
rollout_dir/bace_trace/step_XXXXXXXX/
~~~

主要文件：

| 文件 | 内容 | 排查用途 |
|---|---|---|
| manifest.json | schema、git commit、方法配置 | 实验身份 |
| roots.jsonl | natural root/event | root 和 terminal |
| leaves.jsonl | natural occurrence | action identity/token |
| anchors.jsonl | anchor/action origins | AnchorIndex |
| topology.jsonl | posterior、R/Q、capacity | topology |
| acquisition_rounds.jsonl | ERV candidates/selected action | 选择决策 |
| replay_attempts.jsonl | 每次 Replay 结果 | failure category |
| branches.jsonl | origin、reward、suffix IDs | branch lineage |
| posterior_snapshots.jsonl | 每轮后验 | ERV 更新 |
| trainable_occurrences.jsonl | PPO occurrence/advantage/log-prob | loss 排查 |
| summary.json | record counts/metrics/diagnostics | 单步审计 |

推荐反查链：

~~~text
trainable_occurrences.leaf_id
  -> branches.branch_id
  -> branches.origin_occurrence_id
  -> leaves.occurrence_id
  -> roots.root_id/environment_reset_key
  -> replay_attempts.request_id
  -> acquisition_rounds.selected_anchor/action
~~~

validator 检查 record index/schema/step、ID 唯一性、R+Q=B、branch capacity、copied origin token/loss-mask/old-log-prob、frozen support 和 log-prob 差异阈值。

## 11. 计时指标和排查顺序

当前 collector 记录：

~~~text
root_reset_key_probe_seconds
pilot_root_waves / pilot_root_trajectories / pilot_root_generation_seconds
completion_root_waves / completion_root_trajectories / completion_root_generation_seconds
capacity_correction_waves / capacity_correction_generation_seconds
root_event_logging_seconds
root_batch_concat_seconds
training_batch_concat_seconds
branch_suffix_generation_seconds
replay_validation_seconds
artifact_write_seconds
staged_root_batching_packed
~~~

排查顺序：

1. packed root_generation_waves 是否约为 2；
2. pilot/completion generation seconds 是否因 batch 变宽下降；
3. root_batch_concat_seconds 是否从多次复制变为一次；
4. branch suffix 是否只在 branch-active step 增长；
5. Replay 是否仍保持秒级；
6. artifact write 是否与 token-array 开关成比例；
7. old log-prob、reference log-prob、actor update 是否成为新主瓶颈；
8. 显存、vLLM batch、Ray worker 利用率是否异常。

## 12. 优化路线

### 第一阶段：sequential/packed 短对照

固定同一 parquet、seed、模型、保存策略，先跑 3 step：

~~~bash
BACE_TOTAL_EPOCHS=3 \
BACE_RUN_NAME=bace_staged_sequential_3step \
BACE_STAGED_ROOT_BATCHING=sequential \
bash examples/bace_gigpo/run_alfworld_npu_8card_50epoch.sh

BACE_TOTAL_EPOCHS=3 \
BACE_RUN_NAME=bace_staged_packed_3step \
BACE_STAGED_ROOT_BATCHING=packed \
bash examples/bace_gigpo/run_alfworld_npu_8card_50epoch.sh
~~~

两次实验应设置相同 BACE_DATA_ROOT，避免 parquet 差异进入对照。

### 第二阶段：active-only generation

当前 multi-turn loop 在 wave 内只要有一个环境未 done，就会为全部 rows 调用模型；已 done rows 最后被 mask，但生成计算已经发生。后续可新增可选 active-only 路径：

~~~text
active_indices
  -> preprocess/generate/step
  -> environment worker mapping
  -> restore original positions
~~~

必须保留 fixed-shape 旧路径，并测试 task/worker/occurrence 映射。

### 第三阶段：branch suffix batching

按 remaining horizon 对同一 ERV round 的 suffix 分桶，减少 done row 和 horizon padding。不能破坏 branch session isolation、action identity 和 suffix lineage。

### 第四阶段：公平方法实验

当前 50 epoch 对照主要用于工程诊断，并非最终效果结论，因为 GiGPO 使用 rollout.n=8，BACE 使用 B=4。后续分别做：

~~~text
equal budget:   GiGPO n=4 vs BACE B=4
original scale: GiGPO n=8 vs BACE B=8
~~~

同时冻结 train/val parquet、seed、upstream commit、环境版本、validation、checkpoint 规则和 artifact 策略。

## 13. 最终判断

旧版 BACE 的方法链路是完整的，但 staged sequential 的 root 调度粒度不适合 NPU 高吞吐。当前 packed 版只改变 root 编排和数据合并方式，保留原 sequential/preallocated 路径以及 topology、ERV、Replay、advantage 语义。

正确的下一步是先比较 sequential 与 packed 的 wave-level timing；在 packed 性能确认前，不应先修改 Replay identity 或 advantage 公式。
