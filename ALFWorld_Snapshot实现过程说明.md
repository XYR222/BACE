# ALFWorld Snapshot 实现过程说明

> **状态提示（2026-08-26）：历史/设计记录。** 本文描述的 `snapshot/state_id.py`、`StateIDSnapshotBackend` 和 `get_state_id/set_state_id` 当前不在本 Git tree 中，不能按已实现功能使用。当前有效实现仍是 prefix/fast replay。请先阅读 [`docs/01_当前实现状态与未完成事项.md`](docs/01_当前实现状态与未完成事项.md) 和 [`docs/06_文档有效性清单.md`](docs/06_文档有效性清单.md)。

本文说明当前 BACE-GiGPO 中 ALFWorld snapshot 的实现思路、代码改动、运行流程、关键校验和已知限制。这里的 snapshot 只用于加速 branch 恢复，不改变 BACE 的 anchor、ERV、branch 分配、优势或 loss 定义。

## 1. 目标与基本原则

原来的 ALFWorld replay 是：

```text
reset 同一个 game
  -> 依次执行 anchor 前缀动作
  -> 校验恢复后的 observation / admissible actions
  -> 执行 branch-origin action
  -> 生成 branch suffix
```

anchor 深度为 $d$ 时，恢复成本近似为 $O(d)$。BACE 中同一 anchor 可能被多个 branch 使用，重复 replay 会成为明显的系统开销。

实现目标是把恢复改为：

```text
在自然 root 执行到 anchor 时保存 StateID
  -> branch 请求到达
  -> 在同一个 Fast-Downward registry 中 set_state_id(StateID)
  -> 同步 Python wrapper 状态
  -> 校验恢复结果
  -> 执行 branch-origin action 和 suffix
```

核心原则：

1. 不序列化整个 ALFWorld/TextWorld 环境。
2. 不把 `StateID` 当作跨进程、跨 reset 的全局 ID。
3. snapshot 必须绑定产生它的 physical context 和 registry generation。
4. prefix replay 保留为正确性基线和 `auto` 后端的回退路径。
5. snapshot 失败时，显式 `state_id`/`snapshot` 后端直接报错；只有 `auto` 后端允许回退到 fast replay。

## 2. 为什么使用 StateID

ALFWorld 的 TextWorld PDDL 环境结构大致是：

```text
AlfredTWEnv
  -> TextworldBatchGymEnv
  -> TextWorld PddlEnv
  -> PddlState
  -> Fast-Downward native library
  -> StateRegistry / StateID
```

Fast-Downward 已经维护了符号状态注册表，并暴露：

```python
get_state_id()
set_state_id(state_id)
```

因此 snapshot 保存的不是完整 native 对象，而是当前 registry 中的状态索引，再保存恢复 Python 层所需的少量元数据。只要 registry 仍然存活，恢复可以绕过整个动作前缀，接近常数时间。

重要限制是：

```text
StateID 只在原来的 StateRegistry 生命周期内有效。
```

环境 reset 后 registry generation 会改变，旧 snapshot 必须失效；不同 Ray worker 之间也不能直接传递 StateID。

## 3. Physical Context 与 Logical Session

实现中区分两类对象。

### 3.1 Physical Context

由一个 `AlfworldWorker` 持有，包含：

- 一个 batch size 为 1 的 TextWorld 环境；
- Fast-Downward native library；
- 当前 `StateRegistry`；
- 当前 `registry_generation`；
- snapshot store。

每个 Ray worker 是独立 physical context。worker 内部可以复用同一个 registry，但不能把 StateID 直接交给另一个 worker。

### 3.2 Logical Session

一个 logical session 表示一个 root/branch 当前的恢复游标：

```text
logical_session_id
snapshot_id
context_id
registry_generation
closed
```

同一个 natural-root snapshot 可以 fork 出多个 logical session。每个 session 恢复、执行一步、再保存自己的 continuation snapshot，因此一个 branch 的推进不会改变另一个 branch 的游标。

## 4. 主要代码改动

### 4.1 新增 `snapshot/state_id.py`

核心类是 `StateIDSnapshotBackend`，主要职责如下：

#### 上下文发现与能力检查

`_resolve_context()` 从 wrapper 链中定位：

- `SyncBatchEnv`；
- 唯一的底层子环境；
- TextWorld `PddlEnv` 和 `_pddl_state`；
- `Limit` wrapper；
- native `downward_lib`。

它还为 ctypes 函数设置签名：

```python
native.get_state_id.argtypes = []
native.get_state_id.restype = c_int
native.set_state_id.argtypes = [c_int]
native.set_state_id.restype = None
```

如果不是单环境 `SyncBatchEnv`、找不到 PDDL 状态、找不到 episode limit 或 native library 没有 StateID API，则抛出 `SnapshotUnavailableError`。

#### reset 时建立新 generation

`on_reset()` 在 wrapper reset 成功后调用：

```text
registry_generation += 1
清空旧 snapshots
清空旧 logical sessions
记录 game_file
```

这样可以防止旧 game 或旧 registry 的 StateID 被错误复用。

#### capture

`capture(metadata)` 保存：

- `state_id`；
- `context_id` 和 `registry_generation`；
- `env_step`、`remaining_horizon`、`done`；
- observation hash；
- admissible actions hash；
- PDDL facts hash；
- observation、admissible actions、feedback/raw；
- `last_command`、`_moves`、`_last_action`；
- `extra.*` state fields；
- BACE 关联信息：`task_id`、`root_id`、`occurrence_id`、`anchor_key`。

snapshot handle 只是 store 的 key 和校验信息，不是可脱离 worker 的序列化环境。

#### restore

恢复分为四步：

1. 校验 snapshot 是否存在、context 是否相同、registry generation 是否仍有效。
2. 调用 `set_state_id(handle.state_id)` 恢复 native symbolic state。
3. 根据保存的 facts 重建 `PddlState` 的 Python 索引：`_facts`、变量表、计数器等。
4. 重建 TextWorld `GameState`、`feedback`、`raw`、`last_command`、`_last_action`、`_moves`、infos、episode step 和 batch observation。

恢复后必须通过以下检查：

```text
state_id 一致
observation hash 一致
admissible actions hash 一致
facts hash 一致
remaining horizon 一致
```

任一检查失败都抛出 `SnapshotRestoreError`，不允许带着未经验证的状态继续生成 branch。

#### fork 与 continuation

`fork(snapshot_id, logical_session_id)` 创建 session，但不复制整个环境。

`restore_session(session_id)` 先恢复 session 当前 snapshot。

`advance_session(session_id)` 在 branch 执行一步后捕获新 snapshot，并删除该 session 自己的旧 continuation snapshot。这样 session 游标始终只保留最新位置，而共享的 natural-root snapshot 不会被误删。

`close_session()` 和 `release_snapshot()` 负责释放 session-owned 或独立 snapshot；`close()` 清空 worker 内全部 snapshot/session。

### 4.2 修改 `alfworld/envs.py`

#### worker 层

`AlfworldWorker` 初始化一个 `StateIDSnapshotBackend`，并新增：

- `snapshot_capabilities()`
- `capture_snapshot()`
- `restore_snapshot()`
- `fork_snapshot()`
- `restore_logical_session()`
- `logical_step()` / `logical_step_batch()`
- `release_snapshot()`
- `close_logical_session()`

`reset()` 在环境 reset 后调用 `snapshot_backend.on_reset()`。若当前环境不支持 StateID，只记录不可用原因，不影响原始 fast replay。

`logical_step()` 的顺序是：

```text
restore_session
  -> 执行一个原始环境 action
  -> advance_session 保存下一位置
  -> 返回 observation/reward/done/info/新 handle
```

#### vector/Ray 管理层

`AlfworldEnvs` 新增 selected-worker API：

- `snapshot_capabilities_selected()`
- `capture_snapshots_selected()`
- `restore_snapshots_selected()`
- `fork_snapshots_selected()`
- `restore_logical_sessions()`
- `logical_step_sessions()`

这些接口会把请求按 worker 分组，再恢复原请求顺序。一个 physical worker 可以承载多个 logical session，因此不能简单假设“一个请求对应一个 worker”；同一 worker 的 session 操作必须串行执行并保持各自游标。

### 4.3 修改 `env_manager.py`

新增 `AlfWorldSnapshotBranchManager`，它把 BACE 的 branch request 映射为 logical sessions。

`replay(requests)` 的流程：

1. 检查每个 request 都有 natural-root snapshot 和 worker index。
2. 关闭上一轮遗留 session。
3. 使用 request 的 snapshot ID fork logical session。
4. 恢复所有 session。
5. 为每个 branch 构造 wrapper 状态：任务描述、prefix history、当前 observation、admissible actions、done。
6. 返回 branch manager 需要的 observation/done/info。

后续 `step(text_actions)`：

1. 用 ALFWorld projection 将模型文本动作解析成环境动作。
2. 按当前 admissible action pool 判断格式有效性和环境有效性。
3. 对每个 logical session 调用 `logical_step_sessions()`。
4. 更新 Python wrapper history、observation、admissible actions 和 done。
5. 返回 reward、done、validity 信息。

因此 branch suffix 的每一步都恢复自己的 logical session，再执行一步；不会把不同 branch 的状态混到一起。

## 5. Collector 中的接入方式

`BACE` collector 根据 `algorithm.bace.replay.backend` 选择：

```text
fast_replay -> reset + prefix action replay
state_id    -> ALFWorld StateID snapshot
snapshot    -> 当前显式 snapshot backend
auto        -> 优先 snapshot，失败后回退 fast replay
```

当后端是 `state_id`、`snapshot` 或 `auto` 时，collector：

1. 创建 snapshot branch manager；
2. 用 `ReplayAdapter.replay_and_validate()` 对整轮请求做 preflight；
3. 只有全部 request 通过恢复校验后，才执行 branch-origin action 和 suffix；
4. 将 restore profile、restore 时间、失败类别写入 trace；
5. round 结束后关闭 manager。

显式 snapshot 后端遇到错误会抛出，避免悄悄改变实验语义；`auto` 才允许记录失败并使用 prefix replay。

collector 的 `multi_turn_loop()` 使用 `finally` 清理 root env snapshot store，因此成功、异常和中断路径都不会长期保留旧 snapshot。

## 6. 与 BACE 训练语义的边界

snapshot 只替换环境恢复实现，不改变以下内容：

- natural root 的生成与筛选；
- anchor key 和有效动作 identity；
- branch-origin occurrence 的复制规则；
- branch suffix 的冻结策略；
- ERV / Batch-ERV 计算；
- root、branch、suffix 的 reward 计算；
- 轨迹级优势和 step-level 优势；
- PPO/LoRA loss；
- replay prefix 是否进入训练 buffer 的规则。

恢复本身不产生训练 token，也不应把 prefix replay 的动作重复加入训练样本。只有 branch-origin 和 suffix 按现有 BACE 逻辑进入后续统计/训练。

## 7. 生命周期与内存管理

snapshot 分三类理解：

1. **共享 natural-root snapshot**：由 root occurrence 创建，供一个或多个 branch fork 使用。
2. **session-owned continuation snapshot**：branch 每执行一步后生成，只属于该 logical session。
3. **临时 fork/session 元数据**：记录 session 与当前 snapshot 的映射。

释放规则：

- root reset 时清空上一 registry generation 的全部 snapshot；
- session 前进时删除自己旧的 continuation snapshot；
- session close 时删除最后一个 session-owned snapshot；
- natural-root snapshot 在显式 release 前保留，避免仍被其他 branch 使用；
- collector 每轮结束统一 `clear_snapshots()`。

## 8. 正确性验证方法

### 8.1 单环境 StateID 测试

`tests/bace_gigpo/test_alfworld_snapshot.py` 覆盖：

- 执行多个 state-changing action 后 capture；
- 修改环境，再 restore；
- observation、admissible actions、facts、StateID、episode horizon 一致；
- restore 后继续执行相同动作，与 reference continuation 完全一致；
- 从同一 snapshot fork 两个 session；
- 两个 session 的 continuation 相互隔离；
- session cursor 推进后旧 continuation snapshot 被回收；
- reset 后旧 handle 失效。

### 8.2 20-game 差分验证

ALFWorld 验证脚本以 prefix replay 作为 oracle，比较 StateID restore：

```text
prefix 执行结果
扰动后的环境
snapshot restore 结果
restore 后 continuation
facts / observation / admissible actions / horizon
```

验收不只看进程退出码，还要检查每个 game 的语义结果。已有 replay-equivalence 运行产物位于：

```text
experiments/replay_equivalence/
```

### 8.3 训练 smoke

训练 smoke 至少检查：

- StateID backend 成功初始化；
- snapshot capture 数量大于零；
- BACE branch request 的 restore preflight 通过；
- restore steps 为 0，而不是 prefix replay 的动作数；
- branch-origin 和 suffix 能继续执行；
- artifact trace 中无 fallback 或 restore mismatch；
- reward、advantage、old/ref log prob、loss 和 actor update 正常完成。

## 9. 当前限制与注意事项

### 9.1 只支持特定 ALFWorld 模式

当前实现针对 TextWorld `.tw-pddl`，并要求底层结构是单环境 `SyncBatchEnv`。Thor/multimodal 环境不应直接假设支持同一个 StateID backend。

### 9.2 不支持跨 worker 直接搬运 StateID

StateID 和 registry 都属于 source worker。若 branch 必须在另一个 worker 上执行，不能只传 `state_id`；需要额外的可移植 snapshot 格式，或回退到 prefix replay。当前 ALFWorld 实现的 snapshot manager 在 natural-root 所属 worker 上创建 logical sessions。

### 9.3 reset 会使旧 snapshot 失效

任何 reset 都会推进 `registry_generation` 并清空 store。不能把旧 handle 缓存到下一轮训练继续用。

### 9.4 Python wrapper 同步不能省略

仅调用 `set_state_id()` 不足以保证 TextWorld wrapper 正常工作。必须同步 facts、GameState、moves、last action、infos、batch observation、episode step 和 horizon，否则可能出现 observation 看似正确但 admissible actions、reward 或 done 错误的情况。

### 9.5 branch 数量受 physical worker 容量限制

同一 worker 的 logical sessions 虽然可以复用一个 context，但 `logical_step_batch()` 仍在该 worker 内串行恢复和执行。实际吞吐取决于 worker 数量、请求分组和 branch wave；不能把 StateID snapshot 理解成无限并行。

## 10. 一句话总结

当前 ALFWorld snapshot 的本质是：

> 在同一个存活的 TextWorld/Fast-Downward 状态注册表中保存 natural-root 的 `StateID`，用 `set_state_id()` 直接恢复 native 状态，再完整重建和校验 Python wrapper 与 episode bookkeeping；通过 logical session 支持多个相互隔离的 branch，并在生命周期结束时回收 snapshot。它改变的是 replay 的执行成本，不改变 BACE-GiGPO 的训练语义。
