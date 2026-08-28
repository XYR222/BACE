# BACE-GiGPO 早期训练信号与零优势差异调研

日期：2026-08-16  
对象：`bace_alfworld_npu_8card_150epoch_baseline_fix3_mb4` 与  
`gigpo_qwen2.5_1.5b_npu_8card_150epoch_tp1_mb8_s40_lowval`

## 1. 调研问题

本调研验证以下预期是否成立：

> 训练初期成功率很低，BACE 应该几乎不产生 branch，因此应当接近 GiGPO。

结论是：这个预期只在 **rollout topology 数量** 上大致成立，在 **PPO advantage / gradient 信号** 上不成立。早期差异的主要来源不是 branch 数量，而是 BACE 与 GiGPO 使用了不同的 global credit 定义。

## 2. 数据与统计口径

### 2.1 BACE

读取每个已完成 step 的：

```text
bace_artifacts/<run>/step_xxxxxxxx/trainable_occurrences.jsonl
bace_artifacts/<run>/step_xxxxxxxx/summary.json
bace_artifacts/<run>/step_xxxxxxxx/branches.jsonl
```

统计：

- natural root 数、branch leaf 数；
- root/branch 成功率；
- 每个 task 的 8-leaf outcome 是否同质；
- `leaf_advantage`、`local_advantage`、`occurrence_advantage` 的零比例；
- occurrence 零比例和 response-token 零比例；
- branch 与 natural origin 成败是否一致；
- strict identity 下的 invalid branch 比例。

零值定义为：

```text
abs(advantage) < 1e-8
```

### 2.2 GiGPO

读取：

```text
anchor_groups/<run>/anchor_groups_step_xxxxxx.jsonl
```

每个 action occurrence 中直接使用：

```text
macro_adv
micro_adv
combined_gigpo_adv
terminal_reward
```

GiGPO artifact 没有完整 response-token mask，因此 GiGPO 对照主要是 occurrence-level；BACE 同时报告 occurrence-level 和 token-level。

### 2.3 重要可比性限制

两次训练不是同一个进程，不能把下面的结果解释为严格的 paired causal experiment：

- 不保证每个 step 的模型采样完全相同；
- BACE 使用 actor micro-batch 4，GiGPO 使用 8；
- BACE 的 branch 使最终 occurrence 集合不同；
- GiGPO artifact 的 zero 指标来自 action occurrence，不是 token。

但是，两次运行使用相同的 ALFWorld 任务规模、8 卡基座、`max_steps=40`、验证频率 10，统计足以定位训练信号结构差异。

## 3. 训练初期 topology 是否接近 GiGPO

GiGPO 每个 task 固定为 8 个 natural roots，即每个 step 为：

```text
16 tasks × 8 leaves = 128 natural roots
```

BACE 旧版实际使用 `pilot_roots=2`，再根据 readiness 和 capacity 动态确定 `R + Q = 8`。

### 3.1 前 30 step 的实际 topology

| Step | Natural roots | Branches | Branch 占 128 slots | Root success | Branch success |
|---:|---:|---:|---:|---:|---:|
| 1 | 110 | 18 | 14.1% | 5.5% | 11.1% |
| 2 | 120 | 8 | 6.3% | 7.5% | 25.0% |
| 3 | 125 | 3 | 2.3% | 9.6% | 0.0% |
| 4 | 128 | 0 | 0.0% | 10.2% | - |
| 5 | 123 | 5 | 3.9% | 10.6% | 80.0% |
| 6 | 125 | 3 | 2.3% | 12.0% | 100.0% |
| 7 | 122 | 6 | 4.7% | 10.7% | 50.0% |
| 8 | 122 | 6 | 4.7% | 9.8% | 16.7% |
| 9 | 123 | 5 | 3.9% | 8.9% | 40.0% |
| 10 | 123 | 5 | 3.9% | 9.8% | 60.0% |

Step 1 的 18 branches 是初始化 outlier。Step 2--20 平均约为：

```text
122.8 natural roots + 5.2 branches
```

整个 step 1--20 平均约为：

```text
122.45 natural roots + 5.55 branches
```

也就是约 95.7% 的 leaves 仍然是 natural roots。因此，用户关于“训练初期不应大量 branching”的直觉，在 **slot 数量** 上基本正确，但 step 1 确实存在一次由初始化 prior 造成的较大 branch outlier。

### 3.2 18 条 branch 是否手动指定

不是手动指定。旧版 planner 的逻辑是：

```text
readiness = P(task competence > threshold | pilot + history)
planned_Q = round((B - pilot_roots) × readiness)
planned_R = B - planned_Q
```

随后如果 natural roots 暴露出的 effective anchors 不足以支撑 `Q` 条 branch，capacity correction 会把 branch slot 转成 root slot。

因此 step 1 的 `110 + 18` 是 planner 计算结果，不是脚本中写死的 `18`。

### 3.3 topology 结论

训练初期 branch 数量确实较少，不能单独解释 BACE 与 GiGPO 的巨大差距。必须继续检查 advantage 定义。

这里的“成功率低”是整个 batch 的平均。Planner 是按 task 分别估计 readiness，不是看 batch 平均成功率做一次总开关。因此，少数 task 在 pilot roots 中成功，或其 history posterior 仍高于阈值时，仍会得到 1--数个 branch slot；其他低 readiness task 则仍使用 8 个 natural roots。所以 step 2--20 平均 5.2 个 branch 与低成功率并不矛盾，它只占 128 slots 的 4.1%。

## 4. 全量分阶段结果

下面是 step 1--116 的分段平均。BACE 的 `root_s` 是 natural-root 成功率，GiGPO 的 `leaf_s` 是其全部 natural leaves 成功率，二者不是完全相同的统计对象；这里只用于显示训练阶段变化。

| Steps | BACE R/B | BACE root_s | GiGPO leaf_s | BACE 同质 task | GiGPO 同质 task | BACE zero global | BACE zero local | BACE zero combined | GiGPO zero combined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1--10 | 122.1/5.9 | 9.5% | 7.7% | 12.30/16 | 11.00/16 | 81.3% | 56.4% | 52.5% | 27.2% |
| 11--20 | 122.8/5.2 | 11.2% | 9.9% | 12.00/16 | 10.90/16 | 80.7% | 63.3% | 59.0% | 51.8% |
| 21--30 | 116.4/11.6 | 15.6% | 17.3% | 11.10/16 | 9.30/16 | 72.8% | 58.4% | 50.2% | 38.0% |
| 31--40 | 112.8/15.2 | 18.6% | 24.1% | 10.70/16 | 7.30/16 | 70.9% | 52.7% | 45.6% | 34.7% |
| 41--50 | 111.4/16.6 | 20.5% | 37.2% | 9.50/16 | 7.00/16 | 65.0% | 54.1% | 45.4% | 29.5% |
| 51--60 | 107.7/20.3 | 22.7% | 45.0% | 9.40/16 | 7.10/16 | 63.1% | 51.7% | 43.1% | 23.5% |
| 61--70 | 104.4/23.6 | 22.1% | 59.1% | 10.20/16 | 7.80/16 | 69.0% | 50.1% | 44.3% | 22.8% |
| 71--80 | 100.4/27.6 | 23.7% | 58.9% | 10.10/16 | 5.60/16 | 66.1% | 51.0% | 44.3% | 19.3% |
| 81--90 | 102.3/25.7 | 30.5% | 60.2% | 10.00/16 | 6.40/16 | 64.0% | 47.6% | 39.9% | 17.5% |
| 91--100 | 105.5/22.5 | 39.4% | 68.1% | 9.80/16 | 7.90/16 | 66.0% | 54.6% | 46.8% | 19.8% |
| 101--110 | 100.1/27.9 | 47.1% | 79.8% | 10.90/16 | 9.80/16 | 65.7% | 52.8% | 44.2% | 14.6% |
| 111--116 | 100.7/27.3 | 50.0% | 80.7% | 10.17/16 | 11.00/16 | 57.0% | 55.1% | 43.1% | 21.9% |

### 4.1 精确 checkpoint 对照

下表不做滑动平均，直接列出原始 artifact 中的单 step 值。前 10 step 逐 step 展开，之后每 5 step 取一个 checkpoint，并额外展开 step 111--116。

`BACE zero occ` 和 `GiGPO zero occ` 都是 action occurrence 口径；`BACE zero token` 按 response token 数加权。GiGPO artifact 没有完整 token mask，因此没有伪造 GiGPO token 口径。GiGPO success 按 `trajectory_id` 去重后的 128 条 terminal leaves 计算，不是按较长 trajectory 中更多的 action occurrences 加权。

| Step | BACE R/B | BACE root success | BACE zero occ | BACE zero token | GiGPO success | GiGPO zero occ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 110/18 | 5.5% | 20.6% | 20.7% | 3.1% | 0.0% |
| 2 | 120/8 | 7.5% | 27.2% | 26.9% | 7.0% | 0.0% |
| 3 | 125/3 | 9.6% | 48.2% | 47.0% | 4.7% | 31.9% |
| 4 | 128/0 | 10.2% | 53.4% | 52.9% | 10.9% | 13.1% |
| 5 | 123/5 | 10.6% | 64.9% | 63.7% | 6.2% | 32.0% |
| 6 | 125/3 | 12.0% | 54.9% | 53.4% | 9.4% | 39.2% |
| 7 | 122/6 | 10.7% | 62.3% | 61.6% | 7.0% | 39.0% |
| 8 | 122/6 | 9.8% | 59.6% | 59.0% | 8.6% | 38.8% |
| 9 | 123/5 | 8.9% | 60.2% | 59.2% | 7.0% | 39.0% |
| 10 | 123/5 | 9.8% | 74.4% | 73.1% | 13.3% | 40.1% |
| 15 | 126/2 | 7.9% | 56.0% | 55.4% | 8.6% | 58.6% |
| 20 | 121/7 | 17.4% | 52.1% | 50.6% | 7.0% | 51.4% |
| 25 | 118/10 | 16.1% | 40.7% | 40.8% | 8.6% | 52.0% |
| 30 | 115/13 | 16.5% | 57.7% | 57.3% | 21.1% | 14.4% |
| 35 | 112/16 | 21.4% | 45.2% | 46.0% | 29.7% | 23.2% |
| 40 | 110/18 | 18.2% | 38.9% | 39.0% | 22.7% | 42.3% |
| 45 | 112/16 | 18.8% | 49.6% | 50.7% | 36.7% | 15.9% |
| 50 | 110/18 | 22.7% | 46.6% | 46.8% | 39.1% | 50.5% |
| 55 | 112/16 | 23.2% | 46.9% | 45.6% | 26.6% | 37.0% |
| 60 | 106/22 | 27.4% | 43.4% | 41.7% | 44.5% | 18.2% |
| 65 | 104/24 | 23.1% | 41.1% | 41.1% | 63.3% | 25.3% |
| 70 | 103/25 | 24.3% | 44.1% | 42.8% | 58.6% | 11.5% |
| 75 | 104/24 | 24.0% | 36.1% | 36.1% | 53.9% | 29.2% |
| 80 | 101/27 | 23.8% | 45.1% | 44.3% | 35.9% | 22.9% |
| 85 | 102/26 | 30.4% | 43.9% | 45.2% | 66.4% | 21.9% |
| 90 | 103/25 | 32.0% | 34.7% | 34.8% | 50.0% | 9.7% |
| 95 | 106/22 | 40.6% | 37.8% | 36.8% | 75.0% | 4.3% |
| 100 | 107/21 | 43.9% | 55.2% | 53.8% | 68.0% | 14.7% |
| 105 | 104/24 | 48.1% | 50.9% | 50.7% | 87.5% | 14.6% |
| 110 | 100/28 | 48.0% | 43.8% | 41.5% | 69.5% | 28.3% |
| 111 | 98/30 | 48.0% | 44.2% | 43.4% | 85.9% | 5.6% |
| 112 | 104/24 | 52.9% | 36.3% | 34.3% | 93.0% | 18.3% |
| 113 | 97/31 | 50.5% | 45.7% | 44.6% | 77.3% | 30.9% |
| 114 | 100/28 | 52.0% | 34.5% | 34.3% | 89.8% | 10.9% |
| 115 | 102/26 | 45.1% | 56.3% | 55.8% | 56.2% | 33.2% |
| 116 | 103/25 | 51.5% | 39.8% | 40.3% | 82.0% | 22.2% |

这张表显示了两个不能被分段平均代替的细节：

1. Step 1--2 的 BACE zero 并不高，这与初始化分支和它们形成的 local comparison 有关；从 step 3 开始快速升高，step 10 达到 74.4%。因此“早期差异”是连续阶段结论，不是声称每一个早期 step 都更差。
2. 单 step 方差很大，例如 step 40 和 50 的大小关系会短暂反转。因此判断系统性趋势应使用上一张 10-step 分段平均，精确 checkpoint 主要用于追溯异常。

### 4.2 关键观察

1. 在 step 1--20，BACE branch slot 只有约 4%，但 1--10 和 11--20 两个阶段的 combined-zero 均值分别比 GiGPO 高 25.3 和 7.2 个百分点。
2. 从 step 21 开始 branch 比例升高，但 BACE zero combined 长期仍在 39--50%，并没有因为 branch 增多而消失。
3. GiGPO 的 zero combined 随训练总体下降；BACE 没有同步下降，在成功率较高的 step 101--110 仍约 44.2%。
4. BACE 的同质 task 数始终较高，但这只能解释 global leaf term 归零，不能解释 BACE 与 GiGPO 的全部差距。

## 5. 为什么低成功率时期两者仍不等价

### 5.1 BACE global term 的实际定义

见 [recipe/bace_gigpo/advantage.py](/home/naie/work/work-BACE/verl-agent-src/recipe/bace_gigpo/advantage.py:43)：

```python
leaf_rewards = data.non_tensor_batch["episode_rewards"]
```

BACE 先按 `(task_id, leaf_id)` 去重，再对 8 个 terminal leaves 做归一化：

```text
A_leaf = normalize({terminal reward of each unique leaf})
```

因此，只要一个 task 的 8 个 terminal outcomes 全部相同：

```text
all success -> A_leaf = 0
all failure -> A_leaf = 0
```

这个归零是代码明确实现的行为，不是数值 bug。

### 5.2 GiGPO macro term 的实际定义

GiGPO 在 [core_gigpo.py](/home/naie/work/work-BACE/verl-agent-src/gigpo/core_gigpo.py:160) 中使用：

```python
token_level_rewards
```

而不是 BACE 的 unique `leaf_rewards`。

在训练循环中，Invalid Action penalty 先被注入：

[ray_trainer.py](/home/naie/work/work-BACE/verl-agent-src/verl/trainer/ppo/ray_trainer.py:201)

```python
reward_tensor[..., last_valid_token] -= 0.1 * invalid_action
step_rewards -= 0.1 * invalid_action
```

然后 GiGPO 的 macro normalization 使用这些 occurrence-level reward scores。

因此，哪怕终局成功率全部为 0，只要不同 occurrence 的格式 invalid 数不同，GiGPO 仍然可能获得非零 macro advantage。

BACE 的 local term 会收到这项 penalty，因为 `step_rewards` 被原地修改；但是 BACE 的 global leaf term 仍只读取未包含该 penalty 的 `episode_rewards`。

### 5.3 GiGPO 还不是 unique-leaf normalization

GiGPO 的 `episode_norm_reward` 默认 `compute_mean_std_cross_steps=True`，会对当前 task 的每个 physical occurrence 追加 score，而不是先按 unique leaf 去重：

```text
GiGPO: occurrence-level macro normalization
BACE:  unique-leaf global normalization
```

因此两者在以下方面都不同：

- BACE 每个 leaf 只贡献一次 global statistic；
- GiGPO 较长 trajectory 会贡献更多 occurrence；
- BACE global term 不含 invalid-action penalty；
- GiGPO macro term 含 token-level penalty；
- BACE branch 的 copied origin/suffix 会形成新的 occurrence，但 global reward 仍按 leaf 去重。

这意味着“成功率低，所以 BACE 应该和 GiGPO 一样”在数学上并不成立。低成功率只说明 terminal reward 的变化少，并不能保证两种 normalization 得到相同的 advantage。

## 6. 一个简化例子

假设一个 task 有 8 条轨迹，全部最终失败：

```text
terminal reward = [0, 0, 0, 0, 0, 0, 0, 0]
```

### BACE

```text
unique leaves 全部为 0
std = 0
A_leaf = [0, 0, ..., 0]
```

如果某个 anchor 只有一个 occurrence，local term 也为 0，于是这些 token 完全没有 policy-gradient signal。

### GiGPO

如果某些 occurrence 产生了格式 invalid penalty：

```text
occurrence scores = [0, 0, -0.1, 0, -0.1, 0, ...]
```

GiGPO 可以从这些 occurrence score 中计算非零 macro advantage。它学习的是“在当前 task group 中哪些 response 的格式/动作结果相对更好”，而 BACE 当前 global term 不使用这部分信息。

## 7. 反事实隔离实验：只替换 global term

为了判断差异是否主要来自 advantage，而不是 branch topology，对 BACE 当前 flat training batch 做了离线反事实：

1. 保留 BACE 原有 roots、branches、occurrences 和当前 local advantage；
2. 不重新训练，不改变 Replay；
3. 把 BACE 的 global leaf term 临时换成 GiGPO 风格的 occurrence-level macro score：

```text
score_i = terminal_reward_i - 0.1 × action_format_invalid_i
macro_i = normalize(score_i within task occurrences)
counterfactual = macro_i + current_BACE_local_i
```

这里的 `action_format_invalid_i` 对应训练循环中的 `is_action_valid` 路径；ALFWorld 环境同时保存了 `action_environment_valid` 和 `action_format_valid`，两者不应混为一谈。本反事实验只按训练循环实际注入的 format-invalid penalty 计算，并未把“环境不接受但格式合法”额外当作该 penalty。

结果：

| Steps | BACE 当前 zero occurrence | 反事实 zero occurrence | 反事实 zero token |
|---|---:|---:|---:|
| 1--10 | 52.5% | 10.1% | 9.9% |
| 11--20 | 59.0% | 20.6% | 19.8% |
| 21--30 | 50.2% | 9.7% | 9.8% |
| 31--40 | 45.6% | 9.3% | 8.4% |

若干单 step 结果：

| Step | 当前 BACE zero token | 反事实 zero token |
|---:|---:|---:|
| 50 | 45.3% | 0.14% |
| 70 | 44.2% | 0.13% |
| 90 | 39.9% | 0.03% |
| 110 | 43.2% | 20.5% |
| 115 | 约 55.8% | 约 20.0% |
| 116 | 40.3% | 1.3% |

这不是最终性能因果实验，因为没有用反事实 advantage 更新 actor；但它说明：

```text
当前 BACE 的高 zero-gradient 比例，主要可以由 global credit 定义解释，
而不能仅归因于 branch 数量。
```

## 8. Branch correlation 的影响

BACE branch 与其 natural origin 的 terminal outcome 高度同向：

| Step/阶段 | Branch-origin 与 branch 成败一致 |
|---|---:|
| Step 1 | 18/18 |
| Step 70 | 23/25 |
| Step 110 | 26/28 |
| Step 115 | 21/26 |
| 阶段平均 1--10 | 66.1% |
| 阶段平均 101--110 | 90.0% |

这反映了 branch 的采样性质：origin 来自 natural roots 中已经观察到的 anchor/action edge，branch 是对相同 edge 的 continuation resampling，不是从初始状态重新独立采样一条 trajectory。

因此 branch 主要增加的是局部 continuation evidence，而不是独立 global outcome diversity。随着 branch 比例从早期约 4% 增加到后期约 20%--22%，BACE 的 global leaf diversity 反而相对减少。

## 9. 验证结果与训练信号的关系

| Step | BACE episode success | GiGPO episode success | BACE val | GiGPO val |
|---:|---:|---:|---:|---:|
| 10 | 0.1172 | 0.1328 | 0.0938 | 0.1563 |
| 30 | 0.2188 | 0.2109 | 0.1563 | 0.2031 |
| 50 | 0.2891 | 0.3906 | 0.2031 | 0.3750 |
| 70 | 0.3672 | 0.5859 | 0.2031 | 0.4219 |
| 90 | 0.3750 | 0.5000 | 0.4063 | 0.7500 |
| 110 | 0.5078 | 0.6953 | 0.3594 | 0.8125 |

step 10 时 BACE natural-root success 并不低于 GiGPO，但 BACE advantage signal 已明显更稀疏；step 50 之后 signal density 和 validation gap 同时扩大。这支持“优化信号不足导致后续策略学习落后”的解释。

## 10. 当前结论

### 已确认

1. 训练初期 BACE 的 branch 数量确实较少，除 step 1 外大部分 leaves 仍是 natural roots。
2. step 1 的 18 branches 是初始化 planner 计算结果，不是手动指定。
3. BACE 从第一步开始就不等价于 GiGPO，因为 global term 的 normalization 对象不同。
4. BACE global term 按 unique terminal leaves 归一化；GiGPO macro term 按 physical occurrences 使用 `token_level_rewards`。
5. Invalid Action penalty 会进入 GiGPO macro/local reward，但不会进入 BACE unique-leaf global reward。
6. BACE local group 中 singleton 或同回报 group 会产生零 local advantage；branch suffix 很多是新 anchor，不能保证 local group 变大。
7. Branch outcome 与 origin 高度相关，因此 branches 不能简单视为额外独立 GiGPO samples。

### 尚不能直接断言

- 不能仅凭这个反事实结果断言“GiGPO-style global term 一定带来最终更高 success”；需要真实 1--10 epoch controlled run。
- 不能把 GiGPO occurrence zero 与 BACE token zero 当作完全相同的 token-level指标。
- 不能把当前旧版 legacy run 的结果当作 BatchERV Exact 主方案结果。

## 11. 后续实验顺序建议

在修改代码前，建议先做一个最小 controlled diagnostic：同一个已经保存的 BACE batch 同时计算并记录：

```text
A. 当前 BACE：unique-leaf global + current local
B. occurrence-level terminal global + current local
C. occurrence-level terminal + invalid penalty global + current local
D. 原 GiGPO calculator
```

每个版本记录：

```text
zero occurrence/token
positive/negative token count
per-task advantage std
root/origin/suffix contribution
effective token-weighted leaf count
```

然后再进行短程 controlled training：

```text
same checkpoint
same data order
same 1--3 epoch
same actor micro-batch
same validation tasks
only change advantage definition
```

在这项隔离实验完成前，不应直接用完整 150 epoch 去判断 BACE topology 或 BatchERV 是否有效。

## 12. 文件与代码证据

- BACE advantage：`verl-agent-src/recipe/bace_gigpo/advantage.py`
- GiGPO advantage：`verl-agent-src/gigpo/core_gigpo.py`
- Invalid Action penalty：`verl-agent-src/verl/trainer/ppo/ray_trainer.py`
- Episode reward placement：`verl-agent-src/agent_system/reward_manager/episode.py`
- BACE topology planner：`verl-agent-src/recipe/bace_gigpo/topology.py`
- BACE frontier branch scheduling：`verl-agent-src/recipe/bace_gigpo/frontier.py`
- BACE artifact：`/opt/dpcvol/datasets/8165423358032568398/AESC-exp/bace_artifacts/bace_alfworld_npu_8card_150epoch_baseline_fix3_mb4`
- GiGPO artifact：`/opt/dpcvol/datasets/8165423358032568398/AESC-exp/anchor_groups/gigpo_qwen2.5_1.5b_npu_8card_150epoch_tp1_mb8_s40_lowval`
