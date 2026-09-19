# Search BACE Step-0 Anchor 与阈值离线诊断

日期：2026-09-17

## 1. 结论

1. Search 应显式允许 step 0 成为 BACE anchor。实现采用
   `algorithm.bace.allow_initial_search_anchor=true`，只对
   `task_family=search` 生效；ALFWorld 和 WebShop 的历史语义不变。
2. `batch_erv_threshold=0.005` 不应直接从 ALFWorld 沿用到 Search。
   在当前真实 Search smoke trace 中，全部 4 个 structural task 的第一条
   branch marginal ERV 都低于 `0.005`，因此总 information capacity 从 8 降为 0。
3. 若继续采用“至少保留 97% 离线 ERV value”的保守规则，当前样本支持把
   Search 的首个线上阈值设为 `0.001`；它在该 trace 中保留 100% capacity 和
   100% ERV value。`0.0025` 能保留所有 task 的第一条 branch，但总 capacity
   retention 为 87.5%、ERV value retention 为 93.35%，不满足 97% 规则。
4. `competence_threshold=0.5` 不会在 200-step 参考曲线下导致长期完全没有
   branch，但会造成约 37-step 的 warm-up，并把参考期内的计划 quota 限制为
   `Q<=1`。这不是代码错误，却会使 Search 的 BACE acquisition 偏弱。
5. 细粒度、闭环复算后，建议 Search 的首个正式对照把
   `competence_threshold` 小幅下调为 `0.45`：首次 `Q>0` 约在 step 20，200 步中
   42 步为 `Q=2`，同时仍保留 77.8% natural-root leaf budget。`0.4` 从 step 2
   起几乎持续 branching，并有 109 步为 `Q=2`，不建议作为首个主配置。
6. `batch_erv_threshold` 仍需从 `0.005` 下调；若坚持至少保留 97% 离线 ERV
   的标准取 `0.001`，若接受更强过滤则 `0.0025` 也可行。competence quota 只是
   上限，最终 branch 仍必须通过当前 batch 的 Exact-ERV capacity 检查。

## 2. 输入和范围

真实 BACE trace：

```text
/hpcwork/rwth2089/xsz96350/work-BACE/experiments/search-qwen2.5-3b-exact/
validation/bace_search_exact_smoke4g_4149084/bace_artifacts/step_00000001/roots.jsonl
```

- 8 个 Search tasks；
- 每任务 5 roots；
- 共 40 roots；
- Qwen2.5-3B 冷启动策略；
- Search horizon 为 4。

competence 代理曲线：

```text
/hpcwork/rwth2089/xsz96350/work-BACE/experiments/gigpo-search-reference/
runs/seed_0_v1/segments/3708759/trainer.log
```

- 包含 400 个真实 GiGPO training-step success rates；
- 本报告只使用前 200 步，与正式目标对齐；
- 每步按原版 global batch `256 × 5 = 1280` 个 natural episodes 更新共享
  `search` family history。

机器可读结果：

```text
/hpcwork/rwth2089/xsz96350/work-BACE/experiments/search-qwen2.5-3b-exact/
analysis/search_bace_threshold_diagnostic_20260917.json
```

复算脚本：

```text
verl-agent-src/examples/bace_gigpo/analyze_search_bace_offline.py
```

## 3. Step-0 anchor

旧实现无条件跳过所有 `event.step_index == 0`。新实现只有同时满足以下条件才纳入：

```text
allow_initial_search_anchor = true
task_family = search
format_valid = true
action_identity_kind = valid
```

因此它不会改变格式错误的处理，也不会把 terminal answer 当作 branch candidate。

旧 smoke 的 step-0 分布为：

| 指标 | 数量 |
|---|---:|
| step-0 occurrences | 40 |
| format-valid | 10 |
| valid SEARCH candidate | 5 |
| terminal ANSWER | 5 |
| format-invalid/unparsed | 30 |

35/40 的环境投影实际选择了 search，但其中大量原始 response 同时生成了 search 和
answer，按当前 GiGPO parser 属于格式错误。这些 occurrence 继续进入 PPO并接受格式
惩罚，但不进入 BACE acquisition。

在只纳入 5 个合法 step-0 Search actions 后：

| 策略 | structural anchors |
|---|---:|
| 旧规则：排除 step 0 | 3 |
| 新规则：允许合法 Search step 0 | 4 |

新增的 1 个 anchor 来自同一初始问题状态下至少两个不同的合法搜索 query。该结果证明
step-0 开关在现有真实 trace 上并非空操作。

selected-worker replay 已增加空 prefix 专项测试：reset 后直接返回原始问题 anchor，
不执行任何机械 prefix step，再执行选中的首次 query。

## 4. Batch-ERV threshold

对同一批 roots，以真实 task outcome、`local_prior_strength=2`、
`max_branches_per_anchor=2` 重建全部 Beta posterior 和 Exact Batch-ERV design：

| threshold | total capacity | capacity retention | ERV value retention | task cap≥1 | task cap≥2 |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 100% | 100% | 4 | 4 |
| 0.001 | 8 | 100% | 100% | 4 | 4 |
| 0.0025 | 7 | 87.5% | 93.35% | 4 | 3 |
| 0.005 | 0 | 0% | 0% | 0 | 0 |
| 0.01 | 0 | 0% | 0% | 0 | 0 |

8 个 marginal ERV 的范围为：

```text
0.0017703, 0.0026960, 0.0026960, 0.0026960,
0.0027483, 0.0046712, 0.0046712, 0.0046712
```

其中四个第一分支 marginal 分别为：

```text
0.0027483, 0.0046712, 0.0046712, 0.0046712
```

所以 `0.005` 并不是轻度过滤，而是连每个 task 的第一条 branch 都全部拒绝。

`0.0025` 保留了四个 task 的第一 branch，仅过滤最弱的第二 branch；如果实际 quota
始终为 1，它与 `0.001` 在该样本上的首分支可用性相同。但为了延续此前“至少 97%
ERV value”的保守标准，主推荐仍为 `0.001`。

## 5. Competence threshold

共享 family 的状态转移按正式配置复算：

```text
initial = Beta(0.2, 1.8)
forgetting = 0.8
transfer_fraction = 0.1
strength range = [2, 8]
B = 5
R_min = 2
flexible slots = 3
Q = round(3 × P(p > competence_threshold))
```

使用历史 GiGPO 前 200 步 success rate 作为 natural-root success rate 代理。
复算是闭环的：每一步先由 lagged history 得到 `Q`，随后只用
`256 × (5-Q)` 条预计 natural roots 更新 history，branch outcomes 不进入 history。

| competence threshold | 首次 Q>0 | Q=0 | Q=1 | Q=2 | Q=3 | 平均 Q | branch slots | root slots | root 占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | 2 | 1 | 31 | 162 | 6 | 1.865 | 373 | 627 | 62.7% |
| 0.40 | 2 | 1 | 90 | 109 | 0 | 1.540 | 308 | 692 | 69.2% |
| 0.42 | 2 | 2 | 111 | 87 | 0 | 1.425 | 285 | 715 | 71.5% |
| 0.43 | 2 | 11 | 119 | 70 | 0 | 1.295 | 259 | 741 | 74.1% |
| 0.44 | 2 | 17 | 132 | 51 | 0 | 1.170 | 234 | 766 | 76.6% |
| **0.45** | **20** | **20** | **138** | **42** | **0** | **1.110** | **222** | **778** | **77.8%** |
| 0.46 | 26 | 25 | 163 | 12 | 0 | 0.935 | 187 | 813 | 81.3% |
| 0.47 | 33 | 32 | 164 | 4 | 0 | 0.860 | 172 | 828 | 82.8% |
| 0.48 | 34 | 33 | 167 | 0 | 0 | 0.835 | 167 | 833 | 83.3% |
| 0.49 | 36 | 35 | 165 | 0 | 0 | 0.825 | 165 | 835 | 83.5% |
| 0.50 | 38 | 37 | 163 | 0 | 0 | 0.815 | 163 | 837 | 83.7% |
| 0.55 | 64 | 67 | 133 | 0 | 0 | 0.665 | 133 | 867 | 86.7% |

这里的 slot 是 200 steps × 每 task 5 个 leaves，共 1000 个 task-level slots；正式
batch 每步有 256 个 tasks，所以 `0.45` 对应上限约 56,832 个 branch leaves 和
199,168 个 natural-root leaves。它比 `0.5` 多 15,104 个 branch leaves，即总 leaf
budget 增加 5.9 个百分点用于 branch。

`threshold=0.5` 的分段结果：

| steps | 平均 success rate | 平均计划 Q | Q=0 比例 |
|---|---:|---:|---:|
| 1–50 | 0.3161 | 0.26 | 74% |
| 51–100 | 0.3991 | 1.00 | 0% |
| 101–150 | 0.4385 | 1.00 | 0% |
| 151–200 | 0.4547 | 1.00 | 0% |

这说明 `0.5` 是保守 warm-up，而不是永久关闭 acquisition。但是 Search 的参考
success curve 在前 200 步均值逐步接近约 0.45；在 strength 上限 8 下，`0.5`
因此只能产生 `Q=0/1`。降低到 `0.45` 会把首次 branch 从约 step 38 提前到约
step 20，并在后期允许有限的 `Q=2`；最低仍有 3 条 natural roots/task，且 Exact-ERV
capacity 仍可把缺乏信息价值的 branch slot 转回 root。降低到 `0.4` 则基本从 step 2
起立即 branch，明显削弱冷启动保护。

`0.45` 是 Search 专用的 benchmark calibration，不表示原先 `0.5` 的数学实现有错，
也不应无条件回写到 ALFWorld/WebShop。若研究目标要求完全冻结跨环境的 competence
定义，应继续用 `0.5` 并把它视为保守主线；若目标是让 Search 在 200-step 预算内
实际覆盖 `Q=2` refinement regime，则 `0.45` 是当前证据支持的折中值。

## 6. 证据边界

- ERV 结果来自真实 BACE roots、actions、outcomes 和当前 Exact engine，但只有 8 个
  冷启动 tasks；它足以否定“无条件沿用 0.005”，不足以证明 0.001 对所有训练阶段最优。
- competence 结果不是已经完成的 BACE 训练。它使用 GiGPO success curve 模拟共享
  family controller，并已根据每步计划 `Q` 缩减 history 的 natural-root evidence
  数量；BACE policy 的真实成功率、离散成功数和 capacity correction 仍可能偏离。
- 历史 GiGPO 后期 `valid_action_ratio` 接近 1，未来 step-0 structural anchor 数量预计
  高于冷启动 smoke，但缺少对应 BACE root trace，不能伪造其 ERV posterior。
- 正式上线后应在前 50 steps 持续记录 readiness、planned/final Q、capacity correction、
  marginal ERV 和 step-0 anchor share；这属于诊断记录，不应再设置会自动停止训练的门禁。

## 7. 当前执行状态

- 旧定义的正式作业 `4186008` 已在运行前取消，没有训练 step 或 checkpoint。
- C0 credit 保持 `credit_mode=current`，本次没有改为 C4/C8。
- 正式脚本已经显式启用 Search step-0 anchor，但尚未修改
  `competence_threshold=0.5` 和 `batch_erv_threshold=0.005`，也尚未重新提交正式作业。
