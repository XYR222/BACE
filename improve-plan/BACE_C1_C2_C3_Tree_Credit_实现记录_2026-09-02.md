# BACE C1/C2/C3 Tree Credit 实现记录（2026-09-02）

## 1. 实现口径

本次实现保留 C0（`tree_credit_mode=current`）作为默认逻辑，不删除或替换旧代码。C1/C2/C3 共享同一套 rollout、Exact Batch-ERV、packed root、selected-worker replay、PPO loss 和 checkpoint 逻辑；唯一的算法变量是 tree credit：

| 变体 | `tree_credit_mode` | local | macro | copied origin |
|---|---|---|---|---|
| C1 | `o1_local` | direct continuation 的 G 均值后重新归一化 | 每个保留 edge 的 frozen C0 stable macro | evidence only |
| C2 | `o1_tree_macro` | 与 C1 完全相同 | descendant frozen macro 均值 | evidence only |
| C3 | `o1_full_tree` | descendant return-to-go 的 G 均值后重新归一化 | 与 C2 完全相同 | evidence only |

三个新变体均强制使用 `macro_normalization_mode=stable_occurrence`。`strict_leaf_uniform` 被 fail-closed 拒绝，防止一次实验同时改变 tree credit 和 macro normalizer。

## 2. 数据与树索引

collector 在完整 physical batch 合并完成后写入两项精确 sidecar：

- `tree_origin_occurrence_id`：branch 指向被复制的 concrete natural occurrence；
- `tree_parent_root_id`：natural edge 或 branch 所属的原始 root。

`TreeCreditIndex` 只使用 concrete occurrence identity、root identity 和 step index 建树。因此：

- 相同 `(observation, action)` 出现在不同 root 或同一 root 的循环位置时不会合并；
- C1 的 `Direct(e)` 只包含从 `e` 直接产生的 branches；
- C2/C3 的 `Desc(e)` 只包含同一 root 上、branch origin 不早于 `e` 的 leaves；
- earlier branch 不会错误进入 later edge；
- origin 后立即终止、没有 suffix 的 branch 仍保留为证据；
- fresh suffix 始终是独立 trainable edge。

训练前用于并行整除而复制的 `_adjust_batch_padding` 只参与原 C0 physical macro snapshot，不会被误识别为新 edge。去重后若 batch 不能按 GPU world size 均分，只添加 loss mask 为 0 的 `ppo_padding`，它不产生梯度，也不写入 trainable-occurrence trace。

## 3. Frozen macro 的实现细节

C1/C2/C3 都先在完整 C0 physical records 上直接调用 production `episode_norm_reward`，没有重写 mean/std 或 zero-variance 规则。

当前训练还会在 token reward 上加入 occurrence-level invalid-action penalty，因此实际 production macro 在同一 leaf 内可能存在轻微或显著的 row 差异。为保持严格 controlled comparison：

- C1 对每个保留 edge 使用它自己在 frozen C0 snapshot 中的 macro 值；
- fresh suffix 同样保持自己的 frozen C0 macro；
- C2/C3 不直接把 copied-origin macro 当作祖先 edge 的 macro；而是采用
  `M(target) + M(branch_origin) - M(natural_origin)`。branch origin 与其 natural
  origin 是同一 state、同一 action 的 deterministic replay，故两者共享的
  invalid-action 局部惩罚会在差值中抵消；目标 edge 自己的惩罚仅保留一次。
- local G 采用同一语义：direct 为 `G(target) + G(branch)-G(origin)`；C3 对
  earlier ancestor 使用 `G(target)+gamma^(t_branch-t_target)*(G(branch)-G(origin))`。
  因而 invalid action 是当前错误 action 的局部 GiGPO 项，不会被当作 trajectory
  reward shaping 回传给祖先。
- 不使用“root 第一行”替代整条 root，否则会额外改变 invalid penalty 语义。

## 4. C3 return 重建

对 natural edge `e@t` 和 origin 位于 `t_b` 的 descendant branch：

```text
G(e, branch) = G(e) + gamma^(t_b-t) * [G(branch_origin) - G(natural_origin)]
```

当 `e` 正好是 natural origin 时，上式与原始 direct continuation 等价；当 `e` 是更早 ancestor 时，它只回传 branch 相对 origin 的 continuation 变化。由于 branch origin 和 natural origin 复制同一动作，二者共有的 invalid-action 局部项在方括号中抵消。随后把 original root continuation 与所有 descendant branch continuation 等权平均。fresh suffix 使用自身原始 G，不做 branch-of-branch 推断。

## 5. Trace 与失败保护

每个 trainable unique edge 记录：edge/root/step、direct/descendant leaf IDs 和计数、original/direct/descendant G、base/descendant macro、current/C1/C3 local、current/C1/C2/C3 final advantage，以及两个模式字段。`tree_credit_copied_origins_removed` 与 `tree_credit_adjustment_padding_removed` 分开计量，避免把真实 copied-origin evidence 与 GPU 整除 padding 混为一谈。

每条 branch 另写 `tree_credit_branches.jsonl`，记录 origin、parent root、origin step、leaf、branch-origin G 和 terminal reward。validator 会拒绝：

- C1/C2/C3 中 copied origin 进入 trainable support；
- 重复 concrete edge；
- 缺失 tree-credit 字段；
- 选择的 advantage component 与 mode 不一致；
- branch evidence 与实际成功执行的 branch 集合不一致；
- 非 `stable_occurrence` 的 macro normalization。

## 6. 测试与运行入口

- 单测：`tests/bace_gigpo/test_tree_credit.py`
- trainer 集成测试：`tests/bace_gigpo/test_trainer_advantage_integration.py`
- 单卡三模式烟测：`examples/bace_gigpo/slurm_tree_credit_c123_1gpu_smoke.sbatch`
- 烟测验收：`examples/bace_gigpo/validate_tree_credit_smoke.py`
- 2×H100 正式入口：`examples/bace_gigpo/run_alfworld_h100_2gpu_reference_aligned.sbatch`
- C1/C2/C3 seed0 提交器：`examples/bace_gigpo/submit_tree_credit_c123_2gpu_seed0.sh`

正式三次训练均使用相同的 optimized S3 调度、`capacity_correction_batch_size=4`、GiGPO 对齐参数、seed 0、150 steps、每 5 step 保存，并只轮转保留最近两份 checkpoint。提交器为每个变体同时设置一个 `afterany` 恢复作业；主作业完成后恢复作业会验证 step 150 并直接退出，主作业超时或节点故障时则从最新 checkpoint 续跑。
