# BACE-GiGPO experiment entry points

本目录同时含历史调试脚本和当前正式方法入口。新的 2×H100 C0/tree-credit 实验只使用下表中的独立目录；不要从旧作业号、旧计划文档或通用 launcher 名称推断参数。

| 方法 | 当前独立目录 | 信用模式 | capacity correction |
| --- | --- | --- | ---: |
| C0 | `c0_optimized_2gpu/` | `current` | 1 |
| C1 | `c1_tree_credit_corr1_2gpu/` | `o1_local` | 1 |
| C2 | `c2_tree_credit_corr1_2gpu/` | `o1_tree_macro` | 1 |
| C3 | `c3_tree_credit_corr1_2gpu/` | `o1_full_tree` | 1 |
| C0.5 | `c05_tree_credit_corr1_2gpu/` | `c0_5_origin_family_local_mean` | 1 |
| C4 | `c4_tree_credit_corr1_2gpu/` | `c4_macro_strict_ancestor` | 1 |
| C7 | `c7_tree_credit_corr1_2gpu/` | `c7_flat_leaf_gigpo` | 1 |
| C8 | `c8_tree_credit_corr1_2gpu/` | `c8_macro_local_strict_ancestor` | 1 |

这里的 `capacity_correction_batch_size=1` 表示：对同一个 deficient task，每次容量评估只把一个 branch slot 转成 natural-root slot，生成后立即重新计算容量。不同 task 各自新增的一条 root 仍可被合并到同一个 packed GPU wave。

每个目录包含完整 `run_*.sh`、批量提交辅助脚本和方法 README。训练入口直接构造 Hydra 命令，不调用另一个训练 launcher。普通集群可在已经取得 2 GPU 的 allocation 中用 `bash` 执行；Slurm 用户可使用目录中的提交脚本，并按站点修改 partition、account 和 time。

曾用于历史结果的 correction=4 目录和提交器已从本目录移出，保存在：

```text
work-BACE/archived_launchers/deprecated_capacity_correction_4_20260908/
```

它们只用于审计旧实验，不再作为可提交的当前配置。历史结果本身仍应以对应 `run_metadata/<run>/<invocation>/resolved_command.sh` 为准。

修改参数后先运行 `DRY_RUN=1`，并检查生成的 `resolved_command.sh`。使用新的参数或 seed 必须使用新的 `BACE_RUN_NAME`；只有恢复同一实验时才复用 run name。
