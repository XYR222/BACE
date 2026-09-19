# Search 环境 BACE 接入与 smoke 方案

## 目标与范围

本阶段只验证 Search/HotpotQA 的 BACE 数据链、action identity、Exact
Batch-ERV、selected-worker replay 和一次 PPO 更新，不提交正式训练，不保存
checkpoint。训练参数以原版 Search GiGPO 为基线，BACE 只增加方法本身所需的
动态 root/branch 调度与 trace。

## Action 语义

| 模型输出 | invalid penalty | PPO 数据 | BACE 候选 |
|---|---:|---:|---:|
| 合法 `<search>q</search>` | 否 | 是 | 是 |
| 合法 `<answer>a</answer>` | 否 | 是 | 否（terminal） |
| 空标签、混合标签、多标签或无完整标签 | 是 | 是 | 否 |

合法 search 的 identity 是精确执行字符串
`valid::<search>q</search>`。不做小写化、标点清理或语义 query 合并，避免把
两个实际不同的检索调用写进同一个 posterior。思维链不属于环境动作，不进入
identity。

## Anchor 与 replay

1. Search anchor 沿用 GiGPO 的 `SequenceMatcher >= 0.9` observation 聚类；
   ALFWorld 和 WebShop 的 exact anchor 默认不受影响。
2. Exact Batch-ERV 可在 observation cluster 上统计候选，但请求保存被选中
   occurrence 的实际 observation、query、task reset key 和 transition。
3. reset key 使用版本化 JSON，包含 question、ground truth 和 data source；
   parquet 中的 NumPy array/scalar 会无损规范化为 JSON list/scalar。
4. selected worker 先按 reset key 重建同一问题，再机械执行 occurrence 之前的
   精确 action prefix，核验 anchor 后执行 origin action。
5. replay 前缀不进入梯度；复制 origin 与新生成 suffix 的 PPO 语义保持现有
   BACE 规则。
6. 固定 retriever 的成功响应按 endpoint、top-k 和精确 query 缓存，使同一进程
   中的自然 rollout 与 replay 获得相同文档结果；失败响应不缓存。

## 测试和 smoke

- `tests/bace_gigpo/test_search_replay.py`：覆盖 projection、search/answer/invalid
  分类、parquet reset key、0.9 anchor 聚类、具体 occurrence 分配、selected-worker
  replay 和检索缓存。
- 完整回归：`python -m pytest -q tests/bace_gigpo`。
- 真实环境探针：两个 Search worker 对同一问题执行自然 prefix/origin 和
  reset/replay/origin，要求 anchor、action identity、post observation、reward 和
  done 全部一致。
- GPU smoke：`slurm_search_h100_4gpu_smoke.sbatch`。4×H100 加载并分片完整
  60 GiB FAISS index，使用 Qwen2.5-3B-Instruct 跑 1 step；要求
  `global_step=1`、退出码 0、trace `ok=true`、Exact/packed 标志为 1，并通过
  真实 replay 探针。

Smoke 结果位于：

`/hpcwork/xsz96350/fu_project/work-BACE/experiments/search-qwen2.5-3b-exact/validation/`

物理存储位于 `rwth2089` 的 3 TiB 项目空间；当前 BACE 路径使用符号链接访问。
Smoke 设置 `save_freq=-1`，不会生成 checkpoint，且脚本中没有 `sbatch` 或任何
正式作业提交逻辑。
