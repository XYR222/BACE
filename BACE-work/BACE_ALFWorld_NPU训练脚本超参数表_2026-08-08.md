# BACE ALFWorld NPU 训练脚本超参数表

两份入口均使用当前 upstream 基座、`algorithm.adv_estimator=bace_gigpo`、`algorithm.bace.enabled=true` 和主方案 `strict_identity`。不包含 Invalid Action 三组消融。末尾可追加 Hydra 参数覆盖脚本默认值。

| 参数 | 单卡轻量默认值 | 8 卡默认值 | 说明 |
|---|---:|---:|---|
| `BACE_EXP_ROOT` | `/opt/dpcvol/datasets/8165423358032568398/AESC-exp` | 同左 | 所有数据、checkpoint、日志、trace 的唯一根目录 |
| `BACE_RUN_NAME` | 自动附加 UTC 时间 | 自动附加 UTC 时间 | 默认创建独立实验；设置为已有名称时可用于显式续训 |
| `BACE_RESUME_MODE` | `disable` | `disable` | 防止同名目录被意外续训；续训时显式设置为 `auto` |
| `ENGINE` | `vllm` | `vllm` | rollout engine；也兼容旧脚本的第一个位置参数写法 |
| `TRAIN_DATA_SIZE` / `VAL_DATA_SIZE` | `4 / 4` | `8 / 8` | 小规模现场验证；正式实验可增大 |
| `BACE_TOTAL_EPOCHS` | `10` | `20` | 中等长度但保持轻量；数据量等于 batch 时每 epoch 约一步 |
| `ENV_MAX_STEPS` | `12` | `12` | ALFWorld 单 episode 最大步数 |
| `env.rollout.n` | `4` | `4` | 每个 prompt 的 rollout 数，也是 BACE 候选来源 |
| `BACE_TOPOLOGY` | `dynamic` | `dynamic` | 动态拓扑，按叶预算分配 roots/branches |
| `BACE_DYNAMIC_ROOT_GENERATION` | `staged` | `staged` | 分阶段生成 root，避免一次预分配浪费 rollout |
| `BACE_TOTAL_LEAF_BUDGET` | `4` | `4` | 每 task 满足 `R + Q = B` |
| `BACE_PILOT_ROOTS` | `2` | `2` | 第一阶段 pilot roots 数 |
| `BACE_BRANCH_COUNT` | `2` | `2` | 固定每 anchor 最大 branch 数 |
| `algorithm.bace.invalid_action_mode` | `strict_identity` | `strict_identity` | 主方案；按具体 action identity 区分合法/非法动作 |
| `data.max_prompt_length` | `2048` | `2048` | prompt 截断上限 |
| `data.max_response_length` | `256` | `256` | action trajectory 响应上限，可用 `MAX_RESPONSE_LENGTH` 覆盖 |
| actor PPO mini batch | `16` | `32` | 单卡显存安全优先；8 卡提高吞吐 |
| actor micro batch / GPU | `1` | `1` | NPU 显存保护值 |
| rollout tensor parallel | `1` | `1` | 8 卡采用 data/distributed worker 并行，不切分 1.5B 模型 |
| rollout GPU memory utilization | `0.45` | `0.55` | vLLM 显存水位，OOM 时优先下调 |
| max batched tokens | `4096` | `8192` | vLLM 批处理上限 |
| `trainer.save_freq` / `test_freq` | `5 / 5` | `5 / 5` | 定期 checkpoint 和验证，便于定位中间状态 |

## 产物布局

每个 `run_name` 会创建 `data/<run_name>/text/*.parquet`、`checkpoints/<run_name>/`、`rollout_trajectories/<run_name>/bace_trace/`、`tensorboard/<run_name>/`、`logs/<run_name>.log`、`run_metadata/<run_name>/run_metadata.txt` 和 `trace_validation/<run_name>.json`。trace 包含 roots、leaves、topology、posterior、Replay、branch、token ids、loss mask、old log-prob 及环境动作合法性字段；训练失败也会保存退出码和“不完整 trace”报告。

## 启动示例

```bash
bash examples/bace_gigpo/run_alfworld_npu_1card_light_train.sh
bash examples/bace_gigpo/run_alfworld_npu_8card_train.sh
```

例如只调整长度：

```bash
ENV_MAX_STEPS=16 BACE_TOTAL_EPOCHS=30 bash examples/bace_gigpo/run_alfworld_npu_1card_light_train.sh
```
