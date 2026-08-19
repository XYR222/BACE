# BACE H100 可迁移工程包

本目录是 `/home/naie/work/work-BACE` 在 2026-08-14 的代码迁移副本，并新增了独立的 CUDA/H100 运行层。原有 BACE、BatchERV Exact、legacy、NPU 脚本和历史文档均保留，没有删除或替换原逻辑。本次 A100 过渡包有意不携带大模型和完整 ALFWorld 数据，详见 [README_A100_TRANSITION.md](README_A100_TRANSITION.md)。

## 已随包携带的内容

- `verl-agent-src/`：完整 Git 工作树、`.git`、所有已修改和未跟踪的 BACE 实现、测试及新旧启动脚本。
- `BACE-work/`、`BACE-work-2/`、`ICLR2027/`、`情况说明.md`：全部方案、排查和实验文档。
- `assets/alfworld/`：可选的 ALFWorld PDDL、TextWorld games 和 detector 数据，需单独准备。
- `assets/models/Qwen2.5-1.5B-Instruct/`：可选的当前实验模型，需单独准备。
- `assets/training_data/alfworld_text_16_64/`：16 条训练占位数据和 64 条验证占位数据，不依赖目标机 Hugging Face cache。
- `deploy/gpu/`：H100 环境定义、安装、预检、清单和校验工具。
- `verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_gpu*.sh`：1/2/4/8 GPU 通用启动器、smoke 和完整训练入口。

没有携带完整模型、ALFWorld 数据、当前 Ascend conda 环境、HCCL/torch_npu、编译缓存、旧 Ray session 和训练输出。目标机必须预先具备可用的 NVIDIA Driver；CUDA 用户态依赖由安装脚本创建，大资产由 `deploy/gpu/prepare_runtime_assets.sh` 按需准备。

## 目标机首次安装

要求：Linux x86_64、NVIDIA H100、`nvidia-smi` 正常、Miniconda/Miniforge、能够访问 Python package index。若目标机离线，需要提前制作与其 Linux/CUDA/Python ABI 匹配的 wheelhouse，不能使用本机 NPU wheel cache。

```bash
cd /path/to/work-BACE-2
bash deploy/gpu/verify_bundle.sh
bash deploy/gpu/install_gpu_env.sh
conda activate bace-gpu
bash deploy/gpu/check_gpu_environment.sh
```

环境基线来自当前 upstream README：Python 3.12、vLLM 0.11.0、FlashAttention 2.7.4.post1、PyTorch 2.8 系列。不要混用仓库内旧 Dockerfile 的 vLLM 0.6/0.8 依赖。

## 第一次真实运行

假设目标机有 8 张可用 H100：

```bash
cd /path/to/work-BACE-2/verl-agent-src
conda activate bace-gpu
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPU_COUNT=8 \
  bash examples/gigpo_trainer/run_bace_alfworld_gpu_smoke.sh
```

先检查 smoke 输出目录中的 `run_metadata`、`bace_artifacts`、`trace_validation` 和日志。smoke 正常后运行完整训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPU_COUNT=8 \
  BACE_EXP_ROOT=/data/experiments/BACE \
  bash examples/gigpo_trainer/run_bace_alfworld_gpu_full.sh
```

卡数不是 8 时只需要改 `CUDA_VISIBLE_DEVICES` 和 `GPU_COUNT`。例如 4 卡：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 GPU_COUNT=4 \
  bash examples/gigpo_trainer/run_bace_alfworld_gpu_smoke.sh
```

## 当前 GPU 主方案

GPU 迁移没有改变算法语义，默认配置为：

```text
variant=batch_erv_exact
topology=dynamic
dynamic_root_generation=staged
staged_root_batching=packed
acquisition=batch_erv_exact
total_leaf_budget=8
min_natural_roots=2
batch_erv_threshold=0.01
invalid_action_mode=strict_identity
local_credit_mode=occurrence
```

`strict_identity` 保留当前主方案：自然执行中有效 action 和可识别的 invalid/no-op action 都参与相应统计与选择，不改成 Exact valid-only。

## 常用参数

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `GPU_COUNT` | 自动检测 | 当前作业使用的 GPU 数量 |
| `CUDA_VISIBLE_DEVICES` | `0..GPU_COUNT-1` | 物理 GPU 映射 |
| `MODEL_PATH` | 包内模型 | 可替换成目标机模型目录 |
| `ALFWORLD_DATA` | 包内数据 | ALFWorld 数据根目录 |
| `BACE_EXP_ROOT` | `work-BACE-2/experiments` | checkpoint、trace、日志总目录 |
| `BACE_TOTAL_EPOCHS` | full 为 150 | 训练 epoch |
| `ROLLOUT_TP` | 1 | 1.5B 模型不需要 TP；必须整除卡数 |
| `ACTOR_MICRO_BATCH` | full 为 8 | 每卡 actor micro batch |
| `LOGPROB_MICRO_BATCH` | full 为 8 | rollout/ref log-prob micro batch |
| `GPU_MEMORY_UTILIZATION` | full 为 0.60 | vLLM 显存占用比例 |
| `MAX_NUM_BATCHED_TOKENS` | 16384 | vLLM 单次调度 token 上限 |
| `MAX_NUM_SEQS` | 128 | vLLM 单次 sequence 上限 |
| `SAVE_FREQ` | 30 | checkpoint 频率 |
| `TEST_FREQ` | 10 | 与当前 150 epoch 配置一致的验证频率 |

底层脚本支持追加 Hydra override。例如：

```bash
bash examples/gigpo_trainer/run_bace_alfworld_gpu_smoke.sh \
  algorithm.bace.batch_erv_threshold=0.02
```

## 运行数据保存

每次运行按唯一 `BACE_RUN_NAME` 保存：checkpoint、完整 rollout、BACE artifact、anchor dump、TensorBoard、标准输出、最终 trace 验证报告，以及以下排查信息：

- 完整环境变量、pip freeze、Git HEAD/status/diff；
- 最终解析后的训练命令和用户 Hydra overrides；
- `nvidia-smi` 运行前后快照、GPU 拓扑；
- 周期性 GPU 利用率、显存、功耗、温度采样；
- Batch plan、root/branch、Replay identity、token arrays 和最终 leaf metadata。

## 边界说明

迁移包能固定项目、算法、模型、数据和 Python 依赖，但不能打包目标 H100 的内核驱动。不同服务器的 Driver、GPU 数量、H100 80GB/94GB、NCCL 网络拓扑仍需由 `check_gpu_environment.sh` 在现场确认。首次 smoke 前不要直接提高显存利用率或 micro batch。
