# BACE A100 过渡设备迁移说明

## 1. 为什么先用 A100

A100 与 H100 都是 NVIDIA CUDA 平台，可以先验证绝大多数高风险内容：PyTorch CUDA、vLLM、Ray worker、NCCL、ALFWorld 环境、BACE 序贯 rollout、Replay identity、BatchERV Exact、artifact 保存和 checkpoint。H100 阶段主要再做显存、吞吐和并行度调优。

因此本包只携带代码、文档、测试、安装规格和小型 parquet fixture；约 2.3 GB 的 ALFWorld 数据、约 2.9 GB 的模型以及训练输出按需准备。

## 2. 包内内容

- `verl-agent-src/`：完整工作树和 `.git`，包括所有未提交 BACE 修改；
- `BACE-work/`、`BACE-work-2/`、`ICLR2027/`：方案、审计、实验和论文文档；
- 原 NPU/legacy/Exact 脚本：保留作对照，不作为 A100 入口；
- `verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_gpu.sh`：通用 CUDA 启动器；
- `run_bace_alfworld_a100_smoke.sh`：单张 A100 的保守 1 epoch smoke；
- `deploy/gpu/`：环境安装、CUDA 预检、运行资产准备、迁移校验；
- `assets/training_data/alfworld_text_16_64/`：小型 parquet fixture；
- `assets/README_RUNTIME_ASSETS.md`：模型和 ALFWorld 数据准备说明。

不包含：模型权重、完整 ALFWorld 数据、旧训练输出、Ascend conda 环境、NPU 编译缓存和 A100/H100 专属 CUDA wheel。

## 3. 拿到设备后的操作

### 3.1 解包和结构检查

```bash
cd /path/to/work-BACE-3
bash deploy/gpu/verify_bundle.sh
```

此时脚本会报告模型和 ALFWorld 为 optional missing，这是预期状态；核心代码、文档和小型 parquet 必须通过检查。

### 3.2 安装 CUDA Python 环境

先确认主机驱动：

```bash
nvidia-smi
```

然后安装环境：

```bash
bash deploy/gpu/install_gpu_env.sh
conda activate bace-gpu
bash deploy/gpu/check_gpu_environment.sh
```

预检在大资产缺失时会失败，这表示尚未进入真实训练阶段，不是 BACE 代码错误。

### 3.3 准备模型和 ALFWorld

从已有资产机器导入：

```bash
SOURCE_ASSET_ROOT=/path/to/assets \
  bash deploy/gpu/prepare_runtime_assets.sh
```

或在 A100 上联网下载：

```bash
DOWNLOAD_MODEL=1 DOWNLOAD_ALFWORLD=1 \
  bash deploy/gpu/prepare_runtime_assets.sh
```

下载结束后重新运行：

```bash
bash deploy/gpu/check_gpu_environment.sh
```

### 3.4 A100 单卡 smoke

```bash
CUDA_VISIBLE_DEVICES=0 GPU_COUNT=1 \
  BACE_EXP_ROOT=/data/experiments/BACE-A100 \
  bash verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_a100_smoke.sh
```

该脚本使用保守配置：1 epoch、`max_steps=12`、actor/logprob micro batch 2、vLLM 显存比例 0.50、8192 batched tokens。它验证正确性，不代表 A100 的最终性能配置。

### 3.5 重点检查结果

在实验目录检查：

```text
run_metadata/
  preflight.json
  resolved_command.sh
  pip_freeze.txt
  nvidia_smi_before.txt
  nvidia_smi_after.txt
  gpu_samples.csv
bace_artifacts/
trace_validation/
rollout_trajectories/
checkpoints/
```

必须确认：`trainer.device=cuda`、GPU 数量为 1、`R+Q=B=8`、Replay action identity 正常、artifact summary 完整、trace validation 通过、没有 silent dropped leaves。

### 3.6 A100 后续短跑

smoke 通过后，建议先做 3～5 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 GPU_COUNT=1 \
  BACE_TOTAL_EPOCHS=3 ENV_MAX_STEPS=40 \
  ACTOR_MICRO_BATCH=2 LOGPROB_MICRO_BATCH=2 \
  BACE_EXP_ROOT=/data/experiments/BACE-A100 \
  bash verl-agent-src/examples/gigpo_trainer/run_bace_alfworld_gpu.sh
```

只有在这一步稳定后，才提高 A100 micro batch 或运行完整 50/150 epoch。A100 的单卡结果主要用于功能和数值正确性；H100 上还需要重新测显存、吞吐、NCCL 和 Tensor Parallel。

## 4. 迁移到 H100 时保留什么

可以直接复用：源码、BACE 文档、模型、ALFWorld 数据、parquet、实验 artifact schema 和 checkpoint（需确认 PyTorch/vLLM 兼容）。

需要重新安装或重新确认：NVIDIA Driver、CUDA/PyTorch/vLLM/FlashAttention、GPU 数量、`CUDA_VISIBLE_DEVICES`、NCCL 拓扑和显存参数。

H100 机器上使用 `run_bace_alfworld_gpu_smoke.sh` 或 `run_bace_alfworld_gpu_full.sh`，不要复制 A100 的硬编码设备配置。

## 5. 完整性与版本

迁移包的 Git HEAD、依赖锁和 SHA256 清单均随包保存。重新下载模型或 ALFWorld 后，应额外记录来源、版本和 SHA256；不要修改核心代码后复用旧的整包 checksum。
