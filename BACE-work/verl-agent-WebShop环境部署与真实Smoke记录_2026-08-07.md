# verl-agent WebShop 专用环境部署与真实 Smoke 记录

日期：2026-08-07  
仓库：`/home/naie/work/work-BACE/verl-agent-src`  
架构：`aarch64`

## 1. 专用环境

按照 `agent_system/environments/README.md` 的 WebShop 安装规范创建独立 Conda 环境：

```text
环境名：verl-agent-webshop
路径：/home/naie/Asend/miniconda3/envs/verl-agent-webshop
Python：3.10.20
Java：OpenJDK 11.0.29
```

主要版本：`gym==0.24.0`、`torch==2.6.0+cpu`、`pyserini==0.17.0`、`spacy==3.7.2`、`numpy==1.26.4`、`faiss-cpu==1.14.3`、`ray==2.50.0`、`verl==0.3.1.dev0`。由于 spaCy 与 Gradio 的 typer 约束冲突，显式固定 `typer==0.9.4`。为使仓库顶层环境导入可用，另安装 `gymnasium==1.0.0` 与 ARM 兼容的 `torchvision==0.21.0`。

## 2. 数据与镜像

README 的 Google Drive 下载链接在当前环境返回 `Cannot retrieve the public link of the file`。使用已核验 OID/SHA 与官方文件一致的 Hugging Face 镜像：

`https://huggingface.co/datasets/YWZBrandon/webshop-data/resolve/main/<filename>?download=true`

已完成并校验可读：

```text
data/items_human_ins.json       5,137,548 bytes
data/items_ins_v2_1000.json       147,099 bytes
data/items_shuffle_1000.json    4,467,013 bytes
```

`items_ins_v2.json` 的一次断点续传出现服务端分段错位，JSON 完整性检查已发现并隔离为 `.partial`；重新下载因网络不稳定暂存为 `.download.partial`。`en_core_web_lg==3.7.1` 已完成安装。当前真实 smoke 使用 1000 商品小模式，不依赖全量文件。

## 3. 搜索索引

执行 `convert_product_file_format.py`，再用 Java 11/Pyserini 构建：

```text
search_engine/indexes_1k  1000 docs, errors=0
search_engine/indexes     1000 docs, errors=0
```

同时保留 `resources`、`resources_1k` 及标准索引目录，未删除原逻辑或原始数据。

## 4. 最小兼容修复

上游 `web_agent_site/app.py` 调用新版 `load_products()` 时遗漏必需的 `attrpath`，导致 Flask 首次请求 500。已在网页入口传入已有的 `DEFAULT_ATTR_PATH`；这是参数补齐，不改变产品/目标加载逻辑。

## 5. 真实 Smoke 证据

在 `verl-agent-webshop` 中直接运行 `WebAgentTextEnv-v0`：

- `reset(session=0)` 成功，加载 1000 商品、6910 goals；
- `search[wireless headphones]` 成功，返回结果页和合法 clickables，reward=0、done=False。

在仓库包装器 `WebshopWorker` 中运行：

- `reset(0)` 成功并返回 `available_actions/session_idx/won`；
- `step('search[wireless headphones]')` 成功，`task_score=0.0`；
- `replay(0, ['search[wireless headphones]'])` 成功，重放 observation 与 step 一致。

Flask 服务已启动：

```text
http://127.0.0.1:3000/ABC
```

HTTP 返回 200，并输出包含 WebShop instruction 的 HTML 页面；服务进程保持运行于当前终端会话。

## 6. 2026-08-08 真实一致性验证

固定使用最新版显式协议：`use_small=true`、`human_goals=false`、验证 goal `0:500`、训练 goal `500:len(goals)`。验证脚本为：

```text
examples/bace_gigpo/validate_webshop_real_env.py
```

脚本对 session 500、501、502 保存原始 observation、available-action 完整列表及顺序、模型原始输出、投影 action、reward、done、数据 SHA256 和 HTTP 服务状态。下列四项对三个 session 均为 `true`：

1. `session_reset_identity`
2. `available_action_restoration`
3. `raw_action_identity`
4. `post_transition_consistency`

CPU 专用环境证据：

```text
/home/naie/work/work-BACE/BACE-work/experiment_traces/webshop_real_env_2026-08-08.json
```

NPU 训练环境独立执行证据：

```text
/home/naie/work/work-BACE/BACE-work/experiment_traces/webshop_real_env_npu_python_2026-08-08.json
```

两份报告的顶层 `passed=true`。CPU/fake-env 回归在加入 staged subset 用例后为 `18 passed`，JUnit 证据保存在：

```text
/home/naie/work/work-BACE/BACE-work/experiment_traces/webshop_cpu_fake_env_2026-08-08.xml
```

## 7. NPU 环境补齐

沿用已验证的单卡 NPU 环境，避免用 WebShop 全量 requirements 覆盖 Torch/NPU 配对：

```text
/opt/dpcvol/datasets/8165423358032568398/verl-agent-alfworld
Python 3.11
torch 2.8.0+cpu（torch_npu 注入 NPU backend）
torch_npu 2.8.0.post2
NPU available=true, device_count=1
```

新增 WebShop 最小运行依赖：`gym==0.24.0`、`pyserini==0.17.0`、`beautifulsoup4==4.15.0`、`scikit-learn==1.9.0`、`selenium==4.2.0`、`cleantext==1.1.4`、`rank_bm25==0.2.2`、`thefuzz==0.19.0`、`faiss-cpu==1.14.1`、`en_core_web_sm==3.8.0`。NumPy 在 Conda 安装 Faiss 后被立即恢复并固定为 `1.26.4`，Torch 和 torch_npu 未被替换。

Pyserini 的 Anserini class version 为 55，系统 Java 8 无法加载。因此在此 Conda 环境安装并显式使用 OpenJDK `11.0.27`，`JVM_PATH=$CONDA_PREFIX/lib/server/libjvm.so`。

## 8. 单卡 NPU Smoke

新增可复用 launcher：

```text
examples/bace_gigpo/run_webshop_npu_1card_smoke.sh
```

默认参数为 1 个训练 task、2 个验证 task、1 张 NPU、1 epoch、最多 2 个环境 step；BACE 使用 `dynamic + staged`、总 leaf budget 3、pilot roots 2、branch count 1、`strict_identity`，GiGPO 使用 `mean_norm`。launcher 在训练前验证 HTTP 服务与三组真实 session，并保存 `pip freeze`、Java 版本、训练前后 `npu-smi`、完整 console log、rollout JSONL 和 BACE trace。

有效运行名：

```text
bace_webshop_npu_1card_smoke_20260808_v3
```

结果为训练退出码 0、完成 1 个 NPU optimizer step。主要观测值：生成 3 条 natural roots、丢弃 0 条、6 个 natural occurrences、峰值 NPU allocated memory 约 49.046 GiB、单步耗时约 49.479 秒。该随机小样本未形成 effective anchor，因此实际 branch/replay 请求为 0；这不伪装成分支覆盖，真实 replay identity 由第 6 节的独立 validator 覆盖。

关键证据：

```text
日志：/opt/dpcvol/datasets/8165423358032568398/AESC-exp/logs/bace_webshop_npu_1card_smoke_20260808_v3.log
元数据：/opt/dpcvol/datasets/8165423358032568398/AESC-exp/run_metadata/bace_webshop_npu_1card_smoke_20260808_v3/
rollout：/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/bace_webshop_npu_1card_smoke_20260808_v3/1.jsonl
BACE trace：/opt/dpcvol/datasets/8165423358032568398/AESC-exp/rollout_trajectories/bace_webshop_npu_1card_smoke_20260808_v3/bace_trace/
trace 校验：/opt/dpcvol/datasets/8165423358032568398/AESC-exp/trace_validation/bace_webshop_npu_1card_smoke_20260808_v3.json
identity：/opt/dpcvol/datasets/8165423358032568398/AESC-exp/trace_validation/bace_webshop_npu_1card_smoke_20260808_v3_real_env_identity.json
```

trace validator 返回 `ok=true`，记录数为 roots 3、leaves 6、topology 1、trainable occurrences 6，无 errors 或 warnings。

## 9. Smoke 中发现并修复的问题

1. v1 在 Conda OpenJDK activate hook 因 `set -u` 读取未定义 `JAVA_LD_LIBRARY_PATH` 而停止；launcher 现在先设置空默认值。
2. v1 的 vLLM `max_num_batched_tokens=4096` 小于自动得到的 `max_model_len=4224`；现固定为 8192。
3. v2 首个 staged wave 发现 WebShop manager 忽略 `_bace_worker_indices`，导致 batch 1 对 observation 3。`WebshopEnvironmentManager.reset()` 现仅在 staged 参数存在时调用已有 `reset_subset()`，原来的普通全量 reset 路径未删除或改变。对应 fake-env 测试验证 worker 索引、session key 和 observation 数量。

v1、v2 为失败诊断目录，不能作为 NPU 通过证据；v3 是有效通过目录。

## 10. 当前非阻塞项

全量商品 `items_shuffle.json`（约 5.48 GB）、全量属性文件和全量 Lucene 索引仍未完成。当前固定实验协议明确使用 1000 商品小模式，因此它们不是本次训练与验证的依赖，也不能将未来全量结果与当前 1k 结果混合比较。

## 11. 2026-08-08 WebShop NPU 独立环境

为避免 WebShop 训练脚本继续引用名为 `verl-agent-alfworld` 的环境，已创建独立前缀：

```text
/opt/dpcvol/datasets/8165423358032568398/verl-agent-webshop
```

现有 `/home/naie/Asend/miniconda3/envs/verl-agent-webshop` 是 Python 3.10、Torch 2.6 CPU 环境，直接复制无法执行 NPU 训练，也不能直接混用 Python 3.11 的 torch_npu 二进制包。因此新环境从已经补齐 WebShop 全部运行依赖、并完成真实 WebShop/NPU smoke 的 NPU 环境克隆，再作为独立 WebShop NPU 环境验证。关键版本为 Python 3.11.15、NumPy 1.26.4、Torch 2.8.0、torch_npu 2.8.0.post2、Gym 0.24.0、Pyserini 0.17.0、spaCy 3.8.14、Faiss 1.14.1、Ray 2.46.0 和 OpenJDK 11.0.27。

克隆时 Conda 根据其包元数据短暂恢复了 NumPy 2.4.6；已在目标环境重新固定为 1.26.4，以保持旧 Gym、spaCy、Faiss 和 NPU 训练栈的已验证兼容组合。

使用该独立环境执行 session 500、501、502 的真实 identity validator，四项检查全部通过，报告为：

```text
/home/naie/work/work-BACE/BACE-work/experiment_traces/webshop_real_env_dedicated_npu_conda_2026-08-08.json
```
