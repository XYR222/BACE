# BACE-GiGPO 代码实现相关 GitHub 仓库链接

> 更新时间：2026-08-07  
> 用途：配合《BACE_GiGPO_代码实现架构与Replay工程规范》使用。  
> 原则：`verl-agent` 作为唯一代码基座；其他仓库主要用于按模块参考或最小移植。

---

## 1. 核心代码基座：verl-agent / GiGPO

### verl-agent

- GitHub：<https://github.com/langfengQ/verl-agent>
- 作用：BACE-GiGPO 的主代码基座。
- 主要复用：
  - GiGPO rollout 与 advantage；
  - ALFWorld / WebShop 环境；
  - Ray / PPO trainer；
  - `DataProto`；
  - multi-turn rollout；
  - actor / reference model / KL / old log-probability 等训练基础设施。

### GiGPO 核心代码

- 目录：<https://github.com/langfengQ/verl-agent/tree/master/gigpo>
- `core_gigpo.py`：<https://github.com/langfengQ/verl-agent/blob/master/gigpo/core_gigpo.py>

重点参考：

- exact `anchor_obs` grouping；
- `to_hashable`；
- trajectory-level / state-level advantage；
- GiGPO 原始训练数据组织。

---

## 2. HGPO：BACE recipe 与 trainer 组织方式参考

HGPO 已集成在 `verl-agent` 中，不需要单独 fork 一个仓库。

- HGPO recipe：<https://github.com/langfengQ/verl-agent/tree/master/recipe/hgpo>
- `core_hgpo.py`：<https://github.com/langfengQ/verl-agent/blob/master/recipe/hgpo/core_hgpo.py>
- `hgpo_ray_trainer.py`：<https://github.com/langfengQ/verl-agent/blob/master/recipe/hgpo/hgpo_ray_trainer.py>
- `env_manager.py`：<https://github.com/langfengQ/verl-agent/blob/master/recipe/hgpo/env_manager.py>

主要参考：

- 如何在 `recipe/` 下建立独立算法；
- 如何扩展 trainer 而尽量不修改 `verl/` 核心；
- group-based advantage 与 batch balancing 的执行顺序；
- ALFWorld / WebShop 独立训练配置与启动脚本。

BACE 不采用 HGPO 的 history-aware anchor 定义，仅参考其工程组织方式。

---

## 3. GraphGPO：Tree / Graph metadata 与调试工具参考

GraphGPO 同样已集成在 `verl-agent` 中。

- GraphGPO recipe：<https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO>
- `graphgpo_env_manager.py`：<https://github.com/langfengQ/verl-agent/blob/master/recipe/GraphGPO/graphgpo_env_manager.py>

主要参考：

- state node / action edge 数据结构；
- trajectory-to-graph mapping；
- 跨 trajectory 的 state/action 聚合；
- tree / graph metadata；
- 可视化与 advantage diagnostics。

BACE 不采用 GraphGPO 的 shortest-path reward 或 reverse-Dijkstra credit。

---

## 4. PivoARL：在线环境 Replay 的主要参考仓库

- GitHub：<https://github.com/yuki-younai/PivoARL>

这是 BACE 环境 replay 最值得直接审计的外部仓库之一。

### ALFWorld Replay

- 环境 worker：<https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/alfworld/envs.py>
- Environment manager：<https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/alfworld/env_manager.py>

重点关注：

- `restart()`；
- `restart_from_turn(...)`；
- same-game reset；
- recorded action-prefix replay；
- per-environment target turn；
- action journal；
- memory truncate / copy；
- replay 提前 terminal 的异常检查。

### WebShop Replay

- 环境 worker：<https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/webshop/envs.py>
- Environment manager：<https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/webshop/env_manager.py>

重点关注：

- 保存同一 `session_idx`；
- reset 到同一 session；
- replay search / click / page actions；
- 恢复 `available_actions`；
- 不同 worker 使用不同 replay depth。

BACE 只借鉴 replay infrastructure，不采用 PivoARL 的 reflection、pivotal-turn selection 和 retry-specific credit。

---

## 5. Prefix-GRPO：Replay Validation 与 Prefix 数据审计参考

- GitHub：<https://github.com/HappynessI/Prefix_GRPO>
- README：<https://github.com/HappynessI/Prefix_GRPO/blob/main/README.md>
- ALFWorld prefix 数据构建 / replay validation：  
  <https://github.com/HappynessI/Prefix_GRPO/blob/main/scripts/build_data/build_alfworld_prefix_rl_change_top3.py>

主要参考：

- `prefix_actions`；
- environment reset key；
- replay 后 observation 与 expected observation 比较；
- replay failure / mismatch 分类；
- 只让验证通过的 replay record 进入训练；
- 同一 trajectory 多个 cut point 的 grouped / incremental replay 思路；
- prefix span / mask 的数据组织方式。

BACE 不采用其 teacher-prefix selection 和 historical-prefix token objective。

---

## 6. 3SPO：动态 Rollout 调度参考

- GitHub：<https://github.com/genalyu/3SPO>

主要参考：

- variable rollout allocation；
- state-level statistics；
- 动态 rollout bookkeeping；
- verl 上的多阶段训练组织。

当前 BACE 不依赖其作为 replay 实现来源；我们的中间状态恢复优先参考 PivoARL 与 Prefix-GRPO。

---

## 7. EnvRL：低侵入扩展 verl-agent 的参考

- GitHub：<https://github.com/zt-wang19/EnvRL>

主要参考：

- 如何在 `verl-agent` 上增加额外 transition metadata；
- 如何保持原 GiGPO/PPO 主训练框架不变，同时加入新的环境级信息；
- ALFWorld / WebShop 的低侵入式扩展方式。

EnvRL 不作为 BACE replay 的主要实现来源。

---

## 8. Tree-GRPO：Tree bookkeeping 参考

- GitHub：<https://github.com/AMAP-ML/Tree-GRPO>

主要参考：

- parent / child node ID；
- shared prefix；
- tree rollout metadata；
- leaf-to-root lineage；
- tree batch flattening。

其主要环境是搜索 / QA 型 ReAct agent，因此不直接提供我们需要的 ALFWorld / WebShop simulator replay adapter。

---

## 9. BPO：Branching Policy Optimization

工程规范中将 BPO 作为 branching 方法和 matched-compute baseline 的参考。

截至 2026-08-07，本次检索没有确认到与论文 **Branching Policy Optimization: Sandbox-Native Language Agent Reinforcement Learning** 对应、且可以审计 ALFWorld / WebShop replay 实现的官方 GitHub 仓库。

因此当前不把第三方非官方仓库作为 BACE 的代码来源。

实现时主要参考其论文中的：

- branch / sibling rollout 设计；
- sandbox snapshot / restore 抽象；
- matched-compute evaluation。

---

## 10. TRACE：Tree Rollout Allocation for Contrastive Exploration

工程规范中主要将 TRACE 用作 root / prefix rollout budget allocation 的实验与方法参考。

截至 2026-08-07，本次检索没有确认到与该 TRACE 论文对应的官方 GitHub 实现仓库。

注意不要与以下同名项目混淆：

- `microsoft/trace`：<https://github.com/microsoft/trace>

该仓库是另一项 AI system optimization 工作，并不是本文所讨论的 Tree Rollout Allocation for Contrastive Exploration。

---

# 11. 推荐的实际阅读顺序

如果目标是开始实现 BACE-GiGPO，建议按以下顺序阅读代码：

1. **verl-agent / GiGPO**  
   <https://github.com/langfengQ/verl-agent>

2. **HGPO recipe**  
   <https://github.com/langfengQ/verl-agent/tree/master/recipe/hgpo>

3. **GraphGPO recipe**  
   <https://github.com/langfengQ/verl-agent/tree/master/recipe/GraphGPO>

4. **PivoARL ALFWorld replay**  
   <https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/alfworld/envs.py>

5. **PivoARL ALFWorld manager**  
   <https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/alfworld/env_manager.py>

6. **PivoARL WebShop replay**  
   <https://github.com/yuki-younai/PivoARL/blob/main/agent_system/environments/env_package/webshop/envs.py>

7. **Prefix-GRPO replay validation**  
   <https://github.com/HappynessI/Prefix_GRPO/blob/main/scripts/build_data/build_alfworld_prefix_rl_change_top3.py>

8. **3SPO**  
   <https://github.com/genalyu/3SPO>

9. **EnvRL**  
   <https://github.com/zt-wang19/EnvRL>

10. **Tree-GRPO**  
    <https://github.com/AMAP-ML/Tree-GRPO>

---

# 12. BACE 实现时的来源对应关系

| BACE 模块 | 首选参考来源 |
|---|---|
| 基础 rollout / PPO / Ray / environment | `langfengQ/verl-agent` |
| BACE recipe 和 trainer 组织 | `verl-agent/recipe/hgpo` |
| tree metadata / visualization | `verl-agent/recipe/GraphGPO` |
| ALFWorld online replay | `yuki-younai/PivoARL` |
| WebShop online replay | `yuki-younai/PivoARL` |
| replay consistency validation | `HappynessI/Prefix_GRPO` |
| prefix/action replay 数据 schema | `HappynessI/Prefix_GRPO` |
| variable rollout scheduling | `genalyu/3SPO` |
| verl-agent 低侵入扩展方式 | `zt-wang19/EnvRL` |
| 通用 tree bookkeeping | `AMAP-ML/Tree-GRPO` |
| branching / matched-compute baseline | BPO 论文，当前未确认官方 GitHub |
| root/prefix allocation baseline | TRACE 论文，当前未确认官方 GitHub |

---

## 最终工程原则

BACE-GiGPO 不应把多个外部仓库直接合并成一个代码库。推荐的方式是：

```text
verl-agent / GiGPO        ← 唯一 upstream base
        │
        ├── HGPO           ← recipe / trainer 结构
        ├── GraphGPO       ← tree metadata / diagnostics
        ├── PivoARL        ← restart + action-prefix replay
        ├── Prefix-GRPO    ← replay validation / schema
        ├── 3SPO           ← dynamic scheduling 参考
        ├── EnvRL          ← low-invasive extension 参考
        └── Tree-GRPO      ← generic tree bookkeeping 参考
```

外部代码若实际移植，应记录：来源仓库、源文件、commit、license、修改内容和对应单元测试。
