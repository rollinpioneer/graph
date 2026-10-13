# FLOQ-MIN-REPRO-V2：floq 最小可信复现实验卡

**日期：** 2026-10-13  
**实验状态：** DRAFT / 待批准 / 尚未执行  
**实验性质：** Baseline 复现、工程验证与模块拆分准备  
**目标论文：** *floq: Training Critics via Flow-Matching for Scaling Compute in Value-Based RL*（ICLR 2026）

**官方代码：** https://github.com/CMU-AIRe/floq  
**候选固定 Commit：** `3d60e638b42bbef018c56cb1199ff37c3470520d`

## 一、核心研究目标

本实验只复现 floq 中与后续 Critic 模块替换研究直接有关的部分，不追求完整重现论文全部实验。

### H1：基础复现假设

使用作者公开源码、原始配置及相同训练预算，可以在选定 OGBench 任务上获得与论文报告合理一致的结果。

### H2：模块准入假设

floq 的 Flow Critic 及其相关训练组件可以被准确拆分、记录依赖关系，并在后续研究中作为可替换的价值学习模块。

本次只验证复现可靠性及工程结构，不验证与 GFP、FAN 或 Value Flows 组合后的性能。

## 二、实验范围

| 项目 | 实验安排 |
|---|---|
| 主任务 | `antmaze-large-navigate-singletask-v0` |
| Baseline | floq-default |
| 训练种子 | `0, 1` |
| 训练更新 | 每种子 2,000,000 updates |
| 总正式训练 | **2 条** |
| 评价 | 作者原始评价协议 |
| FQL | 已独立复现，本卡不训练 |
| 第二任务 | `cube-double-play-singletask-v0`，暂不执行 |
| 在线微调 | 不做 |
| 超参数搜索 | 不做 |
| Value Flows/GFP 模块替换 | 不做 |
| 训练时长/GPU·小时限制 | **不设置** |

第二任务只有在主任务复现完成、并再次获得批准后才增加。若增加，仍使用两个种子。

**重要限制：** 取消时间预算不等于允许随意延长训练。2M updates 是固定算法实验条件，不得因指标不足而擅自延长。

## 三、模型结构及冻结配置

### 3.1 核心模块

| 模块 | 功能 | 是否保留 |
|---|---|---|
| Flow Critic | 通过 Flow Matching 学习 Q 值计算 | 保留 |
| Target Flow Critic | 通过价值积分产生 TD 目标 | 保留 |
| Distilled Scalar Critic | 从 Flow Critic 蒸馏普通 Q 读出 | 保留 |
| BC Flow Actor | 学习离线行为分布 | 保留 |
| One-step Actor | 蒸馏动作并进行 Q 优化 | 保留 |

本次禁止调整网络结构、删除损失项或更换 Actor。

后续模块替换的候选单元为：

**Flow Critic + Target Flow Critic + Distilled Scalar Critic，以及相应的训练损失与更新逻辑。**

### 3.2 参数配置

| 参数 | 固定值 |
|---|---|
| Environment | `antmaze-large-navigate-singletask-v0` |
| Offline updates | 2,000,000 |
| Online updates | 0 |
| Learning rate | 3e-4 |
| Batch size | 256 |
| Discount | 0.99 |
| Target update τ | 0.005 |
| BC coefficient α | 10 |
| Actor Flow steps | 10 |
| Critic Flow steps | 8 |
| Flow Critic ensembles | 2 |
| Distilled Critic ensembles | 2 |
| Noise samples | 8 |
| Noise coverage | 0.1 |
| Q aggregation | mean |
| Evaluation interval | 20,000 |
| Evaluation episodes | 50 |
| Checkpoint interval | 1,000,000 |
| Random seeds | 0、1 |

其他模型参数使用固定作者源码中的 `get_config()` 默认设置。

环境优先使用作者提供的 `conda-environment.yaml`，其中 OGBench 版本为 `1.1.2`。若原始环境无法在现有硬件安装，允许提出兼容方案，但必须单独记录依赖变更，并验证是否影响训练或评价语义。

## 四、运行命令

下列为根据作者仓库配置整理的执行模板，未经过本地运行验证。

```bash
# Seed 0
python main.py \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --agent=agents/floq.py \
  --agent.alpha=10 \
  --offline_steps=2000000 \
  --online_steps=0 \
  --seed=0 \
  --save_dir="$EXPERIMENT_ROOT" \
  --project_name=CP_FLOQ \
  --run_group=FLOQ_MIN_REPRO_V2

# Seed 1
python main.py \
  --env_name=antmaze-large-navigate-singletask-v0 \
  --agent=agents/floq.py \
  --agent.alpha=10 \
  --offline_steps=2000000 \
  --online_steps=0 \
  --seed=1 \
  --save_dir="$EXPERIMENT_ROOT" \
  --project_name=CP_FLOQ \
  --run_group=FLOQ_MIN_REPRO_V2
```

`EXPERIMENT_ROOT` 应指向新建、可写且不会覆盖既有结果的位置。

正式运行前核实 W&B 设置、保存路径、环境加载和随机种子传递。

## 五、执行阶段

### P0：静态核验与环境准备

允许进行源码、配置和依赖检查，不启动正式训练。

需要确认：

1. 作者 Commit、全部模型配置和训练入口。
2. 数据集版本、下载来源及数据完整性。
3. 两个 seed 的初始随机性和评价随机性。
4. GPU/JAX/CUDA 兼容性。
5. Checkpoint 保存、恢复与评价流程。
6. 作者结果的评价时点、聚合方式及参考数据。
7. 与已经开展的 FQL 复现是否具有可比的任务与评价协议。

产物：源码及环境清单、最终配置快照、数据标识和问题记录。

### P1：功能试运行

使用 floq seed 0 进行独立短试运行，建议 5,000 updates。

检查：

- 模型可正确初始化，能够执行训练与评价。
- 各 Critic、Actor 损失数值正常。
- 目标网络更新与价值积分能够执行。
- 无持续 NaN/Inf、非法动作或明显数值异常。
- Checkpoint 能够保存和恢复。
- 日志完整，结果路径独立。

**不设时间上限或 GPU·小时上限。**

若出现问题，可以排查环境或非算法工程错误，但不得偷偷改变模型、训练损失或评价规则。

P1 只证明工程功能可用，不产生正式性能结论。

### P2：正式双种子复现

完成 P0/P1 并获训练批准后，执行两条独立完整训练：

| Run | Seed | Updates |
|---|---:|---:|
| FLOQ-S0 | 0 | 2,000,000 |
| FLOQ-S1 | 1 | 2,000,000 |

全部使用同一源码和冻结配置。

原则上完成全部预定更新。训练途中不根据指标提前挑选最终模型，也不因为性能未达预期自动追加训练。

保留完整评价曲线、最终权重及中间 checkpoint。

## 六、评估指标与复现判定

### 6.1 主指标

**OGBench Success Rate**

报告两个 seed 的逐种子值、算术均值和离散程度。

主结果使用作者正式结果采用的 checkpoint 和聚合方式；若作者协议不能完全核实，使用预先固定的 2M checkpoint，并明确标为本实验评价口径。

不得使用训练中最高 Success Rate 替代主结果。

### 6.2 辅助指标

| 类别 | 指标 |
|---|---|
| Flow Critic | `floq_loss` |
| 蒸馏 | `distilled_critic_loss` |
| TD 误差 | `td_error` |
| 价值波动 | `var_flow_returns` |
| Actor | BC、distillation、Q loss |
| 计算效率 | 稳态更新耗时、评价耗时、显存峰值 |
| 实验完整性 | 完成步数、日志、Checkpoint、协议一致性 |

计算成本仅用于记录和分析，**不作为时间预算或停止阈值**。

### 6.3 结果判定

本次只有两个训练种子，而作者使用三个种子，因此无法完整复现其三种子统计结论。

采用以下判定标签：

**A — LIMITED_REPRODUCED**

两次训练完整，配置与作者一致，聚合口径已核对，表现处于作者结果的合理波动范围。

**B — FUNCTIONAL_REPRODUCED**

算法成功运行、完整训练并产生有效策略，但统计结果尚不足以确认论文性能复现。

**C — PERFORMANCE_MISMATCH**

代码和协议经核验有效，但性能与作者结果出现明显且可重复的差异。

**D — INCONCLUSIVE / TECHNICAL_FAILURE**

存在环境、协议、日志、训练完整性或参考指标上的关键缺口，无法可靠判定。

评价标准不得在看到结果之后改变；不得把两个 seed 的均值当作跨种子稳健性的充分证明。

## 七、与正在复现的 FQL 的关系

本实验不再分配 FQL 训练任务，也不操作其已有权重或实验进程。

在 FQL 结果可用后，对照以下项目：

| 项目 | FQL | floq |
|---|---|---|
| Task | 核对一致性 | 主任务固定 |
| 数据和版本 | 核对 | 作者固定版本 |
| Updates | 核对 | 2M |
| Seeds | 记录 | 0、1 |
| Actor/BC 设置 | 核对 | 作者 floq 配方 |
| Critic 类型 | Scalar TD Critic | Flow + Distilled Critic |
| 评价协议 | 核对 | 作者 floq 协议 |

若 FQL 与 floq 的数据版本、训练更新数、评价任务或 Actor 配方不同，必须明确说明其混杂因素，不能将性能差异单独归因于 Critic。

**FQL 尚未完成不阻碍 floq 的独立复现。**

## 八、停止条件

不设置训练时间或 GPU·小时上限。

仅在以下情况暂停、停止或申请修订：

- 关键依赖和数据环境无法正确使用。
- 持续 NaN/Inf 或明显错误的梯度、状态、动作。
- 训练日志、Checkpoint 或评价结果无法可信保存。
- 运行条件偏离冻结配置，且偏差不能作为非算法工程修复处理。
- 继续运行可能破坏已有结果或干扰共享资源中的其他实验。

发生以上情况时，保留原始日志并分析，不直接追加新方法、改变超参数或放宽成功标准。

## 九、最终交付产物

正式实验完成后应提供：

1. **复现身份清单：** Git SHA、依赖版本、数据版本、随机种子和完整命令。
2. **两条完整训练记录：** checkpoint、日志、评价曲线及异常记录。
3. **性能复现报告：** 作者参考结果、本地结果、统计局限、协议差异。
4. **Critic 模块拆分说明：** Flow Critic、Target Critic、Distilled Critic 的计算图、损失函数、梯度关系和输入输出。
5. **研究结论：** floq 是否通过 Baseline 准入，以及后续与 Value Flows 或 GFP 进行模块替换时必须保留的接口约束。

## 十、权限与实验状态

**当前状态：DRAFT，等待批准。**

本卡只规定 floq 的复现工作，不授权更改已有 FQL 实验、不覆盖既有产物、不干预其他进程。

批准正式复现后，按 P0 → P1 → P2 执行。本卡不自动授权第二任务训练、Critic 替换实验、超参数搜索或新的研究方向。

**最终正式训练范围：floq × 1 个任务 × 2 个种子 = 2 条完整训练；每条 2M updates，无时间预算限制。**