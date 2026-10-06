---
title: "CP-DISR C1 跨任务主实验：Blocksworld 候选—目标可达性验证"
subtitle: "Cross-Task Main Experiment Runbook v1"
date: "2026-10-06"
lang: "zh-CN"
card_id: "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
status: "PLAN_ONLY"
authorized: false
repository: "rollinpioneer/graph"
base_branch: "codex/cp-disr-c1-mechanism-confirmation-v1"
base_commit: "1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e"
new_branch: "codex/cp-disr-c1-blocksworld-main-v1"
default_training_runs: 3
---

# CP-DISR C1 跨任务主实验 Runbook v1

## 0. 一页式决策摘要

本阶段停止继续扩旧 T_B/Fresh Confirm，不再把“显式生成后继状态是否必要”作为主问题。

Claude 的交叉核查指出：

1. B2、QMARK 和规划器在 Fresh Confirm 的**决策层都是 32/32**；剩余失败来自物理执行、公开观测或评估器，而不是选错动作。
2. Fresh Confirm 的四个 cell 中，三个是训练目标在交换两个同类型物体后的镜像；因此旧结果不能直接等同于一般意义的组合泛化。
3. +E 的主要问题不是“训练习惯太强”这种模糊说法，而是：它无法让候选动作的效果沿关系图到达具体目标命题，在镜像条件下只能依赖名字通道。
4. 唯一真正新的旧 T_B 组合是 `BUF_T+BUF_S`；该 cell 又混入了物理观测和评估器不一致，不能继续承担下一阶段的主要区分任务。

因此，下一阶段的主问题改为：

> **已知动作合同和当前状态时，一个候选动作的效果需要怎样到达带目标标记的命题，策略才能迁移到非同构的新目标结构？**

本卡只做：

- 一个公开的第二任务族：**符号 Blocksworld**；
- 三条学习训练：
  - B2-CACHED；
  - QMARK；
  - ASNET-READOUT（ASNet-style 适配读出）；
- 一个不训练的精确合同规划器；
- 一次冻结后的正式评估。

默认新增训练总数：

```text
3
```

不自动增加 seed，不自动安装多套外部方法，不重跑旧 T_B，不恢复 C2/C3。

---

# 1. 本卡依据与研究定位

## 1.1 现有基线

### Structural Generalization

```text
branch:
codex/cp-disr-tb-structural-generalization-v1

result commit:
5a2b3d18d21cbab19b9d09c3c430b3203e873730

card:
CP-DISR-TB-STRUCT-GEN-V1
```

主要结果：

```text
B2   = 0.922
ABS  = 0.733
NC   = 0.222
+E   = 0.000
```

冻结分类：

```text
OUTCOME_A
STRUCTURAL_GROUNDING_HYPOTHESIS_STRENGTHENED
```

### C1 Mechanism Confirmation

```text
branch:
codex/cp-disr-c1-mechanism-confirmation-v1

result commit:
1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e

card:
CP-DISR-C1-MECH-CONFIRM-V1
```

Fresh Confirm：

| 方法 | 成功 | 决策层从未偏离公开最优动作 |
|---|---:|---:|
| B2 | 27/32 | 32/32 |
| QMARK | 27/32 | 32/32 |
| B_PLAN | 26/32 | 32/32 |
| ABS | 23/32 | 24/32 |
| NC | 8/32 | 8/32 |
| +E | 0/32 | 0/32 |

冻结分类：

```text
M3
RELATIONAL_QUERY_IS_SUFFICIENT
NARROW_AND_CONTINUE
```

## 1.2 对旧主张的正式修订

本卡不再使用下列宽泛主张作为研究前提：

```text
模型已经证明学会一般性的动作重组。
B2 与 QMARK 学到了不同的内部规则。
B2/QMARK 在 Fresh Confirm 的物理成功差异代表决策能力差异。
显式构造后继是结构泛化的必要条件。
+E 只是因为探索不够或经验不足而失败。
```

本卡采用的收窄结论是：

> T_B 表明：当候选动作的信息能够沿动作—命题关系传播到目标命题时，B2、QMARK 等表示可以完成对象角色互换，并处理一个新的双目标组合；显式写入后继事实不是必要条件。

进一步需要由第二任务族回答：

> 这种能力是否能超出对象镜像对称，进入颜色模式反转、非同构目标形状、多目标干扰和规模增长？

## 1.3 创新判断原则

本研究不追求“从未有人碰过”的绝对原创。

ASNet、关系 GNN、广义策略、后继状态打分等已有工作均与本研究共享部分思想。它们的存在不自动否定本方向。

本阶段判断新增贡献时，关注：

1. 是否把“动作执行暴露”“目标绑定未见”“对象置换同构”“非同构目标”和“规模增长”分开测量；
2. 是否提出并检验“候选到目标可达性”这一统一解释；
3. 是否能比较直接动作节点读出、候选查询传播和显式后继差分的条件性优劣；
4. 是否发现已有关系策略在多目标协调、目标保持或有限传播深度上的明确不足；
5. 是否形成比“某方法分数更高”更完整的机制证据链。

---

# 2. 研究问题与假设

## 2.1 核心研究问题

### RQ1：超出镜像对称后，关系表示是否仍能迁移？

训练目标与测试目标在颜色标签或目标形状上不再能通过对象改名互相得到时，B2、QMARK、ASNET-READOUT 是否仍能正确选择动作？

### RQ2：候选条件化是否必要？

直接读取候选动作节点的表示，能否达到 QMARK 和 B2 的水平？

若能，则当前现象可能主要属于关系图方法的共同能力。

若不能，则候选查询或后继构造提供了额外作用。

### RQ3：显式后继何时产生额外价值？

在候选到目标的传播距离仍处于四层编码器感受野以内时，B2 是否在非同构目标或目标干扰条件下稳定领先？

### RQ4：真正的新算法信号是什么？

若 B2、QMARK、ASNET-READOUT 在规划器可解、传播距离不超过四跳的多目标干扰问题上共同失败，是否表现出一致的目标破坏或错误顺序？

只有出现这种信号，才进入新算法设计。

### RQ5：学习方法相对搜索的合理位置是什么？

在纯符号、规则完全已知的 Blocksworld 中，精确规划器是强参照。学习方法不要求击败规划器，而是检验：

- 是否可以学习一个可迁移的摊销策略；
- 决策延迟和模型前向成本如何；
- 随任务规模增加时，学习策略与搜索成本分别如何变化。

## 2.2 可证伪预测

### H1：同构正对照

在对象改名但结构完全相同的 A0 切片上，三种关系方法应保持高表现。

若失败，优先判断实现、置换等变或数据绑定错误，不进入科学解释。

### H2：候选到目标传播

当候选动作到相关目标命题的最短关系距离不超过四跳时：

- B2；
- QMARK；
- ASNET-READOUT；

至少应有一种方法能够稳定完成 A1/A2。

### H3：直接动作节点读出

若 ASNET-READOUT 追平 B2/QMARK，则说明：

> 直接读取经过关系传播的候选动作节点已经足够。

此时不把 QMARK 包装成新主方法。

### H4：显式后继额外价值

若在非同构、四跳以内的条件下，B2 明显领先 QMARK 和 ASNET-READOUT，则显式后继差分具有可继续发展的额外价值。

### H5：传播深度限制

若三种方法在 `hop > 4` 条件下共同下降，而在 `hop <= 4` 条件下保持较强，则将其标记为已知感受野限制，不直接视为新算法贡献。

### H6：多目标协调信号

若三种方法在 `hop <= 4`、规划器可解的目标干扰任务中共同失败，并反复表现为：

- 破坏已完成目标；
- 错误安排目标顺序；
- 在局部正确动作之间循环；

则允许进入“多目标协调”算法设计。

---

# 3. 范围与非目标

## 3.1 本卡包含

- 纯符号 Blocksworld；
- 三种学习方法；
- 一个精确规划器；
- 同构正对照；
- 颜色模式反转；
- 非同构目标形状；
- 多目标干扰；
- 规模增长；
- 决策层评价；
- 参数量、编码次数和推理时间记录。

## 3.2 本卡不包含

- 视觉输入；
- 真实机器人或 MuJoCo；
- UNKNOWN 观测；
- VLM prior；
- C2/C3；
- 新奖励模型；
- learned world model；
- 多套外部框架复现；
- 新的注意力、记忆或规划模块；
- T_B 新 seed；
- QMARK seed1/seed2 的 Fresh Confirm 评估；
- 旧 test30/Fresh Confirm 重跑；
- 为证明某方法获胜而修改任务。

---

# 4. Git、分支和保护边界

## 4.1 仓库

```text
repository:
rollinpioneer/graph
```

## 4.2 基线

```text
base branch:
codex/cp-disr-c1-mechanism-confirmation-v1

base commit:
1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e
```

开始前必须验证：

```bash
git rev-parse HEAD
git status --short
git branch --show-current
```

## 4.3 新分支

```text
codex/cp-disr-c1-blocksworld-main-v1
```

从完整 base commit 创建独立 worktree。

不得在旧 T_B 或 Mechanism Confirmation worktree 上直接开发。

## 4.4 旧证据保护

下列内容只读：

```text
runs/final_master/2.1.1/structgen/
runs/final_master/c1_route_b/mech_confirm_v1/
```

要求：

- 不覆盖；
- 不改名；
- 不删除；
- 不重新生成原结果；
- 不把新 Blocksworld 结果写入旧目录。

## 4.5 新输出目录

```text
runs/final_master/c1_route_b/
blocksworld_main_v1/<UTC>_<prep_sha8>/
```

建议结构：

```text
prep/
registration/
runs/
eval/
results/
receipts/
```

---

# 5. Blocksworld 任务定义

## 5.1 环境性质

环境必须是：

- 纯符号；
- 完全可观测；
- 确定性；
- 无 UNKNOWN；
- 所有状态变化仅来自冻结动作合同；
- 不调用仿真器、GPU 渲染或外部 provider。

## 5.2 动态谓词

```text
On(x, y)
OnTable(x)
Clear(x)
Holding(x)
HandEmpty
```

约束：

- 一个 block 最多位于一个支撑物上；
- 一个 block 最多支撑一个其他 block；
- 手中最多持有一个 block；
- 非法状态不得进入任何 split。

## 5.3 静态属性谓词

每个 block 恰有一个颜色：

```text
Red(x)
Blue(x)
```

颜色在整个 episode 中不改变。

为了让颜色模式在关系图中可表示：

- 每个 grounded action 必须与其参数对应的颜色事实建立合同关系；
- 允许将 `Red(x)` / `Blue(x)` 作为始终为真的静态前提加入该 grounded contract；
- 这些静态前提不改变动作合法性；
- 不得仅把颜色藏在对象字符串或哈希中。

## 5.4 动作合同

### PICK_UP(x)

前提：

```text
OnTable(x)
Clear(x)
HandEmpty
Color(x)
```

效果：

```text
ADD Holding(x)
DEL OnTable(x)
DEL Clear(x)
DEL HandEmpty
```

### PUT_DOWN(x)

前提：

```text
Holding(x)
Color(x)
```

效果：

```text
ADD OnTable(x)
ADD Clear(x)
ADD HandEmpty
DEL Holding(x)
```

### UNSTACK(x, y)

前提：

```text
On(x, y)
Clear(x)
HandEmpty
Color(x)
Color(y)
```

效果：

```text
ADD Holding(x)
ADD Clear(y)
DEL On(x, y)
DEL Clear(x)
DEL HandEmpty
```

### STACK(x, y)

前提：

```text
Holding(x)
Clear(y)
Color(x)
Color(y)
x != y
```

效果：

```text
ADD On(x, y)
ADD Clear(x)
ADD HandEmpty
DEL Holding(x)
DEL Clear(y)
```

`Color(x)` 表示与 x 对应的唯一 `Red(x)` 或 `Blue(x)`。

## 5.5 终止与奖励

成功条件：

```text
全部 goal atoms 为 TRUE
```

动作失败：

```text
不允许执行 mask=false 的动作
```

episode step cap：

\[
\text{cap(case)} = 2L^* + 4
\]

其中 \(L^*\) 是冻结规划器在初始状态的最短计划长度。

训练 reward：

- 第一次达到完整目标：`1.0`；
- 其他步骤：`0.0`；
- 不增加中间子目标奖励；
- 不用 planner distance 作为 reward；
- 不用颜色匹配作为 reward。

每个动作时长视为 1 个符号时间单位。

折扣半衰期：

```text
H = 训练 split 最短计划长度的中位数
```

H 在任何训练前计算并冻结。

---

# 6. 图表示与候选输入

## 6.1 统一图

三种方法使用相同的：

- ACTION 节点；
- PROPOSITION 节点；
- PRE_POS / PRE_NEG；
- ADD / DEL；
- 反向消息边；
- 目标标记；
- 四层 RGCN；
- LayerNorm；
- CandidateReadout；
- GRU；
- Actor / Value / Q heads。

## 6.2 禁止对象名捷径

Blocksworld 的 `candidate_features` 不得包含：

- block 名字哈希；
- grounded action ID 哈希；
- case ID；
- split；
- 目标类别编号；
- canonical signature；
- 颜色字符串哈希。

固定 8 维候选特征：

```text
[one_hot(PICK_UP, PUT_DOWN, UNSTACK, STACK), arity/2, 0, 0, 0]
```

即：

- 前四维：动作 schema one-hot；
- 第五维：参数个数除以 2；
- 后三维：0。

具体 grounded 绑定只能通过关系图进入策略。

## 6.3 base_input

纯符号环境没有 proprioception。

固定 48 维：

```text
step_index / step_cap
remaining_steps / step_cap
previous_action_succeeded
其余 45 维为 0
```

若动作均为合法执行，则 `previous_action_succeeded=1`；初始为 0。

不得编码：

- block 身份；
- goal hash；
- planner distance；
- 最优动作；
- split 信息。

---

# 7. 三种学习方法

## 7.1 B2-CACHED

数学定义保持 B2：

\[
r_i = E(F_i^{after}) - E(F)
\]

要求：

- 每个决策只计算一次 `E(F)`；
- 每个合法候选计算一次 `E(F_i^{after})`；
- 总编码次数为 `1 + K`；
- 不再按候选重复编码同一个当前状态；
- 参数、CandidateReadout 和 heads 与原 B2 一致。

必须增加两类等价测试：

1. 同一权重和 snapshot 下，原 B2 与 B2-CACHED 的 logits/value/Q 在容差内一致；
2. 同一标量 loss 下，所有共享参数梯度在容差内一致。

建议标识：

```text
method: B2-CACHED
paper label: B2
```

## 7.2 QMARK

沿用已冻结 QMARK 的定义：

\[
r_i = E(G(F,q=a_i)) - E(G(F,q=\varnothing))
\]

要求：

- 不修改事实值；
- 不调用 nominal_apply；
- 只在 queried ACTION 节点增加共享二值标记；
- 总编码次数 `1 + K`；
- 不增加 candidate-specific 参数；
- Blocksworld 任务适配不得改变 QMARK 数学形式。

建议标识：

```text
method: B1-K+QMARK-BW
paper label: QMARK
```

## 7.3 ASNET-READOUT

这是 **ASNet-style 适配读出**，不是完整复现原 ASNet。

它使用：

- 同一个四层关系图编码器；
- 当前状态图；
- 同样的目标标记；
- 同样的候选特征；
- 同样的 GRU、Actor、Value、Q heads；
- 同样的 PPO。

每个决策只编码一次当前图，得到所有节点的最终隐藏表示：

\[
H = E_{node}(G(F))
\]

对候选动作 \(a_i\)：

\[
r_i = H[\text{ACTION}(a_i)]
\]

然后交给与其他方法相同的候选读出和策略 heads。

禁止：

- query marker；
- nominal successor；
- planner feature；
- LM-cut feature；
- 教师动作标签；
- 额外候选 MLP；
- 原 ASNet 的模仿学习作为默认训练。

建议标识：

```text
method: ASNET-READOUT
paper label: ASNet-style readout
```

必须在论文和结果中声明：

> 同编码器、同 PPO 的动作节点读出适配；不是原论文完整系统，不包含其教师模仿、专用交替层或启发式特征。

## 7.4 参数与计算记录

训练前输出：

```text
effective_trainable_parameters
parameters_by_module
encoder_forwards_per_decision
nominal_apply_calls
node_feature_calls
CPU median forward time
GPU median forward time（若训练设备支持）
```

预期：

| 方法 | 编码次数 |
|---|---:|
| B2-CACHED | 1 + K |
| QMARK | 1 + K |
| ASNET-READOUT | 1 |

不为了“等计算”而让 ASNET-READOUT 重复无用前向。

---

# 8. 精确规划器与决策标签

## 8.1 规划器

建立 Blocksworld 专用精确规划器。

推荐：

- A*；
- admissible `h_max`；
- 固定字典序 tie-break；
- 状态缓存；
- 对小规模问题用 BFS 独立交叉验证最短长度。

规划器输入仅包括：

- 当前符号状态；
- 当前目标；
- 冻结动作合同。

规划器不读取模型 logits、Q 或 hidden state。

## 8.2 冻结前资格

每个 train/dev/test case 必须：

- 有解；
- 精确最短长度已知；
- 至少一个最优计划已保存；
- 在冻结资源上搜索完成；
- 无搜索 timeout；
- 无节点上限停止。

建议上限仅作保护：

```text
max_nodes = 2,000,000
cpu_time_limit = 30 s / planning query
max_depth = 32
```

若某 case 不满足，必须在任何训练前重新生成。

训练开始后不得删除难例。

## 8.3 每个决策的规划器标签

正式评估时，对策略访问到的每个状态计算并缓存：

```text
optimal_remaining_length
optimal_action_set
selected_action_is_optimal
selected_action_excess_cost
satisfied_goals_before
satisfied_goals_after
destroyed_satisfied_goal
```

定义：

### 首次偏离

```text
first_divergence_step =
第一个 selected_action 不在 optimal_action_set 的决策编号
```

### 决策完美 episode

```text
decision_perfect =
整个 episode 中每一步 selected_action 都在 optimal_action_set
```

### 目标破坏

```text
destroyed_satisfied_goal =
执行后，执行前为 TRUE 的某个 goal atom 变为 FALSE
```

### 超额动作数

成功 episode：

\[
\text{excess steps} = \text{executed steps} - L^*
\]

失败 episode 记录 step cap，不伪造 excess 值。

---

# 9. 候选到目标的关系距离

## 9.1 定义

在策略实际使用的有向消息图中，定义：

```text
candidate_goal_hops(a, F, G)
```

为候选 ACTION 节点到任一尚未满足的目标 PROPOSITION 节点的最短有向路径长度。

使用：

- PRE/ADD/DEL 边；
- 实际存在的反向消息边；
- 与生产编码器完全相同的边集合。

如果无路径：

```text
INF
```

## 9.2 分层

```text
H1: 1–2 hops
H2: 3–4 hops
H3: >4 hops
H4: INF
```

四层编码器的主要机制结论只使用：

```text
hop <= 4
```

`hop > 4` 仅作为规模和感受野范围结果。

## 9.3 冻结前覆盖要求

A1 与 A2：

- 初始最优动作必须存在 `hop <= 4` 路径；
- planner 最优轨迹中至少 80% 决策属于 `hop <= 4`；
- 若不满足，生成器必须在训练前修订。

B 规模切片：

- 至少 30 个最优决策属于 `hop <= 4`；
- 至少 30 个最优决策属于 `hop > 4`；
- 若无法构造，停止并报告，不在结果后改变分层。

---

# 10. 目标干扰标签

## 10.1 `requires_goal_destruction`

若每条最优计划都至少一次将已经满足的 goal atom 变为 FALSE，则：

```text
requires_goal_destruction = true
```

## 10.2 `monotone_solution_exists`

若存在一条最优计划，使满足的目标集合从不缩小，则：

```text
monotone_solution_exists = true
```

## 10.3 `coordination_slice`

定义：

```text
coordination_slice =
hop <= 4
AND planner_solvable
AND (
  requires_goal_destruction
  OR monotone_solution_exists = false
)
```

该切片承担“是否需要新多目标协调算法”的主要判定。

---

# 11. 数据生成与切分

## 11.1 精确同构检查

对每个问题实例生成两个哈希：

### goal_iso_hash

仅包含：

- block 颜色标签；
- goal `On` / `OnTable` 关系；
- 在所有保持颜色的对象置换中取字典序最小表示。

### problem_iso_hash

包含：

- block 颜色；
- 初始状态；
- goal；
- 保持颜色的对象置换下的最小表示。

由于 block 数量不超过 8，允许精确枚举同色对象置换。

不得只使用近似图哈希作为唯一检查。

## 11.2 Train

```text
n_blocks: 3, 4, 5
cases: 144
每个规模 48
```

目标形状限制：

- 每个目标最多一个非平凡 tower；
- 允许形状：
  - `[2, 1, ...]`
  - `[3, 1, ...]`
- 不允许两个非平凡 tower；
- tower 颜色从顶到底交替；
- 顶部固定为 Red。

因此训练目标形成可表示的颜色模式：

```text
Red → Blue → Red ...
```

初始状态要求：

- 覆盖多种 tower 形状；
- 至少 50% case 初始状态包含某个 `Blue on Red` 关系；
- 颜色反转关系在状态图中出现，但不作为训练目标模式。

## 11.3 ID Dev

```text
n_blocks: 3, 4, 5
cases: 36
每个规模 12
```

与 Train 相同分布，但：

- case ID 不重复；
- initial state 不重复；
- `problem_iso_hash` 不与 Train 重复。

## 11.4 A0：同构正对照

```text
cases: 24
```

从 ID Dev 选取 24 个实例，对同色 block 做对象改名。

要求：

- `goal_iso_hash` 与来源实例相同；
- `problem_iso_hash` 与来源实例相同；
- 原始对象字符串全部改变；
- 不参与主机制分类。

作用：

> 验证模型和 Blocksworld 适配是否保持预期的置换等变性。

## 11.5 A1：颜色模式反转

```text
cases: 32
n_blocks: 3, 4, 5
```

目标 tower 与训练使用相同的无颜色形状分布，但颜色从顶到底改为：

```text
Blue → Red → Blue ...
```

要求：

- 忽略颜色时，可与训练 goal shape 同构；
- 保持颜色时，`goal_iso_hash` 不得出现在 Train/Dev；
- `problem_iso_hash` 不得出现在 Train/Dev；
- 所需动作合同全部在训练中存在；
- 不使用对象名泄漏颜色或角色。

A1 回答：

> 当训练中的颜色—目标关联在图内可表示时，策略是否真正遵循当前目标，而不是固定颜色习惯？

## 11.6 A2：非同构新目标形状

```text
cases: 32
n_blocks: 4, 5
```

目标形状：

- n=4：
  - `[2,2]`
- n=5：
  - `[3,2]`
  - `[2,2,1]`

要求：

- 忽略颜色后，`goal_iso_hash` 仍不得出现在 Train/Dev/A1；
- 16 个 case 为 `coordination_slice=true`；
- 16 个 case 为 `coordination_slice=false`；
- 两组最短计划长度分布尽量匹配；
- 不根据任何模型结果调整。

A2 是下一阶段最重要的非同构组合测试。

## 11.7 B：规模增长

```text
cases: 48
n_blocks: 6, 7, 8
每个规模 16
```

目标必须：

- 至少两个非平凡 tower；
- `goal_iso_hash` 不出现在前述任何 split；
- 同时包含 `hop <= 4` 与 `hop > 4` 决策；
- 同时包含有干扰和无干扰实例；
- 最短计划长度建议覆盖 8–20。

B 用于范围分析，不单独决定某方法是否“创新”。

## 11.8 测试隔离

文件建议：

```text
configs/splits/c1_bw_train_dev_v1.json
configs/splits/c1_bw_a0_iso_v1.json
configs/splits/c1_bw_a1_color_reverse_v1.json
configs/splits/c1_bw_a2_noniso_v1.json
configs/splits/c1_bw_b_scale_v1.json
```

训练 loader 只能打开：

```text
c1_bw_train_dev_v1.json
```

A0/A1/A2/B 只在三条训练全部完成后，由独立评估入口打开。

---

# 12. 训练协议

## 12.1 默认训练矩阵

| Run ID | Method | Seed |
|---|---|---:|
| `R-C1-BW-B2-0` | B2-CACHED | 0 |
| `R-C1-BW-QMARK-0` | QMARK | 0 |
| `R-C1-BW-ASNET-0` | ASNET-READOUT | 0 |

总训练：

```text
3
```

不得自动增加：

- 其他 seed；
- ABS；
- +E；
- NC；
- 更深网络；
- imitation 版本。

## 12.2 共同 PPO

冻结设置建议：

```yaml
rollout_n: 1024
minibatch: 64
update_epochs: 4
lr: 0.0003
entropy_coef: 0.01
value_coef: 0.5
lambda_q: 0.1
grad_clip: 0.5
sequence_length: 16
Ncap: 131072
max_updates: 128
Tcap_wall_seconds: 28800
eval_points: [0, 16384, 32768, 65536, 131072]
```

停止条件：

```text
先到 Ncap 或 Tcap
```

三条训练可并行，但：

- 不共享 optimizer；
- 不根据中途 dev 调整预算；
- 不重启失败 run；
- 不用 best dev checkpoint；
- 正式测试只用 final checkpoint。

## 12.3 ID 资格门

final checkpoint 必须同时满足：

```text
ID Dev success >= 33/36
ID Dev decision_perfect >= 33/36
NaN = 0
hard failure = 0
```

若任一学习方法未通过：

```text
PPO_ID_GATE_FAIL
```

此时：

- 保存三条训练；
- 不进行 A1/A2/B 正式评价；
- 不自动切换到 imitation；
- 停止并请求用户决定是否授权三条统一 planner-imitation 训练。

这样默认仍只有三条训练。

---

# 13. 正式评估

## 13.1 评估顺序

仅在三种方法都通过 ID Gate 后：

1. A0；
2. A1；
3. A2；
4. B；
5. planner baseline。

每个 final checkpoint 对每个 case 只评价一次。

策略：

```text
deterministic argmax
```

不得：

- 挑 checkpoint；
- 按切片换模型；
- 在测试结果后修网络；
- 删除失败 case。

## 13.2 主指标

### Episode Decision-Perfect Rate

\[
D(m,s)=
\frac{\#\{\text{该切片中全程未偏离最优动作集合的 episode}\}}
{\#\{\text{该切片 episode}\}}
\]

这是本卡主指标。

### First Divergence

报告：

- 发生率；
- 决策位置；
- 动作类型；
- 当时 hop；
- 是否已存在满足目标；
- 是否属于 coordination slice。

### Success Rate

作为系统结果保留。

纯符号环境中，成功和决策质量应高度一致；若不一致，必须解释执行器或 cap。

## 13.3 次指标

- excess steps；
- destroyed satisfied goal；
- repeated state-action cycle；
- training color-pattern error；
- success by n_blocks；
- decision-perfect by hop；
- decision-perfect by coordination；
- planning nodes；
- planner CPU；
- model forward time；
- encoder forwards；
- parameters。

## 13.4 A0 实现门

A0 对任一方法：

```text
decision_perfect < 0.90
```

则标记：

```text
EQUIVARIANCE_OR_BINDING_GATE_FAIL
```

停止主结果解释。

---

# 14. 冻结结果分类

本卡输出两个相互独立的标签：

1. 表示排序标签；
2. 失败范围标签。

所有阈值是预注册的研究决策规则，不冒充统计显著性。

## 14.1 Core

主比较只用：

```text
A1 + A2
hop <= 4
```

定义：

```text
D_core(m) =
A1 与 A2 的 decision-perfect rate 等权平均
```

## 14.2 表示排序标签

按下列顺序判断。

### REP_SUCCESSOR_ADVANTAGE

满足：

```text
D_core(B2) - max(D_core(QMARK), D_core(ASNET)) >= 0.15
```

并且 B2 在 A1、A2 两个切片均至少领先 0.10。

解释：

> 显式后继差分在非同构且感受野内出现额外价值。

### REP_QUERY_ADVANTAGE

满足：

```text
D_core(QMARK) - max(D_core(B2), D_core(ASNET)) >= 0.15
```

并且 QMARK 在 A1、A2 两个切片均至少领先 0.10。

解释：

> 查询传播在当前范围内优于显式后继和直接动作读出。

### REP_CANDIDATE_CONDITIONING_NEEDED

满足：

```text
abs(D_core(B2) - D_core(QMARK)) <= 0.10
min(D_core(B2), D_core(QMARK)) - D_core(ASNET) >= 0.15
```

解释：

> 直接动作节点读出不足，候选查询或候选后继条件化具有额外作用。

### REP_DIRECT_READOUT_SUFFICIENT

满足：

```text
D_core(ASNET) >= max(D_core(B2), D_core(QMARK)) - 0.10
```

且 ASNET 在 A1、A2 任一切片均不落后最佳方法超过 0.15。

解释：

> 在当前范围内，直接读取关系传播后的动作节点已经足够；B2/QMARK 的优势属于更广的关系表示能力。

### REP_MIXED

以上均不满足。

解释：

> 排序依赖切片或错误类型，暂不选择主表示。

## 14.3 失败范围标签

### LIMIT_COORDINATION

在 A2 的 `coordination_slice` 且 `hop <= 4` 中：

```text
D(B2) < 0.50
D(QMARK) < 0.50
D(ASNET) < 0.50
```

并且至少 60% 首次偏离属于：

- 破坏已完成目标；
- 错误目标顺序；
- 多目标循环。

解释：

> 出现值得设计新算法的多目标协调信号。

### LIMIT_RECEPTIVE_FIELD

每种方法均满足：

```text
D(hop <= 4) >= 0.70
D(hop <= 4) - D(hop > 4) >= 0.25
```

解释：

> 主要限制来自四层传播范围；不单独作为新算法贡献。

### LIMIT_TRAINING

ID Gate 未通过。

### LIMIT_NONE

没有明显共同失败。

### LIMIT_MIXED

失败来源不同或证据不足。

---

# 15. 主线决策映射

| 表示标签 / 限制标签 | 下一步 |
|---|---|
| `REP_SUCCESSOR_ADVANTAGE` | 以显式后继为算法主线，研究其在多步组合中的额外作用 |
| `REP_QUERY_ADVANTAGE` | 以查询传播为主要实现候选，收紧相对 ASNet-style 的差异 |
| `REP_CANDIDATE_CONDITIONING_NEEDED` | 发展统一的候选条件化关系计算，不急于在 B2/QMARK 二选一 |
| `REP_DIRECT_READOUT_SUFFICIENT` | 不包装 QMARK 为新算法；论文走统一解释、诊断协议与条件比较 |
| `LIMIT_COORDINATION` | 允许设计针对目标保持/顺序的最小新算法 |
| `LIMIT_RECEPTIVE_FIELD` | 只在研究范围需要时增加层数或循环传播；不称为核心创新 |
| `LIMIT_TRAINING` | 先决定是否统一改用 planner imitation，不解释 OOD |
| `REP_MIXED` / `LIMIT_MIXED` | 不扩矩阵，先按已有日志定位一个最小可判问题 |

---

# 16. 执行阶段

## E0：实现与冻结，不训练

### E0.1 Blocksworld 逻辑

建议新增：

```text
src/cp_disr/blocksworld/
  contracts.py
  state.py
  generator.py
  canonical.py
  planner.py
  environment.py
  snapshot.py
  metrics.py
```

### E0.2 方法

建议新增：

```text
src/cp_disr/c1_blocksworld_policies.py
```

包含：

- B2-CACHED；
- QMARK task adapter；
- ASNET-READOUT。

### E0.3 构建脚本

```text
scripts/c1_bw_build.py
scripts/c1_bw_train.py
scripts/c1_bw_eval.py
scripts/c1_bw_results.py
```

### E0.4 测试

```text
tests/test_c1_bw_contracts.py
tests/test_c1_bw_planner.py
tests/test_c1_bw_canonical.py
tests/test_c1_bw_splits.py
tests/test_c1_bw_policies.py
tests/test_c1_bw_metrics.py
```

### E0.5 冻结文件

`prep/` 至少包含：

```text
task_spec.json
contract_registry.json
train_dev_split.json
a0_iso_split.json
a1_color_reverse_split.json
a2_noniso_split.json
b_scale_split.json
canonical_audit.json
solvability_audit.json
hop_audit.json
coordination_audit.json
candidate_feature_audit.json
method_identity.json
training_plan.json
outcome_rules.json
source_identity.json
verify_prep.json
```

prep commit 后，split、规则和方法实现冻结。

## E1：三条训练

并行启动：

```text
R-C1-BW-B2-0
R-C1-BW-QMARK-0
R-C1-BW-ASNET-0
```

训练期间禁止修改 tracked source。

## E2：ID Gate

三条结束后统一检查。

若失败：

```text
PPO_ID_GATE_FAIL
```

停止。

若通过，进入 E3。

## E3：一次性正式评估

依次执行 A0/A1/A2/B 和 planner。

每个模型×case 只运行一次。

## E4：汇总与停止

生成：

```text
representation_label
limitation_label
mainline_recommendation
```

状态：

```text
C1_CROSS_TASK_MAIN_EXPERIMENT_COMPLETE
```

下一动作：

```text
WAIT_FOR_ALGORITHM_AND_PAPER_MAINLINE_DECISION
```

不得自动开发新算法。

---

# 17. 必须通过的测试和反向测试

## 17.1 合同正确性

- 每个动作前提正确；
- ADD/DEL 正确；
- 状态不变量保持；
- 非法动作 mask=false；
- planner 与环境使用同一合同。

## 17.2 规划器正确性

- n<=5 时 A* 最短长度与 BFS 一致；
- optimal_action_set 完整；
- tie-break 不影响集合；
- 每个 split case 可解；
- planner 轨迹最终满足目标。

## 17.3 同构

- 同色对象改名不改变 canonical hash；
- 颜色反转改变 color-preserving hash；
- A2 goal shape 与 train 非同构；
- A0 只作为正对照；
- test 与 train 无未授权重叠。

## 17.4 候选身份泄漏

以下任一恢复时测试必须失败：

```text
bound object hash
grounded action ID hash
case ID
split ID
goal hash
```

## 17.5 B2 缓存等价

- forward 一致；
- gradients 一致；
- nominal_apply 次数 K；
- encoder 次数 1+K。

## 17.6 QMARK

- nominal_apply trap 下仍可 forward；
- facts 字节不变；
- 只有 queried action 获得 query channel；
- 共享 query projection；
- encoder 次数 1+K。

## 17.7 ASNET-READOUT

- 无 query marker；
- 无 nominal_apply；
- 一次 encoder；
- 候选 row 来源于对应 ACTION 节点；
- 不读取 planner 标签。

## 17.8 测试隔离

训练 loader 若尝试打开 A0/A1/A2/B，测试必须失败。

## 17.9 指标

构造已知最优和已知偏离轨迹，验证：

- first divergence；
- decision perfect；
- goal destruction；
- excess steps；
- hop；
- coordination label。

---

# 18. 停止条件

以下任何一项发生即停止，不继续训练：

1. 精确同构检查无法实现；
2. A2 无法构造足够非同构目标；
3. 颜色属性只能通过名字哈希进入；
4. planner 无法稳定求解正式 case；
5. B2-CACHED 无法与 B2 保持 forward/gradient 等价；
6. ASNET-READOUT 必须使用额外启发式才可运行；
7. 三方法无法共享同一训练入口和任务数据；
8. 需要修改 reward 才能让某一方法单独学会；
9. 需要看测试结果后调整 split；
10. 正式评估文件被训练 loader 打开；
11. NaN 或 source identity 漂移；
12. 任一方法发生技术错误后想要无授权重跑。

---

# 19. 结果文件要求

`results/` 至少包含：

```text
final_summary.md
claim_boundary.md
representation_classification.json
limitation_classification.json
train_learning_table.csv
id_dev_by_method.csv
a0_by_case.csv
a1_by_case.csv
a2_by_case.csv
b_scale_by_case.csv
decision_trace_summary.json
hop_strata.csv
coordination_strata.csv
planner_summary.json
parameter_compute_identity.json
split_identity.json
source_identity.json
verify.json
```

每个 run 保存：

```text
resolved_config.yaml
training_accounting.json
checkpoint metadata
eval_id_final.json
eval_a0_final.json
eval_a1_final.json
eval_a2_final.json
eval_b_final.json
decision_log.jsonl
```

权重可保存在服务器或外部制品目录，但必须记录：

```text
absolute path
SHA-256
size
source commit
config hash
```

---

# 20. 允许与禁止的论文主张

## 20.1 在本卡运行前已允许

- T_B 中，QMARK 不写入后继事实，也能在决策层追平 B2；
- 旧 T_B 的三个主要 Fresh cell 是对象置换镜像；
- +E 在旧结构下无法候选级识别目标绑定；
- ABS 存在重复合法空操作 OPEN 的失败模式；
- 旧 B_PLAN 是强参照。

## 20.2 本卡成功后可能允许

取决于结果：

- 候选到目标的关系距离是否解释迁移；
- 直接动作节点读出是否已足够；
- 查询/后继条件化是否有额外价值；
- 显式后继是否在非同构目标中产生优势；
- 四层传播在哪个距离开始失效；
- 多目标干扰是否构成共同失败；
- 机器人 T_B 结果与纯符号 Blocksworld 结果如何互补。

## 20.3 当前仍禁止

- “首次提出关系图策略”；
- “所有已有方法都不能组合泛化”；
- “QMARK 是全新算法”；
- “学习方法优于精确规划器”；
- “未见动作”；
- “Fresh Confirm 已证明一般性组合泛化”；
- “hop>4 失败证明需要新方法”；
- “一个 seed 的 episode 数就是独立训练样本数”；
- “ASNET-READOUT 是原 ASNet 完整复现”。

---

# 21. Git 提交与停止

建议提交顺序：

```text
1. research: add frozen C1 Blocksworld cross-task specification
2. method: add B2 cache, QMARK adapter and ASNet-style readout
3. research: freeze C1 Blocksworld splits and registration
4. research: record C1 Blocksworld cross-task results
```

规则：

- 不创建 PR；
- 不 merge；
- 不 force-push；
- 不修改旧结果分支；
- 每次训练前推送 prep/freeze commit；
- 结果只在全部正式评估完成后提交；
- 最终确认本地与远端 HEAD 一致；
- 工作树干净；
- 无训练、评估、monitor 进程；
- 停止。

---

# 22. 最终回执模板

```yaml
card: CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1
repository: rollinpioneer/graph
base_commit: 1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e
branch: codex/cp-disr-c1-blocksworld-main-v1
prep_commit: <full sha>
result_commit: <full sha>

state: C1_CROSS_TASK_MAIN_EXPERIMENT_COMPLETE
next_action: WAIT_FOR_ALGORITHM_AND_PAPER_MAINLINE_DECISION

old_results_modified: false
training_runs:
  planned: 3
  completed: <n>
  rerun: 0

id_gate:
  B2: <PASS/FAIL>
  QMARK: <PASS/FAIL>
  ASNET_READOUT: <PASS/FAIL>

a0_equivariance_gate: <PASS/FAIL>

representation_label:
  <REP_SUCCESSOR_ADVANTAGE |
   REP_QUERY_ADVANTAGE |
   REP_CANDIDATE_CONDITIONING_NEEDED |
   REP_DIRECT_READOUT_SUFFICIENT |
   REP_MIXED>

limitation_label:
  <LIMIT_COORDINATION |
   LIMIT_RECEPTIVE_FIELD |
   LIMIT_TRAINING |
   LIMIT_NONE |
   LIMIT_MIXED>

main_results:
  id_dev: {}
  A0_iso: {}
  A1_color_reverse: {}
  A2_nonisomorphic: {}
  B_scale: {}

planner:
  solved: <n/total>
  mean_nodes: <value>
  mean_cpu_ms: <value>

compute:
  parameters: {}
  encoder_forwards: {}
  forward_time: {}

claim_boundary:
  allowed: []
  prohibited: []

next_recommendation:
  <text>
```

---

# 23. 当前授权状态

本文档本身不授权执行。

```yaml
authorized: false
training_authorized: false
formal_eval_authorized: false
external_provider_authorized: false
```

用户明确授权本卡后，才可从 E0 开始。

不得仅凭本文档存在而启动训练。
