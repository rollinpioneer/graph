---
title: "CP-DISR C1 Blocksworld Amendment 02：三方法统一规划器模仿训练"
date: "2026-10-06"
lang: "zh-CN"
amendment_id: "CP-DISR-C1-BW-IMITATION-A02"
parent_card: "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
decision: "AUTHORIZE_OPTION_A"
status: "AUTHORIZED"
repository: "rollinpioneer/graph"
branch: "codex/cp-disr-c1-blocksworld-main-v1"
base_commit: "72c0fd2cdc4394dcdee3b5bd6bba210060535d57"
ppo_result_state: "PPO_ID_GATE_FAIL"
new_training_run_cap: 3
---

# Amendment 02：三方法统一规划器模仿训练

## 0. 决定

正式选择 **A**：

> 对 B2-CACHED、QMARK、ASNET-READOUT 统一进行一次规划器模仿训练；只有三种方法全部通过原有 ID gate 后，才执行一次冻结的 A0/A1/A2/B 正式评估和规划器基线。

不选择 B，因为当前停止只说明：

> 在固定 PPO 与稀疏终端奖励下，ASNET-READOUT 最终仍能完成 34/36 个 ID 任务，但其决策最优性从中途的 33/36 下降到最终 26/36。

这没有回答第二任务族的核心表示问题。

不选择 C，因为只对 ASNET 放宽决策完美门，会在看到结果后对单一方法降低标准，使成功率和决策质量被不对称地混用。

本 Amendment 不改变已经发生的 PPO 结果。PPO 结果永久保留为：

```text
PPO_ID_GATE_FAIL
```

规划器模仿是一个新的、三方法统一的训练阶段，不是对旧 PPO run 的重跑或补丁。

---

# 1. 决策依据

## 1.1 已完成的 PPO 结果

| 方法 | Final ID success | Final ID decision-perfect | Gate |
|---|---:|---:|---|
| B2-CACHED | 34/36 | 34/36 | PASS |
| QMARK | 35/36 | 35/36 | PASS |
| ASNET-READOUT | 34/36 | 26/36 | FAIL |

ASNET-READOUT 的主要现象：

- 34/36 能到达目标；
- 只有 26/36 全程选择最优动作；
- 10 个 episode 出现重复 `(state, action)` 循环；
- 8 个 episode 成功但平均多走 3.5 步；
- decision-perfect 在 `N=32768` 达到 33，随后下降至 31，最终为 26；
- 不允许选择中途 checkpoint 代替 final checkpoint。

因此当前证据更接近：

> PPO 训练目标与“始终选择最短计划中的动作”不完全一致，且这种偏差在 ASNET-READOUT 上更明显。

它尚不能说明：

> ASNET-READOUT 的表示本身无法完成 A0/A1/A2/B 的迁移。

## 1.2 为什么必须三种方法统一切换

本阶段研究问题是表示，而不是探索或 PPO 稳定性。

如果只对 ASNET 使用规划器监督：

- ASNET 获得更强训练信号；
- B2/QMARK 保留 PPO；
- 后续 OOD 排名同时混入表示差异与训练协议差异。

因此三种方法必须：

- 从头训练；
- 使用相同 planner-labeled 数据；
- 使用相同模仿损失；
- 使用相同数据聚合轮次；
- 使用相同训练预算和 final-checkpoint 规则。

---

# 2. Git 与证据保护

## 2.1 仓库与分支

```text
repository:
rollinpioneer/graph

branch:
codex/cp-disr-c1-blocksworld-main-v1

base commit:
72c0fd2cdc4394dcdee3b5bd6bba210060535d57
```

开始前确认：

```bash
git rev-parse HEAD
git branch --show-current
git status --short
```

必须满足：

```text
HEAD == 72c0fd2cdc4394dcdee3b5bd6bba210060535d57
branch == codex/cp-disr-c1-blocksworld-main-v1
working tree clean
```

## 2.2 PPO 结果只读

下列目录永久保持字节不变：

```text
runs/final_master/c1_route_b/blocksworld_main_v1/
20261006T015313Z_490a3b11/
```

特别是：

```text
results/final_summary.md
results/claim_boundary.md
results/checkpoint_identity.json
runs/
eval/
```

不得：

- 覆盖 PPO checkpoint identity；
- 将 imitation 结果写入旧 `results/`；
- 将旧 state 从 `PPO_ID_GATE_FAIL` 改成 PASS；
- 删除 ASNET 的循环或退化记录；
- 把 PPO 与 imitation 数字拼成一个同训练协议排行榜。

## 2.3 新输出目录

```text
runs/final_master/c1_route_b/blocksworld_main_v1/
imitation_a02/<UTC>_<prep_sha8>/
```

建议结构：

```text
prep/
datasets/
runs/
eval/
results/
receipts/
```

---

# 3. 允许复用与禁止变更

## 3.1 允许复用

直接复用冻结内容：

- Blocksworld 合同；
- train144 / dev36；
- A0 / A1 / A2 / B split；
- exact planner；
- optimal-action-set 计算；
- B2-CACHED；
- QMARK；
- ASNET-READOUT；
- candidate features；
- base input；
- A01 hop 收口；
- 结果分类规则；
- 正式评估入口。

复用前重新核验所有 hash 与 freeze commit 一致。

## 3.2 禁止变更

不得修改：

- 三种方法的表示定义；
- 图节点和消息边；
- 颜色属性编码；
- reward 或正式评估指标；
- train/dev/test split；
- ID gate；
- A0/A1/A2/B 分类；
- planner tie-break；
- step cap；
- 目标干扰标签；
- 候选身份泄漏保护；
- A01 的 `NOT_TESTABLE_STRUCTURALLY_EMPTY` 状态。

不得增加：

- 新 seed；
- 第四种学习方法；
- 更深网络；
- 额外记忆模块；
- 额外启发式；
- planner distance 输入；
- LM-cut 特征；
- 测试集监督；
- 新算法。

---

# 4. 新训练矩阵

只授权三条新训练：

| Run ID | Method | Init seed |
|---|---|---:|
| `R-C1-BW-IL-B2-0` | B2-CACHED | 0 |
| `R-C1-BW-IL-QMARK-0` | QMARK | 0 |
| `R-C1-BW-IL-ASNET-0` | ASNET-READOUT | 0 |

规则：

- 全部从头初始化；
- 不加载 PPO checkpoint；
- 共享模块使用相同初始化 seed；
- 架构特有参数使用确定性派生 seed；
- 不重跑；
- 不挑 checkpoint；
- 不按 dev 结果调超参数。

总新增训练：

```text
3
```

---

# 5. 规划器监督定义

## 5.1 标签

对每个训练状态 `s` 和目标 `g`，exact planner 给出：

```text
A*(s,g)
```

其中 `A*` 是所有满足下式的合法动作集合：

\[
d^*(T(s,a),g)=d^*(s,g)-1
\]

即所有能保持最短计划长度的动作，而不是任意选择一个 planner tie-break 动作。

终止状态不产生动作损失。

## 5.2 模仿损失

对策略概率 `π(a|s)`：

\[
L_{\mathrm{IL}}
=
-\log \sum_{a\in A^*(s,g)} \pi(a|s)
\]

要求：

- 非法动作保持 mask；
- 多个最优动作全部视为正确；
- 不使用 value target；
- 不使用 Q target；
- 不使用 reward；
- 不使用 entropy bonus；
- 不使用 PPO clipping；
- 不使用 planner distance 作为输入。

训练参数仅包括会影响 policy logits 的模块。

`v_head` 与 `q_head`：

```text
frozen
```

正式评价仍可忽略它们，不得用未训练的 V/Q 解释结果。

---

# 6. 共享状态聚合协议

为了避免只在专家轨迹上训练而在自己访问的状态中循环，本卡使用固定的两轮共享状态聚合。它对三种方法完全相同，不是方法贡献。

## 6.1 D0：专家轨迹

对 train144 的每个 case：

1. 从冻结初始状态开始；
2. 由 exact planner 按固定字典序选择一个最优动作；
3. 执行至成功；
4. 每一步保存：
   - 当前状态；
   - 当前目标；
   - legal mask；
   - 完整 `A*`；
   - chosen canonical expert action；
   - 最短剩余长度；
   - case ID；
   - trajectory ID。

D0 仅来自 train144。

不得打开：

```text
dev36
A0
A1
A2
B
```

## 6.2 Stage 0 训练

三种方法分别在完全相同的 D0 上训练。

固定：

```yaml
epochs: 100
optimizer: Adam
lr: 0.0003
betas: [0.9, 0.999]
eps: 1e-8
weight_decay: 0
grad_clip: 0.5
trajectory_batch_size: 32
shuffle_seed: 0
```

序列处理：

- 每条 trajectory 开始时 hidden=0；
- 在轨迹中按顺序更新 hidden；
- 不在每个 decision 重新清零；
- padding 决策不计入 loss；
- 每条 trajectory 在 epoch loss 中总权重相同；
- 不因轨迹更长而自动获得更大总权重。

Stage 0 结束使用最后一个 epoch，不选择最佳 epoch。

## 6.3 Aggregation Round 1

使用 Stage 0 final checkpoint。

对每个方法、每个 train case：

1. deterministic argmax rollout；
2. 最多运行冻结 step cap；
3. 保存该方法实际访问的完整轨迹；
4. exact planner 为每个访问状态标注完整 `A*`。

构造共享数据：

```text
D1 = D0
   + B2 访问轨迹
   + QMARK 访问轨迹
   + ASNET 访问轨迹
```

三种方法使用完全相同的 D1。

源平衡：

- 每个 `source_method × case` trajectory 总权重相同；
- 循环导致的长轨迹不得因决策数更多而占更大总权重；
- 相同状态允许来自不同历史轨迹重复出现，因为 GRU hidden 可能不同；
- 不使用 dev 或测试状态。

继续训练：

```yaml
epochs: 50
optimizer_state: continue
shuffle_seed: 1
```

使用最后一个 epoch。

## 6.4 Aggregation Round 2

使用 Round 1 final checkpoint，重复一次训练集 rollout 与 planner 标注：

```text
D2 = D1
   + round2 B2 trajectories
   + round2 QMARK trajectories
   + round2 ASNET trajectories
```

三种方法使用完全相同的 D2。

继续训练：

```yaml
epochs: 50
optimizer_state: continue
shuffle_seed: 2
```

Round 2 最后一个 epoch 是唯一 final imitation checkpoint。

## 6.5 禁止自适应

不得根据：

- train loss；
- ID dev；
- ASNET 是否已过门；
- 中途 checkpoint；
- 方法间排序；

增加轮次、epoch 或学习率调整。

固定总计：

```text
100 + 50 + 50 epochs
2 aggregation rounds
```

---

# 7. 训练完整性与收据

每个阶段记录：

```text
dataset hash
trajectory count
decision count
unique (goal,state) count
source-method counts
mean / max trajectory length
cycle-containing trajectory count
planner-label coverage
epoch loss
optimal-action probability mass
argmax-in-optimal-set rate
gradient norm
NaN count
checkpoint SHA-256
```

共享数据必须验证：

```text
D0/D1/D2 hashes identical across the three method runs
```

方法只允许在模型参数、前向结果和优化状态上不同。

训练完成后输出：

```text
imitation_training_accounting.json
imitation_dataset_identity.json
imitation_checkpoint_identity.json
imitation_verify.json
```

---

# 8. ID Gate

三条训练全部完成后，才读取 dev36。

每个 final imitation checkpoint 在 dev36 上评价一次：

```text
success >= 33/36
decision_perfect >= 33/36
NaN = 0
hard failure = 0
```

使用原 gate，不放宽 ASNET。

## 8.1 全部通过

状态：

```text
IMITATION_ID_GATE_PASS
```

随后授权：

- A0；
- A1；
- A2；
- B；
- exact planner baseline。

每个模型×case 只评估一次。

## 8.2 任一失败

状态：

```text
IMITATION_ID_GATE_FAIL
```

立即停止。

不得：

- 再加 aggregation round；
- 增加 epoch；
- 挑中途 checkpoint；
- 单独修改失败方法；
- 自动切换其他监督；
- 打开 A0/A1/A2/B；
- 开发新算法。

此时下一动作：

```text
WAIT_FOR_USER_MAINLINE_DECISION
```

---

# 9. 正式评估规则

仅在 `IMITATION_ID_GATE_PASS` 后执行。

## 9.1 使用的模型

只使用：

```text
三种 final imitation checkpoints
```

不得将 PPO checkpoint 加入主 OOD 表。

PPO 结果单独作为训练协议观察：

> 在相同 PPO 下，ASNET-READOUT 成功率较高但最终决策最优性退化。

## 9.2 评估顺序

```text
A0
A1
A2
B
planner baseline
```

全部使用原冻结文件和原分类规则。

## 9.3 结果解释

主 OOD 比较回答：

> 在统一最优动作监督下，B2、QMARK 和 ASNET-READOUT 的表示分别如何迁移？

不能回答：

> PPO 下哪一种方法总体最好。

训练协议结论与表示结论分开：

### PPO 结论

```text
ASNET-READOUT under the frozen PPO objective:
high ID task success, unstable final decision optimality.
```

### Imitation 结论

```text
representation comparison under matched exact-planner action supervision.
```

---

# 10. 结果分类保持不变

继续使用冻结的：

```text
REP_SUCCESSOR_ADVANTAGE
REP_QUERY_ADVANTAGE
REP_CANDIDATE_CONDITIONING_NEEDED
REP_DIRECT_READOUT_SUFFICIENT
REP_MIXED
```

以及：

```text
LIMIT_COORDINATION
LIMIT_TRAINING
LIMIT_NONE
LIMIT_MIXED
```

固定：

```text
receptive_field_status =
NOT_TESTABLE_STRUCTURALLY_EMPTY
```

若 imitation ID gate 失败：

```text
limitation_label = LIMIT_TRAINING
```

不生成 REP 标签。

---

# 11. 允许与禁止的结论

## 11.1 当前 PPO 阶段允许

可以写：

- B2-CACHED 和 QMARK 通过最终 PPO ID gate；
- ASNET-READOUT 在相同 PPO 预算下达到 34/36 成功，但只有 26/36 决策完美；
- ASNET 的 decision-perfect 从中途 33 下降到 final 26；
- ASNET 出现成功绕路和状态—动作循环；
- 当前没有 OOD 结果。

## 11.2 当前禁止

不能写：

- ASNET 表示不如 B2/QMARK；
- ASNET 无法泛化；
- B2/QMARK 在 A0/A1/A2/B 更强；
- planner baseline 更强或更弱；
- 中途 ASNET checkpoint 已经证明该表示足够；
- PPO final 排名就是表示排名。

## 11.3 Imitation 正式评估后可能写

取决于冻结分类：

- 直接动作节点读出是否足够；
- 候选查询/后继条件化是否必要；
- 显式后继是否在非同构目标上产生额外价值；
- 多目标协调是否构成共同失败；
- 规模增长下各方法的条件性边界。

---

# 12. Git 提交顺序

建议：

```text
1. research: authorize uniform Blocksworld planner imitation
2. training: record uniform Blocksworld planner imitation
3. research: record Blocksworld cross-task evaluation
```

第一条 commit 必须在训练前完成并推送，至少包含：

- 本 Amendment；
- imitation trainer；
- dataset builder；
- unit tests；
- registration；
- frozen training config；
- source hashes。

规则：

- 继续使用当前分支；
- 不创建 PR；
- 不 merge；
- 不 force-push；
- 不修改旧 PPO 结果；
- 不把 checkpoint 二进制提交 Git；
- 记录 checkpoint path、size、SHA-256、source commit 和 config hash。

---

# 13. 测试要求

## 13.1 Planner labels

- `A*` 包含全部最优合法动作；
- 每个 `a∈A*` 都满足距离下降 1；
- 非最优动作不得进入标签；
- 多解 case 不得只保留字典序动作。

## 13.2 Loss

构造两个最优动作的 fixture：

```text
π(a1)+π(a2)
```

必须作为正确概率质量计算，不能把另一个最优动作当负类。

## 13.3 Sequence

- hidden 只在 trajectory 起点清零；
- padding 不参与 loss；
- trajectory-normalized weighting 正确；
- 三方法读取同一轨迹序列。

## 13.4 Shared aggregation

- 每轮三个 source method 都有 144 条 rollout；
- step cap 冻结；
- D1/D2 对三种训练完全相同；
- eval split loader 在训练期间不可调用。

## 13.5 Model boundaries

- B2-CACHED 保持原等价；
- QMARK 不调用 nominal_apply；
- ASNET-READOUT 不使用 query marker 或 successor；
- planner labels 不进入模型输入；
- V/Q heads 不参与训练。

## 13.6 Negative tests

以下任一发生，测试必须失败：

- 打开 A0/A1/A2/B；
- 加载 PPO checkpoint；
- 选择 ASNET `N=32768`；
- 只为 ASNET 加 planner supervision；
- 对不同方法使用不同聚合数据；
- 用单一 tie-break 动作替代完整 `A*`；
- 因循环轨迹更长而增加其总权重；
- 根据 dev 调 epoch 或轮次。

---

# 14. 授权

```yaml
authorized: true
decision: AUTHORIZE_OPTION_A
resume_from: PLANNER_IMITATION_PREP
base_commit: 72c0fd2cdc4394dcdee3b5bd6bba210060535d57
branch: codex/cp-disr-c1-blocksworld-main-v1

new_training_runs:
  - R-C1-BW-IL-B2-0
  - R-C1-BW-IL-QMARK-0
  - R-C1-BW-IL-ASNET-0

new_training_run_cap: 3
aggregation_rounds: 2
formal_eval_authorized_conditionally: true
formal_eval_condition: ALL_THREE_PASS_IMITATION_ID_GATE
external_provider_authorized: false
new_algorithm_authorized: false
additional_seed_authorized: false
```

开始前 Agent 必须记录：

```text
AMENDMENT_A02_ACCEPTED
PPO_RESULT_STATE=PPO_ID_GATE_FAIL
NEW_TRAINING_PROTOCOL=UNIFORM_PLANNER_IMITATION
AUTHORIZED_TRAINING_RUNS=3
FORMAL_EVAL_CONDITION=ALL_THREE_PASS_IMITATION_ID_GATE
```

---

# 15. 最终回执模板

```yaml
card: CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1
amendment: CP-DISR-C1-BW-IMITATION-A02
repository: rollinpioneer/graph
branch: codex/cp-disr-c1-blocksworld-main-v1
base_commit: 72c0fd2cdc4394dcdee3b5bd6bba210060535d57

prep_commit: <sha>
result_commit: <sha>

ppo_state_preserved: PPO_ID_GATE_FAIL
old_results_modified: false

imitation:
  runs_planned: 3
  runs_completed: <n>
  aggregation_rounds: 2
  shared_dataset_hashes:
    D0: <sha>
    D1: <sha>
    D2: <sha>
  id_gate:
    B2: <PASS/FAIL>
    QMARK: <PASS/FAIL>
    ASNET_READOUT: <PASS/FAIL>

formal_evaluation:
  executed: <true/false>
  condition_met: <true/false>
  A0: {}
  A1: {}
  A2: {}
  B: {}
  planner: {}

representation_label: <label or NOT_ISSUED>
limitation_label: <label>
receptive_field_status: NOT_TESTABLE_STRUCTURALLY_EMPTY

state:
  <IMITATION_ID_GATE_FAIL |
   C1_CROSS_TASK_MAIN_EXPERIMENT_COMPLETE>

next_action:
  WAIT_FOR_USER_MAINLINE_DECISION
```
