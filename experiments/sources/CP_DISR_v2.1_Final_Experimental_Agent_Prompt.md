# Prompt：为 CP-DISR / M1 v2.1 制定最终版分阶段实验执行计划

请基于当前已经冻结的 **CP-DISR / M1 v2.1** 研究规范与此前全部讨论，生成一份最终版：

# CP-DISR v2.1 Minimal Experimental Execution Plan

这不是普通的“实验建议”，而是一份可以长期作为后续实验 Agent 工作手册使用的 **分阶段实验执行规范**。

最终文档必须详细到：以后我只需要对 Agent 说：

> **“执行 Stage 1A”**  
> **“执行 Stage 2B”**  
> **“执行 Part IV”**

Agent 就能够仅依赖这份文档完成对应阶段，不需要我再次解释：

- 当前阶段要解决什么；
- 前置条件是什么；
- 需要读取哪些文件；
- 用哪个方法版本；
- 用哪些任务；
- 用哪些 seed；
- 用哪些模型；
- 用哪些参数；
- 哪些参数锁死；
- 哪些参数可调；
- 跑哪些配置；
- 训练多少；
- 多久评估一次；
- 每次评估多少 episode；
- 保存什么日志；
- 输出到什么目录；
- 画哪些图；
- 生成哪些表；
- 什么情况 PASS；
- 什么情况 STOP；
- 什么情况需要重跑；
- 如果结果不好，按什么顺序排查/调节；
- 阶段完成后生成什么报告；
- 下一阶段是什么。

文档最终必须成为：

> **“我说执行 Stage XX，Agent 就能直接开工”的实验总规范。**

---

# 0. 总目标与最高优先级原则

## 0.1 当前论文性质

当前这篇论文首先是一篇：

> **供我自己阅读、整理研究思路、完整呈现研究过程和欣赏的私人论文。**

当前目标不是：

- 正式匿名投稿；
- 毕业审核；
- 科研考核；
- 对外作为严格科学证据；
- 一开始就构建投稿级完整统计包。

因此，本轮实验优先级是：

1. 尽快把 CP-DISR v2.1 的完整 pipeline 跑起来；
2. 实验量尽量控制；
3. 尽快看到基本现象；
4. 尽快获得能填入论文的真实结果；
5. 实验与论文写作并行；
6. 不因为小幅结果不理想反复重开方法设计；
7. 尽量通过合理的参数调整、任务筛选、checkpoint 选择、seed 选择和结果组织，让最终私人论文完整、清晰、好看。

## 0.2 如果旧实验规范与本 Prompt 冲突

如果此前 CP-DISR v2.1 研究规范中的正式投稿式实验要求，与本 Prompt 的“私人论文实验目标”冲突：

> **以本 Prompt 作为当前实验执行最高优先级。**

尤其不要机械恢复下列约束：

- 所有尝试过的 seed 都必须进入主文；
- 所有 checkpoint 必须统一；
- 所有任务都必须展示；
- 所有方法必须完全共享所有优化参数；
- 所有参数必须在第一次实验前永久锁死；
- 看过开发结果以后绝对不能调参；
- 必须完成大量 seed 才能继续写论文；
- 所有失败运行必须进入主结果；
- 所有模块必须做完整统计显著性验证。

这些不是当前私人版本的主要目标。

## 0.3 当前允许的实验调整空间

在不凭空制造实验结果的前提下，允许：

- 根据探索结果调参；
- 重跑实验；
- 延长或缩短训练；
- 选择更合适的 checkpoint；
- 增加新 seed；
- 从多个正常运行 seed 中选择更适合展示的 seed；
- 筛选更能体现 CP-DISR 机制的任务；
- 不把解释价值很低的任务放进主文；
- 选择更有解释力的 metric；
- 使用合理 moving average / EMA；
- 合理选择坐标范围；
- 展示 hard-task / structured-task 子集；
- 将异常或次要运行留在 Raw Archive；
- 选择最清楚的 qualitative episode；
- 根据真实现象收缩论文 claim；
- 对不同方法进行少量、合理、对其训练稳定性有必要的参数适配。

允许的思路是：

> run → 看结果 → 做有限调整 → 重跑 → 形成更清晰的真实结果。

## 0.4 不直接伪造实验数据

虽然当前是私人论文，但仍不要：

- 手工把 0.61 改成 0.75；
- 修改原始 reward；
- 修改原始 success；
- 手工改 episode 结果；
- 编造从未运行过的 baseline；
- 伪造不存在的日志；
- 修改原始 CSV/JSONL 使其符合故事。

如果结果不好：

优先通过：

> 实现排查 → 合理调参 → 重跑 → checkpoint 选择 → seed 选择 → task 选择 → metric 选择 → 合理平滑与展示

获得更适合私人论文的真实结果。

## 0.5 Raw Archive 与 Private Paper View 分离

必须维护两套视图。

### Raw Experiment Archive

完整保存：

- 所有运行；
- 所有 seed；
- 所有 checkpoint；
- 所有 config；
- 所有失败运行；
- 所有 VLM cache；
- 原始日志；
- 环境版本；
- Git commit；
- 软件版本；
- 失败原因。

### Private Paper View

从 Raw Archive 中选择：

- 最适合主故事的 task；
- 正常、代表性或图表清楚的 seed；
- 更合适的 checkpoint；
- 更清楚的 metric；
- 更适合解释机制的 case；
- 更适合论文展示的曲线。

最终所有主文结果仍必须能够通过 selection manifest 追溯到 Raw Archive 中的真实运行。

---

# 1. 当前方法冻结状态

当前方法版本：

> **CP-DISR / M1 v2.1**

除非后续发现真正的结构性错误，否则实验阶段不要继续重设主体 Method。

## 1.1 VLM

采用：

**Frozen Strong API VLM**

原则：

- 不训练；
- 不 LoRA；
- 不 SFT；
- 不 teacher–student；
- 不做多 VLM ensemble；
- API 价格不是当前主要限制。

VLM 只在：

> Task + Initial Scene

阶段生成 soft relations。

生成后：

> offline cache

PPO 训练过程不反复调用 VLM。

## 1.2 Relation Schema

固定：

`m1_soft_relations_v2`

只允许：

### SOFT_SUPPORTS

Action A → Action B

语义：

> A 的某项注册名义效果可能在当前场景中支持 B 后续执行。

不代表：

- hard prerequisite；
- 必然成功；
- 必须按 A→B 顺序执行。

### SOFT_RELEVANT_TO_GOAL

Action A → Goal Proposition

语义：

> A 的某项注册名义效果可能影响该目标的实现、保持或恢复。

不代表：

- 静态相关性分数；
- 一定正向；
- VLM 直接指定下一动作。

每条 relation 必须包含：

`effect_fact_ref`

且该 fact 必须属于 source action 的已注册 ADD / DEL effect。

已经删除：

`SOFT_ORDER`

实验中不要重新加入。

## 1.3 Skill Contract

Skill Contract 固定为：

- skill schema；
- argument types；
- initiation / preconditions；
- nominal ADD；
- nominal DEL；
- limited unknown effects；
- termination；
- failure；
- verification；
- duration/time limits；
- provenance。

来源：

> controller interface + 研究人员一次性补全 + 执行记录校验。

不由 VLM 生成可信合同。

## 1.4 Contract Graph

采用：

**grounded Action–Proposition Graph**

Node：

- Action；
- Proposition。

Contract relation：

- PRE_POS；
- PRE_NEG；
- ADD；
- DEL；
- 对应 reverse message edges。

不单独加入：

- Object Node；
- Goal Node；
- 普通 Logic Node。

Goal 使用：

- proposition goal sign；
- fixed goal_refs。

事实状态：

- TRUE；
- FALSE；
- UNKNOWN。

UNKNOWN 不能自动当 FALSE。

## 1.5 Graph Encoder

使用：

**标准共享 R-GCN**

当前默认：

- hidden dim = 128；
- layers = 4；
- num bases = 4；
- aggregation = mean；
- root transform = true；
- ReLU；
- per-node LayerNorm；
- dropout = 0。

四路图：

- \(G^K\)
- \(G^H\)
- \(	ilde G^{K,i}\)
- \(	ilde G^{H,i}\)

共享同一个 encoder。

## 1.6 Candidate Nominal Intervention

对每个当前可执行候选 skill：

假设：

> 该 skill 按 Skill Contract 的 nominal success branch 成功完成。

生成临时事实 patch。

例如：

OPEN(box)

可能：

`Open(box): FALSE → TRUE`

但这个 patch：

不能修改：

- 真实环境；
- 真实 Fact Store；
- 真实 observation；
- reward；
- 实际时间；
- attempt count。

它只是：

> 一步 symbolic nominal success intervention。

## 1.7 双差分

固定：

\[
D_i^K
=
E(	ilde G_i^K)-E(G^K)
\]

\[
D_i^H
=
E(	ilde G_i^H)-E(G^H)
\]

\[
D_i^P
=
D_i^H-D_i^K
\]

解释：

### \(D^K\)

合同结构中：

candidate nominal effect 导致的结构表示变化。

### \(D^P\)

soft prior 如何改变对同一 candidate-induced structural change 的编码响应。

当前采用：

> **Pure Interaction Prior**

不加入：

- Static VLM score；
- Static prior branch；
- VLM action ranking；
- 当前增强图直接 bias。

## 1.8 Actor

Actor：

### Contract branch

产生：

\[
b_i
\]

### zero-anchored bounded prior branch

使用：

\[
D_i^P
\]

产生：

\[
\Delta_i
\]

最终：

\[
\ell_i=b_i+\Delta_i
\]

要求：

\[
D_i^P=0
\Rightarrow
\Delta_i=0
\]

默认：

\[
B=0.5
\]

## 1.9 Structural Q

Structural Q 只监督真正执行的 action。

目标：

\[
y_t^Q
=
r_t+\Gamma_tV_{	ext{old}}(x_{t+1})
\]

只对：

\(a_t\)

计算真实 transition target。

不得给未执行候选伪造 Q 标签。

其作用：

> 通过真实执行结果给共享结构表示提供动作条件化价值压力。

## 1.10 PPO

采用：

**skill-level duration-aware PPO**

默认：

- lr = `3e-4`
- PPO clip = `0.2`
- GAE lambda = `0.95`
- value coef = `0.5`
- Structural Q coef \(\lambda_Q\) = `0.1`
- entropy coef = `0.01`
- grad clip = `0.5`
- PPO epochs = `4`
- rollout = `1024 skill transitions`
- recurrent sequence length = `16`
- minibatch = `64`
- optimizer = `Adam`

duration discount：

按 Semi-MDP 处理。

## 1.11 Prior Training

正式协议：

### 80%

Original VLM prior

### 20%

No-prior

按 episode 抽样。

episode 内固定。

训练阶段不做：

- edge deletion；
- target swap；
- artificial wrong relation；
- irrelevant edge；
- relation rewrite。

这些只用于测试 robustness。

---

# 2. 实验总组织方式

实验必须使用：

# Part → Stage

结构。

最终至少包含：

# Part I — Infrastructure & Method Validation

### Stage 0A — Environment & Manifest Freeze
### Stage 0B — Unit & Invariant Tests
### Stage 0C — VLM Cache Sanity Check

# Part II — Minimal Learning Validation

### Stage 1A — One-Task Smoke Run
### Stage 1B — Minimal Tuning（仅必要时）

# Part III — Core Paper Evidence

### Stage 2A — Core Exploration Runs
### Stage 2B — Presentation Runs
### Stage 2C — Main Figure & Table Generation

# Part IV — Core Mechanism Evidence

### Stage 3A — Dual-Difference Ablation
### Stage 3B — Structural Q Ablation
### Stage 3C — Bounded Prior Ablation

# Part V — Test-Time Mechanism & Robustness

### Stage 4A — Actor Dependence Diagnostics
### Stage 4B — Prior Robustness Evaluation

# Part VI — Minimal Generalization

### Stage 5A — Compositional Generalization

# Part VII — Private Paper Packaging

### Stage 6A — Raw Archive Review
### Stage 6B — Private Paper View Selection
### Stage 6C — Final Result Package

---

# 3. 每个 Stage 必须使用统一执行模板

这是最终计划最重要的格式要求。

每个 Stage 必须完整包含：

## Stage ID

例如：

`Stage 2A`

## Stage Name

## Purpose

只回答本阶段要解决的核心问题。

## Stage Type

从下列选择：

- Infrastructure
- Unit Test
- Cache Validation
- Smoke
- Exploration
- Presentation
- Ablation
- Mechanism
- Robustness
- Generalization
- Packaging

## 是否属于论文正式结果

明确：

YES / NO / OPTIONAL。

## 前置条件

列：

- 必须完成的 Stage；
- 必须存在的文件；
- 必须存在的 checkpoint；
- 必须存在的 cache。

## 输入

明确：

- method version；
- code version；
- git hash；
- task list；
- skill contracts；
- VLM cache；
- config；
- seed；
- checkpoint；
- dataset/split；
- hardware；
- software env。

## Frozen Parameters

本 Stage 禁止调整的参数。

## Tunable Parameters

本 Stage 允许调整的参数。

如果不允许：

写：

`None`

## Run Matrix

使用明确表格：

| Run Group | Method | Task | Seed | Budget | Prior | Notes |
|---|---|---|---:|---:|---|---|

不要只写“跑若干组合”。

## Training Budget

必须给具体默认值。

不能只写：

“训练到收敛”。

必须写：

- skill transitions；
- PPO updates；
- environment/simulator time；
- evaluation interval。

如果当前平台没有绑定：

给推荐默认值 + `MUST_BIND` 字段。

## Evaluation Protocol

必须明确：

- evaluation episodes；
- stochastic / deterministic；
- checkpoint interval；
- evaluation metric；
- evaluation seed；
- prior setting。

## Required Logging

根据阶段列出：

- success_rate
- episode_return
- task_success
- skill_count
- physical/simulated duration
- actor_entropy
- policy_loss
- value_loss
- q_loss
- DK_norm
- DH_norm
- DP_norm
- prior_residual
- candidate_count
- relation_count
- selected_skill
- failure_reason
- checkpoint_step
- run_id
- seed

## Output Directory

必须给目录模板。

例如：

```text
experiments/
  part_2_core/
    stage_2a/
      task_<name>/
        full/
          seed_0/
```

## Required Files

至少按需包含：

```text
config.yaml
manifest.json
train_metrics.csv
eval_metrics.csv
episode_log.jsonl
run_summary.md
checkpoints/
plots/
tables/
```

## Required Plots

明确：

- 文件名；
- x 轴；
- y 轴；
- smoothing；
- 是否 mean/median；
- 哪些方法叠加。

## Required Tables

明确：

- 表名；
- 列名；
- 聚合方式。

## PASS / STOP Criteria

标准必须宽松。

早期阶段只要求：

- pipeline 正常；
- 没有 NaN；
- reward/success 有学习迹象；
- D 不是全部失效；
- 没有明显实现 bug。

不要要求：

Full 显著优于 baseline。

## Failure Handling

给固定排查顺序。

## Rerun Allowed

YES / NO

## Tuning Allowed

明确：

哪些参数。

## 是否阻塞论文写作

一般：

NO。

如果阻塞某个结果章节：

具体说明。

## Next Stage

明确 Stage ID。

## Agent Deliverables

本阶段 Agent 必须提交的文件列表。

## Agent Execution Template

每个 Stage 最后必须给一句：

> 当用户说“执行 Stage XX”时，Agent严格执行以下步骤：

然后给 8–15 条实际执行步骤。

---

# 4. Agent 不得自动跨阶段

如果我说：

> 执行 Stage 1A

Agent只能执行 Stage 1A。

完成后：

- 输出本阶段报告；
- 更新 stage_status；
- 停止。

不能自动进入：

Stage 1B / Stage 2A。

只有我明确说：

> 继续执行 Stage 2A

才进入下一阶段。

---

# 5. 每个 Stage 必须生成状态文件

目录：

```text
stage_status/
```

例如：

```text
stage_0a.json
stage_0b.json
stage_0c.json
stage_1a.json
...
```

Schema 至少包含：

```json
{
  "stage": "2A",
  "status": "PASS",
  "method_version": "CP-DISR-v2.1",
  "git_hash": "...",
  "started_at": "...",
  "completed_at": "...",
  "runs_completed": [],
  "runs_failed": [],
  "selected_tasks": [],
  "selected_config": null,
  "issues": [],
  "tuning_changes": [],
  "next_stage": "2B"
}
```

允许状态：

- NOT_STARTED
- RUNNING
- PASS
- PASS_WITH_NOTES
- NEEDS_RERUN
- BLOCKED

---

# 6. 实验目录结构必须现在统一

请设计最终目录，例如：

```text
experiments/
├── manifests/
│   ├── method_manifest.yaml
│   ├── task_manifest.yaml
│   ├── software_manifest.yaml
│   └── presentation_manifest.yaml
│
├── stage_status/
│
├── vlm_cache/
│   ├── train/
│   ├── dev/
│   └── test/
│
├── part_0_validation/
│   ├── stage_0a/
│   ├── stage_0b/
│   └── stage_0c/
│
├── part_1_smoke/
│   ├── stage_1a/
│   └── stage_1b/
│
├── part_2_core/
│   ├── stage_2a/
│   ├── stage_2b/
│   └── stage_2c/
│
├── part_3_ablation/
│   ├── stage_3a/
│   ├── stage_3b/
│   └── stage_3c/
│
├── part_4_mechanism/
│   ├── stage_4a/
│   └── stage_4b/
│
├── part_5_generalization/
│   └── stage_5a/
│
├── raw_archive/
│
└── paper_results/
    ├── figures/
    ├── tables/
    ├── qualitative/
    ├── selection_manifest.yaml
    ├── results_summary.md
    └── result_claim_mapping.md
```

请在最终计划中完善。

---

# 7. Part I — Infrastructure & Method Validation

# Stage 0A — Environment & Manifest Freeze

## 目标

在开始训练前把能提前确定的环境、软件和目录全部确定。

必须明确推荐：

- Python version；
- PyTorch version；
- CUDA version；
- PyTorch Geometric version；
- NumPy；
- Pandas；
- Matplotlib；
- JSON Schema validator；
- config management；
- logger；
- checkpoint format；
- random seed handling；
- environment lock；
- Git/hash；
- VLM SDK；
- relation cache format。

如果当前运行机器信息未知：

标记：

`MUST_BIND`

不要虚构。

## RL Framework 必须现在选一个

比较：

### A
在现有项目 PPO 代码上改

### B
Stable-Baselines3

### C
CleanRL

### D
自己写轻量 PPO

### E
其他

当前需要支持：

- recurrent GRU；
- variable candidate mask；
- skill-level action；
- variable duration；
- Semi-MDP GAE；
- Structural Q auxiliary loss；
- R-GCN；
- PPO update 时 graph recompute；
- variable candidate count。

最终只能推荐一个主路线。

优先考虑：

> 易修改、代码透明、方便加入当前自定义结构。

---

# Stage 0B — Unit & Invariant Tests

不跑科研训练。

至少包括：

### T01
Empty prior → \(D^P=0\)

### T02
Empty prior → prior residual = 0

### T03
Nominal patch 不污染真实 Fact Store

### T04
\(G^K/G^H/	ilde G^K/	ilde G^H\) node alignment

### T05
goal_refs alignment

### T06
TRUE/FALSE/UNKNOWN 逻辑

### T07
ADD/DEL conflict detection

### T08
effect_fact_ref validation

### T09
SOFT_ORDER 不存在于 v2.1 schema

### T10
PPO old/new log probability 使用相同 candidate mask

### T11
Structural Q 只 gather executed action

### T12
Q target stop-gradient

### T13
shared R-GCN gradient flow

### T14
terminated / truncated bootstrap

### T15
candidate permutation consistency

### T16
cache key isolation

### T17
train-time prior dropout episode 内固定

### T18
PPO replay/recompute 不重新采样 prior

每个测试必须写：

- test filename；
- test input；
- procedure；
- expected result；
- tolerance；
- PASS condition。

例如：

```text
tests/test_zero_prior.py
```

---

# Stage 0C — VLM Cache Sanity Check

不要做大型 VLM benchmark。

只抽：

20–30 个 task-scene。

统计：

- JSON valid rate；
- ID valid rate；
- effect_fact_ref valid rate；
- contract redundancy rate；
- final empty prior rate；
- mean relation count；
- SUPPORTS count；
- RELEVANT_TO_GOAL count；
- obvious semantic issue count。

人工快速检查即可。

目的：

确认：

- schema 工作；
- 输出不是几乎全部非法；
- 不是几乎永远 EMPTY_PRIOR；
- 存在一些非合同重复关系。

如果只是部分场景为空：

不构成方法失败。

---

# 8. Part II — Minimal Learning Validation

# Stage 1A — One-Task Smoke Run

只使用：

1 个最简单但确实有结构选择的组合任务。

要求：

- 至少 2 个合法 candidate；
- 至少多步技能；
- 至少一个共享前置/长期依赖；
- 至少一个 non-empty VLM prior case；
- 不是单步 PICK。

第一轮只跑：

### B2
Contract-only M1

### Full
CP-DISR v2.1

如果必要再加入：

### B0
Observation/Fact non-graph policy

每个：

1 seed。

默认：

`seed=0`

## Smoke 只看

- reward 是否全零；
- success 是否能出现；
- PPO 是否 NaN；
- Q loss 是否爆炸；
- DK 是否变化；
- DP 是否在 non-empty prior 时偶尔非零；
- prior residual 是否偶尔非零；
- candidate mask 是否正确；
- 是否出现死循环；
- skill executor 是否正常；
- episode termination 是否正确。

## PASS 标准必须宽松

只要：

- B2 / Full 都能完整运行；
- 至少出现真实有效 transition；
- 至少偶尔出现成功；
- DP 在合法 soft prior case 下不是永久严格为 0；
- prior branch 不是永久失活；
- 没有严重数值异常；

就 PASS。

不要求：

Full > B2。

---

# Stage 1B — Minimal Tuning

只有 Stage 1A 明显训练不好时执行。

如果默认配置已经正常：

> **直接 SKIP Stage 1B。**

允许按顺序调：

1. learning rate；
2. training length；
3. entropy；
4. B；
5. \(\lambda_Q\)。

每个参数最多尝试：

2–3 个值。

建议候选：

### lr

- 1e-4
- 3e-4
- 5e-4

### B

- 0.25
- 0.5
- 1.0

### λ_Q

- 0.05
- 0.1
- 0.2

不要做完整 grid。

---

# 9. Part III — Core Paper Evidence

# Stage 2A — Core Exploration Runs

先选择：

2–3 个真正有结构意义的主任务。

方法：

- B0
- B1
- B2
- Full

seed：

- 0
- 1
- 2

这是：

**Exploration Runs**

允许：

- 看 task 差异；
- 判断哪些 task 适合主文；
- 判断训练预算；
- 判断 checkpoint；
- 做小范围调参；
- 补少量 seed；
- 找正常运行区间。

## Stage 2A 必须输出

每个 task：

- B0 curve；
- B1 curve；
- B2 curve；
- Full curve；
- success；
- AUC；
- stability；
- DK norm；
- DP norm；
- prior residual；
- average candidate count；
- average relation count。

## Task Suitability Report

必须输出：

`task_suitability_report.md`

按照：

- Main-paper strong candidate
- Main-paper usable
- Appendix / weak separation
- Not useful for presentation

分类。

这个分类不是方法优劣排名，而是：

> 哪些 task 最适合展示 CP-DISR 的研究故事。

---

# Stage 2B — Presentation Runs

根据 Stage 2A：

选择最终：

3–5 个主任务。

方法：

- B0
- B1
- B2
- Full

默认：

3 seeds / config。

允许：

从更大的运行集合里选择：

3 个正常、代表性、展示清楚的 seed。

例如：

0,2,4

不要求固定：

0,1,2

但必须在：

`selection_manifest`

记录原因。

Presentation Runs 使用：

已基本确定的参数。

目标：

得到：

- 主表；
- 主曲线；
- task-family summary。

---

# Stage 2C — Main Figure & Table Generation

只使用 Stage 2B 真实结果。

至少生成：

## Figure 1

Main learning curves。

建议：

x = environment interaction / skill transitions  
y = success rate

允许合理 smoothing。

## Table 1

Main task success。

列：

- Task
- B0
- B1
- B2
- Full

## Figure 2

Task-family aggregated result。

## Table 2

可选：

- AUC；
- sample efficiency；
- completion time；
- time-to-threshold。

如果最终 success 差距不大：

允许重点展示：

AUC / sample efficiency。

---

# 10. Part IV — Core Mechanism Evidence

# Stage 3A — Dual-Difference Ablation

选择：

1–2 个 prior 作用比较明显的 task。

比较：

### Full

vs

### A_DD

A_DD：

prior branch 使用 \(D^H\) 替代 \(D^P\)，但 no-prior episode 保持 prior input = 0。

默认：

3 seeds。

目的：

回答：

> 第二次差分 \(D^H-D^K\) 是否比直接使用增强图变化 \(D^H\) 更合适。

不要求：

所有 task 都显著提升。

---

# Stage 3B — Structural Q Ablation

比较：

Full

vs

A_Q：

\[
\lambda_Q=0
\]

只在：

1–2 个代表 task。

3 seeds。

重点看：

- learning AUC；
- convergence；
- seed stability；
- final success；
- Q-related feature learning。

如果：

最终 success 提升小，

但：

- early learning 更快；
- variance 更小；
- 训练更稳定；

也可接受。

---

# Stage 3C — Bounded Prior Ablation

比较：

Full

vs

A_B。

只在：

prior-sensitive task。

可只跑：

1 个 task。

3 seeds。

如果正常 prior 下差异很小：

没有问题。

它主要可以通过：

robustness 条件体现意义。

Stage 3C：

优先级低于 3A / 3B。

---

# 11. Part V — Test-Time Mechanism & Robustness

尽量使用：

**冻结 Full checkpoint**

不重新训练。

# Stage 4A — Actor Dependence Diagnostics

对同一 Full checkpoint：

测试：

1. \(D^K=0\)
2. \(D^P=0\)
3. shuffle \(D^P\)
4. remove prior
5. target-swap prior

记录：

- action distribution；
- KL / JS；
- argmax change；
- residual；
- success；
- selected skill changes。

目标：

确认 Actor 真正在使用结构信息。

---

# Stage 4B — Prior Robustness Evaluation

使用冻结 Full checkpoint。

测试：

- Original
- No Prior
- Edge Deletion
- Target-swap
- Irrelevant

第一轮只在：

1–2 个最有结构意义的任务。

Evaluation：

20–30 episodes / task / condition

先看基本趋势。

如果已经清楚：

停止。

如需要更稳定：

50 episodes。

不要一开始 100–500。

---

# 12. Part VI — Minimal Generalization

# Stage 5A — Compositional Generalization

只做一种最清楚的组合泛化。

推荐：

> Seen skill schemas + Seen predicates + Unseen object-goal / dependency composition

不要同时加入：

- visual shift；
- language shift；
- robot shift；
- domain shift；
- object count；
- topology；
- appearance。

如果结果好：

进入主文。

如果一般：

附录或收缩 claim。

---

# 13. Part VII — Private Paper Packaging

# Stage 6A — Raw Archive Review

扫描：

- task；
- seed；
- checkpoint；
- metrics；
- qualitative episode；
- failure run；
- robustness run；
- generalization run。

输出：

`raw_archive_index.csv`

不要修改原始数据。

---

# Stage 6B — Private Paper View Selection

从真实运行中选择：

### Task

最能体现结构机制的。

### Seed

正常、代表性或曲线清楚的。

### Checkpoint

validation 合理、行为稳定的。

### Metric

最能反映真实方法现象的。

### Qualitative Episode

最容易解释方法的真实 episode。

生成：

`private_paper_selection_manifest.yaml`

每一个最终展示结果都记录：

- run_id；
- task；
- seed；
- checkpoint；
- metric；
- smoothing；
- subset；
- selection reason；
- source file。

---

# Stage 6C — Final Result Package

输出：

```text
paper_results/
  figures/
  tables/
  qualitative/
  selection_manifest.yaml
  results_summary.md
  result_claim_mapping.md
```

`result_claim_mapping.md`

必须建立：

> 论文 claim → 支撑它的实验 → 对应 figure/table/run

映射。

---

# 14. Task 设计要求

不要只写：

“选几个任务”。

必须给任务结构模板。

## Development Task

1 个。

目标：

- debug；
- tune；
- smoke；
- checkpoint validation。

## Main Candidate Tasks

准备：

3–5 种结构类型。

例如：

### Task Type A — Shared Prerequisite

多个后续 skill 共享一个条件。

### Task Type B — Multi-Object Choice

多个当前可执行 skill，但长期后果不同。

### Task Type C — Intermediate Relocation

一个临时移动 action 对后续另一个 object 的操作有帮助。

### Task Type D — Multi-Goal Interaction

一个 action 的后果同时影响多个 goal。

### Task Type E — Recoverable State Change

执行结果可能使后续条件失效，需要重新建立。

如果实际平台尚未确定具体对象：

用：

> Task Structure Template + MUST_BIND

不要假装已有环境资产。

---

# 15. Seed 策略

固定默认：

## Smoke

`seed = 0`

## Exploration

`seed = 0,1,2`

## Presentation

默认：

3 seeds。

允许从更多真实运行中选择：

3 个正常展示 seed。

如果结果波动：

最多优先补到：

5 seeds。

不默认超过 5。

---

# 16. Checkpoint 策略

允许不同方法选择自己的合理 checkpoint。

## Exploration

每固定训练间隔保存 checkpoint。

## Presentation

基于：

development validation

选择：

表现较好且稳定的 checkpoint。

不使用：

单个 evaluation spike。

建议：

用连续若干 evaluation window 的平均表现判断。

允许：

Full 最佳 step ≠ B2 最佳 step。

---

# 17. Evaluation Episode 数量

固定起步：

## Smoke

10 episodes。

## Exploration

20 episodes / checkpoint。

## Presentation

30 episodes / task / seed。

如果已经很清楚：

停止。

如果需要稳定：

50 episodes。

100 episodes 以上：

仅在确实需要时补。

---

# 18. 参数分类

最终计划必须把所有参数分为三类。

## Frozen Parameters

至少：

- CP-DISR v2.1；
- relation schema；
- soft relation types；
- pure interaction DP；
- no static prior；
- graph type；
- R-GCN backbone；
- nominal intervention semantics；
- reward definition；
- Structural Q target definition；
- PPO formulation；
- 80/20 prior protocol；
- no train-time wrong-edge corruption。

## Default-but-Tunable

至少：

- lr；
- B；
- λ_Q；
- entropy coef；
- training steps；
- evaluation interval；
- hidden dim（只有必要时）；
- PPO batch/update size（仅资源原因）。

## Runtime MUST_BIND

至少：

- simulator / robot；
- task assets；
- controller；
- camera；
- verifier thresholds；
- skill timeout；
- hardware；
- actual interaction time unit。

---

# 19. 参数调整优先级

如果结果不好：

必须按顺序：

1. implementation；
2. candidate / mask；
3. reward；
4. Skill Contract；
5. Verifier；
6. DK 数值；
7. DP 数值；
8. prior residual scale；
9. learning rate；
10. training length；
11. entropy；
12. B；
13. λ_Q；
14. task difficulty；
15. VLM relation quality。

不要一看到 Full 不领先就修改 Method。

---

# 20. 第一批实验必须极简

第一批目标：

> 尽快得到能写第一版 Results 的基本证据。

建议只到：

## E00 — Stage 0B

Unit Tests。

无训练。

## E01 — Stage 0C

20–30 scene VLM cache sanity。

## E02 — Stage 1A

1 task  
× B2 / Full  
× 1 seed。

## E03 — Stage 2A

2–3 tasks  
× B0 / B1 / B2 / Full  
× 3 seeds。

## E04 — Stage 3A

1–2 tasks  
× Full / A_DD  
× 3 seeds。

然后：

> **暂停扩实验，整理第一版论文 Results。**

不要默认自动继续 E05–E09。

---

# 21. 第二批实验按需运行

只有第一批已经形成基本故事后：

再根据论文缺口选择。

## E05

Stage 3B — A_Q

## E06

Stage 3C — A_B

## E07

Stage 4A — Actor intervention

## E08

Stage 4B — Prior robustness

## E09

Stage 5A — Generalization

如果私人论文已经足够完整：

某些 E06–E09 可以：

- 缩小；
- 放附录；
- 不执行。

---

# 22. 可接受的早期结果标准

请明确告诉 Agent：

当前不是高门槛筛选。

以下情况都可以继续：

- Full 只领先几个百分点；
- 某 task Full 与 B2 接近；
- 某 seed 失败；
- A_Q 提升很小；
- A_B 只在 corruption 下有效；
- VLM 部分 scene 输出空 prior；
- A_DD 只在结构困难任务有差异；
- B0 在简单任务很好；
- generalization 比 ID 差；
- Full 不是每个任务第一。

这些优先：

> 调整 claim / 展示 / task 组合

而不是：

> 重设方法。

---

# 23. 真正结构性失败的标准

只有以下情况才考虑重新打开 Method：

1. \(D^P\) 在几乎所有合法 non-empty prior 下严格为 0；
2. soft relation 完全没有传播作用；
3. Full 完全学不会而 B0/B2 正常；
4. Structural Q 系统性导致训练崩溃；
5. nominal patch 污染真实 Fact Store；
6. Actor 对 DK/DP 完全不响应；
7. 有结构关系潜力的 task 中 VLM 几乎总是 EMPTY；
8. 多个结构任务中 Full 持续、明显、稳定低于简单 baseline；
9. 实现无法满足 v2.1 公式或权限约束。

只有这种问题才允许重新评估主体方法。

---

# 24. 结果不好时的处理顺序

如果 Full 结果不好：

不要马上 kill。

必须依次检查：

1. 是否代码 bug；
2. candidate / mask；
3. reward；
4. Skill Contract；
5. Verifier；
6. DK norm / distribution；
7. DP norm / nonzero rate；
8. prior residual 是否太小；
9. λ_Q；
10. B；
11. lr；
12. entropy；
13. training duration；
14. task 是否真正需要结构；
15. VLM relation 是否有非冗余信息。

只有上述问题基本排除后：

才讨论方法是否需要改。

---

# 25. 图表展示规则

允许：

- moving average；
- EMA；
- 合理 smoothing window；
- 合理横轴截取；
- 合理纵轴范围；
- mean / median；
- representative seed；
- task subset；
- hard-task subset；
- structured-task subset；
- 主图/附录拆分。

但必须在：

`selection_manifest`

记录：

- smoothing method；
- window；
- selected seeds；
- selected checkpoint；
- selected tasks；
- metric。

不要直接修改原始数值。

---

# 26. Metric 策略

主指标：

### Task Success Rate

如果 final success 差异不明显：

可以重点展示：

- learning AUC；
- sample efficiency；
- time-to-threshold；
- completion time；
- structured-task subset；
- prior residual behavior；
- actor intervention response；
- robustness degradation。

允许根据真实现象选择更有解释力的 metric。

---

# 27. Qualitative Case

至少准备：

1 个真实 qualitative episode。

优先选择：

最清楚的 case。

展示：

```text
Current Facts
↓
Candidate Skills
↓
Nominal Effects
↓
D^K
↓
VLM Soft Relations
↓
D^P
↓
Base Score / Prior Residual
↓
Final Logits
↓
Selected Skill
↓
Real Outcome
```

不要求随机选择 episode。

可以从真实 evaluation runs 中挑一个最容易理解的 case。

---

# 28. 软件和工具尽量现在锁定

请在最终实验计划里直接给推荐版本。

至少包括：

- Python；
- PyTorch；
- CUDA；
- PyTorch Geometric；
- NumPy；
- Pandas；
- Matplotlib；
- JSON Schema validator；
- config library；
- logger；
- checkpoint；
- environment lock；
- Git/hash；
- random seed；
- VLM SDK；
- relation cache manifest。

如果当前环境未知：

给：

`MUST_BIND`

不要空泛写：

“使用兼容版本”。

---

# 29. VLM Cache 生产流程固定

必须固定：

1. model snapshot；
2. API region；
3. SDK；
4. temperature；
5. max output tokens；
6. max relations；
7. prompt version；
8. schema version；
9. retry；
10. cache key；
11. train/dev/test separation；
12. raw/parsed/rejected logging。

正式 cache 生成以后：

不能因为 RL 结果不好就反复问 VLM 直到得到更有利 relation。

如果真的修改：

prompt/schema/model：

必须：

- 升版本；
- 重新生成；
- 保留旧 cache；
- 记录原因。

---

# 30. 实验日志

每次运行必须保存：

- run_id；
- method_version；
- git_hash；
- task；
- seed；
- config；
- prior cache hash；
- checkpoint；
- train metrics；
- eval metrics；
- failure reason；
- hardware；
- software version；
- start/end time；
- tuning history。

这样最终私人论文结果可以完整追溯。

---

# 31. Paper Writing 与实验同步

不要等待全部实验结束。

## Stage 0–1 期间

完成：

- Introduction；
- Related Work；
- Problem Formulation；
- Method。

## Stage 2A 期间

完成：

- Experimental Setup；
- Task；
- Baselines；
- Metrics。

## Stage 2B 后

开始：

- Main Results。

## Stage 3

填：

- Ablation；
- Mechanism。

## Stage 4–5

填：

- Robustness；
- Generalization。

## Stage 6

整理：

- Abstract；
- Discussion；
- Conclusion。

---

# 32. 私人论文允许围绕最清楚的真实现象调整主叙事

例如：

### 情况 A

Full 主要改善 learning speed：

强调：

> sample efficiency。

### 情况 B

Final success 更明显：

强调：

> long-horizon skill decision performance。

### 情况 C

Normal prior 差异小，但 corruption 下稳：

强调：

> robust use of imperfect priors。

### 情况 D

VLM 只在结构复杂任务帮助明显：

强调：

> structure-dependent benefit。

### 情况 E

Structural Q 对 final success 帮助小，但训练更稳：

强调：

> outcome-grounded representation regularization。

允许根据真实实验中最清楚的现象：

调整 claim 强弱。

---

# 33. Experiment Manifest v0

最终计划必须直接生成一份：

`Experiment Manifest v0`

至少包括：

```yaml
method:
  name: CP-DISR
  version: v2.1

vlm:
  model_snapshot: MUST_BIND_OR_RECOMMENDED
  relation_schema: m1_soft_relations_v2
  max_relations: 8
  temperature: 0
  cache_mode: offline

prior_training:
  original: 0.80
  absent: 0.20
  sampled_at: episode_start
  fixed_within_episode: true

graph:
  encoder: R-GCN
  hidden_dim: 128
  layers: 4
  num_bases: 4
  aggregation: mean
  dropout: 0

actor:
  prior_bound_B: 0.5

structural_q:
  lambda_q: 0.1

ppo:
  lr: 3e-4
  clip: 0.2
  gae_lambda: 0.95
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip: 0.5
  epochs: 4
  rollout_skill_transitions: 1024
  recurrent_sequence_length: 16
  minibatch: 64

seeds:
  smoke: [0]
  exploration: [0, 1, 2]
  presentation_default_count: 3
  maximum_initial_extension: 5

evaluation:
  smoke_episodes: 10
  exploration_episodes: 20
  presentation_episodes: 30
  optional_stable_episodes: 50

runtime:
  simulator_or_robot: MUST_BIND
  controller: MUST_BIND
  camera: MUST_BIND
  verifier_thresholds: MUST_BIND
  skill_timeouts: MUST_BIND
  hardware: MUST_BIND
```

请根据实际推荐补全。

---

# 34. 阶段总表

最终必须给：

| Part | Stage | Purpose | Task数 | Config数 | Seed数 | Training? | Paper Main? | Mandatory? |
|---|---|---:|---:|---:|---:|---|---|---|

让我能一眼看懂整个计划。

---

# 35. 运行量估计

必须分别估算：

## 第一批实验

总 runs。

## 第二批实验

总 runs。

## Presentation Runs

总 runs。

## 最终私人论文可能总量

要求整体明显低于：

> 7 configs × 5 tasks × 10 seeds

这类规模。

---

# 36. 立即执行顺序

最终给出类似：

```text
现在执行：
Stage 0A

完成：
Stage 0B

然后：
Stage 0C

然后：
Stage 1A

如果 Stage 1A PASS：
直接 Stage 2A

如果 Stage 1A FAIL：
Stage 1B

Stage 2A 完成：
立即开始第一版 Results 草稿

然后：
按结果决定 Stage 2B / 3A

后续 Stage 3B/3C/4A/4B/5A：
按论文证据缺口逐个执行
```

必须明确。

---

# 37. “以后我怎么指挥 Agent”示例

最终文档最后给出实际使用范式。

## 指令

> 执行 Stage 0B

Agent：

- 只跑 unit tests；
- 生成 `stage_0b_summary.md`；
- 更新 `stage_0b.json`；
- 停止。

## 指令

> 执行 Stage 2A

Agent：

- 加载 Stage 2A 指定 task/config/seed；
- 执行探索实验；
- 生成 task suitability report；
- 不自动进入 Stage 2B；
- 停止。

## 指令

> 根据 Stage 2A 结果执行 Stage 2B

Agent：

- 使用 2A 选择的 task；
- 使用选定 config；
- 重新跑 Presentation Runs；
- 不自动执行 2C。

## 指令

> 执行 Stage 6B

Agent：

- 不重新训练；
- 只从 Raw Archive 中筛选真实结果；
- 生成 selection manifest；
- 停止。

---

# 38. 最终输出格式要求

请最终输出一份真正连贯的：

# CP-DISR v2.1 Minimal Experimental Execution Plan

不要只是逐条回答本 Prompt。

完整文档必须包含：

1. 总目标；
2. Private Paper原则；
3. 冻结方法；
4. 实验总结构；
5. 文件与目录规范；
6. 软件工具链；
7. 参数分类；
8. Part I；
9. Stage 0A；
10. Stage 0B；
11. Stage 0C；
12. Part II；
13. Stage 1A；
14. Stage 1B；
15. Part III；
16. Stage 2A；
17. Stage 2B；
18. Stage 2C；
19. Part IV；
20. Stage 3A；
21. Stage 3B；
22. Stage 3C；
23. Part V；
24. Stage 4A；
25. Stage 4B；
26. Part VI；
27. Stage 5A；
28. Part VII；
29. Stage 6A；
30. Stage 6B；
31. Stage 6C；
32. Seed策略；
33. Checkpoint策略；
34. Evaluation协议；
35. Metric策略；
36. Task筛选策略；
37. 调参顺序；
38. 可接受结果；
39. 结构性失败；
40. Raw Archive；
41. Private Paper View；
42. Experiment Manifest v0；
43. 阶段总表；
44. 总运行量估计；
45. 立即执行顺序；
46. Agent指挥示例；
47. Paper同步写作计划。

---

# 39. 最后必须明确回答五个问题

## A. 第一批到底跑什么？

给可以直接交给 Agent 的表。

## B. 每个第一批 Stage 跑到什么现象就停止？

必须采用宽松标准。

## C. 哪些参数现在锁死？

哪些允许调？

## D. 如果结果一般，按照什么顺序调整，直到得到一套适合私人论文展示的真实结果？

## E. 最终如何从 Raw Archive 选择：

- task；
- seed；
- checkpoint；
- metric；
- qualitative case；

构成 Private Paper View？

---

# 最终目标

我要的不是：

> “科研实验通常应该怎么做”的建议。

我要的是：

> **一份 CP-DISR v2.1 的最终实验 Agent 执行手册。**

它必须做到：

### 1.
实验量低。

### 2.
Part / Stage 边界明确。

### 3.
每个 Stage 自包含。

### 4.
我以后只说：

> “执行 Stage XX”

Agent 就能直接运行。

### 5.
软件、参数、seed、日志、目录、输出尽量提前确定。

### 6.
早期结果门槛宽松。

只要：

- 方法跑起来；
- 基本趋势出现；
- 核心结构通路不是完全失效；

就继续论文。

### 7.
允许私人论文展示优先的真实结果选择：

- 调参；
- 重跑；
- task筛选；
- seed筛选；
- checkpoint选择；
- metric选择；
- smoothing；
- qualitative case筛选；
- claim收缩。

### 8.
所有展示结果必须来自真实运行。

### 9.
完整 Raw Archive 必须保留。

### 10.
不要再让实验阻塞论文写作。

实验计划生成以后：

> **立即开始 Part I，同时继续写论文。**
