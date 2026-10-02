---
title: "CP-DISR 新 T_P 主任务族设计文档"
subtitle: "Evidence-First Prior-Diagnostic Task Family · 证据优先的先验诊断任务族"
document_id: CP-DISR-TP-EF-DESIGN-1
task_family_id: T_P_EVIDENCE_FIRST_PREPARATION_REWORK
short_name: T_P-EF
version: "1.0"
status: DESIGN_ONLY_NOT_AUTHORIZED
execute_now: false
method_version: "2.1.1"
baseline_evidence_commit: "9422b837cf1dfc43afc056a77bdb59d63512b3b6"
execution_base_commit: "TBD_AFTER_SOURCE_IDENTITY_FREEZE"
global_s4: NOT_COMPLETE
global_s5: NOT_COMPLETE
---

# 0. 文档定位

本文件定义新的 **T_P 主实验任务族**。T_P 的角色不是提出一个新的算法，也不是制造一个“看起来复杂”的机器人场景，而是为论文的 C2/C3 核心问题提供一个稳定、可辨识、可重复的实验载体：

> 当多个高层技能都合法，而 VLM 提供的语义关系可能有用、无关、缺失或误导时，候选后果条件化的 prior 接口是否能改善策略；策略又是否会在真实结果训练下形成更符合动作效用的 prior 依赖？

本文件只完成设计和未来执行边界，不自动授权：

- 环境构造；
- 新物理 attempt；
- VLM/provider 请求；
- RL 训练；
- optimizer step；
- test；
- Family B 续改；
- RoboCasa 迁移。

后续必须先执行一张独立的 `T_P-EF-DISCOVERY` 资格卡；只有资格门通过后，才允许冻结任务族和申请 Full/B2/A_STAT/A_CAT/A_Q 的 seed0 pilot。

---

# 1. 为什么需要新的 T_P

## 1.1 T_B 已完成的作用

T_B 已经给出 C1 的真实证据：

- B1-K 在两个 seed 及冻结 holdout 上均未形成有效策略；
- B2 和 B1-K+E 最终都能形成可迁移策略；
- B2 在预先固定的开发检查点上更早进入有效策略区域；
- 因此，候选后果表示的主要价值更像是有利的学习归纳偏置，而不是直接效果读取无法达到的唯一终点能力。

T_B 不包含 semantic prior，不能回答：

- prior 是否有整体增量；
- 后果条件化 prior 是否比静态 prior 更合适；
- 灵活融合和纯交互有什么取舍；
- 真实结果训练后，策略是否减少有害 prior 的影响。

这些问题必须由 T_P 承担。

## 1.2 Family B 留下的结论

Family B 的研究目标与 T_P 相同，但具体场景路线暴露了三类问题：

1. **固定候选偏置**：某个 pad 几乎总是更快，任务不具诊断性；
2. **可达性失败**：为了制造偏好反转而移动物体，setup 在决策前超时；
3. **夹爪碰撞失败**：静态 AABB 间隙通过，但完整抓取扫掠空间不足。

因此，本设计不再做 `Family B v4`，也不再用坐标微调强行制造理想反转。

新的原则是：

> **先从已经稳定的任务和决策状态中测出真实候选效用差异，再决定它是否适合作为 T_P。**

---

# 2. 新 T_P 的核心方案

## 2.1 名称

**Evidence-First Preparation-and-Rework Task Family**

中文：**证据优先的准备—返工决策任务族**

任务族优先复用当前 robosuite/LIBERO 运行链、公开 RGB-D、现有 Skill Contract、Verifier、Evaluator 与稳定低层技能。

## 2.2 基础任务语义

优先复用已经稳定运行的 T_B 式任务元素：

- 一个可操作容器；
- 一个主要目标物 `target`；
- 一个次要物体 `secondary_object`；
- 一个合法 buffer / staging 区；
- 固定技能库中的 `OPEN / PICK / PLACE / PLACE_BUFFER`；
- 完整任务目标，例如：

```text
Inside(target, container) = TRUE
AtBuffer(secondary_object, buffer) = TRUE
GripperEmpty = TRUE
```

这只是任务模板。T_P 的研究单位不是整张场景图，而是其中满足资格条件的 **decision state**。

## 2.3 研究单位：decision state，而不是人工镜像场景

一个合格 decision state 必须满足：

```text
当前公开状态 s
        ↓
至少两个真实合法候选 a / b
        ↓
两个候选均能稳定执行
        ↓
候选后使用同一冻结 continuation
        ↓
真实结果存在可复核差异
        ↓
自然 VLM relation 对该差异有潜在信息
```

候选可以来自现有技能，例如：

```text
OPEN(container)
PICK(target)
PICK(secondary_object)
```

具体候选对不在设计阶段预设赢家，也不要求所有状态使用同一个候选对。

## 2.4 两类优先见证

任务族至少应尝试覆盖以下两类中的两类；若自然环境只能稳定形成一类，也可进入后续，但必须缩小 claim。

### W1：Preparation / Rework Witness

两个动作均合法，但不同顺序导致：

- 是否需要额外搬运；
- 是否产生返工；
- 是否更接近 deadline；
- 是否增加完整任务技能数；
- 是否增加真实完成时间。

例：

```text
候选 A：先准备容器
候选 B：先处理次要物体
```

两条分支都能完成任务，但一个顺序会多一次搬运或等待。

### W2：Context-Dependent Prior Witness

同一类 semantic relation 在不同状态下：

- 有时帮助概率移向更优候选；
- 有时几乎无关；
- 有时把概率推向较差候选。

它不要求人工制造严格镜像：

```text
state 1: A > B
state 2: B > A
```

也允许更自然的：

```text
state 1: prior helpful
state 2: prior neutral
state 3: prior harmful
```

关键是 utility 由真实执行给出，而不是按语义标签预设。

---

# 3. 一个简单例子

以下只是说明机制，不是冻结场景或预设结果。

任务：

```text
打开容器
把 target 放入容器
把 secondary_object 放到 buffer
```

某个公开状态中有两个合法候选：

```text
A = OPEN(container)
B = PICK(secondary_object)
```

从同一个 snapshot 分支：

```text
Branch A
restore(s)
→ OPEN(container)
→ 固定 continuation
→ 成功，5 skills，23 s
```

```text
Branch B
restore(s)
→ PICK(secondary_object)
→ 同一个固定 continuation
→ 成功，7 skills，34 s
```

那么真实参考效用满足：

```text
Q_ref(s, A) > Q_ref(s, B)
```

若 VLM 给出的合法、非冗余关系使策略增加 A 的概率，则该 prior 在这个状态上是 helpful。

另一个状态可能测出：

```text
Q_ref(s2, B) > Q_ref(s2, A)
```

即使某条关系在语义上“说得通”，如果它仍把策略推向 A，它在当前状态就是 harmful。

这正是论文真正要研究的：

```text
relation correctness
≠
decision utility
```

---

# 4. 反 Family B 设计原则

## 4.1 不从理想赢家开始

禁止：

```text
先决定 state U 必须 A 胜
先决定 state V 必须 B 胜
再调几何直到出现预期结果
```

必须：

```text
先固定候选状态
再真实执行两个分支
最后读取效用方向
```

## 4.2 不新增危险低层控制

默认禁止：

- 新低层 controller；
- 新抓取轨迹；
- reach 边界坐标；
- 靠近夹爪扫掠空间的障碍物；
- 为制造差异增加实体 pad；
- 为让 prior 有用删除 hard precondition。

优先只使用已经在 T_B 或其他现有任务中稳定完成过的技能、对象和工作空间。

## 4.3 Verifier 后置条件优先于 controller exit

```text
NORMAL_TERMINATION
≠
逻辑后置条件成立
```

例如 `PICK(obj)` 后必须由公开 Verifier 确认：

```text
Held(obj) = TRUE
```

否则该 setup / branch 判失败，不能继续执行或补 nominal fact。

## 4.4 完整抓取扫掠空间，而非静态 AABB

若任务确实需要新物体布局，资格门必须检查：

- 张开夹爪；
- DESCEND；
- PRESS；
- CLOSE；
- LIFT；
- RETREAT；

整个运动扫掠空间。

只检查对象中心距离、AABB 静态间隙或可达半径不够。

## 4.5 不把硬约束缺失包装成 prior 价值

两个候选必须在同一个真实 mask 中合法。

若候选差异来自：

- 一个动作缺 hard precondition；
- planner horizon 不足；
- unsafe action 被 mask；
- controller 根本无法完成；
- observation UNKNOWN；

则该状态不进入 T_P。

---

# 5. 资格流程总览

```text
Phase 0：冻结来源与候选池
    ↓
Phase 1：稳定 decision-state 发现
    ↓
Phase 2：同 snapshot 的真实候选分支
    ↓
Phase 3：E1–E3 relation / representation 检查
    ↓
Phase 4：E4–E6 consequence / insufficiency / prior 检查
    ↓
Phase 5：冻结任务族、生成规则和 split
    ↓
Phase 6：申请 seed0 pilot
```

本设计不允许跳过 Phase 1–4 直接生成 train/dev/test 或开始 RL。

---

# 6. Phase 0 — 来源与候选池冻结

## 6.1 平台优先级

```yaml
primary_platform:
  existing_robosuite_libero_stack

reuse:
  - current public RGB-D path
  - existing stable skills
  - Skill Contracts
  - FactStore
  - Verifier
  - Evaluator
  - B_PLAN
  - branch restore / recorder infrastructure

not_in_scope:
  - Family B coordinate continuation
  - RoboCasa migration
  - new low-level policy training
```

## 6.2 候选状态来源

优先扫描：

1. 现有稳定 dev 场景；
2. 现有运行轨迹中的中间 snapshot；
3. 现有任务生成器能产生、但尚未作为 test 使用的开发配置。

第一轮最多读取 24 个已有合法 dev scene / snapshot。

只有已有状态无法形成最少两个见证时，才允许一轮有界的新 dev 生成；仍不得生成 test。

## 6.3 候选状态登记

每个候选状态至少记录：

```yaml
state_id:
scene_id:
snapshot_hash:
public_observation_hash:
facts_hash:
goal_hash:
candidate_ids:
candidate_mask:
candidate_count:
setup_trace:
source_role: existing_dev | generated_dev
```

不得根据 Full 或任何学习模型输出选择状态。

---

# 7. Phase 1 — 决策状态稳定性门

每个状态先做零 prior、零训练的工程检查。

## Gate G1：多候选合法性

要求：

```text
legal candidate count >= 2
```

候选必须来自同一真实 Safety / Contract mask。

## Gate G2：候选前状态一致

对于配对分支：

- 同一 snapshot；
- 同一公开 RGB-D；
- 同一 FactStore；
- 同一 goal；
- 同一 candidate IDs/mask；
- 同一 qpos/qvel 容差；
- 同一 RNG / restore identity。

## Gate G3：候选技能稳定

每个候选必须先独立证明：

- controller 进入并返回；
- Verifier 后置条件成立；
- 无未知事实 carry-over；
- 无 hidden QA 进入正式事实；
- 无工程异常；
- 不处于 reach / collision 边界。

若任一候选不稳定，该状态立即淘汰，不改坐标继续试。

---

# 8. Phase 2 — 真实分支与 Q_ref 见证

## 8.1 配对执行

对状态 `s` 和候选 `a,b`：

```text
restore(s)
→ execute(a)
→ fixed continuation C
→ record outcome
```

```text
restore(s)
→ execute(b)
→ same fixed continuation C
→ record outcome
```

continuation 必须：

- 在看分支结果前冻结；
- 对 a/b 相同；
- 不读取 VLM truth；
- 不使用 Full / A_STAT / A_CAT 输出；
- 能真实完成或真实失败；
- 所有失败进入分母。

每候选最多两次配对 continuation，用于估计重复误差；不是无限重试。

## 8.2 参考效用

优先使用冻结的任务回报：

```text
Q_ref(s,a) = fixed continuation 下的真实 return
```

它是有限 continuation 的动作效用，不是全局最优 `Q*`。

同时记录：

- success；
- start-discounted return；
- task completion time；
- skill count；
- rework count；
- failure reason；
- controller / Verifier 结果。

## 8.3 差异阈值

在查看跨候选差异前，先由同候选重复结果冻结 `epsilon_Q` / repeatability tolerance。

候选对只有在：

```text
|Q_ref(s,a) - Q_ref(s,b)| > epsilon_Q
```

或有可靠的 success / rework / cost 差异时，才可作为效用见证。

若候选结果相同或差异落在重复误差内：

```text
LOW_DIAGNOSTIC_VALUE
```

不进入主要见证。

## Gate G4：至少两个独立真实见证

至少两个不同配置 / snapshot 必须满足：

- 两候选均真实合法；
- 两分支可审计；
- 有可靠效用差异；
- 差异不是工程异常；
- 差异不是删除 hard constraint 造成。

两个见证只证明“存在”，不代表整个任务族机会密度足够。

---

# 9. Phase 3 — VLM relation 与表示入口

## 9.1 VLM 输入

冻结：

- 初始公开 RGB；
- 对象绑定；
- goal；
- 允许 Action ID；
- 同源 Skill Contract；
- provider / model；
- prompt；
- few-shot；
- relation schema；
- decoding 参数。

VLM 只输出：

```text
SOFT_SUPPORTS(Action A, Action B)
SOFT_RELEVANT_TO_GOAL(Action A, goal proposition)
```

VLM 不输出：

- 下一动作；
- 完整计划；
- 新对象；
- 新技能；
- hard mask；
- reward；
- relation truth。

## 9.2 关系准入

每条 relation 必须：

- ID合法；
- effect_fact_ref 指向 source action 已注册的 ADD/DEL；
- 非非法自环；
- 非直接 hard contract 冗余；
- 不把 hard precondition 改名为 soft prior；
- raw / accepted / rejected 全部保存。

如果自然 relation 只重复合同，例如：

```text
OPEN(container) supports PLACE(target, container)
```

而该支持已完全由硬 PRE/ADD 路径直接表达，则它不能单独作为 prior-diagnostic 见证。

## Gate G5：Relation existence

至少两个独立配置出现：

- 自然生成；
- 合法；
- 非冗余；
- 可裁定或保留 U；
- 与候选决策有实际关联的 relation。

手写 relation / oracle relation 不能替代该门。

## Gate G6：Representation entry

至少两个配置满足：

- legal candidate >= 2；
- 至少一个候选有 changed nominal patch；
- admitted R 进入增强图；
- 对候选表示产生可追踪输入变化；
- mask / facts / reward 未被 R 修改。

## Gate G7：Candidate discriminability

关系影响不能只是所有候选共同平移。

至少观察到：

- candidate-relative representation difference；
- candidate-relative residual / logit difference；
- 或梯度可达输入路径。

随机初始化不要求选对动作，但必须证明通道不是结构上无效。

---

# 10. Phase 4 — E4/E5/E6 资格判定

## E4：真实任务后果

使用 Phase 2 的物理分支证明：

- 候选都合法；
- 后果存在可靠差异；
- 不是增加无意义动作后人为制造成本。

## E5：合同合法但不足

必须检查：

- B_PLAN 深度足够；
- planner 不是因 horizon / bug 失败；
- hard precondition 已完整登记；
- 两候选在合同层均合法；
- 缺失信息属于 soft cost / rework / relevance，而不是安全事实。

## E6：有用但不完美的 prior

relation truth、当前 opportunity 和 utility 分开记录。

允许分类：

```yaml
BENEFICIAL_IMPERFECT:
  prior 有可验证帮助，同时存在缺失/错误/无关机会

HARMFUL_BUT_INFORMATIVE:
  prior 有真实信号，但固定使用会被部分关系误导

NEAR_PERFECT:
  prior 很准确；C2可研究，但自然错误 claim 降级

HEURISTIC_INCONCLUSIVE:
  planner启发式未显示差异，但物理分支有真实差异

CONTRACT_SUFFICIENT:
  合同与物理结果均无 prior 增量空间，不训练

PROVIDER_SCHEMA_LIMITATION:
  R*有信号，R无信号；仅允许一次有依据的来源/schema修订

REPRESENTATION_LIMITATION:
  R*也无法进入表示；转方法决策，不强行训练
```

## Gate G8：进入任务族冻结

只有以下类别允许进入 S2：

- BENEFICIAL_IMPERFECT；
- HARMFUL_BUT_INFORMATIVE；
- NEAR_PERFECT（需缩小错误prior主张）；
- HEURISTIC_INCONCLUSIVE（必须有物理见证）。

---

# 11. Phase 5 — 任务族冻结

## 11.1 生成规则，而不是筛赢家

冻结能够生成合格 decision opportunity 的因素，例如：

- 合法候选分支数；
- preparation / rework 负担；
- 次要物体是否影响工作区；
- relation coverage；
- witness type。

禁止按：

- Full是否领先；
- VLM relation是否“漂亮”；
- 某方法结果；
- test utility；

筛选场景。

## 11.2 初始规模

```yaml
train: 32
dev: 16
test: 32
total: 80
```

这是启动规模，可在冻结前因实际任务族能力调整，但不能看训练结果或 test 后再改。

S1 discovery witness 只进入 discovery / dev，不进入 test。

split 按：

- generator seed；
- scene identity；
- snapshot provenance；

隔离，禁止轻微复制跨 split。

## 11.3 机会密度

训练前冻结：

```yaml
all_decision_states:
eligible_states:
opportunity_density:
multi_candidate_rate:
natural_R_positive_rate:
changed_patch_rate:
representation_entry_rate:
```

`eligible_state` 要求：

- 有效决策状态；
- 至少两个真实合法候选；
- 自然 admitted R；
- 至少一个候选有 changed nominal patch；
- 有真实计算入口。

不能以 Full 赢不赢或最终 residual 是否非零定义 eligible。

## 11.4 共享 utility bank

冻结 16 个 development snapshot：

```text
16 states
× 2 candidates
× 2 continuations
= 64 physical branches
```

供：

- Full；
- A_STAT；
- A_CAT；
- A_Q；
- E2/E4/E8 checkpoint；

共享，不为每个模型重复收集。

---

# 12. Endpoint 与评价

训练前冻结：

```yaml
primary_endpoint:
  metric: mean_start_discounted_return
  symbol: J
  role: frozen_independent_test

secondary_endpoints:
  - success
  - task_cost
  - rework_count
  - skill_count
  - physical_completion_time

learning_evidence:
  - raw learning curve
  - AUC on shared N or T interval
```

若新任务的重要代价无法由 J 或这些次终点稳定测量，必须在训练前修订并记录；不能等 Full 不领先后再换主指标。

失败 episode 的 J 可能为0，因此还必须报告：

- 失败时长；
- failure reason；
- 已执行技能数；
- rework；
- technical denominator。

---

# 13. 训练矩阵（仅未来申请，不在本文件执行）

资格通过后，先申请 seed0 pilot：

```text
Full seed0
B2 seed0
A_STAT seed0
A_CAT seed0
A_Q seed0
```

回答：

```text
Full vs B2:
prior接口整体增量

Full vs A_STAT:
后果条件化 prior vs 静态 prior

Full vs A_CAT:
限制性交互 vs 灵活融合

A_Q:
辅助Q/grounding作用及clip混杂
```

seed0 development evidence、E1–E6、opportunity density、Q_ref 与 readout 分析完成后，才进入 Method Decision Gate。

本设计不授权这些训练。

---

# 14. 预算与停止纪律

## 14.1 资格阶段

设计/离线检查：

```text
RL = 0
optimizer = 0
```

未来 S1 discovery 的上限应沿权威计划单独授权：

- 最多24个已有 dev scene / snapshot；
- 必要时最多12个新 dev scene；
- VLM 最多12次首次请求＋12次合法格式/传输重试；
- 必要物理见证最多8 episode；
- 不得语义重问到满意。

S2 时间 profile：

- 最多10 attempt；
- 取得5个合法 reference 成功；
- 不进入 PPO / BC / few-shot。

## 14.2 停止条件

立即停止并报告：

1. 24个已有状态中没有两个真实候选效用见证；
2. 一次有据修订后仍无自然非冗余 relation；
3. 候选差异只来自 hard constraint / planner bug；
4. 低层执行稳定性不足；
5. 关系不能进入候选表示；
6. 只能通过手写/oracle关系建立任务；
7. 需要继续调坐标才能维持差异；
8. 需要新低层controller或大规模平台迁移。

不得：

- 自动创建 Family C；
- 自动转 RoboCasa；
- 无限新增 witness；
- 为获得正结果换 provider；
- 看 test 后修改生成器。

---

# 15. 工程检查清单

每个候选见证必须同时保存：

```text
source / environment / profile identity
snapshot restore receipt
public RGB-D
facts
goal
candidate IDs and mask
candidate nominal patches
natural raw / admitted / rejected R
branch A / B action traces
fixed continuation identity
Verifier / Evaluator records
real return / success / time / rework / skills
Q_ref uncertainty / repeatability
hidden QA separation
provider request identity and payload hash
```

明确区分：

```text
contract effect description K(a)
nominal effect application T_K^nom(F,a)
real observed consequence F_{t+1}
```

三者不得混用。

---

# 16. 必须输出的设计/资格文件

未来资格卡至少生成：

```text
authorization.json
source_identity.json
candidate_state_registry.json
decision_state_screening.csv
branch_registration.json
paired_branch_results.csv
continuation_contract.json
repeatability_and_epsilon_q.json
relation_schema.json
relation_admission_receipts.json
e1_e6_gate.json
eligibility_manifest.json
provider_cost_ledger.json
technical_events.jsonl
final_discovery_summary.md
verify.json
```

若进入 S2，再生成：

```text
task_family_manifest.yaml
generator_spec.yaml
train_dev_test_split.json
public_input_contract.json
cache_index.json
truth_protocol.json
opportunity_density_definition.json
reference_bank_manifest.json
endpoint_freeze.yaml
runtime_profile.yaml
candidate_seed0_pilot_request.md
```

---

# 17. Git 和版本纪律

建议分支：

```text
codex/cp-disr-tp-evidence-first-design
```

设计提交：

```text
research: define evidence-first prior-diagnostic T_P
```

未来 discovery 与 task-family freeze 分开提交：

```text
research: qualify T_P evidence-first mechanism witnesses
research: freeze T_P evidence-first task family
```

禁止：

- amend 旧 Family B 结果；
- 覆盖 T_B 结果；
- force push；
- git clean；
- 把 provider secret 写入证据；
- 在同一版本内边看结果边改 generator。

---

# 18. 下一张可执行卡的范围

下一张卡应命名为：

```text
CP-DISR-TP-EF-DISCOVERY-1
```

只允许：

1. 读取最多24个已有开发scene/snapshot；
2. 登记多候选状态；
3. 对预先冻结的少量候选状态做真实配对分支；
4. 在看结果前冻结 continuation；
5. 对通过物理门的状态调用有限 VLM/provider；
6. 完成 E1–E6；
7. 输出 ELIGIBLE / NOT_ELIGIBLE。

它不允许：

- 构造完整80配置任务族；
- 开始RL；
- 生成test cache；
- 运行Full/A_STAT/A_CAT；
- 调Family B坐标；
- 迁移RoboCasa。

---

# 19. 最终设计结论

新的 T_P 不是 Family B 的坐标修订，也不是另造一个庞大 benchmark。

它是：

> **从已有稳定任务中，以真实候选分支效用为先，筛选自然 semantic prior 真正有机会有用、无用或有害的决策状态，再将这些状态冻结成 prior-diagnostic task family。**

与 Family B 的根本区别：

```text
Family B:
scene-first
→ 先设计几何
→ 再希望它产生决策差异

T_P-EF:
evidence-first
→ 先验证稳定候选和真实效用
→ 再冻结任务与prior接口
```

只有完成这一步，论文才能从已经完成的 C1 证据，进入真正的 C2/C3 核心实验。
