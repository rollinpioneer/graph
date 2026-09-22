---
title: "CP-DISR"
subtitle: "Paper-Oriented Research Specification v3.1"
date: "2026-09-22 · Current Method: CP-DISR / M1 v2.1.1"
lang: zh-CN
---

**面向长期技能选择的合同—先验交互对比**

*Contract–Prior Interaction Contrasts for Long-Horizon Skill Selection*

## 文档控制

本文为当前完整研究维护源：**文档v3.1；方法CP-DISR/M1 v2.1.1；实验计划v1.1**。本轮独立核验Claude审查后，保持图、DP算子、Full Actor和关系schema；重新裁定Actor训练权重、duration discount、baseline和分阶段实验。改动来源与拒绝理由见`review/CP_DISR_Independent_Adjudication_v1.md`。

最新本轮用户指令与本文件明确决议优先于历史规范；原v3.0及Claude原文归档为source，不自动成为正确结论。旧实验Agent Prompt的有限开发调参与Private Paper View原则保留；主文定量方法比较默认报告完整预定seed，精选illustration另标。未挂载的旧生成执行包未被冒称已核查，历史总预算不一致由Plan1.1明确重新裁定。

四种状态保持：方法定义、非最优保证的实现默认、真实资源MUST_BIND、经验收益待验证。没有新增RL、VLM调用或机器人操作；数学和文件检查不等于生产单测。平台、控制器、标定、资产、软件lock均需真实绑定。

文中[S#]/[R#]保留历史项目与基础参考索引；新增[W#]对应本包`review/Sources.md`与web证据ledger。查到全文仅表示核对相应方法段，不是复现全部实验或穷尽首创审查。M1+只在Future Work，不进入当前算法或预算。

# 1. Executive Summary

CP-DISR研究一个明确问题：在固定技能库、来源明确的技能合同和固定但不完美的VLM关系下，如何学习长期技能选择。合同定义可执行技能的名义结构语义；VLM补充可能相关的跨技能或技能—目标联系；模型不直接执行VLM给出的计划，而是比较各候选在两种结构下引起的名义变化，学习这些变化什么时候应影响决策。

对于每个当前允许执行的候选，系统构造“若该技能按名义成功分支结束”的临时事实副本。合同当前图、增强当前图及其两个临时后继视图由同一个R-GCN编码。合同差分$D_i^K$表达合同结构对该变化的响应；先验交互$D_i^P$是增强响应与合同响应之差。Actor由合同主分数与零锚定有界先验残差组成，随后只执行一个真实技能。[S2，§8–10；S3，§1]

学习使用真实奖励、真实耗时与真实下一状态。PPO、独立V和辅助Structural Q联合训练共享观测、图与候选特征。Q仅监督实际执行动作，标签包含旧V的bootstrap估计；它既不是关系真假标签，也不是精确长期价值。VLM、合同、缓存内容和名义规则不接受这些梯度。[S2，§11、§13]

**本方法学习先验的条件化决策用途，不学习修改先验。**同一关系模式可以在不同情境下增加候选分数、影响接近零或产生负向修正；架构提供这种能力接口，实际是否学到由实验判断。缺少某条图边不意味着策略必然学不会相应行为，因为观测、历史与交互经验也可能包含线索。[S1，§26、§34]

研究定义、主要公式与训练权限已经足够进入论文写作。当前实验要验证结构表示和辅助监督的增量，而不是决定是否允许写Method。具体平台、安全、数据和运行环境仍须完成实际绑定；方法可写不等于系统已部署。

# 2. Research Problem

## 2.1 Problem

给定技能集合$\mathcal U$、合同集合$\mathcal K$、独立目标$\mathcal T$、当前及历史观测和初始化时缓存的软关系$R_e$，学习高层策略$\pi_\theta$，以固定低层控制器执行技能，在有限实际任务时间内最大化折扣成功回报。技能可以失败，事实可以失效或变为UNKNOWN，任务允许多个合法选择而不是唯一预定序列。

“固定”有两层含义：合同与VLM参数在训练期间保持不变；每个episode的软边版本在该episode内固定。不同初始任务与场景可以有不同缓存关系；真实事实、执行历史和策略参数在各自允许的阶段更新。固定关系不等于永远固定状态或输出。

## 2.2 Research question

**候选名义结构干预及其合同—先验交互，能否在匹配可用信息与交互预算时，形成比只读取当前图更有用的技能决策表示？**研究进一步比较合同干预、VLM增量、纯交互与四视图融合、Q辅助及必要的有界残差作用，而不把全部效果归给单一模块。

## 2.3 Core hypothesis

名义效果提供可计算的候选差异，关系编码可把局部事实变化与后续技能、目标相联系，双差分组织软关系对这种响应的影响；真实序贯经验可能使这些特征成为可用的决策依据。这是假设，不预设潜在差分天然具有价值语义，也不预设Full优于所有基线。[S2，§1.3]

# 3. Motivation

长时程技能选择难在局部合法性不足以决定长期效用。先抓哪个对象、是否先打开容器、是否需要临时搬移，可能影响后续可执行性与返工。技能库提供“能调用什么”，合同提供“何时允许调用、名义成功后改变什么”，但不必包含每个场景的全部空间联系和执行风险。技能组合研究已经关注跨技能依赖，不能把“考虑后续影响”本身作为新发明。[R1，R7]

VLM可能根据初始场景提供合同未显式编码的联系，例如一个已注册搬移动作的名义后果可能支持另一对象的后续放置。但语义上可能成立的关系，未必对当前选择有用：目标可能已完成、准备动作成本可能太高、执行条件可能不利。关系语义正确性与情境中的决策用途因此必须分开。

核心矛盾是：直接把软关系作为硬计划或启动要求会给予它过高权限；完全忽略又可能失去有用联系；仅读取整张增强图则不显式分离“候选会改变什么”与“软关系怎样改变对这一变化的解释”。CP-DISR选择候选条件化表示：先在合同上构造变化，再让先验影响变化的编码，而非直接让VLM输出动作优先分。

例如OPEN的名义效果使Open(box)成立，合同本身即可联系到后续PLACE；这属于合同推理，不能重复包装成VLM知识。真正额外的场景联系可以是MOVE(red,buffer)可能支持PLACE(blue,box)，前提是MOVE、其效果及对象绑定已经存在。该软关系不把PLACE成功、ClearPath真值或新的机器人能力写入系统。[S2，§5.3、§9.6；S3，§1.2]

# 4. Paper Positioning

正式名称保留 **CP-DISR：Contract–Prior Differential Intervention for Skill Reasoning**。论文题目采用：**CP-DISR: Contract–Prior Interaction Contrasts for Long-Horizon Skill Selection**。M1只作内部代号；全文将intervention限定为nominal symbolic intervention。

中心问题是：**在相同技能合同与观测条件下，让固定语义先验通过“候选名义变化的编码响应”影响策略，是否比不做干预或直接读取四视图更适合长期技能选择？**固定不完美先验仍是问题背景，candidate-conditioned prior interaction是算法组织中心。

关键观察不是“关系只有改变物理后果才有价值”。输入关系不修改环境动力学，也不修改合同的名义patch。更准确地说，关系的决策用途依赖候选、状态、目标与后续选择；本文检验一种有意限制：先验只经候选变化的表示交互参与决策。静态信息可能有用，A_CAT直接检验排除它的代价与收益。

**One-sentence story.** We study whether restricting fixed semantic priors to their interaction with candidate-induced nominal representation changes improves skill selection under matched observations, contracts, and interaction budgets.

主方法没有显式逐边utility、关系真假或知识修订模块。不能将集合R的编码交互写成已经识别“每条关系的真实贡献”。固定cache隔离生成与利用并支持复现；在线重新query并不必然破坏on-policy，故不以这一错误理由论证固定cache。[W05，W14；本轮设计]

# 5. Core Contributions

本文以**一个统一设计、两个方法层面的贡献方面**组织，不把名义后继、减法和辅助Q各自包装成独立新算法。

## C1. Matched contract–prior interaction representation

我们采用已有afterstate/后继状态思想，在合同/增强与当前/名义后继两个维度建立共享、目标对齐的四视图，并用interaction contrast组织固定先验对候选变化的编码影响。新颖性主张落在该表示及其特定用途，而非后继状态计算或2×2代数本身。[W03，W04]

**English.** We develop a matched contract–prior representation that evaluates the same nominal candidate change with and without fixed semantic relations, yielding a candidate-conditioned interaction contrast.

## C2. Pure-interaction policy design and outcome-grounded implementation

将该交互作为先验的唯一直接特征入口，以零锚定有界残差调制合同策略；给出精确null、可加分离抵消、单步概率比界与置换条件，并以真实执行支持的PPO及辅助候选Q训练共享表示。性质证明设计约束，不证明性能；辅助Q是训练机制而不是第三个原创学习范式。[W16–W19]

**English.** We restrict direct prior modulation to the interaction representation, with an exactly zero-anchored and bounded policy residual, and train the shared decision features using actual executions.

## 实证贡献的位置

有真实结果后，第三点写成**对应任务与预算范围内的实证发现**：受控合同阶梯、四视图融合对照、同checkpoint先验变化和Actor依赖共同说明设计何时有用。没有数据时只写评价问题，不把常规baseline/消融操作宣称为新的evaluation protocol。

当前不是“只有一个公式的论文”：名义轴、matched views、共享坐标、信息限制、执行权限、有限影响和实际训练是一个可检验的整体设计。但组件组合是否值得论文主张，仍须A_CAT等证据，不能由组合复杂度推出新颖或有效。

# 6. System Assumptions

## 6.1 任务与能力范围

主版本针对有限对象、固定技能词表和有限谓词的组合操作。每个高层动作是技能schema与有序参数绑定；连续抓取姿态和运动轨迹由共享的固定参数生成器及低层控制器给出。未实际注册的技能不能由VLM或论文例子扩展成可执行能力。

前置条件和目标使用带符号的合取；有限条件效果与派生规则必须事先注册。一般嵌套OR、量词或复杂条件语法不被默认为支持。它们若不可语义保持地编译到当前接口，应在任务绑定时报告不支持，不偷偷改变目标。

## 6.2 输入与权限

决策只使用当时已经到达的第三视角RGB-D、本体状态、对象绑定、感知特征、事实记录、执行摘要和任务剩余时间。当前MVP使用冻结感知前端和固定Verifier，可训练的是其后的观测融合与GRU。仿真隐藏真值可以用于明确隔离的独立评价或状态诊断，不能暗中用于Actor、候选参数或Verifier。[S2，§4、§12]

安全检查独立运行。关键前置为UNKNOWN时暂不许可相应技能；它不等于世界中一定不可行。等待或观察必须是已注册且有界的安全技能；没有允许动作时使用独立安全终止，不放开mask进行探索。

## 6.3 三层信息结构

| 层级 | 内容 | 回答的问题 |
|---|---|---|
| Information Source | 合同、观测、经验证事实、固定VLM软关系 | 系统有什么证据与接口可用？ |
| Structural Representation | 当前图、名义视图、共享R-GCN、$D^K/D^P$ | 信息如何围绕同一个候选组织？ |
| Decision Learning | Actor、独立V、Structural Q、PPO | 这些表示何时应影响技能选择？ |

真实结果既使运行时重新确认事实，也用于训练神经模块；两种更新不同于知识源修订。即使后两层可以学习，缺少必要可辨认证据或必要动作仍可能形成硬瓶颈。

# 7. Skill-Level Semi-MDP

## 7.1 状态、观测与候选

设底层物理状态为$s_\tau$，高层第$t$次决策发生在$\tau_t$。实际观测与基础决策输入定义为

$$
o_t=(I_t^{ext},D_t^{ext},p_t,\ell_{t-1}^{exec},T_t^{remain}),\qquad
x_t=(o_t,\mathcal T,F_t,L_t^{summary}).
$$

$F_t$为当前三态事实，$L_t$为实际执行记录。任务$\mathcal T$包含实体、带符号目标、明确约束、deadline和独立评价器版本。Episode级动作实例集合为$\mathcal A_e$，当前候选表为$\mathcal A_t$，执行mask为$m_t$。已注册动作、对象与参数由固定程序实例化，mask只由真实能力、合同与安全条件确定。

令$X_t$表示带不可变引用的决策snapshot，包括$x_t$、图模板、固定先验版本、候选、mask和goal_refs。策略可以依赖当前及历史信息；下文$V(X_t)$、$Q(X_t,a)$省略相应历史前缀。有限状态摘要不被声称必然构成完整马尔可夫状态；GRU的具体实现见Method。技能的可变执行时间采用SMDP/Options形式，而信息接口仍可能部分可观测。[R1]

## 7.2 时间与奖励：共同折扣半衰期

**Final Duration Discount Rule：**以秒为唯一日志换算单位，指定半衰期$H>0$，

$$
\boxed{\Gamma_t=2^{-d_t/H},\qquad \bar\gamma_{1s}=2^{-1/H}.}
$$

技能内真实奖励事件发生在偏移$\epsilon_k$时，$r_t=\sum_k2^{-\epsilon_k/H}\rho_k^{env}$。控制tick先用实际tick秒数换算，不将高层技能次数冒充物理秒。事件奖励只在首次独立确认整项任务成功时为1，其他事件0；子目标、恢复、图变化与VLM判断均不另给分。

**H的确定算法而非无限调参：**Stage0A在每个正式训练family取得5条合法成功的固定reference执行，最多尝试10条，使用独立development案例和已授权固定技能；取实际完成时间中位数$T_f^{ref}$。该校准不向PPO buffer、BC或VLM few-shot提供轨迹。一个比较suite使用共同

$$H=\max_{f\in\mathcal F_{train}}T_f^{ref}.$$

因此每个family典型reference完成时的保留率至少为0.5。同family全部方法、seed与消融共用H；跨family训练和组合holdout也继承该suite H。只有一个family时H等于其reference中位数。reference无法执行时暴露真实接口/任务绑定问题，不猜H，也不阻塞Method写作。

0.5是可解释的时间偏好，不是理论最优阈值。Claude的$\bar\gamma^{T}\ge0.3$同样只是工程规则，不能从理论推出。运行开始前冻结H和来源hash，不根据Full成绩调gamma；新family需要新suite时登记版本，旧run不静默重算奖励。独立family使用不同H在各自匹配比较中不违反公平，但其discounted return不能直接跨family平均作为同一物理时间偏好，故本版默认共同H。

若某一任务独立标定为30/60/120/180/300秒，对应每秒gamma为0.97715997/0.98851402/0.99424042/0.99615659/0.99769218。该表是公式示例，不是实际已测的任务时长。

## 7.3 目标与结束标志

$$
J(\theta)=\mathbb E_{\pi_\theta}\!\left[\sum_t w_t r_t\right],
\qquad w_0=1,\quad w_{t+1}=w_t\Gamma_t.
$$

目标/报告准则是时间折扣下的任务成功，不等同完全忽略时间的成功概率。第17.2节使用当前采样分布的经验PPO surrogate，不宣称其无偏优化本J。真实成功、任务期限耗尽、安全终止及定义明确的不可恢复失败属于terminated；单技能失败或超时通常只是一次转移，若任务仍可继续就不终止episode。

有效外部截断与rollout切段需要在最后有效状态bootstrap，不能使用reset后的观测；跨reset不递推优势。无有效末状态的基础设施异常单独标注，不能补造下一状态。这些边界与常见环境的termination/truncation区分一致，但具体任务结束规则必须绑定。[R15]

## 7.4 符号索引

| 符号 | 含义 |
|---|---|
| $\mathcal U,\mathcal K,\mathcal P,\mathcal O_e$ | 技能库、合同、谓词注册表、episode对象目录 |
| $F_t,\mathcal T,R_e$ | 当前三态事实、独立任务、固定软关系 |
| $G_t^K,G_t^H,\widetilde G_t^{K,i},\widetilde G_t^{H,i}$ | 合同/增强当前与候选名义后继四视图 |
| $Z_t^K,Z_t^H,D_i^K,D_i^H,D_i^P$ | 结构编码、两种变化及其交互差分 |
| $z_t^o,h_t,c_i,u_i^K,u_i^P$ | 观测特征、历史状态、候选上下文及两分支特征 |
| $b_i,\Delta_i,\ell_i,\beta$ | 合同评分、先验修正、总评分、修正上界 |
| $V_\omega,Q_\eta,\operatorname{sg}$ | 状态价值、辅助动作价值、停止梯度 |
| $\Gamma_t,w_t,\kappa_t,c_t$ | 时长折扣、起点折扣权重、bootstrap标志、连续记录标志 |

# 8. Skill Contract

## 8.1 定义与来源

技能合同$K_k$是schema、参数、类型约束、启动条件、名义效果、执行结束与验证协议及来源版本的集合：

$$
K_k=(\sigma_k,\mathcal A_k,\mathcal C_k,\mathcal E_k,\mathcal B_k,\mathcal V_k,\mathcal D_k,\mathcal P_k).
$$

$\mathcal C_k$包含带正负符号的启动前置，$\mathcal E_k$包含ADD、DEL与注册UNKNOWN效果，$\mathcal P_k$记录来源与版本。$\mathcal B_k$区分正常退出、失败、中断，$\mathcal V_k$是后置事实验证规则，$\mathcal D_k$是实际规划/执行/验证时限。成功分支语义并不宣称每次启动都成功；正常控制器退出也不自动构成已验证成功。

合同来源固定为“已有控制器接口＋研究者一次性明确规格＋执行级校验”。一次定义library-level合同；episode开始按对象类型自动grounding，如PLACE(?object,?target)绑定为PLACE(red,box)。合同不是正确任务顺序，不由VLM生成可信定义。[S2，§5]

## 8.2 可直接实现的字段

```yaml
schema_version: skill_contract_v1
skill_schema: PLACE
arguments:
  - {name: object, type: movable_object}
  - {name: target, type: openable_container}
controller_ref: MUST_BIND
preconditions:
  required_true: ["Held(object)", "Open(target)"]
  required_false: []
  execution_checks: [binding_resolved, parameters_available, safe_to_start]
nominal_success:
  add: ["Inside(object,target)", "GripperEmpty()"]
  delete: ["Held(object)"]
  unknown: []
  conditional_effects: []
frame_assumption: unrelated_registered_facts_unchanged_nominally
termination:
  normal: [controller_finished]
  failure: [execution_failed, verification_failed]
  interrupt: [safety_stop, critical_fact_lost]
verification:
  postconditions: ["Inside(object,target)", "NotHeld(object)"]
  rule_ref: MUST_BIND
time_limits_seconds:
  planning: MUST_BIND
  execution: MUST_BIND
  verification: MUST_BIND
provenance:
  source: controller_interface_and_researcher_review
  version: MUST_BIND
  validation_record: MUST_BIND
```

字段名为本文统一的数据契约示例；与既有项目不同命名的字段通过显式adapter映射，不能只改名称而改变语义。执行启动器拒绝未绑定的控制器、验证和时限。

## 8.3 例子与约束

| 技能示例 | 正前置 | 名义ADD | 名义DEL |
|---|---|---|---|
| PICK(o) | GripperEmpty、OnTable(o) | Held(o) | GripperEmpty、OnTable(o) |
| OPEN(c) | GripperEmpty | Open(c) | 无 |
| PLACE(o,c) | Held(o)、Open(c) | Inside(o,c)、GripperEmpty | Held(o) |

这些例子沿用原规范的受限接口，不是任意机器人的能力保证。OPEN是否真的要求空夹爪取决于实际控制器；放回、搬移、等待等只有真实注册后才可以出现。重抓与PICK合同相同就复用schema，不凭名称添加新恢复能力。

有限条件效果的guard在原始$F_t$同时求值；TRUE应用，FALSE不应用，UNKNOWN时仅保留各可能分支共同确定的赋值，其余受影响事实置UNKNOWN。冲突ADD/DEL/UNKNOWN赋值与单夹爪等不变量在注册和patch阶段检查。未知条件不自动取最有利分支，也不展开未声明的概率树。

# 9. Fact Verification

事实链为：实际观测、本体与执行日志，经冻结感知/跟踪和几何/时间规则，形成TRUE、FALSE或UNKNOWN，再提交不可变决策snapshot。VLM关系和名义patch不作为事实真值证据。[S2，§12]

例如Held使用夹爪、对象相对位置、随动及历史证据；Inside使用对象估计几何与目标区域；Open使用真实开度；LossDetected需区分先前已确认持有、非主动释放和当前可靠分离。控制器返回NORMAL不等于Inside已经为TRUE，未看到对象也不等于对象不存在。

```yaml
fact_id: p:Held:red
value: UNKNOWN
reason: OCCLUDED
last_confirmed_value: TRUE
capture_time: MUST_BIND
available_time: MUST_BIND
evidence_refs: [frame_id, robot_state_id]
verifier_version: MUST_BIND
hold_epoch: MUST_BIND
```

三值逻辑中$\neg U=U$；正前置要求T，负前置要求F，U不满足任何一方。历史确认与当前值分开保存，过期或冲突证据使当前值变为UNKNOWN，不删除历史。已有成果也可在新证据下从TRUE变为FALSE；不存在永久累计的“曾经完成就算完成”。

低层安全高频运行，事实按实际感知节奏更新，高层在技能结束/中断及有界验证后提交状态。不同frame ID和真实时间跨度构成持久化证据；不能重复读一帧假装多帧确认。观测的采集时间和可用时间分别记录，迟到数据不回写历史PPO输入。

# 10. Contract Graph

## 10.1 图对象

采用SLG式grounded Action–Proposition表示。本文使用其动作/命题及PRE/ADD/DEL关联思想，并适配三态事实和目标读出；不继承原文所有表示或实验结论。[R2]

$$
G_t^K=(V_A\cup V_P,E_K,X_t^{node},\mathcal M_e),
$$

其中$E_K$包含PRE_POS、PRE_NEG、ADD、DEL及各自独立的反向消息类型，$\mathcal M_e$保存节点索引、合同引用、对象绑定和固定goal_refs。PRE为命题→动作，ADD/DEL为动作→命题；反向消息只服务编码，不反转逻辑效果。

Object不额外建节点，参数绑定和外部对象特征表达对象；Goal是命题上的goal sign以及读出索引，不单独造Goal节点。普通合取由合同程序精确求值，不新增通用Logic节点。

## 10.2 Grounding与状态更新

首先按参数类型和静态约束枚举动作，将所有前置和名义效果命题注册为稳定节点，再加入goal命题。模板不依赖当前真假和VLM建议；保留FALSE、UNKNOWN、已完成目标及当前不可执行动作。每次决策只绑定当前真实事实特征和mask，不根据任务进度删掉节点。

PICK(red)与PICK(blue)共享PICK的参数化规则；Held(red)与Held(blue)共享Held。实例对应由有序参数及连接确定，不把任意对象编号作为可学习语义索引。参数共享思想与ASNets相关，但本方法不复制其专用交替网络或规划器教师训练。[R3]

## 10.3 两对象示意

```text
PICK(red) --ADD--> Held(red) --PRE_POS--> PLACE(red,box)
                                                |
                                               ADD
                                                v
                                      Inside(red,box) [goal+]

OPEN(box) --ADD--> Open(box) --PRE_POS--> 两个PLACE实例

PICK(blue) --ADD--> Held(blue) --PRE_POS--> PLACE(blue,box)
```

完整图还含DEL、GripperEmpty及反向消息边，图式只为阅读省略。要求Inside为目标并不把当前Inside设为TRUE；初值必须有实际观测或明确诊断输入支撑。单个开放条件只使某个前置成立，不保证全部PLACE前置或几何可行性同时成立。

# 11. VLM Semantic Prior

## 11.1 冻结输入与输出

输入为任务与目标、初始场景RGB、允许的事实摘要、对象目录、已实例化技能ID与合同摘要、goal映射和固定schema/few-shot。VLM只在初始化为任务—场景生成一次软关系；有限初始池可预生成，PPO期间只读cache。它不得获得真实未来、正确整条动作序列、最优Q或测试成绩。[S2，§7；S3，§1.1–1.2]

当前候选快照保留`qwen3.8-max-0902`。截至2026-09-21，官方页面列出该快照、图像理解及非思考/结构化输出相关能力；这是文档核对，不是账户授权或本项目调用成功证据。实际region、endpoint、SDK行为和权限为MUST_BIND。[R16–R17]

## 11.2 唯一关系词表

| 类型 | 端点 | 语义 |
|---|---|---|
| SOFT_SUPPORTS | Action A→Action B，A≠B | A某项已注册名义后果在当前场景中可能支持B后续执行 |
| SOFT_RELEVANT_TO_GOAL | Action A→已有goal proposition g | A某项名义后果可能影响带符号目标g的实现、保持或恢复 |

两者都不是充分可行、成功承诺、硬前置或静态排序。第二类不预设影响为正。`SOFT_ORDER`不属于v2.1。每条关系必须有`effect_fact_ref`：引用source的注册ADD/DEL效果命题，且已有source→该命题的ADD/DEL邻接。纯UNKNOWN且无该邻接的命题不作软关系锚点，名义干预本身仍支持UNKNOWN。[S3，§1.2；D1]

`effect_fact_ref`仅用于出处、引用校验和审计，不是新节点、hyperedge、真值赋值、独立神经特征或新硬边。锚点合法不证明软关系语义正确，也不要求该事实在所有后续状态都变化。

```json
{
  "schema_version": "m1_soft_relations_v2",
  "relations": []
}
```

空集合是合法输出。接口包提供完整JSON Schema；另须做注册表语义校验，字符串合法不等于Action、Fact或Goal真实存在。

## 11.3 校验与去重

流程固定为raw→JSON→schema→ID/type→source效果与邻接→局部合同冗余→精确去重→明确冲突隔离。相同(type,source,target)只保留一条；多个合法效果引用按规范化ID排序保留一个用于审计，不产生同型平行边以重复加权。

局部冗余只移除明确同义的直接合同联系，例如source ADD p且target PRE_POS p，或source DEL p且target PRE_NEG p，并引用同一p。源动作直接改变目标命题的同义goal关系也去重。不求任意逻辑等价，不把所有有合同路径的关系一概删除。不存在ID、支持自指或违反独立明确禁止项的关系隔离；剩余关系仍允许语义错误。[D1]

## 11.4 Cache合同

缓存key为规范化输入的SHA-256，至少含task、初始图像及预处理hash、bindings、允许ID、合同/谓词版本、goal映射、模型snapshot、prompt/few-shot/schema版本、解码参数、地区与接口配置。不同场景不能仅因task名称相同复用cache。

保存raw response、解析关系、拒绝项、去重原因、输入引用、request ID、模型、token/结束原因（服务实际返回时）、解码参数和hash。格式/传输最多一次受控重试，不因语义不满意重问；合法空输出不重试。两次传输失败记录API_FAILURE，不冒称模型返回了空关系。

起步配置为temperature=0、最多8条关系、输出上限2048 tokens、非思考JSON对象，无网页搜索或工具调用；完整prompt与开发few-shot必须落盘。若协议或快照改变，升cache版本并保留旧文件；不按RL测试成绩重新生成有利输入。

## 11.5 增强图

$$
V_t^H=V_t^K,\qquad E_t^H=E_K\cup E_{soft}^e.
$$

$G_t^H$与$G_t^K$具有同节点、同事实、同目标和同观察来源，只多独立类型的软消息边及其逆向边。Action→Action软边使增强图不必严格二部，但不改变合同语义。去重后软边为空时直接复用合同编码，得到严格零先验交互。

## 11.5 Consequence-grounding admission metadata

`effect_fact_ref`只校验已有source名义ADD/DEL入口，提供provenance；不是特定effect的hard gate，也不证明语义关系正确或一定在DP中非零。当前网络以skill节点为软边source，不能宣称其只解释引用p而忽略其他效果。A有多个效果且p已满足时，其他变化仍可经A传播。正文不以随机初始化范数说明效果级归因成立。

SOFT_ORDER删除是因为本schema聚焦后果关联而非order-only建议，不是因为静态Action→Action边必在DP中抵消。本轮不改prompt/schema，保留缓存协议；论文中可能不适用的关系不等于实现了逐效果关闭开关。

# 12. Candidate Nominal Intervention

## 12.1 名义而非物理预测

对允许候选$a_i$，构造

$$
\widetilde F_t^i=T_{\mathcal K}^{nom}(F_t,a_i).
$$

这一步只问：**若技能按给定名义成功分支完成，哪些注册事实改变？**不预测成功概率，不生成未来图像，不克隆环境执行，不调用VLM模拟，不训练世界模型。实际失败和隐藏副作用由之后的真实转移体现。[S2，§8]

ADD设临时TRUE，DEL设FALSE，未知效果设UNKNOWN；未受影响事实按名义frame assumption保留。不得更新真实Fact Store、观测、时钟、attempt count、execution ID、证据置信度、目标、奖励或done。临时Held=TRUE不是一次传感器确认。

## 12.2 算法：只读名义patch

```text
NOMINAL_PATCH(F, a, K, registry):
  1. 检查a已注册、参数已绑定、F满足其明确逻辑前置。
  2. 所有条件效果的guard只在原始F上同时求值。
  3. 合并ADD / DEL / UNKNOWN赋值；冲突直接报错。
  4. 建立不可变overlay，不修改F的存储对象。
  5. 只按已注册派生规则重算受影响命题。
  6. 校验不变量，输出实际有变化的稀疏patch。
```

同一个patch应用于两种结构：

$$
\widetilde G_t^{K,i}=\operatorname{View}(G_t^K,\widetilde F_t^i),\qquad
\widetilde G_t^{H,i}=\operatorname{View}(G_t^H,\widetilde F_t^i).
$$

当前mask只用于选择真实可执行候选；临时视图不重写历史mask或新增假想执行许可。合同求值与真实启动mask不一致时记录接口错误，不免费改选其他动作。

# 13. Shared Structural Encoder

## 13.1 节点特征与信息位置

标准结构输入为

$$
h_a^{(0)}=e_A+e_{\sigma(a)}+f_A(\operatorname{argroles}(a)),
$$

$$
h_p^{(0)}=e_P+e_{\rho(p)}+
 f_P^{node}(\operatorname{onehot}(F_t(p)),\operatorname{goalsign}(p),\operatorname{argroles}(p)).
$$

这里$f_P^{node}$与后文先验候选读出$\phi_P$不同。schema参数共享，实例ID用于稳定引用而不成为任意学习标签。

| 信息 | 冻结v2.1中的位置 | 名义视图处理 |
|---|---|---|
| skill/predicate schema、参数角色 | 图节点特征 | 不变 |
| 三态事实与goal sign | 命题特征 | 仅注册事实按patch改变；goal不变 |
| 实际current applicability、执行检查 | 候选状态与mask | 保留真实快照，不随假设生成新许可 |
| 实际对象几何、外观与观测质量 | 共同观测及候选上下文 | 四视图使用同一来源，不假造未来几何 |

**兼容说明。**本轮附件列出的观测关联和可执行性信息必须被说明，但既有冻结定义把它们主要置于观测/候选通路。本文不静默把它们新增为可随名义干预变化的节点通道；若实际实现另有观测附着特征，必须声明新feature profile并保持四视图同值。当前主profile维持上式。[S1，§18；S2，§6.2、§9.2、§10.1]

## 13.2 R-GCN

使用标准关系卷积，再接既定ReLU和逐节点LayerNorm：

$$
\bar h_v^{(l+1)}=
W_0^{(l)}h_v^{(l)}+
\sum_{r\in\mathcal R}\sum_{u\in\mathcal N_r(v)}
\frac{1}{|\mathcal N_r(v)|}W_r^{(l)}h_u^{(l)},
$$

$$
h_v^{(l+1)}=\mathrm{LayerNorm}(\operatorname{ReLU}(\bar h_v^{(l+1)})).
$$

关系区分使PRE、ADD、DEL和软联系可以具有不同变换。采用现成RGCNConv、root transform和basis分解；无邻居关系项为0。所有关系类型在初始化登记，缺边不改变参数集合。[R4–R5]

默认hidden=128、4层、4 bases、mean、dropout=0。四视图共享同一个$E_\psi$，不分K/H参数、不使用不同BatchNorm统计或分支embedding。GNN不是精确AND求值器，启动逻辑由合同程序负责。有限层数不保证任意长证明。

## 13.3 Goal-aligned readout

设本episode目标为$(p_j,s_j)_{j=1}^m$：

$$
z_j=f_g(h_{p_j}^{(L)},e_{sign}(s_j)),\quad
z_0=f_0\bigl(\operatorname{Mean}(H_A),\operatorname{Mean}(H_P),\operatorname{Mean}_j z_j\bigr),
$$

$$
E_\psi(G\mid\mathcal T)=Z=[z_0;z_1;\ldots;z_m]\in\mathbb R^{(m+1)\times d}.
$$

$z_0$是读出而不是新节点。goal_refs固定，完成后也不删行；跨任务目标数不同则padding并mask。当前与名义、合同与增强视图保持同一坐标和行语义，使相减是对应目标的比较。$E(G)$只是学习到的结构表示，不是成功率、任务价值、成本或剩余技能数；不人为给latent每维命名。

# 14. Contract–Prior Interaction Contrast

## 14.1 四视图与同一个表示空间

在固定实际观测、目标、节点索引、关系R和编码器参数下，记

$$
Z_{00}=E(F,\varnothing),\quad Z_{10}=E(\widetilde F_i,\varnothing),\quad
Z_{01}=E(F,R),\quad Z_{11}=E(\widetilde F_i,R).
$$

第一个下标是当前/名义后继，第二个是无/有软关系；K合同始终存在，空集合仅指没有VLM软边。

$$
D_i^K=Z_{10}-Z_{00},\qquad D_i^H=Z_{11}-Z_{01},
$$

$$
\boxed{D_i^P=Z_{11}-Z_{01}-Z_{10}+Z_{00}=D_i^H-D_i^K.}
$$

相减对象是目标对齐的表示，不是邻接矩阵、真值表或真实环境结果。该定义与v2.1逐项相同；正式名称用**representation interaction contrast**，实现继续沿用D_K/D_H/D_P。

## 14.2 对称解释

仅为推导定义$S_R(F)=E(F,R)-E(F,\varnothing)$，则

$$
\boxed{D_i^P=S_R(\widetilde F_i)-S_R(F).}
$$

因此可以读成“软关系怎样改变候选变化的编码响应”，也可以读成“候选名义变化怎样改变软关系的编码作用”。S不是新网络、训练目标或缓存，也不必另造prior signature品牌名。

这是共享网络计算的2×2有限交互对比，不是从观察数据识别处理效应的causal difference-in-differences；没有平行趋势假设，也没有将表示坐标解释为世界因果变量。事实patch本身不受R影响。

## 14.3 有意限制与不能过度宣称的性质

若$E(F,R)=A(F)+C(R)$，则$D_i^P=0$。消去的是对所比较变化可加分离的表示成分，不是所有拓扑静态边，也不是所有可能有用的行动先验。

在线性诊断$E(F,R)=M_Rx(F)+b_R$中，

$$D_i^P=(M_R-M_\varnothing)[x(\widetilde F_i)-x(F)],$$

边不变仍可改变变化消息的传播。SOFT_ORDER在数学上也可能非零，主schema删除它是因order-only职责不合，而非必被消去。

完整$(Z^K,D^K)$与$(Z^K,\widetilde Z^K)$可逆，但当前Actor只把Pool$(Z^K)$放进上下文，并继续压缩候选特征。因此不能据上述可逆恒等式声称整个网络对当前/后继信息无损；也不能因$D^H=D^K+D^P$而称A_DD与Full必然同信息。

目标局部没有路径时，全局非线性读出仍可能混合不同组件。非零DP不是关系有用性的证明；DP大小与动作概率变化也不是单调关系。

# 15. Policy Architecture

## 15.1 观测、记忆与候选上下文

冻结前端的真实特征经可训练融合器和GRU：

$$
(z_t^o,h_t)=f_{obs,\phi}(x_t,h_{t-1}),\qquad
c_{t,i}=[z_t^o,e(a_i),\operatorname{Pool}(Z_t^K)].
$$

GRU只读实际观测、任务、核心事实与执行摘要，不直接读取VLM图embedding。候选$e(a_i)$含共享技能embedding、有序对象/目标角色、真实合同状态与允许的当前几何。保留当前$Z^K$的池化上下文，保留一部分绝对任务背景但不宣称全量无损。[S2，§10.1]

## 15.2 合同主分支

$$
u_{t,i}^K=\phi_K(c_{t,i},D_{t,i}^K),\qquad
b_{t,i}=w_K^\top u_{t,i}^K+b_K.
$$

$\phi_K$对固定goal行作候选查询与池化，再融合上下文。它不是observation-only分支，而是合同推理分支。其分数不直接等于Q，不受先验幅度上界$\beta$限制。

## 15.3 零锚定先验分支

$$
\boxed{u_{t,i}^P=
 \phi_P(c_{t,i},u_{t,i}^K,D_{t,i}^P)
 -\phi_P(c_{t,i},u_{t,i}^K,0).}
$$

$$
\boxed{\Delta_{t,i}=\beta\tanh(w_P^\top u_{t,i}^P),\qquad
\ell_{t,i}=b_{t,i}+\Delta_{t,i}.}
$$

两次$\phi_P$共享参数、上下文、goal mask和确定计算。$w_P$之后无独立bias、无额外无界倍率。因此$D_i^P=0$时$u_i^P=\Delta_i=0$。禁止旁路读取当前$Z^H$、增强图候选局部embedding、静态VLM分数或抽样模式标签。

$\Delta_i\in[-\beta,\beta]$允许正向利用、接近零和负向修正。这里的“有用”是当前决策用途，不是关系语义真伪。共享参数仍可以依据不同输入产生不同响应，但同类型边可能发生梯度干扰，训练也可能主要依赖其他特征；上述能力不能写成“自动识别并拒绝所有错边”。[S1，§26–28]

## 15.4 执行mask与策略

$$
m_t=\mathrm{ExecMask}(\mathcal A_t,\mathcal K,F_t,\mathcal S),\qquad
\pi_\theta(a_i\mid X_t)=
\frac{m_{t,i}\exp(\ell_{t,i})}{\sum_j m_{t,j}\exp(\ell_{t,j})}.
$$

实现用masked log-softmax。mask来自真实能力、参数、合同和安全规则，不能被VLM关系放宽。全部mask为0时不构造非法Categorical，不伪造默认动作；交由独立终止规则记录。图内保留不可执行动作供消息传播，但只为真正允许的候选构造名义视图。

## 15.5 Properties Box：适用条件与精确结论

共同条件：固定参数、相同真实观测与历史前缀、相同候选与执行mask、相同goal/node索引、相同确定性编码；只改变明确比较的输入。softmax只在至少一个合法候选上定义。

**P1 Exact null.** 若$R=\varnothing$，复用合同编码：$D_i^P=u_i^P=\Delta_i=0$对全部候选成立，因而$\pi_\theta=\pi_K^\theta$。$\pi_K^\theta$是**同参数**关闭先验修正的策略，不是独立训练B2。

**P2 Empty nominal patch.** 某候选$i$的全部编码输入未变时，$D_i^K=D_i^H=D_i^P=\Delta_i=0$。这只使该候选无直接残差；其他候选的非零残差仍可经softmax分母改变它的概率。只有全部候选满足null时才得到整项policy equality。effect_fact_ref对应p不变不等于整个patch为空。

**P3 Additive separability.** 若对当前与名义输入有$E(F,R)=A(F)+C(R)$，则$D_i^P=0$。不要求线性E，也不推得静态edge一律无影响。

**P4 Bounded direct deviation.** $\Delta_i=\beta\tanh(w_P^\top u_i^P)$，$\beta\ge0$，合法动作有

$$
\frac{\pi_\theta(a_i\mid X)}{\pi_K^\theta(a_i\mid X)}
=\frac{e^{\Delta_i}}{\sum_j\pi_K^\theta(a_j\mid X)e^{\Delta_j}}
\in[e^{-2\beta},e^{2\beta}].
$$

因为分子、加权分母都在$[e^{-\beta},e^{\beta}]$。默认$\beta=0.5$时界为$[e^{-1},e]\approx[0.367879,2.718282]$。这是对合同reference的单步界，不是任意两个先验之间同样的界（后者只能直接给出$e^{\pm4\beta}$粗界），也不保证轨迹回报或argmax不变。mask为0的动作不计算0/0比值。

**P5 Candidate permutation equivariance.** 在逐候选共享网络、集合读出置换不变、候选和mask同步置换、无候选位置编码时，$\pi(P\mathcal A,Pm)=P\pi(\mathcal A,m)$。若先执行argmax，并列需按canonical ID解决；分布等变不自动使按数组首项打破并列的动作选择等变。对象重命名是另一项更强检查，不能由候选重排直接推出。

P1/P4/P5也适用于按本版定义的A_CAT；A_CAT不要求P2/P3成立，因为它允许静态表示成分。A_B保留null而移除P4。全部是代数/架构条件，不作为全局安全或有效性定理。

# 16. Auxiliary Candidate Q（实现模块：Structural Q）

## 16.1 三个并行输出头

Actor负责策略分布；V负责“从当前状态继续当前策略”的价值基线；Q负责“先执行指定候选，再继续当前策略”的辅助动作条件化价值。三者不是Actor→V→Q的串行关系，也不以最大Q替代Actor选动作。

$$
\bar u_t^K=\operatorname{MaskedMean}_i u_{t,i}^K,\qquad
\bar u_t^P=\operatorname{MaskedMean}_i u_{t,i}^P,
$$

$$
V_\omega(X_t)=f_V(z_t^o,\operatorname{Pool}(Z_t^K),\bar u_t^K,\bar u_t^P),
$$

$$
Q_\eta(X_t,a_i)=f_Q(u_{t,i}^K,u_{t,i}^P,z_t^o,\bar u_t^K,\bar u_t^P).
$$

集合汇总只含有效候选，不含padding。Q读取候选集合背景；空patch候选仍可凭观测、身份和后续选择背景得到非零价值。不用$V=\sum_i\pi_iQ_i$，因为未执行候选Q没有当前样本标签。[S2，§11.1–11.2]

## 16.2 实际动作的监督

令$\kappa_t=1-\mathrm{terminated}_t$：

$$
\boxed{y_t^Q=\operatorname{sg}
\left[r_t+\Gamma_t\kappa_tV_{old}(X_{t+1})\right],}
$$

$$
\mathcal L_Q=\mathbb E_t\left[
\operatorname{Huber}\bigl(Q_\eta(X_t,a_t)-y_t^Q\bigr)\right].
$$

只gather实际执行动作$a_t$，不为未执行动作填0或虚拟成功标签；不使用名义后继V、VLM分数、max-Q或DQN式目标。真实失败也进入监督。旧V与target在该批更新固定、detach；这是实际转移支持的bootstrap估计，不是精确长期结果。Huber回归同样不构成无偏真实价值定理。

## 16.3 作用与边界

Q误差经共享特征更新候选读出、R-GCN与观测编码，但不直接更新Actor最后的评分头。它可能促进与实际动作结果相关的表示，也可能与其他学习目标干扰；Q loss下降不能单独证明策略变好。共享特征可使未执行候选受到间接梯度影响，这不等于它们获得了真实标签。

Q没有监督“哪条边错了”，也不能可靠分解VLM错误、感知错误、控制器失败、探索噪声与旧V误差。模型可能通过多处参数降低误差，而不一定压低某条关系的影响。该head属于actual-outcome-supported candidate-value训练机制，不单列原创C3，更不是prior correction。[S1，§30–31]

# 17. PPO / Training Objective

## 17.1 Duration-aware GAE与V目标

使用采样模型的旧V：

$$
\delta_t=r_t+\Gamma_t\kappa_tV_{old}(X_{t+1})-V_{old}(X_t),
$$

$$
\widehat A_t=\delta_t+\Gamma_t\lambda c_t\widehat A_{t+1},\qquad
 y_t^V=\operatorname{sg}[V_{old}(X_t)+\widehat A_t].
$$

$c_t$仅在下一条是同episode、同轨迹且本批可用的连续记录时为1。GAE的$\lambda$按技能决策衰减，$\Gamma$按实际时间衰减；不跨reset递推。真实终止bootstrap为0，外部有效截断或采样切段用最后有效下一状态。GAE是既有估计方法，本项目改变的是时间抽象及输入表示，不将其作为新贡献。[R12，R15]

## 17.2 PPO目标：有效transition均值，不加episode前缀权重

$$
\varrho_t(\theta)=\frac{\pi_\theta(a_t\mid X_t,h_{t-1}^{\theta})}{\pi_{old}(a_t\mid X_t,h_{t-1}^{old})},
$$

令$v_t\in\{0,1\}$标记非padding的有效transition，

$$
\boxed{L_{clip}=\frac{1}{\sum_t v_t}\sum_t v_t
\min\{\varrho_t\widehat A_t,
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)\widehat A_t\}.}
$$

$$
\mathcal L_{total}=-L_{clip}+c_V\mathcal L_V+\lambda_Q\mathcal L_Q
-c_H\mathbb E_{valid}[\mathcal H(\pi_\theta)].
$$

**Final Actor Weighting Rule：**`actor_episode_discount_weight=false`；Actor/V/Q/entropy均按有效transition平均。$w_t=\prod_{j<t}\Gamma_j$保留为实际起点折扣回报核算和诊断字段，不进入Actor loss，不用它重采样，不在别处隐式补乘。GAE的$\lambda$仍按技能决策，$\Gamma$按实际时间；不是把时长折扣删除。

我们采用当前on-policy采样分布上的经验PPO surrogate，并使用时长折扣的优势与价值target。它**不是**episode起点折扣目标$J$的无偏精确梯度，也不能简单重解释成无折扣J的精确梯度。[W01，W02] 每批冻结的surrogate本身可微；Nota–Thomas指出的是省略外层折扣的一般期望更新场通常不是某个固定总体目标的梯度，不是说代码里没有loss。

选择有效样本均值是减少额外时间位置重加权的工程决定。原来prefix加权不属于P0错误；归一化batch中的共同尺度可抵消，V/Q也未因此失去监督，所以不能声称300秒时后段“完全学不到”。统一采用此规则后，旧weighted checkpoint只能标历史profile，不与新profile混作同设置。

其余默认保留：Adam单参数owner，lr3e-4、clip0.2、GAE0.95、cV0.5、cQ0.1、entropy0.01、gradclip0.5、4 epochs、1024转移/rollout、seq16、minibatch64。Actor advantage在有效batch标准化，不标准化V/Q标签；不加入新reward、teacher、pretraining或课程。

## 17.3 80/20先验协议

设$R_e^0$为经过固定解析、校验与去重，但未按测试语义人工纠错的原始关系：

$$
k_e\sim\operatorname{Bernoulli}(0.8),\qquad
R_e=\begin{cases}R_e^0,&k_e=1,\\ \varnothing,&k_e=0.\end{cases}
$$

首次决策前抽样，episode内、跨rollout边界及PPO重算保持不变。模式标签只记录日志，不额外输入网络。原始为空时不补边，实际空prior episode率为$0.2+0.8\Pr(R_e^0=\varnothing)$。80/20是按episode的采样概率，不保证transition或物理时间的比例。[S3，§1.11]

这是所有prior消费者共享的轻量输入缺省协议，不作为独立创新。训练不做删边、target-swap、人工错边、无关边或relation rewrite。训练看到整份缺省，不表示模型已经学会识别所有错误关系；同参数关先验也不等于独立训练B2。

# 18. Forward Information Flow

**图1：完整前向系统图式。**箭头表示数据依赖；同一共享特征产生三个并行head，仅Actor输出动作分布。

```text
Task + Initial Scene + Registered IDs
                 |
             Frozen VLM
                 |
          Soft Relation Cache -----------------+
                                               |
Contracts + Current Verified Facts             |
                 |                             |
          Contract Graph G^K ---- add soft ----> G^H
                 |                             |
                 +----- same candidate i ------+
                               |
                Nominal Fact Patch from Contract
                               |
                     temporary G^K_i / G^H_i
                               |
        current + temporary views -> one shared R-GCN
                               |
                         Z^K, D^K, D^P
                               |
Observation + History -> Obs/GRU -> Candidate Features
                               |
                     +---------+---------+
                     |         |         |
                   Actor       V         Q
                     |
                Independent Mask
                     |
               Selected Real Skill
                     |
              Fixed Skill Controller
                     |
                Real Outcome
                     |
       Verification / Reward / Duration / Next Snapshot
```

初始化阶段ground合同与目标，读取匹配关系缓存并固定本episode版本。每个高层step获取真实事实与候选，先建临时事实/图视图再编码；批处理次序可优化，但信息依赖不能反转。技能执行期间仅低层控制器与独立监测运行，不进行新的VLM规划。执行后Verifier确认新事实，独立评价器计算结果及边界。当前状态变化后重新做下一次候选计算。

$K$个有效候选通常只需当前两视图加每候选两视图，共$2+2N_t$，不是每个候选重复计算两张当前图。缓存相同参数版本内的当前编码是计算复用，不改变上述数据依赖。

# 19. Backward Gradient Flow

**图2：真实转移支持的训练图式。**环境不作为可微分模块；真实记录先生成固定训练目标，再反向传播损失。

```text
Real transitions -> on-policy buffer
          |
old V + reward + duration + termination
          |
     fixed advantage / yV / yQ
          |
   +------+----------------+
   |                       |
PPO/Entropy loss        V loss / Q loss
   |                       |
Actor head              V head / Q head
   +------------+----------+
                |
       shared candidate features
                |
       F_K / F_P, R-GCN, Obs/GRU

No gradient update to:
VLM / cache content / contracts / nominal rules
Graph builder / verifier / fixed controller / evaluator
```

| 参数或组件 | Actor／Entropy | V loss | Q loss | 是否在推理更新参数 |
|---|---|---|---|---|
| 观测融合与GRU | 可更新 | 可更新 | 可更新 | 否 |
| 四路共享R-GCN与readout | 可更新 | 可更新 | 可更新 | 否 |
| 候选编码、$\phi_K/\phi_P$ | 可更新 | 可更新 | 可更新 | 否 |
| Actor最终评分头 | 可更新 | 不直接更新 | 不直接更新 | 否 |
| V head | 不直接更新 | 更新 | 不更新 | 否 |
| Q head | 不直接更新 | 不更新 | 更新 | 否 |
| 旧V与固定target | 否 | 否 | 否 | 否 |
| VLM与关系内容 | 否 | 否 | 否 | 否 |
| 合同、构图、名义规则 | 否 | 否 | 否 | 否 |
| 冻结感知、Verifier、控制器、评价器 | 否 | 否 | 否 | 否 |

“可更新”表示存在梯度路径，不保证每个样本每个参数都有非零梯度。四路$E$共享一个参数所有者，before与after都允许梯度；不能只detach before而改变目标含义。一个去重参数优化器统一backward、裁剪和step，不用三个包含重复共享参数的优化器循环更新。[S2，§11.7]

推理时虽然参数不学习，事实、跟踪、GRU和真实时钟仍更新。源知识不接受梯度，是研究对象与权限的选择；不是当前闭环缺了一条必须到VLM的梯度。共享经验可以改变对固定关系的使用，但不会形成显式永久edge truth或trust。

# 20. Training Algorithm

## 20.1 Algorithm 1 — CP-DISR联合训练

```text
Input:
  training tasks/splits; fixed skills, contracts, perception/verifier
  prior cache; independent evaluator; budget; neural parameters theta

Initialize one model and one optimizer over unique trainable parameters.
Initialize per-environment runtime, prefix history and rollout state.

while the declared real-interaction budget is not exhausted:
  Freeze theta_old for sampling, old log-probabilities and old V.
  B <- empty on-policy buffer

  while B has not reached the rollout boundary:
    if a new episode starts:
      reset the actual environment and its runtime state
      bind objects/tasks, ground contracts, freeze node/goal indices
      load exactly matched original prior R0
      sample one episode keep flag; R <- R0 or empty (0.8 / 0.2)
      h <- zero; w <- 1

    x <- actual arrived observation + verified facts + real history
    candidates, mask <- shared generator and independent checks
    if there is no safe allowed candidate:
      apply independent terminal rule; record an environment event
      do not invent an executed action or Q label
      continue

    X <- immutable aligned snapshot
    dist_old, V_old, _, h_next <- FORWARD(X, theta_old, h, True, False)
    a <- sample masked dist_old
    record old_logp(a); commit h_next once

    execute a using the fixed controller and safety monitor
    collect actual reward events, duration and execution outcome
    verify actual next facts and construct X_next (not reset state)
    determine terminated / truncated from the independent rules
    Gamma <- exp(-log(2) * actual_duration_seconds / frozen_suite_half_life_H)
    r <- within-skill discounted actual reward events
    old_V_next <- 0 if terminated, else
      FORWARD(X_next, theta_old, h_next, True, False).V  # no hidden commit
    store X, a, mask, old_logp, old_V, old_V_next, r, Gamma, w,
          X_next, flags, prior version and recurrent prefix references
    w <- w * Gamma

  compute duration-aware GAE, yV and executed-action yQ
  detach all targets; freeze historical candidates/masks/prior versions

  for each PPO epoch:
    for each continuous recurrent minibatch:
      reconstruct prefix state with current parameters (prefix no-grad)
      reconstruct original graph views and deterministic nominal patches
      recompute four-view encodings, differences and current heads
      compute valid-transition-mean clipped actor objective (no prefix w), V loss and gathered Q loss
      add entropy term with the declared coefficients
      zero unique optimizer gradients; backward; clip; optimizer step

  discard used on-policy data; do not pretend continuing episodes ended
  rebuild current-parameter hidden states for any continuing episodes
```

高层学习为一阶段联合训练，不先训练图、Q或VLM，不进行行为克隆暖启动，不加人工奖励或新课程。rollout期间可以跨多个episode，同一episode也可跨多个rollout；这不改变其先验版本。真实交互预算包括所有失败尝试及其耗时。

## 20.2 FORWARD的精确接口

```text
FORWARD(X, theta, prefix_hidden, need_V, need_Q):
  obs_feature, hidden_next <- observation_fusion_and_GRU(X.base_input, prefix_hidden)
  GK <- contract_view(X.template, X.real_facts)
  GH <- add_valid_soft_edges(GK, X.episode_prior)
  for each allowed candidate i:
    patch_i <- NOMINAL_PATCH(X.real_facts, candidate_i, contract_i)
    GKi, GHi <- readonly views with the same patch_i
  encode the two current views and all valid successor views with E
  if prior is empty: reuse contract encodings exactly
  for each allowed candidate i:
    DK_i <- E(GKi) - E(GK); DH_i <- E(GHi) - E(GH)
    DP_i <- DH_i - DK_i
    c_i <- [obs_feature, candidate_feature_i, pool(E(GK))]
    uK_i <- F_K(c_i, DK_i)
    uP_i <- F_P(c_i,uK_i,DP_i) - F_P(c_i,uK_i,0)
    logits_i <- actorK(uK_i) + B*tanh(actorP_no_bias(uP_i))
  dist <- masked categorical(logits, X.execution_mask)
  V, Q <- independent heads if requested
  return dist, V, Q, hidden_next
```

Bootstrap forward是纯查询，不额外提交GRU；真实一个决策只推进一次历史状态。PPO重算GRU时使用当前参数与原始前缀，片段内截断反向传播，不把不同episode随机拼成连续序列。

# 21. Inference Algorithm

```text
Input: frozen CP-DISR weights, task, fixed runtime interfaces
Bind task/objects and construct the contract template.
Read matching cached prior; for a new input generate once and freeze.
Use Original prior unless an explicitly declared robustness condition.
Initialize actual facts, tracking, time and recurrent state.

repeat until the real task terminates:
  observe -> frozen perception -> verify current facts
  generate candidates and independent execution mask
  construct current and nominal graph views
  shared encode -> DK / DP -> candidate features -> Actor
  select a permitted skill by the fixed evaluation rule
  execute the fixed controller; monitor safety and critical facts
  acquire new evidence; commit verified facts, history and duration
  check independent task success and termination

Return the real outcome and traceable execution/decision logs.
```

主评价采用deterministic argmax并以规范化候选ID打破并列；训练使用masked categorical采样。推理无需输出V/Q，不做optimizer step，不按失败重新询问更满意的VLM关系。原始软边不变，但真实事实和候选状态改变，使其编码响应可以改变。

# 22. Implementation Specification

## 22.1 模块与数据流

以下是供Agent实现的接口定义，不是宣称存在已接通的机器人仓库。

| 模块 | 输入→输出 | 主要接口 | 是否训练 |
|---|---|---|---|
| contracts/ | schema、对象→grounded合同 | SkillContract、Registry、ground_contract | 否 |
| facts/ | 实际测量/日志→T/F/U与snapshot | FactRecord、Verifier、FactStore | 校准后冻结 |
| vlm/ | 初始允许输入→合法关系/cache | PriorClient、validate_relations、CacheKey | 否 |
| graph/ | 模板、事实、关系→图视图 | GraphTemplate、GraphView、build_template | 否 |
| intervention/ | 候选与合同→只读稀疏patch | NominalPatch、apply_nominal | 否 |
| models/ | 图/观察→共享特征 | SharedRGCN、GoalReadout、ObsGRU | 是 |
| policy/ | context、DK/DP→logits/V/Q | CandidateFeatures、Actor、ValueHead、QHead | 是 |
| rl/ | 真实transitions→固定targets与更新 | Collector、RolloutBuffer、SMDPGAE、PPO | 神经参数更新 |
| env/ | 真实命令→结果/耗时/flags | SkillExecutor、RuntimeAdapter、TaskEvaluator | 否 |
| evaluation/ | 冻结模型、case→指标/日志 | evaluate、intervene_prior、aggregate | 否 |
| configs/、tests/ | 版本/fixtures→配置/检查 | resolve_config、validate_manifest、pytest | 否 |

每条真实转移至少包含snapshot/next_snapshot引用、candidate IDs和所选ID、mask、原始事实/观测/历史、prior/cache hash、oldlogp/oldV/oldVnext、reward事件、duration/Gamma/w、terminated/truncated及recurrent前缀引用。动作索引只是本快照位置，不能跨快照冒充技能身份。

## 22.2 张量与确定性默认

一批有效快照的图读出形状为$[B_{batch},m_{max}+1,d]$；候选差分为$[B_{batch},K_{max},m_{max}+1,d]$；候选特征为$[B_{batch},K_{max},d]$；logits/Q为$[B_{batch},K_{max}]$，V为$[B_{batch}]$。batch维符号不与先验上界$\beta$混淆。goal padding、candidate padding和实际执行mask分别管理。

**标准读出实现默认（不新增算法）：**$\phi_K$以候选context投影为query，对差分目标行作单头scaled-dot attention，合并全局差分行后接MLP；$\phi_P$同类读出并加入$u^K$，两次锚定共用计算。可采用query/key/value宽128、两层MLP中间宽256、输出128、ReLU、dropout0。$f_g/f_0$与V/Q使用普通共享MLP，具体层表写入resolved config。观测前端输出维度、几何字段与尺度由实际adapter绑定，不能凭空规定传感器数目。

所有四路结构编码做FP32减法；若执行混合精度，仍须显式保留这一减法精度并重新验证零性质。候选/图分块只改变内存峰值，不减少候选、不截掉某些名义视图。训练时可缓存静态图、raw facts、冻结视觉特征和patch，不跨参数更新缓存可学习的Z/D或候选embedding。

## 22.3 默认配置与必填项

| 项目 | 当前默认或状态 |
|---|---|
| Graph | RGCNConv，hidden128，4层，4 bases，mean，root=True，逐节点LayerNorm，dropout0 |
| Observation history | 可训练GRU，hidden128；输入adapter MUST_BIND |
| Actor | 零锚定有界残差，$\beta=0.5$，无static prior |
| Q | Huber，$\lambda_Q=0.1$，只监督实际动作，target detach |
| PPO | lr$3\times10^{-4}$，clip0.2，GAE$\lambda=0.95$，value0.5，entropy0.01；Actor prefix=false |
| 更新 | 单一Adam，grad clip0.5，4 epochs，1024 skill transitions/rollout |
| 序列 | sequence16，minibatch64有效转移，前缀重算 |
| Prior | episode级Original0.8、Absent0.2；训练无删边/错边 |
| Discount | 共同suite half-life H=max reference median；Gamma=2^(-d/H)；time adapter MUST_BIND |
| VLM | 固定候选qwen3.8-max-0902；region/SDK/endpoint/权限 MUST_BIND |
| Runtime | simulator/robot、资产、controller、camera、Verifier阈值、timeouts、hardware、software lock、Git、splits MUST_BIND |

采用可修改的轻量PyTorch recurrent SMDP-PPO项目实现路线；不把某个现成PPO包默认当作已支持全部动态候选、图重算和Q接口。PyG API以已核对的2.6.1 RGCNConv文档为接口参考，实际torch/PyG/CUDA组合和依赖锁以目标机验证为准。本轮不猜测目标机器的软件lock；总训练预算由Plan v1.1重新明确为smoke16,384、主训练65,536技能上限并配套真实时间cap。实际软件组合与接口以运行manifest记录。[R5；附录A]

## 22.4 失败、终止与归档

控制器退出、验证结果、事实值分字段：TIMEOUT仍可能伴随Held=TRUE；NORMAL也可能UNCONFIRMED。不能因失败把事实回滚到执行前。没有真实动作的reset级终止不伪造Q训练样本；必要时把终止与上一条实际转移按时间正确对齐，保留实现记录。

每run保存resolved config、来源hash、code commit及修改patch、cache manifest、原始episode/decision日志、train/eval metrics、全部checkpoint元数据和失败记录。恢复训练需要RNG、环境/跟踪/事实/GRU前缀及pending buffer；实际环境不能回滚时不声称精确续训。API密钥不写入共享产物。

## 22.5 实现级验收

| 测试族 | 必须验证的结果 |
|---|---|
| 空prior／空patch | Full的DP及相应残差为0；CAT仅空prior必须为0，空patch可保留静态影响 |
| 事实隔离 | 构造所有名义视图后真实Fact Store、reward、time未变 |
| 节点／目标对齐 | 四视图索引一致，完成目标不删行 |
| 三值与效果 | UNKNOWN不当FALSE；冲突效果拒绝；不变量成立 |
| 关系schema | effect引用/source邻接合法；SOFT_ORDER拒绝 |
| Mask与候选 | old/new概率用同mask；重排后概率相应置换 |
| Q与梯度 | 只gather实际动作；target无梯度；四路E共享且可反传 |
| 时间与边界 | terminated无bootstrap；truncation用有效末状态；GRU仅提交一次 |
| Cache与dropout | key区分输入；episode内固定；PPO不重新抽先验 |
| 新对照与训练profile | B1-K不构造后继且DKslot=0；CAT用合同重复anchor；B0无位置/邻接mask；Actor均值无prefix；时长单位和H可复算 |

验收失败需修复实现或真实接口，但不要求先获得性能提升才准写论文。本文附带的文件一致性检查也不等于上述生产单测已经通过。

# 23. Experimental Questions

| RQ | 核心问题 | 受控比较与解释边界 |
|---|---|---|
| RQ1 | 同合同下候选名义变化是否有用？ | B1-K/B2；同c与head，仅改变DK通路；B0作非显式图系统参考 |
| RQ2 | 固定语义关系交互是否带来增量？ | B2/Full，包含prior通路与共同缺省协议的整体效用 |
| RQ3 | 纯交互限制是否优于完整四视图融合？ | Full/A_CAT；透明报告额外输入/投影参数，不称任意架构完全等价 |
| RQ4 | 辅助候选Q是否值得保留？ | Full/A_Q，1个任务三seed；不只看Q训练误差 |
| RQ5 | Actor是否响应结构，响应是否有益？ | 同snapshot敏感性与配对初始case闭环结果分开 |
| RQ6 | 缺省/扰动下如何变化？ | Full和A_CAT同checkpoint各条件，不额外训练；No-prior为missing-prior评价 |
| RQ7 | 是否支持一种未见组合？ | 已有source模型、已见原子技能/谓词，非纯ID重命名；无新模型拼接 |

本矩阵以更直接的归因替换冗余实验，而不是把所有配置与task做笛卡尔积。旧B1-H/A_DD不再强制，A_B只对应经验bounded claim。

# 24. Final Baseline Matrix

| ID | 最终定义 | prior | 名义后继 | 默认位置 |
|---|---|---|---|---|
| B0 | 两层SAB的非显式图集合策略；完整原始合同/事实/目标/候选字段 | 无 | 无 | Tier1主阶梯 |
| B1-K（表内仍可简写B1） | 同B2上下文和head，$u_i^K=\phi_K(c_i,0)$ | 无 | 不计算 | Tier1主阶梯 |
| B2 | 同一$\phi_K(c_i,D_i^K)$，独立训练 | 始终空 | 合同后继 | Tier1主阶梯 |
| Full | 现有合同干预＋纯$D_i^P$＋零锚有界残差 | episode80/20 | 四视图 | Tier1主方法 |
| A_CAT | Full合同分支不变，prior读四视图拼接及合同重复anchor | episode80/20 | 同四视图 | 核心必跑，但先1seed |
| A_Q | Full仅置$\lambda_Q=0$ | episode80/20 | 同四视图 | Tier2训练机制 |
| A_B | Full仅移除tanh，保留$\beta$尺度和zero anchor | episode80/20 | 同四视图 | Tier3，仅有界收益claim需要 |
| B1-H | 旧B1：只读当前GK/GH的静态图策略 | episode80/20 | 无 | 可选，不默认跑 |
| A_DD | 旧非空DH、空prior显式0消融 | episode80/20 | 同四视图 | 可选，被A_CAT替代必跑位置 |

**B1的精确匹配。**$c_i=[z^o,e(a_i),\operatorname{Pool}(Z^K)]$与B2一致；B1将DK输入slot置零，而不把ZK或额外action-local embedding塞入该slot。这样B1→B2只增加已定义的候选变化特征及其计算。B1不是宣称最强的任意current-graph网络，而是匹配的no-intervention对照。换一种更丰富的current-graph readout是另一个架构实验，不默认加入。

**归因等级。**B1→B2是该实现下的受控干预信息对照；B0→B1是非显式图attention与图编码的系统比较，不声称仅“是否有图”一个自由度。B2→Full检验先验交互路径与其训练协议的整体增量；Full→A_CAT检验纯交互限制与四视图融合，不是同一输入信息。共同数据预算不意味着相同轨迹；相同结构不等于所有参数完全一样。

除A_Q，全部主配置保持匹配的实际动作Q监督、同reward/感知/合同/mask/时钟与训练预算。所有使用prior的方法共享相同cache和80/20协议。报告有效trainable参数、前向视图数、延迟和内存；不使用死参数填充冒充等容量，不给baseline额外oracle或剥夺原始合同字段。

## 24.1 B0：最终两层Set Transformer实施合同

输入为当前事实（含T/F/U与goal sign）、签名目标、全体grounded技能及其原始PRE_POS/PRE_NEG/ADD/DEL/UNKNOWN文字项记录、当前候选角色和同源实际观测。每条合同记录包含owning action、predicate、参数的有序角色/类型及一致对象绑定信息；不得只保留效果类别而丢掉“谁影响谁”。任意显示ID不当作可学习语义标签；对象特征来自所有方法共同可见的对象表/观测。

所有token通过共享字段投影进入$d=128$。使用两个标准SAB：multi-head self-attention，4 heads，逐token FFN128→256→128、ReLU、残差、LayerNorm，dropout0，无集合位置编码。候选query对全部编码token做一层cross-attention，输出128，再与共同obs/GRU及候选上下文用MLP256→128融合。V/Q/Actor各自head与其他方法保持相同输出角色。SAB/MAB属于现有Set Transformer组件。[W15]

attention只使用padding mask，不输入graph adjacency、PRE/ADD稀疏attention mask、最短路、层级答案或名义赋值后的事实；不调用NOMINAL_PATCH。token包含原始合同关联，因此它可以通过attention学习关系；“非图”指无显式graph message-passing拓扑，不是禁止一切关系计算。ADD/DEL原字段也不等于已经应用当前T/F/U、条件效果和全局frame后的名义后继。

实现前冻结宽度/层数，不按Full领先与否搜索B0容量；参数差异透明报告。若已有更强且信息匹配的标准attention基线，可在未跑任何主结果前替代一次，但不能增加另一项必跑。第一版采用上述唯一实现。

# 25. Core Ablations

## 25.1 A_CAT：四视图融合替代纯交互输入

按goal/global对应行，沿**feature维**拼接，不能把goal行平铺成固定任务长度：

$$
C_i=[Z^K,\widetilde Z_i^K,Z^H,\widetilde Z_i^H],\qquad
C_i^0=[Z^K,\widetilde Z_i^K,Z^K,\widetilde Z_i^K].
$$

$$
\boxed{u_i^{P,CAT}=\phi_{CAT}(c_i,u_i^K,C_i)-\phi_{CAT}(c_i,u_i^K,C_i^0),}
$$

$$\Delta_i^{CAT}=\beta\tanh(w_P^\top u_i^{P,CAT}).$$

合同FK、c、E、名义patch、goal mask、shared candidate output、V/Q/Actor功能、优化参数和数据预算均与Full相同。CAT必须从头独立训练；“相同head”指下游架构/宽度，不表示两个独立run共享已经训练好的参数。

输入首投影4d→d，而Full为d→d；之后使用相同候选attention/MLP/head。若只此处增宽，CAT额外参数为$3d^2=49,152$（d128，不含相同bias）。实际层表不同则重新精确计数；不谎称严格等参数。Full是在共同输入上固定使用$[I,-I,-I,I]$对比的受限归纳偏置，CAT可学出这一映射，但有限训练不保证自动学会。

无prior时C=C0，残差严格0。空patch或可加分离的静态prior响应则**可以非零**，这是比较目的，不能额外用DP=0 gate把CAT变回Full。锚定到全零而非合同重复视图不是本消融定义。CAT不允许从effect_fact_ref额外读几何或语义标签。

本消融回答：明确丢弃可加分离静态成分是否值得？它不单独隔离“减法计算”与容量、完整信息保留的全部区别。若CAT在同预算主任务/后续评价中稳定更好，纯交互优越性不成立，应报告、缩小claim或在未来v2.2重新选择融合；不能为了保住Full挑选seed/删task。写作仍可继续，但结果结论须改变。

## 25.2 A_Q和A_B

A_Q仅将$\lambda_Q$设为0，保留初始化顺序/网络接口，不把Q输出送Actor。它是保留Q作为正式训练设计时最小必要的训练机制对照，默认只在TC用3seed。

A_B仅使用$\Delta_i=\beta w_P^\top u_i^P$，无tanh、无补回clamp或归一化。零锚定保留。若论文只声明代数上界而不声称其性能/鲁棒收益，A_B不列必跑；一旦主张bounded带来行为收益，就需要它。

A_DD仍可定义：R非空用DH，R为空用0；但不与A_CAT一起强制运行。旧B3、static-prior独立模块、Full-clean、effect-anchored图等不列当前默认任务。

## 25.3 对照复用

Full可复用先前完全匹配的task/split、code行为、runtime/软件、cache、seed、common hyperparameters、预算、时间H与checkpoint规则的run。复用写control_reuse_manifest，不复制成新训练。新参数profile改变后，旧seed0不能与新seed1/2凑成同profile三seed；受影响范围显式补跑并计成本。其他Stage不得自动扩实验。

# 26. Mechanism Diagnostics

对冻结Full checkpoint，在同一真实snapshot上改变一种输入并重算，观测、历史、goal和执行mask保持一致。随后可用配对初始case执行闭环测试；闭环后状态自然分化，不能把它们仍称相同状态比较。[S2，§18.2；S3，§11]

| 操作 | 精确行为 | 主要观察 |
|---|---|---|
| $D^K=0$ | 置零后重算$u^K$及依赖它的先验上下文 | 合同变化信息影响；不改真实图事实 |
| $D^P=0$ | 对齐形状置零，零锚定使prior残差为0 | 同参数去掉先验直接通路 |
| shuffle $D^P$ | 仅在有效候选间交换整块差分，不打乱goal行 | 候选—变化对应是否重要 |
| remove prior | 删除语义边及逆向边，重算视图 | 与同状态DP置零的接口一致性 |
| target-swap | 按第27节合法规则更换一条target | 语义对应变化的响应 |

记录动作分布、JS/KL、argmax与排序变化、base/prior/final logits、真实闭环成功和有效分母。单步非零JS只说明敏感，不能证明有益；小残差也不代表无影响，因为基础分差可能更小。所有候选共同偏置不改变softmax。

判读时分开“无先验”“空patch”“多个候选差分相同”“prior非空且变化可传播”等组别。不能把理论应为0的case当失效，也不能用Q误差下降代替Actor依赖证据。与任务相关的关系干预、实际行为及最终结果应共同支撑利用结论。

# 27. Robustness

四个常规条件及一个适用性限定的Irrelevant条件使用同一冻结checkpoint和配对初始case，不分别重训，也不在普通评价中随机执行训练20%dropout。操作只影响语义soft edges，完成后统一生成逆向消息边；合同、事实、候选、mask和reward不变。

| 条件 | 冻结操作 | 适用性与日志 |
|---|---|---|
| Original | 使用原始解析/校验/去重后的关系 | 原始不等于gold；不人工修成正确 |
| No Prior | 整份软关系为空 | missing-prior / dependence；训练见过缺省，不保证全闭环状态在分布内 |
| Edge Deletion | 每条正向语义边独立以0.5概率删除 | 允许全空或恰未删边，不强制重抽 |
| Target-swap | 均匀选一个合法“原边—新target”对，仅改一条target | 无合法对则NOT_APPLICABLE |
| Irrelevant | 在已有、独立确认无关的组件内追加1条合法支持边 | 无证据/无合法边/已达8条预算则不适用 |

Target-swap保持source、type及effect_fact_ref不变，target类型合法且与原target不同；排除supports自指、已有同型边、局部合同重复和明确任务禁止项。SOFT_RELEVANT_TO_GOAL仍只能指向现有goal_refs。操作保证合法端点错配，不保证语义必错或“合理错误”；只有独立标注子集才可称semantic-wrong。[D1]

Irrelevant要求预先任务声明其不影响目标、关键资源或执行路径，同时其组件在原始消息图中与目标不连通。仅图不连通不足以证明物理无关。不得临时新建节点、删原边腾预算或把任意随机边命名无关。全局读出仍可能受这些边影响，因此它是实际测试，不是预定必须不变的等式。

每个case的编辑seed、前后边集、hash、删除/交换/追加数与适用性在查看成绩前保存，并在episode内固定。报告绝对成功率、对相同适用子集Original的百分点差、有效样本数及动作变化。不适用不是成功，也不把未改图记成已完成错边测试。

# 28. Generalization

优先只检验：**seen skill schemas＋seen predicates＋unseen object–goal / dependency composition**。训练与测试保持低层技能、感知接口和谓词语义不变，组合任务在独立split中出现，不同时改变视觉域、语言域、机器人、对象数量与外观。[S1，§48]

需验证新组合不是任意对象ID重命名后的同构图，也不只是新的环境随机seed。目标数、共享条件和绑定等具体holdout规则在实际资产绑定后写入split manifest。若只测试换位置，应按位置泛化报告，不扩大为依赖组合泛化。

输入新初始场景调用冻结VLM并生成缓存属于推理，不是允许依据测试成绩选择prompt。模型权重、合同、Verifier与预处理冻结。每个checkpoint独立评价，不拼接若干单任务策略冒充一个组合策略。任务和数据尚未绑定时，具体分布与结果为MUST_BIND/NOT_MEASURED；可以借鉴VIMA区分泛化层次的评价思想，但不声称其基准天然提供本项目合同或技能。[R14]

# 29. Experimental Setup Draft

## 29.1 Task Family最终选择与奖励暴露

默认核心family仅**T_B：Multi-Object Choice**和**T_C：Intermediate Relocation**。D0作多步smoke；T_A保留为合同重复/共享前置参考而非强制prior增量；T_D在确有组合评价需要时绑定；T_E成本较高，后置。模板不是已存在资产，平台/控制器/观测/goal/deadline/split仍MUST_BIND。

主任务应在合法参考轨迹中至少出现两个有≥2合法候选的决策点，并且有实际长期后果差异。仅candidate数量大或mask非空不证明有真正决策。合同保留所有真实必要启动/安全条件，不能故意删掉前置制造VLM优势；额外软关系限于当前抽象未完整建模的场景关联、代价与后续联系，而非隐藏的硬安全许可。

优先采用**错误通常可恢复，但消耗真实时间**的任务：增加探索时仍完成的机会，又由同一终端成功的时间折扣区分绕路与高效执行。它仍是稀疏奖励，不会自动给每次返工产生即时负奖励。少量错误即终止case可以保留为边界，不要求全部可恢复，不放宽安全阈值。

记录从实际改变的Fact到目标/候选的message-path长度；4层是否覆盖由具体图核对。PICK→Held→PLACE→Inside含3条边，从变化起点Held算才是2跳。局部覆盖不等于完整长期规划；不提前加candidate-local row或Graph Transformer。

包含prior有用、当前无用及自然误导case，但不按Full测试分数挑选cache或刻意给训练造错边。原始输出可能为空/错误，这些case照实保留。任务分层与test-independent设计先登记；相同任务下所有方法获得同源可观测线索。

## 29.2 Revised Experimental Route v1.1

保留已有Stage ID语义大体兼容，并增加0D。**0A/0B/0C/0D只阻塞依赖它们的真实运行，不阻塞Problem/Method写作。**本轮只生成计划与代数文件验收，不宣称已经执行这些Stage。

| Stage | 目标 | 默认新增训练 |
|---|---|---:|
| 0A | 真实接口/软件/权限/任务绑定；每训练family最多10次reference尝试取得5次成功，确定H与d_ref | 0 |
| 0B | 原生产不变量＋A_CAT null、B1无后继、Set permutation、unweighted Actor及时长单位检查 | 0 |
| 0C | D0/TB/TC各8个初始dev场景，固定VLM协议和cache质检 | 0 |
| 0D | 每family100个uniform masked episodes，低成功时扩到200；必要时50个contract-aware诊断 | 0 |
| 1A | D0，B2/Full，seed0，16,384转移上限 | 2 |
| 1B | 仅学习/数值明显异常时，有限单项优化，最多6个额外job | 条件性，非默认 |
| 2A | TB/TC × B0/B1-K/B2/Full × seed0，65,536/运行 | 8 |
| 3A-pilot | TC × A_CAT × seed0；Full复用2A | 1 |
| **第一暂停点** | **11个默认训练；形成探索Results；不等3seed全部完再写** | **累计11** |
| 2B | 原两任务四方法补seed1、2；不重跑已匹配seed0 | 16 |
| 3A-completion | TC A_CAT补seed1、2；Full继续复用 | 2 |
| 2C | 按明确source生成主表/主曲线 | 0 |
| 3B | TC A_Q，0/1/2 | 3 |
| 3C | A_B仅在bounded经验收益claim需要时，TC，0/1/2 | 可选3 |
| 4A/4B | 同checkpoint Actor依赖与先验变化；Full/A_CAT配对 | 0 |
| 5A | 一种已有source模型对应的实际未见组合 | 0；无source则不自动增训 |
| 6A/6B/6C | Raw审计、展示manifest与论文包 | 0 |

第一批11个，上限622,592训练transition；补齐两任务三seed和A_CAT后累计29个，上限1,802,240；再含A_Q为32个，上限1,998,848；A_B也做则35个，上限2,195,456。表内不含评价、API和条件性重跑，均另记成本。不同stage复用同一Full只计一次。

**Tier1：**B0/B1-K/B2/Full及A_CAT；先单seed再补齐。**Tier2：**A_Q及同checkpoint诊断，必要的一种组合评价。**Tier3：**A_B、B1-H、A_DD及资源/表示诊断。不存在7方法×所有task×多seed一次性全跑，也取消无条件新建36个Presentation训练的要求。

总训练预算在旧记录存在冲突，本版**重新明确选择**上述16,384/65,536，未声称它们本来就是唯一冻结旧值。smoke每4096评价/保存，主训练每8192评价/保存；rollout1024，因此16/64个PPO updates上限。smoke最早8个updates后满足可用性可停；主对比预算完成或明确数值/安全错误才停止，不因Full先赢单独提前停。

每family预算还设$T_{cap}=N_{cap}d_f^{ref}$，$d_f^{ref}$为0A参考执行中实际每技能时长中位数。所有同family方法共用Ncap/Tcap；先达到一个cap即停，最后不足buffer用有效mask处理。实际任务秒数、技能数、control ticks、wall time分别保存。主学习曲线在共同实际时间区间计算raw AUC，并附skill-transition横轴；不得向未运行尾部外推。

## 29.3 Seed、checkpoint与评价：小规模但可解释

Smoke和探索pilot用seed0；主比较补齐0/1/2，而非默认新增完全独立Presentation组。配置/任务未改变且hash可追溯时复用pilot；必要的开发改动记录后，只补跑受影响的比较组，不静默混profile。最多补到5seed是明确可选项，不反复加到获胜。

主文定量对照默认包含全部3个预定有效训练seed，显示逐seed点及均值/样本SD；零成功但数值正常属于有效结果。Private Paper View仍可单独选seed/task作illustration，但标selected exploratory view，不取代“整体方法更好”所需的完整对照表。基础设施无效run及其重试均在Raw中保存。

dev与test分离。每checkpoint：smoke10个dev episode，主训练20个dev；默认从step0开始评价。按最近3个dev窗口平均score选checkpoint，并列取窗口最小值更大，再并列取更早step。默认score为**实际起点时长折扣return**，同时报告success；这避免所有方法success饱和时按偶然尖峰/最早step选择而忽略真实时间差。规则在训练前统一，test前写checkpoint/hash；每task/seed最终30个独立test episode，可共同扩到50，不只补赢家。

主结果同时给success、discounted return、raw learning AUC和成功条件完成时间/成功分母。不同gamma的return不能直接跨family混成同指标，默认共同suite H避免该问题。图显示可用窗口3 trailing moving average，但表与AUC只用原始值，保存raw版；所有选择/平滑写manifest。

Actor-dependence与闭环分开。推荐≤100个真实snapshot，比较DK0、DP0、候选间DP置换和合法先验变化；闭环每条件20episode。same-checkpoint先验测试Original、No-prior、Deletion、Target-swap各25case，Full与A_CAT使用同seed的冻结checkpoint和配对case；Irrelevant仅有原定义合法独立组件时运行。无需新训练。

No-prior正式称**missing-prior / prior-dependence evaluation**：整份输入缺省机制在训练见过，但不能保证关先验后整条闭环状态分布都在训练分布内。80/20对Full和CAT相同不是“该正则完全没有作用”的证明；不做Full-clean，因为本论文不独立归因dropout收益。

## 29.4 指标定义

| 指标 | 计算与边界 |
|---|---|
| Task success | 独立评价器成功数/有效完整episode数；零成功是0 |
| Learning AUC | 原始success和discounted return分别在共同实测预算区间积分并归一；不对平滑值积分、不外推 |
| Discounted task return | 按共同H、真实事件时刻计算起点回报；不从图分数生成奖励 |
| Time-to-threshold | 预声明阈值与连续窗口；未达到记NOT_REACHED |
| Completion time | 成功条件下时间均值/中位数，伴随成功分母与总体失败率 |
| Recovery completion | 独立失效机会后的任务完成率，必须报告机会分母；无机会为NA |
| DK/DP及残差 | 有效候选/goal的norm、非零率、分位数；无适用量为NA，不捏造0 |
| Actor sensitivity | 同状态JS/KL、argmax/排序变化；与闭环成功分开 |
| Cost | 实测编码延迟、峰值显存、视图数与离线API成本，不给虚假硬件估算 |

主表来自原始episode，曲线可用明确记录的EMA或moving average，同时保存原始点和无平滑版。本轮重新采用窗口3 trailing moving average作可选显示默认；原始值与无平滑版保留，figure manifest必须记录。数据选择与显示变换不改原始reward、success或CSV。

## 29.5 两套数据视图

Raw Archive完整保存所有run、失败、中断、seed、checkpoint、config、cache、split和版本。Private Paper View是派生视图，每个图表至少记录run_id、task、seed、checkpoint/hash、split、source文件/行键、metric、聚合、smoothing、subset和selection reason。

代码/感知/控制器bug先修再运行相关比较；有差距但不显著、Full与B2接近、某个有效seed零成功，都不是自动重开Method的理由。按实际观察写有限、相近或负向结果。工程异常与算法正常失败分别归档，不能把后者重新分类成可以删掉的无效数据。

# 30. Paper Results Templates

下列模板为空白证据位置，**NOT_MEASURED表示未有可追溯测量，不能替换为0或预期优势。**最终task行由实际绑定和selection manifest决定。

## 30.1 Table 1 — Main Results

| Task family | B0 | B1 | B2 | Full | seeds / episodes |
|---|---|---|---|---|---|
| T_A（绑定后） | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |
| T_C（绑定后） | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |
| 第三个主family | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |

位置：主结果。回答RQ1/RQ2。结论句模板：“在[任务/预算/所选seed范围]内，Full相对[对照]的任务成功率表现为[真实差值或接近/更低]，其范围与分母见表1。”只有误差与重复证据允许时才用稳定提升等措辞。

## 30.2 Figure 3 — Learning Curve

每个task一幅，横轴为实际交互时间并辅以技能次数版，纵轴为原始任务成功率，叠加B0/B1/B2/Full。图注写seed集合、窗口与平滑方式，原始版本保留。回答RQ1/RQ2；Q的RQ4须用A_Q对照，不能凭主阶梯回答；AUC用原始数据和共同预算计算。

结论句模板：“在相同[预算口径]下，[方法]于[实际区间]达到[真实阈值/未达到]，曲线面积差为[实测]。”不因最终成功接近就从主表删除成功率；可将AUC作为更有信息的补充。

## 30.3 Table 2 — Ablation

| Task | Full | A_CAT | A_Q | A_B | Full control来源 |
|---|---|---|---|---|---|
| 代表结构任务 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |

位置：Ablation。回答RQ3/RQ4及有界残差作用。伴随raw AUC、逐seed原始点和共同配置。结论句模板：“仅改变[因素]后，在[范围]观察到[改善/无明显差异/下降]；这一证据支持[受限表述]，不支持[更强表述]。”未运行某消融就删经验结论或标NOT_RUN。

## 30.4 Figure 4 — Actor Dependence

展示同状态干预的JS/KL、argmax变化及独立闭环成功率，各指标单独作图，不用多坐标轴混合。回答RQ5。结论句模板：“[干预]在[n个有效快照]改变了[分布/动作]；对应闭环评价为[真实结果]，因此证据分别说明[敏感性]和[有益性/未确认]。”

## 30.5 Table 3 — Prior Robustness

| Condition | Success | Paired Original | Difference (pp) | Applicable / valid |
|---|---|---|---|---|
| Original | NOT_MEASURED | — | — | MUST_BIND |
| No Prior | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |
| Deletion | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |
| Target-swap | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |
| Irrelevant | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED | MUST_BIND |

位置：Robustness。回答RQ6。结论句模板：“在固定checkpoint、配对case与明确扰动下，成功率变化为[实际百分点]；不适用case为[分母及原因]。”不写“模型因此识别并修复错边”。

## 30.6 Figure 5 — Qualitative Episode

选择一个真实、最易解释的episode，展示真实观测/Facts→候选→名义效果→DK/DP→base/residual/final logits→实际技能→真实结果。附episode/run/checkpoint/snapshot索引。图可展示负向修正或失败，但不能给latent维度虚构语义，也不能用手绘名义成功当真实机器人证据。

位置：Mechanism/Qualitative。关联RQ2/RQ5。结论句模板：“该选定实例中，[事实与候选模式]对应[实际评分与动作]，其后真实结果为[记录]；该例解释机制，不代表总体频率。”

## 30.7 组合泛化与证据模式

RQ7使用一个ID/组合holdout补充表，列已见schema/predicate核查、非同构检查、评价分母和真实成功。没有数据时只保留任务划分定义，不在摘要声称泛化已验证。

**Expected evidence pattern仅是假设。**干预可能在结构复杂任务更有优势；prior增量可能集中于非冗余关系；Q可能主要影响early learning；bounded残差可能在扰动下更明显。反向、无差异或局部有效结果同样必须保留。论文不绑定固定10%提升，不把轻微点估计差写成普遍成立的机制。

# 31. Scope and Boundary

本方法研究固定不完美先验的条件化利用，不进行关系修复、知识库修订、逐边正确率估计、VLM适配或在线图重写。没有显式edge memory不等于完全没有历史学习：神经参数和GRU可以保留规律，但不是一份可审计的逐边经验表。[S1，§34]

缺少VLM关系不等于相关行为无法学会。如果实际观测、历史与经验足以区分情境，策略可以通过其他通路学习选择；它不会因此自动把新知识写回图。真正的硬瓶颈是全部允许输入都无法区分必要情况，或必要动作不存在/被错误mask。反之，存在一条语义关系也不保证读图器能保留或学会利用它。

可靠合同表示来源和接口经过检查，而不是完整物理模型。若控制器与独立证据确认合同确实写错，应修改合同版本并重新验证相关接口；不能期待PPO自动修合同，也不能以回报提高为由学习放宽安全限制。技能实际执行可靠性和名义成功语义分开；当前用真实return/Q/PPO学习其决策影响，不新增成功概率校准器。

结构差分及其范数只说明输入编码响应，不是关系真伪或因果值。多个来源的错误可能被同一组共享参数吸收，学习不保证逐条归因；这限定了论文的解释强度，不使当前决策问题不完整。

# 32. Discussion

固定知识来源使“先验生成质量”与“先验利用机制”可以分开研究。相同合同和cache下，方法差异主要来自候选信息组织与训练，便于判断表示是否产生额外效用。同时，固定输入不会自动消除上游错误，必须报告输入可用性、关系产出和运行接口，而不把任何失败都归咎于策略。

语义正确性与决策有用性不同。正确关系可能当前不相关，未完成目标的依赖也可能被更低成本路径替代；模型有理由弱化其影响。负向prior修正也不等于建立负向事实，单次失败不等于证伪支持关系。当前“use/suppress/negative adjustment”是输入—动作的条件化映射，不是在线知识验证过程。

实际学习效果可能来自合同干预、观测历史、共享表示与Q等多种因素，因此采用独立训练对照和同参数机制干预。前者检验训练后整体效果，后者检验一个固定模型怎样响应输入，两者不能互相替代。先验缺省训练提供独立使用合同通路的经验机会，但不保证其参数与独立B2相同。

方法性能应在实际任务、先验质量、预算和传感器范围内表述。论文级写作追求概念和证据对应，不要求每个任务Full都第一；也不能因私人展示选择而把有限证据升级为普遍结论。结果接近时可以讨论成本、结构机会与适用范围，而不增加新模块补叙事。

# 33. Future M1+ Direction

**仅作为Discussion / Future Work，不进入当前算法。**未来可研究Candidate × Relation Error Learning：同一候选的价值预测在保留关系时为$q_t^{\mathrm{keep}}$，移除关系$e$时为$q_{t,e}^{\mathrm{drop}}$，以同一实际结果参照$Y_t$构造

$$
u_{t,e}=(Y_t-q_{t,e}^{\mathrm{drop}})^2-(Y_t-q_t^{\mathrm{keep}})^2,
$$

再学习$\hat u_\phi(\chi_{t,i,e})$，检验关系引起的判断偏差是否在新情境中可迁移。该量是特定参照读图器的预测效用，不是关系真假或物理因果效应；参照模型、参考策略、标签数据隔离及删边输入支持都需要独立定义。

Current M1的完整问题是fixed prior→candidate reasoning→real outcome→decision learning。Future M1+关注relation-specific predictive-error experience，两者分别成立，不是为了修补“梯度没有传到VLM”才需要升级。本规范不为M1+新增损失、参数、数据采集或前置实验。[S1，§59–60；S5；D3]

# 34. Paper Outline and Related Work

采用Claude的重心调整，并把精确性质与学习边界放在对应位置：

| 小节 | 内容 |
|---|---|
| 4.1 Overview | 候选条件化问题与2×2总览，不从工程模块清单开篇 |
| 4.2 Matched Structural Views | 合同/事实/soft relation、同节点与goal、metadata与权限 |
| 4.3 Candidate Nominal Interventions | 简短说明采用afterstate思想、同patch、名义而非真实未来 |
| 4.4 Contract–Prior Interaction Contrast | DK/DH/DP、S对称恒等式、可加分离抵消及其有意限制 |
| 4.5 Prior-Modulated Contract Policy | 零锚、有界残差、mask、Properties Box的条件 |
| 4.6 Outcome-Grounded Training | unweighted empirical PPO、duration half-life、独立V/auxiliary candidateQ、80/20 |
| 4.7 Implementation and Complexity | 标准R-GCN、2+2N视图、重算/缓存/计算成本；详细层表移附录 |

4.4–4.5为视觉与论述中心，不机械规定40%篇幅。Problem只定义历史可依赖的策略与任务目标，不提前放GRU实现、80/20比例、evaluation选择或未绑定硬件。

题目/Introduction/Abstract统一使用“nominal representation response”，不写prior改变合同名义effect；方法定义可以肯定陈述，性质和经验claim各有必要边界，不采用“每节最多一句否定”的机械规则。


Related Work按四条轴组织：语言/视觉语义先验及表示RL；符号动作模型与afterstate策略；行动差分与有限交互表示；时间抽象与辅助价值学习。具体一手文献核验等级见review/Sources。来源数量不作35–50的硬目标；API/PyG等实现文档移实施附录，不与算法新颖性依据混淆。

论文结构：Introduction → Related Work → Problem → Method → Experiments → Discussion → Conclusion。Method两核心小节先写，Setup按本版baseline和Plan1.1填真实绑定，Results用原始记录补入；Future M1+只在Discussion简述。

# 35. Abstract v1

**English.** Long-horizon skill selection requires distinguishing locally executable actions from actions that support future task completion. Fixed skill contracts describe nominal effects, while vision-language models may supply additional but imperfect semantic relations. We present CP-DISR, which conditions the direct use of these relations on the encoded changes induced by candidate nominal interventions. A shared encoder processes contract-only and prior-augmented views before and after the same nominal change; their interaction contrast enters a zero-anchored bounded residual on a contract policy. The design yields exact null conditions and a bound on one-step policy deviation under fixed inputs and parameters. We train the decision features using an empirical duration-aware PPO surrogate and auxiliary candidate-value regression from actual executions. Experiments on [bound task families and platform] compare matched contract baselines and four-view concatenation, and assess [measured results, not yet available].

**中文。**局部可以执行的技能未必有利于整项任务完成。CP-DISR在固定技能合同和不完美语义关系下，对每个候选构造名义变化，并比较有无先验时共享表示的前后响应。该交互对比作为先验的唯一直接特征入口，经零锚定有界残差调制合同策略。网络由真实执行数据支持的经验PPO与辅助候选价值回归训练。实验将以匹配合同阶梯和四视图拼接检验这一限制的用途；实际平台、任务和结果待填，不预设改善幅度。

# 36. Introduction v1

长时程机器人任务包含多个当下合法、却有不同后续代价的选择。一个搬移动作可能为另一对象的放置创造条件，也可能只是多余绕路；相同关系在目标已经完成、缓冲区域不同或剩余时间紧张时，决策用途也会改变。高层策略因此不仅要知道“哪些动作允许执行”，还要理解候选动作的名义变化如何与当前任务中的其他联系相互作用。

固定技能合同提供来源明确的前置与名义效果，但不需要完整建模每个场景的全部空间关联。VLM可以提供额外关系，然而这些关系只是软信息，不能重写事实、安全许可或任务奖励。既有研究已经通过状态条件化行动先验、教师指导和VLM表示辅助RL；候选后继状态评价与图策略也有成熟基础。[W03，W05–W07，W14] 本文不把“语言模型可能犯错”或“预测一个后继状态”单独作为创新。

我们研究一个更具体的表示选择：是否值得将固定语义先验的直接影响限制在候选名义变化的编码响应上？对每个可执行技能，CP-DISR在合同结构和语义增强结构中应用同一个名义事实patch，得到当前/名义与合同/增强的四个匹配视图。共享编码器下的交互对比分离软关系对这一变化响应的影响，再通过零锚定有界残差作用于技能选择。

这一约束不表示静态信息在所有任务里都无用。它是一种需要验证的归纳偏置：减少不随所比较变化改变的表示成分，同时可能舍弃有用的上下文。为检验这一取舍，我们用同样四视图的零锚定拼接作为核心对照，而不是只比较一个较弱的当前图策略。真实执行数据训练Actor、独立V和辅助候选Q，关系本身不被当作已验证知识，也不被回报自动修订。

本文的方法贡献集中为matched contract–prior interaction representation与其pure-interaction policy design两个方面。受控baseline、同checkpoint输入变化和行为分析将提供具体任务与预算下的实证判断；当前不将常规实验协议或辅助Q head单独宣称为第三项原创算法。实验结果占位将在实际日志与评价完成后替换。

# 37. Problem Formulation v1

给定固定技能库$\mathcal U$、经过接口校验的合同$\mathcal K$、独立任务$\mathcal T$与每episode缓存的软关系$R_e$。第t次高层决策利用截至实际时刻$\tau_t$已到达的观测、三态事实及历史，形成snapshot $X_t$；允许策略依赖完整可得历史，不假设有限摘要必为Markov状态。候选数记$N_t$，执行mask由实际能力、合同、安全与参数绑定独立确定，R不改变mask。

技能$a_t$持续$d_t>0$秒。指定训练suite共同折扣半衰期H，$\Gamma_t=2^{-d_t/H}$；技能内实际事件奖励按发生偏移折扣为$r_t$。环境仅在首次独立确认全部目标时给事件奖励1；子目标与名义图不另给分。实际任务成功/失败、deadline耗尽和外部截断分别管理，外部有效末状态用于bootstrap。

任务评价准则为$J(\theta)=\mathbb E[\sum_t w_t r_t]$，$w_0=1,w_{t+1}=w_t\Gamma_t$。Method使用有效transition平均的经验PPO surrogate及duration-aware targets，不宣称其是J的无偏精确梯度。[W01] 关系语义是否成立与当前候选是否值得执行不同；本文学习后者的结构输入用途，而不输出逐边正确率。

# 38. Method v1

## 4.1 Overview

CP-DISR针对每个可执行候选计算名义结构变化，以固定语义关系对变化的编码响应调制合同策略。信息来源与表示学习分离，目标是技能选择而非关系修订。

## 4.2 Matched structural views

当前合同图$G^K$具有Action/Proposition两类节点、PRE_POS/PRE_NEG/ADD/DEL及其反向消息边。增强图$G^H$只增加合法软关系，不改当前事实、goal索引和节点集合。effect_fact_ref是已有后果入口的admission metadata，不是单效果神经门控。

## 4.3 Candidate nominal interventions

采用已知动作模型后继状态思想，[W03] 对候选a_i仅在只读事实副本中应用合同名义成功效果，形成$\widetilde G_i^K,\widetilde G_i^H$。两个后继使用同一patch，既不预测实际成功概率，也不取得真实未来；实际结果另由执行及Verifier更新。

## 4.4 Contract–prior interaction contrast

同一个E输出固定global/goal行Z。定义

$$D_i^K=\widetilde Z_i^K-Z^K,\quad D_i^H=\widetilde Z_i^H-Z^H,\quad D_i^P=D_i^H-D_i^K.$$

若记$S_R(F)=E(F,R)-E(F,\varnothing)$，则$D_i^P=S_R(\widetilde F_i)-S_R(F)$。这一2×2表示交互不是因果DiD估计。它消去对该名义变化可加分离的先验成分，而不意味着一切静态边无影响。相同编码器和索引提供共同坐标；不声明每一维有固定语义。

## 4.5 Prior-modulated contract policy

令$c_i=[z^o,e(a_i),\operatorname{Pool}(Z^K)]$，

$$u_i^K=\phi_K(c_i,D_i^K),\qquad b_i=w_K^\top u_i^K+b_K,$$

$$u_i^P=\phi_P(c_i,u_i^K,D_i^P)-\phi_P(c_i,u_i^K,0),$$

$$\Delta_i=\beta\tanh(w_P^\top u_i^P),\quad \ell_i=b_i+\Delta_i,\quad \pi=\operatorname{masked\_softmax}(\ell).$$

当前ZH不能绕过DP进入Full context。无先验时残差为0，同参数policy退化为合同reference；对合法动作$e^{-2\beta}\le\pi_i/\pi_{K,i}\le e^{2\beta}$。空patch保证该候选直接残差0，但其他候选仍可改变其softmax概率。其他性质和置换条件见Properties Box。

## 4.6 Outcome-grounded training

真实执行数据构造时长TD误差$\delta_t=r_t+\Gamma_t(1-terminated_t)V_{old}(X_{t+1})-V_{old}(X_t)$，GAE为$\widehat A_t=\delta_t+\Gamma_t\lambda c_t\widehat A_{t+1}$。独立V以固定$V_{old}+\widehat A$回归，Auxiliary Candidate Q仅gather实际动作并回归$\operatorname{sg}(r_t+\Gamma_t(1-terminated_t)V_{old}(X_{t+1}))$。Q不是关系真假监督，candidate-set context仍可使梯度影响其他候选特征。

总loss为有效transition均值的clipped PPO＋$c_VL_V+\lambda_QL_Q-c_HH(\pi)$；Actor不额外乘episode-prefix折扣。它是当前采样分布下的经验surrogate，不宣称精确J梯度。[W01–W02] 旧targets和先验版本在PPO重算期间固定，四视图和GRU前缀按当前参数重算。VLM/合同/Verifier/低层控制器不接受梯度。

训练在episode起点按80%原始、20%整份空prior抽样并保持episode内固定，模式标签不送网络。不训练人工错边，不加入奖励塑形。H按训练suite的真实reference时长标定并固定，详见Setup。

## 4.7 Implementation and complexity

使用标准4层128维R-GCN、4bases、mean、ReLU与逐节点LayerNorm、dropout0。观测前端固定，后接可训练融合器与GRU。$N_t$候选通常需要$2+2N_t$个共享图视图，空prior复用为$1+N_t$；报告实测延迟/显存，不跨参数更新缓存可学习表示。具体runtime与完整层表从授权实际项目绑定。

# 39. Experimental Setup v1

采用同一固定技能库、合同、控制器、观测与Verifier、执行mask及独立终端奖励。默认T_B多对象选择与T_C中间重定位，D0用于smoke；所有模板的真实simulator/robot、资产、camera、时限和safe execution permission在0A绑定，不把符号toy当机器人结果。

0A用固定reference执行校准各family典型完成时间，共同H取中位时长的最大值。0D用100–200个uniform masked episodes检查奖励暴露，低暴露时最多另50个contract-aware random诊断；5%不是硬阈值。所有这类数据不送PPO、BC或VLM few-shot。

B0使用两层Set Transformer；B1-K与B2同context但DKslot为0且不生成后继；B2做合同干预；Full增加DP先验路径。A_CAT使用同四视图行对齐拼接和合同重复锚点，作为纯交互限制的核心对照。A_Q在一个任务检验辅助loss，A_B仅当声明bounded经验增益时运行；旧A_DD/B1-H降为optional。

第一轮D0两个smoke加两family四方法单seed及TC单seed CAT，共11个新训练。配置正常后补seed1/2，累计29；再加TC三seed A_Q为32。smoke16,384/主训练65,536 skill cap，另以reference每技能中位时长定义真实时间cap，先达到者停止。所有失败交互记账，数据预算不能用未测安全5秒代换。

每checkpoint主评价20个dev，smoke10个；按三个dev窗口真实discounted return均值选择模型，test前冻结选择，最终每task/seed30个独立test。定量比较默认完整0/1/2三seed；精选Private View另标并保留Raw。报告success、discounted return、raw AUC、完成时间与分母；显示曲线可窗口3平滑，表和AUC不平滑。

相同checkpoint比较先验缺省、每边p=.5删除和1条合法targetswap；Irrelevant仅在已有独立合格组件时测试。Full与CAT配对评价，不分别重训。缺省模式训练见过，不写成完全未知OOD；语义targetswap不自动等于错误关系。具体数据、软件hash、总秒数与所有结果仍待实际运行。

# 40. Discussion v1

CP-DISR检验一种纯交互归纳偏置：固定语义先验通过候选名义变化的编码响应进入选择，而不是直接提供动作推荐分数。它不假设全部静态信息无用，也不要求该限制在所有任务上更优；A_CAT使这种取舍能够被直接检验。

合同属于来源明确的能力接口，VLM关系属于可错软输入。名义干预不保证执行成功，真实回报学习也不提供逐边真假或物理中介证据。effect_fact_ref验证合法后果入口，神经消息仍以skill为单位。4层R-GCN和global mean带来有限表达与聚合边界，其实际影响由任务与机制诊断判断，不预先加新网络。

模型只从实际交互学习。稀疏终端奖励可能使探索困难，辅助Q并不制造成功标签；采用时长半衰期和有效样本PPO平均是明确工程选择，不作为全局收敛或最优性定理。选择性利用和泛化范围均由实际数据支持，单步latent变化不是有效性证明。

未来可研究Candidate×Relation Error Learning，用输入关系对照与真实结果形成关系特定的预测误差经验。该方向独立于当前方法，不进入本版loss、图、cache或必跑矩阵。当前方法已经足以写作；未来增强不是开始论文的条件。

# 41. Final Frozen Research Snapshot

| 项目 | 冻结定义 |
|---|---|
| Version | 文档3.1；Method2.1.1；Experimental Plan1.1 |
| Problem | 固定合同与软关系下，学习候选条件化长期技能选择 |
| Core design | candidate nominal change × prior的matched四视图交互，作为直接先验入口 |
| Knowledge | VLM/合同不由RL修改；真实事实由观察与Verifier更新 |
| Graph | SLG两节点、标准4层R-GCN；两种soft关系；effect_ref只作admission metadata |
| DK/DP | 公式不变；DP=S_R(Ftilde)−S_R(F)，不是真实因果/价值贡献 |
| Actor | 合同分＋beta tanh零锚残差，beta=.5；独立mask |
| V/Q | 独立V＋Auxiliary Candidate Q；真实a_t one-step bootstrap Huber，不加关系truth标签 |
| PPO | 有效样本mean，不乘prefix w；时长Gamma仍入reward/TD/GAE/targets |
| Discount | 共同suite half-life H=max reference median，gamma=2^(-1/H) |
| Prior train | episode80/20原始/整份空，fixed；无人工错边 |
| Core baselines | B0-SAB/B1-K/B2/Full；A_CAT取代A_DD必跑；A_Q Tier2；A_B conditional |
| First batch | 11个单seed/烟雾运行先暂停；补齐3seed累计29；含A_Q累计32 |
| New preflight | 0D奖励暴露与决策结构审计，无5%硬门槛 |
| Evidence | 没有本轮RL/API/机器人结果；数学与文件检查不代替生产验证 |
| Future | M1+独立，不阻塞当前论文 |

# Paper Readiness

**No method-level blocker remains.** Current Method is frozen for writing; subsequent experiments validate claims rather than decide whether the paper can be written.

Introduction/Problem/Method已提供可继续编辑的正文；Setup采用Plan1.1明确规则，真实平台、资产和运行manifest待绑定；Results只有模板及问题，不填写未测数字。贡献不要求预先保证Full获胜，A_CAT更好时要改变经验主张而非掩盖。正式写作现在开始，任务安全与软件验收仍须先于真实执行。

# 附录A. 来源、版本与兼容性记录

## A.1 项目来源

| ID | 材料 | 在本文件中的用途 |
|---|---|---|
| S1 | Pasted markdown(10).md | 当前论文导向完整规范、v3.0文档结构、方法v2.1不新增模块 |
| S2 | CP_DISR_M1_Research_Specification_v2.0(1).md | 合同、三态、共享编码、Actor/V/Q、PPO、运行接口的技术底稿 |
| S3 | CP_DISR_v2.1_Final_Experimental_Agent_Prompt.md | 历史阶段组织与Private Paper原则；被本轮Plan1.1明确修订处以新版本为准 |
| S4 | CP_DISR_feedback_chain_review_prompt.md | 反馈链、逐边归因与源头更新问题的范围 |
| S5 | Pasted markdown(9).md | Current Paper与Future Upgrade分离、关系决策用途的讨论要求 |
| S6 | Pasted text(7).txt | 纯DP、删除SOFT_ORDER及训练先验协议决议的原始问题 |
| D1 | 当前对话已确认的v2.1决议 | 两类后果关联schema、必填effect_ref、纯DP、80/20、明确测试编辑 |
| D2 | 当前对话反馈链审查结论 | 固定源头是合理边界；Q监督用途而非关系纠错 |
| D3 | 当前对话M1/M1+结论 | 当前可以负向修正；M1+只作未来预测误差经验方向 |

历史v3.0的原件索引保留在归档ZIP内；本轮实际来源及SHA-256在sources/input_manifest.json。较早回复链接中的生成版v2.1规范和执行包在本轮挂载文件中未提供，因此不声称已重新读取或核验其二进制内容；本稿依据上传原件与明确对话决议重建统一说明，不为缺失文件编造hash。

## A.2 显式兼容选择，不静默改变方法

| 问题 | 本稿处理 |
|---|---|
| 文档版本与方法版本 | 当前文档3.1/方法2.1.1；原文档3.0/方法2.1为历史来源 |
| 节点观测/可执行性输入位置 | 保留冻结v2.1的图结构特征与共同观测/候选通路；第13.1节明示差别 |
| VLM词表与旧schema | 只使用m1_soft_relations_v2、两类关系与effect_fact_ref；不复用旧schema名称 |
| Actor / V / Q串行误解 | 统一写成共享部分特征的并行head |
| Q纠错／DP因果值措辞 | 改为实际结果支持的表示与先验用途学习 |
| 80/20比例地位 | 当前训练协议固定；数值属于实验协议而非差分数学定理 |
| 旧回复训练预算/软件版本不一致 | 本轮Plan1.1重新明确smoke16,384、主训练65,536及评价间隔；实际H、d_ref、时间cap和目标机软件锁仍须绑定，历史冲突预算不混用 |
| v2.0投稿式实验要求与S3 | 继承S3明确的有限探索与Private Paper View；选择证据不冒称未经筛选 |
| 论文式详细头部实现 | 第22节普通attention/MLP作为明确实施默认，不声称已训练或创新 |
| Future M1+ | 仅第33节及Discussion提及，不进入主配置与必跑实验 |

所有MUST_BIND必须由实际资产、代码、设备或运行配置填入；本文不能替代账号授权、机器人安全校准或完整生产测试。

# 附录B. 可复制配置与Agent交接约束

核心配置由随附interfaces/paper_method_manifest.yaml维护：方法版本2.1.1、文档版本3.1、Experimental Plan1.1，future_m1_plus_enabled=false。其他文件含关系schema、固定提示词、runtime绑定模板与结果模板。它们是实现依据，不是完整训练器。

Agent读取本文件后，首先匹配实际代码与schema/contract/feature profile，补齐runtime绑定，再实现或检查第22.5节不变量。只执行用户明确授权的Stage；本轮没有授权或运行新的训练、机器人动作或VLM缓存生产。若用户要求继续写作，直接使用第35–40节并保留未测占位，不等待全部实验。

接口校验需拒绝当前运行中仍出现旧SOFT_ORDER、训练relation_rewrite、static prior或Future M1+损失。实际图/GRU张量应在优化时重算；配置即使可解析，也不证明真实控制器、ID和图registry已经接通。

# 附录C. 参考文献与核对范围

下列基础参考继承v3.0所记录的2026-09-21核对；并非本轮已重新逐项核验的声明。本轮2026-09-22独立研究及全文/摘要/未确认等级另列在W来源索引。引用不证明CP-DISR的效果或首创性，API文档也不证明当前账号与调用结果。

[R1] Sutton, R. S., Precup, D., Singh, S. **Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning.** Artificial Intelligence, 112(1–2):181–211, 1999. DOI: 10.1016/S0004-3702(99)00052-1. https://www.sciencedirect.com/science/article/pii/S0004370299000521

[R2] Chen, D. Z., Thiébaux, S., Trevizan, F. **Learning Domain-Independent Heuristics for Grounded and Lifted Planning.** AAAI 38(18):20078–20086, 2024. DOI: 10.1609/aaai.v38i18.29986. https://ojs.aaai.org/index.php/AAAI/article/view/29986 ; 原文HTML：https://arxiv.org/html/2312.11143v2

[R3] Toyer, S., Trevizan, F., Thiébaux, S., Xie, L. **ASNets: Deep Learning for Generalised Planning.** JAIR, 68:1–68, 2020. DOI: 10.1613/jair.1.11633. https://arxiv.org/abs/1908.01362

[R4] Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., Welling, M. **Modeling Relational Data with Graph Convolutional Networks.** ESWC, 2018. arXiv:1703.06103. https://arxiv.org/abs/1703.06103

[R5] PyTorch Geometric. **RGCNConv, version 2.6.1 documentation.** 接口依据：relation types、mean、root、num_bases。https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.RGCNConv.html

[R6] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. **Proximal Policy Optimization Algorithms.** 2017. arXiv:1707.06347. https://arxiv.org/abs/1707.06347

[R7] Agia, C., Migimatsu, T., Wu, J., Bohg, J. **STAP: Sequencing Task-Agnostic Policies.** ICRA, 2023. DOI: 10.1109/ICRA48891.2023.10160220. https://arxiv.org/abs/2210.12250

[R8] Ahn, M. et al. **Do As I Can, Not As I Say: Grounding Language in Robotic Affordances.** 2022. arXiv:2204.01691. https://arxiv.org/abs/2204.01691

[R9] Rana, K., Haviland, J., Garg, S., Abou-Chakra, J., Reid, I., Suenderhauf, N. **SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning.** CoRL, PMLR 229:23–72, 2023. https://proceedings.mlr.press/v229/rana23a.html

[R10] Jaderberg, M. et al. **Reinforcement Learning with Unsupervised Auxiliary Tasks.** ICLR, 2017; arXiv:1611.05397. https://arxiv.org/abs/1611.05397

[R11] Cobbe, K. W., Hilton, J., Klimov, O., Schulman, J. **Phasic Policy Gradient.** ICML, PMLR 139:2020–2027, 2021. https://proceedings.mlr.press/v139/cobbe21a.html

[R12] Schulman, J., Moritz, P., Levine, S., Jordan, M., Abbeel, P. **High-Dimensional Continuous Control Using Generalized Advantage Estimation.** ICLR, 2016; arXiv:1506.02438. https://arxiv.org/abs/1506.02438

[R13] Nota, C., Thomas, P. S. **Is the Policy Gradient a Gradient?** AAMAS, 2020; arXiv:1906.07073. https://arxiv.org/abs/1906.07073

[R14] Jiang, Y. et al. **VIMA: Robot Manipulation with Multimodal Prompts.** ICML, PMLR 202:14975–15022, 2023. https://proceedings.mlr.press/v202/jiang23b.html

[R15] Farama Foundation. **Handling Time Limits.** Gymnasium官方文档。https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/

[R16] Alibaba Cloud Model Studio. **qwen3.8-max Model Info.** 包含qwen3.8-max-0902快照的公开说明。https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max

[R17] Alibaba Cloud Model Studio. **Structured Output / 结构化输出.** JSON Object只保证格式而非项目语义；本地验证仍必要。https://www.alibabacloud.com/help/zh/model-studio/qwen-structured-output

[R18] Huang, W. et al. **Inner Monologue: Embodied Reasoning through Planning with Language Models.** 2022. arXiv:2207.05608. https://arxiv.org/abs/2207.05608


# 附录D：本轮独立核验与执行版本

本轮修订理由、Math/配置测试、来源核验等级分别在review与validation目录；Plan1.1为执行依据。以下预算、Actor prefix=false、half-life与A_CAT不是旧source内容，而是用户本轮授权后的明确决议。

# 来源与核验等级

[U] 当前用户复核指令；[C] Claude审查稿；[V3] 原v3.0研究规范；[EP0] 原实验Agent Prompt。四者不混为外部研究证据。原始字节和hash见sources。

**[W01] Nota & Thomas, Is the Policy Gradient a Gradient?** — AAMAS 2020；FULL_TEXT_METHOD_CHECKED。

依据：外层折扣省略与起点折扣目标不等价；不能据此要求所有实现必须加权，也不能称去权仍无偏。

地址：`https://arxiv.org/html/1906.07073`

**[W02] Stable-Baselines3 PPO source** — version 2.7.0；OFFICIAL_SOURCE_CHECKED。

依据：clipped surrogate使用mean而无episode-prefix权重；不据此声称它原生支持本项目GRU/动态候选/SMDP。

地址：`https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/ppo/ppo.html`

**[W03] Ståhlberg, Bonet & Geffner, Learning General Policies with Policy Gradient Methods** — KR 2023；FULL_TEXT_METHOD_CHECKED。

依据：GNN state-transition classifier与策略梯度/actor-critic；arXiv2512.19366是后续deposit，不把编号年份当会议年份。

地址：`https://proceedings.kr.org/2023/63/kr2023-0063-stahlberg-et-al.pdf`

**[W04] Counterfactual Quotient Models: Learning What Actions Change, Not What the World Does** — arXiv 2608.22092, 2026；FULL_TEXT_METHOD_CHECKED。

依据：行动间中心化的未来量/同步反事实rollout目标；不是本项目prior×nominal编码2×2；其定理不移植。

地址：`https://arxiv.org/html/2608.22092`

**[W05] Yan et al., Efficient Reinforcement Learning with Large Language Model Priors** — ICLR 2025; arXiv 2410.07927；FULL_TEXT_METHOD_CHECKED。

依据：状态条件化动作先验p(a|s)参与KL/采样/Q；不能称为状态不变static prior。

地址：`https://arxiv.org/html/2410.07927`

**[W06] Guiding Reinforcement Learning Using Uncertainty-Aware Large Language Models** — arXiv 2411.14457, 2024；FULL_TEXT_METHOD_CHECKED。

依据：MC dropout与动态熵调节guidance，并涉及LLM定制/微调；不是本项目固定图对照。

地址：`https://arxiv.org/html/2411.14457`

**[W07] Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents / LLM4Teach** — IJCAI 2024; arXiv 2311.13373；FULL_TEXT_METHOD_CHECKED。

依据：不完美teacher软动作指导/蒸馏与衰减；不作四视图结构交互。

地址：`https://arxiv.org/html/2311.13373`

**[W08] The Mechanism Matters: When Knowledge Graphs Help Reinforcement Learning** — arXiv 2607.19616, 2026；FULL_TEXT_METHOD_CHECKED。

依据：比较feature/mask/shaping及KG质量；有限实验或最优策略不变性不证明本项目坏prior无害。

地址：`https://arxiv.org/html/2607.19616`

**[W09] Leveraging Approximate Symbolic Models for Reinforcement Learning via Skill Diversity / ASGRL** — ICML 2022；FULL_TEXT_METHOD_CHECKED。

依据：近似符号模型、landmarks、技能多样性；不完美符号知识+RL是已有设定。

地址：`https://proceedings.mlr.press/v162/guan22c/guan22c.pdf`

**[W10] Subgoal Graph-Augmented Planning for LLM-Guided Open-World Reinforcement Learning** — arXiv 2511.20993, 2025；FULL_TEXT_METHOD_CHECKED。

依据：子目标图与critique/refinement路径，与固定源关系使用不同。

地址：`https://arxiv.org/html/2511.20993`

**[W11] Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning** — NeurIPS 2023; arXiv 2305.14909；FULL_TEXT_METHOD_CHECKED。

依据：显式规划模型构造、验证/修正、规划；非本项目软关系残差。

地址：`https://arxiv.org/html/2305.14909`

**[W12] InterPreT: Interactive Predicate Learning from Language Feedback for Generalizable Task Planning** — RSS 2024; arXiv 2405.19758；FULL_TEXT_METHOD_CHECKED。

依据：根据交互语言反馈学习谓词与动作模型；不是冻结关系决策利用。

地址：`https://arxiv.org/html/2405.19758`

**[W13] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Robot Task Planning** — arXiv 2507.21545, 2025；FULL_TEXT_METHOD_CHECKED。

依据：从示范学习/构造可复用PDDL domain，与soft message edges不同。

地址：`https://arxiv.org/html/2507.21545`

**[W14] Vision-Language Models Provide Promptable Representations for Reinforcement Learning / PR2L** — arXiv 2402.02651, 2024；FULL_TEXT_METHOD_CHECKED。

依据：本轮独立补充；VLM提供语义表示，由RL学其用途，因此非动作指令式VLM+RL并非新领域。

地址：`https://arxiv.org/html/2402.02651`

**[W15] Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks** — ICML 2019；FULL_TEXT_METHOD_CHECKED。

依据：SAB/MAB/PMA集合注意力与置换性质；本项目仅使用标准两层SAB+candidate query适配。

地址：`https://proceedings.mlr.press/v97/lee19d/lee19d.pdf`

**[W16] High-Dimensional Continuous Control Using Generalized Advantage Estimation** — ICLR 2016; arXiv 1506.02438；FULL_TEXT_METHOD_CHECKED。

依据：GAE/TD(lambda)偏差方差取舍；one-step并非普遍偏差最小。

地址：`https://arxiv.org/html/1506.02438`

**[W17] Reinforcement Learning with Unsupervised Auxiliary Tasks / UNREAL** — ICLR 2017; arXiv 1611.05397；FULL_TEXT_METHOD_CHECKED。

依据：共享表示的辅助任务是已有原则；并非声称本Q严格复现UNREAL。

地址：`https://arxiv.org/html/1611.05397`

**[W18] Residual Reinforcement Learning for Robot Control** — ICRA 2019; arXiv 1812.03201；FULL_TEXT_METHOD_CHECKED。

依据：本轮独立补充；既有控制+学习残差，区别在本项目logit界与交互输入。

地址：`https://arxiv.org/html/1812.03201`

**[W19] Modeling Relational Data with Graph Convolutional Networks** — ESWC 2018; arXiv 1703.06103；FULL_TEXT_METHOD_CHECKED。

依据：关系权重共享、basis与aggregation；不是对当前task门控的证明。

地址：`https://arxiv.org/html/1703.06103`

**[W20] Phasic Policy Gradient** — ICML 2021；ABSTRACT_OR_OFFICIAL_PAGE。

依据：辅助阶段与特征共享干扰的背景；未据摘要细节证明当前lambda=0.1无风险。

地址：`https://proceedings.mlr.press/v139/cobbe21a.html`

**[W21] Time Limits in Reinforcement Learning** — ICML 2018；ABSTRACT_OR_OFFICIAL_PAGE。

依据：终止与时间限制处理背景；本项目具体flag定义由规范负责。

地址：`https://proceedings.mlr.press/v80/pardo18a.html`

**[W22] Planning with Learned Object Importance in Large Problem Instances using GNNs / PLOI** — AAAI 2021；ABSTRACT_OR_OFFICIAL_PAGE。

依据：对象重要性筛选，不是直接候选后继策略的同义工作。

地址：`https://ojs.aaai.org/index.php/AAAI/article/view/17421`

**[W23] LLM+P: Empowering Large Language Models with Optimal Planning Proficiency** — arXiv 2304.11477, 2023；ABSTRACT_OR_OFFICIAL_PAGE。

依据：自然语言到规划问题/外部planner；不把全部LLM-PDDL都概括为学习新的动作模型。

地址：`https://arxiv.org/abs/2304.11477`

**[W24] Graph Learning for Planning: The Story Thus Far and Open Challenges** — arXiv 2412.02136, 2024；ABSTRACT_OR_OFFICIAL_PAGE。

依据：只作综述导航，不以综述取代原创论文核验。

地址：`https://arxiv.org/abs/2412.02136`

**[W25] Afterstate RL for Continuous Control（Claude给出的条目）** — NOT_INDEPENDENTLY_CONFIRMED；UNCONFIRMED_ACCESS_BLOCKED。

依据：访问返回浏览器验证页，本轮未独立确认该题目/作者/正文，不作为已确认核心依据。

地址：`https://openreview.net/forum?id=XO944P8prc`
