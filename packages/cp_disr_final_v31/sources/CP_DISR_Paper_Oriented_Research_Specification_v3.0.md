---
title: "CP-DISR"
subtitle: "Paper-Oriented Research Specification v3.0"
date: "2026-09-21 · Current Method: CP-DISR / M1 v2.1"
lang: zh-CN
---

**固定不完美语义先验下的候选条件化结构决策学习**

*Learning to Use Fixed Imperfect Semantic Priors for Long-Horizon Skill Selection*

## 文档控制

本文件是当前方法的统一论文底稿与实现规范。**文档版本为v3.0，算法版本仍为CP-DISR / M1 v2.1**；版本号变化表示论述、信息流、实现说明和证据结构的整合，不表示加入新的算法模块。M1+仅在第33节作为未来研究方向出现，不进入当前前向、损失、配置或必跑实验。[S1，§1–2、§59–60]

本文件以本轮《Pasted markdown(10).md》为最高内容要求，以现存v2.0规范为技术基底，应用当前对话已经确定的v2.1关系schema、纯交互先验、80/20训练协议和反馈链解释。实验组织继承用户的分阶段实验指令；**论文级组织目标不自动撤销私人探索中已明确允许的开发调参和展示选择，选择后的证据须如实标注。**来源、取舍和运行配置边界见附录A。[S1–S6]

全篇采用四种状态：**方法定义**可立即用于写作；**实现默认**用于确定性实现但不是最优参数结论；**MUST_BIND**表示实际资源或运行配置尚需绑定；**待验证**表示需要真实实验才能主张的效果。本文不继承历史PathGraph/V6/P2B的性能数字作为CP-DISR结果，不宣称本轮已运行训练、调用VLM、验证机器人或获得策略收益。

文中[S#]引用项目材料及其章节，[R#]引用已核对的一手论文或官方文档。算法定义与本文推导不冒称外部文献的原始贡献；文献结论不外推为本方法的性能保证。英文Abstract与中英文贡献可直接作为论文初稿；中文正文草稿仍需根据最终投稿语言、平台与结果编辑。

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

**候选名义结构干预及其合同—先验交互，能否在匹配可用信息与交互预算时，形成比只读取当前图更有用的技能决策表示？**研究进一步分离合同干预、VLM增量、第二次减法、Q辅助以及有界残差的作用，而不把全部效果归给单一模块。

## 2.3 Core hypothesis

名义效果提供可计算的候选差异，关系编码可把局部事实变化与后续技能、目标相联系，双差分组织软关系对这种响应的影响；真实序贯经验可能使这些特征成为可用的决策依据。这是假设，不预设潜在差分天然具有价值语义，也不预设Full优于所有基线。[S2，§1.3]

# 3. Motivation

长时程技能选择难在局部合法性不足以决定长期效用。先抓哪个对象、是否先打开容器、是否需要临时搬移，可能影响后续可执行性与返工。技能库提供“能调用什么”，合同提供“何时允许调用、名义成功后改变什么”，但不必包含每个场景的全部空间联系和执行风险。技能组合研究已经关注跨技能依赖，不能把“考虑后续影响”本身作为新发明。[R1，R7]

VLM可能根据初始场景提供合同未显式编码的联系，例如一个已注册搬移动作的名义后果可能支持另一对象的后续放置。但语义上可能成立的关系，未必对当前选择有用：目标可能已完成、准备动作成本可能太高、执行条件可能不利。关系语义正确性与情境中的决策用途因此必须分开。

核心矛盾是：直接把软关系作为硬计划或启动要求会给予它过高权限；完全忽略又可能失去有用联系；仅读取整张增强图则不显式分离“候选会改变什么”与“软关系怎样改变对这一变化的解释”。CP-DISR选择候选条件化表示：先在合同上构造变化，再让先验影响变化的编码，而非直接让VLM输出动作优先分。

例如OPEN的名义效果使Open(box)成立，合同本身即可联系到后续PLACE；这属于合同推理，不能重复包装成VLM知识。真正额外的场景联系可以是MOVE(red,buffer)可能支持PLACE(blue,box)，前提是MOVE、其效果及对象绑定已经存在。该软关系不把PLACE成功、ClearPath真值或新的机器人能力写入系统。[S2，§5.3、§9.6；S3，§1.2]

# 4. Paper Positioning

正式名称保留为 **CP-DISR：Contract–Prior Differential Intervention for Skill Reasoning**，中文为“合同—先验差分干预技能推理”。CP指合同与先验两种结构来源，DI指候选名义干预后的差分计算，SR指这些表示用于技能推理。M1仅作项目内部编号；名称中的reasoning指具体结构计算与学习，不意味着完备逻辑证明、物理因果识别或VLM自由规划。

建议论文题目为：**CP-DISR: Learning to Use Fixed Imperfect Semantic Priors for Long-Horizon Skill Selection**。这一名称与当前算法吻合，无需重命名为trust、repair或correction。命名是项目工作名，不声称通过了穷尽式命名或新颖性检索。

论文领域定位为 **structured decision learning with fixed imperfect semantic priors**。Skill Contract、SLG式图、标准R-GCN、PPO、Verifier和缓存都是采用或适配的底座。拟议方法贡献在候选计算、双差分接口及实际结果支持的表示训练组合，而不在创建新的知识图谱、VLM微调或新规划器。[S1，§3–4；S2，§2.3]

**一句话论文故事：**我们在来源明确的技能合同上显式比较候选名义后果，分离固定语义关系对这些变化的编码影响，再由真实执行经验学习这种信息何时值得影响长期技能选择。

# 5. Core Contributions

## C1. Candidate Nominal Structural Intervention

**中文。**在统一的grounded Action–Proposition合同表示中，对每个允许执行的技能构造只读名义成功事实副本，显式保留增加、删除及注册未知效果，以不访问实际未来的方式提供候选后果的结构输入。

**English.** We formulate candidate nominal structural interventions over grounded skill contracts. For each executable skill, a read-only temporary fact assignment represents its nominal success effects without accessing actual future states or modifying the real environment.

## C2. Contract–Prior Dual Difference

**中文。**在合同结构和语义增强结构下，用同一编码器与目标索引计算相同候选干预的响应，并以双差分组织合同变化及先验交互；合同信息进入主评分，先验交互仅通过零锚定、有界的候选残差影响选择。

**English.** We introduce a contract–prior dual-difference representation that compares the same candidate-induced change under contract-only and semantically augmented structures. A zero-anchored bounded residual incorporates this interaction into a contract-based skill policy.

## C3. Outcome-Grounded Structural Learning

**中文。**以真实执行动作、任务奖励、耗时和旧V的bootstrap目标训练辅助Structural Q，并与PPO和独立V共享结构特征，使候选表示接受实际长期决策结果的学习压力；该监督不验证、修复或更新源关系。

**English.** We train the shared candidate representations with skill-level PPO and auxiliary action-value supervision derived from actual executions. This outcome-supported learning targets the decision use of fixed priors, rather than the correctness or revision of individual source relations.

以上是方法性贡献描述，不包含尚未获得的性能结论。名义前瞻、关系卷积、向量减法与辅助价值监督分别具有既有基础；组合是否带来增量由第23–30节的实验辨别。[R2–R6，R10–R11]

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

令$X_t$表示带不可变引用的决策snapshot，包括$x_t$、图模板、固定先验版本、候选、mask和goal_refs。策略实际还依赖GRU历史状态；下文$V(X_t)$、$Q(X_t,a)$等写法均省略已明确的因果历史前缀。有限GRU或事实表不被声称必然构成完整马尔可夫状态。技能的可变执行时间采用SMDP/Options形式，而信息接口仍可能部分可观测。[R1]

## 7.2 时间与奖励

动作$a_t=(k_t,\operatorname{args}_t)$真实执行$d_t>0$个统一基本时间单位。令$\bar\gamma\in(0,1)$为每基本单位的折扣，则

$$
\Gamma_t=\bar\gamma^{d_t},\qquad
r_t=\sum_{k=0}^{d_t-1}\bar\gamma^k\rho^{env}_{\tau_t+k}.
$$

以上离散记法假定$d_t$以基本tick计数。若使用秒，则记实际时长$\Delta\tau_t$与参考单位$u$，实现为$\Gamma_t=\bar\gamma^{\Delta\tau_t/u}$；带时间戳的奖励事件按各自偏移折扣。每秒0.99只是当前起步配置，不是已校准的物理参数。

环境事件奖励仅在首次独立确认整项任务成功时为1，其余为0。技能成功、恢复、子目标、图变化和VLM判断不另给奖励。成功事件的原始1与折扣后的技能奖励$r_t$分别保存，不能把发生在技能中途的事件一律搬到动作起点。

## 7.3 目标与结束标志

$$
J(\theta)=\mathbb E_{\pi_\theta}\!\left[\sum_t w_t r_t\right],
\qquad w_0=1,\quad w_{t+1}=w_t\Gamma_t.
$$

目标是时间折扣下的任务成功，不等同完全忽略时间的成功概率。真实成功、任务期限耗尽、安全终止及定义明确的不可恢复失败属于terminated；单技能失败或超时通常只是一次转移，若任务仍可继续就不终止episode。

有效外部截断与rollout切段需要在最后有效状态bootstrap，不能使用reset后的观测；跨reset不递推优势。无有效末状态的基础设施异常单独标注，不能补造下一状态。这些边界与常见环境的termination/truncation区分一致，但具体任务结束规则必须绑定。[R15]

## 7.4 符号索引

| 符号 | 含义 |
|---|---|
| $\mathcal U,\mathcal K,\mathcal P,\mathcal O_e$ | 技能库、合同、谓词注册表、episode对象目录 |
| $F_t,\mathcal T,R_e$ | 当前三态事实、独立任务、固定软关系 |
| $G_t^K,G_t^H,\widetilde G_t^{K,i},\widetilde G_t^{H,i}$ | 合同/增强当前与候选名义后继四视图 |
| $Z_t^K,Z_t^H,D_i^K,D_i^H,D_i^P$ | 结构编码、两种变化及其交互差分 |
| $z_t^o,h_t,c_i,u_i^K,u_i^P$ | 观测特征、历史状态、候选上下文及两分支特征 |
| $b_i,\Delta_i,\ell_i,B$ | 合同评分、先验修正、总评分、修正上界 |
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

这里$f_P^{node}$与后文先验候选读出$F_P$不同。schema参数共享，实例ID用于稳定引用而不成为任意学习标签。

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

# 14. Contract–Prior Dual Difference

## 14.1 正式定义

$$
D_{t,i}^K=E_\psi(\widetilde G_t^{K,i})-E_\psi(G_t^K),
$$

$$
D_{t,i}^H=E_\psi(\widetilde G_t^{H,i})-E_\psi(G_t^H),
$$

$$
\boxed{D_{t,i}^P=D_{t,i}^H-D_{t,i}^K.}
$$

$D^K$是名义事实变化在合同图中的编码响应；$D^H$是同变化在软关系参与下的响应；$D^P$表示两种响应的交互差异。这不是两张邻接矩阵直接相减，也不直接输出某条边的贡献。[S1，§22–25]

## 14.2 Pure Interaction Prior与零性质

简写$E(F,R)$，有

$$
D_i^P=E(\widetilde F_i,R)-E(F,R)
      -E(\widetilde F_i,\varnothing)+E(F,\varnothing).
$$

若$E(F,R)=A(F)+B(R)$完全可加分离，则$D_i^P=0$；这是有意抵消与本次变化可加分离的静态项，而非无损保存一切VLM偏好。空prior直接复用合同视图，使$D^P=0$；空patch在没有伪造其他变化时使$D^K=D^H=D^P=0$。

静态边不变不等于消息不变。在线性诊断$E(F,R)=M_Rx(F)+b_R$下，有$D_i^P=(M_R-M_\varnothing)\delta x_i$，因此无需非线性也可能产生交互。关系权重为零、投影抵消、归一化不敏感或读出忽略该分量时，响应也可能很小。该推导是可表达性说明，不是主网络必然有效的证明。

## 14.3 对照与解释边界

例如已注册效果命题$q$变化，可经$q\to\operatorname{MOVE}$的反向ADD消息，再经软支持边到下游动作。这是传播可能性，而非逻辑效果沿软边传播。即使目标局部无路径，全局非线性读出仍可能混合其他分量；不能仅数goal邻接跳数就断言整个$D^P$为零。

相同编码器为相减提供共同坐标，不能保证latent差异天然语义线性。$D^P$不是关系正确率、VLM confidence、因果效应、成功率增量或动作价值；错误关系也可能产生很大的差分。当前方法提供候选表示，最终用途由策略学习和实验评估。

# 15. Policy Architecture

## 15.1 观测、记忆与候选上下文

冻结前端的真实特征经可训练融合器和GRU：

$$
(z_t^o,h_t)=f_{obs,\phi}(x_t,h_{t-1}),\qquad
c_{t,i}=[z_t^o,e(a_i),\operatorname{Pool}(Z_t^K)].
$$

GRU只读实际观测、任务、核心事实与执行摘要，不直接读取VLM图embedding。候选$e(a_i)$含共享技能embedding、有序对象/目标角色、真实合同状态与允许的当前几何。保留当前$Z^K$，避免只有变化量而丢失绝对任务背景。[S2，§10.1]

## 15.2 合同主分支

$$
u_{t,i}^K=F_K(c_{t,i},D_{t,i}^K),\qquad
b_{t,i}=w_K^\top u_{t,i}^K+b_K.
$$

$F_K$对固定goal行作候选查询与池化，再融合上下文。它不是observation-only分支，而是合同推理分支。其分数不直接等于Q，不受先验幅度上界$B$限制。

## 15.3 零锚定先验分支

$$
\boxed{u_{t,i}^P=
 F_P(c_{t,i},u_{t,i}^K,D_{t,i}^P)
 -F_P(c_{t,i},u_{t,i}^K,0).}
$$

$$
\boxed{\Delta_{t,i}=B\tanh(w_P^\top u_{t,i}^P),\qquad
\ell_{t,i}=b_{t,i}+\Delta_{t,i}.}
$$

两次$F_P$共享参数、上下文、goal mask和确定计算。$w_P$之后无独立bias、无额外无界倍率。因此$D_i^P=0$时$u_i^P=\Delta_i=0$。禁止旁路读取当前$Z^H$、增强图候选局部embedding、静态VLM分数或抽样模式标签。

$\Delta_i\in[-B,B]$允许正向利用、接近零和负向修正。这里的“有用”是当前决策用途，不是关系语义真伪。共享参数仍可以依据不同输入产生不同响应，但同类型边可能发生梯度干扰，训练也可能主要依赖其他特征；上述能力不能写成“自动识别并拒绝所有错边”。[S1，§26–28]

## 15.4 执行mask与策略

$$
m_t=\mathrm{ExecMask}(\mathcal A_t,\mathcal K,F_t,\mathcal S),\qquad
\pi_\theta(a_i\mid X_t)=
\frac{m_{t,i}\exp(\ell_{t,i})}{\sum_j m_{t,j}\exp(\ell_{t,j})}.
$$

实现用masked log-softmax。mask来自真实能力、参数、合同和安全规则，不能被VLM关系放宽。全部mask为0时不构造非法Categorical，不伪造默认动作；交由独立终止规则记录。图内保留不可执行动作供消息传播，但只为真正允许的候选构造名义视图。

## 15.5 有界直接影响的准确性质

同参数、同实际历史、同输入和同mask下，把先验残差置零得到$\pi_K$。因为$|\Delta_i|\le B$，对允许动作有

$$
e^{-2B}\le\frac{\pi_\theta(a_i\mid X_t)}{\pi_K(a_i\mid X_t)}\le e^{2B}.
$$

证明只需写成$\exp(\Delta_i)/\sum_j\pi_K(a_j)\exp(\Delta_j)$；分子与分母都位于$[e^{-B},e^B]$。这是本文对既定残差的代数整理，不是新安全定理。基础logit差超过$2B$时该残差不能单独翻转顺序；小差距仍可翻转。所有候选共同增加同一个常数不改变softmax，因此残差大小不等于行为影响。

该界不限制先验经共享参数训练和已执行轨迹产生的全部间接影响，不保证任务成功或长期回报不下降。B的当前默认0.5是工程初值，不是已证明最优。

# 16. Structural Q

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

Q没有监督“哪条边错了”，也不能可靠分解VLM错误、感知错误、控制器失败、探索噪声与旧V误差。模型可能通过多处参数降低误差，而不一定压低某条关系的影响。当前C3应称为actual-outcome-supported action-value learning，而不是prior correction。[S1，§30–31]

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

## 17.2 PPO目标

$$
\varrho_t(\theta)=
\frac{\pi_\theta(a_t\mid X_t,h_{t-1}^{\theta})}
     {\pi_{old}(a_t\mid X_t,h_{t-1}^{old})},
$$

$$
L_{clip}=\mathbb E_{w_t}\left[
\min\left(\varrho_t\widehat A_t,
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)
\right],
$$

$$
\mathcal L_V=\mathbb E_t\left[\operatorname{Huber}(V_\omega(X_t)-y_t^V)\right],
$$

$$
\boxed{\mathcal L_{total}=-L_{clip}+c_V\mathcal L_V+\lambda_Q\mathcal L_Q
-c_H\mathbb E_t[\mathcal H(\pi_\theta)].}
$$

PPO按真实采样与多轮小批优化交替进行。[R6] 此处$\mathbb E_{w_t}$实施为批内按$w_t$归一的加权有效transition平均；V/Q与entropy使用有效样本均值。该起点权重与TD内的时长折扣职责不同；若采样已按该权重重采样，不再重复加权。[R13]

优势可在有效样本上标准化，仅作用Actor，不改V/Q标签。PPO clip、有限GAE、归一化和辅助损失共同构成实际训练近似，不能声称总梯度严格等于原始$J$梯度。Entropy保留一定探索压力，但不会创造从未发生的真实成功标签。

## 17.3 80/20先验协议

设$R_e^0$为经过固定解析、校验与去重，但未按测试语义人工纠错的原始关系：

$$
k_e\sim\operatorname{Bernoulli}(0.8),\qquad
R_e=\begin{cases}R_e^0,&k_e=1,\\ \varnothing,&k_e=0.\end{cases}
$$

首次决策前抽样，episode内、跨rollout边界及PPO重算保持不变。模式标签只记录日志，不额外输入网络。原始为空时不补边，实际空prior episode率为$0.2+0.8\Pr(R_e^0=\varnothing)$。80/20是按episode的采样概率，不保证transition或物理时间的比例。[S3，§1.11]

这是所有prior消费者共享的轻量输入缺省协议，不是第四项创新。训练不做删边、target-swap、人工错边、无关边或relation rewrite。训练看到整份缺省，不表示模型已经学会识别所有错误关系；同参数关先验也不等于独立训练B2。

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

$K$个有效候选通常只需当前两视图加每候选两视图，共$2+2K$，不是每个候选重复计算两张当前图。缓存相同参数版本内的当前编码是计算复用，不改变上述数据依赖。

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
| 候选编码、$F_K/F_P$ | 可更新 | 可更新 | 可更新 | 否 |
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
    Gamma <- duration discount
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
      compute clipped actor objective, V loss and gathered Q loss
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

一批有效快照的图读出形状为$[B_{batch},m_{max}+1,d]$；候选差分为$[B_{batch},K_{max},m_{max}+1,d]$；候选特征为$[B_{batch},K_{max},d]$；logits/Q为$[B_{batch},K_{max}]$，V为$[B_{batch}]$。batch维符号不与先验上界$B$混淆。goal padding、candidate padding和实际执行mask分别管理。

**标准读出实现默认（不新增算法）：**$F_K$以候选context投影为query，对差分目标行作单头scaled-dot attention，合并全局差分行后接MLP；$F_P$同类读出并加入$u^K$，两次锚定共用计算。可采用query/key/value宽128、两层MLP中间宽256、输出128、ReLU、dropout0。$f_g/f_0$与V/Q使用普通共享MLP，具体层表写入resolved config。观测前端输出维度、几何字段与尺度由实际adapter绑定，不能凭空规定传感器数目。

所有四路结构编码做FP32减法；若执行混合精度，仍须显式保留这一减法精度并重新验证零性质。候选/图分块只改变内存峰值，不减少候选、不截掉某些名义视图。训练时可缓存静态图、raw facts、冻结视觉特征和patch，不跨参数更新缓存可学习的Z/D或候选embedding。

## 22.3 默认配置与必填项

| 项目 | 当前默认或状态 |
|---|---|
| Graph | RGCNConv，hidden128，4层，4 bases，mean，root=True，逐节点LayerNorm，dropout0 |
| Observation history | 可训练GRU，hidden128；输入adapter MUST_BIND |
| Actor | 零锚定有界残差，$B=0.5$，无static prior |
| Q | Huber，$\lambda_Q=0.1$，只监督实际动作，target detach |
| PPO | lr$3\times10^{-4}$，clip0.2，GAE$\lambda=0.95$，value0.5，entropy0.01 |
| 更新 | 单一Adam，grad clip0.5，4 epochs，1024 skill transitions/rollout |
| 序列 | sequence16，minibatch64有效转移，前缀重算 |
| Prior | episode级Original0.8、Absent0.2；训练无删边/错边 |
| Discount | 起步每1秒0.99；实际time adapter MUST_BIND |
| VLM | 固定候选qwen3.8-max-0902；region/SDK/endpoint/权限 MUST_BIND |
| Runtime | simulator/robot、资产、controller、camera、Verifier阈值、timeouts、hardware、software lock、Git、splits MUST_BIND |

采用可修改的轻量PyTorch recurrent SMDP-PPO项目实现路线；不把某个现成PPO包默认当作已支持全部动态候选、图重算和Q接口。PyG API以已核对的2.6.1 RGCNConv文档为接口参考，实际torch/PyG/CUDA组合和依赖锁以目标机验证为准。本研究文档不重新裁定此前回复中不一致的软件补丁号或总训练预算；保留明确方法初值，并要求运行manifest给出实际值及hash。[R5；附录A]

## 22.4 失败、终止与归档

控制器退出、验证结果、事实值分字段：TIMEOUT仍可能伴随Held=TRUE；NORMAL也可能UNCONFIRMED。不能因失败把事实回滚到执行前。没有真实动作的reset级终止不伪造Q训练样本；必要时把终止与上一条实际转移按时间正确对齐，保留实现记录。

每run保存resolved config、来源hash、code commit及修改patch、cache manifest、原始episode/decision日志、train/eval metrics、全部checkpoint元数据和失败记录。恢复训练需要RNG、环境/跟踪/事实/GRU前缀及pending buffer；实际环境不能回滚时不声称精确续训。API密钥不写入共享产物。

## 22.5 实现级验收

| 测试族 | 必须验证的结果 |
|---|---|
| 空prior／空patch | DP及相应残差为0；不可制造时钟或事件变化 |
| 事实隔离 | 构造所有名义视图后真实Fact Store、reward、time未变 |
| 节点／目标对齐 | 四视图索引一致，完成目标不删行 |
| 三值与效果 | UNKNOWN不当FALSE；冲突效果拒绝；不变量成立 |
| 关系schema | effect引用/source邻接合法；SOFT_ORDER拒绝 |
| Mask与候选 | old/new概率用同mask；重排后概率相应置换 |
| Q与梯度 | 只gather实际动作；target无梯度；四路E共享且可反传 |
| 时间与边界 | terminated无bootstrap；truncation用有效末状态；GRU仅提交一次 |
| Cache与dropout | key区分输入；episode内固定；PPO不重新抽先验 |

验收失败需修复实现或真实接口，但不要求先获得性能提升才准写论文。本文附带的文件一致性检查也不等于上述生产单测已经通过。

# 23. Experimental Questions

实验围绕方法主张而不是模块数量组织。下列问题均待真实数据回答；architecture allows不替代实验结果。

| RQ | 问题 | 核心对照／证据 |
|---|---|---|
| RQ1 | 候选结构干预相对仅读当前状态/图是否有用？ | B0、B1与Full的整体比较；共同输入、实际学习曲线 |
| RQ2 | 固定语义prior是否提供额外决策信息？ | 独立训练B2与Full；另作同参数关先验 |
| RQ3 | 第二次差分是否优于直接$D^H$？ | Full与同结构A_DD |
| RQ4 | Structural Q是否改善表示用途、学习效率或稳定性？ | Full与A_Q；不能只看Q loss |
| RQ5 | Actor是否实际使用$D^K/D^P$？ | 冻结模型的置零、置换与语义边干预 |
| RQ6 | prior缺失或扰动时行为如何变化？ | 同一checkpoint的五种先验条件 |
| RQ7 | 已见原子技能/谓词能否用于未见组合？ | 无再训练的组合holdout |

B0/B1与Full的差异包含信息组织与网络计算路径，不能声称一次比较就孤立了所有因素。B2检验先验增量，A_DD更精确地隔离第二次减法。当前矩阵不新增更多必跑配置，只把能够和不能够归因的范围写准确。[S1，§43–46]

# 24. Baselines

| ID | 方法 | 输入与计算 | Q监督 |
|---|---|---|---|
| B0 | Observation / Fact Skill Policy | 共同观测、目标、候选、原始合同字段的非图列表/集合表达；不做图干预 | 匹配的实际动作Q |
| B1 | Current Graph Policy | 当前$G^K/G^H$及共同观测，不生成名义后继 | 匹配的实际动作Q |
| B2 | Contract-only Intervention | 独立训练；$G^K$及$D^K$，始终无prior | 与Full一致 |
| Full | CP-DISR v2.1 | 四视图双差分、零锚定有界prior、真实Q | $\lambda_Q$默认0.1 |
| A_DD | No second difference | 保持Full结构，非空prior输入$D^H$而非$D^P$ | 与Full一致 |
| A_Q | Without Structural Q loss | 保持Full前向和参数初始化，仅$\lambda_Q=0$ | 不反传Q loss |
| A_B | Unbounded prior residual | 保持零锚定，去掉tanh | 与Full一致 |

**B0实施默认。**将同一原始事实、goal与候选合同的PRE_POS/PRE_NEG/ADD/DEL/UNKNOWN字段分别作共享token/集合编码和masked pooling；加入相同观测与候选角色后评分。不故意删掉合同字段让它成为弱输入对照，不用任意对象ID嵌入替代可用对象信息。

**B1实施默认。**当前合同候选特征读取$G^K$及观测；当前先验特征由同一函数读取当前$G^H$与当前$G^K$的差获得，再接同尺度的有界头。它可以利用当前图静态语义，这正是该对照与Full的区别；不得把B1的这一输入权限复制到Full。该标准头部组织是实现约定，不声称原文已指定唯一层表。

B2与“独立训练的Full without VLM”同定义，不重复作为另一组。测试时把Full的$D^P$置零是同参数干预，不等于B2的训练结果。旧B3不再作为必跑配置；A_DD保留合同分支，不能称其为旧Single-Enhanced-Graph模型。

所有组共享技能、控制器、参数生成、感知/Verifier、任务目标、候选、mask、奖励与时间口径。prior消费者共享80/20协议和同源cache；B0不读prior时该抽样为no-op，B2始终空。报告实际参数量、计算成本和优化配置，不以填充无用参数冒充严格等容量。

# 25. Ablations

A_DD的输入必须定义为

$$
D_i^{P,\mathrm{A\_DD}}=
\begin{cases}D_i^H,&R_e\ne\varnothing,\\0,&R_e=\varnothing.\end{cases}
$$

若空prior仍输入$D^H=D^K$，就同时改变了缺省退化接口，不能仅归因第二次减法。A_DD其余context、合同分支、Q、mask和预算与Full匹配。

A_Q只移除Q损失权重，不把Q输出重新送给Actor，也不因删除模块改变共享层随机初始化次序。可以保留Q头作不反传诊断，明确其loss不参与总梯度。A_B只用

$$
\Delta_i=B w_P^\top u_i^P
$$

替代$B\tanh(\cdot)$，保持零锚定和相同小信号尺度，不暗加另一种clip或归一化恢复上界。

消融控制共享task/split、运行接口、共同超参数、训练seed、预算、缓存及checkpoint规则；仅改变目标因素。可复用已有匹配Full训练，不重复计作新run；不匹配则显式补跑或报告缺失。当前不增加static-prior或Full-clean-train必跑消融，不独立声称20%缺省训练带来因果收益。

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

五个条件使用同一冻结checkpoint和配对初始case，不分别重训，也不在普通评价中随机执行训练20%dropout。操作只影响语义soft edges，完成后统一生成逆向消息边；合同、事实、候选、mask和reward不变。

| 条件 | 冻结操作 | 适用性与日志 |
|---|---|---|
| Original | 使用原始解析/校验/去重后的关系 | 原始不等于gold；不人工修成正确 |
| No Prior | 整份软关系为空 | 保留其他输入 |
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

## 29.1 任务结构与真实绑定

| family模板 | 结构作用 | 实际需要的资源 |
|---|---|---|
| D0：Development | 简单但多步、至少2个合法候选、一个有效非空prior case | 已有技能/观测；用于debug、smoke和开发 |
| T_A：Shared Prerequisite | 多个后续技能共享一个条件 | 实际OPEN/PLACE等；合同重复不算prior增量 |
| T_B：Multi-Object Choice | 多个当下可执行对象，长期后果不同 | 对象角色、目标绑定与实际可达性 |
| T_C：Intermediate Relocation | 临时移动与另一对象后续操作有关 | 已注册MOVE/缓冲位置及真实效果 |
| T_D：Multi-Goal Interaction | 候选影响多个目标或共享条件组织 | 已绑定多目标任务与相应执行接口 |
| T_E：Recoverable State Change | 已有成果可能失效，需重建 | 独立确认的失效机会和安全恢复能力 |

这些是task structure templates而非已经存在的场景资产。具体simulator/robot、对象数量、控制器、姿态生成、相机、感知、阈值、timeouts、deadline、初始case池、硬件和版本都为MUST_BIND。不要用纯符号toy结果冒充视觉或真机性能。

## 29.2 运行组织与预算

继续使用Part→Stage：0A绑定、0B生产不变量、0C缓存质检、1A最小学习、必要时1B有限调参、2A探索、2B展示、2C主图表、3A/3B/3C消融、4A机制、4B鲁棒、5A组合泛化、6A/6B/6C归档与论文视图。用户只授权一个Stage时完成该阶段即停止，不能因报告了next stage自动跨阶段。[S3，§2–4]

训练起步仍为1024真实技能转移/rollout；每任务总技能上限$N_{cap}$、实际任务时间上限$T_{cap}$、评价和checkpoint间隔为运行manifest必填。**此前对话的总训练预算存在不一致表述，当前附件未指定其中一组作为最终值，本文不静默选择一组冒充已冻结预算。**这些值不影响Method公式，但执行和正式Setup必须精确填写。

每组记录实际技能次数、控制tick、任务/仿真秒数及wall time；失败尝试计入交互。不同方法预算匹配，不能仅按技能次数相同就假定物理时间相同。在线规划、执行与确认是否计入任务时钟由真实adapter定义；不以暂停仿真规避延迟后宣称真机等价。

## 29.3 Seed、checkpoint与评价

Smoke默认seed0、每次10个评价episode；探索默认0/1/2、每checkpoint20个；Presentation默认3个训练seed、每task/seed30个，必要时扩到50。第一轮鲁棒性20–30个case/condition即可，是否补到50由已声明的评价修订记录决定，不预设100–500。[S3，§15–17、§11]

训练采样masked categorical，主评价deterministic argmax。开发、ID测试与组合holdout分开。Checkpoint依据开发集多个连续窗口的表现与稳定性选择，可采用最近3窗口均值而不是单次尖峰；各方法允许不同step，但规则共同、test前登记。

探索阶段可以有限调参、重跑和选择结构任务。Private Paper View允许从完整真实记录选择适合展示的task、seed与checkpoint，保存候选池、选择理由及是否看过结果；不能把所选seed均值与SD当作未经选择的总体估计，也不能为不同方法各挑不相同的有利task集合。[S3，§0、§13、§15–17]

## 29.4 指标定义

| 指标 | 计算与边界 |
|---|---|
| Task success | 独立评价器成功数/有效完整episode数；零成功是0 |
| Learning AUC | 原始成功率曲线在共同实测预算区间积分并归一；不对平滑值积分、不外推 |
| Time-to-threshold | 预声明阈值与连续窗口；未达到记NOT_REACHED |
| Completion time | 成功条件下时间均值/中位数，伴随成功分母与总体失败率 |
| Recovery completion | 独立失效机会后的任务完成率，必须报告机会分母；无机会为NA |
| DK/DP及残差 | 有效候选/goal的norm、非零率、分位数；无适用量为NA，不捏造0 |
| Actor sensitivity | 同状态JS/KL、argmax/排序变化；与闭环成功分开 |
| Cost | 实测编码延迟、峰值显存、视图数与离线API成本，不给虚假硬件估算 |

主表来自原始episode，曲线可用明确记录的EMA或moving average，同时保存原始点和无平滑版。当前研究文档不把历史不同平滑参数统一猜成一个值，选用参数需写入figure manifest。数据选择与显示变换不改原始reward、success或CSV。

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

每个task一幅，横轴为实际交互时间并辅以技能次数版，纵轴为原始任务成功率，叠加B0/B1/B2/Full。图注写seed集合、窗口与平滑方式，原始版本保留。回答RQ1/RQ2/RQ4；AUC用原始数据和共同预算计算。

结论句模板：“在相同[预算口径]下，[方法]于[实际区间]达到[真实阈值/未达到]，曲线面积差为[实测]。”不因最终成功接近就从主表删除成功率；可将AUC作为更有信息的补充。

## 30.3 Table 2 — Ablation

| Task | Full | A_DD | A_Q | A_B | Full control来源 |
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

## 34.1 建议论文组织

| 正文章节 | 主要内容与衔接 |
|---|---|
| 1 Introduction | 技能长期选择→合同与先验的角色→候选后果视角→三项贡献 |
| 2 Related Work | 技能组合、结构策略、语言先验、关系编码、辅助监督；定位利用而非修图 |
| 3 Problem Formulation | 技能时间抽象、观测/事实/候选、合同、prior、独立奖励与目标 |
| 4 CP-DISR | 概览→合同/增强图→候选名义视图→共享编码→双差分→Actor/V/Q→训练 |
| 5 Experiments | Setup与任务→主比较→消融→机制→prior扰动→组合泛化 |
| 6 Discussion | 固定先验利用、信息与执行边界、按实际结果界定适用范围；简短M1+ |
| 7 Conclusion | 回到候选条件化先验利用，概括真实获得的证据，不提前填优势 |
| Appendix | 合同/schema、完整伪代码、时间与GRU处理、配置、验收和完整来源索引 |

正文不需要把工程目录、每个Stage和全部缓存字段都放进Method。第7–22节提供完整实现规范，第35–40节提供可直接编辑的论文正文初稿，二者使用同一符号与权限。

## 34.2 Related Work v0

**长时程技能规划与层次控制。**Options将可持续多个底层时间步的闭环策略纳入时间抽象，为技能级折扣提供基础。[R1] STAP研究技能间几何依赖并在规划中组合策略、Q与动力学组件。[R7] CP-DISR不训练新的技能库，也不以多步动力学规划为核心；关注固定能力之上的候选表示与实际经验训练。

**符号动作模型与结构化策略。**SLG类表示将规划状态与动作—命题关系输入图学习；ASNets利用schema结构共享参数学习通用策略。[R2–R3] 本方法采用这些结构化建模基础，但名义前瞻、三态观测与目标对齐双差分是本文具体组合，不把两类节点或参数共享单独列为新贡献。

**LLM/VLM先验与机器人规划。**SayCan结合语言知识与技能价值提供的可行性信息；SayPlan利用3D场景图和规划反馈；Inner Monologue把环境反馈送入语言推理过程。[R8–R9，R18] 本方法不让语言模型在线决定序列或修订计划，而在初始缓存固定语义关系，后续策略学习关系条件化的决策用途。这里是控制权限与计算组织的差异，不是宣称所有语言规划方法都盲目信任模型。

**图关系推理。**R-GCN提供成熟的多关系消息传递与basis分解，本文使用标准实现。[R4–R5] 原文还研究知识库补全，但使用其编码器不意味着自动具备补边或修图目标；本项目没有链接预测解码器或逐边真值监督。

**辅助监督与表示共享。**UNREAL研究共享表示上的辅助任务，PPG讨论策略/价值学习中的共享收益与目标干扰。[R10–R11] CP-DISR采用的是实际执行动作的辅助bootstrap价值监督，不使用它们的伪奖励、额外辅助任务或分阶段训练。Structural Q的贡献需由对应消融体现，而不是仅凭“辅助监督”这个概念声称新颖。

**不完美先验。**当前问题把软关系视为可能遗漏、冗余或误导的固定条件输入。相关具身规划研究通过可行性与反馈约束语言建议，说明先验与执行证据需要区分。[R8，R18] 本文的具体位置是候选名义后果视角下的先验利用，而非逐关系正确率估计。语义正确但当前无用、语义未证实但具有预测用途两种情况均需允许。以上构成正文相关工作骨架，不是穷尽式截至当前全部文献的新颖性审查。

# 35. Abstract v0

## 35.1 English abstract

Long-horizon robotic tasks require selecting skills whose immediate effects support subsequent decisions. Skill contracts provide explicit nominal preconditions and effects, while vision-language models can supply additional scene-dependent semantic relations. Such relations may be incomplete, redundant, or misleading, making their decision use distinct from their semantic correctness. We present CP-DISR, a candidate-conditioned method for learning to use fixed imperfect semantic priors. For each executable skill, we apply its nominal success effects to temporary fact assignments under both contract-only and semantically augmented graphs. A shared relational graph encoder produces goal-aligned representations, and a dual difference captures how the soft relations modulate the encoder's response to the same candidate-induced change. A contract-based policy incorporates this interaction through a zero-anchored bounded residual. Skill-level PPO and auxiliary action-value supervision derived from actual executions jointly train the shared representations, while the VLM, source relations, contracts, and low-level controllers remain fixed. The method targets conditional prior utilization rather than relation verification or repair. Experiments across [MUST_BIND: task families and platform] will evaluate task completion, learning efficiency, structural dependence, sensitivity to prior perturbations, and compositional generalization. Numerical findings and evidence-qualified conclusions will be inserted after the corresponding experiments are completed.

## 35.2 中文摘要

长时程机器人任务要求选择既能完成局部操作、又支持后续目标的技能。技能合同提供明确的名义前置与效果，视觉语言模型可补充场景相关语义联系，但这些联系可能不完整、冗余或误导，因而语义正确性不能直接等同于决策用途。本文提出CP-DISR，在固定不完美先验条件下学习候选条件化技能选择。对每个可执行技能，我们在合同图和语义增强图中施加相同的临时名义成功事实变化，通过共享关系图编码器得到目标对齐表示，再以双差分刻画软关系如何改变对该变化的编码响应。合同主分支结合零锚定有界先验残差形成策略。技能级PPO与实际执行数据支持的辅助动作价值监督共同训练共享表示，VLM、关系内容、合同和低层控制器保持固定。该方法研究先验的条件化利用而非关系验证或修复。后续将在[MUST_BIND：任务与平台]上评价任务完成、学习效率、结构依赖、先验扰动及组合泛化；结果部分将在获得真实证据后补入。

# 36. Introduction v0

长时程机器人任务通常需要组合多项已有技能。即使每个技能都能独立执行，选择下一步仍不简单：一个局部可行的动作可能消耗后续所需的条件，提前完成的目标可能再次失效，暂时搬移某个物体也可能是完成另一目标的必要准备。高层决策因此需要考虑动作的后续意义，而不只是当前技能是否允许启动。已有技能组合与具身规划研究已经表明，低层能力与任务层面的依赖需要共同考虑。[R1，R7]

技能合同为这一问题提供了明确基础。合同规定技能的参数、启动条件以及名义成功后增加或删除的事实，使系统能够在实际执行前计算一个候选会怎样改变结构状态。但合同不是完整的场景物理模型，也不为每个任务指定唯一顺序。场景中的额外空间联系、不同准备动作的用途及执行风险，未必都被有限合同显式表达。

视觉语言模型可以从任务与初始场景中补充这些联系。与语言知识和机器人能力相结合的已有方法表明，语义先验需要与实际可执行能力相联系。[R8–R9] 然而，VLM关系可能遗漏、重复或误导；即使一条关系语义合理，它在当前状态下也可能没有选择价值。因此，直接把模型建议当作硬约束并不等同于可靠决策，完全忽略先验也可能丢失有用信息。

我们关注的是候选后果，而不是直接的VLM动作推荐。对于每个当前可执行技能，系统依据合同构造一个只读的名义成功事实副本，并分别在合同结构与语义增强结构中编码同一变化。两次前后差及其进一步差分，使模型能够区分合同变化与软关系对这种变化的编码影响。这里的差分不被预先解释为价值或因果贡献，其用途需要通过真实执行经验学习。

据此，我们提出CP-DISR。共享R-GCN产生目标对齐的候选变化表示，合同分支形成基础动作分数，先验交互仅通过零锚定有界残差影响选择。技能级PPO、独立状态价值和辅助Structural Q使用实际动作、真实奖励、真实耗时与下一状态共同训练共享特征，而不更新VLM或改写关系。该架构允许情境相关的正向利用、弱化与负向修正，但不把这种能力接口写成对所有坏先验的拒绝保证。

本文围绕三个方法组成展开：候选名义结构干预、合同—先验双差分，以及真实结果支持的结构表示学习。实验将分别检验总体效用、先验增量、第二次减法、Q辅助、Actor依赖和有限组合泛化。我们的目标不是把固定先验修成正确知识库，而是研究其在什么条件下能够成为长期技能选择的有效信息。

# 37. Problem Formulation v0

我们考虑固定技能库上的部分可观测组合任务。技能schema及其控制器在策略学习中保持固定，每个高层动作$a_t=(k_t,\operatorname{args}_t)$是一个已实例化技能。任务$\mathcal T$给出有限实体、带符号的合取目标、明确约束与时间期限。系统在第$t$个技能边界接收实际观测$o_t$，并从冻结感知与几何/时间验证形成事实状态$F_t\in\{T,F,U\}^{|V_P|}$。当前与过去已到达的信息构成历史，GRU用以提供有限历史表示，但我们不假定该表示使底层过程完全可观测。

固定合同$\mathcal K$规定技能前置、名义效果、结束与验证接口。当前候选集$\mathcal A_t$由类型合法的技能绑定构成，执行mask$m_t$只依据当前真实事实、参数与安全规则。初始化时，冻结VLM根据任务、初始场景和允许ID生成软关系$R_e^0$并缓存。该关系可错但不改变合同、目标或mask。

令$X_t$为包含任务、事实、观察引用、历史摘要、图模板、候选和固定先验版本的决策snapshot。技能持续$d_t$个统一时间单位，折扣为$\Gamma_t=\bar\gamma^{d_t}$，技能奖励为其中真实环境事件的折扣和$r_t$。环境只在首次独立确认整项任务成功时产生单位事件奖励，其他技能进展不单独计分。目标为

$$
\max_\theta J(\theta)=\mathbb E_{\pi_\theta}\left[\sum_t w_t r_t\right],
\qquad w_{t+1}=w_t\Gamma_t,
$$

其中$w_0=1$。真实终止与有效外部截断分开处理，避免把reset状态用于bootstrap。[R1，R15]

研究问题是在上述固定知识与执行接口下学习策略$\pi_\theta$。源关系不被监督为真或假，合同也不接受回报驱动的修改；可学习部分是观测融合、结构编码、候选读出及策略/价值头。训练时每个episode以0.8概率保留原始关系、以0.2概率使用空关系，episode内部固定。评价采用声明的冻结先验条件。

# 38. Method v0

## 38.1 Overview

CP-DISR将系统分为信息源、结构表示和决策学习三层。合同、实际事实和固定语义关系提供信息；共享图编码与候选名义干预组织结构变化；Actor及实际执行数据支持的价值学习决定如何使用这些变化。算法前向不访问真实未来，反向不修改知识源。

## 38.2 Contract and semantic graph views

我们将grounded动作和命题表示为两类节点，合同的正/负前置、ADD与DEL构成类型边，并加入独立的逆向消息关系。目标通过命题符号与固定索引表达。当前图保留全部相关节点，包括已完成目标、FALSE/UNKNOWN命题和暂时不可执行动作。该结构采用已有规划图表示与schema共享思想，编码底座为标准R-GCN。[R2–R5]

增强图与合同图具有相同节点和事实，仅增加经过schema、ID、效果引用和局部冗余校验的软关系。词表只有SOFT_SUPPORTS和SOFT_RELEVANT_TO_GOAL；effect_fact_ref用于确保source关联到既有名义效果，不创建新事实或赋予真实状态。初始化后的关系内容固定。

## 38.3 Candidate nominal interventions

对允许候选$a_i$，合同生成只读事实副本$\widetilde F_t^i$，表示其名义成功后的ADD/DEL/UNKNOWN变化。相同patch用于合同图和增强图。临时操作不修改实际观测、事实存储、时钟、reward或done，也不预测技能成功概率。它为策略提供候选结构输入，实际失败与耗时随后通过真实经验学习。

## 38.4 Shared encoder and dual difference

四视图由相同$E_\psi$编码为目标对齐矩阵$Z=[z_0;z_1;\ldots;z_m]$，其中$z_0$为全局读出，其余为固定目标行。我们定义

$$
D_i^K=E_\psi(\widetilde G_i^K)-E_\psi(G^K),\qquad
D_i^H=E_\psi(\widetilde G_i^H)-E_\psi(G^H),
$$

$$
D_i^P=D_i^H-D_i^K.
$$

$D_i^P$刻画软关系如何改变编码器对同一名义变化的响应。其数值不等于关系正确率、因果贡献或价值增量。若先验影响与干预可加分离，则交互为零；空prior通过复用合同编码得到严格零差分。该设计排除独立静态VLM动作分数，但不保证学得的特征不利用统计相关性。

## 38.5 Skill policy

当前观测与历史形成$z_t^o$，候选上下文为$c_i=[z_t^o,e(a_i),\operatorname{Pool}(Z^K)]$。合同候选特征$u_i^K=F_K(c_i,D_i^K)$产生基础logit$b_i$。先验特征采用相同上下文的零输入锚定：

$$
u_i^P=F_P(c_i,u_i^K,D_i^P)-F_P(c_i,u_i^K,0),
\qquad \ell_i=b_i+B\tanh(w_P^\top u_i^P).
$$

该结构使$D_i^P=0$时先验残差为0，并限制其直接幅度。残差可以为正、负或接近零；是否学会相应条件化行为需要实验。动作从独立执行mask上的softmax分布产生，不允许先验改变执行资格。

## 38.6 Outcome-grounded learning

Actor、V和Q是共享部分特征的不同输出头。Q读取指定候选及候选集合背景，仅对实际执行动作$a_t$回归

$$
y_t^Q=\operatorname{sg}\left[r_t+\Gamma_t(1-\mathrm{terminated}_t)V_{old}(X_{t+1})\right].
$$

该目标由实际转移支持但含bootstrap误差；未执行候选不填标签。Q损失更新共享表示，不直接识别某条关系是否错误，也不直接更新Actor最终头。PPO采用时长折扣的GAE、独立V回归和熵正则，总目标为

$$
\mathcal L=-L_{clip}+c_V\mathcal L_V+\lambda_Q\mathcal L_Q-c_H\mathbb E[\mathcal H(\pi)].
$$

源合同、VLM、缓存、Verifier和控制器冻结；可训练的图、观测、候选与输出头联合优化。该监督针对固定先验的决策用途，而非知识修订。

## 38.7 Training and inference

训练按照Algorithm 1交替采样真实技能转移与PPO更新，固定每episode先验并保存历史mask、节点索引和原始输入。优化阶段以当前参数重算图特征与GRU前缀，不用缓存旧差分作为新模型输入，也不重新查询VLM。真实终止不bootstrap，有效外截断使用末状态；名义视图始终不写入环境。

推理时参数冻结，Actor根据当前事实与候选重复上述结构计算；V/Q输出非动作选择必需。执行后的实际证据更新事实和历史，下一次决策重新编码同一关系在新状态下的作用。系统因此具有真实观察—决策—执行闭环，但没有自动修图闭环。

# 39. Experimental Setup v0

我们将在[MUST_BIND：simulator/robot及版本]上的组合技能任务评价CP-DISR。低层技能、控制器、感知与Verifier、任务评价器和安全检查在各高层方法之间共享并固定。任务覆盖共享前置、多对象选择、中间重定位、多目标交互及可恢复状态变化中的已绑定子集，具体对象、时限和初始case清单见[MUST_BIND：task/split manifest]。

主要比较为B0非图Observation/Fact策略、B1仅读当前图策略、B2合同干预以及Full。所有方法获得匹配的原始任务与合同信息，除显式A_Q外保留同来源动作价值辅助监督。消融A_DD在非空prior时以$D^H$替代$D^P$，空prior仍输入0；A_B仅移除tanh。训练使用共同实际交互预算[MUST_BIND：$N_{cap}$、$T_{cap}$]并分别报告实际技能次数与时间。

共享R-GCN默认4层、128维、4 bases、mean aggregation、root transform、逐节点LayerNorm、dropout0。PPO默认学习率$3\times10^{-4}$、clip0.2、GAE0.95、每1024技能转移执行4个epochs，sequence16、minibatch64；$c_V=0.5$、$\lambda_Q=0.1$、$c_H=0.01$、$B=0.5$。实际硬件、软件锁和全部优化覆盖值由运行manifest记录，不能把初值写成已实测配置。

VLM只从初始允许输入生成关系，采用固定snapshot并保存原始输出及hash；训练每episode使用80%原始/20%整份缺省，不人为改错边。探索默认3个训练seed，正式展示每task/seed起步30个评价episode；checkpoint由开发集连续窗口选择。任何task/seed子集选择在Private Paper View manifest中明确，原始运行完整归档。

主要指标为独立任务成功率与原始学习AUC，辅以真实完成时间、逐seed差异和计算成本。机制分析区分同状态分布响应与闭环结果；鲁棒性使用同一checkpoint的五种明确先验条件；组合泛化只改变已见技能/谓词形成的目标或依赖组合。所有经验结论将在真实测量后按相应样本与任务范围表述。

# 40. Discussion v0

CP-DISR的学习对象是固定语义先验的用途，而不是先验本身的正确性。先验生成与策略学习解耦，使系统可以在稳定、可追溯的关系输入下比较结构表示；真实回报通过Actor、V与辅助Q影响共享特征，但不产生逐关系真值标签。因此，模型弱化某种关系模式，并不等同于已经识别该关系为假；负向候选修正也不等同于建立新的否定事实。

合同与软关系具有不同地位。合同是来源明确的能力接口，其名义成功语义不承诺每次执行成功；实际失败、耗时与恢复机会进入真实价值学习。合同被独立证据确认写错时需要版本修订，而不是由PPO自行放宽。VLM关系则作为额外可错输入，当前架构允许策略结合观测与经验改变其影响。

方法范围也由允许输入与能力决定。缺边本身不必然导致行为不可学习，观测和历史可能提供替代线索；但必要证据不可辨认或必要动作不存在时，表示学习不能保证补救。有界残差约束一次前向中的直接作用，不保证整个训练过程或长期任务回报免受坏先验影响。最终讨论应依据实际消融、结构依赖与闭环结果界定增量，而不由网络可反传直接推导有效性。

未来可以单独研究候选—关系层面的预测误差经验，把关系引起的判断改变与实际结果对照，检验这些偏差模式能否迁移。这是比当前利用问题更精细的监督对象，不是当前方法必须增加的修复环节。当前论文的独立闭环仍是固定prior、候选结构计算、实际执行和决策学习。

# 41. Final Frozen Research Snapshot

| 项目 | 当前冻结定义 |
|---|---|
| Problem | 固定合同与不完美语义关系下的长期技能选择。 |
| Core Idea | 候选名义干预、合同/先验双差分、真实结果支持的决策学习。 |
| Skill Contract | 来源明确的名义成功与执行接口；不承诺100%成功。 |
| VLM | 初始化生成少量关系并缓存，模型与关系内容冻结。 |
| $G^K$ | Action–Proposition合同图；事实T/F/U，goal用固定索引。 |
| $G^H$ | 与合同图同节点同事实，只增加合法soft edges。 |
| Nominal Intervention | 只读临时ADD/DEL/UNKNOWN，不访问物理未来或写真实状态。 |
| $D^K$ | 合同图名义后继减当前编码，不是任务价值。 |
| $D^P$ | 增强前后差减合同前后差，不是真实因果贡献。 |
| Actor | 合同基础分＋零锚定有界先验残差，独立mask内选择。 |
| Structural Q | 实际动作的bootstrap辅助监督，不辨认或修正具体关系。 |
| PPO | 技能级时长折扣，独立V与GAE，只有真实终端成功事件奖励。 |
| Training | 高层一阶段联合训练；episode级Original80%/No-prior20%。 |
| What is learned | 观测/历史、图/候选表示，以及Actor、V、Q的参数。 |
| What is frozen | VLM、cache内容、合同、名义规则、Verifier、低层与评价器。 |
| What it does | 提供结合情境与经验使用固定先验的能力接口。 |
| What it does NOT do | 不在线修图、不学逐边真值、不校准合同成功率、不微调VLM。 |
| Main paper claim | Candidate-conditioned utilization of fixed imperfect priors。 |
| Main experiments | 七配置、Actor干预、五种prior评价及一种组合holdout。 |
| Future M1+ | 单独研究候选×关系预测误差经验，不进入当前Method。 |

## 研究口径速读

**10秒版本。**合同告诉机器人“做成这一步会改变什么”，VLM补充可能的联系；CP-DISR先比较候选后果，再用真实成败学习这些联系什么时候值得用。

**30秒版本。**系统先根据真实事实和固定技能合同建图，VLM只在初始场景补少量软关系。对每个能执行的候选，它临时假设技能按合同成功，用同一个图网络比较有、无先验时的前后变化。策略用合同判断加有限先验修正选一个技能，之后用真实执行回报训练表示与决策。它学习利用先验，不修正VLM。

**2分钟版本。**固定技能库解决“机器人能做什么”，合同进一步定义这些技能的启动条件和名义成功效果，但不决定任务下一步必须做什么。VLM根据初始图片与任务补充少量可能相关的联系，输出可能遗漏或错误，所以它不能决定事实、mask或奖励。每次真实决策时，系统重新读取观测与事实，并为每个允许技能建立两个临时后继图：一个只含合同，一个还含软关系。四个视图经过同一个R-GCN；合同前后差说明该候选在合同中的结构响应，增强前后差再减合同前后差，说明软关系怎样改变这种响应。Actor先给合同基础分，再用有界、零锚定的先验残差修正，因此先验模式可以被正向利用、弱化或负向使用，但不能保证网络已经识别它真假。机器人随后只真实执行一个技能，事实由新的传感器证据更新，整项任务的独立成功事件提供奖励。PPO、V与实际动作Q监督共同更新可训练网络；合同、VLM与关系内容不改。当前研究验证这种候选条件化表示与学习是否比仅读图更有效，未来逐关系误差经验是另一条独立研究路线。

# Paper Readiness

以下百分比是**定义与本稿可支撑的写作覆盖度估计**，不是实证完成率、质量评级或投稿录用概率。本稿已经实际提供Abstract、Introduction、Problem、Method、Setup与Discussion初稿，性能结果不预填。

| 章节 | 写作就绪度 | 已有内容与仍需补入 |
|---|---|---|
| Introduction | 约90–100%初稿范围 | 完整问题链、方法概述与贡献；最后按结果收束措辞 |
| Related Work | 约75–85% | 六条对话轴与一手依据；仍需针对最终投稿精炼和补充最直接比较 |
| Problem Formulation | 约95–100% | 技能时间抽象、观测/事实/候选、reward与目标统一；实际任务实例待绑定 |
| Method | 约90–100% | 全部主公式、算法、信息/梯度权限已写；具体adapter与层表需按代码确认 |
| Experimental Setup | 约60–75% | 任务模板、七配置、指标和评价已定义；实际平台、预算、硬件、split/hash待绑定 |
| Results | 仅模板，经验结论未填写 | 需要真实训练/评价日志、checkpoint、分母、图表与选择记录 |
| Discussion | 约80–90% | 利用而非修复、来源与执行边界已写；实际失败/收益范围待实验 |
| Abstract | 方法性内容约90%；结果句待填 | 已有完整草稿；任务范围与结果必须由实际证据替换占位 |

**当前CP-DISR已经达到可以正式进入论文撰写阶段的程度。**软件接口、传感器与安全验收仍是运行条件，不是要求先证明算法提升才允许写作的研究gate。

**Current Method is frozen for writing; subsequent experiments validate claims rather than decide whether the paper can be written.**

# 附录A. 来源、版本与兼容性记录

## A.1 项目来源

| ID | 材料 | 在本文件中的用途 |
|---|---|---|
| S1 | Pasted markdown(10).md | 当前论文导向完整规范、v3.0文档结构、方法v2.1不新增模块 |
| S2 | CP_DISR_M1_Research_Specification_v2.0(1).md | 合同、三态、共享编码、Actor/V/Q、PPO、运行接口的技术底稿 |
| S3 | CP_DISR_v2.1_Final_Experimental_Agent_Prompt.md | 最新用户实验组织、v2词表、80/20、七配置与Private Paper原则 |
| S4 | CP_DISR_feedback_chain_review_prompt.md | 反馈链、逐边归因与源头更新问题的范围 |
| S5 | Pasted markdown(9).md | Current Paper与Future Upgrade分离、关系决策用途的讨论要求 |
| S6 | Pasted text(7).txt | 纯DP、删除SOFT_ORDER及训练先验协议决议的原始问题 |
| D1 | 当前对话已确认的v2.1决议 | 两类后果关联schema、必填effect_ref、纯DP、80/20、明确测试编辑 |
| D2 | 当前对话反馈链审查结论 | 固定源头是合理边界；Q监督用途而非关系纠错 |
| D3 | 当前对话M1/M1+结论 | 当前可以负向修正；M1+只作未来预测误差经验方向 |

本轮实际可读取的上传原件与SHA-256保留在sources/source_manifest.json。较早回复链接中的生成版v2.1规范和执行包在本轮挂载文件中未提供，因此不声称已重新读取或核验其二进制内容；本稿依据上传原件与明确对话决议重建统一说明，不为缺失文件编造hash。

## A.2 显式兼容选择，不静默改变方法

| 问题 | 本稿处理 |
|---|---|
| 文档v3.0与方法v2.1 | 分别标识，不把论述升级当算法变更 |
| 节点观测/可执行性输入位置 | 保留冻结v2.1的图结构特征与共同观测/候选通路；第13.1节明示差别 |
| VLM词表与旧schema | 只使用m1_soft_relations_v2、两类关系与effect_fact_ref；不复用旧schema名称 |
| Actor / V / Q串行误解 | 统一写成共享部分特征的并行head |
| Q纠错／DP因果值措辞 | 改为实际结果支持的表示与先验用途学习 |
| 80/20比例地位 | 当前训练协议固定；数值属于实验协议而非差分数学定理 |
| 旧回复训练预算/软件版本不一致 | 本文不选择不明来源数值；总预算/间隔/软件锁MUST_BIND |
| v2.0投稿式实验要求与S3 | 继承S3明确的有限探索与Private Paper View；选择证据不冒称未经筛选 |
| 论文式详细头部实现 | 第22节普通attention/MLP作为明确实施默认，不声称已训练或创新 |
| Future M1+ | 仅第33节及Discussion提及，不进入主配置与必跑实验 |

所有MUST_BIND必须由实际资产、代码、设备或运行配置填入；本文不能替代账号授权、机器人安全校准或完整生产测试。

# 附录B. 可复制配置与Agent交接约束

核心配置由随附interfaces/paper_method_manifest.yaml维护；其method_version为2.1、document_version为3.0，future_m1_plus_enabled=false。其他文件含关系schema、固定提示词、runtime绑定模板与结果模板。它们是实现依据，不是完整训练器。

Agent读取本文件后，首先匹配实际代码与schema/contract/feature profile，补齐runtime绑定，再实现或检查第22.5节不变量。只执行用户明确授权的Stage；本轮没有授权或运行新的训练、机器人动作或VLM缓存生产。若用户要求继续写作，直接使用第35–40节并保留未测占位，不等待全部实验。

接口校验需拒绝当前运行中仍出现旧SOFT_ORDER、训练relation_rewrite、static prior或Future M1+损失。实际图/GRU张量应在优化时重算；配置即使可解析，也不证明真实控制器、ID和图registry已经接通。

# 附录C. 参考文献与核对范围

下列原始论文和官方文档于2026-09-21核对，用于成熟底座和相关工作定位，不证明CP-DISR的效果或首创性。没有将图表或PDF中的未读内容作为依据；API文档只说明公开接口，不说明当前账号或调用结果。

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
