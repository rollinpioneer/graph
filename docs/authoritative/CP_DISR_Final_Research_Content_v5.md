---
title: "CP-DISR 最终研究内容与论文技术路线"
subtitle: "Final Research Content v5 · FROZEN RESEARCH SPECIFICATION"
date: "2026-09-28"
lang: zh-CN
master_id: CP-DISR-FINAL-EXEC-3.0
document_version: "5.0"
default_method_version: "2.1.1"
conditional_method_versions: ["2.2-B", "2.2-C"]
status: AUTHORITATIVE_RESEARCH_SPECIFICATION
freeze_status: FROZEN_AFTER_FINAL_AUDIT
execution_status: PLAN_ONLY
authorized: false
paired_document: CP_DISR_Final_Experimental_Plan_v3.md
---

# 文档控制与冻结声明

本文件与《CP-DISR 最终实验执行计划》v3共同构成后续研究、执行和写作的唯一权威依据。本文件定义研究问题、方法、结构性质及贡献边界；配套实验计划定义必要比较、预算、评价与停止规则。Markdown为权威源。Research Content v4、Experimental Plan v2、此前Master和两轮外部审查保留为输入历史，不再并行授权工作。[S1–S7]

```text
AUTHORITATIVE_RESEARCH_SPECIFICATION
FROZEN_AFTER_FINAL_AUDIT
Default Method: 2.1.1
Conditional Method: 2.2-B OR 2.2-C — S4正式触发后方可启用
Master ID: CP-DISR-FINAL-EXEC-3.0
```

**目标不变：足够强的创新＋完整核心证据链＋最少冗余实验。** CCF-A会议／SCI一区为研究强度目标，不是录用保证。先把研究做强，再从真实结果优化私人展示。论文与实验并行，结果未齐时使用RESULT_PENDING，不把未知写成正结果或零结果。

当前默认算法仍为Method 2.1.1；“v5”是研究文档版本，不是新算法。内部C1/C2/C3保留，Introduction按“统一方法—结果驱动的依赖学习／分析—可选诊断评价”组织，不把afterstate、四项相减和辅助Q分别包装成三个新算法。

未来基础新增训练**17个配置**，条件性新增统一使用**4个elastic attempts**，第一研究周期硬上限**21**。所有升级、重跑、调参、补seed都记账；先替换未启动配置，后使用同一elastic pool，不设其他可叠加桶。超过上限时必须停止自动执行并请求人工研究决策，不能删必要control来凑数。具体排程与成本以配套计划为准。

负结果依次经过诊断性、实现正确性、公平性、训练预算、一次有限修订、必要时S4方法升级，最后才调整经验claim。不得因Full未赢而直接降级主线，也不得无限寻找使Full获胜的task或seed。

本次仅阅读和编辑源文档、核对指定近邻文献、计算文档预算及检查两文档一致性；未SSH、未修改服务器、未调用实验provider，未执行RL、仿真、模型前向或生产测试。历史只读源码结论沿用其原ref，不声称本轮重新核验了运行实现。

## 阅读路线

- Introduction / Related Work：第1–5、20、23–25节。
- Problem / Method：第6–18节。
- baseline与公平性：第19节和实验计划第3–4节。
- 性质、读出与依赖校准：第13、16节和实验计划第15–16节。
- 唯一方法决策：第22节和实验计划S4。
- 文档执行治理、账号与provider历史身份：配套计划Runbook附录，不写入论文主体。

## 冻结的技术边界

1. C2以Full/B2、Full/A_STAT、Full/A_CAT三种不同问题闭环；B1-H只保留可选静态无干预范式，不再承担主C2归因。
2. B1-K+E在T_B seed0为MUST；seed1仅按配套计划冻结规则申请elastic名额。源码审计不能替代“效果描述vs施加”的实验。
3. R-null、bound、权限隔离是各prior对照共享性质；Full相对于这些对照的结构区别是empty-patch null和条件性additive cancellation。
4. 节点级message passing局部性不等于最终策略局部性。指定生产global nonlinear readout不支持无条件端到端严格P4；默认扰动名为GRAPH_REMOTE_PRIOR_INJECTION。
5. 全候选Q前向不等于全候选真实监督。跨接口依赖校准使用同一独立Q_ref，MODEL_Q_PROXY不自证正确。
6. 所有公式定义、17＋4≤21、seed、主终点和升级规则在两文档中同步；本次修改后冻结，普通新结果仅更新实验状态和论文证据。

# 1. Research Motivation：为什么这是一个独立问题

固定技能库使机器人已经能执行OPEN、PICK、PLACE、PLACE_BUFFER等底层技能，但“能调用”不等于“现在调用哪个长期更好”。前置条件确定当下的允许集合，ADD/DEL/UNKNOWN确定名义成功后的符号变化；它们不必包含所有场景相关的代价、风险与返工。技能名义成功也不保证实际控制和后置验证成功。[S3]

不完美的semantic prior能从初始场景给出跨技能或技能—目标联系。关系正确与关系当下值得使用是两回事：一项准备效应可能确实支持后续动作，但准备成本过大、目标已完成、实际参数不利，都会改变其决策价值。相反，语义错误的关系有时也可能偶然把动作推向一个好选择，不能只按关系真假评价策略。

四个信息层必须分开：

| 层 | 内容 | 权限 |
|---|---|---|
| 真实观测／事实 | 当下可观测证据、T/F/U、历史执行 | 更新真实Fact Store，供安全与决策 |
| 技能合同 | 注册能力、前置、名义效果 | 定义逻辑启动和名义解释，不伪造执行 |
| 语义prior | 固定但可能不完整／错误的软联系 | 改变表示；不能写真值、放开mask或奖励 |
| 执行学习 | 实际状态、耗时、终端reward | 更新共享神经参数，不改VLM关系文本 |

问题不是“能否让大模型规划”，而是：**在已有可靠能力和合同之上，怎样让一个不被信任为事实的知识源，以可检查的方式影响候选选择，并从真实结果学习它何时有用。**

## 1.1 Imperfection—Design—Evaluation

| Imperfection | Design response | Evaluation |
|---|---|---|
| missing prior | R-null；episode级80/20 Original/Absent训练 | NoPrior，合法空图机会分层 |
| wrong target | 有界直接残差与不改facts/mask/reward的权限限制 | Target-swap；harmful mass；独立真实后果 |
| graph-remote／unrelated relation | 后果条件化交互；不声称global readout下严格屏蔽 | GRAPH_REMOTE_PRIOR_INJECTION、distance sensitivity、读出分解 |
| provider shift | 固定关系schema与同源输入接口 | 冻结模型Cross-VLM：Full/A_STAT |
| state-dependent usefulness | 仅真实执行结果训练，不给relation-level监督 | 共享Q_ref下的G_ref、Action Flip Quality及跨checkpoint变化 |

这些是设计与检验的对应关系，不是“方法已保证免疫错误先验”。本方法不检测或纠正关系真假，不从VLM自报confidence推导安全性。

# 2. Core Research Question

一句话：**固定且不完美的语义关系，应怎样通过候选名义后果影响高层策略，才能在有限真实交互下实现有效、可分析的技能选择？**

正式问题：给定固定技能库、带版本的合同、真实观测历史、有限目标与episode级固定关系R，在相同任务、信息来源和交互预算下，比较当前状态读出、候选后果差分、静态prior读出、纯交互以及四视图融合；检验它们的学习、动作依赖、错误prior敏感度与执行成本。

研究问题编号与实验计划一致：

| RQ | 问题 | 核心证据／比较 | 对应贡献 |
|---|---|---|---|
| RQ1 | 直接读取候选效果描述与把效果施加到当前图后重新编码，有何不同？ | T_B B1-K/B1-K+E/B2、已有P1、B_PLAN | C1 |
| RQ2 | 合法非冗余prior有无整体增量；该增量是否需要后果条件化而非静态prior上下文？ | Full/B2；Full/A_STAT，同一T_P、匹配seed | C2 |
| RQ3 | 纯交互与灵活四视图融合有何表达力、效用与敏感度取舍？ | Full/A_CAT；A_STAT用于辨别静态来源；读出分解 | C2 |
| RQ4 | 无关系标签的结果训练，是否产生更符合独立动作效用的prior依赖？Q的作用是什么？ | Full/A_STAT/A_CAT/A_Q共享Q_ref，E2/E4/E8轨迹与裁剪审计 | C3 |
| RQ5 | 缺失、错误、图结构远端与provider变化下，先验影响和代价如何？ | NoPrior、Target-swap、GRAPH_REMOTE_PRIOR_INJECTION、条件Cross-VLM | C2/C3 |
| RQ6 | 相对合同搜索、朴素使用prior和逐步LLM推理，成功、成本与摊销效率如何？ | B_PLAN、B_PLAN+R、分析R*、LLM-Planner | 整体 |
| RQ7 | 结论如何随真实任务族难度及有合法来源的未见条件改变？ | Full/B2难度轴；条件holdout | 范围 |

# 3. Scope与非目标

## 3.1 当前方法做什么

学习固定技能集合中的高层选择。每次只执行一个真实技能；根据真实后置证据与独立Evaluator取得转移。名义干预只用于生成特征，prior每episode固定。研究条件允许部分可观测，GRU是有限历史摘要，不宣称完全马尔可夫。

## 3.2 当前方法不做什么

不学习低层轨迹控制；不训练视觉基础模型；不在线改合同；不预测未来图像；不建立可被信任的完整世界模型；不由VLM输出主方法的下一动作；不做关系真假监督、逐边错误修复或VLM微调；不将R\*或QA真值输入训练；不把图差分作reward；不增加通用trust gate、ensemble或未经证据触发的模块。

LLM-Planner/VLM-Planner属于外部比较，不改变主方法的固定prior设定。关系级删边分析不是relation-error learner。名义副本也不是用于采样真实回报的环境克隆。

# 4. Related Problem Positioning

下面比较研究机制，不比较异构任务上的原论文数字。既有文献沿用v4的核验记录；本轮仅对最终审计点名的近邻补查原始书目／摘要，不进行开放式文献搜索，不据此宣称穷尽或首创。[W1–W27]

| 近邻 | 已有思想 | CP-DISR的区别与不能冒称的内容 |
|---|---|---|
| afterstate / post-decision state | 决策后的确定性部分可用于评价 [W19] | 名义后继属该思想；创新不在“看下一步”，而在后果差分作为prior查询锚 |
| successor representation | 预测策略条件下未来访问或特征占用 [W17] | DK是一次名义效果的表示差，不是多步占用预测 |
| Value Prediction Network | option条件的抽象未来价值预测 | 当前不学latent dynamics；合同只给名义成功副本，真实失败通过RL处理 [W1] |
| GNN generalized planning | KR 2022后继价值／GNN策略；KR 2023使用policy gradient、状态转移分类及关系GNN | 是C1的危险近邻；C1定位为prior的候选后果锚，而非“RL＋GNN＋后继”的首创 [W2,W20] |
| ASNets | 动作schema与命题连接组织策略 | 同样利用结构；本研究新增固定prior的候选后果交互约束 [W3] |
| symbolic / neuro-symbolic planning | 用显式规则搜索与连续采样验证 | B_PLAN必须实测；名义图不是已验证世界，不宣称搜索一定失败 [W4] |
| SayCan | 语言任务适合度与技能affordance结合 | 主方法R是关系而非动作分布；不逐步调用模型选技能 [W5] |
| SayPlan / scene-graph LLM planning | 场景图压缩搜索空间与重规划 | A_STAT是在匹配合同干预路径下的静态prior对照；可选B1-H代表无名义干预的静态图范式；外部LLM另比较逐步推理 [W6] |
| LLM action priors | 固定LLM动作分布与RL融合 | 当前R不直接给π先验分数；四视图检验其对名义变化的编码影响 [W7] |
| graph world model / GAVEL | 图模型检查、预测并修复LLM计划 | 本方法不把VLM图当验证器或世界模型；学习的是实际技能选择 [W8] |
| Build on Priors | VLM构建抽象、规划域并进行模仿学习 | 当前能力与合同先给定，不从示范合成规划域，不做该式IL [W9] |
| 辅助表示学习 / UNREAL | 共享表示由辅助目标学习 | Q辅助不是新的学习范式；贡献需落在无关系标签的prior用途校准 [W10] |
| Q-shaping / LLM heuristic Q-learning | 用LLM给出的状态—动作启发式影响Q初始化或学习；相关工作在自身假设下提出无偏／最优性分析 | CP-DISR使用固定关系R，不提供LLM action value，不改变reward或Q目标；单步bound不能冒充这些工作的最优性保证，也不暗示其启发式一定需要逐步在线调用 [W21,W22] |
| LLM4Teach | LLM教师动作指导、学生RL与随训练调整的教师依赖 | 不接收teacher action target；检验状态与候选条件下固定关系的utility-aligned prior reliance，不把“能摆脱错误教师”占为首创 [W23] |
| Flax / LLM-Flax | 神经符号任务简化与规则；LLM-Flax生成relaxation/complementary规则、恢复与object importance | 知识服务规划抽象／启发式，而本方法R不构建可信symbolic dynamics；用B_PLAN+R作范式参照，不新增复杂复现 [W24,W25] |
| Plan-Seq-Learn / BOSS | LLM指导低层控制学习或扩展技能库 | CP-DISR固定技能库与低层controller，学习高层候选选择；层级与训练对象不同 [W26,W27] |

Q-shaping类论文的收敛／无偏结论属于其特定学习规则和前提，本研究不借用该保证。要比较的是关系接口、信息类型与依赖行为，而不是把概率界与最优性界直接比较为“更强”。此轮仅补写研究定位，不新增这些工作的训练baseline。[W21,W22]

GAVEL、Build on Priors在此按2026预印本处理，不冒称已在指定会议录用。LLM-Planner是本项目的信息边界比较，不冒称完整复现SayCan/SayPlan。已有方法可能更强，结果必须保留；可比较成功—推理成本折中，但不能预言LLM必受幻觉伤害而Full必安全。

# 5. Final Contributions：统一方法、学习／分析、诊断评价

内部C1/C2/C3用于追踪证据；论文Introduction不机械列成三个独立算法。

## 5.1 Method contribution — Consequence-conditioned prior interaction for skill reasoning

**C1 Candidate Consequence Anchoring**是构造步骤。afterstate和后继编码已有明确先例；这里用只读、并不被信任为实际结果的名义变化DK，为不完美关系提供候选查询锚。T_B的B1-K+E seed0是MUST，专门检验直接读取PRE/ADD/DEL/UNKNOWN/guards与真正施加到当前事实后重编码的区别。没有该结果前，不宣称已排除效果可访问性的解释。

**C2 Consequence-Conditioned Prior Interaction**是方法的核心设计空间。三个比较各有唯一主要问题：

| 比较 | 主要研究问题 | 不混同的其他问题 |
|---|---|---|
| Full vs B2 | 增加prior接口及其训练协议有无整体增量？ | 不能单独说明后果条件化比静态上下文好 |
| Full vs A_STAT | 保留同一DK、合同分支和参数形状，只把DP换成S_R，后果条件化是否有价值？ | 不将静态张量误说成候选logit完全相同 |
| Full vs A_CAT | 纯交互限制相对灵活四视图融合的效用与敏感度取舍是什么？ | 不是严格单变量或严格等参数实验 |

R-null、bound和权限隔离共同保护所有匹配prior方法，不是Full独占的优势。Full的可检验区别是empty-patch null和条件性additive main-effect cancellation；默认实现没有严格端到端locality保证。

## 5.2 Learning / analysis contribution — Utility-aligned prior reliance

**C3**研究仅凭真实执行结果、没有关系级真假监督时，不同prior接口是否学出更符合独立动作效用的概率重分配。比较Full、A_STAT、A_CAT、A_Q共享同一批Q_ref，以及同一run的E2/E4/E8 checkpoint。是否随执行训练改善，是待检验结果，不是定义保证。

Structural Q是grounding通道而非单独的新RL范式。A_Q检验其训练作用，同时记录全局gradient clipping差异；不能把任何A_Q差异都归因为表示质量。

## 5.3 Evaluation contribution — Diagnostic evaluation of imperfect priors

E1–E6、D0成绩天花板观察、合同／LLM外部范式、机会密度、robustness与独立utility参考形成诊断评价框架。论文是否将其列为Introduction第三项贡献，可在写作中决定；不因此增加训练或创造新的benchmark claim。

统一逻辑是：C1定义候选后果锚，C2限制prior入口，C3检验该接口在结果训练下学会了怎样依赖prior。实验未完成的经验句统一保留RESULT_PENDING。

# 6. Formal Problem Definition

采用时间扩展技能的semi-Markov描述；它沿用options/SMDP思想，不将时间抽象本身作为创新。[W18] 技能级决策时刻记τ_t，动作a_t为grounded技能，真实耗时d_t=τ_{t+1}−τ_t>0。隐藏物理状态s_t仅用于环境和独立QA；决策snapshot X_t由观测、事实、任务、合同、候选及版本引用构成。

| 符号 | 定义 |
|---|---|
| U,K,P,O | 技能库、合同库、谓词注册表、实体目录 |
| F_t:P→{T,F,U} | 当前真实三态事实 |
| T=(G,deadline,Evaluator) | 签名目标与独立任务协议 |
| A_t,m_t | 稳定ID候选及真实执行mask |
| R_e^0,R_e | episode原始admitted关系及Original/Absent处理后的关系 |
| P_i | 名义效果后实际发生变化的事实patch，含派生闭包 |
| Z00,Z10,Z01,Z11 | 当前／名义后继 × 无／有prior编码 |
| DK,DH,DP | 合同变化、增强变化、交互差分 |
| z^o,c_i,u_i^K,u_i^P | 实际历史摘要、上下文与两类候选特征 |
| b_i,Δ_i,ℓ_i,B | 合同logit、prior残差、总logit与幅度上界 |
| V,Q,R\* | 状态基线、辅助候选价值、仅分析的真值参考关系 |
| H,Γ_t,w_t | 折扣半衰期、技能时长折扣、起点核算权重 |

\[
\Gamma_t=2^{-d_t/H},\qquad
r_t=\sum_k2^{-\epsilon_{t,k}/H}\rho^{env}_{t,k},\qquad
w_t=\prod_{j<t}\Gamma_j.
\]

目标报告量为：

\[
G_e=\sum_t w_t r_t,\qquad J=\mathbb E[G_e].
\]

**T_P primary endpoint固定为mean start-discounted return J**，在S2 task-family manifest中、S3训练之前登记，并在冻结独立test上作为主终点。success、完整任务cost/rework/skill count/physical completion time是提前声明的次终点；只保留当前环境确实可测的字段，未测为NA。learning AUC是预声明的学习过程证据，不在success持平后才改称主终点。

ρ仅在独立首次确认完整任务成功时为1，其他为0。没有子目标、搬运、图变化或VLM奖励。J可区分同为成功但更快完成的情况；失败时纯终端reward下J仍为0，不能声称它完整覆盖失败成本。Episode任务deadline与训练总N/T预算不是同一个量。

H按同family匹配实验固定；T_B延续可信历史H，T_P若需要新H，则以新family profile登记。不同H的raw discounted return不直接跨task取总体均值。跨task汇总优先success、归一化cost和逐task效果，不回改旧reward。

# 7. Skill Contract

每个grounded技能a有schema、有序参数及类型、PRE_POS、PRE_NEG、ADD、DEL、UNKNOWN、条件效果、frame规则、timeout、controller与Verifier来源。

正前置要求F(p)=T；负前置要求F(p)=F；U不能作为允许关键安全动作的证据。一般skill失败是合法真实转移，是否终止episode由独立任务协议决定。安全拒绝或无正时长技术异常不能补成虚假0.05秒训练转移。

名义效果同步求值：所有guard用同一原始F；合并赋值，冲突拒绝；ADD设T，DEL设F，UNKNOWN设U；未改变项frame保留；只按已注册派生规则更新。不能为让prior“有用”删除真正的硬前置，或把不安全行为变成探索候选。

效果描述、效果施加和真实后果分别记为K(a)、T_K^nom(F,a)、真实F_{t+1}。它们在文中不得互换。

# 8. Semantic Prior

## 8.1 两类关系及语义

| 类型 | 合法端点 | 含义 |
|---|---|---|
| SOFT_SUPPORTS | Action A→Action B，A≠B | A的注册名义效果可能支持B后续执行；不是充分可行、硬前置或必须先做 |
| SOFT_RELEVANT_TO_GOAL | Action A→签名目标命题g | A的注册名义效果可能影响g的实现、保持或恢复；不预定影响一定为正 |

每条关系有effect_fact_ref，必须指向source已经注册的ADD/DEL效果。它是admission provenance，不是新增神经truth标签，也不是只让该effect传播的精确gate。一个source有多个效果时，其他变化也可能经过该action节点影响编码。[S3; S4]

## 8.2 生成与固定

VLM读取冻结初始RGB、对象绑定、goal、允许ID及同源合同描述，输出关系JSON；不输出下一动作、完整计划或新对象。模型、prompt、few-shot、schema与解码参数随cache版本冻结。训练每episode一次Bernoulli(0.8)决定保留整份R或置空；episode内与PPO重算中固定。

具体provider/endpoint/SDK的历史身份及目标机绑定放在实验计划Runbook附录；研究定义只要求固定且可追溯的请求接口。服务更换必须产生新的输入身份，不能沿用旧cache标签。

每scene首次请求一次，最多一次传输/格式重试，禁止语义重试。合法空prior不是API失败；缺失cache也不允许伪装合法空。保留raw/accepted/rejected、冗余理由、request ID、实际payload与hash。

## 8.3 错误与标注

规则校验只确认schema、ID、效果出处和局部冗余，不保证语义正确。R\*只在封闭且可独立裁定的候选关系宇宙中定义；不能从soft “may support”直接臆造一套全知真值。关系正确性、当下机会、对固定continuation的效用分别标注，未知保持UNKNOWN。

# 9. Graph Construction

G^K=(V,E_K,F,T)包含ACTION和PROPOSITION节点，PRE_POS/PRE_NEG为命题→动作，ADD/DEL为动作→命题，使用反向消息边。UNKNOWN保留为名义赋值，不凭空增加现有未定义edge type。

G^H与G^K使用相同节点、事实与目标，只增加R及逆向软消息边。四视图固定节点/goal行、相同事实特征规则、同一encoder。实体标识用于一致绑定，不应靠不透明ID泄漏目标或任务答案。

当前指定源码GraphTemplate有全体合同的效果边；因此B1-K并非完全未接收ADD/DEL。SnapshotBuilder的候选8维向量只含稳定schema/参数标识、参数数和timeout，没有显式候选效果列表。NodeFeatures的实例绑定表达能力、对称节点碰撞、goal绑定和当前图pooling可访问性须在S0分别审计。[G1–G3]

若原实现与声明的同源观测/对象表不符，标记输入profile缺陷；对将要比较的方法统一修正并留旧结果，不只强化Full或削弱baseline。

# 10. Candidate Nominal Intervention

对每个m_i=true的候选：

\[
\widetilde F_i=T_K^{nom}(F,a_i),\quad
\mathcal P_i=\{p:\widetilde F_i(p)\ne F(p)\}.
\]

构造immutable overlay，不修改真实FactStore、时钟、执行次数、reward、done、安全许可或输入图像。两种图使用完全相同的patch，增强图R不随名义结果改写。

名义mask不是新的真实许可表。图上保留尚不可执行技能用于结构传播，但不为这些动作执行真实环境操作。条件效果、derived closure及互斥检查都必须来源于固定合同，而非模型编造。

每个决策所需encoder调用理论上可复用当前K/H；候选名义分支按实际候选数计数。报告实现调用次数、额外内存和前向时延，不宣称共享概念就已实现最优缓存。

# 11. Contract Difference DK

\[
Z_{00}=E_\psi(G^K),\qquad
Z_{10}^{i}=E_\psi(\widetilde G^{K,i}),\qquad
D_i^K=Z_{10}^{i}-Z_{00}.
\]

DK是目标对齐潜在表示的变化，不是reward、成功概率、最短路径或真实因果效应。它把候选的注册效果与当前T/F/U及合同拓扑共同考虑，而显式效果表B1-K+E只提供描述，不施加到当前图。

B1-K与B2比较保留原匹配定义：c_i相同，前者phi_K(c_i,0)，后者phi_K(c_i,DK)。现有结果支持这一匹配实现的增量；若要声称超越“给模型一个可直接读的效果表”，必须完成基础MUST的T_B B1-K+E seed0；seed1按已冻结规则从elastic pool申请。

# 12. Prior Interaction DP：匹配2×2，而非因果DiD估计

\[
Z_{01}=E_\psi(G^H),\quad
Z_{11}^{i}=E_\psi(\widetilde G^{H,i}),
\]
\[
D_i^H=Z_{11}^{i}-Z_{01},\qquad
\boxed{D_i^P=Z_{11}^{i}-Z_{01}-Z_{10}^{i}+Z_{00}}.
\]

第一轴是相同事实的当前/名义后继，第二轴是同源K上的无/有R。共享权重、节点和goal行使相减有对齐语义。R不修改patch。

定义S_R(F)=E(F,R)−E(F,∅)，则DP=S_R(tilde F_i)−S_R(F)。其含义是prior怎样调制当前候选名义变化的编码，或候选变化怎样调制prior的编码作用。它不是从观察数据估计causal difference-in-differences，不需要平行趋势假设，也不提供物理因果识别。

静态图边并不必然被消去。例如E(F,R)=M_R x(F)+b_R时，DP=(M_R−M_∅)(x(tilde F_i)−x(F))。被抵消的是可加分离的表示主效应，不是所有固定关系，也不是所有有用的静态知识。

表达力的有意代价：若一个场景偏好有用却不改变任何候选名义变化的响应，Full可能无法直接使用它。A_STAT与A_CAT都不是为了必然失败：A_STAT保留同一合同干预，只将先验输入换成\(S_R(F)=Z_{01}-Z_{00}\)；A_CAT允许灵活四视图融合。B1-H是可选的无名义干预静态图范式，不承担这项单变量C2归因。

A_STAT的静态张量S_R在当前state共享，但\(c_i,u_i^K\)仍因候选而不同；它仍能输出候选不同的prior残差。因此“static”只指prior输入没有按该候选的nominal patch再作差，不等于无候选query或共同logit平移。

# 13. Structural Properties：已知性质、条件命题与不成立的推广

所有性质固定参数、真实snapshot、候选表/mask、输入索引和相同历史hidden。只讨论明示输入干预，不比较两个独立训练模型。

## P1 Exact null

若R=∅，K/H视图复用，DP=0。零锚定phi_P保证uP=Δ=0，所以π等于同权重的合同参考策略π_K^θ，**不等于独立训练B2**。空patch也使该候选DP=0，但其他候选残差仍可能经softmax改变它的概率。

## P2 Additive cancellation

若所比较输入上E(F,R)=A(F)+B(R)，代入四式得DP=0。它是条件性代数恒等式，不证明“有用的prior都是不可加”或“静态prior都无用”。该前提必须对整个被相减的E成立，包括goal逐点MLP与global nonlinear readout；当前网络不被假定实际可加分离。因此不得将P2写成“生产Full总能滤掉全部静态prior”。

## P3 Bounded direct effect

\[
\Delta_i=B\tanh(w_P^\top u_i^P),\quad |\Delta_i|\le B.
\]

对同权重同mask的π_K^θ：

\[
\frac{\pi_\theta(i)}{\pi_K^\theta(i)}
=\frac{e^{\Delta_i}}{\sum_j\pi_K^\theta(j)e^{\Delta_j}}
\in[e^{-2B},e^{2B}].
\]

两个不同prior的直接粗界为e^{±4B}，不是e^{±2B}。这些是单步概率比／logit约束，不是轨迹安全、回报下界、错误动作概率为零或argmax不变保证。多步分布变化可累积，训练也会改变合同分支参数。

## P4a Node-level consequence locality

若L层message passing只使用局部邻居、逐节点变换和逐节点LayerNorm，且patch仅改节点输入，那么第ℓ层因patch改变的节点只可能位于其ℓ跳前向传播锥内。可由层数归纳证明。边方向、逆边、self-loop和每关系度归一化必须纳入实际依赖图。

该性质只针对**节点编码器**，不自动延伸到图读出、Actor和闭环轨迹。

## P4b Conditional mixed-interaction locality

对prior边编辑r定义其计算影响footprint B_r：至少含直接受消息和度归一化变化影响的端点。对patch定义P_i。使用所有待比较图的并图，而非只在K上算距离。

一个保守充分条件是：在进入固定线性读出前，P_i与B_r的扰动锥从未在任何非线性计算节点相遇；等价的简化充分距离条件可取无向并图d(P_i,B_r)>2L，并要求节点集合、pool权重及分母固定。于是逐节点混合有限差分为零，固定线性／逐节点后固定线性汇聚仍保持零。

这不是必要条件，也不接受未经证明的“距离≥2L一定不变”。边添加可能改变最短路、局部归一化或候选集合；任何这些变化都要更新footprint。全policy不变还要求所有合法候选的输出差不变且没有其他R旁路；只证明一个候选的残差不变不能推出它的softmax概率不变。

## P4c 指定生产实现不能直接使用严格端到端命题

在R1源码快照55a3b7c的GoalReadout中，先分别mean所有ACTION/PROPOSITION和goal行，再经非线性MLP得到global_row；differences()对**包含这个global_row的Z**做差。局部状态与远端prior可以在该MLP中相互作用，即使图上无传播路径。[G1]

分析性反例，不是实验：设不连通的两区域分别贡献x和y，全局读出g(x+y)=ReLU(x+y)。取x=0、y=−0.5，候选使x增加1，远端关系使y增加1，则四输出为0、0.5、0.5、1.5，混合差为0.5。不存在图上近邻关系，也可能有非零全局DP。

所以当前只能正式使用P4a，并将P4b写成条件命题。**Method 2.1.1不得称为已经consequence-local的完整策略。** S0需验证后续源码是否相同，S7分开记录node-local delta与global-readout DP。GRAPH_REMOTE_PRIOR_INJECTION或sensitivity曲线仍有研究价值，但不预设Full在远处严格零。若图过小／有hub导致没有合法远端关系，记NOT_APPLICABLE，不添假节点制造漂亮曲线。

## 13.5 Property Comparison Table

下表“✗”表示该性质不是接口保证，不表示所有输入上一定违反；“✓”以本文件明确的共享权重、零锚、相同真实输入和mask条件为前提。

| Property | Full | A_STAT | A_CAT | B1-H-MATCHED-v1 |
|---|---|---|---|---|
| \(R=\varnothing\) exact null | ✓ | ✓ | ✓ | ✓ |
| bounded direct residual | ✓ | ✓ | ✓ | ✓ |
| prior不能修改mask/facts/reward | ✓ | ✓ | ✓ | ✓ |
| **empty-patch prior null** | **✓** | ✗ | ✗ | N/A：无名义patch |
| **additive prior main-effect cancellation** | **✓，仅可加分离前提下** | ✗ | ✗ | ✗ |
| production end-to-end strict locality | ✗ | ✗ | ✗ | ✗ |

Full在\(\widetilde F_i=F\)时保证该候选的\(D_i^P=u_i^P=\Delta_i=0\)。这不保证该候选最终softmax概率完全不变：其他候选的非零残差仍会改变归一化。所谓“不能直接被prior推动”限定为该候选的直接残差。

不能将共享的R-null、bound和权限隔离当作Full击败A_STAT/A_CAT的唯一解释。经验辨识依靠Full/A_STAT、Full/A_CAT、独立效用校准和读出分解，不承诺比较方向。

## 13.6 S0性质检查与重排条件

未来获准S0至少核验Full的Exact R-null、Empty-patch null、Bound和Candidate reorder equivariance。当前文档没有运行这些测试。

重排要求candidate IDs、features、mask、contract lookup和输出inverse permutation同步；policy历史、事实、prior、权重不变，deterministic tie-break按相同canonical ID。合法列表只改顺序时应按ID对齐保持输出，数值差按冻结dtype容差核验；浮点归约次序不许被声称bitwise一致。若代码隐含索引依赖，报告实际能成立的条件并修复索引错误，不先给无条件定理。

候选重排不等于对象重命名／图同构等变；稳定ID标量不能自动保证后两者。A_STAT/A_CAT/B1-H也检查其共享null/bound，但不得为使empty-patch测试通过，给这些对照补Full专属DP gate。

bitwise exact null来自显式分支复用；一般浮点代数使用明确atol/rtol，不用检测阈值偷改真实网络输出。

# 14. Actor / V / Structural Q

## 14.1 实际历史与合同上下文

\[
(z_t^o,h_t)=f_{obs}(x_t,h_{t-1}),\qquad
c_i=[z_t^o,e(a_i),\operatorname{Pool}(Z_{00})],
\]
\[
u_i^K=\phi_K(c_i,D_i^K),\qquad b_i=w_K^\top u_i^K+b_K.
\]

GRU不直接读取prior图embedding；但past action受prior影响后，真实历史可以间接不同。冻结shadow分析必须固定同一输入前缀，不把两个已经分叉的episode当作同状态反事实。

## 14.2 Zero-anchored prior branch

\[
u_i^P=\phi_P(c_i,u_i^K,D_i^P)-\phi_P(c_i,u_i^K,0),
\quad\Delta_i=B\tanh(w_P^\top u_i^P),
\quad\ell_i=b_i+\Delta_i.
\]

两次phi_P共享权重、上下文与goal mask。最后先验线性头无独立bias或额外无界scale。当前Full不另外输入ZH、VLM评分、抽样mode或oracle label。只在真实m_i=true上softmax；无候选时安全终止，不默认选第一个动作。

## 14.3 三个并行输出头

\[
\bar u^K=\operatorname{MaskedMean}_i u_i^K,\quad
\bar u^P=\operatorname{MaskedMean}_i u_i^P,
\]
\[
V(X)=f_V(z^o,\operatorname{Pool}(Z_{00}),\bar u^K,\bar u^P),
\]
\[
Q(X,a_i)=f_Q(u_i^K,u_i^P,z^o,\bar u^K,\bar u^P).
\]

Actor不等于argmax Q；V不强制等于ΣπQ。Q在代码中对全候选前向，但当前loss只gather执行动作。mean特征允许参数共享产生间接梯度，不能说未执行候选获得了同一步真实标签。[G1,G4]

## 14.4 默认架构与实现审计

默认L=4、hidden128、4basis、RGCN mean/root、ReLU＋逐节点LN、dropout0；GRU128；candidate attention128。语义规格的MLP描述与指定源码实际层表可能不完全一致，S0导出真实层表和参数计数，不按文档摘要默默重建一套不同网络。

指定Policy类分配了不被每个method使用的模块。报告total allocated、active trainable及实际梯度参数三种计数，不用未用模块假装容量对齐。A_CAT现有4d→d首投影的理论增量为3d²=49,152，但实际active百分比仍须导出，不能仅凭一个10%阈值自动缩宽CAT。

# 15. Outcome Grounding与标签边界

真实执行提供当前状态、选定动作、实际时长、独立reward和真实下一状态。它们同时支持策略学习和Q/V监督；名义副本不直接提供成功标签。失败、返工和超时若是有效真实转移，也保留在buffer。

当前Q目标：

\[
y_t^Q=\operatorname{sg}\{r_t+\Gamma_t\kappa_tV_{old}(X_{t+1})\},\qquad
\kappa_t=1-\mathrm{terminated}_t,
\]
\[
\mathcal L_Q=\mathbb E_{valid}\operatorname{Huber}
(Q(X_t,a_t)-y_t^Q).
\]

没有未执行动作的当步target，没有max-Q、nominal-success reward或relation真假标签。旧V bootstrap有估计误差，不能称yQ为精确长期真值。仅凭all-candidate forward不能把C3写成“所有候选获得更密真实监督”。

R\*、语义真假和独立行动价值只在分析分区。训练代码不得读取这些sidecar；计算图和数据loader都应可审计。研究者用development分析设计任务不等于逐transition向训练泄漏oracle，但所有数据选择理由要留痕，test不可用于方法升级。

# 16. Utility-Aligned Prior Reliance / Prior-Reliance Calibration

## 16.1 定义、比较对象与非主张

本研究中的calibration指prior诱发的概率重分配与独立决策效用的适当性，不是ECE或关系正确概率校准。Δ不是confidence、correctness probability或物理因果效应。

固定各模型自己的θ、同一真实X、同一观测前缀，在该模型内定义\(\pi^0\)为只把最终prior残差置零的策略，合同logits不变；\(\delta\pi_i=\pi_i-\pi_i^0\)。DP-Zero、Δ-Zero和R-removed是不同诊断，必须分开命名。

**比较接口固定为Full、A_STAT、A_CAT、A_Q。**每个模型使用自身的π与π0，但全部共享同一组物理状态、合法候选和Q_ref。不能只展示“Full的G_ref>0”而没有接口参照。

## 16.2 独立真实效用bank

沿用16个预先冻结development snapshot、每state两个合法候选、每候选两次配对continuation，共最多64条物理分支；不为每个模型或checkpoint重做。bank在S2固定选择与continuation，S3/S4可用其development分析，S7复用；它不是blind test，不能将已用于方法决策的状态称为未见最终测试。

必须能核验分支恢复状态、随机流和continuation。Q_ref是有限共同continuation下的动作价值，不是全局最优Q*。环境真值和关系标签仅在分析分区；不写入训练loader、reward、prior内容或Q训练target。

优先使用正好两个合法候选、从而全候选均有独立标签的状态。若合法候选更多，两个物理分支只覆盖子集\(\mathcal B_s\)：不得对未执行候选补零或填模型Q冒充Q_ref。此时按子集内条件概率计算明确标注的\(G_{ref}^{\mathcal B_s}\)，并报告每种策略在该子集上的原始概率质量；不把它写成完整策略的期望增益。无法覆盖／恢复时使用实际有标签动作的描述性分析或NA。

## 16.3 三个主要量

全合法候选有独立reference时：

\[
G_{ref}(s)=\sum_{i\in\mathcal A(s)}\delta\pi_i Q_{ref}(s,i).
\]

它是该固定continuation下的一步概率搬运价值差，不是完整闭环policy value差。

\[
H_{ref}(s)=\sum_i[\delta\pi_i]_+\,
\mathbf1\{Q_{ref}(s,i)<\max_jQ_{ref}(s,j)-\epsilon_Q\}.
\]

H_ref称Harmful Mass，即新增到比参考最佳候选差至少ε_Q的候选上的质量；它不是总损害的无偏估计，须和G_ref一起读。ε_Q在development按J/return同单位及reference误差冻结，不能混用秒与回报。

Action Flip Quality比较π与π0的deterministic argmax：两动作Q_ref都可识别时，按其差大于ε_Q、在±ε_Q内、小于−ε_Q分为beneficial、neutral、harmful；reference不完整或不确定度覆盖边界时标UNKNOWN。关系true不强制Δ为正；真实错误关系也可能偶然诱发好动作。

## 16.4 跨接口与训练轨迹

在同一bank上对Full/A_STAT/A_CAT/A_Q计算G_ref、H_ref、Action Flip Quality、TV、有效机会和残差。再在**同一run**的E2、E4、E8 post-update checkpoint上重复前向，比较这种依赖是否随真实结果训练改善。基础为11模型身份：Full3＋A_STAT3＋A_CAT3＋A_Q2；没有到达的checkpoint写NOT_REACHED，不能从文件名推测完成update。

同一物理state下各模型的hidden由相同真实观测前缀、各自冻结参数重算，不把一个模型的latent hidden复制给另一个。保存训练中间checkpoint不要求S5重跑低信息E2闭环dev；E2校准只需离线前向。

只有实际轨迹支持时，才写“outcome-grounded reliance随训练改善”。三checkpoint比较本身不构成时间方向因果证明，也不能以中途最好点代替预定终点。若Full和A_STAT/CAT均改善，报告共同现象；若A_Q相当，C3仍可成立，但Q的独立经验收益不能虚构。

## 16.5 Q-proxy与作用分层

\[
\widehat G_Q(s)=\sum_i[\pi(i|s)-\pi^{\Delta=0}(i|s)]
\operatorname{sg}(\widehat Q(s,i)).
\]

它始终标MODEL_Q_PROXY。先核查独立已执行动作Q误差，再与有物理分支的G_ref比较方向和排序；不以模型Q给自身prior背书，不用Full训练过的Q-head优于A_Q未训练Q-head证明表示更好。

每条关系／状态分别记录合法性、语义T/F/U、结构机会、utility helpful/harmful/neutral/unknown。全数据、R+、多合法候选、真实changed patch、graph-remote、训练已见／未见各有分母；初始化但未更新的counter为NOT_MEASURED，不是观测0。

跨接口和跨checkpoint只是共享reference上的额外前向，不新增物理分支、不增加训练。参数fingerprint固定，任何gradient reach诊断不得调用optimizer。

# 17. Training Objective与预算语义

## 17.1 GAE和V目标

\[
\delta_t=r_t+\Gamma_t\kappa_tV_{old}(X_{t+1})-V_{old}(X_t),
\]
\[
\widehat A_t=\delta_t+\Gamma_t\lambda c_t\widehat A_{t+1},\qquad
 y_t^V=\operatorname{sg}(V_{old}(X_t)+\widehat A_t).
\]

c_t仅在同episode、连续且本批可用时为1；deadline是真实任务terminal，不bootstrap；有效外部截断可bootstrap最后有效观测；跨reset不可递推。[W11,W12]

## 17.2 PPO与Q联合目标

\[
\rho_t=\exp(\log\pi_\theta(a_t|X_t)-\log\pi_{old}(a_t|X_t)),
\]
\[
L_{clip}=\mathbb E_{valid}\min\{\rho_t\widehat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\},
\]
\[
\mathcal L=-L_{clip}+c_V\mathcal L_V+\lambda_Q\mathcal L_Q
-c_H\mathbb E_{valid}\mathcal H(\pi_\theta).
\]

Actor/V/Q/entropy按有效transition均值，actor_episode_discount_weight=false；起点权重w只用于return核算，不在loss偷偷补乘。当前经验PPO surrogate不声称是起点折扣目标J的无偏精确梯度。[S3 §17; W11]

默认Adam lr3e-4、clip0.2、GAE0.95、cV0.5、lambdaQ0.1、entropy0.01、gradclip0.5、4epochs、rollout1024、seq16、minibatch目标64；尾batch保留，实际optimizer steps可不同于64。Advantage按有效batch标准化，V/Q target不标准化。一个去重optimizer管理共享参数。

### Gradient Clipping Audit（A_Q解释所必需，不改变训练）

每次实际optimizer step前记录全部有效参数的pre-clip global L2 norm，在现有clip操作后记录post-clip norm，并记录触发次数／实际step总数。默认阈值仍为0.5，不能为消除混杂本轮换优化器或裁剪协议。

若\(g=g_{Actor/V/H}+\lambda_Qg_Q\)，global norm clipping的缩放约为\(c=\min(1,0.5/(\|g\|_2+\varepsilon))\)。令λ_Q=0不仅去掉Q梯度，也可能改变c，从而改变Actor/V的有效梯度大小；后续Adam又有自身预条件与历史状态。因此Full/A_Q的成绩差不能自动只归因为representation learning。

S0核验日志字段确实在clip前后取得，训练与Results按方法／seed报告pre/post分布和clip-trigger rate；两个值不是把同一个返回值抄两遍。诊断不改变grad，不补额外run。

## 17.3 前缀与重算

采样存原始snapshot、prior版本、candidate/mask、old logp/value及真实转移。PPO使用当前参数重新编码图和名义差分；不把旧learned embedding当新输入。不重抽prior，不修改历史mask。GRU序列块前缀以当前参数no-grad重算，块内按既有规则反传；prefix长度可能造成性能成本，进度监测需区分采样、前缀和backward。

## 17.4 中间预算不是新方法

E2/E4/E8/E16分别是2,048/4,096/8,192/16,384真实技能转移目标点；同一run分段，不给新seed，不重置累计N/T。最终envelope和Tcap在开跑前声明，中间pause_at_n不得调用会重算Ncap*d_ref的旧快捷入口。评价必须记录样本数、已完成update、残片、评的是更新前还是更新后，不能只靠文件名。

# 18. Inference与实际执行算法

```text
输入：真实snapshot X、历史hidden h、固定episode R、冻结θ
1. 从真实能力、facts、合同和Safety计算candidate IDs与mask。
2. 如无合法候选，交独立安全终止；不构造虚假动作。
3. 由实际base input推进GRU一次，得到z^o与h'。
4. 编码当前合同图Z00；需要prior时编码Z01。
5. 对每个合法候选构造只读nominal patch；同patch生成Z10/Z11。
6. 计算DK、DH、DP；合同主分数b与零锚定残差Δ。
7. masked categorical采样（训练）或canonical argmax（评价）。
8. 只执行选定技能，记录真实sim端点、controller exit。
9. 由实际观测→perception→Verifier更新真实facts；独立Evaluator给reward/done。
10. 持久化真实转移；更新训练参数仅在获准PPO边界进行。
```

shadow forward复用调用前hidden，不再次提交GRU或环境step。LLM推理延迟不推进冻结仿真时，分别报告simulation seconds与wall seconds，不能把逐步API开销当0来宣称更快。

# 19. Information Boundaries与方法合同

本节是baseline定义权威源。A_STAT是本轮最终审计确定的对照，尚未声称生产实现或实验已经完成。

| 方法 | 当前合同／效果输入 | prior | 名义候选干预 | 当前职责 |
|---|---|---|---|---|
| B0 | 原始PRE/ADD/DEL/UNKNOWN和同源facts/goal，Set Transformer | 无 | 无 | 复用历史seed0系统参照，不是“无结构信息” |
| B1-K | 当前K图、共同c_i；phi_K(c_i,0) | 无 | 无 | T_B匹配no-intervention，补s1 |
| B1-K+E | B1-K＋候选PRE/ADD/DEL/UNKNOWN/guards及grounded ownership | 无 | 无 | T_B s0 MUST；s1只按冻结条件申请elastic |
| B2 | 当前K＋DK | 无 | 有 | 合同干预控制 |
| Full | 当前K＋DK＋DP唯一直接prior入口 | 80/20 | 有 | 默认Method 2.1.1 |
| A_STAT | 与Full相同K、DK、context、Actor/V/Q，只将DP输入换为S_R | 同Full | 同Full | C2单变量、参数形状匹配对照，T_P三seed |
| A_CAT | K＋DK＋四视图／合同重复anchor | 同Full | 同Full | 灵活融合设计空间，T_P三seed |
| A_Q | Full | 同Full | 同Full | 仅λ_Q=0；T_P s0/s1，裁剪混杂须报告 |
| B1-H | B1-K合同路径＋当前H/K静态差分 | 80/20 | 无 | OPTIONAL / elastic-pool diagnostic baseline |
| B_PLAN | 公开facts、完整合同、有限规划与代价 | 无 | 搜索中符号后继 | 零训练合同参照 |
| B_PLAN+R | 同一规划器＋冻结R启发式／tie-break | 真实R | 符号搜索 | 零训练朴素prior使用 |
| B_PLAN+R* | 同一规划器＋有限宇宙oracle关系 | 仅分析 | 符号搜索 | oracle-informed reference，不是保证upper bound |
| LLM-Planner | 同源facts/goal/candidates/合同/R及公共摘要 | 同一R | 自主推理 | MUST逐步基础模型参照 |
| VLM-Planner | 同上＋当前RGB | 显式声明 | 自主推理 | OPTIONAL，额外观察与调用成本 |

## 19.1 B1-K+E — 候选效果描述，不施加后继

\(e_i^E\)为同一grounded候选的PRE_POS/PRE_NEG、ADD/DEL/UNKNOWN、条件guards与效果ownership/binding组成的token集合。每条保留predicate、参数有序角色、类型、对象绑定和字段类型，通过共同字段编码与置换不变pool读出。

保留B1-K已有全部当前信息；不先对F执行效果赋值，不运行derived closure，不构造名义afterstate，不把“当前事实会怎样改变”的计算结果作为隐藏特征。允许读取公共当前事实，不允许结果标签或oracle几何。

效果读出的位置、宽度与实际active参数在S0、首次+E采样前固定，不能看到B2/+E成绩后改变。+E是有针对性的经验控制，不是偷偷替换历史B1-K，也不是自动严格等参数。**T_B seed0 MUST**；只有冻结的非平凡学习规则触发时才申请seed1，所有新增attempt共用4个elastic名额。

## 19.2 B1-H-MATCHED-v1 — 保留定义，非核心C2对照

\[
u_i^{K,H}=\phi_K(c_i,0),\quad S_R=Z_{01}-Z_{00},
\]
\[
u_i^{P,H}=\phi_H(c_i,u_i^{K,H},S_R)-\phi_H(c_i,u_i^{K,H},0),
\quad\Delta_i^H=B\tanh(w_H^\top u_i^{P,H}).
\]

只读当前K/H，不生成Z10/Z11。旧源码phi_K(c_i,ZK)分支不能冒充该matched定义。此baseline仍有candidate query，但没有候选名义后继。

它与Full同时在合同干预和prior输入机制上不同，所以不能作为“prior后果条件化是否必要”的单变量证据。仅在有独立问题时由elastic pool授权；T_P B1-K/B1-H四格不在基础矩阵。若另行做四格，只作描述性包级比较，两行prior模块不同，不称因果析因。

## 19.3 A_STAT — Full的静态prior输入控制

\[
u_i^K=\phi_K(c_i,D_i^K),\qquad S_R=Z_{01}-Z_{00},
\]
\[
u_i^{P,S}=\phi_P(c_i,u_i^K,S_R)-\phi_P(c_i,u_i^K,0),
\qquad \Delta_i^S=B\tanh(w_P^\top u_i^{P,S}).
\]

与Full保留同一contract branch、DK、nominal patch、c_i、phi_P、Actor/V/Q架构、bound、parameter shape、优化与预算；**唯一设计干预是先验输入张量：DP换为S_R**。独立run参数从头训练，不共享已经学好的权重；“同参数形状／数量”不等于“训练后数值相同”。S0导出active参数表验证精确匹配，不能新加一套静态head。

A_STAT可缓存Z00/Z01/Z10并省略不影响其输出的Z11计算，计算量较小需单独报告；这不改变其信息内容，不以算力耗时相同冒充信息公平。S_R在状态内共享，但phi_P仍读取候选context和DK，因此它可以产生候选不同的静态prior残差。

R为空时S_R=0，uP=Δ=0。空patch时S_R一般不为0；可加分离E时S_R仍可能保留B(R)−B(∅)，所以它不具有Full的empty-patch null或additive cancellation保证。不得额外补DP gate。

## 19.4 A_CAT — 灵活四视图融合

\[
C_i=[Z_{00},Z_{10},Z_{01},Z_{11}],\quad
C_i^0=[Z_{00},Z_{10},Z_{00},Z_{10}],
\]
\[
u_i^{P,CAT}=\phi_{CAT}(c_i,u_i^K,C_i)-\phi_{CAT}(c_i,u_i^K,C_i^0),
\qquad\Delta_i^{CAT}=B\tanh(w^\top u_i^{P,CAT}).
\]

同R、同patch、同训练目标，保留原定义的更宽首投影。R为空精确零，空patch／E可加分离时可以非零；不给CAT补Full专属gate。该比较同时涉及输入形式、非线性位置、null性质与首投影容量，不叫单变量；A_STAT负责补足单输入变量问题。

## 19.5 归因层级与基础配置

Full/B2是prior接口及训练协议整体增量；Full/A_STAT是保留合同干预与参数形状的prior输入控制；Full/A_CAT是限制性交互与灵活融合；B0/B1-K为系统架构比较；B1-K+E/B2为效果描述与施加的代表性经验控制。

基础新增训练固定为T_B三条（B1-K s1、B2 s1、B1-K+E s0）加T_P十四条（Full/B2/A_STAT/A_CAT各三seed，A_Q两seed），合计17。B1-H不在基础；全部条件性新增统一消耗4个elastic attempts，第一周期17＋4≤21。

LLM输入尽量匹配但pretraining知识与表示能力不等同。VLM-Planner额外当前RGB单列。不能把“给了同源字段”写成所有模型具有相同先验知识，也不能将LLM胜出隐去。

# 20. C1→C2→C3的必要性与依赖

C1使候选变成一个对当前事实的明确假设变化，而不是只用不透明ID推荐动作。C2检查关系怎样改变这个变化的表示；C3用实际成败与成本决定这种表示应如何影响策略。这条链的每一步都可能失效：无patch→无变化；R没有可达或非冗余作用→无可识别先验问题；有响应但学习不正确→噪声被传播。

C2不能只靠“DP有范数”证明，C3不能只靠“Q loss有限”证明，C1不能仅靠弱baseline证明。对应证据分别是Full/B2、Full/A_STAT与Full/A_CAT的三类比较；Full/A_STAT/A_CAT/A_Q共享独立Q_ref及同run训练轨迹；T_B B1-K/+E/B2、P1与B_PLAN。C2→C3由跨接口utility-aligned reliance连接，不再是只测Full自己的一个独立量。

# 21. Failure Modes与研究解释

| 观察 | 优先解释范围 | 不能直接推出 |
|---|---|---|
| patch为空 | 该步没有名义输入变化 | 整个任务prior无用 |
| source R非空但无DP | schema、reach、readout、模型权重或可加抵消 | VLM一定错／方法已失败 |
| DP非零但Δ≈0 | downstream读出、零锚、初始化或学习 | 接口完全没作用 |
| Δ非零但动作不变 | margin大、共同shift、近似同值动作 | prior完全无用或有收益 |
| Full/CAT clean持平 | ceiling、代价指标或确实无差异 | 局部性／鲁棒性自动成立 |
| A_STAT或CAT优于Full | 静态信息、灵活非线性、读出／容量或优化的候选解释 | 仅凭胜负定位唯一原因或自动升级 |
| wrong R仍被使用 | 推理未接触错误模式、训练覆盖或错关系偶然有用 | bounded必然失效或有安全保证 |
| A_Q无差异 | Q冗余、欠训练、预算、其他loss或裁剪缩放差异 | C3必然消失或Q必然提升表示 |
| B_PLAN足够强 | 名义合同已解决task，学习作用在摊销或别处 | 搜索必然弱于神经方法 |
| LLM强于Full | 更大模型推理与外部知识可更有效 | 应隐藏baseline或改更弱模型 |

唯一统一处理顺序见实验计划S4；不同原因走不同分支。不把任何单一负现象设成唯一能“证伪”的判据，也不预言某种升级必然修复。

# 22. Conditional Method Upgrade：本轮不启用

## 22.1 一次方法决策门

仅S4可批准一个方向。核心输入包括Full、B2、A_STAT、A_CAT、A_Q、E1–E6、prior opportunity density、DP-Zero、readout decomposition、跨接口／跨checkpoint G_ref、Q预测质量和gradient clipping audit。B1-H不是该门的必需输入。

基础17＋单一elastic4＝第一周期硬上限21。全部技术重跑、匹配调参、分歧seed、+E s1、可选B1-H与upgrade净增共用这4个名额。升级先替换尚未启动配置；已发生的pilot和失败attempt仍计成本，不因改名消失。若必要同版本control导致超过21，停止自动执行，请求人工研究决策；不削弱control，也不自动突破上限。

A_STAT优于Full提示静态信息价值，但仍需CAT、读出分解、robustness与G_ref排除其他解释；CAT优于Full且A_STAT约等于Full，才更支持非线性交互方向，但不是单凭这组排序就证明locality。静态影响分解在本周期先作为分析；不能凭该结果另开未经定义的第三升级路线。

## 22.2 Upgrade B：Consequence-Local Nonlinear Interaction

触发：CAT的差异不由不公平配置解释；A_STAT没有同等静态收益，读出分解与合法局部输入诊断支持有价值的非线性交互；当前global readout或纯DP压缩确有机制性不足。不能仅把冻结CAT换一个pool后仍成功，就证明新局部模型训练一定有效。

本文件修改review的简单局部CAT形式，以保留empty-patch null。先由合同patch构造固定区域N_i，区域选择规则R-independent或在所有待比较条件的固定闭包上预先定义。四视图**在区域内重算局部消息传递**；不能对已被全局／外部消息污染的节点embedding做末端mask后宣称严格局部。越界边、degree normalization、目标行和边界特征一致处理，边界仅取K侧公共信息。

令U_ab^i是区域内四视图，令共享非线性映射Ψ_i得到局部读出：

\[
J_i^{CL}=\Psi(U_{11}^i)-\Psi(U_{01}^i)-\Psi(U_{10}^i)+\Psi(U_{00}^i),
\]
\[
u_i^{P,CL}=\phi_{CL}(c_i,u_i^K,J_i^{CL})-\phi_{CL}(c_i,u_i^K,0),\quad
\Delta_i^{CL}=B\tanh(w^\top u_i^{P,CL}).
\]

非线性发生在交互差分之前，可能保留被原表示差分丢掉的局部组合信息。它保持R-empty和patch-empty null、bound及**按新区域构造明确保证的输入支撑限制**；不保证原E可加分离就仍抵消，因为Ψ可以将主效应非线性混合。若把review的φ(C_local)−φ(C_local^0)直接作为CL定义，则patch-empty一般不为零；该版本只能称局部prior融合，不能冒称保留全部原性质。

原Full不是已证明局部的“线性特例”：它含global非线性readout。论文只能把原Full、全局CAT与新CL放在“交互位置×支撑区域”的设计空间中比较，不能未经公式等价证明说三者严格嵌套。

升级标Method 2.2-B。优先用新CL的扩展seed替换尚未启动的Full s1/s2；原Full s0保留为已发生的2.1.1成员。示例：CL s0/s1/s2加一个必要matched-readout控制，相对被替换的两个名额净增2，消耗elastic2；这只是记账示例，不自动证明其control已经充分。S4必须明确同版本static/fusion/Q控制是否仍可解释；旧A_Q不能冒充新CL的Q消融。需要超过剩余elastic或21总上限时暂停，请求人工决定，不增加隐含桶。

## 22.3 Upgrade C：Counterfactual Prior Credit

触发同时要求：合法且有效prior机会；原A_Q作用弱／prior梯度欠训练；独立分析显示用途校准差；全候选Q可前向且在相关数据上有足够可靠性。当前全候选输出不是自动许可，未执行动作Q偏差可能被利用。

可选新loss原型：

\[
L_{PC}=-\lambda_{PC}\mathbb E_s\sum_i
[\pi(i|s)-\operatorname{sg}(\pi^{\Delta=0}(i|s))]
\operatorname{sg}(\widehat Q(s,i)).
\]

仅作用于prior-specific参数，明确detach合同context／共享encoder／critic输出的边界；否则就不是“只给prior通路credit”。先用滞后critic与warm-up、有限权重，配套原Full及lambda_PC=0控制。所有真实label仍只来自执行，Q对未执行候选是模型估计，不是真实counterfactual target。

Bound只限制当下残差，不能保证不被偏Q误导。没有Q可靠性证据时仅计算G_Q作代理分析，不将它加入训练。方法标2.2-C，第一周期最多一个升级方向。原Full必须保留为λ_PC=0匹配控制；PC的净新增run、必要同版本Q控制和技术重跑全部占同一elastic4。PC三seed净增3已经消耗三个名额；若其他必要控制使总量超过21，停止而不删除控制。不要顺便加trust gate、relation classifier或world model。

# 23. Novelty Defense：问题、证据与不可越界回答

| 质疑 | 正确的研究回答 | 对应证据 |
|---|---|---|
| “就是afterstate／KR 2023的GNN后继策略” | C1采用已有思想作为prior查询锚，不宣称RL＋GNN＋后继本身新 | B1-K+E、DK、Full/A_STAT及[W20]定位 |
| “B2只是更容易读到效果” | 代码只证明可访问性，实际效果描述对照必须跑 | T_B B1-K+E s0 MUST，按规则s1 |
| “C2只是四项相减” | 代数不新；接口区别是empty-patch null及条件可加抵消，共享null/bound不冒称独有 | Full/A_STAT单输入对照、Full/A_CAT设计空间、性质表 |
| “收益只是静态prior或多参数” | A_STAT保留DK、同phi_P和参数形状，只有prior输入张量不同 | 三seed Full/A_STAT；active参数表 |
| “融合更强” | 它是研究要测的反例，不预设Full或CAT失败 | A_CAT、utility-aligned reliance、readout decomposition |
| “为何不用planner或LLM直接选” | 固定关系接口是摊销选择，不是否定搜索／大模型推理 | B_PLAN、B_PLAN+R、LLM-Planner、API与wall成本 |
| “Q-shaping有最优性保证，你只有bound” | 研究对象和假设不同：LLM启发式Q指导vs固定关系表示接口；不借用也不贬低其保证 | 正面讨论[W21,W22]；不增新复杂baseline |
| “就是LLM teacher” | 不蒸馏teacher action，不按统一时间表代替状态／候选依赖判断 | [W23]与跨接口、跨checkpoint G_ref |
| “Q创新在哪里” | Q是现有grounding通道；C3是无关系标签下可独立测量的prior reliance | A_Q、共享Q_ref、clip审计，不以自身Q自证 |
| “局部性被global读出破坏” | 当前只给节点／条件命题，不作端到端严格声明 | GRAPH_REMOTE、行分解、保留真实远端响应 |
| “任务为Full定制” | E1–E6、模型结果前冻结family、全分母、独立test与规划器R增量 | 公开B_PLAN/B_PLAN+R的J与success差，不要求必须正 |
| “有界等于安全吗” | 不等于；界只约束固定同状态直接残差／概率比 | 权限隔离与真实robustness分别呈现 |
| “seed少、全是自定义task” | 关键Full/B2/A_STAT/A_CAT三seed、Q两seed；任务族和外部参照明确但不抹去范围限制 | 逐seed和paired test，不能承诺录用 |

任何措辞不能替代必要比较；也不将两个自定义任务或17个run等价为某会议认可的固定门槛。

# 24. Paper Story与实验并行写作

一句话：**CP-DISR以候选名义后果为锚，研究静态上下文、后果条件化与灵活融合三种固定关系prior接口，并用独立动作效用检验真实结果训练下的prior reliance。**

## Abstract / Introduction

问题是能力／合法性不等于长期效用，语义关系可能有帮助也可能误导。贡献层级为一个统一方法、一个学习／分析贡献、可选一个诊断评价贡献；C1和辅助Q分别作为构造与grounding步骤，不各写成全新算法。

Related Work正面讨论afterstate、KR 2023策略梯度广义规划、Q-shaping、LLM4Teach、静态scene-graph输入、Flax/LLM-Flax及层级不同的PSL/BOSS。结果句以\([RESULT_PENDING:Sx]\)占位，不预先承诺Full最强。

## Method

真实facts与合同→名义patch→DK查询锚→四视图DP→零锚／bound→Actor/V/Q→真实执行训练。随后给Full/A_STAT/A_CAT/B1-H性质表与utility-aligned reliance定义。先解释共享性质，再解释Full独有的empty-patch null和条件性抵消；不以local theorem代替现有global读出分析。

## Results

1. T_B效果描述与施加、已有P1及第二seed；
2. T_P E1–E6、自然非冗余prior与机会密度；
3. 固定primary endpoint J：Full/B2、Full/A_STAT、Full/A_CAT；
4. Full/A_STAT/A_CAT/A_Q共享reference和同run训练轨迹；
5. NoPrior/Target-swap、GRAPH_REMOTE及readout decomposition；
6. 搜索／LLM的成绩与在线成本，SHOULD难度轴和有来源的扩展。

D0、T_C作为诊断适用边界保留；不将non-diagnostic任务当整个prior研究的否定，也不重复训练到正结果。

## Discussion / 写作锁定

讨论静态有用信息可能被纯交互遗漏、global nonlinear混合、辅助Q与clip的共同作用、有限reference和seed带来的不确定性。Methods、问题、协议及已有Results现在写；S2填family和primary endpoint manifest；S3/S4填探索证据和方法选择；S6填独立终点；S7填机制与robustness。不等全部实验结束才写稿。

只有S4正式触发方法2.2-B/C时调整相应技术段；常规新结果只填数字、图表和有证据的经验句，不重新发明Master。

# 25. Claim Boundary与最终写作规则

| 可以作为定义／条件性质写 | 需要真实实验后决定 | 当前不能写 |
|---|---|---|
| 固定R、同patch、只读nominal | Full利用prior优于B2 | prior普遍带来性能增益 |
| exact null、bound、可加条件抵消 | Full/A_STAT/A_CAT性能、效用与敏感度取舍 | 2×2代数新颖 |
| 真实执行目标、无关系label | Q改善AUC或用途校准（含clip解释） | 全候选Q有全候选真值监督 |
| 节点级locality／条件命题 | 全流程距离敏感度 | 现有global readout保证严格off-path不变 |
| C1已有局部seed0发现 | 跨seed、任务族和holdout | B1-K失败证明所有current-graph方法不足 |
| independent R\* analysis boundary | 外部planner相对摊销效率 | oracle planner必为理论性能上界 |

最终展示可选择清楚的task、qualitative、图表布局与合理平滑，但主比较逐预定seed和有效反例必须有正文或附录去处。选取illustration不能篡改主结果集合。失败测试、负研究结果、方法升级前后数据进入Raw Archive，不全部强塞主表，也不悄悄消失。

# 附录 A — 最终审计落实记录（内部，不作为论文结果）

| Required patch | 冻结落点 | 实验影响 |
|---|---|---|
| X1 A_STAT | §5、12、19.3；EP §3、7、S3–S7 | T_P A_STAT三seed；B1-H/B1-K四格移出基础 |
| X2 B1-K+E | §5、11、19.1；EP §3.1、S0、§7 | T_B s0 MUST，s1条件占elastic |
| D2 性质比较 | §13.5 | 共享null/bound/权限；Full区别仅空patch null与条件抵消 |
| D3 primary endpoint | §6；EP S2/§20 | 训练前固定J；次终点和机会分母预声明 |
| D4 17＋4≤21 | 文档控制、§19.5、22；EP §7、24 | 所有新增attempt只有同一4槽池 |
| D5/D6 术语与测试 | §2、13、16、25；EP S0/S7 | 默认GRAPH_REMOTE；四类Full性质单测 |
| A1 readout decomposition | §13边界；EP §15.5 | 0新RL，共享prior读出行干预，不宣称因果份额 |
| A2/A4 校准与clip | §16、17.2；EP §16 | 同64分支、跨接口/阶段、真实pre/post clip |

低成本写作建议已纳入：指定近邻、贡献层级、Problem–Method表、utility-aligned prior reliance命名、baseline差距透明报告和评价去重。原Review的强P4与旧四格义务不恢复。

两项数学／计数限定：goal-aligned行并非天然严格“后果局部”，非线性行干预不可加成100%来源份额；跳过S5九个T_P新seed的E2 dev点节省9×16=144个episode，不是288。它们仅纠正解释与算术，不新增研究方向。

# 附录 B — 来源与核验层级

## 项目输入

- [S1] `Pasted text(20260928-083117).txt`：v4/v2整合输入；与最新冻结指令冲突时按[S7]。
- [S2] `CP_DISR_Research_Review_of_v1_1.md`：2026-09-28外部审查；其自身注明未读源码／未执行实验，建议不是既成事实。
- [S3] `CP_DISR_Final_Research_and_Experimental_Package_v3.1(1).zip`内`research/CP_DISR_Paper_Oriented_Research_Specification_v3.1.md`、`interfaces/paper_method_manifest.yaml`：默认Method2.1.1定义；尤其§13.3、14.3、15–17、24–25。
- [S4] `CP_DISR_Experimental_Plan_v1.1(1).md`：B0/B1-K、A_CAT、H及Actor规则；历史预算不再授权。
- [S5] `cp_disr_master_upgrade/CP_DISR_Master_Plan_Research_Completion_v1_1.md`与历史Master Ledger：事实与上一研究矩阵输入。
- [S6] `CP_DISR_Final_Audit_RCv4_EPv2.md`：最终审计，SURGICAL REVISION THEN FREEZE；8组必要修改。
- [S7] `Pasted text(20260928-103411).txt`：本轮最终修订、17＋4≤21、冻结与禁止实验要求。
- 底稿：`CP_DISR_Final_Research_Content_v4.md`、`CP_DISR_Final_Experimental_Plan_v2.md`；原始字节hash保存在配套Freeze Manifest中，未覆盖。

## 只读源码证据（不等于生产测试已通过）

仓库均为rollinpioneer/graph，ref=`55a3b7ce35edebbbb8a587fe1e3a98ca1967db07`。

- [G1] `src/cp_disr/neural.py`，blob `e8e0cd8d37ad37ce3f96d3bf2faef4b46639f5ee`：GoalReadout、GraphEncoder、B1-H、A_CAT、V/Q。
- [G2] `src/cp_disr/graph.py`，blob `f8c9c5e8711e82887a30174daa99796bd32595ea`：ADD/DEL边、reverse、four_views固定R。
- [G3] `src/cp_disr/platforms/libero/snapshot.py`，blob `5fb054f8de355907062ccb2ee846446429a389ea`：候选8维、稳定hash、base input。
- [G4] `src/cp_disr/torch_rl.py`，blob `d1d1b01b8393b61e6773455739814995cd6b4257`：Q gather、TD/GAE目标和PPO实现。

该ref用于证据固定，不意味着后续必须从这一过时worktree原样启动。S0只读diff后冻结未来active source；历史R3修复与generation入口分别核验，不能由一份文件存在推断所有runner已正确集成。

## 已有实验定位

- [E1] R1 `55a3b7ce35edebbbb8a587fe1e3a98ca1967db07`，`runs/v13_r1/20260926T112613Z/`。
- [E2] R2 `eaa43e97d2808c27b4c16c29011aa678f06d1b0f`，`runs/v13_r2/20260927T070500Z/`；实际旧账号运行。
- [E3] P1/P2 `5a0ef69d2b236c8ba6ba109774ce6a0940ce5fdf`，`runs/v13_parallel_mechanism/20260927T134414Z_fc6cbf1f/`。
- [E4] R3 `dbaf23ba7c8137b7cfcf31b7c9a1c28af7c4b447`；Recovery `cd95646ee0502419052210590b280750374a0401`。
- [E5] 历史D0 test `f34789d6a5e6953b9416d06eb43dfaa14b6530e1`；非当前profile重复seed。
- [E6] T_C nonoverlap `66614acb443dcacee2b16c5472d4ecbd697bf561`；不是成功的prior主任务。

## 原始文献（2026-09-28；W1–W19沿用底稿，W20–W27仅补核验指定近邻）

- [W1] Oh, Singh, Lee. *Value Prediction Network*. NeurIPS 2017. https://arxiv.org/abs/1707.03497
- [W2] Ståhlberg, Bonet, Geffner. *Learning Generalized Policies Without Supervision Using GNNs*. KR 2022. https://arxiv.org/abs/2205.06002
- [W3] Toyer et al. *Action Schema Networks: Generalised Policies with Deep Learning*. https://arxiv.org/abs/1709.04271
- [W4] Garrett, Lozano-Pérez, Kaelbling. *PDDLStream*. https://arxiv.org/abs/1802.08705
- [W5] Ahn et al. *Do As I Can, Not As I Say*. https://arxiv.org/abs/2204.01691
- [W6] Rana et al. *SayPlan*. https://arxiv.org/abs/2307.06135
- [W7] Yan et al. *Efficient Reinforcement Learning with Large Language Model Priors*. https://arxiv.org/abs/2410.07927
- [W8] Wang et al. *GAVEL: Graph World Models for Verified and Efficient Long-Horizon LLM Task Planning*. 2026预印本. https://arxiv.org/abs/2609.19315
- [W9] Lorang et al. *Build on Priors: Vision–Language–Guided Neuro-Symbolic Imitation Learning for Data-Efficient Real-World Robot Manipulation*. 2026预印本. https://arxiv.org/abs/2604.03759
- [W10] Jaderberg et al. *Reinforcement Learning with Unsupervised Auxiliary Tasks*. https://arxiv.org/abs/1611.05397
- [W11] Schulman et al. *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
- [W12] Schulman et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. https://arxiv.org/abs/1506.02438
- [W13] Schlichtkrull et al. *Modeling Relational Data with Graph Convolutional Networks*. https://arxiv.org/abs/1703.06103
- [W14] Lee et al. *Set Transformer*. https://arxiv.org/abs/1810.00825
- [W15] Agarwal et al. *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. https://arxiv.org/abs/2108.13264
- [W16] Liu et al. *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*. https://arxiv.org/abs/2306.03310
- [W17] Dayan. *Improving Generalization for Temporal Difference Learning: The Successor Representation*. Neural Computation 5(4), 1993. https://doi.org/10.1162/neco.1993.5.4.613 （本轮核对出版社书目与摘要。）
- [W18] Sutton, Precup, Singh. *Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning*. Artificial Intelligence 112, 1999. https://doi.org/10.1016/S0004-3702(99)00052-1 （本轮核对出版社书目与摘要。）
- [W19] Sutton, Barto. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. https://reinforcementlearning.pubpub.org/ （本轮核对出版社开放版入口；不是重新审阅全书的声明。）

- [W20] Ståhlberg, Bonet, Geffner. *Learning General Policies with Policy Gradient Methods*. KR 2023, 647–657. https://proceedings.kr.org/2023/63/ DOI:10.24963/kr.2023/63 （本轮核对官方论文页；不把后上传arXiv年份当会议年份。）
- [W21] Xiefeng Wu. *From Reward Shaping to Q-Shaping: Achieving Unbiased Learning with LLM-Guided Knowledge*. 2024预印本. https://arxiv.org/abs/2410.01458 （本轮核对原始摘要；其保证按该文自身假设理解，不移植到CP-DISR。）
- [W22] Xiefeng Wu. *Enhancing Q-Learning with Large Language Model Heuristics*. 2024预印本v3. https://arxiv.org/abs/2405.03341 （原始摘要和版本核验；不声称本轮逐项复证全部收敛定理。）
- [W23] Zhou et al. *Large Language Model as a Policy Teacher for Training Reinforcement Learning Agents*. IJCAI 2024, 5671–5679. https://www.ijcai.org/proceedings/2024/627 （LLM4Teach；官方书目及arXiv:2311.13373v6 §1/§3中的teacher/student、annealing机制已作窄核验。）
- [W24] Kim, Lee. *LLM-Flax: Generalizable Robotic Task Planning via Neuro-Symbolic Approaches with Large Language Models*. 2026预印本. https://arxiv.org/abs/2604.26569
- [W25] Du et al. *Fast Task Planning with Neuro-Symbolic Relaxation*. https://arxiv.org/abs/2507.15975 （原始v1为2025；原始页面v2列RA-L 2026及DOI:10.1109/LRA.2026.3662556，按实际引用版本记录。）
- [W26] Dalal et al. *Plan-Seq-Learn: Language Model Guided RL for Solving Long Horizon Robotics Tasks*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/2e9f9cde1b709281a06dd14f679e4c51-Abstract-Conference.html
- [W27] Zhang et al. *Bootstrap Your Own Skills: Learning to Solve New Tasks with Large Language Model Guidance*. CoRL 2023, PMLR229, 302–325. https://proceedings.mlr.press/v229/zhang23a.html

# 附录 C — Cross-Document Consistency Audit

| 检查 | 本文件 | Experimental Plan v3 | 冻结结果 |
|---|---|---|---|
| 默认算法／升级 | 控制、§22 | 控制、S4 | 2.1.1；仅S4正式触发2.2-B/C |
| RQ与贡献 | §2、5、20 | §2、25 | 方法＋utility-aligned learning/analysis＋评价 |
| A_STAT | §19.3 | §3.2 | 同公式、同DK、同参数形状，仅DP→S_R |
| B1-K+E | §19.1 | §3.1、§7 | PRE/ADD/DEL/UNKNOWN/guards、grounded binding；s0 MUST |
| B1-H | §19.2 | §3.3、§7.2 | OPTIONAL / elastic-pool；无基础四格 |
| 预算 | 控制、§19.5、22 | §7、8、24 | 基础17、唯一elastic4、硬上限21 |
| seed | §19.5 | §7、11 | T_B三新增；T_P F/B2/STAT/CAT各3、AQ2 |
| primary endpoint | §6 | S2、§20 | T_P的J先冻结；次终点与机会密度单列 |
| 性质 | §13 | S0、S7 | P1/空patch/bound/条件重排；global不保证strict locality |
| 静态／图远端 | §12、13、19 | RQ、S3/S4/S7 | A_STAT静态对照；GRAPH_REMOTE默认 |
| 校准 | §16 | §16 | 共享64 development分支，四接口及同run E2/E4/E8 |
| clip | §17.2 | S0、§16.4 | pre/post真实norm、触发率，不新增run |
| readout | §13边界 | §15.5 | goal-aligned/global行分解，不推断严格局部性或可加百分比 |
| 写作／执行 | §24、25、附录D | §26、Runbook | 现在写；RESULT_PENDING；authorized=false |

配套自动审计核算矩阵、预算、来源hash及两文档的公式文本相等；这不表示S0生产单元测试已经执行。

# 附录 D — 冻结与授权边界

本文件定义研究，不授权训练、仿真、API或模型前向。实验执行只从Experimental Plan v3的S0–S8领取获准工作。账号、provider历史绑定、目录与Git操作见其Runbook附录，不在论文正文展开。

本次局部修订不改变默认Method 2.1.1，不证明尚未运行的A_STAT/+E或结果。普通新证据只更新状态、RESULT_PENDING、图表和经验解释；只有S4明确触发2.2-B或2.2-C时才允许方法版本改变。已冻结文件以hash归档，不与旧Master并行派发。

**Research Content v5 and Experimental Plan v3 are now the sole authoritative project specifications unless S4 explicitly triggers a method-version change.**

**研究未完成不等于论文不能写；研究未得到正结果也不等于可以改写数据。先完成有辨识度的检验，再按真实结果升级、保留或调整贡献。**
