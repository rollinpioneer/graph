---
title: "CP-DISR（M1）研究定稿"
subtitle: "合同／先验双差分结构推演与真实回报监督的技能级策略"
date: "2026-09-20 · Research Specification v2.0"
lang: zh-CN
---

# 文档控制与结论

**文档身份：本项目当前方法、MVP实现、实验协议和论文写作的统一规范。**主方法正式工作名为 **CP-DISR：Contract–Prior Differential Intervention for Skill Reasoning**，内部编号继续使用M1。本名称是本项目拟定的工作名，不表示首创性或命名唯一性已经获得外部确认。

**本轮结论：论文写作前必须完成的研究实验为0项。**Problem Formulation和Method已有完整可写的定义；性能、机制收益、鲁棒性与泛化由后续真实实验验证。可以立即开始论文写作，同时进行实现与实验。这个结论不等于代码已跑通、感知已验证、技能已安全部署，也不等于方法有效性已经得到证明。

本文件只保留当前主方案：**冻结强API VLM＋离线关系缓存；SLG式Action–Proposition合同图；共享标准R-GCN；候选名义成功干预；合同／先验双差分；零锚定有界先验残差；独立V与辅助Structural Q；真实终端奖励下的技能级PPO。**不增加VLM微调、LLM teacher、额外文本语义编码、世界模型、在线图修复、图奖励、M2/M3或新损失。

**来源与补齐规则。**[S0]为本轮《Pasted markdown(8).md》，其明确决定优先于历史讨论；最近的M1实现、SLG定型和强API选型讨论作为补充设计依据。本文的数学推导是对既定公式的整理；标为“实现默认”的数值、异常处置和配置是为实施补齐的约定，不是已有实验事实。硬件、具体任务和底层控制器未由材料给定，必须标为待绑定，不以通用知识补造成已部署资源。外部核对仅用于引用成熟底座及确认API文档，不引入新研究方向。参考文献列于附录D。

**文件使用。**Markdown是后续版本维护源；Word是同内容阅读版。任何改变动作层级、奖励、图底座、差分算子或监督来源的修改，都需要新版本记录。超参数和硬件设置在开发集确定后写入运行manifest，不能查看测试结果后无记录地调整。

| 状态 | 含义 |
|---|---|
| 方法冻结 | 当前采用的算法与信息权限定义 |
| 实现默认 | 可开始编码的初值或协议选择，不是最优值 |
| 运行前必填 | 真实硬件、控制器、标定、时限等未给定字段 |
| 待实验证明 | 性能、样本效率、鲁棒性、泛化与具体模块贡献 |

# 1. 研究问题与正式定位

## 1.1 Problem Statement

长时程机器人组合任务中，固定技能库提供基本执行能力，但高层策略仍需根据目标、当前环境和执行历史选择下一项技能。技能合同给出经过检查的启动要求和名义成功效果；视觉语言模型可以补充合同未直接表达的语义关系，但这些关系可能重复、遗漏或错误。

**本研究学习一种候选条件化的结构决策机制：在合同模型上对每个可执行技能施加临时名义效果，分离合同结构变化与VLM软关系对该变化的额外交互，再以实际执行所得回报训练这些表示的决策用途。**

“可信合同”在本文中指来源明确、经校验、版本冻结的能力接口；它不是对机器人全部物理过程的完美模型。“不完美先验”也不意味着故意选择差模型，而是方法不以先验绝对正确为前提。

## 1.2 唯一论文主定位

**Contract–Prior Differential Structural Reasoning for Long-Horizon Skill Decision。**

研究中心不是新的图本体，也不是VLM规划能力；中心是**候选干预—合同／先验分离—结果监督**这一计算与学习闭环。PathGraph保留为项目与运行时结构名称；论文算法使用CP-DISR（M1）。

备选名称仅供最终标题选择，不对应新方法：CP-SIR（Contract–Prior Structural Intervention Reasoning）；D2-Skill（Dual-Difference Skill Reasoning）；OG-SIR（Outcome-Grounded Structural Intervention Reasoning）。

## 1.3 主研究问题与核心假设

主问题：给定固定技能和可错语义关系，能否通过候选名义结构变化及其先验交互，学到比只读取当前图更有用的技能选择表示？

核心假设：候选成功分支的符号效果提供可计算的结构差异；共享编码器下的双差分突出软关系如何改变对该差异的响应；实际动作价值监督和PPO使这些响应有机会成为决策相关特征。**假设描述学习动机，不预设一定优于普通前瞻或信息匹配网络。**

# 2. 核心动机、贡献与边界

## 2.1 核心矛盾

直接服从语义关系可能沿着错误依赖行动；完全忽略关系可能丢失跨技能的有用联系；只编码当前增强图，又难以明确区分“候选会改变什么”和“语义先验额外影响了什么”。M1将这两个变化拆成可计算对象，并把最终评价交给真实执行回报。

## 2.2 拟写入Introduction的三项贡献

**C1．候选名义结构干预。**在统一的动作—命题合同图中，以只读临时副本表示每个候选按合同成功后的逻辑变化，显式保留增加、删除及未知状态，不调用环境真实未来。

**C2．合同／先验双差分算子。**使用同一编码器、节点集合与目标索引，分别计算合同图和增强图的前后差，再分离语义关系对候选变化的交互；合同推理进入主评分，先验交互进入零锚定有界通路。

**C3．真实结果监督的结构学习。**以实际执行动作的TD目标训练辅助Structural Q，与PPO的Actor和V共享结构特征；更新的是网络对关系和上下文的解释，而非把虚拟成功或VLM判断当作奖励。

名义前瞻、向量减法、辅助Q学习各自都有既有基础。拟议创新在三者的具体组合、隔离方式和可验证作用，不声称其中任何常见运算单独为首创。结果完成后才能将具体经验结论写入贡献段。

## 2.3 明确不是核心创新

Skill Contract整理、SLG图表示、schema参数共享原则、标准R-GCN、Fact Verifier、PPO/GAE、API调用与缓存均为底座或基础设施。SLG保留动作、命题及PRE/ADD/DEL关联；ASNet提供schema级共享的参考；R-GCN提供成熟多关系卷积。[R1–R4]

图表示被采用，不代表原论文已验证本项目的三态感知、机器人技能和双差分组合。Compiler/规则管理仅承担绑定与一致性处理，不单列贡献。

# 3. 方法概览与五个运行阶段

## 3.1 初始化阶段

独立任务、技能注册表和对象目录先确定。程序ground技能与命题，建立固定episode合同模板和goal_refs。采集初始RGB-D及允许事实摘要，冻结强API VLM生成少量软关系；解析、验证、去重并缓存。训练从缓存选择原始或扰动后的关系版本，整个episode内保持该版本固定。

## 3.2 每次高层技能决策

真实观测产生当前事实F_t；共享候选生成器与合同/安全检查给出候选和mask。构造$G_t^K$及$G_t^H$。对允许的每个候选应用同一名义事实patch，得到两种临时后继图。共享编码器输出四组目标对齐表示，计算$D^K$、$D^H$、$D^P$。Actor给候选评分并采样。

## 3.3 技能执行阶段

固定低层控制器依据当前估计几何执行技能；关键事实监测可以触发中断。此期间没有新的VLM调用，没有名义图写入真实事实。技能失败、拒绝或超时仍算真实尝试，并计入实际耗时。

## 3.4 执行后更新

控制器结束后，在有界窗口内获取新证据，Verifier确认后置条件，写入当前事实及执行记录。独立评价器计算真实任务成功与终止；生成下一个技能决策快照。

## 3.5 训练阶段

将真实转移写入on-policy buffer；用旧V、实际时长和结束标志计算GAE、V目标与Q目标。PPO更新Actor、共享编码器和独立V/Q head。训练图表示重新计算，不重复调用API或重新扰动历史先验。

```text
Task + Initial Scene -> Frozen API VLM -> Cached Soft Relations
Skill Contracts + Current Facts -> G^K
G^K + Cached Soft Relations -> G^H
Candidates -> Nominal Success Patches -> Four Aligned Graph Views
Shared R-GCN -> D^K / D^H / D^P
Contract Reasoning + Bounded Prior Residual -> Skill Selection
Fixed Controller -> Actual Observation / Facts / Task Reward
Real Transition Buffer -> PPO + V + Structural Q -> High-level Update
```

# 4. 系统假设、时间尺度与符号

## 4.1 任务范围

主版本针对固定技能库、有限已绑定对象、有限谓词词表的桌面组合操作。动作参数包括技能类型、对象和目标角色；连续抓取姿态和轨迹由固定参数生成器及控制器给出。主合同支持带符号的合取前置和目标、显式ADD/DEL，以及注册的有限派生规则。一般嵌套OR、量词和未展开的复杂条件效果不被默认为已支持。

同一事实可以失效或回到UNKNOWN。多个技能建立同一条件表示候选替代，不意味着所有技能都必须执行。任务中复杂逻辑如果不可编译到当前受限表示，应在任务注册时报不支持，不静默改写为合取。

## 4.2 观测与独立任务

$$
o_t=(I_t^{ext},D_t^{ext},p_t,\ell_{t-1},T_t^{remain}).
$$

RGB-D来自固定第三视角；p_t是本体状态；上一技能摘要包含真实结束原因；剩余时间属于任务输入。对象几何和感知质量来自冻结前端，不默认已知环境状态。纯状态输入仅作明确标注的程序诊断，不代替视觉主实验。

$$
x_t=(o_t,\mathcal T,F_t^{core},L_t^{summary}),\qquad
 a_t=(k_t,\operatorname{args}_t).
$$

任务目标、核心事实查询、候选全集、安全规则和奖励不依赖VLM先验。Gripper或低层规划同样不得暗用模拟器对象真值；仿真真值仅可用于独立reward、离线标注、评价或显式状态诊断。

## 4.3 时间抽象

真实物理状态为s_τ；t为高层决策索引。技能从τ_t开始，持续d_t>0个统一基本时间单位，下一次决策在τ_{t+1}=τ_t+d_t。持续时间包括被任务期限计入的规划、运动、确认和等待。技能时间抽象沿用SMDP/Options基础；部分观测和GRU并不自动构成完整马尔可夫状态。[R5]

## 4.4 统一符号表

| 符号 | 含义 |
|---|---|
| $\mathcal U,\mathcal P,\mathcal O_e$ | 技能schema、谓词注册表、episode对象目录 |
| $\mathcal T$ | 独立目标、实体、明确约束和任务期限 |
| $F_t,L_{\le t}$ | 当前三态事实、实际执行历史 |
| $\mathcal A_e,\mathcal A_t,\bar m_t$ | episode实例动作、当前候选、独立执行mask |
| $G_t^K,G_t^H$ | 当前合同图、当前语义增强图 |
| $E_{prior}^e$ | 当前episode固定的软边版本 |
| $\widetilde F_t^i$ | 候选i的名义成功事实副本 |
| $Z^K,Z^H,\widetilde Z^{K,i},\widetilde Z^{H,i}$ | 共享E产生的四路表示 |
| $D_i^K,D_i^H,D_i^P$ | 合同变化、增强变化、先验交互差分 |
| $z_t^o,h_t$ | 观测编码与基础GRU状态 |
| $u_i^K,u_i^P$ | 合同候选特征、零锚定先验候选特征 |
| $b_i,\Delta_i,\ell_i$ | 合同logit、先验修正、最终logit |
| $B$ | 先验修正绝对上界 |
| $V_\omega,Q_\eta$ | PPO状态价值、辅助动作条件化价值 |
| $\Gamma_t,w_t$ | 技能时长折扣、episode起点累计折扣 |
| $\operatorname{sg}$ | stop-gradient |

# 5. Skill Contract：定义、获得与实例

## 5.1 合同不是任务计划

合同规定技能接受什么参数、哪些条件允许启动、名义成功时改变哪些事实、何时结束，以及怎样验证结果。它不规定某个具体任务必须下一步执行该技能。固定技能库层面一次定义，episode自动grounding。

来源采用“已有控制器接口＋研究人员一次性补全＋独立执行记录校验”。接口可直接提供参数和退出条件；前置与效果通过研究人员明确语义并验证。成功轨迹中常出现的条件不能直接推断为必要条件；VLM草稿不获得可信合同权限。

## 5.2 完整schema

```yaml
schema_version: skill_contract_v1
skill_type: PLACE
arguments:
  - {name: object, type: movable_object}
  - {name: target, type: openable_container}
controller_ref: REQUIRED_CONTROLLER_ID
initiation:
  required_true: ["Held(object)", "Open(target)"]
  required_false: []
  execution_checks: [binding_resolved, parameter_available, safe_to_start]
  unknown_policy: DEFER_CRITICAL_REQUIREMENTS
nominal_success_effects:
  add: ["Inside(object,target)", "GripperEmpty()"]
  delete: ["Held(object)"]
  unknown: []
  conditional_effects: []
effect_scope:
  arguments: [object, target, gripper]
  frame_assumption: unrelated_registered_facts_unchanged_nominally
verification:
  postconditions: ["Inside(object,target)", "NotHeld(object)"]
  rule_ref: REQUIRED_VERIFIER_RULE
execution:
  normal_termination: [controller_finished]
  failures: [execution_failed, verification_failed]
  interrupts: [safety_stop, critical_fact_lost]
  max_planning_seconds: null
  max_execution_seconds: null
  max_verification_seconds: null
repeat_policy: ALLOW_IF_INITIATION_CONFIRMED
provenance:
  source: controller_interface_and_review
  version: REQUIRED_VERSION
  calibration_record: REQUIRED_RECORD
```

null时限及REQUIRED字段表示当前材料没有给出的设备配置；真实执行启动器必须拒绝未绑定值。时间不得凭空填成“安全5秒”。

图只使用符号核心：参数、前置和ADD/DEL；unknown写入名义patch，不单独新增一种图节点。执行条件、时限、历史和验证规则保留于注册表及运行时接口。

## 5.3 三个示例合同

下表仅定义一个受限桌面技能集合，不是对任意机器人的能力承诺。

| Schema | 全部正前置 | ADD | DEL | 执行后验证 |
|---|---|---|---|---|
| PICK(?o) | GripperEmpty, OnTable(?o) | Held(?o) | GripperEmpty, OnTable(?o) | 确认Held；未确认不能当成功 |
| OPEN(?c) | GripperEmpty | Open(?c) | 无 | 实际开度达到固定阈值 |
| PLACE(?o,?c) | Held(?o), Open(?c) | Inside(?o,?c), GripperEmpty | Held(?o) | 区域、释放及必要稳定性证据 |

示例OPEN要求空夹爪取决于控制器；若实际控制器要求不同，合同在正式运行前按真实接口绑定。真实任务还必须有必要的放回或缓冲技能；示例五个Action节点只是构图示例，不保证任意失误都有恢复路径。恢复与PICK执行合同相同则不另造RECOVER能力。

## 5.4 条件效果与不变量

主MVP优先注册无条件ADD/DEL。若存在有限条件效果，其guard在干预前F_t上同时求值；TRUE时应用，FALSE时不应用，UNKNOWN时只保留所有可能分支共同确定的赋值，其余受影响事实置UNKNOWN。不展开概率树，不假定未知guard为FALSE。

互相冲突的同次赋值在合同校验阶段报错；单夹爪不能同时持有两个对象等不变量由固定规则检查。派生事实仅通过注册规则重算，不沿任意软边传播真假。

# 6. Contract Graph：成熟底座与自动构图

## 6.1 两类节点，三类合同语义

沿用SLG式grounded action–proposition表示。它具有Action和Proposition节点，以及PRE、ADD、DEL关系；原SLG的当前状态/goal标记在本项目中适配为三态值与goal sign。[R1]

$$
G_t^K=(\mathcal A_e\cup\mathcal P_e,
 E_{\mathrm{pre,pos}},E_{\mathrm{pre,neg}},E_{add},E_{del},X_t,\mathcal M_e).
$$

采用PRE_POS/PRE_NEG：Fact→Action；ADD/DEL：Action→Fact。为神经消息传播显式添加反向关系ID；反向消息不意味着逻辑效果反转。

不保留单独Object、Goal、普通Logic节点。对象通过grounded arguments、外部object table和观测特征表示；普通前置为合取，由规则层精确检查；目标使用命题goal_sign和固定goal_refs。

## 6.2 节点属性与稳定性

Action初始特征：类型、共享skill schema embedding、参数角色/类型。Proposition特征：共享predicate schema embedding、三态one-hot、goal sign、参数角色。不得学习任意episode对象编号作为语义。四路图的真实几何、观测质量和历史由共同观测上下文提供，不在名义后继中伪造测量。

保留FALSE/UNKNOWN事实、已完成目标和当前不可执行Action节点；只对当前真实执行候选使用mask。若提前删去不可执行技能，名义干预后“后续技能的条件改变”将无法被表示。

## 6.3 自动grounding

```python
def build_template(schemas, object_table, goals, registry):
    actions, facts, edges, contracts = {}, {}, set(), {}
    def fact_id(atom):
        registry.validate(atom)
        key = canonical_atom(atom)  # 谓词＋有序参数
        facts.setdefault(key, make_fact_node(atom))
        return key
    for schema in schemas:
        for binding in typed_bindings(schema, object_table):
            c = ground_contract(schema, binding)
            validate_contract(c)
            aid = canonical_action(c)
            actions[aid] = make_action_node(c)
            contracts[aid] = c
            for lit in c.preconditions:
                rid = 'PRE_POS' if lit.positive else 'PRE_NEG'
                edges.add((fact_id(lit.atom), aid, rid))
            for atom in c.add_effects:
                edges.add((aid, fact_id(atom), 'ADD'))
            for atom in c.delete_effects:
                edges.add((aid, fact_id(atom), 'DEL'))
            for atom in c.unknown_effects:
                fact_id(atom)  # 状态patch目标，非新增核心关系
    goal_refs = []
    for lit in goals:
        pid = fact_id(lit.atom)
        validate_goal_sign(facts[pid], lit.sign)
        facts[pid].goal_sign = lit.sign
        goal_refs.append((pid, lit.sign))
    return freeze_template(actions, facts, edges, goal_refs,
                           contracts, object_table)
```

按类型和真正静态约束ground，不按当前Fact真假或VLM建议ground。包含任意额外监测事实时也要在模板生成阶段统一注册，四路编码节点集合保持一致。

## 6.4 red / blue / box示例

目标为Inside(red,box)与Inside(blue,box)同时成立。Action实例：PICK(red)、PICK(blue)、OPEN(box)、PLACE(red,box)、PLACE(blue,box)。Fact实例：GripperEmpty、OnTable(red)、OnTable(blue)、Held(red)、Held(blue)、Open(box)、Inside(red,box)、Inside(blue,box)。一共13个节点。

| Action | PRE | ADD | DEL |
|---|---|---|---|
| PICK(red) | GripperEmpty, OnTable(red) | Held(red) | GripperEmpty, OnTable(red) |
| PICK(blue) | GripperEmpty, OnTable(blue) | Held(blue) | GripperEmpty, OnTable(blue) |
| OPEN(box) | GripperEmpty | Open(box) | 无 |
| PLACE(red,box) | Held(red), Open(box) | Inside(red,box), GripperEmpty | Held(red) |
| PLACE(blue,box) | Held(blue), Open(box) | Inside(blue,box), GripperEmpty | Held(blue) |

```text
PICK(red) --ADD--> Held(red) --PRE--> PLACE(red,box)
                                             | ADD
                                             v
                                   Inside(red,box) [goal+]

OPEN(box) --ADD--> Open(box) --PRE--> 两个PLACE实例

PICK(blue) --ADD--> Held(blue) --PRE--> PLACE(blue,box)
                                             | ADD
                                             v
                                   Inside(blue,box) [goal+]
```

图示省略的DEL和GripperEmpty边见上表，不代表它们不存在。若只给出Open=FALSE及两个Held=FALSE，其他事实先为UNKNOWN；不能仅凭任务描述将Inside初始化为FALSE。目标标记TRUE表示要求，不表示当前已经成立。

## 6.5 目标与逻辑范围

每个Fact的goal_sign取0、+1、−1。goal_refs保存固定、版本化的命题索引与要求符号。所有目标行在四路编码中保持，不按完成状态删行。

普通AND无额外节点；多个生产者ADD到同一Fact不要求全部执行。一般OR目标不能通过同时设置两个goal flag近似实现。本主版本使用合取目标；复杂任务逻辑若必须保留，需先提供语义保持编译，而不能暗中升级图本体或改变任务目标。

# 7. Frozen API VLM与先验缓存

## 7.1 主方案

冻结强API VLM，固定prompt、few-shot和输出协议，离线生成有限软关系并缓存。**不训练本地4B，不做LoRA/SFT，不做teacher–student，不加入多VLM投票。**API价格不是当前选择变量。

本次外部核对日期为2026-09-20。官方模型说明列有支持图像输入的快照`qwen3.8-max-0902`（别名`qwen3.8-max-2026-09-02`），因此保留它作为当前具体配置。[R10]这只确认公开接口存在，不表示当前账户已授权、调用已完成或生成质量已验证。账户地区、接口版本与访问权限属于运行前必填；不可用时固定一个同能力级别替代快照并重新生成整套缓存，不静默混用。

## 7.2 输入与禁止输入

| 必须提供 | 作用 |
|---|---|
| 初始RGB与图像预处理配置 | 场景语义证据；深度通过既有几何摘要使用 |
| 独立任务指令与目标 | 限制目标不能被模型重写 |
| 对象目录与绑定状态 | 明确现有ID和参数角色 |
| 已实例化Action ID、合同摘要 | 只引用真实技能能力 |
| goal ID到目标Proposition的映射 | 目标端点可定位 |
| 必要Fact ID与初始三态摘要 | 允许中间条件引用，不授权模型确认事实 |
| 三类关系schema、最多8条 | 限制输出空间；允许空集合 |
| 固定3个few-shot | 展示空输出、合法支持和不过度排序 |

禁止提供真实未来、正确完整执行序列、最优Q、测试回报、隐藏模拟器干预标记。few-shot来自独立开发材料，不从测试轨迹抽取。关掉API网页搜索、工具调用和外部检索，记录所有请求参数。

## 7.3 关系schema

仅允许SOFT_SUPPORTS、SOFT_ORDER（Action→Action）和SOFT_RELEVANT_TO_GOAL（Action→已标为目标的Proposition）。中间Fact仅为校验元数据，MVP不由它新建合同边。source由程序填写为VLM_SOFT，不接受模型自报可信度。

```json
{
  "title": "M1SoftRelations",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "relations"],
  "properties": {
    "schema_version": {"const": "m1_soft_relations_v1"},
    "relations": {
      "type": "array", "maxItems": 8,
      "items": {
        "type": "object", "additionalProperties": false,
        "required": ["relation_id", "type", "source_ref", "target_ref"],
        "properties": {
          "relation_id": {"type": "string", "minLength": 1},
          "type": {"enum": ["SOFT_SUPPORTS", "SOFT_ORDER", "SOFT_RELEVANT_TO_GOAL"]},
          "source_ref": {"type": "string", "minLength": 1},
          "target_ref": {"type": "string", "minLength": 1},
          "optional_mediating_fact_ref": {"type": ["string", "null"]}
        }
      }
    }
  }
}
```

本地validator对source/target再检查允许ID和端点类型；JSON Schema中的字符串类型不证明引用有效。goal外部ID先规范化为目标Proposition索引。

例如：

```json
{
  "schema_version": "m1_soft_relations_v1",
  "relations": [{
    "relation_id": "r1",
    "type": "SOFT_SUPPORTS",
    "source_ref": "a:MOVE:red:buffer",
    "target_ref": "a:PLACE:blue:box",
    "optional_mediating_fact_ref": null
  }]
}
```

该例的MOVE必须实际已注册；否则隔离。关系表达潜在支持，不赋予MOVE额外物理效果。

## 7.4 固定prompt与few-shot协议

系统prompt固定为：

```text
You provide bounded, imperfect semantic relations for a skill policy.
Use only supplied action, proposition and goal identifiers.
Do not create skills, facts, goals, truth values, rewards or action answers.
Return at most 8 relations; an empty list is valid.
Prefer scene-relevant relations not already entailed by the supplied contracts.
SOFT_SUPPORTS means possible support, not sufficient feasibility.
SOFT_ORDER means a suggested order, not a mandatory constraint.
Do not turn one legal ordering into the only legal ordering.
If references are unresolved or evidence is insufficient, omit the relation.
Output only the JSON object specified by m1_soft_relations_v1.
```

固定示例一：仅有合同已明确的OPEN ADD Open和Open PRE PLACE，且没有其他依据；输出空relations，说明不要为凑数量重复合同。

固定示例二：独立开发场景显示red占据blue的接近区域，真实技能有MOVE(red,buffer)；可输出MOVE到PLACE(blue,box)的SOFT_SUPPORTS，不能写“执行MOVE保证PLACE成功”。

固定示例三：两个对象独立可放置且没有支持某顺序的额外线索；不输出强制红先蓝的顺序边，允许空集合。示例图像和JSON必须在正式缓存生成前固定，其身份写入prompt hash。示例只示范关系，不提供完整任务答案。

## 7.5 输出处理与可核查API边界

```text
raw response -> JSON parser -> schema validation -> ID/type validation
-> fixed contract redundancy check -> exact dedup
-> explicit conflict isolation -> final E_prior
```

官方文档当前支持图像输入的JSON Mode，并说明多模态输入的严格json_schema约束可能降为json_object；因此本版使用JSON Object输出并始终进行本地严格验证，不把服务端格式支持当作语义保证。[R11]

工程非法关系直接隔离；格式合法但可能语义错误的关系继续作为soft prior。不人工修正测试语义边，不反复生成到自己满意。默认只对传输/语法失败做最多一次按固定错误反馈的重试，保存两次输出；第一次完全可解析输出即接受。内容错误不触发语义挑选式重试。全体无效则返回显式EMPTY_PRIOR而非伪造新关系。

## 7.6 冗余与冲突的有限定义

精确重复按类型、端点、参数和中间Fact规范化去重。若a增加p且b要求p，VLM的同义SOFT_SUPPORTS及同一中间p可判为合同已有；不能因此把绝对SOFT_ORDER也视为等价。“某技能可能提供条件”不等于“总要先执行它”。只使用冻结的局部规则，不求任意逻辑等价。

不存在的端点、类型错误、明确违反独立任务禁止项的关系隔离；合同未写出的关系不因未写出就判FALSE。软顺序相反或出现环不自动删光，除非与明确任务硬顺序直接冲突。关系处理日志保留原因。

## 7.7 缓存与复现

缓存键为以下规范化内容的SHA-256：task定义、初始RGB原始内容/预处理结果hash、对象绑定、允许ID、合同/谓词版本、模型快照、prompt与few-shot版本、schema、解码配置、API地区与接口版本。原始响应、解析关系、隔离项、去重日志、request ID、生成日期和所有hash均保存。

```text
relations_cache/{split}/{cache_key}/
  manifest.json
  prompt.txt
  input_refs.json
  raw_response.json
  parsed_relations.json
  rejected_relations.json
  processing_log.json
```

先在开发集固定生成协议，训练和测试缓存严格分开。测试初始观测上的一次冻结生成是推理，不是允许用测试成绩重新选prompt。有限初始化池提前全部缓存；新场景第一次生成后冻结。不能按task名称给不同初始场景复用缓存。

缓存后PPO不调用API。数据、代码、环境资产、随机种子和缓存完整公开时，下游实验不需访问原API；这不保证GPU逐bit一致，也不保证将来还能从闭源模型重新生成相同raw response。公开API输入图像须满足数据使用权限，不保存或分享API密钥。

# 8. Candidate Nominal Intervention

## 8.1 它是什么

对每个当前允许执行的候选a_i，以固定技能合同构造“如果按名义成功分支结束”的事实赋值：

$$
\widetilde F_t^i=T_{\mathcal K}^{nom}(F_t,a_i).
$$

这是一部给定的符号名义转移模型，不是生成式世界模型rollout，不使用物理模拟、真实未来状态或成功概率。本文不把它称为完全model-free推理；它是**利用给定符号能力模型进行的一步条件计算**，真实交互学习仍为PPO。

一个技能实际可能失败。名义分支仅是候选特征来源；实际失败、隐藏副作用和执行时长均在真实转移中进入回报监督。

## 8.2 干预范围

ADD将相应临时命题设为TRUE；DEL设为FALSE；unknown effects设为UNKNOWN。对未影响命题采用合同声明的名义frame assumption。不复制新的相机观测，不把假设的Held=TRUE写为“传感器确认”。

技能名义效果不增加attempt count、execution ID、证据置信度或实际时钟，不改变目标、VLM边和当前候选mask，不生成done/reward。几何位置继续由真实观测上下文提供，临时图不虚构移动后的3D坐标。

## 8.3 干预算法

```python
def nominal_patch(facts, candidate, contract, registry):
    status = evaluate_three_valued_preconditions(facts, contract)
    if status != TRUE:
        return InvalidBranch('required_precondition_not_confirmed')

    # 所有guard都在原始facts上求值，不能顺序污染。
    effects = resolve_registered_effects(contract, facts)
    patch = merge_assignments_or_raise(
        true_atoms=effects.add,
        false_atoms=effects.delete,
        unknown_atoms=effects.unknown,
    )
    virtual = ReadOnlyOverlay(facts, patch)
    virtual = recompute_registered_derived_facts(virtual)
    registry.check_invariants(virtual)
    return FrozenPatch(diff(facts, virtual))
```

如果真实启动mask和逻辑检查不一致，应记录接口错误，而不是把该动作事后免费换成另一动作。安全观察/等待必须有已注册的有界执行接口，名义逻辑patch可以为空；它的实际价值仍可通过当前状态、技能身份和真实回报学习。

## 8.4 UNKNOWN与复杂效果

必须满足的正文字项要求TRUE，负文字项要求FALSE；UNKNOWN不能满足任何一方。观察不足不等于物理绝对不可行，而是暂时不能获得启动许可。非关键未知不禁止无关动作。

有限条件效果按第5章规则处理。一般不确定效果不擅自选最有利分支，不建立概率树；只有所有成功结果共同确定的效果可写定值，其余为UNKNOWN。未支持的条件语法在合同注册时报错。

## 8.5 同一个patch用于两类图

$$
\widetilde G_t^{K,i}=\operatorname{View}(G_t^K,\widetilde F_t^i),\qquad
\widetilde G_t^{H,i}=\operatorname{View}(G_t^H,\widetilde F_t^i).
$$

四个图具有同一节点、目标索引和当前观测来源。唯一对照因素是“原始/名义事实”和“无/有额外软边”。真实环境不会被克隆执行K次；只复制或覆盖逻辑特征。

# 9. 增强图、共享编码与双差分

## 9.1 相同节点，只增加软边

$$
V_t^H=V_t^K,\qquad E_t^H=E_t^K\cup E_{prior}^e.
$$

SOFT_SUPPORTS与SOFT_ORDER为Action→Action；SOFT_RELEVANT_TO_GOAL为Action→目标Proposition。软边有独立关系ID及反向消息边，绝不混入PRE/ADD/DEL或名义转移程序。加入Action→Action后$G^H$不再严格为二部图，但仍共享同一节点集合和R-GCN。

## 9.2 节点初始化与schema共享

$$
h_a^{(0)}=e_{action}+e_{\sigma(a)}+f_A(\operatorname{argtypes}(a)),
$$

$$
h_p^{(0)}=e_{prop}+e_{\rho(p)}+
 f_P(\operatorname{onehot}(\nu_t(p)),\operatorname{goalsign}(p),
 \operatorname{argtypes}(p)).
$$

同一PICK schema的所有实例共享embedding与投影；同一Held predicate的实例也共享。参数角色按固定顺序存储。实例身份通过grounded绑定、连接和候选对应体现，而非用任意ID学习表。额外的对象外观/几何由观测与候选编码提供。

这继承ASNet的schema参数共享原则，但不声称完整复现ASNet的专用交替网络。[R2]主版本不使用冻结文本embedding或LLM teacher。

## 9.3 标准R-GCN

$$
h_v^{(l+1)}=\sigma\!\left(W_0^{(l)}h_v^{(l)}+
 \sum_{r\in\mathcal R}\sum_{u\in\mathcal N_r(v)}
 \frac{1}{|\mathcal N_r(v)|}W_r^{(l)}h_u^{(l)}\right).
$$

使用现成RGCNConv，root transform、关系权重与basis decomposition沿用成熟实现。[R3–R4]建议起步：128维、4层、mean aggregation、4个basis、ReLU、逐节点LayerNorm，Dropout=0。LayerNorm是标准稳定化配置，不增加新的消息机制。四路同用一个E_ψ，不按K/H或before/after分别训练网络、BatchNorm统计或分支embedding。

没有邻居的关系贡献为零。所有关系在模型初始化时登记，当前缺边不改变参数结构。当前不可执行的技能仍参与结构传播。GNN不是AND真值求解器；精确可执行性由合同程序给出。

## 9.4 Goal-aligned readout

令m为本episode固定原子目标数，目标Proposition及要求符号为(p_j,s_j)。

$$
z_j=f_g(h_{p_j}^{(L)},e_{sign}(s_j)),\quad j=1,\ldots,m,
$$

$$
z_0=f_0\!\left(\operatorname{Mean}(H_A),
 \operatorname{Mean}(H_P),\operatorname{Mean}(\{z_j\})\right),
$$

$$
\boxed{E_\psi(G\mid\mathcal T)=Z=[z_0;z_1;\ldots;z_m]
 \in\mathbb R^{(m+1)\times d}.}
$$

z_0是整体读出而非新节点。goal_refs在episode内固定。跨任务m可变时使用padding及goal mask；候选读取按集合处理目标行，而不按位置学习固定“红先蓝”权重。

E不是价值、成功概率或剩余步数。图编码器具有有限传播半径，不宣称4层已经完成任意长逻辑证明；整体任务价值仍由真实序贯学习估计。

## 9.5 双差分的正式定义

$$
Z_t^K=E_\psi(G_t^K),\quad Z_t^H=E_\psi(G_t^H),
$$

$$
\widetilde Z_t^{K,i}=E_\psi(\widetilde G_t^{K,i}),\quad
\widetilde Z_t^{H,i}=E_\psi(\widetilde G_t^{H,i}),
$$

$$
\boxed{D_{t,i}^K=\widetilde Z_t^{K,i}-Z_t^K,}
$$

$$
D_{t,i}^H=\widetilde Z_t^{H,i}-Z_t^H,
$$

$$
\boxed{D_{t,i}^P=D_{t,i}^H-D_{t,i}^K.}
$$

$D^K$表示合同结构中对候选名义事实变化的编码响应。$D^H$表示额外软关系参与后对同一变化的响应。$D^P$表示软关系对该响应造成的交互差异。**它不等于VLM真实因果贡献、正确概率、额外成功率或价值提升。**

## 9.6 为什么比较两次

以OPEN为例，第一层after-before问“Open由FALSE变TRUE后，目标对齐表示怎样变化”；第二层enhanced-contract问“加入语义关系后，这个动作引起的变化被怎样不同地组织”。重点不是整张$G^H$总体是否更好，而是先验怎样影响对某个候选变化的解释。

若合同已表达OPEN通过Open支持两个PLACE，这一共享前置属于$D^K$，不冒充VLM新知识。经检查保留的例如MOVE(red,buffer)支持PLACE(blue,box)的软边，可以改变编码响应，但不在名义模型中保证蓝色一定放好。

## 9.7 差分性质与边界

**零先验。**若所有软边为空或被去重，直接复用$Z^K$及合同后继编码，使$G^H$=$G^K$、$D^P$=0。相同输入复用避免随机数和数值路径造成虚假差分。

**空patch。**如果技能没有改变参与编码的逻辑输入，且没有伪造时间、来源或证据字段，则$D^K$=$D^H$=$D^P$=0；WAIT仍可通过观测和候选身份得到非零动作分数。

**共同坐标。**同encoder、节点、goal顺序与无随机Dropout为相减提供一致坐标基础；不赋予每个维度人工语义。共同仿射变换Z'=AZ+c下，差分变为AD，常数抵消；任意非线性重参数化不保持差分几何。

**交互范围。**写为E(F,R)时：

$$
D_i^P=E(\widetilde F^i,R)-E(F,R)
      -E(\widetilde F^i,\varnothing)+E(F,\varnothing).
$$

若E(F,R)=A(F)+B(R)可分离，则$D^P$=0。因此该算子刻画事实干预与软结构的交互，并非无损保留所有静态先验偏好。以上是代数性质，不是行为有效性结论。

# 10. Policy Architecture

## 10.1 观测与候选上下文

冻结视觉前端提供实际测量与特征；可训练观测融合器和GRU编码：

$$
(z_t^o,h_t)=f_{obs,\phi}(x_t,h_{t-1}).
$$

GRU读取实际观测、独立目标、核心事实和执行摘要，不直接读取VLM图embedding。候选e(a_i)包含共享技能embedding、有序对象/目标角色特征、当前合同状态和实际可获得的几何信息，不含真实未来或正确下一动作。

$$
c_{t,i}=[z_t^o,e(a_i),\operatorname{Pool}(Z_t^K)].
$$

保留当前$Z^K$，避免只给变化量而丢掉绝对任务上下文。

## 10.2 合同推理分支

F_K对目标对齐的$D_i^K$进行共享候选查询和池化，再与c_i融合：

$$
u_{t,i}^K=F_{\zeta_K}(c_{t,i},D_{t,i}^K),\qquad
b_{t,i}=w_K^\top u_{t,i}^K+b_K.
$$

这里的base正式命名为**Contract-Reasoning Branch**，不再称observation-only。候选attention采用标准机制，空padding用mask；它不是新的价值传播算子。

## 10.3 零锚定先验分支

$$
\boxed{u_{t,i}^P=
F_{\zeta_P}(c_{t,i},u_{t,i}^K,D_{t,i}^P)
-F_{\zeta_P}(c_{t,i},u_{t,i}^K,0).}
$$

两个F_P共用参数、上下文和确定计算；零输入使用同一goal mask。随后：

$$
\boxed{\Delta_{t,i}=B\tanh(w_P^\top u_{t,i}^P),\qquad
\ell_{t,i}=b_{t,i}+\Delta_{t,i}.}
$$

w_P之后不加独立bias，不增加无界外部倍率，不重新引入固定prior bias。$D^P$=0时u^P和Δ严格为0。合同分支不受该先验幅度上界限制，但合同也不是完美现实模型。

## 10.4 候选和执行mask

$$
\mathcal A_t=\operatorname{Instantiate}(\mathcal U,\mathcal O_t,\mathcal T),
\qquad
\bar m_t=\operatorname{Check}(\mathcal A_t,\mathcal K,F_t,\mathcal S).
$$

mask只依据实际能力、参数绑定、固定合同与安全；VLM软顺序不影响候选全集和mask。图内没有某动作不禁止选择。UNKNOWN关键启动条件要求确认；其他无关UNKNOWN不阻塞全部技能。OBSERVE_WAIT只在可安全保持时可用，有非零时长。

$$
\boxed{\pi_\theta(a_i\mid X_t)=
\frac{\bar m_{t,i}\exp(\ell_{t,i})}
{\sum_j\bar m_{t,j}\exp(\ell_{t,j})}.}
$$

实际分布用masked log_softmax实现，禁止用零乘无穷得到NaN。所有动作都不允许时不生成非法Categorical，转独立安全终止规则。对被mask的动作不必计算名义视图。

## 10.5 有界影响的准确保证

同参数、同输入、同候选、同mask下关闭先验修正得到π_K。由于$|\Delta_i|\le B$：

$$
 e^{-2B}\le\frac{\pi_\theta(a_i\mid X_t)}{\pi_K(a_i\mid X_t)}\le e^{2B}.
$$

概率界来自exp(Δ_i)与其概率加权归一化之比，只对允许动作成立。不保证argmax不变、不保证整个episode回报不退化、不约束图通路对共享参数和训练轨迹的全部影响。基础logit差超过2B时不能由该残差单独翻转；小差距可被小残差翻转。

主版本固定全局B，0.5只是实现起步值。B校准和bounded/unbounded消融属于并行实验，不是开始写Method的先决条件。

# 11. Structural Q、PPO Critic与总训练目标

## 11.1 Actor、V、Q职责

Actor的ℓ_i是选择分数，不是Q。V估计当前状态下继续采用当前策略的回报，用作GAE基线。Structural Q估计先选择指定技能，再跟随当前策略的期望回报，是**辅助动作条件化价值监督**。

V与Q共享观测、图及候选特征，输出head独立。不同图快照、先验版本和执行时钟属于可观测算法状态X_t的一部分。Q不以成功分支为条件；它必须平均真实技能失败与成功的后果。

## 11.2 两个head

$$
\bar u_t^K=\operatorname{MaskedMean}_i u_{t,i}^K,
\quad \bar u_t^P=\operatorname{MaskedMean}_i u_{t,i}^P,
$$

$$
V_\omega(X_t)=f_V(z_t^o,\operatorname{Pool}(Z_t^K),
\bar u_t^K,\bar u_t^P),
$$

$$
Q_\eta(X_t,a_i)=f_Q(u_{t,i}^K,u_{t,i}^P,z_t^o,
\bar u_t^K,\bar u_t^P).
$$

池化只覆盖真实有效候选，padding不计入均值。Q读取候选集合上下文，避免WAIT零patch时丢失后续选择背景。主实现不使用V=ΣπQ，因未执行候选Q可能不可靠；V仍独立训练。

## 11.3 环境奖励和目标

$$
\rho_\tau^{env}=\mathbf1\{\text{此刻首次独立确认整项任务成功}\},
$$

$$
r_t=\sum_{k=0}^{d_t-1}\bar\gamma^k\rho_{\tau_t+k}^{env},
\qquad\Gamma_t=\bar\gamma^{d_t},
$$

$$
J(\theta)=\mathbb E\!\left[\sum_t w_t r_t\right],
\quad w_0=1,\quad w_{t+1}=w_t\Gamma_t.
$$

只在最终成功给+1；技能成功、子目标、恢复、失败和图进展不另行给分。任务超时为真实终止，回报0；失败尝试消耗真实时间并影响成功机会。目标是时间折扣下任务成功，不等同完全不计时的成功概率。

## 11.4 结束标志与GAE

令b_t=1−terminated_t；c_t只在下一条记录是同一episode连续转移时为1。使用采样模型的旧V：[R6–R8]

$$
\delta_t=r_t+\Gamma_t b_tV_{old}(X_{t+1})-V_{old}(X_t),
$$

$$
\widehat A_t=\delta_t+\Gamma_t\lambda c_t\widehat A_{t+1},
\qquad y_t^V=\operatorname{sg}[V_{old}(X_t)+\widehat A_t].
$$

λ按高层决策衰减，Γ按实际时间衰减。真实终止无bootstrap；buffer切段和有效外部截断使用最终有效状态值，不跨reset递推。优势标准化若使用，只作用于actor优势，不修改V/Q目标；配置需记录。

## 11.5 Structural Q目标

$$
\boxed{y_t^Q=\operatorname{sg}
[r_t+\Gamma_t b_tV_{old}(X_{t+1})].}
$$

$$
\boxed{\mathcal L_Q=\mathbb E_t
[\operatorname{Huber}(Q_\eta(X_t,a_t)-y_t^Q)].}
$$

只对实际a_t取Q输出计算损失。其他候选不填0，不使用虚拟后继V作标签，不用VLM打分，不做max-Q或DQN目标。目标是实际转移支撑的bootstrap估计，并非精确长期价值。所有真实失败转移也纳入监督。

Q误差经共享特征反传到R-GCN、候选读出和观测编码。它提供“如何解释此类结构与回报关系”的学习压力，不直接监督D的某一维，也不保证Actor一定使用D。网络更新不会给具体缓存边永久写confidence，推理期不学习per-edge trust memory。

## 11.6 PPO和总损失

$$
\varrho_t(\theta)=
\frac{\pi_\theta(a_t\mid X_t,h_{t-1}^{\theta})}
{\pi_{old}(a_t\mid X_t,h_{t-1}^{old})},
$$

$$
L_{clip}=\mathbb E_{w_t}\!\left[
\min\{\varrho_t\widehat A_t,
\operatorname{clip}(\varrho_t,1-\epsilon,1+\epsilon)\widehat A_t\}\right],
$$

$$
\mathcal L_V=\mathbb E_t[\operatorname{Huber}(V_\omega(X_t)-y_t^V)],
$$

$$
\boxed{\mathcal L_{total}=-L_{clip}
+c_V\mathcal L_V+\lambda_Q\mathcal L_Q-c_H\mathbb E_t[H(\pi_\theta)].}
$$

对当前声明的episode起点折扣目标，访问权重w_t与TD中的Γ_t含义不同，不是重复折扣。[R8]默认实现加权策略surrogate；V/Q以采样transition均值回归，entropy也按声明的采样均值作为正则，不声称总体梯度精确等于原始J梯度。采样已经按w重采样时，不再次重复加权。PPO裁剪、有限GAE和辅助损失均是明确的算法近似/正则。

**不新增teacher、contrastive、semantic consistency、graph reward或人工shaping损失。**Q辅助项不改环境奖励，但确实改共享表示的优化目标，不能称总梯度仍是纯PPO梯度。

## 11.7 梯度与优化器

| 参数 | Actor/Entropy | V loss | Q loss |
|---|---:|---:|---:|
| 观测融合与GRU φ | 是 | 是 | 是 |
| 四路共享R-GCN ψ | 是 | 是 | 是 |
| 合同候选特征 ζ_K | 是 | 是 | 是 |
| 零锚定先验特征 ζ_P | 是 | 是 | 是 |
| Actor最终head | 是 | 否 | 否 |
| V head ω | 否 | 是 | 否 |
| Q head η | 否 | 否 | 是 |
| 旧V与固定target | 否 | 否 | 否 |
| 合同、图构造、干预、Verifier、VLM | 否 | 否 | 否 |

```text
Actor loss -> Actor head -> uK/uP -> Obs Encoder 与四路Graph Encoder
V loss     -> V head     -> 同一共享候选/状态特征
Q loss     -> Q head     -> 实际a_t的共享特征 -> 共享编码器
固定yV/yQ  -> stop-gradient
固定符号程序 -> 不可学习，但其输出送入可微编码器
```

四个E分支均允许梯度，不能只对after分支训练而随意detach before。一个参数所有者管理共享层；一个去重参数优化器，统一前向、总loss、backward、裁剪和step。不用三个含重复共享参数的优化器反复更新。c_V=0.5、λ_Q=0.1为开发初值，不是已证明最佳。

# 12. Fact Verification与运行时更新

## 12.1 固定事实链

```text
第三视角RGB-D + proprioception + execution log
-> frozen perception/tracking
-> geometric/temporal predicate verification
-> TRUE / FALSE / UNKNOWN records
-> deterministic Fact Store and Graph Snapshot
```

事实判定器是基础设施。模型软边不作为真值证据；执行命令不直接产生TRUE；控制器正常结束不等于世界事实成立。

## 12.2 事实定义与证据

| 谓词 | 主要依据 | 重要边界 |
|---|---|---|
| Held(A) | 夹爪实际状态、对象相对位置、随动和持有历史 | 闭夹爪或对象变高不单独构成充分证据 |
| Inside(A,B) | 对象估计几何与目标区域 | 靠近边界、遮挡或身份不确定时UNKNOWN |
| StableIn(A,B) | 有效时间窗内的空间稳定 | 只确认观察窗，不保证未来永不失效 |
| Open(B) | 实际开度/姿态 | 半开不自动等于Opened或Closed |
| Observed(A) | 当前可用对象测量 | 未检测到不证明物理上不存在 |
| Active/Ended | 真实execution ID与日志 | 不需要VLM判断 |
| LossDetected(A,epoch) | 本次可靠持有后非主动丢失证据 | 主动释放与旧持有周期不误触发 |

支持Placed等复合谓词时按已注册三值逻辑推导；纯RGB-D不将几何邻近冒充真实接触。

## 12.3 当前值与历史值

```yaml
fact_id: Held:red
value: UNKNOWN
reason: OCCLUDED
last_confirmed_value: TRUE
last_confirmed_time: measured_timestamp
capture_time: measured_timestamp
available_time: measured_timestamp
evidence_quality: null
p_true: null
evidence_ids: [frame_id, robot_state_id]
hold_epoch: current_epoch
```

confidence默认是测量质量，不伪装为已校准事实概率。证据过期、冲突或缺失时当前为UNKNOWN，历史仍保留。可靠反证经短时确认后允许TRUE→FALSE。负前置要求可靠FALSE，UNKNOWN的否定仍UNKNOWN。

## 12.4 更新节奏与安全边界

底层安全独立高频运行；关键事实按感知周期更新；高层在技能结束/中断并确认后提交快照。10Hz、约0.2秒多帧窗口只是慢速任务起步配置，需硬件标定，不能作为安全保证。多帧必须有不同frame ID及实际覆盖时间，不能重复处理旧帧累计证据。

固定相机遮挡可能导致长期UNKNOWN。OBSERVE_WAIT只是有界等待新帧，不隐含主动腕部相机；若持续无法确认关键条件，记录该运行失败或传感器不可用原因，不解除mask赌执行。需要额外视角时属于所有方法共享的新观测配置，不能私下加入主方法。

## 12.5 执行结果三分

| 字段 | 例子 |
|---|---|
| controller_exit | NORMAL / TIMEOUT / REJECTED / INTERRUPTED / SAFETY_ABORT |
| verified_outcome | VERIFIED_SUCCESS / VERIFIED_FAILURE / UNCONFIRMED |
| world_fact | Held(A)=T/F/U |

TIMEOUT与Held=TRUE可同时成立。正常退出但证据不足是UNCONFIRMED。失败可能推动物体，不能恢复执行前图。重试只由当前合同与真实条件决定，不按一次失败永久mask。开始恢复、改变阶段标签、计数增加均不产生奖励。

## 12.6 时间和数据一致性

观测有采集与到达两个时间。决策只使用已到达证据；迟到数据不能回写历史PPO输入。图、候选、mask和frame引用共享snapshot ID。并行环境分别维护跟踪、事实、持有周期和GRU状态。

真实任务终止包括成功、预先定义的不可恢复失败、安全终止和任务期限用尽；单技能超时通常不是episode终止。外部截断及buffer边界要bootstrap到最后有效状态，不用reset后状态。[R9]无有效末状态时标为基础设施异常，不补造数据、不只保留成功试验。

# 13. Training Algorithm

## 13.1 阶段划分

准备阶段只做接口校准和实现验收：冻结技能合同、观测/Verifier、独立奖励、缓存规则与任务清单。没有Graph预训练、Q预训练、VLM微调、policy分阶段冻结或复杂课程。

学习阶段一阶段联合训练Obs Encoder、GRU、R-GCN、候选特征、Actor、V和Q。只用实际on-policy交互；有限先验扰动按episode固定。评价阶段全部模型权重冻结，状态、跟踪、事实与GRU仍随运行更新。

终端奖励的探索可行性需要真实数据判断；Q head不凭空制造成功监督。若训练长时间全零回报，先如实记录实现或探索问题，而不是偷偷加奖励或删除失败seed。任何新增课程或暖启动要作为后续明确版本/共同训练协议变更，不是当前隐藏默认。

## 13.2 统一前向算法

```text
M1_FORWARD(snapshot, theta, hidden, need_value, need_q):
    z_obs, hidden_next = ObsEncoder(snapshot.base_input, hidden)
    GK = contract_view(snapshot.template, snapshot.real_facts)
    GH = add_soft_edges(GK, snapshot.episode_prior)
    ZK = E(GK, snapshot.goal_refs)
    ZH = E(GH, snapshot.goal_refs)  # 空prior直接复用ZK

    for each allowed candidate i:
        patch_i = nominal_patch(snapshot.real_facts, candidate_i,
                                frozen_contract_i)
        # 相同patch；仅事实逻辑不同，不伪造物理观测。
        GK_i = readonly_view(GK, patch_i)
        GH_i = readonly_view(GH, patch_i)
    batch_encode valid GK_i and GH_i with the same E

    for each allowed candidate i:
        DK_i = E(GK_i) - ZK
        DH_i = E(GH_i) - ZH
        DP_i = DH_i - DK_i
        context_i = [z_obs, candidate_features_i, pool(ZK)]
        uK_i = F_K(context_i, DK_i)
        uP_i = F_P(context_i, uK_i, DP_i)
               - F_P(context_i, uK_i, zero_like(DP_i))
        logits_i = actorK(uK_i) + B * tanh(actorP_no_bias(uP_i))

    dist = masked_categorical(logits, snapshot.mask)
    V = value_head(shared_features) if need_value else None
    Q = q_head(shared_candidate_features) if need_q else None
    return dist, V, Q, hidden_next
```

forward必须是纯函数式状态转换；用于计算next V的调用不得重复推进已提交GRU。相同观测快照的隐藏状态前缀按一次真实决策序列管理。

## 13.3 完整训练伪代码

```text
输入：任务训练split、冻结技能/合同/Verifier、关系缓存、配置
初始化一个包含全部可学习参数的模型与优化器
初始化各环境独立的事实、跟踪、日志与序列缓存

while 尚有统一真实交互预算：
    theta_old = 当前模型冻结快照
    buffer = empty

    while 未到本轮采样边界：
        对每个处于高层决策点的环境：
            if 新episode：
                reset环境与该环境的运行时缓存
                绑定对象和任务；构造合同模板与goal_refs
                按初始输入读取已生成relation cache
                选择本episode的原始/空/删除/改写软边版本
                hidden = zero; episode_discount = 1

            获取当前实际已到达观测
            perception -> 当前测量
            verifier -> 当前真实三态facts
            从固定技能库ground候选并求独立执行mask
            形成原始snapshot，保留任务/图/候选/观测版本

            if 没有安全可用动作：
                按独立安全结束合同处理；不放开全部mask
                记录原因并结束该episode
                continue

            with no_grad：
                dist_old, oldV, _, next_hidden =
                    M1_FORWARD(snapshot, theta_old, hidden, True, False)
                action = sample(dist_old)
                oldlogp = dist_old.log_prob(action)
            只在本次真实决策提交next_hidden

            执行固定skill并做关键监测/有界结束确认
            收集真实奖励序列、持续时间d、结束原因
            verifier提交实际next facts
            独立评价器给出terminated/truncated和成功
            构造next_snapshot，不替换为reset后的状态
            Gamma = gamma_base ** d
            reward = duration_discounted_env_rewards

            with no_grad：
                oldVnext = 0 if terminated else
                    M1_FORWARD(next_snapshot, theta_old,
                               next_hidden, True, False).V
            保存snapshot、action、oldlogp、oldV、oldVnext、
                 reward、d、Gamma、episode_discount、next_snapshot、
                 terminated、truncated、序列边界
            episode_discount *= Gamma
            追加实际执行日志，不用命令直接写Fact=True

    按oldV、Gamma和真实边界计算GAE、yV、yQ；目标全部detach
    在本轮更新期间固定targets、候选、mask和先验版本

    for epoch in PPO_epochs：
        for continuous_minibatch in buffer：
            以当前参数重算episode前缀GRU状态；前缀可no_grad
            当前片段内截断BPTT，不随机打散转移
            重建原始图和名义patch，重算四路E和差分
            得到dist_new、V_new、Q_new
            actor_loss = clipped_surrogate(stored_action, oldlogp,
                                          dist_new, advantage,
                                          episode_discount)
            value_loss = Huber(V_new - stored_yV)
            q_selected = gather(Q_new, stored_action_index)
            q_loss = Huber(q_selected - stored_yQ)
            total = actor_loss + cV*value_loss + lambdaQ*q_loss
                    - cH*entropy(dist_new)
            optimizer.zero_grad()
            total.backward()
            clip_grad_norm(unique_shared_parameters)
            optimizer.step()

    丢弃已用on-policy buffer，不伪造任务结束
    对仍在同一episode运行的环境重建当前参数下GRU状态
    不重置实际事实、对象跟踪、任务时钟和先验版本
```

旧模型仅为每批PPO采样/目标快照，不新增EMA target网络或replay训练系统。实际Q目标在整轮更新固定，未选动作无Q标签。

## 13.4 Buffer与缓存权限

| 保存内容 | 跨优化更新能否复用 |
|---|---|
| 合同模板、节点索引、goal_refs | 可以，需版本一致 |
| 原始事实、观测引用、执行摘要 | 可以，是当时输入 |
| 冻结视觉特征 | 可以，前端权重与预处理固定 |
| 软边及episode扰动seed/结果 | 必须保存，不重新采样 |
| 名义patch | 可缓存，合同和原事实相同才能复用 |
| 候选ID、mask、参数角色 | 必须保存 |
| oldlogp、oldV、oldVnext、r、d、Γ、w | 必须保存 |
| Z、$D^K$、$D^P$、可学习候选特征 | 不能作为新模型固定输入，必须重算 |
| RNN边界与必要前缀 | 必须保存，用当前参数重建 |

# 14. Inference Algorithm

```text
输入：独立任务；冻结M1；技能/合同/感知/事实接口
采集允许的初始场景，绑定对象和任务
读取精确匹配cache；新场景只初始化调用一次冻结API并缓存
实例化合同图和固定goal_refs
不施加训练用人工扰动
初始化事实、跟踪、历史和GRU

while 未真正终止：
    当前观测 -> 冻结感知 -> Verifier facts
    从技能库生成候选与独立mask
    snapshot = G^K/G^H的共同节点、当前facts、固定先验
    dist = M1_FORWARD(snapshot, frozen_model,
                      hidden, need_value=False, need_q=False).dist
    按冻结评价规则选择技能（主评价argmax）
    固定低层执行器执行，必要时由监测/安全中断
    获得新实际观测，做有界后置确认
    更新facts、历史、任务时间
    独立评价器检查成功或终止

输出：结果、全部执行、事实、先验及决策日志
```

V/Q输出head非动作选择必需；共享结构特征仍需计算。推理不更新权重、不在线微调关系、不为失败重新生成更满意的先验。原始先验固定但当前事实可回退，因此同一关系在不同状态中得到不同编码响应。

# 15. Implementation Details与Agent执行规范

## 15.1 模块职责

| 模块 | 是否训练／loss | 数据 | 推理运行 |
|---|---|---|---|
| Frozen API VLM | 不训练 | 初始化允许图像与任务 | 仅新输入一次或读cache |
| Parser/Validator/Dedup | 固定程序 | raw response与注册接口 | 初始化 |
| Skill Contract | 接口整理、独立校验后冻结 | 控制器说明与校验记录 | 配置读取 |
| Contract Graph Builder | 固定程序 | 合同、实体、目标 | 初始化与快照绑定 |
| Fact Verifier | 感知/规则独立校准后冻结 | 真实传感器与日志 | 是 |
| Nominal Intervention | 固定程序，无梯度 | 当前事实与合同 | 每候选 |
| Standard R-GCN | Actor/V/Q联合 | 四路结构视图 | 是 |
| Observation Encoder/GRU | Actor/V/Q联合 | 观测、任务、核心事实、历史 | 是 |
| Candidate Encoder F_K/F_P | Actor/V/Q联合 | 结构差分、对象角色、状态 | 是 |
| Actor/Prior Head | Actor/Entropy | 真实PPO转移 | 是 |
| V Critic | V回归 | 真实GAE target | 非必需 |
| Structural Q | 实际a_t的Q回归 | 真实TD target | 非必需 |
| Low-level Controller | 冻结 | 实际状态与目标参数 | 是 |
| Task Evaluator | 固定、独立 | 真实任务结果 | 是 |

## 15.2 数据契约

```text
TaskSpec:
  task_id, split, instruction, typed_entities, signed_goal_atoms,
  explicit_constraints, deadline, evaluator_version
SkillSpec:
  schema_name, typed_arguments, pre_pos, pre_neg,
  add, delete, unknown, conditional_effects,
  effect_scope, checks, controller_ref, verification_ref,
  termination_rules, time_limits, version, provenance
FactRecord:
  atom_id, value, evidence_quality, reason,
  capture_time, available_time, last_confirmed_value,
  last_confirmed_time, evidence_ids, hold_epoch, verifier_version
GraphTemplate:
  node_ids, node_types, action_schema_ids, predicate_schema_ids,
  ordered_argument_refs, contract_edge_index, relation_ids,
  goal_refs, grounded_contracts, object_table_ref, version
PriorCache:
  key, model_snapshot, prompt_hash, schema_hash, input_hashes,
  raw_response, parsed_relations, rejected, dedup_log, final_soft_edges
Snapshot:
  env_id, episode_id, time, template_ref, fact_values,
  observation_ref, execution_summary, prior_variant_ref,
  candidate_ids, candidate_features, execution_mask, goal_mask
Transition:
  snapshot_ref, next_snapshot_ref, selected_candidate_id,
  old_logp, old_V, old_V_next, reward, duration, Gamma, w,
  terminated, truncated, reason, recurrent_sequence_refs
```

候选ID为技能类型＋有序参数＋必要参数生成版本，不把当前索引“第3项”当跨时间动作语义。名义视图与真实Fact Store使用不同不可变类型，任务评价器schema不得含D或prior字段。

## 15.3 建议目录

```text
cp_disr/
  contracts/
    schema.py          # 合同与类型
    registry.py        # 技能/谓词绑定
    invariants.py      # 固定逻辑与单夹爪约束
    ground.py          # episode实例化
  graph/
    slg.py             # 两类节点，PRE/ADD/DEL
    snapshots.py       # 不可变图视图
    intervention.py    # 只读patch
    derived.py         # 注册三值派生规则
    rgcn.py            # 标准算子包装
    readout.py         # goal_refs对齐
    differences.py     # DK/DH/DP与空先验快捷路径
    batching.py        # 双拓扑，多候选视图
  vlm/
    prompt.txt
    relation_schema.json
    client.py          # 固定API快照
    parse_validate.py
    dedup.py
    cache.py
    corruption.py      # 只改软边
  perception/
    adapter.py
    fact_verifier.py
    fact_store.py
  skills/
    candidates.py
    mask.py
    executor.py
    controllers/       # 必须绑定真实实现
  policy/
    observation.py
    candidate_features.py
    actor.py
    critic.py
    structural_q.py
    model.py           # 唯一共享参数所有者
  rl/
    collector.py
    rollout.py
    targets.py
    losses.py
    recurrent.py
    ppo.py
  env/
    adapter.py
    task_evaluator.py
  evaluation/
    protocols.py
    metrics.py
    actor_interventions.py
    reporting.py
  configs/
    method.yaml
    run_manifest.yaml
    splits.json
  tests/
    test_grounding.py
    test_no_mutation.py
    test_zero_prior.py
    test_goal_alignment.py
    test_masks_and_ids.py
    test_gradients.py
    test_smdp_boundaries.py
```

## 15.4 起步配置与运行前必填

以下“默认”来自既定设计或本轮为实施补齐，未被标为最优。

| 参数 | 起步配置 | 固定时机 |
|---|---|---|
| VLM | qwen3.8-max-0902，非思考JSON模式 | 缓存生成前核查权限 |
| 每输入关系预算 | 8，允许0 | prompt固定时 |
| VLM temperature | 0或服务greedy等价设置 | cache manifest |
| 输出token上限 | 2048，截断判无效而非补造尾部 | cache manifest |
| 格式重试 | 最多1次；只修语法/传输，记录全部 | 生成协议 |
| 图宽度/层数 | 128 / 4 | 开发集确认后 |
| R-GCN | mean，root=True，basis=4 | 模型配置 |
| 图/零锚定Dropout | 0 | 冻结 |
| GRU宽度 | 128 | 开发配置 |
| B | 0.5 | 开发集确认后，不测试时调 |
| lr / clip / λ_GAE | 3e-4 / 0.2 / 0.95 | 开发配置 |
| c_V / λ_Q / c_H | 0.5 / 0.1 / 0.01 | 开发配置 |
| 梯度裁剪 | 0.5 | 开发配置 |
| 优化器 | 单一Adam，共享参数去重 | 冻结实现 |
| 时钟单位 | 秒或固定控制tick，必须一致 | 环境适配时 |
| 折扣起步 | 每1秒0.99，再按实际d换算 | 任务时长确认后 |
| 先验训练模式 | 原始50%、空20%、删15%、改写15% | 开发协议固定 |
| PPO更新起步 | 4 epochs；1024个技能转移／轮 | 受显存与时长适配 |
| 连续片段/小批量 | 片段16步；总64转移，视四路显存分块 | 运行前profiling |
| 任务/控制器/相机/阈值/时限 | **材料未提供，必须实际绑定** | 真机或论文正式运行前 |
| 最终seed与评价数 | 预注册，8 seeds可作预算起点 | 正式评价前 |

不沿用旧P2B高层步预算当新底层交互预算。运行manifest还需写硬件、torch/PyG/仿真版本、模型和配置hash、随机种子、数据split。API客户端选用服务商当前SDK，未知接口参数不能靠历史样例猜测。

## 15.5 计算成本

K个允许候选需要2+2K个图视图，共享静态节点和两种边拓扑。K=20时为42个视图，可批量处理而非42个独立Python对象。空prior复用增强分支，变成1+K。相同patch可复用同一forward内编码，但不跨参数更新缓存可学习表示。

粗略复杂度为O((2K+2)L(RNd²+Md))，N节点、M边、R关系数、L层数。逐关系卷积性能依设备而异，应报告实际峰值显存和延迟。[R4]MVP不实现复杂局部梯度缓存，不声称线性近似effect embedding等价于完整GNN重算。

训练中可保存固定视觉特征和稀疏patch，图编码在PPO更新时重算；FP32完成最终减法。需要混合精度时，零先验路径显式复用，并以固定容差记录数值一致性。

## 15.6 实现验收不是写作前置实验

必须检查：空prior使DP/Δ为0；临时patch不污染真实Fact Store；同节点/goal对齐；UNKNOWN不当FALSE；ADD/DEL冲突被拒绝；goal保留；PPO旧新概率用同mask；Q只gather实际动作；target无梯度；四路共享E能收到梯度；terminated/truncated正确；对象/候选重排后概率一致；缓存键区分初始场景。

这些是代码正确性与安全门槛，随实现完成。本文不预填“验收通过”。它们不要求取得论文级性能，也不阻止现在撰写算法定义。

# 16. 最终论文实验体系

本章设计最终证据，不要求当前立即全跑。研究问题：整体方法是否有效；合同干预是否有价值；VLM是否有额外作用；双差分是否优于直接增强图；Q监督是否有帮助；错误先验下如何退化；未见组合是否复用；Actor是否实际使用差分。

| 实验 | 自变量 | 关键不变项 | 主要证据 |
|---|---|---|---|
| Main results | 第17章核心方法 | 技能、观测、reward、预算、任务split | 成功率、学习曲线、实际执行时间 |
| Contract reasoning | 当前合同图与合同干预 | 相同合同信息、网络容量和Q监督 | 干预是否超出多给合同信息 |
| Prior increment | 合同M1与完整M1 | 同编码器结构及输入权限 | 非冗余先验增量 |
| Differential mechanism | 双差分与单增强图干预 | 同名义patch、E、目标与真实Q标签 | 差分组织的独立作用 |
| Structural Q | Full与λ_Q=0 | 同Actor、数据预算、图与reward | 辅助动作价值监督 |
| Prior robustness | 原始、空、删除、合理错边、无关边 | 真实任务/合同/事实/mask不变 | 绝对成功率及相对退化 |
| Generalization | 未见绑定/依赖/任务组合 | 原子技能与接口不变 | 组合而非仅seed泛化 |
| Mechanism | 固定状态差分干预与闭环干预 | 参数、观测、候选固定 | Actor是否有意义地使用差分 |
| Cost | 对象/候选/图规模 | 相同硬件、时钟口径 | 编码耗时、memory、API摊销 |

## 16.1 任务层级

原子层检验共享技能与Verifier，不承担创新主张；主任务族使用2–3对象、4–10个必要高层操作、多个当前合法选择、共享前置或成果失效；结构泛化层增加未见组合和部分依赖变化，不能只把运输步长变小。

具体平台、机器人和任务资产由材料尚未给出，不虚构已选基准或可直接调用控制器。可引用VIMA等组合评价原则，但不能声称它天然提供本项目全部技能合同。[R14]

## 16.2 公平性与运行预算

所有核心selector使用同一技能库、控制器、参数生成、termination/timeout、感知/Verifier、候选、mask、任务目标、reward和实际时长预算。不同策略轨迹自然不同；公平是相同状态下同一生成规则，不强迫动作序列相同。

Observation-only也获得独立目标、事实、候选与原始合同字段的非图表达，不能只看一张图而被剥夺任务要求。若另设不提供全部合同字段的普通PPO，将它标为系统对照，不用于结构编码的严格归因。

对比训练预算以底层交互/模拟时间为主，高层技能次数另报。在线推理和确认实际消耗的时间计入实机任务，不以暂停仿真忽略延迟后宣称实机等价。

## 16.3 指标与统计

任务成功由独立评价器计算；先按family平均，再跨family和训练seed汇总。报告底层交互横轴的成功率曲线与AUC、达到预定阈值的交互数（未达记未达）、成功条件下完成时间以及全体运行成功/失败分布，不用少量成功样本的快速完成掩盖大量失败。

图鲁棒性报告每条件绝对成功率和百分点退化。恢复统计同时记录loss发生率、机会分母和失效后最终完成率；零机会不是100%。Q TD误差可报告，但下降不等于结构有效。没有可信最优计划时不报告伪造的“比最优多几步”。

用训练seed和任务family体现重复层次，预先固定主要比较、区间覆盖、checkpoint选择、缺失数据和多重比较处理。不按测试最高分选checkpoint、不删除零成功seed、不反复生成关系直到获胜。错误分类区分感知、事实、先验、selector、低层执行与基础设施。

# 17. Baselines：精简并去除重复名称

## 17.1 核心配置

| ID | 配置 | 真正比较什么 |
|---|---|---|
| B0 | Observation/Fact Skill Policy；同任务、候选、原始合同字段，非图列表编码 | 图结构和候选干预的整体效果 |
| B1 | Current-Graph Policy；读取当前GK/GH，不生成名义后继 | 干预相对静态增强图读取 |
| B2 | Contract-Reasoning M1；仅GK和DK，prior为空 | 合同推演的价值；等于独立训练的Full M1 without VLM |
| B3 | Single-Enhanced-Graph Intervention；只用当前GH、名义GH和DH表征 | 双图分离相对普通增强图前瞻 |
| Full | GK/GH四路双差分＋零锚定有界prior＋真实Q | 主方法 |

B2与“Full M1 without VLM prior”的独立训练版同定义，不重复运行两组来扩充表格。测试时把Full的DP置零是同参数干预，不等于独立训练B2。

核心比较为避免额外Q训练收益混淆，B0/B1/B2/B3均可使用相同形式的动作条件化Q辅助头、同一真实目标与λ_Q；Q输入根据各自实际可得特征调整。Full vs Full-noQ独立识别Q作用。若报告标准纯PPO基线，要另标其没有Q辅助，不能将差值全归因于图干预。

B3使用共享候选读出与匹配容量，输出单图策略；它改变信息组织与融合。为更精确隔离第二次减法，可在Full上保留DK主分支，将先验输入DP替换为DH，其他模块不变，标为“no second difference”消融，而不宣称B3一组已经单独识别所有因素。

## 17.2 必要的强参考，而非无限扩张

若完整合同能支持闭环符号规划，可加入一个相同技能接口的规划器参考；若不能，选择一个共享候选、真实反馈的冻结VLM技能规划参考。只需一种最有解释力的非学习/语言规划对照，不机械加入全部HTN、行为树和VLA。STAP与SayPlan等用于Related Work定位，不能未经适配就把论文名称当作可运行baseline。[R12–R13]

低层端到端策略或学习自身技能的方法属于系统级参考，需单列示范、预训练、观测、执行能力和预算，不能用其胜负直接归因M1。

# 18. Ablations与Actor机制诊断

## 18.1 紧凑消融集合

| 机制 | 最小对照 | 是否需重新训练 |
|---|---|---:|
| 合同干预 | Full/Contract对相应当前图读取 | 是 |
| VLM | B2与Full | 是；另作同参数置零 |
| 双差分 | B3或精确no-second-difference | 是 |
| Structural Q | Full-noQ | 是 |
| 有界影响 | Full与相同结构unbounded | 是 |
| 先验扰动训练 | Full-clean-train与Full | 是，若主claim强调此项 |
| Actor依赖 | 以下固定模型干预 | 否 |

不为每种候选数、每条关系错误和每个B重新训练全套模型。数值敏感性可先使用冻结模型作诊断，但最终B机制比较需要对应训练，不只推理临时改幅度。

## 18.2 Actor是否使用D

在同一模型、同一状态、相同候选/mask下依次做：DK=0；DP=0；只在有效候选间置换DP；删除某些软边并重新编码四路图；保留软边数量/类型但替换语义对应作为诊断。goal行与对象绑定不随意错配到非法输入。

记录动作分布JS或KL变化、argmax变化、候选排序、prior残差分布，并在闭环执行中测成功变化。单步快照和闭环干预分开：闭环状态分布改变是干预真实后果，不能假定其后状态仍相同。

**分布有变化只证明网络敏感，不证明变化有益。**需要观察任务相关修改是否产生与真实完成相符的影响；置零/置换也可能带来分布外输入，因此与语义可解释的边干预共同使用。Q误差下降但Actor不响应D，应如实报告，不能只凭共享梯度宣称使用了图。

## 18.3 数值与结构验收

四路共享参数、goal对齐、空prior的DP=0、Δ=0是可直接检查的性质。共同仿射变换下差分协变，不能由此给latent维度命名。每层Dropout关闭、训练时重编码，不把缓存旧D用于新参数更新。

# 19. Robustness与Compositional Generalization

## 19.1 先验质量

主先验来自现实可用强冻结API，不故意选择差模型制造故事。评价Original、No prior、删除、类型合法且合理的错关系、无关关系；保持任务、合同、事实、候选与reward不变。原始关系的语义错误不手工修成gold。

训练扰动只修改软边，episode固定，PPO重算使用同一先验版本。测试采用独立位置、强度和组合。原始生成图的错误审计可与最终实验并行，用于描述分布，不作为重新修正测试输入的依据。

可选增加一个固定不同来源缓存，优先评价同一模型对来源变化的表现；不按通用模型榜单命名“强弱”，也不把另一个大模型当关系真值。关系质量包括合法率、重复、明显错误、未决比例和非冗余产出，格式与语义分开。

## 19.2 泛化划分

训练中原子技能和谓词已出现；测试改变未见对象—目标组合、任务依赖组合或共享条件组织。重新命名后等价的图不当作全新拓扑。外观变化、随机位置、对象数量扩展和不同失效位置分别报告，不混称全部为组合泛化。

所有split在训练前固定，技能/Verifier适配与VLM few-shot也不得使用测试结果。测试缓存允许读取测试初始观测并调用冻结生成器，这是推理，不等于允许用其得分选模型。

## 19.3 能声称与不能声称

允许主张某组任务、先验质量和预算范围内的有效性与稳健性；不能声称任意错误VLM、任意未知对象、所有技能词表和永久遮挡都可处理。合同尚未覆盖的能力、未可观测事实和全新谓词不属于本版的隐含保证。方法主题不绑定某一个百分点，但“能有效使用结构”必须由数据支持；持续无效时应修正经验claim，而不是通过表达优化隐藏结果。

# 20. 论文写作前必须完成的最小前置实验

**当前主体框架已经足以开始论文写作，前置实验为0项；以下验证全部可以与论文写作并行。**

这里的“0项”仅针对“必须先拿到性能数据才能写Problem/Method”的研究实验。它不免除代码单测、设备安全验收、感知校准和正式实验配置登记。实际MVP不运行，就不能写“已跑通”；没有结果，就不能写“优于baseline”。

## 20.1 三类门槛

| 类别 | 内容 | 是否阻塞方法写作 |
|---|---|---:|
| A．设计可冻结 | 图底座、公式、权限、损失、输入输出和时间定义 | 否 |
| B．并行证据 | 效果、模块增益、鲁棒性、泛化、数据效率 | 否；阻塞最终经验结论 |
| 运行门槛 | 控制器、标定、超时、安全、代码一致性 | 否；阻塞未经验证的真实执行 |
| C．必须先做研究实验 | 无：当前没有定义必须靠性能选择才能成立的分支 | **0项** |

## 20.2 十四个问题的逐项判断

| 问题 | 当前分类 | 并行处理与结论边界 |
|---|---|---|
| Strong API relation质量 | 建议早测，不阻塞 | 在生成正式cache时记录质量；未测不写高精度 |
| 非合同重复关系是否存在 | 建议早测，不阻塞 | 去重日志可直接显示；全空时仍可定义模型，但不能声称先验增量 |
| Contract-only是否能学习 | 建议早测，不阻塞 | 合并主训练起步，不要求先跑到显著 |
| M1能否学习 | 建议早测，不阻塞 | 正常MVP训练即提供证据；若全零先检查探索/接口 |
| Actor是否使用DP | 正式实验，不阻塞 | 冻结模型干预，与效用证据一起判断 |
| Structural Q是否有效 | 正式消融，不阻塞 | Full vs noQ；误差下降不是充分证据 |
| Latent subtraction有效性 | 公式可定义，效用待测 | 共享坐标与零性质做单测，收益由消融验证 |
| B取值 | 开发配置，不阻塞 | 开发集固定，不测试后选择 |
| R-GCN层数 | 开发配置，不阻塞 | 不新增网络算法 |
| relation数量 | 开发配置，不阻塞 | 预算起步8，固定后复现 |
| Verifier准确度 | 运行前验收，建议早测 | 硬件可用性必查，不需先获得研究性提升 |
| terminal PPO能否探索 | 建议早测，不阻塞 | 纳入学习曲线；不以Q辅助替代探索证据 |
| compositional generalization | 最终实验，不阻塞 | 划分现在固定，结果后填 |
| prior corruption robustness | 最终实验，不阻塞 | 不宣称任意错误安全 |

## 20.3 不另开前置实验队列

关系合法性检查本就是cache生产；学习smoke run本就是MVP开发；梯度/零差分检查本就是单元测试。它们不升级为要求“完整多seed优于baseline后才准写作”的审查门槛。

若并行实现发现真正的结构性错误，例如四路对不齐、真实数据被名义分支污染、target梯度泄漏，应修复实现或局部定义并记录版本；这不是用实验选论文方向。若有效性没有建立，则限制结果与claim，不改写数据来配合故事。

# 21. 立即写论文计划与Method草稿骨架

## 21.1 写作状态

以下比例表示“当前定义足以支撑的初稿范围”，不是声称已经写完投稿稿。

| 章节 | 现在可写范围 | 还缺什么 |
|---|---|---|
| Introduction | 完整论证和贡献骨架（约100%初稿） | 经验性结果句与最终数字 |
| Related Work | 主线完整可写 | 投稿前文献更新与具体引用格式 |
| Problem Formulation | 当前范围可完整写 | 具体任务域的实例说明 |
| Method | 约90–100%算法初稿 | 绑定后的参数与软件细节 |
| Experimental Setup | 约80–90%协议初稿 | 平台、对象、任务、预算和统计清单 |
| Results | 仅表格位置和问题定义 | 所有真实数值与置信区间 |
| Discussion | 先写模型/观测/先验边界 | 实际失败模式和适用范围 |
| Conclusion | 先写方法性骨架 | 经数据支持的最终结论 |

**从现在开始论文写作与实验并行。**不等待所有结果才写Introduction、Problem、Method和实验协议。投稿前仍必须用最终真实数据核对每项经验主张。

## 21.2 Method章节

3.1 Problem Formulation and Skill-level SMDP

3.2 Skill Contracts and Grounded Action–Proposition Graphs

3.3 Frozen VLM Semantic Relations and Reproducible Caches

3.4 Candidate Nominal Structural Interventions

3.5 Contract–Prior Dual Differences

3.6 Zero-anchored Bounded Prior Policy

3.7 Outcome-grounded Structural Q and PPO Training

Fact Verifier、数据契约和完整伪代码可在主文简述并放附录展开。R-GCN和SLG引用原工作，不占创新叙述。

## 21.3 可以直接使用的方法定位段

> We adopt a grounded action–proposition representation of skill contracts and use a standard shared R-GCN encoder. Our contribution is not a redesigned symbolic graph or a VLM planner. For each executable skill, we apply a nominal success intervention to a temporary fact assignment and compare the resulting representations under contract-only and semantically augmented structures. Their dual difference encodes how additional VLM relations interact with the candidate-induced structural change. A zero-anchored bounded residual incorporates this interaction into skill selection, while a separate action-conditioned value head trains the shared features using targets derived from actual skill executions.

这里的nominal明确区别真实未来；“trains using actual targets”包含旧V bootstrap误差，不替换为已证明的因果或安全描述。学习是否有效由实验章节承担。

# 22. 最终贡献、冻结清单与论文故事

## 22.1 冻结的主体

技能级结构化action；共享冻结低层；第三视角RGB-D及真实事实；强冻结API与离线cache；SLG两节点底座；标准共享R-GCN；目标对齐E；一步名义成功干预；DK/DH/DP；合同主分支与零锚定有界prior；独立V、selected-action Q；真实终端reward；duration-aware PPO；高层一阶段联合训练；无VLM训练和新辅助loss。

“全局Skill Schema＋每episode grounding”固定。Object在参数中、Goal在Fact标记和读出索引中、普通AND在合同求值器中；不重新引入五类节点。

## 22.2 仍需填入的配置，不重新打开研究方向

具体机器人、仿真平台、对象资产、相机标定、冻结感知模型、每技能时限、谓词阈值、数据split、训练预算、最终模型超参和统计计划尚待真实资源绑定。材料没有提供的内容不得写成已经完成。它们不妨碍先实现类型、构图、差分、模型与buffer，也不妨碍写算法正文。

## 22.3 贡献边界表

| 部分 | 来源 | 定位 |
|---|---|---|
| Skill Contract | 真实技能接口及一次性研究者校验 | 基础设施 |
| SLG Action–Proposition | 既有规划学习表示[R1] | 不作为创新 |
| Schema参数共享 | ASNet等原则[R2] | 不作为创新 |
| R-GCN/PPO/GAE | 成熟算法[R3–R8] | 不作为创新 |
| VLM软关系与cache | 本方法固定输入协议 | 不单独作为核心创新 |
| Candidate nominal intervention | M1的候选结构计算 | 拟议核心组成 |
| Contract/prior dual difference | M1的交互表示算子 | 拟议核心 |
| Outcome-grounded Structural Q | 真实动作目标监督共享结构特征 | 拟议核心训练组成 |
| Verifier与图管理器 | 运行时事实基础设施 | 不作为核心创新 |

## 22.4 一句话论文故事

**我们不让VLM决定机器人必须怎样行动，而是在给定技能合同上对候选动作施加名义结构干预，分离合同变化与不完美语义先验的交互，并以真实执行回报学习这些结构差异何时值得影响技能选择。**

这一故事不依赖某个固定提升百分点，但它要求最终数据支持结构机制确有价值。不能把论文主题不绑定单一数字理解为不需要实证。

# 附录A：固定prompt样例与可追溯输入

本附录是生成协议的实施样例，来源为本轮按既定schema补齐的设计。few-shot完整材料须在正式cache生成前落盘，不能仅保存“用了3个示例”而没有具体内容。

## A.1 每个请求的用户输入结构

```text
TASK:
  instruction: <独立原始指令>
  goals: <目标ID、对应命题、要求符号>
  explicit_constraints: <仅用户/基准明确要求>
INITIAL_SCENE:
  image: <实际初始RGB输入>
  objects: <ID、类型、参数角色、绑定状态>
  observed_facts: <必要T/F/U与观测质量，不用隐藏真值>
AVAILABLE_ACTIONS:
  <全部类型合法实例ID＋参数>
CONTRACTS:
  <schema与PRE/ADD/DEL简要定义>
ALLOWED_FACT_IDS:
  <可引用注册命题>
RELATION_SCHEMA:
  <第7章JSON结构和端点类型>
OUTPUT:
  JSON object only, 0 to 8 relations, no new identifiers.
```

## A.2 三个固定示例

**示例1：合同重复，不凑关系。**开发输入已给OPEN(box)增加Open(box)，Open(box)为PLACE(red,box)前置，没有其他有依据的联系。期望输出：

```json
{"schema_version":"m1_soft_relations_v1","relations":[]}
```

**示例2：额外但不保证成功的场景支持。**开发图像中red占据blue接近区域，已有MOVE(red,buffer)与PLACE(blue,box)，合同未表示该场景联系。输出允许：

```json
{
  "schema_version":"m1_soft_relations_v1",
  "relations":[{
    "relation_id":"ex2_r1",
    "type":"SOFT_SUPPORTS",
    "source_ref":"a:MOVE:red:buffer",
    "target_ref":"a:PLACE:blue:box",
    "optional_mediating_fact_ref":null
  }]
}
```

该标签只示范“可能支持”。不同时输出“PLACE一定成功”或新的ClearPath事实。若实际开发图像并不支持该关系，不能只复制文本假装有视觉依据。

**示例3：两种顺序都合法，不添加唯一顺序。**目标为放好red和blue，观测未显示优先级依据；合同路径对称。输出仍可为空。空输出用于示范拒绝无依据推断，不在所有场景强制为空。

few-shot通过固定开发场景提供输入图像与上述输出格式；模型收到的不是完整正确skill sequence。所有示例进入prompt_hash。若输入场景超出允许证据，系统不让VLM扩写真实技能或对象。

# 附录B：实现配置模板与验收门槛

以下为可复制的配置样例。它是本文新增的实施默认，不是已执行实验的配置。

```yaml
method:
  name: CP-DISR
  internal_name: M1
  specification: v2.0
  prereq_research_experiments: 0
vlm:
  provider: alibaba_model_studio
  model_snapshot: qwen3.8-max-0902
  region: null
  thinking: false
  response_format: json_object
  relation_schema: m1_soft_relations_v1
  max_relations: 8
  temperature: 0
  max_output_tokens: 2048
  format_retries: 1
  allow_web_search: false
  trainable: false
  cache_required_for_training: true
graph:
  representation: grounded_action_proposition_slg
  node_types: [ACTION, PROPOSITION]
  contract_relations: [PRE_POS, PRE_NEG, ADD, DEL]
  reverse_relations: true
  encoder: RGCNConv
  hidden_dim: 128
  layers: 4
  num_bases: 4
  aggregation: mean
  root_weight: true
  normalization: per_node_layer_norm
  dropout: 0
  goal_alignment: fixed_proposition_indices
m1:
  nominal_branches: success_only
  nominal_depth: 1
  shared_encoder: true
  zero_anchor_prior: true
  prior_bound: 0.5
  difference_dtype: float32
  use_text_initialization: false
  use_teacher: false
training:
  optimizer: Adam
  learning_rate: 0.0003
  ppo_clip: 0.2
  gae_lambda_per_skill: 0.95
  ppo_epochs: 4
  rollout_skill_transitions: 1024
  recurrent_sequence_length: 16
  minibatch_transitions: 64
  recurrent_hidden_dim: 128
  value_coef: 0.5
  structural_q_coef: 0.1
  entropy_coef: 0.01
  grad_clip_norm: 0.5
  reward: independent_terminal_success
  gamma_reference_seconds: 1.0
  gamma_reference_value: 0.99
  actor_episode_discount_weight: true
  target_detach: true
  graph_pretraining: false
  value_pretraining: false
  policy_warm_start: false
prior_training:
  original: 0.50
  absent: 0.20
  deletion: 0.15
  relation_rewrite: 0.15
  sampled_at: episode_start
runtime_required:
  robot_or_simulator: null
  environment_version: null
  controller_manifest: null
  calibration_manifest: null
  verifier_manifest: null
  skill_time_limits: null
  task_splits: null
  task_evaluator_version: null
  total_interaction_budget: null
  seeds: null
```

配置校验器需拒绝runtime_required为空的真实训练/执行启动。纯逻辑单元测试可使用模拟结构输入，但必须标明不是机器人性能实验。学习参数表不是新增VLM训练任务；API实例化、传感器权限和执行controller仍须由实际项目接入。

# 附录C：科学可主张范围与运行可复现性

**可在实验前陈述：**图与节点的定义；四路同坐标结构；名义干预不修改真实事实；空prior严格零交互；局部有界概率性质；target和梯度来源；基础算法引用；可运行的接口约定。

**必须有结果才陈述：**成功率提升、样本效率提升、Q有益、比普通前瞻更好、VLM提供真实增量、错误先验更稳健、组合泛化、实机可靠性。表示可计算不等于已具备价值语义；不同维度没有固定人工标签不是自动失败，也不是自动有效。

**不保证：**合同完美物理预测；任意未知对象/技能；永久遮挡中的真值确认；任意先验错误安全；错误图无法改变argmax；本方法超过所有规划器；辅助Q获得无偏真实价值；上游API永久可再生成。

复现包应包含：代码与环境依赖锁、任务资产可发布部分、控制器与Verifier版本、所有split、模型checkpoint、初始化图像hash、API raw/parsed/rejected缓存、完整prompt/few-shot、超参数、seed、日志、评价器版本、checkpoint选择规则和失败运行列表。仅有relation cache不足以保证全部训练可复现；它解决的是再次访问原API的依赖。

# 附录D：来源与参考文献

**[S0] 用户本轮附件。**《Pasted markdown(8).md》。本文件将其最终主方案统一为规范；附件中的“例如”与缺省参数均标为设计/配置，不当作已有实验结论。方法名称、版本号、工程默认与缓存格式为本轮整理时明确补齐的内容。

**[R1] Chen, D. Z., Thiébaux, S., Trevizan, F.** Learning Domain-Independent Heuristics for Grounded and Lifted Planning. AAAI 2024, 38(18): 20078–20086. DOI: 10.1609/aaai.v38i18.29986. arXiv:2312.11143。直接依据：SLG两类节点、PRE/ADD/DEL关联、状态和goal标记；本项目三态和输出接口是明确适配。

**[R2] Toyer, S. et al.** Action Schema Networks / ASNets: Deep Learning for Generalised Planning. AAAI 2018；扩展版JAIR 2020，arXiv:1908.01362。依据：ground action/proposition结构及schema级参数共享。本文不复制完整ASNet训练法或专用层。

**[R3] Schlichtkrull, M. et al.** Modeling Relational Data with Graph Convolutional Networks. ESWC 2018. arXiv:1703.06103。依据：多关系图卷积与关系参数共享。

**[R4] PyTorch Geometric官方文档。** torch_geometric.nn.conv.RGCNConv。核对日期2026-09-20。依据：mean aggregation、root transform、num_bases和edge_type接口；实际软件版本须写入manifest。

**[R5] Sutton, R. S., Precup, D., Singh, S.** Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence, 112:181–211, 1999. DOI: 10.1016/S0004-3702(99)00052-1。

**[R6] Schulman, J. et al.** Proximal Policy Optimization Algorithms. 2017. arXiv:1707.06347。本文采用PPO，不将其作为新算法。

**[R7] Schulman, J. et al.** High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR 2016. arXiv:1506.02438。依据：优势估计的偏差—方差折中与GAE形式。

**[R8] Nota, C., Thomas, P. S.** Is the Policy Gradient a Gradient? AAMAS 2020. arXiv:1906.07073。依据：折扣目标与访问分布权重的区别；不是M1性能保证。

**[R9] Gymnasium官方文档。** Handling Time Limits。核对日期2026-09-20。依据：任务终止与外部截断的bootstrap差异。

**[R10] 阿里云Model Studio官方模型说明。** qwen3.8-max；快照qwen3.8-max-0902，别名qwen3.8-max-2026-09-02。核对日期2026-09-20。只确认公开快照与视觉输入能力，不代表本项目账户调用或relation质量已验证。

**[R11] 阿里云Model Studio官方文档。** 如何让千问生成JSON字符串／结构化输出。核对日期2026-09-20。依据：图像JSON Mode与本地schema验证需求。服务文档可能变化，运行manifest需记录实际接口行为。

**[R12] Agia, C. et al.** STAP: Sequencing Task-Agnostic Policies. ICRA 2023. arXiv:2210.12250。Related Work：技能组合及后果评价；不作为M1独有新颖性证明。

**[R13] Rana, K. et al.** SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable Robot Task Planning. CoRL 2023, PMLR 229。Related Work：图场景、语言规划与执行验证。

**[R14] Jiang, Y. et al.** VIMA: Robot Manipulation with Multimodal Prompts. ICML 2023, PMLR 202。参考组合评价的分层设计，不默认采用其完整benchmark作为本项目环境。

**[R15] Farahmand, A.-M., Barreto, A., Nikovski, D.** Value-Aware Loss Function for Model-based Reinforcement Learning. AISTATS 2017, PMLR 54。参考决策相关表示/模型监督的背景，不将M1辅助Q等同于该方法或继承其保证。

以上为本轮核对的主要一手依据，不是穷尽式新颖性审查。后续Related Work应在投稿前更新，但不因此继续添加方法模块。

# Final Frozen Research Snapshot

| 项目 | 冻结定义 |
|---|---|
| **Problem** | 固定技能库中，如何利用合同与可错语义先验，学习长期技能选择。 |
| **Core Idea** | 对每个候选构造名义成功逻辑副本，分离合同变化与先验交互，用真实执行回报监督其决策用途。 |
| **VLM** | 固定强API快照；主配置qwen3.8-max-0902。只生成初始少量软关系，离线缓存，不训练、不在线重规划。 |
| **$G^K$** | SLG式Action–Proposition合同图，PRE_POS/PRE_NEG/ADD/DEL及反向消息边，goal在命题标记与固定索引中。 |
| **$G^H$** | 与$G^K$同节点、同目标、同事实，只增加受限VLM软边；去重后空先验等于合同图。 |
| **Nominal Intervention** | 给定合同的成功分支符号干预，只改临时事实；不是物理预测，不写真实状态和reward。 |
| **$D^K$** | 共享编码器下合同后继减合同当前，表示候选诱发的结构编码变化。 |
| **$D^P$** | 增强图前后差再减合同前后差，表示软关系与同一干预的编码交互，不是真实因果价值。 |
| **Actor** | Contract-Reasoning主分支＋零锚定有界prior残差；DP=0时Δ=0，执行mask与VLM无关。 |
| **Structural Q** | 只对实际动作，用真实reward和旧V bootstrap监督共享结构特征；不修改具体边的永久trust。 |
| **PPO** | 技能级、按实际时长折扣；独立V用于GAE；高层一阶段联合训练，只有最终环境成功奖励。 |
| **Verifier** | 冻结感知、几何/时间规则和执行日志产生T/F/U；与软先验和名义副本隔离。 |
| **Required Pre-Experiments** | **0项写作前置研究实验。**单测、标定、安全验收与正式效果实验并行，不冒充已经通过。 |
| **Paper Status** | 可以立即写Introduction、Related Work、Problem、Method和实验协议；Results待真实数据，经验claim随证据填写。 |
