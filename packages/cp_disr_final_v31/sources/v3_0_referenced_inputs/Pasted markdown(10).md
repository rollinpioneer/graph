请基于截至目前我们关于 CP-DISR / M1 的全部讨论，重新整理一版：

# CP-DISR 最新完整研究文档 —— Paper-Oriented Research Specification

这次的目标和之前不同。

不要再写成：

- brainstorming记录；
- 问题清单；
- 方法审查报告；
- 多个方案并列讨论；
- “以后可以考虑A/B/C”的研究备忘录。

我要的是：

> **一份已经明显向正式论文靠拢的、统一的、完整的研究文档。**

它既要能够作为：

1. 当前研究方案的最终说明；
2. 后续实验实现的依据；
3. 论文 Method / Problem Formulation / Experimental Setup 的直接底稿；
4. 后续写 Introduction、Abstract、Results 时的统一事实来源。

详细程度要足够高。

新的研究人员如果只读这一份文档，
应该能够基本理解：

- 我们究竟在研究什么；
- 为什么这个问题值得研究；
- 当前方法到底解决什么、不解决什么；
- VLM、Skill Contract、Graph、D^K、D^P、Structural Q、PPO各自扮演什么角色；
- 信息如何前向传递；
- 真实执行结果如何参与训练；
- 哪些模块被冻结；
- 哪些模块学习；
- 如何实现；
- 如何训练；
- 如何评价；
- 论文最终应该主张什么。

同时：

Agent如果只读这份文档，
应该能够据此继续实现当前方法。

目标论文水平仍然按照：

**CCF-A会议 / SCI一区期刊的方法论文**

来组织。

但是：

当前文档不需要假装已经拥有尚未完成的实验结果。

实验结论没有数据支撑的地方：

写成待验证命题，
不要伪造结果。


==================================================
一、首先明确：当前主方法继续冻结
==================================================

当前论文版本继续采用：

**CP-DISR / M1 v2.1**

不要因为最近讨论了反馈链问题，
就重新加入新的模块。

当前主版本不加入：

- per-edge trust；
- relation memory；
- online graph repair；
- VLM feedback loop；
- VLM fine-tuning；
- contract calibration；
- relation-level reward；
- 新world model；
- 新planner；
- LLM teacher；
- 新辅助loss。

这些都不属于当前论文主方法。

最近对信息传导链的审查最终结论是：

> 当前“真实回报不修改VLM / Skill Contract源头”是一个合理的研究边界，而不是必须修补的结构性错误。

当前研究的对象不是：

> 如何纠正VLM。

而是：

> **如何在固定Skill Contract与固定但不完美的VLM semantic prior条件下，通过真实执行经验学习这些信息什么时候、以什么方式影响长期skill selection。**

请把这个定位贯穿整篇文档。


==================================================
二、必须纠正旧版本中可能过强的表述
==================================================

不要再写：

- correct VLM prior；
- repair VLM relation；
- identify wrong relation；
- learn true relation；
- Structural Q corrects erroneous graph；
- D^P extracts the true causal contribution of VLM；
- reward反向告诉系统哪条VLM边是错的。

这些都超过了当前方法真正做到的事情。

当前更准确的表述是：

### VLM

提供：

**fixed imperfect semantic relations**

### D^P

表示：

**soft relations如何改变模型对candidate nominal structural change的编码响应**

### PPO / Structural Q

使用真实执行经验：

**训练共享表示及这些表示的决策用途**

而不是：

验证或修改源关系。


==================================================
三、先给论文级核心定位
==================================================

请重新定义：

### 1. Problem

### 2. Motivation

### 3. Core contradiction

### 4. Research question

### 5. Core hypothesis

### 6. Paper-level one-sentence story

当前核心问题应接近：

> 在长时程机器人任务中，skill library提供固定执行能力，Skill Contract提供可靠的名义结构知识，而VLM能够补充合同无法显式编码的语义关系。然而VLM关系是不完美的，直接将其作为计划或硬约束容易导致错误。我们研究如何以可靠合同为基础，对候选skill的名义后果进行结构推演，并学习固定不完美语义先验应如何影响这些候选后果的解释与最终skill选择。

请重新打磨成论文级表达。

不要把论文定位成：

“VLM机器人规划”。

也不要定位成：

“自动修图”。

也不要定位成：

“新的知识图谱”。

更准确的是：

**structured decision learning with fixed imperfect semantic priors**。


==================================================
四、重新给当前方法起正式论文名称
==================================================

“M1”只保留为内部代号。

请根据最终方法重新判断：

CP-DISR

是否仍然是最合适正式名称。

如果是：

解释缩写和名称含义。

如果不是：

给：

### 主推荐名称

以及：

### 2–3个备选名称。

名称重点考虑：

- Candidate-conditioned；
- Intervention；
- Contract；
- Semantic Prior；
- Structural Reasoning；
- Outcome-grounded learning。

不要突出：

“repair”
“correction”
“trust”

因为当前方法并不做这些。


==================================================
五、给一版论文摘要草稿
==================================================

请写：

### Abstract v0

要求接近真正论文摘要。

包括：

1. 背景；
2. 问题；
3. 当前方法；
4. 核心机制；
5. 训练方式；
6. 实验将验证什么。

由于目前实验未全部完成：

最后结果部分使用：

可替换占位表达，

例如：

“Experiments across ... evaluate ...”

而不是伪造数值。


==================================================
六、给一版Introduction完整逻辑
==================================================

不要只是列提纲。

请接近论文正文地写出：

### Introduction v0

需要形成清晰的故事：

1. 长时程skill selection为什么困难；
2. Skill Contract能解决什么；
3. Contract缺少什么；
4. VLM为什么有帮助；
5. 为什么不能直接相信VLM；
6. 当前已有简单方案可能是什么；
7. 为什么我们关注candidate consequence而不是直接VLM action recommendation；
8. CP-DISR的核心想法；
9. 为什么真实执行经验仍然重要；
10. 三项贡献。

注意：

不要把“VLM错误”写成必须在线修复的问题。

当前主张是：

> VLM关系可以有用，也可能无用或误导，因此policy应该学习其条件化决策用途。


==================================================
七、最终贡献必须重新写准确
==================================================

当前建议仍保持三项贡献。

请重新写成论文正式措辞。

大致方向：

### C1. Candidate Nominal Structural Intervention

基于Skill Contract：

对每个候选skill构造名义成功干预，

显式表示：

“如果当前执行这个candidate并按合同成功，会改变什么结构状态？”


### C2. Contract–Prior Dual Difference

分别在：

可靠合同图

和：

合同 + 不完美semantic prior

中计算相同candidate intervention的结构响应，

通过：

D^K
D^H
D^P

组织：

合同后果

与：

prior-conditioned interaction。


### C3. Outcome-Grounded Structural Learning

通过：

PPO
+
Structural Q

使用真实执行结果：

训练这些结构表示如何被用于skill selection。

注意：

C3不是：

relation correction。

而是：

**outcome-grounded utilization / representation learning。**

请给最终英文/中文贡献描述。


==================================================
八、系统总体架构
==================================================

请重新画出完整系统：

Task Instruction
+
Initial Scene
+
Skill Library
↓
Frozen VLM
↓
Soft Relation Cache

Skill Contracts
+
Current Verified Facts
↓
Contract Graph G^K

G^K
+
Soft Relations
↓
G^H

Current Observation
+
History
+
Candidate Skills
↓
Candidate Nominal Intervention
↓
tilde G^{K,i}
tilde G^{H,i}

Shared R-GCN
↓
E(G^K)
E(G^H)
E(tilde G^{K,i})
E(tilde G^{H,i})

↓
D_i^K
D_i^H
D_i^P

↓
Contract Candidate Feature
+
Prior Interaction Feature
+
Observation / History

↓
Actor
V
Structural Q

↓
Skill Selection
↓
Fixed Low-Level Skill Controller
↓
Real Environment Outcome
↓
Verifier / Reward / Duration
↓
PPO + V + Structural Q Learning

请分别解释：

### 初始化阶段

### 每个高层决策step

### skill执行阶段

### 执行后事实更新

### PPO update阶段。


==================================================
九、重新明确三层信息结构
==================================================

请正式加入：

### Level 1 — Information Source

- Skill Contract；
- Verified Fact；
- Observation；
- VLM soft relation。

解决：

“系统有什么证据可用？”


### Level 2 — Structural Representation

- G^K / G^H；
- R-GCN；
- nominal intervention；
- D^K / D^P。

解决：

“这些信息如何围绕candidate被组织？”


### Level 3 — Decision Learning

- Actor；
- PPO；
- V；
- Structural Q。

解决：

“这些表示什么时候应该影响skill decision？”

这三层非常适合作为整篇方法的概念骨架。

请在Method Overview中明确使用。


==================================================
十、Skill Contract正式定义
==================================================

请给论文级定义。

至少包含：

- skill schema；
- arguments；
- type constraints；
- initiation / preconditions；
- nominal ADD；
- nominal DEL；
- unknown effects；
- success termination；
- failure termination；
- verification；
- timeout；
- provenance。

说明：

合同表示：

> **skill成功分支的名义结构语义**

而不是：

> skill每次执行都一定成功。

这点必须明确。

例如：

PLACE成功
→
Inside(object,target)=TRUE

不等于：

PLACE成功率=100%。

真实执行可靠性最终由真实经验影响动作价值。


==================================================
十一、Skill Contract来源
==================================================

保持：

existing skill controller interface
+
researcher one-time specification
+
execution-level validation。

区分：

### Library-level contract

一次定义。

### Episode-level grounding

自动绑定object。

例如：

PLACE(?obj,?target)

→

PLACE(red,box)

PLACE(blue,box)

请写实现流程。


==================================================
十二、Fact Verifier
==================================================

正式定义：

Observation
+
proprioception
+
execution log
+
geometric / temporal rules

→

TRUE / FALSE / UNKNOWN。

Verifier不接受：

VLM relation

作为事实真值来源。

它属于：

运行时基础设施，

不是核心算法贡献。


==================================================
十三、Contract Graph G^K
==================================================

采用：

grounded Action–Proposition Graph。

Action nodes：

PICK(red)
OPEN(box)
PLACE(red,box)

Proposition nodes：

Held(red)
Open(box)
Inside(red,box)

relation：

PRE_POS
PRE_NEG
ADD
DEL

以及reverse message edges。

正式定义：

\[
G_t^K=(V_A,V_P,E_K,X_t)
\]

或者更合适的数学表达。

说明：

Object不单独建立节点。

Goal通过：

goal proposition flag
+
fixed goal_refs

表示。


==================================================
十四、VLM Semantic Prior
==================================================

正式说明：

VLM输入：

- initial scene image；
- task instruction；
- instantiated skill IDs；
- relevant fact / goal IDs；
- relation schema。

只生成：

### SOFT_SUPPORTS

### SOFT_RELEVANT_TO_GOAL

并且必须包含：

effect_fact_ref。

删除：

SOFT_ORDER。

强调：

VLM不能：

- 新增skill；
- 修改Skill Contract；
- 宣布Fact真假；
- 产生hard mask；
- 修改reward；
- 直接选最终动作。


==================================================
十五、effect_fact_ref的角色要准确
==================================================

明确：

effect_fact_ref主要用于：

- relation validation；
- provenance；
- audit；
- 确保soft relation与source skill某个已有名义effect有关。

当前MVP中：

它不是新的fact node；
不是额外hyperedge；
不单独扩展Graph topology。

如果当前实现确实如此：

请准确写出边界。


==================================================
十六、VLM Cache
==================================================

完整定义：

task
+
initial scene
+
bindings
+
allowed IDs
+
model snapshot
+
prompt version
+
schema version

→

cache key。

保存：

- raw response；
- parsed relation；
- rejected relation；
- dedup log；
- model；
- prompt；
- decoding setting；
- hash。

PPO训练期间：

只读取cache。

VLM不参与backprop。


==================================================
十七、G^H
==================================================

定义：

\[
V^H=V^K
\]

只增加：

\[
E_{\text{soft}}
\]

因此：

\[
G^H=G^K\cup E_{\text{soft}}
\]

VLM不能新建节点。

如果relation与合同结构重复：

程序去重。

若：

所有soft relation都被去重：

\[
G^H=G^K
\]

因此：

\[
D^P=0
\]

请把这个性质作为方法设计中的重要 sanity property。


==================================================
十八、Graph Encoder
==================================================

主版本：

Standard shared R-GCN。

不要把R-GCN写成核心创新。

解释：

为什么需要relation-aware message passing。

节点初始化至少说明：

Action：

- schema embedding；
- argument/object feature；
- current applicability；
- observation-linked features。

Proposition：

- predicate schema embedding；
- truth state；
- goal sign；
- observation-linked features。

schema-level parameter sharing：

PICK(red)
PICK(blue)

共享：

PICK。

Held(red)
Held(blue)

共享：

Held。


==================================================
十九、E(G)正式定义
==================================================

明确：

E(G)

不是：

- success probability；
- task value；
- cost；
- remaining horizon。

它只是：

**learned structural representation。**

采用：

goal-aligned representation。

例如：

\[
E(G)
=
[z_{\text{global}},z_{g_1},...,z_{g_m}]
\]

\[
E(G)\in\mathbb R^{(m+1)\times d}
\]

请说明：

goal alignment为什么对于不同graph view做latent subtraction很重要。


==================================================
二十、Candidate Nominal Intervention
==================================================

这是Method核心。

对于当前candidate：

\[
a_i
\]

读取：

Skill Contract nominal ADD / DEL。

构造：

\[
\tilde F_t^i
\]

再得到：

\[
\tilde G_t^{K,i}
\]

和：

\[
\tilde G_t^{H,i}
\]

强调：

这不是：

- physics rollout；
- learned world model；
- success prediction；
- environment simulation；
- VLM simulation。

只是：

> **nominal symbolic success intervention。**

请给正式算法和伪代码。


==================================================
二十一、四图必须由同一个R-GCN编码
==================================================

当前顺序必须准确：

不是：

R-GCN先编码
→
再做intervention。

而是：

当前事实
→
构造base graph

candidate nominal effect
→
构造temporary graph

然后四种view：

G^K
G^H
tilde G^{K,i}
tilde G^{H,i}

全部经过：

同一个shared R-GCN。

请把forward chain彻底写准确。


==================================================
二十二、D^K
==================================================

定义：

\[
D_i^K
=
E(\tilde G_i^K)-E(G^K)
\]

解释：

> candidate的名义成功后果，在可靠合同结构中造成的representation change。

不是：

真实价值。


==================================================
二十三、D^H
==================================================

定义：

\[
D_i^H
=
E(\tilde G_i^H)-E(G^H)
\]

解释：

> 在soft relations也参与消息传播时，相同candidate intervention造成的representation change。


==================================================
二十四、D^P
==================================================

定义：

\[
D_i^P
=
D_i^H-D_i^K
\]

等价：

\[
D_i^P
=
[E(\tilde F_i,R)-E(F,R)]
-
[E(\tilde F_i,\varnothing)-E(F,\varnothing)]
\]

准确解释：

> **soft relations如何改变Graph Encoder对该candidate-induced nominal structural change的响应。**

它不是：

- relation correctness；
- causal contribution；
- success increment；
- true value；
- VLM confidence；
- relation trust。


==================================================
二十五、Pure Interaction Prior
==================================================

正式写清：

当前v2.1不使用：

Static VLM Prior。

VLM不能直接产生：

“OPEN +0.5分”。

它必须通过：

candidate consequence representation

影响policy。

如果：

VLM影响只是一个与candidate intervention完全可分离的静态bias：

\[
E(F,R)=A(F)+B(R)
\]

则：

\[
D^P=0
\]

这是有意设计。


==================================================
二十六、当前所谓“有用prior就学、没用prior就少用”应如何准确表述
==================================================

请把最近讨论正式纳入文档。

当前架构允许：

### 正向利用

某种prior-conditioned结构模式经常与更好决策相关：

prior residual可以增加candidate logit。


### 弱化 / 忽略

该模式没有帮助：

residual可以趋近0。


### 负向修正

当前：

\[
\Delta_i\in[-B,B]
\]

因此：

某种prior-conditioned模式也可以降低candidate logit。


但是：

不要写：

> 模型一定能够稳定识别所有坏prior。

更准确：

> **当前架构提供了根据上下文和经验选择性利用固定先验的能力接口，其实际学习效果由实验验证。**


==================================================
二十七、Actor最终结构
==================================================

Contract branch：

输入：

- observation/history；
- candidate；
- current G^K representation；
- D^K。

得到：

\[
b_i
\]

Prior branch：

输入：

- candidate context；
- D^P。

输出：

zero-anchored bounded residual：

\[
\Delta_i
=
B\tanh(f_P(...))
\]

且必须保证：

\[
D_i^P=0
\Rightarrow
\Delta_i=0
\]

最终：

\[
\ell_i=b_i+\Delta_i
\]

然后在合法candidate mask内：

softmax。


==================================================
二十八、为什么需要bounded prior
==================================================

准确解释：

Skill Contract属于系统定义的可靠能力结构。

VLM relation只是额外软信息。

因此：

prior可以：

影响policy，

但其直接logit作用受到限制。

不要声称：

bounded residual保证绝对安全或鲁棒。


==================================================
二十九、Actor / V / Structural Q不是串行关系
==================================================

明确：

它们是：

共享部分feature的不同head。

不是：

Actor → V → Q。

Actor：

负责policy。

V：

负责PPO / GAE baseline。

Structural Q：

动作条件化辅助价值监督。


==================================================
三十、Structural Q重新准确定义
==================================================

输入：

\[
Q(X_t,a_i)
\]

包含：

- observation/history；
- candidate；
- D^K；
- D^P；
- candidate-set context。

只对实际执行的：

\[
a_t
\]

产生监督。

目标：

\[
y_t^Q
=
\operatorname{sg}
[
r_t
+
\Gamma_t
(1-\text{terminated}_t)
V_{\text{old}}(X_{t+1})
]
\]

Loss：

Huber。


==================================================
三十一、Structural Q真正做什么
==================================================

这一部分必须根据最新讨论重写。

Structural Q不是：

> 哪条VLM边错了的识别器。

它提供的是：

> **actual-outcome-supported action-value learning signal**

该信号可以更新：

- Q head；
- candidate feature；
- R-GCN；
- observation encoder。

因此它可以：

帮助共享表示更符合真实长期动作结果。

但它不能直接区分：

- VLM错误；
- low-level controller失败；
- perception error；
- exploration noise；
- value estimation error。

所以不要说：

Structural Q corrects the VLM prior。


==================================================
三十二、Forward Chain
==================================================

请给：

论文图式的Forward Chain。

准确顺序：

Task / Scene
→
VLM cache

Contracts + Facts
→
G^K / G^H

Candidate
→
Nominal Fact Patch
→
temporary graphs

四图
→
Shared R-GCN

→
D^K / D^P

Observation / History
→
Candidate Features

→
Actor / V / Q

→
Skill

→
Low-level controller

→
Real outcome。


==================================================
三十三、Backward Chain
==================================================

真实环境本身不是可微模块。

流程应该写成：

真实transition
→
rollout buffer
→
advantage / V target / Q target
→
Actor loss
+
V loss
+
Q loss
→
trainable neural modules。

更新：

- Observation Encoder；
- GRU；
- R-GCN；
- candidate feature；
- Actor；
- V；
- Q。

冻结：

- VLM；
- VLM relation cache；
- Skill Contract；
- Graph Builder；
- Nominal Intervention rules；
- Verifier；
- low-level controllers；
- task evaluator。

请做完整表格。


==================================================
三十四、Current Paper的能力边界
==================================================

请正式写一个：

### Scope and Boundary

但不要写成自我批评。

正常说明：

当前方法研究：

> fixed imperfect prior utilization。

不研究：

- relation repair；
- knowledge base revision；
- per-edge correctness estimation；
- VLM adaptation；
- online graph rewriting。

VLM relation缺失：

不等于policy必然无法学会该行为。

因为：

Observation / history / real experience

也可能提供相关信息。

真正硬瓶颈是：

> 必要证据在所有可用输入中不可辨认，或者必要action不存在/被错误mask。


==================================================
三十五、Skill Contract错误如何处理
==================================================

合同与VLM地位必须分开。

### Skill Contract

是：

能力接口。

如果通过控制器/证据确认合同写错：

应该：

修订contract版本。

不能期待：

PPO自动修合同。


### VLM relation

是：

不完美软输入。

策略可以：

根据经验调整其决策影响。


==================================================
三十六、不要加入Contract Calibration
==================================================

当前nominal intervention表示：

> 如果skill按名义成功分支完成，会发生什么。

不是：

> 当前skill的成功概率。

实际skill可能：

成功；
失败；
耗时不同；
失败后恢复。

这些都通过：

真实return / PPO / Q

进入实际动作价值学习。

当前论文不增加：

\[
P(success|state,skill)
\]

除非未来实验证明这是主要瓶颈。


==================================================
三十七、训练目标
==================================================

给最终：

\[
L_{\text{total}}
=
L_{\text{PPO}}
+
c_VL_V
+
\lambda_QL_Q
-
c_HH(\pi)
\]

解释：

Actor loss；

Value loss；

Structural Q loss；

Entropy。


==================================================
三十八、Semi-MDP训练
==================================================

skill duration不同。

使用：

\[
\Gamma_t=\bar\gamma^{d_t}
\]

GAE、V target、Q target：

统一使用duration-aware discount。

请正式定义。


==================================================
三十九、Training Algorithm
==================================================

请给接近论文Algorithm 1级别的伪代码。

包括：

1. task reset；
2. load VLM cache；
3. fact verification；
4. candidate generation；
5. construct G^K/G^H；
6. candidate nominal intervention；
7. batch graph views；
8. shared R-GCN；
9. compute D^K/D^P；
10. actor；
11. sample skill；
12. low-level execution；
13. reward / duration；
14. next observation/facts；
15. rollout buffer；
16. GAE；
17. Q target；
18. PPO epochs；
19. backward；
20. optimizer update。


==================================================
四十、Inference Algorithm
==================================================

也给一个独立算法。

Inference：

不需要Q参与动作选择。

流程：

Observation
→
Verifier
→
Candidate
→
Graphs
→
Nominal Intervention
→
D^K / D^P
→
Actor
→
Skill
→
Execute。


==================================================
四十一、实现架构
==================================================

请详细到Agent可实现。

建议目录：

contracts/
facts/
vlm/
graph/
intervention/
models/
policy/
rl/
env/
evaluation/
configs/
tests/

给每个模块：

- input；
- output；
-主要class/function；
- 是否训练；
- 数据流。


==================================================
四十二、默认实现参数
==================================================

如果没有明显问题：

保留目前默认：

Graph：

R-GCN
4 layers
hidden 128
4 bases
mean aggregation
dropout 0

Actor：

B=0.5

Structural Q：

λ_Q=0.1

PPO：

lr=3e-4
clip=0.2
GAE λ=0.95
value coef=0.5
entropy=0.01
grad clip=0.5
epochs=4
rollout=1024 skill transitions
sequence=16
minibatch=64
Adam

Prior training：

80% Original
20% No-prior。

这些属于：

默认实验配置，

不是方法理论定义。


==================================================
四十三、实验问题重新围绕论文claim设计
==================================================

最终实验应回答：

### RQ1

Candidate structural intervention是否比：

只看当前状态/当前图

更有价值？


### RQ2

固定VLM semantic prior是否能给contract reasoning带来额外决策信息？


### RQ3

dual difference：

\[
D^P=D^H-D^K
\]

是否比直接使用：

\[
D^H
\]

更合理？


### RQ4

Structural Q是否帮助：

representation learning / stability / sample efficiency？


### RQ5

训练后的policy是否真的使用：

D^K / D^P？


### RQ6

prior缺失/扰动时：

policy如何变化？


### RQ7

方法是否具有基本组合泛化？


==================================================
四十四、Baseline体系
==================================================

至少定义：

### B0

Observation / Fact Skill Policy

无graph structural reasoning。


### B1

Current Graph Policy

使用当前图表示，

不做candidate nominal intervention。


### B2

Contract-only Intervention

使用：

G^K
+
D^K

无VLM prior。


### Full

完整 CP-DISR。


### A_DD

使用：

D^H

而不是：

D^P。


### A_Q

\[
\lambda_Q=0
\]


### A_B

去掉bounded prior约束。

公平原则：

- 同Skill Library；
- 同low-level controller；
- 同observation；
- 同reward；
- 同训练预算；
- 高层decision mechanism不同。


==================================================
四十五、当前论文不要承诺“坏prior一定被忽略”
==================================================

实验中应验证：

当前architecture是否真正做到：

### useful prior

能够影响行为；

### unhelpful / misleading prior

影响减弱或发生不同方向的修正。

但是论文在没有实验前只说：

> architecture allows context-dependent utilization。

不要提前写：

> automatically rejects wrong relations。


==================================================
四十六、Actor Dependence Diagnostics
==================================================

训练后对同一checkpoint：

- D^K=0；
- D^P=0；
- shuffle D^P；
- remove prior；
- target-swap relation。

观察：

- action distribution；
- KL/JS；
- argmax change；
- success。

证明：

图和prior不是装饰输入。


==================================================
四十七、Robustness
==================================================

测试：

- Original；
- No Prior；
- Edge Deletion；
- Target-swap；
- Irrelevant。

不需要重新训练。

不要把它写成：

“模型识别并修正错边”。

只评价：

policy behavior under imperfect prior changes。


==================================================
四十八、Generalization
==================================================

优先只做：

compositional generalization。

例如：

seen skill schemas
+
seen predicates
+
unseen object-goal / dependency composition。

避免一次扩成：

视觉、语言、跨机器人、跨域全部泛化。


==================================================
四十九、Expected Results部分如何写
==================================================

因为当前实验还没全部完成：

不要虚构结果。

可以写：

### Expected evidence pattern

例如：

- intervention类方法应在结构复杂任务更明显；
- VLM增益应在存在非冗余soft relation的任务更明显；
- Structural Q可能主要影响early learning或稳定性；
- bounded residual可能主要在prior corruption下体现。

明确标记：

这些是：

hypotheses / expected observations

而不是：

observed results。


==================================================
五十、论文主张强度必须和实际证据匹配
==================================================

如果最后：

Full只比B2略好，

论文可以写：

VLM提供有限但可测额外信息。

如果：

Full主要改善AUC，

强调：

sample efficiency。

如果：

robustness更明显，

强调：

conditional utilization of imperfect priors。

不要让论文主题绑定：

“必须提升10%”。


==================================================
五十一、Related Work结构
==================================================

请设计论文Related Work。

至少围绕：

### 1. Long-horizon robotic skill planning / hierarchical control

### 2. Skill contracts / symbolic action models / structured policy

### 3. VLM / LLM priors for robot planning

### 4. Graph-based relational reasoning

### 5. RL with auxiliary value supervision / structured decision learning

### 6. Imperfect / noisy priors

但不要把当前方法错误定位成：

knowledge graph repair。

需要说明：

我们的重点是：

> candidate-conditioned utilization of fixed semantic prior。


==================================================
五十二、给一版Paper Outline
==================================================

最终论文可以考虑：

# 1 Introduction

# 2 Related Work

# 3 Problem Formulation

## 3.1 Skill-Level Semi-MDP

## 3.2 Skill Contract

## 3.3 Imperfect Semantic Prior

# 4 CP-DISR

## 4.1 Overview

## 4.2 Contract Graph

## 4.3 VLM Semantic Relations

## 4.4 Candidate Nominal Structural Intervention

## 4.5 Contract–Prior Dual Difference

## 4.6 Candidate Policy with Bounded Prior Residual

## 4.7 Outcome-Grounded Structural Learning

## 4.8 Training

# 5 Experiments

## 5.1 Tasks

## 5.2 Baselines

## 5.3 Main Results

## 5.4 Ablation

## 5.5 Mechanism Analysis

## 5.6 Prior Robustness

## 5.7 Generalization

# 6 Discussion

# 7 Conclusion

请判断并优化。


==================================================
五十三、给Introduction v0
==================================================

这一版研究文档不仅给提纲。

请直接写一版接近论文正文的：

### Introduction v0

长度可以：

800–1200中文字符等价内容

或者对应英文论文结构。

重点是：

逻辑完整。


==================================================
五十四、给Problem Formulation v0
==================================================

完整写：

- Semi-MDP；
- observation；
- fact state；
- skill candidate set；
- duration；
- reward；
- policy objective；
- contract；
- prior。

数学符号统一。


==================================================
五十五、给Method v0
==================================================

这是文档最重要部分。

尽量做到：

以后可以直接改成论文Method。

包括：

公式；
定义；
伪代码；
信息流；
梯度流；
权限边界。


==================================================
五十六、给Experimental Setup v0
==================================================

即使具体任务还未全部绑定，

也要写成：

论文实验章节的骨架。

对尚未确定部分：

标：

MUST_BIND。

例如：

- simulator；
- concrete tasks；
- exact hardware；
- VLM model snapshot；
- training budget。

不要为了完整而虚构。


==================================================
五十七、给Results章节模板
==================================================

请直接设计：

### Table 1
Main Results

### Figure 3
Learning Curve

### Table 2
Ablation

### Figure 4
Actor Dependence

### Table 3
Robustness

### Figure 5
Qualitative Case

给每张图/表：

它回答什么RQ；

预计放在哪里；

结果出来以后如何写一句结论。

不要填假数字。


==================================================
五十八、给Discussion v0
==================================================

Discussion正常讨论：

### 当前方法真正学习的是：

prior utilization

而非：

prior repair。


### 固定knowledge source的意义

使生成prior与学习决策解耦。


### 方法边界

当前不显式：

- 修关系；
- 学relation truth；
- 更新VLM。

但不要写成严重缺陷。

写成：

scope。


==================================================
五十九、Future Work必须与当前论文严格分离
==================================================

当前我们已经有一个值得未来研究的升级方向：

### Candidate × Relation Error Learning

核心思想：

不只是：

有用prior → use

无用prior → suppress

而是进一步研究：

> 某条relation如何改变某个candidate的判断，以及这种改变与真实结果之间是否存在可迁移的偏差模式。

例如：

with relation：

\[
q_t^+
\]

remove relation：

\[
q_{t,e}^-
\]

真实结果：

\[
Y_t
\]

关系预测效用：

\[
u_{t,e}
=
(Y_t-q_{t,e}^-)^2
-
(Y_t-q_t^+)^2
\]

未来可以学习：

\[
\hat u_\phi(\chi_{t,i,e})
\]

但是：

### 这个Future M1+绝对不能并入当前Method。

只在：

Discussion / Future Work

简短说明：

当前方法学习prior utilization；

未来可以研究：

experience-backed relation-level error patterns。


==================================================
六十、不要让Future M1+削弱Current M1
==================================================

文档必须明确：

Current M1是独立完整研究问题：

> fixed prior → candidate reasoning → real outcome → decision learning。

Future M1+是：

> relation-specific predictive-error experience。

不是：

“Current M1逻辑不完整，所以以后补洞”。

请写清两者关系。


==================================================
六十一、最终研究一句话解释
==================================================

最后请给：

### 10秒版本

### 30秒版本

### 2分钟版本

分别解释当前研究。

10秒版本要非常简单。

例如：

> VLM和技能合同先告诉机器人“可能有哪些长期关系”，CP-DISR不直接相信VLM，而是对每个候选skill先做一次名义结构推演，再通过真实执行经验学习这些关系什么时候值得影响skill选择。


==================================================
六十二、最终文档结构
==================================================

最终请不要逐条回答上面的问题。

把所有内容重新组织成一份真正连贯的：

# CP-DISR Paper-Oriented Research Specification v3.0

建议结构：

# 1. Executive Summary

# 2. Research Problem

# 3. Motivation

# 4. Paper Positioning

# 5. Core Contributions

# 6. System Assumptions

# 7. Skill-Level Semi-MDP

# 8. Skill Contract

# 9. Fact Verification

# 10. Contract Graph

# 11. VLM Semantic Prior

# 12. Candidate Nominal Intervention

# 13. Shared Structural Encoder

# 14. Contract–Prior Dual Difference

# 15. Policy Architecture

# 16. Structural Q

# 17. PPO / Training Objective

# 18. Forward Information Flow

# 19. Backward Gradient Flow

# 20. Training Algorithm

# 21. Inference Algorithm

# 22. Implementation Specification

# 23. Experimental Questions

# 24. Baselines

# 25. Ablations

# 26. Mechanism Diagnostics

# 27. Robustness

# 28. Generalization

# 29. Experimental Setup Draft

# 30. Paper Results Templates

# 31. Scope and Boundary

# 32. Discussion

# 33. Future M1+ Direction

# 34. Paper Outline

# 35. Abstract v0

# 36. Introduction v0

# 37. Problem Formulation v0

# 38. Method v0

# 39. Experimental Setup v0

# 40. Discussion v0

# 41. Final Frozen Research Snapshot


==================================================
六十三、最终Frozen Snapshot
==================================================

最后再用一页总结：

### Problem

### Core Idea

### Skill Contract

### VLM

### G^K

### G^H

### Nominal Intervention

### D^K

### D^P

### Actor

### Structural Q

### PPO

### Training

### What is learned

### What is frozen

### What the method does

### What the method does NOT do

### Main paper claim

### Main experiments

### Future M1+

每项1–3句话。


==================================================
六十四、写作风格要求
==================================================

整篇文档要逐渐接近论文语言。

不要大量使用：

“我们之前讨论过……”
“上一版……”
“原来方案……”
“这一轮决定……”

除非真的需要解释版本变化。

正文应该像：

**当前方案从一开始就是这样定义的。**

不要主动写成：

方法自我批评报告。

但所有数学边界和claim必须准确。


==================================================
六十五、特别重要：不要继续发散新创新
==================================================

当前目标是：

### 冻结Current Paper。

不是：

再找M2、M3、M4。

所以：

除已经明确单独保留的 Future M1+ 外，

不要主动提出新的：

- trust；
- memory；
- world model；
- planner；
- VLM adaptation；
- 新graph architecture；
- 新reward；
- 新teacher。

如果发现某个真正逻辑错误：

指出。

否则：

保持当前Method。


==================================================
六十六、最后给出“Paper Readiness”判断
==================================================

文档末尾给：

# Paper Readiness

分别判断：

### Introduction

现在可完成多少？

### Related Work

多少？

### Problem Formulation

多少？

### Method

多少？

### Experimental Setup

多少？

### Results

缺什么？

### Discussion

多少？

### Abstract

多少？

最后明确回答：

> **当前CP-DISR是否已经达到可以正式进入论文撰写阶段的程度？**

如果答案是：

是，

请直接写：

> **Current Method is frozen for writing; subsequent experiments validate claims rather than decide whether the paper can be written.**

不要再设置新的前置研究gate。


==================================================
最终目标
==================================================

这一次最重要的是：

**把当前CP-DISR从“研究讨论方案”升级成“论文级方法定义”。**

当前主线已经明确：

> **不是纠正VLM，而是学习如何利用固定的不完美VLM semantic prior。**

Current Paper：

继续CP-DISR v2.1。

Future Upgrade：

独立保留Candidate × Relation Error Learning，
但不阻塞当前论文。

最终文档应该让我能够：

1. 直接开始写论文；
2. 直接交给Agent继续实现；
3. 直接根据文档设计最终图表；
4. 不再反复解释“VLM到底有没有被纠正”；
5. 后续实验只负责验证方法主张，而不是重新决定整套研究是什么。