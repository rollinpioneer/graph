---
title: "CP-DISR C1：双模型交叉审查后的最终研究决策与首张最小验证卡"
date: "2026-10-09"
status: "PLAN_ONLY_AWAITING_USER_APPROVAL"
execution_authorized: false
repository: "rollinpioneer/graph"
base_commit: "511b117495920234cb646016eaf504c7371233aa"
base_branch: "codex/cp-disr-c1-depots-fast-alt-v3"
last_completed_card: "C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3"
proposed_card: "CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1"
new_training_runs_in_first_card: 0
new_external_model_fits_in_first_card: 0
new_problem_generation_in_first_card: 0
storage_owner: "xushijie3"
---

# CP-DISR C1：最终研究决策与最小验证实验

> 本文综合研究状态快照、A（GPT独立方案）、B（Claude独立方案）、RA（Claude对A的审查）、RB（本对话中GPT对B的审查），作出下一项投入决定。本文不是新实验结果，也不表示任何方案已达到CCF-A录用标准。
>
> **本次选择：暂不训练。主推“搜索比较范围与监督范围是否失配”的低成本验证；只有得到可定位、与实际选择有关的证据，才考虑短微调。唯一备选是固定WL深度的精确后继监督适配。** 不自动合并两项，不恢复REL，不开始新编码器或新领域。
>
> **授权边界：批准首卡后，只执行本文第二部分的零训练诊断。第二、三层以及备选WL拟合均需要新的明确批准。低分不是技术失败；证据不足是合法终点。**

## 阅读导航

- 第一部分：研究负责人决策、分歧处理、三层实验路线与论文风险。
- 第二部分：Coding Agent可直接使用的首卡规格、配置、记录格式和验收要求。
- 第三部分：固定版本证据与材料身份。

---

# 第一部分　最终研究决策

## 1. 当前结论：保留哪些事实，撤回哪些过度解释

### 1.1 当前工作对象

仍研究已知动作合同下的状态／动作评价：训练目标结构简单，评价时目标组合、对象规模或世界配置可能变化。当前工作评分器是冻结Depots DENSE-G（更新4100），部署为共同eager GBFS中的状态启发值。评分越小，OPEN优先级越高。[S1][G1][G2]

当前不是恢复机器人部分观测研究，也不是发展一个未知动力学模型。本文不改变任务领域或动作语义。

### 1.2 三项已被本项目记录支持的事实

| 事实 | 适用范围 | 不允许扩大的解释 |
|---|---|---|
| 同搜索中DENSE在Joint128全部完成，平均计划长度比1.095、平均展开24；WL为1.219、133 | 同一Depots世界配置、项目划分、单种子，两个完整训练配方 | 不是纯注意力因果证明，也不是公开IPC全面领先 |
| FAST吞吐约1.74倍，IPC仍14/22；p06/11/12/17相对WL已有明显节点引导劣势 | 固定评分器、同搜索、开发数据 | 不能说只差速度；也不能说无限时间一定解不出 |
| ALT 1:1没有合理净收益；REL跨Depots没有超过DENSE | 已运行的具体配方 | 不否定全部搜索组合、关系学习或计算共享家族 |

来源：[S1]第3.10—3.12节，[G1][G2]。

### 1.3 本轮代码核对进一步确定的事实

当前PDDL训练器使用 `SerialTrainer(..., "ACTION", ...)`；动作分数为

\[
\ell(s,a)=V(s,G)-V(T(s,a),G).
\]

同一父状态中的softmax会消去共同项 \(V(s,G)\)，两动作分数差也会消去它。因此，**当前ACTION损失没有显式父—子比较，也没有显式跨父状态比较**。源码：[C1][C2][C3]。

这不等于：

- 父状态的网络参数永远没有梯度；参数共享及该状态在其他样本中作为后继，仍可更新它。
- 不同父状态的分数完全自由或独立；共享网络、共享后继和训练图会施加间接约束。
- 该缺口已被证明是IPC失败的原因。

另一方面，固定GOOSE版本的计划排序数据**包含父状态、计划选定后继、其他合法后继**。父状态不是被完全忽略，兄弟后继也不是从未出现。WL缺少的是与DENSE相同的完整精确后继顺序及偏离状态覆盖，而不是“从来没看反例”。[C4][C5]

### 1.4 当前不得当作已证结论的内容

- IPC引导失败主要由跨组水平漂移导致。
- 托盘稀缺或某一拓扑轴是唯一原因。
- 提供更充分标签后，WL一定追平或一定追不上DENSE。
- DENSE超过增强WL就证明注意力／表示贡献，WL追平就只能写分析稿。
- 颜色细化的理论可共享量等于净墙钟加速，或DENSE最多只能因此多解一道题。

---

## 2. 四份材料如何定位

| 材料 | 内容 | 本次用途 |
|---|---|---|
| A | GPT：前沿可证排序纠错、紧凑对象中介、精确共享 | 方法候选及原始最小实验 |
| B | Claude：监督匹配WL、世界偏移分轴、停滞组合 | 基线风险与其他候选 |
| RA | Claude审查A：监督范围、简单题饱和、最近邻文献、失败题定位 | 修正A首卡及训练边界 |
| RB | GPT在本对话中审查B：基线实际代码、归因不足、世界轴耦合、组合成本 | 修正B结论与实验规模 |

RB目前是本对话中的完整审查正文，没有单独上传的RB文件；不得虚构其仓库路径或附件哈希。本文已经记录影响决策的RB观点。

---

## 3. 共识与分歧清单：逐项裁决，不取平均

| 编号 | 分歧／共识 | 类型 | 哪项证据更充分 | 本次裁决／仍需什么 |
|---|---|---|---|---|
| D1 | 保留原研究方向，不再无依据叠模块 | 价值判断上的共识 | 历史资产与连续负结果 | 采纳；Depots和现有冻结模型不变 |
| D2 | DENSE当前是否显式学习父子／跨父比较 | 事实 | RA的代数分析、现行PDDL训练路径 | 已解决：没有这些显式项；因果作用未解决 |
| D3 | WL是否只看到计划状态，缺少兄弟反例 | 事实 | RB直接读取固定作者代码 | 已解决：它看到父、计划后继和其他后继；不能沿用B的绝对说法 |
| D4 | 先做WL监督适配，还是失败题错误定位 | 决策价值 | B/RB说明基线混淆；RA说明A的采样区间不对 | 本次先定位真实失败；WL适配为唯一备选，后续正式比较仍必要 |
| D5 | 24道新简单题能否否定前沿错位 | 实验解释 | RA的饱和观察；但未在本轮独立复算全部waste脚本 | 拒绝作为放行门；改用已知失败题，简单题只作负控 |
| D6 | 多余展开为0是否说明不存在排序／质量问题 | 实验解释 | 引擎生成时判目标语义与数学 | 只能说明没有额外扩展最终返回路径外的节点；返回计划仍可能不最优，评分生成开销也不为0 |
| D7 | 前沿排序是否具有很强的首次创新 | 文献 | RA指出NeurIPS2023直接近邻；本轮核查其定义1及式1—4 | 已解决：普通OPEN排序不是新机制；若继续，必须超过已有排序目标／普通数据增加 |
| D8 | 已知完整最优参考长度是否能支持Joint8任意状态精确标签 | 事实／工程 | 原训练标签覆盖只含64个6箱Joint任务的完整oracle；根部参考长度与任意状态查询不同 | 拒绝默认ExactOracle能轻易处理8箱；首卡不新建8箱完整状态空间 |
| D9 | T2能否用Joint8、高4、多堆题训练 | 研究定义 | RA第6.6节与当前受限训练主张冲突 | 本轮拒绝；诊断题永不进入训练。若未来放宽训练结构，应明确更改主张并重新批准 |
| D10 | p11/p14等“只差托盘”是否证明拥挤度原因 | 实验解释 | 对象统计只是相关证据；初始布局／目标绑定未被隔离 | 未解决；不据此直接训练。托盘数、初始堆高和可行目标形状互相制约 |
| D11 | CAL旧负结果是否已否定跨组监督 | 历史解释 | RA指出部署从一步／两步变为GBFS；旧结果和代码 | 未否定。不能把CAL损失原样搬回就称新方法；必要对照须匹配新部署 |
| D12 | 紧凑/lifted图是否必然丢掉合同 | 文献事实 | RB与AAAI2024原论文：有动作模式子图、前提及效果 | 已解决：不必然丢失；但高工程成本使其本轮暂缓 |
| D13 | 精确共享是否确定有速度收益、覆盖最多+1 | 实验解释 | RA提供无权重结构统计，但未测端到端开销，样本来自WL路径 | 未解决。仅记录“报告的潜在复用率”；不把理论比率称为已测加速／覆盖上限 |
| D14 | R1/R2/R3比例能否直接判定H1成立、H2主导 | 因果解释 | 证书条件、选择偏差、多解及OPEN动态 | 拒绝硬因果标签；只输出定位信号、混合证据或证据不足 |

**文献的重要补充：**NeurIPS2023已经把计划状态与同一时刻OPEN的非计划状态比较；它并不要求相邻计划状态之间一定比较。这提示我们：父子项缺失是一条具体事实，但“每一步V必须严格下降”不是高效GBFS的一般必要条件。[P1]

**材料读取边界：**RA中“Struct30/32和Joint96/128零多余展开”、颜色共享表、B的动作分解与托盘配对，由审查者报告。本文不宣称已经运行其 `notes/` 脚本独立复现；首卡不依赖这些精确比例作为完成门槛。

---

## 4. 候选比较与最终取舍

| 候选 | 信息价值／可复用性 | 新增贡献目前处于何种状态 | 成本与主要风险 | 本次决定 |
|---|---|---|---|---|
| 搜索比较范围监督纠错 | 直接对应代码缺口与失败搜索；能用冻结值、旧计划诊断 | 有条件的方法候选；普通跨组排序已有强近邻 | 证书可能稀少；局部错与跨组错可能同时出现 | **主推，但当前只批设计、不训练** |
| WL精确后继监督适配（固定L2） | 现成标签、便宜拟合，检验重要基线混淆 | 主要是更强对照，不是新算法 | 状态覆盖、词表、损失和权重仍不同；不能一次完成归因 | **唯一备选，不在首卡运行** |
| 世界偏移五格＋检查点矩阵 | 可分解数据分布，但原设计量大且耦合 | 数据干预／评价设计，不自动形成模型创新 | 约千次搜索；低托盘与低目标堆约束可能不可同时满足 | 不执行 |
| 对象中介紧凑模型 | 方法潜力较高，可针对成本和资源绑定 | 未验证结构候选 | 重写编码与重新训练；无法用首卡结果直接证明必要性 | 暂缓，不列为本轮备选 |
| 精确共享编码 | 有可核对的代数目标 | 潜在算法性优化，实际成本未知 | 不改变固定评分的节点排序；签名维护可能抵消收益 | 暂缓 |
| 停滞后DENSE/WL组合 | 容易兼容但与失败ALT并不完全等价 | 成熟系统机制 | 并集不是实际组合成绩；队列与成本未完全定义 | 不执行 |

### 为什么不继续沿用RB的“先WL”排序

RB是在没有RA失败分布与最近邻修正时作出的阶段决定。现在RA提供了更直接的问题：A首卡在容易成功的小题上可能缺乏判别力，DENSE部署则恰好消费未直接监督的比较。先定位失败，能决定下一笔神经训练是否有明确对象；先拟合WL主要决定旧正结果的解释和对照强度。

因此本次排序调整为：**一次有限失败定位 → 若有信号，再短微调；若证据不足，则停止神经训练提议，只保留CPU-WL适配作为备选。** 这不是把两条路线拼成一个“联合创新”。

---

## 5. 主推方案与可证伪假设

### 5.1 主推方案的功能定义

研究“状态评分的训练比较范围是否应该覆盖实际搜索中的竞争关系”。保留当前DENSE结构与共同GBFS；有必要时仅增加同问题内、可靠的父子／跨生成组排序监督。

后续候选最小形式为：

\[
\mathcal L = \mathcal L_{\text{原ACTION}}
+\lambda\frac{1}{|\mathcal P|}\sum_{(i,j)\in\mathcal P}
\operatorname{softplus}\!\left(\frac{V(s_i,G)-V(s_j,G)}{\tau}\right),
\qquad d^*(s_i)<d^*(s_j).
\]

它是**已知排序学习路线内的受控改进**，不是本轮已经成功的方法。首卡不实现该训练损失。

### 5.2 三个层次的假设

| 层次 | 假设 | 哪层能检验 |
|---|---|---|
| H-code | 原ACTION损失没有显式跨组约束 | 已可由代码与代数判断 |
| H-locate | 在DENSE真实失败的搜索中，已经可选且可证剩余代价更低的其他分支被持续延后；不只是参考路径上的局部选错 | 首卡诊断提供支持／不足／反向证据，不提供唯一因果结论 |
| H-intervene | 保持结构、训练题和预算不变，补充明确的跨组监督能改善部署，同时保住旧能力 | 第二层匹配微调 |

最终更强的“预算内前沿定向采样优于普通跨组采样”只在第三层检验；第一层不得据错排数量直接宣布该创新有效。

### 5.3 最大风险

- **论文风险**：只是补上已知排序约束，或只是多了数据。与Chrestien2023、Hao2024/2025的实质差异可能不足。
- **科学风险**：局部状态表示错误／世界配置外推占主导，补跨组约束无法修复。
- **测量风险**：已知参考计划并非最优，LM-cut较松，能证明的比较过少；不能把未证明当正确。
- **执行风险**：分数偏好与总体计划代价并非完全一致；修复离线排序可能损害已强的Joint搜索。

---

## 6. 三个决策层级：不能自动升级

### 第一层：冻结失败题的比较范围定位

- **目标**：判定H-locate是否有足够的、数值稳健的实例证据，识别局部／链式／跨生成组比较的差别。
- **操作**：最多10题、DENSE和WL；验证参考计划，短搜索前缀，只读记录OPEN；用可靠上下界产生诊断证书。
- **资源**：0模型训练、0WL拟合、0新任务；最多20条主前缀、8条很短的固定夹具。总GPU进程占用墙钟累计不超过1800秒，标签CPU累计不超过3600核秒，详见首卡配置。
- **成功定义**：不是“解出更多题”，而是技术有效且出现分散的跨组错排见证，达到预先给定的进入短微调的筛选线。
- **停止定义**：资产／语义问题、达到资源上限、证书不足、只观察到局部错误、数值并列主导等均收口；不改指标凑通过。
- **论文意义**：定位可检验机制，不是新方法有效性结论。

### 第二层：最多两条短微调，只测“补比较范围”是否有用

仅在第一层输出 `CROSS_SCOPE_SIGNAL`、训练数据可用，且用户另行批准后讨论实施：

| 条件 | 数据与初始化 | 训练内容 |
|---|---|---|
| T0 | 同一冻结DENSE起点、原Train96、同一批次和预算 | 原ACTION损失继续训练 |
| T1 | 与T0相同 | 原损失＋原训练题内的父子／跨生成组可靠排序 |

- **上限建议**：每条512次更新；不因损失未收敛续到4100。选模只在128/256/512三个点，规则相同。
- **训练边界**：禁止使用IPC、Joint、Struct的诊断状态／证书；禁止用高4、多堆或8箱题训练。原训练题中的合法偏离状态属于训练问题，可用已有精确Oracle标注，但新增状态、标签和计算必须记账。
- **匹配条件**：T0/T1使用同一状态池、编码调用与旧任务展示预算；T0不使用新增比较标签。两者都要记录实际梯度与状态出现次数。不能把对照的零损失解释为它“理论能力差”。
- **最小开发板建议**：既有Struct中固定16题＋Joint固定32题（8格各4题）＋首卡6个IPC开发题，合计54题。按题ID哈希选取而非挑结果；T0、T1和冻结基线使用同一60秒／4096展开协议。旧300秒结果不能直接填入该表。
- **进入正式阶段的工程筛选线**：T1对T0在共同成功题的展开量几何平均比≤0.85，至少5题有非并列修复；54题总完成数不能少超过1题；48个内部题的共同成功长度比均值增幅≤0.03。冻结模型也必须同时报告。各项为开发投入线，不是显著性或非劣效证明。
- **停止**：只降低新损失、不改善执行；仅T0/T1共同改善；明显损害简单能力；增益只出现在已经看过的个别IPC题。分别记录，不自动追加新头、温度、种子或数据。
- **论文意义**：即使成功，也首先是“补监督范围有效”，不主张新的前沿挖掘算法。

上述数字是下一层预算与决策概览，并非本首卡的执行配置；若获批层二时要改，必须在任何训练或新评测前重新固定。

### 第三层：仅当前两层均支持时，检验真正的算法增量

不是直接扩大训练总量，而是在同一训练问题、同标签预算内比较：

1. 原ACTION／同步数继续训练；
2. 已有范式的通用父子和OPEN排序；
3. 面向实际竞争前沿的可靠冲突采样。

主要比较第3对第2。数据池、标签质量、距离间隔分布与参数预算匹配，才讨论选择机制的额外价值。若第2≈第3＞第1，保留普通范围补全，放弃“新采样机制”的主张。

正式阶段还需要：

- 固定L2、精确后继监督适配的WL强对照；原作者模型保留。拟合计入训练，不称零训练。
- 同一GBFS、相同节点档位和真实时间成本；参考值和已证明最优分开。
- 一份方法冻结后生成的独立Depots结构／世界分层测试，不再拿IPC22作盲测。
- 在已有Blocksworld资产上进行相同机制的领域内重训／评价；不声称积木权重零样本迁移到Depots，或历史结果已经证明新机制跨域。
- 开发仍可单种子；若主方法与最关键对照已有一致收益，再单独批准补至3个种子，不能把3视为自动可信门槛。

**第三层没有被授权，也不允许Agent自行展开。** 当前没有批准一次覆盖所有正式臂、多个种子和两域的训练矩阵。

---

## 7. 唯一备选：固定L2的WL精确后继监督适配

若首卡无法支持继续神经微调，可单独批准一个CPU拟合：保持ILG、L=2、作者rank-SVM与共同搜索，使用原训练题的完整精确后继排序；不同时扫L4、换合同图、扩大世界或重训DENSE。

它回答“便宜的更充分监督基线能否复现主要优势”，而不是“论文是否只能变成分析稿”。状态／标签构造、沿计划父子约束是否保留、去重与权重必须在该备选卡中先固定。

此备选不在首卡运行。也不把它强行称为主方法创新。

---

## 8. CCF-A贡献潜力：客观判断

目前没有足够证据批准“一定可形成CCF-A方法”的完整训练投入。

潜在可成立的链条是：

> 明确的部署比较错位 → 可验证的错排见证 → 同结构、同状态与预算下的监督改进 → 超过普通排序／更多数据对照 → 在新目标结构和第二领域保留收益。

其中第一步的代码事实已支持，第二步未完成，后面更未完成。NeurIPS2023及IJCAI2024已有直接排序先例，所以“加跨组loss”本身不够；但这也不是放弃该问题的理由。它应当先被验证为有用组件，再根据真正的额外能力界定贡献。

本次拒绝的是**无依据立即训练**，不是否定成熟方向。首卡无信号时就停止该假说的当前配方，不能修改样本和阈值直到出现正结果。

---

# 第二部分　Coding Agent首卡

## 9. 卡片身份与不可扩展的授权范围

```yaml
card_id: CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1
status: PLAN_ONLY_AWAITING_USER_APPROVAL
execution_authorized: false
base_commit: 511b117495920234cb646016eaf504c7371233aa
proposed_branch: codex/cp-disr-c1-search-scope-diagnostic-v1
mode: frozen_model_readonly_diagnostic
new_neural_trainings: 0
new_wl_fits: 0
optimizer_steps: 0
new_problem_generation: 0
new_diagnostic_certificates: permitted
use_diagnostic_certificates_for_training: false
allow_auto_layer2: false
allow_auto_backup: false
```

批准后可以在新分支增加只读诊断适配、测试、日志与报告。不能修改原模型权重、原训练标签、原搜索结果、外部模型或旧配置。不能启动拓扑训练、短微调、WL拟合、新世界生成、ALT比例扫描或新编码器。

## 10. 资产与环境核对

### 10.1 历史路径与固定身份

```text
历史仓库根：/home/xushijie3/work/graph_cp_disr
历史Python：/home/xushijie3/envs/cpdisr/bin/python
存储所有者：xushijie3
```

这些是已有运行清单中的路径，不是本次在线确认。Agent必须在服务器核验；若路径不同，明确记录实际路径，不能猜一个存在的xushijie2目录继续写。

冻结DENSE：

```text
selected_update: 4100
sha256: ffaad0f4e0e536afbef3f66109e3ff4f36563212fd428762436bb3f5305bcd38
```

主要资产目录（仓库相对路径）：

```text
runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e/
runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1/
runs/final_master/c1_route_b/depots_fast_alt_v3/20261008T191927Z_fastalt/
```

读取 `runs/lock.json`、各卡运行清单、旧WL模型身份、PDDL与计划哈希；权重不在Git中，实际路径只能从服务器清单解析。不指定未经确认的`.pt`文件名。

### 10.2 工作区规则

- 建新分支／独立worktree，不切换或破坏正在运行的旧工作区。
- 保留根目录已有未跟踪 `sas_plan`，不删除、不提交；新临时文件放新run目录。
- 只显式暂存本卡代码与轻量结果，不使用仓库级通配`git add .`。
- 不强推、不合并、不创建PR；不修改旧回执以消除历史偏差。
- 未经批准不终止其他人或旧实验；本卡预先批准的子进程限时终止属于预算控制，必须记账，不得扩展到其他进程。

### 10.3 复用接口与需要新增的部分

复用：

```text
src/cp_disr/pddl/task.py
src/cp_disr/pddl/data.py
src/cp_disr/pddl/train.py
src/cp_disr/pddl/search_match/engine.py
src/cp_disr/pddl/search_match/evaluators.py
src/cp_disr/pddl/fast_alt/fast_value.py
src/cp_disr/pddl/fast_alt/evaluators.py
```

已核对：旧 `observer(s, expanded)` 只能看到被弹出的状态，**没有OPEN内容**。不能假称原接口已足够完成R3。需要新增只读事件观察接口，或本卡局部搜索包装，保留原语义。须通过第18节夹具。

现有 `evaluators.py` 已有SAS转换、状态映射和 `sas_with_init`；它没有已证实的任意状态LM-cut接口。先实现／核验一个只读取根启发值的后端，再进入正式采集。若需要大规模重写外部规划器，按工程上限停止，不自动扩张。

---

## 11. 题目集合：固定失败区间，保留正反对照

### 11.1 IPC主组

| 组 | 题目 | 历史角色 | 参照计划 |
|---|---|---|---|
| F | p06、p11、p12、p17 | DENSE节点引导劣于WL的4题 | 截至base commit已保存、可回放的最短计划，不限生成它的方法 |
| R | p14、p15 | DENSE能解、WL在旧共同协议失败的反向组 | 同一规则选已保存计划 |

“历史角色”不强制本次短前缀重复相同成功／失败。结果按实际记录，不因本次意外解出而换题。

### 11.2 小问题负控组C：最多4题

从**已有Train96**选3／5箱×需运输／无需运输四格，每格按 `SHA256("scope-diag-v1|case_id")` 最小者1题。要求原精确Oracle资产可访问；题ID在任何新模型评价前冻结。

选择Train而非新小题，是为了用已知简单分布验证证书实现，不把其低错误率当作主假说反证。若原Oracle缓存不含任意状态查询，只可在这些小问题上按第15节预算重建。某格无资产则记录缺额，禁止改用看起来更有错误的题替换。

**不使用RA提议的8箱高4任意状态作为“已免费精确标签”。**已有最优参考长度并不等于已保存完整距离表。若某条已证明最优计划存在，它的路径后缀可提供精确距离，但这不扩展到该路径所有兄弟或OPEN状态。

---

## 12. 参考计划与状态身份

1. 只从base commit对应的既有运行资产中挑计划；不先运行一个新规划器以寻找更有利的参照。
2. 选择规则：经本任务语义回放有效者中长度最短；并列按文件SHA-256、再按规范化相对路径排序。
3. 记录完整动作ID、状态序列、domain/problem哈希、单位代价、最终目标检查。
4. 若计划不是已证明最优，明确 `optimality=UNKNOWN`。
5. 对参考状态 \(w_k\)，剩余计划长度是 \(U(w_k)\)，不是默认的 \(d^*(w_k)\)。同状态在计划中重复出现时取最短有效后缀，并保存证据来源。
6. 计划状态序列供离线诊断，不给搜索器提供优先级、额外剪枝或提示。

所有状态以 `(domain_sha, problem_sha, dynamic_bitmask_hex)` 标识。JSON保存十六进制字符串，加载时显式以16进制转整数；不沿用此前hex/int混淆。目标、静态事实、动作代价和翻译身份进入缓存键。

---

## 13. 冻结搜索与采样协议

### 13.1 搜索条件

每题各运行一次冻结DENSE-FAST和当前作者WL评分，共最多20条主前缀。

- 单OPEN，以原 `(h, insertion_serial)` 排序。
- CLOSED不重开，OPEN发现较短路径则更新父链接与g。
- 规范动作顺序；生成时检测目标；同父的新后继一起评分。
- 不使用C3/G1、两步前瞻、ALT、学习门控、参照路径引导。
- 没达到节点档位就超时，记录截断，不能补称同节点比较。

### 13.2 每题预算

```yaml
ipc_prefix:
  max_expansions: 4096
  wall_seconds: 60
  memory_gib: 8
control_prefix:
  max_expansions: 128
  wall_seconds: 15
  memory_gib: 8
main_traces_max: 20
fixture_traces_max: 8
fixture_max_expansions: 16
```

不再承诺“一分钟一定达到WL解题节点数两倍”。4096与60秒取先到者；不同条件的完整节点证据范围分别报告。首卡不以谁解出为主指标。

### 13.3 OPEN事件采样

每条trace在以下预算比例位置记录不超过32个快照：

\[
I=\operatorname{unique}\left\{\left\lceil N_{cap}^{j/31}\right\rceil:j=0,\ldots,31\right\}.
\]

在弹出下一个有效节点**之前**记录，此时它与其他竞争节点都仍在OPEN。

每个快照保留：

- 实际将被弹出的状态及入队分数、插入序号、当前g；
- 尚在OPEN、未进CLOSED的参考路径状态中，已知有效后缀最短者；并列按状态哈希；
- 另最多3个按固定状态哈希选择的OPEN竞争者；不能按模型错得多来事后选；
- 首次生成父状态、当前最短路径父状态、生成动作、创建组；
- OPEN/CLOSED大小及参考路径最深已生成、最深已弹出位置。

不得把“曾经进入OPEN的最深参考状态”一直当作仍可选择的竞争者。已弹出／已关闭的状态必须退出锚点候选。没有参考锚点就记录 `NO_ACTIVE_REFERENCE_ANCHOR`，不是“没有排序错误”。

新增观察可增加运行开销，故本卡不宣传速度提升；日志开销计入预算。不要每次展开对整个OPEN再跑网络。非驱动评分器在冻结后的采样状态上离线评分。

### 13.4 参考路径局部采样

每题参考计划选择最多16个父状态位置，按位置等距取样并去重。对每个位置：

- 记录父及计划下一状态，供链式比较。
- 生成全部合法后继，DENSE和WL在同一集合打分，保留原始候选排序。
- 距离界标注对象最多8个后继：计划后继、DENSE首选、WL首选，其余按固定哈希补齐。
- 多动作通向同一状态时共享状态界，保留所有动作ID。

参考下一动作没有被选择不自动算错；只有可靠代价界能够证明的比较才计入“可证错误”。

---

## 14. 诊断标签：上界、下界、未知

### 14.1 定义

\[
L(s)\le d^*(s,G)\le U(s),\qquad
U(x)<L(y)\Rightarrow d^*(x,G)<d^*(y,G).
\]

- 优先复用有效参考后缀作U。
- 小负控题使用原ExactOracle时，L=U=d*。
- IPC用LM-cut作L；采用与任务一致的单位动作代价、无不兼容变换。`h_add`和目标计数不是本卡的可采纳下界。[P4]
- 无计划上界时U=∞；界重叠时 `UNKNOWN`。
- 状态距离无穷只有在后端提供可靠不可解证据时才单独报告；不得把超时标成不可解。

### 14.2 LM-cut后端与语义检查

复用已安装Fast Downward／Scorpion的兼容接口；允许SAS更换initial state读取根LM-cut值，或等价只读服务。必须：

1. 核验每个状态合法、与原静态事实一致；所有SAS变量映射覆盖且无歧义。
2. 保留目标与动作代价，不根据待评状态重写任务目标。
3. 在小Oracle题上验证 `0 <= L(s) <= d*(s)`，并在已知目标状态验证0。
4. 同状态重复查询一致；状态批次与逐个查询一致。
5. 不新增有利剪枝或抽象代价变换；引用外部可采纳性条件，而非仅靠少量测试声称普适证明。

没有可核验后端时，写 `LABEL_BACKEND_BLOCKED` 并停止本卡；不悄悄用h_add代替，不为跑满主卡无限改造外部软件。

### 14.3 有界的额外U查询

允许在**搜索采样全部冻结以后**，对至多64个状态请求一次短可行续接，以增加证书覆盖：

- 优先顺序按 `(case_id, category, state_hash)` 轮转，按问题均分；不根据分数差挑状态。
- 每次最多5秒CPU墙钟、1线程；总CPU预算另受3600核秒约束。
- 计划必须回放；成功只提供上界，失败仍UNKNOWN。
- 使用已安装的一个固定规划配置，在登记中记录；本卡不调规划器配置。
- 这些查询不改变已记录的搜索轨迹，也不进入任何训练。

上界来源（旧后缀、新有界计划、精确Oracle）必须分别报告，防止把证书选择偏差隐藏掉。

---

## 15. 总资源上限与“不足即收口”

```yaml
global_caps:
  gpu_process_wall_seconds: 1800
  label_cpu_core_seconds: 3600
  unique_bound_states: 4096
  additional_upper_bound_requests: 64
  upper_bound_request_seconds: 5
  concurrent_cpu_label_workers: 4
  per_worker_threads: 1
  per_worker_memory_gib: 8
  simultaneous_gpu_workers: 1
  new_problem_count: 0
  oracle_rebuilds_large_joint_or_ipc: 0
```

GPU预算含本卡预热、路径／离线评分、主前缀和GPU夹具；CPU预算含翻译、界查询、计划搜索和小题Oracle重建，不能只记求解器内部用时。并行累计按核秒，不拿并行墙钟替代总CPU消耗。

这些是上限，不是“预计只用这么久”的承诺。若每状态启动外部进程的固定开销很高，少量状态耗尽预算也应报告，不能删去启动开销或自动扩容。

最多4096个bound state须在执行标注前按每题、R1/R2/R3类别公平分配。达到上限，后续条目写 `BUDGET_UNLABELLED`。标签信息量不够是预料中的合法阴性结果。

---

## 16. 读数定义：只测能证明的内容

### 16.1 R1：链式比较

对路径父子 \((w_k,w_{k+1})\)，仅在 `U(child)<L(parent)` 时有可靠严格距离关系。分别记录评分顺序：正确、明显倒置、数值并列。

R1不是充分的“GBFS应该改善”指标，也不是GBFS高效的必要条件。它检验一种当前训练没有显式施加的约束。

### 16.2 R2：兄弟局部比较

在同一父状态下：若 `U(better)<L(worse)`，但模型选择的状态为可证更差者，记录局部选择错误。

- 只对实际首选与可证更好候选的关系统计“决策级错误”；额外全配对错误另列，避免重复放大。
- 保留所有最优／等价选择可能性，不强制模仿唯一参考动作。
- 按动作模式分类，仅为描述。不能因drive/load频繁就认定架构缺资源推理。

### 16.3 R3：实际OPEN竞争

对快照中将弹出的s和仍在OPEN的候选t：

```text
若 U(t)<L(s)，且 t/s 不属于同一已记录生成组：
    记录跨生成组的“可证剩余代价更差者先出队”见证。
若有共同生成父状态：
    分类为SIBLING_IN_OPEN，不进入纯跨组计数。
生成父状态信息不足：
    分类为GROUP_UNKNOWN，不冒充跨组见证。
```

状态可以有多个父状态；必须保留已记录的首次生成组和路径更新，不能把某个父指针当作该状态唯一可能的父。

额外记录一个更强、但可能更稀少的证书：

\[
g(t)+U(t)<g(s)+L(s).
\]

前一式只证明剩余代价顺序；后一式在当前父链下还证明已知完整方案代价界更好。两者不能混称“当前选择必然导致更长最终计划”。

### 16.4 分母与不确定性

每题、每trace、每R类都至少报告：

- 预定样本数、实际采到数、缺锚点数；
- 有完整映射的状态数；
- 已证严格顺序数、等距可证数、未知数；
- 明显评分倒置数、数值并列数、正确数；
- 不同根状态／不同OPEN事件／不同状态对数量；
- 参考计划来源、L/U来源与证书覆盖率；
- 同一见证上DENSE与WL的偏好。

**禁止只报“已证对里有多少倒置”而不报未知比例。**上界只覆盖参考计划时，证书选择有偏；可证错排计数是观察到的见证，不是所有OPEN错误率的无偏估计。

简单负控没错不能否定失败组假说；失败组没有足够界也不能证明模型正确。一个参考锚点被100次延后，要同时报告“100个事件”和“1个被延后状态”，不能当100个独立问题。

### 16.5 数值规则

- DENSE明显分离容差采用 `1e-5 + 1e-6*max(abs(v_i),abs(v_j))`；并列／近并列单列。
- 实際OPEN比较仍使用原始浮点与原插入序号，不为了诊断换成容差比较。
- WL保留原适配的rounding；不能拿未取整预测解释实际队列优先级。
- R3优先用入队时真实使用的分数；离线新前向只作交叉检查，不覆盖搜索中的原值。
- 不由有限数值通过再次声称全程轨迹等价。

---

## 17. 预先固定的科学结果标签与后续分支

这些标签是**研究投入筛选**，不是统计显著性、因果分类或方法族生死判决。

### 17.1 `CROSS_SCOPE_SIGNAL`

需同时满足：

1. F组至少2道题，每道出现至少8个不同的DENSE出队事件，存在仍在OPEN的、非同一已记录生成组的可证更好状态；
2. 每道至少涉及4个不同的被延后状态，且明显倒置而非只靠近并列；
3. 见证来自实际搜索，不只是沿参考路径离线比较；
4. 资产、计划、状态映射与界检查均通过。

不要求把R2压到10%以下，也不将“R2低”在低覆盖情况下称为局部能力正确。R2若也有大量见证，另标 `MIXED_LOCAL_AND_CROSS`；此时只支持两种问题并存，不支持跨组主导。

输出：**可以讨论层二匹配短微调；仍需用户批准。** 不输出“H1已成立”。

### 17.2 `LOCAL_WITNESS_PRESENT_NO_CROSS_LOCALIZATION`

F组至少2题，各有至少6个不同父状态上的R2可证选择错误，但没有达到跨组见证线。

输出：不放行层二跨组微调。报告跨组证书覆盖是否足够，不能由此写“前沿纠错必然修不了”。本轮不自动开始拓扑训练或新编码器。

### 17.3 `INCONCLUSIVE_CERTIFICATES` / `REFERENCE_ANCHOR_MISSING`

F组多数题无足够活跃参考锚点、界重叠或后端预算耗尽。输出测量失败／覆盖不足，拒绝当前神经训练申请；不能叫算法假说被证伪。唯一备选WL拟合可供用户另行决定。

### 17.4 `NUMERICAL_TIE_DOMINATED`

多数潜在见证只处于并列容差内。先报告数值与队列处理的限制；不以微调“纠正跨组语义”解释，不能自行改精度／并列规则救结果。

### 17.5 `NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX`

读取范围与证书覆盖较完整，但没有达到明确见证线。只对这组前缀降级当前解释；不得宣称所有前沿类方法无效。达到预算即停止，不延长到4万展开寻找正例。

**WL读数的用途**：同一组状态上的比较可以增强或削弱“DENSE特有比较弱点”的解释。WL与DENSE的R值相近，不是自动停止所有跨组方法的逻辑证明，因为两者真实搜索轨迹不同；同路径读数不等于同策略续接。

---

## 18. 必要测试，不扩成历史审计

最多8条短搜索fixture，全部计入资源账本；不回放全部历史卡。

1. **无副作用观察器**：在确定性假评分／冻结WL的同一小题上，有无观察器的出队序列、最终计划一致。
2. **OPEN锚点生命周期**：已弹出、已关闭的参考状态不能仍被算作可选；未出现锚点应返回UNKNOWN。
3. **生成组与路径更新**：同一状态多个父状态时不丢失来源；同父兄弟不能进入纯跨组统计。
4. **状态序列化**：hex字符串往返、静态事实与SAS变量映射均正确。
5. **证书方向与未知**：`U(x)<L(y)`正确；等号不当严格顺序；重叠不当相等；无穷与超时不混。
6. **可采纳性小题检查**：LM-cut不超过已知d*，目标值0，验证计划长度不小于d*。
7. **同样最优动作／近并列**：不因参考动作没被选就判错；数值并列单独输出。
8. **授权与预算**：无优化器、无fit、无新问题；限额触发只结束本卡自有子进程，回执计数不重读合并日志。

源代码算式不变性可以用合成分数验证：对所有合法动作共同加常数，NLL与成对差不变。这是公式夹具，不是模型训练结果。

---

## 19. 结果文件与记录格式

新结果根（待未来实际执行时生成）：

```text
runs/final_master/c1_route_b/search_scope_diagnostic_v1/<UTCtimestamp>_<registration_shortsha>/
  plan/runbook.md
  registration/config.yaml
  registration/asset_identity.json
  registration/case_manifest.json
  registration/source_identity.json
  checks/fixture_results.json
  references/plans.json
  states/states.jsonl.gz
  traces/<case_id>/<driver>.jsonl.gz
  labels/bounds.jsonl
  labels/certified_pairs.csv
  results/per_problem.csv
  results/diagnostic_summary.json
  results/resource_accounting.json
  results/final_summary.md
  results/claim_boundary.md
  results/method_decision.json
  results/verify.json
  receipts/final_receipt.json
```

### 19.1 bounds.jsonl

```json
{
  "case_id": "ipc_p06",
  "state_hash": "...",
  "state_hex": "...",
  "problem_sha256": "...",
  "goal_sha256": "...",
  "cost_model": "unit",
  "lower": null,
  "upper": null,
  "lower_source": "LM_CUT|EXACT|UNAVAILABLE",
  "upper_source": "VERIFIED_SUFFIX|BOUNDED_PLAN|EXACT|UNAVAILABLE",
  "plan_sha256": null,
  "status": "OK|UNKNOWN|BUDGET_UNLABELLED|UNSUPPORTED",
  "cpu_seconds": 0.0
}
```

`null`表示不可得；不能将null序列化成0。无穷使用明确字符串或状态字段，保持标准JSON兼容。

### 19.2 certified_pairs.csv最少字段

```text
case_id,trace_id,event_id,pair_class,better_hash,worse_hash,
better_parent,worse_parent,shared_recorded_parent,
L_better,U_better,L_worse,U_worse,
g_better,g_worse,certificate_remaining,certificate_total,
dense_v_better,dense_v_worse,wl_h_better,wl_h_worse,
dense_order,wl_order,actual_popped,
anchor_active,lower_backend,upper_source,reference_plan_sha
```

### 19.3 method_decision.json

```json
{
  "card_status": "DIAGNOSTIC_COMPLETE|INCOMPLETE|BLOCKED",
  "scientific_label": "CROSS_SCOPE_SIGNAL|LOCAL_WITNESS_PRESENT_NO_CROSS_LOCALIZATION|INCONCLUSIVE_CERTIFICATES|REFERENCE_ANCHOR_MISSING|NUMERICAL_TIE_DOMINATED|NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX",
  "mixed_local_and_cross": false,
  "eligible_for_layer2_discussion": false,
  "layer2_authorized": false,
  "backup_authorized": false,
  "supporting_cases": [],
  "counterevidence_cases": [],
  "unresolved": [],
  "next_action": "WAIT_FOR_USER_RESEARCH_DECISION"
}
```

所有表由同一报告程序生成。不要手工调整一项率以让它满足第17节。

---

## 20. 自动执行顺序与停止规则

```text
收到对本卡的明确批准
  → 核验仓库、旧资产、权重、外部模型、存储与进程权限
  → 固定10题以内的清单、参考计划选择与全部预算
  → 实现只读观察器、界接口与必要fixture
  → 若语义／资产不通过，输出BLOCKED并停止
  → 登记提交成功推送后，才启动主前缀
  → 在同一预算内采样DENSE/WL、冻结样本清单
  → 离线算界和另一评分器读数
  → 统一报告、科学标签与UNKNOWN比例
  → 验证旧资产未变、产物大小和回执计数
  → 显式提交轻量文件、推送、停止
```

- 输入资产哈希不符、计划回放失败、L>d*反例、标签进入队列优先级、旧数据被修改：技术性停止。
- 某题超时／节点上限、界不够紧、没有信号：科学或资源结果，不是可重跑故障。
- 不因结果不理想从60秒增到300秒；不生成新的失败题补名额；不改变证书标准。
- 仅允许同配置、恢复完整进度的基础设施中断恢复，记录全部尝试；没有恢复状态时报告缺口，不从头挑更有利结果。
- 低分不阻止其他独立题完成；不会在每道题后征求是否继续，但层二／备选必须另行批准。

### 最终摘要必须回答的六个问题

1. 哪些训练约束确实缺失，哪些只是模型共享参数的间接约束？
2. 在真实OPEN中是否存在跨组可证错排，分散在哪些题、哪些状态？
3. 局部错误、链式顺序和跨组见证是否共存？
4. UNKNOWN有多少，参考锚点和上下界造成什么选择限制？
5. 同一状态上WL有何不同；这些差异是否足以支持下一项干预，而非单纯归因？
6. 本卡建议是否进入层二讨论，还是停止并保留备选？不能写成“自动开始训练”。

---

## 21. 验收标准与最终回执

技术验收与科学结论分开。即使 `scientific_label=INCONCLUSIVE_CERTIFICATES`，只要协议完成且所有UNKNOWN正确保留，技术验收也可以通过。

```yaml
card: CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1
approval_reference: REQUIRED_BEFORE_EXECUTION
base_commit: 511b117495920234cb646016eaf504c7371233aa
registration_commit: TO_BE_CREATED_ONLY_AFTER_APPROVAL
result_commit: RESOLVE_FROM_GIT_AFTER_COMMIT
weights_sha_before: RECORDED
weights_sha_after: RECORDED
weights_modified: false
old_results_modified: false
new_neural_trainings: 0
new_wl_fits: 0
optimizer_steps: 0
new_problems: 0
main_prefixes_planned: AT_MOST_20
main_prefixes_completed: ACTUAL
fixture_prefixes: AT_MOST_8
unique_bound_states: ACTUAL_AT_MOST_4096
upper_bound_searches: ACTUAL_AT_MOST_64
gpu_process_wall_seconds: ACTUAL_AT_MOST_1800
label_cpu_core_seconds: ACTUAL_AT_MOST_3600
budget_stops: []
technical_gaps: []
scientific_label: FROM_RULES
training_data_exported_from_diagnostic_states: false
layer2_started: false
wl_backup_started: false
next_action: WAIT_FOR_USER_RESEARCH_DECISION
```

验收要点：所有指标分母可复算；每个严格标签有计划/下界证据；没有把已离开OPEN的状态计作可选；不重复计算合并日志；原始权重／旧结果不变；未知和失败未删除；新`.pt`和大二进制不提交。

本首卡只提供是否值得花下一笔训练预算的证据，不要求输出“赢家”，不对CCF-A录用作承诺。

---

# 第三部分　证据与可核查来源

## 22. 材料读取范围

- A、B、RA和research_state均为用户给出的Markdown材料；RB为本对话上一轮GPT审查正文。
- 对关键训练路径与共同搜索器作了只读源码核对；没有加载服务器权重、查询实时进程、运行实验或执行RA/B的整套统计脚本。
- RA报告的共享率、逐题waste和B的动作拆分作为审查材料来源保留，不能写成本文独立实验结果。
- 本轮补读NeurIPS2023方法部分及官方页面；ECAI2025用于判断近邻范围的内容来自作者机构摘要，未声称全文实现复现。

### 附件内容身份（本地文件字节核对）

| 材料 | 文件 | SHA-256 |
|---|---|---|
| STATE | `research_state.md` | `38d9dfa42af232548704d3af415ad2b794de450e079afa48d6280929ffffb822` |
| A | `CP_DISR_C1_Independent_Method_Proposals_20261009.md` | `ffba9e7ae00af6c2fab8d6e2a1acbc06f19cdfbb29a5940147e0b7e34dad15f6` |
| B | `CP_DISR_C1_下一阶段候选方案_独立评估_20261009.md` | `1f179b87bfcbe77f919e4ce4758ec06545770802c4dc1b7430f8f0e6cf046f28` |
| RA | `CP_DISR_C1_GPT方案交叉审查_20261009.md` | `59e4f9b85bee641a300ca92acf60497cf29f3ee0153a229aeeac34ddd4873452` |

RB无独立上传文件，见本对话审查正文；不为其编造哈希。

## 23. 固定仓库证据

[S1]: research_state.md
[A]: CP_DISR_C1_Independent_Method_Proposals_20261009.md
[B]: CP_DISR_C1_下一阶段候选方案_独立评估_20261009.md
[RA]: CP_DISR_C1_GPT方案交叉审查_20261009.md

- **[S1]** 用户提供的 `research_state.md`，截至 `511b117495920234cb646016eaf504c7371233aa`。
- **[A]** GPT独立方案文件。
- **[B]** Claude独立方案文件。
- **[RA]** Claude对A的交叉审查文件。
- **RB**：本对话中“Claude方案总体判断／最多三个重要问题／监督匹配WL优先”这一轮GPT审查；没有独立附件。

[G1]: https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1/results/final_summary.md
[G2]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/runs/final_master/c1_route_b/depots_fast_alt_v3/20261008T191927Z_fastalt/results/final_summary.md
[G3]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/runs/final_master/c1_route_b/depots_fast_alt_v3/20261008T191927Z_fastalt/results/claim_boundary.md
[G4]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e/training_label_coverage.json
[C1]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/src/cp_disr/pddl/train.py
[C2]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/src/cp_disr/blocksworld/method_serial/model.py
[C3]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/src/cp_disr/blocksworld/goal_progress.py
[C4]: https://github.com/DillonZChen/goose/blob/04dbe8f27a8ec027626e6c6b5d6e7610b81f9d57/goose/learning/dataset/heuristic/creator/classic_ranking_dataset_creator.py
[C5]: https://github.com/DillonZChen/goose/blob/04dbe8f27a8ec027626e6c6b5d6e7610b81f9d57/goose/learning/predictor/linear_model/rank_util.py
[C6]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/src/cp_disr/pddl/search_match/engine.py
[C7]: https://github.com/rollinpioneer/graph/blob/511b117495920234cb646016eaf504c7371233aa/src/cp_disr/pddl/search_match/evaluators.py

| 证据键 | 说明 |
|---|---|
| [G1] | 共同GBFS的正结果、IPC差距、数据角色 |
| [G2]、[G3] | FAST/ALT的结果、轨迹与成本边界 |
| [G4] | 原精确Oracle标签覆盖，不应误读成8箱任意状态都已有精确距离 |
| [C1]—[C3] | 当前PDDL训练器、logits、同父动作集合与成对排序 |
| [C4]、[C5] | 固定GOOSE的父／计划后继／其他后继数据与实际配对 |
| [C6] | 同搜索器的队列、CLOSED、判目标和现有observer接口 |
| [C7] | 评分适配、SAS转换与已有h_add/WL接口 |

## 24. 关键文献

[P1]: https://proceedings.neurips.cc/paper_files/paper/2023/file/50ea4dbd1cff6bd3daef939eff10c092-Paper-Conference.pdf
[P2]: https://www.ijcai.org/proceedings/2024/0743
[P3]: https://digitalcollections.anu.edu.au/entities/publication/4feec7b1-5d22-43b9-9d8f-433b4400b2e5
[P4]: https://www.fast-downward.org/latest/documentation/search/Evaluator/
[P5]: https://ojs.aaai.org/index.php/AAAI/article/view/29986

| 来源 | 与本次决定的关系 | 阅读边界 |
|---|---|---|
| [P1] Chrestien等，NeurIPS2023，Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal | 定义1、式1—4已经覆盖计划状态与OPEN竞争者的排序；不能把普通OPEN排序当首创 | 核对官方PDF方法页，包括第3—5页 |
| [P2] Hao等，IJCAI2024，Guiding GBFS through Learned Pairwise Rankings | 直接排序与计划外状态监督的强近邻 | 官方论文入口；具体本项目WL行为以固定代码为准 |
| [P3] Hao等，ECAI2025，Effective Data Generation and Feature Selection in Learning for Planning | A*树中更多排序信息；“增加排序样本”并非未被研究 | 作者机构摘要，不冒称全文复现 |
| [P4] Fast Downward官方Evaluator文档 | LM-cut/h_max等下界性质与h_add非可采纳性质需区分 | 当前官方文档；执行固定服务器版本并核验语言支持 |
| [P5] Chen等，AAAI2024，Learning Domain-Independent Heuristics for Grounded and Lifted Planning | 提升式图可保留动作模式和合同结构，不等于丢掉已知合同 | 本文只用于纠正文献前提，不启动编码器实验 |

---

**结束声明：**本文件为最终研究决策与待批准的首卡设计。没有创建研究分支、修改仓库、启动任何训练、拟合或服务器搜索。所有预算、阈值和读数格式均是未来执行规格，不是本轮新获得的数据。
