# CP-DISR 论文开写前最终审查（Pre-Writing Review）

审查对象：CP_DISR_Paper_Oriented_Research_Package_v3.0（**文档 v3.0 / 方法 CP-DISR·M1 v2.1**）
审查日期：2026-09-22
审查角色：CCF-A / SCI 一区审稿人，按机器人学习、VLM、RL、结构化推理四个方向同时审
读过的材料：README、主规范 v3.0（41 节 + Paper Readiness + 附录 A–C）、6 份 paper_drafts、全部 interfaces、两份 figure、validation；sources/ 中与 D^P/SOFT_ORDER/effect_fact_ref 决议相关的原始讨论。
补充检查：(1) 在联网状态下做了 2023–2026 文献撞车检索；(2) 用随机初始化的 4 层 R-GCN（numpy）跑了一个 toy 数值检查，用来回答 effect_fact_ref 的问题（第 9 节）。toy 检查只看表示层行为，不代表训练后的效果。

> **结论先行：No method-level blocker remains。v2.1 可以冻结，今天就可以开始写。** 需要改的地方集中在论文定位、贡献排序、baseline 矩阵和写作语气上，另有两处很小的训练实现默认值要改。

---

## 0. Source of Truth 确认

- 文档版本 v3.0 ≠ 方法版本。Current Method = **CP-DISR / M1 v2.1**。
- Future M1+（Candidate × Relation Error Learning）只放在 Discussion / Future Work：不进 Current Method，不进 loss，不进实验。
- v3.0 和 sources/ 冲突时以 v3.0 为准。整合过程中没有发现 v3.0 引入的错误。v2.0 → v2.1 的三项变化（删除 SOFT_ORDER、纯交互 D^P、80/20）已经一致地落到了 manifest、schema、prompt、drafts 和 validation 里。

---

## 1. 总体判断（第三部分）

| 问题 | 判断 |
|---|---|
| 1. 问题定义是否成立 | **成立**。建议把重心从“VLM 会犯错”移到“语义正确 ≠ 决策有用，而且二者都依赖具体候选动作”（见第 4 节）。 |
| 2. M1 是否形成方法闭环 | **形成**。固定 prior → 候选名义干预 → 四视图共享编码 → D^K / D^P → 合同主分 + 有界零锚残差 → 真实执行 → PPO / V / Q → 共享表示。前向不访问真实未来，反向不改知识源，训练和推理的定义一致。 |
| 3. 创新是否足以写一篇方法论文 | **够**，但真正需要守的核心只有一个：**C2（contract–prior 交互对比）建立在 C1（候选名义干预）之上**。C1 单独看新颖性偏弱（afterstate / successor-state GNN 已经很成熟），C3 更接近训练机制。 |
| 4. 是否有写作前必须改的结构问题 | **没有**。effect_fact_ref 的错位是真实存在的，但属于表述问题，不是结构错误（第 9 节）。 |
| 5. 最大风险 | 见下表。 |

**风险分类**

- **Novelty risk（中）**：C1 容易被看成 afterstate / successor-state 评价（Ståhlberg–Bonet–Geffner 系列、PLOI、Sutton & Barto 的 afterstate 概念）；C3 容易被看成普通 auxiliary value head。防守重点放在 C2。
- **Methodological risk（中低）**：审稿人会问 “D^P 是不是人工特征工程，直接把四个 embedding concat 给 MLP 行不行”。目前的 A_DD 回答不了这个问题（第 7 节）。
- **Experimental risk（高）**：reward 只有终端成功、没有课程、没有行为克隆，全靠 PPO 从零学。如果随机策略的成功率接近 0，整套实验可能学不起来。另外，现在的 B1 和 Full 之间不能干净地归因（第 18 节）。这一项影响的是实验，不影响开写。
- **Writing / positioning risk（高）**：现有草稿的防御性否定句太多（“不等于 / 不保证 / 不声称”），读起来像技术规范，不像论文。故事也容易被读成 “VLM + R-GCN + PPO” 的拼装。

## Overall Verdict

### **A. 可以直接进入论文写作，只需要局部修订。**

两处 v2.1 内的实现默认值（PW-05 actor 折扣加权、PW-06 γ̄ 的取值规则）会出现在 Method 的公式里，所以建议写 §4.6 之前先定下来，花十分钟就够。它们不改变方法结构。

---

## 4. 论文真正的研究问题（第四部分）

**现在的表述**：“Learning to utilize fixed imperfect semantic priors for long-horizon skill selection.”

- 明确性：明确。
- 重要性：重要。
- 方法味：一般。问题出在 “VLM 可能出错” 是常识，审稿人读到这里会想 “那就做 filtering / confidence 吧”。

**更强、但不夸大的中心**：

> 一条语义关系对决策有没有用，不由关系本身决定，而取决于**它是否改变了某个候选动作的后果**。

这句话同时解释了三件事：为什么要做候选干预（A）；为什么 prior 只以交互的形式进入（C，也就是纯交互 D^P）；为什么“用不用”要从结果中学（D）。

**推荐的中心排序**：**A + C 融合为一个中心**，写成 candidate-conditioned prior interaction，D 作为学习手段，B（contract / prior 分离）作为使 C 成立的前提。

**推荐的 Problem statement 一句话**：

> Given fixed skill contracts and a fixed, imperfect set of VLM-proposed relations, learn a skill policy whose use of each relation is conditioned on how that relation alters the nominal consequences of each executable candidate.

这样写，审稿人看到的是一个关于“表示的算子”的问题，而不是“VLM + 图 + PPO”的系统。

---

## 5. 核心贡献强度（第五部分）

| 贡献 | 是否算方法贡献 | 最接近的已有工作 | 真正新的地方 | 建议 |
|---|---|---|---|---|
| **C1** Candidate Nominal Structural Intervention | 单独看**弱**。“用已知动作模型生成后继状态，再用 GNN 评估”是 generalized planning 的标准做法（Ståhlberg, Bonet, Geffner KR 2022/2023 把策略定义为 state-transition 分类器；PLOI；afterstate value） | successor-state GNN policy、afterstates | 只取名义成功分支，与部分可观测、真实失败共存；更关键的是它是 C2 的“干预轴” | **降级为 C2 的构件**，不作为独立卖点。写法：“we build on the classic afterstate idea, but…” |
| **C2** Contract–Prior Dual Difference + 零锚有界残差 | **是**。这是最有算法味的一项 | LLM prior 作为 action prior / KL / policy shaping（Yan et al. ICLR 2025；uncertainty-aware LLM guidance；LLM4Teach）；KG 注入 RL 的机制比较（2607.19616） | 把 prior 的作用**限定在 2×2（干预 × prior）交互对比**上，并通过零锚定得到精确的 null 性质。检索中没有找到做同样事情的工作 | **作为唯一的核心算法贡献**，放在 Method 视觉和文字的中心 |
| **C3** Outcome-Grounded Structural Learning | 更像**训练机制** | UNREAL、PPG、各种 auxiliary Q/V head | 只监督实际执行的候选，把 D^K / D^P 特征直接绑定到真实回报。这点合理，但新颖性有限 | **不再作为第三项方法贡献**，改成 §4.6 的训练设计，配一个消融 |

**建议的三项贡献（替换当前的 C1–C3）**：

1. **问题 / 表述**：把 “fixed imperfect prior utilization” 形式化为**候选条件化**的问题，指出 relation correctness 和 decision utility 的差别体现在候选后果层面。
2. **方法**：contract–prior interaction contrast（原 C1 + C2），包括四视图共享编码、纯交互 D^P，以及零锚定有界残差带来的精确 null 性质；训练时由真实结果监督（原 C3 并入这里）。
3. **实证 / 分析协议**：matched baseline 阶梯、同一 checkpoint 下的 prior 扰动（No / Deletion / Target-swap / Irrelevant）、Actor 依赖诊断。这类“可控评估协议”在 CCF-A 论文里通常可以作为第三点。

如果坚持保留三项方法贡献，也可以写成 C1 / C2 / C3，但请把 C1 的措辞从 “We formulate” 改为 “We adopt and adapt …”，C3 从 “We train …” 改为训练设计的描述，避免审稿人觉得这些是 “natural combination”。

---

## 6. 联网撞车审查（第六部分）

检索时间：2026-09。覆盖了 candidate-conditioned / afterstate / successor-state GNN、action-effect graph、symbolic action model + RL、LLM / VLM prior + RL、noisy prior、KG 注入 RL、difference / counterfactual 表示、auxiliary value 等方向。

**核心问题**：有没有论文已经做过“对 candidate 施加名义效果 → 比较前后图表示 → 再比较有 / 无 semantic prior → 用真实 RL 回报学习这种差异”？

**答案：没有检索到。** 离得最近的是下面这些工作，但每一项只覆盖了其中一部分：

| 工作 | 重合部分 | 不同部分 | 撞车风险 |
|---|---|---|---|
| Ståhlberg, Bonet, Geffner, *Learning General Policies with Policy Gradient Methods*（KR 2023；arXiv 2512.19366），以及 KR 2022、ICAPS 2022 的同系列工作 | 用 GNN 评估 successor states，策略是 transition classifier，用 actor-critic 训练 | 没有 prior；没有 before / after 差分；完全可观测的经典规划 | **C1 高**，C2 低 |
| Afterstate value（Sutton & Barto 2018 §6.8；*Afterstate RL for Continuous Control*，OpenReview） | 用已知的确定性效果评估动作后状态 | 没有图、没有 prior | C1 中 |
| Counterfactual Quotient Models（arXiv 2608.22092，2026-08） | 只保留“动作之间有差异的部分”，公共部分相消，思想上和 D^K 同类 | 学到的动力学、没有 prior、没有符号合同 | **D^K 的思想层面中**。需要引用，说明“差分消去共享成分”有先例，我们的差分对象是 prior × intervention |
| Yan et al., *Efficient RL with LLM Priors*（ICLR 2025） | 把 LLM 当作不完美的 action prior，在 RL 中使用 | prior 直接作用在动作分布上（这正是“static prior”），不经过候选后果 | C2 **中**，是必须正面对比的 prior-as-action-bias 路线 |
| *Guiding RL Using Uncertainty-Aware LLMs*（arXiv 2411.14457） | 根据可信度调节 LLM 对策略的影响 | 按不确定性加权，不做结构交互 | 中低 |
| LLM4Teach（IJCAI 2024） | LLM 作为不完美的教师，RL 学生逐步脱离 | 蒸馏式，不做结构表示 | 低 |
| *The Mechanism Matters: When KGs Help RL*（arXiv 2607.19616，2026-07） | 比较 KG 注入 RL 的方式（state feature / mask / reward shaping）；发现 hard mask 在错误 KG 下脆弱、soft 注入更稳健 | 合成 KG，MiniGrid，没有候选条件化 | 中低。**对 motivation 很有用**：可以支持“prior 不应改变 mask”这一设计，并且是 2026 年的新证据 |
| SGA-ACR（arXiv 2511.20993） | LLM 子目标图不完美，结合 RL | 在线 critique / refine 图，也就是修图路线 | 低，并且是“修 prior 而不是用 prior”的反例 |
| ASGRL（Guan et al., ICML 2022） | 用近似（不完美）符号模型辅助 RL | 符号模型用来生成 landmark / 技能多样性 | 中低，**需要引用**，它是 “imperfect symbolic knowledge + RL” 的直接前身 |
| LLM 生成 PDDL / 世界模型（Guan et al. NeurIPS 2023；LLM+P；InterPreT；UniDomain 2025） | VLM / LLM 提供结构知识 | 构造硬的动作模型再做规划 | 低。**但如果把 soft edge 改成 fact → action，会变成 “VLM 补 soft precondition”，这条线的撞车风险会上升**（第 9 节） |
| PLOI（AAAI 2021）、*Graph Learning for Planning* 综述（arXiv 2412.02136） | GNN + 规划结构 | 没有 prior 交互 | 背景 |
| Plan-Seq-Learn（ICLR 2024）、BOSS（CoRL 2023）、SayCan / Inner Monologue（CoRL 2022） | LLM 与技能组合 | LLM 决定序列或提供评分 | 背景 |

**结论**：C2 的核心形式（**prior × candidate intervention 的 2×2 交互对比，作为唯一的 prior 通路**）在本次检索中没有找到重合。C1 本身有成熟的先例，必须主动引用 afterstate / successor-state 工作并降低 C1 的调子。这样做反而能加强可信度。

局限：公开检索看不到仍在审稿中的投稿；上表中没有逐篇读全文的条目，投稿前请逐条核对 bib。

---

## 7. D^K / D^P 是否站得住（第七部分）

**1. latent 相减有没有基础？**
有，但它是一种归纳偏置，不是语义线性。四视图共享编码器、目标行对齐，这保证了“同一坐标系”。对合同分支来说，(E(G), D^K) 是 (E(G), E(G̃)) 的可逆重参数化，加上 context 里已经有 Pool(Z^K)，所以**D^K 没有信息损失，只是提供了一个零点**。D^P 则是**有意的信息限制**（丢掉静态项）。论文里要把这两者区分开说。

**2. D^K 最准确的解释**：候选 i 的名义效果在合同结构下引起的**目标对齐表示位移**（contract-induced representation shift）。

**3. D^P 最准确的解释**：有一个很好用的对称恒等式，应当写进论文：

$$
D_i^P=\underbrace{[E(\tilde F_i,R)-E(\tilde F_i,\varnothing)]}_{S(\tilde F_i)}-\underbrace{[E(F,R)-E(F,\varnothing)]}_{S(F)}
$$

其中 $S(F)$ 是 prior 在事实状态 $F$ 下的**结构签名（prior signature）**。所以 D^P 同时可以读成：
- prior 如何改变编码器对这个干预的响应（现在文档里的写法）；
- **候选干预如何改变 prior 的作用**（新写法，更直观）。

这正是 2×2 析因设计里的 **interaction contrast**。建议论文正式称它为 *contract–prior interaction contrast*，用这个名字替代 “difference of differences”，并明确说明它**不是因果 DiD 估计**。

**4 / 5. 和直接 concat 四个 embedding 相比有没有实质优势？**
这是审稿人最容易提出的替代方案。可以从两方面回答：
- concat 同样能做到零锚定（见下面的 A_CAT 定义），所以“零性质”**不是** D^P 独有的优势；
- D^P 真正的区别在于**把静态 prior 签名 S(F) 排除在外**。这是一个可以证伪的设计主张：如果静态信息有用，concat 应该更好；如果纯交互这种限制更利于泛化和抗扰，Full 应该持平或更好。

**6. A_DD 够不够？**
不够。A_DD（非空时输入 D^H）只检验“是否减掉 D^K”，而 D^H = D^K + D^P，且 F_P 已经能看到 u^K，A_DD 的信息量几乎和 Full 一样。它回答不了 “D^P 是不是人工特征工程”。

**建议（P1）：新增 A_CAT，把 A_DD 降为可选。**

$$
u_i^{P,\mathrm{CAT}}=F_P\big(c_i,u_i^K,[Z^K,\tilde Z_i^K,Z^H,\tilde Z_i^H]\big)-F_P\big(c_i,u_i^K,[Z^K,\tilde Z_i^K,Z^K,\tilde Z_i^K]\big)
$$

它和 Full 一样零锚定、有界、使用同一个 head；空 prior 时严格为 0；可以学到静态签名和交互项。**这一个配置同时回答第七部分的第 5 问和第八部分**（纯交互是否丢失有用信息），配置总数不增加（见第 18 节）。

---

## 8. Pure Interaction Prior（第八部分）

**判断：这是一个漂亮而且明确的设计原则，应当保留，并在论文中把它当作核心卖点。**

- **创新性**：它是 C2 与 “LLM 作为 action prior”（Yan et al. 2025 等）划清界线的地方。那类方法本质上都是 static prior。
- **合理性**：v2.1 的关系词表（只保留后果关联，删除 SOFT_ORDER）、prompt（必须给出 effect_fact_ref），和算子是配套的：**schema 只允许那些 D^P 能够表达的关系进入**。这种“schema 与算子一致”的设计本身就值得在论文里写一段。
- **实际 VLM 关系**：SOFT_RELEVANT_TO_GOAL 带有一部分静态意味（“这个动作和某个目标有关”）。在纯交互设计下，它只有在候选改变了源动作邻域时才起作用。这是有意为之。
- **长时程技能选择**：静态偏好（“先做 A”）在长时程任务中恰恰是最容易出错、最依赖状态的 prior，丢掉它是一个合理的取舍。

**论文中的说服力写法（建议）**：

> A relation should influence a decision only insofar as it changes what the decision does. We therefore let the prior enter the policy exclusively through its interaction with candidate-induced change; any component of the prior that is invariant to the candidate cancels by construction. This excludes static action biases, which are precisely the form of prior most likely to be stale in long-horizon execution, and it gives an exact null: without relations, or without change, the policy reduces to the contract policy.

同时把 A_CAT 作为这一取舍的实证检验。

---

## 9. effect_fact_ref（第九部分，重点）

### 9.1 错位是否真实存在？**存在。**

relation 的语义是“A 的效果 p 支持 B”，但计算时只有一条 A → B 的软边。A 的节点嵌入由它**所有**相邻命题的消息聚合而成（前置、ADD、DEL），因此软边传给 B 的是“A 周围发生的任何变化”，不是“p 的变化”。

### 9.2 这个错位在哪些情况下有影响？

| 情形 | 当前 A → B 设计 | 影响 |
|---|---|---|
| 候选 = 源动作 A | 名义 patch 会**同时**改变 A 的全部效果，所以“到底是哪个效果”在计算上没有区别 | **无** |
| 候选 = C，C 的效果是 A 的前置（例如 PICK 使 MOVE 可执行） | A 的邻域变化经软边传到 B，相当于“C 使 A 可行，而 A 支持 B”，是一种两步传播 | 更像**优点**，只是噪声更大 |
| p 已经为真（支持作用已经实现） | A 的其他效果仍然变化，软边照常起作用，**不受 p 的状态门控** | **有语义泄漏**：prompt 里写的 “A relation may be inactive in later states” 在计算上并不成立 |

### 9.3 toy 数值检查

设置：随机初始化的 4 层 R-GCN，mean 聚合 + LayerNorm，20 个 seed，目标行加全局读出。表中数值为 ‖D^P‖/‖D^K‖ 的中位数。

| 状态 | 设计 | 候选 = MOVE（源动作） | 候选 = PICK(blue)（无关） |
|---|---|---|---|
| p 尚未实现 | A → B | 1.47 | 0.80 |
| p 尚未实现 | p → B | 2.03 | 0.91 |
| **p 已实现** | A → B | **1.46（不变）** | 0.83 |
| **p 已实现** | p → B | **1.08（下降约 47%）** | 0.93 |

说明：
- 效果锚定（p → B）**确实提供了部分门控**，但不是干净的开关，多跳消息仍然会泄漏。
- 在小图上，任何一条软边都会让所有候选的 D^P 明显非零。门控最终依赖学到的权重，**两种设计都得不到干净的“按效果归因”**。
- 这是随机初始化下的结果，只能说明表示层面的趋势。

### 9.4 如果改成 p → B，需要付出什么代价

1. **符号问题**：DEL 锚定的关系表示“p 变为假会支持 B”。R-GCN 没有边特征，需要把 SOFT_SUPPORTS 拆成 via-ADD 和 via-DEL 两类，GOAL 类同理，关系类型从 2 类变成 4 类（加上反向共 8 类）。
2. **语义变化**：关系不再属于 A，任何能产生 p 的动作都会继承这份支持。这会让 VLM prior 变成 “soft precondition 补全”，和 LLM 生成 PDDL / 动作模型的工作（Guan 2023、LLM+P、InterPreT、UniDomain）**撞车风险明显上升**，也会稀释 “skill-level consequence relation” 的故事。
3. **工程**：schema 语义、去重键、冗余规则（p 已经是 PRE_POS B 时要去重）、prompt、validator 和图构建都要改，还要重新做验证。
4. **公式**：D 的定义不变。

### 9.5 裁决

**这不是 P0，也不需要在写论文前升级到 v2.2。保留 Action → Action。** 理由：
- 论文的主张是“编码器对候选变化的响应”，没有主张“按效果归因”，所以不存在逻辑错误；
- 对主要情形（候选 = 源动作），这个错位在计算上没有影响；
- 改成效果锚定只能带来部分门控，却要付出语义和新颖性方面的代价。

**需要做的只是改表述（P1）**：
- 把 effect_fact_ref 写成 **consequence-grounding admission constraint**：它保证每条进入模型的关系都能用一个已注册的名义效果来解释，从而把 static / order-only 关系挡在外面。这就是 schema 层面的纯交互原则，也正好回答审稿人会问的 “不用它为什么要它”；
- 在 Method 或 Limitations 里用一句话说明：“The relation enters at the skill level; which of the source's effects mediates it is not resolved by the encoder.”；
- 删除或修改 prompt 中 “A relation may be inactive in later states” 这句。它是给 VLM 看的，但会误导读者以为存在门控，可以改成 “Relations are not required to hold in later states”；
- 如果 Stage 4A 的诊断显示软边在锚定效果已实现时照样主导 D^P，再把 effect-anchored 版本作为 **P2 可选消融**。

---

## 10. 固定初始 VLM relation（第十部分）

**合理，而且这是一个有意思的设定。** 审稿人几乎一定会问 “为什么不重新 query VLM”。建议从四点回答：
1. **研究隔离**：把 prior 的生成质量和 prior 的利用机制分开，才能在同一份 cache 下比较不同的表示；
2. **非平稳性**：训练中反复 query 会让输入分布跟着策略漂移，PPO 的 on-policy 假设会进一步被破坏；
3. **成本和延迟**：每个技能边界都调用一次 VLM，在真机上并不现实；
4. **状态条件化已经内建**：关系是后果关联的，它是否起作用由当前事实和候选决定（D^P 随状态变化），因此“固定关系 ≠ 固定影响”。

**不需要做动态重新 query 的实验。** 在 Discussion 里写一段，放到 Future Work（P3）即可。

---

## 11. “好 prior 用、坏 prior 少用”的 claim 强度（第十一部分）

监督是间接的（PPO / Q），没有逐条关系的标签，所以**“learn when and how semantic priors should influence decisions” 说得太强**，而 “provide a learnable interface” 又太弱，不像贡献。

**推荐的中间强度**：

> CP-DISR **learns a candidate-conditioned, bounded modulation** of the contract policy by fixed semantic relations, **trained only from task outcomes**; we **test whether** this modulation is used selectively via same-checkpoint prior perturbations and actor-dependence diagnostics.

也就是说，claim 写“学习一个 prior 调制”（这是方法本身做的事），至于是否“选择性地使用”，交给实验去检验。

---

## 12. Structural Q（第十二部分）

1. **是否有意义**：有。它提供的是**候选级回归信号**。Actor 的梯度经过 softmax 分散到所有候选上，方差也高；Q 只沿着**被执行候选的 u^K / u^P**，用真实回报做回归，把 D^K / D^P 特征直接和结果绑定。这是目前最合理的故事，应当保留。
2. **和普通 auxiliary Q head 的区别**：区别只在于它读取的是候选结构特征。概念本身不新。
3. **是否作为 C3**：**不作为独立贡献**（见第 5 节）。
4. **定位**：training mechanism，配一个消融（A_Q）。
5. **和 PPO critic 是否重复**：部分重复。因为 $y^Q_t - V_{old}(X_t)=\delta_t$，Q 实际上是在学 V 加上一步 TD 优势。这一点可以作为 “why both” 的解释。
6. **为什么既要 V 又要 Q**：V 是 GAE 的基线，需要状态级的量；Q 提供候选级的表示压力。不采用 $V=\sum\pi Q$，因为未执行的候选没有标签。现在文档里的这段解释是对的。
7. **one-step bootstrap target 是否合适**：合适，偏差小、实现简单。它和 GAE 的 λ-return 不一致，这是有意为之，写一句说明即可。
8. **稀疏 reward 下的注意点**：训练早期 V_old≈0，Q 的目标只在成功转移上非零，所以 Q 的作用可能主要集中在中后期。这属于实验解读问题，不需要改方法。

**最小写作调整**：§4.6 的标题改为 “Outcome-grounded training”；C3 的内容并入贡献 2 的最后一句；把 “Structural Q” 这个名字保留为模块名。

---

## 13. Actor / V / Q 梯度（第十三部分）

没有明显危险。λ_Q=0.1、Huber 损失、目标 detach，这些已经比较保守。PPG 讨论过共享表示下的干扰，但那是在 λ 较大、价值损失占主导的情况下。**不需要引入 stop-grad 或 gradient coefficient 设计。** 作为 P2 诊断，可以在训练日志里记录共享层上 actor / Q 梯度的余弦相似度，一行代码，出问题时可以直接定位。

---

## 14. Semi-MDP 与奖励（第十四部分）

- **数学一致性**：Γ_t=γ̄^{d_t}；技能内的奖励按事件偏移折扣；GAE 的递推 $\hat A_t=\delta_t+\Gamma_t\lambda c_t\hat A_{t+1}$ 是标准的 SMDP 写法。terminated 时 κ=0；truncated 时 κ=1，用最后一个有效状态 bootstrap，并令 c=0。**以上正确。**
- **问题 1：actor 的 w_t 折扣加权（P1，PW-05）**
  - 现在按 Nota & Thomas 的做法，用 $w_t=\prod\Gamma$ 对 actor 样本加权。在时长折扣下，这会**系统性地压低后期决策的权重**：γ̄=0.99/s 时，第 60 s 的权重约 0.55，120 s 约 0.30，200 s 约 0.13，300 s 约 0.05。长时程任务的后半段几乎学不到东西，而这恰好是论文最关心的部分。
  - 主流 PPO 实现都不这么做。
  - **建议**：默认 `actor_episode_discount_weight: false`（标准做法），在论文里用一句话脚注说明引用了 Nota & Thomas 并作出这一选择。
- **问题 2：γ̄ 的取值（P1，PW-06）**
  - γ̄ 应当按任务的典型时长来标定。建议规则：典型成功时长 $T$ 下，$\bar\gamma^{T}\ge 0.3$。
  - 0.99/s 只适合 ≤120 s 的任务。
- **稀疏 reward**
  - 这是实验上最大的风险。reward 只来自终端成功，又没有课程，PPO 必须靠随机探索偶尔成功才能起步。例如 6 步决策、每步 3–4 个合法候选、每步 1 个正确选择，随机成功率约 0.1%。
  - 这**不需要改方法**，而是一条任务设计的门槛（第 22 节）：**每个训练任务上，随机 masked 策略在 deadline 内的成功率应 ≥5%**（在 Stage 0 预检）。满足这个条件，PPO 才现实。
  - 最适合 PPO 的任务是：选错会付出时间代价，但仍然可以恢复。这样学习信号来自折扣回报的差异，而不是 0 / 1 的可行性。

---

## 15. Graph Encoder 与 Goal-Aligned Readout（第十五部分）

1. **多目标**：固定目标行加候选 query 的 attention，足以支撑多目标推理。
2. **Mean pooling 是否过弱**：z_0 中的 Mean(H_A)、Mean(H_P) 在大图上会**稀释局部变化**。对那些效果在 4 跳内到不了任何目标命题的候选，D 几乎只剩下被稀释后的全局行。这是一个真实的弱点，但目前不必修改（P2）。如果 Stage 4A 发现这类候选的 ‖D^K‖ 远小于其他候选，再加一行“候选动作节点自身的差分”，改动很小，不属于新模块。
3. **目标行 + 全局行能否支撑相减**：能。
4. **padding / mask**：合理。
5. **4 层的感受野**：典型链条的跳数是 PICK → Held → PLACE → Inside（2 跳），MOVE → p → A →soft→ B → goal（3 跳），都在 4 层以内。更长的依赖在 Limitations 里说明即可。**不需要换 Graph Transformer。**

---

## 16. soft relation 词表（第十六部分）

两类关系**恰好够用**，而且有利于保持论文干净。删掉 SOFT_ORDER 损失的主要是静态顺序偏好，而这类信息在纯交互设计下本来就进不了模型（D^P 会把它消掉），v2.1 的删除是自洽的。**不要重新加回来。** 在论文里可以用一句话说明：“Order-only relations are excluded because they are invariant to candidate-induced change and would cancel in D^P.”

---

## 17. 80% Original / 20% No-prior（第十七部分）

- 这是合理的正则化，本质上和 modality dropout / conditioning dropout 同类。
- 它应当写进 Method 的 Training 小节，作为训练协议的一部分，但**不是贡献**。
- 它**不会**成为混淆变量，因为所有使用 prior 的配置都采用同样的协议。
- **不需要**单独做消融或 Full-clean-train。
- 需要注意一点：由于训练时见过 20% 的空 prior，鲁棒性实验中的 No Prior 条件对 Full 来说是 in-distribution 的，论文里要明说。
- 另外，Problem Formulation 草稿里现在写着 80/20，这属于训练协议，应当移到 Method。

---

## 18. Baseline 公平性（第十八到二十部分）

### 18.1 RQ1（候选干预是否有用）：**现在无法干净地隔离。**

- Full vs B1：同时改变了“有没有干预”和“prior 如何组织”（B1 读的是静态的 G^H − G^K）；
- B1 vs B2：同时改变了“有没有干预”和“能不能用 prior”。

**建议选 C：重新定义 B1**，得到一条“每一步只加一个因素”的阶梯：

| ID | 定义 | 相对上一行新增的因素 |
|---|---|---|
| B0 | 非图 set-Transformer（见 18.3），输入原始合同字段 | — |
| **B1（重定义）** | 当前 **G^K**，不做干预，**不用 prior** | 图结构 |
| B2 | 合同干预，不用 prior | **候选干预**（B1 → B2：RQ1 的干净对比） |
| Full | + 纯交互 prior | **prior**（B2 → Full：RQ2） |
| **A_CAT（新增）** | Full，但 prior 输入改为四视图 concat 并零锚定 | 检验 D^P 的限制（RQ3，并回答第八部分的问题） |
| A_Q | λ_Q=0 | RQ4 |
| A_B | 去掉 tanh | 有界性 |

必跑配置仍然是 7 个。旧 B1（静态 G^H 策略）改名为 **B1-H**，和 **A_DD** 一起降为 P2 可选。B1-H 回答的是“不做干预，用静态图 prior 行不行”，这不是核心主张，而且 A_CAT 已经覆盖了“静态信息”这个问题。

### 18.2 各 RQ 的归因等级（写论文时必须标清）

| RQ | 对照 | 归因等级 |
|---|---|---|
| RQ1 干预 | B1 → B2 | **isolated**（重定义之后） |
| RQ1 图 | B0 → B1 | isolated（B0 足够强的前提下） |
| RQ2 prior 增量 | B2 → Full | isolated（训练时是否使用 prior，包括 80/20 协议作为整体） |
| RQ3 交互对比 | Full vs A_CAT（A_DD 可选） | isolated |
| RQ4 Q | Full vs A_Q | isolated |
| RQ5 Actor 依赖 | 同一 checkpoint 置零 / 置换 | 机制敏感性，**不是收益归因** |
| RQ6 扰动 | 同一 checkpoint 的五种条件 | 行为对比，不是因果识别 |
| RQ7 组合泛化 | ID vs holdout | overall empirical |

### 18.3 B0 是否足够强（第二十部分）

B0 的输入里已经有每个候选的 ADD / DEL 字段，等于**已经以扁平 token 的形式拿到了名义变化**。所以 B0 → B2 实际检验的是“图传播”是否优于“扁平效果 token”，这是一个公平的对比。

**最小改进**：把 B0 的 DeepSets / MLP 换成一个 2 层的 set-Transformer，让候选 token 对事实和目标 token 做 cross-attention，参数量和 Full 匹配。这样可以挡住 “Graph 只是赢了弱 MLP” 的质疑。改动约几十行代码。

---

## 21. 写作 readiness 与实验并行（第二十一、三十三部分）

| 类别 | 内容 |
|---|---|
| **写论文前必须确定** | 贡献重组（PW-02）；baseline 阶梯（PW-08）；w_t 和 γ̄ 两个默认值（PW-05、PW-06）；符号表（PW-09） |
| **可以和实验并行补** | Related Work 深挖；Experimental Setup 的平台和预算绑定；Figure 1 / 2 的定稿；A_CAT 和 set-Transformer B0 的实现 |
| **投稿前才需要** | 主表、学习曲线、消融、鲁棒性、组合泛化的数字；seed 数量；Figure 5 的真实案例；Abstract 最后一句 |

---

## 22. 任务设计原则（第二十二部分）

确实存在“mask 已经把答案筛出来”的风险。例如 Shared Prerequisite：如果 OPEN 是唯一的合法候选，就没有决策可言。原则如下：

1. **决策密度**：每个 episode 至少有 ≥2 个决策点，这些点上有 ≥2 个合法候选，且它们的长期后果不同。在 Stage 0 统计“有效决策点比例”并在论文中报告。
2. **合同不给答案**：合同只描述局部的前置和效果；长期好坏由**合同之外但可观测的依赖**决定。例如红块压在盖子上会让 PLACE(blue) 执行失败或耗时，但 PLACE 的合同里没有 ¬OnLid(red) 这个前置。
3. **prior 非冗余**：有价值的 VLM 关系应当表达第 2 条里的那种依赖；合同里已有的联系会被去重。
4. **错误可恢复但有代价**：选错通常损失时间，而不是直接导致 episode 失败，这样折扣回报才能提供梯度。
5. **可学习性门槛**：随机 masked 策略的成功率 ≥5%（不满足就缩短 horizon 或放宽 deadline，而不是加课程）。
6. **依赖在 4 跳以内**：关键依赖应当在 R-GCN 的感受野内。
7. **有意包含 prior 无用或误导的 case**：例如目标已经完成，或者关系语义正确但当前成本过高。这是“correctness ≠ utility”的直接展示。

五个模板里，**T_B（多对象选择）和 T_C（中间重定位）最能展示方法**，建议作为主表的两个核心 family；T_E 的实现成本最高，可以放到最后。

---

## 23. “高级感”与叙事顺序（第二十三部分）

**Method 的中心应该是一个算子，而不是一条流水线**：*contract–prior interaction contrast over candidate nominal interventions*。

叙事顺序：
1. 一个具体的反例：局部合法的技能破坏了后续条件，而一条语义关系是否有用取决于选择了哪个候选；
2. 观察：prior 的效用是**候选条件化**的；
3. 算子：2×2 视图（合同 / 增强 × 当前 / 名义）及其交互对比；
4. 性质：精确的 null、有界影响、置换等变；
5. 学习：只从结果中学；
6. 最后才提到 R-GCN 和 PPO，并且明确是“标准组件”。

写作上**避免**用 “we use VLM + R-GCN + PPO” 开头的句式，模块名不要出现在 Introduction 的贡献列表里。

---

## 24. 方法名称（第二十四部分）

- CP-DISR 的缩写尚可，但 “Differential Intervention” **容易被读成因果干预（do-calculus）**，DISR 也不好读。
- **主推荐：保留缩写 CP-DISR，不改算法**。在正文里把 intervention 一律写成 *nominal intervention*，首次出现时定义，并加一句 “not a causal intervention”。已有的材料都用这个名字，改名的收益有限。
- 如果愿意改名，备选：
  1. **PRICE**：PRior–Intervention Contrast Encoding（直接点出核心算子，好读）；
  2. **CPIC**：Contract–Prior Interaction Contrast；
  3. **NICE-Skill**：Nominal Intervention Contrast Encoding for Skill selection。

---

## 25. Abstract 审查（第二十五部分）

逐句看 Abstract_v0：

| 句 | 问题 | 建议 |
|---|---|---|
| 1 “Long-horizon robotic tasks require…” | 正确但平淡 | 加上“局部合法 ≠ 长期有用”的冲突 |
| 2 “Skill contracts provide… VLMs can supply…” | 好 | 保留 |
| 3 “Such relations may be incomplete… making their decision use distinct from their semantic correctness.” | **这是最强的一句**，但被埋在中间 | 升级为问题陈述，并加上“depends on the candidate” |
| 4 “We present CP-DISR…” | 好 | 保留 |
| 5 “For each executable skill, we apply its nominal success effects…” | 技术细节太早 | 合并为一句算子描述 |
| 6 “A shared relational graph encoder … dual difference…” | “shared relational graph encoder” 不需要出现在摘要里 | 改为 “an interaction contrast isolates how the relations alter each candidate's consequences” |
| 7 “zero-anchored bounded residual” | 好，但要点出它的作用 | “…so that the policy provably reduces to the contract policy when relations are absent and deviates from it by a bounded factor otherwise” |
| 8 “Skill-level PPO and auxiliary action-value supervision…” | **写得过重** | 压成半句：“trained end-to-end from task outcomes only” |
| 9 “targets conditional prior utilization rather than relation verification or repair” | 防御性表述，放在摘要里偏弱 | 改为正面陈述，或者删除 |
| 10–11 实验占位 | 可以 | 改成下面的结构 |

**Recommended Abstract Structure（6–7 句）**：

1. 场景和冲突：长时程技能选择中，局部合法的技能可能破坏后续条件。
2. 两种知识来源：合同（可靠但有限）和 VLM 关系（丰富但不完美）。
3. **关键观察**：一条关系是否有用，取决于它是否改变了某个候选的后果。
4. 方法：对每个候选做名义干预，用 contract–prior interaction contrast 得到“prior 对该候选后果的改变”，再通过零锚定、有界的残差调制合同策略。
5. 性质和训练：精确 null 与有界影响；只从任务结果端到端训练，不修改 prior。
6. 实验占位：在 [N 个任务族 / 平台] 上，对照 matched baselines 和同一 checkpoint 的 prior 扰动…… [结果]。
7. （可选）一句话收束 “correctness ≠ utility” 的意义。

---

## 26. Introduction 审查（第二十六部分）

现在的 Introduction 结构合理，但有两个问题：**技术细节出现得太早**（第 4 段就出现了“只读名义成功事实副本”），而且**缺少一个具体的例子**。

**最终 Introduction 叙事（5 段 + 贡献列表）**：

1. **一个具体例子（配 Fig. 1 左半部分）**：桌上有红、蓝两个方块和一个盒子，目标是把蓝块放进盒子。红块压在盒盖边缘。PICK(blue) 和 MOVE(red, buffer) 都合法。VLM 给出了关系 “MOVE(red) supports PLACE(blue)”。在这个状态下这条关系有用；但如果盒盖已经打开，或者红块已经移走，它就没用了。**关系没有变，效用变了。**
2. **两种知识**：合同告诉我们动作会改变什么；VLM 补充合同之外的联系；两者各自的局限。
3. **现有方法的不足**：把 prior 当作 action bias / 计划 / mask，要么过度信任，要么不区分“候选是否改变了这条关系的作用”。这里引用 2607.19616 关于 hard mask 在错误 KG 下脆弱的证据，以及 Yan et al. 2025 等 action-prior 路线。
4. **我们的想法（只讲算子）**：2×2 视图和交互对比、零锚定、只从结果学习。
5. **边界**：我们研究的是如何利用 prior，不是修复 prior。**一句话讲清楚即可**，不要连续三句否定。
6. 贡献列表（第 38 节），以及一句结果占位。

---

## 27. Problem Formulation 审查（第二十七部分）

- **整体严谨**：SMDP、部分可观测、三值事实、候选、mask、固定 prior、奖励、duration 都已具备。
- **符号冲突（P1，PW-09）**：
  - 上界 **B** 和 baseline 名 B0 / B1 / B2、B_batch 冲突 → 改为 **β**；
  - **K** 既表示合同上标又表示候选数（2+2K）→ 候选数改为 **N_t**；
  - **F** 既表示事实又表示网络 F_K / F_P / f_P^node → 网络改为 **φ_K, φ_P**。
- **Method 内容混进了 Problem**：GRU 和 80/20 协议应当移到 Method。Problem 里只保留“一个可以依赖历史的策略”。
- **实验协议混进了 Problem**：“评价采用声明的冻结先验条件”应当移到 Experiments。
- 可以在 Problem 中正式定义 relation correctness 和 decision utility 的区别（一段话即可），为贯穿全文的主张打下基础。

---

## 28. Method 结构（第二十八部分）

建议的顺序（与现有顺序大体一致，只调整重心）：

- 4.1 Overview（Fig. 1）
- 4.2 Structural Views：合同图 + 语义关系 + 增强图（把原 4.2 和 4.3 合并，VLM prompt 等细节移到附录）
- 4.3 Candidate Nominal Interventions（短，定位为 afterstate 的名义版本）
- **4.4 Contract–Prior Interaction Contrast（Fig. 2，全文中心）**：定义、对称恒等式、纯交互原则、schema 与算子的一致性、零性质
- **4.5 Prior-Modulated Contract Policy**：零锚定有界残差，给出 Properties box
- 4.6 Outcome-Grounded Training：PPO、V、实际动作 Q、80/20、SMDP 折扣
- 4.7 Implementation Summary（重算、计算量 2+2N，具体放附录）

4.4 和 4.5 应当占 Method 篇幅的 40% 左右。

---

## 29. 主图设计（第二十九部分）

### Figure 1（方法总览，一栏宽或整页宽）

**必须出现**：
- 左侧：一个真实场景缩略图，标出红块、盒盖、蓝块；
- 两个输入：合同（小卡片：PRE / ADD / DEL）和 VLM 关系（一条虚线箭头，写明 “MOVE(red) ⇢ PLACE(blue)”）；
- 中间：**2×2 网格**（行：G^K / G^H；列：current / nominal after candidate i），四个小图共用一个 “shared encoder E” 标志；
- 网格右侧：D^K（横向差）和 D^P（交互对比），用颜色区分；
- 右侧：合同分数 b_i 加上有界残差 β·tanh(·)，得到 logit；再经过 mask、选择技能、真实执行；
- 一条从 outcome 回到网络的**虚线**，标 “learn from outcome only”；从 outcome 指向 VLM 的方向画一个 “✗ no update”。

**不应该出现**：GRU 细节、Verifier、cache、hash、Huber、PPO clip、V / Q 三个 head 的分叉、MUST_BIND 字段、工程模块名。

### Figure 2（交互对比的概念图）

- 用同一个例子，候选 = MOVE(red, buffer)。
- 四个面板对应四个视图，每个面板上高亮名义 patch 改变的命题（OnLid(red): T → F）和那条软边。
- 横向箭头标 D^K 和 D^H，纵向箭头标 S(F) 和 S(F̃)，中心标 **D^P = D^H − D^K = S(F̃) − S(F)**。
- 右侧小插图做对照：
  - (a) 候选 = PICK(blue)，没有触及源动作的邻域，D^P 小；
  - (b) 红块已经移走，同一条关系、同一个候选，D^P 变小。
  这对应第 9 节的门控讨论。示意图不画数值，也不给 latent 维度命名。

---

## 30. Related Work（第三十部分）

**现有引用的准确性**：
- R2（Chen, Thiébaux, Trevizan AAAI 2024）、R3、R4、R6、R10–R13 均正确。
- R8 SayCan 应该改引 **CoRL 2022（PMLR 205）**，不要只引 arXiv。
- R18 Inner Monologue 同样改引 **CoRL 2022**。
- R16 / R17（Qwen 文档）、R5（PyG 文档）、R15（Gymnasium 文档）**不要放进正文参考文献**，移到附录的实现说明或脚注。R15 可以换成 Pardo et al., *Time Limits in RL*（ICML 2018）。

**建议分为 4 节**：

1. **Language and vision-language priors for robot decision making**
   - 核心：SayCan、Inner Monologue、Text2Motion、PSL（ICLR 2024）、BOSS（CoRL 2023）、ELLM（ICML 2023）、LLM4Teach（IJCAI 2024）、Yan et al.（ICLR 2025）、uncertainty-aware LLM guidance（2411.14457）、SGA-ACR（2511.20993）、KG mechanisms（2607.19616）。
   - 差异：这些工作把 prior 当作计划、action prior 或 shaping，或者在线修改 prior；我们只让 prior 通过候选后果的交互进入策略，而且不修改 prior。
   - 最大撞车：**Yan et al. 2025**（imperfect LLM action prior），需要写清楚 static vs interaction 的区别。
2. **Symbolic action models, generalized policies, and afterstates**
   - 核心：ASNets、SLG / GOOSE、Ståhlberg et al.（KR 2022 / 2023、ICAPS 2022）、PLOI、Graph Learning for Planning 综述（2412.02136）、afterstates、ASGRL（ICML 2022）、LLM 生成 PDDL（Guan et al. NeurIPS 2023、LLM+P、InterPreT）。
   - 差异：它们评估完整的后继状态；我们只用名义成功分支，与部分可观测、真实失败共存，重点在 prior 如何改变这个后继。
   - 最大撞车：**Ståhlberg et al. 与 C1**。
3. **Action-effect and difference representations**
   - 核心：afterstate / successor features、Counterfactual Quotient Models（2608.22092）、action representation learning（Chandak et al. ICML 2019）。
   - 差异：它们消去的是动作之间的共同成分；我们做的是 prior × intervention 的交互对比。
   - 撞车风险：中，需要引用。
4. **Temporal abstraction and auxiliary value learning**
   - 核心：Options / SMDP、STAP、UNREAL、PPG。
   - 差异：标准组件，不作为贡献。

目标文献量：35–50 篇。上面列出的不少条目来自本次检索的摘要页和已知文献，**投稿前请逐篇核对 bib**。

---

## 31. Novelty Map（第三十一部分）

| Component | Closest Prior Work | Existing Idea | CP-DISR Difference | Novelty Strength | Collision Risk |
|---|---|---|---|---|---|
| Skill Contract | STRIPS / PDDL operators；skill 前后置接口 | 前置 + 名义效果 | 三值事实、名义成功 ≠ 真实成功 | **Not Novel**（适配） | 低 |
| Grounded action–proposition graph | SLG（Chen et al. 2024）、ASNets | 动作 / 命题二部图 | 三值特征 + 目标行读出 | **Not Novel** | 低 |
| R-GCN | Schlichtkrull 2018 | 多关系消息传递 | 无 | **Not Novel** | 无 |
| Candidate nominal intervention | Ståhlberg et al.；afterstates；PLOI | 用已知模型生成后继并评估 | 名义分支、部分可观测、真实失败 | **弱** | **高** |
| Latent before / after difference（D^K） | CQM 2608.22092；successor-state 比较 | 差分消去共享成分 | 目标对齐、零点性质 | 弱–中 | 中 |
| **Contract / prior interaction contrast（D^P）** | 未发现直接对应 | 2×2 交互对比（统计学概念） | 用作 prior 进入策略的**唯一**通路；schema 与算子一致 | **强（核心）** | 低（检索范围内） |
| Zero-anchored bounded residual | residual policy learning、KL-regularized priors | 在基础策略上加有界修正 | 输入为交互对比，具有精确的 null 性质 | 中（作为 C2 的一部分） | 中低 |
| Structural Q | UNREAL、PPG、auxiliary Q | 辅助价值监督 | 候选级，只监督执行动作 | **弱**（训练机制） | 中 |
| VLM semantic prior | SayCan、SayPlan、KG-RL | 语言知识辅助决策 | 后果锚定的 schema、只生成一次 | 弱 | 中 |
| Fixed-prior utilization（问题设定） | Yan et al. 2025、LLM4Teach、2607.19616 | 不完美 prior + RL | 候选条件化、不修改 prior、可控扰动协议 | 中 | 中 |

**真正需要守住的**：D^P 算子、纯交互原则、与之配套的 schema、零锚定有界调制，以及它们在“固定不完美 prior 利用”这个问题上的组合。

---

## 32. 是否需要 Theory（第三十二部分）

**不需要定理，保留一个 Properties box（或 Proposition 1）即可。** 内容（每条证明一两行）：

- (i) **Exact null**：R=∅ ⇒ D^P=0 ⇒ π=π_K；空 patch ⇒ D^K=D^P=0；E 可加分离 ⇒ D^P=0。
- (ii) **Bounded deviation**：$e^{-2\beta}\le\pi/\pi_K\le e^{2\beta}$。
- (iii) **Symmetric reading**：D^P = S(F̃_i) − S(F)。
- (iv) **Candidate permutation equivariance**：候选重排时，概率相应置换。

它们的价值在于**把设计原则变成可检验的陈述**，并且和实现级验收（22.5）一一对应。不要为了显得高级而写收敛性或最优性定理。

---

## 34. 问题分级汇总（第三十四部分）

### P0 — 写论文前必须解决

**无。**

### P1 — 写论文过程中应修正

贡献重组；交互对比的命名和对称恒等式；effect_fact_ref 的表述；claim 强度；baseline 阶梯（B1 重定义 + A_CAT + set-Transformer B0）；w_t 默认值；γ̄ 标定规则；符号冲突；Problem / Method 的边界；Related Work 重建；Figure 1 / 2 重画；草稿去防御化。

### P2 — 实验出来后再决定

- A_DD / B1-H 是否补跑；
- effect-anchored 消融（仅当 4A 诊断显示门控问题时）；
- 候选节点差分行（仅当 D 被稀释时）；
- seed 扩展；
- 梯度余弦诊断；
- T_E 是否纳入。

### P3 — Future Work

- 动态重新 query VLM；
- M1+；
- 更长依赖（>4 跳）和 Graph Transformer；
- 真机扩展。

---

## 35. 方法修改的严格检查（第三十五部分）

本次审查**不建议修改 Current Method 的结构**。唯一涉及 v2.1 配置的两项（PW-05、PW-06）都是训练默认值，对照第三十五部分的六个问题：

1. **哪里不成立**：两者都不是逻辑错误，但在长时程任务上会带来实际训练风险（后期样本权重被压低、有效 horizon 过短）；
2. **性质**：属于“可以更好”；
3. **不改的后果**：不会影响论文是否成立，但可能影响实验能否学起来；
4. **最小修改**：各改一个配置值或一条规则；
5. **重写量**：Method 公式中去掉一个下标（$\mathbb E_{w_t}$ → $\mathbb E_t$），外加一句脚注；
6. **新实验**：不需要。

---

## 36. Future M1+（第三十六部分）

- **是合理的 Future Work。** keep / drop 对照加真实结果得到的预测效用，是比 M1 更细粒度的监督对象，和 M1 之间是“升级方向”，不是“补丁”。
- **值得保留一段**，建议 4–6 句，放在 Discussion 的最后；公式可以只保留 u_{t,e} 一行，也可以完全不写公式。
- 不要写成“当前方法缺少逐关系归因”，而是写成“当前方法回答的是结构能否被利用；下一个问题是能否形成逐关系的效用经验”。

---

## 37. Final Pre-Writing Change List

| ID | Priority | File / Section | Current Issue | Exact Change | Why | Blocks Writing? |
|---|---|---|---|---|---|---|
| PW-01 | P1 | Spec §4、Intro、Abstract | 中心问题被读成 “VLM 会犯错” | 中心句改为：“relation utility is candidate-conditioned: a relation matters only insofar as it changes a candidate's consequences.” | 提供非常识性的问题中心 | 否（写 Intro 时执行） |
| PW-02 | P1 | Spec §5、Intro 贡献段 | C1 / C3 新颖性弱，贡献列表像三个模块 | 改为 {问题表述；contract–prior interaction contrast 方法（含名义干预和结果训练）；可控评估协议}。C1 写成 “adopt and adapt afterstates”；Structural Q 降为训练设计 | 避免 “natural combination” 质疑 | **写 Intro / Method 之前定** |
| PW-03 | P1 | Method §4.4、Spec §14 | “dual difference / difference of differences” 形式感强、解释弱 | 正式命名为 **contract–prior interaction contrast**；加入对称恒等式 D^P=S(F̃)−S(F)；声明不是因果 DiD；加入 “schema ⇄ operator consistency” 段落 | 把 C2 从特征工程提升为一个有原则的算子 | 否 |
| PW-04 | P1 | Method §4.2、system_prompt.txt、relation_schema 描述 | effect_fact_ref 的语义与计算错位，表述不清 | 定义为 consequence-grounding admission constraint；Limitations 加一句 skill-level mediation；prompt 中 “may be inactive in later states” 改为 “are not required to hold in later states”。**不改图结构** | 预先回应审稿人 “为什么要它、它进不进网络” | 否 |
| PW-05 | P1 | manifest `ppo.actor_episode_discount_weight`、Spec §17.2 | 时长折扣下 w_t 会严重压低后期决策的权重（300 s 时约 0.05） | 默认值设为 `false`；L_clip 改用普通有效样本均值；脚注引用 Nota & Thomas 说明这一选择 | 长时程任务后段可学习 | 写 §4.6 前定 |
| PW-06 | P1 | manifest `gamma_reference_value`、Setup | γ̄=0.99/s 只适合约 120 s 以内的任务 | 规则：γ̄^{T_typ} ≥ 0.3，按任务绑定并写入 runtime manifest | 避免有效 horizon 过短 | 否 |
| PW-07 | P1 | Abstract、Intro、Discussion | claim 强度在 “learn when” 和 “interface” 之间摇摆 | 统一为 “learns a candidate-conditioned bounded modulation from outcomes; we test selectivity” | 可以辩护，也可以被实验支持 | 否 |
| PW-08 | P1 | Spec §23–25、manifest `variants` | RQ1 无法隔离；A_DD 回答不了 concat 质疑；B0 可能被说弱 | B1 重定义为 G^K-only、无 prior；新增 A_CAT（零锚定四视图 concat）；B0 改为 2 层 set-Transformer；旧 B1 改名 B1-H，与 A_DD 一起降为可选。必跑仍是 7 个 | 每个 RQ 都有干净对照 | 写 Experiments 前定；不阻塞 Method |
| PW-09 | P1 | 全文符号 | B / K / F 多重含义 | 上界 B → β；候选数 K → N_t；网络 F_K / F_P → φ_K / φ_P | 审稿可读性 | **写 Problem 之前定** |
| PW-10 | P1 | Problem_Formulation_v0 | GRU、80/20、评价条件写进了 Problem | 移到 Method §4.6 / Experiments；Problem 中加入 correctness vs utility 的定义 | Problem 和 Method 分离 | 否 |
| PW-11 | P1 | 全部 drafts | 防御性否定句密集，像规范文档 | 每节最多一句边界声明；其余集中到 Limitations 一段；主动语态陈述设计 | 论文的可读性和说服力 | 否 |
| PW-12 | P1 | Related Work、附录 C | 只有 18 条；把 API 文档当参考文献；SayCan / IM 引用的是 arXiv | 按第 30 节 4 个小节重建到 35–50 条；文档类引用移到脚注；更正 venue | 正式投稿的底线 | 否（可并行） |
| PW-13 | P1 | figures/ | 现有两张图是工程信息流 | 按第 29 节重画 Fig. 1（2×2 为中心）和 Fig. 2（交互对比概念图）；现有 mermaid 移到附录 | 让审稿人第一眼看到的是算子 | 否（与 Method 同步） |
| PW-14 | P1 | Experimental Setup | 没有可学习性门槛，没有决策密度指标 | 加入：随机 masked 策略成功率 ≥5%；有效决策点比例；把 T_B / T_C 作为核心 family | 降低实验风险 | 否 |
| PW-15 | P1 | Method | 缺少形式化的性质陈述 | 加入 Properties box（第 32 节的 i–iv） | 把设计原则变成可检验的陈述 | 否 |
| PW-16 | P2 | 实验 | — | A_DD、B1-H、effect-anchored 变体、候选节点差分行、梯度余弦诊断：按诊断结果决定是否加入 | — | 否 |

> **No method-level blocker remains.** 可以开始写。

---

## 38. 最终论文定位（第三十八部分）

**Final Paper Title（推荐）**
*CP-DISR: Candidate-Conditioned Use of Imperfect Semantic Priors for Long-Horizon Skill Selection*

备选：*Learning to Use Fixed Imperfect Semantic Priors through Contract–Prior Interaction Contrasts*

**One-sentence Story**
一条语义关系有没有用，要看它是否改变了某个候选动作的后果。所以我们对每个候选做名义干预，只提取 prior 与这个干预之间的交互，把它作为对合同策略的有界调制，并且只从真实任务结果中学习如何使用它。

**Problem Statement**
在固定技能合同和固定、不完美的 VLM 关系下，学习一个长时程技能策略，使它对每条关系的使用都以该关系如何改变各个可执行候选的名义后果为条件；训练中不修改 prior，也不使用关系正误标签。

**Central Algorithmic Idea**
Contract–prior interaction contrast：
$D^P_i=[E(\tilde G^H_i)-E(G^H)]-[E(\tilde G^K_i)-E(G^K)]$，
它是 prior 进入策略的唯一通路，再经零锚定有界残差调制合同策略。

**Three Contributions**
1. 把固定不完美 prior 的利用形式化为候选条件化问题，区分 relation correctness 与 candidate-level decision utility。
2. CP-DISR：基于名义干预的 contract–prior interaction contrast，配套后果锚定的关系 schema；零锚定有界调制带来精确 null 和有界偏离；只从结果训练（PPO 加候选级价值监督）。
3. 一套可控评估协议：matched baseline 阶梯、同一 checkpoint 的 prior 扰动、Actor 依赖诊断，并在 [任务] 上给出实证 [结果占位]。

**Main Claim**
在匹配的信息和交互预算下，与只使用合同干预、只读当前图、非图表示以及不受限的 prior 融合相比，以交互对比方式使用固定不完美 prior 能够 [提升 / 不降低]长时程技能选择的 [成功率 / 效率]，并且其行为对 prior 扰动有可测量、有界的敏感性。方括号内的内容以实验结果为准。

**Claims We Should NOT Make**
- 模型能识别或修复错误关系，或能估计逐条关系的正确率；
- D^P 是因果效应、价值增量或关系重要性；
- 有界残差保证坏 prior 不会降低回报；
- 名义干预是物理预测或世界模型；
- 候选干预本身是新的（afterstate 早已存在）；
- 结果适用于真机或任意 VLM（除非实验覆盖）；
- Full 在所有任务上都最好。

**Best Reviewer Mental Model**
“他们注意到 prior 的价值取决于候选动作，于是设计了一个 2×2 的名义干预 × prior 视图，只让交互项影响策略，而且没有 prior 时严格退化为合同策略。消融检查了这个限制是不是比直接 concat 更好。”

---

## 39. Final Paper Outline（第三十九部分）

| 节 | 目的 | 核心内容 | 公式 | 图 | 引用 | 现在能写？ | 还缺什么 |
|---|---|---|---|---|---|---|---|
| **1 Introduction** | 建立问题中心 | 1.1 例子；1.2 两种知识；1.3 现有做法的缺口；1.4 我们的算子；1.5 贡献 | 无 | Fig. 1 左半部分 | 8–12 | **能** | 最后一句结果 |
| **2 Related Work** | 定位与划界 | 2.1 LLM / VLM priors；2.2 Symbolic models & afterstates；2.3 Difference representations；2.4 SMDP & auxiliary learning | 无 | 无 | 30+ | 能（并行） | 文献精读 |
| **3 Problem Formulation** | 形式化 | 3.1 Skill SMDP 与部分可观测；3.2 Contracts & facts；3.3 Fixed imperfect priors；3.4 Correctness vs utility；3.5 Objective | SMDP 折扣、J(θ)、mask | 无 | Options、Pardo 2018 | **能**（先定 PW-09） | 无 |
| **4 Method** | 核心 | 4.1 Overview；4.2 Structural views；4.3 Nominal interventions；**4.4 Interaction contrast**；**4.5 Prior-modulated policy + Properties**；4.6 Outcome-grounded training；4.7 Complexity | D^K、D^P、S(F)、u^P、ℓ、ratio bound、y^Q、L_total | Fig. 1、Fig. 2、Alg. 1 | R-GCN、PPO、GAE | **能**（先定 PW-02 / 05） | 层表（附录） |
| **5 Experiments** | 验证 | 5.1 Tasks & platform；5.2 Baselines（阶梯）；5.3 Main results（RQ1–2）；5.4 Ablations（RQ3–4）；5.5 Mechanism（RQ5）；5.6 Prior perturbation（RQ6）；5.7 Composition（RQ7） | 指标定义 | Tab. 1–3、Fig. 3–5 | VIMA 等 | 5.1–5.2 能写骨架 | 平台绑定、全部结果 |
| **6 Discussion & Limitations** | 边界 | correctness vs utility；固定 prior 的得失；skill-level mediation；4 跳限制；稀疏 reward；M1+ | 无 | 无 | 少量 | 能写 80% | 按结果调整 |
| **7 Conclusion** | 收束 | 3–4 句 | 无 | 无 | 无 | 结果后 | 结果 |
| **Appendix** | 可复现 | A 合同 / schema / prompt；B 完整算法和 GRU 重算；C 超参数和层表；D 扰动协议；E 实现验收；F 信息 / 梯度流（现有 mermaid） | — | 现有两张 mermaid | 文档类脚注 | 能 | 运行绑定 |

---

## 40. 写作顺序（第四十部分）

1. **符号表和贡献定稿**（PW-09、PW-02，半天）：后面每一节都依赖它们。
2. **Method §4.3–4.5**（核心算子和 Properties）：这部分已经冻结、最确定，写完也就定下了全文的语言。
3. **Fig. 2 → Fig. 1**：先画概念图，再画总览图。画图时能暴露 Method 中叙述不清的地方。
4. **Problem Formulation**：Method 写完后，回头只保留 Method 真正需要的形式化内容。
5. **Method §4.1、4.2、4.6、4.7 和 Algorithm 1。**
6. **Introduction**：此时中心句、例子和图都已经就位。
7. **Related Work**：可以和 1–6 并行做文献精读，最后成稿。
8. **Experimental Setup 与 baseline 定义**：同步推动 Stage 0 的平台绑定和可学习性预检。
9. **Discussion / Limitations**：先写 80%，结果出来后调整。
10. **Abstract**：最后写，结果句留占位。

这样排的原因是：先写最确定、最核心的部分，让它决定全文的语言；Introduction 和 Abstract 依赖中心句和图，放在后面；实验相关的章节和实验同步推进，不让实验阻塞写作。

---

# Final Decision

1. **当前研究是否已经达到可以正式开始写论文的程度？** 是。
2. **是否还有 Method 级别的问题必须先解决？** 没有。effect_fact_ref 的错位是真实的，但不是结构错误，保留 Action → Action，用表述解决。w_t 和 γ̄ 是 v2.1 内的训练默认值，写 §4.6 前定下即可。
3. **（不适用）**
4. **Freeze Current Method and start paper writing now.**

后续实验负责验证 claim，不再决定研究是什么。最需要尽早启动的实验侧工作是：Stage 0 的可学习性预检（随机成功率 ≥5%）和 baseline 阶梯的实现（B1 重定义、A_CAT、set-Transformer B0）。

---

### Sources（本次联网核对）

- [Ståhlberg, Bonet, Geffner — Learning General Policies with Policy Gradient Methods (KR 2023)](https://proceedings.kr.org/2023/63/) · [arXiv 2512.19366](https://arxiv.org/abs/2512.19366)
- [Learning Generalized Policies without Supervision Using GNNs (KR 2022)](https://proceedings.kr.org/2022/49/kr2022-0049-stahlberg-et-al.pdf)
- [Learning General Optimal Policies with GNNs (ICAPS 2022)](https://ojs.aaai.org/index.php/ICAPS/article/view/19851)
- [Counterfactual Quotient Models (arXiv 2608.22092)](https://arxiv.org/abs/2608.22092)
- [Efficient RL with LLM Priors (ICLR 2025)](https://openreview.net/forum?id=e2NRNQ0sZe)
- [Guiding RL Using Uncertainty-Aware LLMs (arXiv 2411.14457)](https://web3.arxiv.org/abs/2411.14457?context=cs)
- [LLM as a Policy Teacher / LLM4Teach (IJCAI 2024)](https://www.ijcai.org/proceedings/2024/0627.pdf)
- [The Mechanism Matters: When Knowledge Graphs Help RL (arXiv 2607.19616)](https://arxiv.org/abs/2607.19616)
- [Subgoal Graph-Augmented Planning for LLM-Guided Open-World RL (arXiv 2511.20993)](https://arxiv.org/abs/2511.20993)
- [Leveraging Approximate Symbolic Models for RL via Skill Diversity (ICML 2022)](https://proceedings.mlr.press/v162/guan22c.html)
- [Planning with Learned Object Importance (AAAI 2021)](https://ojs.aaai.org/index.php/AAAI/article/view/17421/17228)
- [Graph Learning for Planning: The Story Thus Far (arXiv 2412.02136)](https://arxiv.org/abs/2412.02136)
- [Afterstate RL for Continuous Control (OpenReview)](https://openreview.net/forum?id=XO944P8prc)
- [Plan-Seq-Learn (arXiv 2405.01534)](https://arxiv.org/abs/2405.01534)
- [Leveraging Pre-trained LLMs to Construct and Utilize World Models (NeurIPS 2023)](https://openreview.net/pdf?id=zDbsSscmuj)
- [InterPreT](https://interpret-robot.github.io/) · [UniDomain (arXiv 2507.21545)](https://arxiv.org/html/2507.21545v1)
- [Can Graph Learning Improve Planning in LLM-based Agents? (NeurIPS 2024)](https://arxiv.org/abs/2405.19119)
- [LLM-Guided RL through Policy Modulation (arXiv 2505.20671)](https://pith.science/paper/2505.20671)
