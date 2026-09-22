# CP-DISR 独立复核与最终研究／实验路线裁决

2026-09-22 · 文档3.1 · Method v2.1.1 · Experimental Plan v1.1

# 1. Executive Verdict

**接受“可以立即写论文、无需重构核心架构”的总体裁决；不接受Claude把若干工程选择提升成数学结论或普遍必要条件。**

按Claude末尾PW-01至PW-16的主建议分类，本次4项主建议可直接采纳方向（PW-07/09/10/15，性质仍须补足适用条件），12项修改后吸收；这不是16项论证全部正确，更不表示所有支持理由被接受。特别否决：随机成功率5%的硬准入；静态SOFT_ORDER必在DP消失；Pool(ZK)+DK必无信息损失；去掉Actor prefix仍精确优化原J；逐步query必破坏on-policy；随机图范数证明effect gate；常规baseline ladder天然是新评价协议。

**最终版本：文档3.1，方法CP-DISR/M1 v2.1.1，Experimental Plan v1.1。**图、DP、主Actor、关系schema不变；patch版本记录真实发生的训练权重/折扣规则变化。第一暂停点从原多seed矩阵拆为11个默认新训练，不一次性运行7配置。

本轮没有执行RL、VLM调用或机器人操作，也没有Claude toy源码，因此不复述其20seed数字为已复现证据。已检索一手文献并检查相关方法；代数/配置检查单独记录，不能代替生产单测。

当前可用材料包括v3.0主文与接口包、Claude审查全文和原实验Agent Prompt；旧生成版实验执行包未挂载，v3.0也明确记录了历史预算冲突。本轮预算是**新裁定**，不是假称恢复了唯一旧冻结值。[V3 §29.2；C §0、35]


---

# 2. Current Method是否继续冻结

**继续冻结核心架构；没有发现必须重构DP、graph、Actor或schema的P0方法矛盾。No method-level blocker remains.** 这不是代码验收、新颖性证书或论文录用保证。

真正改变Method实现的是Actor loss的采样权重与duration discount规则：它们改变优化过程，必须同步公式、配置与版本。Baseline、A_CAT、B0增强、任务/奖励暴露预检主要改变Experimental Setup。贡献重组、interaction命名、图示、边界和Related Work主要改变Writing。

Claude遗漏或说错的地方主要在**论证层**：R不改变合同名义后果；缺少完整ZK时无损重参数化不成立；一个候选的空patch不保证其最终softmax概率等于合同reference；共享Q因candidate-set context仍会向未执行候选特征间接回传。把这些句子原样写进论文会产生错误主张，但修正它们不要求新增网络。

冻结源信息的边界保留：合同错误须按证据修订接口版本，VLM软关系不自动修图；真实结果学习其使用方式。Future M1+不进入当前版本。


---

# 3. Final Research Positioning

正式名称保留 **CP-DISR：Contract–Prior Differential Intervention for Skill Reasoning**。论文题目采用：**CP-DISR: Contract–Prior Interaction Contrasts for Long-Horizon Skill Selection**。M1只作内部代号；全文将intervention限定为nominal symbolic intervention。

中心问题是：**在相同技能合同与观测条件下，让固定语义先验通过“候选名义变化的编码响应”影响策略，是否比不做干预或直接读取四视图更适合长期技能选择？**固定不完美先验仍是问题背景，candidate-conditioned prior interaction是算法组织中心。

关键观察不是“关系只有改变物理后果才有价值”。输入关系不修改环境动力学，也不修改合同的名义patch。更准确地说，关系的决策用途依赖候选、状态、目标与后续选择；本文检验一种有意限制：先验只经候选变化的表示交互参与决策。静态信息可能有用，A_CAT直接检验排除它的代价与收益。

**One-sentence story.** We study whether restricting fixed semantic priors to their interaction with candidate-induced nominal representation changes improves skill selection under matched observations, contracts, and interaction budgets.

主方法没有显式逐边utility、关系真假或知识修订模块。不能将集合R的编码交互写成已经识别“每条关系的真实贡献”。固定cache隔离生成与利用并支持复现；在线重新query并不必然破坏on-policy，故不以这一错误理由论证固定cache。[W05，W14；本轮设计]


推荐研究问题顺序：RQ1同合同下显式候选变化是否有帮助（B1-K/B2）；RQ2软关系交互是否增加用途（B2/Full）；RQ3纯交互限制对比完整四视图融合是否值得（Full/A_CAT）；RQ4辅助候选Q是否提供增量（Full/A_Q）；其余Actor依赖、先验缺省/扰动、组合holdout为机制和范围证据。

该中心比“VLM会错”更能体现算法，但不是语义改名就自动更强。A_CAT必须能证伪pure-interaction假设；不能先断言所有静态信息无用再用实验印证既定答案。


---

# 4. Final Contribution Structure

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


---

# 5. D^P / Interaction Contrast最终定义

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

若$E(F,R)=A(F)+B(R)$，则$D_i^P=0$。消去的是对所比较变化可加分离的表示成分，不是所有拓扑静态边，也不是所有可能有用的行动先验。

在线性诊断$E(F,R)=M_Rx(F)+b_R$中，

$$D_i^P=(M_R-M_\varnothing)[x(\widetilde F_i)-x(F)],$$

边不变仍可改变变化消息的传播。SOFT_ORDER在数学上也可能非零，主schema删除它是因order-only职责不合，而非必被消去。

完整$(Z^K,D^K)$与$(Z^K,\widetilde Z^K)$可逆，但当前Actor只把Pool$(Z^K)$放进上下文，并继续压缩候选特征。因此不能据上述可逆恒等式声称整个网络对当前/后继信息无损；也不能因$D^H=D^K+D^P$而称A_DD与Full必然同信息。

目标局部没有路径时，全局非线性读出仍可能混合不同组件。非零DP不是关系有用性的证明；DP大小与动作概率变化也不是单调关系。


---

# 6. A_CAT审查

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



**裁决：A_CAT成为必跑核心消融，替代A_DD的默认位置，不增加主要配置总数。**也不把CAT当成一个更弱的“随便concat大向量”基线：必须row-aligned、同context、合同重复锚点、同下游head和真实匹配预算。

若A_CAT仅在ID更好而Full在已预定组合/扰动更好，可以报告准确trade-off；若CAT在主要相关条件中持续占优，则“纯交互限制有性能优势”应撤回。仍可以报告设计与负结果，或将未来融合改动明确升v2.2；不能把负结果重新解释成“我们理论上正确所以更好”。


---

# 7. Baseline Ladder最终版

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



---

# 8. B0 Set Transformer最终设计

输入为当前事实（含T/F/U与goal sign）、签名目标、全体grounded技能及其原始PRE_POS/PRE_NEG/ADD/DEL/UNKNOWN文字项记录、当前候选角色和同源实际观测。每条合同记录包含owning action、predicate、参数的有序角色/类型及一致对象绑定信息；不得只保留效果类别而丢掉“谁影响谁”。任意显示ID不当作可学习语义标签；对象特征来自所有方法共同可见的对象表/观测。

所有token通过共享字段投影进入$d=128$。使用两个标准SAB：multi-head self-attention，4 heads，逐token FFN128→256→128、ReLU、残差、LayerNorm，dropout0，无集合位置编码。候选query对全部编码token做一层cross-attention，输出128，再与共同obs/GRU及候选上下文用MLP256→128融合。V/Q/Actor各自head与其他方法保持相同输出角色。SAB/MAB属于现有Set Transformer组件。[W15]

attention只使用padding mask，不输入graph adjacency、PRE/ADD稀疏attention mask、最短路、层级答案或名义赋值后的事实；不调用NOMINAL_PATCH。token包含原始合同关联，因此它可以通过attention学习关系；“非图”指无显式graph message-passing拓扑，不是禁止一切关系计算。ADD/DEL原字段也不等于已经应用当前T/F/U、条件效果和全局frame后的名义后继。

实现前冻结宽度/层数，不按Full领先与否搜索B0容量；参数差异透明报告。若已有更强且信息匹配的标准attention基线，可在未跑任何主结果前替代一次，但不能增加另一项必跑。第一版采用上述唯一实现。


**裁决：本轮升级为上述两层Set Transformer，不再额外跑原DeepSets版本。**它不是唯一可能的强baseline，但用一个标准成熟attention实现替换弱读出，能直接回应“只赢了弱MLP”，成本比追加更多对照小。所有方法都必须获得完整原始合同、目标、实际观测；参数完全匹配既不是公平性的充分条件，也不是本轮硬目标。


---

# 9. Structural Q最终定位

正式名称采用 **Auxiliary Candidate Q（代码模块可保留structural_q）**，归入Outcome-Grounded Training，不再单列原创C3。

$$
y_t^Q=\operatorname{sg}[r_t+\Gamma_t(1-\mathrm{terminated}_t)V_{old}(X_{t+1})],
\qquad \mathcal L_Q=\mathbb E\operatorname{Huber}(Q(X_t,a_t)-y_t^Q).
$$

保留one-step目标和lambda_Q=0.1；不另加lambda-return搜索。V与Q部分重叠，因为$y_t^Q-V_{old}(X_t)=\delta_t$，但V是状态基线，Q是实际候选回归，两者职责不同。当前Q还读取候选集合均值，故监督索引只选a_t，不等于梯度只到a_t的特征。[V3 §16]

Claude称one-step“偏差小”不普遍正确：它通常减少长序列采样波动，却更依赖下一状态V估计；多步/λ-return属于偏差与方差的取舍，不存在本任务未测就成立的全面优劣。[W16] 早期没有真实成功且V接近0时，Q没有凭空创造奖励的能力；随机非零value也不是可靠任务监督。

**A_Q需要做，但放Tier2，仅TC三seed。**其目的是检查保留这项辅助损失是否值得，不因理论故事好就省掉，也不要求所有family跑。共享梯度余弦/范数只作为异常时的稀疏诊断，不承诺“一行代码就准确归因所有梯度冲突”。


---

# 10. Actor Weighting最终规则

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


对gamma=0.99/s，60/120/200/300秒的prefix权重约0.5472/0.2994/0.1340/0.0490，Claude数值基本正确。但“后期几乎不参与更新”的程度取决于batch中早晚样本比例、归一化、梯度尺度及辅助loss，不是单凭0.049即可判定。关闭它是本轮有限工程选择，不宣称理论必胜，也不新增weighted/unweighted必跑消融。


---

# 11. Duration Discount最终规则

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


| 独立任务的H或典型reference时长 | gamma每秒 | H时刻保留 | e-folding horizon H/ln2 |
|---:|---:|---:|---:|
| 30s | 0.97715997 | 0.5 | 43.28s |
| 60s | 0.98851402 | 0.5 | 86.56s |
| 120s | 0.99424042 | 0.5 | 173.12s |
| 180s | 0.99615659 | 0.5 | 259.69s |
| 300s | 0.99769218 | 0.5 | 432.81s |

多family比较不是照此表各自偷偷换gamma，而是先取suite共同H再换算。两个策略同样都成功但相差ΔT秒，回报比为$2^{-\Delta T/H}$；这个时间偏好解释了恢复绕路的真实代价，但仍不提供额外中间奖励。


---

# 12. Task Learnability Gate最终规则

## 目的和边界

不再采用“random success必须≥5%”硬门槛。随机策略成功是当前初始化、deadline、动作分支和执行器下的**奖励暴露代理**，不是PPO可学习性的必要或充分条件。检查在合法仿真/受限技能接口内进行；真实机器人若未获安全授权，BLOCKED，不以随机探索之名绕过安全。

## 运行合同

每个启用训练family，从预留dev/preflight池均匀有放回采样初始化，使用独立seed流，合法candidate上uniform采样。先100个完整episode；若有效成功≤2，扩至200；若在100时已≥3且覆盖至少2个初始case，不再扩。保留计划数、有效数、基础设施异常数与实际case分布，不能把失败删掉。

若200个仍≤2成功，最多另跑50个**contract-aware random**诊断：在合法且名义patch非空候选上uniform采样；该集合为空时回到全部合法候选。它只是检查大量空patch/无效重复是否淹没探索，不声称其最优或总比uniform强，特别观察动作的空patch可能有真正价值。不用VLM评分、最优路径、Q或隐藏正确动作；不把诊断数据送入PPO/BC。

## 判断

- **PASS（工程暴露充分）：**默认至少3次真实成功且至少2case，存在合法恢复、至少两个真正有后果差异的选择点，关键日志完整；3不是理论阈值。
- **PASS_WITH_NOTES（低暴露）：**1–2次成功或集中单case，但已有独立成功reference和合法任务路径。允许先进行单seed smoke，不启动多seed矩阵；不称“不可学”。
- **NEEDS_TASK_REVIEW：**uniform200和可选contract-aware50仍无成功，且重复失败原因明确；用stage status=NEEDS_RERUN加该reason。暂停扩主训练，完成一轮有限任务/接口核查。
- **BLOCKED：**安全/观测/奖励/合同错误、没有合法可执行成功路径或现实资源未绑定。只阻塞执行，不阻塞论文写作。

若独立同分布Bernoulli近似适用，0/200的单侧95%成功率上界约为1.49%，不是0；若p=.01，100次全无成功的概率约36.6%，200次仍约13.4%。因此固定5%会错误排除部分可学习任务。实际相关初始化/非独立执行时不能把上述区间当严格统计保证。

## 必填观测

success有效数/分母与初始case覆盖；每episode决策点数、≥2合法候选比例、平均/分位branching factor、强制动作比例、空patch比例、重复loop/失败类型、成功/失败耗时；raw prior非空率与关系数；改变Fact到目标路径覆盖；实际非空prior+changed-patch中DP RMS与residual，后两者来自单独未训练forward而非策略成绩。

候选数≥2仅表示选择机会，不能证明长期后果不同。有效结构决策由任务定义、已注册可执行路径和受控开发执行证据确认，不由VLM或Full价值估计充当真值。

## 有限调整顺序

检查reward/flags/time→合同/mask/控制器/观测绑定→恢复技能与无意义分支→deadline是否留了真实恢复机会→减少一项无关决策或使用较短的训练模板。benchmark固定deadline不擅改；自定义任务默认deadline=2*T_ref，仅允许在主训练前一次改至3*T_ref并记录版本、重跑0D。安全阈值与单技能超时不放宽。仍困难时如实标低暴露/暂不启用该family，不直接加shaping、专家暖启动或复杂课程。

## 产物及停止

输出`stage_0d_summary.md`、`random_episode_log.jsonl`、`reward_exposure.csv`、`decision_structure.csv`、`task_revision_log.jsonl`、status JSON。日志中preflight/source标签与RL训练分离。完成100/200及条件性50的明确预算后停止，不自动进入下一Stage。


---

# 13. Task Family最终选择原则

默认核心family仅**T_B：Multi-Object Choice**和**T_C：Intermediate Relocation**。D0作多步smoke；T_A保留为合同重复/共享前置参考而非强制prior增量；T_D在确有组合评价需要时绑定；T_E成本较高，后置。模板不是已存在资产，平台/控制器/观测/goal/deadline/split仍MUST_BIND。

主任务应在合法参考轨迹中至少出现两个有≥2合法候选的决策点，并且有实际长期后果差异。仅candidate数量大或mask非空不证明有真正决策。合同保留所有真实必要启动/安全条件，不能故意删掉前置制造VLM优势；额外软关系限于当前抽象未完整建模的场景关联、代价与后续联系，而非隐藏的硬安全许可。

优先采用**错误通常可恢复，但消耗真实时间**的任务：增加探索时仍完成的机会，又由同一终端成功的时间折扣区分绕路与高效执行。它仍是稀疏奖励，不会自动给每次返工产生即时负奖励。少量错误即终止case可以保留为边界，不要求全部可恢复，不放宽安全阈值。

记录从实际改变的Fact到目标/候选的message-path长度；4层是否覆盖由具体图核对。PICK→Held→PLACE→Inside含3条边，从变化起点Held算才是2跳。局部覆盖不等于完整长期规划；不提前加candidate-local row或Graph Transformer。

包含prior有用、当前无用及自然误导case，但不按Full测试分数挑选cache或刻意给训练造错边。原始输出可能为空/错误，这些case照实保留。任务分层与test-independent设计先登记；相同任务下所有方法获得同源可观测线索。


---

# 14. effect_fact_ref最终裁决

**保持Action→Action的SOFT_SUPPORTS，以及Action→Goal Proposition的SOFT_RELEVANT_TO_GOAL；不改v2关系schema或图。** `effect_fact_ref`定位为 **consequence-grounding admission metadata**：要求引用source已注册效果及已有ADD/DEL邻接，提供可审计入口；不作网络额外输入、效果专属门控或逐效果归因。

Claude正确指出当前消息不是只传指定p。但其两个支持理由需要否决。

**候选等于source也不保证没问题。**A有ADD p和ADD q，p已TRUE而q未成立，此次名义patch可能只改变q。A→B仍可传递q引发的变化，即使relation引用p。因此“source的所有效果同时变化，所以无区别”不严格。[C §9.2]

**改成p→B也不保证硬门控。**p自身truth不变，隐藏表示仍可能经q→A→p等消息改变；直接连p可以缩短某些路径、改变归纳偏置，却不是p已完成时强制关闭。DEL涉及符号和来源归属，需设计；但不一定非得增加某个固定数量的relation types，也不必然变成硬PDDL precondition。

Claude报告的20seed随机R-GCN范数比只是一份未附可运行源码/图/初始特征的报告，本轮不声称复现。随机权重下的DP/DK比值还受分母大小、读出、初始化和seed影响，不能推断训练后用途，更不能证明p→B有稳定语义优势。

保留现图的理由是：当前问题是**skill-level representation response**，不要求独占效果中介识别；改图涉及语义、输入与缓存协议，而未发现必须修复的逻辑矛盾。论文加一句：

> The effect reference validates consequence grounding; neural messages are routed at the skill level and are not gated by that individual effect.

现prompt中的“may be inactive”是允许关系可能不适用，不是计算硬门控保证。本轮只澄清文档，保留prompt/schema字节与缓存key协议；不为修辞启动新缓存。以后真改prompt仍须新hash。**不新增effect-anchored实验。**


---

# 15. 80/20 prior training最终裁决

**保留80%Original、20%整份No-prior；比例不搜索；不做Full-clean必跑。**它属于共享输入缺省正则，不是论文创新。

No-prior条件应命名为**missing-prior / prior-dependence evaluation**，而不是“模型从未见过的完全OOD条件”。但训练见过空输入，不保证闭环后所有状态都处于训练分布。[V3 §17.3]

所有prior消费者采用相同协议降低比较不公平，但不说明dropout无影响或与结构设计没有交互。因此论文结果限定在共同80/20协议下，不单独主张20%比例优越或缺省正则的因果增益；只有未来要主张它的独立作用才需要clean对照。

Original不等于正确prior；空cache不补边；PPO重算不重抽；训练没有人工错边/删边。测试Deletion/Target-swap保持已冻结算法，Irrelevant保留适用性条件，不为了凑表新造不相关组件。


---

# 16. R-GCN / Readout是否需要修改

**现在不改：4层、128维、4bases、mean、逐节点LayerNorm、dropout0，以及goal rows＋global row全部保留。**

mean pooling确实可能使局部变化随节点数稀释；同一变化经过非线性global读出后也不保证严格按1/N缩小。关键变化到不了goal的L跳范围时，goal行可能不响应，但整个DP仍可通过global读出非零。[V3 §13–14]

这值得在0B验证已设计toy的传播，在4A记录按candidate分层的DK/DP、patch大小、路径长度、图规模和Actor响应；**不需要在得到瓶颈证据之前加入candidate-local row**。

若后来决定把action-node difference追加到E或Actor输入，那就是表示/架构变化，尽管代码少，也应明确新feature profile或v2.2并重新做匹配对照，不称为无语义影响的小参数调整。四层适合的只是已核验的局部路径，不是“足以理解所有多目标长时程任务”的保证。


---

# 17. Updated Novelty Map

核验等级：**F**=已获取全文并核对相关方法段；**A**=仅独立确认摘要/官方主页；**U**=Claude给出但本轮未成功独立确认。F不表示所有定理和实验已复现。

| 组件/对话对象 | 文献与核验 | 已有内容 | CP-DISR保留的区别与判断 |
|---|---|---|---|
| Candidate afterstate | Ståhlberg–Bonet–Geffner KR2023 [W03] F | 状态转移分类、GNN与策略梯度 | 候选后继不是独立新颖点；作为nominal轴继承 |
| before/after/去公共成分 | CQM 2608.22092 [W04] F | 候选行动间中心化、同步反事实rollout学习行动效果 | 本项目对比prior×nominal表示，不用真实环境多分支标签；不继承CQM充分性定理 |
| fixed/imperfect LLM action prior | Yan ICLR2025 [W05] F | 状态条件化动作先验、KL/采样等RL利用 | 不可称对方是状态不变static；区别是动作分布入口与结构交互入口 |
| uncertainty guidance | 2411.14457 [W06] F | LLM定制、MC dropout与动态熵指导 | 本项目不学习LLM不确定性或trust |
| imperfect teacher | LLM4Teach IJCAI2024 [W07] F | 软动作teacher/蒸馏并逐渐减弱依赖 | 不完美教师＋RL早已有，不是本项目独占问题 |
| KG输入机制 | 2607.19616 [W08] F | feature/mask/shaping与KG质量比较 | 支持机制需要受控测试；其PBRS不变性不是本项目有界残差保证 |
| approximate symbolic knowledge | ASGRL ICML2022 [W09] F | 近似符号模型、landmarks、技能多样性 | 固定不完美符号知识＋RL是已有设定；本项目直接学习候选表示 |
| 在线子目标图 | SGA-ACR [W10] F | 图规划、critic/refinement | 修改图与使用固定图不同，不因此否定其state conditioning |
| LLM/PDDL模型 | Guan NeurIPS2023 / InterPreT / UniDomain [W11–13] F | 模型/谓词构造、反馈修订、外部规划 | 本图soft边不赋真值，不成为硬前置；partial observation也不能单独当新意 |
| 固定VLM表示＋RL（独立补充） | PR2L [W14] F | 从VLM读取任务相关表示，由RL学动作用途 | “不让VLM直接选动作”本身也有先例；本项目特定2×2结构约束是区别 |
| bounded/residual路径（独立补充） | Residual RL [W18] F | 固定控制与学习残差相加 | 残差不新；本项目是logit有界且输入为交互，不是低层控制残差 |
| Auxiliary Q | UNREAL [W17] F；PPG [W20] A | 辅助学习/共享表示 | 当前候选Q保留为训练机制；不当独立原创算法 |
| 对象筛选 | PLOI [W22] A | 学习对象重要性辅助规划 | 不把它误列为直接successor-state策略方法 |
| 规划文献背景 | LLM+P [W23] A；Graph Learning综述[W24] A | 规划接口/综述导航 | 只用于相应已核验概述 |
| Claude的连续控制afterstate条目 | OpenReview XO944P8prc [W25] U | 当前访问被验证页阻止 | 不填作者、年份或方法细节，不作已确认依据 |

**核心novelty结论：**在本轮核对的直接相关全文中，未找到“固定软关系×合同名义干预的共享四视图交互，作为唯一直接prior特征通路，再零锚有界调制策略”的同一定义。这支持将其作为待实证的核心设计，但不证明全领域不存在，也不等于某一减法公式本身新颖。

“C2最核心”接受；“C2已经被证明强创新、碰撞风险一定低、其他内容无关”不接受。论文应对afterstate、状态条件化LLM先验、VLM表示RL和辅助价值学习逐项准确对话，不强行凑35–50篇。全文/摘要/未确认条目的URL、版本和核验说明见web_evidence_ledger.json。


---

# 18. Properties Box最终公式

共同条件：固定参数、相同真实观测与历史前缀、相同候选与执行mask、相同goal/node索引、相同确定性编码；只改变明确比较的输入。softmax只在至少一个合法候选上定义。

**P1 Exact null.** 若$R=\varnothing$，复用合同编码：$D_i^P=u_i^P=\Delta_i=0$对全部候选成立，因而$\pi_\theta=\pi_K^\theta$。$\pi_K^\theta$是**同参数**关闭先验修正的策略，不是独立训练B2。

**P2 Empty nominal patch.** 某候选$i$的全部编码输入未变时，$D_i^K=D_i^H=D_i^P=\Delta_i=0$。这只使该候选无直接残差；其他候选的非零残差仍可经softmax分母改变它的概率。只有全部候选满足null时才得到整项policy equality。effect_fact_ref对应p不变不等于整个patch为空。

**P3 Additive separability.** 若对当前与名义输入有$E(F,R)=A(F)+B(R)$，则$D_i^P=0$。不要求线性E，也不推得静态edge一律无影响。

**P4 Bounded direct deviation.** $\Delta_i=\beta\tanh(w_P^\top u_i^P)$，$\beta\ge0$，合法动作有

$$
\frac{\pi_\theta(a_i\mid X)}{\pi_K^\theta(a_i\mid X)}
=\frac{e^{\Delta_i}}{\sum_j\pi_K^\theta(a_j\mid X)e^{\Delta_j}}
\in[e^{-2\beta},e^{2\beta}].
$$

因为分子、加权分母都在$[e^{-\beta},e^{\beta}]$。默认$\beta=0.5$时界为$[e^{-1},e]\approx[0.367879,2.718282]$。这是对合同reference的单步界，不是任意两个先验之间同样的界（后者只能直接给出$e^{\pm4\beta}$粗界），也不保证轨迹回报或argmax不变。mask为0的动作不计算0/0比值。

**P5 Candidate permutation equivariance.** 在逐候选共享网络、集合读出置换不变、候选和mask同步置换、无候选位置编码时，$\pi(P\mathcal A,Pm)=P\pi(\mathcal A,m)$。若先执行argmax，并列需按canonical ID解决；分布等变不自动使按数组首项打破并列的动作选择等变。对象重命名是另一项更强检查，不能由候选重排直接推出。

P1/P4/P5也适用于按本版定义的A_CAT；A_CAT不要求P2/P3成立，因为它允许静态表示成分。A_B保留null而移除P4。全部是代数/架构条件，不作为全局安全或有效性定理。


---

# 19. Revised Experimental Route

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


---

# 20. 第一批实验

第一批先做0A–0D实际运行准备，然后只运行：

| 任务 | 配置 | seed | Ncap | 新run |
|---|---|---:|---:|---:|
| D0 smoke | B2、Full | 0 | 各16,384 | 2 |
| TB | B0、B1-K、B2、Full | 0 | 各65,536 | 4 |
| TC | B0、B1-K、B2、Full | 0 | 各65,536 | 4 |
| TC核心消融 | A_CAT；Full复用TC | 0 | 65,536 | 1 |
| **合计** | | | **622,592上限** | **11** |

先写探索Results，不用等显著差异。Single-seed只能给基本现象和排查依据，不能假装稳定平均效果。A_CAT包含在第一轮，使核心假设早接受检验，但不抢先跑A_Q/A_B/Full-clean/effect anchored。

任务/接口有错误则修复并记录；只是Full不领先不算实现错误。没有暴露真实成功时，暂停多seed扩展、做有限诊断，而不是把坏结果直接抹去或自动加reward。


---

# 21. 第二批实验

核心输入、配置和预算不变时，TB/TC四方法补seed1和2，共16个；TC A_CAT补seed1和2，共2个。加第一批后累计29个，形成两任务、三seed主阶梯及核心CAT对照。**无需另建36个Presentation训练。**

保留Auxiliary Candidate Q时，TC的A_Q补0/1/2共3个，累计32个。若主张bounded带来经验收益，TC A_B再3个，累计35个；仅写代数界则不必。

同checkpoint机制与先验测试、已有source支持的一种组合holdout均不重训。没有合适source checkpoint、未见组合只是重命名或必须新增技能时，5A标NOT_APPLICABLE/未执行，不自动造新任务矩阵。

每阶段执行到报告停止。不同环境profile、预算H或超参已经变更的旧run不能凑在新三seed中；重新训练需求必须列新run清单和原因，不能藏进“复用”。


---

# 22. 哪些实验删除或降级

**从默认队列删除：**旧A_DD必跑、旧B1-H必跑、旧B3、原DeepSets额外对照、Full-clean、effect-anchored图、candidate-local readout、全部任务一次性7配置、无条件36次新Presentation运行。

**降为条件性：**A_B仅对应经验bounded claim；A_DD只在CAT结果需要分辨“静态信息”与“减去DK”时考虑；B1-H只在评审确实要求当前prior-only基线时考虑；梯度余弦/更大图/更长链仅诊断。

**保留但不增加训练：**同checkpoint的DP0/remove（同状态应等价）、shuffle、删除、target-swap；已完成的Original评价精确条件一致可引用，不重复收费。Irrelevant无独立证据或满8边预算则NA，不额外造资产。

没有证据证明某个理论性质带来性能收益时，选择收缩claim，而不是为了“理论box每项必须有一组实验”无限加run。


---

# 23. Paper Method最终结构

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


---

# 24. Figure 1 / Figure 2最终设计规范

## Figure 1 — 2×2核心总览

采用双栏横向版式：左侧20%为任务与合同/soft relation来源；中间50%为**行K/H、列current/nominal-after**的四个对齐图；右侧30%为DK合同评分与DP有界残差、mask、真实skill execution。底部用一条细反馈线从实际执行记录指向可训练网络，标“training on real executions”。VLM/cache标frozen，不必画一串红叉或工程模块。

同节点位置在四格一致；真实事实变化仅在名义列高亮；软关系仅在H行出现，两个H视图内容相同。一个shared E标记覆盖四视图；箭头说明先构造图再编码。图中可以使用蓝色代表合同、橙色代表soft，黑白线型同时区分；不能依赖颜色才可读。

场景图必须来自实际可授权输入，否则明确标schematic，不把示意图称为真实机器人实验。MOVE、effect与对象必须已有注册；不能为了好看新增OnLid/ClearPath事实。每候选重复2×2的批处理用“for each allowed candidate”标注，不画全部候选导致拥挤。

## Figure 2 — 交互代数与状态依赖

四格用同一E的Z00/Z10/Z01/Z11标注，横向箭头DK/DH，纵向S_R(F)/S_R(Ftilde)，中心完整等式DP=DH−DK=S_R(Ftilde)−S_R(F)。S仅临时记号，不画第五个网络。

右侧示意采用两个**确实成立**的对照：R为空时DP=0；整个候选编码patch为空时该候选DP/Delta=0。再放一个“不同candidate/state可产生不同响应”的定性可表达性面板，不画必然增大/减小箭头。

**删除Claude原图要求的两种暗示：**PICK(blue)看似无关所以DP必小；p已TRUE所以MOVE的DP必变小。这些不由当前A→A消息图保证。需要展示它们时，只能用真实checkpoint测得值和来源作为机制结果图，不把随机RGCN数字或未测绿色箭头混成方法性质。

最终出图应保留可编辑向量源；本轮交付的是上述draw specification，不伪造真实场景图、实验效果或latent维度语义。


---

# 25. Final Change List

| Current Item | Claude Suggestion | Your Verdict | Change Now? | Reason |
|---|---|---|---|---|
| B0 | 换两层Set Transformer | 接受，定义无显式图而非无关系能力 | 是 | 用替换而非追加解决弱基线质疑 |
| B1 | 改current GK-only | 接受并精确定义FK(c,0) | 是 | 同B2context，避免同时改prior/干预 |
| B2 | 合同干预 | 保留 | 否 | 先验增量与intervention参考 |
| Full | 保持主体 | 保留结构，训练profile升级2.1.1 | 训练默认改 | 无DP/graph/schema重构 |
| A_DD | 降optional | 接受 | 是 | CAT直接检验完整信息融合 |
| A_CAT | 新增必跑 | 接受，合同重复zero anchor | 是 | 替代A_DD，不加配置总数 |
| A_Q | Q降训练机制配消融 | 接受，Tier2 TC三seed | 分阶段 | 检验辅助loss是否值得 |
| A_B | 继续必跑之一 | 修改为conditional Tier3 | 是 | 代数界本身不需性能归因 |
| Random success | >=5%硬准入 | 否决硬阈值，改奖励暴露审计 | 是 | 无理论必要性；1–2%仍可能可学 |
| Decision density | 多个真实决策点 | 接受，区分机会与后果 | 是 | candidate数不等于有效选择 |
| Gamma | 保留率>=0.3 | 改suite半衰期H规则 | 是 | 可解释且所有方法同时间偏好 |
| Actor weighting | prefix=false | 接受工程选择，否决精确J等价 | 是 | 统一empirical surrogate并记录版本 |
| effect_fact_ref | admission constraint | 接受metadata定位，否决硬门控/源动作无差异 | 文字改 | 不改A→A/schema/prompt字节 |
| 80/20 | 保持、无clean | 接受，但不能称无任何交互效应 | 否 | 限定比较协议，不宣称dropout独立收益 |
| R-GCN/readout | 4层暂不改 | 接受，不保证任意长期推理 | 否 | 先诊断真实瓶颈 |
| Task families | TB/TC优先 | 接受，以实际任务绑定为准 | 是 | 两任务足够首轮，不硬补第三 |
| Robustness | 五条件 | 保留四常规＋irrelevant按适用性 | 评价名改 | No-prior不是全新缺省条件 |
| Seed count | 七配置各3seed | 分tier，先11run单seed，再补齐 | 是 | 早得现象，主对照最终3seed |
| Contribution | 问题/方法/协议三项 | 改两个方法方面＋数据后的实证发现 | 是 | 不强凑三项算法或自封协议创新 |
| Properties | null/bound/permutation | 接受并补充假设及softmax分母边界 | 是 | 不比较独立B2，不给CAT错误P2要求 |
| Online requery | 会破坏on-policy | 否决一般性因果推论，仍不做 | 文字改 | 记录条件输入即可定义on-policy |
| Novelty | C2强且未撞车 | 限于核对范围内未见相同定义 | 是 | 查全文不等于穷尽首创证明 |

## P0

**No method-level blocker remains.** 本轮没有发现必须改核心图/DP/Actor才能形成方法闭环的错误。实际系统仍须通过事实隔离、mask、targets、时间、共享梯度等生产检查。Claude的过强句子不进入新版本，不将其当作现有方法的必修结构缺陷。

## P1 — 本轮已经写入规范/计划

中心改为candidate-conditioned representation interaction；两项方法方面＋后续实证贡献；B1-K重定义；B0两层SAB；A_CAT替代A_DD；Actor按有效transition均值；suite half-life；0D奖励暴露审计；TB/TC优先；11→29→32分批；同参数Properties与准确图示；主文数值使用完整预定3seed，Private View另标；metadata澄清而非effect-gating。

## P2 — 只有证据出现后决定

A_B、A_DD/B1-H补跑、更多seed、candidate-local readout、更长传播、梯度干扰和具体任务难度修订。新增readout/图语义属于新profile或v2.2，不因几行代码就称无方法变化。

## P3 — Future Work

M1+、逐关系经验与utility、online repair/regeneration、VLM adaptation、新world model/可靠性模型、大规模Graph Transformer与跨机器人扩展。均不进入当前实验必跑队列。

## 两个版本号的最终裁决

**Method：CP-DISR / M1 v2.1.1。Experimental Plan：v1.1。研究文档：v3.1。**

2.1.1记录surrogate权重与折扣配置真实改变，核心算子/架构保持2.1。旧2.1结果不能无标识混用。只有改DP、图/读出输入、Actor结构或关系语义才考虑2.2；A_CAT是比较方法，不把它的输入权限偷渡进Full。

## 十个最终答案

| 问题 | 裁决 |
|---|---|
| C2中心、C1/C3降级？ | **接受中心排序，修改“唯一公式创新”叙事**；名义干预是继承构件，Q是训练机制 |
| A_CAT必跑？ | **是**，替代A_DD，先1seed再补齐 |
| B1重定义？ | **是**，B1-K，current合同context＋零DKslot，无prior |
| B0升级Set Transformer？ | **是**，两层标准SAB，不另跑旧弱版本 |
| Actor prefix weighting关闭？ | **是**，明确经验PPO surrogate，非精确J梯度 |
| Gamma按时长标定？ | **是**，共同suite half-life，不保留全局0.99/s |
| Stage0 random check加入？ | **是**，但**否决5%硬门槛** |
| effect_fact_ref仍Action→Action？ | **是**，Goal关系保持Action→goal；引用只作admission metadata |
| 实验路线修改？ | **是**，替换归因对照、分批而不是叠加 |
| 现在能正式写论文？ | **能，现在开始**；实验验证经验claim，不决定有没有资格写Method |

**最终核心：保持同一个结构交互方法，用更准确的论证、更直接的A_CAT对照和更合理的稀疏回报训练协议替代多余实验；先11个训练取得基本现象，再补齐需要的证据，不再无限审查。**


---

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
