---
title: "CP-DISR 最终实验执行计划"
subtitle: "Final Experimental Plan v3 · FROZEN EXECUTION PLAN"
date: "2026-09-28"
lang: zh-CN
master_id: CP-DISR-FINAL-EXEC-3.0
document_version: "3.0"
research_content: CP_DISR_Final_Research_Content_v5.md
default_method_version: "2.1.1"
conditional_method_versions: ["2.2-B", "2.2-C"]
status: PLAN_ONLY
freeze_status: FROZEN_PLAN
authorized: false
current_evidence_version: "2026-09-28 / through cd95646ee0502419052210590b280750374a0401; R1/R2/P1 on separate refs"
---

# 文档控制、冻结与唯一授权体系

本文件与Research Content v5构成后续研究、实验和写作的唯一权威依据。Research Content v4、Experimental Plan v2、旧Master、局部卡及审计文本作为历史输入保留，不再并行授权未来工作。默认Method仍2.1.1；只有S4正式批准时启用2.2-B或2.2-C，第一周期最多一方向。

```yaml
master_id: CP-DISR-FINAL-EXEC-3.0
status: PLAN_ONLY
freeze_status: FROZEN_PLAN
authorized: false
default_method_version: "2.1.1"
base_new_rl_runs: 17
single_elastic_attempt_pool: 4
first_cycle_attempt_hard_cap: 21
```

目标保持CCF-A会议／SCI一区级别的方法研究强度：充分创新、完整核心证据与最少冗余实验，不代表录用保证。保留C1/C2/C3证据链；负结果按诊断性→实现→公平性→预算→一次有限修订→S4方法升级→最后调整claim处理。论文现在继续写，未完成部分标RESULT_PENDING。

**基础矩阵固定为17个新增RL配置：T_B 3＋T_P 14。**所有条件性训练、技术重跑、匹配调参、分歧seed、可选B1-H和方法升级净增只使用一个4-attempt弹性池，第一周期总量17＋4≤21。先替换未启动配置，后新增；必要control超预算时人工决策，不通过删control或自动突破硬上限处理。

本文件被冻结但不构成实验许可。每张卡默认PLANNED、authorized=false；用户必须指定S0–S8或子卡及资源才可执行。表中是未来上限，不是已发生计数。本轮未SSH、未运行RL／仿真／模型前向／生产单元测试、未调用实验provider、未修改服务器或启动monitor。

动态执行唯一账号仍是`xushijie2@gpu03`，具体路径、import、存储与凭据规则集中在Runbook附录。禁止fallback旧账号。新结果只改变卡状态和论文结果，不自动产生新Master Plan。

阅读顺序：§1已有事实→§2–4对照和S0→§5–6存在性／T_P冻结→§7–11训练与单一决策门→§12–20测试／分析→§24成本→§26写作同步。

# 1. Current Evidence Snapshot

| 来源 | 可信观察 | 复用方式 | 不允许解释 |
|---|---|---|---|
| T_B R1 B1-K/B2 | N13623/14464，13+1/14+1更；共同采样N8192开发0/10 vs10/10 | 主线seed0；保留原N/T与模型身份 | 单seed说明所有当前图方法不足 |
| T_B R2 B0 | N14705，14完整+1残片，dev点均0/10；selected dev20 0/20 | 系统参照，保留实际旧账号偏差 | 伪写xushijie2运行或自动废弃重训 |
| selected common dev20 | B0/B1-K选N0，B2选final14464 | 单独展示选择列和原规则 | “三个终点训练模型”或独立test |
| P1 | 同B2 Original10/10，DK-Zero0/10，30动作翻转，TV约0.10997 | C1机制依赖证据 | DK-Zero等于独立B1-K或跨seed必然性 |
| 历史D0 | Method2.1.1 B2/Full各8更N8192；test各29/30 | 历史smoke／有限负观察 | 当前profile独立重复seed |
| 新D0 R3 | Full u2耐久N2048；n2048评价10/10；第三更未完成 | 保留部分结果与stall | n2048必等于u2模型 |
| R3-Recovery | B2 N1935，1完整+1残片；Full Original/Absent dev均10/10 | 非精确预算匹配的描述；同Full性能不变可用 | 严格matched2048；机制机会0的结论 |
| T_C穿透版本 | 启用84case重叠；E16两run0更新 | 输入／工程Archive | 有效prior主结果 |
| T_C nonoverlap | 10个direct/relocation都成功，搬运更慢 | task-design boundary | Full或prior一般无用 |

## 1.1 三个已知口径问题，不重新实验来“修历史”

1. Recovery脚本把N_CAP改2048后bind_profile，实际Tcap约9318.4；原授权要求保留E8安全Tcap。论文记录实际，不说同预算。
2. Recovery prior评价中的dp_opp/dp_nz/delta_nz仅初始化未累加；派生视图改为NOT_MEASURED，原文件保留。selected_candidates为None的记录不补猜动作。
3. Full n_002048的eval行记update1，评价调用发生在第二更之前；u2是另一耐久代。按真实checkpoint hash区分采样位置与优化位置。

不做重复完整考古；直接采用既有Master Ledger的来源清单和上述结论。后续仅在新结果复用所需文件缺失时补核验。

## 1.2 永久不自动恢复的历史义务

EEP v1.0的29/71、Plan1.1的11/29/32/35、旧T_A/T_C 24-job、旧T_B事故续跑、无益T_C重定位搜索、D0到E8长尾、历史P2兼容环境，都不因曾写入计划而进入本表剩余量。未来工作只领S0–S8。

# 2. Experiment Questions与证据结构

| RQ | 问题 | 核心证据／比较 | 对应贡献 |
|---|---|---|---|
| RQ1 | 直接读取候选效果描述与把效果施加到当前图后重新编码，有何不同？ | T_B B1-K/B1-K+E/B2、已有P1、B_PLAN | C1 |
| RQ2 | 合法非冗余prior有无整体增量；该增量是否需要后果条件化而非静态prior上下文？ | Full/B2；Full/A_STAT，同一T_P、匹配seed | C2 |
| RQ3 | 纯交互与灵活四视图融合有何表达力、效用与敏感度取舍？ | Full/A_CAT；A_STAT用于辨别静态来源；读出分解 | C2 |
| RQ4 | 无关系标签的结果训练，是否产生更符合独立动作效用的prior依赖？Q的作用是什么？ | Full/A_STAT/A_CAT/A_Q共享Q_ref，E2/E4/E8轨迹与裁剪审计 | C3 |
| RQ5 | 缺失、错误、图结构远端与provider变化下，先验影响和代价如何？ | NoPrior、Target-swap、GRAPH_REMOTE_PRIOR_INJECTION、条件Cross-VLM | C2/C3 |
| RQ6 | 相对合同搜索、朴素使用prior和逐步LLM推理，成功、成本与摊销效率如何？ | B_PLAN、B_PLAN+R、分析R*、LLM-Planner | 整体 |
| RQ7 | 结论如何随真实任务族难度及有合法来源的未见条件改变？ | Full/B2难度轴；条件holdout | 范围 |

三类C2问题必须分开：Full/B2是prior接口整体增量；Full/A_STAT是同DK和参数形状的后果条件化vs静态prior输入；Full/A_CAT是限制性交互vs灵活融合。B1-H不承担这三个问题的核心归因。

主结果只来自真实执行，oracle容量、结构性质与推理时输入干预单列。没有新结果时用RESULT_PENDING，不填预期赢家；未运行、不适用、未测和观测零严格区分。

# 3. Method Matrix与公平性合同

正式定义以Research Content v5 §19为准；同名方法不可存在另一套未登记的实现。

| 方法 | input／prior入口 | 名义干预 | 学习与地位 | 公平性控制 |
|---|---|---|---|---|
| B0 | 原始PRE/ADD/DEL/UNKNOWN、facts/goals、公共观测和候选；无R | 无 | 复用历史s0系统参照 | 非显式图，不剥夺原始效果关联 |
| B1-K | K图＋共同c；phi_K(c,0)；无R | 无 | T_B补s1 | 当前合同图含效果，不冒称它完全没看ADD/DEL |
| B1-K+E | B1-K全部输入＋PRE/ADD/DEL/UNKNOWN/guards、grounded ownership/binding | 无 | **T_B s0 MUST**；s1按条件占elastic | 只读效果描述，不应用效果或closure，不生成afterstate |
| B2 | K、DK；effective R为空 | 有 | T_B s1；T_P s0/s1/s2 | DP/uP/Δ精确零 |
| Full | K、DK；唯一直接prior输入DP | 有 | T_P s0/s1/s2 | Method2.1.1，episode80/20 |
| A_STAT | 与Full相同K/DK/c/phi_P/V/Q/Actor；prior输入仅换S_R | 同Full | **T_P s0/s1/s2 MUST** | 同parameter shape与active数量；没有empty-patch gate |
| A_CAT | 同源四视图C和合同重复C0 | 同Full | T_P s0/s1/s2 | 相同R和budget；增宽首投影透明，不假称单变量 |
| A_Q | Full，λ_Q=0 | 同Full | T_P s0/s1 | 保留Q-head分配；记录全局clip影响 |
| B1-H | phi_K(c,0)＋当前H/K静态S_R，80/20 | 无 | **OPTIONAL / elastic-pool diagnostic baseline** | matched-v1，不用旧phi_K(c,ZK)；不承担C2主claim |
| B_PLAN | 同源facts/goal/合同／公共时限；无R | 搜索中后继 | 0训练MUST | 共同执行与reward，足够规划预算 |
| B_PLAN+R | 同一搜索＋冻结R软规则 | 符号搜索 | 0训练MUST | 不改hard facts，不用truth过滤R |
| B_PLAN+R* | 同一搜索＋有限宇宙oracle关系 | 符号搜索 | ANALYSIS_ONLY | oracle-informed reference，不保证upper bound |
| LLM-Planner | 当前公共facts/goal/合法candidate/合同/R／公共摘要 | 模型推理 | 0训练MUST | input-matched不等于相同pretraining知识 |
| VLM-Planner | 上述＋当前RGB | 模型推理 | OPTIONAL | 更大观察权限，API成本另授权 |

## 3.1 B1-K+E s0 MUST与s1唯一触发规则

S0审计原始字段→编码器消费→候选可读性，但不以“源码里有ADD/DEL”免除+E s0。+E得到候选PRE/ADD/DEL/UNKNOWN及guards和grounded效果ownership，读出形式、宽度、归一化及插入位置在s0训练前冻结。它不执行nominal赋值或derived closure，不输入未来执行结果。参数差异据实报告。

**s1触发条件在本文件固定：**s0的任一预先固定、post-update dev10评价点出现至少1个经独立Evaluator确认的完整任务成功，即记NONTRIVIAL_LEARNING，并申请同profile s1以降低单seed解释风险；该新增从elastic4扣1，必须另有授权。不得事后根据它是否击败B2改规则。s0全零仍需检查训练真实完成与指标，不将一次失败自动称为+E普遍无效；也不自动为“找到非零”再换seed。

若S0发现公共输入真的遗漏，修复必须对受影响比较公平，旧结果另列；不能只给+E补bug后宣称原B1-K/B2严格匹配。公共修复造成的重训仍受单一elastic池约束。

## 3.2 A_STAT：唯一prior输入张量替换

\[
u_i^K=\phi_K(c_i,D_i^K),\qquad S_R=Z_{01}-Z_{00},
\]
\[
u_i^{P,S}=\phi_P(c_i,u_i^K,S_R)-\phi_P(c_i,u_i^K,0),
\qquad \Delta_i^S=B\tanh(w_P^\top u_i^{P,S}).
\]

Full仍使用\(D_i^P=S_R(\widetilde F_i)-S_R(F)\)。A_STAT保留相同DK、contract branch、c_i、phi_P、Actor/V/Q、残差bound、参数形状、optimizer和交互预算；它唯一改变的是phi_P的prior输入DP→S_R。各run独立训练，不共享已学权重。

S0核验active参数数量精确相同；不得增加额外静态head或把参数“匹配”成共享值。A_STAT可以省略不被其输出使用的Z11计算，视图数／延迟差如实报告。S_R不因候选而重新计算，但candidate context与DK仍不同，所以输出可有候选特异性。

它保留R-null与bound，不保证empty-patch null或additive cancellation。不要加DP==0 gate把A_STAT变成Full。其三seed与Full/B2/A_CAT同等级。

## 3.3 可选四格只作描述性诊断

B1-H保留Research v5的matched-v1定义，但不进入基础训练、S3核心pilot、S4必需输入或默认S7模型集合。T_P B1-K s0也不在基础矩阵；C-FACT独立预算桶取消。

若未来在elastic4内获得独立授权，B1-K/B1-H/B2/Full四格只能描述两种prior机制各自的包级增量：无干预行和有干预行采用不同接口，不是严格单变量因果析因。无对应T_P B1-K时不得用T_B数据填格，宁可不计算I_IP。

## 3.4 参数、优化与统计公平

导出allocated/active/实际有梯度参数、视图数、wall和memory；不以死参数填容量。Full/A_STAT参数形状匹配，Full/A_CAT的首投影和零锚性质差异透明报告，不因为CAT强就缩小它。

一次有依据的调参保持关键比较机会对等，但同样消耗elastic4；不能为每个用途另留预算。不同profile的seed不混均值。Full/A_Q结合gradient clipping audit解释，不能把所有变化都归给Q的表示训练。

# 4. Stage 0 — Scientific Code Audit（S0）

S0只关闭会使归因错误的实现缺口，不重新跑整套0A–0D。历史只读source结论可以复用；本文件没有实际执行S0。

| 审计项 | 既有线索／问题 | 输出与检查 | 执行后影响 |
|---|---|---|---|
| 1 B1-K效果权限 | K图有ADD/DEL，candidate向量无显式表 | 字段→张量→候选读出的追踪与绑定碰撞 | 解释历史范围；不取消+E MUST |
| 2 B1-K+E实现 | 效果描述vs施加 | PRE/ADD/DEL/UNKNOWN/guards/ownership齐全；无nominal/closure调用 | 冻结s0与非平凡学习触发规则 |
| 3 A_STAT实现 | 同DK，只DP→S_R | 同parameter shape/active count、同heads/目标/80-20；不加empty-patch gate | 核心C2公平性 |
| 4 encoder与global DAG | 4层RGCN、逐节点LN；mean后非线性global row | 实際层表、差分前后位置、graph/goal读取路径 | 不授予端到端strict locality |
| 5 hub／距离 | 需真实统计 | degree、diameter、components、patch/edit footprint距离 | 无合法远端bin时N/A |
| 6 同patch与R | 四视图R应固定 | source哈希、节点/行/mask一致、nominal不写真值 | 先修实际接口错误 |
| 7 Q | all-forward、executed-only label | 目标、detach、gradient reach、覆盖面 | Q_proxy不冒充真值 |
| 8 A_Q clipping | 总梯度改变会影响global clip | 实测pre-clip/post-clip norm、clip_trigger、actual_steps字段 | Full/A_Q Results必须解释 |
| 9 T_B外部规划 | 合同搜索是否足够 | B_PLAN dev最多10episode及名义搜索预算 | 如实定位学习相对搜索用途 |
| 10 relation truth | T_P仍未绑定 | finite universe、T/F/U、utility、opportunity各自依据 | E6与Q_ref，不进训练 |
| 11 A_CAT容量 | 4d首投影增宽、零锚不同 | allocated/active参数表与计算成本 | 不假称严格单变量 |
| 12 B1-H | 旧slot=ZK | 仅在未来可选实现时使用matched-v1 | 不成为S3启动硬前置 |
| 13 budget／checkpoint／计数 | 历史Recovery口径问题 | envelope不被pause改写、post-update eval、计数真实累加 | 未测counter不得写0结论 |

## 4.1 必须安排的四类Full性质单元检查

1. **Exact R-null**：R为空时DP/uP/Δ为零，同权重合同参考一致；不等于独立训练B2。
2. **Empty-patch null**：指定候选\(\widetilde F_i=F\)时该候选DP/uP/Δ为零；其他候选残差导致其softmax概率间接变化不算违反。
3. **Bound**：有限合法输入下\(|\Delta_i|\le B\)，不将它扩为轨迹回报或安全保证。
4. **Candidate permutation/reorder equivariance**：candidate ID/feature/mask/contract索引同步重排，按ID逆置换比较输出、Q、logits和canonical deterministic action；hidden和全部真实输入固定。允许声明的浮点归约容差；不将候选重排扩写成对象重命名／图同构等变。stable-ID实现限制必须写明。

A_STAT/A_CAT另核验共享R-null、bound与权限隔离；empty-patch非零可以合法，不能强行改其定义来通过Full专属测试。

## 4.2 Gradient Clipping Audit字段

在正式clip调用前读取pre_clip_global_grad_norm，调用后读取post_clip_global_grad_norm；记录clip_threshold、clip_triggered、optimizer_step_id及method/seed。clip-trigger rate的分母是实际optimizer steps，含合法尾batch，不固定假设每更64步。S0只核验测量路径，不能声称已有Full/A_Q差异。

默认global gradclip0.5不变，不新增“无clip”run。诊断读取不得改变grad、RNG或optimizer；不能把clip_grad_norm返回的同一pre值同时记录为post。

## 4.3 交付与子授权

输出`runs/final_master/S0/<stamp>/`中的audit、input/shape、layer/DAG、property_tests、clip_logging_contract、B_PLAN_dev、source_diff和未绑定项。未执行测试保持NOT_RUN，不用文件存在替代通过。

S0审计本身0RL。获单独S0-TB子授权后，三条基础T_B run（B1-K s1、B2 s1、+E s0）可在各自直接相关检查通过后运行；它们计入17，不依赖T_P，也不因本文件被冻结而自动启动。

# 5. Stage 1 — Prior Existence E1–E6（S1）

## 5.1 范围与分母

先读取最多24个已有合法开发scene/中间snapshot。最多两种候选结构，不启用穿透T_C动态环境。只在已有输入无法回答时，预先固定最多12个新开发scene，请求一次原provider；最多一次格式/传输重试，禁止语义重问。

先冻结关系schema、候选关系宇宙、E5的公共信息抽象和任务后果判据，再看模型成绩。不能为获得strict ranking筛选case；所有排除和不适用都有分母。

阶段预算：0RL/0optimizer；T_B B_PLAN10ep归S0；S1三planner×最多12dev=36ep；必要独立物理见证最多8ep。VLM最多12首次＋12合法重试。算子容量只离线前向、不更新参数。

## 5.2 E1–E6操作表

| 门 | 定义／测量 | 进入下一步的条件 | 失败／不确定时的后果 |
|---|---|---|---|
| E1 Relation existence | raw/accepted/nonempty、局部冗余、可标注precision/recall | 至少两个独立配置有自然合法且可裁定的非冗余关系 | 只有手写/oracle不算；查admission或来源，不伪装成功 |
| E2 Representation entry | legal candidate≥2、真实changed patch、R与编码footprint | 至少两个配置有进入候选表示的路径／输入变化 | 区分无patch、mask、拓扑、读出；不只看source count |
| E3 Candidate discriminability | C_D、C_Δ、相对logit改变、梯度到输入 | 多候选响应不是全部共同平移；random权重不需选对 | 当前零输出也可能权重偶然；用E2*与结构检查，不直接否定 |
| E4 Real task consequence | 同初始化合法候选真实执行＋共同continuation | 至少两个独立见证有可靠可行性、cost或rework差 | 开关没后果是non-diagnostic；不把增加搬运动作本身算收益 |
| E5 Contract legitimate insufficiency | K抽象相同/近同合法排序，真实后果不同；planner充分性检查 | 缺的是soft risk/cost/relevance，不是删掉的hard precondition | 搜索截断／错误cost模型不能证明合同原则不够 |
| E6 Useful but imperfect prior | R与R\*的结构质量＋B_PLAN/R/R\*表现，真/错/未知与效用分离 | 存在可验证有用信号，并明确自然错误/缺失或误导机会；分类通过而非固定不等式 | R近完美→C2可试、自然imperfect claim受限；R无信号→不训练；R有害但有信号→校准任务可试 |

以上“两个见证”等为本项目筛查最低操作标准，不是统计显著性或物理定理。T_P最终总体分布必须另行量化，不能把两个见证当所有scene都具有机会。

## 5.3 E5不能以缺少hard constraint造问题

B_PLAN须有足够深度解决注册名义任务；将名义规划失败归因于horizon不足时先处理planner。关系如果本来应是启动安全必要条件，写回可靠合同来源或修task，不能让Full通过语义猜测绕过它。

对环境soft风险的“truth”不等于skill永远不可执行。优先寻找同一合同层可执行集合下，真实时间、返工或目标保持有差别的状态。事后知道Full赢才称其为soft关系不允许。

## 5.4 E6不是要求三条曲线严格递增

理想窗口：B_PLAN < B_PLAN+R < B_PLAN+R\*，指标可为return或同成功下cost。但这不是必要门。

| 模式 | 分类 | 处理 |
|---|---|---|
| R改善，仍有缺漏/错误 | BENEFICIAL_IMPERFECT | 进入pilot |
| R有可靠真关系，但固定启发式被错边误导 | HARMFUL_BUT_INFORMATIVE | 可进入校准pilot，预先说明不保证Full收益 |
| R几乎完美 | NEAR_PERFECT | C2可试；错误prior部分用明确合成challenge，不冒称自然错误充分 |
| 所有planner一样，但物理分支有差异 | HEURISTIC_INCONCLUSIVE | 不直接判task无效；改一次冻结启发式或用见证判断 |
| B_PLAN与R\*都相等且无物理差异 | CONTRACT_SUFFICIENT | 不做该T_P长训练 |
| R无有效信号、R\*能区分 | PROVIDER/SCHEMA_LIMITATION | 一次有依据的来源/schema修订，升版本 |
| R\*也不能进入表示，但外部能区分 | REPRESENTATION_LIMITATION | 去S4方法路径，不强行label prior质量差 |

R\*仅在有限注册关系宇宙和指定任务语义下定义；B_PLAN+R\*受其heuristic、search及controller限制，一般称oracle-informed reference，不称理论upper bound。

## 5.5 E2* Oracle relation capacity probe

使用R\*替换R仅在独立分析进程中，比较reachable footprint、C_D、C_Δ、各goal/global行响应与candidate相对分数。无新RL、不向训练提供R\*。随机初始化或冻结权重对R\*没响应并不独自证明架构容量为零；还需输入依赖、mask、rank/梯度和不同可复用初始化的有限诊断。没有必要训练一个oracle Full来制造正结果。

## 5.6 S1决策与停止

存在性通过写ELIGIBLE，所有量化/局限进入eligibility_manifest。没有通过时写具体类别，不记RL失败。允许最多一次基于失败来源的修订，仍不能建立可诊断问题则暂停T_P训练授权，进入S4研究决策。不是自动删C2，也不是自动制造第三种task。

# 6. Stage 2 — T_P Construction与任务族冻结（S2）

T_P是prior-diagnostic task role，当前尚未绑定，不能假定任何位置/资产已完成。优先使用现有合法技能、几何、观测、控制器和合同语义。新数据版本不得复用旧图像key；旧T_C保留不覆盖。

## 6.1 任务族与生成规则

至少一条有机制含义的难度轴：合法候选分支数、准备/返工负担、自然关系覆盖或两种见证类型。能共享同一controller与图schema才合并为一族；仅改名称不算不同task。尽量让T_P包含不止一个有效见证类型，但不得为达到“至少两类”随意改环境。

T_B保留旧核心分布，难度评价可用合法初始前置满足程度或真实分支数。依赖深度、物体数若当前binding无法改变，不宣称已经实现，先选择现成可观测轴；分层分析不冒充受控新难度生成。

## 6.2 数据规模与隔离

起步train32/dev16/test32，总80配置；不是所有情况下的最优规模。S1见证属于discovery/dev，不转入test。按generator seed和场景身份划分，禁止同scene轻微复制跨split。训练数据不按VLM非空率过采样；如以“prior-eligible族”为总体，预先定义它的条件，主表说明条件分布，同时保留全量eligible生成分母。

test配置和生成规则在训练前冻结；训练前不向研究者暴露test模型输出、test relation truth或成功率。test cache在模型/方法冻结后由数据进程物化，失败/API缺失单列，不看test关系有利程度重选scene。

## 6.3 关系truth／utility只分析

schema容许的有限关系宇宙独立枚举并去除直接合同冗余；标注T/F/U与判断依据，至少两人/两次独立核对争议（可为研究者与独立规则，不强制LLM仲裁）。模型输出不是标注真值来源。对“可能支持”制定可检验任务语义，无法证实时保留U；recall只在完整标注的有限宇宙上定义，不能对开放世界报召回率。

记录opportunity、helpful/harmful/neutral、relation truth分别字段。truth sidecar不被train loader读取。训练集可以在QA阶段标注，但不能作为学习目标、prior内容或选择特定关系的手工过滤器。

## 6.4 运行和时间profile

新T_P只复核真正变动的reset/contract/Verifier/controller路径，不再全套重跑0A–0D。固定少量reference真实技能，最多10attempt取得5合法成功；同family各方法共享H、d_ref和deadline，参考不进PPO/BC/few-shot。

T_B继续历史H约23.1和d_ref约4.2；T_P优先复用suite H，若不足以表达新任务典型时长，单独冻结新family H，不重算T_B原reward。跨familyraw return不直接平均。不能因Full不领先放宽timeout或删失败。

## 6.5 cache成本

原provider、prompt、few-shot、schema维持，除非S1明确批准修订。80配置最坏需80次新首次请求；实际可核验同输入复用扣除。S1最多12首次与80相加的保守上界92，重试另外最多92。新seed/CAT/Q均复用同一cache，不重新采样VLM。

cache记录模型身份、实际payload、scene与图像hash、ID/contract/schema、raw/parsed/rejected，R冻结后不看RL结果删错边。test32只在S6获准后调用，不提前用于S4。

## 6.6 Primary Endpoint、Opportunity Density与功效边界

在S2 task_family_manifest中、任何S3训练前冻结：

```yaml
primary_endpoint:
  metric: mean_start_discounted_return
  symbol: J
  definition: "mean over episodes of sum_t w_t * r_t"
  evaluation_role: frozen_independent_test
secondary_endpoints:
  - success
  - task_cost
  - rework_count
  - skill_count
  - physical_completion_time
learning_evidence: raw_learning_auc_on_shared_N_or_T_interval
outcome_relabeling_after_results: false
```

实现不能稳定测量的次终点在S2就标NA及原因；不得等success无差再把某个成本指标改称主终点。J沿Research v5定义，真实首次完整成功奖励与原有时间折扣不变。失败时J可能为0，因此失败成本还要按预声明次终点报告，不声称J包含所有潜在代价。

同时冻结并输出机会定义，而不是仅统计cache非空：
- `all_decision_states`：各method/seed实际可核验决策状态数，包含单候选／无候选状态，排除接口未初始化记录并单列失败；
- `eligible_states`：有效可决策状态、至少两个真实合法候选、有自然admitted R，且至少一个合法候选有changed nominal patch与真实计算入口；不以最终Δ非零或模型胜负作为结构资格；
- `opportunity_density=eligible_states/all_decision_states`，并逐项报告多候选、R+、changed-patch和入口的各自分母；
- scene/episode/state三个层次的R+比例分别报告；
- helpful/harmful/neutral/unknown只能在有独立utility或可裁定关系标签的覆盖集上定义，未覆盖不能填0；
- common冻结snapshot bank与各policy实际访问分布分开，避免把不同轨迹机会率当同状态比较。

S2报告development估计与采样规则；**test opportunity density和test关系效用统计只能在S6/S7 release后计算**。不得为满足密度阈值提前查看或重选test。两个S1见证仅证明存在，不代表最终test有足够机会。

3训练seed×32test case是有限设计，episode共享训练模型、不是96个独立训练重复；通常只对较大success差异有辨识力。没有误差模型／方差／相关性假设时，不声称固定15–20百分点检测能力，也不伪造严格power analysis。因此J预先为主，success为重要次终点；报告不确定性和机会密度，不因负结果事后换主指标。

## 6.7 共享校准bank与启动资格

S2冻结16个**development**真实snapshot、各两个合法候选与两次continuation的选择规则。64条独立分支在S3/S4获准的开发分析中生成后，S7和所有模型/checkpoint共享。标签不进训练、cache或Q target；它们可以服务方法决策，因此不称blind test。完整规则见§16，不再为每个接口另收集物理分支。



输出task_family_manifest、generator、split、reference、cache_index、truth_protocol、public_input_contract、geometry/runtime变更证据和dev规则。资格要求真实可运行、可诊断和公平，不要求所有scripted dev20均成功或预期Full领先。

# 7. Training Runs：基础17与单一Elastic Pool

## 7.1 基础MUST 17个新增RL配置

| Run ID | Task | Method | Seed | 中间观察 | 默认研究终点 |
|---|---|---|---:|---|---|
| R-TB-K-1 | T_B | B1-K | 1 | 0/4k/8k/final | 可信历史E16 envelope或Tcap先到 |
| R-TB-DK-1 | T_B | B2 | 1 | 同上 | 同上 |
| R-TB-E-0 | T_B | B1-K+E | 0 | 同上 | 同T_B对照 |
| R-TP-F-0 | T_P | Full | 0 | 0/E2/E4/E8 | matched E8；S4可批准E16 |
| R-TP-F-1 | T_P | Full | 1 | 0/E4/E8 | S4冻结的相同终点；E2只保存checkpoint |
| R-TP-F-2 | T_P | Full | 2 | 同上 | 同上 |
| R-TP-K-0 | T_P | B2 | 0 | 0/E2/E4/E8 | 同Full |
| R-TP-K-1 | T_P | B2 | 1 | 0/E4/E8 | 同Full |
| R-TP-K-2 | T_P | B2 | 2 | 同上 | 同Full |
| R-TP-STAT-0 | T_P | A_STAT | 0 | 0/E2/E4/E8 | 同Full |
| R-TP-STAT-1 | T_P | A_STAT | 1 | 0/E4/E8 | 同Full |
| R-TP-STAT-2 | T_P | A_STAT | 2 | 同上 | 同Full |
| R-TP-CAT-0 | T_P | A_CAT | 0 | 0/E2/E4/E8 | 同Full |
| R-TP-CAT-1 | T_P | A_CAT | 1 | 0/E4/E8 | 同Full |
| R-TP-CAT-2 | T_P | A_CAT | 2 | 同上 | 同Full |
| R-TP-Q0-0 | T_P | A_Q | 0 | 0/E2/E4/E8 | 同Full |
| R-TP-Q0-1 | T_P | A_Q | 1 | 0/E4/E8 | 同Full |

T_B新增3；T_P为Full3＋B2 3＋A_STAT3＋A_CAT3＋A_Q2＝14；合计17。历史T_B B0/B1-K/B2 seed0、P1及有效D0观察复用。没有基础T_P B1-H或B1-K，没有新的Presentation训练矩阵。

E2/E4/E8是同run的中间checkpoint，不各计run。未到评价点写NOT_REACHED，实际Tcap先到就记录真正final N/T及残片。

## 7.2 唯一Elastic Pool = 4 attempts

固定资源槽为`ELASTIC-01`至`ELASTIC-04`，初始UNALLOCATED/authorized=false。以下用途**共用**它们，不设其他bucket：
- B1-K+E s1（§3.1触发）；
- 可选B1-H或其他经批准的描述性控制；
- 技术故障的新attempt、一次匹配调参、解释方向分歧的seed；
- S4升级净新增。

一个slot只能分配一次用途；失败attempt仍计数。真正从完整可核验状态按同配置连续恢复，可保留原attempt，但已消耗N/T/墙钟不得清零；从头重跑、回滚后重新采样或改profile要登记新的attempt成本，不能以resume名义绕过池。

**第一研究周期硬上限：17＋4＝21个新增RL attempts。**所有条件项只有触发和用户授权后才运行；文件出现或某个结果出现不自动派发。池耗尽或必要control会超过21时，停止自动执行，提交研究问题与必要预算；不能自动突破，也不能删除科学必要的control以死守数字。

## 7.3 升级先替换、后新增

S4在S5之前。2.2-B优先替换未启动的Full扩展seed，旧Full s0保留。例：CL三seed替代Full s1/s2，再有一个必要matched-readout control，净新增2，占elastic2；具体控制是否足够须由S4证明，不把此例当自动许可。

2.2-C保留Full作为λ_PC=0匹配控制，PC三seed净增3，占elastic3；若还要同版本Q消融且剩余额度不足，暂停，不用旧A_Q冒充。两种升级在第一周期互斥。所有原pilot、已跑seed和技术失败保留计数。

## 7.4 不默认安排

不恢复旧大矩阵、旧T_C任务工程或D0 E8长尾；不为表格对称给空prior的T_B加Full；不默认A_B、旧A_DD或T_P四格。A_B若将来成为经验claim必要条件，也只能申请elastic而不能成为第五个桶。cross-VLM、generalization和readout/calibration使用冻结模型，不新增RL。

# 8. 冻结矩阵的核心解释

17个基础配置分别关闭三类无法互相替代的问题：T_B的B1-K+E检验效果描述与施加；T_P的A_STAT检验后果条件化与静态prior；A_CAT检验纯交互与灵活融合；A_Q及共享utility reference检验学习／校准。

B1-H同时移除合同干预，不能完成A_STAT的单输入控制，因此只作可选范式参照。它的定义保留，但不再为补四格而增加基础T_P B1-K。

核心T_P Full/B2/A_STAT/A_CAT均三seed，A_Q两seed；T_B B1-K/B2结合历史为两seed，+E先s0。主比较全seed展示，非平凡+E或方向分歧追加只从同一4槽池中申请。

当前权威预算只有17＋4≤21。此前配置数仅在历史输入或审计记录中出现，不构成未来并行队列。

# 9. E2/E4/E8/E16与共同预算

E2=2048、E4=4096、E8=8192、E16=16384真实技能转移。它们是一个run的interim节点，不是四次训练。每1024有效转移完整PPO；样本边界应在更新完成、generation持久化后再评价。

在新run配置中区分：

```yaml
study_envelope_ncap: 16384
study_envelope_tcap: 16384 * d_ref_task
pause_at_n: [2048, 4096, 8192]
authorized_stop_n: 8192
max_complete_updates_envelope: 16
```

Tcap在采样前固定，pause/authorized_stop变化不得重新绑定H或Tcap。真正最终scientific budget在S4冻结：若停E8，各方法以同一N释放点、同一安全Tcap比较；若一条先Tcap结束，报告实际N/T，不强制凑N。skill开始前检查剩余训练资源，安全结束后记录阈值跨越量，不伪造T正好等于cap。

Full/B2/A_STAT/A_CAT是主要比较组，若CAT尚明显增长而Full已平，允许该组或预先声明需匹配的对照组累计延至E16；最终主表不能把E8与E16当等预算。A_Q要解释同终点效应时匹配；可选B1-H有单独批准的描述性scope；预算差异另作早期学习切片。

PPO保持Method2.1.1默认：Adam3e-4、clip.2、GAE.95、cV.5、lambdaQ.1（AQ0）、entropy.01、gradclip.5、4epochs、seq16、minibatch目标64、Actor无episode外层权重。prior消费者80/20，共同输入／环境流与独立参数/RNG。调整只走S4并保留尝试。

S3 pilot固定为T_P的Full/B2/A_STAT/A_CAT seed0，A_Q seed0紧随或资源允许时同批启动，共5个run，已计入基础17。默认最多2个worker；逻辑同批不等于需要5张GPU同时启动。

S3在0/E2/E4/E8做dev16；S5 profile已冻结的九个T_P扩展seed在0/E4/E8做dev16，不重复低信息E2开发评价。所有run仍按真实更新边界保存E2 checkpoint并做必要在线安全/数值检查，供跨checkpoint校准使用；省略的是闭环dev，不是checkpoint或安全检查。T_B三条新增沿0/4k/8k/final的共同历史协议。

# 10. Stage 4 — 单一Method Decision Gate（S4）

**时间：同profile seed0代表实验和必要development分析完成之后、大规模扩seed之前。**S4不得读取独立test结果。

核心输入固定为Full、B2、A_STAT、A_CAT、A_Q、E1–E6、prior opportunity density、DP-Zero、readout decomposition、共享Q_ref的跨接口/跨checkpoint calibration、Q预测质量与gradient clipping audit。B1-H不是门的必需项。

先统一检查诊断性、输入/计数/梯度、信息和参数公平、训练预算及是否仍在学习。证据不足时沿既有S1/S2的一次有据修订处理，不把task缺机会混成算法失败。所有成绩主判据使用**S2预先冻结的J**，success、cost等为既定次终点，不在结果后更换主指标。

| Case | 真实观察 | 决策依据与允许动作 | 不能据此直接宣称 |
|---|---|---|---|
| A | Full优于B2、不弱于A_STAT，且相对CAT有clean或已测robustness/calibration优势 | 保持2.1.1，冻结profile与seed扩展；seed0若仍不确定就保留其不确定性 | 普遍优势、strict locality或每个组件独立有益 |
| B | A_STAT不弱于或优于Full | 检查CAT、行分解、G_ref与扰动；可能是静态信息重要，也可能统计／优化未分开；先做已有分析的prior influence decomposition | 排序本身证明静态是唯一原因，或自动新增第三升级路线 |
| C | CAT优于Full，而A_STAT约等于Full | 非线性交互/表达力是优先假说；结合容量、读出分解、局部输入证据后才考虑2.2-B | CAT胜就证明consequence locality，或原Full是严格局部线性特例 |
| D | Full约等于B2，但结构机会充分、真实动作翻转存在 | 看既定J/cost、difficulty和G_ref，区分success ceiling、等效动作及有害重分配 | success无差就自动换主终点或删prior贡献 |
| E | A_Q无清楚增益或Q/校准不足 | 联合pre/post clip、Q真实预测质量、prior梯度与E2/E4/E8校准轨迹；可靠Q且prior欠训练时才考虑2.2-C | 任意成绩差只来自Q representation，或C3必因AQ持平消失 |

“≥／≈”是待结合逐seed、case不确定性和预定义指标解释的观察，不是机械统计判定。多个现象可同时存在，S4记录主要解释和未排除因素，不再扩成几十个自动分支。

## 10.1 外部强参照和C1方向不一致

B_PLAN在T_B足够强时保留其成绩，讨论摊销计算、实际执行非名义因素或T_P，不削弱planner。LLM-Planner强时报告成本与信息界，不预言它必受幻觉影响。

T_B seed1未复现，或+E显示非平凡学习，按冻结规则申请elastic，不连续补seed到同向。可选B1-H若执行，只作其批准范围解释，不重新恢复基础析因矩阵。

## 10.2 Upgrade B — 先验证条件，再替换名额

采用Research Content v5 §22.2的区域内重算＋四项非线性contrast。冻结N_i、边界、degree normalization、goal/global处理、null/empty-patch/bound及新locality条件。不能对已全局污染的embedding做末端mask就宣称严格局部。

优先替换尚未启动的Full s1/s2；旧Full s0保留。S4逐项写明新CL、A_STAT/CAT/B2和必要Q/control的匹配关系及哪些旧结果仍可复用。任何净新增只占elastic4，已发生attempt不删除。必要control超21则暂停人工决策，不削control，也不自动延伸。

## 10.3 Upgrade C — 原Full作为λ_PC=0控制

G_Q先标MODEL_Q_PROXY并由独立Q_ref验证；若启用credit，明确prior-specific参数、detach、滞后critic、warm-up和权重。Full保留为同条件λ_PC=0控制，PC净新增从同一elastic池扣除。2.1.1 A_Q不能自动充当2.2-C的Q消融；credit在无Q时无法定义的问题要在同版本控制中解决。超上限则暂停，不伪装成本不变。

## 10.4 决议与冻结

`method_decision.json`记录观察、主要Case、依据文件、未排除解释、修订attempt、最终method/input/optimizer、主终点、剩余run、已用elastic及用户批准。第一周期最多一个B/C升级方向。S4之后不因test结果改方法；常规新结果只更新状态和论文，不能重新打开Master设计。

# 11. Stage 5 — Seed Expansion（S5）

默认最终身份：T_B B1-K/B2 s0/s1、B0历史s0、B1-K+E s0；T_P Full/B2/A_STAT/A_CAT各s0/s1/s2，A_Q s0/s1。B1-H只可选；+E s1按§3.1触发并申请elastic。没有独立factorial追加桶。

所有预定有效seed都有正文或附录位置，representative seed只用于illustration。SHA派生model/environment/action/prior/evaluation流分别固定；多一次prior抽样不改变公共环境流。同结构公共模块按同规则初始化、参数和Adam对象独立，不要求不同方法实际轨迹相同。

S5沿S4最终profile与scientific budget运行；九个T_P扩展seed闭环dev点为0/E4/E8或批准的final，不重复E2开发评价。E2 checkpoint仍保存，用于离线校准；必要interim safety和first-update会计检查不省略。若批准E16，新增final点成本明列。

S0发现实质性公共输入修复时，历史seed0不能与修复seed1伪装同profile。重训只从elastic4申请；没有足够名额且科学上需要时暂停人工决定，不掩盖兼容性。

不能因为一个seed表现差单独给它更多预算或删掉；技术失败与算法负结果分别记录。两／三seed不能当录用或严格统计功效保证。

# 12. Stage 6 — Independent Test（S6）

## 12.1 冻结顺序

1. S4冻结方法、输入、优化与研究终点；S5完成规定seed。
2. 各run主评价checkpoint默认最后有效的post-update generation。若需要best-dev部署比较，单独预先声明选择规则并记录N/U，不能看test再选；主表同一规则。
3. 保存model/Adam索引、源码与所有input hash及`test_release_manifest`。
4. 只在此后允许生成/读取test cache、关系truth和运行test。开发选模与最终test完全分开。
5. 每个method/seed/checkpoint在同一test场景顺序只跑一次；R复用、evaluation RNG独立。不重复到满意。

## 12.2 基础独立test数量

T_B共6个模型：历史B0 s0、B1-K s0/s1、B2 s0/s1，加新增B1-K+E s0；6×30＝**180 episode**。历史权重与输入不能证明可用时，不用sidecar假装可评、不自动重训；该对照缺项如实保留并按elastic/人工决策处理。

T_P基础14模型×32＝**448 episode**。基础学习模型独立test总上限**628 episode**。conditional +E s1、可选B1-H、重跑或升级产生的有效新终点按实际批准模型数另计；它们不隐含在628内，也不增加基础run。

D0旧test仅引用原身份，不重跑；T_B test若曾用于开发，不能改名blind test，必须先处理来源资格。test不是用来确认Full必须获胜的最后调参集。

## 12.3 统计与预算比较

每task报告各seed，case层paired差与seed层不确定性分开。可用分层bootstrap，但2/3seed区间不能解决样本少的问题；主张范围按证据。不同N/T/更新时点列清楚。训练raw曲线、best-dev及blind test为不同数据角色，不能拼接一个“最好数”主表。

# 13. External Baselines执行协议

## 13.1 共享真实执行接口

所有baseline使用同一个能力目录、facts、goal、Safety/Verifier/Evaluator、真实skill时间、deadline。不给规划器隐藏真值，不给LLM任意生成新技能。先在dev冻结search/提示和预算，再test。

## 13.2 B_PLAN

有限代价search/replanning在三态注册合同上展开名义后继，每次执行一个首动作后用真实facts重规划。默认深度6、最多4096节点、每decision2秒CPU搜索，canonical tie-break；S0若合法名义解超过深度6，可依据reference将深度最多调到8并冻结一次，不能因为Full领先与否改。

初始skill cost来自与模型同源的独立reference中位时长，正数；不使用每test真实未来time。启发式以未满足目标数／合同最短松弛步数实现，明确不保证admissible。搜索失败、timeout、NO_PLAN与实际skill失败分别计数。

## 13.3 B_PLAN+R的固定软映射

保持search、节点预算和hard constraints完全一致。默认首先按合同cost排序；对cost处于一个预先冻结容差内的候选计划作R tie-break。默认容差为0.1×同task d_ref，不根据test调。

关系计分q_R(plan)使用合法SOFT_SUPPORTS中source在destination前、source注册效果在名义步骤确有变化且destination出现在剩余计划的计数；SOFT_RELEVANT_TO_GOAL只给与未完成签名目标关联的salience计数，不声称其效果一定正向。总分按关系数归一，最多一个固定权重，无relation truth和LLM自报confidence。

这是一种明确的naive-use参考，可能被错边误导，不能当最强prior planner或用它失败证明任何学习方法必要。S1若计分在所有plan上常数，记录HEURISTIC_NONDISCRIMINATIVE，最多一次更有依据的soft-cost映射修订。最终rule freeze后R与R\*使用相同算法，不给oracle更长搜索。

## 13.4 B_PLAN+R\*

只在有独立关系标注的T_P dev/test分析。R\*只改变relation集合，不提供未来轨迹、最优动作、Q_ref分值或success。标ORACLE_INFORMED_REFERENCE；除非另有最优规划与真值完备证明，不写upper bound。

## 13.5 LLM-Planner — MUST

输入固定序列化公共事实T/F/U、签名goal、合法candidate列表、各PRE/ADD/DEL/UNKNOWN和guard、同一R、公共对象/本体/历史摘要与剩余真实task时间。没有oracle truth、nominal Q、未来结果或额外实时图像。每步只返回`{"candidate_id":"..."}`，必须属于已给mask。

模型provider/snapshot、region、prompt、examples、temperature、token cap在dev冻结；优先同prior provider，但不未经权限核验假定服务可用。最多16个开发决策供提示语法校验，不能挑输出最差的模型作baseline。

每次决策一次API，最多一次格式/传输修复，禁止语义重试；两次无合法回答记接口失败，不静默fallback成B_PLAN或删该episode。调用90秒墙钟上限，输出2048token上限；不记录私钥/authorization header。保留请求hash、响应、request ID、latency、token及实际模型回显（无回显写未知）。

初始预算D_eval=12个技能决策/episode，必须在S2按reference确认能完成合法计划；若不足，正式比较前统一修订所有final-eval方法相同decision cap与API总预算。物理task deadline与decision cap都记录。不能只限制LLM步数而学习模型无限重试，再称严格同预算。

input-matched不等于相同pretraining知识和表示损失。prompt历史摘要的范围必须与公共观察可追溯，不能给LLM额外完整失败诊断oracle。若共同公开信息没有文本无损表达，写明差别，必要时分别展示information-matched与practical面板。

## 13.6 VLM-Planner — OPTIONAL

只在T_P test32，用同技能接口每步增加当前RGB。称stronger-observation practical baseline，不称严格same-input。上限32×12=384首次调用，含一次重试最多768attempt。费用额外授权，不因0训练而说免费。

## 13.7 外部baseline数量与复用

基础final：B_PLAN在两task62episode；B_PLAN+R最多62（T_B R为空且算法完全退化同B_PLAN时复用30，不重复）；B_PLAN+R\*仅T_P32；LLM两task62。名义上限218ep，其中可能直接去重30。

S1同case、同算法、同版本的开发输出可用于E5/E6和baseline开发报告，不算新独立样本；不能复用dev替代test。外部baseline胜出如实报告，比较训练采样成本与在线API/latency，而不是仅截取一个有利轴。T_P test明确报告B_PLAN与B_PLAN+R的J、success、cost及成对差：这是与Full无关的任务prior使用证据，正向、负向或无差都保留，不另增加同case评价。

# 14. Stage 7a — Prior Robustness

## 14.1 基础闭环：NoPrior与Target-swap

使用Full、A_STAT、A_CAT各3个已训练有效seed，共9个模型。对预先冻结的16个T_P test子case，Original复用相同模型／checkpoint／随机流的独立test，不重评；新增NoPrior和合法Target-swap两条件：9×16×2＝**288 episode**上限。

case按hash/已声明见证分层选择，不按模型成功或响应大小选择。报告全test和条件子集的关系；无有效机会的平坦曲线是低诊断性，不自动是鲁棒。编辑不得改task goal、真实facts、candidate registry或hard mask。

Deletion若与NoPrior完全相同，直接复用；无合法target替换时N/A，不以schema拒绝当神经鲁棒性。A_Q在calibration中比较，若另主张其robustness作用需指定授权，不默认扩全部方法。

## 14.2 GRAPH_REMOTE_PRIOR_INJECTION：先前向，再决定闭环

默认名称固定为**GRAPH_REMOTE_PRIOR_INJECTION**：图结构上远离candidate patch的关系编辑，不意味着实际网络的计算路径隔离。先执行§15的固定state前向敏感度。

只有有合法remote关系且同状态TV超过S0冻结的数值重复误差容差、显示非平凡policy变化，才可申请该组闭环；不按它是否损害CAT或有利Full决定。闭环对预先选定的最多16case、9模型、一个remote条件，上限**144 episode**；未触发则0。相同case集合规则不能在看到TV后挑选最有利几个。若strict locality的升级条件未成立，不使用“off-path invariance”命名结果。

OPTIONAL LLM扰动保留最多8case×Target-swap/GRAPH_REMOTE两条件＝16episode，API另计，不作为新的复杂baseline。

## 14.3 Cross-VLM — SHOULD，条件授权

保持训练provider A与权重冻结，在同一32个T_P test初始输入请求provider B一次，最多32首次＋32合法重试；B选择和请求语义在看其test结果前冻结，不挑有利provider。

主要比较**Full与A_STAT**各3seed、16固定case，共**96 episode**。它直接对照conditioned与static prior面对provider变化的行为；不默认增加CAT或B1-H。

新R的schema/ID/provenance和实际调用成本记录，不能重训后仍称0训练swap；天然关系质量变化与合成Target-swap分开解释。未来服务可用性需获准时核验，不在本轮宣称已授权。

# 15. Stage 7b — Graph-Remote Sensitivity与Readout Decomposition

## 15.1 先区分图距离与计算影响

当前指定global nonlinear readout不保证端到端strict locality。S0导出的节点传播、goal行和global行要分开；图远端编辑也可能通过非线性global row影响DP与policy。无真实远端bin时N/A，不造新对象、predicate或假边。

## 15.2 合法编辑与距离

同节点、goal、facts和真实mask，仅编辑schema允许的关系及其反向边。距离以包含所有比较编辑的并图、归一化footprint到candidate changed-patch为准，另存有向reachable、度数变化与node/global读出。

当前统一叫GRAPH_REMOTE_PRIOR_INJECTION。只有S4批准2.2-B且新区域／完整计算图确实满足条件后，才允许将对应试验称为off-path invariance；该性质不能回写2.1.1。

## 15.3 Sensitivity–Distance

冻结32个development真实state作为S3/S4/S7共享机制bank（可复用同一真实episode中的多个状态，统计按case聚类），每state最多8个合法编辑，Full/A_STAT/A_CAT共9模型。不得用最终test状态参与S4方法决策。记录DP或对应接口输入、Δ/logit与TV、候选flip分母。A_STAT使用S_R；CAT使用C/C0，不能把它们的输入一律重命名为DP。

按state和seed保留原始点，编辑数不当独立episode。距离与语义／度数可能混杂，因此不作仅由距离导致影响的因果定律。Full远端非零是需要解释的真实结果，不自动是实现错误。

## 15.4 Relation-level leave-one-out

每state最多8条真实R逐条删除，重算同模型输出，报告关系—候选关联。它不是可加因果归因，也不训练relation error learner。不根据test删边后的收益重新选择R。

Sensitivity与LOO共用未编辑baseline：32state×9模型×(1基线＋≤8remote＋≤8LOO)，最多**4896次policy forward**。它们仍受原有6000联合上限约束，不将两项各给一个6000桶。

## 15.5 Readout Decomposition Analysis — MUST，0新RL

用同一32-state bank和9个模型，不增加环境episode。在**prior支路的共同goal/global读出行接口**做两种干预，真实合同分支、DK、c_i、u_i^K、hidden、模型参数和mask保持固定：

| 模型 | 原prior输入／anchor | global-row置零 | goal-aligned rows置零 |
|---|---|---|---|
| Full | DP与零anchor | 仅DP的global行置零；anchor仍为零 | 仅DP的全部goal行置零 |
| A_STAT | S_R与零anchor | 仅S_R的global行置零 | 仅S_R的全部goal行置零 |
| A_CAT | 同一投影后的C、C0两路 | 两路同一global行一起置零 | 两路同一goal行一起置零 |

CAT不得只改C而不改C0，也不得重算一个局部encoder来替代本分析。一次条件对该state的全部合法候选施加同一行选择，单次policy forward仍返回全候选输出，不另为每candidate创建一组预算。行数量与attention mask不变，不删行改变归一化。选定切点为进入prior candidate readout前、共享行投影后；切点、bias、复用和实际hook hash在分析前冻结。

保存各行norm（Full:DP，STAT:S_R，CAT:C/C0及其差作描述）、原始Δ、两干预的Δ变化、TV和action flip。新增最多32×9×2＝**576次前向**；未编辑baseline复用§15.3。无goal行时对应干预N/A。

**解释边界：**goal-aligned行有其节点感受野，但并不自动是严格consequence-local；global context与attention仍可能耦合。两种置零效应通常不可加，不能除以总Δ后宣称“x%物理因果来自局部、y%来自全局”。该分析给出读出通路依赖的经验替代证据，而不是补出已不存在的端到端定理。

## 15.6 前向总预算

§15基础最多4896＋576＝5472次；§16跨接口/轨迹最多1056次；其余既有DP-Zero、E2*容量、Q-proxy及少量gradient-reach输入合计剩余最多472次，**研究policy-forward总上限7000**。可从一次原始logit同时计算π与π0时扣除实际复用。S0生产单元测试调用另列工程计数，不冒充研究前向结果；gradient backward次数单列且optimizer=0。没有新state、seed或任务被此算术新增。

# 16. Stage 7c — Utility-Aligned Prior Reliance

## 16.1 同一批独立物理分支，全接口共享

S2冻结16个development真实snapshot、每state两个合法候选、每候选两次配对continuation，合计**64条物理分支**。这些分支可在S3/S4开发诊断中先生成，S7继续复用；不是每model×checkpoint各64条，也不是曾用于S4后仍叫blind test。

固定continuation、分支恢复与随机流必须有证据。Q_ref估计有限continuation下的真实动作效用，不是最优Q*；relation truth不是Q_ref。任何label不进入训练、cache或Q目标。

优先选择恰有两个合法候选的state以取得完整支持。合法候选更多时，只能在已测子集上计算明确标为conditional/subset的G_ref，并同时报告原概率覆盖；未测候选不能填零、填模型Q或从一个执行动作反推出排序。无法恢复state则保留实际标签分析／N/A，不能伪装反事实。

## 16.2 四接口主要量

在同一state、同一真实观测前缀和同一Q_ref上，比较**Full、A_STAT、A_CAT、A_Q**。每个模型分别用自己的冻结参数重算hidden，再取得π与仅将最终prior residual置零的π0。Full@Absent、DP-Zero、Δ-Zero和A_STAT不是同一个条件，不混命名。

完整支持时：

\[
G_{ref}(s)=\sum_i(\pi_i-\pi_i^0)Q_{ref}(s,i),
\]
\[
H_{ref}(s)=\sum_i[\pi_i-\pi_i^0]_+
\mathbf1\{Q_{ref}(s,i)<\max_jQ_{ref}(s,j)-\epsilon_Q\}.
\]

分别报告G_ref、Harmful Mass、Action Flip Quality（beneficial/neutral/harmful/unknown）、TV、有效候选分母和实际参考误差。ε_Q在dev按回报同单位冻结；有噪声覆盖分类边界时unknown，不假装精确效用真值。

关系真／错／未知、R+、机会、helpful/harmful/neutral、graph-remote与seen/unseen分开。不能要求“真关系必须正Δ”或“错关系必须零Δ”。正G_ref仅表示该固定continuation上的一步概率搬运，不直接等于全episode policy改进。

## 16.3 同run E2/E4/E8轨迹

对Full3、A_STAT3、A_CAT3、A_Q2共11个模型身份，在同一bank的E2/E4/E8 post-update checkpoint计算同样量：16×11×3＝**528个state–model–checkpoint比较单元**，每单元π/π0最多两次前向，保守上限**1056次policy forward**。如果同一次forward可由base logits直接得到π0，按实际扣除，不把528比较单元误当永远528次调用。

未达到的checkpoint标NOT_REACHED，sample N与完成update单独记录。S5只省略E2闭环dev，仍保留E2 checkpoint。额外模型和checkpoint只增加前向，不增加64条物理分支。

结果只有真实显示随执行训练改善时，才支撑outcome-grounded reliance；Full/A_STAT/CAT都改善、AQ不差或某接口更好均保留。不能挑中途最漂亮checkpoint替换预定终点，也不能没有参照只写“Full校准良好”。

## 16.4 Q-proxy与Gradient Clipping Audit

G_Q=Σ(π−π0)sg(Q)只标MODEL_Q_PROXY。先看独立已执行动作的预测质量，再对照G_ref；A_Q未训练Q-head的误差不用于证明Full表示更强。

逐method/seed/update报告pre-clip norm、post-clip norm、clip-trigger rate。Full与A_Q的总梯度不同，global clipping可能改变Actor/V有效梯度缩放，所以AQ差异应结合这一机制解释，不只归因为Q的表示学习。保持默认训练过程，不新增无clip或改lr对照。

gradient reach可在已冻结的实际batch上计算，no optimizer；记录backward次数与sample来源。未更新counter为NOT_MEASURED。上述分析、64分支、所有失败与估计不确定性均单列成本；0新RL不等于0计算。

# 17. Task Family Difficulty Axes

## 17.1 T_B合同族

优先轴是**合法初始前置满足程度／真实候选分支数**。若已有controller和模板可支持，才再扩依赖深度或可交换对象数；必须保留同目标语义、注册合同及安全执行。通过修改mask或给B2更多候选不属于难度控制。

不能改变实例规模时，用已有state按分支数分层，名称为within-task stratification，而不是“泛化到新任务规模”。至少有两档实际非空水平才出曲线。

## 17.2 T_P先验族

优先两个S1真实见证类型及关系覆盖／candidate ambiguity轴；自然prior质量只分层不改训练标签。扰动质量轴属于robustness，与自然关系密度难度分开。

难度轴为SHOULD，默认三个预注册level×每level8case，core级与已有test完全相同时复用。T_B B1-K/B2有效2seed最多96ep；T_P只做Full/B2各3seed最多144ep；合计上限240ep。A_CAT负责representation design comparison，不默认承担breadth。实际重复条件扣除，不为每level另训模型。

所有level若已知只有一档有效，报告轴不可用，不通过重新调几何制造更漂亮下降趋势。若任务族难度实质性改变训练分布，进入新任务版本而非“0训练无条件复用”。

# 18. Low-cost Generalization

优先：未见对象类别且合同结构不变→未见布局→真正未见组合。前提是既有冻结观测前端、对象绑定和技能controller能处理该类别；更换颜色或ID不是类别泛化，依赖新的检测器则必须另计成本与输入版本。

一个20case holdout，默认Full/B2/CAT全部有效主seed共9模型=180episode，0训练；若仅代表seed展示则明确illustrative，不写跨seed泛化。没有合法来源时该卡N/A，不开发大量新资产，转为难度轴与cross-VLM深度证据。

组合泛化必须在生成器中明确训练未见的依赖/目标组合，不能把新随机布局改名composition。与主test隔离，不能看其结果调方法。

# 19. 两条Qualitative Analysis

MUST各一条真实链，优先复用：

1. **Contract chain**：P1中同状态DK正常/置零的候选排名、被选技能和后续执行。
2. **Prior chain**：T_P最终Full中一条真实关系→某candidate DP/残差→概率变化→独立执行后果。正确利用或错误依赖均可成为解释例子。

每条包含scene/frame引用、T/F/U、合法mask、candidate ID、真实R及effect出处、DK/DP/Δ摘要、real action与Verifier/Evaluator。避免为latent维度起未经证实的物理名称。

优先按预定义规则选择有多个合法候选、记录齐全的第一个机制见证；选择后保存理由，完整分布仍在统计表。若缺画面最多各一个有界replay，新增2episode；不能把replay的更好成绩替换原统计。

# 20. Evaluation Metrics字典

| 指标 | 定义／单位 | 必须的分母与限制 |
|---|---|---|
| success | 独立首次完整目标确认 | valid/planned/API/技术失败均公开 |
| start return G / primary J | episode G=Σw_t r_t；J为其均值 | T_P primary，S2在S3前冻结；同H同task，不跨H混合 |
| learning AUC | 共同实测N或T区间、原始点梯形积分除区间长度 | 稀疏点插值只派生估计，不造中间观测 |
| task duration | reset任务起点到独立首次成功 | 失败的成功时长NA，另报失败episode时长 |
| terminal skill duration | 最末技能end−start | 不当全任务完成时间 |
| physical cost | 实际time/skills/预注册能耗代理 | 能耗未测则不编造 |
| rework | 重复执行本已完成子目标／恢复事件的预定义计数 | 不靠事后看赢家改定义 |
| action flip | 同X/hidden下argmax改变次数 | comparison total、multi-legal total、tie次数 |
| TV | \(\tfrac12\sum_i\lvert\pi_i-\pi'_i\rvert\) | 相同candidate/mask；轨迹分叉后不能当同状态 |
| DP/Δ | RMS、max、saturation和非零数 | opportunity=R合法非空＋真实patch＋多候选等分别报，不混成一个无解释率 |
| utility-aligned prior reliance | G_ref、Harmful Mass、Action Flip Quality，另列G_Q proxy | 共享64分支；Full/STAT/CAT/AQ及E2/E4/E8；覆盖不足用subset或NA |
| relation precision/recall | finite truth universe中的T/F/U | unsupported denominator=NA；不开放世界recall |
| redundancy | 按admission直接合同同义项剔除 | 不扩大为所有可推导路径都删 |
| distance sensitivity | GRAPH_REMOTE编辑footprint到patch距离条件下的TV等 | 并图、degree和global readout；不是strict off-path不变 |
| readout decomposition | prior支路global/goal-aligned行置零后的norm、Δ、TV、flip | 两branch anchor同步；非线性不保证可加百分比 |
| gradient clipping | pre/post global norm、trigger次数/实际optimizer steps | Full/A_Q同记录，读取不改变grad |
| opportunity density | eligible decision states / 全部可核验decision states | R+、多候选、patch、入口各分母；不以模型Δ非零来定义结构资格 |
| latency | 每decision p50/p95、policy/API/env分段 | GPU同步测量范围固定 |
| API成本 | 首次、重试、token、wall | 0训练baseline也消耗API |
| 参数量 | allocated/active/有梯度参数 | 不用死参数填等容量 |
| budget | N、实际T、完整/残片update、optimizer steps | checkpoint pre/post、sample boundary与durable boundary分开 |

未运行、不适用、未测、数值零是四种不同状态。所有图表依据原始事件；新派生CSV不覆盖旧Raw。平滑只加可见overlay，主表与test不平滑。

# 21. Negative Result Handling

任何负观察按以下七层一次处理，不能绕过也不能无限循环：

1. **Diagnostic**：是否至少两个合法候选？patch/关系/真实后果是否有变化？
2. **Implementation**：输入路径、R固定、counter、Q目标、eval时点、mask及时间是否正确？
3. **Information fairness**：效果字段、hidden truth、extra observations和参数/优化机会是否匹配？
4. **Sufficiency**：曲线是否仍增长、目标预算是否真的执行、早停是否算法而非基础设施？
5. **One justified adjustment**：最多一个局部、可解释、对关键比较公平的修订，使用同一elastic4。
6. **Method decision**：真正的表达力／学习信号问题才由S4选择B或C，不是为救某个seed。
7. **Claim decision**：有效实验仍否定某项经验优势，保留负结果，准确修改贡献；不能将“保持创新”理解为禁止结论变化。

绝不以“Full没有赢”判实施失败。也不以D0 ceiling或未测opportunity作为C2退出依据。论证来源与证据强度比结果方向优先。

# 22. Stop Rules：问题回答后即停

- E1–E6：限定来源与见证完成即可，不等先验越来越多；一次修订后仍无诊断性就做研究决策，不第3种task自动搜索。
- C1：T_B匹配B1-K/B2第二seed、+E s0和现有P1齐；+E s1只按规则申请elastic，不因方向不理想无限追加。
- C2：Full/B2/A_STAT/A_CAT的预定三seed及实际用途与错误敏感度完成即收束；不要求每个指标Full第一，不再补基础B1-H四格。
- C3：A_Q两seed、四接口共享Q_ref／跨checkpoint和clip解释完成即收束，不为同一结论全task补Q。
- Locality：条件证明可审查或经验曲线已如实说明global混合，即停止；无远端bin就NA，不造hub-free假图充正式任务。
- Robustness：固定条件/强度/seed/子集跑完即停；不继续加最有利噪声等级。
- Test：只作冻结终点评价，不因不理想反向调参。
- 升级：第一周期最多一方向，全部净增／重跑使用elastic4，总量不超过21；必要control超限须停止自动执行并人工决定，不以删control守数字。
- 私人展示：核心研究尚缺时不靠排版宣布收工；核心比较完成后不为表格对称补无信息实验。

完成状态须分：ENGINEERING_COMPLETE、EVIDENCE_COMPLETE、METHOD_DECISION_PENDING和PAPER_VIEW_FROZEN。真实负结果可EVIDENCE_COMPLETE，未运行对照不能以“设计合理”替代。

# 23. Resource / Parallel Plan

资源并行不改变stage依赖：S1/S2是T_P训练前置，S4在大规模扩seed之前，S6在方法／权重冻结之后。动态执行只在Runbook规定的xushijie2工作区。

默认2个训练槽，一张获准且足够空闲GPU一个worker。S3逻辑同批五run按Full/B2、A_STAT/A_CAT、A_Q顺序排入空槽；不要求5GPU并发，也不把多进程堆到一张争用GPU。

S0科学审计与获独立S0-TB授权的T_B三条可分开调度。S5在profile冻结后资源足够可用4槽，但不能将S1标注、API、写作和方法决策时间按GPU数机械等分。所有评估process与训练参数/RNG隔离。

具体用户/Python/import/输出断言、phase-aware heartbeat、generation持久化和Git规则放Runbook附录。它们是内部执行纪律，不进入论文的创新或Results叙事。

# 24. Estimated Workload — 17基础＋4弹性≤21

所有数字是未来获准任务的容量上限或规划算例，不是本轮已运行值。按训练run、独立环境episode、仅前向和API分别核算，避免把“0新RL”写成免费。

## 24.1 新RL训练与attempt

| 项 | 计算 | 上限 |
|---|---|---:|
| T_B新增基础 | B1-K s1＋B2 s1＋B1-K+E s0 | 3 runs |
| T_P新增基础 | Full3＋B2 3＋A_STAT3＋A_CAT3＋A_Q2 | 14 runs |
| **基础合计** | 3＋14 | **17 runs** |
| **单一elastic** | +E s1、B1-H、故障重跑、匹配调参、分歧seed、upgrade净增共享 | **4 attempts** |
| **第一周期硬上限** | 17＋4 | **21 attempts** |
| 基础默认转移释放 | 3×16384＋14×8192 | **163,840 transitions** |
| T_P全部批准延至E16的基础释放 | 17×16384 | **278,528 transitions** |

elastic转移和评价按被批准scope另记，不预填为完成，也不再给upgrade另建桶。sample/state未恢复的回滚重采不得消除已有成本。超21的必要科学控制只能人工新决策，不能自动启用。

## 24.2 环境评价：默认去重调度

S3五个T_P seed0有0/E2/E4/E8四个dev16点；S5九个扩展seed省E2闭环dev、保留0/E4/E8三个点，E2 checkpoint继续保存。下表不包含训练episode本身。

| 工作 | 优先级 | 计算 | episode上限 |
|---|---|---|---:|
| S0 T_B B_PLAN开发 | MUST | 10 | 10 |
| S1三planner与必要QA | MUST | 3×12＋8 | 44 |
| T_P训练dev监测 | MUST | 5×4×16＋9×3×16 | **752** |
| T_B三条新训练dev | MUST | 3×4×10 | **120** |
| 学习模型独立test | MUST | T_B 6×30＋T_P 14×32 | **628** |
| 外部final baseline | MUST／R*分析 | B_PLAN62＋R62＋R*32＋LLM62 | **218** |
| NoPrior／Target-swap | MUST | 9模型×16case×2 | **288** |
| DP-Zero闭环 | MUST机制包，优先复用前向 | Full3×8 | **24** |
| 独立calibration物理分支 | MUST | 16state×2候选×2重复；跨接口共享 | **64** |
| qualitative补画面 | MUST材料，优先已有 | 最多2条 | **2** |
| **核心保守合计** | 不要求凑满 | 无重复条件扣除前 | **2,150** |
| 难度轴 | SHOULD，需明确释放 | T_B96＋T_P Full/B2 144 | **240** |
| **核心＋推荐难度轴** | 默认规划情景 | 2150＋240 | **2,390** |

若所有14个T_P模型都做四个dev点，会是896个T_P dev episode和2534个含难度总量；本冻结计划不采用该重复调度。省去九个扩展seed的E2点节省**9×16＝144 episode**，不是288。保留这一算术对照仅解释账本口径，不重新授权被删监测点。

T_B的+R与B_PLAN完全退化且身份相同可直接减30；已有core难度点与test一致、已有qualitative素材、合法条件不存在等均按真实去重扣除。不能把两个seed的不同真实运行当重复删掉。

批准所有T_P基础run累计延至E16时，最多新增14×16＝**224个final dev episode**，不是新增14个run。部分延长按实际匹配组计算；conditional模型和upgrade的test／分析开销在同一账本单列，不隐含在628中。

## 24.3 可选与条件性评价（不自动叠加）

| 项 | 上限episode | 触发／限制 |
|---|---:|---|
| GRAPH_REMOTE closed-loop | 144 | 固定forward先显示超数值噪声的非平凡TV；9模型×16 |
| Cross-VLM | 96 | Full/A_STAT各3seed×16；另32新provider输入 |
| 合法generalization holdout | 180 | 既有20case×Full/B2/CAT9模型；无来源不建新任务 |
| LLM扰动 | 16 | OPTIONAL，两个条件各8case |
| VLM-Planner | 32 | OPTIONAL，当前RGB更强权限 |
| **这些扩展全部开启的额外上界** | **468** | 不包括已单列的SHOULD难度240 |

一个闭环条件不存在合法关系时N/A，不另造关系或多跑case补数量。所有技术失败、API失败和截止仍在实际attempt账内，不以排除失败压低成本。

## 24.4 仅前向／裁剪／标注

- sensitivity＋LOO共享baseline：最多4896次policy forward；
- readout decomposition：最多576次追加forward；
- calibration：16×11×3＝528个比较单元，π/π0保守最多1056次forward；
- 其余既有DP-Zero、E2*、Qproxy和gradient reach输入共用剩余472次；**研究前向总cap=7000**，实际复用扣除；
- gradient-reach的backward次数另记，optimizer=0；clip audit读取现有训练日志，不开新run；
- 关系标注仍是有限宇宙两次独立核对：最多36发现scene＋dev16＋test32＝84场景；如果每场景约10关系且全量二次核对，约1680次判断，这是工作量算例，不是实际标注数；
- 64物理分支的reference标签跨接口／checkpoint共享；不为四接口再乘物理成本。

前向耗时依赖实际候选数和encoder调用，不承诺“几千次必少于两小时”；先记录有限窗口吞吐，再更新计划，不借0RL隐藏GPU开销。

## 24.5 VLM／LLM首次与尝试上限

| 工作 | 首次调用上限 | 含每次最多一次合法重试的attempt上限 |
|---|---:|---:|
| 主prior cache：S1≤12＋T_P80 | **92** | **184** |
| LLM-Planner dev语法＋main | 16＋62×12＝**760** | **1520** |
| **基础合计** | **852** | **1704** |
| OPTIONAL LLM扰动 | 16×12＝192 | 384 |
| SHOULD Cross-VLM cache | 32 | 64 |
| OPTIONAL VLM-Planner | 32×12＝384 | 768 |
| **全部上述API扩展均获准时总上界** | **1460** | **2920** |

该调用总表覆盖已批准的S1发现输入、T_P80和显式cross-VLM/LLM/VLM条件。difficulty与generalization只在现有冻结输入及精确cache可复用时按此预算释放；若它们另需新scene的provider请求，必须先列出实际新增输入与调用预算请求人工决定，不能暗含在“0训练”或1460中自动调用。重复cache输入可精确复用扣除；不能只用文件名或近似图像判相同请求。final-eval默认12decision/episode在S2用合法reference核验，修改必须在正式比较前对所有对应方法公平并同步重算调用上限。token、费用与API wall单列。本轮没有任何provider请求。

## 24.6 墙钟情景（不是完成保证）

沿底稿规划假设：T_B E16每run10–16小时；尚未绑定的T_P E8每run4–12小时；闭环episode平均0.5–3分钟。原R1仅提供历史B1-K约10.07h、B2约15.91h，不是新模型吞吐保证。

- 基础训练：3×(10–16)＋14×(4–12)＝**86–216 worker-hours**；
- 1训练槽纯训练约3.6–9天；2槽理想43–108小时，但不能并行跨过S1/S2/S4依赖；
- 核心2150评价约17.9–107.5 worker-hours；含推荐难度2390评价约19.9–119.5 worker-hours；
- 基础训练＋核心及推荐难度约**106–336 worker-hours**，另加API等待、仅前向、标注、周转和必要实现；
- 1槽完整周期可按约5–16天初始预留；2槽因阶段依赖与人工环节可按约5–13天预留，均不是截止保证；4槽只在S5或已释放评价时有益；
- T_P延至E16、elastic、条件评价和未知资源问题必须按实际另算，不把上面默认范围当所有情景的硬时间上限。

不会为了兑现估计而削弱baseline、改变reward/timeout或扩大未经授权并行。最终报告写真实消耗，而不是计划上界。

# 25. Experiment → Paper Mapping

| 已授权后产生的材料 | RQ／claim | 论文位置 | 不能替代 |
|---|---|---|---|
| T_B B1-K/B2 s0/s1、+E s0、P1、B_PLAN | RQ1：效果描述与施加／candidate anchor | Main Results、C1机制 | 普遍否定所有current-graph方法 |
| E1–E6、R质量和机会密度 | 任务可诊断且prior不被当真值 | Setup、Prior Eligibility | Full性能提升 |
| Full/B2三seed | RQ2：prior接口整体增量，primary J | 主表、learning curves | 条件化优于static |
| Full/A_STAT三seed | RQ2：同DK/参数形状，仅prior tensor不同 | Core Ablation、设计空间 | 四视图融合比较 |
| Full/A_CAT三seed | RQ3：pure interaction与flexible fusion | Core Ablation | 单变量或bound独立收益 |
| Full/A_STAT/CAT/A_Q的G_ref与E2/E4/E8 | RQ4：utility-aligned prior reliance | C3分析 | 关系真假概率校准／model-Q自证 |
| Full/A_Q两seed＋clip | RQ4：Q通道与优化机制 | Learning Ablation、Discussion | 所有差异仅来自representation |
| readout decomposition | RQ3/RQ5：prior行通路的输入依赖 | Mechanism Figure | 端到端严格locality或可加贡献比例 |
| NoPrior/Target-swap/GRAPH_REMOTE | RQ5：不完美prior敏感度 | Robustness、Sensitivity | 无机会时的平坦成绩等于稳健 |
| B_PLAN/R/LLM；R*仅分析 | RQ6：搜索、naive R、FM与摊销policy | External Baselines | 直接复现所有原文、理论upper bound |
| Full/B2难度、Full/A_STAT cross-VLM、条件holdout | RQ7／RQ5 | 范围与迁移 | 临时造第三RL task的必要性 |
| D0／T_C历史 | ceiling与task-design边界 | Setup/Discussion/Appendix | 整个prior方法已无效 |
| 可选B1-H | 独立批准的static-without-intervention问题 | Diagnostic/Appendix | 本计划核心C2归因 |

每个MUST都有上述RQ/claim；不为表格对称增加训练。完整已计划且有效的反例在正文或附录有位置，不用Private View掩盖。

# 26. Paper Writing Synchronization

| 时间 | 可以立即写／更新 | 不等待 |
|---|---|---|
| 现在 | 研究内容v5的Problem/Method骨架、Related Work、已有T_B/P1、D0/T_C局限、全部baseline定义 | 不等S1或新训练 |
| S0完成 | +E/A_STAT实现公平、实际layer/readout/Q、四类Full性质、clip字段和B_PLAN定位 | 不等3seed |
| S1完成 | T_P witness、E5/E6、R precision/recall的可识别范围 | 不先填Full赢 |
| S2完成 | 具体task family/split/cache、公开输入、primary J/次终点/机会定义和development校准bank | 不等pilot |
| S3完成 | seed0假设—结果、Full/STAT/CAT/Q图、行分解/校准/clip与S4问题 | 不把探索等同最终test |
| S4冻结 | 确定Method2.1.1或唯一升级；Introduction贡献措辞按已证／待证区分 | 不在不同method混表 |
| S5完成 | 逐seed点和learning区间、主比较草稿 | 不只展示代表seed |
| S6完成 | 主表替换成独立test；dev曲线仍为learning数据 | 不反向调方法 |
| S7完成 | locality/calibration/robustness/难度与Discussion | 不必有所有optional结果 |
| S8 | Abstract/Conclusion与Private Paper View | 不新开Presentation训练 |

占位统一：`[RESULT_PENDING:Sx]`、`[TABLE_PENDING:T#]`、`[FIG_SPEC:F#]`、`[SOURCE_UNVERIFIED:...]`。公式／性质用条件语言，结果句待真实数据再落定。图表美化不能隐藏A_STAT/CAT、强LLM或无效Q的反例。主要结果方向变化要更新相应论述，不能只改数字却保留不再成立的旧结论。

# 27. Stage卡、交付与执行纪律

| Stage | 输入 | 交付 | 允许自动推进的边界 |
|---|---|---|---|
| S0 | 两份冻结文件、指定source/历史ledger | +E/A_STAT/信息/性质/clip/预算审计＋B_PLAN dev | 不自动执行S1或T_B三run，除子授权 |
| S1 | S0审计、最多两类来源 | E1–E6、R\*容量、truth与费用 | 不自动批量scene/cache |
| S2 | ELIGIBLE witness及批准生成规则 | family/split/cache/runtime、primary J/机会定义、校准bank | 不自动开seed0 |
| S3 | 已冻源、S2和seed0授权 | Full/B2/A_STAT/A_CAT/A_Q五方法的开发证据 | 同runE2→E4在授权内；E8/E16释放按卡 |
| S4 | 五方法、E1–E6、分解/校准/clip/opportunity | 单一方法决定、替代矩阵、17＋elastic已用/剩余 | 没用户确认不升级／扩seed |
| S5 | S4冻结配置和17-run清单 | 预定seed及checkpoint；E2保留不重复dev | 不看test，条件项只领elastic |
| S6 | 模型/方法冻结、test-release | 独立test、外部baseline、调用账 | 不调参 |
| S7 | frozen模型、分析sidecars、指定子卡 | robustness/locality/calibration等 | optional单独授权，不超预算 |
| S8 | 全核心证据、负结果和来源 | Final Evidence View、完成声明 | 不新增实验 |

每Stage目录：`runs/final_master/<stage>/<UTC_configsha8>/`。训练目录：`runs/final_master/<method_version>/<task>/<method>/seed_<k>/<attempt>/`。保留raw、checkpoint、input与source identity；.pt不进Git时仍需健康存储上的字节和hash，sidecar不是模型。

状态表只有PLANNED/RUNNING/COMPLETE/STOPPED/DROPPED，证据另记positive/negative/non-diagnostic/engineering/unknown。方法不占优可以COMPLETE，缺失基线不能“PASS_WITH_NOTES”冒充完整。所有run需明确plan_id与attempt_id，不把恢复/中间checkpoint当新seed。

不git clean、不reset --hard、不删历史、不强推、不创建/合并PR（除明确批准）。不要复制environ.txt、SSH key或token进入研究包；用户名/路径来源记录可去敏，但不能伪造。

# 28. Final Completion Checklist与冻结路线

核心证据闭环要求：
- T_B的B1-K/B2第二seed、B1-K+E s0、现有B0/P1与外部合同参照有真实身份；
- T_P的E1–E6、训练前primary J及机会密度定义成立或其限制被明确处理；
- Full/B2/A_STAT/A_CAT各三seed、A_Q两seed按最终同profile规则完成；
- 同64 development物理分支的跨接口／checkpoint utility-aligned reliance及clip审计完成，测不到的项目明确影响claim而不是填0；
- 权重冻结后的独立test、外部planner／LLM、NoPrior/Target-swap、GRAPH_REMOTE前向和readout decomposition有实际证据；
- SHOULD/CONDITIONAL/OPTIONAL按来源和授权执行或写NOT_RUN/N_A，不恢复大矩阵；
- 总attempt不超过基础17＋唯一elastic4；必要control超限由人工决策，不自动突破；
- 论文已有部分持续写，主claim在真实结果后定向，不要求Full在所有指标获胜。

```text
NOW：Research Content v5 + Experimental Plan v3冻结，authorized=false
  ↓ 获准S0
科学代码／+E／A_STAT／Full性质／clip／B_PLAN
  ├─ 获子授权的T_B三条新增run可独立排队
  ↓
S1 Prior Existence E1–E6 + R* capacity
  ├─ 非诊断：原限定内一次有据修订／研究决策，不开放式扩task
  ↓
S2 冻结T_P family/split/cache/primary J/opportunity/calibration bank
  ↓
S3 seed0 Full / B2 / A_STAT / A_CAT / A_Q
  ↓
S4 单一方法门：2.1.1，或一个2.2-B/2.2-C
  └─ 先替换未启动配置；净增、重跑共用elastic4；总cap21
  ↓
S5 扩到基础17配置，保留E2 checkpoint，省重复E2 dev
  ↓
S6 冻结权重 → 独立test + B_PLAN/R + LLM
  ↓
S7 frozen robustness / GRAPH_REMOTE / readout decomposition
   / 跨接口与训练轨迹calibration / 既定扩展
  ↓
S8 Final Evidence View → 研究完成后的私人展示优化
```

此后普通新实验结果只更新卡的PLANNED/RUNNING/COMPLETE/STOPPED/DROPPED状态、结果占位和论文证据，不自动触发新Master。只有S4正式批准Method 2.2-B或2.2-C时才升方法版本；必要科学工作超cap时暂停请求人工决定，不给新编号继续派发。

**Research Content v5 and Experimental Plan v3 are now the sole authoritative project specifications unless S4 explicitly triggers a method-version change.**

# 附录 A — 最终Cross-Document Audit

| 检查 | Research Content v5 | 本计划v3 | 结论 |
|---|---|---|---|
| 核心claim→证据 | §2/5/20/23 | §2/25/28 | 每项核心RQ都有匹配训练或分析 |
| MUST→RQ | §19 | §3/7/25 | 无新增无主张重实验 |
| 方法名 | §19 | §3 | Full/B2/A_STAT/A_CAT/A_Q/+E一致 |
| A_STAT公式 | §19.3 | §3.2 | 完全相同；只有DP→S_R，保留DK/参数形状 |
| B1-K+E | §19.1 | §3.1 | PRE/ADD/DEL/UNKNOWN/guards/binding；s0 MUST |
| 17＋4＝21 | 控制/§22 | 控制/§7/§24 | 单一pool，无C-FACT或升级独立bucket |
| seed | §19.5 | §7/§11 | T_B新增3；T_P 3/3/3/3/2 |
| primary endpoint | §6 | §6.6/§20 | J在S3前冻结，success次终点 |
| GRAPH_REMOTE | §13/§16/§25 | §14/§15 | 默认图远端，不承诺strict invariance |
| B1-H | §19.2 | §3.3/§7.2 | OPTIONAL / elastic-pool，不入基础 |
| Upgrade B/C | §22 | S4 | 仅S4、一方向、替换优先、同一21cap |
| 当前Method | 控制 | 控制 | 2.1.1，未执行升级 |
| readout | §13 | §15.5 | goal/global行依赖，不伪造严格局部比例 |
| C3 bank和clip | §16/§17 | §16 | 64分支共享、11模型、三checkpoint、pre/post clipping |
| 执行/写作 | §24/附录D | §26/Runbook | PLAN_ONLY/authorized=false，RESULT_PENDING同步 |

本次自动审计检查静态文档、矩阵、算术与公式文本，不执行生产测试。结果和输入文件hash随交付归档。

# 附录 B — 来源、数学限定与本轮修订范围

来源[S1–S7]、历史源码[G1–G4]、结果[E1–E6]及文献[W1–W27]使用Research Content v5同一索引。v4/v2原件及最终审计保留，当前未来执行只以本两文件为准。

修改范围限定于X1、X2、D2、D3、D4、D5/D6、A1、A2/A4和其低成本写作／计数同步，不新建Master研究方向或复杂baseline。

明确的数学／算术处理：
1. readout行置零测量非线性通路依赖，不证明goal行严格consequence-local，不给可加来源百分比；
2. Full empty-patch保证该候选直接Δ为零，不保证其softmax概率不受其他候选改变；
3. Full/CAT/STAT成绩排序是机制假说线索，不是对locality／static原因的逻辑证明；
4. 3seed×32case不伪装成96个独立训练重复，不给未经统计模型支持的精确检测效应量；
5. S5实际九个扩展seed省E2：144episode；最终默认含难度2390而非未去重2534；
6. calibration有两个实测候选不等于所有候选均有Q_ref，支持不足时条件化或NA；
7. 528校准比较单元不必等于528单次forward，保守按最多1056调用并去重。

这些是必要解释和计数边界，不额外增加训练、物理分支、task或seed。

# 附录 C — Internal Runbook（不写入论文研究主体）

## C.1 用户、源码和存储

唯一动态账号`xushijie2@gpu03`；目标Python为`/home/xushijie2/envs/lerobotpi0-xfs/bin/python`。仓库或独立worktree和所有active runs/logs/TMPDIR/cache在`/home/xushijie2`的已核验XFS。启动先检查真实用户、hostname、sys.executable、cp_disr导入、realpath和mount，不可fallback旧账号／旧venv。

历史旧账号仅按已有合法授权只读归档，不能在那里执行新Python或写新报告。R2旧账号偏差保留，不改写成新账号历史，也不因用户名自动重训。

## C.2 阶段授权与并行

用户明确指定stage/subcard及资源后才可改变authorized。S0只读审计不能自动开始T_B训练，S2存在不代表S3授权。默认2worker，隔离GPU、CPU线程、输出和RNG；S5资源足够才申请更高并行。

S3排程Full/B2、A_STAT/A_CAT、A_Q不创建第二派发器。S4前不扩主要seed；S6前不运行test。方法/预算/数据更改全部通过已定义决策门。

## C.3 Phase-aware进度与持久化

区分reset、collection、PPO epoch/minibatch、optimizer、fsync/publish和eval case。PPO期间没有新transition不是停滞证明。记录最后完成事件与wall/sim时间，不单纯以stdout静默决定重新启动。

使用已验证的不可变generation、全文件hash、COMPLETE、LATEST最后更新和上一代保留；N/T/模型/Adam/RNG来自同一代。checkpoint记录pre/post update与实际N/U/fragment，不能以文件名当真实性。I/O、非法mask、安全越界、非有限或native无进展时保全并停，禁止自动循环重训。

## C.4 Provider历史身份与调用

历史prior绑定为qwen3.8-max-0902、cn-beijing、`https://dashscope.aliyuncs.com/api/v1`、DashScope1.27.6；仅作为历史输入身份，不保证服务未来可用。获准调用前验证权限与model/endpoint，不自动改模型后沿旧cache标签。最大一次合法格式／传输重试，禁止语义重试到非空。

不读取旧environ.txt或私钥内容，不将authorization header、token、API key复制进日志/Git。LLM等待不推进simulation时，wall/API成本独立报告。

## C.5 Raw Archive / Paper View / Git

训练目录`runs/final_master/<method_version>/<task>/<method>/seed_<k>/<attempt>/`；阶段目录`runs/final_master/<stage>/<UTC_configsha8>/`。每项记录master/run/attempt_id、源/输入/model hash、实际成本和证据类别。

PLANNED/RUNNING/COMPLETE/STOPPED/DROPPED只表示执行状态；positive/negative/non-diagnostic/engineering/unknown另列。没有结果不填0；负研究结果可COMPLETE。

不git clean、不reset --hard、不删历史、不强推。新source在正式采样前冻结，运行中不热改；不在另一活跃训练worktree切分支或改共享editable导入。未经明确授权不建/合PR。

.pt不入Git时仍须位于健康存储并有hash与manifest；sidecar不是模型字节。源码与去敏报告可按stage授权提交推送。Private Paper View可选清楚图例与布局，但预定seed和有效反例必须可追溯，不篡改Raw。

## C.6 当前冻结状态

```yaml
master_id: CP-DISR-FINAL-EXEC-3.0
status: PLAN_ONLY
freeze_status: FROZEN_PLAN
authorized: false
stages_S0_to_S8: PLANNED
base_new_rl_runs: 17
elastic_attempts_total: 4
elastic_attempts_allocated: 0
first_cycle_attempt_hard_cap: 21
default_method_version: "2.1.1"
method_upgrade_triggered: false
```

本次只生成与核查文档，没有授权或执行S0。除S4明确触发方法版本变化外，Research Content v5与Experimental Plan v3是唯一项目权威规范。
