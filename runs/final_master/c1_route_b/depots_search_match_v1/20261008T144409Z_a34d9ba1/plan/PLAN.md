---
title: "CP-DISR C1：Claude审查复核与同搜索评分器比较实验卡 v2"
date: "2026-10-08"
status: "PLAN_ONLY"
execution_authorized: false
card: "C1-DEPOTS-SEARCH-MATCH-V1"
repository: "rollinpioneer/graph"
base_branch: "codex/cp-disr-c1-public-depots-rel-v1"
base_commit: "e10e80f56236cb7c43cd4113c8bc345d932f629f"
proposed_branch: "codex/cp-disr-c1-depots-search-match-v1"
storage_owner: "xushijie3"
new_training_runs: 0
new_gate_labels: false
new_test_generation: false
scope: "旧结果口径补充、冻结评分器的同搜索比较、原始问题拓扑统计、完整报告"
---

# 0. 决定与本卡要回答的问题

**采纳Claude的主要优先级调整：暂停把学习门控作为下一项默认训练；先将现有冻结评分器用于同一个完整搜索框架。**

这里的“完整搜索框架”指保留待探索的其他分支，而非只沿当前选中的一条执行路线走下去；本卡仍有节点、时间和内存限制，不承诺完备或最优。

锁定方向不变：**动作规则已知，训练目标结构受限，测试更复杂的目标组合。**改变的是评分器的使用方式，不引入部分观测、合同学习、机器人恢复或新领域。

本卡核心问题：

> 在同一个状态生成器、同一个搜索器和相同资源上限下，DENSE-G学到的状态评分，能否比WL-GOOSE和经典启发式更有效地引导搜索？

同时回答一个更有限的问题：**当前公开题的差距，有多少能通过改变执行框架缩小？**不能预先断言差距主要来自无回溯，也不能保证接入搜索后就能赢。

本卡明确不做：
- 新的VOC门控训练、GateTrain标签收集；
- 新REL规则、注意力/容量重训、第三领域；
- 更大拓扑训练、更多seed或自动调用精确规划器作策略；
- 根据当前题的成绩选择搜索规则、检查点或启发式混合权重。

**本卡新增训练为0。**结束后必须给出结果与下一方法决定，不能自动沿本文件的后续候选启动新训练。

# 1. 来源与本轮复核范围

## 1.1 来源分级

- **[A] 用户提供的Claude审查**：`CP_DISR_C1_Depots结果与VOC方案审查_20261008.md`，及`depots_paired.py`。审查中的新配对数字按其报告引用；本轮没有重新执行该脚本或全量复算CSV。
- **[R] 本轮直接读取的固定仓库代码**：`pddl/stages.py`、`pddl/model.py`、`scripts/c1_public_depots.py`、既有运行的`suite_runtime.json`。用于核对执行接口、旧参考长度规则、作者软件版本与资产路径。
- **[W] 本轮核验的作者代码/官方资料**：固定版本GOOSE的`plan.py`与`classic.toml`；Fast Downward搜索文档；计算选择的原始论文页。
- **[D] 本文判断与新设计**：所有修正、预算、搜索语义、停止规则均为拟议方案，尚未运行。

已有的Depots结果仍以固定提交下的`final_summary.md`、`final_summary_numbers.md`、`claim_boundary.md`、`receipts/final_receipt.json`为入口。本轮不是再次认证服务器运行完整性的“PASS”。

## 1.2 固定资产

```text
repository: rollinpioneer/graph
base branch: codex/cp-disr-c1-public-depots-rel-v1
base commit: e10e80f56236cb7c43cd4113c8bc345d932f629f

old result root:
runs/final_master/c1_route_b/public_depots_rel_v1/
20261008T063544Z_9898106e/

recorded repo path: /home/xushijie3/work/graph_cp_disr
recorded python: /home/xushijie3/envs/cpdisr/bin/python
```

以上绝对路径来自旧`suite_runtime.json`，Agent仍应验证实际目录，不对历史文件做全局字符串替换。

旧运行固定的外部版本：

```text
GOOSE: 04dbe8f27a8ec027626e6c6b5d6e7610b81f9d57
GOOSE Scorpion: 4c16ce05b31d68b67cb126a3a7c31e9123ba50c6
WLPlan: 9d9fc99165135bfe8e2817145888ede162e4a97a
Fast Downward: 9b81c7e422fdf7be9f73b96e9a7a969c483dd5d4
pddl-generators: d5c22c9ab21ecaf90db82daf2a0537973c661009
downward-benchmarks: e21d49c2cb61d147a46c5966f2581bf6fd422b9f
```

# 2. Claude审查：采纳什么，修正什么

## 2.1 采纳的主要意见

| 审查意见 | 本文决定 | 理由及范围 |
|---|---|---|
| 先做冻结评分器+GBFS，不先训练VOC | 采纳 | 不新增训练，可在相同搜索框架下比较已学评分；不是保证能解决公开大题 |
| 门控必须同时对比一步，而不能只对固定两步 | 采纳 | DENSE门控对一步的共同成功成本为29更短/27更长；对固定两步较强并不足以证明还有很大可学收益 |
| 局部最优距离差与实际门控收益可能不一致 | 采纳 | C3历史、剩余预算和后续策略都会影响回合结果 |
| IPC同时改变目标、对象规模、地点/车辆配置 | 采纳，需逐题统计 | 这是一组混合变化；相关性不是拓扑导致失败的因果证明 |
| 旧神经执行的参考解长步数上限与外部规划器不同 | 采纳 | 新同搜索比较不再使用由参考解长推导的深度上限 |
| 不继续给REL加领域规则救主张 | 采纳 | 当前负结果针对此配方；不扩大成所有关系方法都无效 |
| 预算应报告曲线，不能只挑一个胜出的点 | 采纳 | 使用同一次搜索的节点里程碑，并列时间、评价数和内存 |

## 2.2 必须修正的九处判断

### C1. “逐题三选一的最好结果”不是逐动作门控的上界

Claude报告DENSE在三种完整执行方式中逐题选最好，计罚比为1.074，原门控为1.119。[A §0,附表]

这只界定**每题开始时从这三种固定执行方式选一个**的事后上界。逐动作切换会形成三者都没走过的新轨迹，可以更好也可以更差。因此：

- 可用于说明已有三路执行的互补程度；
- 不能据此证明逐动作VOC最多改善0.045；
- 也不能据此推导“必须改善0.02”这个统计门槛。

本卡不再生成新的门控标签来穷尽上界；直接开展搜索比较。

### C2. 旧IPC参考长度可以补充更新，但不能改写原实验预算和结果

据审查，p06原参考长度192，而已有有效计划长度58；因此旧参考并非全系统最好已知。[A F1]

可以另建：`R_best_observed = min(所有已验证有效计划长度)`，并保留`R_proven_optimal`字段。**旧`R_registered`与`cap_registered=2R_registered+4`不可覆盖。**

例如p06：旧上限388，新参考若为58，对应假设上限120。不能不重跑就声称旧策略原本是在120步预算下运行。超过120步的旧成功轨迹，只能写“超过新假设上限”；未执行过的结果不称为实测。

本卡只补充长度归一化表与上限敏感性说明；不重算一个伪装成新运行的成功率。

### C3. 同一个搜索器只固定执行因素，不会让两种学习方法只剩一个内部差异

DENSE与WL-GOOSE的输入表示、监督构造、容量、优化和选模均不同。共同使用Train96也不等于训练状态、标签粒度完全一致。

本卡的准确表述是：**在共同搜索器中比较完整的评分器配方。**不是“证明注意力比WL好”的单模块消融。

### C4. WL-GOOSE当前固定配方是排序学习，不是简单的绝对代价回归

本轮读到的`classic.toml`明确为`optimisation="rank-svm"`、`state_representation="downward"`；`plan.py`对应`eager_greedy([wlgoose(...)])`。[W1,W2]

因此不能把对比解释成“我们的排序网络，对它的h*回归器”。两者都会面临训练比较关系如何迁移到搜索状态的问题。

### C5. 评分V用于搜索没有额外训练，但改变了评分的使用分布

原动作分数为`V(s)-V(s_a)`，主要比较同一父状态的兄弟后继。GBFS会在开放表上比较来自不同父状态、不同深度的V。

接入是合法的，但跨分支排名是否有效正是待测问题。失败不能唯一归因于拓扑、执行器或V“未校准”；成功也不证明V是准确剩余距离。

### C6. 同节点预算不是同计算预算，“零训练”也不是“只用CPU”

神经V需要GPU推理或较慢的CPU推理；每个展开生成的新状态数也不同。必须并列：展开数、生成数、唯一h评价数、逻辑图编码数、实际前向调用数、时间、内存。

相同展开数下更好，只能说明该搜索预算下的引导质量，不等于更快或更省计算。

### C7. 原门控作用不应被直接定性为无效；也不能由局部计数判定新标签必然无信息

DENSE已有门控确实在本套数据上提高了完成率、减少了二阶叶评价。29/27不证明两者等价，6箱上的修复/损害计数也不能直接代表8箱回合收益。

本轮选择暂停VOC是**投入优先级判断**，不是根据当前汇总表已证明其不可学。

### C8. 回合收益标签需要完整历史和剩余预算；局部缓存键不够

如果未来用`Δ_roll`，它取决于后续执行，因此缓存键至少包括：问题、状态、完整已访问集合的稳定摘要、剩余动作预算、后续策略版本、评分器权重、随机性设置。

`V∩succ(s)`与`V∩succ²(s)`足以影响当前有限选择，却不足以决定更远的策略续接。HandGate续接产生的是**相对HandGate的单次干预收益**，不等于部署学习门控后整段轨迹的真实收益。

### C9. 遮掉已训练注意力和拓扑相关性都不是干净的因果控制

测试时将注意力输出置0可作敏感性探针，不能代替同预算重新训练的容量匹配模型；它引入了测试时分布变化。

IPC失败与更多卡车/地点相关，只能支持后续问题选择，不能证明“扩大拓扑训练一定修复”。同样，模型+搜索没赢不意味着应自动回到VOC或否定整个C1方向。

# 3. 已有证据如何影响本卡设计

以下数字来自审查及已完成结果，不是新实验输出。[A,R5]

| 对比 | 关键结果 | 对本卡的意义 |
|---|---|---|
| DENSE门控 vs 一步，joint | 127 vs125成功；共同成功成本29更短/27更长 | 不再把小门控收益当作默认的大方法空间 |
| DENSE门控 vs固定两步，joint | 39更短/10更长，叶状态约少78% | 保留原门控作为既有执行参照 |
| REL一步 vs DENSE，不运输 | 19更短/5更长 | 关系配方可能有任务条件边界，但该层是事后分析 |
| REL一步 vs DENSE，运输 | 10更短/22更长，56 vs61成功 | 不能靠纯堆叠有利层替代整个主比较 |
| DENSE门控 vs WL-GOOSE，IPC | 11 vs20成功；共同成功1更短/7更长 | 公开系统差距值得正面检验 |
| DENSE/REL/MG与WL的执行框架 | 神经方沿单条路线，WL用开放表搜索 | 比较同一执行框架，比再训练门控更直接 |

旧joint共128题，所有初始参考长度已有最优证明；“8箱没有标签”应解释为**没有完整逐状态最优动作/拆除标签**，不是没有初始最优长度。[R5]

# 4. 实验假说与解释边界

本卡只测试三个有限假说：

**H1：执行方式贡献。**冻结评分器接入保留其他分支的搜索后，其完成能力和路径质量是否改善？相对旧执行的变化是整套执行协议变化，包含移除参考长度步数上限，不能唯一归为“会回溯”。

**H2：搜索引导质量。**共同搜索与节点上限下，DENSE评分是否能比WL、h_add及目标计数更快找到有效且较短的路径？

**H3：质量—成本分离。**神经评分是否有节点预算上的优势，但在相同时间预算下受制于编码开销？

本卡不是新搜索算法提案。接入成熟搜索可以证明既有评分器的使用价值，但不能仅因“加了搜索后成功更多”就认定完成了CCF-A级方法贡献。

# 5. 本卡完整范围与默认预算

## 5.1 运行条件

六个评分器，**同一搜索器内只替换评分回调**：

| ID | 评分 | 权重/训练 | 角色 |
|---|---|---|---|
| V_DENSE | 冻结DENSE-G的状态V | 旧选中更新4100 | 主候选 |
| V_MG | 冻结MG-G状态V | 旧选中更新820 | 原学习基座 |
| V_REL | 冻结REL-G状态V | 旧选中更新3280 | 历史结构候选，不重新选模 |
| H_WL | 已拟合WL-GOOSE评分 | typed/IPC各复用对应模型 | 主外部学习对照 |
| H_ADD | 删除放松的h_add | 不训练 | 有信息的经典启发式 |
| H_COUNT | 未满足最终目标数量 | 不训练 | 基础参照，不代表最强经典规划 |

本卡不默认追加h_FF，不增加REL变体，不将原LAMA改造成同搜索条件。

## 5.2 题集和数量

- Struct32：目标组合变化、4/5箱；
- Joint128：6/8箱×1/2堆×目标高2/4，每格运输/不运输各8题；
- IPC22：公开原题，全部保留。

共182题。现有材料已用于本设计，**本卡全部属于开发/比较，不称独立确认**。

六种评分×182题＝**1,092次搜索运行**。

每次运行只启动一次，节点里程碑为`1,000 / 10,000 / 100,000`，得到至多3,276条预算快照，不将三档重新独立跑三遍。先找到有效解即终止；不是随预算增加自动继续优化首解。

额外必要夹具仅使用预选的6道训练小题与少量合成图；不计入论文成绩。原生WL-GOOSE最多追加6道训练小题的接口对照运行，不扩张为第二张全任务系统矩阵。

## 5.3 资源

- 每题总壁钟硬上限：300秒，包括该题解析、接地/翻译、图模板构造、状态转换、搜索和评分；
- 每题搜索有效展开上限：100,000；
- Host内存上限：8 GiB/题级worker；神经GPU预算在夹具阶段按可用设备统一固定并登记；
- 已加载权重作为可复用常驻服务，加载/预热单列，不在每个节点重复加载；
- 同一GPU默认只允许一个神经搜索任务活跃；CPU worker默认最多4个，实际资源可在开跑前整体调低；
- 六种条件搜索语义一致，GPU批大小只改变计算分块，不改变下一次pop时机；
- 不承诺“几分钟结束”。1,092×300秒是91小时题级串行最坏时限总和；实际常因提前解出而减少，可利用互不冲突设备并行。

节点与时间是同时约束，不是忽略超时只看展开数。未达到下一节点档的超时记录为该预算联合约束下未解，并标明`TIMEOUT_BEFORE_NODE_LIMIT`。

# 6. 搜索接口：真正共享实现，不假装完全复现作者搜索器

## 6.1 推荐的默认实现

为避免本卡被C++/Python跨语言插件工程吞没，主比较采用一个**明确记录语义的共享STRIPS图搜索器**，所有六种h均在其中运行。优先复用项目已有`StripsTask`、合法动作与状态表示。

WL评分必须调用作者模型/特征实现，通过适配层读取旧`.model`；**不得手写一个“类似WL”的模型替代**。h_add使用同一STRIPS动作与目标，计算实现应与小训练题上的已有参考核对。

为什么不要求逐题复现原生GOOSE展开数：旧GOOSE通过Downward/Scorpion执行，翻译、动作排序、状态表示及数值处理可能不同。既然主比较已把全部h放进同一新搜索器，就不应伪称和原生展开轨迹完全相同。

报告分开：

1. **共同搜索评分表**：H_WL注明“作者模型＋共同搜索适配”，不把旧20/22直接填进新表；
2. **历史原生系统表**：旧WL/LAMA完整系统结果保留，说明协议不同，不能作为等节点结构比较。

若Agent能直接把全部评分器接入同一个原生搜索器，也可采用原生路径，但必须在读取本卡任务成绩前锁定，并对全部六行一致。不得中途因为某个成绩不好更换搜索器。

WL适配无法读取真实评分或验证接口时，报告`H_WL_TECHNICAL_INCOMPLETE`；其他条件可完成，但最终必须明确外部主比较未完成。不能靠旧表或简化替代补成“完整公平比较”。

## 6.2 统一的搜索语义

采用eager型GBFS：新状态进入开放表前计算h，优先选择h较小的节点。

共同规则：
- 使用完整任务状态作重复检测；同一问题内静态事实固定，不省略会改变动作合法性的事实；
- 一个OPEN优先队列，键为`(h_value, insertion_id)`，FIFO处理同分；
- 后继按固定的原始动作ID排序；不使用学习策略偏好、helpful action优先队列或G1；
- 根节点先检查真实目标；后继生成时检查真实目标，若同批多个目标后继则按动作序列取第一个并返回；
- 对非终止后继先做重复过滤、再按固定顺序批量评分、全部完成后插入OPEN，再pop下一个节点；
- CLOSED节点不重开；尚在OPEN的状态若发现更短路径可更新g和不可变路径记录，使用版本号跳过过期堆项；
- 禁止Beam、限制只看前k个候选、跨多个OPEN节点批量展开或自动多启发式混合；
- 不再使用C3的“当前真实路线的已访问集合”限制搜索。完整状态重复检测仍保留；一个分支没出口时继续处理OPEN中的其他分支，不立即判整题失败；
- 不使用`2L_ref+4`作为路径深度限制，也不以已知最优路径长决定停止；
- 模型有限的负值是合法排序值，不裁剪成0；非有限神经输出为技术错误，不能解释成不可达；
- h_add为无穷时，本卡统一作为排序末尾，不独享死路剪枝，避免把额外剪枝能力混入“评分差异”；
- 返回首个有效计划，回放验证。首解不等于最优解。

本实现的“生成时终点检查”等细节与Fast Downward默认行为可能不同，必须写为**共同GBFS适配协议**，不是未经核验的作者原生配置。所有条件享有相同差异。

## 6.3 最小伪代码

```python
if task.is_goal(initial):
    return solved([])
OPEN = priority_queue()
CLOSED = set()
best_open_g = {initial: 0}
path_record = immutable_root(initial)
OPEN.push(h(initial), serial(), initial, 0, path_record)

while OPEN and within_wall_memory_budget():
    h_s, _, s, g, record = OPEN.pop()
    if stale(record) or s in CLOSED:
        continue
    if expanded >= 100_000:
        return NODE_LIMIT
    CLOSED.add(s)
    expanded += 1
    pending = []
    for a in sorted(task.legal_actions(s), key=canonical_action_id):
        t = task.apply(s, a)
        generated += 1
        if task.is_goal(t):
            return validate_and_return(record.append(a, t))
        if t in CLOSED:
            continue
        if unseen(t) or g + cost(a) < best_open_g[t]:
            pending.append_or_update(t, g + cost(a), record.append(a, t))
    # 同一个父节点所有待评后继完成后，才能进行下一次pop。
    scores = evaluator.evaluate_uncached(pending.states)
    for item, ht in zip(pending, scores):
        register_best_open(item)
        OPEN.push(ht, serial(), item.state, item.g, item.record)
    write_completed_expansion_milestones_if_due()

return OPEN_EXHAUSTED or TIMEOUT or MEMORY_LIMIT
```

父记录不可变，避免更新某个OPEN父节点时，已生成后代的计划链、g与实际动作数不一致。小图夹具覆盖这一点。

## 6.4 评分器接口与批处理

```python
class StateEvaluator:
    def prepare(self, task, model_identity): ...
    def evaluate(self, states: list[int]) -> list[float]: ...
    def metrics(self) -> dict: ...
```

- 神经方调用已存在的`PddlSerialModel.state_values(template, task, states)`；只要状态V，不调用一个会额外生成全部动作后继的策略包装；
- 同一问题固定目标，因此直接以`V(s)`从小到大排序。不使用局部`softmax`概率作为跨父节点启发式；
- 神经V不需要是距离或下界，但必须确定、有限且方向正确；
- 禁止按当前batch、OPEN或兄弟集合单独归一化V，这会改变跨批次比较；
- GOOSE作者实现若有缩放、离散化或截断，记录变换；新共同搜索保留实际作者预测语义，不能偷偷重新调尺度或取反；
- 固定精度，关闭dropout，权重只读。每次任务后清理该题的状态值缓存，不跨方法共享缓存结果；
- 仅同一父节点的后继批量推理；大批次可分块，顺序不变。额外分块不算新的搜索方法。

# 7. 预处理与外部模型转换的最低正确性检查

仅做会影响比较正确性的检查，不重做全历史审计。

1. 固定6道Train96小题，覆盖typed/untyped、多个可行动作、运输和不运输；
2. 对对应状态核对合法动作、状态转移、目标真值；
3. 确认神经`state_values`与原一步/前瞻中同状态V一致；
4. 读取作者WL模型，验证当前状态转成作者表示后预测一致。仅初始状态一致不足，需要至少数个非初始状态；
5. h_add小例子检查：已满足目标贡献0、不可达标记、前提求和与多行动作传播；
6. 合成图检查：保留另一分支能解、重复状态不重复展开、OPEN中更短路径更新、CLOSED不重开、目标检测、同分顺序；
7. 预算快照检查：低档未解不删除OPEN，继续同一运行；首解时间早于超时时有效；未生成/未验证的计划不算成功；
8. 结果计划用项目语义完整回放，必要时调用已有验证器。验证不参与挑动作。

由于旧WL使用Downward状态表示，适配层可能需要由事实状态恢复SAS变量、静态事实与互斥组。必须使用作者转换或经小例子核验的转换；不能忽略静态事实、把遗漏变量随便填0，或把对象名称改成训练外常量后仍称等价。

同一搜索器不代表三种神经模型与WL具有相同信息编码；这些仍是被比较的评分器差异。

# 8. 旧IPC参考长度补充：只做派生报告

输出`reference_length_sensitivity.csv`，每题含：

```text
case_id
registered_reference_length / registered_reference_kind
registered_step_cap
proven_optimal_length（若无则null）
best_observed_valid_length / source_method / source_plan_hash
counterfactual_step_cap_from_best
old_plan_exceeds_counterfactual_cap
old_success_preserved
```

规则：
- 先验证用于更新参考的计划，不能只取日志里一个整数；
- 有最优证明时，任何更短“有效计划”都是需要查明的矛盾，不直接取更短值覆盖最优；
- 历史成功、超时、DEADLINE及原指标不变；
- 新附表可报告成功计划`length / best_observed_valid_length`；它是相对观察到的最好计划，不是最优比；
- 若附加失败计罚敏感性表，分子仍用旧`cap_registered+1`，明确是固定旧惩罚下更换分母，不是新执行预算；
- 本卡新搜索不依赖任何参考长度停止。IPC主要以覆盖率和共同成功原始计划长度比较；最优比仅用于已证明的7题及已证最优的struct/joint。

参照其他方法求出的较短计划是合理的事后归一化，但不能把用全体结果选出的分母说成事前固定主指标。

# 9. 问题范围：目标、数量和世界拓扑分开记录

不增加新的题目，只读取旧manifest和实际PDDL。

每题记录：

```text
n_crates, n_goal_towers, max_goal_height, needs_transport
n_depots, n_distributors, n_places, n_trucks, n_hoists, n_pallets
n_ground_actions, n_dynamic_atoms, n_template_nodes, n_edges
reference_kind, destruction_label(TRUE/FALSE/UNKNOWN)
```

审查说Train96和Joint128的拓扑都是1 depot、1 distributor、1 truck、2 hoist；本卡从实际问题文件再次枚举，不能从文件名推断IPC的数量。[A §8]

IPC原题路径在旧生成代码中指向固定`downward-benchmarks`目录。本地副本缺失时按旧记录固定版本恢复，核对SHA，不换成网上同名新版题。

拓扑与箱数、目标高度、初始布局可能共同变化。报告分层用于发现限制；不能用22题上的相关性宣布已识别唯一瓶颈。

特别注明：目标塔高2/3是训练范围，4为训练外；目标高度不等于初始摆放高度。

# 10. 评价：先看覆盖，再看成本和计算，不只看成功子集

## 10.1 主比较

主要条件：`V_DENSE`对`H_WL`；同时保留`H_ADD`与`H_COUNT`标尺。V_MG/V_REL为辅助评分器比较。

主节点档：10,000；1,000和100,000为预算曲线。每档均受300秒硬时限和内存限制。

每个数据集单独报告：

1. 覆盖率、独有成功和共同失败；
2. 共同成功题的计划长度差、较短/较长/相同题数、差值总和与中位数；
3. 全体题的失败原因；
4. 首解所需展开数、唯一h评价数、图编码数、批调用数；
5. 解析/接地/转换/搜索/神经推理耗时，Host/GPU峰值；
6. 计划验证率；
7. 按箱数、目标堆数、高度、运输与拓扑分层。

`同节点预算胜出`与`同时间预算胜出`分别写。模型在大图超时、无法达到节点上限时，不能声称已经完成了该节点上限的纯引导质量比较。

## 10.2 失败处理

不从计划成本均值中“消失”：主表始终把覆盖率与共同成功成本放在一起。

辅助质量分可用：成功且有已证最优长度时`L*/L`，失败为0；IPC非最优参考题只提供独立的“最好已知参考质量分”，不叫最优性。

`OPEN_EXHAUSTED`是该表示与搜索耗尽；`TIMEOUT`、`NODE_LIMIT`、`MEMORY_LIMIT`是资源失败；`MODEL_NONFINITE`、`INVALID_PLAN`、适配失败是技术问题。不能把预算超限全部标成“模型没学会”。

## 10.3 统计

对固定模型做配对题目分析，允许报告不一致计数、符号检验及效果大小；按题/生成问题类聚合，不把每个状态当独立样本。

事后层的p值标为探索性；不把p>0.05写为等价，不用“净优势≤0”自动判死研究方向，也不从先前0.045的事后oracle差值推导新功效阈值。

单训练种子仍限制训练稳定性的结论，但不是拒绝一切方法选择的理由。本卡不新增训练seed。

# 11. 时间、节点快照和缓存的执行约定

- 1,000和10,000节点只记录快照，不中止搜索、不重启计时、不清空OPEN；
- 如果已在低档找到首解，后续档记同一个首解记录，不伪造额外搜索；
- 批评分过程中超时，未完成结果不得用于继续决策；记录已开始与已完成h评价数；
- 单父节点可能产生大量子状态，故同时记录generated及evaluated，必要时因内存而失败，不静默截断候选；
- 时间硬截止由题级监督进程执行，常驻模型服务不被全局清杀；仅回收该题进程/请求，绝不影响他人作业；
- 权重加载计时单列，题级编译/模板与状态转换纳入题级总耗时。不得为了神经方把昂贵预处理移出主时间而把WL的预处理算进去；
- 缓存仅是同一纯状态函数的memoization：键为task/goal版本、权重SHA、state完整身份、数值配置；不存储规划器标签；
- 不因GPU空闲与否临时修改搜索深度、阈值或checkpoint。运行硬件变化单列；多卡并行只并行独立题/方法，不改变单次搜索顺序。

# 12. 执行顺序与自动接续

后续明确授权执行本卡后，按以下顺序自动接续至完整报告，不因低分停下等待批准：

```text
资产与路径核对
    ↓
旧IPC参考长度派生表 + PDDL拓扑表（不修改历史结果）
    ↓
共同搜索器、六个h适配、训练小题夹具
    ↓
锁定搜索语义、数值处理、预算、模型身份
    ↓
六个h在Struct32上运行
    ↓
六个h在Joint128上运行
    ↓
六个h在IPC22上运行
    ↓
校验计划、生成三档节点与时间表
    ↓
写总结、结论边界、方法建议、回执和产物清单
    ↓
检查轻量产物并提交结果，停止
```

接口错误修复可在成绩前进行；成绩出现后语义修复只作明确新版本，不挑保留有利旧结果。某个h技术失败，可让独立条件完成；外部关键比较缺失必须显式标识，不能把卡写成全部验证通过。

**范围错误或数据泄漏：**停止放行后续任务、保留记录，并按用户权限请求决定。不得擅自终止正在运行的训练。题级硬时限是本卡执行授权的一部分，不等于可任意杀死现有服务器任务。

# 13. Agent需要改哪些文件

建议新增，不覆盖已完成实验实现：

```text
src/cp_disr/pddl/search_match/
    engine.py              # 单一OPEN搜索与统一重复策略
    evaluators.py          # neural / WL作者模型 / h_add / count适配
    budget.py              # 时间、节点、内存与快照
    report.py              # 所有表格由日志生成

scripts/c1_depots_search_match.py
configs/c1_depots_search_match_v1.yaml
tests/test_depots_search_match.py
```

复用：
- `src/cp_disr/pddl/task.py`：STRIPS状态、接地、动作和回放；
- `src/cp_disr/pddl/model.py`：`state_values`，不得重新初始化替代旧checkpoint；
- 旧运行`runs/{mg,dense,rel}/selection.json`与`training_accounting.json`：选中权重路径和SHA；
- 旧运行`goose/train_typed/wl_goose.model`、`goose/train_ipc/wl_goose.model`：同编码对应模型；
- 旧`data/manifest.json`或实际manifest路径：以目录发现结果为准，不能猜一个不存在的路径后静默生成新题。

不得在模型forward中加入最优距离、测试参考长度、问题难度格编号、未来搜索结果或人工正确动作。

# 14. 输出要求与回执

新运行目录：

```text
runs/final_master/c1_route_b/depots_search_match_v1/<UTC_time>_<codehash>/
```

最小产物：

| 文件 | 内容 |
|---|---|
| plan/registration.json | 任务范围、授权原文、搜索规则、资源上限、旧资产身份 |
| assets/evaluator_identity.json | checkpoint/WL模型/作者软件版本、编码形式、数值变换 |
| results/reference_length_sensitivity.csv | 历史参考与最好已知参考对照，旧结果不变 |
| results/topology.csv | 所有题的对象、目标、资源规模 |
| results/search_by_case.csv | 逐题首解、失败、展开、时间、内存、计划长度 |
| results/search_budget_curve.csv | 同一次运行的三档快照 |
| results/paired_comparisons.csv | 覆盖与成本/展开配对 |
| results/compute.csv | 生成、评价、前向、图编码、转换开销 |
| plans/ | 有效计划和验证记录 |
| results/final_summary.md | 全部结果，不只保留最好h |
| results/claim_boundary.md | 数据角色、同搜索与原生系统、可说/不可说 |
| results/method_recommendation.md | 下列决策树的一项明确建议 |
| receipts/final_receipt.json | 完成数、缺失、无新训练、进程和提交情况 |

回执最低字段：

```yaml
card: C1-DEPOTS-SEARCH-MATCH-V1
base_commit: e10e80f56236cb7c43cd4113c8bc345d932f629f
new_training_runs: 0
optimizer_steps: 0
new_gate_labels: 0
new_problem_generation: 0
search_conditions_planned: 6
search_cases_planned: 182
search_runs_planned: 1092
completed_search_runs: <actual>
technical_incomplete: <actual>
resource_limited: <actual>
common_engine: <identity>
wl_identity_verified: <true/false>
historical_results_modified: false
path_owner: xushijie3
next_action: WAIT_FOR_METHOD_DECISION
```

资源失败是实验结果，不要求全部题解出才PASS；外部h缺失、无效计划或输入泄漏不能混成正常低分。提交前检查文件大小、路径、秘密信息和权重扩展名；普通Git只存代码、配置、表格和轻量日志，不存`.pt`或大状态缓存。默认不创建PR、不合并、不强推。

# 15. 结果出现后如何决定，不自动换方向

## 情况A：共同搜索下，DENSE对WL/h_add有质量或展开优势

若在不损害覆盖的情况下更少展开或计划更短，保留“受限目标训练的状态评分器可用于搜索”的证据。**不能叫发明了GBFS，也不能把所有差异归给注意力。**

若这种优势只存在于某个公开域的项目划分，下一张卡再设计独立确认；不把本开发题重新称为确认。

## 情况B：节点预算质量好，但时间/显存明显落后

优先研究昂贵评分调用的选择或共享编码，针对真实的运行成本。此时可以比较“便宜h为多数状态评分、只有候选进入关键队列时调用GNN”等计算设计。

这只是下一张方法卡候选；不能未经比较就在本卡偷偷加入top-k重排、延迟评分或异步多节点展开。

## 情况C：搜索补上了覆盖，但DENSE没有超过WL或h_add

可说明旧执行限制贡献了一部分差距，但现有学习配方未证明新的搜索优势。停止把“接了搜索”当论文主创新，不自动恢复VOC，更不换域救结果。

下一步应针对跨分支排序、训练监督或编码成本提出一个明确方法；先根据本卡曲线确定实际瓶颈，再写新的有限实验。

## 情况D：神经搜索大多在少量展开前超时

本卡不足以判断其在大节点预算下的引导能力。先把成本问题和启发式质量分开；不能把墙钟失败直接解释成不会处理目标结构。

不补一个排除时间的无限计算表来制造胜利。可在小问题有完整节点曲线的范围内报告质量，明确规模边界。

## 情况E：失败在更多地点/资源的题中集中

这只能支持后续“拓扑覆盖”的研究理由。若下一卡要新Train96′，保持目标结构受限，DENSE与WL同数据重训、相同数据量和选模机会；还需保留原拓扑与新拓扑的测试分层。

**本卡不自动启动这两项训练。**不能据相关性写“已排除目标结构因素”或“已证明是拓扑导致失败”。

## 情况F：所有评分都与目标计数接近

检查共同搜索实现、并列和分数调用是否正确；若正确，就承认本配置下没有显示学习评分的有效增量。不是简单给所有神经模型加深后重跑。

# 16. VOC保留为后续候选，但本卡不执行

若以后确实需要学习门控，至少遵守以下修订：

1. GateTrain使用同一简单生成分布的新问题，不借用Struct/Joint/IPC；新增数据和标签成本单列；
2. 若标签目标是回合改进，缓存完整访问历史、剩余预算、后续策略与模型身份；局部d*差只作辅助；
3. 后续HandGate策略与学习门控不是同一个策略，标签不是部署收益真值；
4. `a1==a2`时动作路径相同、任务代价差可能为0，但展开计算已花费，净计算价值应为负，不能把这些案例全删掉后声称学会节约计算；
5. “30个有信息问题”“+0.02非劣界”“减少30%叶数”是工程决策参数，不是由当前证据自动推出的统计标准；
6. 问题内打乱RICH表示保留边缘分布，却仍破坏联合分布，不是完美的容量控制；需要正确命名并按问题划分验证；
7. 学习门控必须对比手工阈值的合理预算曲线，而非仅对比每步都搜索；叶数、前向数、时间不应被强迫混成一个“等计算”点；
8. 64/128题的功效不能仅凭题数判断；先报告配对差值分布与可识别效应范围；
9. 若选择仍然改变端到端行为，不能说“评分器没变，所以泛化不能改变”；能否提升泛化由独立任务评价决定，但不能声称评分器本身变强。

# 17. 简明结论边界与文章定位

本卡成功时可以说明：

> 固定的、在简单目标结构上训练的状态评分，在共同搜索和明确预算下仍保留可用的决策信息；与同题训练的另一种评分器及经典启发式相比，在某些覆盖率、计划长度或展开预算上具有具体优势。

不可仅凭本卡声称：
- 新的搜索算法或新的注意力原理；
- 纯粹由执行无回溯造成全部IPC差距；
- 同节点比较等于同算力、神经方更快；
- 相对最好已知参考的长度比就是最优性；
- 在两个域都已完成这项新搜索比较（本卡只做Depots）；
- 旧REL主张已恢复，或某个未来VOC已经验证；
- 已达到录用标准。

**这不是让项目退回低水平机制稿，而是先把已有评分器放进公平的使用框架，确定下一项方法该改善评分质量还是评分成本。**有实质优势才深化；没有优势也不靠改名或换协议包装成功。

# 18. 证据入口

## 用户审查

[A1] `CP_DISR_C1_Depots结果与VOC方案审查_20261008.md`：主要结论§0，口径F1–F5，VOC问题M1–M8，搜索建议§4–§8。

[A2] `depots_paired.py`：`paired()`、三路逐题事后选择、分层统计。本轮未执行。

## 固定项目文件

固定根：
`https://github.com/rollinpioneer/graph/blob/e10e80f56236cb7c43cd4113c8bc345d932f629f/`

[R1] `src/cp_disr/pddl/stages.py`：`reference_length`、旧预算、原训练/评价。

[R2] `src/cp_disr/pddl/model.py`：`state_values`、图分块、领域词表。

[R3] `scripts/c1_public_depots.py`：`_goose_worker`、`load_model`、`cmd_lock`、固定外部调用。

[R4] `runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e/suite_runtime.json`：软件SHA、服务器记录路径。

[R5] 同运行下`results/final_summary.md`、`results/final_summary_numbers.md`、`results/claim_boundary.md`、`receipts/final_receipt.json`：已完成实验来源；本文未重新复算全部数字。

## 作者代码与官方材料

[W1] 固定GOOSE执行入口：
https://github.com/DillonZChen/goose/blob/04dbe8f27a8ec027626e6c6b5d6e7610b81f9d57/plan.py

[W2] 固定GOOSE训练配方（rank-svm / downward）：
https://github.com/DillonZChen/goose/blob/04dbe8f27a8ec027626e6c6b5d6e7610b81f9d57/configurations/classic.toml

[W3] Fast Downward官方搜索语义：
https://www.fast-downward.org/latest/documentation/search/SearchAlgorithm/

[W4] Hay等，Selecting Computations: Theory and Applications，UAI 2012。这里只用来说明计算选择已有成熟研究，不声称穷尽方法近邻：
https://proceedings.mlr.press/r10/hay12a.html

---

**最终执行建议：只授权本卡时，做六个冻结评分器的共同搜索比较、旧指标补充和完整报告；训练数量为0。拓扑训练、学习门控、昂贵评分调度与独立确认属于下一张卡，不自动展开。**
