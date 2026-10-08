---
title: "CP-DISR C1：同搜索审查整合与 FAST＋ALT 零训练实验卡"
version: "v3"
date: "2026-10-09"
card: "C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3"
status: "PLAN_ONLY"
execution_authorized: false
repository: "rollinpioneer/graph"
base_branch: "codex/cp-disr-c1-depots-search-match-v1"
base_commit: "57f011656ea2ccb6915869eeead265d6c98514f2"
proposed_branch: "codex/cp-disr-c1-depots-fast-alt-v3"
storage_owner: "xushijie3"
new_neural_trainings: 0
new_external_fits: 0
optimizer_steps: 0
new_training_labels: 0
new_problem_generation: 0
max_primary_search_runs: 452
---

# 审查后的决定

**保持研究方向不变；下一张卡同时处理两个已经有证据的问题：同一评分的执行成本，以及部分公开题上较差的搜索引导。**

执行两项零训练实验：

1. **FAST：**保留冻结 DENSE-G、原评分函数和原 eager GBFS，减少状态转换、静态张量构造和无用读出的开销。
2. **ALT：**在同一搜索骨架中轮流按 DENSE 与 `h_add` 选取待展开状态，并运行匹配的 WL＋`h_add` 版本。它检验互补引导，不是等价加速，也不声称减少了神经评分次数。

**暂缓旧 SELECT“每次提升16个状态”的方案、学习门控、REL改规则、深层增量GNN和新训练。**拓扑多样训练列为本卡结束后的明确候选，但不自动执行。

本文件替代旧《同搜索结果收口与冻结评分器降本方案》中下一卡的执行设计，不覆盖旧实验、旧参考长度、旧权重和旧回执。用户后续明确授权本文件后，Agent应在本文件范围内自动完成准备、实现、评价和报告；阶段间不因分数低而反复等待主线决定。

---

## 1. 证据范围和文件依据

### 1.1 本次审查实际使用了什么

- 用户提供的 Claude 审查全文：`CP_DISR_C1_同搜索收尾与评分降本方案审查_20261009.md`。[A1]
- 配套只读脚本：`search_match_check.py`。[A2]
- 旧方案全文：`CP_DISR_C1_SearchMatch_Closeout_and_ScoringCost_Plan_20261009.md`。[A3]
- 固定提交中 `search_by_case.csv` 的全部 DENSE IPC行、全部WL IPC行，以及可见的 `h_add` IPC行；核对失败节点数、状态评价数和调用数。[R1]
- 固定提交的评分调用实现；结合此前已读取的同搜索代码、总结与回执。[R2–R6]
- Fast Downward关于交替开放表的官方文档、PyTorch关于Profiler与异步计时的官方材料，限于校验标准实现和计时原则。[W1–W3]

**没有在本轮运行服务器搜索、加载权重或重新执行完整CSV复核脚本。**对Struct/Joint的整套配对计数，本文沿用仓库报告与Claude的复核，不称为又一次独立计算。上传脚本只做了本地语法检查，没有执行其文件读取与统计流程。

本文明确区分：

- **记录事实：**源文件直接给出的数值或代码行为。
- **审查判断：**这些证据能够支持或不能支持的解释。
- **本卡设计：**下面新提出的实现、预算与资源选择，均尚未执行。

### 1.2 当前冻结基线

```text
仓库：rollinpioneer/graph
提交：57f011656ea2ccb6915869eeead265d6c98514f2
旧运行根：runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1/
神经评分器：上一卡选定的 DENSE-G，update 4100
WL：上一卡已拟合的作者模型，typed和IPC编码各有对应资产
经典评分：上一卡的 h_add 实现
```

模型路径、SHA-256、外部仓库版本、引擎文件哈希从旧卡资产清单和回执读取；不按文件名猜测权重，不重新选检查点。

---

## 2. 审查结论：哪些采纳，哪些修改

| 审查意见 | 本版处理 |
|---|---|
| IPC不能只解释为评分慢 | **采纳。**增加“资源停止原因”和“相对引导证据”两列，不能互相替代。 |
| FAST保留，按调用与批量拆成本 | **采纳。**但区分引擎逻辑批次、物理前向分块和编码器内部拆分。 |
| FAST最多影响2—4题 | **不采纳为上界。**加速可能使任何超时题继续探索；实际需要多少节点未知。 |
| 每次16个的SELECT不应优先 | **采纳暂缓。**但不能从平均新后继少于16就证明它一定每轮多算16个；`pop_up_to(16)`可能取不满。 |
| 加入DENSE＋h_add交替开放表 | **采纳。**保持全状态去重、相同预算和明确的轮换语义。 |
| h_add会救回四个引导不佳题 | **只作待检验假说。**它在p06、p12、p17也达到旧100k节点上限，并非已知可靠的解法。 |
| 交替必然损害Joint计划质量 | **改为风险。**轮换改变路径，但计划质量可能改善也可能退化，应实际测量。 |
| 同一批题、两种执行器中出现同方向分层效应，证明REL机制 | **只保留探索性重复观察。**没有独立题集，也没有隔离关系遗漏与其他因素。 |
| best_h下降较少证明平台而非虚假低值 | **不成立为机制结论。**不同问题的V尺度和起点不同，极值摘要不能识别平台结构。 |
| 后续扩展世界拓扑、仍限制目标结构 | **保留为后续方案。**不是本卡默认训练，不因ALT失败就自动证明拓扑是原因。 |
| 拓扑训练仍失败便证明编码器有问题 | **不采纳这种排他归因。**标签覆盖、优化、模型、搜索预算仍可能影响结果。 |

### 2.1 同搜索的正结果保留，但不夸大数量级

| 指标 | DENSE | WL评分 | h_add |
|---|---:|---:|---:|
| Struct成功 | 32/32 | 32/32 | 32/32 |
| Struct平均计划/最优 | 1.022 | 1.174 | 1.296 |
| Joint成功 | 128/128 | 128/128 | 128/128 |
| Joint平均计划/最优 | 1.095 | 1.219 | 1.485 |
| Joint平均展开 | 24 | 133 | 510 |
| IPC最终成功 | 14/22 | 18/22 | 10/22 |

这些是旧卡结果，不是本卡结果。DENSE相对WL在Joint上更短/更长/相同为90/16/22；展开更少/更多为125/3。[R3,A1]

**133/24约为5.5倍，不宜统一写“少一个数量级”。**同搜索固定了执行框架，不代表评分器在表示、监督组织、容量和选模上只有一个差异。

### 2.2 IPC失败逐题证据：不再二分成“纯引导/纯成本”

下表的DENSE均是旧卡300秒超时前已完成的展开数。`未解`不是不可解证明。[R1,A1]

| 题目 | DENSE已展开仍未解 | WL解出所需展开 | h_add旧记录 | 合理解释 |
|---|---:|---:|---|---|
| p06 | 13,779 | 593 | 100,000仍未解 | 对WL已有明显的节点效率劣势；继续加速后能否找到解仍未知。 |
| p11 | 15,567 | 1,556 | 19,068解出 | 对WL已有明显劣势；存在另一启发式的可用解法。 |
| p12 | 10,652 | 978 | 100,000仍未解 | 对WL已有明显的节点效率劣势。 |
| p17 | 7,915 | 588 | 100,000仍未解 | 对WL已有明显劣势，不需要等DENSE到10k才承认这一事实。 |
| p19 | 12,047 | WL在100k仍未解 | 100,000仍未解 | 对WL的原因无法区分；Claude报告REL能解出，不代表DENSE同样需要其节点数。 |
| p20 | 5,367 | WL在100k仍未解 | 80,657时超时 | 所有本卡旧评分器都未解，原因未定。 |
| p21 | 1,174 | 25,037 | **139解出** | 相对WL的可探索深度被时间严重截断，但相对h_add已显现引导劣势；不是“纯成本”。 |
| p22 | 347 | 48,652 | 11,338时超时 | 时间截断很强，尚不知道DENSE的最终节点需求。 |

必须同时保留：

> **八题的程序停止原因都是超时；至少四题已有相对WL的明显引导劣势。资源限制与引导劣势可以在同一题并存。**

不写“各一半”，不把脚本的`COST`标签当作经过因果识别的分类，不给FAST能新增几题规定理论上界。FAST若让p06等题解出，说明速度改善有用，但不会抹去旧轨迹比WL多耗节点的事实。

### 2.3 共同成功集合与REL分层

- IPC双方共同成功的12题上，DENSE展开全都更少，但这不能代表那些DENSE未解、WL很快解出的题。
- REL对DENSE在Joint的运输/非运输分层，应按审查补报：不需运输17题更短、4题更长；需运输13题更短、25题更长。[A1]
- 这是事后分层、同一题集与单种子。可以提出“运输中介关系覆盖不足”假说，不能称为已证明的稳定机理；本卡不恢复REL改规则。

---

## 3. 本卡要回答的三个问题

**Q1——成本：**不改变模型判断和搜索轨迹，当前实现能减少多少实际耗时？时间主要花在每次调用的固定开销，还是随状态数和图规模增长的计算？

**Q2——互补：**轮流使用学得评分与经典启发式，能否改善DENSE单独引导不佳的部分公开题？代价是什么？

**Q3——保留价值：**这种组合是否保住原来Struct/Joint较短计划的优势，并在相同调度下提供WL组合没有的收益？

不研究：新的注意力原理、学习门控、随机掩码、更多前瞻深度、训练更大网络或新任务方向。

---

## 4. 授权、目录和资产边界

### 4.1 目录

已在历史运行配置中出现的工作区是：

```text
/home/xushijie3/work/graph_cp_disr
/home/xushijie3/envs/cpdisr/bin/python
```

以上是待执行Agent核验的已知路径，不表示本轮检查过服务器挂载。新工作建议使用从固定提交建立的独立工作树或新分支：

```text
分支：codex/cp-disr-c1-depots-fast-alt-v3
结果：runs/final_master/c1_route_b/depots_fast_alt_v3/<UTC_run_id>/
缓存：/home/xushijie3/cp_disr_cache/
临时：/home/xushijie3/cp_disr_tmp/
```

新写入均在xushijie3。检查真实路径和符号链接，不回写xushijie2；不全局替换历史路径。工作树不干净时，不自动reset、stash或删除用户文件。

### 4.2 必须冻结

- DENSE选中权重、两份WL模型及SHA-256；不重训、不改选模。
- 现有182个问题的PDDL、任务编码和原最优/最好已知参考。
- eager引擎的后继顺序、FIFO并列规则、去重、目标检查与路径更新语义。
- `h_add`实现、WL原生取整方式、模型加载与预热口径。
- FAST保留float32及原后端数值设置，不自动升级PyTorch、CUDA、外部求解器。

### 4.3 一次授权后的自动范围

用户后续明确“执行本v3”后，允许完成本卡全部452次以内的主搜索、有限测试和报告，不在性能无提升时停下来征求是否继续另一条件。

本次请求只是生成实验文件，因此`execution_authorized: false`。不得根据旧卡授权启动本卡。

---

## 5. 统一执行流程

```text
资产核验＋固定配置
    ↓
复读已有IPC行与调用统计，写更正说明（不搜索）
    ↓
有界Profiler及固定输入状态库
    ↓
FAST实现＋等价检查          ALT引擎实现＋合成测试
    └──────────────────┬──────────────────┘
                       ↓
确定唯一FAST候选；固定ALT轮换规则；提交登记
                       ↓
IPC22六条件交错执行＋Struct/Joint两个ALT条件
                       ↓
统一报表、逐题解释、结论边界、回执与校验
                       ↓
轻量结果提交；停止，不自动开始训练
```

ALT可以在FAST实现时准备，但不得依据完整新评价结果再改轮换比例。FAST失败不阻塞独立ALT：后者统一退回原DENSE实现，报告性能与成本，不再声称使用了加速版本。

---

## 6. 阶段A：有限Profiler和调用成本模型

### 6.1 先把三个“调用”分开

1. **引擎逻辑评价批次：**展开一个父节点后，给全部新后继打分的调用。
2. **神经外层物理分块：**`NeuralEval`按48个状态拆分的一次`state_values`调用。
3. **编码器内部前向：**大图还可能因为边预算再次拆批。

旧IPC记录已显示：p21有1,175个逻辑批次、1,604次外层前向；p22有347个逻辑批次、630次外层前向。因此“每次展开只有一次网络调用”对大题不成立。[R1,R2]

所有报告分别计数，不把`eval_batches`直接当GPU前向数。

### 6.2 探针范围

- Train96中按模板边数选最小、中位、最大三题，遇并列按case_id选。
- IPC p15、p21、p22；它们已用于开发，不能声称盲测。
- 每题最多采集32次有效展开的自然后继批次、1,024个不同可达状态；采集阶段另限60秒。到限就保留已有材料，不以“采不满”为技术失败。
- 若旧记录没有状态，运行一次这个有界采集；它是新增前向/搜索活动，单独记账，不说成只读日志。
- 不用最优标签挑选容易加速的状态。

### 6.3 拆分和微基准

拆分主机状态打包、输入张量构造、传输、静态特征/边准备、关系层、目标注意力/值读出、结果取回与同步。

在每个模板上测物理批大小`1,4,8,16,32,48`；再用逻辑64状态批次确认48＋16拆分行为。微基准不足64个不同状态时可重复输入，但必须标明仅用于计时，不能利用状态值缓存走捷径。自然批次结果另外保留。

每种输入预热2次、记录3次，REF/FAST用交错顺序、相同状态和相同设备。Profiler只开在短片段；最终速度测量关闭Profiler。

- 主机计时须覆盖取回结果；使用CUDA事件或正确同步区分核启动与完成时间。[W2,W3]
- 不能把CPU等待同步的时间再与同一段GPU执行时间重复求和。
- 可以消除已确认冗余的同步，但不能靠不等GPU做完来“加速”。
- 不启用新的半精度、TF32设置、编译器或跨父节点投机批处理。

### 6.4 成本拟合

先给实测表，再拟合一个描述模型：

\[
T_{call}(B,E,G)\approx\alpha+\beta B+\gamma BE+\delta BG^2.
\]

其中B为该物理调用状态数，E为模板边数，G为目标数。使用相同模板内改变B的测量，避免只靠跨题平均值猜固定开销；样本不足或变量共线时改报分模板`alpha+beta*B`，不硬解释每个系数。

报告原始与FAST的每批耗时、残差、拟合范围和分项Profiler结果。模型是性能描述，不是因果定理；不外推未测硬件/图规模，也不预承诺3—10倍加速。

若一次评价还被内部拆批，用各实际子调用求和，不能拿逻辑批大小代入单次公式。至少额外记录CPU/GPU负载和实际批大小分布。

---

## 7. 阶段B：FAST——保持评分函数的实现加速

### 7.1 唯一候选包含的有限改动

**A. 批量状态打包。**

预先缓存动态原子到模板命题位置的索引、静态真值、目标标记。把Python任意长整数状态打包成定长字节或无符号字序列，再批量生成与原实现完全相同的TRUE/FALSE/UNKNOWN张量。

必须支持超过64位的状态；禁止直接转单个int64而截断。静态事实、命题顺序、字节序、padding位和设备dtype必须测试。Depots可见状态的UNKNOWN通道按原逻辑保持，不自行改变语义。

**B. 固定特征与边索引缓存。**

固定模型在`eval`模式时，节点类型/谓词/参数嵌入组合可以按模板缓存；分块边索引可以按批大小缓存。`goal_free_static`已有缓存，不能把重复调用它等同于每次从头构图；实际重点是反复构建的内容。

缓存键包括模型SHA、模板/目标身份、设备、dtype及批大小。默认额外缓存上限主机512MiB、GPU256MiB，可用LRU逐出；计入总内存。题目专属缓存准备必须在题级时钟内，题结束清理。

**C. 跳过没有使用的旧读出。**

为当前值评分提供只返回节点H的编码路径，跳过`_encode`中最终赋给`_ro`而未使用的旧读出。保留DENSE当前的目标特征、目标注意力、phi/rho和最终状态值，不能删错分支。[R5,R6]

**D. 安全的输入/输出同步整理。**

只有在结果取回已完成必要同步时，才移除额外冗余等待；用完整端到端微基准确认，不改变输出精度。默认不改48状态外层分块和原内部边预算。

### 7.2 明确不做

- 不近似复用父状态隐藏特征，不裁剪图/目标/候选。
- 不跨多个OPEN父节点批量展开以换取大批量。
- 不做量化、半精度、重新训练、蒸馏或新损失。
- 不把已有全状态去重、直接调用V、已有模板缓存当作新贡献。
- 不在看到IPC完成率后改批大小或增加优化版本矩阵。

### 7.3 等价性与搜索前缀

必要测试：

1. 真值与目标输入张量逐项相同，包括高位bitmask、静态事实、typed/IPC两种输入。
2. 相同单状态及相同批次下，报告所有值差；参考容差为`abs_diff <= 1e-5 + 1e-6*abs(reference)`，并另外报告bitwise是否一致。容差是检查尺度，不是搜索等价证明。
3. 对固定状态库，比较排序、并列和候选首选，不静默改变tie-break或给分数取整。
4. 搜索前缀夹具：Struct固定4题；Joint每格取运输/不运输各1题，共16题。REF/FAST各最多1,000次展开或找到解，另限60秒。合计最多40次有界前缀运行，单独登记，不计入452次主搜索。
5. 比较每次真实展开的状态ID序列、生成次序、父指针、首解计划；记录第一处分歧。
6. 有相同输入REF自身的重复读数，区分既有GPU非确定性与新代码改变。
7. IPC主运行保存低开销的有效展开序列滚动哈希，在1/10/100/1k/10k节点及结束点记录。REF与FAST只比较二者都实际到达的相同前缀；若正式运行才发现分歧，保留结果但撤销该行“轨迹等价”的解释，不事后挑一个更好前缀。

**通过采样检查，只能说“在已验证样本中数值/前缀一致”，不是全状态机器证明。**若出现新排序/前缀分歧，修复实现；不能通过改旧分数或并列规则把分歧藏起来。

如无法满足必要条件，FAST标记`NOT_EQUIVALENT_OR_INCOMPLETE`，不进入其正式计时行；ALT回退原DENSE。性能较慢不是技术失败：数值正确但没有加速仍需如实报告。

---

## 8. 阶段C：ALT——两个开放表交替引导

### 8.1 它和旧SELECT的不同

旧SELECT先按便宜分数筛哪些节点有资格精评，预期减少昂贵调用；当前自然批次较小，收益不清楚，因此本卡不做。

ALT则是：**每个新状态都由两个评分器评价，各自进入一个OPEN表；轮流从两个表里取状态展开。**它不以减少精评数量为机制，相反还增加`h_add`评价和第二个堆的开销。

Fast Downward已有交替开放表及相应eager greedy配置。[W1] 本卡是在已有框架中检验评分器互补，不声称新搜索算法，也不冒称逐项复现LAMA的全部机制。

### 8.2 两个条件

- `ALT_DENSE_ADD`：主表按冻结DENSE排序，辅表按`h_add`排序。
- `ALT_WL_ADD`：主表按已拟合WL评分排序，辅表按同一个`h_add`排序。

共同控制：`EAGER_DENSE`、`EAGER_WL`、`EAGER_HADD`。DENSE使用通过验证的FAST；若FAST不可用，全卡涉及DENSE的ALT统一使用REF并披露。

不增加DENSE＋REL、DENSE＋WL、三个队列或多个轮换比。若这一个标准组合没有价值，不现场搜索更好组合。

### 8.3 共同搜索语义

继承旧引擎：完整状态去重；CLOSED不重开；OPEN中发现较短g只更新父路径；规范动作顺序；根及生成时检查目标；遇到第一个目标后继立即返回；不加C3、G1、参考长度步数上限、beam或helpful-action裁剪。[R4]

两个队列共享：

- `node[state] = (g, parent, action)`；
- 一个CLOSED集合；
- 一个首次发现序号`serial`；
- 同一个题级时间、内存和总展开预算。

两个堆分别用`(h_main, serial, state)`与`(h_add, serial, state)`排序。**两个分数不相加、不求平均、不比绝对大小。**WL有限负值仍是普通键；`h_add=+inf`只排末尾，不裁剪；神经NaN/Inf为技术错误。

### 8.4 轮换细节（不得由Agent自行猜）

- 根先从主队列弹出。以后每完成一次**有效展开**，轮到另一队列。
- 一个状态从任一队列展开后，另一个队列中的副本成为过期条目。
- 清理过期条目不算展开、不消耗轮次，但算运行时间。
- 指定队列清理后为空，则尝试另一个；两者都无有效条目才`OPEN_EXHAUSTED`。
- 如果实际从另一个队列完成展开，下一轮切到它的对面。所有异常回退次数要记录。
- 同一状态只真实展开一次；两个堆各存一份索引不算两个状态展开。
- 同一父节点的全部新非目标后继，分别完成两种评分，再插入两个堆，之后才允许下次pop。
- 不因某个队列正在耗时而异步绕开它，避免无登记地改变算法。

### 8.5 参考伪代码

```python
# 设计伪代码；需要接入旧Budget、SuccessorIndex、状态/计划验证接口。
# 两个h均为纯状态函数；主h可为DENSE或作者WL。
OPEN = [heap(), heap()]
node, closed, next_serial = init_common_records()
turn = 0

if goal(init):
    return valid_empty_plan()
insert_both(init, h_main(init), h_add(init), serial=0)

while True:
    budget.check()
    i = turn
    discard_closed_entries(OPEN[i], closed, budget)
    if not OPEN[i]:
        i = 1 - i
        discard_closed_entries(OPEN[i], closed, budget)
    if not OPEN[i]:
        return OPEN_EXHAUSTED
    if expanded >= 100_000:
        return NODE_LIMIT

    s = pop(OPEN[i]).state
    closed.add(s)
    expanded += 1
    turn = 1 - i
    pending = []

    for a in canonical_legal_actions(s):
        t = apply(s, a)
        generated += 1
        if goal(t):
            set_goal_parent(t, s, a)
            return reconstruct_and_validate(t)
        if t in closed:
            record_duplicate()
        elif t not in node:
            record_new_node(t, s, a, next_serial)
            next_serial += 1
            pending.append(t)
        else:
            update_shorter_open_path_if_needed(t, s, a)

    if pending:
        main_values = main_evaluator.evaluate(pending, budget)
        add_values = add_evaluator.evaluate(pending, budget)
        for t, hm, ha in zip(pending, main_values, add_values):
            push(OPEN[0], (hm, discovery_serial(t), t))
            push(OPEN[1], (ha, discovery_serial(t), t))
    record_milestone_if_reached()
```

实际代码应复用旧引擎的计划父指针更新和资源守护，不直接把此伪代码当生产实现。

### 8.6 必要夹具

复用原引擎已有测试，只新增：

- `ALT(h,h)`在稳定数值下与单队列h的有效展开前缀相同。
- 同状态双堆条目仅展开一次；过期条目不偷走轮换次数。
- OPEN较短路径更新后，两堆取出时使用同一份新父路径。
- 分支死路后另一分支仍保留；空队列回退可终止。
- FIFO并列、负值、正无穷、目标生成返回和资源限制。
- 两评分器每个新状态各评价一次，缓存未跨题污染。

### 8.7 不预设互补会成立

`h_add`在旧p21能很快解题，在p06/p12/p17却不能在100k节点内解出。双队列可能使已有弱引导互补，也可能让较好评分被较差次序打断。

不得承诺：交替必然救回失败、IPC覆盖必达二者逐题并集、Joint计划必定变长或必定保持。它也不是两个独立搜索各分一半时间，节点共享会改变各自看到的状态分布。

---

## 9. 主评价矩阵和精确预算

### 9.1 新增主搜索

| 条件 | Struct32 | Joint128 | IPC22 | 新增次数 |
|---|---|---|---|---:|
| `REF_DENSE` 原评分＋原eager | 旧结果参照 | 旧结果参照 | 新跑22 | 22 |
| `FAST_DENSE` 等价加速＋原eager | 不重跑完整集 | 不重跑完整集 | 新跑22 | 22 |
| `ALT_DENSE_ADD` | 新跑32 | 新跑128 | 新跑22 | 182 |
| `ALT_WL_ADD` | 新跑32 | 新跑128 | 新跑22 | 182 |
| `EAGER_WL` | 旧结果参照 | 旧结果参照 | 新跑22 | 22 |
| `EAGER_HADD` | 旧结果参照 | 旧结果参照 | 新跑22 | 22 |
| **合计** | **64** | **256** | **132** | **452** |

- 新主搜索：452次上限，不是六条件全部182题的1,092次。
- 有界状态采集、微基准、最多40次前缀运行和合成夹具另行记账，不能隐藏在“零训练”中。
- Struct/Joint旧eager行只作计划质量和节点参照；**不拿其隔日时间比较新ALT速度**。
- FAST数值/前缀通过后，旧eager的质量行可作为相同算法参照，并注明复用。若有不一致，不假装这些旧行代表FAST。
- FAST不可用则省去其22次主搜索，不用复制REF行冒充完成；ALT仍可用REF。最终回执报告实际次数与缺项。

### 9.2 固定资源和计时

- 每题：300秒、100,000次真实展开、8GiB主机RSS增长；内存与时间同样应用到ALT两个队列和所有评分成本。
- 1k/10k/100k为同一次运行快照，不重启三次。
- 时间从题级解析/接地或WL翻译开始，包含模板准备、缓存、转换、搜索、两个评分器和结果回传。
- 模型载入与统一预热按旧卡放在题级时钟外，单独给出冷启动成本。不能把题目专属预计算放出去。
- 保留现有已登记资源守护；记录真实结束时刻及超限量，不承诺毫秒级严格命中300秒。监控只终止本卡自己创建、已获预算终止授权的子进程。
- 不由最优参考设置搜索深度/动作上限。

### 9.3 公平计时安排

IPC同题REF/FAST在同一GPU上相邻执行，按case_id确定AB或BA顺序；ALT_DENSE也放进同题时间块。不同题可分到不同空闲GPU，但同卡不并行两个自己的计时任务。

WL、h_add与ALT_WL在相同CPU限制下安排，记录核数/线程数、后台负载及翻译开销。不得以管理员方式终止其他用户工作。

只做一轮主搜索，不因某次共享负载不利而挑选最快重跑。共享负载明显影响结论时写不确定；微基准的三次交错测量用于补充，不冒充多个训练种子。

---

## 10. 主指标、分层与解释

### 10.1 FAST主比较

- 同输入每次逻辑调用/物理前向的耗时、每状态耗时、批大小。
- 配对加速比及各模板的分项变化；缓存准备和峰值内存。
- 数值、排序、搜索前缀与共同终止前计划一致性。
- IPC 10/60/300秒覆盖、到限时的展开/生成/评价状态数。

**等价加速的首要证据是保持判断且降低成本，不是必须新增几道IPC成功。**若加速让原先较弱的搜索探索更远而最终成功，也如实保留，不能因不符合先验而排除。

### 10.2 ALT主比较

主比较：`ALT_DENSE_ADD`对`FAST_DENSE`（或明确回退的REF）及`ALT_WL_ADD`。

必须再对照`EAGER_WL`与`EAGER_HADD`，避免只超过自己的弱条件。报告：

1. IPC全部22题的完成/超时/节点上限/内存限制，成功时间曲线。
2. 共同成功题的计划长度、更短/更长/相同、展开数与评价状态数。
3. DENSE新增成功和失去成功的具体题号；p06/p11/p12/p17只是预先标记的诊断子集，不替代全IPC结果。
4. Struct/Joint计划/已证最优、真正最优计划数；运输/不运输、目标堆高、堆数分层。
5. 两队列有效pop数、过期pop数、队列峰值、重复状态和路径更新。
6. 两评分器各自评价状态数、物理调用数和时间。ALT不能将一份状态的两次评价算成一次总评价。

### 10.3 Joint已有能力保护线

预先将`平均逐题计划/最优比值增加0.05`作为**开发期工程取舍线**，不称为统计非劣效证明，不为达到该线调参数。

- 完成数不下降且全部128题成功时，计算ALT相对原DENSE的平均比值差；≤0.05表示没有超过本卡事先愿意接受的平均长度代价。
- 任一条件失败时，成功集合均值不能替代完整比较。主表同时列失败数、配对成本，并给失败惩罚敏感性；不宣称已满足保护线。
- 无论是否达到保护线，都完成剩余评价并报告Pareto取舍；它不是成绩放行门。

### 10.4 IPC参考和统计

- 7题的最优证明与其他题的最好已知计划分别标注。
- 成本比只能使用固定的基准版本；新更短有效计划放到追加best-known表，不追改旧预算或旧成败。
- 配对符号检验/成功不一致检验如报告，要说明是单种子、开发题和未校正多重比较。p大不证明等价，p小也不证明通用机制。
- 未达到节点档位的超时必须保留为截断，不能当成完整同节点预算试验。
- 单题`best_h`和跨题V数值不用于判断“虚假低值/平台”；V不是距离，且不同题有不同尺度。

---

## 11. 结果如何决定下一步

| 观察 | 本卡决定与下一步 |
|---|---|
| FAST保持前缀并降低调用耗时 | 保留工程优化。是否改善覆盖另报，不把缓存写成新规划理论。 |
| FAST只在小图快、大图图层仍重 | 保留有效部分；后续可研究精确的共享/增量计算或延迟评价，但不现场展开新矩阵。 |
| FAST使原四个引导劣势题也解出 | 说明速度与节点质量共同影响限时结果；仍报告其相对WL的节点劣势，撤销“只能改善两题”的说法。 |
| ALT_DENSE增加IPC完成、保住Joint，且优于同调度WL | 有互补组件价值的开发证据；先独立确认，不称为新搜索算法。 |
| ALT_DENSE与ALT_WL接近，二者都比单队列好 | 主要支持组合调度的价值，未显示DENSE特有的增量。 |
| ALT只有h_add已有成功，且神经成本使它更慢 | 不宣称神经组合价值，停止该配方。 |
| ALT损害Joint或IPC覆盖，不形成有用取舍 | 停止1:1交替配方，不自动尝试3:1、8:1、第三评分器或调度网络。 |
| 加速/ALT后仍有明显相对WL的节点劣势 | 下一张训练卡优先考虑世界拓扑覆盖；不需要等所有超时都消失才研究质量。 |
| 大题仍只探索很浅，没法判定引导 | 暂不因节点不足判模型失效；依据实测剩余成本提出一项有界计算改进。 |

一次ALT失败不能证明新拓扑训练一定有效，也不能证明关系表示本身失效。本卡结束必须给一个优先级明确的下一步决定，而不是自动返回“五个模块都试”。

---

## 12. 后续训练方案预案——不在本卡执行

这是对Claude“拓扑多样、目标仍受限”意见的保留，不是本卡训练授权。

### 12.1 何时值得优先考虑

FAST/ALT之后，部分已消费公开题或新开发诊断仍显示明显节点引导劣势；同时原同拓扑Struct/Joint优势保留。

候选训练目标仍为单座非平凡目标堆、堆高2或3、少量箱子；变化在地点/车辆/吊车配置。可考虑1—3辆车、2—6地点、相应可用吊车，但必须保持任务合法、目标可达并先估计标签成本。

不能因为IPC显示某种拓扑，就从IPC轨迹取训练标签；新题由公开生成器独立生成。当前IPC用于设计已被消费，未来结论需要独立测试。

### 12.2 最小可解释的预算建议

若要识别“拓扑覆盖”本身，优先安排：

- 原拓扑数据下的一条匹配DENSE训练；
- 多拓扑、同目标限制和同更新/样本预算的一条DENSE训练；
- WL在相同新训练问题上的一次拟合。

两个DENSE共用初始化种子和优化流程。不能只把新数据新种子模型与旧模型相比，就把收益全部归因于拓扑。Claude建议的多种子可在候选有效后安排；不同架构在IPC上解不同题，不是同一方法训练种子方差的直接测量。

完整可达状态枚举可能随地点/车辆爆炸；不能承诺仍用旧BFS成本。若改用最优规划器轨迹或有界标签，需要单独说明监督、完整最优动作集合是否仍可得，并给匹配控制相同标签权限。未证明最优的计划不能当作精确距离。

### 12.3 独立确认应分开什么

至少区分世界拓扑是否见过、目标结构是否见过。不能只生成“所有因素都更大”的集合后称为纯目标结构迁移。

新拓扑训练仍不成功，只能说明这套训练干预不足；表示、监督、优化、规模与时间仍需分开，不能直接判定“问题一定在编码器”。

**本卡只在最终`method_recommendation.md`中决定是否提出此训练卡，不生成新训练题、不拟合WL、不自动增加种子。**

---

## 13. 工程实现落点

建议新建独立子包，不覆盖旧运行入口：

```text
src/cp_disr/pddl/fast_alt/
    profiling.py        # 有界输入库、调用计数与成本拟合
    fast_value.py       # 静态缓存、真值批量转换、仅节点编码路径
    alt_engine.py       # 两开放表，复用旧后继索引和Budget
    runner.py           # 主搜索、资源守护与按题恢复/跳过
    report.py           # 唯一报表来源
scripts/c1_depots_fast_alt.py
tests/test_depots_fast_alt.py
```

这些是拟议新文件，尚未存在。原始接口入口：

```text
src/cp_disr/pddl/search_match/engine.py
src/cp_disr/pddl/search_match/evaluators.py
src/cp_disr/pddl/search_match/budget.py
src/cp_disr/pddl/model.py
src/cp_disr/pddl/task.py
src/cp_disr/c1_blocksworld_policies.py
src/cp_disr/blocksworld/method_serial/model.py
```

CLI建议提供`prepare / profile / verify-fast / freeze / run-all / report`。只在`freeze`通过后进行主评价。某阶段没有性能收益也可`DONE`，不能把科学负结果标成程序故障。

---

## 14. 记录、自动接续与异常处理

### 14.1 最少产物

```text
plan/registration.json
assets/frozen_identity.json
prep/ipc_evidence_review.csv
prep/workload_counts.csv
profile/natural_batches.jsonl
profile/timings.csv
profile/call_cost_model.json
checks/fast_equivalence.json
checks/search_prefixes.json
checks/alt_fixtures.json
runs/search_results.jsonl
results/primary_tables.md
results/ipc_by_case.csv
results/paired_quality.csv
results/compute_costs.csv
results/decision_summary.md
results/final_summary.md
results/claim_boundary.md
results/method_recommendation.md
results/verify.json
receipts/final_receipt.json
```

报表从同一逐题结果脚本生成，性能表不得手改。记录方案、代码、模型、数据哈希及实际设备；没有必要每一个子任务都另开登记提交。

### 14.2 自动执行和停止

- 低分、变慢、未新增成功均不是提前停止其他有效条件的理由。
- 单个搜索到300秒、节点或内存上限，是正常资源结果；保存后继续。
- 非有限神经值、非法计划、资产错配属于技术问题。标记缺口，停止依赖错误部件的新任务；独立条件可以完成。
- 发现方案错误时停止投放新任务，不擅自杀正在运行的其他实验。当前卡已明确授权的题级预算守护可结束其自己的搜索进程。
- 不因低分或共享负载从头重跑。基础设施中断后的新尝试需要明确记录；默认无自动完整重试，已经完成的case直接跳过。
- 全套结束时取消本卡自己的监控，检查本卡进程；不触碰其他人的进程。

### 14.3 提交与大小检查

提交前列出新目录文件数、总字节、最大文件及扩展名清单。权重和缓存不入Git；轻量计划、配置、逐题结果、代码和测试入新分支；大Profiler轨迹保留服务器路径和哈希。

默认单个Profiler原始轨迹大于20MiB不进Git；轻量结果总量超过25MiB先压缩逐步日志或改为哈希清单，不删除结论依据、不强行提交二进制。

不得PR、merge、force-push；不得覆盖57f0116或以前结果。若登记后的修改只涉及报告，也应保留修改记录；评分/引擎/预算变化则需要在受影响正式任务之前更新登记。

### 14.4 最终回执模板

```yaml
card: C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3
base_commit: 57f011656ea2ccb6915869eeead265d6c98514f2
result_commit: TO_BE_WRITTEN_AFTER_PUSH
storage_owner: xushijie3
new_training_runs: 0
new_external_fits: 0
optimizer_steps: 0
new_training_labels: 0
new_problem_generation: 0
primary_searches_planned_max: 452
primary_searches_completed: ACTUAL
bounded_probe_searches: ACTUAL
microbenchmark_calls: ACTUAL
fast_status: PASS_OR_INCOMPLETE
alt_dense_backend: FAST_OR_REFERENCE
solved: ACTUAL
resource_limited: ACTUAL
technical_incomplete: ACTUAL
old_results_modified: false
weights_modified: false
no_pt_committed: true
next_action: WAIT_FOR_METHOD_DECISION
```

`verify.json`应检查计数与计划回放、身份和数据角色；其PASS只表示技术一致性，不表示方法提升成立。

---

## 15. 最终报告必须回答的八个问题

1. 原先关于“IPC主要只差速度”的解释怎样修正？哪些题已有引导劣势，哪些仍受截断影响？
2. FAST分别减少了哪一类开销？是否真正保留已验证的数值与搜索前缀？
3. 同时间下FAST实际多探索了多少，完成率和计划质量如何变化？
4. ALT_DENSE相对单DENSE，修复和损害了哪些IPC题？Joint的好计划是否保留？
5. 相对ALT_WL和单h_add，加入DENSE是否增加了实际价值，还是只增加计算？
6. 按总展开、总状态评价、昂贵前向和时间，结论分别是什么？有没有口径相互冲突？
7. 哪些当前主张支持、哪些仅为假说、哪些应停止？
8. 下一张卡优先独立确认、拓扑覆盖训练，还是精确计算降本？只能给一个主优先项，不自动执行。

**可以形成的结论：**现有受限目标训练的评分器在部分结构测试中有引导优势；实现降本和标准互补搜索是否把这种优势转化成更好的实际预算表现，需要本卡检验。

**不能形成的结论：**新注意力原理、新搜索算法、公开域全面SOTA、所有IPC问题只差速度、ALT证明已学会任务依赖，或一次失败就证明某一类模型永远无效。

---

## 16. 证据入口

### 用户提供材料

- [A1] `CP_DISR_C1_同搜索收尾与评分降本方案审查_20261009.md`：本次Claude审查；其中“事实/强推断/假说”的标记保留，但本文对若干标记作了收紧。
- [A2] `search_match_check.py`：只读配对与IPC调用统计；脚本`GUIDANCE/COST`输出是比较标签，不是因果诊断。
- [A3] `CP_DISR_C1_SearchMatch_Closeout_and_ScoringCost_Plan_20261009.md`：旧方案，本v3替换其下一卡执行部分。

### 固定仓库来源

统一固定提交：`57f011656ea2ccb6915869eeead265d6c98514f2`。

- [R1] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1/results/search_by_case.csv
- [R2] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/src/cp_disr/pddl/search_match/evaluators.py
- [R3] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1/results/final_summary.md
- [R4] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/src/cp_disr/pddl/search_match/engine.py
- [R5] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/src/cp_disr/c1_blocksworld_policies.py
- [R6] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/src/cp_disr/blocksworld/method_serial/model.py
- [R7] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/src/cp_disr/pddl/model.py
- [R8] https://github.com/rollinpioneer/graph/blob/57f011656ea2ccb6915869eeead265d6c98514f2/runs/final_master/c1_route_b/depots_search_match_v1/20261008T144409Z_a34d9ba1/results/claim_boundary.md

### 外部核验（只用于标准计算/计时依据，不替代项目结果）

- [W1] Fast Downward，OpenList与SearchAlgorithm官方文档：
  https://www.fast-downward.org/HEAD/documentation/search/OpenList/
  https://www.fast-downward.org/latest/documentation/search/SearchAlgorithm/
- [W2] PyTorch，Profiler官方文档：
  https://docs.pytorch.org/docs/stable/profiler
- [W3] PyTorch，CUDA asynchronous execution官方文档：
  https://github.com/pytorch/docs/blob/site/stable/notes/cuda.md

**本文件为实验设计。未连接服务器开展新计算，未修改仓库、提交代码或更新任何权重。**
