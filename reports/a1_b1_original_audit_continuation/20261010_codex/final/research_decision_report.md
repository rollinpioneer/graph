# A1/B1补充审计与获准A-V：工程状态报告

状态：`COMPLETE_WITH_BOUNDED_UNKNOWNS`。补充原论文/官方代码审计已完成；用户批准的A-V已完成。**固定安装构建中的剪枝差异已复现：none恢复813/836目标对，6399训练状态的不同向量数从95升至3074。** 原因尚未定位，全部证据仍为E0，没有拟合、评分或搜索收益证明。

## 1. 仓库、输入与权限

服务器真实代码仓库为 `/home/xushijie3/work/graph_cp_disr`，分支 `codex/cp-disr-c1-compact-diagnosis-v2`，HEAD `cc7d79d6fcb053bf0c16033e765a2e808de2aaac`，origin为 `git@github.com:rollinpioneer/graph.git`。受控文件无改动，已有未跟踪checkpoint/结果资产仍存在；不能简称整个工作区为空或全部已提交。运行前后HEAD、分支及完整status相同。本地CP-DISR是master尚无提交/HEAD、无remote的报告工作目录，不是服务器工程checkout。

V2输入位于 `runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z`；SCOPE为 `search_scope_diagnostic_v1/20261009T063905Z_511b117`；旧公共实验为 `public_depots_rel_v1/20261008T063544Z_9898106e`。路径均相对服务器仓库。旧A1/B1诊断和最新Claude审计在独立work目录，版本分别登记于 `../02_original_method_identity.md`，不能声称后者在cc7d79d6f中。

本轮先只读审计、双分支并行推进。用户随后明确批准“A-V，要尽量并行运行”，协调者只执行该特征卡。没有改原代码/配置/权重、安装依赖、切分支、fetch/pull/push、训练、拟合、模型评分、搜索或查询。SSH通过一个持久会话完成，最终清理状态见 `../00_receipt.json`。

## 2. 原作者方法还原到什么程度

A1实际读到2024 Return to Tradition、WLPlan、超参数研究、Expressiveness正文及历史tag/branch源码。2024原论文是L4/多重集/h*回归，CP采用后续L2/set/a-m/rank-SVM配方。历史CPU WL的取整、FIFO、不重开和目标时机已静态核实。没有运行2024作者完整系统，受限Zenodo原包未取得。

B1已取得OptRank后续公开包12575733及Chrestien作者仓库e7cd65b的内嵌NeuroPlanner包，完成指定代码静态审计。旧“未找到官方实现”已更新。OptRank(GOOSE)是LLG神经模型，不是WL/rank-SVM。后续公开包与论文在默认训练、划分、repeat、引擎上有差异，不追认为论文原始运行。NeurIPS作者保存表只核实Table1一个单元格：Elevators/GBFS/Lgbfs统计85.2747%取整85%，与发表值一致；无完整运行复现身份。

详细页码、版本、代码行号见两份 `original_method_audit.md`；合并差异表为 `../03_original_vs_cp_diff.csv`。

## 3. A1：信息丢失与获准A-V

官方ILG对象色不自动带类型，静态事实仅在实际state.atoms中才进入图；Problem第四参为负目标。候选16交换hoist，但静态位置不同，不是完整任务合法重命名。此前误传静态参数的解释已更正。

A-V固定96题6399状态，不追加词表样本。载入W1、同一安装库独立a-m、none各重复两次，ILG/2轮/set不变；另外9个已有候选状态仅嵌入。最多两个单线程worker并行，一次正式调用成功。

| 核查 | 冻结W1 | 独立a-m | 独立none |
|---|---:|---:|---:|
| 特征数 | 147 | 147 | 202 |
| 6399训练状态不同向量数 | 95 | 95 | 3074 |
| 836目标对可区分数 | 0 | 0 | 813 |
| 836对照对可区分数 | 836 | 836 | 836 |
| 24旧候选可区分数 | 0 | 0 | 23 |

冻结/重建a-m完整矩阵一致；每条流水线双重复逐位一致。95个a-m训练组全部被none拆开，反向拆分0。目标恢复率97.25%来自836全量核查，已不依赖23/24外推；剩余23目标的原因未知。旧探索候选仍不是确认集。

结论升级为 `FIXED_BUILD_PRUNING_DIFFERENCE_REPRODUCED`，根因 `UNRESOLVED_ORIGIN`。可把差异定位到当前a-m收集/剪枝/重映射/嵌入路径整体，不能锁定“MaxSAT是首个丢失点”或直接判官方源码bug。安装扩展哈希固定，源码HEAD9d9fc99受控文件无改动，但其逐字节构建对应关系未证明。静态MaxSAT设计应保持训练矩阵划分，与实测矛盾仍待解释。

已有none选项足以恢复这些局部特征差异；未拟合none读出，也未验证评分与搜索。原作者原生配置是否避免全部现象仍未知；当前现象不能叫2024论文固有限制。A1保持 `USE_EXISTING_METHOD`，新表示及AB2不触发。

完整结果与保护证据见 `../A_V/result_report.md`、`receipt.json`、`pairs.csv`、`saved_matrix_verification.json`。特征阶段墙钟2.899秒，worker CPU3.781秒，RSS采样峰588.3MiB。四次符号collect、六条管线；其他实验计数均0。

## 4. B1：监督差异与离线—搜索脱节

OptRank约束选定最优计划下一状态优于父/兄弟，并可传递到更早祖先层。Chrestien内嵌PDDL损失比较下一路径状态与累计离轨竞争者；网格实现只用同g层，有版本差别。任意off-off严格d*排序不是两者理论要求的全部对象。有限训练、采样、目标感知和整数部署都可能改变保证适用条件。

对取得的173行既有派生表重统计：跨父112修复/36损伤，同父17修复/8损伤，全部EXACT/EXACT，无并列，保存margin符号无不一致。112跨父修复中57对两端均为训练状态；这不是独立泛化或搜索收益证据。未重算原模型评分。

旧0/112仅为启发式粗覆盖：没有绑定selected plan successor，也没有作者约束闭包。112已全部核实reference_kind=OPTIMAL，BEST_SAVED不是该子集的限制。不能再写成“作者不能构造112对”或“全是两端离轨”；精确可构造性仍UNKNOWN。详见 `../B1/saved_pair_recount.json`、`pair_to_search_evidence.csv`。

库B与IPC22不交、D0/T1缺真实OPEN/轨迹的结论来自旧审计；本轮没有取得新的搜索记录。训练集拟合、监督目标不同、数据覆盖、检查点/部署差异及未发生实际竞争都是解释候选，不作因果结论。B1维持 `DEPRIORITIZE`，无新监督开发；B-V无需新实验，但精确构造性连接并未伪称已完成。

## 5. 缺陷归属、方法空间与停止边界

已核实的工程风险是固定安装构建a-m路径改变了训练状态划分；类型/静态输入和原版身份也有历史解释错误。论文理论仅在其明确定义的表示、模型、最优计划和理想排序条件下成立；Expressiveness定理3是有界存在性，不能自动套到当前训练RGCN。没有证明作者论文不实或方法固有失败。

作者已有none、多重集、完整状态、更深/更高阶WL和路径/祖先离轨监督；这些内容不能包装为新算法贡献。仍可能存在的表示—搜索或监督—OPEN问题只是未证实空间，需要实际竞争和可隔离证据才能重新裁决，方向由科研主线决定。

当前建议：A1保留为已有配置与工程来源问题；B1降级。无新增待审批实验卡，AB2=`NOT_TRIGGERED`。未执行后续拟合、搜索、编译或换库。未知边界和逐项更正见 `../05_conflicts_and_missing_evidence.md`。

## 可直接交给科研主线GPT

> CP服务器冻结在cc7d79d6f，原代码/配置/模型和既有结果未改。补充原论文及官方历史代码已静态审计；2024 WL原版与当前L2/set/a-m/rank-SVM不同，OptRank实为LLG神经模型。获准A-V完成六条特征流水线、双单线程并行、0拟合/评分/搜索：a-m训练向量组95，none3074；none恢复813/836目标对，冻结/重建一致，重复逐位一致。固定构建的剪枝现象已复现，源码/构建根因未定位，搜索证据仍E0。A1维持USE_EXISTING_METHOD；B1维持DEPRIORITIZE，旧0/112精确不可构造说法撤回，112全OPTIMAL但selected successor与闭包仍缺。一个作者已保存表格单元格统计一致，不是完整原生复现。AB2未触发，无新算法或后续实验授权；工程执行到此停止。
