执行 Stage 0B — Unit & Invariant Tests。

严格以以下文件为唯一执行规范：
1. CP_DISR_v2.1_Minimal_Experimental_Execution_Plan.md
2. CP_DISR_M1_Research_Specification_v2.1.md
3. CP_DISR_v2.1_Experiment_Manifest_v0.yaml
4. CP_DISR_v2.1_Implementation_Interfaces_v2.1
5. 当前仓库内已冻结的schema、fixtures、method configs和Stage执行卡

当前仓库：
- path: /home/__compress_data/xushijie/graph_cp_disr_v2_1
- branch: codex/cp-disr-v2.1-stage-0a
- starting commit: 3669e5a11e59b4b327db714f092b2d9ca14d78d3

当前已验软件环境：
- Python 3.10.19
- PyTorch 2.7.1+cu126
- PyTorch Geometric 2.6.1
- GPU R-GCN forward/backward: passed
- checkpoint round-trip: passed

Stage 0A仍为BLOCKED，不得在本轮将其改写为PASS。
Stage 0B允许在真实任务资产尚未绑定时，使用明确标记的synthetic unit fixtures验证生产组件。

==================================================
一、本轮唯一目标
==================================================

执行Stage 0B规定的T01–T25全部单元与不变量测试，验证：

- CP-DISR v2.1公式实现；
- 名义干预与真实状态权限隔离；
- G^K/G^H及名义图的节点和目标对齐；
- TRUE/FALSE/UNKNOWN逻辑；
- VLM relation schema v2；
- 空prior零差分与零残差；
- candidate mask和canonical candidate ID一致性；
- Structural Q仅监督真实执行动作；
- target stop-gradient；
- 四路共享R-GCN梯度；
- duration-aware SMDP target；
- terminated/truncated/buffer boundary；
- episode级80/20 prior固定协议；
- PPO重算时不得重新抽prior；
- recurrent prefix和GRU状态管理；
- bounded prior residual；
- Full、A_DD、A_Q、A_B开关语义。

本Stage不评价任务性能，不证明方法优于baseline。

==================================================
二、禁止事项
==================================================

1. 不执行科研训练。
2. 不执行Stage 0C。
3. 不发出任何VLM API请求。
4. 不执行机器人或仿真任务动作。
5. 不生成success rate、learning curve或任何性能结论。
6. 不修改CP-DISR v2.1冻结Method。
7. 不通过放宽公式、改变fixture语义或删除测试来使错误实现通过。
8. 不使用assert True、空测试或全量skip冒充通过。
9. 不把四个toy graph测试冒充T01–T25完整生产验收。
10. 不将synthetic fixture写入论文实验结果。
11. 不自动进入Stage 0C。
12. 不推送GitHub。

==================================================
三、测试必须调用生产组件
==================================================

在运行前先确认：

- tests调用src/cp_disr中的生产实现；
- 测试没有复制一套独立的“正确答案实现”替代生产代码；
- fixtures均包含：
  synthetic_unit_fixture: true
- production relation registry不存在SOFT_ORDER；
- training config不存在relation_rewrite；
- 主训练prior协议只有：
  original: 0.80
  absent: 0.20
- no-prior episode内，Full prior通路确实退化为零，而不是切换到另一套隐藏输入；
- runtime未绑定时train/evaluate入口继续fail closed。

若测试没有调用生产代码，Stage 0B直接记为BLOCKED，不得继续宣称验收完成。

==================================================
四、按三组执行T01–T25
==================================================

先运行：
- T01–T09：逻辑、图、目标和relation校验。

再运行：
- T10–T18：PPO数据链、mask、Structural Q、梯度、边界、prior固定和重算。

最后运行：
- T19–T25：时长折扣、权限隔离、bounded residual、GRU、soft message path、空候选和消融开关。

至少覆盖：

T01 Empty prior -> D^P = 0
T02 Empty prior -> prior residual = 0
T03 Nominal patch不污染真实Fact Store
T04 四路图node alignment
T05 goal_refs alignment
T06 TRUE/FALSE/UNKNOWN逻辑
T07 ADD/DEL/UNKNOWN冲突拒绝
T08 effect_fact_ref验证
T09 SOFT_ORDER不存在
T10 PPO old/new使用同一candidate IDs和mask
T11 Structural Q只gather executed action
T12 Q/V target stop-gradient
T13 四路共享R-GCN gradient flow
T14 terminated/truncated/buffer boundary
T15 candidate/object/goal permutation consistency
T16 cache key isolation
T17 prior episode内固定
T18 PPO recompute不重新抽prior且不复用旧D
T19 duration discount和episode起点累计折扣
T20 prior不得修改candidate/mask/reward/evaluator
T21 bounded prior residual及概率比界
T22 recurrent prefix和多环境GRU隔离
T23 soft relation message path及D^P响应
T24 empty candidate安全终止
T25 Full/A_DD/A_Q/A_B开关语义

==================================================
五、设备和精度
==================================================

必须执行：
- CPU逻辑测试；
- CPU FP32；
- 适用测试的CPU FP64手算对照；
- GPU FP32数值和梯度子集。

默认容差：
- FP32：atol=1e-6，rtol=1e-5
- FP64：atol=1e-10，rtol=1e-9

但以下性质不得用容差掩盖：
- 空prior显式复用下D^P严格为0；
- 零锚定下uP和Delta严格为0；
- nominal overlay不得修改真实Fact Store；
- 未执行Q输出项的直接梯度严格为0；
- detached target不得拥有grad_fn；
- episode内prior hash不得变化。

CPU与GPU不要求逐bit完全一致，但必须报告最大绝对误差、最大相对误差及失败位置。

==================================================
六、失败修复规则
==================================================

允许在Stage 0B内修复实现bug，但必须：

1. 记录失败测试、traceback和根因；
2. 保留修复前日志；
3. 每次修复生成attempt记录；
4. 保存代码diff；
5. 只修实现错误，不改Method；
6. 修复后先重跑受影响测试；
7. 再重跑完整CPU T01–T25；
8. 适用时重跑GPU子集。

优先处理顺序：

1. T03真实Fact Store污染；
2. T10 candidate/mask不一致；
3. T12 target梯度泄漏；
4. T14 terminated/truncated边界；
5. 节点和goal alignment；
6. 三值逻辑与effect冲突；
7. prior抽样/重算；
8. GRU状态；
9. 数值、梯度与soft path。

不得进行超参数搜索。

==================================================
七、输出目录
==================================================

使用：

experiments/part_0_validation/stage_0b/

必须生成：

- junit.xml
- test_results.json
- test_results.csv
- fixture_manifest.json
- invariant_report.md
- stage_0b_summary.md
- logs/
- attempts/
- environment_snapshot.json
- code_hashes.json
- schema_hashes.json

更新：

experiments/stage_status/stage_0b.json

test_results.csv至少包含：

- test_id
- test_name
- production_component
- fixture
- device
- dtype
- expected
- actual
- tolerance
- status
- failure_location
- attempt
- code_commit
- notes

==================================================
八、状态判定
==================================================

只有以下情况可以写PASS：

- T01–T25生产组件CPU检查全部通过；
- 没有测试被不合理skip；
- 没有权限、mask、target或状态污染错误；
- 全部Required Files存在且可追溯；
- 测试实际调用生产组件。

如果CPU全部通过，但某些GPU补测由于明确设备原因没有完成：
- PASS_WITH_NOTES。

以下情况写NEEDS_RERUN：
- 任一测试发现可修复的实现错误；
- mask、target、prior固定、状态隔离或边界行为错误；
- 修复后尚未完成完整CPU回归。

以下情况写BLOCKED：
- 核心生产组件缺失；
- 测试调用的是独立伪实现；
- fixture/schema不完整，无法进行实际验收；
- 测试大量skip，无法判断实现正确性。

不要因为Stage 0A仍BLOCKED而自动把Stage 0B也写BLOCKED。
两者状态独立记录。

==================================================
九、Git要求
==================================================

完成后：

1. 将Stage 0B测试、修复、报告和状态文件提交到当前分支；
2. 输出最终commit hash；
3. 确认git status干净；
4. 不推送GitHub；
5. 不修改旧仓库；
6. 停止，不执行Stage 0C。

==================================================
十、最终报告必须明确回答
==================================================

1. T01–T25分别是什么状态？
2. CPU通过多少、失败多少、skip多少？
3. GPU补测通过多少、失败多少、未运行多少？
4. 是否发现真实Fact Store污染？
5. 空prior时D^P和Delta是否严格为0？
6. mask和candidate ID重算是否一致？
7. Structural Q是否只监督executed action？
8. target是否完全detach？
9. prior是否只在episode开始抽样一次？
10. PPO更新是否重新计算E/D而不重新抽prior？
11. terminated/truncated处理是否正确？
12. Full/A_DD/A_Q/A_B是否只改变各自规定的部分？
13. 本轮修复了哪些生产代码bug？
14. Stage 0B最终状态是什么？
15. 是否具备进入Stage 0C的代码和schema条件？
16. Stage 0C仍缺哪些真实输入或API验证？
17. Stage 1A仍缺哪些真实runtime绑定？

完成报告、提交代码、更新stage_status后立即停止。