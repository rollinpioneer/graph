# L2RAR2 R12 执行审计

engineering_status = R12_STATIC_AUDIT_COMPLETE_WITH_GAPS
scientific_status = L2RAR2_PARTIAL_KEEP_G1
retained_graph = G1_predicate_bound
selected_candidate_id = null
confirmation_run = false
l3_entry_allowed = false

## 结论

本轮是零物理静态审计。R12 新增物理执行严格为 **0**，没有创建 MuJoCo model/data、simulator 或 renderer，也没有调用 physics step。证据范围是入口审计、永久拒绝登记簿、经审阅的 fake factory/step 纯测试，以及 R11 固定的 `physical_execution_accounting.json`。R11 历史事实仍为 5 次调用、40 个实例、上限 8，状态为 `BUDGET_EXCEEDED_BLOCKED`；没有用 R12 拒绝请求抵扣历史计数。

CLI 的 `physics-probe` 子命令、`run_physics_probe()`、ordinary/instrumented 辅助入口和 R12 maintenance deny 入口均在物理依赖或 factory 之前拒绝。纯测试中 fake factory 与 fake step 计数均为 0。该门禁是同一 Git clone worktree 范围内的合作式入口保护，不是操作系统级沙箱；其他 clone、任意外部 Python 进程不在本次运行时覆盖范围内。

## 历史调用与保存状态

五次历史调用只有 R11 的不可拆分总账和最后保留的部分 ledger 可追溯。`historical_execution_inventory.csv` 保留一条 `AGGREGATE_ONLY` 记录（40 个实例）和一条 `PARTIAL_LAST_BATCH_ONLY` 记录；没有制造前四次调用或 40 条伪明细。原 R11 文件未修改。

R11 保存了仪表版 `after_mj_step` trace，以及缓存等价性摘要；没有保存 ordinary action-end 状态，也没有得到跨来源 callback/order 的一一映射。因此首差异状态为 `NOT_LOCALIZABLE_FROM_SAVED_ARTIFACTS`。R11 摘要中四个 case 的 action/control/event 相等与 action-end geometry 不等仍仅按原作用域保留，不能被改写为首个物理根因。

## 审计语义与禁止主张

本轮新增了历史来源与当前来源分离、sampling point 显式映射、同时间戳顺序未知、窗口前驱缺口、contact/support/loss 分列和机制主张隔离。时间边界没有改变旧参考合同；没有调整阈值、补采样、回放、重标旧事件或改变 G1。

仍禁止从 weld-off、手部接触、绝对物体速度、单一 contact_lost 事件或缓存 geometry mismatch 推出真实 task loss、稳定支撑、根因或正确检测延迟。没有候选、没有 confirmation、没有 L3，也没有训练或 API 调用。

## 验证与交接

包内辅助纯测试实际运行 48 项；仓库新增纯测试实际运行 3 项。代码只做门禁和只读审计，没有读取密钥、外置原始 RGB 或服务器 ZIP。结果 ZIP 的文件完整性验证与科学有效性分开报告。

下一阶段固定为 `HUMAN_REVIEW_OF_STATIC_AUDIT`；未来物理预算仍为 0，任何新的 replay 必须经人工复核并另立协议。
