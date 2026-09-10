# L2RAR2 R13 静态恢复与 R14 准备度审计

## 工程状态

`R13_WAITING_FOR_HUMAN_REVIEW`

R13 新增物理执行严格为 `0`。本轮没有 import 或构造 MuJoCo model/data、simulator、renderer，没有调用 `mj_step`/`mj_forward`，没有训练、API 调用或密钥读取。R12 合作式门禁只做状态复核和既有纯测试范围记录，没有通过新 replay 验证。

## 科学状态

- scientific status: `L2RAR2_PARTIAL_KEEP_G1`
- retained graph: `G1_predicate_bound`
- selected candidate: `null`
- confirmation: `false`
- L3: `false`

## 已恢复证据

已复核 R12 入口、R12 结果 ZIP、R11/R12 固定审计文件，并对受控目录生成静态证据清单。R12 ZIP 完整性结果为 `PASS`，实际 SHA256 为 `eb555e663efa45c572a4bcf0971201c81a73757c2bed4c020e2d1421e70727f8`。历史调用恢复为一条 40/8 aggregate 记录、一条保留的 last-batch ledger 记录和一条 R12 零物理静态审计记录；没有伪造 40 条明细。R11 三个机制字段均保留为 quarantine，不用于科学机制或旧事件重标。

Round-9 generation lock 已恢复：`artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/rounds/l2rar2_9_attach_relpose_repair/repair_generation_lock.json`，SHA256 `74b6accfc12b3865b53531ef02b1a0e4ad28cd8d54e5f7ea86c3be47797a059f`；Round-9 validation manifest 为 `artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/rounds/l2rar2_9_attach_relpose_repair/run_manifest.json`，SHA256 `0684fe22d4834c96a1ed7f8cfd56a2800ac2f561512dc140cde60171b51a5868`。其记录的外置 rollout manifest SHA256 为 `0708e6408aa95a83c42b8fcd9269577442b62e093ff008e3aab1597a3eb3e50b`，当前注册路径 `/home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1/rollout_manifest.csv` 的实际 SHA256 为 `0708e6408aa95a83c42b8fcd9269577442b62e093ff008e3aab1597a3eb3e50b`，状态为 `VERIFIED_HASHED_ARTIFACT`。因此 generation lock、rollout manifest hash、collection/repair version、source collection commit、case order、family seed 和 rollout seed base 已纳入 R13 证据范围。

## 未恢复证据

仍缺少前四次 R11 调用的独立日志、ordinary action-end 状态、跨来源 callback/order 一一映射、Round-9 采集时的 Python/NumPy/platform/renderer 有效配置、模型 XML 哈希以及不能从现有 manifest 证明的完整采样记录。R11 的缓存 geometry mismatch 仍是已保存摘要事实，不是首个物理差异或根因。相邻 runtime 文件只说明 R10/R11 cache-only interface diagnostic 的依赖盘点，不能冒充 Round-9 collection runtime。

## 采样点合同

源码可以证明：每个 control 段执行五次 `mj_step` 后触发 control callback；动作完成后触发 action-end callback，再返回结果。源码不能证明历史保存文件中的 `instrumented_after_mj_step` 与 `cached_action_end` 是同一采样点。crosswalk 因此保留 `SOURCE_ORDER_ONLY`、`NO_SAVED_COUNTERPART` 或 `NOT_COMPARABLE`，不做 nearest-time 对齐、插值或重编号。

## 环境合同

simulator class 和当前入口源码可由 Git 验证；Round-9 generation provenance、case order、family seed 和 rollout seed base 已恢复，但历史 collection runtime、NumPy/platform、模型 XML、有效 renderer 配置及普通 action-end 采样合同仍不完整。因此环境合同不是 R14 执行授权，也未达到可直接复现级别。相邻记录中的 Python 3.10.19、MuJoCo 3.4.0、OpenCV 4.13.0、PyTorch 2.7.1+cu126 和 CUDA unavailable 均标记为 `RECORDED_ADJACENT_RUNTIME`，不是 `VERIFIED_COLLECTION_RUNTIME`。

## 人工复核与 R14 申请

`human_review_decision.json` 保持模板空字段，`r13_physical_budget_authorized=0`、`r14_physical_budget_authorized=0`。R14 草案仅准备单 case `K3_normal_hold_pause_resume`、请求 2 个实例：先 ordinary，只有执行1所有等价性门通过才允许讨论 instrumented。R13 没有批准它，也没有修改 R12 永久拒绝登记簿。

因此当前 R14 申请状态为 **未就绪，等待人工复核**；下一阶段是 `HUMAN_DECISION_ON_R14_MICRO_REPLAY_REQUEST`。

## 禁止主张

本轮没有定位 geometry mismatch 根因，没有证明 renderer、积分器、warmstart、callback 顺序或 weld 是差异来源，没有证明 K4/K5/K6 的真实 task-level loss，没有选择候选，没有 confirmation/L3。交付文件通过校验只代表文件完整性。
