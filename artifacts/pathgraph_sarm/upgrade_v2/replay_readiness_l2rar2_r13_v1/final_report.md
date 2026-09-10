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

## 未恢复证据

仍缺少前四次 R11 调用的独立日志、ordinary action-end 状态、跨来源 callback/order 一一映射、generation lock、rollout manifest、历史运行时版本、模型 XML 哈希和完整 seed/program 记录。R11 的缓存 geometry mismatch 仍是已保存摘要事实，不是首个物理差异或根因。

## 采样点合同

源码可以证明：每个 control 段执行五次 `mj_step` 后触发 control callback；动作完成后触发 action-end callback，再返回结果。源码不能证明历史保存文件中的 `instrumented_after_mj_step` 与 `cached_action_end` 是同一采样点。crosswalk 因此保留 `SOURCE_ORDER_ONLY`、`NO_SAVED_COUNTERPART` 或 `NOT_COMPARABLE`，不做 nearest-time 对齐、插值或重编号。

## 环境合同

simulator class 和当前入口源码可由 Git 验证；历史 Python/MuJoCo/NumPy/OpenCV、模型 XML、generation lock、rollout manifest、seed 和有效 renderer 配置不完整。因此环境合同不是 R14 执行授权，也未达到可直接复现级别。

## 人工复核与 R14 申请

`human_review_decision.json` 保持模板空字段，`r13_physical_budget_authorized=0`、`r14_physical_budget_authorized=0`。R14 草案仅准备单 case `K3_normal_hold_pause_resume`、请求 2 个实例：先 ordinary，只有执行1所有等价性门通过才允许讨论 instrumented。R13 没有批准它，也没有修改 R12 永久拒绝登记簿。

因此当前 R14 申请状态为 **未就绪，等待人工复核**；下一阶段是 `HUMAN_DECISION_ON_R14_MICRO_REPLAY_REQUEST`。

## 禁止主张

本轮没有定位 geometry mismatch 根因，没有证明 renderer、积分器、warmstart、callback 顺序或 weld 是差异来源，没有证明 K4/K5/K6 的真实 task-level loss，没有选择候选，没有 confirmation/L3。交付文件通过校验只代表文件完整性。
