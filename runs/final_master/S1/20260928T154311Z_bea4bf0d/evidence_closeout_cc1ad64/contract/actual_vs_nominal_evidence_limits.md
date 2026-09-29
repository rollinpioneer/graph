# 实际观测 vs 名义推演：证据限制

- 首动作后的真实 Verifier facts：NOT_RECORDED_OR_NOT_FOUND（journal 仅保存 `verified_fact_count=10`，没有 10 条 facts 的值；r3 branch journals and branch_results (fact_id/facts keys); sidecars hold hashes and QA coordinates only）。
- 因此本轮不声称重演了运行时 NO_PLAN 的原因；名义后继标为 NOMINAL_DERIVED，不能当成真实 Verifier 输出。
- 隐藏 QA 坐标（sidecar 的 hidden_truth）只用于配对恢复 QA，未转成正式公开 facts 来补缺口。

- T_A_dev_14：初态绑定 EXACT_INITIAL_STATE_VERIFIED_AGAINST_R3_RECEIPTS；合法性与保存 mask 一致={'a:OPEN:container:v1': True, 'a:PICK:second_object:v1': True}。
- T_A_dev_18：初态绑定 EXACT_INITIAL_STATE_VERIFIED_AGAINST_R3_RECEIPTS；合法性与保存 mask 一致={'a:OPEN:container:v1': True, 'a:PICK:second_object:v1': True}。

- 精确闭包只说明名义合同模型下的可达性；运行时观察到的 NO_PLAN 与之一致但不是同一证据。
- 闭包使用 3^predicate_count 的严格去重状态；若超过 100000 会写 ENUMERATION_LIMIT_REACHED 而不是“不可达”。
