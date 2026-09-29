# S1-REV1 r3 证据收口摘要（零新增样本）

- 预算：累计预留 20/20，剩余 0，状态 CONSISTENT；UNKNOWN 未退款。
- r3：8 个分支、4 组配对、2 个独立 case 配置。
- OPEN 分支：Evaluator TASK_SUCCESS；PICK 分支：Evaluator CONTINUE 后 planner NO_PLAN（不是物理不可行证明）。
- 合同层：{'T_A_dev_14': True, 'T_A_dev_18': True}；诊断原因 CONTRACT_EXPLAINABLE_BRANCH_DIFFERENCE。
- 关系：admitted 共 6 条（14/18/19 各 2 条：True）；truth/utility/opportunity=UNKNOWN。
- 关键门状态：{"E1_relation_existence": "NOT_ESTABLISHED", "E2_E3_representation": "PRIOR_VALUES_RETAINED_NOT_REVALIDATED", "E4_real_consequence": "LIMITED_SUPPORT_FEASIBILITY_ONLY", "E5_contract_insufficiency": "NOT_SATISFIED", "E6_prior_classification": "UNKNOWN"}
- 资格：tp_training_authorized=false；额外物理尝试=0；方法升级=false；下一步=EXPLICIT_RESEARCH_DECISION_REQUIRED。
- 权威文档：Final Experimental Plan v3 与 Research Content v5 已从原始权威附件恢复并完成对照；执行当时服务器未找到文档的历史记录保留不变。
- E1：NOT_ESTABLISHED；自然合法关系存在，但缺独立可裁定truth证据。
- E2/E3：沿用历史保存值，本轮未重新验证。
- E4：LIMITED_SUPPORT_FEASIBILITY_ONLY；两个独立case有配对协议结果差，但cost/rework未测，PICK为Evaluator CONTINUE后planner NO_PLAN。
- E5：NOT_SATISFIED；当前OPEN/PICK差异可由注册合同长期可达性解释，不满足合同外soft risk/cost/relevance条件。
- E6：UNKNOWN；缺独立truth/utility/opportunity与R*证据。
- S1处置不变：tp_training_authorized=false；additional_physical_attempts_authorized=0；next_action=S4_RESEARCH_DECISION。
- post-action 真实事实未保存；controller 内部状态与 RNG 未比较（NOT_MEASURED）。
