# E5 复核（合同不足）

名义合同下 OPEN/PICK 是否已可区分（各 case，穷举 BFS + 生产 nominal 函数）：{'T_A_dev_14': True, 'T_A_dev_18': True}。

diagnostic_reason=CONTRACT_EXPLAINABLE_BRANCH_DIFFERENCE：r3 的 OPEN 成功 / PICK 后 NO_PLAN 差异可由合同（OnTable(second) 不会被重新加入、OPEN 需 GripperEmpty）在名义层解释。这解释的是顺序结构，不是 E4 的真实后果差异；E5 的“合同之外”条件因此未被建立。

限制：真实 post-action facts 未保存，NO_PLAN 的运行时原因只是限定假设（NOMINAL_DERIVED），未被重演证明。
