# 关系裁定与多跳合同解释（离线，未调用 provider）

8 个 provider cache 均通过 verify_audit_cache；非空 case：T_A_dev_14, T_A_dev_18, T_A_dev_19；被冻结 validator 接纳的关系共 6 条。

OPEN→目标/第二物体 的 SOFT_RELEVANT_TO_GOAL 被接纳，原因是本地冗余规则只在 effect fact 等于目标时判冗余（此处 effect 为 Open(container)，不等于目标）。
同时存在多跳合同路径 OPEN --ADD--> Open(container) --PRE_POS--> PLACE --ADD--> Inside(...)；二者不矛盾：接纳 ≠ 关系提供了合同之外的信息。

truth / utility / opportunity 标签均为 UNKNOWN：保存的数据里没有独立裁定、q_R 或 R*，本轮不发明。
T_A_dev_19 没有物理分支，不套用 14/18 的物理后果。
