# 合同层面的 OPEN / PICK 顺序解释（纯符号，NOMINAL_DERIVED，不是运行时观测）

依据实际运行的 stage_2a 合同（无 derived 规则、无 exclusive groups）。相关效果（生产合同解析）：

- `p:OnTable:second_object`：ADD by []；DEL by ['a:PICK:second_object:v1']；作为前提被 ['a:PICK:second_object:v1'] 使用
- `p:Held:second_object`：ADD by ['a:PICK:second_object:v1']；DEL by ['a:PLACE:second_object:container:v1', 'a:PLACE_BUFFER:second_object:buffer:v1']；作为前提被 ['a:PLACE:second_object:container:v1', 'a:PLACE_BUFFER:second_object:buffer:v1'] 使用
- `p:GripperEmpty`：ADD by ['a:PLACE:second_object:container:v1', 'a:PLACE:target:container:v1', 'a:PLACE_BUFFER:second_object:buffer:v1', 'a:PLACE_BUFFER:target:buffer:v1']；DEL by ['a:PICK:second_object:v1', 'a:PICK:target:v1']；作为前提被 ['a:OPEN:container:v1', 'a:PICK:second_object:v1', 'a:PICK:target:v1'] 使用
- `p:Open:container`：ADD by ['a:OPEN:container:v1']；DEL by []；作为前提被 ['a:PLACE:second_object:container:v1', 'a:PLACE:target:container:v1'] 使用
- `p:Inside:second_object:container`：ADD by ['a:PLACE:second_object:container:v1']；DEL by []；作为前提被 [] 使用
- `p:AtBuffer:second_object:buffer`：ADD by ['a:PLACE_BUFFER:second_object:buffer:v1']；DEL by []；作为前提被 [] 使用

适用条件：初态 `Open(container)=FALSE` 且 `Inside(second_object,container)=FALSE`（下表由保存的公开 facts 核验）。在此条件下：
`Inside(second_object,container)` 只能由 PLACE(second_object,container) 增加，它需要 `Held(second_object)` 与 `Open(container)`；`Held(second_object)` 只能由 PICK(second_object) 增加，它需要 `OnTable(second_object)` 并永久删除之，且没有任何合同再增加 `OnTable(second_object)`；PICK 之后 `GripperEmpty=FALSE`，而 OPEN 需要 `GripperEmpty`，只有 PLACE/PLACE_BUFFER 才能恢复 GripperEmpty，二者都会删除 Held。因此名义合同下 OPEN 必须先于 PICK(second_object)。以上是名义合同的解释，不是物理不可行的证明。

## 精确闭包结果

- T_A_dev_14：OPEN 后 goal 可达=True（最短名义计划 ['a:PICK:second_object:v1', 'a:PLACE:second_object:container:v1', 'a:PICK:target:v1', 'a:PLACE:target:container:v1']，闭包状态 15，状态空间上界 59049）；PICK(second) 后 goal 可达=PROVEN_UNREACHABLE（闭包状态 8，状态 hash 29ea49ebbf36，边 hash 878202ba2d8e，状态: COMPLETE）。
  - 原 B_PLAN 配置离线：初态 PLAN_FOUND ['a:OPEN:container:v1', 'a:PICK:second_object:v1', 'a:PLACE:second_object:container:v1', 'a:PICK:target:v1', 'a:PLACE:target:container:v1']；OPEN 后(NOMINAL_DERIVED) PLAN_FOUND；PICK 后(NOMINAL_DERIVED) NO_PLAN。
- T_A_dev_18：OPEN 后 goal 可达=True（最短名义计划 ['a:PICK:second_object:v1', 'a:PLACE:second_object:container:v1', 'a:PICK:target:v1', 'a:PLACE:target:container:v1']，闭包状态 15，状态空间上界 59049）；PICK(second) 后 goal 可达=PROVEN_UNREACHABLE（闭包状态 8，状态 hash 29ea49ebbf36，边 hash 878202ba2d8e，状态: COMPLETE）。
  - 原 B_PLAN 配置离线：初态 PLAN_FOUND ['a:OPEN:container:v1', 'a:PICK:second_object:v1', 'a:PLACE:second_object:container:v1', 'a:PICK:target:v1', 'a:PLACE:target:container:v1']；OPEN 后(NOMINAL_DERIVED) PLAN_FOUND；PICK 后(NOMINAL_DERIVED) NO_PLAN。

## 诊断字段（非新增官方 E6 类别）

- T_A_dev_14：CONTRACT_ALREADY_DISTINGUISHES_ACTIONS=True；NOMINAL_RECOVERY_GAP=True；SEARCH_HORIZON_LIMITATION=False；SEARCH_NODE_OR_CPU_LIMITATION=False；OBSERVATION_CONTRACT_MISMATCH=NOT_ESTABLISHED；EXTRA_CONTRACT_CONSEQUENCE_PLAUSIBLE=NOT_ESTABLISHED；INSUFFICIENT_SAVED_STATE=True；runtime_NO_PLAN_reason=OBSERVED_NOT_REENACTED (actual post-action facts not saved); nominal model reachability reported separately
- T_A_dev_18：CONTRACT_ALREADY_DISTINGUISHES_ACTIONS=True；NOMINAL_RECOVERY_GAP=True；SEARCH_HORIZON_LIMITATION=False；SEARCH_NODE_OR_CPU_LIMITATION=False；OBSERVATION_CONTRACT_MISMATCH=NOT_ESTABLISHED；EXTRA_CONTRACT_CONSEQUENCE_PLAUSIBLE=NOT_ESTABLISHED；INSUFFICIENT_SAVED_STATE=True；runtime_NO_PLAN_reason=OBSERVED_NOT_REENACTED (actual post-action facts not saved); nominal model reachability reported separately

若合同已能区分 OPEN 与 PICK 的长期可达性，这对候选不支持 E5 所需的“合同近同排序下的 soft 差异”；即便当前 mask 二者均为 true，也不代表长期合同效用相同。本轮不改合同、不补技能、不重跑物理实验。
