# E4 复核（r3，零新增样本）

统计单位：2 个独立 case 配置 × 每 case [2, 2] 个 paired repeats = 4 组配对、8 个 branch attempts；repeat 不是独立配置，不做显著性或泛化率计算。

## 每对结果

- T_A_dev_14 repeat 0：evaluator 确认成功差=True；终止层差=True；cost 差=NOT_MEASURED；rework 差=NOT_MEASURED；动作序列不同（仅描述）=True；配对恢复(qpos/qvel≤1e-09)=True（qpos最大差 0.0，qvel最大差 0.0）
- T_A_dev_14 repeat 1：evaluator 确认成功差=True；终止层差=True；cost 差=NOT_MEASURED；rework 差=NOT_MEASURED；动作序列不同（仅描述）=True；配对恢复(qpos/qvel≤1e-09)=True（qpos最大差 0.0，qvel最大差 0.0）
- T_A_dev_18 repeat 0：evaluator 确认成功差=True；终止层差=True；cost 差=NOT_MEASURED；rework 差=NOT_MEASURED；动作序列不同（仅描述）=True；配对恢复(qpos/qvel≤1e-09)=True（qpos最大差 0.0，qvel最大差 0.0）
- T_A_dev_18 repeat 1：evaluator 确认成功差=True；终止层差=True；cost 差=NOT_MEASURED；rework 差=NOT_MEASURED；动作序列不同（仅描述）=True；配对恢复(qpos/qvel≤1e-09)=True（qpos最大差 0.0，qvel最大差 0.0）

## 状态层级（不合并）

- OPEN 分支：独立 Evaluator 在最后一个动作后给出 TASK_SUCCESS（terminated=true），真实任务成功。
- PICK 分支：首动作后 Evaluator 为 CONTINUE / success=false / 未终止；随后 B_PLAN 返回 NO_PLAN，runner 因 planner 停止。这不是 Evaluator 判定的失败，也不是 DEADLINE，更不是物理不可行的证明。

## 每 case 结论

- T_A_dev_14：2 组配对，配对恢复通过 2，结果差 2；协议层可靠差异=True。支持范围：same public initial state and sim qpos/qvel; shared B_PLAN continuation; evaluator-level outcome；未比较：controller internal state, RNG stream, actual post-action facts (not saved)。
- T_A_dev_18：2 组配对，配对恢复通过 2，结果差 2；协议层可靠差异=True。支持范围：same public initial state and sim qpos/qvel; shared B_PLAN continuation; evaluator-level outcome；未比较：controller internal state, RNG stream, actual post-action facts (not saved)。

冻结 E4 门：CANDIDATE_SATISFIED_PENDING_AUTHORITATIVE_DEFINITION（authoritative Final Experimental Plan v3 §5.2-5.6 not located; E4 stated as in the operation card (>=2 independent witnesses with reliable consequence difference). No E5-style 'beyond contract' condition is added.）

限制：cost/rework 无可比记录（NOT_MEASURED）；PICK 后真实 post-action facts 未保存，不能声称重演了 NO_PLAN 的运行时原因。
