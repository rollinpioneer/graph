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

E4 派生证据状态：LIMITED_SUPPORT_FEASIBILITY_ONLY（derived evidence status against Final Experimental Plan v3 §5.2 (not an official v3 enum); the action-sequence difference is not used as E4 evidence; cost/rework NOT_MEASURED. E4 and E5 stay separate.）

限制：cost/rework 无可比记录（NOT_MEASURED）；PICK 后真实 post-action facts 未保存，不能声称重演了 NO_PLAN 的运行时原因。

## 对照 Final Experimental Plan v3 §5.2

v3 E4要求：

同初始化合法候选真实执行＋共同continuation，并在至少两个独立见证上观察可靠的feasibility、cost或rework差异。

当前证据：

- T_A_dev_14、T_A_dev_18 是两个独立case配置；
- 每个case有两个paired repeat，repeat不计作独立配置；
- paired public state、qpos、qvel一致；
- OPEN分支最终由独立Evaluator确认TASK_SUCCESS；
- PICK分支首动作后Evaluator为CONTINUE，随后B_PLAN返回NO_PLAN；
- cost=NOT_MEASURED；
- rework=NOT_MEASURED；
- PICK的物理不可行性=NOT_ESTABLISHED；
- controller内部状态和RNG流=NOT_MEASURED。

因此本收口对E4只记录：

LIMITED_SUPPORT_FEASIBILITY_ONLY

它表示在冻结的共同continuation协议下存在两个独立case的可靠结果差异；
不表示PICK被Evaluator判失败，不表示物理不可行，也不表示E5的合同外soft差异已经成立。

E4与E5必须保持分离。
