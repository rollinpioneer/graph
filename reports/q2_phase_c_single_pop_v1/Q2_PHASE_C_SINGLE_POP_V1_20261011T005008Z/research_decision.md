# CP-DISR-Q2-PHASE-C-SINGLE-POP-V1 结果

只在两份固定现场改变第一次出队，之后全部使用冻结 CPU/FP32 单状态 D0@820。

| 事件 | 裁决 | A 新增展开 | B 新增展开 | ΔE | A/B 计划长 | ΔL |
|---|---|---:|---:|---:|---|---:|
| struct_n4_004_k008 | VALID_COMPLETE_NO_OBSERVED_EFFECT | 6 | 6 | 0 | 14/14 | 0 |
| ipc_p12_k002 | VALID_BOUNDED_EFFECT | 110 | 108 | UNKNOWN | UNKNOWN/UNKNOWN | UNKNOWN |

逐事件边界：

- struct_n4_004_k008：共同完整窗口 6 步；排除干预当步后的首次决策分歧 3；完整效率 OBSERVED_FOR_THIS_EVENT；汇合 NO_OBSERVED_COALESCENCE_IN_SAVED_WINDOW。
- ipc_p12_k002：共同完整窗口 108 步；排除干预当步后的首次决策分歧 2；完整效率 UNKNOWN；汇合 NO_OBSERVED_COALESCENCE_IN_SAVED_WINDOW。

p12 各支若截断，不把截断值代入真实总展开或加速比。Struct 的计划经过动作合同回放，最优性仍 UNKNOWN。A′/B′为技术重复，不能视为四个独立任务。

H2-C 只按上述两个事件裁决；不支持 T1 整体优劣或监督来源因果。H3 保持 UNRESOLVED。Struct 同父关系形式已有方法覆盖，具体最优路径条件未知；p12 跨父关系缺少最优路径身份。

训练、拟合、距离认证、GPU、完整图构建均为0。执行到本卡预算后停止，没有补跑、扩大面板、R4或新算法。

详细数据：results.csv、event_results.json、repeatability_checks.json、coalescence_checks.json、branch_ledger.jsonl、branches/及plans/。

补充核查：Struct 的 A/B 合法计划动作序列不同，但长度均为14；同处理重复动作序列一致。`VALID_COMPLETE_NO_OBSERVED_EFFECT`仅指预注册主指标 ΔE=0、ΔL=0，不表示轨迹或计划动作完全相同。

评分账本记录3105次真实模型—状态调用，其中4次在墙钟截止时中断，均保存失败/中断记录及对应字节占位，不作为有效评分。没有新训练、拟合或距离认证。

截止处理：完整展开均在120秒内完成；原生调用中断与现场保存产生14—59毫秒的截止处理延迟，已在results.csv逐支列明。没有追加超时展开，半完成展开不纳入比较。
