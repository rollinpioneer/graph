# G0 路线决定

- 状态：SWITCH
- 数据版本：pathgraph_stage1_dataset_v0.1
- 选定任务：square, transport
- 结构证据：现有 manifest 提供完整成功/失败 rollout，但没有可验证的不同合法顺序或 episode 内恢复事件。
- 完整历史：候选 rollout 的起始步为 0，episode 文件存在，已纳入 manifest。
- 当前关键 edge 数量：forward/failure 已覆盖；alternative/recovery=0（未标注，不等于物理上不存在）。
- 需要补采：见 targeted_collection_plan.csv。
- 阶段 2 入口：若补采后达到门控，读取 selected_tasks.yaml，建立 Graph spec v1 和人工 GT 标注协议。
