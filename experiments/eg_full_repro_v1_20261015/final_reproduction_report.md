# EconomicGrasp V1：原始数据通过，完整监督生成中

当前结论：原始数据取得成功，完整来源与内容核验PASS；标签缓存优化在四个固定真实帧上与官方原路径完全相同。全量Graspness当前快照2103/25,600帧，持续运行。

完整监督、真实Smoke、单种子10Epoch/64,000更新、Seen/Similar/Novel官方AP和Bad Case尚未完成；实测AP保持null，不能报告完整复现成功。环境及CUDA算子工程检查此前已通过，官方权重工程检查不算本次训练结果。

用户最新规则：仅正式长时训练期间启用定时监控，其余阶段持续执行。当前监控已暂停，准备阶段保持本轮保活SSH会话。研究进展优先，只修复实际阻塞或影响结论的问题。

分支 `codex/eg-full-repro-v1-20261015`，https://github.com/rollinpioneer/graph/tree/codex/eg-full-repro-v1-20261015 。原始目录 `/home/xushijie3/work/eg_full_repro_v1_20261015`，输出 `runs/EG_FULL_V1`；轻量证据见data_report.md、data_recovery/，后续入口见CONTINUATION.md。
