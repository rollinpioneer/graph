# 评价器逐步一致性审计

结论：A PASS，50 组真实完整初态匹配场景，共 100 条独立仿真。不是仅共享种子或在同一轨迹上比较动作。

官方固定源码 utils.evaluation.evaluate 调用 sample_actions(temperature=0)，项目实际 RD01 common.evaluate 调用 actor.mode()，都按原代码裁剪动作。通过回放 reset 包装器将相同完整 MuJoCo INTEGRATION、qpos/qvel、目标、环境/动作/全局 NumPy/Python RNG 与原包装器计数恢复到两个独立环境。初始化最大差 0，RNG 全部相等。

逐步动作最大绝对差 0（门槛 1e-6）；位置差 0、速度差 0（各门槛 1e-5）。终止、截断、奖励、成功和步数完全相同。按原 PointMaze 距离阈值 1、终止选项与 TimeLimit 1000 步独立重算全部通过。

轻量逐步表：results/A_steps_<model>_task<id>_scene<id>.csv（50 文件）；汇总 A_CASES.csv、A_AUDIT.json；完整匹配场景与双轨迹在 runs/A_*.pkl（服务器）。原函数源码散列与权重身份在 EVIDENCE_INDEX.json。

适配边界：RD01 函数在这里只显示所选任务的单任务视图，真实环境仍保留全部官方任务，回放恢复实际 task ID。该函数固定生成五项任务汇总，未显示任务的空均值会有日志警告；这些未使用的汇总不作为审计结果、不写入成绩。审计采用完整逐步记录和独立判定；两个函数源码均未修改。H1 支持仅限这些冻结权重、温度和所测 50 场景，不扩展到随机温度、视觉任务或所有环境。
