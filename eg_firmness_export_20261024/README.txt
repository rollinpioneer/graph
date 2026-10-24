牢度可学习性数据导出：v2 在 cal_0067_0064 停止
日期：2026-10-24

本目录不是完整数据交付，不能用于运行 firmness_probe.py。
最新执行采用用户 2026-10-24 09:55 修订的 v2 候选级门槛，无视角覆盖。
cal_0067_0000 通过，features/cal_0067_0000.npz 含 50 个候选特征。
下一帧 cal_0067_0064 的候选中心、视角、深度、角度均未通过一致性核验，
宽度最大误差 0.0629182383 m，角度 logit 最大误差 12.17100477。
立即停止；没有跳帧、剔除候选、训练、解析评价或换角统计。
v2 结果见 summary.json、v2_report_zh.txt 和 failure_v2_cal_0067_0064.json。
原情形 C 汇总保留为 summary_case_c.json。以下是之前诊断记录。

最新诊断见 diagnosis_report_zh.txt、diagnosis_cal_0067_0000.json。
中心完全一致，同卡两次 pred 完全一致，但视角相同种子的 pred 最大误差
0.0110011 > 1e-4，不满足情形 A；依新指令归入 C，不覆盖视角，不恢复导出。
以下记录首次失败，原 summary 已保留为 summary_initial_gate.json。
144 帧历史缓存均存在并含 input_points；钩子输出形状符合指令。
首个标定帧 cal_0067_0000 在冻结官方 EG、TF32 开启、eval 模式下执行前向。
angle_index 与缓存完全相同，但 pred 最大绝对误差为 0.36499732732772827，
超过预定 1e-5 门槛。依照用户“不满足就停下告诉我”的要求停止。
该结果仅表明当前重建前向未复现缓存预测，未定位差异来源，不作机制推断。

未导出可用特征或评价标签，未启动批量推理、CPU 解析评价、训练或换角选择。
summary.json 中的 null 表示未执行，不能解释为零缺失或检查通过。
没有修改历史数据、模型源码或权重，没有访问 C4/C5。

infer_features.py：已执行首帧；包含数据重建、前向钩子和复现断言。
evaluate_export.py：仅准备，未上传服务器或执行；按设计直接调用上一轮
evaluate_p2.py 的评价函数，不改其实现。后续操作必须先解决复现不一致。
protocol.json：用户指定的数据划分和原始阈值。
failure_cal_0067_0000.json：失败帧收据，含输入 SHA、权重 SHA、GPU、耗时。
STOP_ON_MISMATCH：原始断言异常记录。
source_provenance.json：历史实现和预注册版本来源。
asset_check.json：缓存可用性核验。

本轮未放宽阈值、未更换缓存或候选，未尝试额外重跑来绕过失败。
