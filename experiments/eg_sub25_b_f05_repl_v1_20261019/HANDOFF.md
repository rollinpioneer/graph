# EG-SUB25-B-F05-REPL-V1 最终交接
计算、统计和最终交付核验已完成。禁止重复启动训练、标签、预测、评价或流水线。
根目录 /aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019；gpu04。
分支 codex/eg-sub25-b-f05-repl-v1-20261019。
B: AP 40.440913→42.280472，+1.839559，95%CI [1.092625,2.578899]，10/12场景正向，AP_CONFIRMED_TARGET。A: +2.767868，12/12场景正向。
B Top-1 −1.540799pp且CI跨0；仅支持固定12场景上的AP跨子集复现，不能声称Top-1改善、官方完整90场景或全量复现。
6144/6144帧、49152配置帧记录、24/24场景评价退出0。所有计算进程已退出。
真实并行标签32CPU，评价峰值24CPU，骨干单GPU，3评分头同时训练，导出各阶段2进程；8为GPU导出配置上限。GPU4.627541/8进程小时，CPU25.428029/50worker小时，产物10.556/25GiB。
16→1024验证调用修复沿用同3权重，0追加训练。每100步头日志保留，异常前全部逐步loss未落盘；详见最终报告。
最终报告 results/final_delivery_report.md；独立重算和完成检查 results/final_delivery_validation.json；大文件身份 results/final_artifact_index.json。
本次只待同分支最终Git归档与SSH退出回执；完成后暂停自动化eg-sub25-b-f05-15，不修改其他监控，不启动下一轮。
