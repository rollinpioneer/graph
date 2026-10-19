# EG-B-GLOBAL-RANK-PILOT-V1 最终交接
P0–P2已完成且技术核验通过；开发闸门未通过，停止当前配方，不重复训练/评价，不进入C4。
分支 codex/eg-b-global-rank-pilot-v1-20261020；服务器 /aita_data/home/aita_xushijie/work/grasp/eg_b_global_rank_pilot_v1_20261020（gpu04）。
六个新M/R头各1000步、共6000更新，配对初始化/帧流/设备一致，旧O完整保留。
C3开发仅384固定帧，20配置共7680记录。R−O：Top-1 −1.5625pp、AP −0.838645；R−M：Top-1 −0.347222pp、AP −0.240597。代理反序下降未转为部署收益，不能归因为MSE或特征单一原因。
GPU0.0261/1进程小时、CPU4.9370/10worker小时；训练峰值5进程，离线9CPU，开发峰值48CPU。所有计算进程已结束，监控保持暂停。
最终报告 final_report.md；判据 results/development_decision.json；核验 results/independent_validation.json；原始权重/逐帧/日志索引 results/server_artifact_index.json。
当前只待同分支最终归档及SSH69371退出回执；禁止启动下一轮或P3。
