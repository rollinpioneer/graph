# EG-C3-B-TOP1-RANK-DIAG-V1
已完成零训练诊断与必要核验；不要重复启动。
服务器 /aita_data/home/aita_xushijie/work/grasp/eg_c3_b_top1_rank_diag_v1_20261020，gpu04。
分支 codex/eg-c3-b-top1-rank-diag-v1-20261020；起点50305f9c6919fe536b36704dd9f6149515fe3029（卡片登记），父研究40d705ba6ed41032121850604a3b2f73d7425658。
带CD 1106次退化、964次恢复，512个去重退化帧。主要原因类别为无碰撞但质量不足802次（72.51%），非空碰撞302，空抓2。退化对Q全偏向更低真实质量的新候选，但不能据此证明MSE或特征缺失是根因。
完整报告 diagnosis_report.md。全部事件索引 results/regressions_cd_index.csv；去重帧 results/unique_regression_frames_cd.csv。
原始缓存不变；0训练、0推理、0新几何评价。现有监控保持暂停。只待最终同分支归档和SSH退出回执，不启动下一轮。
