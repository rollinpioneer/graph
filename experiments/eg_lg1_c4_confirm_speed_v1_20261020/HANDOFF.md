# EG-LG1 C4确认与计算加速最终交接

两阶段计算、统计与必要核验完成：CONFIRMATION_PASS；EXACT_BUT_BELOW_2X_TARGET。
服务器：aita_xushijie@172.19.164.161；根目录/aita_data/home/aita_xushijie/work/grasp/eg_lg1_c4_confirm_speed_v1_20261020。
分支：codex/eg-lg1-c4-confirm-speed-v1-20261020；https://github.com/rollinpioneer/graph/tree/codex/eg-lg1-c4-confirm-speed-v1-20261020。最终Git发布回执待写入results/archive_receipt.json。
C4 G−O AP+2.3135，95%CI[1.6581725313436673, 3.082495007195055]；Top1+3.3420pp，95%CI[1.5190972222222223, 5.360243055555555]。
单CPU几何中位数1.385078→1.072328秒（1.292倍）；96帧所有输入/输出精确一致。
C4已使用，禁止再当未见确认集。未新增训练/监督标签或调融合，未自动进入下一轮。原方法/权重只读。所有计算进程已结束后核验并退出本轮SSH37582；本地结果目录保存实际退出回执。
详见FINAL_REPORT.md，逐场景结果和所有必要验证在results。原始大文件留服务器，全索引由summary文件散列定位。
