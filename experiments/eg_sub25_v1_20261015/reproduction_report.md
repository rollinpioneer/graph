> Historical training-completion snapshot. Current final outcome: [final_reproduction_report.md](final_reproduction_report.md) and [bad_case_report.md](bad_case_report.md).

# EG-SUB25-V1：训练完成并通过最终检查，评价待执行

2026-10-15台北16:44:05，唯一seed20261015的25%子集训练正常退出（原生返回码0）。50场景x128帧=6400，10epoch、batch4、原Adam0.001和epoch cosine，严格16000连续更新、每epoch1600，全部记录损失和诊断指标有限。全程使用GPU1单卡共享，没有新增seed、改参数或重新正式训练。

墙钟2.8673小时（约2小时52分钟），epoch计时合计2.8607小时；Torch allocated峰值2.562GiB，采样驱动级自有显存峰值3656MiB，进程组RSS采样峰值4.690GiB。驱动资源每15秒采样，峰值是已观测值，完整序列留服务器。

Epoch平均总损失从4.681914降至3.267090，全部7项分损失和耗时见epoch_results.csv/epoch_metrics.jsonl。原cosine与源码补丁身份核验通过。10个checkpoint均严格加载，模型参数有限且优化器状态存在；最终开发scene8/frame0真实推理输出1024x17有限抓取表示，旋转正交与行列式检查通过。该帧仅作有效性检查，没有计算AP。原评分为logit加权和，未改变解码方式。

训练权重与完整日志保存在 `/home/xushijie3/work/eg_sub25_v1_20261015/runs/EG_SUB25_V1/train` 及 `/home/xushijie3/work/eg_sub25_v1_20261015/logs`；每个checkpoint的SHA256、源码/划分身份、检查回执见TRAINING_COMPLETE.json、train_config.json和receipts/。最终权重为economicgrasp_epoch10.tar，未上传Git。

原始数据、所选6450监督哈希与80帧Smoke此前PASS。顺序Smoke的空候选FPS异常及改为官方shuffle/worker播种后的PASS均留存，阈值0.1、正常输入算法/损失未改；正式16000更新未触发该保护。旧全量训练结果已按用户要求删除。

开发集10场景完整AP、固定官方Seen119/122和Novel188/189的子集/官方预训练配对诊断，以及最多12个探索性Bad Case尚未执行，AP为null。当前仅完成子集训练与有效性核验，不能称完整论文复现或已获得泛化性能。后续按冻结evaluation_protocol.json继续，不换场景或自动追加75场景训练。定时跟进保持PAUSED。
