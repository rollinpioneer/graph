# 2026-10-15最新状态：训练已完成并核验PASS

训练于台北16:44:05正常退出，10epoch/16000更新，10个checkpoint严格加载、最终真实开发帧有效解码PASS。实时状态为TRAINING_VALIDATED_PENDING_EVALUATION，入口阶段DEV_AP_AND_PAIRED_DIAGNOSIS_PENDING。正式训练和训练管理器均结束，不要再次启动run_card*.py或重训。下次直接按已冻结evaluation_protocol.json执行开发AP、两模型官方场景配对诊断与Bad Case；当前没有AP结果。详见reproduction_report.md、TRAINING_COMPLETE.json、epoch_results.csv。定时跟进保持PAUSED。

以下为训练启动时的交接记录，状态应以上述最新记录为准。

# EG-SUB25-V1 当前交接

新卡分支codex/eg-sub25-v1-20261015，服务器根目录 `/home/xushijie3/work/eg_sub25_v1_20261015`；归档工作区仍为 `/home/xushijie3/work/eg_full_repro_v1_20261015/archive_worktree`，当前已切本卡分支。旧全量训练按用户要求在1000更新取消，自训练结果和完整训练日志已删除，停止/删除证据归档。旧根目录中的数据、监督、Conda/CUDA、官方预训练权重仍被新卡复用，绝不能删除旧根目录。

冻结划分见frozen_split.json，SHA bdf75f905271f4672de2407da3c095f257082fa46cd4099d9f16ebecbd258157；50x128=6400训练，10x256开发，其余40暂不训练，物体覆盖40种。元数据选择未用模型结果，不能根据性能调整。原始/全部监督已有完整核验，这次6450个选中监督文件哈希复查PASS。Dataset和预处理显式限制清单，复用缓存不重复生成。

80帧Smoke（5x16，20个batch4更新）已PASS。初次顺序Smoke及诊断确认原FPS空候选非法访问；改为相同80帧且对齐官方shuffle和worker初始化后PASS，仅加空候选明确报错，未改变阈值/模型/损失或补点。失败证据和修改前代码保留，正式若再触发空候选应停止受影响阶段，不跳样本或改阈值绕过。

当前唯一正式训练seed20261015，GPU1，随机初始化、原模型/损失，10epoch/batch4/Adam0.001/points20000/voxel0.005/官方epoch cosine，1600更新每epoch、总16000。Smoke不用于初始化。运行控制器run_card_attempt2.py，状态data_recovery/PIPELINE_STATUS.json，训练receipt为receipts/formal_training.json，输出runs/EG_SUB25_V1/train；过程身份和时间样本见ETA_SAMPLE_START.json及LAUNCH_ATTEMPT2.json。检查/proc启动时间防PID复用，不重启旧run_card.py或重复训练。

用户要求后台运行、不挂定时监控，economicgrasp-v1保持PAUSED，不自行恢复。必要资源/共享负载日志在后台继续；GPU0/1共享已有授权，其他卡须当前空闲。结束本轮关闭自己的SSH客户端，不终止训练清理连接。

run_card_attempt2.py在训练结束停在TRAINING_EXITED_PENDING_FINAL_CHECKS。下次继续先验证10个checkpoint严格加载、16000连续有限更新、每epoch1600和官方cosine、最终真实帧有效预测、日志/权重哈希和实际资源。中断优先从同实验最新checkpoint恢复optimizer/epoch/RNG，不换seed/配方，不以不理想指标重训。

后续按evaluation_protocol.json做开发10场景完整256帧AP；再在Seen119/122、Novel188/189完整256帧分别测试子集模型和官方全量预训练模型，保持官方GraspNetAPI定义并保存同一输入用于配对比较。开发集来自官方训练划分，官方预训练不能在其上被声称独立泛化；四个官方场景是探索性开发材料，不称完整Seen/Novel AP。

最多12个配对Bad Case（每固定官方场景2低AP+1中位帧，按ID破同分），保存RGB/depth/抓取图/分数/碰撞与质量/场景AP/相关中间输出，区分子集失败官方成功、双方失败、点合理但方向或评分异常；这些观察不直接证明机制。无需自动扩大75场景或开发新方法。最终补齐报告与实际资源，相关代码和轻量结果沿用本分支提交推送核验，大权重/完整日志留服务器。
