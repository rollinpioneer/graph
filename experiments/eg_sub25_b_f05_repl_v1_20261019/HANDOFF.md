# EG-SUB25-B-F05-REPL-V1运行交接
分支codex/eg-sub25-b-f05-repl-v1-20261019；继续沿用。根目录/aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019，gpu04。
EG-B及3个1000步评分头均训练完成，禁止再启动训练。最终骨干SHA 533d065386e96a2f76ad2317f0376321a3d2c350a88027d8310e9fd7cd99bc98。
原流水线在评分接口验证误传16候选处技术停止；已有三个权重未变，以1024候选完成验证，0追加训练。旧记录见results/head_validation_recovery。
新接续控制器PID3380582，脚本code/continue_after_head_validation.py；日志logs/continue_after_head_validation.log。它只执行已有头验证及C3后续，不训练。状态results/pipeline_status.json。
C3_method_lock.json已冻结，评价结果不得调参。当前A导出，A/B各3072帧；后续24CPU场景评价。实际GPU需读每阶段assignment，配置最多8进程不代表已使用8。
完成状态COMPLETED_PENDING_GIT_ARCHIVE后核验并最终归档；TECHNICAL_STOP保留现场并查日志，不自动追加训练。SSH结束只退出客户端，不结束后台任务。

用户最新要求（2026-10-19）：启用本线程15分钟监控。自动化id eg-sub25-b-f05-15，状态ACTIVE，每15分钟检查阶段、真实并行、错误及资源；完成交付后暂停。此前“无定时监控”约定由本次明确要求取代。旧economicgrasp-v1和lab保持暂停，未复用旧节点。当前A预测3072/3072完成，B预测进行中；CPU评价最多24路将按空闲资源启动。

2026-10-19 22:55最新：A/B两套C3 GPU预测均已完成，各3072帧，共6144；GPU推理任务已无待执行。官方几何/碰撞/摩擦评价正在24个独立场景任务全并行运行，均CUDA_VISIBLE_DEVICES为空，控制器3380582。无需追加GPU任务；15分钟监控保持ACTIVE，不能重复启动评价。

15分钟心跳检查2026-10-19T23:06:49.499663+08:00：24路评价存活并推进，完成1281/6144帧，0/24完整场景；GPU累计4.628/8小时，CPU含活动任务11.336/50worker小时，新增运行产物10.404/25GiB。无新报错，不汇总不完整AP。
