# EG-FULL-REPRO-V1 持续执行交接

本卡分支 `codex/eg-full-repro-v1-20261015`，服务器根目录 `/home/xushijie3/work/eg_full_repro_v1_20261015`；Git为archive_worktree。每轮复用一个保活SSH长会话，结束关闭自己的客户端，不以结束实验来清理SSH。

原始数据全量核验PASS；无需Google登录。官方ZIP CRC与镜像全TAR SHA及完整数据内容检查均已通过，不重复下载。当前快照 2026-10-15T04:14:54.654677+00:00：Graspness 9868/25600，4个CPU分片运行，resume_cpu_preparation.py依次继续Economic和全量监督检查，成功状态PREPARATION_COMPLETE_PENDING_REAL_SMOKE。以服务器实时状态和回执为准。

旧execution_pipeline.py协调器和两个GPU标签任务已结束，不可盲目重启。新CPU适配/抽样逐哈希对照证据和有意取消原因见CPU_PREPARATION_ADAPTATION.json、CPU_KNN_CROSS_SCENE_CHECK.json、GPU_CONTENTION_PAUSE.json。现有7,820帧已保留，CPU继续原官方标签含义；全部监督PASS才可放行。

准备PASS后运行data_recovery/resume_training_after_preparation.py：它检查固定源码补丁身份、真实8帧两个batch4更新的Smoke（有限损失/梯度、参数更新、模型+优化器保存重载），然后唯一正式训练（seed20261015，随机初始化，10epoch/batch4/64,000更新）。Smoke权重不进入正式训练。此脚本当前尚未启动；禁止重复启动。训练中断优先恢复本实验checkpoint与optimizer/RNG，不能换参重训。

用户明确允许本卡共享GPU0、1并记录负载；GPU_SHARING_POLICY.json约束共享卡0/1启动时至少24GiB空闲。其他卡继续当前空闲检查。resource_jobs记录进程组、GPU显存/利用率和共享compute进程；不影响其他任务。每个大产物保留24GiB磁盘余量。

用户要求只有正式长时训练启用economicgrasp-v1定时监控（20分钟），目前PAUSED。准备、Smoke、测试、分析持续在当前轮执行。实际正式训练启动且首批更新有限后才激活，结束后暂停。

训练退出后，当前轮或下一次训练监控回合持续执行summarize_training.py检查10个checkpoint、64,000连续更新、官方cosine和最终真实帧预测，然后依次官方完整Seen/Similar/Novel（各30场景x256，API proc4，collision_thresh0，保存确切输入）。沿用execution_pipeline.py已注册函数，不能重跑其整条主流程。summarize_evaluation.py检查AP张量、23,040预测和90场景结果。

最后做有界探索性Bad Case：每划分3个最低AP+1中位场景，每场景2最低+1中位帧（最多36），按ID破同分；用确切保存输入重放中间输出、官方eval_grasp核对质量/碰撞/AP，保存RGB/depth/抓取图与分数，区分表现与未证实原因；不修改模型重训。补齐最终报告、哈希索引、训练耗时和峰值显存，沿用本分支提交轻量结果并推送核验。当前尚无本卡正式训练权重或AP，不能报完整复现成功。
