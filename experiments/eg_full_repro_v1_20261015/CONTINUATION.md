# EG-FULL-REPRO-V1 持续执行交接

所有后续工作沿用本卡分支 `codex/eg-full-repro-v1-20261015`，服务器根目录 `/home/xushijie3/work/eg_full_repro_v1_20261015`，Git检出为 `archive_worktree`。每轮复用一个保活SSH长会话，最后关闭自己的连接，后台作业保持运行。

## 当前任务

数据获取已经恢复；不需要Google Drive账号。镜像全TAR SHA与独立官方ZIP逐文件CRC双重核验，随后打开全部图像/meta和原始标签。模型官方CRC已PASS；其他组件正在下载。

`data_recovery/data_driver.py`负责下载、SHA验证、选择性解包及删除本轮已核验TAR；`raw_validation_driver.py`在全量解包结束后自动执行官方CRC比对和全部原始内容检查；`execution_pipeline.py`正在等待RAW验证PASS，然后按本卡顺序自动继续。实际状态读远端 `data_recovery/PIPELINE_STATUS.json`、`DATA_ACQUISITION_STATUS.json`、`RAW_VALIDATION_DRIVER.json` 及 `logs/`；归档SNAPSHOT只是本次提交时点，不能作实时依据。进程身份见 BACKGROUND_PROCESS_IDENTITIES.json，核对/proc启动时间和命令防PID复用。不要重复启动正在运行的管理器或作业。

## 后续已实现的门

1. scene0和99前2帧：原计算路径与场景缓存生成标签逐元素、dtype、文件哈希完全相同后才启用缓存。
2. 3个当前空闲GPU按场景模3分片生成全部Graspness，再用2个CPU作业按场景模2生成全部Economic标签；阈值/筛选/网络/损失均保持官方。写入原子化，恢复须有摘要回执，拒绝未知覆盖。
3. 所有25600帧的工作区掩码对应、标签摘要、Economic索引/数值/对象点坐标；scene0/50/99固定原始监督语义探针。
4. 真实8个预登记训练样本、两个batch4更新；所有分项损失/梯度/参数检查，模型及优化器checkpoint重载。Smoke权重不能进入正式训练。
5. 唯一种子20261015，原训练脚本10epoch/batch4，严格64000更新，逐batch/epoch日志、原子checkpoint、RNG保存。只在全部门通过后启动。
6. 检查10个checkpoint、日志更新连续性、官方epoch cosine及最终真实帧有效解码，然后按seen/similar/novel完整测试，proc4，collision_thresh0；保存原始预测、实际采样输入、AP张量、分项计时和抓取数。

GPU在每个作业启动时实查空闲，无其他compute进程且显存占用<=100MiB/利用率<=1%；不占用历史空闲卡，不干扰别人。resource_jobs保存独立进程组资源采样及退出回执。三分片会等待当前可用资源，不能将正常等待当失败。所有大文件写入保留24GiB磁盘余量，触及门槛应停在该阶段，不能填满盘或删旧资产。

## 评价完成后仍需处理

执行器以 `EVALUATED_PENDING_BAD_CASE_AND_FINAL_ARCHIVE` 结束，不能称全卡完成。继续按卡选每划分3最低AP+1中位场景、每场景2最低帧+1中位帧，最多36帧，按ID破同分；利用保存的确切输入重放中间输出，用官方eval_grasp重新诊断所选帧并核对原AP。保存RGB/depth/抓取姿态图、分数/碰撞/质量和场景AP，人工检查视觉与定量证据，明确探索性选择和机制不确定性。最后补齐所有报告、checkpoint/数据/日志哈希索引和研究决策意见，不增加模型训练。

## 故障/恢复

门失败先读对应log和receipt，保留现场、定位原因。允许确定性工程修复；不改主要配方、模型、监督阈值、AP或跳样本。执行器故障不会自动重新训练；正式训练如中断优先从本实验最新完整checkpoint恢复optimizer/epoch/RNG，保留中断尝试、日志尾与失败checkpoint。不得因指标差而重训换参。source patch SHA在实际数据门后记录于PIPELINE_STATUS；实质改动需更新补丁及说明。

用户最新要求：只有正式长时训练期间才启用每20分钟的监控 `economicgrasp-v1`；下载、核验、监督生成、短检查、测试和分析阶段持续在当前轮执行，不以定时跟进替代持续工作。当前监控已PAUSED，训练启动后才启用，训练结束时暂停并持续接续评价。远端nohup作业独立继续。科研进展优先，只修复实际阻塞或影响科研结论的问题，不扩展执行框架。
