# EG-FULL-REPRO-V1 当前执行交接

服务器根目录 `/home/xushijie3/work/eg_full_repro_v1_20261015`，Git archive_worktree，本卡分支codex/eg-full-repro-v1-20261015，每轮保活SSH长会话并在结束核验客户端退出，不终止实验清理连接。历史执行器不可盲目重启，禁止重复训练。

## 当前状态

原始完整Kinect数据、25,600帧Graspness、100场景Economic和全量监督核验PASS；真实8帧两个batch4更新、七损失/梯度/参数更新、模型与优化器checkpoint重载PASS。全部缓存保留；不重复下载或预处理，不需要Google登录。

正式训练恢复尝试2运行，唯一种子20261015，随机初始化、原模型/损失、10epoch、batch4、Adam0.001和原epoch cosine、目标64,000更新。实际实时状态读data_recovery/PIPELINE_STATUS.json、receipts/formal_training_attempt2.json及runs/EG_FULL_V1/train/batch_metrics.jsonl，原生进程身份见TRAINING_LAUNCH_VERIFICATION.json和回执，核对/proc启动时间。

初次训练9更新后NOFILE1024耗尽且无checkpoint；train_attempt1_nofile_failure、日志和回执完整保留。仅升本卡进程软上限65536，恢复同种子同配方，属于同实验技术恢复，不是新seed或超参试验。若再次中断，优先从本实验最新完整checkpoint恢复optimizer/epoch/RNG；没有checkpoint才如实记录相同配方重启，不能更换参数。

GPU0、1已获用户明确共享授权，启动时至少24GiB空闲；其他卡须当前空闲。resource_jobs记录GPU显存/利用率、共享compute进程、自有资源和退出回执；不干扰他人，保留24GiB磁盘余量。当前训练用GPU0；CPU准备对照与目录竞态/检查器修复见对应证据，官方训练源码补丁仍为train_config.json登记的e084db...，算法、主要配方和监督阈值未改。

## 长时训练与后续

用户最新要求：后台训练即可，给预计完成时间，不必挂监控。economicgrasp-v1已PAUSED，不得自行恢复定时跟进；必要的实验日志和共享负载记录在后台继续。当前训练继续运行，本轮归档后关闭SSH。当前初估训练于2026-10-16约05:00（台北）完成，按近期约1.13更新/秒；短窗口及共享负载导致不确定性，见TRAINING_ETA_AND_MONITOR_POLICY.json。训练结束停在后续检查入口，等待用户下次要求接续。

训练管理脚本resume_training_nofile.py在训练退出后停在TRAINING_EXITED_PENDING_FINAL_CHECKPOINT。该状态且训练receipt PASS后，运行data_recovery/resume_evaluation_after_training.py：检查10个checkpoint、64,000连续更新、官方cosine和最终真实帧有效预测，随后依次完整Seen/Similar/Novel（各30x256，官方test.py和API，proc4，collision_thresh0，保存确切推理输入），最后核验23,040预测、AP张量、90场景CSV。脚本只在前门通过后运行，不覆盖既有预测；失败保留现场并定向恢复。

评价后仍需有界探索性Bad Case：每划分3最低AP+1中位场景，每场景2最低+1中位帧，最多36，按ID破同分；重放确切输入/中间输出，用官方eval_grasp核对AP/碰撞/质量，保存RGB/depth/抓取图、分数、场景AP，区分表现与未证实原因，不据此改网络重训。补齐最终报告、实际资源/日志/权重哈希索引和研究决策。

每轮明确文件清单暂存、提交同分支、用已登记部署密钥推送并核验远端SHA，同步本地交接。大数据/权重/完整日志留服务器。当前无完整训练权重或AP，不能称完整复现成功。
