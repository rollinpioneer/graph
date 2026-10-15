# EconomicGrasp V1：完整原始数据通过，监督准备持续中

快照 2026-10-15T04:14:54.654677+00:00：原始数据PASS；Graspness 9868/25,600。GPU原路径与场景缓存四帧一致；CPU预处理与GPU对照（原四帧、另12场景36帧）数组和文件哈希一致。完整监督、真实Smoke、10epoch/64,000更新、三划分AP和Bad Case尚未完成，实测AP保持null。

已保留原GPU生成7,820帧，4个CPU任务继续；暂停与有意取消仅涉及本卡预处理。用户已授权共享GPU0、1并要求记录负载。所有阈值、模型和训练配方保持；CPU准备适配与抽样验证局限完整披露于data_report.md及对应代码证据。

仅正式长时训练启用定时监控，其余阶段持续执行。分支 https://github.com/rollinpioneer/graph/tree/codex/eg-full-repro-v1-20261015 ，服务器输出 `/home/xushijie3/work/eg_full_repro_v1_20261015/runs/EG_FULL_V1`。仍为部分完成，不能宣称复现成功。
