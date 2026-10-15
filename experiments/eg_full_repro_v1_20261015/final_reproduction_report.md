# EconomicGrasp V1：准备与Smoke通过，正式训练运行中

快照2026-10-15T05:24:57.570602+00:00：原始完整数据PASS；25,600帧Graspness、100场景Economic和全量监督核验PASS；真实8样本两个batch4更新、有限损失/梯度/参数更新和模型+优化器重载PASS。

原模型、seed20261015、随机初始化、10epoch/batch4/Adam0.001/官方cosine，目标64,000更新。首次9更新后文件句柄上限失败且无checkpoint，完整保留；提高本卡子进程NOFILE后同配方同种子恢复，快照尝试2为141更新，GPU0共享运行，用户授权和负载均留痕。尚无完整epoch checkpoint或AP，不能称完整复现成功。

CPU监督准备优化与GPU对照（原4帧、12场景36帧）数组及文件哈希一致，全部监督核验通过；抽样对照并非全量双后端验证。目录竞态和检查器错误已定位并留存；模型/损失/阈值/主要配方未改。

用户最新要求后台训练、不挂监控，economicgrasp-v1已PAUSED，不自行恢复。必要实验日志与共享负载记录继续。按近期速度初估训练还需约16小时，2026-10-16凌晨约05:00台北完成，实际随负载变化；测试分析耗时另计。训练退出停在后续入口，下次接续最终checkpoint/64,000更新检查、三划分完整官方评价和探索性Bad Case。分支 https://github.com/rollinpioneer/graph/tree/codex/eg-full-repro-v1-20261015 ，产物 `/home/xushijie3/work/eg_full_repro_v1_20261015/runs/EG_FULL_V1`。完整日志、数据和权重留服务器，轻量证据与补丁归档Git。

发布前最新快照：尝试2为321个连续更新，仍在后台运行。
