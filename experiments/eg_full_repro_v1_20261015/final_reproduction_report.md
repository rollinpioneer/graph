# EconomicGrasp V1：数据获取恢复，完整复现仍在执行

环境和主要CUDA扩展已通过工程检查；官方权重加载及合成点云前向已通过。此前官方下载阻塞已找到可核验的数据镜像解决，当前完整下载进行中。88个物体模型的路径、大小、CRC已与官方ZIP逐项一致。

完整数据内容检查、监督生成、真实数据Smoke Test、10 Epoch/64,000更新、三划分官方AP及Bad Case尚未完成，不能将当前状态称为复现成功。全部实测AP保持null。

分支：`codex/eg-full-repro-v1-20261015`， https://github.com/rollinpioneer/graph/tree/codex/eg-full-repro-v1-20261015 。数据恢复证据见 data_source_recovery.md 和 data_recovery/；之前阻塞报告保留于 checks_history/phase0_*。大型原始文件和运行日志留在 `/home/xushijie3/work/eg_full_repro_v1_20261015`。
