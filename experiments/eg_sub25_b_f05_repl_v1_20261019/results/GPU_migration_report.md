# GPU迁移结果（2026-10-19）

已从GPU6切换到GPU1，沿用同一EG-B实验和seed20261015。第7轮Checkpoint（476def0dc78626bf53ef7d30489baea4bea2e64158532395a689dd776374c510）保留11200更新，原始train.py恢复模型、Adam、Python/NumPy/PyTorch及单卡CUDA随机状态，再训练第8–10轮；batch4、原Cosine和目标16000更新不变。未训练第二个骨干。

切换前重新采样3次GPU1均为0%利用率、8798MiB空闲，满足6GiB余量和均值≤60%条件。已通过nvidia-smi核验新训练PID2849958实际位于GPU1。原训练和等待控制器按PID身份核验后有意退出；旧资源回执中的-15退出码是计划切卡，不是数值故障。其他任务未终止。

Checkpoint保存到停止原进程之间，已记录4次未入Checkpoint的更新，原始完整指标文件留在/aita_data/home/aita_xushijie/work/grasp/eg_sub25_b_f05_repl_v1_20261019/runs/B/migration_original_GPU6/batch_metrics_unmodified.jsonl，SHA256=0191d7b169344b21209ee6f1773181836f70f1665d727a52520e4418f79b204a；规范训练流从11200恢复，重放4次已记录尾部更新，另有至多1次未记录的在途更新可能需要重算。未覆盖既有权重。

续训校验：第8轮已推进90步，所有分项损失有限、步号连续，源Checkpoint哈希不变。30秒窗口GPU6 0.832步/秒→GPU1 1.697步/秒，约2.04倍。短窗口预计骨干结束2026-10-19T21:56:02.716054+08:00，共享负载变化可改变ETA，后续标签/头训练/C3另计。

新迁移控制器2806952；训练PID2849958；依赖流水线PID2849959。累计GPU8小时及其他预算不变，后台继续，无定时监控。
