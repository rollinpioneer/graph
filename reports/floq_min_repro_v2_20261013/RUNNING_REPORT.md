# FLOQ-MIN-REPRO-V2 运行回执

状态：P0/P1 通过；两条正式训练运行中，尚无 2M 成绩。

固定官方 SHA：3d60e638b42bbef018c56cb1199ff37c3470520d。分支：[codex/floq-min-repro-v2-20261013](https://github.com/rollinpioneer/graph/tree/codex/floq-min-repro-v2-20261013)。服务器 ROOT：/home/xushijie3/work/floq_min_repro_v2_20261013。

Snapshot 2026-10-14T00:01:28.930281 (server time).

| Run | GPU | updates/2M | phase | own VRAM peak MiB |
|---|---:|---:|---|---:|
| floq-0 | 3 | 255700 | TRAINING_RUNNING | 954 |
| floq-1 | 6 | 260000 | NATIVE_EVALUATING | 954 |

## P1 工程证据

独立 5k 更新、50 回合原生评价（第1步）、checkpoint save/restore 通过；参数/优化器/JAX RNG树逐字节身份一致，固定采样 key 的动作完全一致。所有 Actor/Flow/Scalar Critic 梯度有限，Target Flow 梯度为0；目标更新按原版一拍延迟实现，误差0。试跑不计入正式成绩。

P1 GPU 驻留 0.06664 小时，进程生命期 0.06871 小时，显存峰值 980 MiB，进程树 RAM 峰值 2.055 GiB。CPU 真值见 p1/cpu_time.txt；GPU 驻留来自约2秒轮询，存在采样边界误差。

P1 原生评价耗时 104.96 秒；5k 训练循环日志耗时 153.16 秒，扣除该评价后 48.20 秒，约 103.7 更新/秒。此数值仍包含循环日志/验证等开销，非纯内核计时。以此粗估每条2M更新 5.36 小时，加101次原生评价按P1失败回合估计约 2.94 小时；共享资源与策略回合长度会变化，不作为预算或停止阈值。

## 评价与统计限制

Task1 作者参考94±4%，3 seeds；作者检查点聚合窗口尚未核实。按卡片授权冻结主结果为2M原生50回合，报告两个种子、均值、总体/样本标准差；不能使用最高分。完整训练之前不标记A/B/C，不声称完成论文复现。详见 EXECUTION_FREEZE.md 与 CRITIC_INTERFACES.md。

已有FQL和GFP共四条1M训练及全部Matched评价均完成。它们的不同更新数、配置/环境和评价口径是混杂因素，不单独归因于Critic。旧实验归档分支本轮不推送。

## 定时监控与后续交接

Codex当前任务的heartbeat Lab 全训练巡检（id=lab）已启用，每30分钟运行。脚本已实测覆盖所有本账号训练/评价/GPU进程及已登记任务，检查损失有限性、验证CSV、checkpoint、Matched完成数量与资源。电脑及Codex应用须保持运行；服务器的已脱离SSH的正式队列不依赖客户端保持连接。

后续读取 ROOT/receipts/monitor/latest.json、ROOT/receipts/queue.json 与各 runs 的 status/progress/native_evaluations/checkpoint_index。最终汇总逐种子2M原生结果、全曲线、1M/2M文件散列及GPU/CPU成本；不再训练FQL、不启动Q-Flow、不新增条件或自动重启。新卡后续归档沿用上述分支。

大型权重、全stdout、原生逐回合回执和资源逐样本日志留 ROOT/runs；Git仅保存轻量配置、摘要、回执、源码身份和相关工程脚本。
