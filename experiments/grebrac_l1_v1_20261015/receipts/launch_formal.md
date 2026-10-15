# 正式启动记录

- 启动时间：2026-10-15 10:56:48–10:57:03 UTC，gpu03，账号 xushijie3。
- 启动脚本：`code/launch_formal.sh`（远端 `scripts/launch_formal.sh`，sha256 `a03e181c…866c`，与仓库一致）。每条作业用 `setsid -f` 脱离 SSH 会话。
- 汇总脚本：`code/summarize.py`（远端 sha256 `6450d42f…e8ef`，与仓库一致）。
- 启动前资源：可用内存 351 GiB，数据盘剩余 88 GB，1 分钟负载 269（112 核，多人共享）。
- 放置原则：同一任务同一种子的 R/G 两臂放在同一张卡上同时运行，使两臂的运行条件尽量一致。避开了几乎满载的 GPU 3（显存剩余约 2 GB）和 GPU 6（利用率 100%）。

| 作业 | GPU | GPU UUID 前缀 | 绑核 |
|---|---|---|---|
| R_task1_s0 / G_task1_s0 | 7 | GPU-71e35dc1 | 0-5 / 6-11 |
| R_task1_s1 / G_task1_s1 | 7 | GPU-71e35dc1 | 12-17 / 18-23 |
| R_task1_s2 / G_task1_s2 | 4 | GPU-dc70602e | 24-29 / 30-35 |
| R_task1_s3 / G_task1_s3 | 4 | GPU-dc70602e | 36-41 / 42-47 |
| R_task4_s0 / G_task4_s0 | 2 | GPU-eb22e795 | 48-53 / 54-59 |
| R_task4_s1 / G_task4_s1 | 2 | GPU-eb22e795 | 60-65 / 66-71 |
| R_task4_s2 / G_task4_s2 | 1 | GPU-61a23cf4 | 72-77 / 78-83 |
| R_task4_s3 / G_task4_s3 | 5 | GPU-1f8161f4 | 84-89 / 90-95 |

所有 16 条作业的 `stdout.log` 都显示从官方 `rebrac_params.csv` 的 `humanoidmaze_medium_navigate` 行恢复了超参数。

## 重启记录（按冻结规则：非算法原因的启动失败可重启一次）

- 11:30 UTC 的检查发现 5 条 task4 作业在 11:03:33–11:03:35 UTC 以 exit 1 退出：R_task4_s0、G_task4_s0、R_task4_s1（GPU 2），R_task4_s2、G_task4_s2（GPU 1）。
- 原因：`main.py:207` 把数据集搬上 GPU（`dataset_on_gpu=true`，默认值）时 `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 552000276 bytes`。这发生在第一次更新之前，属于共享 GPU 显存不足，与算法无关。其余 11 条作业不受影响。
- 处理：失败目录原样移到 `runs/failed_attempt1/`，不删除。配置不变，只换到当时空闲显存更多的卡上，每条重启一次，绑核不变：

| 作业 | 新 GPU | 绑核 | 重启时间（UTC） |
|---|---|---|---|
| R_task4_s0 / G_task4_s0 | 0 | 48-53 / 54-59 | 11:30:51 / 11:31:06 |
| R_task4_s1 | 2 | 60-65 | 11:31:28 |
| R_task4_s2 / G_task4_s2 | 6 | 72-77 / 78-83 | 11:31:40 / 11:31:53 |

- 11:36 UTC 确认 16 条作业都已写出 `train.csv`，进入训练。这 5 条不能再因同类原因重启；若再次失败，会报告给用户。
- R_task4_s1 与 G_task4_s1 不在同一时间、同一张卡上运行。种子决定初始化和批次序列，换卡只影响墙钟时间，不影响配对。

## 汇总脚本修正（无正式结果之前）

`success.csv` 中的数值已经是百分数（`utils/evaluation.py:40`：`100*eval_info["success"]`）。第一版 `summarize.py` 又乘了 100，现已去掉，并加上墙钟字段。修正后 sha256 为 `7049958c…c3c3`，远端副本一致。修正时还没有任何作业跑完，汇总结果没有用过旧版脚本。
