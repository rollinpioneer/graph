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
