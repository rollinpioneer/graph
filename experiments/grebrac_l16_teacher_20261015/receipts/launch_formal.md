# 正式启动记录

- 启动时间：2026-10-15 16:46:50–16:48:36 UTC，gpu03，账号 xushijie3。
- 启动方式：远端 `scripts/launch_all.sh`（sha256 前缀 `90ed4409413925b8`）用 `setsid -f` 脱离 SSH 会话，逐条调用 `scripts/launch_formal.sh`（与仓库 `code/launch_formal.sh` 一致），间隔 15 秒。
- 启动前资源：1 分钟负载 50（112 核，多人共享），可用内存 382 GiB，数据盘剩余 71 GB。
- 启动前空闲显存：GPU 1 40.5 GB、GPU 7 32.4 GB、GPU 5 30.8 GB、GPU 4 29.9 GB。避开了 GPU 3（2.7 GB）、GPU 6（利用率 100%）和 GPU 0（上一轮出现过他人显存突增）。
- 放置原则：同任务的相邻种子两两放在同一张卡上，每卡 2 条，每条单独绑 6 核。

| 作业 | GPU | GPU UUID 前缀 | 绑核 | 启动（UTC） |
|---|---|---|---|---|
| T_task1_s0 | 1 | GPU-61a23cf4 | 0-5 | 16:46:50 |
| T_task1_s1 | 1 | GPU-61a23cf4 | 6-11 | 16:47:05 |
| T_task1_s2 | 5 | GPU-1f8161f4 | 12-17 | 16:47:20 |
| T_task1_s3 | 5 | GPU-1f8161f4 | 18-23 | 16:47:36 |
| T_task4_s0 | 4 | GPU-dc70602e | 24-29 | 16:47:51 |
| T_task4_s1 | 4 | GPU-dc70602e | 30-35 | 16:48:06 |
| T_task4_s2 | 7 | GPU-71e35dc1 | 36-41 | 16:48:21 |
| T_task4_s3 | 7 | GPU-71e35dc1 | 42-47 | 16:48:36 |

- 所有作业的 `stdout.log` 都显示从官方 `rebrac_params.csv` 的 `humanoidmaze_medium_navigate` 行恢复了超参数。
- 后 5 条作业的数据加载比前 3 条慢 2–3 分钟（8 条同时解压同一数据文件，服务器负载升到约 80），但没有报错。16:52:45 UTC 确认 8 条作业都在训练，没有错误，没有重启。

## 配对对照

R 臂不重跑，汇总时在 `runs/paired/` 下为 T_* 和 L1 的 `runs/formal/R_task{1,4}_s{0..3}` 建软链接，再运行 `scripts/summarize.py`。
