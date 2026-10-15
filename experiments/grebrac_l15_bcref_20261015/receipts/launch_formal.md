# 正式启动记录

- 启动时间：2026-10-15 15:26:22–15:28:08 UTC，gpu03，账号 xushijie3。
- 启动方式：远端 `scripts/launch_all.sh`（sha256 前缀 `168dce736c864a1d`）用 `setsid -f` 脱离 SSH 会话，逐条调用 `scripts/launch_formal.sh`（与仓库 `code/launch_formal.sh` 一致），间隔 15 秒。
- 启动前资源：1 分钟负载 52（112 核，多人共享），可用内存 388 GiB，数据盘剩余 76 GB。
- 启动前空闲显存：GPU 5 32.4 GB、GPU 7 28.9 GB、GPU 4 27.3 GB、GPU 0 25.7 GB。避开了 GPU 3（2.7 GB）、GPU 1（14 GB）、GPU 6（利用率 91–100%）。
- 放置原则：同任务的相邻种子两两放在同一张卡上，每卡 2 条，每条单独绑 6 核。

| 作业 | GPU | GPU UUID 前缀 | 绑核 | 启动（UTC） |
|---|---|---|---|---|
| B_task1_s0 | 5 | GPU-1f8161f4 | 0-5 | 15:26:22 |
| B_task1_s1 | 5 | GPU-1f8161f4 | 6-11 | 15:26:37 |
| B_task1_s2 | 7 | GPU-71e35dc1 | 12-17 | 15:26:52 |
| B_task1_s3 | 7 | GPU-71e35dc1 | 18-23 | 15:27:07 |
| B_task4_s0 | 4 | GPU-dc70602e | 24-29 | 15:27:23 |
| B_task4_s1 | 4 | GPU-dc70602e | 30-35 | 15:27:38 |
| B_task4_s2 | 0 | GPU-4120c94f | 36-41 | 15:27:53 |
| B_task4_s3 | 0 → 2（重启） | GPU-4120c94f → GPU-eb22e795 | 42-47 | 15:28:08 → 15:30:07 |

所有作业的 `stdout.log` 都显示从官方 `rebrac_params.csv` 的 `humanoidmaze_medium_navigate` 行恢复了超参数。

## 重启记录（按冻结规则：非算法原因的启动失败可重启一次）

- B_task4_s3 于 15:29:21 UTC 以 exit 1 退出。原因与 L1 相同：`main.py` 把数据集搬上 GPU 时 `RESOURCE_EXHAUSTED: Out of memory while trying to allocate 552000276 bytes`，发生在第一次更新之前。当时 GPU 0 的空闲显存已从 25.7 GB 降到约 6.5 GB。
- 处理：失败目录原样移到 `runs/failed_attempt1/`。配置和绑核不变，只换到 GPU 2（当时空闲 25.6 GB），15:30:07 UTC 重启。该条不能再因同类原因重启。
- 15:31:49 UTC 确认 8 条作业都在训练，没有错误。

## 配对对照

R 臂不重跑，汇总时在 `runs/paired/` 下为 B_* 和 L1 的 `runs/formal/R_task{1,4}_s{0..3}` 建软链接，再运行 `scripts/summarize.py`。
