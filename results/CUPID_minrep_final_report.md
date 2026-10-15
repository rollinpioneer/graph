# CUPID 最小复现结果

## 1. 最终状态

**FAIL**

用户明确授权忽略 6 小时资源门槛后，实验继续到正式训练阶段。正式训练在第 16 轮因 DataLoader worker 被 `SIGKILL` 终止，命令返回 1。按照“正式训练失败立即停止”和“不得启动第二次正式训练”的规则，本次未继续重试、未从检查点恢复，也未进入试跑或影响计算阶段。

## 2. 固定设置与授权变更

- 官方提交：`2941eba9427a3398d3cbe26b10c01e3ece56bd18`
- 任务：RoboMimic Square-MH，低维状态
- 策略：Diffusion Policy
- 训练种子：0
- 数据集：300 条示范，检查通过
- 环境：独立 Conda 环境 `cupid`
- PyTorch：1.12.1
- CUDA：11.6
- GPU：物理 GPU 1，A100-SXM4-40GB
- W&B：离线
- 时间门槛：用户明确授权忽略 6 小时限制

任务、种子、数据划分、正式轮数 1751、检查点策略、试跑种子和分析阈值均未改变。物理 GPU 0 因共享进程占满后，经用户授权改用物理 GPU 1。

## 3. 门槛零与正式训练

- 50 轮短跑：完成
- 短跑墙钟：944 秒
- 短跑最后轮数：49
- 短跑有限损失记录：9,850 条
- 短跑 `latest.ckpt`：存在
- 短跑训练进程峰值显存：1,778 MiB
- 正式训练启动时间：2026-07-21 09:45:14
- 正式训练失败时间：2026-07-21 09:52:03
- 正式训练失败前墙钟：409 秒
- 失败轮次：第 16 轮训练中
- 直接错误：`RuntimeError: DataLoader worker (pid 3484358) is killed by signal: Killed.`
- GPU 失败时显存：约 9.0GB 总占用，非 CUDA OOM
- 主机诊断：约 393GiB 可用内存，`/dev/shm` 约 252GiB 可用；当前权限下未能获得内核 OOM 记录

正式输出目录产生了 epoch 0 检查点和部分日志，但它们不是完成的 1751 轮正式策略，不能用于后续试跑或分析。

## 4. 按要求汇报

1. 最终状态：FAIL
2. 失败门槛：正式训练在第 16 轮因 DataLoader worker 被 SIGKILL 失败
3. 代码提交：`2941eba9427a3398d3cbe26b10c01e3ece56bd18`
4. 训练墙钟时间：正式训练 409 秒后失败；50 轮短跑 944 秒
5. 保存的成功/失败试跑数量：0 / 0，未进入试跑阶段
6. 影响计算墙钟时间：未启动
7. 10 条与 50 条排序相关差：N/A
8. 10 条与 50 条前 20% 名单重合差：N/A
9. 是否允许进入交叉拟合控制变量改进：否
10. 最终报告绝对路径：`/home/xushijie/CUPID/results/CUPID_minrep_final_report.md`
11. 关键日志绝对路径：`/home/xushijie/CUPID/logs/formal_train1751.log`
12. 累计 GPU 小时：约 0.376 小时（短跑加正式训练失败前）

## 5. 关键文件

- 正式训练日志：`/home/xushijie/CUPID/logs/formal_train1751.log`
- 正式训练时间：`/home/xushijie/CUPID/logs/formal_train_timing.txt`
- 正式训练显存采样：`/home/xushijie/CUPID/logs/formal_train_gpu_memory.csv`
- 失败系统诊断：`/home/xushijie/CUPID/logs/formal_train_failure_system_diagnosis.log`
- 短跑日志：`/home/xushijie/CUPID/logs/gate0_train50_gpu1.log`
- 数据集检查：`/home/xushijie/CUPID/logs/dataset_check.log`
- 状态文件：`/home/xushijie/CUPID/status/formal_train.fail`

本阶段未保存 100 条固定试跑，未运行 TRAK 影响计算，未运行重复子集分析，也未运行筛选后的第二次训练。
