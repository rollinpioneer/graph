# r2 Wave 1 停止记录（E4 恢复，补偿重跑）

- 结论：r2 首批 2 个 attempt 已预留（RESERVED），但 worker 进程在约 1 秒内退出（exit 1），**从未 claim，未创建环境、未 reset、未执行 skill、无 provider 调用**。attempt 收尾为 UNKNOWN（已保留）。
- 原因（执行侧错误，第二次）：协调器启动 worker 子进程时没有转发 `--round r2`，worker 在 r1 目录的登记里找不到 r2 的 branch_id，报 `witness execution requires frozen branch registration`。
- 账本：r2 physical_witness_episodes used=2 / cap=8；r1 为 2/8（2 FAILED）；原 8/8。累计已预留 8+2+2=12，累计上限 18。
- 调度：故障后停止，未预留、未启动其余 6 个分支。
- 代码修复（已提交，不改变实验方法）：worker 命令统一由 `_worker_cmd` 生成并转发 round；run_wave 在预留任何额度前，先用完全相同的命令行执行 `worker --dry-run`（只解析登记与静态预检，不预留、不 claim、不建环境）。对当前 r2 登记做 dry-run 已通过。
- 后续动作需要用户决定，见对话。
