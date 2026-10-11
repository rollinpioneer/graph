# 代码、证据与复现

- 归档仓库：https://github.com/rollinpioneer/graph
- 本轮分支：`codex/pc-scale-02-20261011`；冻结登记提交 `d83c65c9e`，起点 `070b2bfa3863b93bb350578a57bfae91f7f5562a`。
- 服务器本轮目录：`/home/xushijie3/work/pc_scale02_20261011/experiments/pc_scale02_20261011`。
- 只读官方环境、代码和资产：`/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011`；大型官方权重、归一化数据及 IsaacGym 软件包不包含在 Git 归档中。

在已核验的 lab-xushijie3 上，CPU 离线重算入口：

```bash
cd /home/xushijie3/work/pc_scale02_20261011/experiments/pc_scale02_20261011
/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011/env/bin/python analyze.py
/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011/env/bin/python process_audit.py
```

这些命令需要 evidence_inventory.json 索引中的原始 NPZ 和完整 GPU 监控日志。仅有 Git 中的轻量文件可以检查端点、配置、散列、事件汇总和结论，但不能冒充重新执行仿真。

每个实际 GPU 作业的精确参数与启动时间见 `runs/<job>/launch.json`、`data/config.json`、`guard_result.json`；GPU 选择快照在 `logs/*_reservation.json`。正式四条件为 `A_3_sparse`、`A_3_crowded`、`A_6_sparse`、`A_6_crowded`。恢复对照为 `recovery6_retry1`，它是失败门槛的校准检查，没有运行 B1/B2。

源码入口：prepare.py（旧轨迹、身份与几何）、evaluate.py（冻结官方策略、环境、记录及恢复检查）、launch_job.py/guard.py（独立作业与资源守护）、takeover_guard.py（仅替换自有监控、保持同一 worker 与阶段总额）、analyze.py/process_audit.py（离线指标及探索性过程敏感性）、historical_cases.py（历史事件图像及状态索引）。

全仿真复现必须另建运行目录和分支，保留原始目录，取得并校验所列官方资产与环境，再按冻结场景/配置和回执执行。在新目录中调整 run_job.sh 的 ROUND 指向新目录；重新登记启动时刻及预算。不能直接对已有 runs 名称重复启动或覆盖结果，也不能因当前显卡负载改写科学条件。较慢共享卡的守护额度更替需在阶段总预算内登记并核验；原更替详情见两份 monitor_takeover.json。

主要轻量结果：results/RESULTS.md、DECISION.md、A_endpoints.csv、A_summary.json、process_audit.json、resource_summary.json、stage_status.json；全部正式回合图像在 results/A_cases，历史事件图像与状态段在 results/historical_cases。完整原始文件的路径、大小、SHA256 和是否纳入 Git 在 evidence_inventory.json。

原实验卡里的历史相对链接保持原样，部分引用未在本分支复制；历史来源的位置以上述只读目录与本地交接目录为准。
