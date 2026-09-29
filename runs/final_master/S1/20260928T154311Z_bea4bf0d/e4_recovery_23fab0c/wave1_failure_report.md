# Wave 1 停止记录（E4 恢复）

- 结论：首批 2 个分支在 bundle_create 阶段失败，均为 `BindingError: MUST_BIND: runtime_factory verified source hash`，耗时约 1 秒，**未创建真实环境、未 reset、未执行任何 skill、无 provider 调用**。
- 原因（执行侧错误）：恢复登记与 receipt 绑定了 `input_binding/T_A_s1_rev1_runtime_manifest.yaml`，其中 runtime_factory.sha256 为旧值 5db83fe6…；23fab0c 之后 runtime_factory.py 的实际 hash 为 4bd45b0a…，因此哈希门失败。收口验收生成的 `integration_closeout_10971a4/derived/runtime_manifest.current.json` 才是与当前源码一致的 manifest（离线核验：旧 manifest PREFLIGHT_FAIL，current manifest PREFLIGHT_PASS）。
- 账本：恢复账本 physical_witness_episodes used=2 / cap=8，两个 attempt 状态 FAILED，均保留，不改 ID、不清账本。原 8/8 与旧目录未改。
- 调度：按规则在故障后停止调度未启动的分支，剩余 6 个分支未预留、未启动。
- 代码修复（已提交，不改变实验方法）：run_wave 在预留任何额度前增加零成本 preflight（manifest 的 runtime_factory hash 必须与磁盘源码一致）。
- 后续动作需要用户决定：是否补偿这 2 个未产生任何环境交互的 attempt，见对话。
