# PathGraph P1-Core 阶段报告（当前收口）

## 已完成

- 固定基点 `ade63b375f64b31dfe6bb50cc893335d2dfbfea4`、源文件锁和包完整性核验通过。
- 创建正式工作树：`/home/__compress_data/xushijie/graph_pathgraph_p1_core_v1_worktree`，分支 `research/pathgraph-p1-state-conditioned-reward-v1`。
- `run_first_pass.sh` 完成：12 个合成 episode、33 个 transition、8 种方法；独立表达式与冻结引擎最大误差为 0。
- 包测试通过：70 passed。
- 旧参考图的结构/数值审计通过；当前 G1 明确发现 9 节点、3 顶层边、数值字段为空、7 个节点无成功路径，不能视作完整奖励图。
- 外置资源目录已定位；当前可见文件为摘要 JSON，不满足逐前缀因果状态合同。

## 已发现的反例（仅合成算术输入）

- `unit_scale_clipped_closed_cycle`：`FULL_FROZEN` signed return = 1.0。
- `phi_reset_closed_cycle`：`FULL_FROZEN` signed return = 0.5。
- 正权重和不是 signed return；例如首个循环 positive weight sum = 2.5。

这些结果说明“冻结债务账本局部算术成立”不能推出“任意完整闭环安全”，也不能直接写成真实物理漏洞。

## 未完成/不可声称

- 没有可用的 raw state prefix，因此未运行真实 CAUSAL_STATE_REPLAY、25/50/75/100% 前缀一致性或 phi 几何绑定。
- 当前生成图的 runtime 展开、数值 cost、历史 guard 和恢复边来源尚未闭合；状态为 `CURRENT_GRAPH_BINDING_INCOMPLETE`。
- 必测语义场景、真实轨迹上的八方法公平比较、正式消融与最终贡献表尚未完成。
- 本阶段没有新物理采集、训练或大模型调用。

结论限定为：旧参考机制在提供的合成算术夹具上可复现，但全路径安全存在输入级反例；当前 G1 生成图绑定和真实状态因果回放仍是未解决项。
