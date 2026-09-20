# Stage 0A 实现清单

状态：BLOCKED。生产入口不以建议路径冒充已存在实现。

| 计划 §3.3 模块 | 盘点结论 |
|---|---|
| `contracts/registry.py` | 技能、谓词、类型 grounding 与真实合同来源未绑定 |
| `graph/snapshots.py, intervention.py` | 不可变快照、三态 overlay 与不变量生产实现未建立 |
| `graph/rgcn.py, readout.py` | 仅执行第三方 RGCN 库探针；尚无生产四路编码和 goal readout |
| `vlm/client.py, cache.py, validate.py` | 只有冻结 schema/prompt；无已绑定 API adapter、few-shot 和缓存入口 |
| `policy/model.py` | 冻结 v2.1 七配置、GRU、V/Q 与单一参数所有者未实现 |
| `rl/collector.py, rollout.py` | 无真实平台 on-policy skill transition 接口 |
| `rl/targets.py, losses.py, recurrent.py` | 旧 PPO/reward 实现未认证为本版 SMDP/当前参数 prefix 重算 |
| `env/adapter.py, skills/executor.py` | 无 D0/T_A/T_C 的正式平台和固定技能接口绑定 |
| `evaluation/run.py, metrics.py` | 无本版独立 task_success evaluator 和配对 case 入口 |
| `reporting/build.py` | 本次仅有 Stage 0A 审计报告生成器；无科研结果报告入口 |

## 已有历史代码的实际核查

- `/home/__compress_data/xushijie/graph_github_upload`，commit `087a7b2770c8130c9d0b29e3a6b703f19c8c50e5`；路径检索 CP-DISR/M1 候选 0 项，定点源码审查 0 个文件。
- `/home/__compress_data/xushijie/graph_pathgraph_p2c_rl_evaluation_repair_v1_worktree`，commit `999958422d876270337afa469e3047f605dceb84`；路径检索 CP-DISR/M1 候选 0 项，定点源码审查 2 个文件。
- `/home/__compress_data/xushijie/graph_pathgraph_p1_v6gm1_worktree`，commit `980997e69c914287349e51233c8fff0d50934406`；路径检索 CP-DISR/M1 候选 0 项，定点源码审查 2 个文件。

完整范围、真实路径、SHA-256 和行号证据见 `legacy_resource_review.json`。该检索不宣称穷尽服务器所有文件。历史 reward shaping、旧离散任务或控制器可供后续适配审查，但未直接绑定为 v2.1 训练器或 D0/T_A/T_C 资产。

因外部任务与控制器合同仍未绑定，本轮在 Stage 0A 核心资源停止条件处结束；没有用自造平台、toy nominal transition 或空壳 CLI 填补缺项。后续补做 0A 时需实现/绑定上述生产模块。
