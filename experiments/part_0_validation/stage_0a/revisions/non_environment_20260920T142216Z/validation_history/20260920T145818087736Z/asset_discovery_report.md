# 服务器资产盘点

只读检查；未导入历史模块、未读取 API 凭据、未启动仿真或机器人。范围限清单中的项目与搜索深度，不声称服务器任何位置都不存在缺项。

| 源项目 | commit | tracked dirty |
|---|---|---|
| `/home/__compress_data/xushijie/graph_github_upload` | `087a7b2770c8130c9d0b29e3a6b703f19c8c50e5` | False |
| `/home/__compress_data/xushijie/graph_pathgraph_p2c_rl_evaluation_repair_v1_worktree` | `999958422d876270337afa469e3047f605dceb84` | False |
| `/home/__compress_data/xushijie/graph_pathgraph_p1_v6gm1_worktree` | `980997e69c914287349e51233c8fff0d50934406` | False |
| `/home/__compress_data/xushijie/CUPID/repo` | `2941eba9427a3398d3cbe26b10c01e3ece56bd18` | True |
| `/home/__compress_data/xushijie/LIBERO` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | True |

## 复用判断

- LIBERO `env_wrapper.py:ControlEnv` 提供 reset/step/check_success、相机名称和频率配置。默认同时含 agentview/腕部视角、depth=False、20 Hz、horizon=1000；与本版固定第三视角 RGB-D、skill 级时长、受控终止不同，必须显式 adapter，不能照搬默认。
- LIBERO `base_predicates.py:In/On/Open` 与各任务 `_check_success` 可供独立任务 evaluator 适配；其模拟器真值不能冒充视觉 Fact Verifier 或控制器的实测几何。
- LIBERO BDDL/XML/初始化数据是实际资产候选，尚未完成 D0/T_A/T_C 的角色、固定技能和真实 case 池映射。
- 历史 P1 regrasp/return 控制器有候选代码；尚未证明可执行本版通用 OPEN/PICK/PLACE/MOVE，也未迁入新项目。
- 历史 P2C gym adapter/PPO/checkpoint 可参考通用工程；reward shaping、固定 37 动作以及旧算法状态语义不能直接复用为本版方法。
- 搜索到 checkpoint 文件不代表它是本版合格的冻结视觉前端；未加载这些权重，未生成 resolved task/runtime 绑定。

共记录 65 个代码候选和 3229 个资产候选。逐项 absolute path、commit/dirty、AST 签名、I/O 注解（无注解时明确未知）、hash、证据行、adapter 要求和语义差异见 `asset_discovery.json`。
