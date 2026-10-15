# 条件卡 B：高回报行为与安全约束冲突
日期：2026-10-15。状态：条件和公开接口成立；暂不选择为新模块方向。近邻重叠高，本服务器数据访问尚未打通。

| 字段 | 核查结果 |
|---|---|
| 条件定义 | 数据/部署中任务回报与累计安全成本存在权衡。累计成本预算与“任何时刻都不能违规”是不同问题，必须先选择一种。 |
| 原方法边界 | 本次 GFP 的 guidance 和 Actor 目标使用任务回报 Q，没有独立成本约束目标。这是目标定义的边界，不是原论文实现出错。不能从 HumanoidMaze 的低姿态直接构造安全标签。 |
| 具体数据入口 | 固定 DSRL 71064f21976f59c8f7139d00f1294963a1558369；可候选 OfflineCarCircle-v0。注册器绑定 Bullet RaceCar/CircleTask/Octagon，horizon300；源码映射到 SafetyCarCircle-v0-100-1450.hdf5。 |
| 字段和轨迹 | 官方加载器要求 observations、next_observations、actions、rewards、costs、terminals、timeouts。终止或超时分割轨迹，回合 reward/cost 分别累加；get_normalized_score 以 target_cost 归一化成本。文件名中的100不能自动当作我们批准的预算。 |
| 验证状态 | 代码和下载映射已核查；服务器对官方 Hugging Face 文件的 HEAD 请求报 Network is unreachable，未取得实际数据、未运行环境，也未核验成本分布。不能标为实验准入完成。 |
| 可验证预测 | 在具有同样安全标签的对照中，任务 Q 偏好的高回报高成本行为是否获得较高学习权重；安全机制是否改变这项中间量并降低真实违规，同时保留可比回报。当前尚未测量。 |
| 简单替代解释 | 优势来自额外安全标签或普通 reward-cost 惩罚，而不是新机制；所有方法须获得同样信息，并比较合理惩罚系数的 GFP 及成熟离线安全方法。 |
| 最近邻工作 | EpiFlow 已使用 epigraph feasibility value 加权行为并训练 Flow；SafeFQL 已将 reachability safety value 和一步 Flow Actor 结合。仅“安全评分+GFP/Flow”有直接重叠。 |
| 最小后续试验 | 首先只检查可取得的数据中 reward/cost 的联合分布、预算下的行为覆盖和成本定义；随后如仍有独立问题，再冻结累计成本预算、原版惩罚基线和最接近安全方法。当前不启动训练。 |
| 退出条件 | 无充分安全标注/可评环境、优势仅由更多标签解释、普通惩罚已足够，或机制等价于已有安全 Flow 方法，则放弃。 |

证据：[DSRL 注册器](https://github.com/liuzuxin/DSRL/blob/71064f21976f59c8f7139d00f1294963a1558369/dsrl/offline_bullet_safety_gym/__init__.py)、[数据加载与成本评价](https://github.com/liuzuxin/DSRL/blob/71064f21976f59c8f7139d00f1294963a1558369/dsrl/offline_env.py)、[公开数据映射](https://github.com/liuzuxin/DSRL/blob/71064f21976f59c8f7139d00f1294963a1558369/dsrl/infos.py)、[基准主页](https://www.offline-saferl.org/)。

## 最近邻比较不能省略的差异

- [EpiFlow v1](https://arxiv.org/html/2602.08054v1) 面向状态安全约束，其 feasibility/epigraph value 参与行为加权和 Flow Matching；它直接占据“用安全价值重加权 Flow 行为”的机制位置。不能把换宿主名称作为创新。
- [SafeFQL v2](https://arxiv.org/html/2603.15136v2) 使用 max-backup 形式的安全价值与一步策略；其附录明确采用连续 signed safety function，而不只是二值 cost。因此不能默认为 DSRL cost 字段可无差别代入，也不能让新方法独享连续安全信息。
- 这两篇的逐状态安全问题不自动等于 DSRL 的期望累计成本预算。近邻机制必须比较，指标也须对齐；不同目标下的论文数字不能直接排名。
- 本轮按论文当前固定版本核查，不沿用早期版本可能存在、但 v2 未核实的 conformal calibration 说法。

当前只支持“该条件有真实公开基准和清楚目标边界”，不支持“GFP 已在这里表现差”或“新增安全模块有原创性”。
