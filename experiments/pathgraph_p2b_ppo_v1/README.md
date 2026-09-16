# PathGraph P2B：Graph-PPO Reward Utility

本包提供实验操作手册、协议、折扣PBRS参考实现、冻结P2A/V6源码加载器、SB3训练/评价入口和测试。

**目的**：固定一个成熟PPO算法，只比较任务势函数来源。数据由策略在线交互产生，不再给旧示范加权。

- 6种奖励；8个训练seed；48个任务。
- 每任务524,288个环境transition。
- BASE64与LONG_CHAIN128；最终每模型2,048个新test episode。
- 状态条件抽象技能环境，不是机器人或MuJoCo物理。
- P1/P2A全部历史结果只读。

**验证边界**：算术/冻结源码适配可本地测试；真实SB3训练依赖需在服务器安装并运行smoke。本包不包含已训练模型，不表示实验已完成。

先读`MANUAL.md`与`LOCAL_VALIDATION.json`。本包不包含已训练模型或用户原始轨迹。
