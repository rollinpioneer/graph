# floq Critic 准入的静态模块说明

依据固定官方提交 `3d60e638b42bbef018c56cb1199ff37c3470520d` 的 `agents/floq.py`、`utils/networks.py` 和 `utils/flax_utils.py`。本卡不做替换实验；下述不是新方法贡献声明。

## 网络与计算路径

| 模块 | 输入 → 输出 | 学习信号和状态 |
|---|---|---|
| `modules_floq` | observation `[B,29]`、action `[B,8]`、return `[2,B,1]`、time `[B,1]` → velocity `[2,B,1]` | 两个独立 ensemble；Fourier time 64 维、HL-Gauss return 51 bins/σ16；默认 4×512 MLP、LayerNorm |
| `modules_target_floq` | 同上，经 8 步 Euler 积分和 ensemble mean → `[B,1]` | TD backup，无反传；初始化复制 floq，Polyak τ=.005 |
| `modules_critic` | observation、action → Q `[2,B]` | 普通标量双 Critic；拟合当前 Flow 积分的 ensemble/noise 均值 |
| `modules_actor_bc_flow` | observation、noisy action、time → action velocity | BC flow matching；默认 10 步 Euler 生成蒸馏目标 |
| `modules_actor_onestep_flow` | observation、Gaussian noise → action `[B,8]` | α=10 动作蒸馏 + 标量 Critic 上的 Q 最大化；动作裁剪 [-1,1] |

Flow 积分初始 return 来自均匀噪声，范围为奖励范围除以 `1-discount` 后乘 `.1`。在 -1/0 奖励下为 [-10,0]；HL-Gauss 映射的范围另带 `.01` reward offset，为 [-101,1]。最终以实际展开配置为准。每条 batch 256，噪声重复 8，Critic 对内部 2048 条样本计算；不能仅用外层 batch 比较成本。

## 损失和梯度依赖

TD target = `mean_noise(reward + .99 * mask * target_flow_return(next_observation, next_action, noise))`。next_action 为当前 one-step Actor 的随机动作；本 loss 对 TD target 路径不回传到 Actor 或 Target Flow。

`floq_loss` 对 Flow velocity 与直线插值目标 velocity 的平方误差先按 **8 个 noise 求和**，再对 ensemble 和 batch 求平均。不是把全部维度取平均；换实现时若改变求和，会改变隐含损失权重。`train_at_zero_only=False`，time 均匀采样，保持多步积分。

`distilled_critic_loss` 为两个 Scalar Critic 对当前在线 Flow 的积分均值的平方误差。Flow 积分使用存储参数作为蒸馏目标，梯度只更新 Scalar Critic；不能把它误写成同时反传的联合蒸馏。

`actor_loss = bc_flow_loss + 10 * distill_loss + q_loss`；`q_loss = -mean_ensemble_batch(Q_scalar(s, clip(actor(s,z))))`，不做 Q loss normalization。Actor 的 Q loss 对 action 回传，Scalar Critic 参数在该路径固定。BC 多步动作目标固定，distill loss 只更新 one-step Actor；BC 自身从 BC flow matching 学习。

所有可训练网络登记在同一个 ModuleDict/Adam TrainState 内，`total_loss` 同时形成 Actor 和 Critic 梯度；target 模块在 optimizer 参数树中但梯度为零，再由 Polyak 更新。更新实现 `target_update(new_network,'floq')` 读取的是 `self.network` 的**更新前在线参数**和更新前 target；保留这一拍延迟，不能为“修正”而悄悄换成优化后的参数。P1 检查按此实际实现验证。

独立替换 Critic 时必须一起定义 Flow/Target Flow/Scalar Distillation 的状态、噪声范围、bootstrap、encoder 绑定和 target 时序，并保留 Actor 输入/输出和 Scalar Q 对 action 的导数接口。只替换 online Flow 而保留旧 target 或旧读出会造成状态或目标不匹配。当前低维任务 `encoder=None`；视觉 encoder 的额外路径未经本卡运行验证，不能泛化声称已准入。

## 可替换性与现有贡献边界

该价值组件可以作为一个整体受控替换，Actor 两支通常可保留；但不具有“更换单个函数就完全独立”的条件。先建立返回数值、action 梯度、target 时序、损失尺度、优化器状态和 RNG 消费的接口契约，再做单一受控替换。

论文已经讨论 flow critic 的逐步计算、t=0 限制、Flow steps、噪声范围、time/return embedding 和 ResNet 对照；这些不能直接作为新贡献。其标量 Critic 蒸馏和 FQL BC/one-step Actor 组合亦已实现。后续最容易检验的失败假设是：Flow return 的噪声方差与蒸馏读出的误差积累，造成 Actor 的 Q 梯度偏离 Flow 值；本轮仅记录 `var_flow_returns`、`distilled_critic_loss`、`td_error` 和 Actor loss，不开展新训练或声称证实该假设。与 GFP/Value Flows 的创新性需要另行检索具体方法及批准实验后判断。

方法来源：https://arxiv.org/html/2509.06863v2 、https://github.com/CMU-AIRe/floq/tree/3d60e638b42bbef018c56cb1199ff37c3470520d 。
