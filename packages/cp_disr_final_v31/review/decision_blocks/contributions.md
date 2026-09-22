# 5. Core Contributions

本文以**一个统一设计、两个方法层面的贡献方面**组织，不把名义后继、减法和辅助Q各自包装成独立新算法。

## C1. Matched contract–prior interaction representation

我们采用已有afterstate/后继状态思想，在合同/增强与当前/名义后继两个维度建立共享、目标对齐的四视图，并用interaction contrast组织固定先验对候选变化的编码影响。新颖性主张落在该表示及其特定用途，而非后继状态计算或2×2代数本身。[W03，W04]

**English.** We develop a matched contract–prior representation that evaluates the same nominal candidate change with and without fixed semantic relations, yielding a candidate-conditioned interaction contrast.

## C2. Pure-interaction policy design and outcome-grounded implementation

将该交互作为先验的唯一直接特征入口，以零锚定有界残差调制合同策略；给出精确null、可加分离抵消、单步概率比界与置换条件，并以真实执行支持的PPO及辅助候选Q训练共享表示。性质证明设计约束，不证明性能；辅助Q是训练机制而不是第三个原创学习范式。[W16–W19]

**English.** We restrict direct prior modulation to the interaction representation, with an exactly zero-anchored and bounded policy residual, and train the shared decision features using actual executions.

## 实证贡献的位置

有真实结果后，第三点写成**对应任务与预算范围内的实证发现**：受控合同阶梯、四视图融合对照、同checkpoint先验变化和Actor依赖共同说明设计何时有用。没有数据时只写评价问题，不把常规baseline/消融操作宣称为新的evaluation protocol。

当前不是“只有一个公式的论文”：名义轴、matched views、共享坐标、信息限制、执行权限、有限影响和实际训练是一个可检验的整体设计。但组件组合是否值得论文主张，仍须A_CAT等证据，不能由组合复杂度推出新颖或有效。
