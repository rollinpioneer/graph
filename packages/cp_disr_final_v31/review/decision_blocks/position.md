# 4. Paper Positioning

正式名称保留 **CP-DISR：Contract–Prior Differential Intervention for Skill Reasoning**。论文题目采用：**CP-DISR: Contract–Prior Interaction Contrasts for Long-Horizon Skill Selection**。M1只作内部代号；全文将intervention限定为nominal symbolic intervention。

中心问题是：**在相同技能合同与观测条件下，让固定语义先验通过“候选名义变化的编码响应”影响策略，是否比不做干预或直接读取四视图更适合长期技能选择？**固定不完美先验仍是问题背景，candidate-conditioned prior interaction是算法组织中心。

关键观察不是“关系只有改变物理后果才有价值”。输入关系不修改环境动力学，也不修改合同的名义patch。更准确地说，关系的决策用途依赖候选、状态、目标与后续选择；本文检验一种有意限制：先验只经候选变化的表示交互参与决策。静态信息可能有用，A_CAT直接检验排除它的代价与收益。

**One-sentence story.** We study whether restricting fixed semantic priors to their interaction with candidate-induced nominal representation changes improves skill selection under matched observations, contracts, and interaction budgets.

主方法没有显式逐边utility、关系真假或知识修订模块。不能将集合R的编码交互写成已经识别“每条关系的真实贡献”。固定cache隔离生成与利用并支持复现；在线重新query并不必然破坏on-policy，故不以这一错误理由论证固定cache。[W05，W14；本轮设计]
