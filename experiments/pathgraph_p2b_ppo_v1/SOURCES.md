# 来源与本轮设计分界

核对日期：2026-09-16。官方版本链接用于固定可复现实现，并非声称采用最新版本。

## 仓库/用户实验事实

[R1] P2A正式报告（固定BASE）：
https://github.com/rollinpioneer/graph/blob/577db67c64343850965fa4e0f6bc7cac0c15674d/artifacts/pathgraph_sarm/upgrade_v2/p2a_state_weighted_bc_utility_v1/final/report.md

[R2] P2A零训练取证：
https://github.com/rollinpioneer/graph/tree/577db67c64343850965fa4e0f6bc7cac0c15674d/artifacts/pathgraph_sarm/upgrade_v2/p2a_state_weighted_bc_utility_v1/forensics_v6_loss_v1

[R3] 冻结动作环境，Git blob `9688f2788fc716b2bc623fd04fb2f0e8f418ccaf`：
https://github.com/rollinpioneer/graph/blob/577db67c64343850965fa4e0f6bc7cac0c15674d/experiments/pathgraph_p2a_utility_v1/p2a/env.py

[R4] P1冻结方法哈希：
https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/independent_state_holdout_v1/method_source_hashes.json

本轮源码加载器另逐字节核对已提供操作包中的reward_v6.py、reeval_core.py、reward_contract.json。不会把外置路径的存在当作在本地执行过服务器实验。

## 算法/接口一手来源

[S1] Schulman et al., Proximal Policy Optimization Algorithms, 2017.
https://arxiv.org/abs/1707.06347

[S2] Ng, Harada, Russell, Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping, ICML 1999.
https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf

[S3] Stable-Baselines3 2.7.1官方PPO文档：
https://stable-baselines3.readthedocs.io/en/v2.7.1/modules/ppo.html

[S4] Stable-Baselines3 2.7.1 on-policy rollout源码（terminal observation与超时bootstrap）：
https://stable-baselines3.readthedocs.io/en/v2.7.1/_modules/stable_baselines3/common/on_policy_algorithm.html

[S5] Gymnasium官方Handling Time Limits：
https://gymnasium.farama.org/main/tutorials/handling_time_limits/

[S6] Agarwal et al., Deep Reinforcement Learning at the Edge of the Statistical Precipice, NeurIPS 2021.
https://arxiv.org/abs/2108.13264

[S7] 选定包版本：
https://pypi.org/project/stable-baselines3/2.7.1/
https://pypi.org/project/gymnasium/1.2.2/

## 不属于上述来源的结论

六方法矩阵、固定beta、非图强势函数系数、训练规模、family命名、LONG_CHAIN128、终止适配选择和+3pp实用门槛均为本操作包新设计。来源不保证本实验会学到成功策略，更不保证图奖励胜过基线。
