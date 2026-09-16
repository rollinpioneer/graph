# 来源与新设计分界

核对日期：2026-09-15。固定项目提交：`8e86b246e900e1e12dbb21c88b95046a06c8e4c5`。

## 项目一级来源

- P1状态holdout决定：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/independent_state_holdout_v1/final/decision.json>
- P1实际评分入口：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/upgrade_v2/p1_independent_state_holdout_v1/scorer.py>
- 冻结方法字节哈希：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/independent_state_holdout_v1/method_source_hashes.json>
- 构造状态字段，不是action训练数据：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/upgrade_v2/p1_independent_state_holdout_v1/state_builder.py>
- 强非图事件基线：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/upgrade_v2/p1_mainline_grasp_v6gm1/extra_methods.py>
- 旧动作环境：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/tools/stage6/policy_env.py>
- 旧策略评价器：<https://github.com/rollinpioneer/graph/blob/8e86b246e900e1e12dbb21c88b95046a06c8e4c5/tools/stage6/evaluate_policy_checkpoint.py>

## 方法背景一级来源

- SARM, 原v1（2025）：<https://arxiv.org/abs/2509.25358v1>。支持“奖励辅助示范重加权是相关下游用途”；不支持P2A特定任务、权重floor、模型架构或结果。
- imitation官方BC文档：<https://imitation.readthedocs.io/en/latest/algorithms/bc.html>。支持“从观测动作对监督学习策略”的定义；本包并未声称调用了该库的BC实现。
- Ng, Harada & Russell, 1999：<https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf>。潜势变换与策略不变性背景；不把该定理用于保证当前正权重BC必定改善。

## 本包新增设计

参考技能MDP、混合质量数据、分割与种子、POSITIVE_PROGRESS_FLOOR_V1、45任务训练矩阵、统计口径，全部是本轮建议，不是来源中已有实测成果。
