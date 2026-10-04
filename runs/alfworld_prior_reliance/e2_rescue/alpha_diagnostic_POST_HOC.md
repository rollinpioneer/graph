# POST_HOC_DIAGNOSTIC：PRIOR_BIAS 先验强度扫描（仅评价，不训练）

标记：POST_HOC_DIAGNOSTIC。数据：`alpha_diagnostic_POST_HOC.json`；脚本：`scripts/alfworld_alpha_diagnostic.py`。
所有区间均基于**单个训练 seed**（PRIOR_BIAS seed 0），只反映 dev episode / scene 抽样，不代表跨训练 seed 的稳定性。最佳 alpha 不是方法结果，不覆盖原 pilot 结果。

设置：权重冻结（无 backward / optimizer）；`logits = base_logits + alpha * 现有中心化并缩放到[-1,1]的 prior 分数`，其中 base_logits 为去掉 prior 残差后的合同 logits；dev 125 局 × 2 次重复，随机数与 pilot 评价相同（逐 (game, repeat, seed) 固定）。
Harness 检查：用训练得到的 alpha = B·tanh(β)（最早 0.0055，最晚 0.0065）重放，250/250 个 episode 与 pilot 自带评价完全一致。

| checkpoint（真实 transition 数） | alpha | J | raw steps | CHECK 数 |
|---|---|---|---|---|
| 最早（2074） | 0 | 0.9004 | 7.70 | 3.71 |
| | 0.25 | 0.9009 | 7.66 | 3.66 |
| | 0.5 | 0.9028 | 7.51 | 3.56 |
| 最晚（8265） | 0 | 0.9240 | 5.73 | 2.46 |
| | 0.25 | 0.9248 | 5.67 | 2.41 |
| | 0.5 | 0.9247 | 5.68 | 2.41 |

相对 alpha=0 的配对差（同一 episode）：

| checkpoint | alpha | ΔJ | game-cluster 95% CI | scene-cluster 95% CI | Δsteps | ΔCHECK |
|---|---|---|---|---|---|---|
| 最早 | 0.25 | +0.0006 | [−0.0013, +0.0024] | [−0.0030, +0.0037] | −0.048 | −0.048 |
| 最早 | 0.5 | +0.0024 | [−0.0022, +0.0065] | [−0.0055, +0.0053] | −0.192 | −0.152 |
| 最晚 | 0.25 | +0.0008 | [−0.0003, +0.0019] | [−0.0001, +0.0020] | −0.060 | −0.048 |
| 最晚 | 0.5 | +0.0007 | [−0.0010, +0.0022] | [−0.0013, +0.0019] | −0.044 | −0.052 |

读法：外加 prior 强度到 0.5，J 的变化只有 +0.0006 至 +0.0024，所有区间都包含 0（最晚 checkpoint、alpha=0.25 的 scene 区间下界为 −0.0001，也未排除 0）。方向一致地略微减少 CHECK，但幅度远小于不确定性。该诊断不支持“prior 被学到的强度偏低导致无收益”这一解释，也不足以支持相反结论。
