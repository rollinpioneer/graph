# CUPID 最小复现实验简报

日期：2026-07-22  
任务：Robomimic Square-MH，低维输入  
最终状态：PASS

## 1. 实验结论

CUPID 最小复现流程已完整跑通，包括模型训练、100 次在线 rollout、TRAK 样本影响计算、demo 级影响聚合、官方 demonstration score 计算，以及不同 rollout 预算下的稳定性分析。训练、评估和影响分析产物均已保存并进行哈希冻结。

模型在正式 100 次 rollout 中成功 71 次，成功率为 71%。影响排名随 rollout 预算增加而明显稳定：50 次 rollout 时，与完整 100 次结果的 Spearman 排名相关系数约为 0.75–0.77；但 Top-20% 集合的 Jaccard 仅约 0.42–0.43，说明若目标是稳定识别最重要的一小批 demo，50 次 rollout 仍有较明显不确定性。

## 2. 关键设置与数据

| 项目 | 数值 |
|---|---:|
| 原始 demonstrations | 300 |
| 训练 / 验证 / holdout | 192 / 12 / 96 |
| 训练 epoch | 1751（编号 0–1750） |
| 正式 rollout | 100 |
| Rollout seed | 100000–100099 |
| 成功 / 失败 | 71 / 29 |
| Rollout 决策点 | 4066 |
| TRAK 投影维度 | 4000 |
| TRAK 时间步 | 64 |
| TRAK 原始矩阵 | 75129 × 4066，float32 |
| Rollout-demo 影响矩阵 | 100 × 192，float32 |
| TRAK 计算时间 | 4326 秒（约 1 小时 12 分） |
| 正式训练时间 | 39788 秒（约 11 小时 3 分） |
| 100 次 rollout 时间 | 5670 秒（约 1 小时 34 分） |

训练阶段最后一次周期性测试成功率为 0.68；独立的正式 100 次 rollout 成功率为 0.71。两者使用的评估样本规模不同，因此不要求数值完全一致，但结果处于相近区间。

## 3. Demo 影响评分

最终为 192 个训练 demo 生成了 net influence 分数。排名最高的五个原始数据集 demo 为：

| 排名 | Demo 索引 | 分数 |
|---:|---:|---:|
| 1 | 131 | 22.251995 |
| 2 | 87 | 21.715597 |
| 3 | 255 | 21.703617 |
| 4 | 270 | 19.112383 |
| 5 | 298 | 18.788118 |

排名最低的三个 demo 为 230、247、243，对应分数分别为 -48.183853、-48.909950、-49.843810。该分数用于当前模型和评估分布下的相对排序，不应直接解释为跨任务的绝对质量值。

## 4. Rollout 预算稳定性

每种预算重复抽样 20 次，以完整 100 次 rollout 的排序为参照。

| 预算 | 随机 Spearman | 分层 Spearman | 随机 Top-20% Jaccard | 分层 Top-20% Jaccard |
|---:|---:|---:|---:|---:|
| 5 | 0.214 | 0.199 | 0.156 | 0.161 |
| 10 | 0.332 | 0.393 | 0.190 | 0.220 |
| 25 | 0.540 | 0.497 | 0.280 | 0.251 |
| 50 | 0.768 | 0.754 | 0.431 | 0.424 |
| 100 | 1.000 | 1.000 | 1.000 | 1.000 |

主要观察：

- 5–10 次 rollout 不足以稳定恢复完整 demo 排名。
- 25 次 rollout 可获得中等程度的整体排序一致性。
- 50 次 rollout 已能较好恢复整体排序，但 Top-20% demo 集合仍不够稳定。
- 本实验中，分层抽样在预算 10 时优于随机抽样，但在 25 和 50 时没有形成一致优势。

## 5. 结果位置

- Demo 分数：`/home/xushijie/CUPID/results/influence_layers/final_demo_scores.csv`
- Rollout-demo 影响矩阵：`/home/xushijie/CUPID/results/influence_layers/rollout_demo_influence.npy`
- 随机预算分析：`/home/xushijie/CUPID/results/rollout_budget_random/rollout_budget_summary.csv`
- 分层预算分析：`/home/xushijie/CUPID/results/rollout_budget_stratified/rollout_budget_summary.csv`
- 冻结哈希：`/home/xushijie/CUPID/frozen/influence/all_files_sha256.txt`

总体而言，本次最小复现实验证明了从训练数据到 rollout 结果，再到 demo 影响分数的完整证据链可以运行并产生有限、可审计的结果。若后续使用较小 rollout 预算进行 demo 筛选，建议至少采用 50 次 rollout，并对 Top-k 选择的不稳定性进行额外复核。
