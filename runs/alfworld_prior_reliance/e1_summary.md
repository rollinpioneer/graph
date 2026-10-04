# E1 总结（dev，125 局；冻结 prior cache：qwen3.7-flash-2026-07-15，238 key，100% 有效）

结果文件：`e1_result.json`（含 `task_prior_admission` 与 `qref_validation` 两节）；数据审计：`data_audit.json`；缓存：`prior_cache.json`。

## 1. Task / prior admission
| 准入条件 | 结果 | 判定 |
|---|---|---|
| 1 搜索决策中 ≥2 个合法 CHECK | 98–100% | 通过 |
| 2 分歧发生在合法 soft choice 上 | 100%（构造上两者都取自公开合法集合；全部分歧均有 ≥2 个合法 CHECK） | 通过 |
| 3 prior 在 canContain 可行集内非均匀 | 100% 的 episode | 通过 |
| 4 Top-1 / Miss@3 各 ≥10% | top1 40.0%（50 局），top-k only 42.4%，miss 17.6%（22 局） | 通过 |
| 5 Belief-Optimal 相对最佳静态脚本 ≥0.03 | 最佳静态=Fixed Prior；差值 +0.0092，95% CI [0.0026, 0.0165]；全知上限相对 Fixed Prior 也只有 +0.0265 | **未通过** |
| 6 Belief-Optimal 与 Fixed Prior 动作分歧 ≥10% | 73.6%（1367 个搜索状态） | 通过 |

均值 J：Belief-Optimal 0.9268，Fixed Prior 0.9176，Random 0.8650，Contract 0.8535，全知上限 0.9441。成功率均为 100%。

停止条件命中：「Fixed Prior 与 Belief-Optimal 几乎等价」。同时，条件 5 在该任务设定下不可达：即便全知策略相对 Fixed Prior 也只有 +0.0265 < 0.03（J 被 H=50 与固定宏动作开销压缩）。未改布局、奖励、目标位置或 prior prompt。

prior 不是复述 canContain：prior 质量中平均 14.2% 落在 canContain 不可行类别；71.2% 的 episode 中 canContain 过滤会改变排序；10.4% 的 episode 未过滤 top-1 本身不可行。

## 2. Q_ref validation（单独汇报，状态：PENDING_RESEARCH_DECISION）
Q_ref 已改为多 holder（独立 Bernoulli、至少一个 holder）模型；验证改为同初始观察签名（同房间+同目标）下的条件期望比较，按 scene 做 cluster bootstrap；ε_Q=0.001 以下标 neutral/unknown。
- 签名 cell 116 个（48 组，平均每 cell 2.5 个隐藏世界）；decided 56 个，neutral/unknown 60 个。
- decided：top1−top2 的真实 J 差均值 +0.0042，scene-cluster 95% CI [0.0019, 0.0069]（12 个 scene 簇），64% 的 cell top1 更好。
- 绝对校准：平均 Q 预测 0.9102 vs 实现 0.9110；动作级相关 0.77。
- neutral cell 均值 −0.0023（预期无方向）。

## 3. 数据审计要点
多目标实例比例 58–60%，多 holder 比例 44–48%，目标 receptacle 有多个实例的比例 23–38%（train/dev/test），均非平凡，Q_ref 已按此修正。test 中有 33/126 局的 holder (目标类, receptacle 类) 组合在 train 中未出现，1 局目标类（knife）未在 train 出现；dev 为 22/125 与 0。
