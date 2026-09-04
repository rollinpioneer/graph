# CUPID Transport-MH 未通过门槛分析与方案调整建议

版本日期：2026-07-25  
工作目录：`/home/xushijie/CUPID`  
分析性质：正式结果复核 + 事后探索性诊断 + 文献调研；未修改预注册结论，未启动重训练

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent validate; paper-search with scholar-search fallback
- Origin Mode: validate + experiment planning
- Origin Date: 2026-07-25
- Verification Status: ANALYZED
- Evidence Scope: Transport-MH 的 100 条 rollout、192 条训练 Demo、50/50 独立池结果及相关方法论文
- Reproducibility Scope: 正式汇总已从冻结产物复核；新增方法比较是同一有限池上的事后探索，不是独立复现

## 1. 结论先行

未通过的门槛是：

```text
Bootstrap 稳定核心相对同数量直接最低分集合的配对精度增益 >= 0.03
```

它对“Bootstrap 成员选择比直接排序更好”这一主张是关键门槛，不能删除或事后降低；但它不是“CUPID 分数是否有用”的必要门槛，也不是“直接最低分过滤是否能改善重训练”的最终因果检验。

当前数据支持：Transport 上存在可靠的低分排序信号，小规模直接最低分集合的跨池精度很高。当前数据不支持：Bootstrap 能在该排序之上提供额外成员识别价值。

因此最合理的调整不是挽救本轮 Bootstrap 分支，而是冻结其失败结论，另开预注册的作者方法复现分支：使用 CUPID 原分数和 CUPID-Quality 直接排序，按固定比例过滤后重训练。最终门槛从“Bootstrap 是否胜过排序”改成“作者式过滤是否胜过未过滤和随机同数量过滤”。

## 2. 正式结果说明什么

八项强门槛中七项通过，唯一失败项如下：

| 指标 | 要求 | 实际 | 差距 |
|---|---:|---:|---:|
| Bootstrap `p>=0.90` 独立精度 | `>=0.70` | `0.870824` | 通过 |
| 相对固定最低 38 条增益 | `>=0.15` | `+0.249508` | 通过 |
| 相对同数量直接最低分增益 | `>=+0.03` | `-0.011352` | 距门槛 `-0.041352` |

在 200 个方向性评价中，Bootstrap 相对直接排序为：

```text
胜 / 平 / 负 = 38 / 95 / 67
平均增益 = -0.011352
中位数 = 0
按 100 个 split repeat 聚类的描述性 bootstrap 95% 区间
  = [-0.02043, -0.00222]
P(平均增益 >= 0) = 0.0081
P(平均增益 >= 0.03) = 0
```

这些区间只是对固定 100-rollout 池中重复拆分的描述，不是跨训练 seed 或跨任务的总体置信区间。即便如此，结果也表明失败并非由少数异常拆分造成。把门槛从 `+0.03` 降到 `0`，本轮仍不会通过。

## 3. 为什么 Bootstrap 会输

### 3.1 它优化了不同的目标

独立池把“另一半 rollout 中分数最低 20%”当作参考。直接最低分方法使用源池均值排序，正好对齐这一评价目标。Bootstrap `p>=0.90` 则要求一条 Demo 在反复重采样中都落入底部 38 条，因此同时偏好低均值和低方差。

两种方法每次平均只交换约 `1.98` 条 Demo，但交换成员的特征很有规律：

| 仅被某方法选中的 Demo | 源池平均分数 | 源池标准误 | 在独立池进入底部 20% 的比例 | 独立池平均排名百分位 |
|---|---:|---:|---:|---:|
| Bootstrap-only | `-0.03173` | `0.01093` | `0.72769` | `0.16642` |
| Direct-bottom-only | `-0.03949` | `0.02250` | `0.82163` | `0.14677` |

Bootstrap 换入了“没那么低、但方差更小”的条目；直接排序保留了“更低、但方差更大”的条目。独立池的评价规则更奖励后者。这支持“稳定性惩罚与评价目标错配”的解释，但不等同于证明真实 Demo 质量就是如此。

### 3.2 阈值越严格，错配越明显

| 稳定阈值 | 平均选择数 | Bootstrap 独立精度 | 同数量直接排序精度 | 配对增益 |
|---|---:|---:|---:|---:|
| `p>=0.80` | `18.585` | `0.8157` | `0.8156` | `+0.00005` |
| `p>=0.90` | `12.845` | `0.8708` | `0.8822` | `-0.01135` |
| `p>=0.95` | `9.215` | `0.8807` | `0.9100` | `-0.02926` |

更严格的“反复入选”要求提高了绝对精度，却没有提高相对直接排序的精度；原因是集合也变小了。匹配集合大小后，Bootstrap 的相对表现反而恶化。

### 3.3 固定 20% 没有自然边界

第 38/39 条的分数差只有 `0.000471`，而配对标准误为 `0.017961`，比值仅 `0.0262`。边界一倍标准误内有 41 条，两倍内有 75 条。固定删 38 条是比例设定，不是数据中出现了恰好 38 条的清晰簇。

### 3.4 不是计算错误，也不是完全没有信号

- 官方分数重建最大误差为 `1.468e-6`，远低于 `0.002` 门槛。
- Bootstrap 核心精度 `0.8708`，明显高于固定最低 38 条的 `0.6213`。
- Transport 的稳定低分核心和独立一致性均明显强于 Square。

所以主要问题在“如何从排序决定具体成员”，不在 TRAK 链路损坏，也不在 Transport 完全没有低分信号。

## 4. 替换重采样方式能否越过门槛

在完全相同的 100-rollout 有限池上做了事后探索；这些结果不能改写正式结论：

| 方法 | 平均选择数 | 独立精度 | 同数量直接排序精度 | 配对增益 |
|---|---:|---:|---:|---:|
| 原普通 Bootstrap | `12.845` | `0.8708` | `0.8822` | `-0.01135` |
| 成功/失败分层 Bootstrap | `13.085` | `0.8650` | `0.8801` | `-0.01515` |
| 分层 50% 无放回子采样 | `12.325` | `0.8731` | `0.8881` | `-0.01502` |
| 分层 80% 无放回子采样 | `22.820` | `0.7676` | `0.7617` | `+0.00595` |

结论：类别比例波动不是主要原因。80% 子采样只得到很小的正增益，远低于 `+0.03`，且是在看过结果后尝试的多个方案之一。继续微调重采样比例和阈值很可能变成选择性报告，不建议作为主线。

## 5. 直接最低 K 条的探索性表现

相同 50/50 拆分下，直接按源池 CUPID 分数选择固定 K 条：

| K | 独立精度 | Recall（相对独立底部 38 条） | Jaccard | 独立平均排名百分位 |
|---:|---:|---:|---:|---:|
| 5 | `0.9260` | `0.1218` | `0.1210` | `0.0851` |
| 8 | `0.9263` | `0.1950` | `0.1926` | `0.0871` |
| 10 | `0.9090` | `0.2392` | `0.2344` | `0.0949` |
| 13 | `0.8812` | `0.3014` | `0.2909` | `0.1075` |
| 19 | `0.8100` | `0.4050` | `0.3717` | `0.1365` |
| 38 | `0.6213` | `0.6213` | `0.4529` | `0.2068` |

这说明低分头部很纯，但“选得更准”和“删后训练更好”不是同一件事。K 越小精度越高、召回越低；真正最优 K 只能由独立的过滤重训练决定，不能从这张表反推。

## 6. 文献中的替代方法及适用性

### 6.0 CUPID 原论文直接改变了门槛的解释

CUPID 原论文（Agia et al., CoRL 2025, arXiv `2506.19121`）的近似线性 datamodel 得出一个直接操作规则：过滤时选 performance influence 最低的 `k` 条，选择新数据时选最高的 `k` 条。作者没有使用 Bootstrap 稳定核心，也没有要求复杂选择器胜过同数量直接排序。

作者 RoboMimic 协议为：

- 从 300 条 Demo 中随机取约 `2/3` 作为 filter-k 训练集；
- 对过滤比例 `0.10, 0.25, 0.50, 0.75, 0.90` 直接排序和重训练；
- 使用 100 条基础策略 rollout 计算 influence；
- 每个设置运行 3 个训练 seed；
- 模拟任务用最后 10 个 checkpoint 各 50 条 rollout，共 500 条评估 rollout。

论文报告 Transport-MH 上 CUPID 和 CUPID-Quality 都优于全数据基础策略；从 Figure 3 目测，约 50% 过滤处，基础策略约 `0.42`，CUPID 约 `0.50`，CUPID-Quality 约 `0.55`。这些是图上近似读数，不是论文表格中的精确数值。论文文字的更稳健结论是：CUPID-Quality 在混合质量任务上取得最高策略成功率，并在 Transport-MH 使用少于原始 300 条数据的 33% 时超过官方 Diffusion Policy。

这与当前实验并不矛盾：当前门槛检验“Bootstrap 能否更准地恢复独立子池底部成员”，而论文最终检验“直接按分数过滤后，重新训练的策略是否更好”。前者是新增的代理指标，不能替代后者的因果结果。

论文也明确承认四个与当前问题一致的限制：合适的 `k` 尚未解决；贪心 top-k 忽略 Demo 之间的交互；REINFORCE 式 performance influence 可能高方差；当前作者实验为降低成本只用 `C=1` policy checkpoint，并指出 `C>1` TRAK ensemble 可能提高准确度。

### 6.1 CUPID-Quality：当前最有根据的替代方法

CUPID-Quality 将归一化后的 performance influence 权重设为 `0.50`，再将 `min_of_max` 和 `max_of_min` 两个 action-influence 质量项各赋权 `0.25`。论文认为该质量项在 Transport 这类精细、混合质量任务中能降低噪声，但在策略鲁棒性或伪相关任务上可能引入偏差。

本次 Transport 的正式 `online_trak_influence.pkl` 已经包含 `sum_of_sum`、`min_of_max` 和 `max_of_min` 的全 100-rollout 分数。因此生成 CUPID-Quality 排名不需要重新训练基础策略、不需要新增 rollout，也不需要重算 TRAK；只需要按官方 notebook 的 `[0.50, 0.25, 0.25]` 归一化权重生成和冻结排序。

基于这份正式产物做的事后诊断表明，CUPID-Quality 与原 CUPID 高度相关，但确实改变了过滤成员：两套全量分数的 Pearson 相关为 `0.87875`，Spearman 排名相关为 `0.85813`。不同过滤数量下的低分集合重合如下：

| 过滤数量（比例） | 重合数量 | Jaccard | 每种方法独有数量 |
|---:|---:|---:|---:|
| 19（10%） | 14 | `0.5833` | 5 |
| 48（25%） | 37 | `0.6271` | 11 |
| 96（50%） | 83 | `0.7615` | 13 |

在建议的 50% 过滤点，人工质量标签均值为：

| 方法 | 被过滤 96 条 | 被保留 96 条 | 保留减过滤 |
|---|---:|---:|---:|
| CUPID | `1.8819` | `2.3403` | `+0.4584` |
| CUPID-Quality | `1.8229` | `2.3993` | `+0.5764` |

CUPID-Quality 因而比 CUPID 更偏向删除低人工质量 Demo，并把保留/过滤质量间隔扩大约 `0.1180`。这支持把它纳入重训练比较，但人工标签分离并不等于策略成功率提升，不能据此宣布方法通过。

判断：这是最便宜、最贴近作者 Transport 结果的替代分支。它仍不能保证通过旧 Bootstrap 门槛；应由过滤重训练的独立成功率判断。

### 6.2 Stability selection / complementary pairs

Stability selection 的原始目标是把子采样与变量选择器结合，对某些误发现率提供有限样本控制；complementary-pairs 版本改进部分误差界。它们并不保证在“预测另一子池的最低均值成员”上优于直接排序。当前结果正是稳定选择常见的保守性/功效权衡：假阳性风险可能下降，但会漏掉高方差的真实极端项。

判断：理论上可以让稳定性解释更规范，但本地探索没有显示其能越过 `+0.03`；不适合继续作为主线挽救。

### 6.3 连续重加权

Ren et al. (ICML 2018) 用干净、无偏 validation 集上的 meta-gradient 学习样本权重；DVRL (ICML 2020) 用 validation performance 的强化学习信号学习选择概率。它们比硬阈值更能处理当前 42--44 条模糊 Demo，因为不必把边界附近条目强制变成 0/1。

限制：现有 12 条 validation Demo 很小；Diffusion Policy 的最终成功率不可直接微分；接入会改变训练算法和算力预算。它们可能改善最终策略，但不能直接“通过当前 Bootstrap 门槛”。

判断：作为第二优先级新分支有研究价值，前提是先定义可信的 validation utility，防止对 12 条 validation 过拟合。

### 6.4 Data Shapley / Data Banzhaf

Data Shapley 定义了相对于模型性能的联盟边际贡献；Data Banzhaf 特别针对随机训练导致的数据价值排名不稳，论文报告其 semivalue safety margin 更大。这与当前“高方差成员”的问题相关。

限制：对 192 条轨迹做联盟式估值通常需要大量子集训练/性能评估。即使使用近似，也会从 CUPID/TRAK 复现变成另一套研究方法，成本远高于一次简单 bottom-K 重训练。

判断：Data Banzhaf 是理论上最贴近“随机训练下鲁棒估值”的替代评分方法，但不是当前最小、最快的下一步，也不能保证在本任务上超过直接 CUPID 排序。

### 6.5 多模型/多 seed 的 TRAK 聚合

TRAK 本身设计为可利用少量已训练模型进行可扩展归因；当前实验只有一个训练 seed、一个 checkpoint（`model_id=0`）。跨训练 seed/模型聚合可以引入真正的新信息，而不是在同一 100-rollout 池中反复重采样。

限制：至少需要新增基础策略训练、rollout 和 TRAK；当前正式 TRAK 单次约 10.2 小时且占约 12 GiB，整体成本高。它更适合确认“跨模型稳定”，不一定改善同一模型上的最低分预测。

判断：如果论文主张必须是“跨训练随机性的稳定归因”，这是最有针对性的确认方法；如果目标只是判断过滤能否提高策略性能，优先级低于直接重训练。

### 6.6 机器人模仿学习基线

CUPID 论文直接比较了 DemInf、Demo-SCORE、Success Similarity、Random 和 Oracle。论文在 RoboMimic mixed-quality 任务中观察到：DemInf 更能提高人工质量标签，但 CUPID 的策略成功率通常匹配或超过 DemInf；Demo-SCORE 和 Success Similarity 在成功/失败 rollout 状态相似时较弱。Demo-SCORE 还需要在训练过程多个 checkpoint 上收集 rollout 并训练分类器，不能直接从当前单 checkpoint 的产物公平复现其作者协议。

领域内其他工作也支持“轨迹质量不能只看单一分类损失”：*Data Quality in Imitation Learning* 从 action divergence 和 transition diversity 分析质量；Demo-SCORE 使用在线经验训练成功/失败分类器；USN 用不确定性感知选择和负学习处理 action noise。这些方法解决的噪声模型与当前 performance influence 不同，适合作基线，不适合为了通过旧门槛临时替换主评分。

### 6.7 GraNd / EL2N / forgetting / AUM / coreset

这些方法主要在有离散标签的监督分类中利用损失、margin 或训练动态识别噪声/冗余样本，或保持数据覆盖。当前单位是长轨迹 Demo，目标是下游机器人策略成功率，低分 Demo 也可能是困难但有价值的行为模式。

判断：可作为辅助特征或误标诊断，不能不经适配就替代 performance influence；直接用于过滤有较高任务错配风险。

## 7. 推荐的方案调整

### 7.1 冻结本轮结论

保持：

```text
PASS_VARIABLE_K_DIAGNOSIS_BOOTSTRAP_MEMBERSHIP_NOT_PROVEN
```

不修改 `+0.03` 门槛，不追溯授权 Bootstrap 过滤重训练。可以在论文中把它报告为负结果：Bootstrap 稳定化没有优于同数量直接排序。

### 7.2 新建作者方法因果复现分支

不再以 `K=13` 作为主方案。它只有约 6.8% 过滤，虽然跨池成员精度高，但与作者 Transport-MH 取得明显策略增益的过滤区间不一致；而且离线成员精度不能决定最优策略性能。

资源受限的第一阶段建议固定为作者曲线中最有信息量的 `50%` 过滤，即从 192 条训练 Demo 中删除 96 条、保留 96 条；保留数量正好是原始 300 条数据的 32%，与论文“少于 33% 原始数据”的 Transport 设置对齐。至少比较四臂：

| 训练臂 | 目的 |
|---|---|
| 未过滤 | 原始性能基线 |
| CUPID 直接最低 50% 过滤 | 复现作者的 performance-influence 排序 |
| CUPID-Quality 直接最低 50% 过滤 | 检验论文在 Transport 上最强的组合方法 |
| 随机 50% 过滤 | 区分“选对成员”与“仅仅少用一半数据” |

使用未参与成员选择的新 rollout seeds；训练预算、validation/holdout、checkpoint 规则和评估 seeds 完全一致。现有未过滤 seed-0 checkpoint 可作为 pilot 基线，但所有四臂都必须在同一组新评估 seeds 上重评。seed 0 只作 pilot；若 CUPID 或 CUPID-Quality 达到预先定义的实际改进，再扩展训练 seeds 1、2。最终主张由跨训练 seed 的配对重训练决定。

若算力允许完整复现作者比例曲线，再增加 `25%` 过滤（删除 48 条）；不建议第一轮同时跑 `10%,25%,50%,75%,90%` 后只报告最佳比例。完整曲线可以做，但必须全部报告并处理多重比较。

### 7.3 新门槛检验不同主张

新协议不再要求作者式直接排序越过“Bootstrap 相对 bottom-K”的旧门槛，而应预注册：

1. CUPID-50% 和 CUPID-Quality-50% 相对未过滤的独立 rollout 成功率差；
2. 两种方法相对随机删 50% 的成功率差；
3. 跨训练 seed 的方向一致性和效应区间；
4. 失败时停止，不再根据同一评估结果换过滤比例或组合权重。

具体最小有意义差异和 rollout 数量应在运行前按可接受算力做功效/精度规划。100 条 rollout 适合作 pilot，不足以对约 5 个百分点的小提升做强确认。

### 7.4 何时再做连续权重

只有在作者式 CUPID/CUPID-Quality hard filtering 出现以下情况时，再进入软权重分支：

- 排序信号高但硬删除导致性能下降；
- 结果对 K 明显敏感；
- 边界附近 Demo 被证明既有正作用也有负作用。

软权重分支应保留未过滤、hard bottom-K 和随机同数量三类基线，并使用新的 validation/rollout 证据，不能继续从当前 100 条有限池调温度参数。

## 8. 最终判断

| 问题 | 判断 |
|---|---|
| 这个门槛关键吗？ | 对 Bootstrap 额外价值的主张关键；对 CUPID 排序和过滤的最终因果价值不是终局门槛 |
| 是门槛太高导致失败吗？ | 不是主要原因；实际均值为负，描述性区间也低于 0 |
| 加更多普通 Bootstrap 有用吗？ | 很可能无用；问题是目标错配，不是 Monte Carlo 重复数不足 |
| 换分层/子采样能直接通过吗？ | 当前探索不能；最好仅 `+0.006`，未达 `+0.03` |
| 换 Data Shapley/Banzhaf 能保证通过吗？ | 不能保证，且成本和方法变化很大 |
| 最值得做什么？ | 冻结 Bootstrap 负结果，另开作者式 CUPID-50% / CUPID-Quality-50% / 未过滤 / random-50% 的独立重训练 pilot |
| 连续权重值得吗？ | 值得作为第二阶段，但需要新协议和可信 validation utility |

## 9. 统计风险检查

总体置信度：`CAUTION`。正式门槛结论证据充分；机制分析和替代方法比较仍受单任务、单训练 seed、固定 100-rollout 池限制。

Fallacy Scan：`11/11` 已检查。

| 风险 | 判断 |
|---|---|
| Simpson's paradox | NOTE：未观察需要汇总/分组方向反转的预设群组 |
| Ecological fallacy | NOTE：结论限定到 Demo/rollout，不外推到人或真实世界总体 |
| Berkson's paradox | CAUTION：MH 数据和固定训练划分是选择后样本 |
| Collider bias | NOTE：成功/失败参与构分，但未据此作因果控制解释 |
| Base-rate neglect | NOTE：44/56 成功失败基率已报告 |
| Regression to mean | CAUTION：对象按极端低分选择；独立子池缓解但不消除 |
| Survivorship bias | NOTE：100 条 rollout 全纳入 |
| Look-elsewhere effect | CAUTION：新增重采样与 K 表是事后探索，不能作确认性通过依据 |
| Garden of forking paths | CAUTION：新方法若不另行预注册会产生较大研究者自由度 |
| Correlation != causation | CAUTION：离线成员一致性不能证明删后性能改善 |
| Reverse causality | NOTE：当前未提出方向性因果机制 |

## 10. 可验证文献

完整检索记录和来源限制见 `allinone.md`。关键来源：

- Meinshausen & Buhlmann, 2010, *Stability Selection*, DOI `10.1111/j.1467-9868.2010.00740.x`.
- Shah & Samworth, 2013, *Variable selection with error control: Another look at stability selection*, DOI `10.1111/j.1467-9868.2011.01034.x`.
- Ren et al., 2018, *Learning to Reweight Examples for Robust Deep Learning*, ICML/PMLR 80.
- Ghorbani & Zou, 2019, *Data Shapley*, arXiv `1904.02868`.
- Yoon et al., 2020, *Data Valuation using Reinforcement Learning*, ICML/PMLR 119.
- Wang & Jia, 2022, *Data Banzhaf*, arXiv `2205.15466`.
- Park et al., 2023, *TRAK*, arXiv `2303.14186`.
- Hammoudeh & Lowd, 2024, *Training data influence analysis and estimation: a survey*, DOI `10.1007/s10994-023-06495-7`.
- Agia et al., 2025, *CUPID: Curating Data your Robot Loves with Influence Functions*, arXiv `2506.19121`, CoRL 2025.
- Belkhale et al., 2023, *Data Quality in Imitation Learning*, arXiv `2306.02437`.
- Chen et al., 2025, *Curating Demonstrations using Online Experience* (Demo-SCORE), DOI `10.15607/rss.2025.xxi.071`.
