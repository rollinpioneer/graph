# CUPID Transport-MH 替代选择方法文献检索记录

检索日期：2026-07-25  
查询范围：数据估值、训练数据选择、稳定选择、连续重加权、数据剪枝；以 2017--2026 为主，并补充 2010/2013 年稳定选择基础论文。

## 检索工具状态

`paper-search` 聚合器未产生完整可用结果，错误/异常原样记录如下：

| 来源 | 状态 |
|---|---|
| OpenAlex（聚合器适配器） | `[open_alex] unavailable (import failed: unsupported operand type(s) for \|: 'type' and 'NoneType'); skipping this source.` |
| Semantic Scholar | 45 秒后 `timeout` 退出码 `124`，无 stderr |
| arXiv | 45 秒后 `timeout` 退出码 `124`，无 stderr |
| OpenReview / Crossref / DBLP | 全来源聚合调用持续无输出后被中止，因此没有完成结果，不能解释为 0 篇 |

后备检索使用 `scholar-search` 的 OpenAlex API 和 Python 3.12，只发送公开主题/标题关键词，不上传本地实验数据。随后使用 `arxiv-analyze` 完整读取 CUPID 原论文 arXiv `2506.19121`。以下是与决策直接相关、已由 OpenAlex 或 arXiv 原文核实的去重结果。引用数是检索时 OpenAlex 记录值，可能因预印本/会议版本拆分而低估或重复。

## OpenAlex 后备结果（10 篇）

| # | 论文 | 年份 | 来源 | 引用数 | 相关性 |
|---:|---|---:|---|---:|---|
| [1](https://doi.org/10.1111/j.1467-9868.2010.00740.x) | Stability Selection | 2010 | JRSS B | 2186 | 子采样稳定选择与误发现控制的基础方法 |
| [2](https://doi.org/10.1111/j.1467-9868.2011.01034.x) | Variable selection with error control: Another look at stability selection | 2013 | JRSS B / Cambridge repository record | 213 | complementary-pairs stability selection 和改进误差界 |
| [3](http://proceedings.mlr.press/v80/ren18a/ren18a.pdf) | Learning to Reweight Examples for Robust Deep Learning | 2018 | ICML | 418 | 用干净 validation 的 meta-gradient 学习样本权重 |
| [4](https://arxiv.org/abs/1904.02868) | Data Shapley: Equitable Valuation of Data for Machine Learning | 2019 | arXiv / ICML work | 152 | 以模型性能联盟边际贡献定义数据价值 |
| [5](http://proceedings.mlr.press/v119/yoon20a/yoon20a.pdf) | Data Valuation using Reinforcement Learning | 2020 | ICML | 24 | 用 validation reward 学习样本选择概率 |
| [6](https://arxiv.org/abs/2001.10528) | Identifying Mislabeled Data using the Area Under the Margin Ranking | 2020 | arXiv / NeurIPS work | 45 | 用训练 margin 动态识别误标样本 |
| [7](https://arxiv.org/abs/2107.07075) | Deep Learning on a Data Diet: Finding Important Examples Early in Training | 2021 | arXiv / NeurIPS work | 81 | GraNd/EL2N 数据剪枝分数，多初始化平均 |
| [8](https://arxiv.org/abs/2205.15466) | Data Banzhaf: A Robust Data Valuation Framework for Machine Learning | 2022 | arXiv / AISTATS work | 10 | 针对随机训练性能噪声设计更大 safety margin 的 semivalue |
| [9](https://arxiv.org/abs/2303.14186) | TRAK: Attributing Model Behavior at Scale | 2023 | arXiv / ICML work | 10 | 可扩展归因，可利用少量训练模型 |
| [10](https://doi.org/10.1007/s10994-023-06495-7) | Training data influence analysis and estimation: a survey | 2024 | Machine Learning | 55 | 影响定义、估计假设、复杂度与局限综述 |

## 机器人模仿学习领域结果（4 篇）

| # | 论文 | 年份 | 来源 | 引用数 | 相关性 |
|---:|---|---:|---|---:|---|
| [11](https://arxiv.org/abs/2506.19121) | CUPID: Curating Data your Robot Loves with Influence Functions | 2025 | CoRL / arXiv | 未取 | 当前复现的原方法；直接 top-k，Transport 上 CUPID-Quality 最强 |
| [12](https://arxiv.org/abs/2306.02437) | Data Quality in Imitation Learning | 2023 | arXiv | 7 | 从 action divergence 和 transition diversity 定义 IL 数据质量 |
| [13](https://doi.org/10.15607/rss.2025.xxi.071) | Curating Demonstrations using Online Experience | 2025 | RSS | 2 | Demo-SCORE；用在线成功/失败经验训练分类器过滤示范 |
| [14](https://doi.org/10.1613/jair.1.15819) | USN: A Robust Imitation Learning Method against Diverse Action Noise | 2024 | JAIR | 2 | 不确定性感知样本选择和负学习，应对 action noise |

### Semantic Scholar（0 个完成结果）

请求超时，不能解释为没有匹配论文。

### arXiv（0 个完成结果）

请求超时；相关 arXiv 条目由 OpenAlex 后备结果核实。

### OpenReview（0 个完成结果）

聚合调用未完成，未得到可报告结果。

### Crossref（0 个完成结果）

聚合调用未完成，未得到可报告结果。

### DBLP（0 个完成结果）

聚合调用未完成，未得到可报告结果。

### Model Knowledge（0 个新增条目）

为避免把未实时核实的记忆条目混入方法决策，本报告不添加纯模型记忆论文。

## 汇总

### Overview

完成检索的语料包含 14 篇直接相关论文，覆盖稳定选择、影响归因、联盟式数据估值、连续权重、训练动态剪枝和机器人示范筛选。没有论文能保证某替代方法在当前 Transport-MH 的“相对同数量直接最低分精度增益 >= 0.03”门槛上通过；CUPID 原论文也不使用这一门槛。

### Trends

2010--2013 年工作重点是高维变量选择的错误控制；2018--2020 年转向用 validation 信号学习训练样本权重/价值；2021--2024 年重点扩展到低成本剪枝、大模型归因和影响估计鲁棒性。多数实证来自监督分类或语言任务，直接迁移到长轨迹机器人模仿学习存在任务错配。

### Key themes

1. 稳定选择与错误控制：[1], [2]。目标是控制选择错误，不保证胜过点估计排序。
2. Validation 驱动的连续选择：[3], [5]。更接近最终性能，但要求可信 validation utility。
3. 联盟式数据价值：[4], [8]。定义原则性强，计算成本高；Banzhaf 特别处理随机性能噪声。
4. 训练动态剪枝：[6], [7]。成本低，但主要面向有离散标签的监督学习。
5. 可扩展影响归因：[9], [10]。与现有 CUPID/TRAK 链路最接近，但归因仍是近似且模型依赖。
6. 机器人示范筛选：[11]--[14]。最终目标是策略成功率，数据质量、成功分类和 action-noise 方法解决的是不同噪声机制。

### Keywords frequency

以下为 10 个去重标题的大小写无关词频：

| Keyword | Count |
|---|---:|
| data | 8 |
| selection | 3 |
| valuation | 3 |
| stability | 2 |
| training | 2 |

### Most cited by accepted paper

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| 1 | Stability Selection | 2010 | 2186 |
| 2 | Learning to Reweight Examples for Robust Deep Learning | 2018 | 418 |
| 3 | Variable selection with error control: Another look at stability selection | 2013 | 213 |
| 4 | Data Shapley | 2019 | 152 |
| 5 | Deep Learning on a Data Diet | 2021 | 81 |

### Most cited by first author

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| 1 | Nicolai Meinshausen | 1 | 2186 |
| 2 | Mengye Ren | 1 | 418 |
| 3 | Rajen D. Shah | 1 | 213 |
| 4 | Amirata Ghorbani | 1 | 152 |
| 5 | Mansheej Paul | 1 | 81 |

### Recommendations for reading

1. [1] 和 [2]：先明确稳定选择真正保证的是选择错误控制，而不是排序精度优势。
2. [9] 和 [10]：理解当前 TRAK 分数的能力、近似性质和模型依赖性。
3. [3]：评估从硬删除转向 validation 驱动连续权重的实现条件。
4. [8]：若研究问题升级为“随机训练下的鲁棒数据估值”，再评估 Banzhaf 的计算代价。
5. [7]：作为便宜辅助分数参考，但不要直接把分类剪枝结论外推到机器人轨迹。
6. [11]--[13]：用 CUPID 原协议和最接近的 Demo-SCORE 基线校准下一轮机器人实验，而不是只依赖通用数据剪枝文献。
