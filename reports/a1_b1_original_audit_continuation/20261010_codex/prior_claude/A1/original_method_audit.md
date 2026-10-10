# A1 原版方法审计：关系表示的信息损失与成本

标签：论文 `READ_PRIMARY`；官方代码 `OFFICIAL_CODE_AUDITED`（静态）；项目 `CP_DISR_ADAPTATION`。无任何官方运行。

## 1 逐项回答（A-P）
| 问题 | 回答 | 证据 |
|---|---|---|
| 图包含动作节点/模式？ | 否。只有对象与命题（谓词×目标状态）节点，边标为参数位置 | P1 Def 3.1；`ilg.cpp:26–139` |
| 对象/类型/静态/目标/当前事实？ | 对象：统一色（无类型）；静态事实：ILG 不使用 `Problem.statics`，只有当它们作为状态原子出现才进入；目标：有（ag/ap/ug）；当前事实：有。“可输入”（`state_representation=all`）与“默认实际使用”（`downward`，只保留翻译原子）不同 | `ilg.cpp`；`classic_dataset_creator.py:34–47`；`classic.toml` |
| 初始色、迭代、聚合 | 初始色=节点色；L 轮（论文实验 L=4，默认 2）；论文定义为多重集，默认配置 `hash=set`（丢计数） | P1 §2–3；`wl.cpp`；`classic.toml` |
| 节点名/ID 是否影响可分性？ | 节点颜色不含名字；词表只含颜色哈希。本卡未发现名字影响，但**未做实验验证** | 代码阅读 |
| 词表与未见颜色 | 训练图上收集；推理未见颜色被忽略，且含未见邻居的节点在之后各轮被整体丢弃（`wl.cpp:48–66`） | P1 §3；代码 |
| 项目 `a-m` 的真实含义 | `PruningOptions::ALL_MAXSAT`：在收集后的训练图上，列完全相同的特征并成一组，MaxSAT 在“被保留特征的祖先必须保留”和“每组至少留一个”约束下最小化保留数，再重映射词表。按设计在训练集上保持划分 | `pruning_options.cpp:6`；`bulk_pruners.cpp:11–113`；`features.cpp:219–300` |
| 评分读出、取整 | 线性读出 `x·w`（无截距）；官方 C++ 启发式取 `round`→int | `features.cpp:493–500`；`wlgoose_heuristic.cc:23–26` |
| 作者是否已有恢复信息的配置？ | 部分：`pruning=none`、`multiset_hash`、`state_representation=all`（数据侧）、更多迭代、其他图生成器。规划器侧只见 FD 事实；其代价（特征数 202 vs 147 等）为已有数字，但任何组合都**未在本项目评测** | `classic.toml`；`wlf_generator.cc` |
理论边界：P1 Cor 4.5 “当前特征生成器无法在所有领域学习 h*”；P1 §5 “有限训练集会使语义不同的特征看起来相同”。这些是针对其 WL/2-LWL 变体的陈述，不能外推到 GNN/RGCN/CP-DISR 的 Action–Proposition 图。

## 2 既有证据复核（A-E）
- **836**：=唯一有序（问题，更好后继状态，更差后继状态）对，占 9875 个唯一对的 8.47%；覆盖 78 题；对应 844 个动作层严格对；819 个更好端属于父状态的精确最优集；d* 差为 2 的 799 个、为 1 的 37 个；8 个来自多个父决策；标签冲突 0；均为 Train96、typed、同问题同父（`A1/zero_diff_836_profile.json`）。“被丢弃的 1 类”是这 836 个，不是 1 个对。
- **逐层链（24 个探索候选，`A1/loss_pipeline_evidence.csv`）**：PDDL 状态 DIFFERENT；适配器原子 DIFFERENT；初始 ILG 颜色直方图 SAME；未剪训练词表的 WL 颜色 DIFFERENT（23）/UNKNOWN（候选16，阶梯内未分）；a-m 投影后 SAME；读出 SAME；部署取整 SAME；搜索比较 UNKNOWN。图同构只对候选16做过（ISOMORPHIC，唯一映射交换 hoist0↔hoist1，其静态位置不同）。
- 选择机制：从 1045 个零差候选（S1 873/S2 172）按固定哈希取 12+12，不是随机样本；**23/24 不能外推 836**。
- 区分：`SAME/DIFFERENT/UNKNOWN` 之外，“图同构”“WL 不可分”“有限深度未分”“哈希碰撞”未被区分——只检验了深度与剪枝/词表/聚合。
- 候选16：静态原子作为状态原子后 set/2 仍不可分，multiset/2 可分；再加类型 set/2 可分。联合变化，不能分别归因；是否等价于官方 `state_representation=all` 取决于类型是否被当作一元原子，未核。
- 特征不同分数相同：W1 训练 1300、库 B 585 对，读出抵消（`readout`）；WL 另有取整并列 1838（训练）/589（库 B）。读出只有 8 个非零权重（7 个第0层）。不自动视为表示缺陷。
- 去剪枝改变维度时旧权重无对应系数，本卡未给新特征补权重，也未做任何“去剪枝旧权重”对照。

## 3 成本（仅既有记录）
历史卡 `cost_breakdown.csv`：IPC22 上神经评分约占墙钟 99%，WL 约 84%（共享 GPU/CPU、300 s，含翻译）；Joint 面板 WL 约 2–11%。特征数：W1 147（未剪训练词表 202），模型文件 ≈15 KB。图规模、单次评分时间与端到端不能由此互相推出，计时环境不一致，不作速度结论。

## 4 来源分类（摘要，见 04）
a-m 现象：UNRESOLVED_ORIGIN；类型/静态盲：METHOD_LIMIT（类型）+ AUTHOR_CONFIG_TRADEOFF（静态）；取整：AUTHOR_CONFIG_TRADEOFF（官方）；读出稀疏：UNRESOLVED_ORIGIN（小训练集）。没有一项被确认为 `CP_ADAPTATION` 造成。
