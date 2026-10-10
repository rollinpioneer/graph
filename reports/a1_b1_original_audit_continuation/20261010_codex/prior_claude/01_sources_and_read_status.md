# 01 来源与阅读状态

状态词：`READ_PRIMARY`（实际读到原文/代码/数据）、`INDEX_ONLY`、`USER_REPORTED`、`UNAVAILABLE`。
说明：网页/论文通过抓取工具取得，工具返回的是对页面文本的抽取摘要而非逐字全文；表中“节”号来自抽取结果，页码拿不到时只写节/定理。PDF 为压缩流的（Return to Tradition PDF、ICAPS 2024 Expressiveness PDF）无法解析，改用 arXiv HTML 或第三方文本镜像，已标明。

## 1 论文
| 编号 | 资料 | 版本/日期/来源 | 状态 | 实际读到 |
|---|---|---|---|---|
| P1 | Return to Tradition（Chen, Thiébaux, Trevizan） | arXiv 2403.16508 v1（2024-03-25）；ICAPS 34(1) 68–76；代码 Zenodo 10.5281/zenodo.10757383 | READ_PRIMARY（arXiv HTML v1，经抽取） | ILG 定义（§3 Def 3.1）、WL 与特征（§2–3，实验 L=4）、未见颜色忽略（§3）、数据（§5：IPC23 学习赛易例，Scorpion 最优计划，计划状态+cost-to-goal）、局限（§4–6） |
| P2 | WLPlan: Relational Features for Symbolic Planning | arXiv 2411.00577 v1（2024-11-01） | READ_PRIMARY（HTML，仅数值 ν-ILG 节：Def 4.1、§5.1–5.2）；经典 ILG 细节未取得 | 数值 ILG 节点/边/WL；“未见颜色被忽略”；未描述剪枝、取整 |
| P3 | WL Features: A 1,000,000 Sample Size Hyperparameter Study（Chen） | arXiv 2508.18515 v1 | READ_PRIMARY（HTML，部分） | 只定义了 none 与 i-mf；a-m 未定义；Table 2 摘要 |
| P4 | State Encodings for GNN-Based Lifted Planners（Horčík 等） | AAAI 2025, 39(25) 26525–26533 | READ_PRIMARY（第三方文本镜像） | 编码比较（object/atom/object-atom/Rel）；不含 ILG 与 GOOSE 的专门结论；类型=一元谓词标签；静态事实未单列 |
| P5 | Expressiveness of GNNs in Planning Domains（Horčík, Šír） | ICAPS 34(1) 281–289 | INDEX_ONLY（摘要）；正文 PDF 无法解析 → 图 7 等未核 | 摘要：不可区分状态与编码的关系 |
| P6 | Learning Domain-Independent Heuristics for Grounded and Lifted Planning（AAAI 2024） | — | UNAVAILABLE（未读，不改变任何裁决） | — |
| P7 | Guiding GBFS through Learned Pairwise Rankings（Hao 等） | IJCAI 2024, 6724–6732, doi 10.24963/ijcai.2024/743 | READ_PRIMARY（第三方文本镜像，公式符号有损） | §3 理论与约束，§4.1–4.3 损失/数据，§6 实验，§7 局限 |
| P8 | Optimize Planning Heuristics to Rank（Chrestien 等） | NeurIPS 2023；arXiv 2310.19463 v1 | READ_PRIMARY（arXiv HTML v1，经抽取） | Def 1/Thm 1，损失 L01/L*/L_gbfs/L_rt，数据，局限 §8.3–8.5 |
| P9 | ECAI 2025 特征选择（Hao 等）、Action Optimality Checking、Policy Comparison Oracles、Efficient Lookahead Encoding | 卡片索引 | INDEX_ONLY（未读） | — |
读取预算：A1 核心 P1/P4/P5 + 后续 P2/P3；B1 核心 P7/P8；各路后续论文 ≤3。

## 2 官方代码（本机副本，静态阅读，未运行）
| 编号 | 资料 | 身份 | 状态 |
|---|---|---|---|
| C1 | `~/ext/goose`，origin https://github.com/DillonZChen/goose.git | main `04dbe8f27a8ec027626e6c6b5d6e7610b81f9d57`（2026-03-27）；论文时分支 `icaps24` 未取得；README 引用 P1 | READ_PRIMARY |
| C2 | `~/ext/goose/ext/wlplan`（子模块） | `9d9fc99165135bfe8e2817145888ede162e4a97a`，`git describe` = v2.1.0-12-g9d9fc99（2026-03-27）；已安装 wheel 的 `package_version` 字符串为 2.2.0，wheel/shim 的 `_wlplan*.so` sha256 `0089e22796904c5e…`（两处相同）；wheel 由 `setup_goose.sh` 用 `uv pip install .` 从该子模块构建 | READ_PRIMARY（源码）；wheel 与源码逐字节一致未验证 |
| C3 | `ext/planners/scorpion`（分支 scorp-goose） | `4c16ce05b31d68b67cb126a3a7c31e9123ba50c6`（2025-11-07） | READ_PRIMARY（goose/wlgoose_heuristic.cc、features/wlf_generator.cc） |
| C4 | GitHub 页面 DillonZChen/goose、DillonZChen/wlplan | 页面登陆页 | INDEX_ONLY |
没有取得官方原生系统的运行记录（`OFFICIAL_EXISTING_RUN_VERIFIED` 不成立）。

## 3 项目资料（冻结提交 `cc7d79d6fcb053bf0c16033e765a2e808de2aaac`，已从仓库核对为该提交）
READ_PRIMARY：V2 的 `final/*`、`diagnostics/{packages,state_manifest,pairs,library_stats}`、`scores_*.json`、`b_pair_outcomes.csv.gz`、`b_repair_damage*.csv`、`training/W1/*`、`registration/panels.json`、`results/panel_by_case.csv`；SCOPE 的 `results/*`、候选对；OLD 的 `data/{manifest,exact}.json`、`goose/train_typed/*`；源码 `compact/{w1,diagb,crossparent,train}.py`、`scripts/c1_compact_w1_fit.py`、`search_match/{engine,evaluators,runner}.py`（含 `WlEval`、取整）。
最新 A1/B1 = 上一张执行卡的输出（未入库）：`/home/xushijie3/work/cp_disr_a1_b1/CP_DISR_A1_B1_20261010T021755Z_cc7d79d6f/`，回执 sha256 `fa6b3adfa469cf1a1ccd77a2bac38ac821f90dc909dfc32730675bc4d94bbd15`；基于冻结提交 `cc7d79d6f`、读取同哈希输入（`input_manifest.csv` 131 项均与提交 blob 相同）；**不属于**该提交，也未提交/推送。

## 4 用户线索核实
| 线索 | 状态 | 证据位置 |
|---|---|---|
| 836 个严格对零特征差 | 已核实：=唯一有序状态对 836（占 9875 的 8.47%，78 题，844 个动作层对，819 个更好端属精确最优集，d* 差 799 个为 2、37 个为 1，8 个来自多个父决策） | `A1/zero_diff_836_profile.json` |
| 抽查 24 个、23 个去剪枝后可分 | 已核实；抽样为哈希确定的非随机抽样（S1 12/S2 12，来自 1045 个零差候选），不外推 836 | `A1/loss_pipeline_evidence.csv` |
| W1 有特征不同但分数相同的对 | 已核实：训练对 1300、库 B 585（跨父 375+同父 210）；读出抵消，不是表示丢失 | 上一卡 `a1/readout_categories.json` |
| 旧 WL 取整产生并列 | 已核实且定位：取整在**官方**规划器启发式内 `wlgoose_heuristic.cc:23–26` | C3 |
| 一个候选需静态位置+类型 | 已核实但需修正：候选 16；加静态原子且 multiset/2 即可分，set/2 需再加类型；联合变化不能分别归因，且上一卡关于 `Problem(...,statics)` 的解释有误（见 05） | 上一卡 `a1/candidates.csv` |
| 尚无实际搜索损失证明 | 已核实：24 个候选无搜索事件 | `A1/loss_pipeline_evidence.csv` |
| T1-S 112 修复/36 损伤 | 已核实：唯一对 ID 148（另有同父 25 对），全部精确标签，无并列介入；修复=反序→正确 | `B1/pair_to_search_evidence.csv` |
| 同父/首选也变化 | 已核实（同父 17/8；首选 4/2） | 上一卡 `b1/decision_changes_summary.json` |
| 库 B 不含 IPC22；D0/T1 无前沿 | 已核实 | 上一卡 `b1/problem_join_summary.json` |
