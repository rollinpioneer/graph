# A1 原作者方法补充审计（2026-10-10）

状态：`COMPLETE_WITH_BOUNDED_UNKNOWNS`。先完成静态审计：阅读论文分页文本、固定源码、历史脚本和已保存表，并派生表格统计。随后用户明确批准 A-V，执行协调者完成固定安装版本的6条特征流水线，本报告真实读取 `../A_V/receipt.json` 并更新剪枝裁决为 `FIXED_BUILD_PRUNING_DIFFERENCE_REPRODUCED`。静态审计结论保留，新增实验单列于 §5。本分支没有 SSH 或计算操作；新增实验没有学习拟合、评分、搜索或计划验证，旧报告和实验资产未修改。

## 1. 实际阅读范围与身份

来源根：`C:/Users/jackx/Desktop/CP-DISR-HANDOFF/Original_Sources_20261010`。下文 PDF 页码是本地文件从 1 开始的页序；Expressiveness 同时给会议页码。实际依据为同名 `papers/*.txt` 中的分页原文，没有把资料包 README 当成论文正文。

| 来源 | 实际阅读范围 | 身份与限制 |
|---|---|---|
| Return to Tradition，arXiv:2403.16508v1，2024-03-25 | PDF 第 2–4、6、9 页；WL、ILG、理论与实验协议、官方代码引用 | `READ_PRIMARY` / `PAPER_ONLY`；没有原生模型运行 |
| Expressiveness of Graph Neural Networks in Planning Domains，ICAPS 2024，Horčík & Šír | PDF 第 1–3、6–8 页，尤其定理 1/2/3、图 7 和实验边界 | `READ_PRIMARY` / `PAPER_ONLY`；会议页 281–288 |
| WLPlan: Relational Features for Symbolic Planning，arXiv:2411.00577v1，2024-11-01 | PDF 第 1–4 页，软件职责与 νILG 定义 | `READ_PRIMARY` / `PAPER_ONLY`；不把 νILG 当 2024 经典 ILG |
| WLFs: A 1,000,000 Sample Size Hyperparameter Study，arXiv:2508.18515v1，2025-08-25 | PDF 第 1、4–6、12 页；表 1、特征剪枝/哈希/状态表示、实验、附录 D | `READ_PRIMARY` / `PAPER_ONLY`；该论文评估 `none/i-mf`，不是 `a-m` |
| GOOSE `icaps-24` tag | `1b1f57f0c1654049605758a8ddd52ea2d657267c`；ILG、1-WL、词表、训练配置、数据入口、CPU WL 搜索代码 | `READ_PRIMARY` / `OFFICIAL_CODE_AUDITED`；没有运行。README 将该仓库关联到论文；Zenodo 元数据也指向该 tag。根许可证为 GPL v3 |
| GOOSE `icaps24` branch | `56a025daae4a6aca576899589ad87181b0ebea71`；与 tag 的上述核心文件静态对照 | `READ_PRIMARY` / `OFFICIAL_CODE_AUDITED`；不是同一版本 |
| WLPlan `v2.1.0` tag | `4aa84678d7a9dde58bac4fe0af1a16d50e5be3b6`；ILG、WL、邻居容器、收集/导出、MaxSAT 剪枝 | `READ_PRIMARY` / `OFFICIAL_CODE_AUDITED`；根许可证为 MIT。不能证明服务器 wheel 来自同一构建 |

版本哈希来自实际读取的 `metadata/archive_validation.json`（ZIP commit comment）和 `goose_refs.txt`。论文的 Zenodo 10757383 元数据实际读取：版本 `icaps-24`、`restricted`、文件列表为空。GitHub tag 源码已取得；不能声称取得或逐字节等同于受限 Zenodo 原包。

2026 main `04dbe8f`、服务器 WLPlan `9d9fc99165135bfe8e2817145888ede162e4a97a`（v2.1.0-12）与扩展二进制身份由协调者另行登记。本地 `v2.1.0` tag 不能代替这些固定输入；本报告不以同一个版本字符串证明 wheel/source 一致。

## 2. 2024 原论文与 2026 配方必须分开

Return to Tradition PDF 第 6 页 §5 的原始实验是 **L=4、邻居多重集、SVR/径向基 SVR/GPR 学习 h\***，另有 2-LWL+SVR。数据取自 Scorpion 在 IPC2023 十个领域训练题上产生的最优计划及其状态/剩余代价；该论文没有 Depots 领域。每题训练参考求解上限 30 分钟，评价 GBFS 上限 1800 秒，CPU 模型单核/8 GB，SVR/GOOSE 五次，GPR 一次。

历史 tag 的 `experiments/models/wl_ilg_{gpr,svr-linear,svr-rbf,lwl}.toml`（第 5–8 行）对应这些模型与 L=4；`learner/train_wl.py:45–99` 默认 linear-svr、ilg、1wl、4 轮、`prune=0`。`learner/models/wl/wl1.py:33–39` 收集所有邻居 `(colour,edge_label)` 后排序，保留重复项。2026 `classic.toml` 的 2 轮/set/a-m/rank-SVM 与 CP 配置一致，至多证明 CP 使用了这一后续配方；它不能说明 CP 已还原 2024 原版的八项协议。

历史 tag 与 branch 也不可互换：静态 diff 发现 tag `learner/train_wl.py:130–132` 使用 33% 验证拆分；branch `learner/train_wl.py:21,142–150` 设置 `_TRAIN_ALL=True`，训练全部输入。branch ILG 将状态节点从新整数 ID 改为谓词-参数 tuple；1-WL 增加 `_k=1`，更新主体仍保留邻居多重集。两者 `eager_search.cc`、`best_first_open_list.h` 和 `run_wl.py` 的 SHA256 相同，但不能因此宣称整个训练/搜索包相同。

## 3. ILG、词表、读出与部署

2024 ILG 定义（PDF 第 3 页 Def. 3.1）：对象与当前真命题/目标的并集，对象统一 `ob` 色，命题按谓词及 ap/ag/ug 着色，边按参数位置标号，**不含动作模式或动作节点**。它没有自动把 PDDL 类型加到对象色；静态事实是否进入取决于实际状态输入。

历史 tag `learner/representation/ilg.py:108–143,145–176` 与定义对应。当前 WLPlan tag `src/graph_generator/graph_generators/ilg.cpp:27–85,103–139,177–181` 同样只从对象、目标、传入 `state.atoms` 建图，且显式忽略动作；函数未使用 `Problem.statics`。普通问题对象统一色；常量对象在 `differentiate_constant_objects` 开启时有单独色（第 37–52 行），故不能把“所有 WLPlan 对象永远统一色”推广到 νILG/常量配置。

四参 `Problem(domain, objects, positive_goals, negative_goals)` 的第四个参数是负目标（`include/planning/problem.hpp:57–60`、`src/planning/problem.cpp:76–80`）。旧脚本 `mk_problem_s` 的静态参数误传构成错误解释；v2.1.0 的 `ilg.cpp:63–80` 确实创建负目标节点而只给正目标连边。原脚本后面的“静态事实作为状态原子”检验与该错误传参是不同路径。

原论文 PDF 第 3 页明确词表来自训练图，推理忽略未见颜色。当前 tag `src/feature_generator/feature_generators/wl.cpp:48–65` 除忽略未见节点颜色外，还令包含未见邻居的节点在下一轮成为未见颜色。`neighbour_containers/wl_neighbour_container.cpp:12–35` 在 set 模式将同 `(edge_label,colour)` 的计数置 1；最终图 embedding 仍为颜色计数直方图，不能说最终图表示是集合。

历史部署已经取整：`planners/downward_cpu/src/search/heuristics/goose_linear.cc:37–42` 计算 `bias + Σ feature*weight` 后 `round`→int；当前 WLPlan `features.cpp:488–506` 的线性预测本身返回 double。旧 CP 的部署取整可归于官方部署取舍，但不是图信息丢失；原始实值评分和取整后排序必须单列。

## 4. 历史 CPU WL 搜索协议已补读

历史 tag/branch 用 `run_wl.py:32–46`（默认 eager、timeout 参数默认 1800）和 `planners/search.py:34–70` 构造 `eager_greedy([h_goose])`，调用包内 Downward CPU。GNN 另走 GPU `batch_eager_greedy`，不能合并。

CPU `plugin_eager_greedy.cc:63–71` 设置单评分器 greedy OPEN，并明确 `reopen_closed=false`；`best_first_open_list.cc:65–84` 是整数键最小优先且同分 FIFO；`eager_search.cc:116–166` 忽略已关闭重复条目，`207–280` 处理新节点/更便宜父路径，`283–287` 在后继生成后判断目标，源码注释说明这个改动使其不再保持 A* 语义。这些是静态作者部署代码证据，不是 Scorpion 的训练计划生成协议，也不是 CP GBFS 逐步相等的证明。规范动作序、缓存、初始目标处理、计时执行器还需分别比对。

## 5. a-m 设计、旧表观察和剩余冲突

`a-m` 的真实枚举是 `ALL_MAXSAT`（`pruning_options.cpp:6`）。`bulk_pruners.cpp:11–17` 在完成收集后的训练图 embedding 上调用 `prune_maxsat`；`features.cpp:462–483` 根据特征列的完整训练计数向量分等价组。`bulk_pruners.cpp:77–94` 硬约束要求每组至少留一个代表、保留某特征时保留其全部祖先；`features.cpp:239–266` 还传播删除依赖，随后重映射颜色键（第 268–318 行）。MaxSAT 调用 PySAT RC2（`maxsat.cpp:103–138`）；静态审计阶段没有调用它，后来获准 A-V 的独立 a-m 重建属于特征收集调用，不能称整轮无特征计算。

静态推论：若输入训练图、词表和部署重映射语义相同，硬约束有效且执行结果正确，则保留等价列代表应保持**该训练矩阵的行可分性**。它不保证训练集合外所有状态，也不保证评分或搜索表现。

本轮实际读取协调者下载的旧 `prior_diagnostic/a1/` 文件，未运行脚本：

- `a1_probe_pruning.py:20–31` 由同一 `W1/data.json` 构造同一 dataset，只有 `pruning` 为 `a-m/none`；2 轮、set、ILG 相同。第 36、49–50 行按整列值匹配，不用不同词表的列 ID 直接配对。因此“两个词表编号不同”或“这里换了训练输入”不符合这个脚本的静态语义。
- `pruning_probe.json` 保存 24 条：17 条两端在训练输入、其中 16 条 none 可分而 a-m 不可分，1 条两者均不分（候选 16）；另外 7 条因成员缺失未做训练列核对。16 条中的全部 399 条候选-区分特征记录 `equal_column_kept=false`。399 不是独立特征数或独立状态对数。派生统计显式只展开这 16 条的 details，避免把 7 个缺字段/null 条目算成特征记录。
- `pipeline_rebuild_check.json` 保存历史重建 a-m=147 特征、无剪枝=202、6399 行，重建 embedding 与载入 W1 完全相同。该 JSON 是历史计算记录；静态审计阶段没有独立重算，随后获准的独立检查见 §5.1。
- `candidates.csv` 的 23/24 在 R1x 记录可分；这与上面的 17 条训练成员子集是不同分母。所有 24 条搜索状态都是 `NO_SEARCH_RECORD_FOR_THIS_PAIR`。

以上为新增 A-V 前的历史证据：它们使“普通列编号或换训练集造成假象”更不可信，但**尚不足以证明官方 a-m 有 bug**。当时缺少保存的完整剪枝前后矩阵/颜色 key 映射、MaxSAT 约束与解证书、运行时实际模块身份及与源码的构建对应关系，仍不能排除底层版本差异、重映射/导出行为、历史输出与脚本版本不一致、输入对象隐藏变异或状态索引问题。`a1_run.py:317–318` 的错误静态传参不在 `a1_probe_pruning.py:20–31` 路径上，不能拿它解释这个剪枝冲突。

836/9875=8.47% 是 `zero_difference_summary_train.json` 保存的唯一训练状态对计数；其权重份额为 7.30%，动作层总分母 10345。覆盖 78 题、799 个距离差为 2 等更细字段由协调者另核，不将这里的摘要 JSON当成这些细字段的独立依据。有限候选和训练内冲突不能外推 836 的机制比例。

### 5.1 本次用户批准并执行的 A-V

A-V 状态为 `APPROVED_EXECUTED`。本分支真实读取本地 `../A_V/receipt.json`；执行协调者另提供训练输入6399个状态、836对照和旧24候选的逐对计数。实验以同冻结输入、同安装构建，比较载入 W1、独立重建 a-m、独立重建 none，每条两次，共6条流水线。

| 检查 | 获准实验结果 | 直接依据 |
|---|---|---|
| 冻结与独立重建 a-m | 矩阵及训练划分完全相同 | receipt 的 `loaded_rebuilt_matrix_equal` / `loaded_rebuilt_partition_equal` |
| 两次重复 | 三条管线各重复逐位相同 | `repeats_bitwise_equal` 和六条矩阵哈希 |
| 训练向量组 | 6399状态：a-m/frozen=95组，none=3074组 | receipt 的各流水线 `training_distinct_vectors`；6399由协调者核实 |
| 训练划分方向 | 95个 a-m 组均被 none 拆开；none组被a-m拆开=0 | `partition_am_vs_none_equal=false`、`am_groups_split_by_none=95`、`none_groups_split_by_am=0` |
| 836目标对 | none区分813，冻结/a-m区分0 | receipt 的 `target_pairs_separated_by_none_only=813`、分母836；三者逐对计数由协调者核实 |
| 836非零对照 | 三者均区分836 | 执行协调者的本次结果核查 |
| 旧24候选 | none区分23；候选16仍不区分 | 执行协调者的本次结果核查 |
| 输入与安装扩展 | 输入哈希不变；扩展SHA不变 | receipt 的 `input_hashes_unchanged=true`；扩展核查由协调者提供 |

冻结/重建 a-m 的6,408×147矩阵哈希均为 `723ff6a3bad2d7d13784770906ea438e663abf29b4d05022ca5ba01bdd591047`；none的6,408×202矩阵哈希均为 `5bd6ec92e3954022751543845f93a7ecc0976f36486f447990c21a3cdc5f18b9`。训练向量组统计仅取6399训练状态；6,408为本次全部嵌入行数，不能与训练分母混用。本分支读取的是回执中的矩阵哈希/核查字段，没有另执行矩阵计算。

因此现象裁决为 **`FIXED_BUILD_PRUNING_DIFFERENCE_REPRODUCED`**：当前固定安装构建下，同输入 a-m 与 none 确实产生不同训练状态划分；813/836是本次目标集合直接检查，不再是从23/24候选外推。冻结重载、独立重建和重复随机性已不能单独解释该差异。候选16在无剪枝set配置中仍不分，不能把全部836归因于a-m。

原因仍为 **`UNRESOLVED_ORIGIN`**。`wheel_source_byte_identity=NOT_PROVEN`，同一安装扩展SHA不证明该二进制对应当前源码；新增实验也未检查MaxSAT硬约束/解证书、颜色映射及其源码执行路径。它不证明2024论文方法固有限制、作者实现bug，或实际搜索收益。已读v2.1.0源码的条件性保持划分推论仍成立；与固定安装行为的差异有待定位。

实验实际为4次特征collect、6条特征流水线。`training`、`learned_fitting`、`model_scoring`、`neural_forward`、`search`、`exact_queries`全部0；2个单线程worker并行，总耗时2.90秒、worker CPU总计3.78秒，采样RSS合计峰值588.3 MiB。回执搜索等级为 `E0_ONLY`，停止条件为 `NO_AUTOMATIC_EXPANSION_OR_REFIT`。

## 6. 理论边界与已有工作覆盖

Expressiveness PDF 第 2 页（会议 282）假设正规化非数值/非时间 PDDL，无条件效果、axiom、负前提，类型作为一元谓词，状态含同一实例的目标关系。第 3 页（283）定理 1 的语义同构必须保留完整 enriched state 的关系；只交换 hoist 而破坏静态位置不满足该假设。

定理 2/3 是该节定义的图消息传递（无边特征、节点 multi-hot、邻居多重集、图级多重集 readout）的 C² 边界。定理 2：C² 等价图对任意该模型参数都输出相同。定理 3：对每个有限规模上界 n，存在一个模型和参数可区分所有 ≤n 的非 C² 等价图；不是任意 RGCN/聚合器/训练模型的能力或学习保证。文中第 7 页（287）也明确其结果仅直接覆盖所选 GNN 子类。CP action-proposition 图、DENSE 注意力、有限词表/剪枝或 set WL 不能不加论证套用。

Return PDF 第 4 页 Theorem 4.1 专门处理同 ILG 上的边标签扩展，反向构造需要可保持注入性的 GIN/MLP，并非实际 4 层 RGCN 自动具备；第 6 页 Cor.4.5 是“存在领域/任务反例”的全域限制，不是 Depots 当前 836 对的来源证明。

已有办法的身份：2025 超参论文第 5 页明确 set/multiset、partial/complete 是速度、未见颜色和表达力取舍；第 6 页分别用 FD24.06 与 Powerlifted，预算 1分钟/4GB，因此其结果不能直接与 CP 300秒预算拼表。`none`、多重集、更多轮、完整状态以及高阶特征都是作者已有选择；该论文讨论 `i-mf`，不能替代 `a-m` 定义。表达恢复、拟合利用、搜索收益分别待验证，不能把这些已有机制作为新表示贡献。

## 7. 本分支交付与未知边界

本分支交付 `original_method_audit.md`、`conclusion.md`、`diff.csv`、`loss_pipeline_evidence.csv`、`pruning_probe_summary.json`。两张 CSV仅整理静态来源和已有结果，保留其原有证据身份；后来获准 A-V 只更新本报告和 `conclusion.md`，新增来源为 `../A_V/receipt.json`。A1 的实际搜索证据仍到 E0；没有 E1/E2，不能宣称局部信息差异造成搜索损失，也不能宣称没有损失。

未获取/未完成：GOOSE 最初 Zenodo 原包；原生逐运行权重及日志身份；服务器 wheel 与审计源码构建一致性；旧当时特征矩阵及新旧运行的颜色映射/MaxSAT证书；D0/T1/这些候选的完整真实 OPEN 轨迹；八项端到端作者复现。本次 A-V 新矩阵的相等性、划分及哈希已由执行协调者核查并记入回执，不再将“新实验未执行”列为未知。AAAI2025 State Encodings 原文不在本次已读四篇范围，不以它承担结论。

A-V 已获准完成，当前固定构建的差异可复现；建议仍为 `USE_EXISTING_METHOD`，优先审视作者已有配置。固定构建重跑不是2024 SVR/多重集/L4/原版状态协议的复现，AB2仍未获得资格证据。源码原因和搜索价值尚待科研主线裁决；本卡完成后停止，不自动扩大验证、拟合、搜索或开发新表示。
