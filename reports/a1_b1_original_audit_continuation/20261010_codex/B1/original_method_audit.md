# B1 补充原作者方法审计（2026-10-10）

状态：COMPLETE_WITH_BOUNDED_UNKNOWNS。裁决维持 DEPRIORITIZE。只做论文、源码静态阅读与既有表统计；本分支训练、拟合、新特征、模型评分、搜索、精确查询、SSH、提交和推送计数均为 0。

## 1. 身份、版本与阅读范围

OptRank：已读 IJCAI 2024 正式论文 §3–4、§6–7，PDF pp.3–7（出版页 6726–6730），以及官方来源记录、README 和下列指定源码。论文指向 Zenodo 11107790；本地取得的是同一 concept 后续公开记录 12575733，**不是同一原包**。归档发布 MD5 为 ae64b6490faff699ce11318082c4020c。本轮以此归档身份固定源码，不能把元数据中的 GitHub 当前 HEAD 6b3d747ebad3d41d79d55fc1000232a568caea26 当作 ZIP commit。身份升级为 OFFICIAL_CODE_AUDITED（指定文件静态范围）；未运行。

Chrestien 等：本地论文是 arXiv v1，2023-10-30，21 个 PDF 文件页，实际读了 §§2–4、6 和附录 8.3–8.5。另从 NeurIPS 官方页面读取正式会议 PDF（20 页），核对定义、代码入口与目标表值。两版本分开登记。作者仓库固定 e7cd65baa820dc8b9267749ff57e384d41bea02a；README 明确对应论文并内嵌 NeuroPlanner.jl.zip。本轮审计的是**内嵌包**，未换用后来 NeuroPlanner main。身份升级为 OFFICIAL_CODE_AUDITED（指定代码静态范围）。

D0/T1-S：本轮读取协调者取得的 Claude 既有报告、汇总 JSON、完整 173 行 pair 表和 b_coverage.py 原字节；没有运行或重新审计 CP 训练源码。CP 对照仍注明“既有审计记录”；旧基准为 cc7d79d6fcb053bf0c16033e765a2e808de2aaac，冻结身份由协调者登记。

正式论文入口：[IJCAI 2024](https://www.ijcai.org/proceedings/2024/0743.pdf)；[NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/50ea4dbd1cff6bd3daef939eff10c092-Paper-Conference.pdf)。官方身份来自论文、作者仓库 README 和 Zenodo 元数据，不来自名称相似推断。

## 2. 原作者监督与理论

### OptRank

- IJCAI PDF p.3 Definition 3 / Theorem 1 是**单位代价、选定最优计划、准确 optimal ranking**条件下的结论：下一路径状态严格优于父状态和兄弟，传递性提供相对更早祖先与祖先兄弟的次序。不是有限训练保证。p.7 §6.3 明确无保证学到 optimal ranking。
- p.4 §4.3 的离轨训练状态是计划父节点的其他一步后继，无需查询它们的 d*。learner/dataset/rank_dataset_gnn.py:79–129 调用 perfect_with_siblings；planners/downward/src/search/search_algorithms/perfect_with_siblings.cc:126–139 先写选定计划后继，再写其他动作后继；learner/models/rank_model.py:31–39 比较选定后继与兄弟、上一个路径状态。
- **兄弟之间、任意两离轨状态之间的真实 d* 排序不是作者必须约束的对象。** “跨父”标签不自动等于由路径递减导出的传递性关系；缺少输入不等于作者失败。
- p.4 §4.1–4.2 用共享嵌入与无偏置读出，经成对学习导出标量。rank_model.py:18–19,37–45 用 sigmoid、目标 1；train_rank.py:231–245 用 MSE、Adam、ReduceLROnPlateau。论文写 sigmoid−0.5/中心化目标；目标同时偏移时不改变成对 MSE，不能把偏移单独当损伤。
- **旧共享身份表把 OptRank(GOOSE) 称作 WL 模型，需更正**：论文 p.5–6 是 GOOSE-LLG 神经模型，4 层、64 维；rank_model.py:9,12–19 继承 models.gnn.Model。不是 A1 的 WL/ILG/rank-SVM。GOOSE 是框架名。

### Chrestien 等

- 论文 pp.3–5 Definition 1 / Theorem 1 / Eq.(4) 针对沿某条最优计划构造的理想 OPEN：下一路径状态严格优于当前离轨竞争者。不是任意失败搜索树上的所有两两 d* 排序。
- 内嵌 src/losses.jl:114–160 逐步生成路径状态的一步后继，累计 states，并用 setdiff(keys(stateids), htrajectory) 排除整条路径；:146–148 构造下一路径状态对累计离轨状态的约束，覆盖此前祖先层候选；:191–193,216–220 使用 mean softplus(h_path−h_off)。默认 max_branch 为整数上限；有限值时 :130–133 抽样，需要单独登记。没有任意 off-off 排序项。
- Lrt 是另一损失，:238–275 仅要求路径相邻状态递减；Lstar 加 g（:163–168），Lgbfs 不加 g（:216–220）。三者不可混称。
- scripts/supervised.jl:46–59,103–112 从已有计划构造每题 minibatch，随机选题更新，AdaBelief，默认 10,000 更新，随机选约一半题训练。代码中其他 bootstrap 分支不表示本论文或本轮使用。
- 论文 pp.5–6 已说明目标感知、多个解路径冲突、代理学习与样本非独立边界。附录 8.4 pp.15–16 的非单位成本例子（含代价 1 和 9）不能直接否定 Depots 单位代价上的准确排序存在性。

## 3. 新的论文—代码冲突与部署边界

**实数排序与整数部署分开。** IJCAI p.4 Lemma 2 用 r(s)=w·nn(s) 的实数次序；官方后续包 rank_model.py:53–62,64–88 实际 h/h_batch 调用 shift_heu，默认加 1 后 np.rint 转整数（:81–87）。量化可能合并实数排名，不能直接承接严格次序保证，也不能说是 CP 适配独有。静态示意：0.10、0.20 均变成整数 1；这不是模型前向或新实验。没有指定真实竞争事件，搜索后果未知。

| 环节 | IJCAI 正式论文 p.6 | 后续公开包默认 |
|---|---|---|
| repeats | 3 | rank_train_test_ipc2023.py:47 为 5 |
| 训练上限 | 500 epochs | train_rank.py:74 为 50；脚本 :120 未覆盖 |
| 调度/选模 | validation accuracy 控制学习率 | train_rank.py:244–245,311–315 val_loss 调度、(train+2val)/3 选模 |
| 划分 | 随机 90/10 | rank_dataset_gnn.py:200–215 每第 val_interval 题验证 |
| 搜索 | eager GBFS、1800 秒 | rank_train_test_ipc2023.py:18,49–72 默认 batch_eager_greedy、downward_gpu、timeout=2000 |

实际命令或已有模型可覆盖默认；这是**版本/配置冲突**，不是证明论文运行不实。需原运行命令、冻结权重、对应日志才能确认历史协议。

本轮读取脚本实际调用的 GPU batch 链：planners/downward_gpu/src/search/search_engines/batch_eager_search.cc:76–97 出队后判目标，:119–143 对新节点批量评分、插入整数优先级；plugin_batch_eager_greedy.cc 设置 reopen_closed=false。与旧 CP 审计记录的生成时判目标不同。CPU downward 的 eager_search.cc:283–286 生成时判目标，不能借其替换 GPU 调用链。

**NeurIPS 网格/PDDL 约束有差异**：作者仓库 sokoban/train/gbfslgbfs.py:188–219 仅 on-path 与 off-path 且 g 相等时建立约束；train.py:114–116 调用该入口。它不同于内嵌 PDDL 累计离轨约束，未追认该版本为论文每个网格结果的确切运行代码。

内嵌 src/heuristic.jl:12–15 返回原标量，搜索委托 SymbolicPlanners。Manifest.toml:804–808 为 0.1.11 / tree b25d1f41e9d145941aa53a6296034046c4b2235c；依赖引擎源码未在本分支审计，tie、reopen、目标时机仍未知。

## 4. 既有 D0/T1 与粗覆盖结论的修正

已读 prior_claude/B1/pair_audit_summary.json、optrank_coverage_summary.json、原报告、b_coverage.py 及完整 pair_to_search_evidence.csv。本轮重新统计该 173 行派生表：173 唯一变化对，跨父 112 修复/36 损伤，同父 17/8，全部 EXACT/EXACT，无并列，均为 E0；保存 margin 的符号与修复/损伤标签 0 个不一致。112 修复中 57 个两端训练状态，85 train / 27 struct。未取得全部原评分/包索引，不能宣称再次独立核验原始模型评分。库 B 与 IPC22 不交仍引旧审计。详见 saved_pair_recount.json。

**旧 0/112 不能当精确作者规则证明**：

- b_coverage.py 的 is_opt_succ 是 package 后继局部最小 d*，不是 selected plan successor 身份，原 caveat 已承认。
- reference_kind 被读取但未用于筛掉 BEST_SAVED；总体汇总含 56 OPTIMAL 与 8 BEST_SAVED 参考题，但本轮已核实重点 112 个跨父修复对全部为 OPTIMAL，此项不影响该子集，不能作为其 0/112 结论的限制原因。
- 不在参考父后继/不是局部最优后继等启发式否定，不等于完整约束或传递闭包中的不可构造证书。
- 脚本未实际构造作者约束边和闭包。因此 0/112 的准确含义是**旧粗分类返回 0 个 covered**，不能改写为全部是两端离轨、或作者不可能覆盖。

本轮对 112 行重计粗否定原因：19 行 better 不在参考父后继、86 行 better 不是局部最优后继、7 行 worse 不在更早参考父后继。**这 112 行全部 reference_kind=OPTIMAL，所以 BEST_SAVED 未筛除没有影响这个重点子集**；它仍是其他统计的潜在边界。不能仅由这三种启发式原因恢复真正路径成员身份。新 pair_to_search_evidence.csv 保留旧字段并新增 exact_author_constructibility=UNKNOWN_NOT_RECONSTRUCTED 与 LEGACY_COARSE_HEURISTIC_ONLY 标签。

新源码确认任意 off-off 严格 d* 不是必需约束，但不能从这个事实反推 112 对的真实轨迹成员身份。缺 selected successor 与完整已保存包时，精确可构造性标为 UNKNOWN。进一步已保存表连接属于本卡只读范围，无需运行作者监督生成逻辑。

## 5. 作者既有结果的最小核验

内嵌 README 指定 scripts/super/results.csv 为论文原始结果；scripts/show.jl:38–88,225–230 按组、问题路径排序交替折、两折互换选模、三种子平均和 5 秒阈值制表。本轮没有执行作者 Julia 脚本。

独立标准库解析器 inspect_saved_elevators_cell.py **只核查 Table 1 Elevators × GBFS × Lgbfs 一个单元格**。CSV SHA256 为 8106b257f46baf8a36388cfd45408f5b569b4d6ee621b7f63890d1e2812ffad7。扫描已有 1,613,790 行，仅保留 4,860 行/54 配置；组内问题重复 0，≤5 秒却未解为 0。三种子交叉折覆盖 0.8666007905、0.7351190476、0.9565217391，均值 85.27471924%，取整 85，匹配正式 Table 1（PDF p.9）。壁时 3.994 秒，单进程，GPU 0。

身份仅为**原始结果已取得＋一个表单元格统计一致性核实**；不授全部 OFFICIAL_EXISTING_RUN_VERIFIED：权重哈希、计划正确性、引擎二进制、逐运行日志未核。未复算其他表/方法。详见 official_saved_cell_check.json 和 official_saved_cell_configs.csv。

## 6. 科研去留

已有方法提供路径递减、下一路径状态对兄弟、及对跨祖先层离轨候选的机制；普通增加跨父项或困难样本不能凭 T1-S 离线净修复包装成新机制。源码补齐新增量化和版本边界，但没有 D0/T1 的 E1/E2，更没有 E3/E4。85% 作者表值匹配、112>36 或粗 0/112 不能提升为搜索因果或作者失败。

B-V 若为已保存计划/包的准确连接，可继续作为只读统计；本轮没有新的 B-V 实验申请。AB2 为 NOT_TRIGGERED。B1 保持 DEPRIORITIZE，到此停止等待研究负责人。
