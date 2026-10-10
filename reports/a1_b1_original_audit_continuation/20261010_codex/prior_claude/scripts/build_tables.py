#!/usr/bin/env python3
"""Builds 03_original_vs_cp_diff.csv and 04_limitations_and_attribution.csv from the audit notes (hand-entered rows; evidence locations given per row). No data access."""
import csv
import sys

AUD = sys.argv[1]
G = "goose main 04dbe8f"
W = "wlplan 9d9fc99"
S = "scorpion 4c16ce05"
H03 = ["method_version", "stage", "paper_definition_and_page", "official_code_file_line_commit", "author_default_config", "cp_disr_location_commit", "actual_difference", "evidence_status", "potential_impact", "existing_solution", "next_check"]
R03 = [
    ["WL-GOOSE ICAPS24 / %s" % G, "特征", "WL 颜色在 0..L 轮的计数向量（P1 §2–3；实验 L=4）", "%s src/feature_generator/feature_generators/wl.cpp:embed_impl（约197–215）" % W, "classic.toml iterations=2", "W1/旧WL: wlplan 同库，opts iterations=2 @cc7d79d6f training/W1/*.opts", "论文实验 L=4，现行默认与项目均为 2", "READ_PRIMARY(论文,代码)", "更少轮数可能少分开状态；上一卡 23 例在 0 轮相同、1 轮即可分，深度非主因", "iterations 是公开选项", "无需新查"],
    ["WL-GOOSE", "图编码(ILG)", "P1 §3 Def 3.1：节点=对象+初始/目标命题；对象色 ob；命题色=谓词×{已达目标,已达非目标,未达目标}；边标位置；无动作节点", "%s src/graph_generator/graph_generators/ilg.cpp:26–139" % W, "graph_representation=ilg", "同库", "无差异", "READ_PRIMARY", "对象无类型色；动作关系不入图", "nilg/iilg/ploig/aoag 等其他生成器存在（未评测）", "无"],
    ["WL-GOOSE", "静态事实", "ILG 的命题=状态中为真的命题；静态事实只有当它们在状态里才出现（P2 Def 4.1 推断，P1 未单列）", "ilg.cpp 全文无 get_statics 使用；goose/learning/dataset/heuristic/creator/classic_dataset_creator.py:34–47；goose/planning/util.py:17–42；scorpion features/wlf_generator.cc:14–75（仅 FD 变量事实）", "state_representation=downward（只保留翻译原子，始终为真的原子被翻译删除）", "compact/w1.py:26–59 StateExporter（同样用翻译原子）@cc7d79d6f", "无差异（项目=官方默认）。官方另有 state_representation=all/no-statics（数据集创建侧）；C++ 规划器只见 FD 事实", "READ_PRIMARY(代码)", "位置固定的对象（hoist/pallet）位置不可见 → 上一卡候选16", "state_representation=all 可保留静态原子（数据侧）；规划侧是否可用未核", "A-V 之外无需新查"],
    ["WL-GOOSE", "对象类型", "ILG 对象统一色 ob（P1 §3；P2 同）", "ilg.cpp:48–52 对象色=0", "无类型", "同库；项目 typed 领域的类型不入图", "无差异", "READ_PRIMARY", "不同类型对象在图上可互换（候选16同构映射的原因之一）", "类型可作为一元静态原子进入状态（all）；未评测", "A-V 可选"],
    ["WL-GOOSE", "目标", "目标命题带 ag/ap/ug 颜色（P1 §3）", "ilg.cpp:55–67,102–118", "—", "w1.py:50–55 目标来自翻译", "无差异", "READ_PRIMARY", "—", "—", "无"],
    ["WL-GOOSE", "邻居聚合", "论文定义为邻居(颜色,边标)的多重集（P1 §2–3，抽取结果）", "wl.cpp:refine；neighbour_containers/wl_neighbour_container*.cpp（multiset_hash 开关）", "classic.toml hash=\"set\"", "opts hash 为 set", "论文定义多重集 vs 现行默认 set —— 页码待核；项目=官方默认", "READ_PRIMARY(代码)/抽取(论文)", "set 丢计数；上一卡候选16 在静态原子+多重集下可分、set 下不可分", "multiset_hash=True 是公开选项", "A-V（单因素）"],
    ["WL-GOOSE", "词表与未见颜色", "训练图上颜色集合；之后未见颜色被忽略（P1 §3）", "features.cpp:194–209；wl.cpp:48–66（遇到未见颜色，该节点及其邻居在之后各轮被丢弃）", "—", "W1 词表来自 6399 个 Train96 状态（data.json）", "训练数据分布不同", "READ_PRIMARY", "OOD 状态（IPC）可能丢失更多颜色", "—", "无"],
    ["WL-GOOSE", "剪枝", "P1 未描述剪枝；P3 只定义 none 与 i-mf，a-m 未定义", "pruning_options.cpp:6（a-m=ALL_MAXSAT）；pruning/bulk_pruners.cpp:11–113；features.cpp:219–300(remap)", "feature_pruning=\"a-m\"", "opts a-m；上一卡重建 202→147 特征", "代码设计：把训练集上列完全相同的特征并成一组、MaxSAT 保留最少特征并保证祖先被保留 → 按设计训练集内划分应不变；上一卡观察：23 个训练内候选在 a-m 下特征相同、未剪词表下可分（且 202 列里仅 22 列与保留列相等）→ 与设计矛盾", "READ_PRIMARY(代码)；矛盾=UNRESOLVED_ORIGIN", "若观察属实：官方默认剪枝会让训练内严格序对不可分；若不实：上一卡结论需撤回", "pruning=none 是公开选项", "**A-V 首要**：先判定观察是否为分析假象"],
    ["WL-GOOSE", "损失/估计器", "P1：GPR/SVR/rank-SVM 等（rank-SVM 细节 P1 页未取得）", "goose/learning/predictor/linear_model/rank_classifier.py:19–62", "optimisation=rank-svm：LinearSVC(hinge,C=1,无截距,max_iter=1e6)；对重复对计数但 fit 不传权重", "scripts/c1_compact_w1_fit.py:81–83：同估计器但传入样本权重；合并相同差向量", "有意改动（D0 对齐）", "READ_PRIMARY", "权重不同于官方；9875 个对只剩 22 个差向量类", "—", "无"],
    ["WL-GOOSE", "标签/比较对", "计划后继优于其他后继；计划状态优于父（P1 §5 原文细节待核）", "classic_ranking_dataset_creator.py:29–76：good=计划后继，maybe=其余后继，bad=父状态", "data_generation=plan", "compact/w1.py:71–119：精确 d* 严格序对，同父，无父子对", "官方把兄弟当作“不确定且视为更差”；W1 只用严格更差", "READ_PRIMARY", "官方监督含无标签的“假定更差”对", "—", "无"],
    ["WL-GOOSE", "采样/数据", "IPC23 学习赛易例，≤99/域，Scorpion 最优计划，计划状态（P1 §5）", "classic_ranking_dataset_creator.py（每题一条计划）", "plan, data_pruning=equivalent-weighted", "Train96，457 轨迹含 361 条偏离", "监督信息量不同", "READ_PRIMARY", "不可作公平监督比较", "—", "唯一状态/对计数已给"],
    ["WL-GOOSE", "评分精度/取整", "P1 未描述取整", "scorpion goose/wlgoose_heuristic.cc:23–26 `static_cast<int>(std::round(h))`", "整数启发式（FD）", "search_match/evaluators.py WlEval `cpp_round`", "无差异（取整位于官方规划器启发式内，不是项目适配）", "READ_PRIMARY", "取整产生并列：旧 WL 库 B 589 对原始读出不同但取整后相同", "缩放后取整等（无公开选项，未核）", "无"],
    ["WL-GOOSE", "读出稀疏性", "—", "LinearSVC 解", "—", "W1: 147 特征中 8 个非零权重（7 个第0层、1 个第1层）", "由合并后仅 22 个差向量类的小训练集所致（推测）", "本项目观察", "第2层 112 个特征从未被读出使用", "—", "需拟合才可评价（不在零拟合范围）"],
    ["WL-GOOSE", "缓存/并列/重复检测/目标检查", "—", "wlgoose_heuristic.cc:14 cache_estimates 参数；FD 搜索细节未核", "—", "engine.py：键 (h,序号)、生成时检查目标、CLOSED 不重开", "官方搜索侧未核 → UNKNOWN", "UNKNOWN", "并列规则影响 WL 的大量并列", "—", "若要作者原生比较需另批"],
    ["WL-GOOSE", "搜索预算与计时", "30 分钟计划求解用于数据（P1 §5）", "—", "—", "300 s/10万节点/共享 GPU", "不可比", "—", "—", "—", "—"],
    ["Opt Rank (Hao 2024)", "约束", "§3：父子约束 s_i≻s_{i-1}；兄弟约束 s_i≻s_{i-1} 的兄弟；跨路径由传递性得到；兄弟之间不排序", "未找到官方代码", "—", "D0: compact/train.py ACTION 损失（集合NLL+严格兄弟对softplus）；V(s) 在 softmax 中抵消，无显式父子项", "D0 无显式父子项；兄弟对用精确 d* 严格序（可含非计划兄弟之间）", "READ_PRIMARY(论文镜像)", "—", "—", "无"],
    ["Opt Rank", "理论", "Thm 1：单位代价、全序、固定并列规则下，最优排序的 GBFS 返回最优计划；训练模型无保证（§6.3）", "—", "—", "—", "—", "READ_PRIMARY(论文镜像)", "平均排序误差不对应搜索保证", "—", "—"],
    ["Opt Rank", "数据/损失", "最优计划的状态及其兄弟；偏移 sigmoid + MSE 的成对 Direct Ranker；批量共享嵌入（§4.1,§4.3）", "—", "—", "D0 每决策权重 1/(轨迹长×批)", "损失与模型不同", "READ_PRIMARY(论文镜像)", "—", "—", "—"],
    ["Chrestien 2023", "约束", "只要求每步 OPEN 中计划上的状态严格优于计划外状态（Def 1/Thm 1）；损失 L_gbfs 用 OPEN 中的对 + 距轨迹 1 的状态", "未找到官方代码", "—", "T1-S：25% 的对换成同题不同父的跨父严格序对（worse=d* 更大的其他决策状态）", "T1-S 的跨父对来自精确标签，且两端可均为计划外状态；Chrestien 只比较“计划上 vs 计划外”", "READ_PRIMARY(论文,抽取)", "计划外 vs 计划外的次序在其最优效率定理中不需要", "—", "—"],
    ["Chrestien 2023", "局限", "§8.3–8.5：多条最优计划时零损失不可达；GBFS 完美排序可能不存在；不同搜索不可互换", "—", "—", "—", "—", "READ_PRIMARY(抽取)", "—", "—", "—"],
    ["D0 / T1-S", "数据角色", "—", "—", "—", "库 B: Train32 为 Train96 题（样本内），Struct16/Joint16 为诊断", "T1 的 148 个跨父对：57 个修复 + 15 个损伤的两端都是 Train96 训练状态", "本项目观察", "离线改善部分是样本内拟合，不能当泛化证据", "—", "已在 B1 表分列"],
]
H04 = ["id", "exact_definition", "applicable_tasks_configs", "author_acknowledged", "paper_location", "official_code_evidence", "project_code_evidence", "existing_data_evidence", "source_class", "evidence_identity", "search_evidence_level", "alternative_explanations", "existing_solutions", "remaining_unknown"]
R04 = [
    ["L-A1a", "a-m 剪枝后训练内严格序对特征相同，未剪（保留训练词表）可分", "typed Depots Train96/库B；wlplan 2.1.0-12 副本，classic.toml 默认", "否（P1 未描述剪枝；P3 只讨论 i-mf）", "P1 §5 “有限训练集会使语义不同的特征看起来相同”只是一般性提示", "bulk_pruners.cpp:11–113：设计上保持训练集列等价；矛盾未解", "w1/重建 147 vs 202 特征", "上一卡 pruning_probe.json（16 个训练内候选：未剪 8–36 个特征可分、剪后 0）", "UNRESOLVED_ORIGIN", "本项目观察", "E0", "分析假象（两次独立 collect、列比较）；wheel 与源码版本差；真实的 wlplan 行为", "pruning=none", "是否可复现；机制"],
    ["L-A1b", "ILG 无对象类型且看不到被翻译删除的静态位置，导致候选16（hoist0↔hoist1 交换）同构", "Depots typed、DOWNWARD 表示", "ILG 对象无类型是定义（P1 Def 3.1）；静态事实未单列", "P1 §3 Def 3.1；P2 Def 4.1", "ilg.cpp:48–52；goose/…/classic_dataset_creator.py:34–47；planning/util.py", "w1.py StateExporter 同官方默认", "上一卡候选16：同构；静态+multiset 可分；静态+类型 set/2 可分", "METHOD_LIMIT（类型）+ AUTHOR_CONFIG_TRADEOFF（静态）", "理论(定义)+本项目观察", "E0", "合法重命名要求保持类型与静态事实，该映射不合法", "state_representation=all（数据侧）、multiset_hash", "覆盖率：只有 1 个候选；联合变化未分离"],
    ["L-A1c", "读出把不同特征折叠成相同分数（W1 1300 个训练对抵消）", "W1", "否", "—", "LinearSVC 解", "c1_compact_w1_fit.py", "readout_categories.json：训练 1300、库B 585", "CP_ADAPTATION（监督/权重）/UNRESOLVED_ORIGIN", "本项目观察", "E0", "小训练集；hinge C=1；合并权重", "—", "需拟合才能评价"],
    ["L-A1d", "部署取整制造并列", "旧 WL（和 W1）", "代码层事实，论文未描述", "—", "wlgoose_heuristic.cc:23–26", "evaluators.py WlEval", "库B：589 对（WL）", "AUTHOR_CONFIG_TRADEOFF", "官方代码事实", "E0", "—", "—", "不同取整策略的搜索效果"],
    ["L-A1e", "WL 近似零差训练对占 8.47%（836/9875）", "Train96", "否", "—", "—", "—", "zero_diff_836_profile.json", "UNRESOLVED_ORIGIN（含 L-A1a/b)", "本项目观察", "E0", "同上", "—", "原因比例（24 个探索样本，不外推）"],
    ["L-B1a", "离线跨父平均改善（112/36）未必改变同父候选首选", "库B", "否", "Hao §3：理论只要求排序；§6.3 训练模型无保证", "—", "diagb.py/crossparent.py", "上一卡 b1：384 个全精确 package：4 改善/2 损伤/353 不变且最优；跨父对不属于 package", "UNRESOLVED_ORIGIN", "本项目观察", "E0", "package 首选≠OPEN 出队", "—", "搜索后果"],
    ["L-B1b", "T1-S 的修复对在 Opt Rank 计划规则与传递性下 0/112 可构造（粗分类）", "库B 的 reference 计划", "否", "Hao §3；Chrestien Def 1", "—", "pair_to_search_evidence.csv", "optrank_coverage_summary.json", "UNRESOLVED_ORIGIN", "本项目观察", "E0", "粗规则：只用库内参考计划父状态", "—", "不能把“不可构造”算成作者方法失败"],
    ["L-B1c", "离线改善与搜索代价脱节", "Joint16∩面板、IPC22", "Hao §6.3 / Chrestien 一般性", "—", "—", "—", "Joint16 上 D0→T1 零变化；IPC T1/D0R 展开比 1.84（12 题）", "UNRESOLVED_ORIGIN", "本项目观察", "E0（无 E1–E4）", "单训练运行、检查点、GPU 噪声", "—", "全部"],
    ["L-C1", "神经评分占 IPC 搜索墙钟约 99%", "IPC22", "Hao §6.3 提到推理时间是障碍", "—", "—", "cost_breakdown.csv（历史卡）", "旧卡计时", "CP_ADAPTATION/硬件", "历史数据（共享GPU，时间身份另报）", "—", "共享 GPU 噪声", "—", "不可作精确速度结论"],
]
for name, H, R in (("03_original_vs_cp_diff.csv", H03, R03), ("04_limitations_and_attribution.csv", H04, R04)):
    with open(AUD + "/" + name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(H)
        for r in R:
            assert len(r) == len(H), (name, r[0], len(r), len(H))
            w.writerow(r)
print("ok")
