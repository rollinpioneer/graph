---
title: "CP-DISR C1 Blocksworld Amendment A03：保留 ID 失败，释放一次冻结模型评估"
date: "2026-10-06"
lang: "zh-CN"
amendment_id: "CP-DISR-C1-BW-FROZEN-EVAL-A03"
parent_card: "CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1"
status: "RESEARCH_DECISION_ISSUED_EVALUATION_ONLY"
repository: "rollinpioneer/graph"
branch: "codex/cp-disr-c1-blocksworld-main-v1"
base_commit: "b6804664e0f0ddb2f9b7068d9c49fdf5e6254a6a"
new_training_run_cap: 0
optimizer_steps_authorized: 0
formal_evaluation_authorized: true
original_a02_gate_status: "IMITATION_ID_GATE_FAIL"
---

# Amendment A03：不再为过门重训，评价三个已经冻结的模型

> **决定：保留 A02 的原始 FAIL，不把 33 改成 32，也不单独照顾 ASNET。对三种最终模仿学习模型统一开放一次 A0/A1/A2/B 及规划器评估。新训练为零。**
>
> 本文件是一项在已知 ID 结果后作出的明确协议修订，不是假装 A02 从未改变。A02 的原结果、gate、配置和回执均保持原样；A03 另建登记与结果目录。实施授权只限本文件列出的评估，不包含训练、调参或新算法开发。

## 1. 为什么改为评估，而不是再次训练

A02 仓库记录如下：[R1–R3]

| 方法 | ID dev36 成功 | ID dev36 决策完美 | A02 gate |
|---|---:|---:|---|
| B2-CACHED | 36/36 | 35/36 | PASS |
| QMARK | 34/36 | 33/36 | PASS |
| ASNET-READOUT | 36/36 | 32/36 | FAIL |

三条运行均完成固定 100+50+50 epochs；D0/D1/D2 相同，无 NaN。最终训练集的最优动作 argmax 命中率记录为 B2=1.000、QMARK=1.000、ASNET=0.997。ASNET 的四个非决策完美 dev 案例最终全部成功，其中三例含重复状态—动作。[R1]

这些记录足以支持“模型已形成能够完成多数 ID 任务的策略”，但不支持“它们的 ID 决策质量已经严格匹配”。ASNET 不是没有学会任务，而是仍存在绕路；在当前模型上，这是要报告的现象，不能靠选 checkpoint 或继续训练至过门使其消失。

33/36 是此前选择的操作门槛，不是 32/36 与 33/36 之间存在科学性质突变的证明。此处**不降低原门槛**，而是改变接下来的研究问题：

> 固定训练已经结束、ID 差距已知时，这三个最终策略在预先生成的目标反转、非同构目标和规模测试上分别表现怎样？

这可以推进主研究，但不再声称完成了“先把三种方法训练到相同 ID 水平，再孤立比较表示”的原理想设计。A02 的 gate failure 仍然成立。

## 2. 与 A02、v1.1 的优先级和覆盖关系

本修订只覆盖本节明确列出的执行与解释条款，其余合同、输入、模型、任务及指标沿用冻结实现。

| 原条款 | A03 的处理 |
|---|---|
| A02 §§8–9：三种模型必须全过 ID gate 才能评估 | A02 状态不变；A03 另行授权全部三个指定 final checkpoint，ID 分数不再充当 A03 的放行条件 |
| A02：失败后不得自动继续 | 已正确执行；本文件是停止后的显式研究决策，不是自动续跑 |
| v1.1 / 原脚本：A0 decision-perfect <0.90 就阻止 A1/A2/B | A0 改为诊断切片，低分如实保留，不触发第二道成绩门 |
| 原 REP_* / LIMIT_COORDINATION 分类 | 不作为 A03 的确认性分类输出；采用连续成绩、ID 差距及逐条件行为解释 |
| A01：hop>4 无法构造 | 保留 `NOT_TESTABLE_STRUCTURALLY_EMPTY`，不改变图或 hop 定义 |
| 只使用最终 checkpoint，不改测试题 | 全部保留 |

**A03 不设任何替代成绩门槛。**不是把 ASNET 的最低分改为 32，也不是只允许 ASNET 例外。三种已完成的 final 模型统一进入评估，即使某一模型在某个切片成绩为零，也要完成并公开预定的其他切片。

评分差是结果；源文件不符、权重不符、非法标签、损坏的输入、非有限概率等才是技术完整性问题。

## 3. Git 与冻结资产

### 3.1 仓库状态

```text
repository: rollinpioneer/graph
branch: codex/cp-disr-c1-blocksworld-main-v1
base_commit: b6804664e0f0ddb2f9b7068d9c49fdf5e6254a6a
A02_prep_commit: 8ea6a71a54a1f25fd402b1e14e835728f93b5667
original_split_freeze_commit: 490a3b11d49f6155baf2a6b6c430edac6609d2af
```

本文件起草时已通过 GitHub 读取远端分支和 A02 结果。没有登录服务器、重验 checkpoint 二进制、修改仓库或运行实验。执行 Agent 只需在本机核验下述固定资产一次，不重新做全项目审计。[R1–R4]

### 3.2 旧目录只读

```text
PPO:
runs/final_master/c1_route_b/blocksworld_main_v1/20261006T015313Z_490a3b11/

A02:
runs/final_master/c1_route_b/blocksworld_main_v1/imitation_a02/20261006T053652Z_8ea6a71a/
```

不在旧目录新增 eval 文件、decision_log 或改写 STATE、final_receipt、classification。旧 PPO 和 A02 的原始 FAIL 永久保留。

新输出根：

```text
runs/final_master/c1_route_b/blocksworld_main_v1/evaluation_a03/<UTC>_<specsha8>/
```

### 3.3 唯一允许使用的三个模型

以下相对路径均以仓库根为起点：[R3]

```text
A02_ROOT = runs/final_master/c1_route_b/blocksworld_main_v1/imitation_a02/20261006T053652Z_8ea6a71a
```

| 方法 | final checkpoint 路径后缀 | SHA-256 | 字节数 |
|---|---|---|---:|
| B2-CACHED | `runs/R-C1-BW-IL-B2-0/checkpoints/final.pt` | `29c7eab495b4dea7a3f30f40fe7321619bdbe941153b3a69c2a937a528d61e11` | 19860355 |
| QMARK | `runs/R-C1-BW-IL-QMARK-0/checkpoints/final.pt` | `50fa46d9c1ea39d91839f3980ae0c985549e339cf91317adb1a3505008f3ed87` | 19862899 |
| ASNET-READOUT | `runs/R-C1-BW-IL-ASNET-0/checkpoints/final.pt` | `9b02eb9c4f80204802273fa92df02ea21093fd54e2fbcbf672846fa31df847da` | 19860355 |

实际路径为 `A02_ROOT/后缀`。不使用 stage0、round1、PPO、早期冒烟或其他 seed 的模型，不创建集成，不换掉 final.pt。

### 3.4 Split 文件及哈希

按 A02 的 `prep/c1_bw_imitation_a02.json` 固定：[R4]

| 切片 | 路径 | SHA-256 |
|---|---|---|
| train/dev（仅复用已保存 dev 成绩） | `configs/splits/c1_bw_train_dev_v1.json` | `64225ed50d69aa8c42d9ce36470bdb4705b792f464e97a88b23b2ba8cdc0b913` |
| A0 | `configs/splits/c1_bw_a0_iso_v1.json` | `c15346690583b700766e1adb73c46580d83439506972ddd09b7773637dd168f6` |
| A1 | `configs/splits/c1_bw_a1_color_reverse_v1.json` | `4e31998725be0c81d6e7f9306beaf4c2b7a7629e7400482f4f18541b1a9c1d96` |
| A2 | `configs/splits/c1_bw_a2_noniso_v1.json` | `1207d455afaa7b8c5b8e4d2f62d5b49dbf2c29e07a944ef85a543396b4a7a349` |
| B | `configs/splits/c1_bw_b_scale_v1.json` | `9dcf596cc72d9102182f71963edb90d4fd576bd3a9063be40309be0a0ea89220` |

不重新生成题目或改变类别、颜色、起始状态、步数上限；不因 ID 错误案例改变测试分布。

## 4. 最小实现：新增评估入口，不修改旧 gate

新增一个小型 A03 专用入口，例如：

```text
scripts/c1_bw_eval_a03.py
```

它复用现有 `cp_disr.blocksworld.train.evaluate`、状态引擎、规划器、模型工厂及评估逻辑。只处理四件事：读取 A03 的明确授权；按上表装载三个 final checkpoint；将结果写入新根目录；统一完成所有切片。

原 `scripts/c1_bw_eval.py` 同时存在全员 ID gate、A0 gate，且会把评估结果写在旧 checkpoint 附近。不能直接以 `--only`、`--planner-only` 或伪造旧 `launch_state` 绕过它，也不能删除原 gate 后假称仍是 A02。[R5]

冻结模型 forward、合同/环境、数据、planner、metrics 文件不改。源检查对这些核心文件逐个比较基线哈希；新的入口、A03 spec 和结果脚本单独登记。不将新增评估脚本的 commit 冒充原模型训练 commit。

进入评估前只做必要检查：模型与 split 哈希匹配；三个模型可加载且参数为只读；旧结果不变；输出路径正确；不会构造 optimizer 或调用 backward。测试入口使用人工小 fixture，不能先用正式测试题“干跑”。不重复合同穷举、同构类枚举或训练等价审计。

## 5. 评估范围、资源与一次性规则

| 切片 | 每个学习模型的案例数 | 作用 |
|---|---:|---|
| A0 | 24 | 对象改名诊断，不作为性能放行门 |
| A1 | 32 | 颜色模式反转 |
| A2 | 32 | 非同构目标和已冻结的干扰分组 |
| B | 48 | n=6/7/8 的规模变化 |

每个模型共 136 局，三个模型共 **408 局**。规划器在同样的 136 个初始问题上各求解并执行一次参考计划，共 **136 条参考记录**。总预定记录 **544**；每一步的最优动作标注是同一评估的分析计算，不算新训练，也不算额外模型 episode。

使用原 deterministic argmax、合法动作 mask、step cap、目标判定、hidden 初始化及更新方式；不截断循环来“帮助”模型，不提前终止本来允许恢复的任务。

V/Q heads 未被模仿训练，保持冻结，不从它们的输出解释价值估计或置信度。规划器只标注、只作为参照，不替换策略动作，不为策略填入 planner state、距离或最优动作集合。

每个 `(checkpoint_sha, case_id)` 只运行一次。评估前登记 case 清单，逐例持久化；已完成结果不可覆盖。若进程中断，仅可续跑尚未开始的 case。对开始但未完整记录的 case 标 `TECHNICAL_INCOMPLETE`，不能悄悄重跑或计作算法失败。

不再运行 dev36：从 A02 保存的记录读取。不得以更多 dev 次数挑选“更好的一次”。

## 6. A0：区分“没答好”与“改名后行为改变”

一个策略可以在原问题及改名后的问题上同样绕路，因此 A0 成绩不到 90% 本身不能证明对象绑定或等变性实现错误。

A03 对三种方法统一采用以下处理：

- 记录 A0 原始成功率、决策完美率、绕路和循环；不再调用 `CL.a0_gate` 来阻断 A1/A2/B。
- 复用已有原问题 dev 记录和冻结改名映射，能直接比较的配对结果并列呈现。缺少原动作/概率记录时标 `PAIR_DETAIL_UNAVAILABLE`，不补跑 dev。
- 配对动作不同也不自动代表实现错误：保留同分动作、最优动作多解和名称 tie-break 的可能性。
- 只有可直接证明的对象映射错位、case 混用、非法候选、输入或权重改变才算技术完整性失败。A0 低分或配对差异本身只是待解释结果，不能自动修改方法。

这项变更与取消全员 ID 性能门一起，在任何 A03 测试前冻结；不是等 A0 失败后再次补规则。

## 7. 评价指标：保留原主指标，不换成更好看的指标

### 7.1 主指标

继续使用 **episode decision-perfect rate**，沿用冻结 metrics 的定义、最优动作集合和判定逻辑。最优参考无法求解的决策标 `REFERENCE_UNAVAILABLE`，不能当作“没有检测到错误，所以完美”。

完整列出 success、decision-perfect 的分子和分母，并校验：一个 episode 被认定决策完美时，必须成功，且完整轨迹的每步参考标签有效。若现有字段与这项逻辑存在不一致，原字段原样保留，并单列缺口，不能静默重新定义旧成绩。

A1/A2 等权均值可以作为主测试摘要；A0 不并入此均值，B 按规模单列。不把 A0 的同构成绩用于抬高主要泛化分数。

### 7.2 次指标与行为

成功率、成功局超额步数、失败局实际步数、首次非最优动作位置、重复状态—动作循环、目标破坏和计算成本均报告。失败局不伪造“超额步数=0”，不只报告成功局而隐藏失败分母。

**拆掉已完成目标不一定是错误。**在需要先拆后搭的题中，它可能正是最优动作。仅当选择偏离完整最优动作集合时，才能称为可避免的错误；原始 `destroyed_satisfied_goal` 是行为记录，不是单独的失败原因证明。

短图距离同样不等于充分表达能力：某个动作三跳可到最近一个目标，并不证明所有目标关系、顺序约束或对象身份都已被网络正确区分。不得从 `hop<=3` 直接推出“已排除所有表示限制”。不为此新增训练或改图。

### 7.3 ID 差距必须与新结果放在一起

主表格式：

```text
method | ID success/36 | ID perfect/36 | A1 success/perfect | A2 success/perfect | B6 | B7 | B8
```

对于每对模型、每个测试切片，同时报告：两者都完美、仅 A 完美、仅 B 完美、都不完美。差异基于相同 case 的完整分母，不只挑共同成功案例。

可以描述 `test_rate - ID_rate`，但它不校正训练质量，也不把不同分布变成配对实验。禁止通过除以 ID 分数、删除 ID 难例对应测试题或事后加权，制造“公平归一化”的排名。

这些是一组固定训练所得策略的实测差异，不自动是纯表示因果效应，更不是跨 seed 的显著性结论。A03 不补 seed、不追加显著性检验。[R7]

## 8. 规划器与计算成本

沿用冻结的 exact planner、动作成本及搜索资源限制。报告原始计划长度、搜索节点、成功与 timeout；不能因学习方法更慢而不报告规划器结果。

策略访问状态的 oracle 查询可以缓存，以减少分析耗时；但缓存命中的耗时不能被当作规划器从零求解的速度。初始问题参考单独计时，记录是否冷缓存。

复用现有单次评估计时，不另做大型性能测试。不把 GPU 异步前向时间和 CPU/wall 计时直接写成精确倍数。仅提供实际测量口径及原值；不以推理耗时推断样本效率。

## 9. 已知测试接触史与信息隔离

A02 回执确认：这三个注册的 final imitation checkpoint 尚未接触正式切片。[R2]

同时，PPO 停点总结披露：更早在注册前，未注册的三更新 smoke checkpoint 曾用 A0/A1/A2/B 干跑评估/结果脚本。[R6]

两者必须同时保留，准确标签为：

```text
PREMATERIALIZED_SUITE_WITH_DISCLOSED_SMOKE_EXPOSURE
registered_A02_checkpoint_test_exposure_before_A03: false
protocol_selection_used_ID_results: true
```

不称为整个项目“从未见过的完全盲测”。本轮不重建测试集，不声称已经独立核验那个 smoke 过程是否影响过更早开发。后续任何依据 A03 结果设计的方法，都不能继续把本套题当成完全未接触的最终确认题。

## 10. 结果解释与算法决策出口

A03 不颁发原先以“全部 ID gate 通过”为前提的确认性 `REP_*` 标签，也不把旧 A02 的 `LIMIT_TRAINING` 擦掉。

新结果状态：

```yaml
original_ppo_state: PPO_ID_GATE_FAIL
original_a02_state: IMITATION_ID_GATE_FAIL
original_rep_label: NOT_ISSUED
A03_scope: FROZEN_POLICY_GENERALIZATION_WITH_DISCLOSED_ID_GAP
A03_status: COMPLETE   # 仅在评估完整时
receptive_field_status: NOT_TESTABLE_STRUCTURALLY_EMPTY
```

报告需要回答四个决策问题，不再设新的硬成绩门：

1. **B2 与 QMARK** 在颜色反转、非同构和规模增长上，是否表现出清楚、可重复到多个案例的不同错误模式？
2. **ASNET** 是否与它们接近？若落后，这个差距是否主要延续 ID 已有绕路，还是在特定新条件下出现新的集中失败？这只能作为条件性解释，不能自动归因为架构。
3. **搜索** 在这套题上是否仍简单且强？若是，学习工作的主张应落在表示理解和迁移，而非虚构搜索困难。
4. **下一项最小算法改动** 应解决哪个已实际观察到的问题？描述对象、预期行为变化及一个必要控制，不自动实现或铺开训练。

新算法不必以“所有已有方法都失败”为唯一前提；可针对有意义的稳定差异、协调错误或计算瓶颈提出候选。也不因已有相似方法而否定方向。没有明确问题时，先保留结果，不通过继续训练寻找想要的排序。

完成本轮后应提交一份明确的主线建议，而不是仅重复“需要更多审计或 seed”。

## 11. 输出最小集

新根目录包含：

```text
registration/a03_spec.json          # 决策、模型、split、原 gate 及授权范围
registration/source_identity.json
receipts/asset_check.json           # 单次必要完整性检查
receipts/case_ledger.jsonl          # 未开始/开始/完成/技术未完成

eval/<method>/a0_final.json
eval/<method>/a1_final.json
eval/<method>/a2_final.json
eval/<method>/b_final.json
eval/<method>/decision_log.jsonl
eval/planner_baseline.json

results/final_summary.md
results/id_and_test_table.csv
results/by_case.csv
results/paired_comparisons.csv
results/claim_boundary.md
results/verify.json
receipts/final_receipt.yaml
```

必要时可以合并小文件，不为形式再生成几十个审计文件。final_summary 中保留完整原始分数、协议修改、测试接触史、ID 差异和未完成项。

## 12. Git、停止与回执

继续在当前分支工作。开始时确认 HEAD 为上述 base；A03 spec、最小评估入口和必要 fixture 测试完成后，先提交并推送一次评估登记，再开放正式切片。

建议提交：

```text
research: authorize frozen-policy evaluation with disclosed ID gap
research: record Blocksworld A03 frozen-policy evaluation
```

不创建 PR、不 merge、不 force-push，不提交 `.pt` 或密钥，不修改旧 PPO/A02 目录。模型文件前后哈希一致；新增源码身份单独记录。

**不因低成绩停止。**只因真实输入/模型/标签完整性故障停止并保留已执行证据。超时或资源中断如实记录，不能当成模型能力失败，也不得用无限重试覆盖。

最终回执：

```yaml
amendment: CP-DISR-C1-BW-FROZEN-EVAL-A03
base_commit: b6804664e0f0ddb2f9b7068d9c49fdf5e6254a6a
branch: codex/cp-disr-c1-blocksworld-main-v1
registration_commit: <full_sha>
result_commit: <full_sha_or_external_git_reference>
old_results_modified: false
original_gate_status: IMITATION_ID_GATE_FAIL
new_training_runs: 0
optimizer_steps: 0
checkpoint_selection: A02_FINAL_ONLY
id_gate_relaxed_or_rewritten: false
evaluation_release_rule_changed: true
a0_performance_stop_removed_for_all: true
learned_episodes_planned: 408
learned_episodes_completed: <count>
planner_cases_planned: 136
planner_cases_completed: <count>
technical_incomplete: <count>
original_rep_label: NOT_ISSUED
receptive_field_status: NOT_TESTABLE_STRUCTURALLY_EMPTY
state: C1_BW_A03_EVALUATION_COMPLETE_WITH_ID_GAP
next_action: WAIT_FOR_USER_ALGORITHM_MAINLINE_DECISION
```

若有技术缺口，state 使用 `A03_TECHNICAL_INCOMPLETE`，不能输出完整完成状态。

## 13. 给执行 Agent 的短指令

```text
批准执行 CP-DISR-C1-BW-FROZEN-EVAL-A03，只做评估，新训练为 0。
保留 A02 的 IMITATION_ID_GATE_FAIL，不把 33 改成 32，不单独放宽 ASNET。
从 b6804664e0f0ddb2f9b7068d9c49fdf5e6254a6a，在原分支新建 A03 登记和结果目录。
按清单核验三个 A02 final checkpoint 和 split；不得挑其他 checkpoint 或重跑 dev。
新增最小评估入口，复用原模型与 metrics，不修改旧 gate 或旧结果。
三个模型统一执行 A0/A1/A2/B，规划器执行同一 136 个参考问题。
A0 分数只作诊断，不阻止后续切片；低分不是技术故障。
完整报告 ID 与测试差异，不颁发原 matched-ID REP 标签，不声称纯表示因果排名。
保留早期 smoke checkpoint 接触测试文件的披露，不称为全项目严格盲测。
结束后提交轻量结果，停止；不得训练、调参、换测试题或开发新算法。
```

## 14. 证据来源与核验范围

以下仓库路径为本文件起草时直接读取的记录；绝对服务器文件是否仍在，由执行时的哈希检查确认。文件内容中的统计并不等于本次重新运行了实验。

- **[R1] A02 final summary**：`runs/final_master/c1_route_b/blocksworld_main_v1/imitation_a02/20261006T053652Z_8ea6a71a/results/final_summary.md`，ref `b6804664e0f0ddb2f9b7068d9c49fdf5e6254a6a`。
- **[R2] A02 receipt**：同根 `receipts/final_receipt.yaml`，同 ref。
- **[R3] Checkpoint identity**：同根 `results/imitation_checkpoint_identity.json`，同 ref。
- **[R4] A02 registration/config**：同根 `prep/c1_bw_imitation_a02.json`，同 ref。
- **[R5] 原评估入口**：`scripts/c1_bw_eval.py`，同 ref；确认存在 ID/A0 成绩阻断及向旧模型目录写入结果的路径。
- **[R6] PPO 停点总结**：`runs/final_master/c1_route_b/blocksworld_main_v1/20261006T015313Z_490a3b11/results/final_summary.md`，ref `72c0fd2cdc4394dcdee3b5bd6bba210060535d57`；早期 smoke 接触史沿用此已提供的记录，未独立复查服务器临时文件。
- **[R7] 评价不确定性参考**：Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*, NeurIPS 2021 / arXiv:2108.13264，https://arxiv.org/abs/2108.13264 。只用于说明少量训练下不能把单次分数当作普遍排序；A03 的具体取舍是本项目研究决策，并非该论文规定。
