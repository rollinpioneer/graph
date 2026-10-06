---
title: "CP-DISR C1：最小归因与必要回退能力实验"
subtitle: "结合本轮交叉审查的执行文档 v2"
date: "2026-10-06"
card_id: "C1-BW-GP-MINIMAL-ATTRIBUTION-V1"
document_version: "2.0-audit-integrated"
status: "PLAN_ONLY"
execution_authorized: false
repository: "rollinpioneer/graph"
base_branch: "codex/cp-disr-c1-bw-goal-progress-v1"
base_commit: "b625cfc7f4c83f2c1cf054eef58538d77363c1d5"
proposed_branch: "codex/cp-disr-c1-bw-gp-minimal-controls-v1"
new_training_run_cap: 2
confirmation_target_cases: 112
learned_and_rule_conditions: 9
---

# 0. 执行摘要

**本卡保留两条训练，不换主线、不增加模型家族。改变的是测试的重点：不再只看“能不能多搭几座塔”，还要看“需要暂时拆掉已完成目标时，模型能不能作出正确选择”。**

两条训练分别是：

| 新条件 | 与谁配对 | 唯一有意改变的实现因素 | 首要问题 |
|---|---|---|---|
| **B2-RANK** | 已完成的 M0 | 在原 B2 训练路径上加入排序损失 | 换损失是否已足以获得组合能力？ |
| **M1-GOAL** | 已完成的 M1 | 编码器恢复目标标记 | 逐目标特征不受其他目标影响这一性质，是否是组合能力的重要来源？ |

执行顺序：少量必要检查与一次 G1 开发集评价 → 冻结新题、权重、实现及训练配置 → 两条固定训练 → 九个模型／规则条件的一次确认评价 → 输出明确算法选择建议并停止。

新题目标总数 **112**：32 道可不拆最优多塔题、32 道最优解必经拆目标的多塔题、16 道高度 4 题、16 对颜色互换孪生题（32 道）。

**无 ID 成绩放行门，无临时延长训练，无自动第三条训练。** 审查提到的 M1-NLL 只列为本卡结束后的候选，不在本卡授权范围内。

本文是实验计划，不是本次已执行回执。用户随后明确授权本卡后，执行者才可修改仓库、训练和评价。旧实验的授权不自动延伸到本卡。

---

# 1. 依据、已知结果与本轮采纳的修订

## 1.1 来源层级

- **本次审查材料 [A1]**：`CP_DISR_C1_本轮结果审查_20261006.md`。其中分层重算数字是审查者报告的结果；本文件作者没有在服务器重新执行这些统计。
- **配套脚本 [A2–A3]**：`train_destruction_labels.py`、`confirm112_strata.py`。用于只读标签和历史成绩，不加载模型、不训练。
- **仓库实现与已提交记录 [R1–R8]**：在完整提交 `b625cfc7f4c83f2c1cf054eef58538d77363c1d5` 上核对训练路径、标签定义、规则实现和身份记录。
- **本卡新增设计**：新题配额、组合规则优先级、解释阈值和执行停止边界，均属于本卡事前方案，不冒充审查或旧实验已经证明的事实。

## 1.2 当前结果，应按两类问题分别理解

审查对已消费 Confirm112 的分层结果如下。表中是成功数；“需要拆”严格含义见 §3。[A1 §2.1]

| 方法 | 可不拆的最优方案存在，94题 | 所有最优方案均需拆目标，18题 |
|---|---:|---:|
| 原 B2 / C0 | 39 | 8 |
| M0 / 原 B2 继续训练 | 40 | 8 |
| M1 | 84 | 8 |
| M2 | 84 | 8 |
| G1 / 直接目标进展规则 | 81 | 0 |
| C3 / 已访问后继排除 | 89 | 16 |

这支持：新方案在第一类题上有很大的净改善；在第二类题上的总成功数没有提高。不能由后一项说模型毫无相关能力，也不能把“均为8题”理解成做对了相同的8题。

核心48题中，M1与G1都全程最优42题，共同全程最优40题，各自独有2题。这是**逐题正确性重合**，不是已经证明每一步动作、评分或内部机制相同。[A1 §2.1]

A1-5 中，M1/M2均为3/8成功，原B2为8/8；但该组有3道最优方案必须拆目标的题，G1也只有4/8。暂称“颜色反转题上的退化”，**颜色是否造成退化必须靠成对干预检验**。[A1 §2.3]

## 1.3 M1/M2 的窄义结论

代码定义为：[R3]

\[
V_{M1}(F,G)=\rho\!\left(\sum_{g\in G}\phi(x(F,g))\right),
\quad
V_{M2}(F,G)=\sum_{g\in G}\rho(\phi(x(F,g))).
\]

两者都使用**不含其他目标标记的逐目标特征**，并做求和。当前无明显差别，只说明未观察到“最后一层非线性放在求和前／后”的额外收益；不能扩大成“任何整体模型和任何分解模型都等价”。

B2本来就计算候选后继。M0到M1的实际变化还包括：目标标记清零、注意力读出改为求和式标量评分、移除GRU／时间／候选上下文、显式逐目标真值位、排序损失和不同可训练模块。不能说新方法首次开始看后继状态。[A1 §2.4；R3–R5]

## 1.4 对审查建议的明确处理

| 审查建议 | 本卡处理 |
|---|---|
| 保留 B2-RANK、M1-GOAL 两条训练 | 采纳，硬上限2条 |
| 按是否需要拆目标分层 | 采纳；保持“所有最优方案”的原始标签含义 |
| 加颜色互换孪生题 | 采纳；整对隔离、双向配对报告 |
| G1在dev36上评价 | 采纳；一次评价或复用已有等价记录，不作训练放行门 |
| G1+C3、M1+C3 | 采纳为零训练参照，明确组合顺序，不调参 |
| 确认集“约128题” | 审查列出的配额实际相加为112，本卡采用112，不为了凑数加16题 |
| 32题差5题、16题差3题再解释 | 作为探索性方向阈值采纳，不当成显著性或等效性检验 |
| 条件性第三条M1-NLL | 仅形成后续建议；本卡不自动运行 |
| 两种颜色都失败即可排除颜色 | 不采用这种排他推断；双失败不能排除颜色与结构共同作用 |
| M1-GOAL下降就“证明唯一机制” | 收窄为该受控实现变化的证据；输入分布与训练路径等解释仍保留 |

**不追求绝对首次提出。** 本卡要获得的是可验证的能力差异、实施更简洁的方案或更清楚的适用边界，不以增加模块或消融数量作为贡献本身。

---

# 2. 仓库、起点及旧证据保护

## 2.1 分支

```yaml
repository: rollinpioneer/graph
base_branch: codex/cp-disr-c1-bw-goal-progress-v1
base_commit: b625cfc7f4c83f2c1cf054eef58538d77363c1d5
new_branch: codex/cp-disr-c1-bw-gp-minimal-controls-v1
new_result_root: runs/final_master/c1_route_b/blocksworld_main_v1/gp_minimal_attribution_v1/<UTC>_<prep_sha8>/
```

该基线分支已在撰写时查询。新分支名称只是建议，未在本轮创建。如果执行时已有同名分支，先读取它的状态；不得覆盖已有实验、回退他人提交或强推。

## 2.2 两条训练的共同初始资产

两条都从 **A02 的原 B2 最终权重**出发，不能加载 M0/M1 的训练后权重继续微调。

```text
run: R-C1-BW-IL-B2-0
path: runs/final_master/c1_route_b/blocksworld_main_v1/imitation_a02/20261006T053652Z_8ea6a71a/runs/R-C1-BW-IL-B2-0/checkpoints/final.pt
sha256: 29c7eab495b4dea7a3f30f40fe7321619bdbe941153b3a69c2a937a528d61e11
size_bytes: 19860355
```

执行者须在实际服务器核验文件存在和哈希；仓库路径不是本助手已取得权重的证明。

## 2.3 已有配对参照

以下路径均相对仓库根，GP_ROOT 为：

```text
runs/final_master/c1_route_b/blocksworld_main_v1/goal_progress_v1/train_confirm/20261006T095758Z_90758ab1
```

| 条件 | 路径（接在 GP_ROOT 后） | SHA-256 | 字节 |
|---|---|---|---:|
| M0 | `runs/R-C1-GP-M0-0/checkpoints/final.pt` | `973925b522af2ea53d621b68b96005d23fae45a11c45423d94c0a1a9012aa82f` | 19860291 |
| M1 | `runs/R-C1-GP-M1-0/checkpoints/final.pt` | `8144d0ec6227d0eb8ae5403fe0bf72dacf351ff6a1bbf1051e5adb33ca2ae94f` | 17880803 |

它们只用于新确认题评价，不再训练。身份来自旧运行的 `training_accounting.json`。[R6–R7]

## 2.4 所有旧目录只读

T_B、PPO、A02、A03、A04P、Pilot80、Confirm112、旧M0/M1/M2训练目录均不得写入。保留所有旧gate失败、旧方向标签及烟雾测试接触披露。新增标签写入本卡，不能反向改判旧实验。

---

# 3. 主分层：什么叫“需要先拆目标”

直接复用生产 `planner.destruction_labels(problem, solver)`。[R2]

记 `D*` 为其返回的 `requires_goal_destruction`：

- `D*=False`：至少存在一条**最优**计划，从不把此前已为真的目标原子变为假。简称“可不拆最优层”。
- `D*=True`：每条**最优**计划都会至少一次把此前已为真的目标原子变为假。简称“最优必经拆目标层”。
- `None`：最优计划图枚举未完成或标签不可得；不能当作False。

这里包含初始已经满足的目标，也包含执行途中才满足的目标。标签是初始问题层面的性质，不由某个策略是否碰巧拆了目标决定。

**这不是“所有成功方案都必须拆”的证明。** 本卡不新开搜索项目去证明更强标签。成功率与全程最优率分别报告，避免将这个区别隐藏。

“拆掉已完成目标”本身不是错误；在 `D*=True` 中，最优方案必须这样做。逐步错误仍由“所选动作是否属于全部最优动作集合”判断。复用基线精确规划器及其限制，不给学习模型送入搜索结果。若诊断搜索达到时间、节点或深度限制，记为参考标签不可得而非无解；实际成功率仍可报告。整局最优性若无法判定则记NA，不把未知强转为成功或失败。

### G1 也不是永不允许拆目标

生产G1仅在存在“已满足目标集合严格增大”的合法动作时优先选它；否则退回B2原选择，原选择可以拆目标。因此历史0/18是观测成绩，不是G1结构上必然0分的证明。[R5]

---

# 4. 第0步：仅做影响本卡的必要核验

## 4.1 只读补全旧train/dev标签

优先复用已保存、能对应源提交和题目哈希的审查输出。若没有机器可读记录，用附件 `train_destruction_labels.py` 的逻辑生成**本卡旁置文件**：

```text
prep/old_train_dev_destruction_labels.json
```

预期审查数字为 train19/144、dev5/36，但它们不是资格门。若重算不一致，保留原始题目、标签、脚本版本和差异，不能修改题目使数字吻合。

附件脚本中 `bool(destroy)` 会把None当False；本卡的薄封装必须先检查 `destroy is None`，将它记为UNKNOWN。不要修改旧split的None字段；不要把这些新标签加进训练损失或模型输入。

## 4.2 G1在dev36上一次评价

使用原A02 B2和生产G1规则；原步数上限、平分顺序、输入和GRU保持不变。若已有等价的完整结果，核验哈希后复用，不能为形式重跑。

报告36题总成绩，以及5道审查标为 `D*=True` 的逐题结果。主对比是历史M1在dev36的逐题成绩；只重读，不重新运行M1。

用途：了解学习方法在旧分布内是否做到了简单规则没做到的事情。**G1做得好或差都不自动证明颜色根因，也不触发停训、加训、换题。**

## 4.3 附件统计脚本的使用边界

`confirm112_strata.py` 只重读旧结果。固定GP_ROOT，不使用“匹配多个目录后取第一个”的不确定glob；布尔值显式解析，过滤掉dev行；每个condition×case必须唯一。旧结果已经有可靠导出时直接复用，不必重新生成所有日志。

本步不是扩大审计项目：不加载旧QMARK/ASNET，不重跑旧确认集，不检查无关分支。

---

# 5. 训练一：B2-RANK，严格只增加损失项

## 5.1 问题与边界

与旧M0的唯一科学配置差异是 `rank_weight: 0 → 1`。它测试“把相同排序约束加到原B2读出上是否足够”，不是测试排序损失在所有网络上是否有效。

```yaml
run_id: R-C1-GPA-B2-RANK-0
paper_label: B2-RANK
architecture: original_B2_CACHED
training_reference: R-C1-GP-M0-0
init: A02_B2_final
rank_weight: 1.0
```

## 5.2 实现约束

必须扩展旧M0实际使用的 `imitation.ImitationTrainer`，不能直接改用M1的 `GPTrainer`。[R3–R4]

保持：

- `trajectory_hidden`、完整轨迹的GRU反向传播；不能逐步重置hidden或切断原有梯度。
- 原 `base_input`、候选8维特征、目标标记、CandidateReadout和后继向量差。
- `_group_logits` 的B2路径、原 `decision_weights`、原分块和累积方式。
- M0的可训练模块：encoder、observation、gru、candidate、contract、base；原来冻结的模块继续冻结。
- 优化器从空状态开始；所有可训练参数学习率1e-4。

对同一批B2 logits调用现有 `goal_progress.nll_and_rank`。公式为：

\[
L=L_{\rm IL}+L_{\rm rank},\qquad
L_{\rm IL}=-\log\sum_{a\in A^*}\pi(a\mid F,G),
\]

\[
L_{\rm rank}=\operatorname{mean}_{d(a)<d(b)}
\operatorname{softplus}(z_b-z_a).
\]

其中 `d(a)` 是原标签文件中动作后状态的精确剩余距离；只比较严格有序的合法动作对，距离相等不比较，无有效对时排序项为0。每轨迹总权重相同，不让长轨迹自动权重更大。

排序监督使用原冻结文件，不新增规划器标签种类、不重采轨迹。

## 5.3 训练前等价检查

在同一初始权重、同一固定批次上将新实现 `rank_weight=0`，与旧M0训练路径比较logits、损失及每个可训练参数梯度。此检查只做前向和反向，不更新参数，不构成第三条训练。

同环境下能逐字节一致则记录哈希；浮点累加顺序不同则记录逐张量误差。默认数值判定 `atol=1e-6, rtol=1e-5`，不要求跨GPU梯度字节哈希完全相同。若超差，先修实现，不能扩大容差去包住算法差异。

`diff_vs_M0.json` 分开列出科学配置与路径元数据；科学配置只允许rank_weight不同，run_id、输出路径、源提交等元数据当然不同。

---

# 6. 训练二：M1-GOAL，目标无关与目标条件编码的比较

## 6.1 正确定位

**主要看多塔能力是否变化，随后看需要回退的层、颜色孪生和高度层。** 它不是承诺修复颜色的补丁。

与旧M1相比，只恢复传入图编码器的 `prop_goal_sign`；其余特征、评分、损失、初始化及训练配置保持不变。

```yaml
run_id: R-C1-GPA-M1-GOAL-0
paper_label: M1-GOAL
training_reference: R-C1-GP-M1-0
mode: global
encoder_goal_marks: production_marks
init_encoder: A02_B2_final
head_seed: 20261006
rank_weight: 1.0
```

## 6.2 模型定义

旧M1：[R3]

\[
x_0(F,g)=[h_g,\operatorname{mean}H_{action},
\operatorname{mean}H_{prop},\operatorname{truth}(F,g),\operatorname{sign}(g)],
\]

其中图编码时所有目标标记为0。维度386，头部为同一 `phi/rho`，先求和后评分。

新M1-GOAL使用同样拼接，只让编码器读取生产目标标记：

\[
x_G(F,g;G),\quad
V_G(F,G)=\rho\!\left(\sum_g\phi(x_G(F,g;G))\right).
\]

候选logit继续为 `V_G(F,G)-V_G(T_nom(F,a),G)`，不加B2残差，不恢复GRU、时间输入或候选上下文。目标真值位、goal sign、动作/命题均值全部保留。

新子类可以只覆盖构造静态图张量的函数，让编码使用 `st` 而非目标清零后的 `st0`。当前状态和所有候选后继都用同一规则；不能只给一侧加目标标记。

只恢复已有输入，不新增头、层数、关系边或训练特征。旧M1加载器和路径保持不变。

## 6.3 为什么不能只解释成“时机”

这个开关同时使每个目标的特征依赖整个目标集合，并让编码器输入更接近B2预训练时的分布。这些都是该开关的后果，不是已被进一步隔离的独立原因。[A1 §2.6]

结果能支持“本实现中目标条件化编码的影响”，不能凭一条训练就证明“唯一机理是目标干扰”或“普遍应当晚注入目标”。

## 6.4 匹配检查

- 初始encoder逐张量等于A02 B2。
- 新头按旧M1同一种子与模块创建，初始SHA应为：
  `57ab0cf2512b02f95dee91fe73a9980471e5310548e7e4d1840340758f887f37`。
- 头部参数82,688、可训练总参数预期513,472，与M1相同；不得用总state_dict大小冒充有效可训练参数。
- 新实现关闭目标标记开关后，与原M1的初始logits、损失、梯度在相同容差下等价。
- 开关开启时检查真实 `prop_goal_sign` 被送入消息传播；不要要求某个手写例子必须变好。
- 目标顺序置换只改变索引顺序，不改变输出；换颜色则允许改变输出。

---

# 7. 数据与训练预算：复用、不扩充

## 7.1 复用原 D_train

```text
GP_ROOT/datasets/D_train.json
trajectory_count: 1296
decision_count: 7446
case_scope: train144, n<=5, single_nontrivial_tower
```

身份必须同时验证文件和语义摘要：[R8]

```yaml
d_train_file_sha256: 387e66309c51195ad017b2b90e9f2742090457d108a353f34159acd8f496bf78
d_train_semantic_sha256: 2b3380200d8aaf719871a4b2ae446bcfb1ed992dfb821cd6fb2f0bbb12d31d69
rank_labels_sha256_as_recorded: 24b5e335a727ce8328947451a55f1b6bdb5c686a0715a74f8b17b77f8da90064
```

排序标签按原身份生成函数核验，不把语义摘要误当文件字节哈希。登记时同时保存它们的实际字节SHA。

不增加状态收集，不把新确认题、颜色孪生、dev的破坏标签或首错状态加入训练；n=3旧重复权重保持原状。

## 7.2 固定训练参数

| 参数 | B2-RANK | M1-GOAL |
|---|---|---|
| 初始encoder/原模型 | A02 B2 final | A02 B2 final |
| 新头 | 无 | 与M1相同种子初始化，不加载M1 final |
| 优化器 | Adam重置 | Adam重置 |
| 学习率 | 全部1e-4 | encoder 1e-4；新头3e-4 |
| betas / eps / weight decay | (0.9,0.999) / 1e-8 / 0 | 相同 |
| epoch | 100 | 100 |
| 每批轨迹数 | 32 | 32 |
| 打乱 | 原 `0*100003+ep`，ep=0..99 | 相同 |
| 梯度裁剪 | 0.5 | 0.5 |
| 决策分块 | 原M0的256 | 原M1的192 |
| 权重 | 原逐轨迹等权 | 原逐轨迹等权 |
| 损失 | 最优集损失+1.0排序项 | 与M1相同 |
| 正式权重 | epoch100 final | epoch100 final |

预期每条4,100次优化更新、744,600个决策样本展示。它们不是744,600个独立训练状态。参数量预期 B2-RANK=701,889、M1-GOAL=513,472；**本卡保证各自与M0/M1配对，不宣称两个新模型彼此等容量、等计算。**[R4；R6–R8]

不增加epoch，不调排序权重，不根据训练曲线选中间权重。既有硬件不同则记录，不能从墙钟直接推导样本效率。

---

# 8. 新确认集：112题，按必要回退分层并加入颜色孪生

## 8.1 总体配额

命名空间：`C1-BW-GPA-REVIEW-v2-20261006`。建议文件 `configs/splits/c1_bw_gp_attr_confirm112_v2.json`，不能覆盖旧Confirm112。

| 大层 | 数量 | 目的 |
|---|---:|---|
| T-MONO：可不拆最优多塔 | 32 | 验证组合收益是否保留 |
| T-BREAK：最优必经拆目标多塔 | 32 | 检验是否超出简单进展规则 |
| H4：高度4，可不拆/必经拆各8 | 16 | 不把塔高和需要回退混为一谈 |
| COLOR：n=5高度3，16对红蓝互换 | 32 | 在相同物理问题上识别颜色影响 |
| **总计** | **112** | 新确认题不是旧Confirm112复用 |

审查文字写“约128题”，但给出的32+32+16+16+16实际为112。本卡明确采用112，避免无理由扩样本。

## 8.2 多塔层的结构匹配

T-MONO与T-BREAK采用相同四组目标形状，每组各8题（括号为各塔高度，1表示单独在桌面的积木）：

| 结构组 | 积木数n | 非平凡塔数k | 最大塔高h | 形状 | 每个D*层 |
|---|---:|---:|---:|---|---:|
| T6-2 | 6 | 2 | 3 | (3,2,1) | 8 |
| T6-3 | 6 | 3 | 2 | (2,2,2) | 8 |
| T7-2 | 7 | 2 | 3 | (3,2,1,1) | 8 |
| T8-3 | 8 | 3 | 3 | (3,2,2,1) | 8 |

沿用原交替颜色、塔顶红色生成规则。两层通过独立规划器标签划分，不通过模型表现划分。

记录每题初始目标满足数、最短长度L*、n/k/h，按L*的1–8、9–16、17–24档报告。它们不是完全正交的因果设计；不能仅凭层间平均差断言“只因拆目标更难”。

不为配平所有长度而反复改生成器。相同生成规则、结构配额和固定筛选顺序优先；实际长度差异透明报告。

## 8.3 高度4层

固定 n=6、k=2、形状(4,2)、塔顶红色，D*两层各8题。与T6-2的 n=6、k=2、h=3 条件并列；不跨D*混合解释塔高。

允许表述“这些匹配n、k的目标形状下的差异”；不能宣称完全隔离塔高与目标边数、初始状态、最短长度的影响。

## 8.4 颜色互换孪生

固定 n=5、k=1、h=3、形状(3,1,1)。生成16个物理问题骨架，目标为D*=False/True各8对。

每对包含：

- RED：按旧训练颜色方向，塔顶红色。
- BLUE：把每个积木的红/蓝属性全体互换，塔顶蓝色。

**保持完全相同：**对象名字、动态初始状态、完整目标支撑关系、候选ID、动作顺序、合法集合、环境转移、L*、步数上限、初始hidden和所有评估设置。仅公开颜色属性及相应静态合同颜色前提随颜色一起变换；不能只改显示标签或故意令动作不合法。

逐对验证合法动作与后继状态一致、规划器最优长度与D*标签一致。网络可以受颜色影响；规划器解答应不变。

登记 `pair_id`、`color_variant`、`skeleton_hash`、每个成员的保色问题同构哈希。颜色对不是32个独立骨架，应按16对报告。

## 8.5 隔离与预先不足规则

排除所有已有train/dev、资格题、A0/A1/A2/B、Pilot80、Confirm112以及可获取已登记问题的**保色问题同构类**。同构哈希包含颜色、初始状态与完整目标，不只比较目标形状。[R9]

孪生对的两个成员都分别查重；任一个碰撞，整对剔除。新集内部也整对查重。若两个成员本身落在同一个保色同构类，剔除该对，避免把无实质颜色对比当成有效孪生。

允许沿用旧目标形状，这是新实例确认，不声称新增任务族。查重不以“必须是训练之外的目标形状”误删RED同分布孪生。

生成只可使用合法性、题型、D*标签、精确可解性和去重；**不调用任何模型，不按G1/B2成绩挑题。**固定每个8题单元最多2,000次候选生成；颜色对每个D*配额最多10,000次。上限是生成保护，不是得到坏结果后的重采机会。

标签None、规划器未完成、无解或初始已成功的候选不纳入；记录拒绝原因。若配额不足，在模型调用前保留可得实例、冻结实际分母，不复制补齐、不放宽隔离、不临时换谓词/目标。空层保留为“未覆盖”；样本不足时只做描述，不能下该层结论。

**不为了凑112而重设计整个任务。**实际分母一经登记，之后不能由成绩改变。

---

# 9. 九个条件与规则组合的精确定义

| 条件 | 使用的权重 | 是否训练 | 正式动作选择 |
|---|---|---:|---|
| C0 | A02 B2 final | 0 | 原argmax |
| M0 | 旧M0 final | 0 | 原argmax |
| M1 | 旧M1 final | 0 | 原M1评分argmax |
| B2-RANK | 本卡final | 1条 | 原B2路径argmax，无规则 |
| M1-GOAL | 本卡final | 1条 | 新目标条件编码评分argmax，无规则 |
| G1 | A02 B2 final | 0 | 生产直接进展规则 |
| C3 | A02 B2 final | 0 | 生产已访问后继排除 |
| G1+C3 | A02 B2 final | 0 | 先排已访问后继，再在余集中作G1选择 |
| M1+C3 | 旧M1 final | 0 | 先排已访问后继，再按M1评分选 |

精确规划器是第十种参照，但不训练，不向任何策略输入最优动作或距离。

## 9.1 共同设置

原合同、完全可观测状态、`step_cap=2L*+4`、正成本、确定性argmax、动作平分规则不变。B2系模型沿各自真实轨迹更新GRU；M1系无GRU。原规则读取名义后继，不新增搜索树或回滚环境。

每局访问集合初始包含原始状态，每执行一次动作加入实际新状态，每题清空。状态键按完整具体状态区分对象，不能用跨题同构哈希合并本局不同状态。

## 9.2 G1 与 C3 必须忠实复用

- G1：若存在 `satisfied_before` 的严格超集 `satisfied_after`，在这组动作中按B2分数选；否则回到全部合法动作的B2选择。不升级成“最大化计数增幅”，不增加其他优先级。[R5]
- C3：过滤后继已在本局visited中的合法动作，再按原评分选；没有剩余动作时以 `NO_UNVISITED_SUCCESSOR` 结束，作为控制器失败，不伪装为环境无合法动作或技术缺失。[R5]

## 9.3 组合顺序固定

G1+C3：

```python
available = [a for a in legal if nominal_next(state, a) not in visited]
if not available:
    stop_reason = "NO_UNVISITED_SUCCESSOR"
else:
    improving = [a for a in available
                 if satisfied(state) < satisfied(nominal_next(state, a))]
    selected = b2_argmax(improving if improving else available)
```

M1+C3使用相同 `available`，直接按旧M1评分选择。两者是零训练参照，不叫新主方法。其他优先级、访问信息、兜底行为不允许在测试后调整。

---

# 10. 主指标与解释阈值

## 10.1 三组主读数，不合成一个总冠军分数

**组一：多塔，分开报T-MONO和T-BREAK。**每层都报告成功数S、全程最优数D、步数、循环、首次错误。主要配对是 B2-RANK↔M0、M1-GOAL↔M1。

**组二：颜色孪生。**对S和D分别报告四格：两边都对、RED对/BLUE错、RED错/BLUE对、两边都错；再按D*标签分层。

**组三：相对简单规则的新增能力。**每个模型与G1逐题列两者都对、仅模型对、仅G1对、两者都错；报告T-BREAK层和H4层的变化。C3、G1+C3、M1+C3以成功与绕路共同评价，不能只比较一个指标。

## 10.2 固定方向阈值

以下只是资源受限的研究决策阈值，不是统计显著性、种子波动估计或等效性界限。[A1 §5–§6]

```text
N=32 的层：净成功/全程最优差至少5题，才标注清楚的方向信号。
N=16 的层或P=16孪生对：净差至少3题。
实际16<=N<32：阈值 ceil(5*N/32)。
N<16（或P<16）：只报原始配对数，不颁发方向标签。
```

所有小差距照常报告，只是不据此扩大机制主张。“差距未到阈值”叫**未观察到清楚差异**，不叫“等价”或“没有作用”。

同层对比时记为 `net=仅A成功−仅B成功`，两者同题因此等于成功总数差。成功和全程最优分别算。若二者方向冲突，标记完成与最优性取舍，不挑一个更有利的数字。

颜色配对净差 `RED-only − BLUE-only`；必须同时报告反方向项，不只数BLUE失败的一边。跨模型修复使用同pair结果差，不将两边都失败解释成颜色无关。

**增加题量不能替代独立训练种子。**本卡仅对固定训练条件及固定权重给出单种子证据，不采纳“±2–3题就是测得的种子波动”的说法。

## 10.3 与G1的逐题一致

定义正确性一致率为 `(both_correct+both_wrong)/N`；另报共同正确集合的交并比和各自独有正确题。分别对S、D计算。

这些统计不是动作轨迹一致性。若额外报告同动作序列比例，必须真比较逐步序列；两条策略进入不同状态后不能把对应步骤的动作不同当作同状态决策冲突。

## 10.4 目标破坏指标

逐步记录 `destroyed_satisfied_goal` 及所选动作是否最优。报告“必要的目标拆除”和“非最优动作伴随的拆除”时必须依据规划器集合；不能一概把拆目标当坏事。

辅助记录：G1/C3是否改动原最高分动作、改动前后动作是否最优、访问排除是否耗尽、正确首选被规则替换的局数、每局规划器参照是否完整。

## 10.5 开发集

两条新模型各评价dev36一次，分开S/D并附D*标签。旧C0/M0/M1只复用已保存dev结果。开发集未全对照样完成正式确认，不新增ID放行门；最终解释中并列ID差距。

---

# 11. 执行预算、冻结与测试隔离

目标预算：

```text
新训练：2条
新确认集：112题
模型/规则确认执行：9×112=1008局
精确规划器初始问题参照：112题
G1 dev36：最多36局（可复用已有）
两条新模型 dev36：72局
```

实际预算随事前冻结分母改变，不因结果改变。M2、QMARK、ASNET不重训不重评。M1-NLL不在预算中。

## 11.1 阶段顺序

1. 核验base、工作树与资产；读取旧结果身份，不读取新题给模型。
2. 完成§4与§5–§6的必要fixture。
3. 生成并冻结新确认集、两条训练配置、全部固定控制器、阈值、旧权重身份；提交并推送登记。
4. 两条训练可并行；不能共享optimizer或相互等分数选权重。训练期间不跑新确认题。
5. 两条训练结束后锁定epoch100权重，再统一开放九条件确认评价。基线不能先试新题后再回头调新模型。
6. 条件成绩差不阻止其他条件和切片。规划器仅用于事前资格/步数上限和事后评价标签，不替学习模型选动作。
7. 生成所有配对表、层表、颜色四格和选择建议；提交轻量结果，停止。

新确认集的查重/标签构造可以在登记前由规划器访问，但任何模型前向、烟雾试跑或人为调参都不能使用新确认题。烟雾测试使用人工例子和训练题的固定fixture。

## 11.2 不再制造“冻结检查拦住日志自己增长”的问题

源身份检查覆盖模型/环境/规则/训练器/生成器与注册配置。运行中写入的ledger、日志和结果是显式产物白名单，不参与“必须工作树完全干净”的代码检查；不能因此排除整个源目录或跳过冻结源码核验。

## 11.3 只为真实技术问题停止

哈希错、身份错、非法动作、合法候选上的非有限logits或非有限梯度、测试集误入训练、标签未解析等属于技术问题。非法动作mask位置的负无穷是预期值，不记为数值失败。保留发生过的记录，不将技术缺失计失败或成功。

单条训练崩溃不自动重跑，其余独立运行可完成并保存；不存在的checkpoint不参与测试，匹配对比标记INCOMPLETE。是否修复重启需另行决策，不能悄悄重来。

低训练分、低确认分、空的方向标签、规则胜过模型、颜色/高度退化都不是重训或改题理由。

评价进程断开时，已完成case不重跑；已started但未completed的case记技术缺口。可对尚未started的case继续同版本执行。修复若改变策略或任务语义，停止合并解释，不能拼成一个单版本结果。

---

# 12. 必要fixture：数量受控、只验证本卡开关

1. **B2-RANK零权重还原**：同批次logits/loss/梯度与M0路径一致；原GRU梯度仍能回传。
2. **排序标签**：全部最优动作集合保留；合法严格距离对正确；同距不排序、无对返回0、padding和非法动作不参与。
3. **M1-GOAL开关**：关闭等于旧M1；开启后当前与候选后继都使用真实目标标记；没有恢复其他B2上下文。
4. **参数与初始化**：各自与历史参照匹配；M1-GOAL头部初始SHA一致；optimizer重置。
5. **颜色孪生**：除颜色及其一致的静态前提外字段相同；环境转移/L*/D*不变；双方独立查重。
6. **标签None**：故意给未知标签，必须登记UNKNOWN而非False；不要以bool强转掩盖。
7. **G1/C3组合**：有未访问进展、只有未访问非进展、没有未访问三种情况；组合顺序固定且不泄露最优标签。
8. **产物隔离与ledger**：写日志不触发源码漂移；修改源码仍会失败；同condition×case重复启动被拒绝。

原环境、规划器、B2缓存等已通过且输入未变的检查可以复用，不再次做全仓库审计或大批仿真。

---

# 13. 结果到算法选择：研究决策，不是自动归因证明

所有结果完成后，生成以下字段。阈值使用§10，结论限定为本次固定协议。

| 结果模式 | 可以支持的判断 | 不得扩大成 |
|---|---|---|
| B2-RANK较M0显著方向性改善，且接近M1并保留回退/颜色能力 | 原B2加排序约束已是一条较简洁的候选方案 | 新结构在一切任务上都没价值 |
| B2-RANK接近M0；M1-GOAL在多塔明显弱于M1 | 目标无关编码在这套匹配实现中有实质作用 | 已证明唯一原因，或所有目标条件编码都不能组合 |
| B2-RANK接近M0；M1-GOAL与M1未见清楚差异 | 两个单项开关尚不足以解释整套收益 | 已排除读出、真值位、监督、容量和优化的其他解释 |
| M1-GOAL保多塔，颜色孪生配对缺口变小且必要回退不退化 | 候选的能力兼容改进；优先考虑为工作基座 | 已解决颜色鲁棒性或所有回退问题 |
| M1-GOAL修复颜色，却损失多塔 | 明确的结构取舍，有研究价值 | 直接按题型切模型就获得一般算法 |
| RED成功、BLUE失败的配对净差清楚 | 固定模型对颜色变换具有行为敏感性 | 所有旧退化都由颜色单独造成 |
| 孪生两边都失败 | 仅交换颜色不足以修好这些题，结构/覆盖等解释更重要 | 已证明颜色完全无关 |
| 学习方法仅在可不拆层改善，必要回退层无新增净收益 | 收窄为在这一范围内的组合推广；下一步关注必要回退 | 仍用多数简单题的平均分宣称更强规划能力 |
| M1+C3或G1+C3已经覆盖新模型收益 | 组合规则是必须保留的强参照；比较最优性/成本再作取舍 | 学习模块因此一定没有价值 |

最终必须分别回答：

1. 两个新条件相对各自匹配参照，有没有清楚的新增能力或退化？
2. 哪个条件在必须先退一步的题上，比B2和G1多解决了哪些具体题？
3. 与C3及两个组合规则相比，是更少绕路、更多完成，还是没有明显增量？
4. 颜色改变的配对效应是否被隔离；高度观察是否混入D*？
5. 下一版工作基座选择什么，选择理由是什么，还保留哪些未解释因素？

**M1-NLL的处理：** 若B2-RANK没有清楚收益、M1-GOAL又保留M1能力，可以建议未来补M1去排序损失的训练。但本卡只输出候选，不开始第三条。下一次若用本卡结果设计模型，需要另一套独立确认题，不能再把本卡题当未见测试。

不需要完成所有可能的消融才能形成方法贡献；也不能把相关改进中没有隔离的部分写成唯一原因。

---

# 14. 交付文件、提交与回执

建议新增薄模块，不直接改写旧卡代码定义：

```text
docs/c1_blocksworld/CP_DISR_C1_GP_Minimal_Attribution_Runbook_v2.md
src/cp_disr/blocksworld/gp_attribution.py
scripts/c1_bw_gp_attribution.py
tests/test_c1_bw_gp_attribution.py
```

这是建议接口，不宣称这些文件或命令已存在。执行者实现最小入口即可，不进行无关架构重构。

结果目录：

```text
prep/
  registration.json
  old_train_dev_destruction_labels.json
  g1_dev36.json
  paired_config_diffs.json
  initial_weight_identity.json
  dataset_identity.json
  source_identity.json
  confirm_manifest.json
  generation_shortfalls.json
runs/
  R-C1-GPA-B2-RANK-0/
  R-C1-GPA-M1-GOAL-0/
eval/
  <condition>/                 # 一次性逐题与逐步记录
results/
  final_summary.md
  claim_boundary.md
  dev_and_confirmation.csv
  by_case.csv
  by_destructive_stratum.csv
  by_n_k_h_l.csv
  colour_twin_pairs.csv
  matched_training_comparisons.csv
  g1_agreement.csv
  rule_combination_comparisons.csv
  checkpoint_identity.json
  verify.json
receipts/
  final_receipt.yaml
```

登记至少包含：源提交、用户授权文本、两条训练差异表、D_train/labels/old checkpoints/新split哈希、实际配额、颜色pair_id、规则定义、阈值和训练预算。新题标签旁置，不改历史split。

训练文件保存每epoch损失、NLL/rank分项、梯度范数、优化步数、训练样本展示数、实际参数集合和最终权重身份。中途文件只作故障定位，不用于择优。

提交顺序：登记与实现 → 两条训练及锁定checkpoint身份 → 完整评估结果。可按现有规范合并轻量提交，但登记必须先于正式训练/测试。不创建PR、不merge、不force-push、不提交.pt权重；权重和完整日志在可追踪制品位置保存哈希。

示例回执：

```yaml
card: C1-BW-GP-MINIMAL-ATTRIBUTION-V1
runbook_version: 2.0-audit-integrated
repository: rollinpioneer/graph
base_commit: b625cfc7f4c83f2c1cf054eef58538d77363c1d5
branch: codex/cp-disr-c1-bw-gp-minimal-controls-v1
registration_commit: <full_sha>
result_commit: <full_sha>
old_results_modified: false
training:
  planned: 2
  completed: <n>
  reruns: 0
  extra_data_collection: 0
  rank_zero_equivalence: <PASS/FAIL>
  goal_switch_equivalence: <PASS/FAIL>
confirmation:
  target_cases: 112
  frozen_cases: <N>
  colour_pairs: <P>
  conditions: [C0, M0, M1, B2-RANK, M1-GOAL, G1, C3, G1+C3, M1+C3]
  completed_episodes: <actual>
  technical_gaps: <actual>
  destruction_label_scope: ALL_OPTIMAL_PLANS
  unknown_labels_not_coerced: true
results:
  mono_multitower: {}
  break_multitower: {}
  height4_by_break: {}
  colour_paired_effects: {}
  g1_case_agreement: {}
  matched_pairs: {}
algorithm_recommendation: <chosen base and supported scope>
conditional_third_run_started: false
state: C1_BW_GP_MINIMAL_ATTRIBUTION_COMPLETE
next_action: WAIT_FOR_USER_MAINLINE_DECISION
```

有技术缺口则state加 `WITH_TECHNICAL_GAPS`，完整列分母，不虚构所有完成。最后清理本卡worker/monitor，取消本卡监控，不操作其他项目进程。

---

# 15. 交给执行者的简短说明

> 本段仅在用户明确授权执行本卡后适用，不因文件存在而自动获权。

```text
执行 C1-BW-GP-MINIMAL-ATTRIBUTION-V1，按本v2文档。
从 b625cfc7f4c83f2c1cf054eef58538d77363c1d5 创建独立分支。
只训练 B2-RANK、M1-GOAL 两条，各100 epoch；不起第三条。
两者都从A02 B2 final出发，各自严格匹配历史M0/M1起点和配置。
B2-RANK扩展M0的训练器，只加rank；不要借用GPTrainer改掉GRU/学习率。
M1-GOAL只恢复编码目标标记，保留M1的头、真值位、求和、损失和无时间路径。
先完成G1 dev36与必要等价fixture，再冻结新112题及九条件。
新题按最优计划是否必须拆目标分层；颜色题整对互换并整对隔离。
两条训练结束后完整执行确认，不设ID成绩门，不挑权重，不因低分重采题。
所有旧数据、结果、状态、标签和权重保持只读。
输出配对机制证据和具体算法基座建议，提交轻量结果后停止。
```

---

# 附录：引用、复核范围和可追踪来源

**附件来源**

- [A1] `CP_DISR_C1_本轮结果审查_20261006.md`，本次用户上传。重点§2.1–2.8、§5–§6。报告中的94/18分层、G1重合数及train19/dev5来自该审查，不冒称本文件独立重跑。
- [A2] `train_destruction_labels.py`，本次上传。只读旧train/dev，调用生产规划器标签；使用时需保护None。
- [A3] `confirm112_strata.py`，本次上传。只读旧逐题CSV及标签；使用时固定唯一GP_ROOT。

**仓库来源**（均固定到 `b625cfc7f4c83f2c1cf054eef58538d77363c1d5`）

- [R1] `runs/final_master/c1_route_b/blocksworld_main_v1/goal_progress_v1/train_confirm/20261006T095758Z_90758ab1/results/final_summary.md` 和 `claim_boundary.md`：旧整卡成绩与界限，已在前序审查读取。
- [R2] `src/cp_disr/blocksworld/planner.py`：本轮读取。`destruction_labels` 明确只针对所有最优计划；None处理必须保留。
- [R3] `src/cp_disr/blocksworld/goal_progress.py`：本轮读取模型、特征及评分定义。
- [R4] `src/cp_disr/blocksworld/imitation.py` 与 `scripts/c1_bw_goal_progress.py`：本轮读取M0训练器、GRU梯度、权重和启动配置。
- [R5] `src/cp_disr/blocksworld/a04p_controls.py`：本轮读取G1/C3真实选择与停止规则。
- [R6] GP_ROOT下 `runs/R-C1-GP-M0-0/training_accounting.json`：本轮读取M0参照权重和训练账目。
- [R7] GP_ROOT下 `runs/R-C1-GP-M1-0/training_accounting.json`：本轮读取M1参照权重、初始头哈希和训练账目。
- [R8] GP_ROOT下 `training_dataset_identity.json`：本轮读取D_train文件SHA、语义SHA及轨迹数。
- [R9] `src/cp_disr/blocksworld/canonical.py` 的保色同构接口及原split定义，按本次审查材料引用；执行时复用并核验，不声称本文件重新证明其实现。

阅读入口：
`https://github.com/rollinpioneer/graph/tree/b625cfc7f4c83f2c1cf054eef58538d77363c1d5`

本文件未启动服务器任务、未执行附件分析脚本、未创建Git分支或提交；本轮完成的是有来源、可执行且限制在两条训练内的实验设计。
