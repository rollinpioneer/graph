---
title: "CP-DISR C1：审计整合后的五类方法自动串行实验 Runbook"
date: "2026-10-07"
version: "3.0-audit-integrated-full-serial"
card: "C1-BW-METHOD-SERIAL-SUITE-V3"
status: "READY_FOR_AGENT_IMPLEMENTATION"
execution_authorized_in_this_response: false
activation: "用户后续明确授权执行本文件后，释放完整串行队列；阶段间不重复请求批准"
repository: "rollinpioneer/graph"
base_branch: "codex/cp-disr-c1-bw-scale-order-prototype-v1"
base_commit: "2d008431eb7fb069e617e4a02843bef84c6117d1"
proposed_branch: "codex/cp-disr-c1-bw-method-serial-v3"
server_storage_owner: "xushijie3"
legacy_storage_owner: "xushijie2"
execution_mode: "FULL_SERIAL"
max_active_training_runs: 1
max_new_training_runs: 9
automatic_height4_training: false
automatic_external_training: false
automatic_second_domain: false
automatic_extra_seeds: false
---

# CP-DISR C1：五类方法自动串行实验 Runbook v3

## 0. 本文件作出的决定

**研究方向固定：已知动作合同、状态完全可见，使用受限目标结构训练，检验复杂目标组合和准备步骤上的迁移。**

采用用户提供的 Claude 审计中的主要修正：

1. **先运行零训练的两步前瞻**，不再等子目标训练失败才做；
2. 子目标组增加 **SG-FACT**，区分“下一事件监督”和“事件—动作配对信息”；
3. 明确所有主方法的训练目标塔高只有 **2、3**，高4评价是**目标塔高外推**；
4. 价值组改成“**绝对代价校准 vs 相对代价校准**”，两者都用可学习的正温度解耦动作置信度与步数尺度；
5. 循环推理组把**同一检查点的4轮与8轮比较**列为正式结果；
6. 目标交互组保留强稠密控制，并披露跨塔关系屏蔽可能只是分布外抑制；
7. 所有方法使用共同开发材料，全部训练与选模结束后才生成**同一份独立确认集**；
8. 工作区、数据副本、权重、日志、缓存、临时文件的新写入统一迁到 **xushijie3**。

### 0.1 自动执行的含义与预算

本版按用户“自动接续串行执行这些方法，最后给出完整结论”的要求，采用 **五类方法全覆盖**，不是只执行一个条件分支：

```text
两步前瞻：0条训练
子目标组：BASE / FACT / JOINT，共3条
价值组：ABS / REL，共2条
循环推理组：SELF / REL，共2条
目标交互组：DENSE / REL，共2条
正常合计：9条训练；每条1个种子，100个epoch
```

Claude 建议中的“约5条”对应“3条子目标＋价值或循环中的一组”，**没有覆盖另外两类方法**。本文件没有把9条隐瞒成5条。

两步前瞻的结果只决定**价值组与循环组谁先做**，不决定另一个是否被永久跳过。即使某组低分，也完成剩余独立组。某些训练若在数学上完全等价，可按后文确定规则复用并减少实际运行；不得另加实验填满额度。

本版**不自动加入**：高4训练诊断、首个On事件的新训练变体、KR训练、第二领域、额外种子、组合五模块的“全家桶”。

### 0.2 本轮与服务器执行的边界

本文件及附带代码是计划和数学/调度参考，**不是已完成的服务器实现与结果**。本次回复未进入服务器、未移动服务器文件、未训练模型、未创建Git分支。

用户后续明确说“执行本runbook”后，Agent可从准备到最终报告自动完成整个已列范围，**不因方法分数低而停下来等待选择，也不需要逐阶段再次授权**。

不得利用本版继续重跑已有完全等价实验。启动时只核对一次本地/远端同等产物；现有代码、权重与已完成记录应复用。

---

## 1. 审计意见裁决：采纳什么，修正什么

以下区分：**[A] 用户粘贴的Claude审计；[R] 本轮读取的固定仓库；[P] 旧方案；[D] 本版设计或数学推导；[W] 外部原始材料。**

| 审计意见 | 本版处理 |
|---|---|
| 训练目标高度≤3，高4是外推 | **采纳。** 原生成器 `TRAIN_SHAPES` 支持这一点。需逐题验证train/dev清单并输出goal-height统计。不是声称训练的初始状态绝无高4结构。 |
| 原三个瓶颈忽略了高度外推 | **采纳并收窄。** H1/H2是解释高度外推失败的候选机制；高度外推是任务条件，不与它们构成已经证明的互斥原因。 |
| 把Board48、旧112、状态库合并看 | **采纳综合判断，不直接拼分母。** 三者分别是选模、开发支持和局部检查；不能把240状态当240道独立题与160局相加。 |
| 两步前瞻应前置 | **采纳。** 它零训练但不是零计算；可以产生方法排序信息。 |
| 神经叶值不如目标计数就说明跨状态不可比 | **只作支持线索，不作根因定论。** 也可能来自叶状态分布、同值选择、短视、控制历史和更好的简单启发。 |
| 增加SG-FACT | **采纳并精确定义。** FACT只用Y的两个投影构造笛卡尔标签，使用与JOINT同型损失，以免改变多项优化权重。不是“两个独立网络”的额外结构变体。 |
| 检查不同最优动作的首个事件集合 | **采纳。** 真正配对信息判断是 `Y != projection_goal(Y) × A*`，不只是 `Y != U0 × A*`。 |
| 首个事件可能大量为OnTable | **采纳统计与语义披露。** 本轮不看比例后自动把主标签改成首个On；否则又变成一项新方法选择。 |
| SG-BASE新头可能没有梯度 | **记录，不预判为必然。** 近零动作损失使它风险较高；保存新头梯度和参数变化，不能把低信号冒充实现错误。 |
| 四层已覆盖整张图 | **不采用这个强说法。** 旧测量针对候选到最近未满足目标，不能推出全部节点/全部目标关系已充分覆盖或计算完毕。 |
| REC测试4→8轮应是主检验 | **采纳。** 同一T4选定检查点同时评价T4/T8，SELF也评价两档，不只给主方法增加计算。 |
| 加绝对锚与相对差约束有冗余 | **采纳。** 精确拟合全部绝对距离时，相对差自动成立；原方案最多能解释优化组织，不能宣称新增独立信息。 |
| 价值尺度与动作softmax冲突 | **采纳。** 两个价值条件共同加入正温度β与初始仿射重标定，保证初始动作分布保留。 |
| REL注意力胜过DENSE可能是跨塔屏蔽 | **采纳作为竞争解释。** mask密度是描述证据，单靠密度仍不能区分唯一原因。 |
| 可选高4训练诊断 | **不默认执行。** 本轮不再增加数据覆盖实验；不能把这条没做说成已排除所有高度因素。 |

### 1.1 需要明示的一处理论边界

对单位代价、合法可逆动作且“手空/持有”交替的当前Blocksworld，有：

```text
d*(T(s,a),G) ∈ {d*(s,G)-1, d*(s,G)+1}
```

前提是距离精确、目标/状态合法。这是当前领域性质，不是所有规划域的性质。错误动作的**局部最优后续超额代价**为2，不代表整条实际失败轨迹只多2步。

如果把状态值精确锚定为d*，最优和非最优的基础差分分数约为+1与−1；softmax置信度因合法动作数量而受限。正温度β允许：

```text
代价模型以“步数”为单位；
动作分布仍可有足够置信度。
```

本版不引用“中位logit差约54”为已核验事实；该数来自审计，Agent可在训练数据中顺手测量，不能据此修改测试输入。

---

## 2. 当前基线、训练分布与外推分层

### 2.1 固定仓库

```text
repository: rollinpioneer/graph
base_branch: codex/cp-disr-c1-bw-scale-order-prototype-v1
base_commit: 2d008431eb7fb069e617e4a02843bef84c6117d1
new_branch: codex/cp-disr-c1-bw-method-serial-v3
```

本轮读取的远端 `codex/cp-disr-c1-bw-*` 引用仍包含上述结果，无本轮套件结果。不能仅据此断言服务器没有未推送工作。

### 2.2 工作权重与训练数据

工作权重为原 **M1-GOAL**，不是PAD：

```text
runs/final_master/c1_route_b/blocksworld_main_v1/
gp_minimal_attribution_v1/20261006T143512Z_3cb37301/
runs/R-C1-GPA-M1-GOAL-0/checkpoints/final.pt

file_sha256:
b04b32daa53553dd1632e836fb881df2e3a86181ebba95a80157b56d2c1d145f
bytes: 17880803
```

训练数据：

```text
runs/final_master/c1_route_b/blocksworld_main_v1/
goal_progress_v1/train_confirm/20261006T095758Z_90758ab1/
datasets/D_train.json

file_sha256:
387e66309c51195ad017b2b90e9f2742090457d108a353f34159acd8f496bf78

1296 trajectories / 7446 decision occurrences
```

排序标签从该数据的原清单读取：

```text
expected file_sha256:
5486cea84eeca5efa8cbd73722cd34aa2609b44a369b95f398b9d3cdd5be50f6
```

不可把“文件SHA”和“权重张量摘要”混为同一个值。B2参照使用原加载器/权重清单，文件绝对路径在新服务器解析，不猜测。

### 2.3 必须在所有结果表出现的训练限制

```yaml
train_object_count: [3,4,5]
train_goal_tower_height: [2,3]
train_nontrivial_goal_towers: 1
pad_training_used: false
height4_training_used: false
```

- `goal_height`：最终目标的最高塔；
- `initial_height`：初始物理摆放最高塔；
- `goal_towers`：最终目标非平凡塔数；
- `height_seen`：目标塔高属于{2,3}；
- `height_unseen`：目标塔高为4；
- `structure_seen`：单座非平凡目标塔；
- `structure_unseen`：多座非平凡目标塔。

**高度在训练范围内不等于整个问题分布内。**n=8、高2仍然是对象规模外推；颜色或塔数也可能未见。

本版沿用数据逐题计算这些字段，不为获得某标签重采样、不新增高度4梯度数据。

### 2.4 已有结果的用途

已有开发证据：

```text
Board48：MG 47成功/30全程最优；PAD同样47/30
MG高2最优21/24，高4最优9/24
旧scorer-control112：MG+C3为108/82，B2+G1+C3为109/88
公共240状态：MG首选最优229，纯准备100/106
```

这些不是未来新模型的结果。本版不再重做旧错误分类，只复用字段和脚本；必要缺项在本轮轨迹中补齐。

---

## 3. xushijie3 路径迁移与安全边界

### 3.1 不猜测绝对挂载点

用户指定的是 `xushijie2 → xushijie3`，未提供完整绝对前缀。本文件**不假定它一定是 `/home/xushijie3`、`/data/xushijie3`，也不把目录所有者名称擅自当作SSH主机名**。

Agent在已授权的服务器会话中检查现有挂载、工作区配置和新目录，然后一次性写入：

```yaml
# suite_runtime.yaml — 以下是字段，不是已核验路径
storage_root: <已存在且可写、归属xushijie3的绝对路径>
repo_root: <storage_root下的当前新worktree>
asset_root: <storage_root下的旧资产只读副本或已迁移资产目录>
run_root: <repo_root>/runs/final_master/c1_route_b/blocksworld_main_v1/method_serial_v3/<run_id>
cache_root: <storage_root>/cp_disr_cache
tmp_root: <storage_root>/cp_disr_tmp
venv_python: <实际验证的Python可执行文件>
device: <一张实际可用的GPU>
```

如果存在多个可写候选，以用户已配置工作区为先；否则按能找到完整已验证资产且不与其他实验冲突的目录选择并记录。若仍无可验证目标目录，记录 `BLOCKED_PATH_RESOLUTION`，输出已有准备报告；**不在xushijie2悄悄启动训练，也不全盘复制他人的目录**。

### 3.2 迁移规则

1. 优先读取xushijie3已迁移的旧资产，按相对路径和SHA解析。
2. 缺失时只复制本卡所需的MG、B2、D_train、rank标签、开发题和必要日志；旧目录只读。
3. 不使用 `sed` 全局替换历史JSON/MD内容。旧清单中的绝对路径作为来源记录保留，通过 `path_map.json` 映射。
4. 复制前核对剩余空间；复制至临时文件，校验SHA后原子改名。不得移动/删除旧资产。
5. 所有新模型输出、缓存和临时文件必须在xushijie3。使用 `Path.resolve()` 核对，防止符号链接实际回写xushijie2。
6. 本进程设置 `TMPDIR`、`XDG_CACHE_HOME`、`TORCH_HOME` 等至本卡cache/tmp；不更改全服务器配置。
7. SSH账号、主机别名和已有远程连接权限保持原样；本卡只是存储路径迁移，没有改账户或创建远程账号的授权。

### 3.3 运行目录结构

```text
method_serial_v3/<UTC>_<code8>/
  plan/
  assets/
    path_map.json
    asset_identity.json
  prep/
  stages/
    lookahead/
    next_event/
    calibration/
    recurrent/
    goal_attention/
  runs/<condition>/
  development/
  confirmation/
  results/
  receipts/
```

新分支只从固定基线创建。旧根目录与旧权重保持不变。不提交.pt、优化器状态或大型缓存到普通Git。

---

## 4. 自动串行状态机与固定运行额度

### 4.1 队列

```text
PATH_AND_ASSET_PREFLIGHT
  ↓
MINIMAL_FIXTURES_AND_DATA_PROFILE
  ↓
LOOKAHEAD_DEV                       # 0训练；MG/COUNT两种叶值
  ↓
NEXT_EVENT_LABELS                   # 只在原训练题生成
  ↓
SG_BASE → SG_FACT → SG_JOINT         # 严格串行，最多3条
  ↓
按lookahead读数决定顺序：
  CAL_ABS → CAL_REL → REC_SELF → REC_REL
  或
  REC_SELF → REC_REL → CAL_ABS → CAL_REL
  ↓
GOAL_DENSE → GOAL_REL               # 2条
  ↓
LOCK_ALL_SELECTED_CHECKPOINTS
  ↓
GENERATE_ONE_CONFIRM_SET
  ↓
CONFIRM_ALL_ELIGIBLE_CONDITIONS      # 串行评价
  ↓
FINAL_TABLES_AND_CLAIM_BOUNDARY
  ↓
SIZE_CHECK → LIGHT_COMMIT/PUSH → CLEANUP
```

每条训练完成后立即进行固定的开发评价与选模，然后自动进入下一条。不等待用户选择分支；低分不终止队列。

**同一时刻最多1个训练进程，默认训练与GPU评价也不并行。**标签生成可以在一个阶段内部使用CPU进程池，默认不超过48进程且不超过系统允许资源；不额外启动另一方法的训练。

### 4.2 两步前瞻只改变先后次序

若同树神经叶值在Board48与旧112的高4开发面板上，都没有比目标计数更好的完成数，且计罚成本不低，并在共同根状态的跨根候选选择中更差，则：

```text
next_order = CALIBRATION_THEN_RECURRENT
reason = LEAF_VALUE_WEAKNESS_COMPATIBLE_WITH_CALIBRATION_HYPOTHESIS
```

其余情况：

```text
next_order = RECURRENT_THEN_CALIBRATION
```

材料不足时使用固定回退顺序 `CALIBRATION_THEN_RECURRENT`，记录缺项。该判断**不是根因证明或显著性判据**；全覆盖模式下，两组最终都会完成。

### 4.3 完成状态、数学等价和局部失败

每个stage必须以结构化receipt结束：

```text
DONE
REUSED_EXACT
MATHEMATICALLY_EQUIVALENT_REUSED
UNINFORMATIVE_LABELS
TECHNICAL_INCOMPLETE
BLOCKED_DEPENDENCY
```

- **低成功率：DONE**，不能写成技术失败。
- 同结构、同初值、同数据、同损失完全一致时，可复用一次训练。只需新数学fixture证明，不根据相似分数宣称等价。
- SG某个标签不可用，不阻塞价值、循环或注意力组。
- 某个模型NaN/实现失败时记录该方法 `TECHNICAL_INCOMPLETE`，完成能够独立进行的其他方法，最后明确哪项比较无法回答。
- 公共权重校验失败、训练/确认泄漏、无法解析xushijie3或资源权限问题属于**全局阻断**，停止计算并照实输出不完整回执。
- Agent可以在训练前修正新增代码的纯技术bug并重跑fixture；不能看到结果后改损失、目标标签或网络定义却沿用同一condition名。

### 4.4 中断恢复，不从头重刷

保存 `resume_last.pt`：模型、优化器、随机数状态、epoch、batch位置、数据顺序、累计优化步数、源代码/数据hash。采用临时文件＋原子替换。

硬件中断最多允许每条2次同配置恢复；必须从保存状态继续。若仅有epoch边界快照，保留未完成epoch的尝试日志，回退到边界并披露重复计算，不伪称物理计算严格一次。

**没有可用恢复点时，不自动从头开始新seed或新训练。**记录局部不完整，继续其他独立方法。

正常完成的模型不因低分恢复、迁移、延长或重新初始化。CPU标签缓存和已经完成的评价按条件/数据/权重hash复用。

### 4.5 执行授权的写法

准备阶段生成：

```json
{
  "authorization_source": "用户后续明确执行本runbook的消息",
  "scope": "FULL_SERIAL_5_FAMILIES_MAX_9_TRAININGS_AND_ONE_COMMON_CONFIRM",
  "requires_intermediate_confirmation": false
}
```

只有这条授权成立后，才进入完整计算。不能因为旧卡曾获授权，就悄悄扩成9条。本次只是生成runbook。



## 5. 所有训练共享的开发与评价协议

### 5.1 模型起点与预算

所有学习候选均从**同一个原M1-GOAL最终检查点**初始化，不从前一个候选的训练结果接着练，避免把串行执行变成模块叠加。

```yaml
train_seed: 0
new_module_seed: 20261007
optimizer: Adam
optimizer_reset: true
existing_encoder_and_heads_lr: 0.0001
new_module_lr: 0.0003
betas: [0.9, 0.999]
eps: 1.0e-8
weight_decay: 0
grad_clip: 0.5
rank_weight: 1.0
epochs: 100
trajectory_batch_size: 32
save_epochs: [20,40,60,80,100]
resume_checkpoint: every_epoch_and_clean_shutdown
```

每条仍有1296个轨迹槽，41批/epoch，预期4100次更新、744600个原决策展示。端点值编码、循环中间读出、目标—动作张量的额外计算**另行报告**，不能说全部严格等算力。

既有有效编码器、phi/rho共同微调；未使用的B2时间、历史GRU、旧Q/V头不启用。新模块采用保持初始动作分布的零残差或仿射初始化。

### 5.2 三个开发面板不直接拼分母

**D-SELECT：Board48。**每个学习条件的5个checkpoint都在此选模。选择键固定：

1. 总成功数最大；
2. 高4（未见目标高度）全程最优数最大；
3. 全部题失败计罚长度比均值最小；
4. 全部题全程最优数最大；
5. 较早epoch。

失败计罚长度比：成功为steps/L*；失败为(cap+1)/L*；L*=0单列。空候选提前失败不能获得“省动作”的优势。

**D-SUPPORT：旧scorer-control112。**每个条件只对选定checkpoint评价一次。高4、必要回退、颜色层逐题报告；**不据此回头换checkpoint、调参或追加训练**。可以改变后续方法的执行顺序，但本版已冻结方法定义。

**D-STATE：旧240公共状态库。**只对选定checkpoint评价。该库近饱和，主要用于能力保持、局部纠错和新增错误检查；不把提高1个状态解释为解决深层推理。

此外，原dev36对每个选定模型评价一次，只查分布内能力，不作为资格门或事后挑点集合。

来源不同的结果分别展示。若绘制“开发综合表”，使用面板等权指标并标明是启发式汇总，不能把来自同题的状态和整局记录当成独立样本。

### 5.3 公平选模与既有基线

- 每个新条件有相同5个checkpoint机会；同家族不得单独多扫参数或温度。
- 原MG和B2使用既定已部署checkpoint，作为冻结参照；不把其与额外微调模型的差异宣称为纯结构因果。
- 关键机制结论来自新家族内部匹配条件。
- 所有家族使用相同开发集合，开发比较存在选模偏好；最终共同确认才用于核对方向。
- 本轮不自动恢复额外的“最佳checkpoint扫描”、新seed或GPU从头重训。

### 5.4 一次性的最终确认

五类方法开发全部结束、每个checkpoint和REC的T4/T8评价方式锁定后，生成**一套新的128题**：

```text
n ∈ {6,8}
非平凡目标塔数 k ∈ {1,2}
目标最高塔高 h ∈ {2,4}
8格，每格目标16题
```

h2是高度在训练范围内的测试，h4是目标塔高外推。两者都可能同时包含对象数、塔数外推，不能叫成完全ID与完全OOD的单变量比较。

沿用标准合同和生成器，不增加隐藏动力学；目标形状与原Board48一致但问题实例新。排除所有训练、开发、历史确认问题及其仅改名副本；颜色成对结构若自然存在，按骨架聚合。生成不看任何模型成绩。

每格固定最多2000候选尝试，不足则所有方法用同一实际题集，披露缺额；不因少一题停止全套。记录最短长度、初始高度、初始已满足目标数及全部最优方案是否需要拆目标。UNKNOWN保持UNKNOWN。

**本版为了给出完整方法比较，不再以“开发分数够不够好”决定是否生成确认集。**全部开发终止后统一确认所有技术有效条件；失败方法也保留在表中。确认后不得再训练、重选checkpoint或改变规则。

### 5.5 统一执行控制

- 主要学习模型都用原C3。
- B2+G1+C3仍为强系统参照。
- 不给新模型悄悄加G1，不改变C3空候选终止。
- 根动作预算仍为 `2L*+4`。
- 精确规划器只给标签与参照，不能在策略执行中替模型选动作。
- 两步前瞻明确增加合同展开权限，和同树计数控制比较，不能归为同推理成本。
- 每步同时记录模型raw选择、wrapper选择和C3最终选择；多阶段方法保留每次改写原因。

---

## 6. 方法A：两步合同前瞻——首先运行，零训练

### 6.1 条件

```text
MG_C3               原MG一步评分，冻结参照
B_G1C3              原最佳完整执行系统，冻结参照
LOOK2_MG            相同深度2树，MG状态值评叶
LOOK2_COUNT         相同深度2树，未满足最终目标数评叶
```

两个lookahead条件的根集合、合同、终止处理、去重、tie-break完全一致；只有非终止叶评分不同。

### 6.2 前向定义

对根状态s，先按真实历史C3取得根候选。每个a产生s_a，再展开全部合法第二动作b，产生s_ab。

- 过滤s_ab属于真实访问集或当前小路径的状态；不把假想状态写进真实历史。
- 一步已成功优先于两步成功；有成功分支时按步数最短、固定动作ID破并列。
- 无成功分支时，所有被比较叶深度严格为2。
- 每个根的分数为该根下最优叶分数；若该根没有叶但其他根有，局部忽略无叶根。
- 所有根无叶时退回原MG+C3，不伪称全局无解。
- 根无可用动作仍按原C3终止。
- 只执行第一步，下一步重新展开。

```python
roots = c3_available(s, real_visited)
if not roots:
    return terminate("NO_UNVISITED_SUCCESSOR")

terminals, leaves = [], {}
for a in roots:
    sa = apply(s, a)
    if goal_complete(sa):
        terminals.append((1, a))
        continue
    leaves[a] = []
    for b in legal(sa):
        sab = apply(sa, b)
        if sab in real_visited or sab in {s, sa}:
            continue
        if goal_complete(sab):
            terminals.append((2, a))
        else:
            leaves[a].append(sab)
if terminals:
    return lexicographic_min_cost_root(terminals)
if any(leaves.values()):
    # MG: V(sab,G); COUNT: number of currently false final goal atoms
    return argmin_root_best_leaf(leaves, leaf_score, fixed_tie_break)
return original_mg_c3(s, real_visited)
```

在当前四算子积木中，从同一根走两步后，非终止叶的手部阶段相同，减少了一个“持有/手空”因素。但它不保证V跨父节点已经校准。

### 6.3 “同一棵树”必须准确解释

在**同一状态和同一真实访问历史**下，两种叶评分的候选树完全相同。两控制器完整执行后会走向不同状态、产生不同历史，因此不能宣称两条完整轨迹中每一步的树也相同。

额外使用已有公共状态库，从固定 `visited={s}` 构造共享树，离线计算：
- 神经/计数叶评分各自选择的根动作；
- 完整最优根动作；
- 跨根叶排序错误与根动作超额代价；
- 终止叶是否存在、并列率、根和叶的数量。

这些规划器标签仅用于结果表，不进入选择函数。

### 6.4 解释与自动排序

- LOOK2_MG超过一步MG、也超过同树COUNT：支持学得叶评分在固定额外计算下有用。
- 两者同样改善：优先解释为前瞻/合同展开有用，未证明神经叶值增量。
- COUNT优于MG：与校准不足相容，但也可能是更好的任务启发、叶分布或tie-break。触发价值组优先，**不输出“校准已证明”**。
- 两者失败：不证明所有前瞻无效；本版只检验深度2。
- 无论结果高低，自动继续子目标组。

记录每步合同展开、唯一叶数、逻辑图数、batch调用、推理时间；不将“零训练”写成“免费实验”。

---

## 7. 方法B：下一事件组织——BASE / FACT / JOINT

### 7.1 标签语义保持一致

根状态s的U0为当前未满足的最终目标原子。对每条最优续接，记录：
- 第一动作a；
- 第一次使U0中原子成立的事件g；
- 多解得到所有合法(g,a)，形成Y*。

它仍包含OnTable目标；不声称每个事件都在为“搭某座塔”服务。已满足后拆掉再恢复的目标不算根U0新事件。每步重新判断，完整目标始终可见。

### 7.2 信息量统计——在标签准备时一次完成

对完整标签定义：

```text
A = 全部最优第一动作
Q = projection_goal(Y)
E_a = {g : (g,a) in Y}
Y_fact = Q × A
```

输出**独立goal-state对口径**及**轨迹加权口径**：
1. 完整/UNKNOWN标签数；
2. |U0|、|A|、|Q|、|Y|分布；
3. 事件选择信息比例：`Q != U0`；
4. 配对信息比例：`Y != Q × A`；
5. 不同最优动作有不同事件集合：存在a,b使E_a≠E_b；
6. 单事件矩形 `Y={g*}×A` 的数量；
7. |Y|/(|Q||A|)及事件集合Jaccard分布；
8. OnTable/On占全部合法事件对的比例；
9. 仅含OnTable事件、仅含On事件和混合事件的根状态比例；
10. 标签首次达成步数与训练目标塔高、目标塔数组合。

对于完整标签，`projection_action(Y)==A`必须成立。节点限制导致任一最优续接分支未完成，整根标UNKNOWN，不用部分答案当负例。

**“Y不是U0×A”不是充分的配对信息判据。**例如Y={一个目标}×A只有事件选择信息，没有动作特异的配对监督。

不根据OnTable占比自动改变事件语义。只可零训练输出替代事件定义的覆盖统计；不得加入On-only模型，或在看开发成绩后选择更好事件定义。

### 7.3 模型不变，比较标签的细度

```text
当前/所有一阶后继编码
→ 386维目标特征
→ 原phi给每目标z (128维)
→ 保留原MG分数b
→ q(g|s,G) 与目标条件残差delta(g,a)
→ p(a|g,s,G)
→ 对g边缘化得到p(a|s,G)
→ 原C3
```

公式：

\[
q_g=\operatorname{softmax}_{g\in U_0}\mathrm{Gate}([z_g(s),\bar z(s,G)]),
\]

\[
\delta_{ga}=\mathrm{Head}([z_g(s),z_g(s_a),z_g(s_a)-z_g(s),t_g(s),t_g(s_a)]),
\]

\[
p(a|g)=\operatorname{softmax}_a[b_a+\delta_{ga}],\quad
p(a)=\sum_g q_gp(a|g).
\]

Gate 256→128→1；Head 386→128→1；两个最后层零初始化，其余共享种子。三条件有效参数、初值和数据完全匹配，名义新增82690参数，实际计数另报。

这是**目标—动作联合概率模型**，最终动作分数一般不再来自一个公共状态势函数。C3保留，但不提供最优性或完整性保证。

### 7.4 三个损失——FACT采用“投影标签笛卡尔松弛”

令：

\[
P_A=\sum_{a\in A}p(a),\quad
P_F=\sum_{(g,a)\in Q\times A}q_gp(a|g),\quad
P_Y=\sum_{(g,a)\in Y}q_gp(a|g).
\]

\[
L_\mathrm{BASE}=-\log P_A+L_\mathrm{rank}
\]
\[
L_\mathrm{FACT}=-\log P_F+L_\mathrm{rank}
\]
\[
L_\mathrm{JOINT}=-\log P_Y+L_\mathrm{rank}.
\]

等价地，两种辅助组织损失分别是 `log P_A - log P_F` 与 `log P_A - log P_Y`，共同权重1。

**本版对审计文字的具体化：**FACT没有使用a↔g配对关系，只使用Q和A两个投影；但保留与JOINT同一模型及同型集合概率损失。没有额外叠加两份NLL，使“配对 vs 无配对”的核心区别更干净。FACT并非强制模型概率完全独立，而是**训练标签不包含配对**。

- JOINT>BASE：更多事件相关监督有用，尚不能证明配对必要。
- JOINT>FACT：在相同架构下，配对信息比仅投影信息有用。
- FACT≈JOINT且都>BASE：优先采用简单事件组织解释。
- BASE梯度小：如实报告训练条件，不以此否定对照；首批/每保存epoch记新头梯度范数和参数变化。

计算用logsumexp，非法动作列先安全处理，UNKNOWN以U0×A作占位让组织损失为零，不能对空集合logsumexp后乘0。rank作用在边缘log概率，与原合法候选标签一致。

### 7.5 数学等价时省运行，不制造区分

若全部完整标签满足：
- Q==U0：FACT与BASE同损失，可训练BASE一次并作为FACT精确别名；
- Y==Q×A：JOINT与FACT同损失，可复用FACT；
- 两者都成立：三者只需一条BASE训练，标注其他为等价复用。

必须在训练前用标签清单和数学fixture确定，不能因为结果接近而少算。

有非零但极少配对信息时正常完成，报告比例；不加大难样本权重、不添加两塔训练。主张必须限于真正有信息的层，且不能因事后选择这些层而夸大总体效果。

### 7.6 训练与主要对照

```text
R-C1-SERIAL-SG-BASE-0
R-C1-SERIAL-SG-FACT-0
R-C1-SERIAL-SG-JOINT-0
```

每条使用§5共同预算，原D_train不变；新增标签的最优续接状态只参与标注，不自动加入梯度状态。三条串行。

核心困难层：
- 未见目标高度h4；
- 多塔目标，即训练未出现的塔间选择；
- 中性准备和必要拆除状态；
- 相同状态下合法联合预测；
- 实际绕路与新增错误。

不写“已经学到塔间协调”：训练只有单塔，事件组织从塔内到塔间迁移本来就是被测假说。



## 8. 方法C：绝对代价校准 vs 相对代价校准

### 8.1 撤销旧“绝对锚＋再加一致性”的主对照

若e(t)=V(t)−d*(t)，最优长度k片段满足：

```text
[V(s)-V(t)]-k = e(s)-e(t)
```

精确绝对拟合已经隐含差分一致。因此旧组合不是新增独立标签信息，只改变误差组织/权重。**本版不用这一对照来主张新推理机制。**

改为同架构、同状态、同标签来源的两种校准目标：
- CAL-ABS：用绝对剩余代价；
- CAL-REL：只用同一问题中状态对的代价差，不同时加绝对锚。

两者都是成熟监督方式的匹配比较，不声称首创价值函数。

### 8.2 温度与仿射初始化，保留原动作分布

使用：

\[
W_\theta(t,G)=a_0V_\theta(t,G)+c_0,\quad a_0>0,
\]
\[
b_a=\beta\,[W_\theta(s,G)-W_\theta(s_a,G)],\quad \beta=\exp(\eta)>0.
\]

a0、c0是**仅从原训练状态估计并固定**的共同初始化常数，β为两条件共同新增的可学习标量。

确定方式：
1. 在原训练数据的严格好坏候选对上取正间隔 `V(worse)-V(better)` 的中位m；
2. 若无有效m，使用m=2并标记回退；否则设a0=clip(2/m, 1e-3, 1e3)；
3. c0为训练状态上 `d*(t)-a0 V0(t,G)` 的加权中位数；
4. 设β0=1/a0。

于是初始时：

```text
beta0 * [W0(s)-W0(sa)] == V0(s)-V0(sa)
```

动作概率与原MG一致，数值容差内验证。c0不是允许模型看测试距离；测试无标签输入。

训练η学习率1e-4。为数值保护，更新后η投影到[-9,9]，两条件相同；记录触边次数。若长期触边，报告校准/分类仍不兼容，不扩大范围重训。

不能直接把V压到步数单位却仍以温度1要求近零动作损失；也不能单独只给主方法温度。

### 8.3 共同标签池和数据边界

对原训练根状态s：
- 保留已有全部合法后继及其精确距离；
- 端点池Z包含s、合法一阶后继，以及原记录轨迹中合法最优连续片段的2/4步端点；
- 校准状态对P包含所有(s,sa)合法转移，以及可验证的2/4步最优片段；
- 边目标统一为 `d*(s)-d*(t)`，错误转移可以为负；不能对所有边强制正进展；
- 两条件读取相同Z、P以及同一距离缓存；训练时ABS用Z的绝对值，REL用P的差值；
- 不把端点自动升级成新的动作监督根，不加入开发/确认状态；
- 计算超限标UNKNOWN并同样mask；报告覆盖、端点前向数及额外标签CPU时间。

c为Z内非零训练距离的中位数，至少1，只用于归一化校准损失。每根的节点/边先取均值，再按原轨迹权重聚合，防止分支多的状态获得更大总权重。

### 8.4 损失

\[
L_\mathrm{ABS}
=
L_{A^*}(\beta\Delta W)+L_\mathrm{rank}(\beta\Delta W)
+0.1\,\mathrm{mean}_{t\in Z}\mathrm{Huber}\left(\frac{W(t)-d^*(t)}c\right),
\]

\[
L_\mathrm{REL}
=
L_{A^*}(\beta\Delta W)+L_\mathrm{rank}(\beta\Delta W)
+0.1\,\mathrm{mean}_{(s,t)\in P}\mathrm{Huber}\left(\frac{W(s)-W(t)-[d^*(s)-d^*(t)]}c\right).
\]

REL不加绝对距离锚；同一目标下整体加常数不改变其相对损失和动作排序。它仍使用规划器代价信息，不能称为无监督。

两者端点编码池相同，输出位置/训练状态相同，但损失涉及节点或边的差异是设计变量。REL约束同样可以由真实绝对距离推导，**不主张凭空增加了新信息**；只检验相对组织是否更适合泛化。

不在本卡追加Bellman控制或额外稳定项。若需要，那是以后另一项方法，而不是看到分数后修改这两条。

### 8.5 训练与评价

```text
R-C1-SERIAL-CAL-ABS-0
R-C1-SERIAL-CAL-REL-0
```

各100epoch，分别从同MG起点，温度/仿射常数相同。按**一步C3执行**的统一开发规则选checkpoint，不能给其中一个按两步效果专门选点。

每个选定checkpoint均报告：
- 一步C3；
- 同一冻结的两步前瞻执行；
- 训练内与训练外高度的值误差、相对排序、跨父叶排名；
- β与间隔分布；
- 任务与成本。

两步执行使用W评叶，正温度不改变叶排序。无须把终止V强制设0来混合深度，仍用方法A的显式终止优先。

### 8.6 结果解释

- 只值误差更好、动作/任务不变：不主张性能机制。
- ABS与REL都改善：校准族有用，相对约束未必必要。
- REL超过ABS：相对代价组织在本条件下更好，不说明所有绝对价值学习不适用。
- 只有两步而非一步改善：说明更适合被用于该前瞻，收益包含额外推理。
- 失败：不再追加温度范围、损失权重、步数跨度扫参；继续下一家族。

---

## 9. 方法D：目标锚定的反复关系推理

### 9.1 架构

原MG的四层编码得到H0。新增共享处理器：

\[
m_v^t=\sum_r \mathrm{mean}_{u\in N_r(v)}W_r h_u^t,
\]
\[
h_v^{t+1}=\mathrm{GRU}([H_v^0,m_v^t,\mathrm{goalmark}_v],h_v^t),
\]
\[
\widetilde H_v^T=H_v^0+W_o(h_v^T-H_v^0).
\]

h0=H0，Wo零初始化；每轮重注入原输入与目标标记。更新后的H用于同样的目标特征、phi/rho和状态评分。每个候选通过自身后继图计算同一状态函数，不加执行搜索或轨迹记忆。

两个条件：
- REC-REL：真实类型边的邻居消息；
- REC-SELF：同样Wr/GRU/更新次数，但每类存在的邻居槽使用自身h_v的变换。保留该节点的关系类型存在mask，参数必须实际参与前向，不增加空参数占位。

该比较识别的是**反复运算中的新增关系信息**，并不单独识别“共享参数”相对于“不共享参数”的贡献。

### 9.2 训练深度与稳定性

主训练最大T=4。两条件使用相同的：

```text
L_rec = 0.5 * (L_action+L_rank at T=2)
      + 0.5 * (L_action+L_rank at T=4)
```

一次展开到4同时读取2和4，**T=8不参与训练**。中间监督是本版预先固定的稳定性调整，两条件共同使用；额外读出成本单列。

输入重注入和中间监督不等于完整复现Deep Thinking的渐进训练。作者原方法还专门处理过度迭代退化，因此本原型仍可能在8轮失效，不把失败推成“反复推理路线不成立”。[W1]

### 9.3 开发与正式主比较

```text
R-C1-SERIAL-REC-SELF-0
R-C1-SERIAL-REC-REL-0
```

各100epoch。5个开发checkpoint只在T=4下选择。**同一选定checkpoint**在所有开发支持与正式确认题上同时评价T=4和T=8，SELF与REL都评两档。

核心结果包括：
1. REL4 vs SELF4：新增关系传播的作用；
2. REL8 vs REL4：不再训练、增加计算是否帮助高4外推；
3. SELF8 vs SELF4：额外迭代本身的控制；
4. `(REL8−REL4)−(SELF8−SELF4)`：描述性差中差，按成功/最优/成本分别报告，不当成因果证明。

如果REL4优于SELF4但REL8不再提高，可保留“新增关系计算有效”，**不可主张测试时计算外推**。如果只有增加到8轮才有效，必须与SELF8和真实时间成本比较，不能与旧4轮模型作伪等算力比较。

仍无共享/不共享参数对照，因此即使REL8有效也不称“参数共享是唯一原因”。

### 9.4 感受野表述

不再写“四层已覆盖全部图，所以没有传播问题”，也不写“高4失败证明四层不够”。旧最近目标hop测量和算法依赖深度不是同一个量。所有相关解释使用“关系组合/迭代处理”而不是未经证明的全图覆盖。

不修改HandEmpty边、不制造长路径、不加自适应停止器、T16/32或测试时逐题挑轮数。

---

## 10. 方法E：目标关系约束注意力

### 10.1 两条件

```text
R-C1-SERIAL-GOAL-DENSE-0
R-C1-SERIAL-GOAL-REL-0
```

均在原MG目标向量z之后、求和rho之前增加一个4头、128维的残差注意力层。输出投影Wo=0，两条件初始分布复现MG。原编码器、phi/rho共同微调。

DENSE与REL读取**完全相同的全目标对关系特征**，使用相同参数和计算张量，唯一差别是注意力掩码。

\[
e_{ij}^h=(W_Q^hz_i)^\top(W_K^hz_j)/\sqrt{32}+b_h(r_{ij}(s,G)),
\]
\[
\widetilde z_i=z_i+W_o\,\mathrm{concat}_h\sum_j
\mathrm{softmax}_j(e_{ij}^h+\mathrm{mask}_{ij})W_V^hz_j.
\]

### 10.2 r与mask的可实施边界

r_ij的字段在训练前固定：
- 目标谓词类别、正负号与当前真值；
- 参数位置两两是否绑定同一对象（不读对象名哈希）；
- 当前为真的关系命题，是否直接连接两个目标所涉及的参数对象；
- 是否存在同一公开grounded contract，其ADD/DEL对这两个目标构成直接威胁；
- 自环、共享对象标记。

REL允许：
```text
i==j
OR share_bound_object(i,j)
OR current_true_relational_atom_connects_argument_objects(i,j)
OR direct_contract_add_delete_threat(i,j)
```

DENSE允许全部目标对。零元HandEmpty没有对象参数，不作为“任意两个目标连通”的mask依据；但生产编码器的HandEmpty边**不能删除**。

所有关系只来自当前状态、目标绑定和公共合同。不能使用规划器标注的目标顺序、必须拆、最优动作或题目分层ID作为模型输入。

mask由待评分状态和完整G决定，同一个状态不因来自哪个根动作而得到不同势函数。仍按MG的一阶后继差分排序。

### 10.3 分布外屏蔽与解释

记录训练/开发/确认的：
- 全mask密度、非自环密度；
- 同目标塔/跨目标塔密度；
- OnTable相关目标对密度；
- train与test各类关系频率。

“同塔/跨塔”只用于结果描述，不作为网络输入或手工mask规则。

训练只有一座非平凡塔，但仍有OnTable单块，不能笼统说训练完全没有跨对象目标对。测试的塔间关系可能未被训练覆盖。

REL超过DENSE可能来自：
1. 合同/状态结构约束更合适；
2. 它屏蔽了测试时新增而训练未见的关系；
3. 两者共同造成更稳定的输入分布。

**仅靠mask密度无法唯一分开这些解释。**本轮不增加第三条mask消融，不主张已经证明唯一机制。若所有实际mask都为全通，两个条件数学等价，按训练前fixture复用一次，不浪费第二条。

---

## 11. 全套开发和确认需要报告的指标

### 11.1 统一执行表

每个条件、每个面板和高度层输出：

1. 成功数/分母；
2. 全程最优数/分母；
3. 实际计划长度／最短长度；
4. 成功题的多余动作数、共同成功题配对差；
5. 全部题的失败计罚成本，避免提前失败被认为更高效；
6. 第一次raw错误与第一次executed错误；
7. 首次错误类型；
8. 模型原本选对但被C3/前瞻/G1即时改坏的次数；
9. C3候选中已没有原最优动作的状态数；
10. 必要回退、高4、多塔、简单保护层分别统计；
11. 参数数、逻辑编码图数、batch调用次数、循环次数、叶展开数、推理时间和显存；
12. 新标签/端点/目标对计算成本。

**高4收益统一称为“未见目标塔高上的改善”。**不能仅凭同组涨分直接归因于准备组织、关系深度或交互。方法匹配对照能支持条件差异，但不排除多种机制共同参与。

### 11.2 联合事件模块额外指标

对完整Y*状态，输出：
- 事件投影Q的Top1命中；
- top joint pair是否属于Y；
- P_Y、P_F、P_A；
- P_Y/P_F：给定投影可接受质量中的真实配对质量；
- pairing-informative状态层的动作准确率与联合预测；
- singleton-event层、OnTable-only层、On层；
- SG-FACT与SG-JOINT在同一状态的修复/新增错误。

未知标签单列。其他方法没有事件预测就写N/A，不能从执行轨迹倒推一个预测结果。

### 11.3 共同状态库与困难错误链

旧240库不再承担主要收益门槛。最终确认库每题最多6个固定状态：
- 初始；
- 字典序最优轨迹约1/3和2/3；
- 终止前；
- 最多两个按固定顺序的非最优后继。

选状态不看模型分数；所有无记忆模型在相同状态、同合法动作上评raw排序。构造标签超限时动作/事件指标分别标缺失，整局评价仍完成。

已有MG错误状态可作开发挑战库，按(problem,state)去重，和旧240库分表。它是按旧模型挑出的困难状态，不能与无筛选公共库混成总体准确率。

### 11.4 统计边界

- 所有配对比较保留“共同正确、仅前者、仅后者、共同错误”；
- 同一问题多个状态、同一颜色孪生骨架按组处理；
- 可以输出固定检查点在题目抽样下的配对区间/检验，不能冒充训练种子稳定性；
- 本套多条件比较是探索筛选；显著性检验若报告，同时记录比较总数和校正方式，不依赖单个p值给方向盖章；
- 不显著不等于等价，单seed不能断言普遍胜者；
- 普通和外推指标都报，不为了“完整结论”硬选一个正结果。



## 12. 最终统一确认：所有有效条件一次比较

### 12.1 满额确认矩阵

| # | 条件 | 权重来源 | 推理方式 |
|---:|---|---|---|
| 1 | MG_C3 | 原MG | 一步+C3 |
| 2 | B_G1C3 | 原B2 | 旧强规则组合 |
| 3 | LOOK2_MG | 原MG | 两步、神经叶值 |
| 4 | LOOK2_COUNT | 原MG仅用于规定回退 | 同树、目标数叶值 |
| 5 | SG_BASE | 开发选定 | 一步软目标组织+C3 |
| 6 | SG_FACT | 开发选定或等价别名 | 同上 |
| 7 | SG_JOINT | 开发选定或等价别名 | 同上 |
| 8 | CAL_ABS_C3 | 开发选定 | 一步+C3 |
| 9 | CAL_REL_C3 | 开发选定 | 一步+C3 |
| 10 | CAL_ABS_LOOK2 | 与#8相同检查点 | 两步叶值 |
| 11 | CAL_REL_LOOK2 | 与#9相同检查点 | 两步叶值 |
| 12 | REC_SELF_T4 | T4开发选定 | 4轮 |
| 13 | REC_SELF_T8 | 与#12相同检查点 | 8轮 |
| 14 | REC_REL_T4 | T4开发选定 | 4轮 |
| 15 | REC_REL_T8 | 与#14相同检查点 | 8轮 |
| 16 | GOAL_DENSE | 开发选定 | 稠密目标注意力 |
| 17 | GOAL_REL | 开发选定或等价别名 | 关系mask目标注意力 |

规划器另作128个初始问题参照，不属于学习方法。最多 **17×128=2176局＋128个规划器参照**。

同一检查点相同算法的数学别名只执行一次；表中注明引用同一结果，不把别名伪造成独立运行。技术失败条件保留一行TECHNICAL_INCOMPLETE及原因，不用零分或删除掩盖。

### 12.2 确认后不继续追逐冠军

开发结束时预先写 `preconfirmation_selection.json`：
- 每个家族的候选与匹配控制；
- 按开发信息选择的一个系统候选；
- 固定的关键比较列表；
- 已知副作用与预计收益范围。

确认后报告：
1. 预先选中的候选是否复现；
2. 全17条件的观察结果；
3. 每家族的方法主张是否得到支持；
4. 观察到的最佳系统与预先选中系统是否相同。

可以报告本次确认表中分数最高的系统，但应称“本套题的观察最佳”，**不能在确认后再调它并沿用同一测试证明泛化**。

本卡最后不自动训练、更换数据集、组合获胜模块或启动第二域。最终报告可以推荐下一方法，但不再次请求用户确认本卡尚未执行的阶段。

---

## 13. 最终必须回答的研究问题与结论模板

### 13.1 五家族必须逐项有答案

**两步前瞻：**
- 它是否超过一步？
- 神经叶值是否超过同树计数？
- 额外搜索与模型评分各能解释到什么程度？

**下一事件：**
- 标签是否真正含动作特异的配对信息？
- FACT是否已经覆盖JOINT收益？
- 收益是否主要发生在OnTable事件、准备动作和训练外目标高度？
- 是否改善执行而不只是辅助预测？

**价值校准：**
- 温度与步数尺度是否稳定？
- ABS与REL谁更适合一步或两步？
- 是否只是值误差变小而任务没改善？

**循环推理：**
- REL是否优于SELF？
- 同checkpoint从4到8是否在h4有稳定的配对净收益？
- 是否付出明显简单能力或成本代价？
- 没有未共享参数控制时，哪些主张不能作？

**目标注意力：**
- REL是否超过强DENSE？
- 跨塔mask是否训练外改变？
- 是否只能解释成合理的屏蔽先验，还是已有更直接证据？

### 13.2 每项必须分四层写

```text
事实：实际计数、修复与损害、成本
匹配比较：这项对照真正改变了什么
支持的解释：在这些设置和检查点下成立的范围
仍不支持的解释：替代原因、额外信息/计算、外推限制
```

不允许：
- 所有候选都无优势时强行“包装出一项创新”；
- 没有新领域结果就写跨领域已成立；
- FACT追平JOINT仍写联合配对是关键；
- 只增加搜索就说模型推理能力提高；
- 温度变化改善概率就说动作排序一定改善；
- h4提高就说所有长期依赖已经解决；
- 只因已有论文相似而否定已观察到的真实增量。

### 13.3 最终建议的排序原则

先列所有方案的部署收益、保留能力、额外成本和匹配控制差异，再排序。默认按：
1. 困难层实际执行能力/成本有无改善；
2. 简单与训练内高度是否保留；
3. 相对匹配控制是否有特定增量；
4. 计算与标签代价；
5. 对第二公开领域的适用前提。

不设单一不可更改“最优率必须多5题”门槛。小差异直接写证据弱；必要时推荐保留旧系统。所有原始数字必须可追溯。

建议最终状态：

```text
C1_METHOD_SERIAL_SUITE_COMPLETE
C1_METHOD_SERIAL_SUITE_COMPLETE_WITH_SKIPS
C1_METHOD_SERIAL_SUITE_PARTIAL_TECHNICAL
C1_METHOD_SERIAL_SUITE_BLOCKED
```

前三者必须同时说明已完成多少家族、多少训练和评价；`COMPLETE_WITH_SKIPS`仅用于数学等价/标签无信息的合法跳过，不能掩盖技术失败。

---

## 14. 可实现的总入口与最小代码结构

### 14.1 拟新增模块

```text
src/cp_disr/blocksworld/method_serial/
  assets.py                 # xushijie3路径解析、身份与manifest
  schedule.py               # 状态机、恢复、顺序与预算
  shared_evaluation.py       # 高度分层、共同选模、统一表格
  lookahead.py
  event_labels.py
  event_model.py
  event_losses.py
  calibration.py
  recurrent.py
  goal_attention.py
  report.py
scripts/c1_method_serial.py
tests/test_method_serial_*.py
```

复用而不改写：
```text
goal_progress.py
gp_attribution.py
c1_blocksworld_policies.py
planner.py
imitation.py
state.py
scale_order_prototype现有生成器/状态库/结果入口
```

如需新增“返回节点H、不执行旧readout”的hook，必须用旧输出等价fixture验证；不为了本卡顺带重构全项目。

### 14.2 CLI合同

Agent完成适配后应提供：

```text
python scripts/c1_method_serial.py prepare --config <suite.yaml>
python scripts/c1_method_serial.py run-all --config <suite.yaml>
python scripts/c1_method_serial.py resume --config <suite.yaml>
python scripts/c1_method_serial.py status --config <suite.yaml>
python scripts/c1_method_serial.py report --config <suite.yaml>
```

`run-all`是一个阻塞的串行驱动器，内部依次启动子进程并等待完成，不依赖助手以后再次发消息，不需要每半小时一个重复cron。可以由用户现有tmux/作业系统持有会话，但只允许本卡一个driver实例。

**附带的调度参考不是已接入仓库的训练程序。**Agent必须实现上述CLI与各stage处理器，再执行。不能直接拿数学参考模块当生产训练器。

### 14.3 总循环伪代码

```python
verify_activation_and_assets()
acquire_suite_lock()
try:
    stage("prepare")
    stage("lookahead_dev")
    stage("event_labels")
    family("event", ["SG_BASE", "SG_FACT", "SG_JOINT"],
           exact_equivalence_reuse=True)

    order = choose_priority_from_completed_lookahead()
    for name in order:                    # 两家族都做，只变先后
        family(name, required_conditions(name))

    family("goal_attention", ["GOAL_DENSE", "GOAL_REL"])
    lock_all_checkpoint_selections()
    stage("confirm_prepare")              # 到此才生成/打开最终题
    for condition in eligible_conditions():
        evaluate_confirm_once_or_resume_missing(condition)
    stage("report")
    stage("size_and_publish")
finally:
    write_final_receipt_even_on_failure()
    release_suite_lock()
```

低分不抛异常。一个家族局部技术失败时，记录失败并进入下一个独立家族。全局泄漏、基线hash不符、路径越界和权限错误才抛全局异常。

### 14.4 断点与状态写入

每个stage的状态包括：
```text
stage_id, status, attempt
source_commit, config_hash, inputs_hash
started_at, finished_at, host, device
new_optimizer_steps, effective_optimizer_steps, resumed_steps
output_paths, output_sha256
exception_type, exception_summary
```

状态用临时文件＋原子改名写入。`DONE`必须同时满足产物存在且hash匹配；文件夹存在不等于运行完成。

训练额度按唯一run_id计数；同配置恢复不是新seed，但额外尝试时间与重复计算单列。已完成run_id不可因分数不佳被删除后重跑。

---

## 15. 必需fixture与不再要求的审计

### 15.1 最少正确性测试

**共享层**
- 资产映射到xushijie3、符号链接回写xushijie2被拒绝；
- 训练目标h≤3统计正确；初始高度单独计算；
- 完成receipt可恢复跳过，局部失败能接续，全局失败会结束；
- 训练次数上限、唯一锁与确认集后训练禁令有效。

**下一事件**
- 小型多解穷举与Y一致；
- `Y={g*}×A`时FACT与JOINT数值和梯度相同；
- `Y!=Q×A`时标签确有配对信息；
- Q==U时FACT与BASE相同；
- UNKNOWN安全占位、合法mask、梯度有限；
- 初始动作分布与MG一致；
- 目标/动作重排不改变对应分布。

**价值**
- β0、a0、c0保持初始动作概率；
- 相对损失对全局偏移不变，绝对损失不具备该不变性；
- 不把错误片段强制进展；
- ABS/REL读取相同端点，β为两者共有；
- η正温度及数值有限。

**循环/注意力/前瞻**
- 共享参数每轮重用，不意外创建新参数；
- T8未进训练；SELF/REL选用同一T4选点规则；
- goal mask只依赖待评分状态与G，不依赖标签或前驱动作；
- dense也读取相同关系特征；
- 同根同历史的两种叶评分共享一棵树，假想访问不污染真实访问；
- 提前终止与空叶回退一致。

### 15.2 不再做

不再开展全历史合同审计、旧图一致性重跑、更多规模增补、额外高4训练、巨大文献筛选、每小步单独登记提交、每组成绩不足就新增门槛、每个候选多seed。

测试只用于保证新定义被正确实现，不成为无限推迟主实验的理由。

---

## 16. 计算预算、产物大小与提交

### 16.1 训练预算

正常满额：
```text
3 SG + 2 CAL + 2 REC + 2 GOAL = 9 runs
9 × 4100 = 36,900 accepted optimizer updates
9 × 744,600 = 6,701,400 base decision exposures
```

这些展示数不包含端点/目标对/循环额外前向和被中断尝试；都应另列。

### 16.2 评价预算（满额示例）

```text
9条件×5检查点×Board48 = 2160局开发选模
9个选定条件×旧112      = 1008局开发支持
9个选定条件×dev36       = 324局
两步原MG/COUNT×160       = 320局
REC额外T8两条件×160      = 320局
CAL额外LOOK2两条件×160   = 320局
合计约4452局新开发执行
最终17条件×128          = 2176局
最终规划器初始参照       = 128
```

旧冻结参照可在完全相同代码/题/权重条件下复用；若不满足则增加实际重评并列明。公共状态查询、精确事件标签和校准标签另计。数学等价复用或技术失败会减少物理执行，不伪造满额。

这比原来的两条训练方案大，原因是用户本轮要求五类全覆盖和统一最终结论。不得只报告“9条训练”却隐藏开发评价和标签开销。

### 16.3 产物管理

至少保存：
```text
suite_runtime.yaml
suite_config.yaml
stage_state.json
assets/path_map.json
assets/asset_identity.json
prep/training_scope.json
prep/event_information.json
prep/calibration_targets.json
runs/*/training_accounting.json
runs/*/checkpoint_selection.json
development/all_checkpoint_scores.csv
development/support_by_case.csv
confirmation/manifest.json
confirmation/by_case.jsonl(.gz)
results/by_family.csv
results/by_height_seen_unseen.csv
results/by_goal_structure.csv
results/paired_repairs_harms.csv
results/first_error_types.csv
results/next_event_information_and_accuracy.csv
results/lookahead_leaf_diagnostics.csv
results/calibration_and_temperature.csv
results/recurrent_T4_T8.csv
results/goal_mask_density.csv
results/compute_costs.csv
results/selection_history.json
results/final_summary.md
results/claim_boundary.md
results/method_recommendation.md
results/technical_gaps.md
results/verify.json
receipts/final_receipt.json
```

权重、优化器和大缓存留在xushijie3，路径/大小/SHA进Git。逐题轨迹可gzip；超过轻量阈值保存在资产目录并提供hash，不删除唯一原始记录。

提交前先输出每个文件与目录的大小。默认普通Git单文件≤10MB，超限文件转资产存储；这是归档策略，不是删除结果的许可。

建议两次主要提交：实现＋固定计划；结果＋总结。无需每个函数修一次单独登记。训练运行中的语义源代码不改；纯报告脚本的事后修改单独注明。

可以在既有用户授权范围推送新分支，不创建PR、不合并、不强推。若推送权限失败，保留本地commit SHA及错误，不说远端已同步。

### 16.4 最终回执模板

```yaml
card: C1-BW-METHOD-SERIAL-SUITE-V3
base_commit: 2d008431eb7fb069e617e4a02843bef84c6117d1
branch: codex/cp-disr-c1-bw-method-serial-v3
storage_owner: xushijie3
resolved_storage_root: <actual>
resolved_repo_root: <actual>
path_migration_verified: <true|false>
old_results_modified: false

execution_mode: FULL_SERIAL
new_training_cap: 9
new_trainings_completed: <n>
mathematical_reuse: []
technical_incomplete: []
family_completion:
  lookahead: <status>
  next_event: <status>
  calibration: <status>
  recurrent: <status>
  goal_attention: <status>

training_goal_heights: [2,3]
height4_in_gradient_training: false
event_semantics: FIRST_CURRENTLY_UNMET_FINAL_GOAL_ATOM
pair_informative_unique_states: <n>
ontable_event_fraction: <actual>

confirmation:
  actual_cases: <n>
  actual_conditions_executed: <n>
  mathematical_aliases: []
  episodes_completed: <n>
  planner_references: <n>
  selection_locked_before_confirmation: true

result_commit: <full_sha_or_not_pushed>
best_observed_system: <actual>
preselected_method_confirmed: <actual_or_inconclusive>
specific_method_contributions_supported: []
specific_claims_not_supported: []
next_recommendation: <text>
status: <one_of_section_13>
remaining_processes_owned_by_this_card: 0
```

---

## 17. Agent可直接使用的交接指令

```text
依据本runbook的用户执行授权，在xushijie3完成五类方法的全覆盖串行实验。
不要重新选择研究方向，不追加规模或高4训练，不运行外部方法和第二域。

先解析并验证xushijie3实际路径，迁移本卡必要只读资产；旧xushijie2不再新写入。
核对基线2d008431eb7fb069e617e4a02843bef84c6117d1与原MG权重、D_train。
建立唯一串行driver，可中断恢复；不要依赖每半小时重复cron。

先做两步前瞻，再做SG-BASE/SG-FACT/SG-JOINT。
依据前瞻读数决定CAL与REC两组先后，但两组最终都做。
最后完成GOAL-DENSE/GOAL-REL。正常上限9条训练。
数学等价可复用，技术局部失败记录并继续其他独立项。
低分正常接续；不在每阶段等待用户选下一步。

选模统一使用Board48；旧112与240状态库分表作为支持。
所有训练目标高度仍为2/3；高4统一标为外推。
SG只用原事件定义，不暗改成首个On。
CAL的绝对/相对损失分开、共同使用正温度。
REC同一T4选定检查点评T4和T8；DENSE与REL注意力共享信息与参数。

全部开发结束并锁定后生成一份Confirm128，
全部有效条件都在同一题上评价，最后一次性给完整结论。
最终先查产物大小，再提交轻量结果、结论边界和回执。
保留低分、别名和技术缺口，不虚构成功。
完成后停止，不自动把获胜模块叠加或开展第二域。
```

---

## 18. 来源、核验范围与文件关系

### 18.1 本轮来源

[A1] 用户本轮粘贴的Claude审计。其“logit间隔中位54”等新增统计未由本轮重新加载模型核验，按审计报告标注。

[P1] `CP_DISR_C1_Method_Selection_and_Experiment_Plans_20261007.md`。
[P2] `CP_DISR_C1_Next_Event_Agent_Runbook_v2.md`。
两文件由Files读取；本版覆盖其顺序、授权范围、预算、FACT控制、价值损失和确认安排。旧文件仍保留，不改写。

[R1] 固定提交 `2d008431eb7fb069e617e4a02843bef84c6117d1` 的 `src/cp_disr/blocksworld/generator.py`：训练目标形状最高为3。
[R2] 同提交 `src/cp_disr/blocksworld/goal_progress.py`：状态值、候选评分、动作集合损失与排序。
[R3] 本轮GitHub matching refs：`codex/cp-disr-c1-bw-*`，用于确认基线，不代表服务器未推送状态。
[R4] 前文已提供的scale-order结果及原方法代码；本轮不重跑旧实验。

[W1] Bansal et al., NeurIPS 2022, *End-to-end Algorithm Synthesis with Recurrent Networks: Extrapolation without Overthinking*。官方页与作者摘要说明输入回忆及渐进训练用于缓解过度迭代。本版只借鉴其稳定性警告，不声称完整复现：
`https://papers.nips.cc/paper_files/paper/2022/hash/7f70331dbe58ad59d83941dfa7d975aa-Abstract-Conference.html`
`https://arxiv.org/abs/2202.05826`

### 18.2 附带代码的范围

- `math_reference_v3.py`：FACT/JOINT信息量与损失、温度/校准公式的独立张量参考；
- `test_math_reference_v3.py`：合成CPU数学检查；
- `serial_driver_reference.py`：独立串行/恢复/局部失败接续框架；需要Agent接入真实stage命令；
- `suite_config.yaml`：预算和顺序的机器可读示例；
- `README.md`：接入步骤与权限边界。

**这些文件不是已训练模型，也不是已完成服务器端完整适配。**本轮已完成21项本地参考测试（14项张量数学＋7项虚拟子进程/路径/恢复测试），全部通过；未加载生产权重、未运行任务环境、训练与优化器步数均为0。真实仓库和GPU集成仍由Agent完成。

---

**最终执行原则：不再因为单项低分停住，也不为了给“完整结论”硬判赢家。完成明确的五类方法对照，在同一确认集上给出性能、成本、外推范围和主张边界。**
