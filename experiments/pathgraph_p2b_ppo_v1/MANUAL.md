# PathGraph P2B：图势函数奖励＋PPO强化学习效用实验
## Agent操作手册 · V1.0

**本阶段核心：从“奖励算得对”推进到“它能否帮助策略学得更快、更好”。所有方法使用同一套PPO，主要自变量是奖励的势函数来源。**

固定BASE：`577db67c64343850965fa4e0f6bc7cac0c15674d`  
建议分支：`research/pathgraph-p2b-graph-ppo-utility-v1`  
证据层级：`ONLINE_RL_ACTION_CONDITIONED_ABSTRACT_SKILL_ENV_NOT_PHYSICS`

> P2A负结果及取证保留。P2B不是“继续调BC权重”，不是新的抓取工程，也不是用更强算法掩盖旧结果。

---

## 0. 来源事实与本轮新设计必须分开

### 已核对的事实

- P2A普通BC成功率约0.914，V6加权约0.867；效用未建立。取证发现主体推进动作被相对降权。这些结论不因P2B的成败改变。[R1–R2]
- P2A环境在`experiments/pathgraph_p2a_utility_v1/p2a/env.py`。两物体、13个技能动作、31维共同观测；动作改变环境状态，没有MuJoCo物理。[R3]
- 环境已有8种条件、64步限制、按时间索引生成的随机表；抓取/重抓是声明的概率技能原语，不是真实抓取器。[R3]
- P1势函数由外置`reward_v6.py::CapabilityPotential`提供。它包含对象级能力代价和几何项，具有历史初始距离anchor；本包按真实字节锁加载，不另写同名公式。

### 本轮新增，不是既有结果

PPO、折扣PBRS适配、有限任务期限的Gymnasium边界、LONG_CHAIN128配置、非图强势函数、新family划分、交互步数和统计规则，均是P2B预注册设计。

PPO和势函数奖励塑形均来自已有文献。本轮的研究变量是**任务图产生的奖励表示是否有下游增量**，不是发明PPO或望远镜求和。[S1–S4]

---

## 1. P2A → P2B：到底改变什么

| 维度 | P2A | P2B |
|---|---|---|
| 学习范式 | 固定示范上的加权监督学习 | 当前策略交互的on-policy PPO |
| 数据 | 1,920条固定示范 | 新family上的在线采样；轨迹随策略变化 |
| 专家动作 | BC监督标签 | 不进入训练，不接管评估 |
| 奖励用途 | 正单步奖励→样本权重 | 基础任务奖励＋折扣势差，参与优势/价值估计 |
| 优化目标 | 模仿示范动作 | 最大化预先定义的折扣任务回报 |
| 网络 | BC动作分类网络 | 同构策略网络＋价值网络的actor-critic |
| 时间信用 | 单步权重 | PPO/GAE传播长期后果 |
| 公平性 | 同数据、同初始化、同batch | 同环境分布、初始网络、步数和超参数；不强求相同轨迹 |
| 环境 | 抽象技能环境 | 复用相同核心；新增命名难度配置和终止适配 |
| 主结论 | 权重映射是否有用 | 图奖励相对基础/强非图塑形是否有学习效用 |

**保持不变**：V6状态势函数源码、P1/P2A结果、技能基本含义、所有方法共同观测、任务成功的事实定义。禁止同时引入图专用策略输入、图动作mask、新网络或新抓取器。

---

## 2. 本轮问题与允许结论

H-RL1：同PPO下，图完整势函数是否优于仅基础任务奖励？  
H-RL2：图完整势函数是否优于“计数＋匹配事件＋几何进度”的强非图势函数？  
H-RL3：完整图势函数是否比图能力代价本身更有用？此项为消融，不代替H-RL2。  
H-RL4：差异是否体现在学习速度、最终成功或长动作链稳健性，而不只是训练回报尺度？

不预设主方法必须获胜。成功收口包括可靠正结果、无差异、退化、所有方法平台化等结论。

最强允许结论：在声明的抽象技能任务、固定PPO配置与采样条件下，图势函数塑形改善高层策略学习。不能扩大为真机、视觉、摩擦抓持、完整物理闭环、所有图算法或原SARM论文性能。

---

## 3. 环境与任务难度

### 3.1 复用，不重写物理环境

`p2b/task_core.py`加载冻结P2A `SkillEnv`。保留13动作：WAIT，以及A/B各自的ACQUIRE、ADVANCE、PLACE、START_RECOVERY、REGRASP、RELEASE。

策略观察保持31维：两物体位置、目标、held、valid、开放loss、phase以及剩余时间。所有方法得到相同字段；不把图cost、势函数值、未来扰动、condition ID、专家建议或“正确动作”只提供给图组。

phase是本参考环境已可见的状态，不是新训练的视觉预测。任务存在隐藏的随机因素/条件安排；不得声称这已覆盖真实部分可观测机器人控制，也不能把参考世界叫作物理仿真。

### 3.2 两个预先命名的配置

| Profile | 动作推进步长 | 任务期限 | 检验范围 |
|---|---|---|---|
| BASE64 | 冻结P2A family步长 | 64步 | 原任务复杂度 |
| LONG_CHAIN128 | 同family步长×0.5 | 128步 | 同几何路线需更多连续动作，信用延迟更长 |

其他动力学、技能成功概率、目标判定、干预规则不变。训练episode按profile等比例交替；各方法完全相同。评估分层汇报，也给二profile等权平均。

LONG_CHAIN128不是新任务图拓扑，也不是新增物体泛化。不要为了图胜出临时加入更多失败、删观测或修改技能概率。

### 3.3 八条件原名不变

`FREE_ORDER`、`A_STARTED`、`B_STARTED`、`A_VALID_B_LOSS`、`B_VALID_A_LOSS`、`INVALIDATE_FIRST`、`LATE_LOSS`、`TWO_LOSSES`。

没有`gripper_reopen`条件；RELEASE是动作。A_STARTED/B_STARTED表示部分起始状态，不能自动称为强制顺序任务。

不以每个条件必须成功作为开始训练或评分的门。自然没有发生loss也要保留；只是不进入“已观察loss的恢复”分母，不能伪造事件。

---

## 4. 数据：从示范集改成策略交互流

### 4.1 不再使用旧示范训练

P2A的1,920条示范、P1构造holdout、V5/V6物理/状态缓存全部只作历史诊断，不进入PPO训练buffer、初始化、专家混合或蒸馏。不得用BC checkpoint暖启动主组。

PPO每次更新用当前策略最新采样的rollout，更新后再收新数据。不能为了“同输入公平”，强迫六种不同策略复用同一个旧rollout buffer，那不再是这套on-policy实验。

### 4.2 新family划分

- 工具/工程smoke：`1200000–1200003`。
- train：`1201000–1201031`，32个family。
- validation：`1202000–1202007`，8个family。
- test：`1203000–1203031`，32个family。

这些是本轮拟定ID。`tools/history_scan.py`检查BASE中已提交的registry/split/protocol/manifest；记录扫描范围。服务器外置历史registry由Agent补查一次，不循环等批准。碰撞只在采集前解决：改命名空间、重新冻结，而不是看结果后换失败seed。

种子由命名空间哈希得到64位非负数；训练key含policy_seed、vector_env_rank、episode_index，不含method。验证/测试key含split、family、profile、condition、repeat，不含policy_seed和method。

### 4.3 采样公平，不要求动作相同

相同训练seed跨奖励组：初始actor/critic参数相同；每个vector worker的episode-spec序列相同；各episode的外生随机表相同。因为策略不同、结束时间不同，进入同一episode的全局训练步可能不同，这正常。记录实际环境访问，不伪造“同batch”。

同一状态及同一动作序列，在不同奖励wrapper中必须产生相同后续状态。包内有这一非干预测试。

### 4.4 规模

6方法×8策略seed = **48训练任务**。每任务**524,288个环境transition**，合计**25,165,824个训练交互**，不含validation/test。

每个checkpoint验证：8family×2profile×8condition×4repeat = **512 episode**。  
每个最终策略测试：32×2×8×4 = **2,048 episode**。  
完整最终test矩阵：48×2,048 = **98,304 episode**。

这是计算实验的预注册规模，不是人工授权或nonce制度。不承诺运行耗时；实际资源时间由服务器记录。

---

## 5. 基础任务奖励

统一定义：

\[
r_t^{task}=\mathbf 1\{\text{本步真正完成任务}\}.
\]

成功只来自两个物体当前有效目标事实；每episode仅一次成功奖励。其他步骤，包括等待、无效动作、loss和期限耗尽，基础奖励为0。不额外惩罚某种方法更常选择的动作。

\(\gamma=0.99\)。这会在共同的任务目标中偏好更早成功。记录未折扣成功率和折扣任务回报，二者不是完全相同指标。

**本轮不把纯势差当唯一任务目标。**只用纯势差时，同起终点的累计值可能无法表达实际效用；PBRS作为基础目标之上的学习辅助信号。

---

## 6. 六组奖励，其他组件相同

统一使用：

\[
r_t^{train}=r_t^{task}+\beta F_t,\qquad
F_t=\gamma\Phi(x_{t+1})-\Phi(x_t),\quad\beta=1.
\]

实际实现已把beta写入`shaping_reward`；不要在wrapper外再乘一次。

令\(u=|V|/2\)，\(\ell=\text{当前未匹配恢复的loss数}/2\)，并令

\[
g={1\over2}\sum_{i\in\{A,B\}}\mathbf1[held_i\land\neg valid_i]
\operatorname{clip}(1-d_i/\max(d_{0,i},10^{-9}),0,1).
\]

\(d_{0,i}\)来自该episode初始已知距离，不用未来最大/最小值。初始距离接近零时该物体几何项设0，避免凭空归一化。

| 方法 | 状态势函数\(\Phi\) | 用途 |
|---|---|---|
| TASK_ONLY | 0 | 相同PPO的基础对照 |
| COUNT_PBRS | \(u-1\) | 简单无序计数 |
| COUNT_EVENTS_PBRS | \(u-1-0.5\ell\) | 对象级匹配事件强基线 |
| GEOM_COUNT_EVENTS_PBRS | \(u-1-0.5\ell+0.25g\) | 同样有稠密几何、不用图结构的更强对照 |
| GRAPH_COST_PBRS | \(-C_{cap}^{V6}(x)\) | 去掉几何项的图消融 |
| GRAPH_FULL_PBRS | 冻结V6 `psi` | 主方法 |

事件基线是**新的折扣兼容势函数版本**，不等于旧P2A直接loss扣分/restore返分序列；独立命名避免混称。它必须维护对象级开放事件表，不用累计恢复次数当奖励进度。

三种非图势函数系数全部在正式采样前冻结，不用test估分位数、标准差或最佳尺度。它们不共享图方法cost。统一beta不代表奖励RMS相等，需报告实际幅值/方差；本轮固定配置的收益不等于对全部scale鲁棒。

GRAPH_FULL只复用P1已验证的**状态势函数值**，P2B改变的是如何将势值变成折扣训练奖励。旧P1公式、参数和文件不覆盖。

禁止reward clipping、只保留正reward、reward归一化、按episode挑奖励方法。PPO的策略ratio clipping和梯度裁剪不等于裁剪环境reward。

---

## 7. 三种“结束”必须分清

这是本轮最高风险接口，不得省略测试。[S3–S5]

### 7.1 真正任务终止

P2B明确选择有限任务期限：BASE64为64步，LONG_CHAIN128为128步，剩余时间已包含在观察中。成功或未成功耗尽任务期限都为`terminated=True`，`truncated=False`，但`success`单独保留。

历史P2A底层把到期限标记为`truncated`。P2B wrapper将其映射为**新命名的有限任务终止语义**，保留`legacy_truncated`原值，不改旧源码、旧P1结果或伪造raw `terminal_failure=True`。

到真正任务终止，令吸收状态势值：

\[
\Phi_{terminal}^{effective}=0.
\]

成功和失败期限都适用。仍保存原始\(\Phi_{after}^{raw}\)和有效值，终端清算产生的正分不是“失败反而进步”。

### 7.2 外部时间截断

若未来为非任务原因的TimeLimit等截断，`terminated=False,truncated=True`：不清零末状态势值，需要bootstrap。主实验不套额外TimeLimit。

不要把自动reset后的新episode首状态当作上一episode末状态。SB3向量环境提供terminal observation，版本具体处理要经服务器集成smoke核验；不要手工重复补一次bootstrap。[S4]

### 7.3 PPO rollout buffer分段

n_steps=256到达、检查点保存、日志轮转都不是episode终止。保持环境和势函数bank连续，使用critic对尚未终止状态bootstrap，不能清零anchor或开放loss表。

---

## 8. 折扣账本：不能照搬P1“每个环加总0”的门

连续区间检查：

\[
\sum_{t=0}^{T-1}\gamma^t\beta F_t
=\beta[-\Phi(x_0)+\gamma^T\Phi^{effective}(x_T)].
\]

本包函数已使用`shaping_reward=beta*F`，审计时直接累加该字段。

若\(\gamma=1\)且非终端起终点同势值，净塑形为0，兼容P1。若\(\gamma<1\)，非终端严格状态返回也可能有非零塑形项：

\[
\beta(\gamma^T-1)\Phi(x_0).
\]

这**不自动是刷分**。真正终止且末势置0时，塑形相对基础任务回报仅增加依赖初始状态的常数\(-\beta\Phi(x_0)\)。这也是使用折扣形式并正确处理边界的原因。[S2]

**特别注意label-only**：LOST→RECOVERING原始势值不变时，P1的\(\Delta\Psi=0\)仍成立。但折扣塑形是\((\gamma-1)\Psi\)，未必为0。负势下WAIT也可能得到小的正塑形；这是折扣项，不是给recovery标签新加信用。必须同时输出原始势差、折扣项、终端清算和任务奖励。

因此，本轮不能沿用“所有准备动作0reward”“任何完整物理返回必须未折扣reward和为0”作为PPO门。

理论性质不是效果保证：在满足条件的理想设置中，PBRS不改变基础任务的最优策略排序；实际函数逼近、探索、有限样本会影响学习过程。本实验测的正是这些有限训练效果，而不是把最优策略不变定理当成必胜保证。

---

## 9. PPO实现与固定训练配置

直接调用官方Stable-Baselines3 PPO，不手写替代算法。[S1,S3]

- `stable-baselines3==2.7.1`
- `gymnasium==1.2.2`
- P2A兼容的PyTorch/NumPy在隔离环境解析一次，保存全量版本；不在不同方法中换版本。

这些是选定可复现版本，不宣称是最新版本。

| 参数 | 值 |
|---|---|
| policy | MlpPolicy |
| actor / critic | 各128×128，Tanh，orthogonal初始化 |
| 初始模型 | 同seed跨方法权重哈希相同；不加载BC |
| gamma / GAE lambda | 0.99 / 0.95 |
| learning rate | 常数3e-4 |
| n_envs / n_steps | 8 / 256，每轮2,048 transition |
| batch size / epochs | 256 / 10 |
| clip_range | 0.2 |
| clip_range_vf | None |
| entropy coefficient | 0.01 |
| vf coefficient | 0.5 |
| max_grad_norm | 0.5 |
| target_kl | None |
| advantage normalization | True，所有组一致 |
| observation/reward normalization | False |
| action mask | False |
| train interactions | 每任务524,288，256次完整rollout更新 |
| seed | 211,223,227,229,233,239,241,251 |
| 选checkpoint | 固定最终步，不按test或validation挑最高 |

CPU为小MLP默认；可在正式开始前统一换设备，但必须所有组一致、版本记录齐全。不能某方法GPU某方法CPU再只比较墙钟。

GAE会把后续后果传播到准备动作，但不会自动保证每个准备动作正优势。不要写“上了RL一定解决BC问题”。

---

## 10. 训练流程

1. 核对BASE与冻结源码。
2. 生成并锁定family、episode规格和48-job矩阵。
3. 运行纯测试、冻结源回放与上游PPO集成smoke。
4. runner提交后开始正式on-policy采样。
5. 每个seed的六方法以相同超参数训练至固定524,288步。
6. 在0、32,768、65,536、131,072、262,144、524,288步保存checkpoint并跑validation。
7. 所有训练checkpoint哈希封存后，才运行最终test。
8. 成功、无差异、退化都完整汇总。

不得要求所有环境case训练成功才允许评价。依赖或源码错误不能伪造结果；可完成的基线继续记录，但缺主方法时只能提交不完整效用结论。

一个seed崩溃不得换另一个seed补齐；保留失败。可恢复的基础设施中断需从完整一致的checkpoint/优化器/RNG/env状态恢复；当前参考runner**未提供完整环境训练续跑**，不要用仅模型权重声称精确续跑。

没有授权文件、nonce或逐阶段人工批准流程。学术数据隔离和不可改历史结果仍保留。

---

## 11. 评价指标与归因

### 主指标

最终test任务成功率，按policy seed×family聚合，再对两profile、八condition等权。不是训练 shaped return。

两个预注册主比较：

- GRAPH_FULL − TASK_ONLY。
- GRAPH_FULL − GEOM_COUNT_EVENTS。

给每个比较97.5%双侧crossed bootstrap区间；两项共同报告。实用差异门为点估计至少+3pp且区间下界>0。区间不是跨全部任务的普遍保证。[S6]

### 次指标

- validation成功率学习曲线的归一化AUC（按环境步横轴）；
- 首次达到80%且下一个预定checkpoint仍≥80%的采样步；未达到标右删失，不填0或删除；
- 最终折扣基础任务回报；
- 无效动作、WAIT、成功时步数、所有episode步数；
- 每个对象loss/restore配对计数与恢复延迟；
- 每条件成功率，尤其A_STARTED、INVALIDATE_FIRST、晚loss和二次loss；
- 基础/塑形/终端清算reward的均值、方差、RMS和梯度诊断；
- entropy、KL、clip fraction、explained variance、更新计数及墙钟。

学习曲线仅用validation，不能按它反复挑新超参数。次指标优势不能替代主比较失败；可写“最终效用未建立，学习速度有探索性差异”，不能挑最好的一条曲线当总体胜利。

### 事件分母

不同策略会到达不同状态，因此观察到loss的次数可以不同。报告每策略的实际loss分母和所有计划条件上的最终成功率。零loss时`restore|loss=NA`，不是1。没有完成任务却规避所有loss的策略不能被称为恢复优秀。

### 统计单位

8个训练seed、32个test family为交叉随机因素。每family的条件/重复先聚合；bootstrap重采样seed维和family维并保持配对，不把98,304episode或百万transition当成同等独立重复。

必须给每seed结果、leave-one-seed-out方向检查和BASE64/LONG_CHAIN128分层；若优势只来自一个seed或单一条件，要写清。

---

## 12. 决策不能混写

- 两个主比较均满足门：`RL_REWARD_UTILITY_SUPPORTED_IN_REFERENCE_DOMAIN`。
- 只优于TASK_ONLY：`TASK_BASELINE_GAIN_ONLY_GRAPH_INCREMENT_NOT_ESTABLISHED`。
- 未满足或方向混合：`RL_REWARD_UTILITY_NOT_ESTABLISHED_OR_MIXED`。
- 若负向区间明确，报告对应基线下的退化，不美化。
- 全部接近0：记录探索/训练能力不足；不能把底线全失败叫算法等价。
- 全部接近1：最终分数区分度不足，报告学习曲线，不能事后换更难test制造优势。
- 主训练/评估数据缺失：`RL_UTILITY_MATRIX_INCOMPLETE`，不能将失败job删除后发表完整比较。

这些是P2B新字段。历史P1 `confirmation_passed`不修改。新报告明确`robot_policy_gain_claimed=false`、`physical_cycle_claim=NOT_EVALUATED`；当前不是完整机器人确认。

如果图与强非图势函数持平，只能说该任务上未建立图结构额外效用。若仅有单一skill MDP上的正结果，也不能说广泛跨任务泛化。

---

## 13. 工程变化与不做的事

**新增**：Gymnasium接口、PPO训练runner、折扣PBRS wrapper、终止审计、新family在线采样、长链profile、事件/几何强基线、配对统计。

**删除出主线**：BC示范采样权重、95%分位缩放、positive floor、BC数据打乱权重、示范混合质量比例等。旧实现不删，只是不在P2B使用。

**不做**：MuJoCo、Contact-GraspNet/AnyGrasp部署、控制器改进、视觉、真机、GNN/RNN替换、在线图生成、图专用mask、同时优化reward参数与算法参数。

“工程为创新让路”意味着复用官方PPO和现有抽象任务，减少新自由度；不意味着可以忽略训练正确性、边界错误或测试污染。

---

## 14. 服务器接手与实际命令

```bash
export REPO=/home/__compress_data/xushijie/graph_github_upload
export BASE=577db67c64343850965fa4e0f6bc7cac0c15674d
export WORK=/home/__compress_data/xushijie/graph_pathgraph_p2b_ppo_worktree
export BRANCH=research/pathgraph-p2b-graph-ppo-utility-v1
export DATA=/home/__compress_data/xushijie/graph_pathgraph_p2b_ppo_data/run_v1
export V6_TOOLS=/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools
```

路径是服务器预期位置，不是本对话sandbox地址。Agent先核对存在。

```bash
set -euo pipefail
git -C "$REPO" fetch origin --prune
git -C "$REPO" cat-file -e "$BASE^{commit}"
git -C "$REPO" worktree add -b "$BRANCH" "$WORK" "$BASE"
```

工作树已存在时核对而不删除；父分支/main后来前进不构成阻塞，使用固定BASE且不修改main。

将本包复制到：

```text
$WORK/experiments/pathgraph_p2b_ppo_v1/
```

在独立Python环境安装`requirements-ppo.txt`。优先保留服务器已有可用PyTorch版本，不升级共享环境。依赖解析完成后，保存`pip freeze`、Python/PyTorch版本、OS和CPU信息到`runtime_lock.json`；48任务使用同一环境。

本包选择的版本应可从官方源安装，但本地交付环境未能联网安装SB3，**上游训练smoke必须由服务器实际运行**，不是依据文件存在就宣布完成。

```bash
export PKG=$WORK/experiments/pathgraph_p2b_ppo_v1
export PY=/path/to/isolated/p2b/python
export P2B_SOURCE_REPO=$WORK
export P2B_V6_TOOLS=$V6_TOOLS
cd "$PKG"
PY="$PY" bash tools/run_checks.sh /absolute/new/p2b_checks
"$PY" -B tools/history_scan.py --repo "$WORK" --out /absolute/new/family_scan.json
```

`run_checks.sh`包含真实字节源集成时需要两个环境变量；没有它们只运行算术测试并明确记录NOT_RUN，不能宣称source integration通过。

### 先做服务器smoke，不优化研究效果

```bash
"$PY" -B -m p2b smoke --repo "$WORK" --v6-tools "$V6_TOOLS" \
 --method TASK_ONLY --seed 211 --out /absolute/new/smoke_task
"$PY" -B -m p2b smoke --repo "$WORK" --v6-tools "$V6_TOOLS" \
 --method GRAPH_FULL_PBRS --seed 211 --out /absolute/new/smoke_graph
```

smoke各64交互步，只检查上游API、真实参数更新、存取checkpoint、自主动作及无泄漏，不用于选PPO超参数或证明效用。检查actor和critic初始化跨组相同、权重训练前后改变、`num_timesteps`准确、test未被访问。

任何API修正应先提交runner版本并记录；若修改了科学参数/数据分布则提升协议版本，不能默默让手册与执行不一致。

### 冻结runner后执行正式矩阵

```bash
cd "$WORK"
git diff --check
git add experiments/pathgraph_p2b_ppo_v1
git commit -m "research: add discount-correct graph PPO utility experiment"
git push -u origin "$BRANCH"
export RUNNER_COMMIT=$(git rev-parse HEAD)
```

记录commit后不修改package；本包seal会核对训练期间Python源码哈希。推荐在独立固定执行树运行，并将结果写外置DATA。

```bash
cd "$PKG"
REPO="$WORK" V6_TOOLS="$V6_TOOLS" DATA="$DATA" PY="$PY" \
 bash tools/run_formal.sh
```

此脚本自动：完整性检查→历史family扫描→plan→preflight→smoke→48训练→封存→98,304 test→统计。

没有训练候选选择步骤，不能按validation丢弃某seed。脚本对训练失败逐项记录，继续其它job；若矩阵不完整不生成完整效用通过声明。

单job命令用于排程，不用于择优重跑：

```bash
"$PY" -B -m p2b train-job --repo "$WORK" --v6-tools "$V6_TOOLS" \
 --method GRAPH_FULL_PBRS --seed 211 --out "$DATA/training/GRAPH_FULL_PBRS/seed_211"
```

正式输出目录必须新建。精确训练恢复尚未实现，不在已有目录上覆盖或伪造重启状态。

---

## 15. 必需审计与Agent还要完成的工作

包内提供能直接调用上游PPO的参考runner，不是只列目标文件名。但以下收口不能由包测试代替：

1. 服务器真实SB3/Gymnasium兼容smoke与运行版本锁。
2. 外置历史registry的family去重补查，注明范围，不虚构全盘核验。
3. 验证边界API：success、任务deadline、外部truncation、buffer cut；代码单测和真正上游rollout返回相符。
4. 汇总训练reward组成与on-policy阶段占用，确保没有再次把准备动作按即时正分过滤。
5. 每condition/profile/seed表、AUC、删失阈值、恢复分母、配对区间和所有异常job。
6. 报告基本初始化/环境随机流一致，但不宣称训练轨迹或梯度相同。
7. 对测试结果不利于主方法同样完整交付；没有工程抓取分支前置条件。

`p2b summarize`已产主表、主比较、学习曲线和决定；Agent需要加写人可读`report.md`、reward组成审计、leave-one-seed-out和limitations。不得把参考代码中解释字段当成这些分析已执行。

---

## 16. 文件组织

外置DATA：

```text
run_v1/
  family_history_scan.json
  plan/                         # 预先固定的episode specs与job matrix
  preflight/
  smoke/                        # NOT_RESEARCH_RESULT
  training/<method>/seed_<s>/
    run_identity.json
    initialization.json
    training_logs/episodes_worker_*.jsonl
    training_logs/audit_traces_worker_*.jsonl
    ppo_logs/progress.csv
    policy_<step>.zip
    validation/<step>/episodes.jsonl
    checkpoint_index.json
    complete.json 或 failed.json
  seal/test_ready.json
  test/<method>/seed_<s>/
    episodes.jsonl
    fixed_subset_traces.jsonl
    summary.json
  summary/
```

每个training episode都保留终局、任务/训练回报、配对loss、动作统计和discount audit。完整训练状态trace按不依赖分数的固定episode哈希抽样；不是只保留成功。test固定repeat=0保存完整trace，所有episode均保留结果及在线账本审计。

Git轻量结果：

```text
artifacts/pathgraph_sarm/upgrade_v2/p2b_graph_ppo_utility_v1/
  protocol_and_runtime_lock/
  source_lock.json
  job_matrix.csv
  training_completion.csv
  initialization_parity.json
  boundary_audit.json
  reward_component_audit.csv
  method_metrics.csv
  condition_metrics.csv
  learning_curve_summary.csv
  threshold_censoring.csv
  opportunity_metrics.csv
  primary_comparisons.json
  leave_one_seed_out.csv
  external_artifacts.tsv
  final/decision.json
  final/report.md
  result_manifest.json
```

不上传checkpoint、全量buffer、外置raw或Python环境。只上传索引、hash和必要轻量表。

---

## 17. 最终报告必须回答

- 这次确实是PPO在线学习吗？策略是否决定了训练和test动作？
- 学习数据来自哪些family、多少交互，而非多少旧示范？
- 六组是否只改变势函数来源？实际初始网络和超参数是否一致？
- 任务deadline、外部截断、rollout cut怎样处理？
- 原始势差、折扣项、终端清算怎样区分？
- 图组是否优于同PPO任务奖励及强非图势函数？区间和效应多大？
- 优势是最终性能还是样本效率，在哪些profile和条件出现？
- 两物体参考环境是否容易到所有组接近满分，或难到都无法探索成功？
- 相同函数形式保证核算，但是否产生实际学习差异？
- 尚未验证哪些内容？

最简结论模板：

```text
algorithm: SB3 PPO 2.7.1
new_on_policy_training: true
benchmark: ACTION_CONDITIONED_ABSTRACT_SKILL_ENV_NOT_PHYSICS
training_jobs: .../48
train_environment_steps_per_job: ...
P2A_demo_reused_for_training: false
teacher_takeover: 0
P1_frozen_sources_modified: false
primary_V6_minus_task: ...
primary_V6_minus_nongraph_strong: ...
learning_efficiency_evidence: ...
policy_utility_evidence: ...
robot_policy_gain_claimed: false
physical_cycle_claim: NOT_EVALUATED
historical_confirmation_fields_modified: false
```

**核心：这次让奖励参与真正的长期回报学习；使用同一成熟算法、共同任务目标和强非图对照，检验图势函数本身是否提供额外学习价值。**
