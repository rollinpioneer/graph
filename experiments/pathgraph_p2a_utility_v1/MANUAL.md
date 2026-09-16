# PathGraph P2A：奖励重加权的策略效用实验
## Agent操作手册 · V1.0

**核心问题：奖励算得合理，是否也能让同一个策略模型从同一批示范中学得更好？**

固定父提交：`8e86b246e900e1e12dbb21c88b95046a06c8e4c5`  
建议分支：`research/pathgraph-p2a-state-weighted-bc-utility-v1`  
本轮新增内容：动作驱动的参考技能环境、混合质量示范、真正的策略训练与自主策略评估。  
本轮不做：抓取网络、视觉、MuJoCo控制器、物理返回、真机或新的奖励设计。

---

## 0. 先读范围，避免再被工程问题牵走

P1已交付独立构造状态holdout的表示与核算结果。本轮不重复该验证，也不重新解释历史总PASS字段。P2A把问题从“账算得对”推进到“训练有无收益”。

**本包提供的是一个可执行的动作驱动抽象技能MDP，不是物理仿真器，也不是原SARM复现实验。** 策略真的输出动作，动作真的改变此参考环境中的后续状态；但“抓取、重抓”等原语的成功条件是本轮显式规定的参考动力学，不代表真实机器人抓取能力。新环境是本轮新增基准，不能冒充历史已有物理轨迹。

最强允许结论是：

> 在声明的抽象技能MDP、相同训练数据及训练配置下，冻结V6奖励的重加权是否改善了学习策略的自主任务成功率。

不能扩大为：机器人抓取更稳、视觉更准、真实物理闭环安全、所有图方法优于非图方法。

工程模块不可用时局部记录。**没有第三方抓取模型，也照常完成本轮。**训练或主奖励输入本身缺失时，则诚实标为不完整，不能用公式测试替代训练。

## 1. 来源和本轮新增设计

### 已核对的仓库事实

1. P1 `independent_state_holdout_v1/final/decision.json`：32条、12个状态循环，A–E通过，`policy_gain_claimed=false`。
2. `tools/stage6/policy_env.py`：已有动作驱动的轻量环境思路及实际执行动作记录。
3. 该旧环境的观测编码依赖预设order，干预在固定第7步；故不直接作为本轮主对照，旧源码保持不变。
4. P1 scorer使用外置`reward_v6.py`和仓库`extra_methods.py`。`method_source_hashes.json`提供真实字节锁。
5. P1构造holdout没有训练所需的实际执行动作；其legacy字段有构造占位值。本轮不能从那些状态倒推“示范动作”，也不能把占位`GRAPH_COST_ONLY=0`当作有效强基线。

### 本轮新增、不是已有结果

参考MDP、数据划分、权重映射、MLP、样本数量、优化步数和统计门都属于本手册的预先设计。本轮采用标准加权行为克隆结构，但权重映射不是原SARM逐参数复刻；方法名必须使用P2A名称，不能写“复现原论文SARM失败”。

原SARM以奖励辅助示范筛选和重加权作为下游用途[O1]；行为克隆需要观测—动作对[O2]。P2A直接检验这一用途，先不引入另一个RL算法变量。

## 2. 一轮执行路线

```text
核对P1源代码字节锁
→ 检查新环境动作确实影响结果
→ 新建并锁定混合质量训练示范
→ 只在训练集计算各方法权重
→ 9方法 × 5策略seed训练
→ 模型自主运行新的评估episodes
→ 同family、同seed配对比较
→ 无论增益、无差异、退化，均形成结果
```

不设人工授权/nonce流程，不把全体case成功率作为训练前门禁。保留数据失败、策略失败和不可用方法；不得换失败seed。

## 3. 工作树及精确源入口

```bash
export REPO=/home/__compress_data/xushijie/graph_github_upload
export BASE=8e86b246e900e1e12dbb21c88b95046a06c8e4c5
export WORK=/home/__compress_data/xushijie/graph_pathgraph_p2a_utility_worktree
export BRANCH=research/pathgraph-p2a-state-weighted-bc-utility-v1
export DATA=/home/__compress_data/xushijie/graph_pathgraph_p2a_utility_data/run_v1
export PY=/home/xushijie/.conda/envs/lerobot/bin/python
export V6_TOOLS=/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools
export EXTRA=$WORK/upgrade_v2/p1_mainline_grasp_v6gm1/extra_methods.py
```

路径是服务器接手路径，Agent先核实存在，不将其当成本对话sandbox附件地址。

```bash
set -euo pipefail
git -C "$REPO" fetch origin --prune
git -C "$REPO" cat-file -e "$BASE^{commit}"
git -C "$REPO" worktree add -b "$BRANCH" "$WORK" "$BASE"
```

已有工作树时先检查，不删除、不`reset --hard`。父分支后续前进不构成阻塞，使用固定BASE即可；记录main现状但不要求main永远停留在某个旧SHA。

把本包复制到：

```text
$WORK/experiments/pathgraph_p2a_utility_v1/
```

旧V6、V6GM1、P1holdout、stage6均只读。本轮reward scorer必须加载字节锁匹配的旧文件，不自行照公式重写一个“V6”。

外置源码锁见`contracts/source_lock.json`。本包不附训练好的策略或第三方抓取权重。

## 4. 唯一主环境：Action-conditioned abstract skill MDP

源码：`p2a/env.py`。

两个物体A/B、两个目标；当前状态含物体位置、目标、held、current-valid、开放loss和phase。动作13个：WAIT，以及每个物体的ACQUIRE、ADVANCE、PLACE、START_RECOVERY、REGRASP、RELEASE。

动作的语义是抽象技能原语：例如REGRASP在RECOVERING状态下按预设技能可靠性尝试恢复held。**它不是新实现的机器人抓取器。**此模型有意绕过机械执行层，单独检验高层动作学习和奖励重加权。

三项必须成立：

- 同一当前状态选择不同动作，应当可能产生不同后续状态；WAIT不能被后台脚本偷偷替换成正确动作。
- 任务成功只由两个物体当前确实在目标且不被持有产生，不读取condition的预设答案。
- horizon耗尽是`truncated`，统计成功为0，但不在奖励输入中伪造`terminal_failure`或EOF失败事件。

phase是参考状态原语的当前可见事实，**所有方法的策略都获得相同phase输入**。不得只给PathGraph策略额外历史/身份信息，然后把信息优势归功于奖励。

### 8个评估条件

| 条件 | 内容 |
|---|---|
| FREE_ORDER | 两个物体均未完成，策略自由选择先后 |
| A_STARTED | A已经持有，继续完成任务 |
| B_STARTED | B已经持有，与A_STARTED镜像 |
| A_VALID_B_LOSS | A有效，B操作中出现一次外生loss |
| B_VALID_A_LOSS | 镜像 |
| INVALIDATE_FIRST | 首个子目标完成后，另一个运输时使前者失效 |
| LATE_LOSS | 已运输较远后发生loss |
| TWO_LOSSES | 最多两次loss，由当下运输进度触发 |

扰动影响参考状态，不是weld操作。触发条件是实际held/transport/progress事实，不能因策略动作快就绕过固定第7步干预。若策略始终没建立抓持，loss可能未暴露，必须统计planned/observed分母，不能宣称“0漏检”。

物体运动步长、目标布局、原语可靠性按family生成；同一family跨方法固定。`condition`只控制环境，不进入策略观测或奖励输入。

## 5. 数据分组及数量

| 集合 | family | 每family/condition次数 | 用途 |
|---|---:|---:|---|
| train | 2011000–2011023，24个 | 10 | 1920条示范，训练唯一数据池 |
| validation | 2012000–2012007，8个 | 4 | 每策略256条自主rollout，运行诊断 |
| test | 2013000–2013015，16个 | 8 | 每策略1024条自主rollout，正式比较 |

历史32条状态holdout和112条V5回放都不进入本轮训练或测试。新family不是重新声称新物理分布，只是同一参考动力学下的留出参数与随机种子。

训练示范每组10条：5条当前状态专家、2条带停顿、2条混入随机动作、1条恢复动作不可靠的行为。错误动作实际提交给`step`并保存结果，不从图节点反向编造动作。质量标签只在数据审计里用，不能进入策略或权重。数据生成器不读取任何奖励值。

不得生成一池数据然后按V6得分保留“容易证明优越”的子集。全部1920条均进入共享数据池；包含超时、错误和无效动作。

**当前只使用一份训练数据集和5个策略seed。统计结论不覆盖“换一批训练数据”的随机性。**后续需要再扩为多数据种子时另立版本。

## 6. 九种固定训练方法

| 方法ID | 实际改变 |
|---|---|
| BC_UNIFORM | 每条示范动作权重1 |
| SPARSE_TERMINAL | 终端事件信号经同一映射转权重 |
| LINEAR_A_FIRST_R1 | 既有A-first进度差 |
| LINEAR_B_FIRST_R1 | 既有B-first进度差 |
| UNORDERED_VALID_COUNT | 当前有效数量差 |
| VALID_COUNT_PLUS_MATCHED_EVENTS_V1 | 数量差＋对象级配对loss/restore，强非图对照 |
| V6_CAP_COST_ONLY | 同一V6绑定只使用capability cost差，去几何项消融 |
| V6_CAP_POTENTIAL | 冻结V6完整势差，本轮主方法 |
| V6_WEIGHT_SHUFFLED | 打乱V6训练权重与动作的对应，保留权重多重集合/ESS |

`V6_CAP_COST_ONLY`不是旧图的`GRAPH_COST_ONLY`占位值。本轮所有任务统一是dual_order，两个线性基线均适用；不在不适用任务上填0再比较。

主比较预先固定为：V6 vs BC_UNIFORM；V6 vs VALID_COUNT_PLUS_MATCHED_EVENTS_V1。其余为消融或诊断。

图不是唯一能够保存完整状态的表示；非图方法也能构造相同势函数。即使本轮有增益，也只是相对这里明确实现的基线，不证明图的数学唯一性。

## 7. 权重映射：本轮新接口，不能冒称原SARM完全复现

由冻结奖励取得训练转移信号$r_i^{(m)}$。只用训练集非零绝对值计算：

\[
q_m=Q_{0.95}\{ |r_i^{(m)}|:|r_i^{(m)}|>10^{-12}\}.
\]

定义：

\[
u_i^{(m)}=\operatorname{clip}\left(\frac{\max(0,r_i^{(m)})}{q_m},0,1\right),
\qquad
\tilde w_i^{(m)}=0.2+0.8u_i^{(m)},
\qquad
w_i^{(m)}=\frac{\tilde w_i^{(m)}}{\operatorname{mean}_{train}\tilde w^{(m)}}.
\]

全零信号使用均匀权重并显式标记`all_zero_signal`；不能把退化基线描述为信息充分。BC直接为1。乱序权重用固定seed在训练样本内打乱，不碰validation/test。

底线0.2是本轮预先声明的工程取舍：保留START_RECOVERY等可能是必要但即时奖励为0的准备动作。它不是验证过的最优参数，不允许根据test再调整。采用统一mean-one权重，避免仅因梯度尺度不同造成假优势。

注意：$\sum r=0$不保证$\sum\max(0,r)=0$；本轮正是在实测这个重加权是否有用。势差的代数抵消本身不保证BC效果改善[O3]。外生loss可能给前一条合理动作负信号，这种归因风险也要保留，而不是偷偷让V6独享“知道哪一步该免责”的标签。

训练损失：

\[
\mathcal L_m(\theta)=\frac1B\sum_{i\in\mathcal B}w_i^{(m)}[-\log\pi_\theta(a_i\mid o_i)].
\]

离线权重读取$(s_t,a_t,s_{t+1})$是离线标签生成，允许；策略输入不得包含$s_{t+1}$、事件答案、奖励、势、图代价、数据质量或未来目标结局。

## 8. 真正训练策略

共同架构：31维当前状态输入、两层128单元ReLU、13动作logits，约2万个参数级别；实际参数总数写入报告，不以估数冒充测量。

共同设置：Adam，learning rate=$3\times10^{-4}$，batch=256，5000更新步，梯度范数上限1，最后一步checkpoint用于评估，不按test选最优。

策略seed：17、29、43、71、101。

共9×5=45个训练job。每个seed的初始参数和batch索引序列跨方法完全相同；仅训练权重不同。没有训练中在线采样补数，没有奖励模型更新，没有DAgger专家接管。

CPU可跑，已有CUDA可用则运行前选定一个device并锁定。不得某方法训练更多步、换更大模型，或仅重跑表现差的seed。

本轮默认不做额外25%样本效率实验、不换Diffusion Policy或VLA；先完成这一轮效用判定再扩展。

## 9. 自主rollout评价，不是离线标签准确率

测试调用已训练网络输出动作，再将该动作提交参考环境。禁止脚本替代网络动作、失败后调用专家、用冻结轨迹重放冒充策略控制。

**主要指标：8种条件等权、family等权的最终任务成功率。**评价器只读取参考环境的目标完成事实，不用V6回报定义成功，不以总reward高判策略好。

同时报告：

- 每个condition成功率、episode步数、超时、WAIT比例、无效动作数；
- planned-loss episode数、真实暴露loss数、每个loss是否匹配恢复；
- 至少一次loss的episode内恢复后任务成功率，明确这是条件指标；
- FIRST/SECOND loss分开，防止把未经历第二次loss的策略写成“第二次恢复100%”；
- 各权重的ESS、正质量、零信号比例及重抓/准备动作的权重；
- V6与shuffle、cost-only的差异，检查收益来自奖励语义还是仅来自权重分散；
- 所有checkpoint哈希、初始参数一致性、batch顺序一致性。

本包已提供episode级、逐loss配对、按condition指标、按动作权重及动作/观测日志。Agent只需将这些真实输出按交付清单收口；需要追加按phase的展示时从原始日志计算，不修改算法或训练参数。不能用episode restored计数代替逐loss记录。

## 10. 统计与判定，不追求必胜

配对单位为相同策略seed、相同test family、相同condition和环境seed。先在family内对条件等权平均，再做跨策略seed和family的交叉配对bootstrap，10,000次。transition不是独立样本。

两个主要比较使用97.5%区间（两项比较Bonferroni口径）；本包输出的是bootstrap近似区间，不是有限样本精确保证。

- 两个主比较均下界>0，且均值提升≥3个百分点：`UTILITY_SUPPORTED_VS_TWO_PRESPECIFIED_BASELINES_IN_REFERENCE_MDP`。
- 都明显为负：`UTILITY_DEGRADED_IN_REFERENCE_MDP`。
- 其余：`UTILITY_NOT_ESTABLISHED_OR_MIXED`，保留“优于BC但未优于强基线”等分项解释。

如果各方法都接近满分，写`CEILING_LIMITED_DIAGNOSTIC`；不能根据看过的test调难度再称同一独立测试。方法无显著差异不意味着等价，只有区间足够窄才谈排除某个效应大小。

始终：`robot_policy_gain_claimed=false`、`physical_cycle_claim=NOT_EVALUATED`。原P1的`confirmation_passed`不改。本轮使用单独`policy_utility_evidence`字段，不再拿一个全局false阻塞阶段收口。

## 11. 运行前的最小正确性检查

只检查会破坏因果结论的内容：

1. 环境WAIT/错误动作确实影响结果；成功来自当前几何与held事实。
2. reset种子复现；不同动作的状态分叉测试；一轮episode不自动重置。
3. 数据中动作是实际提交的动作，$T$个动作对应$T+1$个状态。
4. train/validation/test family不交叠；P1 holdout未进入训练。
5. V6和强事件基线字节锁匹配，linear两方向真实不同。
6. 策略输入白名单；修改未来数据不改变此前policy观测。
7. 奖励/权重只用train拟合；全零、负值、NaN有明确处理。
8. 每个方法同一初始参数、相同batch索引，评估没有teacher调用。

这些检查不是“所有场景都必须成功”的工程门。环境参考专家可以失败，全部记录；但若no-op也能自动成功或轨迹无动作因果关系，必须修正数据生成器再进行新版本训练。

## 12. 可执行入口

### 12.1 包内工具与完整性

```bash
cd "$WORK/experiments/pathgraph_p2a_utility_v1"
"$PY" -B tools/verify_package.py --root .
"$PY" -B -m unittest discover -s tests -v
"$PY" -B tools/preflight.py --repo "$WORK" --v6-tools "$V6_TOOLS" --extra-methods "$EXTRA" --out /absolute/new/preflight.json
```

本包提供真实环境、数据生成、冻结奖励调用、加权训练、自主rollout与配对统计，不是空CLI占位符。numpy和torch需要服务器环境已有或正常安装；不安装视觉、抓取或MuJoCo依赖。

### 12.2 冒烟运行（不得混入正式结果）

```bash
"$PY" -B -m p2a prepare --smoke --out /absolute/new/p2a_smoke/data
"$PY" -B -m p2a weights --data /absolute/new/p2a_smoke/data --v6-tools "$V6_TOOLS" --extra-methods "$EXTRA" --out /absolute/new/p2a_smoke/weights
"$PY" -B -m p2a train --weights /absolute/new/p2a_smoke/weights --method BC_UNIFORM --seed 17 --steps 20 --batch-size 32 --out /absolute/new/p2a_smoke/model
"$PY" -B -m p2a evaluate --checkpoint /absolute/new/p2a_smoke/model/policy.pt --data /absolute/new/p2a_smoke/data --split validation --out /absolute/new/p2a_smoke/eval
```

只用于确认文件和训练/控制链连通，不以冒烟表现调test或证明政策收益。

### 12.3 正式冻结和一轮执行

Agent先完成接口核验、必要日志补充及全部单测，将本包、协议和新代码提交为RUNNER_COMMIT。用该commit建立执行worktree，`cd`到执行树包目录，避免`--repo`指向执行树但Python仍导入开发树。

```bash
# 在冻结执行树的包目录内运行
export P2A_DEVICE=cpu
bash tools/run_formal.sh "$WORK" "$V6_TOOLS" "$EXTRA" "$DATA"
```

实际顺序：prepare→保存数据hash→weights→45个train→各自validation与test→analyze。

生成器、训练配置、评价规则在prepare前已冻结。训练数据生成后先写哈希，weights绝不回写数据。validation只诊断，最终使用固定5000步模型；不据validation重新选模型或超参数。若未来需要调参，另起开发版本和全新test。

正式路径已存在时脚本拒绝覆盖。进程中断要核对job日志与哈希，再从未完成的命令继续，不能把失败job删除后无痕重跑。脚本将job失败写入job_status；其余无依赖job继续。不完整矩阵不能输出总体增益，报告`PARTIAL_TRAINING_MATRIX`即可。

## 13. 不同失败怎么处理

| 情况 | 处理 |
|---|---|
| 抓取SDK、EGL、MuJoCo、相机不存在 | 本轮无需它们，继续 |
| V6字节源缺失/不匹配 | 定位已有冻结包，不自造V6；仍可跑环境/BC smoke，但不宣称主比较完成 |
| 单seed训练异常 | 保留失败，继续其他job；主结论标不完整 |
| test策略表现差 | 这是科学结果，不改环境/不换seed |
| 完成率高到都接近100% | 报告当前任务辨别力不足，不把“都通过”叫增益 |
| 冻结奖励核算过关但策略变差 | 保留P1核算结论；P2效用负结果单列 |
| 主方法胜BC、不胜强基线 | 不宣称图带来独有收益 |

## 14. 交付目录与必须补齐的图表

轻量报告根：

```text
artifacts/pathgraph_sarm/upgrade_v2/p2a_state_weighted_bc_utility_v1/
```

外置数据根`$DATA`存训练JSONL/NPZ、policy.pt、全部rollout，轻量报告只索引它们及哈希。

至少交付：

```text
protocol_lock.json / source_lock.json / split_manifest.json
training_matrix.csv / job_status.jsonl
weight_diagnostics.json / action_phase_weight_summary.csv
initialization_batch_parity.json
per_episode_policy_metrics.csv / per_loss_recovery.csv
per_condition_metrics.csv / paired_comparison.json
policy_control_audit.json / claim_to_evidence.csv
final/decision.json / final/report.md / final/next_stage.md
external_artifacts.tsv / result_manifest.json
```

`per_loss_recovery.csv`记录planned条件、observed_loss_id、首次失效、匹配restore、失效时任务进度、恢复后success；未触发loss、超时、未恢复保留，不能只收成功行。

报告解释“算术正确”和“学习有效”的区别；强基线胜出或所有方法无差异必须同样突出。

## 15. 最终汇报模板

```text
BASE / runner commit / result commit / branch / remote match
P1 frozen paths unchanged
benchmark = ACTION_CONDITIONED_ABSTRACT_SKILL_MDP_NOT_PHYSICS
training_data_episodes / transitions / failures / mixture
train-validation-test family disjoint
method source hashes / weight transform ID
methods x seeds completed / failed
same initialization and sample order
checkpoint rule = FIXED_LAST_STEP
learned policy actually controlled env = true/false
teacher takeover calls = 0
per method: family-macro success, conditions, recovery exposure/restore, steps
V6 - BC: difference and paired interval
V6 - count+events: difference and paired interval
V6 - cost-only and shuffled: exploratory comparison
effect interpretation / ceiling limitation / unresolved metrics
policy_utility_evidence
robot_policy_gain_claimed=false
physical_cycle_claim=NOT_EVALUATED
P1 historical confirmation fields unchanged
```

## 16. 科学边界与下一步

势差闭环核算是必要性质之一，不是图方法必胜的证明；PBRS文献讨论的策略不变性也不等于本轮加权BC性能保证[O3]。当前完整状态含人为验证的phase，故不检验图状态从视觉学出来的误差。

若本轮有效用，下一步再选一个现成可训练的模拟任务/既有带动作数据，扩展到更真实策略，而不是回头自研抓取器。若无效用，优先分析reward-to-weight映射、准备动作信用和强基线等价性；不要靠再写更多工程case取得通过。

**本阶段成功完成的标志，是拿到可信的训练与自主评估结果，而不是PathGraph必须获胜。**

## 17. 来源

[S1] rollinpioneer/graph，固定提交8e86b246…：P1 decision、scorer、state_builder、extra_methods以及tools/stage6/policy_env.py；详见`SOURCES.md`。

[O1] SARM原始v1：<https://arxiv.org/abs/2509.25358v1>。这里只借鉴奖励辅助示范筛选/重加权的用途，不借用原论文成功率为本项目背书。

[O2] imitation官方BC说明：<https://imitation.readthedocs.io/en/latest/algorithms/bc.html>。本包采用PyTorch小型离散策略参考实现，不要求部署imitation整个栈。

[O3] Ng, Harada, Russell (1999), Policy invariance under reward transformations：<https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf>。不得把已有势差思想当成新理论贡献。
