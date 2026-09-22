# Stage 0D — Reward Exposure and Decision Structure

## Scope / Purpose
检查稀疏奖励是否暴露，不建立5%学习定理。

## Inputs / Preconditions
0A/0B；0C产物可供关系统计。共同输入为本包research规范、interfaces manifest及Experiment Plan v1.1；目标runtime不得有执行关键MUST_BIND。

## Frozen / Tunable
所有方法定义与权限按共同合同冻结。除1B的有限队列以及明确记录的资源分块，不能当场新增优化变量。消融只改变自己的字段。

## Run Matrix / Budget
不产生默认新训练。实际evaluation/reference/API预算见本阶段细则，不能把它们计作RL训练run。

## Evaluation / Required Logging
0是本阶段指定evaluation默认episode数（0表示不做策略评价）；按主计划精确condition/split执行。每个真实run记录配置/source/Git/cache/H、实际时间、mask与所选动作、损失与D/Delta；无训练量不填伪造0。

## Extra protocol
# Stage 0D — Reward Exposure & Decision-Structure Audit

## 目的和边界

不再采用“random success必须≥5%”硬门槛。随机策略成功是当前初始化、deadline、动作分支和执行器下的**奖励暴露代理**，不是PPO可学习性的必要或充分条件。检查在合法仿真/受限技能接口内进行；真实机器人若未获安全授权，BLOCKED，不以随机探索之名绕过安全。

## 运行合同

每个启用训练family，从预留dev/preflight池均匀有放回采样初始化，使用独立seed流，合法candidate上uniform采样。先100个完整episode；若有效成功≤2，扩至200；若在100时已≥3且覆盖至少2个初始case，不再扩。保留计划数、有效数、基础设施异常数与实际case分布，不能把失败删掉。

若200个仍≤2成功，最多另跑50个**contract-aware random**诊断：在合法且名义patch非空候选上uniform采样；该集合为空时回到全部合法候选。它只是检查大量空patch/无效重复是否淹没探索，不声称其最优或总比uniform强，特别观察动作的空patch可能有真正价值。不用VLM评分、最优路径、Q或隐藏正确动作；不把诊断数据送入PPO/BC。

## 判断

- **PASS（工程暴露充分）：**默认至少3次真实成功且至少2case，存在合法恢复、至少两个真正有后果差异的选择点，关键日志完整；3不是理论阈值。
- **PASS_WITH_NOTES（低暴露）：**1–2次成功或集中单case，但已有独立成功reference和合法任务路径。允许先进行单seed smoke，不启动多seed矩阵；不称“不可学”。
- **NEEDS_TASK_REVIEW：**uniform200和可选contract-aware50仍无成功，且重复失败原因明确；用stage status=NEEDS_RERUN加该reason。暂停扩主训练，完成一轮有限任务/接口核查。
- **BLOCKED：**安全/观测/奖励/合同错误、没有合法可执行成功路径或现实资源未绑定。只阻塞执行，不阻塞论文写作。

若独立同分布Bernoulli近似适用，0/200的单侧95%成功率上界约为1.49%，不是0；若p=.01，100次全无成功的概率约36.6%，200次仍约13.4%。因此固定5%会错误排除部分可学习任务。实际相关初始化/非独立执行时不能把上述区间当严格统计保证。

## 必填观测

success有效数/分母与初始case覆盖；每episode决策点数、≥2合法候选比例、平均/分位branching factor、强制动作比例、空patch比例、重复loop/失败类型、成功/失败耗时；raw prior非空率与关系数；改变Fact到目标路径覆盖；实际非空prior+changed-patch中DP RMS与residual，后两者来自单独未训练forward而非策略成绩。

候选数≥2仅表示选择机会，不能证明长期后果不同。有效结构决策由任务定义、已注册可执行路径和受控开发执行证据确认，不由VLM或Full价值估计充当真值。

## 有限调整顺序

检查reward/flags/time→合同/mask/控制器/观测绑定→恢复技能与无意义分支→deadline是否留了真实恢复机会→减少一项无关决策或使用较短的训练模板。benchmark固定deadline不擅改；自定义任务默认deadline=2*T_ref，仅允许在主训练前一次改至3*T_ref并记录版本、重跑0D。安全阈值与单技能超时不放宽。仍困难时如实标低暴露/暂不启用该family，不直接加shaping、专家暖启动或复杂课程。

## 产物及停止

输出`stage_0d_summary.md`、`random_episode_log.jsonl`、`reward_exposure.csv`、`decision_structure.csv`、`task_revision_log.jsonl`、status JSON。日志中preflight/source标签与RL训练分离。完成100/200及条件性50的明确预算后停止，不自动进入下一Stage。


## Output Directory / Required Files
`runs/stage_0d/`；阶段级`reports/stage_0d_summary.md`、`status/stage_0d.json`、实际run/source清单与异常/调参记录。训练阶段另含train/eval/decision日志、checkpoints、raw metrics和图表；诊断阶段含条件适用性与原始episode；打包阶段含source/hash/selection。

## PASS / STOP
100episode，≤2成功扩至200；仍低则最多50contract-aware。至少3成功/2case为工程绿灯；1–2可带注释先smoke；0则有限核查，硬失败仅安全/接口/不可达。。状态采用NOT_STARTED/RUNNING/PASS/PASS_WITH_NOTES/NEEDS_RERUN/BLOCKED；NOT_APPLICABLE和SKIPPED放execution_reason，不另造状态。

## Failures / Rerun / Writing
安全/接口/数据污染即停并保留证据；正常负结果不删除。重跑新attempt不覆盖旧run；Full对照只有所有匹配字段一致才复用。所有Stage不阻塞论文方法写作，缺结果只限制具体经验claim。

## Agent Execution Template
1. 确认授权范围为Stage 0D及其phase，读取本卡和主计划，不默认跨Stage。
2. 验证前置报告、source/hash和执行关键runtime绑定；缺失写BLOCKED。
3. 解析本阶段真实run/evaluation矩阵，区别planned ID、已有control和新任务。
4. 冻结本轮task/split、seed、H、mask权限、配置与条件清单。
5. 只在已授权实际工作树调用真实入口；本包不假装包含训练器。
6. 按本阶段预算执行并持续写原始日志；失败、安全和不适用条件分别处理。
7. 按既定规则选择checkpoint/聚合指标，保留全部有效失败与预算分母。
8. 生成本阶段summary、必要图表与source引用；未生成材料写NOT_CREATED。
9. 更新stage状态与实际成本账本，记录改动和对照复用。
10. 提出下一Stage建议后停止；没有明确授权不自动继续。
