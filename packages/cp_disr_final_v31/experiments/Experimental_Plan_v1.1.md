# CP-DISR Revised Experimental Plan v1.1

2026-09-22 · Method v2.1.1 · Research Specification v3.1

**执行原则：替换对照、缩小第一批，不叠加实验。先11个新训练取得基本现象，再补齐至29；Auxiliary Candidate Q消融之后核心合计32。A_B需要时35。不自动重跑36个Presentation。**

## Source of truth与实际状态

本计划优先于历史实验预算和方法ID定义；原实验Agent Prompt的有限调参、真实Raw Archive和Private View保持。完整方法见research/v3.1，当前参数见interfaces/paper_method_manifest.yaml。对照替换理由和文献核验见review。此包不是已经接好环境的训练仓库，本轮没有执行任何Stage的机器人/RL/API部分。

所有stage状态初始NOT_STARTED。输入资源缺失时写BLOCKED并停止，不通过猜测填MUST_BIND。用户仅指定一个Stage时只执行该Stage；指定Part才可在其范围内顺序推进。next_stage只是建议，不是后台调度。3A分pilot与completion，未明确completion且未有2B报告时只做pilot；不在同一次默认3A里自动补完所有seed。

## Frozen方法、运行绑定与默认优化

固定：两类soft关系与原prompt/schema；effect_ref只作admission metadata；Action–Proposition合同图；Full四视图DP/zeroanchor/bounded residual；名义only-success且只读；独立真实终端奖励；实际a_t Q；共享E；当前无static prior/repair/memory/world model/teacher。

默认：graph4层128/bases4/mean/root/ReLU/LN/dropout0；GRU128；candidate attention128＋MLP256→128；beta=.5；lr3e-4；clip.2；GAE.95；cV.5；lambdaQ.1；entropy.01；gradclip.5；Adam；4epochs；rollout1024；seq16；minibatch64。Actor advantage按有效batch标准化，V/Q目标不标准化。一个去重optimizer，旧targets和历史mask固定，GRU前缀/图重算，不把旧可学习embedding作为新输入。

Actor prefix=false；loss有效transition均值；w只作回报核算。Gamma=2^(-d_seconds/H)，H取0A每个训练family5次合法reference成功时长中位数的最大值，最多10尝试/family；所有方法同H，holdout继承。reference校准不是teacher/BC数据，不进PPO、VLM few-shot或模型选择。独立任务原有deadline必须保留；自定义deadline默认2*T_ref，一次训练前修订最多3*T_ref、记录并重做0D，安全阈值/技能超时不放宽。

真实资源MUST_BIND：工作树/Git、simulator或robot、资产/对象/目标、controller与连续参数adapter、camera/感知/Verifier、安全授权、技能时限、实际时钟、软件lock/硬件、任务split、API地区/模型实际可用性/凭据引用、测试/训练/评价入口。不把当前聊天容器配置当用户实验机，也不把API公开页面当已获调用权限。

软件路线继续项目内轻量PyTorch recurrent SMDP-PPO；本版不重新猜历史冲突的补丁版本。采用已绑定profile时保存torch/PyG/CUDA等实际版本、import/forward/backward测试与依赖锁；profile尚不存在时0A阻塞，不自动改用SB3的无循环默认实现。

## 任务与split

主训练T_B/T_C，D0 smoke；T_A非冗余负参考/合同共享可选，T_D组合需求再绑定，T_E后置。每family预留train128、dev20、test-ID50、robustness50、composition50个case，实际资产必须存在且独立hash；这是池容量，不是必须跑完全部评价。0D的preflight case由独立development池有放回抽样，不使用test收益。API缓存按实际启用池物化，不按task名跨scene复用。

任务结构至少在合法reference轨迹上有两个≥2候选且后果真正不同的决策点。错误可恢复但有真实时间代价是主要模板；不要从合同删真实硬前置来制造prior作用。软边原始可错，不为凑非空缓存而生成到满意；关系真假、先验当前是否有用与神经响应大小分开。

## Seed与配对

训练seed先0，后补1/2；可选3/4需要明确扩展，不连续试到获胜。不同method在同task用配对initialization、environment、prior-dropout流，动作采样各独立实例。派生seed用SHA-256规范JSON（namespace='cp_disr_exp_v1_1',task,split,training_seed,case_or_episode,stream），不使用Python进程hash。测试case流不包含训练seed，使所有训练seed面对共同初始case；扰动流再加condition并独立于训练dropout。

共享模块按模块名派生初始化流，A_Q删除loss而不打乱网络其他初值。配对种子不保证闭环轨迹相同。保存Python/NumPy/Torch/CUDA RNG和可恢复状态；不能回滚真实环境时恢复记新attempt，不冒称bitwise续训。

## 预算与时钟

| Profile | Ncap技能 | PPO updates上限 | eval/checkpoint间隔 | dev episodes |
|---|---:|---:|---:|---:|
| Smoke | 16,384 | 16 | 4,096 / 4updates | 10 |
| Core/Ablation | 65,536 | 64 | 8,192 / 8updates | 20 |
| 仅必要的length修订 | 131,072总量 | 128 | 8,192 | 20 |

同task Tcap=Ncap*d_ref，d_ref取0A参考执行全部有效技能实际时长的中位数，非假定安全5秒；N/T先达停止。最后有效残片可更新，padding不计loss，外部cap截断bootstrap最后有效状态。真实失败尝试必须计时。训练、evaluation、reference校准、API调用各自记账。

扩长只在明确开发修订后发生；Full与CAT/消融必须匹配预算。继续某seed增加交互，不变成新的独立seed。参数/任务/H改变后需要显式新profile；旧0号与新1/2号不能凑三seed。

## checkpoint与evaluation

保存step0、每评价点、正常结束和安全可恢复中断点。checkpoint含模型/optimizer/RNG、有效N/T/update、code/runtime/config/cache/split/H hash以及归一化状态。开发选模score=真实起点discounted return的最近3窗口均值；并列按最差窗口、再按更早step，少于3点明确标注。test前冻结checkpoint清单。

最终test-ID每task/seed30个episode，可以同比较组扩到50；所有方法使用deterministic argmax和canonical ID tie-break，训练sample masked categorical。普通evaluation使用Original，B2/B1-K/B0不读prior。成功率、原始discounted return、原始AUC、成功条件时长与分母、失败原因同时报告。completion_time在0成功时为NA不是0。图显示可窗口3moving average，原数据/无平滑版并存，表/AUC绝不平滑。

主表默认全部预定0/1/2有效seed，数值正常的失败不能排除。Private View可另选illustrative任务/seed，但使用selected exploratory标签并保留完整来源，不取代总体比较证据。有限调参允许，不能改原始reward、success或结果CSV。

## 两种归档与日志

每真实run目录为runs/<stage>/<task>/<method>/seed_<s>/<UTC_configsha8>/，至少保存resolved_config.yaml、manifest.json、train_metrics.csv、eval_episode.jsonl、decision_log.jsonl、failures.jsonl、tuning_history.jsonl、checkpoints/与run_summary.md。计划行只有planned_id，真实执行后才赋run_id。

每update：N/T/update、loss_actor/V/Q、entropy、KL/clip、gradnorm、learning_rate、candidate/relation数量、DK/DH/DP RMS与exactzero率、Delta分位/饱和率。每决策：真实snapshot、事实T/F/U与证据引用、candidate IDs/mask、实际a、名义patch摘要、D摘要、base/Delta/final分数、r/d/Gamma/w/flags、controller exit和verified outcome。完整张量只在mechanism抽样保存，原输入可重算。

Raw保留所有失败、弃用、续训与中断及hash；Paper View保存task/seed/checkpoint/metric/source/aggregation/smoothing/subset/selection_reason，不覆盖原数据。主结果不能来自只含选优seed的表而假装未经选择。

## 有限调参与停止

排查实现→candidate/mask→reward→合同→Verifier→DK→DP→residual，再lr→length→entropy→beta→lambdaQ。单次只改变一个，1B最多6个新增job；候选lr{1e-4,3e-4,5e-4}、entropy{.005,.01,.02}、beta{.25,.5,1}、lambdaQ{.05,.1,.2}，不是全grid。zero reward不默认加shaping、课程或BC。

PASS表示阶段完成不是Full赢；NEEDS_RERUN针对实施错误/明确未完成目标，正常接近或负结果可PASS_WITH_NOTES。无效mask、真实事实被名义污染、target泄漏、非法安全执行必须停止。所有stage都不阻塞论文定义写作，但缺真实数据的经验句不能写成结果。


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


# 24. Final Baseline Matrix

| ID | 最终定义 | prior | 名义后继 | 默认位置 |
|---|---|---|---|---|
| B0 | 两层SAB的非显式图集合策略；完整原始合同/事实/目标/候选字段 | 无 | 无 | Tier1主阶梯 |
| B1-K（表内仍可简写B1） | 同B2上下文和head，$u_i^K=\phi_K(c_i,0)$ | 无 | 不计算 | Tier1主阶梯 |
| B2 | 同一$\phi_K(c_i,D_i^K)$，独立训练 | 始终空 | 合同后继 | Tier1主阶梯 |
| Full | 现有合同干预＋纯$D_i^P$＋零锚有界残差 | episode80/20 | 四视图 | Tier1主方法 |
| A_CAT | Full合同分支不变，prior读四视图拼接及合同重复anchor | episode80/20 | 同四视图 | 核心必跑，但先1seed |
| A_Q | Full仅置$\lambda_Q=0$ | episode80/20 | 同四视图 | Tier2训练机制 |
| A_B | Full仅移除tanh，保留$\beta$尺度和zero anchor | episode80/20 | 同四视图 | Tier3，仅有界收益claim需要 |
| B1-H | 旧B1：只读当前GK/GH的静态图策略 | episode80/20 | 无 | 可选，不默认跑 |
| A_DD | 旧非空DH、空prior显式0消融 | episode80/20 | 同四视图 | 可选，被A_CAT替代必跑位置 |

**B1的精确匹配。**$c_i=[z^o,e(a_i),\operatorname{Pool}(Z^K)]$与B2一致；B1将DK输入slot置零，而不把ZK或额外action-local embedding塞入该slot。这样B1→B2只增加已定义的候选变化特征及其计算。B1不是宣称最强的任意current-graph网络，而是匹配的no-intervention对照。换一种更丰富的current-graph readout是另一个架构实验，不默认加入。

**归因等级。**B1→B2是该实现下的受控干预信息对照；B0→B1是非显式图attention与图编码的系统比较，不声称仅“是否有图”一个自由度。B2→Full检验先验交互路径与其训练协议的整体增量；Full→A_CAT检验纯交互限制与四视图融合，不是同一输入信息。共同数据预算不意味着相同轨迹；相同结构不等于所有参数完全一样。

除A_Q，全部主配置保持匹配的实际动作Q监督、同reward/感知/合同/mask/时钟与训练预算。所有使用prior的方法共享相同cache和80/20协议。报告有效trainable参数、前向视图数、延迟和内存；不使用死参数填充冒充等容量，不给baseline额外oracle或剥夺原始合同字段。

## 24.1 B0：最终两层Set Transformer实施合同

输入为当前事实（含T/F/U与goal sign）、签名目标、全体grounded技能及其原始PRE_POS/PRE_NEG/ADD/DEL/UNKNOWN文字项记录、当前候选角色和同源实际观测。每条合同记录包含owning action、predicate、参数的有序角色/类型及一致对象绑定信息；不得只保留效果类别而丢掉“谁影响谁”。任意显示ID不当作可学习语义标签；对象特征来自所有方法共同可见的对象表/观测。

所有token通过共享字段投影进入$d=128$。使用两个标准SAB：multi-head self-attention，4 heads，逐token FFN128→256→128、ReLU、残差、LayerNorm，dropout0，无集合位置编码。候选query对全部编码token做一层cross-attention，输出128，再与共同obs/GRU及候选上下文用MLP256→128融合。V/Q/Actor各自head与其他方法保持相同输出角色。SAB/MAB属于现有Set Transformer组件。[W15]

attention只使用padding mask，不输入graph adjacency、PRE/ADD稀疏attention mask、最短路、层级答案或名义赋值后的事实；不调用NOMINAL_PATCH。token包含原始合同关联，因此它可以通过attention学习关系；“非图”指无显式graph message-passing拓扑，不是禁止一切关系计算。ADD/DEL原字段也不等于已经应用当前T/F/U、条件效果和全局frame后的名义后继。

实现前冻结宽度/层数，不按Full领先与否搜索B0容量；参数差异透明报告。若已有更强且信息匹配的标准attention基线，可在未跑任何主结果前替代一次，但不能增加另一项必跑。第一版采用上述唯一实现。


# 25. Core Ablations

## 25.1 A_CAT：四视图融合替代纯交互输入

按goal/global对应行，沿**feature维**拼接，不能把goal行平铺成固定任务长度：

$$
C_i=[Z^K,\widetilde Z_i^K,Z^H,\widetilde Z_i^H],\qquad
C_i^0=[Z^K,\widetilde Z_i^K,Z^K,\widetilde Z_i^K].
$$

$$
\boxed{u_i^{P,CAT}=\phi_{CAT}(c_i,u_i^K,C_i)-\phi_{CAT}(c_i,u_i^K,C_i^0),}
$$

$$\Delta_i^{CAT}=\beta\tanh(w_P^\top u_i^{P,CAT}).$$

合同FK、c、E、名义patch、goal mask、shared candidate output、V/Q/Actor功能、优化参数和数据预算均与Full相同。CAT必须从头独立训练；“相同head”指下游架构/宽度，不表示两个独立run共享已经训练好的参数。

输入首投影4d→d，而Full为d→d；之后使用相同候选attention/MLP/head。若只此处增宽，CAT额外参数为$3d^2=49,152$（d128，不含相同bias）。实际层表不同则重新精确计数；不谎称严格等参数。Full是在共同输入上固定使用$[I,-I,-I,I]$对比的受限归纳偏置，CAT可学出这一映射，但有限训练不保证自动学会。

无prior时C=C0，残差严格0。空patch或可加分离的静态prior响应则**可以非零**，这是比较目的，不能额外用DP=0 gate把CAT变回Full。锚定到全零而非合同重复视图不是本消融定义。CAT不允许从effect_fact_ref额外读几何或语义标签。

本消融回答：明确丢弃可加分离静态成分是否值得？它不单独隔离“减法计算”与容量、完整信息保留的全部区别。若CAT在同预算主任务/后续评价中稳定更好，纯交互优越性不成立，应报告、缩小claim或在未来v2.2重新选择融合；不能为了保住Full挑选seed/删task。写作仍可继续，但结果结论须改变。

## 25.2 A_Q和A_B

A_Q仅将$\lambda_Q$设为0，保留初始化顺序/网络接口，不把Q输出送Actor。它是保留Q作为正式训练设计时最小必要的训练机制对照，默认只在TC用3seed。

A_B仅使用$\Delta_i=\beta w_P^\top u_i^P$，无tanh、无补回clamp或归一化。零锚定保留。若论文只声明代数上界而不声称其性能/鲁棒收益，A_B不列必跑；一旦主张bounded带来行为收益，就需要它。

A_DD仍可定义：R非空用DH，R为空用0；但不与A_CAT一起强制运行。旧B3、static-prior独立模块、Full-clean、effect-anchored图等不列当前默认任务。

## 25.3 对照复用

Full可复用先前完全匹配的task/split、code行为、runtime/软件、cache、seed、common hyperparameters、预算、时间H与checkpoint规则的run。复用写control_reuse_manifest，不复制成新训练。新参数profile改变后，旧seed0不能与新seed1/2凑成同profile三seed；受影响范围显式补跑并计成本。其他Stage不得自动扩实验。


## 29.2 Revised Experimental Route v1.1

保留已有Stage ID语义大体兼容，并增加0D。**0A/0B/0C/0D只阻塞依赖它们的真实运行，不阻塞Problem/Method写作。**本轮只生成计划与代数文件验收，不宣称已经执行这些Stage。

| Stage | 目标 | 默认新增训练 |
|---|---|---:|
| 0A | 真实接口/软件/权限/任务绑定；每训练family最多10次reference尝试取得5次成功，确定H与d_ref | 0 |
| 0B | 原生产不变量＋A_CAT null、B1无后继、Set permutation、unweighted Actor及时长单位检查 | 0 |
| 0C | D0/TB/TC各8个初始dev场景，固定VLM协议和cache质检 | 0 |
| 0D | 每family100个uniform masked episodes，低成功时扩到200；必要时50个contract-aware诊断 | 0 |
| 1A | D0，B2/Full，seed0，16,384转移上限 | 2 |
| 1B | 仅学习/数值明显异常时，有限单项优化，最多6个额外job | 条件性，非默认 |
| 2A | TB/TC × B0/B1-K/B2/Full × seed0，65,536/运行 | 8 |
| 3A-pilot | TC × A_CAT × seed0；Full复用2A | 1 |
| **第一暂停点** | **11个默认训练；形成探索Results；不等3seed全部完再写** | **累计11** |
| 2B | 原两任务四方法补seed1、2；不重跑已匹配seed0 | 16 |
| 3A-completion | TC A_CAT补seed1、2；Full继续复用 | 2 |
| 2C | 按明确source生成主表/主曲线 | 0 |
| 3B | TC A_Q，0/1/2 | 3 |
| 3C | A_B仅在bounded经验收益claim需要时，TC，0/1/2 | 可选3 |
| 4A/4B | 同checkpoint Actor依赖与先验变化；Full/A_CAT配对 | 0 |
| 5A | 一种已有source模型对应的实际未见组合 | 0；无source则不自动增训 |
| 6A/6B/6C | Raw审计、展示manifest与论文包 | 0 |

第一批11个，上限622,592训练transition；补齐两任务三seed和A_CAT后累计29个，上限1,802,240；再含A_Q为32个，上限1,998,848；A_B也做则35个，上限2,195,456。表内不含评价、API和条件性重跑，均另记成本。不同stage复用同一Full只计一次。

**Tier1：**B0/B1-K/B2/Full及A_CAT；先单seed再补齐。**Tier2：**A_Q及同checkpoint诊断，必要的一种组合评价。**Tier3：**A_B、B1-H、A_DD及资源/表示诊断。不存在7方法×所有task×多seed一次性全跑，也取消无条件新建36个Presentation训练的要求。

总训练预算在旧记录存在冲突，本版**重新明确选择**上述16,384/65,536，未声称它们本来就是唯一冻结旧值。smoke每4096评价/保存，主训练每8192评价/保存；rollout1024，因此16/64个PPO updates上限。smoke最早8个updates后满足可用性可停；主对比预算完成或明确数值/安全错误才停止，不因Full先赢单独提前停。

每family预算还设$T_{cap}=N_{cap}d_f^{ref}$，$d_f^{ref}$为0A参考执行中实际每技能时长中位数。所有同family方法共用Ncap/Tcap；先达到一个cap即停，最后不足buffer用有效mask处理。实际任务秒数、技能数、control ticks、wall time分别保存。主学习曲线在共同实际时间区间计算raw AUC，并附skill-transition横轴；不得向未运行尾部外推。


# 20. 第一批实验

第一批先做0A–0D实际运行准备，然后只运行：

| 任务 | 配置 | seed | Ncap | 新run |
|---|---|---:|---:|---:|
| D0 smoke | B2、Full | 0 | 各16,384 | 2 |
| TB | B0、B1-K、B2、Full | 0 | 各65,536 | 4 |
| TC | B0、B1-K、B2、Full | 0 | 各65,536 | 4 |
| TC核心消融 | A_CAT；Full复用TC | 0 | 65,536 | 1 |
| **合计** | | | **622,592上限** | **11** |

先写探索Results，不用等显著差异。Single-seed只能给基本现象和排查依据，不能假装稳定平均效果。A_CAT包含在第一轮，使核心假设早接受检验，但不抢先跑A_Q/A_B/Full-clean/effect anchored。

任务/接口有错误则修复并记录；只是Full不领先不算实现错误。没有暴露真实成功时，暂停多seed扩展、做有限诊断，而不是把坏结果直接抹去或自动加reward。


# 21. 第二批实验

核心输入、配置和预算不变时，TB/TC四方法补seed1和2，共16个；TC A_CAT补seed1和2，共2个。加第一批后累计29个，形成两任务、三seed主阶梯及核心CAT对照。**无需另建36个Presentation训练。**

保留Auxiliary Candidate Q时，TC的A_Q补0/1/2共3个，累计32个。若主张bounded带来经验收益，TC A_B再3个，累计35个；仅写代数界则不必。

同checkpoint机制与先验测试、已有source支持的一种组合holdout均不重训。没有合适source checkpoint、未见组合只是重命名或必须新增技能时，5A标NOT_APPLICABLE/未执行，不自动造新任务矩阵。

每阶段执行到报告停止。不同环境profile、预算H或超参已经变更的旧run不能凑在新三seed中；重新训练需求必须列新run清单和原因，不能藏进“复用”。


# 22. 哪些实验删除或降级

**从默认队列删除：**旧A_DD必跑、旧B1-H必跑、旧B3、原DeepSets额外对照、Full-clean、effect-anchored图、candidate-local readout、全部任务一次性7配置、无条件36次新Presentation运行。

**降为条件性：**A_B仅对应经验bounded claim；A_DD只在CAT结果需要分辨“静态信息”与“减去DK”时考虑；B1-H只在评审确实要求当前prior-only基线时考虑；梯度余弦/更大图/更长链仅诊断。

**保留但不增加训练：**同checkpoint的DP0/remove（同状态应等价）、shuffle、删除、target-swap；已完成的Original评价精确条件一致可引用，不重复收费。Irrelevant无独立证据或满8边预算则NA，不额外造资产。

没有证据证明某个理论性质带来性能收益时，选择收缩claim，而不是为了“理论box每项必须有一组实验”无限加run。


# Stage索引

- Stage 0A: Environment, Interface and Time Binding — `stages/stage_0a.md`
- Stage 0B: Production Mathematical and Interface Invariants — `stages/stage_0b.md`
- Stage 0C: Frozen VLM Cache Sanity — `stages/stage_0c.md`
- Stage 0D: Reward Exposure and Decision Structure — `stages/stage_0d.md`
- Stage 1A: One-task Smoke — `stages/stage_1a.md`
- Stage 1B: Conditional Minimal Tuning — `stages/stage_1b.md`
- Stage 2A: Core Ladder Pilot — `stages/stage_2a.md`
- Stage 2B: Core Ladder Seed Completion — `stages/stage_2b.md`
- Stage 2C: Main Figures and Tables — `stages/stage_2c.md`
- Stage 3A: Four-view Concatenation Ablation — `stages/stage_3a.md`
- Stage 3B: Auxiliary Candidate Q Ablation — `stages/stage_3b.md`
- Stage 3C: Conditional Bound Ablation — `stages/stage_3c.md`
- Stage 4A: Same-snapshot Actor Dependence — `stages/stage_4a.md`
- Stage 4B: Missing-prior and Perturbation Evaluation — `stages/stage_4b.md`
- Stage 5A: One Compositional Holdout — `stages/stage_5a.md`
- Stage 6A: Raw Archive Audit — `stages/stage_6a.md`
- Stage 6B: Private Paper View — `stages/stage_6b.md`
- Stage 6C: Final Result Package — `stages/stage_6c.md`

# 版本与写作

方法训练profile为2.1.1。所有stage可与写作并行；当前paper定义不等待性能达到门槛。本计划不承诺CCF-A录用或Full必胜。运行source、精确预算、原始失败、选择说明是事实链；没有记录不得称为已执行。
