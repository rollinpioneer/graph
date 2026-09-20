# CP-DISR v2.1 Minimal Experimental Execution Plan

**文档版本：EEP v1.0 · 2026-09-20**  
**方法版本：CP-DISR / M1 v2.1**  
**用途：私人论文的分阶段实验 Agent 工作手册。**

> 核心执行原则：先用少量真实交互打通 pipeline，再获取核心比较与双差分证据；性能一般不阻塞写作。默认第一批最多新增29个训练 job，默认完整私人论文路线71个训练 job。原始实验完整保留，论文展示按显式 selection manifest 选择。

## 阅读入口

首次接手阅读第1–8章，再读取相应 Part / Stage。随后用户只需指定 `执行 Stage 1A` 等指令，Agent即可根据本手册及已绑定的项目文件工作。每个 Stage 都列出输入、运行矩阵、预算、评价、文件、停止条件和执行步骤；不依赖重新阅读旧对话理解关键规则。

**可直接执行是有前提的：**项目代码、任务资产、相机、技能控制器和评价器必须已经真实绑定。本手册不能凭空生成不存在的机器人能力。缺少这些资源时，Agent先完成本 Stage 可做的检查，生成 `BLOCKED` 与具体缺项，不假造环境或宣称已训练。随本手册交付的工具是计划检查、清单和预检工具，不是完整机器人训练器。

---

# 1. 文档权威、来源与执行边界

## 1.1 权威顺序

**实验目的、展示规则、规模与调参权限：**本轮 `CP_DISR_v2.1_Final_Experimental_Agent_Prompt.md` 优先于旧研究规范的正式投稿式要求。[S1]

**算法、奖励、图、数据权限与三态逻辑：**继续使用 `CP_DISR_M1_Research_Specification_v2.1.md` 及接口包。[S2–S3] 本计划不恢复 SOFT_ORDER、静态 prior、50/20/15/15训练、图奖励、VLM微调或世界模型。

**本次补齐的执行默认：**软件锁定候选、任务结构模板、具体预算、seed派生、调参次数上限、checkpoint规则、指标算法和阶段状态规则。它们是为了可执行而作出的设计选择，不是已经验证的最佳参数，也不是已有实验结果。

**外部核对：**仅用于软件接口、版本和API公开能力；不替换用户的方法设计。软件版本被核对为可查的发布版本，不代表整套依赖已在目标机器安装并联合验收。[W1–W10]

## 1.2 私人论文与两套视图

`Raw Experiment Archive` 保存全部真实运行、失败、尝试过的seed、config、checkpoint、评价日志、缓存、代码和环境hash。`Private Paper View` 可选任务子集、三个展示seed、各方法自己的开发集checkpoint、有解释力的metric和真实qualitative episode。

可以有限调参、重跑、缩短或延长预算，也可以不把所有失败运行放进主文。不修改原始reward、success或episode结果，不编造baseline，不把挑选后的结果称为未选择的总体均值。挑选最清楚的成功case可标为“illustrative case”；挑选较好seed可标为“selected-seed view”，不自动称为“随机代表性结果”。每张图表保留可追溯的原始文件与选择理由。[S1，§0与§25–27]

不要求默认统计显著性，不以“Full第一”作为完成条件。结果为负也可以完成一个实验Stage；论文主张按实际现象收缩。展示选择提高叙事清晰度，不能保证方法变好。

## 1.3 单阶段、整Part和本轮起点

| 用户指令 | 授权范围 | 停止点 |
|---|---|---|
| 执行 Stage XX | 仅该Stage；检查前置条件、完成本阶段准备与作业、报告 | 写入状态后停止，不启动下一Stage |
| 执行 Part IV | 明确授权本Part的3A→3B→3C，按顺序逐一写状态 | 本Part结束即停；遇BLOCKED/NEEDS_RERUN先停 |
| 执行 Part I | 0A→0B→0C；不训练 | 0C结束即停；0A缺资源则在0A停止 |
| 继续/补跑 Stage XX | 只补该Stage列明的缺口；完整保存旧revision | 本Stage完成即停 |
| 仅生成计划 | 不安装软件、不调用付费模型、不训练、不移动机器人 | 计划交付即停 |

“Next Stage”只表示推荐入口，不是自动执行许可。本轮附件要求生成计划后开始Part I；当前先完成0A的**资料盘点与本地预检**，真实训练host/平台未绑定时止于0A，不能把聊天沙箱当成用户实验机。任何真实机器人动作还必须满足本地既定安全许可；本手册不是安全授权。

Part IV被整体授权时，3C不因总体优先级较低而被默默跳过；单独按证据缺口执行时，3C可不选。

## 1.4 状态与PASS含义

允许 `NOT_STARTED / RUNNING / PASS / PASS_WITH_NOTES / NEEDS_RERUN / BLOCKED`。不额外引入 `FAIL` 或 `SKIP` 枚举：跳过1B写 `PASS_WITH_NOTES`、`skipped=true`、`reason=DEFAULT_SMOKE_PASSED`；尚未被授权的可选阶段保持NOT_STARTED。

`PASS`表示阶段数据和交付满足要求，不表示方法有效性已证明。`PASS_WITH_NOTES`可用于真实零成功seed、负向比较、小样本或明确记录的预算限制。`NEEDS_RERUN`表示可修复代码/数据错误使当前证据不宜使用。`BLOCKED`表示资源、权限、资产或关键前置缺失。`STOP`是结束执行的动作，最终仍写入上述之一。

状态文件保留 `method_version, plan_version, git_hash, source_hashes, started_at, completed_at, runs_completed, runs_failed, runs_reused, selected_tasks, selected_config, issues, tuning_changes, next_stage, skipped, execution_scope, readiness`。上游修订不删除下游结果，只标记其dependency hash已过期。

---

# 2. 冻结方法与七个配置的实施定义

## 2.1 不允许在实验中改动的方法核心

VLM只读独立任务、初始图像和注册ID/合同摘要，离线缓存；不训练、不在线重规划。关系只有 `SOFT_SUPPORTS` 和 `SOFT_RELEVANT_TO_GOAL`，必须带 `effect_fact_ref`，它是source已有ADD/DEL效果的校验元数据，不是新的神经输入或硬前置。

合同图只含Action、Proposition，PRE_POS/PRE_NEG/ADD/DEL及反向消息边；目标是命题上的goal sign及固定goal_refs；UNKNOWN不是FALSE。合同、mask、独立task_success、真实事实均不由VLM决定。

四路共享标准R-GCN，默认128维、4层、4 bases、mean、root=True、ReLU、逐节点LayerNorm、dropout=0。名义成功patch只改临时事实，不改实际观测、几何、时钟、attempt count、奖励和真实Fact Store。

\[
D_i^K=E(\widetilde G_i^K)-E(G^K),\qquad
D_i^H=E(\widetilde G_i^H)-E(G^H),\qquad
D_i^P=D_i^H-D_i^K.
\]

\[
c_i=[z^o,e(a_i),\operatorname{Pool}(Z^K)],\quad
u_i^K=F_K(c_i,D_i^K),
\]
\[
u_i^P=F_P(c_i,u_i^K,D_i^P)-F_P(c_i,u_i^K,0),\qquad
\ell_i=b_i+B\tanh(w_P^\top u_i^P).
\]

这里上式 `u_i^K/u_i^P` 为候选结构特征；所有零锚定分支使用同一上下文和goal mask。**空prior直接复用合同编码，保证 \(D^P=u^P=\Delta=0\)**；不是两次随机前向后期待近似相等。prior head最后不加bias或外部无界倍率。

GRU只读实际观测、独立目标、核心事实、执行摘要，不直接读VLM表示。V、Q输出head独立，共享特征。推理不需要调用Q来做max-Q选择。[S2，§4–14]

## 2.2 时间、奖励与目标的完整约定

统一计时单位为秒。令真实技能决策区间时长为 \(d_t>0\)，默认每1秒折扣0.99：

\[
\Gamma_t=0.99^{d_t/1\mathrm{s}},\quad
r_t=\sum_k 0.99^{\Delta\tau_{t,k}/1\mathrm{s}}\rho_{t,k},\quad
w_0=1,\ w_{t+1}=w_t\Gamma_t.
\]

\(\rho\)只在独立确认整项任务首次成功时为1，其余为0；实际成功发生时刻相对区间起点的偏移必须保留。若平台只在技能结束边界交付奖励，adapter明确记录该边界事件时间，不能暗用名义成功时刻。完整episode return由真实奖励时间计算，主success是独立布尔指标，两者不可混为一谈。

\[
\delta_t=r_t+\Gamma_t(1-\mathrm{terminated}_t)V_{old}(X_{t+1})-V_{old}(X_t),
\]
\[
\widehat A_t=\delta_t+\Gamma_t\lambda c_t\widehat A_{t+1},\quad
y_t^V=\operatorname{sg}[V_{old}(X_t)+\widehat A_t],
\]
\[
y_t^Q=\operatorname{sg}[r_t+\Gamma_t(1-\mathrm{terminated}_t)V_{old}(X_{t+1})].
\]

\(c_t\)只在下一条记录是同一episode的连续转移时为1；技能超时通常不等于任务终止。任务期限耗尽是terminated；有效外部截断与buffer切段使用最后有效状态bootstrap，不读reset后的状态。终止与截断的bootstrap区别沿用Gymnasium接口语义。[S2，§11；W6]

\[
\mathcal L=-\mathbb E_{w_t}[L_{PPO,clip}]+0.5\,\mathcal L_V+
\lambda_Q\mathcal L_Q-0.01\,\mathbb E[H(\pi)],
\]

V/Q使用Huber，delta=1，Q只gather实际执行动作；所有targets detach，未执行候选不填0、不贴虚构标签。actor对每个minibatch的有效transition使用 `sum(w*surrogate)/sum(w)`，V/Q和entropy使用有效transition的普通均值。优势标准化只给actor，不改V/Q targets。共享参数只属于一个Adam optimizer。

## 2.3 四个主配置与三个消融

| ID | 可用结构与实施默认 | prior | Q | 解释边界 |
|---|---|---|---|---|
| B0 | 同样的观测、目标、候选和原始合同字段；按typed fact/contract tuple做共享MLP＋集合池化；不用图卷积或名义后继 | 不读取 | 与Full同真实动作target | 非图系统参考；Full差值不只等于双差分收益 |
| B1 | 编码当前GK/GH，不生成名义图；用当前ZK及ZH−ZK做匹配候选读出和零锚定残差 | 80/20 | 相同 | 当前图读取参考；其静态关系输入只属于baseline，不进入Full |
| B2 | Full的合同通路GK、DK；独立从头训练；prior始终空 | 空 | 相同 | 不是对Full测试时置零 |
| Full | 四路编码、DK/DP、零锚定有界残差 | 80/20 | \(\lambda_Q=0.1\) | CP-DISR v2.1 |
| A_DD | 保持Full其余结构；非空prior时prior输入改为DH，空prior时显式0 | 80/20 | 相同 | 精确“无第二次差分”，不是单图B3 |
| A_Q | Full，仅\(\lambda_Q=0\)；保留Q输出便于同维度诊断但不参与loss | 80/20 | 不训练Q | 不把未训练Q误差当同等质量预测 |
| A_B | Full，仅把\(B\tanh(s_i)\)改为\(B s_i\)；保留相同B数值 | 80/20 | 相同 | 幅度上界消融；不能推理时临时改B冒充重训练 |

B0/B1的上述具体readout是本计划新增的实现默认，原规范只定义其功能角色。实现时记录参数量，不强迫精确相等而增加模块；差距>25%先解释，必要时只适配baseline的MLP宽度。不要故意剥夺B0的任务目标、合同字段或实际观测。

默认标准候选读出：单头scaled dot-product attention，query由候选context投影到128维，key/value来自有效goal行和global行；接两层128维ReLU MLP。层数/宽度、归一化与参数初始化写进model_manifest。B0使用同样深度的MLP和masked mean，不按任意对象ID学习语义。B1用同形状的当前结构量代替差分，不计算未发生的后继。所用具体架构源码hash是最终运行依据。

---

# 3. 软件工具链与唯一RL实施路线

## 3.1 唯一主路线

**选择D：项目内轻量、透明的PyTorch recurrent SMDP-PPO。**CleanRL只作为PPO循环与日志的阅读参考，不直接引入它的固定动作/观测代码、LSTM、reward normalization或完整默认超参。[W4]

| 路线 | 当前决定 | 依据 |
|---|---|---|
| A．沿用现有PPO | 不作为默认；只复用经核对的通用组件 | 当前附件没有已验收的v2.1训练源码；旧实验不能证明新recompute/SMDP接口已有 |
| B．SB3 / sb3-contrib | 不选为主训练底座 | 官方Recurrent PPO以LSTM为主；MaskablePPO文档不支持recurrent policy，不能视为现成GRU＋mask组合。[W5] |
| C．CleanRL | 仅参考 | 单文件实现可审计，但本项目仍需改GRU、变长候选、SMDP target及图重算。[W4] |
| D．轻量PyTorch PPO | **采用** | 在一个清楚的collector、buffer、targets、loss、recurrent模块中显式实现上述接口 |
| E．其他大框架 | 不引入 | 没有当前必须解决的新需求 |

这不是新RL算法贡献；它是最小实施容器。不得因选择自写循环而省略T01–T25验收。

## 3.2 推荐锁定候选

| 项目 | 推荐值 | 绑定/验证要求 |
|---|---|---|
| OS | Linux x86_64，Ubuntu 22.04 LTS兼容环境 | 实际发行版、kernel、容器digest MUST_BIND；不重装用户现有环境 |
| Python | 3.11.13 | 为历史稳定兼容候选，不称最新；正式host准确patch写锁。[W1] |
| PyTorch | 2.7.1 | 官方有cu126及CPU wheel；不与任意torchvision混装。[W2] |
| CUDA wheel runtime | cu126（12.6） | 目标GPU架构与driver MUST_BIND；新GPU不支持时在0A更换整套兼容profile，不在运行中偷偷更换 |
| torchvision | 0.22.1，仅现有视觉前端确实依赖时安装 | 不凭此新增视觉模型；感知checkpoint另行绑定。[W2] |
| PyTorch Geometric | 2.6.1 | 使用标准RGCNConv，先最小安装；额外编译扩展非默认。[W3] |
| NumPy / Pandas | 1.26.4 / 2.2.3 | 执行默认；完整传递依赖由lock解析 |
| Matplotlib | 3.9.4 | 只生成真实数据图；不默认Seaborn |
| JSON Schema | jsonschema 4.23.0，Draft202012Validator | schema检查＋额外跨字段/ID校验 |
| Config | PyYAML 6.0.2＋dataclass | 一个最终resolved config；不再叠加Hydra |
| Logger | 标准logging＋CSV/JSONL＋TensorBoard 2.19.0 | CSV/JSONL是源，TensorBoard是观察副本；不依赖云logger |
| Unit test | pytest 8.3.5 | CPU确定性单测为主，目标GPU补数值与梯度检查 |
| Env API | Gymnasium 1.1.1 | 自定义skill adapter；不使用框架reward wrapper |
| Lock | uv 0.8.22 | `pyproject.toml + uv.lock + uv pip freeze + wheel index记录`；本包不伪造已解析lock。[W8] |
| VLM SDK | dashscope 1.27.6 | 发布记录可查；图像＋JSON实际调用需0C验证。[W9] |
| Checkpoint | PyTorch state_dict字典，`.pt`＋JSON manifest | 不保存完整Python类对象；只载入可信本项目文件 |
| Git/hash | git可执行程序＋SHA-256 | 记录实际git版本、完整40位commit、dirty diff hash；不虚构commit |

上表是唯一首选profile而非“最新软件清单”；未在目标机器联合安装的状态写 `PROPOSED_NOT_VALIDATED`。在0A执行resolver、`uv pip check`、import及RGCN前向/反向后，才升级 `software_profile=VALIDATED`。CPU可用于单测；GPU性能与显存需求由实际profiling决定，不声称固定硬件必能承受任意候选数。

## 3.3 项目最小实现交付清单

0A负责绑定/搭建以下代码入口；0B负责验收，不跑科研训练。现有项目可以映射到不同路径，但必须在 `entrypoints.yaml` 写真实路径和CLI参数。下面是应实现的接口，不表示本包已附带训练器。

| 模块 | 最小职责 |
|---|---|
| `contracts/registry.py` | skill/predicate注册、类型grounding、合法效果和来源校验 |
| `graph/snapshots.py, intervention.py` | 不可变真实快照、只读nominal overlay、三态与不变量 |
| `graph/rgcn.py, readout.py` | 标准共享RGCNConv、固定goal_refs、FP32差分 |
| `vlm/client.py, cache.py, validate.py` | 冻结API、原始/解析/拒绝记录、完整key |
| `policy/model.py` | GRU、候选编码、GK/GH/后继、七种配置、单一参数所有者 |
| `rl/collector.py, rollout.py` | 真执行transition、候选mask、prior版本、时长、完整prefix |
| `rl/targets.py, losses.py, recurrent.py` | SMDP GAE/Q、PPO、target detach、当前参数prefix重算 |
| `env/adapter.py, skills/executor.py` | 真实平台、相机、skill执行、实际结束标志；MUST_BIND |
| `evaluation/run.py, metrics.py` | 冻结checkpoint、配对初始case、task_success、AUC、诊断 |
| `reporting/build.py` | 只读raw→派生表图→selection manifest |

不允许把scripted expert作为训练采样策略，不允许用名义patch构造环境转移来冒充真实执行。程序逻辑fixture属于0B；它可以没有机器人，但不能变成主论文视觉结果。

---

# 4. 目录、文件、hash与恢复协议

`EXPERIMENT_ROOT`在0A绑定，以下相对路径统一：

```text
experiments/
  docs/                        # 本手册、依赖源、实施笔记
  manifests/
    experiment_manifest_v0.yaml
    method_manifest.yaml
    model_manifest.yaml
    software_manifest.yaml
    runtime_manifest.yaml
    task_manifest.yaml
    skill_contracts.yaml
    verifier_manifest.yaml
    splits.json
    vlm_manifest.yaml
    entrypoints.yaml
    presentation_manifest.yaml
    tuning_history.jsonl
  configs/
    resolved_defaults.yaml
    methods/{B0,B1,B2,Full,A_DD,A_Q,A_B}.yaml
    stages/stage_<id>.yaml
  stage_status/
    stage_<id>.json
    history/stage_<id>_<timestamp>.json
  vlm_cache/{train,dev,test}/<cache_key>/
  part_0_validation/{stage_0a,stage_0b,stage_0c}/
  part_1_smoke/{stage_1a,stage_1b}/
  part_2_core/{stage_2a,stage_2b,stage_2c}/
  part_3_ablation/{stage_3a,stage_3b,stage_3c}/
  part_4_mechanism/{stage_4a,stage_4b}/
  part_5_generalization/stage_5a/
  part_6_packaging/{stage_6a,stage_6b,stage_6c}/
  raw_archive/
    runs/<run_id>/
    index.jsonl
    snapshots/
  paper_results/
    figures/
    tables/
    qualitative/
    selection_manifest.yaml
    results_summary.md
    result_claim_mapping.md
    figure_sources.json
    README.md
  paper_draft/
    sections/
    writing_status.md
```

每个训练运行的目录模板：

```text
part_<n>_<name>/stage_<id>/task_<task_id>/<method>/seed_<s>/attempt_<nn>/
  config.yaml
  manifest.json
  train_metrics.csv
  eval_metrics.csv
  episode_log.jsonl
  decision_log.jsonl
  diagnostics.csv
  failure_log.jsonl
  run_summary.md
  checkpoints/step_<N>.pt
  checkpoints/step_<N>.json
  plots/
  tables/
  stdout.log
  stderr.log
  checksums.sha256
```

`run_id = <stage>-<task>-<method>-s<seed>-<UTCtimestamp>-<configSHA8>-a<attempt>`。新调参、新代码、新cache或从头重跑均新增attempt，禁止覆盖旧CSV。原始目录关闭后只读；raw_archive可使用校验过的copy/reflink或内容寻址引用，不要求把大checkpoint物理复制两遍。

复用Full必须通过 `method, architecture, task/split, seed, reward, controller/verifier, observation, prior/cache, training budget, relevant optimizer settings, code semantics`检查，写 `runs_reused`；同一个run不是第二个seed。仅报告脚本变动可复用；语义代码变化则新运行。3A使用2A Full，3B/3C默认使用2B Full，不混合不同预算作严格消融。

checkpoint包含model/optimizer state_dict、RNG状态、总技能transition、PPO update、折扣计数和所有manifest hash。恢复整段on-policy训练必须有一致环境状态、Fact Store、GRU prefix、episode prior和buffer；缺少环境恢复能力时只可在完整update后的全新episode边界恢复，并标记trajectory discontinuity。未完成buffer不得拼接旧策略与新策略数据。新seed不能继承另一个seed的权重。

---

# 5. Task Manifest：结构模板、绑定与split

## 5.1 固定任务ID和候选范围

以下是**任务结构模板，不是已存在的环境资产**。0A将schema、对象、谓词、传感器证据、控制器和实际目标绑定到这些ID。优先使用已有可执行资产，不新造复杂机器人任务来迎合方法。

| ID | 用途与结构 | 最小实例模板与可选技能 | 何以需要结构 | 必填绑定与验收 |
|---|---|---|---|---|
| D0 | 唯一development task；共享前置＋一个缓冲选择 | 两对象、一个可开容器、一个缓冲区域；OPEN/PICK/PLACE及实际存在的缓冲技能 | 至少2个合法候选、4–6个必要技能、一个共享依赖；至少一个缓存场景保留合法非冗余prior | 真实skill注册、观察初始状态、2种可行次序/选择；不得为制造非空prior添加假效果 |
| T_A | Shared Prerequisite | 两个后续操作共同需要Open/Ready类已有条件；4–8技能 | 提前建立共享条件可改变后续选择；合同支路本身也可能已经足够 | 前置是控制器真实要求；失败后存在合法放回/恢复路径 |
| T_B | Multi-Object Choice | 2–3对象、2目标/工作区；同一时刻多个可操作对象；5–10技能 | 局部都合法，但完成后会占用/释放不同已有资源或影响后续顺序 | 长期差异来自实际任务/几何/合同，不人工赋予Full额外信息 |
| T_C | Intermediate Relocation | 一个干扰对象需临时移到已有buffer，随后处理目标对象；5–9技能 | relocation的注册后果可通过非冗余soft relation支持另一个skill/goal | MOVE或等价PICK＋PLACE必须真实存在；几何阻碍/效果不能只写在VLM文本中 |
| T_D | Multi-Goal Interaction | 一个动作建立/撤销影响多个合取目标的事实；6–10技能 | 局部满足一目标可能妨碍另一目标，所有目标当前有效性都要保留 | signed conjunction、DEL语义、独立整项成功评价；不把OR偷偷改成AND |
| T_E | Recoverable State Change | 任务中已建立条件可因实际执行失效，再用已注册skill重建；6–10技能 | 恢复之后还需继续剩余任务，不只识别一次loss | 实际可恢复事件及监测证据；不把未来干预标记给policy；无资源则不选 |

**默认执行组合：**1A使用D0；2A使用T_A、T_C；2B使用T_A、T_B、T_C。T_D/T_E是替换或扩到4–5个主任务的候选，不默认一起训练。3A默认T_C；3B默认T_A（可由2A选出的代表任务替换）；3C默认T_C。

替换规则：先按controller已支持、事实可观测、能终止、多个真实候选、4–10步结构筛选，再考虑已观察到的解释价值。写 `selected_tasks` 和变更理由；不得把没有结构差异的几何随机种子包装成新task family。2B未能绑定第三个真实任务时，2A仍可写结果，但2B完整3-task矩阵为BLOCKED，不虚构第三行。

D0不进入主性能表。主任务的dev初始化用于checkpoint选择；D0不作为判断每个复杂task checkpoint质量的唯一validation任务。

## 5.2 每任务的有限初始化池

0A定义每个task的 `train=64`、`dev=50`、`test_id=50` 个初始case。它们是初始状态/场景，不是示范轨迹，不含正确动作序列。初始图像、对象绑定与全部hash不同；相同任务名不意味着同一个cache。

第一批仅按需要生产缓存：D0的64个train＋10个dev，T_A/T_C各64个train＋20个dev；0C的24个sanity场景尽量从这些dev池中取。后续30/50 episode评估只补齐尚未生产的既定case缓存，不改生成协议。缓存生成量单独计入，不当作RL训练job。

训练在64个case间以独立 `train_case` RNG抽取；可重复初始化，但真实后续rollout由policy执行。增加case池属于新的split revision，不在测试成绩差时悄悄替换测试池。所有方法共享同一任务与初始化定义；动作轨迹无需相同。

## 5.3 唯一组合泛化

提前为一个已绑定task family登记两个held-out composition，记 `G1/G2`，每个50个测试初始case。仅改变**未见的对象—目标角色组合或任务依赖组合**，skill schemas、谓词、对象数、感知、控制器、外观分布保持不变。

每个G1/G2列出 `seen_atoms / seen_skills / unseen_composition_signature / source_training_tasks / canonical_graph_signature`。仅对象重命名后同构、或者只更换随机位置，不计为结构泛化。冻结原来的训练checkpoint，零梯度、零finetune；允许冻结VLM读取新初始场景生成一次cache。若只有重新命名可做，标 `GENERALIZATION_UNSUPPORTED_BY_CURRENT_ASSETS`，不伪造结果。

---

# 6. 预算、评价、seed、checkpoint和指标

## 6.1 预算单位与统一默认

本私人执行计划采用**高层真实skill transitions作为运行配额**，同时记录实际底层tick、模拟/物理时长及wall time。这是相对旧正式实验规范的显式执行口径调整；相同skill次数不等于相同物理交互量。物理样本效率主张必须基于共同物理时域重新计算，不从skill横轴直接推出。

一次PPO rollout默认收集全体env合计1024个真实skill transitions；一次PPO update含4个epochs；minibatch为64个有效transition，连续片段长16。正常满buffer每update有64次optimizer step，padding不算有效数据。默认4个并行仿真env各256条；实机通常1个，资源适配必须写manifest。不能把1024误解成每个env都1024。

| 阶段/用途 | skill transitions上限/新job | PPO updates上限 | checkpoint及dev评估间隔 | 每checkpoint dev episodes |
|---|---:|---:|---:|---:|
| 1A smoke | 16,384 | 16 | 4 updates = 4,096 transitions | 10 |
| 1B单一调参trial | 16,384；仅训练长度trial可32,768 | 16或32 | 4 updates | 10 |
| 2A exploration | 65,536 | 64 | 8 updates = 8,192 transitions | 20 |
| 2B presentation | 131,072 | 128 | 16 updates = 16,384 transitions | 20 |
| 3A第一批A_DD | 65,536，与复用2A Full一致 | 64 | 8 updates | 20 |
| 3B / 3C | 131,072，与复用2B Full一致 | 128 | 16 updates | 20 |
| 4A / 4B / 5A | 0 | 0 | 只用冻结checkpoint | 见各Stage |
| 0A / 0B / 0C / 2C / 6A–6C | 0 | 0 | 不适用 | 0（0C是24个场景，不是episode性能评估） |

初始step=0也做相同dev评估，便于识别学习是否出现。中途评价使用独立env实例；不重置训练中未结束episode，不推进其GRU或改变其prior。

**物理/模拟时间资源上限。**每task绑定 `reference_skill_seconds=d_ref`，来自现有controller记录或0A有限接口试运行的实际中位时长；训练资源cap为 `2 × N_cap × d_ref` 秒，跨并行env求和。规划表暂用 `d_ref=2.0s`，故1A/2A/2B资源cap分别65,536 / 262,144 / 524,288模拟秒；这些不是预计墙钟耗时，也不是已测技能时长。真实执行前d_ref、deadline、每skill timeout必须MUST_BIND，不能用2秒作为实机安全阈值。

默认以skill cap或资源cap先到者结束采样；不在skill中途凭训练cap强制修改任务终止。已开始技能允许结束，记录overshoot；不足1024的尾buffer使用有效样本/序列mask完成至多一次partial update并单报。预算触发不是方法失败；比较AUC取实际共同范围，不能给短run外推后半段曲线。

任务级仿真deadline建议起点120秒，最大高层决策64次，仅供平台绑定时确认；实机必须用已核准值。达到定义中的任务deadline/硬任务步限为真实终止，不作为免费训练截断。3A等控制消融的Full若只有不同预算，先找匹配checkpoint及其实际训练history，不以延长某一方法的结果装成等预算。

## 6.2 Seed与独立随机流

训练seed：smoke `[0]`；exploration `[0,1,2]`；presentation默认 `[0,1,2]`。补seed按 `[3,4]` 顺序，默认最多5个正常尝试的不同训练seed；技术性重跑同seed记新attempt。每配置选择3个展示seed可不为012，但要记录所有候选和理由。

不要使用Python不稳定的内置hash派生实验seed。定义：

```text
seed32(namespace, task_id, case_index, train_seed_or_zero)
 = int.from_bytes(SHA256(canonical_UTF8_JSON([...]))[:4], 'big')
```

namespace固定为 `init_model / train_case / action_sampling / prior_dropout_train / dev_case / test_id_case / generalization_case / prior_robustness_eval / dp_shuffle / unit_fixture`。环境评估case和测试扰动的派生输入不包含method，不包含训练seed，保证不同模型配对初始case与边编辑；训练模型和动作采样则包含train_seed。

dev索引0–49；test_id索引0–49但namespace不同。smoke用dev前10，explore前20，最终presentation用test_id前30。扩大到50时**追加30–49**，合并为50，不把前30又当新增独立样本。PyTorch/NumPy/Python/env/action space各自设种，保存所有RNG state；GPU非确定性另记，不承诺跨硬件逐bit相同。

## 6.3 评价协议

训练采样stochastic；默认所有论文评价使用masked argmax deterministic。相同最大logit的精确tie按canonical candidate ID排序打破，不以列表位置偏置。可另做stochastic诊断但不得与deterministic主表混平均。

主条件Original；B0/B2为无prior。Full/B1/A_DD/A_Q/A_B训练为80/20，而评价**不额外施加20%dropout**。原始cache为空仍然是Original-empty，不另补边。每次episode重置自身跟踪/Fact Store/GRU，保留该次完整执行日志。

最终评价默认30 episode/task/training seed；robustness先20；smoke10；需要更稳定时一次性扩到50。20→50或30→50的触发是预期呈现是否受采样噪声主导、配对差异是否频繁反向等描述性需求，不是“还没显著所以继续”。不默认100以上。

## 6.4 Checkpoint选择

每方法、task、seed在自己的**该task dev池**上选择checkpoint；不同方法step可不同。默认选择指标为最近连续3个dev评估点的success均值，先最大均值，再较小三点标准差，再较早checkpoint。前两个评估点不单独作为best；不足3点用最后完整checkpoint并注明。step=0只作基线，不选作训练成果，除非明确报告零学习。

不按单次尖峰选checkpoint；相同selected checkpoint同时用于主success、robustness和generalization，不对每种扰动再挑一个最优模型。AUC来自整个真实学习轨迹，不只来自选中的checkpoint。Paper View允许改选另一合理checkpoint，但写新selection revision、dev证据和理由；被查看并用于调整的test结果转为exploratory selection data，不再称为未使用holdout。

## 6.5 指标计算与图表约定

设task f、训练seed s的评价成功数为k，总有效episode为n：\(\widehat p_{f,s}=k/n\)。先在每seed内按实际episode计算，再跨selected seeds等权平均；task-family结果先按task等权，再按family等权。不要因为某seed做了50次、另一个30次而让前者权重大。

学习AUC用原始dev成功率点作梯形积分，并除以共同横轴长度：
\[
\operatorname{AUC}_{[0,H]}=\frac1H\sum_j\frac{p_j+p_{j+1}}2(x_{j+1}-x_j).
\]

H取该比较集合实际共同可用区间；规则内允许线性插值对齐已有采样点之间，不外推、不拼接不同seed的最佳片段。缺0点则从共同最早点开始，注明 `auc_start`。skill-AUC与physical-time-AUC分列；未记录物理时间则后者N/A。

`time_to_70`采用连续两次原始dev评价≥0.70的第一次达标位置；未达到记 `NOT_REACHED` 并保留budget，不填0或编造无穷值。0.70是统一展示阈值，不是Stage PASS门槛。

完成时间只统计成功episode，同时列成功数和全部episode结果分布；没有成功写N/A。恢复率必须给机会数，零机会N/A。robustness报告绝对成功率和相对同子集Original的百分点差，不能把缺失case算成成功。

默认曲线：原始点低透明度＋trailing moving average窗口3用于显示；窗口不足3使用已有点。主表、AUC、checkpoint选择都不用平滑后的数据。聚合曲线先对齐原始x，再跨seed均值，最后画MA3；误差范围为selected seed间±1 sample SD，不标95%CI。n=1只画单线，不画“置信区间”。

成功率轴默认[0,1]；局部放大另作明确标为zoom的图。默认统一横轴，不删早期失败段；可另展示structured-task subset和representative seed，但写清子集。图输出SVG和PNG，统一尺寸；每图一个独立画布，不用多个subplot挤在一起。所有绘图字段（x、y、单位、subset、seed、smoothing、aggregation、checkpoint）写selection manifest。

---

# 7. 调参、运行错误与结构性问题

## 7.1 三类参数

| 类别 | 内容 | 权限 |
|---|---|---|
| Frozen Method | v2.1；两种关系及effect_fact_ref；pure DP；无静态prior；Action–Proposition；共享标准R-GCN；名义patch权限；终端奖励；真实executed-action Q target；SMDP PPO；80/20；无训练错边 | 所有Stage禁止为争取好结果改变；结构性错误单独记录方法修订建议，不自动实现新方法 |
| Default-but-Tunable | lr=3e-4；B=0.5；λQ=0.1；entropy=.01；训练配额；评估间隔；必要时hidden=128；资源性batch/num_env | 仅1B、2A的有限trial或新revision；2B/3/4/5每个已启动run内锁定 |
| Runtime MUST_BIND | repo/worktree/commit；simulator/robot版本；对象资产；controller；camera/calibration；frozen perception checkpoint；verifier阈值；skill timeout；实际时间单位与d_ref；reward evaluator；硬件/driver；API地区/权限；split实例 | 0A优先绑定；不得以纸面默认冒充已经存在 |

不把γ当随意提分旋钮：本版每秒0.99与时间目标一起冻结；若实际任务时标完全不匹配，记录协议问题并另行明确版本，不只给Full更有利折扣。hidden或batch调整需通过零性质/梯度/重排回归单测，且训练从明确起点重新开始；不能把两个结构的学习曲线拼起来。

## 7.2 唯一排查与调参顺序

附件§19与§24对后半段调参顺序略有不同；本手册作显式统一：采用§19和Stage1B的一致順序，不保留两条相互矛盾的自动队列。

**implementation → candidate/mask → reward → Skill Contract → Verifier → DK → DP → prior residual scale → lr → training length → entropy → B → λQ → task difficulty → VLM relation quality。**

前八项是诊断，不是立即改参数。检查λQ/B梯度或饱和可以发生在数值诊断，但真正trial仍按上述顺序。先确定是接口失败、无成功探索、数值爆炸、还是正常但无增量，不因Full暂时落后改Method。

1B每次仅改一个量：lr按3e-4→1e-4（不稳定）或5e-4（稳定但学习慢）；长度16,384→32,768；entropy从.01到.02或.005；B从.5到.25或1.0；λQ从.1到.05或.2。每参数含默认最多3值，**整个1B最多6个新增训练job**，不是每个参数各跑6个。出现满足smoke的配置即可停止；预算用完则报告未解决，不无限搜到胜出。

2A允许额外最多4个单配置诊断job，或一次增加第3任务完整12个job，默认二者不自动同时进行。每项调整写hypothesis、旧值、新值、影响方法、观察结果、采用/放弃及数据是否已被查看。不同方法少量lr/entropy适配允许，但记录这是经适配后的比较；严格单因素A_DD/A_Q/A_B默认继承其Full全部其他超参。

## 7.3 技术失败、正常零成功与重跑

NaN/Inf、非法mask、真实事实被名义patch污染、target泄漏、误用reset状态bootstrap、丢失raw文件是技术错误：停止相应job，归档，修复后新attempt。OOM先减同时编码的候选view块，再减并行env或microbatch，通过梯度累积保持有效batch；不先删候选/软关系。

正常完成但零成功是**有效学习结果**，不能标作基础设施异常。可不放Private Paper主图，但Raw Archive保留且selection理由为performance/exploration failure。某seed失败不自动要求全部seed重跑；基础设施异常使用相同seed新attempt，不能用新seed掩盖故障。

Full只领先几个百分点、Full≈B2、B0简单任务表现很好、A_Q帮助小、A_B只在扰动中有差异、部分空prior、generalization较差，均可继续。可以优先讲真实的AUC、速度、稳定性或结构任务局部现象；没有支持就不写相应优势。

## 7.4 何时才讨论重新打开Method

在合法非空prior、真实非空patch、正确goal对齐的受控输入上，DP几乎处处严格0；soft边完全无传播；排除bug后Actor始终不响应DK/DP；Full长期学不会而B0/B2正常；Q在多项检查后系统性导致崩溃；多结构任务持续明显落后；资源接口根本无法满足名义/真实隔离，这些才触发结构问题报告。[S1，§23]

Agent不得自己新增static prior、teacher或shaping。输出 `method_issue_report.md`，给证据、已排除项、影响范围；停止当前有问题作业但不阻塞算法说明与研究过程写作。一次不成功或原始cache部分为空不构成上述结论。

---

# 8. VLM Cache、日志与Agent输入契约

## 8.1 VLM生产协议

保留 `qwen3.8-max-0902`（完整别名`qwen3.8-max-2026-09-02`），不使用自动滚动的max别名。官方当前页面列出该快照、图像输入和结构化输出能力；这不是本账户调用成功或relation质量保证。[W7]

SDK首选dashscope1.27.6。地区建议Singapore，但**实际region及base_http_api_url必须绑定到账户与官方对应端点**，不得默认调用SDK中国区端点却在manifest写Singapore。0A保存endpoint字符串；0C用一张允许的dev图像确认图像输入、非思考、JSON object、2048输出token、temperature=0的真实返回行为。SDK调用适配层通过本机已安装版本签名及服务响应核对，不猜测未支持的参数；无法实现声明行为则BLOCKED，不静默删字段。

固定 `max_relations=8, max_output_tokens=2048, temperature=0, thinking=false, response_format=json_object, web_search=false, tools_disabled=true`。prompt版本 `cp_disr_relations_v2.1_p1`，schema `m1_soft_relations_v2`。3个few-shot从独立开发材料选择并冻结：合同重复→空、额外后果支持→合法边、无顺序依据→空；完整图像与JSON落盘。完整prompt、few-shot和预处理一起hash。

每输入最多2次请求：首次＋一次仅传输/语法/截断失败重试；401/403不重复、429按服务允许等待后最多一次、格式合法但语义差不重问。第一次可解析结果立即进入本地schema/ID/effect校验；非法项隔离，合法语义可错项保留；最终无关系记EMPTY_PRIOR。服务不可达记API_ERROR，不将失败请求冒充VLM主动空输出。不得因RL低分反复问模型。

cache_key为canonical JSON的SHA-256，必须包含 `split, task_definition_hash, initial_RGB_content_hash, preprocessing_hash, object_binding_hash, allowed_ID_hash, contract/predicate_versions, model_snapshot, SDK/API_version, region/endpoint, prompt/fewshot/schema_hash, decoding_config`。输入对象数组按canonical ID排序，有序参数保持顺序。split也进key，防止train/dev/test互相覆盖。

每条cache保存 `manifest.json, request.json, raw_response_attempt_0.json, raw_response_attempt_1.json(实际有重试才生成), parsed_relations.json, rejected_relations.json, dedup_log.json, final_edges.json, processing_log.json`。密钥只用环境变量名，不写值；不存在attempt_1不造空响应冒充运行。

正式生成后改变模型/prompt/schema/region/图像处理必须升cache版本并保留旧输入输出。没有账户权限时0C BLOCKED，不自动切成本地模型。后续Stage可按相同冻结协议为其既定case补缺cache，这属于本Stage输入准备，不自动重做或修改0C。

## 8.2 训练prior版本

每episode开始用独立流抽Bernoulli(0.8)：保留原始或整份空。episode跨rollout/update时不重抽；PPO重算用保存的最终边集和hash。所有prior消费者共享此协议；B0/B2操作为空。mode标签不输给Actor/GRU，不泄漏“人工空”与“自然空”的区别。

记录抽样模式、原始边数、最终边数、原始空比例与实际空episode比例；80/20不是必然每5个episode恰好4/1，也不是transition占比。没有non-empty prior的task可以继续合同学习，但不拿它证明VLM增量。

## 8.3 标准日志字段

**L_RUN（每run）：**run_id、stage、method/plan version、git hash/dirty hash、task/split、training seed、resolved config hash、controller/verifier/perception/evaluator版本、hardware/software、API/cache/prompt/schema hash、预算、实际起止时间、attempt、parent/reuse run、调参历史、失败原因。

**L_TRAIN（每update与真实episode）：**success_rate、episode_return、task_success、skill_count、train_skill_transitions、PPO_update、optimizer_step、low_level_ticks、simulated/physical_seconds、wall_seconds、actor_entropy、policy_loss、value_loss、q_loss、grad_norm_before_clip、approx_KL、clip_fraction、DK_norm、DH_norm、DP_norm、DP_nonzero_rate、prior_residual_mean/max、prior_saturation_fraction、candidate_count、relation_count、selected_skill、failure_reason、checkpoint_step、run_id、seed。

**L_DECISION（每decision，可压缩）：**snapshot/episode/env ID、frame/evidence IDs、真实fact哈希和T/F/U、所有candidate IDs及mask、selected candidate ID/index、old_logp/oldV/oldVnext、r/d/Gamma/w、terminated/truncated、prior原始/最终hash、nominal patch引用、base logits/prior residual/final logits、GRU prefix引用。大张量使用压缩快照按引用存，不重复写JSON像素。

**L_EVAL（每eval episode）：**model/checkpoint hash、task、training seed、eval namespace/case、condition、success、return、技能数、实际耗时、终止/失败原因、prior编辑前后hash、适用性、decision日志引用。相同case的配对关系必须可恢复。

**L_TEST（每单测）：**test ID、fixture/hash、CPU/GPU/dtype、expected/actual、tolerance、pass、异常堆栈。**L_CACHE（每scene）：**请求次数、首答JSON合法、ID合法、effect合法、合同重复数、最终边数、类型计数、空原因、语义人工审计标签。

数值不适用写null及N/A原因，不能把B0的DK缺失写成0，也不能用NaN代表正常缺失。浮点NaN/Inf必须触发数值错误。DK/DH/DP日志同时保存Frobenius norm与按有效goal×维度归一化RMS，padding和masked candidates不计。

DP实际非零检测阈值为 `1e-7 + 1e-6 * max(RMS(DK), RMS(DH))`；另报严格zero比例，避免把舍入噪声当机制作用。threshold是诊断定义，不强迫每个合法prior在每个状态都非零。空patch和原始空prior自然为零。

---


---

# Part I — Infrastructure & Method Validation

## Stage 0A — Environment & Manifest Freeze

| 字段 | 执行定义 |
|---|---|
| Stage ID | 0A |
| Stage Name | Environment & Manifest Freeze |
| Purpose | 将文档默认变为真实可识别的代码、环境、任务和接口；建立后续Agent无歧义的运行入口。 |
| Stage Type | Infrastructure |
| 是否属于论文正式结果 | NO |
| 前置条件 | 当前v2.1规范、v2.1接口包与本计划存在；允许读项目文件。无训练checkpoint前置。API密钥只检查可用性，不读取/打印值。 |
| 输入 | 读取sources中的规范/接口/prompt；manifests初始模板；当前worktree和git（MUST_BIND）；实际训练host、GPU/driver、已有controller/观测/评价器、任务资产与标定记录。历史repo名称只能作查找线索，不作为已确认工作分支。 |
| Frozen Parameters | 第2章方法核心、relation schema、80/20、真实奖励权限；主RL路线为项目内PyTorch SMDP-PPO；不迁移旧奖励方法为主方法。 |
| Tunable Parameters | 仅依赖兼容profile、目录/入口映射、num_env与view分块、资产绑定；无性能调参。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| bootstrap | Infrastructure | D0/T_A/T_B/T_C/T_D/T_E templates | 0 | 0 | 不调用 | 1次host/manifest核验；仅按已有记录估计时长 |

### Training Budget

训练0 transitions、0 PPO updates、0性能episode。无机器人动作默认；若项目既有许可允许，可做≤12次控制器接口调用校准d_ref，标为基础设施校准，不入RL结果。不具备许可时使用既有校验记录或MUST_BIND。

### Evaluation Protocol

不做成功率评估；import、依赖解析、RGCN最小前/反向和checkpoint round-trip各1次。不以这些检查代替0B的T01–T25。

### Required Logging

L_RUN的版本/资源子集＋source_hashes、依赖解析输出、git dirty diff hash、entrypoint检查、资源缺项；无训练loss日志。

### Output Directory

`experiments/part_0_validation/stage_0a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_0a.json`。

### Required Files

- `source_inventory.json`
- `environment_probe.json`
- `binding_report.md`
- `implementation_inventory.md`
- `resolved_defaults.yaml`
- `entrypoints.yaml`
- `software_manifest.yaml`
- `runtime_manifest.yaml`
- `task_manifest.yaml`
- `splits.json`
- `stage_0a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

None；只输出表格和缺项报告。

### Required Tables

binding_table.csv：field、proposed、observed、status、evidence；dependency_table.csv：package、requested、resolved、import_status。无均值。

### PASS / STOP Criteria

PASS：选定host软件能解析/导入，核心代码入口可定位，D0及2A两个任务所需controller、camera、verifier、deadline和evaluator完成真实绑定；T_B/D/E可先标可选未绑定。BLOCKED：核心资产、code、host、账户region或安全许可缺失。可完成资料预检但不称完整0A PASS。

### Failure Handling

先查源文件/hash→项目路径/分支→依赖冲突→核心模块缺失→资产→感知/合同/评价器→安全/时间→API配置。缺代码可按第3.3章在本Stage实现冻结接口，但不假装外部controller存在。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，补绑定后新revision；不覆盖前次报告。 |
| Tuning Allowed | 只允许依赖/资源/绑定调整，不允许性能优化。 |
| 是否阻塞论文写作 | NO；只阻塞尚未绑定平台上的执行，不阻塞Introduction/Method。 |
| Next Stage | 0B；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 0A”时，Agent严格执行以下步骤：

1. 读取本计划与v2.1原文件并记录SHA-256。
2. 定位用户指定worktree；记录完整commit和dirty diff，未指定则写MUST_BIND。
3. 只读盘点host、CUDA、Python、包版本与文件系统权限。
4. 在独立虚拟环境解析第3.2章推荐profile；不覆盖系统Python。
5. 映射或实现第3.3章冻结代码入口，保存入口参数和hash。
6. 绑定技能合同、Fact Verifier、独立任务评价器和安全结束规则。
7. 将D0、T_A、T_C结构模板映射到真实资产并定义train/dev/test初始化池。
8. 登记时钟、deadline、timeout、d_ref及GPU/view分块；标出剩余MUST_BIND。
9. 生成所有manifest与implementation/binding报告，执行轻量import/round-trip检查。
10. 按实际readiness写stage_0a.json；有核心缺项则BLOCKED，不自动执行0B。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 0B — Unit & Invariant Tests

| 字段 | 执行定义 |
|---|---|
| Stage ID | 0B |
| Stage Name | Unit & Invariant Tests |
| Purpose | 验证v2.1公式、权限与PPO数据链，不检验机器人策略性能。 |
| Stage Type | Unit Test |
| 是否属于论文正式结果 | NO |
| 前置条件 | 0A至少readiness.core_code=true、软件可运行；规范/schema及fixture存在。可在robot未绑定时独立执行纯逻辑测试，但不得自动把0A升级PASS。 |
| 输入 | 当前代码commit和resolved_defaults；生产nominal/graph/policy/targets/buffer组件；tests/fixtures；v2.1 schema；seed32(unit_fixture,...,0)；CPU为必需，目标GPU可用时补测。无cache请求、无checkpoint训练。 |
| Frozen Parameters | 全部方法不变量、测试期望和fixture语义；不能通过放宽公式来让错误实现通过。 |
| Tunable Parameters | None；只修bug。数值tol若因设备调整必须给CPU对照和原因，不改变严格零性质。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| invariants | Production components + controlled fixtures | T01–T25 | 0 | 0 | 空/合法/非法fixture | CPU 25项；GPU补对应数值项，不是25个RL runs |

### Training Budget

0真实skill transitions，0科研PPO updates，0性能episode；允许test内dummy backward/optimizer验证梯度。

### Evaluation Protocol

逐项T01–T25执行；CPU FP64/FP32及可用GPU FP32。expected/tolerance在下表。所有实际收集的测试都报告，包括失败与不适用设备。

### Required Logging

L_TEST完整；保留pytest JUnit XML、stdout、traceback、fixture/code/schema hash；统计passed/failed/skipped，不产生训练success率。

### Output Directory

`experiments/part_0_validation/stage_0b/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_0b.json`。

### Required Files

- `junit.xml`
- `test_results.json`
- `test_results.csv`
- `fixture_manifest.json`
- `invariant_report.md`
- `stage_0b_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

None；可选test_status.png（x=test ID，y=0/1，原始无平滑），不得称学习曲线。

### Required Tables

test_results.csv：test_id、device、dtype、expected、actual、tolerance、status、failure_location；按测试计数，不以toy=25项。

### PASS / STOP Criteria

PASS：T01–T25生产组件CPU检查全部通过；目标GPU未绑定可PASS_WITH_NOTES并记录GPU未验。NEEDS_RERUN：任何权限/target/mask错误；BLOCKED：核心生产组件缺失。不能用skip代替通过。

### Failure Handling

按测试失败定位，不做超参搜索；优先T03污染、T10 mask、T12 target、T14边界，再alignment、数值和梯度。修复后重跑受影响测试及完整CPU套件。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，技术修复后重跑。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO；阻塞未经验收的1A训练，不阻塞方法写作。 |
| Next Stage | 0C；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Stage 0B的逐项测试规范（T01–T25）

CPU逻辑期望精确相等；FP32标准比较默认 `atol=1e-6, rtol=1e-5`；FP64手算默认 `atol=1e-10, rtol=1e-9`。空prior复用与零锚定必须严格零；不用宽容差掩盖两条不一致路径。GPU先保持FP32；目标GPU与CPU逐元素结果不要求逐bit相同，但零通路仍显式复用。

| ID / test filename | 输入与procedure | expected result / tolerance / PASS |
|---|---|---|
| T01 `tests/test_zero_prior.py::test_dp_zero` | 固定6–10节点合同fixture；facts含T/F/U；prior为空；至少2个非空候选patch，分别做四路前向 | 直接复用K当前及K后继；所有DP元素严格0；形状、dtype一致 |
| T02 `tests/test_zero_prior.py::test_residual_zero` | 任意合法context与随机已初始化F_P，输入DP=0；另测原始cache为空 | uP和Δ严格0，Full概率与同参数合同分数分布一致；概率tol FP32；不是与独立训练B2权重比较 |
| T03 `tests/test_nominal_isolation.py` | 真实Fact Store含timestamp、attempt、quality、图模板；连续生成多个候选patch并尝试写入overlay | 真实序列化SHA256前后完全相同；不可变写入抛异常；观测/时钟/attempt/reward不变 |
| T04 `tests/test_node_alignment.py` | GK/GH及2个候选的两种后继；加入1条合法soft edge | node IDs、顺序、类型、argument binding逐项相同；只有边集合或声明的临时facts不同，精确通过 |
| T05 `tests/test_goal_alignment.py` | 含一个已完成goal、一个未完成goal和一个负目标；多个goal排列 | goal_refs、sign、行mask在四路相同；完成goal不删；padding重排后读出按ID还原一致，FP32tol |
| T06 `tests/test_three_valued_logic.py` | 正/负文字各测试T/F/U；AND包含未知；条件效果guard为U | U既不满足正前置也不满足负前置；非关键U不阻塞无关动作；UNKNOWN guard仅保留共同确定效果，精确 |
| T07 `tests/test_effect_conflicts.py` | 同一fact同时ADD/DEL、ADD/UNKNOWN、两个互斥Held等 | 注册/patch合并时报可定位异常；不按列表先后覆盖；合法非冲突对照通过 |
| T08 `tests/test_relation_validation.py::test_effect_ref` | 合法ADD锚点、合法DEL锚点、未知ID、target类型错、其他action效果、纯UNKNOWN锚点 | 仅合法source效果且已有ADD/DEL邻接者通过；给goal外部ID需正确规范化；metadata不新增edge/fact |
| T09 `tests/test_relation_validation.py::test_no_order` | SOFT_ORDER JSON以及missing effect_ref JSON；检查registered relation IDs | schema拒绝两个非法样例；生产relation表无ORDER及反向ORDER；历史toy文件出现该字符串不算生产违规 |
| T10 `tests/test_ppo_masks.py` | 存储candidate IDs/mask/action/oldlogp；未更新参数重算；故意更换顺序并以ID映射，再故意改mask | 正确重算old/new logp在tol内相同；合法重排等价；mask hash变化主动报错；不以新mask解释旧动作 |
| T11 `tests/test_q_executed_only.py` | leaf `q_values`形状[2,3]，executed=[0,2]，两个真实targets；loss.backward | 未选择的q_values输出项梯度严格0，selected有非零梯度；共享上游因集合池化收到梯度不是违规 |
| T12 `tests/test_q_target_detach.py` | oldVnext requires_grad=True；非终止真实r/Gamma；构造Q target | yQ.requires_grad=False、无grad_fn；Q backward不更新oldV；V target同样detach |
| T13 `tests/test_shared_gradients.py` | 非退化有效prior fixture；分别只反传actor、V、Q loss；4路E保留autograd路径 | 同一E对象/参数ID；各loss至少一项预期共享参数梯度有限且非零；不是要求所有参数或零prior分支都有梯度 |
| T14 `tests/test_smdp_boundaries.py` | 三种边界：terminated、valid truncation、buffer end；最后状态V=2，reset状态V=99，r=.3，Gamma=.9 | terminated yQ=.3；其他有效bootstrap yQ=2.1；不读取99；GAE不跨reset；FP64tol |
| T15 `tests/test_permutation.py` | 候选重排、对象ID一致重命名、goal行重排；同时重排对应边/参数/mask | 按canonical真实身份逆映射后logit/probability一致，FP32tol；精确tie用ID规则，不按原列表位置 |
| T16 `tests/test_cache_key_isolation.py` | task名相同但图像字节、split、goal、绑定、schema、region分别改变一次；另改JSON key顺序 | 每个语义输入改变都变key；只改变无序键顺序不变；数据split不相互覆盖 |
| T17 `tests/test_prior_episode_fixed.py` | 3个env、4个episode，跨多次rollout边界；固定RNG状态 | 只在episode_start抽一次；同episode prior hash不变；各env独立；不要求小样本恰好80/20 |
| T18 `tests/test_recompute.py` | 同一stored buffer执行4个PPO epochs的前向及一次dummy update，用spy计数API/RNG/E调用 | 重算E且使用当前参数；API调用0、prior抽样0；读取stored edges/mask；不能以detach旧D替代重算 |
| T19 `tests/test_duration_discount.py` | d=1s、2s和奖励在区间内0.5s的手算事件；episode起点累计权重 | Gamma=.99、.9801；reward按真实事件偏移折扣；w递乘；技能耗时非正拒绝；FP64tol |
| T20 `tests/test_permissions.py` | 只改prior边或传入拒绝关系，真实task/facts不变；检查mask/evaluator输入schema | candidate/mask/reward定义与prior无关；evaluator不接DK/DP；静态ZH不能偷进Full基础GRU/context |
| T21 `tests/test_prior_bound.py` | s取大正/负及0；B=.5，多个候选 | abs(Δ)≤B+1e-7；允许动作的概率比在exp(±2B)内（FP32tol）；A_B不要求该界 |
| T22 `tests/test_recurrent_prefix.py` | 两env不同历史，连续片段与全prefix；nextV probe重复调用 | probe不重复提交GRU；当前参数prefix重算与整段一致；reset只清本env；截断BPTT片段有梯度 |
| T23 `tests/test_soft_message_path.py` | 保留附件四个线性toy；另用生产128维、ReLU/LN共享RGCN，预先构造多维非退化权重与有效3跳soft路径 | 线性toy复现代数见证；生产fixture中至少一个DP与边移除响应>1e-6且梯度有限；不是要求随机网络每次都非零；合法但合同重复的边正式validator应剔除 |
| T24 `tests/test_empty_candidates.py` | 全mask false、仅安全WAIT合法、WAIT空patch三种 | 全空按独立安全结束，不创建非法Categorical；WAIT时长>0且patch空；不能给虚假transition奖励 |
| T25 `tests/test_ablation_switches.py` | 同一模型前向分别Full/A_DD/A_Q/A_B，prior空/非空 | A_DD非空用DH、空显式0；A_Q只有Q loss系数改0；A_B只移除tanh且保留B；其余config hash等价 |

**实现要求：**T01–T25必须针对实际生产组件或明确的最小fixture执行。没有生产代码时写BLOCKED，不用`assert True`、skip所有测试或附件4个toy冒充全量通过。T18的dummy backward/optimizer仅用于代码验收，不构成科研RL训练。

**固定fixture目录：**`tests/fixtures/contracts_v1.json`、`facts_tfu.json`、`nonredundant_support_graph.json`、`smdp_transitions.json`、`snapshots_two_envs.json`。均标 `synthetic_unit_fixture=true`，不入经验数据/论文主成功率。


### Agent Execution Template

当用户说“执行 Stage 0B”时，Agent严格执行以下步骤：

1. 读取当前commit、schema、配置和fixture hash。
2. 确认测试调用生产组件，而非独立伪实现。
3. 先运行T01–T09的逻辑与关系校验。
4. 运行T10–T18的数据链、梯度与prior重算测试。
5. 运行T19–T25时间、权限、界、GRU、传播和消融开关测试。
6. 目标GPU可用时补FP32数值/梯度子集并记录设备差异。
7. 修复本Stage内发现的实现bug，保留diff并重跑受影响项。
8. 汇总全部通过、失败、未运行项和容差，生成invariant report。
9. 写stage_0b.json及summary；不启动VLM生成或训练。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 0C — VLM Cache Sanity Check

| 字段 | 执行定义 |
|---|---|
| Stage ID | 0C |
| Stage Name | VLM Cache Sanity Check |
| Purpose | 用24个真实task-scene确认缓存与关系接口能产出可用输入，不做大型VLM能力benchmark。 |
| Stage Type | Cache Validation |
| 是否属于论文正式结果 | NO；统计可选写入Implementation说明 |
| 前置条件 | 0A具备API region、SDK、图像和场景绑定；0B的schema/ID/effect/冗余/key检查通过；D0/T_A/T_C各8个dev图像存在。无RL checkpoint。 |
| 输入 | v2.1 prompt/schema/注册ID/合同、固定3个few-shot；模型qwen3.8-max-0902；dashscope1.27.6；actual API endpoint/region；case seeds派生dev_case；API密钥仅经环境提供。 |
| Frozen Parameters | 第8.1章模型快照、温度0、JSON object、非思考、2048 tokens、≤8关系、一次重试、effect锚点、证据权限与cache key。 |
| Tunable Parameters | 只修接口/格式错误；正式产出前允许一次prompt说明澄清但必须升prompt hash并整批重建，不以RL回报挑关系。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| cache_sanity | Frozen API VLM | D0 dev[0:8] | 0 | 0 | Original | 8场景；三任务合计24；非RL训练 |
| cache_sanity | Frozen API VLM | T_A dev[0:8] | 0 | 0 | Original | 8场景；三任务合计24；非RL训练 |
| cache_sanity | Frozen API VLM | T_C dev[0:8] | 0 | 0 | Original | 8场景；三任务合计24；非RL训练 |

### Training Budget

0训练transitions、0 PPO updates；24次首答，最多24次格式/传输重试，总请求≤48；没有语义重试。

### Evaluation Protocol

每scene记录首答JSON合法率、ID合法率、effect合法率、局部合同冗余率、最终空比例、mean边数、两类型数、obvious semantic issue数。人工复核24份，分CORRECT/OBVIOUS_ISSUE/UNDECIDABLE，仅审计不改正式边。

### Required Logging

L_CACHE全部＋request ID、输入哈希、原始/拒绝/去重文件、时间和API错误原因。合法率按原始候选关系为分母；无候选时ID/effect比率N/A。

### Output Directory

`experiments/part_0_validation/stage_0c/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_0c.json`。

### Required Files

- `sanity_cases.json`
- `cache_sanity.csv`
- `semantic_audit.csv`
- `cache_manifest.json`
- `cache_sanity_report.md`
- `stage_0c_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

cache_relation_counts.svg/png：x=scene ID，y=最终关系数，24个原始柱，无均值/平滑。

### Required Tables

cache_summary.csv：24输入数、成功请求数、first_json_valid_n/rate、raw_relation_n、id/effect_valid_n/rate、redundant_n/rate、empty_n/rate、type_count、semantic_issue_n。空请求和API失败分列。

### PASS / STOP Criteria

PASS：24场景均有处理记录、正常完成请求≥20，首答JSON合法≥80%，有至少3个合法非空且非冗余case（含D0≥1）。只有1–2个可用case可PASS_WITH_NOTES继续有限smoke，但不说普遍高产出。全部空/几乎全非法先NEEDS_RERUN排查输入；API/图像权限缺失BLOCKED。部分自然空不是方法失败。

### Failure Handling

检查真实图像和ID→SDK/region/参数→解析schema→effect出处→局部冗余→任务是否有额外语义关系潜力。不得放松合同冗余、补假effects或反复模型抽样以凑非空。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，仅接口修复/已记录版本变更后；原raw保留。 |
| Tuning Allowed | 不允许模型性能调参；最多一次说明性prompt revision，保持v2关系语义。 |
| 是否阻塞论文写作 | NO；缺少可用prior阻塞Full smoke的先验通路验收，不阻塞论文方法。 |
| Next Stage | 1A；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 0C”时，Agent严格执行以下步骤：

1. 核查API、region、模型快照、SDK和冻结prompt/schema hash。
2. 核查24个dev场景均为真实允许输入且不包含未来或答案。
3. 执行1个场景接口检查，作为24份之一而非额外挑选。
4. 按同一参数完成其余请求，严格执行≤1次失败重试。
5. 解析、校验ID/effect、局部去重，保存所有raw及rejected记录。
6. 逐份审计明显语义问题，仅加审计标签不修改保留关系。
7. 计算明确分母的合法/重复/空关系统计并画关系数图。
8. 检查D0是否至少一个有效非空case，定位全空/非法原因。
9. 冻结生产cache协议，写stage_0c.json和summary；不启动1A。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。


---

# Part II — Minimal Learning Validation

## Stage 1A — One-Task Smoke Run

| 字段 | 执行定义 |
|---|---|
| Stage ID | 1A |
| Stage Name | One-Task Smoke Run |
| Purpose | 仅判断B2和Full是否能产生真实成功、保持数值正常且结构通路未永久失活。 |
| Stage Type | Smoke |
| 是否属于论文正式结果 | NO |
| 前置条件 | 0A D0真实平台与时间/安全字段已绑定；0B PASS；0C有D0合法非空prior；D0 train64/dev10 cache齐全；训练入口真实存在。 |
| 输入 | v2.1 code commit、D0合同/感知/evaluator/split、defaults、seed0、无预训练checkpoint。B2/Full从各自同seed初始化开始；硬件使用已验profile。 |
| Frozen Parameters | 方法核心；D0定义与reward；16,384 cap、10episode dev、Original评估；初次lr/B/λQ默认。 |
| Tunable Parameters | None；资源性view分块允许，不改变有效batch。效果差转1B，不在本run暗调参。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| main | B2 | D0 | 0 | 16,384 | 空 | 两个新job；无B0默认 |
| main | Full | D0 | 0 | 16,384 | 80/20；B0/B2无prior | 两个新job；无B0默认 |

### Training Budget

每job最多16,384 transitions=16 full updates；每4 updates保存/评估；最早8 updates检查提前结束；物理cap=2×16,384×D0.d_ref。

### Evaluation Protocol

step0与每4updates：D0 dev前10、deterministic；Full Original、B2空；成功从训练episode或dev真实记录确认。需要随机动作诊断时单列10episode，不与主口径混合。

### Required Logging

L_RUN＋L_TRAIN＋L_DECISION＋L_EVAL；重点reward全零、有效结束、DK/DP非零、Δ、mask、Q loss/grad、循环与timeout。

### Output Directory

`experiments/part_1_smoke/stage_1a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_1a.json`。

### Required Files

- `config.yaml`
- `manifest.json`
- `train_metrics.csv`
- `eval_metrics.csv`
- `episode_log.jsonl`
- `decision_log.jsonl`
- `diagnostics.csv`
- `run_summary.md`
- `smoke_pair_report.md`
- `stage_1a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

smoke_success.svg/png：x=真实skill transitions，y=dev success，B2/Full原始点、不平滑；smoke_dp.svg/png：x=update，y=Full DP_RMS/Δ_RMS，各自独立图。

### Required Tables

smoke_gate_table.csv：method、completed_updates、valid_transitions、success_episodes、NaN_n、DP_nonzero_rate、residual_nonzero_rate、mask_error_n、verdict；不做显著性。

### PASS / STOP Criteria

最早两job各≥8updates且均有真实成功、无严重数值/权限错误、Full在合法非空prior＋非空patch中出现DP和Δ响应即可PASS并停止。达到16updates仍无学习/永久零通路则NEEDS_RERUN转1B诊断；Full无需>B2。初始策略已常成功只能写pipeline可用，不强称学习提升。

### Failure Handling

按第7.2章顺序；先排除空candidate、假reward、Verifier、零patch和冗余先验。数值错误即时停止；正常零成功不修改reward。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，修bug后新attempt；正常调参在1B。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO；只阻塞进入大批训练，不阻塞方法写作。 |
| Next Stage | PASS→2A；否则1B；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 1A”时，Agent严格执行以下步骤：

1. 核查0A/0B/0C及D0资源、缓存与时间manifest。
2. 生成B2/Full seed0两个resolved config并写run_id。
3. 验证初始候选≥2、patch与真实事实隔离，做step0 dev评价。
4. 执行真实skill rollout和PPO，不缓存旧可学习D作为新输入。
5. 每4updates落checkpoint、10episode dev和数值诊断。
6. 在各job第8update起检查宽松smoke门，满足即停止该job。
7. 未达门则运行到16update或实际资源cap，不偷偷延长。
8. 联合核对两job及Full DP/Δ合法非零证据。
9. 归档原始日志，写pair report、stage_1a.json；不自动进入1B/2A。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 1B — Minimal Tuning

| 字段 | 执行定义 |
|---|---|
| Stage ID | 1B |
| Stage Name | Minimal Tuning |
| Purpose | 在实现正确的前提下，用最多6个单变量trial找到能正常学习的配置，不做grid。 |
| Stage Type | Exploration |
| 是否属于论文正式结果 | NO |
| 前置条件 | 1A完成但学习/数值现象不满足smoke；1A正常PASS则本Stage直接记录skipped，不启动job。若有权限bug先修并重验对应0B测试。 |
| 输入 | 1A全量日志/checkpoint、D0默认config、v2.1源码、同D0 cache/split、seed0；trial从头开始，长度延长可从同配置同seed完整checkpoint继续且记录parent。 |
| Frozen Parameters | 方法/奖励/任务/80-20/schema/patch/目标；single-factor trial；最多6个新增job。 |
| Tunable Parameters | lr→训练长度→entropy→B→λQ；每量含默认最多3值，详第7.2章；只改一个量。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| conditional_trial | B2或Full（只跑受影响者） | D0 | 0 | 16,384 | Full80/20；B2空 | 最多6个新job；length trial cap32,768；每job config显式登记 |

### Training Budget

默认每trial16,384/16updates，length trial32,768/32updates；评估间隔4updates×10episodes；总新增训练job≤6，每个cap物理时间2×N×d_ref。不要求把每个参数都试完。

### Evaluation Protocol

同1A；候选配置优先数值正常、真实成功出现、DP/Δ工作，再看dev最近3点均值；不是比较谁能击败baseline。被选配置若影响两个方法，在剩余trial额度中做另一方法确认。

### Required Logging

L_RUN/L_TRAIN/L_DECISION/L_EVAL＋trial_id、hypothesis、changed_field、old/new、reason、accepted、parent_run与seen_results。

### Output Directory

`experiments/part_1_smoke/stage_1b/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_1b.json`。

### Required Files

- `trial_registry.jsonl`
- `tuning_history.jsonl`
- `tuning_results.csv`
- `selected_config.yaml`
- `tuning_report.md`
- `stage_1b_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

minimal_tuning_success.svg/png：x=skill transitions，y=dev success，按单变量trial叠加、原始点不平滑；无数据时不造图。

### Required Tables

tuning_results.csv：trial、method、changed_field、old/new、updates、success、DP_active、numeric_issue、selected；只作开发选择。

### PASS / STOP Criteria

配置满足1A门即PASS停止；无需穷尽trial。1A已PASS则PASS_WITH_NOTES＋skipped。6个job用完仍无成功/通路失效写NEEDS_RERUN或结构问题报告，不无限搜索。硬资源缺失BLOCKED。

### Failure Handling

先实现与输入链，再按唯一调参顺序；length只延长真实训练；无成功不暗加curriculum/BC/reward。VLM自然关系不足可换已注册适合D0场景模板revision，不能篡改缓存。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，限制在本Stage额度；新增授权另记revision。 |
| Tuning Allowed | 仅上述五类量；hidden/batch只在资源明确原因且回归单测后改。 |
| 是否阻塞论文写作 | NO。 |
| Next Stage | 2A；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 1B”时，Agent严格执行以下步骤：

1. 检查1A；已PASS就写skipped并立即停止。
2. 按固定排查顺序确认不是mask/reward/合同/Verifier或重算bug。
3. 在trial_registry登记一个最有依据的单变量调整。
4. 使用seed0同D0数据链运行受影响方法，保存全部trial日志。
5. 每4updates评估10episode，检查真实成功与结构通路。
6. 满足smoke即停止搜索，保存selected_config。
7. 未满足且额度尚有才进入下一个有依据的值/参数，绝不做grid。
8. 核查是否需要对另一方法确认配置，计入6job上限。
9. 生成tuning_report和stage_1b.json，停止，不运行2A。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。


---

# Part III — Core Paper Evidence

## Stage 2A — Core Exploration Runs

| 字段 | 执行定义 |
|---|---|
| Stage ID | 2A |
| Stage Name | Core Exploration Runs |
| Purpose | 在两个结构任务中观察四个核心方法的学习现象，确定任务展示价值与适合的预算。 |
| Stage Type | Exploration |
| 是否属于论文正式结果 | OPTIONAL：第一版Results可用，必须标exploratory |
| 前置条件 | 1A PASS或1B选定有效配置；T_A/T_C均已绑定；B0/B1代码和Q头已通过相关0B测试；train64/dev20 cache齐全。 |
| 输入 | 当前resolved defaults/1B选定config；T_A/T_C及其同task dev；B0/B1/B2/Full；seeds012；从头初始化；v2.1 commit、平台、合同、evaluator、cache冻结。 |
| Frozen Parameters | 方法核心、各task及split、reward、观测/技能/mask、80/20；每个run内config锁定。 |
| Tunable Parameters | run之间允许第7.2章有限单变量适配；额外≤4诊断job或一次第3task完整12job，均新revision；默认矩阵不自动扩。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| main | B0 | T_A | 0,1,2 | 65,536 | 空 | 每行3个seed，合计24新job |
| main | B1 | T_A | 0,1,2 | 65,536 | 80/20；B0/B2无prior | 每行3个seed，合计24新job |
| main | B2 | T_A | 0,1,2 | 65,536 | 空 | 每行3个seed，合计24新job |
| main | Full | T_A | 0,1,2 | 65,536 | 80/20；B0/B2无prior | 每行3个seed，合计24新job |
| main | B0 | T_C | 0,1,2 | 65,536 | 空 | 每行3个seed，合计24新job |
| main | B1 | T_C | 0,1,2 | 65,536 | 80/20；B0/B2无prior | 每行3个seed，合计24新job |
| main | B2 | T_C | 0,1,2 | 65,536 | 空 | 每行3个seed，合计24新job |
| main | Full | T_C | 0,1,2 | 65,536 | 80/20；B0/B2无prior | 每行3个seed，合计24新job |

### Training Budget

每job65,536 transitions=64updates；每8updates保存/20episode dev；物理cap=2×65,536×task.d_ref。共24主job；无全局“训练到收敛”。

### Evaluation Protocol

dev前20，step0和每8updates，deterministic，Original；B0/B2空。按最近3点选择开发checkpoint；本Stage不必须访问test_id。允许报告zero-success正常seed。

### Required Logging

L_RUN/L_TRAIN/L_DECISION/L_EVAL；每task汇总四条曲线、success、AUC、seed波动、DK/DP/Δ、candidate/relation数及实际交互时间。

### Output Directory

`experiments/part_2_core/stage_2a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_2a.json`。

### Required Files

- `exploration_run_index.csv`
- `task_suitability_report.md`
- `exploration_summary.csv`
- `recommended_stage_2b.yaml`
- `mechanism_task_selection.yaml`
- `results_draft_v0.md`
- `stage_2a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

explore_<task>_success.svg/png：x=skill transitions，y=dev success，四方法seed均值＋各seed原始淡线，显示MA3；explore_<task>_dp.svg/png：Full DP_RMS随update独立图；各方法不适用D写N/A。

### Required Tables

exploration_summary.csv：task、method、seed_count、selected_dev_checkpoint、success、AUC_common、std、zero_success_seed_n、candidate_mean、relation_mean、DP_nonzero_rate、physical_seconds；先seed后task聚合。

### PASS / STOP Criteria

PASS：矩阵主要运行结束并有可解释原始曲线、完整失败记录和task suitability。某正常seed零成功/Full不领先不阻塞。缺个别技术run可PASS_WITH_NOTES并列实际n及未完成项；某方法在某task完全没有有效run则NEEDS_RERUN，不能造比较。

### Failure Handling

先技术问题，再输入与D/Δ，再有限参数；评估task结构意义，不以Full赢家作为唯一task准入条件。若Full持续严重异常，写method_issue_report但不自动改Method。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，有限诊断或基础设施修复；保留所有attempt。 |
| Tuning Allowed | 默认有限lr/长度/entropy/B/λQ；方法间适配写明；超额度停止。 |
| 是否阻塞论文写作 | NO；此时即可写Experimental Setup与探索Results。 |
| Next Stage | 第一批优先3A；需要主展示则显式执行2B；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Task Suitability Report字段
每task给结构类型、asset readiness、可执行候选范围、非空prior率、非冗余关系例、典型失败、四方法真实现象、推荐metric、推荐claim范围；类别限定为Main-paper strong candidate / Main-paper usable / Appendix–weak separation / Not useful for presentation。类别表示展示解释价值，不代表科学方法胜负。

### Agent Execution Template

当用户说“执行 Stage 2A”时，Agent严格执行以下步骤：

1. 读取Stage前置与两个task实际绑定，解析所有config及cache hash。
2. 展开2tasks×4methods×3seeds为24行run registry。
3. 先跑各method一个完整update检查日志与mask，再继续同run。
4. 按64update cap运行全部矩阵，保留每8update的checkpoint和20episode dev。
5. 对技术失败做同seed新attempt；正常零成功不得重标异常。
6. 汇总success/AUC/稳定性/结构诊断，生成每task四方法图。
7. 用结构需求、可执行性、机制可解释性与真实结果分类task suitability。
8. 登记2B的3task建议（第三task须真实绑定）和3A/3B/3C候选task。
9. 形成第一版exploratory Results草稿与stage_2a.json，停止，不自动扩实验。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 2B — Presentation Runs

| 字段 | 执行定义 |
|---|---|
| Stage ID | 2B |
| Stage Name | Presentation Runs |
| Purpose | 以已基本确定的配置获取三个主任务、四个方法、三个seed的私人论文展示结果。 |
| Stage Type | Presentation |
| 是否属于论文正式结果 | YES：标明所选任务/seed/checkpoint视图 |
| 前置条件 | 2A报告及推荐config存在；T_A/T_B/T_C三任务真实绑定（允许报告中明确替换）；选定seed列表和presentation manifest落盘；测试case/cache可用。 |
| 输入 | 2A选定defaults/少量per-method overrides；默认T_A/T_B/T_C；B0/B1/B2/Full；seeds012；从头训练；当前verified commit与平台；test_id前30固定。 |
| Frozen Parameters | 每run配置、观测/合同/任务/reward/cache、checkpoint选择算法；使用同task dev选模型，不用单个test尖峰。 |
| Tunable Parameters | None在已启动run内。补seed3/4、任务扩展或超参变更必须新presentation revision且预算单列。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| main | B0 | T_A | 0,1,2 | 131,072 | 空 | 默认36新job；不是默认复用2A曲线拼接 |
| main | B1 | T_A | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 默认36新job；不是默认复用2A曲线拼接 |
| main | B2 | T_A | 0,1,2 | 131,072 | 空 | 默认36新job；不是默认复用2A曲线拼接 |
| main | Full | T_A | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 默认36新job；不是默认复用2A曲线拼接 |
| main | B0 | T_B | 0,1,2 | 131,072 | 空 | 默认36新job；不是默认复用2A曲线拼接 |
| main | B1 | T_B | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 默认36新job；不是默认复用2A曲线拼接 |
| main | B2 | T_B | 0,1,2 | 131,072 | 空 | 默认36新job；不是默认复用2A曲线拼接 |
| main | Full | T_B | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 默认36新job；不是默认复用2A曲线拼接 |
| main | B0 | T_C | 0,1,2 | 131,072 | 空 | 默认36新job；不是默认复用2A曲线拼接 |
| main | B1 | T_C | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 默认36新job；不是默认复用2A曲线拼接 |
| main | B2 | T_C | 0,1,2 | 131,072 | 空 | 默认36新job；不是默认复用2A曲线拼接 |
| main | Full | T_C | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 默认36新job；不是默认复用2A曲线拼接 |

### Training Budget

每job131,072 transitions=128updates；每16updates checkpoint＋20episode dev；完成后selected checkpoint做30episode test_id。默认36新job；资源cap=2×N×d_ref。扩到5tasks为60主job，不自动执行。

### Evaluation Protocol

训练中dev前20；按3连续窗口选择各method自己的checkpoint；最终test_id前30、deterministic、Original。需50时仅追加20。同一seed三方法/四方法可共享初始case，训练环境轨迹自然不同。

### Required Logging

L_RUN/L_TRAIN/L_DECISION/L_EVAL全套＋checkpoint_selection.json、selected seed标志、presentation revision和全部可用run候选。

### Output Directory

`experiments/part_2_core/stage_2b/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_2b.json`。

### Required Files

- `presentation_manifest.yaml`
- `presentation_run_index.csv`
- `checkpoint_selection.json`
- `presentation_results.csv`
- `seed_selection_notes.md`
- `stage_2b_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

仅每run核查曲线；正式合并图交2C。run_success.svg/png：x=skill transitions、y=dev success，原始点＋MA3。

### Required Tables

presentation_results.csv：task、family、method、training_seed、run_id、checkpoint/hash、step、dev_window_score、test_success_n/n/rate、return、completion_time、actual_budget；不先手工四舍五入再聚合。

### PASS / STOP Criteria

PASS：真实三任务矩阵及可追溯checkpoint/test日志齐备；不要求Full优于基线。某seed失败可PASS_WITH_NOTES并声明实际n或有限补3/4；某任务未绑定则BLOCKED该矩阵，不能拿两任务称三任务。

### Failure Handling

技术错误同seed重跑；普通差异小先保留结果，再在Paper View选择适当metric/子集。反复调参必须回到新探索revision，不在本run中隐式更改。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，技术修复或有记录presentation revision。 |
| Tuning Allowed | 本Stage运行内None；少量已登记方法适配只能在run前生效。 |
| 是否阻塞论文写作 | NO；本Stage未完成只阻塞最终Main Results数字，不阻塞正文其他部分。 |
| Next Stage | 2C；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 2B”时，Agent严格执行以下步骤：

1. 读取2A建议，绑定并核查第三任务，生成presentation manifest。
2. 锁定任务、各方法config与默认seeds，展开36个新job。
3. 从头运行128update配额，保存16update间隔开发评价。
4. 以每个task的dev连续3窗口规则选checkpoint，不按test挑选。
5. 在固定30个test_id case上评价冻结checkpoint。
6. 技术失败用同seed新attempt；需要额外seed只能依已记录理由补3/4。
7. 保存所有seed与checkpoint候选，标出拟展示三个seed及原因。
8. 汇总真实presentation_results和checkpoint selection。
9. 写stage_2b.json与summary，停止，不自动生成2C或消融。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 2C — Main Figure & Table Generation

| 字段 | 执行定义 |
|---|---|
| Stage ID | 2C |
| Stage Name | Main Figure & Table Generation |
| Purpose | 只将2B真实结果转换成主图主表，不增加训练或评价。 |
| Stage Type | Packaging |
| 是否属于论文正式结果 | YES |
| 前置条件 | 2B至少有可用于展示的真实结果；presentation manifest和所有source hash可解析；无test日志则BLOCKED数字图表。 |
| 输入 | 2B run index、train/dev/test logs、checkpoint selection、selected tasks/seeds、plot config；同一已验软件profile中的Pandas/Matplotlib；不载入模型执行。 |
| Frozen Parameters | 原始数值、source hash、指标公式；success分母；不插入未运行baseline；默认MA3仅用于显示。 |
| Tunable Parameters | 仅展示：metric、明确task subset、mean/median（默认mean）、MA3或EMA(.3)二选一、坐标zoom；全部记录，不改raw。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| report | B0/B1/B2/Full | 2B selected 3 tasks | 0,1,2 | 0 | 读取Original结果 | 0训练；0新增episode |

### Training Budget

0 transitions、0 updates、0 environment seconds；只读取文件。

### Evaluation Protocol

无新评价；重算per-seed success、共同区间AUC及task-family均值，与原始episode逐项核对。

### Required Logging

输入/输出SHA、selection manifest、metric/smoothing版本、聚合层级、n/tasks/seeds/episodes、缺失值/截取范围和图表生成命令。

### Output Directory

`experiments/part_2_core/stage_2c/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_2c.json`。

### Required Files

- `table_1_main_success.csv`
- `table_2_efficiency.csv`
- `figure_sources.json`
- `plot_config.yaml`
- `selection_manifest.yaml`
- `stage_2c_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_1_<task>_learning.svg/png：x=skill transitions，y=dev success，四方法seed均值、MA3、SD；fig_2_family_result.svg/png：x=family，y=最终test success，四方法、无平滑。每task独立画布。

### Required Tables

Table1：Task、B0、B1、B2、Full，每格selected-seed mean±SD、n；Table2：Task、Method、AUC_common、time_to_70、success-conditioned completion time、success_n、physical budget。AUC/时间不适用写N/A。

### PASS / STOP Criteria

PASS：每个图表值可回溯到2B真实source，重算误差≤1e-10（展示四舍五入前），图轴/标签/分母齐全。Full不领先仍PASS。缺源/不一致为NEEDS_RERUN报告脚本，不重新训练。

### Failure Handling

先source路径/hash→episode分母→seed/task聚合→共同横轴→平滑→图轴。禁止通过改原始CSV修图。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，仅重生成派生图表；新selection revision。 |
| Tuning Allowed | 仅展示设置，不能训练调参。 |
| 是否阻塞论文写作 | NO；图表不齐只阻塞最终结果插图。 |
| Next Stage | 按证据缺口3A/3B；或6A；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 2C”时，Agent严格执行以下步骤：

1. 加载2B真实source及presentation/selection manifest。
2. 校验所有run/checkpoint/cache hash与样本分母。
3. 用原始episode重算per-seed success与duration。
4. 用原始dev曲线计算共同横轴AUC与time_to_70。
5. 按任务/seed层级聚合，保留所选子集标签。
6. 生成Figure1、Figure2、Table1及可用Table2。
7. 逐图保存smoothing、轴范围、source与selected seed说明。
8. 生成figure_sources和stage_2c_summary；更新状态后停止。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。


---

# Part IV — Core Mechanism Evidence

## Stage 3A — Dual-Difference Ablation

| 字段 | 执行定义 |
|---|---|
| Stage ID | 3A |
| Stage Name | Dual-Difference Ablation |
| Purpose | 在一个先验具有解释价值的任务中比较DP与DH，隔离第二次差分。 |
| Stage Type | Ablation |
| 是否属于论文正式结果 | YES；第一批为exploratory mechanism evidence |
| 前置条件 | 2A有T_C Full三个seed、可追溯config和checkpoint；A_DD开关通过T25。可在2B之前执行；无匹配Full则将补跑Full显式计入本Stage矩阵。 |
| 输入 | T_C默认或2A选定1个有合法非空prior的task；Full源为2A同task seeds012；A_DD同样seeds从头训练；同task cache、同预算64updates、相同其余超参、平台/commit语义。 |
| Frozen Parameters | 只改prior输入DP→DH；空prior仍显式0；DK主通路、F_P、B、Q target、80/20、reward、预算、其余超参不变。 |
| Tunable Parameters | None；允许资源分块，不改有效batch；需不同稳定化适配时降级为适配比较，不称单因素。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| reference | Full | T_C | 0,1,2 | 65,536 | 80/20；B0/B2无prior | 复用2A匹配Full；0新增训练 |
| main | A_DD | T_C | 0,1,2 | 65,536 | 80/20；B0/B2无prior | 3新job；未训练Full不得冒称复用 |

### Training Budget

A_DD每job65,536=64updates；每8updates×20dev episodes。Full复用0新训练；最终比较可对两个method selected checkpoint各做30个test_id，最多180evaluation episodes，日志已有且完全匹配可复用。

### Evaluation Protocol

checkpoint用同task dev连续3点选；test_id前30、deterministic、Original；同seeds、同case；可追加到50但不强制。报告实际empty prior比例，A_DD空输入0。

### Required Logging

L_RUN/L_TRAIN/L_DECISION/L_EVAL＋A_DD具体input-source标记、empty/nonempty分组、DK/DH/DP RMS、Δ饱和、Q loss；Full reuse provenance。

### Output Directory

`experiments/part_3_ablation/stage_3a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_3a.json`。

### Required Files

- `comparison_manifest.json`
- `ablation_dd.csv`
- `reused_full_runs.json`
- `dd_report.md`
- `stage_3a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_3a_dd_learning.svg/png：x=skill transitions，y=dev success，Full/A_DD，seed均值＋MA3；fig_3a_prior_input.svg/png：x=update，y=有效候选输入RMS，按方法独立说明量不同，不当价值。

### Required Tables

ablation_dd.csv：task、method、n_seeds、test_success、AUC_common、checkpoint_step、empty_rate、prior_input_RMS、residual_RMS、source_run；每seed先算再均值。

### PASS / STOP Criteria

PASS：Full/A_DD匹配比较完成、空prior接口正确、真实曲线与差异可解释；无差异或A_DD更好也PASS并限制claim。全DP严格零先排查T23与输入，不能用胜负门替代工程门。

### Failure Handling

先核对A_DD是否空prior误用DK→是否改变额外架构/超参→Full复用hash→target/梯度→数值。无收益不重新加static prior。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，技术错误时；新预算比较须同时有匹配Full。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO；只阻塞“双差分优于直接DH”的经验句。 |
| Next Stage | 第一批在此暂停，整理Results；之后显式2B或3B；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 3A”时，Agent严格执行以下步骤：

1. 读取2A的机制task选择，默认T_C，不按test成绩重新挑task。
2. 校验Full三个源run的全部对照条件与预算，登记复用。
3. 构造A_DD唯一开关并运行T25空prior/非空prior检查。
4. 从头运行A_DD三个seed，每8updates评估20个dev episode。
5. 按相同dev规则选Full/A_DD的checkpoint。
6. 在同一30个test_id case评价或复用完全相同的已有评价。
7. 计算success/AUC与输入残差行为，保留负向/接近结果。
8. 完成dd_report、source manifest和stage_3a.json。
9. 停止扩实验，交付第一版Results证据，不自动跑3B。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 3B — Structural Q Ablation

| 字段 | 执行定义 |
|---|---|
| Stage ID | 3B |
| Stage Name | Structural Q Ablation |
| Purpose | 判断实际动作Q监督对学习速度、稳定性和最终成功的作用。 |
| Stage Type | Ablation |
| 是否属于论文正式结果 | YES；按论文证据缺口执行 |
| 前置条件 | 已有匹配Full（默认2B T_A三个seed128updates）；A_Q开关通过T25；不要求3A性能提升。 |
| 输入 | 默认T_A，或2A选定代表task；Full seeds012源run；A_Q相同seed、task、cache、128update预算；除λQ外相同config和代码。 |
| Frozen Parameters | 仅λQ=0；Actor/V/图/GRU/80-20/reward/优化器其余设置不变；不同时删层或更换数据。 |
| Tunable Parameters | None。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| reference | Full | T_A | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 复用2B匹配Full |
| main | A_Q | T_A | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 3新job |

### Training Budget

A_Q每job131,072=128updates；16update×20dev评估；Full0新训练。最终两个method各3seed×30case=180评价episode，已有匹配Full评价可复用。

### Evaluation Protocol

Original、deterministic、同task dev选checkpoint，test_id前30；以原始学习曲线AUC、达到阈值位置、正常零成功seed数、final success和跨seedSD描述，不强制显著性。

### Required Logging

L_RUN/L_TRAIN/L_DECISION/L_EVAL＋各loss的共享梯度norm定期抽样（每16updates）；A_Q的raw Q误差如记录必须标UNTRAINED_HEAD，不称其较差证明Q监督有效。

### Output Directory

`experiments/part_3_ablation/stage_3b/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_3b.json`。

### Required Files

- `comparison_manifest.json`
- `ablation_q.csv`
- `q_report.md`
- `stage_3b_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_3b_q_learning.svg/png：x=skill transitions，y=dev success，Full/A_Q，mean＋MA3；fig_3b_seed_auc.svg/png：x=method，y=per-seed AUC，原始点无平滑。

### Required Tables

ablation_q.csv：task、method、n、success_mean/std、AUC_mean/std、time_to_70、zero_success_seed_n、actual_budget；Q loss下降只作诊断。

### PASS / STOP Criteria

PASS：匹配比较完整且λQ之外未变；效应小、无效或负效应均可PASS。Q导致技术性NaN则NEEDS_RERUN先修复；多个正常负结果不自动加新loss。

### Failure Handling

核对target detach/边界→只gather已执行→共享参数重复更新→Q权重/梯度尺度→数据预算。需要调λQ只能新的探索revision，不在此消融偷偷调。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，技术修复后；保留原比较。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO；不阻塞方法，未运行则不写Q稳定提高收益。 |
| Next Stage | 3C或4A（需新指令）；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 3B”时，Agent严格执行以下步骤：

1. 读取匹配Full源run和选定代表task。
2. 确认A_Q只将λQ置零，Actor/V图结构不变。
3. 检查真实Q target和共享参数实现已通过单测。
4. 从头训练A_Q三个seed，按16update间隔保存/评价。
5. 按同task dev规则选择checkpoint并固定。
6. 读取/运行配对30episode test评价。
7. 从真实曲线计算AUC、稳定性与最终成功，不只看Q误差。
8. 生成q_report、比较表和stage_3b.json，停止。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 3C — Bounded Prior Ablation

| 字段 | 执行定义 |
|---|---|
| Stage ID | 3C |
| Stage Name | Bounded Prior Ablation |
| Purpose | 在一个prior-sensitive任务中比较有界和无界残差，保存供robustness比较使用的模型。 |
| Stage Type | Ablation |
| 是否属于论文正式结果 | OPTIONAL；低于3A/3B优先级 |
| 前置条件 | 2A确定T_C有先验敏感性；默认2B Full T_C三个seed可复用；A_B开关T25通过。 |
| 输入 | T_C、seeds012；Full复用2B；A_B从头；相同128update、B数值、reward/合同/mask/80-20/cache。 |
| Frozen Parameters | 仅去掉tanh，Δ=B*s；不去掉zero anchor、不换F_P、不增加bias、不改B或其他优化参数。 |
| Tunable Parameters | None。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| reference | Full | T_C | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 复用2B Full |
| main | A_B | T_C | 0,1,2 | 131,072 | 80/20；B0/B2无prior | 3新job |

### Training Budget

A_B每job131,072/128updates；每16updates×20dev；最终Original30episode/seed。测试corruption在4B执行，不在此默认额外扩条件。

### Evaluation Protocol

Original配对评价；正常prior下没差异可完成。本Stage产出冻结A_B checkpoint；若4B随后被授权，可增加Full/A_B相同扰动对照而无需新训练。

### Required Logging

L_RUN/L_TRAIN/L_DECISION/L_EVAL＋abs residual分位数、|Δ|>B比例、base logit margin、actor_entropy、梯度与NaN；A_B不受B界不是实现错误。

### Output Directory

`experiments/part_3_ablation/stage_3c/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_3c.json`。

### Required Files

- `comparison_manifest.json`
- `ablation_bound.csv`
- `boundedness_report.md`
- `ablation_checkpoints.json`
- `stage_3c_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_3c_residual_distribution.svg/png：x=|Δ|，y=有效候选频率，Full/A_B原始直方；fig_3c_learning.svg/png：x=skill transitions，y=dev success，mean＋MA3。

### Required Tables

ablation_bound.csv：method、n、success/AUC、residual_p50/p95/max、over_B_fraction、numeric_issue_n、checkpoint_hash；按run统计再聚合。

### PASS / STOP Criteria

PASS：唯一开关正确且真实运行可比较；Original差异小无需重跑。无界分支数值异常如实记录，不用隐藏clip强行救活后仍称原A_B。

### Failure Handling

先开关/额外bias/倍率→zero anchor→大残差/梯度→是否溢出。技术问题修复；纯算法不稳定保留现象，不更改其他超参掩盖。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，技术修复后；纯性能不强制。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO。 |
| Next Stage | 4A/4B（需指令）；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 3C”时，Agent严格执行以下步骤：

1. 读取T_C Full参考与相同B/config/hash。
2. 确认A_B仅删除tanh，zero anchor仍成立。
3. 从头训练三个A_B seed，保存完整残差和数值日志。
4. 按16update间隔开发评价，观察但不隐藏过大残差。
5. 按同一dev规则冻结checkpoint。
6. 执行/复用Original30episode配对评价。
7. 生成残差分布、成功/AUC表与待4B模型索引。
8. 写stage_3c.json并停止，不自动运行corruption。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。


---

# Part V — Test-Time Mechanism & Robustness

## Stage 4A — Actor Dependence Diagnostics

| 字段 | 执行定义 |
|---|---|
| Stage ID | 4A |
| Stage Name | Actor Dependence Diagnostics |
| Purpose | 分别测同一真实快照上的Actor敏感性与闭环后果，区分使用结构和使用得有益。 |
| Stage Type | Mechanism |
| 是否属于论文正式结果 | OPTIONAL |
| 前置条件 | 至少一个真实Full checkpoint及evaluation decision日志；默认2B T_C三个seed；真实有效候选/goal/mask、GRU prefix可重建；不需新训练。 |
| 输入 | T_C Full seeds012 selected checkpoints；test_id前20；每seed从实际Original episode按episode轮询取至多64快照（每episode≤4，候选≥2）；固定source/history、DP shuffle seed、原prior及合法target-swap编辑。 |
| Frozen Parameters | 参数、任务、感知、合同、mask、reward、原始快照内容；干预只发生在规定的特征/软边输入，不反向写真实事实。 |
| Tunable Parameters | None；可缩减至1个已有seed作首轮，但必须标n=1；默认3seed。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| snapshot+closed_loop | Full frozen | T_C | 0,1,2 | 0 | Original | 64真实快照/seed；闭环20episode/seed；不可编辑条件标N/A |
| snapshot+closed_loop | Full frozen | T_C | 0,1,2 | 0 | DK=0 | 64真实快照/seed；闭环20episode/seed；不可编辑条件标N/A |
| snapshot+closed_loop | Full frozen | T_C | 0,1,2 | 0 | DP=0 | 64真实快照/seed；闭环20episode/seed；不可编辑条件标N/A |
| snapshot+closed_loop | Full frozen | T_C | 0,1,2 | 0 | DP shuffle | 64真实快照/seed；闭环20episode/seed；不可编辑条件标N/A |
| snapshot+closed_loop | Full frozen | T_C | 0,1,2 | 0 | remove prior | 64真实快照/seed；闭环20episode/seed；不可编辑条件标N/A |
| snapshot+closed_loop | Full frozen | T_C | 0,1,2 | 0 | target-swap | 64真实快照/seed；闭环20episode/seed；不可编辑条件标N/A |

### Training Budget

0训练transitions/updates。最多3×64×6=1,152个固定快照前向；最多3×6×20=360个闭环evaluation episodes，已有同checkpoint/case的Original可复用。物理时间另计，不把前向次数算环境步。

### Evaluation Protocol

快照：相同历史/GRU、candidate/goal绑定，逐条件重算logits，记录JS、KL(Poriginal||Pcondition)、argmax变化。闭环：各condition独立reset同初始case，从t0持续施加干预，deterministic；后续状态允许自然分化。

### Required Logging

L_EVAL＋原始/干预action distributions、masked support、DK/DP/RMS、Δ、selected_skill、JS/KL、argmax change、快照/原history引用、condition适用性、before/after模型hash；L_DECISION闭环全量。

### Output Directory

`experiments/part_4_mechanism/stage_4a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_4a.json`。

### Required Files

- `intervention_manifest.json`
- `snapshot_diagnostics.csv`
- `closed_loop_diagnostics.csv`
- `actor_dependence_report.md`
- `stage_4a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_4a_js.svg/png：x=condition，y=per-snapshot JS，按seed汇总原始分布；fig_4a_closed_loop.svg/png：x=condition，y=success，selected-seed mean，无平滑。

### Required Tables

actor_interventions.csv：condition、eligible_snapshots、mean_JS/KL、argmax_change_rate、Δ_RMS、closed_loop_n/success、source_checkpoint；single-step与closed-loop分开。

### PASS / STOP Criteria

PASS表示诊断完整、干预实现正确；Actor变化不明显仍可PASS_WITH_NOTES并报告未见依赖证据。固定快照DP=0与remove-prior应得到相同分布（FP32tol），不一致为NEEDS_RERUN。非零JS只说明敏感，不自动说明有益。

### Failure Handling

确认快照/history一致→candidate/goal对齐→干预是否生效→DP本来是否零→基础margin→mask单候选。只有在多项合法非退化case都不响应时写结构问题报告。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，仅重复评价/修诊断代码，参数冻结。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO；缺证据就不写Actor有益利用。 |
| Next Stage | 4B；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### 干预的精确定义
`DK=0`：只将进入F_K的DK置零，重算uK/b及依赖uK的prior branch；DP本身仍为原始四路结果。它是特征干预，不声称对应某个真实合同图。
`DP=0`：固定c、uK，prior输入置零，因zero-anchor使Δ=0。
`shuffle DP`：仅在同snapshot允许候选间将整个goal-aligned DP矩阵作循环非零offset置换；offset从1..K−1均匀抽取；不乱排goal行/对象绑定，K<2不适用。
`remove prior`：删除全部语义及对应反向边后重新四路编码；同快照的GK/DK/context不变。
`target-swap`：第4B冻结算法；无合法对写N/A。
JS使用自然对数及同一个有效mask支持集；零概率项按0log0=0处理，不对padding计算KL。原始/干预分布均保存，不只输出一个距离。

### Agent Execution Template

当用户说“执行 Stage 4A”时，Agent严格执行以下步骤：

1. 校验Full checkpoint/hash和真实Original decision日志。
2. 按固定round-robin规则取最多64个真实快照/seed，不按reward选快照。
3. 实现DK零、DP零、有效候选DP derangement、先验删除和target-swap。
4. 固定snapshot/history/mask测action分布，验证DP0与remove-prior等价。
5. 记录JS/KL/argmax/残差，目标行与对象绑定不能错位。
6. 对每condition从相同初始case独立跑20episode闭环，持续施加干预。
7. 分别汇总固定状态敏感性与真实最终success，记录N/A分母。
8. 校验模型权重未变，生成actor_dependence_report和stage_4a.json并停止。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 4B — Prior Robustness Evaluation

| 字段 | 执行定义 |
|---|---|
| Stage ID | 4B |
| Stage Name | Prior Robustness Evaluation |
| Purpose | 在冻结模型上测先验缺省、缺边、端点错配及合格无关边的影响，不引入训练corruption。 |
| Stage Type | Robustness |
| 是否属于论文正式结果 | OPTIONAL |
| 前置条件 | Full T_C三个冻结checkpoint、test_id前20和原始cache可用；target-swap编辑与irrelevant适用性仅依task/初始输入判定。3C完成时可加入已训练A_B，不作为Full-only执行的前置。 |
| 输入 | 默认1个task T_C，Full seeds012；Original/No-prior/Deletion/Target-swap/Irrelevant；独立prior_robustness_eval seed；相同20初始cases和固定checkpoint。 |
| Frozen Parameters | 全部模型参数、任务/事实/合同/mask/reward、每条件checkpoint；deletion p=.5，swap1edge，irrelevant最多1edge且总预算≤8。 |
| Tunable Parameters | None；可按需一次扩至50episode，或加第2task。使用已有A_B属显式scope扩展，不新增训练。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| robustness | Full frozen | T_C | 0,1,2 | 0 | Original | 20episode/seed；N/A不计作扰动成功 |
| robustness | Full frozen | T_C | 0,1,2 | 0 | No-prior | 20episode/seed；N/A不计作扰动成功 |
| robustness | Full frozen | T_C | 0,1,2 | 0 | Deletion(p=.5) | 20episode/seed；N/A不计作扰动成功 |
| robustness | Full frozen | T_C | 0,1,2 | 0 | Target-swap(1) | 20episode/seed；N/A不计作扰动成功 |
| robustness | Full frozen | T_C | 0,1,2 | 0 | Irrelevant(+1) | 20episode/seed；N/A不计作扰动成功 |

### Training Budget

0训练updates/transitions；默认最多1task×3seeds×5conditions×20=300评价episode（适用性可能减少）；一次扩50为750。第二task×2；显式加入A_B再×2。不要默认叠加所有扩展。

### Evaluation Protocol

deterministic；每case编辑一次并固定整episode，method/训练seed共用同case边编辑。按条件适用子集重报Original；Original全部为空仍可报告No-prior，但不冒称未见错误类型鲁棒。

### Required Logging

L_EVAL/L_DECISION＋original/edited edge lists与hash、semantic forward/reverse列表、实际删数、EMPTY_SOURCE/NO_EDGE_DELETED/N_A、独立无关性证据、residual/argmax、成功/失败。

### Output Directory

`experiments/part_4_mechanism/stage_4b/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_4b.json`。

### Required Files

- `robustness_cases.json`
- `edited_priors.jsonl`
- `robustness_metrics.csv`
- `applicability_table.csv`
- `robustness_report.md`
- `stage_4b_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_4b_success.svg/png：x=condition，y=绝对success，mean across training seeds；fig_4b_degradation.svg/png：x=condition，y=相对同子集Original的百分点变化；无平滑。

### Required Tables

robustness_metrics.csv：task、method、condition、eligible_case_n、training_seed_n、episodes_n、success_rate、original_matched_rate、delta_pp、empty_rate、residual_p95；N/A理由另表。

### PASS / STOP Criteria

PASS：适用条件真实完成、编辑/分母/对照正确；趋势明显或无明显差异均可停止。Irrelevant不适用并不阻塞；不制造任务来凑表。N/A误算成功、改mask/reward、编辑逆边遗漏为NEEDS_RERUN。

### Failure Handling

先编辑合法性/反向边→源cache→pairing→checkpoint→环境版本。语义错配不一定错误，禁止为让Full更好而挑编辑seed。结果差时收缩robustness claim。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，诊断bug或明确一次扩50；不重训。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO。 |
| Next Stage | 5A或6A；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### 测试编辑算法（唯一版本）
**Original**：用原始解析/校验/局部去重后soft edges，不修语义。
**No-prior**：整份空，不改别的输入。
**Deletion**：每条语义正向边独立以0.5概率删；不分类型、不保证留边或删边；原始空记EMPTY_SOURCE，恰好无删除记NO_EDGE_DELETED并保留，不能拒绝重抽。
**Target-swap**：枚举全部合法(原边,new target)对；固定source/type/effect_ref；排除原target、自指supports、已有同型边、局部合同冗余和独立任务禁止项；relevant-to-goal仍指向现有goal；在全部合法对中均匀选一对，替换1边，保持边数/类型数。不存在候选对则N/A。该操作名是端点错配，不保证语义必错；只有独立提前标注的subset可称semantic-wrong。
**Irrelevant**：只在已有且任务明确定义不影响目标/关键资源/执行路径、并在原GK/GH消息图中与目标不连通的干扰组件内，枚举合法非重复非合同冗余supports及有效source effect锚点，均匀加1边。原prior已有8边、无独立无关性依据或无合法边均N/A；不删原边腾位、不新造节点。
先操作语义边再统一生成反向消息边；对应逆边不能残留。编辑不改真实facts/goal/skill/mask/reward。全局读出可能使无关组件有响应，所以Irrelevant不是强制Δ=0的单测。

### Agent Execution Template

当用户说“执行 Stage 4B”时，Agent严格执行以下步骤：

1. 读取冻结Full、原始cache及20个固定test case。
2. 对每case生成5条件，并验证只编辑语义软边。
3. 对每次编辑重建反向消息边、保存seed/hash/适用性。
4. 先判定Target-swap与Irrelevant是否有合法操作，无则N/A。
5. 在相同初始case执行20episode/condition/seed，不更新参数。
6. 在每个适用子集重算Original参考，避免分母错配。
7. 生成success、退化百分点、残差与适用性表；趋势已够清楚则停止。
8. 写robustness_report和stage_4b.json，不自动追加50或进入5A。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。


---

# Part VI — Minimal Generalization

## Stage 5A — Compositional Generalization

| 字段 | 执行定义 |
|---|---|
| Stage ID | 5A |
| Stage Name | Compositional Generalization |
| Purpose | 仅检验seen skills/predicates下的未见对象—目标或依赖组合，不叠加视觉/机器人/对象数shift。 |
| Stage Type | Generalization |
| 是否属于论文正式结果 | OPTIONAL |
| 前置条件 | 0A splits预先登记G1/G2；绑定证明确非纯ID重命名；默认2B T_C Full/B2三个seed checkpoint；新组合cache通过相同冻结生产协议；不可因测试成绩重定义组合。 |
| 输入 | 一个source task family T_C（实际绑定可替换）；Full/B2 seeds012；两个held-out compositions G1/G2及各前30cases；固定训练checkpoint、同相机/对象数量/低层/谓词/schema。 |
| Frozen Parameters | 全部训练参数、模型权重、技能/谓词词表、观测和控制器；零adaptation；只改变composition字段。 |
| Tunable Parameters | None；追加评价至50可新revision，不能fine-tune后仍称zero-shot。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| generalization | Full | T_C:G1 | 0,1,2 | 0 | Original | 30episode/seed；已见原子＋未见组合 |
| generalization | B2 | T_C:G1 | 0,1,2 | 0 | 空 | 30episode/seed；已见原子＋未见组合 |
| generalization | Full | T_C:G2 | 0,1,2 | 0 | Original | 30episode/seed；已见原子＋未见组合 |
| generalization | B2 | T_C:G2 | 0,1,2 | 0 | 空 | 30episode/seed；已见原子＋未见组合 |

### Training Budget

0训练transitions/updates；2methods×2compositions×3seeds×30=360evaluation episodes；ID比较复用同checkpoint2B日志。扩50=600，不默认增加新training job。

### Evaluation Protocol

deterministic，冻结VLM生产Original cache，Full/B2相同case；报告每composition成功与配对ID差异。使用了测试结果调配composition时改标exploratory-generalization，而非未触碰holdout。

### Required Logging

L_EVAL/L_DECISION＋seen/unseen原子表、canonical composition signature、source training split hash、checkpoint hash前后、ID/OOD case IDs、未支持资产原因。

### Output Directory

`experiments/part_5_generalization/stage_5a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_5a.json`。

### Required Files

- `generalization_manifest.json`
- `composition_audit.md`
- `generalization_results.csv`
- `generalization_report.md`
- `stage_5a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

fig_5a_composition_success.svg/png：x=ID/G1/G2，y=success，Full/B2 mean across seeds，无平滑。

### Required Tables

generalization_results.csv：source_task、composition、method、seen_skills/predicates、unseen_signature、n_seeds/episodes、success_mean/std、ID_matched、delta_pp；不可混外观shift。

### PASS / STOP Criteria

PASS：组合划分真实有效、冻结评价可追溯；结果一般/较差可放附录或缩claim。只有同构重命名、训练见过该组合、需要新skill的case不算本Stage完成，写BLOCKED/不支持并不造数。

### Failure Handling

先查split泄漏与纯重命名→技能/predicate覆盖→controller/camera一致→goal/ID绑定→cache生成→policy行为。不要为好结果同时改变视觉、对象数与任务难度。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，修绑定/评价bug或明确补50，不重训。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO。 |
| Next Stage | 6A；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 5A”时，Agent严格执行以下步骤：

1. 读取G1/G2及source training split，核对seen skills/predicates。
2. 检验composition signature不是仅改ID字符串或纯随机位置。
3. 冻结Full/B2 checkpoint，确认无新训练与finetune。
4. 按既定prompt/schema为新初始case生产/加载一次cache。
5. 对两个composition、两method、三seed各30episode评价。
6. 复用对应ID结果，计算每composition成功与退化。
7. 如组合不成立则报告不支持，不编造泛化能力。
8. 生成manifest/report和stage_5a.json并停止。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。


---

# Part VII — Private Paper Packaging

## Stage 6A — Raw Archive Review

| 字段 | 执行定义 |
|---|---|
| Stage ID | 6A |
| Stage Name | Raw Archive Review |
| Purpose | 建立所有真实运行、失败、模型和评价的可追溯总索引，不修改原数据。 |
| Stage Type | Packaging |
| 是否属于论文正式结果 | NO；为最终可追溯性必需 |
| 前置条件 | 至少一个已完成或失败但留有真实日志的Stage；raw/run目录可读；不要求所有可选Stage完成。 |
| 输入 | 所有part目录与raw_archive；所有attempt/seed/checkpoint/config/cache；stage状态；file checksums；无需API、模型执行或训练设备。 |
| Frozen Parameters | 原始日志和checkpoint只读；不把缺失日志补造成存在。 |
| Tunable Parameters | None；仅修索引工具。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| archive_scan | All actual methods | All attempted tasks | 读取所有实际seed，不抽新seed | 0 | All actual variants | 仅索引；包括失败与零成功run |

### Training Budget

0 transitions、0 updates、0 evaluation episodes。

### Evaluation Protocol

逐run验证必需文件/hash与源引用；区分TRAIN_COMPLETE、NORMAL_ZERO_SUCCESS、INFRA_FAILED、PARTIAL、MISSING_SOURCE；状态不是方法排名。

### Required Logging

扫描时间、文件总数/hash、run/attempt/seed/ckpt索引、缺失项、原始错误分类、来源目录；不输出敏感API key。

### Output Directory

`experiments/part_6_packaging/stage_6a/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_6a.json`。

### Required Files

- `raw_archive_index.csv`
- `checkpoint_index.csv`
- `evaluation_index.csv`
- `archive_integrity_report.md`
- `stage_6a_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

None；可选archive_coverage.svg/png仅计数量，不画伪性能。

### Required Tables

raw_archive_index.csv：run_id、stage、task/method/seed、attempt、code/config/cache hash、budget、status、success可用性、source path、failure_reason；一真实run一行。

### PASS / STOP Criteria

PASS：所有发现的运行均入索引；有少量缺失可PASS_WITH_NOTES并限制可展示集合。某主图源文件丢失则BLOCKED该图，不能靠回忆补数。

### Failure Handling

先路径与manifest→checksums→日志完整性→重复run ID→checkpoint源→密钥脱敏。只修改派生index，原raw保持不动。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES。 |
| Tuning Allowed | None。 |
| 是否阻塞论文写作 | NO。 |
| Next Stage | 6B；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 6A”时，Agent严格执行以下步骤：

1. 扫描全部实际运行目录，不只扫描获胜或成功run。
2. 读取manifest建立run/attempt/seed唯一索引。
3. 核对config、checkpoint、cache及episode日志hash。
4. 识别技术失败、正常零成功、partial和缺源，分别标注。
5. 建立checkpoint/evaluation/qualitative候选索引。
6. 只读验证主结果来源是否仍存在，不补造丢失记录。
7. 输出archive index与完整性报告。
8. 写stage_6a.json，停止，不自动筛选主文。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 6B — Private Paper View Selection

| 字段 | 执行定义 |
|---|---|
| Stage ID | 6B |
| Stage Name | Private Paper View Selection |
| Purpose | 从Raw Archive选出真实且最有解释价值的任务、seed、checkpoint、metric和qualitative case。 |
| Stage Type | Packaging |
| 是否属于论文正式结果 | YES：展示选择规范 |
| 前置条件 | 6A索引完成；有真实可用结果；已有2B时主表只选2B；仅有2A时允许私人探索Results view并标provisional，不冒称2B已经完成。 |
| 输入 | raw_archive_index、checkpoint/evaluation索引、task suitability、dev checkpoint证据、已查看结果范围、paper story draft；不重新执行任何模型。 |
| Frozen Parameters | 原始结果不变；每个selection必须有source；同一学习曲线只属于同一run，不能跨seed/attempt拼成best-of曲线。 |
| Tunable Parameters | 展示层task subset、3个seed选择、每method合理dev checkpoint、metric、MA3或EMA(.3)、axis/zoom、主文/附录分配；全部留记录。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| paper_view | Selected actual methods | 3主task默认；或标注的exploratory subset | 0,1,2 | 0 | 按实际评价条件 | 允许从0..4真实seed选3；不暗示未选择总体 |

### Training Budget

0训练、0新评价；只能选择/重算现有真实文件。

### Evaluation Protocol

选择后重算图表数值并与source逐项核对；checkpoint优先现有dev三窗口结果；换选时保存理由、查看过的数据范围。

### Required Logging

selection_id、view_revision、run_id/task/seed/checkpoint、metric、smoothing/window、subset、reason、source file/hash、selected_from、selection_data_used、n_attempted/n_selected。

### Output Directory

`experiments/part_6_packaging/stage_6b/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_6b.json`。

### Required Files

- `private_paper_selection_manifest.yaml`
- `selection_manifest.yaml`
- `selection_report.md`
- `qualitative_case_manifest.json`
- `stage_6b_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

preview_*仅用已选真实数据，沿用2C/3/4/5对应轴和smoothing；没有数据不生成结果图。

### Required Tables

selection_table.csv：figure/table/cell、run_id、task、seed、checkpoint、metric、source、reason、selection_scope；一项展示证据一行。

### PASS / STOP Criteria

PASS：全部拟展示数值/曲线/case有完整来源且标签准确；无需包含所有seed/失败run。找不到至少一个真实qualitative episode可PASS_WITH_NOTES并标缺口；不能用toy冒充实机case。

### Failure Handling

先查来源→checkpoint开发证据→seed正常与性能选择区别→subset范围→聚合→措辞。差异不清楚可以不写优势，不通过截轴和择seed暗示稳定总体增益。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，新selection revision；raw不变。 |
| Tuning Allowed | 仅展示参数。 |
| 是否阻塞论文写作 | NO。 |
| Next Stage | 6C；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### 最小selection语义
保留selection_mode=all_available / representative / selected_high_performing / illustrative_case；实际选择较好seed时不能写representative。允许每method不同seed集合，但必须标nonpaired_selected_seeds，不能据此画配对改善。默认尽量同集合；不要求把全体未选结果塞进主文。

### Agent Execution Template

当用户说“执行 Stage 6B”时，Agent严格执行以下步骤：

1. 读取6A完整索引和任务解释价值报告。
2. 确定主叙事只围绕已观察到的现象，不预定Full必须赢。
3. 选择任务及主文/附录范围，记录排除与纳入原因。
4. 默认选012三seed；补到5后选3必须登记全部候选和具体理由。
5. 使用各method合理dev checkpoint，记录不同训练step。
6. 选择success/AUC/稳定性等真实metric，冻结展示smoothing和坐标。
7. 选择至少一个可解释的真实episode，保留完整facts→actions→outcome链。
8. 生成private_paper_selection_manifest及同内容规范selection文件。
9. 更新stage_6b.json后停止，不重新训练或自动执行6C。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

## Stage 6C — Final Result Package

| 字段 | 执行定义 |
|---|---|
| Stage ID | 6C |
| Stage Name | Final Result Package |
| Purpose | 将已选择真实证据打包进私人论文，并逐条建立claim到figure/table/run的映射。 |
| Stage Type | Packaging |
| 是否属于论文正式结果 | YES |
| 前置条件 | 6B selection manifest完整；展示所需raw均存在；主结果/消融/机制缺项可显式缺省，不伪造补齐。 |
| 输入 | 6B选择、6A索引、相关Stage图表/数据、paper draft、当前手册与方法版本；无训练checkpoint调用、无API。 |
| Frozen Parameters | 真实证据、source hash、selection规则、方法定义；没有实验不能写成已验证。 |
| Tunable Parameters | 文案、图表顺序、主文/附录范围、claim强弱；任何数值选择变化回写selection revision。 |

### Run Matrix

| Run Group | Method | Task | Seed | Budget（skill transitions/new job或reference history） | Prior | Notes |
|---|---|---|---|---:|---|---|
| final_package | Selected actual evidence | Selected task/family views | 读取所有实际seed，不抽新seed | 0 | 实际条件 | 不新增任何训练或evaluation |

### Training Budget

0 transitions、0 updates、0 environment time；只复制/链接源文件并重生成派生展示。

### Evaluation Protocol

逐claim检查证据类型：方法代数性质/单测、Actor敏感性、成功/AUC增益、robustness、组合泛化分别需要对应证据；不把一种替代另一种。

### Required Logging

package版本、全部figure/table源、claim IDs、完成/未做/不支持项、每图n及selection mode、checksums、生成命令与软件环境。

### Output Directory

`experiments/part_6_packaging/stage_6c/`

训练或逐模型评价使用 `task_<task>/<method>/seed_<s>/attempt_<nn>/` 子目录；Stage总报告保存在本阶段根目录。源文件引用写完整相对路径和SHA-256。状态统一写 `experiments/stage_status/stage_6c.json`。

### Required Files

- `paper_results/figures/`
- `paper_results/tables/`
- `paper_results/qualitative/`
- `paper_results/selection_manifest.yaml`
- `paper_results/results_summary.md`
- `paper_results/result_claim_mapping.md`
- `paper_results/figure_sources.json`
- `paper_results/README.md`
- `paper_results/checksums.sha256`
- `stage_6c_summary.md`

每个实际训练run还必须拥有第4章的config/manifest/metrics/episode/decision/checkpoint完整集合；非训练Stage不制造空训练日志冒充运行。

### Required Plots

复制或重生成已定义的主学习曲线、family图、真实消融/机制/robustness/generalization图；未执行的章节不造空白胜出图。

### Required Tables

最终table_1、可用table_2、必要消融表；claim_mapping：claim_id、准确表述、scope、evidence_stage、figure/table、run_id/checkpoint、supported/limited/not_evaluated。

### PASS / STOP Criteria

PASS：最终package可追溯、无编造结果、claim与实际证据一致；没有必要把可选3C/4/5跑满。证据缺失则标LIMITED/NOT_EVALUATED，不能用过强措辞；源数据损坏为BLOCKED该claim。

### Failure Handling

先来源与数值→selection标签→claim范围→图表可读性→引用与hash。只改派生文案和图表，不能改raw。

| 执行权限 | 规定 |
|---|---|
| Rerun Allowed | YES，修改稿件或补入后续真实结果时升package版本。 |
| Tuning Allowed | 仅写作和展示。 |
| 是否阻塞论文写作 | NO；本Stage完成后可完成Abstract/Discussion/Conclusion。 |
| Next Stage | STOP：交付最终私人论文结果包；仅供建议，不自动启动 |

### Agent Deliverables

提交上述Required Files、指定图表、该Stage summary及实际状态JSON。summary固定包含：完成了什么、实际运行/复用/失败数量、真实结果与限制、相对冻结配置的变更、问题与下一阶段建议。没有运行的数据写未执行，不填写推测数值。

### Agent Execution Template

当用户说“执行 Stage 6C”时，Agent严格执行以下步骤：

1. 加载selection manifest，确认所有source仍存在且hash相符。
2. 复制/生成最终figures、tables与真实qualitative材料。
3. 每个figure/table标出任务、seed、episode、checkpoint与选择方式。
4. 建立claim→Stage→图表→run/checkpoint映射。
5. 将未做或不支持的结果写NOT_EVALUATED/LIMITED，收缩文案。
6. 生成results_summary、README和checksums。
7. 检查所有数字都能从raw重算，不依赖手工改CSV。
8. 交付结果包，写stage_6c.json并停止，不自动安排新实验。

本Stage完成或遇停止条件后更新状态并停止。只有用户明确授权整Part时，才可在该Part边界内继续，不能越过其他Part。

---

# 9. 阶段总表：执行顺序不是自动授权

| Part | Stage | Purpose | Task数 | Config数 | Seed数 | Training? | Paper Main? | Mandatory? |
|---|---|---|---:|---:|---:|---|---|---|
| I | 0A | 实际代码、环境、资产和manifest绑定 | D0＋2个主任务先绑定 | 基础设施1组 | 0 | NO | NO | 首次运行必需 |
| I | 0B | T01–T25不变量与数据链 | 25类测试fixture | 生产组件测试 | fixture 0 | NO | NO | 学习前必需 |
| I | 0C | VLM缓存生产链检查 | 24 task-scene | 1冻结快照 | 不适用 | NO | OPTIONAL setup | 初始缓存必需 |
| II | 1A | 最小学习smoke | 1：D0 | 2：B2/Full | 1：0 | YES | NO | 第一批必需 |
| II | 1B | 定向最小调参 | 1：D0 | 每次1个trial；总≤6新job | 1：0 | YES | NO | 仅1A未通过时 |
| III | 2A | 核心探索与task suitability | 默认2；可3 | 4：B0/B1/B2/Full | 3：0/1/2 | YES | 可写第一版Results；标探索 | 第一批必需 |
| III | 2B | 主展示运行 | 默认3；可3–5 | 同上4 | 每配置3；候选最多5 | YES | YES | 完整主展示时 |
| III | 2C | 主曲线和主表 | 使用2B已完成任务 | 4结果组 | 使用已选seed | NO | YES | 展示包必需 |
| IV | 3A | 双差分消融 | 默认1：T_C；可2 | Full/A_DD | 3 | YES：默认仅A_DD新训 | YES/探索版 | 第一批核心机制 |
| IV | 3B | Structural Q消融 | 默认1：T_A；可2 | Full/A_Q | 3 | YES：默认仅A_Q新训 | YES/OPTIONAL | 按Q证据缺口 |
| IV | 3C | bounded residual消融 | 默认1：T_C | Full/A_B | 3 | YES：默认仅A_B新训 | OPTIONAL | 优先级低于3A/3B |
| V | 4A | Actor固定输入/闭环依赖诊断 | 默认1：T_C | Full及5种干预 | 3个冻结模型 | NO | OPTIONAL | 按机制证据缺口 |
| V | 4B | prior测试鲁棒性 | 默认1；最多先2 | Full×5条件 | 3个冻结模型 | NO | OPTIONAL | 按鲁棒性claim缺口 |
| VI | 5A | 单一轴组合泛化 | 1源family＋2新组合 | Full/B2 | 3个冻结模型 | NO | OPTIONAL | 泛化claim需要时 |
| VII | 6A | Raw Archive清点与完整性 | 全部已有 | 全部已有 | 全部已有 | NO | NO | 最终打包必需 |
| VII | 6B | Private Paper View选择 | 已有task子集 | 已有方法 | 真实选定seed | NO | YES | 最终打包必需 |
| VII | 6C | 图表/claim/source完整结果包 | 已选结果 | 已选配置 | 已选seed | NO | YES | 最终打包必需 |

表中“必需”是相应执行/打包目标的前置，不是“必须完成才允许写Method”。同一Part内的Stage可有不同证据优先级；用户只说“执行Stage3A”不等于授权3B或3C。

# 10. 第一批、第二批与最终运行量

## 10.1 第一批：默认29个新增训练job

| 实验ID | Stage | 直接执行矩阵 | 新增训练job | 每job上限 | 何时停止 |
|---|---|---|---:|---:|---|
| 准备 | 0A | 定位实现/资产并绑定manifest | 0 | 0 | 核心绑定齐全PASS；缺项BLOCKED并停止 |
| E00 | 0B | CPU T01–T25；已绑定GPU补相应数值测试 | 0 | 0 | 必须不变量通过；发现bug停止修复，不跑性能 |
| E01 | 0C | D0/T_A/T_C各8个dev初始场景，共24 | 0 | 最多24初次＋24固定重试请求 | 缓存链正常且有少量合法非冗余关系，不扩大为benchmark |
| E02 | 1A | D0 × B2/Full × seed0 | 2 | 16,384 transitions | 最早各8 updates、出现真实成功和有效结构响应且无严重异常即停 |
| E03 | 2A | T_A/T_C × B0/B1/B2/Full × seeds0/1/2 | 24 | 65,536 transitions | 完成矩阵或合法资源cap，输出趋势与task报告；不追到Full第一 |
| E04 | 3A | T_C × A_DD × seeds0/1/2；复用2A的Full | 3 | 65,536 transitions | 匹配对照齐全即可，正/负/接近均记录，随后暂停扩实验 |
| **总计** |  | **2＋24＋3** | **29** | **最多1,802,240真实skill transitions** | **整理第一版Results** |

0C的24个task-scene不是24个RL运行，也不是全部后续cache。默认1B不进入此总数；若确需1B，最多6个新增单trial job，逐次满足停止条件即停止调参。2A扩成3个task、3A扩成2个task时，第一批为 `2+36+6=44` 个新增训练job。

默认First Batch中Full参考已有3条run；它们在消融表可出现，但不增加训练job数。若旧Full不满足复用条件，新增匹配Full必须单独登记并更新运行量，不能继续沿用29的名义预算。

## 10.2 Presentation Runs：单列，不冒充第一批复用

默认Stage2B为3 tasks ×4 methods ×3 seeds＝**36个新增job**，每个最多131,072 transitions，共4,718,592 transitions。

主任务扩到5个时为60个；默认不把2A的65,536-step探索run直接算作2B的独立新seed或完整131,072-step presentation run。0/1/2可重新运行，但attempt和purpose不同。经过explicit reuse/continue审查也可减少重训，但必须记录共享历史和真实新增训练量，不能将继承权重的运行当成独立重复。

## 10.3 第二批：默认6个新增训练job，其他主要是冻结评价

| 项目 | 默认新增job | 冻结评价规模 | 备注 |
|---|---:|---:|---|
| 3B：T_A A_Q ×3 seeds | 3 | 与2B相同dev和最终30 episode | 复用2B Full |
| 3C：T_C A_B ×3 seeds | 3 | Original先评；robustness只有单独授权时补 | 复用2B Full |
| 4A：T_C Full ×3模型 | 0 | ≤1,152同快照前向；≤360闭环episode | 64快照/模型×6条件；20episode/模型/条件 |
| 4B：T_C Full ×3模型×5条件 | 0 | ≤300 episode | 条件不适用则更少；扩50为≤750 |
| 5A：Full/B2 ×2新组合×3模型 | 0 | 360 episode | 每组合30；扩50为600 |
| 6A–6C | 0 | 不产生环境episode | 只读归档、选择与制图 |

3B增加第二个任务时第二批新增训练为9个。4A和4B相同模型、相同checkpoint、相同case、相同操作与口径的Original/No-prior/Target-swap评价可以按episode ID复用；**不可将复用数据算作新的独立样本**。上表给不复用的评价上界，非强制全跑。

## 10.4 总规模

默认完整私人论文路线（含3B和3C、但不含条件调参）为：

\[
N_{\mathrm{train}}=29+36+6=\boxed{71}.
\]

\[
N_{\mathrm{skill,max}}=1{,}802{,}240+4{,}718{,}592+786{,}432
=\boxed{7{,}307{,}264}.
\]

按任务数在允许区间扩展时为71–113个新增训练job；条件1B另计≤6，额外seed/不兼容Full重跑另计，不能把所有可选扩展默认叠加。作为对照，`7 configs×5 tasks×10 seeds=350` 个job；本计划不采用该笛卡尔积。

**最小可用版本不是71个全部跑完，而是先29个形成第一版Results。**实际早停、跳过低优先级模块、复用已匹配评价可进一步减少运行；这些都要通过manifest报告，而不是预先保证一定看到何种性能。

# 11. 立即执行顺序与Agent指挥范式

## 11.1 顺序

```text
Stage 0A：在实际项目绑定实现、环境和资产
  ↓ [用户继续授权，或明确执行Part I]
Stage 0B：生产组件不变量
  ↓
Stage 0C：24-scene缓存链
  ↓ [不自动跨到Part II]
Stage 1A：B2/Full最小学习
  ├─ PASS/PASS_WITH_NOTES：1B记skip；下一项为2A
  └─ NEEDS_RERUN：1B诊断/有限调参；无自动方法升级
Stage 2A：两主任务探索 → 立即整理第一版Results框架
Stage 3A：双差分核心证据 → 暂停扩实验，整理结果

随后按用户明确指令：
Stage 2B → Stage 2C（主展示）
Stage 3B / 3C / 4A / 4B / 5A（按论文证据缺口）
Stage 6A → 6B → 6C（归档与最终展示包）
```

“下一项为2A”表示推荐下一条指令，不代表Agent已经获得运行权限。Stage失败时也不能以“修复”为名自动启动其他Stage的完整矩阵；只做本Stage允许的排查、有限修复与重试。

## 11.2 指令与预期行为

| 用户指令 | Agent实际范围 | 必交付与停止点 |
|---|---|---|
| 执行Stage0B | 只运行T01–T25及需要的fixture检查 | stage_0b_summary、JUnit、逐项结果、stage_0b.json；不调用VLM/不训练 |
| 执行Stage1A | D0、B2/Full、seed0、smoke预算 | 真实日志、两run总结、stage_1a.json；不自动调参或做2A |
| 执行Stage2A | 默认T_A/T_C、四方法、012 | 探索曲线/表、task_suitability_report、配置建议、stage_2a.json；停止 |
| 根据Stage2A执行Stage2B | 读取真实2A报告，登记主task/config/seed选择，跑Presentation | 主结果原始记录与选择理由；不自动2C |
| 执行Part IV | 按3A→3B→3C执行本Part已授权配置，前置缺失就BLOCKED | 每Stage单独状态；Part IV报告；不自动做Part V |
| 执行Stage4B | 只用指定/按本计划选定的冻结checkpoint做测试扰动 | absolute success、退化、适用分母和错误日志；不训练 |
| 执行Stage6B | 只读Raw Archive并生成Private Paper View selection | private_paper_selection_manifest.yaml；不补跑差的数据 |
| 执行Part VII | 6A→6B→6C，归档、选择、重生成展示 | 可追溯paper_results；不暗中启动缺失实验 |

## 11.3 执行前的最小解析协议

Agent先从指令解析scope，再读取相应stage_status及Stage卡；解析真实输入路径与hash，检查该Stage必填字段。存在可通过已有文件解析的信息时先读，不重复问用户。核心资源确实没有绑定时，只输出一次集中缺项表、BLOCKED状态和本轮可完成的资料工作，不虚构环境或运行结果。

本包的 `tools/plan_stage.py --stage 2A` 只导出计划，**不是训练入口**。实际train/eval入口在 `manifests/entrypoints.yaml` 中定位；没有实现时由0A按冻结架构接入/实现，完成后才可启动学习。

# 12. Private Paper View：最后五个问题的执行答案

## A. 第一批到底跑什么？

采用第10.1节表：0A绑定、0B测试、0C24场景、1A两job、2A24job、3A三新增job并复用Full。默认仅29个新训练job；不自动加入A_Q、A_B、泛化或大规模鲁棒性。

## B. 每个第一批Stage出现什么就停止？

0A按真实readiness结束；0B按不变量而不是性能结束；0C只需证明生产链可用和存在少量合法非冗余关系。1A只需真实成功出现、数值正常、DK和合法DP/残差不永久失活；2A和3A按有限矩阵/预算结束，即使Full与B2接近、某seed零成功或A_DD无明显差异，也先保留并整理结果。

`PASS`表示完成本Stage目标，不代表证明方法优越。严重NaN、事实污染、错误mask、target泄漏等必须停止；学习差异小不是严重错误。没有成功信号时不能把“transition存在”写成已经学会。

## C. 哪些参数现在锁死，哪些允许调？

**锁死算法语义：**v2.1、两种relation与effect_fact_ref、纯DP、无static prior、Action–Proposition图、标准共享R-GCN、名义成功副本、真实终端奖励、执行动作Q目标、duration-aware PPO、80/20episode prior、不训练人工错边。

**允许有限开发调整：**lr、训练长度、entropy、B、lambdaQ；hidden dim只在必要时，rollout/minibatch/view分块只在资源适配时。层数/关系类型/权限不作为常规调参项。参数一旦进入单个消融比较，除被消融项外保持一致；其他方法可有明确记录的少量稳定性适配，但结果解释包含这项差异。

**Runtime MUST_BIND：**实际平台/代码/主机、对象资产、控制器、相机与感知、Verifier、时限/安全、时钟、APIregion/账户、split和所有源hash。MUST_BIND不是建议参数，也不是允许Agent自行猜测的实验事实。

## D. 结果一般时按什么顺序处理？

附件两处调参顺序不完全相同。本版明确采用附件§19与Stage1B顺序为统一执行顺序，§24中B/lambdaQ的提前项仅用于数值检查，不抢先启动参数搜索：

```text
实现 → candidates/mask → 独立reward → Skill Contract → Verifier
→ DK数值/传播 → DP数值/非零条件 → prior residual量级
→ lr → training length → entropy → B → lambdaQ
→ task难度/结构适配 → VLM非冗余关系质量
```

之后进行有限重跑与checkpoint/seed/task/metric选择。只调整当前有证据支持的首个问题；1B新job总上限6，不跑完整grid。若只是优势小或无优势，优先缩小claim、选择真正有结构意义的已有task、展示真实AUC或机制响应，不以“直到获胜”为无限停止准则。

## E. 怎样由Raw Archive组成Private Paper View？

先在6A索引所有真实run与失败原因，保留Raw不可变；6B建立结果级选择记录，再由6C生成派生图表。

| 选择对象 | 默认操作 | 必须记录 | 不能混淆 |
|---|---|---|---|
| Task | 从3–5候选类型中选有真实机制与清楚行为的任务 | 全部候选、所选task、结构理由、是否看过结果再选 | chosen structured subset不代表全部任务分布 |
| Seed | 3个正常/代表性/清楚的真实seed；必要时先补到5 | 候选seed、所选seed、排除原因、各方法是否配对 | 低success但有效的seed不是基础设施异常 |
| Checkpoint | 各方法自己的dev最近3点评分，允许不同step | checkpoint SHA、dev窗口、选择指标、预算 | 不把单次test尖峰当稳定最优 |
| Metric | Success为基础；按现象重点展示raw-AUC等 | metric定义、分母、聚合、selection_data_seen | 只成功样本completion-time不代表全体更快 |
| Curve | 默认MA3 trailing，raw保留；mean/SD或代表seed | sourceCSV、选seed、窗口、坐标范围、聚合和图注 | 平滑值不回写raw；seedSD不是置信区间 |
| Qualitative | 至少一个真实、最容易解释的episode | episode ID、checkpoint、frame/fact/action/result链、挑选原因 | 不把几个不同episode拼成同一次成功 |

允许为了私人呈现选择更好的真实结果，但选后均值必须称为**所选run/子集的描述性结果**，不包装成未经选择的总体估计。涉及测试后选择的结果，写 `selection_data_seen: test` 和 `inference_scope: selected_exploratory_view`。不强制所有失败都进入主文，但raw index与selection manifest保留它们的位置及排除理由。

# 13. Paper Writing同步计划

| 实验阶段 | 同步写作产物 | 此时不能预填 |
|---|---|---|
| 0A–1B | Introduction、Related Work工作稿、Problem、Method | 方法已部署、成功率优势、鲁棒保证 |
| 2A | Experimental Setup、Task、Baselines、Metrics及第一版探索Results | 未运行task、未运行seed或显著性 |
| 2B–2C | 主表、主曲线、主结果描述 | “所有任务都优于”或不存在的平均增益 |
| 3A–3C | 有结果的消融和机制解释 | 未做的消融贡献；共享梯度本身不证明行为有益 |
| 4A–5A | 鲁棒性/泛化对应小节；未做部分可省略 | 任意prior错误安全、全领域泛化 |
| 6A–6C | Abstract结果句、Discussion、Conclusion和claim映射 | 比raw证据范围更强的结论 |

论文可围绕最清楚的真实现象调整：final success、learning AUC、结构复杂任务的局部收益、或错prior条件下退化。若某模块没有观测到增益，也可写成边界结果；不因此自动回到Method重新设计。

本包附 `paper_draft/sections/method_working_draft.md`，仅为冻结方法的中文起步稿，不含实验数值；它不代表完整论文已写完。Related Work中的独立学术定位仍需实际写作时核对原始文献。

# 14. 本轮实际完成状态与启动边界

本轮生成手册、17张Stage卡、配置/manifest模板、方法变体配置、默认计划矩阵、状态schema、选择模板及计划/预检工具。对当前聊天沙箱执行**资料hash与软件metadata预检**；实际记录见 `part_0_validation/stage_0a/`。

这个沙箱不等于用户训练host。生产模型/collector/executor、任务资产、相机、控制器、标定、安全、真实cache和checkpoint尚未通过本包绑定核验，因此Stage0A为 `BLOCKED`，并标 `scope: local_inventory_only`。其他Stage保持 `NOT_STARTED`。没有执行VLM请求、科研训练、机器人动作，也没有将文档校验冒充T01–T25生产单元测试。

完整0A下一次在真实项目上先读取已有资源；若仓库已实现某组件，应映射复用，而不是因为本包没有该代码就重新实现。只有真实证据齐全才写PASS。缺项不会阻止继续写Problem与Method。

# 附录A. Experiment Manifest v0

下面是本轮提出的完整默认配置，不是已执行日志，也不是已解析的目标host环境锁。`MUST_BIND`必须在相应Stage启动前由真实资源解析。每次真实run另写resolved config，不能直接将本模板改名为运行证据。

```yaml
manifest_version: experiment_manifest_v0
plan_version: EEP-v1.0
created_date: '2026-09-20'
state: PROPOSED_NOT_EXECUTED
method:
  name: CP-DISR
  internal_name: M1
  version: v2.1
  specification_ref: sources/CP_DISR_M1_Research_Specification_v2.1.md
  core_frozen: true
  no_static_prior: true
  nominal_success_only: true
  nominal_depth: 1
  prereq_research_experiments: 0
rl:
  framework: project_local_pytorch_recurrent_smdp_ppo
  cleanrl_role: reading_reference_only
  trainer_implemented_in_this_kit: false
vlm:
  provider: alibaba_model_studio
  model_snapshot: qwen3.8-max-0902
  full_snapshot_alias: qwen3.8-max-2026-09-02
  region_recommendation: Singapore
  region: MUST_BIND
  base_http_api_url: MUST_BIND
  api_account_authorized: MUST_BIND
  sdk: dashscope
  sdk_version: 1.27.6
  relation_schema: m1_soft_relations_v2
  schema_ref: sources/v2.1_interfaces/relation_schema.json
  prompt_ref: sources/v2.1_interfaces/system_prompt.txt
  prompt_version: cp_disr_relations_v2.1_p1
  fewshot_count: 3
  fewshot_manifest: MUST_BIND
  max_relations: 8
  temperature: 0
  max_output_tokens: 2048
  thinking: false
  response_format: json_object
  max_retries: 1
  retry_only:
  - transport_error
  - syntax_error
  - truncated_output
  semantic_retry: false
  allow_web_search: false
  tools_enabled: false
  cache_mode: offline
  cache_required: true
  cache_key_hash: sha256
  key_includes_split: true
  seed_supported_by_api: MUST_VERIFY_OR_UNSUPPORTED
prior_training:
  original: 0.8
  absent: 0.2
  sampled_at: episode_start
  fixed_within_episode: true
  resample_during_ppo: false
  expose_sampling_mode_to_network: false
  shared_across_prior_consumers: true
  rng_namespace: prior_dropout_train
  edge_corruption_enabled: false
graph:
  representation: grounded_action_proposition_slg
  node_types:
  - ACTION
  - PROPOSITION
  contract_relations:
  - PRE_POS
  - PRE_NEG
  - ADD
  - DEL
  soft_relations:
  - SOFT_SUPPORTS
  - SOFT_RELEVANT_TO_GOAL
  reverse_message_edges: true
  encoder: RGCNConv
  hidden_dim: 128
  layers: 4
  num_bases: 4
  aggregation: mean
  root_weight: true
  activation: ReLU
  normalization: per_node_LayerNorm
  dropout: 0
  difference_dtype: float32
  goal_alignment: fixed_proposition_indices
  empty_prior_encoding_reuse: true
actor:
  prior_bound_B: 0.5
  zero_anchor: true
  prior_input: DP_only
  enhanced_context_in_full: false
  final_prior_bias: false
  candidate_readout: single_head_scaled_dot_product_attention
  mlp_hidden_dims:
  - 128
  - 128
structural_q:
  lambda_q: 0.1
  target: stop_gradient(real_r + Gamma * (1-terminated) * old_V_next)
  executed_action_only: true
  loss: Huber
  huber_delta: 1.0
  label_unexecuted_candidates: false
ppo:
  optimizer: Adam
  lr: 0.0003
  adam_eps: 1.0e-08
  adam_betas:
  - 0.9
  - 0.999
  weight_decay: 0.0
  clip: 0.2
  gae_lambda_per_skill: 0.95
  value_coef: 0.5
  entropy_coef: 0.01
  grad_clip: 0.5
  epochs: 4
  rollout_skill_transitions: 1024
  recurrent_sequence_length: 16
  recurrent_hidden_dim: 128
  minibatch_valid_transitions: 64
  num_envs_recommendation_sim: 4
  learning_rate_schedule: constant
  value_loss: Huber
  value_clipping: false
  advantage_normalization: valid_minibatch_actor_only
  actor_episode_discount_weight: true
  surrogate_reduction: sum(w * surrogate)/sum(w)
  critic_and_entropy_reduction: valid_transition_mean
  gamma_reference_seconds: 1.0
  gamma_reference_value: 0.99
  graph_recompute_during_ppo: true
  gru_prefix_recompute_with_current_parameters: true
  amp: false
  torch_compile: false
  reward_shaping: false
  reward_normalization: false
  graph_pretraining: false
  q_pretraining: false
  bc_warm_start: false
seeds:
  smoke:
  - 0
  exploration:
  - 0
  - 1
  - 2
  presentation_default:
  - 0
  - 1
  - 2
  presentation_default_count: 3
  extension_order:
  - 3
  - 4
  maximum_initial_extension: 5
  derive_algorithm: first4bytes_big_endian_SHA256(canonical_json(namespace,task,case,seed))
  eval_and_corruption_omit_method_and_train_seed: true
tasks:
  development: D0
  exploration_default:
  - T_A
  - T_C
  presentation_default:
  - T_A
  - T_B
  - T_C
  optional_candidates:
  - T_D
  - T_E
  dd_default: T_C
  q_default: T_A
  bound_default: T_C
  diagnostics_default: T_C
  generalization_source_default: T_C
  train_case_pool: 64
  dev_case_pool: 50
  test_id_case_pool: 50
  generalization_compositions:
  - G1
  - G2
  generalization_cases_per_composition: 50
budget:
  primary_execution_axis: real_skill_transitions
  physical_budget_equality_claimed: false
  smoke: 16384
  tuning_trial: 16384
  tuning_length_trial: 32768
  exploration: 65536
  presentation: 131072
  dd_first_batch: 65536
  q_and_bound: 131072
  tuning_max_new_jobs: 6
  exploration_optional_diagnostic_jobs: 4
  physical_cap_formula: 2 * N_cap * bound_task_reference_skill_seconds
  planning_only_reference_skill_seconds: 2.0
  real_reference_skill_seconds: MUST_BIND_PER_TASK
  partial_rollout: valid_samples_only_single_partial_update
  default_first_batch_new_training_jobs: 29
  default_full_route_new_training_jobs: 71
evaluation:
  policy_mode: deterministic_masked_argmax
  tie_break: canonical_candidate_id
  default_prior: original
  no_train_dropout_in_evaluation: true
  smoke_episodes: 10
  exploration_episodes: 20
  training_presentation_dev_episodes: 20
  presentation_test_episodes: 30
  robustness_initial_episodes: 20
  mechanism_closed_loop_episodes: 20
  snapshot_max_per_training_seed: 64
  snapshot_max_per_episode: 4
  generalization_episodes: 30
  optional_stable_episodes: 50
  append_only_episode_extension: true
  checkpoint_interval_updates:
    smoke: 4
    exploration: 8
    presentation: 16
    dd: 8
    q: 16
    bound: 16
  checkpoint_selection: maximize trailing_3_dev_success_mean; tie smaller_sd; tie
    earlier_step
  success_threshold_for_efficiency: 0.7
  threshold_consecutive_points: 2
robustness:
  conditions:
  - original
  - absent
  - edge_deletion
  - type_preserving_target_swap
  - irrelevant_edge_addition
  deletion_p: 0.5
  target_swap_edges: 1
  irrelevant_add_edges: 1
  max_relations: 8
  edit_semantic_forward_then_rebuild_reverse: true
  fixed_within_episode: true
  no_eligible_edit: NOT_APPLICABLE
  semantic_wrong_requires_independent_annotation: true
  irrelevant_requires_task_and_graph_evidence: true
presentation:
  raw_archive_immutable: true
  selection_allowed: true
  manifest_required: true
  default_seed_aggregation: mean
  error_bar: selected_seed_sample_standard_deviation_not_CI
  default_smoothing: trailing_MA3
  smoothing_for_display_only: true
  alternate_smoothing: EMA_alpha_0.3
  raw_AUC: true
  success_axis:
  - 0
  - 1
  no_cross_run_curve_splicing: true
  explicitly_label_selected_seed_view: true
software:
  profile_status: PROPOSED_NOT_VALIDATED
  python: 3.11.13
  torch: 2.7.1
  cuda_wheel: cu126
  torchvision_optional: 0.22.1
  torch_geometric: 2.6.1
  numpy: 1.26.4
  pandas: 2.2.3
  matplotlib: 3.9.4
  jsonschema: 4.23.0
  PyYAML: 6.0.2
  tensorboard: 2.19.0
  pytest: 8.3.5
  gymnasium: 1.1.1
  uv: 0.8.22
  dashscope: 1.27.6
  lockfile: MUST_GENERATE_ON_TARGET_HOST
runtime:
  experiment_root: MUST_BIND
  repository_path: MUST_BIND
  repository_url: MUST_BIND
  git_hash: MUST_BIND
  simulator_or_robot: MUST_BIND
  environment_version: MUST_BIND
  task_assets: MUST_BIND
  controller_manifest: MUST_BIND
  camera: MUST_BIND
  calibration_manifest: MUST_BIND
  perception_checkpoint: MUST_BIND
  verifier_thresholds: MUST_BIND
  skill_timeouts: MUST_BIND
  task_deadlines: MUST_BIND
  actual_interaction_time_unit: MUST_BIND
  reference_skill_seconds_by_task: MUST_BIND
  task_evaluator_version: MUST_BIND
  hardware: MUST_BIND
  gpu_driver: MUST_BIND
  os: MUST_BIND
  safety_authorization: MUST_BIND
  entrypoints: MUST_BIND
  task_splits: MUST_BIND
agent_scope:
  single_stage_auto_advance: false
  explicit_part_allows_internal_stages: true
  stop_on:
  - BLOCKED
  - NEEDS_RERUN
  skip_1b_when_smoke_passed: true
  skip_status: PASS_WITH_NOTES
  skipped_boolean: true
```

# 附录B. 状态、选择与claim文件

状态文件模板及机器schema分别在`stage_status/`和`schemas/`。未执行写NOT_STARTED；受阻写BLOCKED；不设含糊的FAIL枚举；需重跑用NEEDS_RERUN。跳过1B用PASS_WITH_NOTES与skipped=true，不能声称运行通过。

选择模板（以下全为待填项，不是实验记录）：

```yaml
schema_version: private_paper_selection_v1
status: TEMPLATE_NO_RESULTS
view_revision: 0
selection_mode: MUST_SELECT_ALL_AVAILABLE_OR_REPRESENTATIVE_OR_SELECTED_HIGH_PERFORMING
selected_from_run_ids: []
items: []
item_template:
  artifact_id: MUST_BIND
  run_id: MUST_BIND
  task: MUST_BIND
  training_seed: MUST_BIND
  checkpoint_ref: MUST_BIND
  checkpoint_sha256: MUST_BIND
  metric: MUST_BIND
  smoothing:
    method: trailing_MA
    window: 3
    display_only: true
  subset: MUST_BIND
  selection_reason: MUST_BIND
  selection_data_used: MUST_BIND
  source_file: MUST_BIND
  source_sha256: MUST_BIND
  n_attempted: MUST_BIND
  n_selected: MUST_BIND
```

每个claim必须有`claim_id / wording / evidence_stage / run_ids / checkpoint_hashes / figure_table / scope / caveat / status`。无证据标NOT_EVALUATED，不生成带数字的默认结论。完整映射由6C填充。

# 附录C. 原文件与外部核对依据

本手册S1为实验执行要求，S2为方法语义，S3为接口；三者均保留原件。新增预算、软件profile、任务ID、阈值和筛选规则属于本轮执行设计，不冒称来源中已经做过的实验。

**[S1] CP_DISR_v2.1_Final_Experimental_Agent_Prompt.md**。SHA-256：`e74eb0dc879e538c23ef3436ca8243fc3bd00a2d96dc50e913c99972d25c01eb`；本包路径：`sources/CP_DISR_v2.1_Final_Experimental_Agent_Prompt.md`。

**[S2] CP_DISR_M1_Research_Specification_v2.1.md**。SHA-256：`c552592d9d201e54332344379ceb130fd740dcd1b43134f94f18ad222399fdae`；本包路径：`sources/CP_DISR_M1_Research_Specification_v2.1.md`。

**[S3] CP_DISR_M1_Implementation_Interfaces_v2.1.zip**。SHA-256：`d46dcbe3032da4b987001f0ee2394d1a1eb5ecbc11b4b70a4071c62596dd783f`；本包路径：`sources/CP_DISR_M1_Implementation_Interfaces_v2.1.zip`。

外部核对日期：2026-09-20。以下一手文档仅支持软件/API接口事实，不支持CP-DISR性能。版本存在不等于组合已完成目标host兼容测试；实际lock由0A生成。

**[W1] Python 3.11.13 release**。Version existence; not a latest-version claim.

```text
https://www.python.org/downloads/release/python-31113/
```

**[W2] PyTorch previous versions**。2.7.1/torchvision0.22.1/cu126 wheel pairing.

```text
https://pytorch.org/get-started/previous-versions/
```

**[W3] PyG 2.6.1 installation and RGCNConv**。Standard R-GCN interface; package integration not tested here.

```text
https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.conv.RGCNConv.html
```

**[W4] CleanRL PPO**。Single-file PPO reference; LSTM variant is not our GRU implementation.

```text
https://docs.cleanrl.dev/rl-algorithms/ppo/
```

**[W5] SB3-contrib MaskablePPO**。Recurrent policies not supported in documented MaskablePPO; consult separate Recurrent PPO.

```text
https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
```

**[W6] Gymnasium Handling Time Limits**。Termination/truncation bootstrap distinction.

```text
https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
```

**[W7] Alibaba qwen3.8-max Model Info**。Snapshot0902, image input, structured output public support; account not verified.

```text
https://www.alibabacloud.com/help/en/model-studio/qwen3-8-max
```

**[W8] uv 0.8.22 package release**。Pinned resolver candidate.

```text
https://pypi.org/project/uv/0.8.22/
```

**[W9] DashScope release history**。Release history lists1.27.6 on2026-09-17; no SDK/API call executed.

```text
https://pypi.org/project/dashscope/1.25.2/
```

**[W10] PyG2.6.1 minimal installation**。Optional extension libraries and supported Python range.

```text
https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html
```


# 冻结执行摘要

**Method维持v2.1。先完成Part I真实绑定与接口验收；第一批默认29个新增训练job后暂停扩实验。每次只执行明确授权的Stage/Part，所有展示来自可追溯真实记录；论文写作与实验并行。**
