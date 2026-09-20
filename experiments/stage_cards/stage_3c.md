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
# 本卡的共同执行规则（随卡附带）

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
