# GREBRAC-L16-TEACHER：确定性教师版 GFP

授权：2026-10-15 16:37 UTC，用户在决策卡"要不要测'确定性教师版 GFP'，看流是不是必需的？"上选择"测教师版"。卡上写明的范围："task1+task4 各 4 种子共 8 条新作业，约 1 小时，复用 ReBRAC 作对照，判据先冻结再启动。"
前置结果：`/mnt/project-files/gfp_simplify_review/GREBRAC_L15_BCREF_RESULT_20261015.md`（L1.5 判为"独立单峰参考不够"，第 4、6 节）。
远端根目录：`/home/xushijie3/work/grebrac_l16_teacher_20261015`（主机 gpu03，账号 xushijie3）。

## 要回答的问题

L1 和 L1.5 都把 guidance 加在 actor 自己的 BC 项上，与 GFP 的结构不同。这一轮在 ReBRAC 底座上照搬 GFP 的结构，只把流（VaBC）换成一个确定性网络：

- g 加在**教师**的 BC 上，参考值沿用 GFP 官方配置的 min 规则；
- actor 不再直接 BC 到数据，而是以固定强度贴近教师的动作（对应 GFP 的蒸馏）。

如果 task1 的增益回来了，说明在 ReBRAC 底座上 GFP 的增益不需要流。如果仍然没有增益，说明增益依赖流（多峰教师）或 GFP 其余的 FQL 式结构。

## 代码与环境（冻结）

- **源码：** GFP 官方 `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`，依次打 p1（L1）、p2（L1.5）、p3（本轮）。
  - `code/grebrac_p3.patch`：相对官方原版的完整差异，sha256 `26814964…aab7`。
  - `code/grebrac_p3_incremental.patch`：相对 p2 的增量（去掉了文件头时间戳），sha256 `adbc33c9…`。远端 `receipts/` 中是同一文件。
  - 打完后的文件：`agents/rebrac.py` sha256 `1d64de0414836cd3…`，`config/agent/rebrac.yaml` sha256 `3bcb080700c6ffda…`，远端已核对。
- **p3 新增 `guidance_ref=teacher`：**
  - **教师 T：** 与 actor 同结构（4×512，tanh 输出），取 `.mode()` 作确定性动作。损失为 mean_batch(g · ‖T(s) − a‖²)，用原始 g，不做 batch 归一化（`guidance_normalize=false`），与 GFP 的 VaBC 损失 `mean(guidance * flow_error)` 相同（`agents/gfp.py:183`）。
  - **g：** sigmoid((λ/η)(Q̄(s,a) − min(Q̄(s, π(s)), Q̄(s, T(s)))))。这是 GFP 的 softmax 形式加 `guidance_agg=min`（`agents/gfp.py:32-54, 97-118`；`hyperparams/gfp_params.csv` 的 humanoidmaze_medium_navigate 行）。Q̄ 为 target critic，按 q_agg=mean 聚合；λ = 1/mean|Q(s,π(s))|；η = 1e-3（GFP 同环境行）；π(s) 与 T(s) 都 stop-grad。
  - **actor：** −λQ(s,π(s)) + α_actor·‖π(s) − sg(T(s))‖²，α_actor = 0.01（ReBRAC 官方值），只把 BC 目标从数据动作换成教师动作。GFP 的 α = 0.3 作用在对 21 维取均值的 MSE 上，折合按维求和约 0.014，与 0.01 同一量级。
  - T 与 actor 在同一次更新里训练（actor_freq=2），同一个 Adam 优化器、同一学习率。critic、目标网络、采样都是 ReBRAC 原样。
- **本地检查（CPU，`tests/test_teacher.py`，输出见 `receipts/local_tests.txt`）：** 全部通过。
  - 旧模式不变：`guidance=False` 与官方原版逐位相同；`guidance_ref=actor` 与 L1 逐位相同；`guidance_ref=bc` 与 L1.5 逐位相同。
  - 新增 `teacher` 模块后，其余模块的初始化不变。
  - 梯度隔离：教师的梯度与 α_actor 无关（actor 的 BC 目标对教师 stop-grad）；actor 和 critic 的梯度与 η 无关（g 只进教师损失）。
  - 40 次更新内参数有限，教师 mse 6.80 → 5.70；fused 更新与验证集日志路径正常，日志里有 `teacher_mse`、`mse_teacher`、`teacher_loss`、`guidance`、`mse`。
- **环境、数据、入口：** 与 L1、L1.5 相同。
- **脚本：** `code/launch_formal.sh`（sha256 `f730cadc…2719`）、`code/summarize.py`（`15150ef8…2c2d`）、`code/extract_curves.py`（`02d2aa9b…6d05`），远端副本已核对。

## 与 GFP 仍然不同的地方（本轮不改）

- 教师是单峰确定性网络，GFP 是 10 步积分的流。
- actor 是确定性网络，GFP 的 actor 以噪声为输入（FQL 结构）。
- critic 是 ReBRAC 的（目标动作加噪声、对数据下一动作的 BC 惩罚），GFP 在本环境用混合 TD 目标。
- α 用 ReBRAC 的 0.01，不用 GFP 的 0.3（见上，量级相同）。

## 实验矩阵（冻结）

| 臂 | agent 参数 | 来源 |
|---|---|---|
| T：确定性教师版 | `agent=rebrac agent.guidance=true agent.eta_temperature=0.001 agent.guidance_ref=teacher agent.guidance_normalize=false` | 本轮新跑 |
| R：ReBRAC | `agent=rebrac` | 复用 L1 的 `runs/formal/R_task{1,4}_s{0..3}`，不重跑 |

- **任务：** `humanoidmaze-medium-navigate-singletask-task1-v0`、`…-task4-v0`。
- **种子：** 0、1、2、3。新作业共 2 任务 × 4 种子 = **8 条**，每条 1,000,000 次离线更新，online 0。
- **超参数：** 官方 `rebrac_params.csv` 的 humanoidmaze_medium_navigate 行，与 L1、L1.5 相同。
- **公共参数：** 与 L1 相同：`is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 log_interval=5000 save_interval=100000 save_last_checkpoint=true log_on_wandb=false`，`n_steps_fused=4`、`dataset_on_gpu=true` 为默认。
- **配对：** 同种子。T 臂的 actor 目标与 ReBRAC 不同，所以不存在逐位相同的退化情形；配对的含义是同种子的初始化与批次序列。
- **不做：** 其他 η、归一化 g 的变体、改 α、混合 TD 目标、cube-double、加种子、调参。这些都需要另行授权。

## 主指标与判据（冻结，先于任何 T 臂结果）

- **主指标：** 官方 `final_Actor`（800k、900k、1M 三次评价的均值）。每个种子算 d_s = T_s − R_s，报告 4 个种子的均值和配对 t 95% 区间，并列出每个种子、每个检查点。
- **判定**（单位：百分点，阈值与 L1、L1.5 相同）：

| 结果 | 判定 |
|---|---|
| task1 均值 d ≥ +25，且 task4 均值 d ≥ −10 | **确定性教师即可**：在 ReBRAC 底座上，GFP 的增益不需要流 |
| task1 均值 d < +10 | **确定性教师不够**：增益依赖流（多峰教师）或 GFP 其余的 FQL 式结构（带噪声的一步 actor、混合 TD 目标） |
| task1 均值 d ≥ +25，但 task4 均值 d < −10 | **伤绕行**：带 guidance 的教师结构压制绕行行为 |
| task1 均值 d 在 +10 与 +25 之间 | **灰区**：不下结论，报告后由用户决定 |
| task1 达标，task4 落在 −10 附近 | 按上表照判，同时报告区间 |

- **R 臂基准（已知，来自 L1）：** task1 均值 25.1，task4 均值 22.8。作者 CSV 8 种子参照（不参与判定）：GFP task1 83.5 / task4 3.0；ReBRAC 34.1 / 19.5。L1.5 的 B 臂：task1 33.7 / task4 4.1。
- **次要诊断（不参与判定）：**
  - 教师与数据的 mse（`teacher_mse`）、actor 与教师的 mse（`mse_teacher`）、actor 与数据的 mse（`mse`）。
  - g 均值、g > 0.1、g > 0.75 的比例，与你自己的 GFP task1 日志（800k–1M：g 均值约 0.48，g > 0.75 约 0.17）比较。
  - q_mean，与 R 臂比较。

## 工程停止条件（冻结，与 L1、L1.5 相同）

- 任一指标、参数或损失出现 NaN/Inf：停止该条作业，保留状态。
- 非算法原因导致的启动失败（环境、EGL、显存不足）：允许修正后对该条重启一次，并记录在 receipts。
- 低成功率本身**不是**停止或调参理由。
- Q 值发散（|q_mean| > 1000）只记录、不停止，在报告中标注。

## 资源

- 服务器多人共享，使用时不查看、不打断他人进程。
- 正式启动前，在一张卡上跑 T 臂 task1 seed 0 的 2k 步（评价关闭），确认单步耗时、g 没有塌、教师 mse 在下降，结果写入 `receipts/timing_2k.md`。这一步只检查工程是否正常，不改设计。
- 启动前查看各卡空闲显存，每卡至少留 10 GB 余量，每卡不超过 3 条，间隔约 15 秒错开启动，每条作业单独绑 6 核。放置记录写入 `receipts/launch_formal.md`。
