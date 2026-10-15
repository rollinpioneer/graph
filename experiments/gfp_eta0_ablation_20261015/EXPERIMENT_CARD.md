# GFP-ETA0-ABL：在 GFP 内部关掉 guidance

授权：2026-10-15 17:49 UTC，用户在决策卡"要不要在 GFP 内部关掉 guidance 做对照，看它到底有没有用？"上选择"两个任务都测"。卡上写明的范围："GFP 与关掉 guidance 的 GFP，task1+task4 各 4 种子共 16 条，约 2 小时，判据先冻结再启动。"
前置结果：`/mnt/project-files/gfp_simplify_review/GREBRAC_L16_TEACHER_RESULT_20261015.md`。L1、L1.5、L1.6 都说明，在 ReBRAC 单 actor 底座上，任何形式的 guidance 都拿不到 GFP 在 task1 上的增益，而且 guidance 伤 task4。
远端根目录：`/home/xushijie3/work/gfp_eta0_ablation_20261015`（主机 gpu03，账号 xushijie3）。

## 要回答的问题

guidance 在 GFP 自己的结构里到底有没有用：

- 如果关掉 guidance 后 task1 仍有约 80%，GFP 的增益来自 FQL 式结构（流 + 带噪声的一步 actor + 混合 TD 目标），可以直接删掉 guidance，把 GFP 简化为这个结构。
- 如果关掉后 task1 明显下降，guidance 有用，但只有和流配合才有用（L1–L1.6 已经说明它搬不到单 actor 上）。
- 同时看 task4：GFP 在 task4 上弱（作者 CSV 3.0），这里检查弱点是否来自 guidance。

## 代码与环境（冻结）

- **源码：** GFP 官方 `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`，**不打任何补丁**。远端 `source/gfp` 由官方 tarball 解压，`receipts/source_manifest.txt` 记录了 50 个文件的 sha256，与本地官方副本逐一相同（只有排序不同）。
- **关掉 guidance 的方式：** CLI 参数 `agent.eta_temperature=0`。
  - `agents/gfp.py:176-178`：`eta_temperature == 0` 时 `VaBC_loss = mean(flow_error)`，不调用 `compute_guidance`。其余部分（critic、混合 TD 目标、蒸馏、Q 损失、α = 0.3）完全相同。
  - eta 只在这里和在线阶段（`eta_temperature_online`，本实验 online_steps=0）用到。
  - CLI 参数优先于 CSV：`hyperparams/hyperparams.py` 只覆盖 CLI 没有给出的 agent 键（`key not in agent_overrides`）。
- **g 的梯度路径（核对过代码）：** g 用 target critic 计算（`critic_eval`，`agents/gfp.py:27-30`），actor 动作 stop-grad，流动作不带 `grad_params`，所以 g 只按样本重新加权流的 BC 误差，只影响 vabc 网络的参数。优化器是单个 Adam（`agents/gfp.py:378`），按参数归一化，所以 g 的整体平均水平（约 0.48）基本被抵消，F 臂与 G 臂的差别只是**样本之间的相对权重**。
- **RNG：** guidance 不消耗随机数（`actor_loss` 无论 eta 取何值都切 4 份 rng），所以同种子下 G 与 F 的初始化和批次序列相同，配对有意义。
- **超参数：** 官方 `gfp_params.csv` 的 humanoidmaze_medium_navigate 行：discount 0.995、flow_steps 10、q_agg mean、target_agg mean、guidance_agg min、α 0.3、η 0.001（F 臂由 CLI 改为 0）。
- **评价：** `eval_flow_policy=false`，只评价 actor（与作者报告的 `final_Actor` 相同口径，也与 L1–L1.6 相同）。
- **环境、数据、入口：** 与 L1–L1.6 相同（Python 环境 `flow_baselines_preflight_20261013/envs/gfp312`，数据通过 `home/.ogbench/data` 软链接到 `flow_critic_dual_host_prep_v1_20261014/data`，入口 `scripts/entry_spawn.py`）。
- **脚本：** `code/launch_formal.sh`、`code/summarize.py`、`code/extract_curves.py`、`code/entry_spawn.py`，远端副本在启动前核对 sha256。

## F 臂不是纯 FQL

F = GFP 去掉 guidance。它与 FQL 的差别：本环境的 GFP 行用 `target_agg=mean`（TD 目标取 actor 动作与流动作 Q 值的均值），FQL 只用 actor 动作。所以本实验回答的是"guidance 在 GFP 里是否必要"，不回答"GFP 是否等于 FQL"。

## 实验矩阵（冻结）

| 臂 | agent 参数 |
|---|---|
| G：官方 GFP | `agent=gfp eval_flow_policy=false` |
| F：关掉 guidance 的 GFP | `agent=gfp agent.eta_temperature=0 eval_flow_policy=false` |

- **任务：** `humanoidmaze-medium-navigate-singletask-task1-v0`、`…-task4-v0`。
- **种子：** 0、1、2、3。共 2 臂 × 2 任务 × 4 种子 = **16 条**，每条 1,000,000 次离线更新，online 0。
- **公共参数：** `is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 log_interval=5000 save_interval=100000 save_last_checkpoint=true log_on_wandb=false`，其余为官方默认（`log_locally=true`）。
- **配对：** 同任务同种子的 G 与 F。
- **不做：** 其他 η、调参、cube-double、加种子、评价流策略、纯 FQL 对照。这些都需要另行授权。

## 主指标与判据（冻结，先于任何结果）

- **主指标：** 官方 `final_Actor`（800k、900k、1M 三次评价的均值）。每个种子算 d_s = G_s − F_s，报告 4 个种子的均值和配对 t 95% 区间，并列出每个种子、每个检查点。
- **复现检查（先看）：** G 臂 task1 均值 ≥ 60（作者 CSV 8 种子 83.5）。不满足时不对 guidance 下结论，只报告结果，由用户决定。
- **task1 判定**（单位：百分点，阈值与 L1–L1.6 相同）：

| task1 均值 d = G − F | 判定 |
|---|---|
| ≥ +25 | **guidance 是 task1 增益的主要来源**：底座保留 GFP（流 + guidance） |
| < +10 | **guidance 不必要**：GFP 可删掉 guidance，简化为"FQL 结构 + 混合 TD 目标" |
| +10 到 +25 之间 | **灰区**：不下结论，报告后由用户决定 |

- **task4 判定：**

| task4 均值 d = G − F | 判定 |
|---|---|
| ≤ −10 | **guidance 压制绕行**：GFP 在 task4 上的弱点来自 guidance |
| \|d\| < 10 | **task4 的弱点不来自 guidance** |
| ≥ +10 | **guidance 对 task4 也有帮助** |

- **参照（不参与判定）：**
  - 我们的 ReBRAC（L1）：task1 均值 25.1，task4 均值 22.8。
  - 作者 CSV（8 种子）：GFP task1 83.5 / task4 3.0；ReBRAC 34.1 / 19.5；GFP-AWR 40.2；VaBC 28.8。FQL 的 19 出自 FQL 论文，GFP 仓库没有在本环境重跑 FQL。
- **次要诊断（不参与判定）：**
  - G 臂的 g 均值、g > 0.1、g > 0.75 的比例，与你自己的 GFP task1 日志（800k–1M：g 均值约 0.48，g > 0.75 约 0.17）比较。
  - 两臂的 q_mean、tgt_q_mean、distill_loss、VaBC_loss。

## 工程停止条件（冻结，与 L1–L1.6 相同）

- 任一指标、参数或损失出现 NaN/Inf：停止该条作业，保留状态。
- 非算法原因导致的启动失败（环境、EGL、显存不足）：允许修正后对该条重启一次，并记录在 receipts。
- 低成功率本身**不是**停止或调参理由。
- Q 值发散（|q_mean| > 1000）只记录、不停止，在报告中标注。

## 资源

- 服务器多人共享，使用时不查看、不打断他人进程。
- 正式启动前，在一张空闲卡上依次跑 G 与 F 两臂 task1 seed 0 的 2k 步（评价关闭），确认 F 臂的 `eta_temperature` 为 0、日志里没有 guidance 键，G 臂有 guidance 键，并记录单步耗时，结果写入 `receipts/timing_2k.md`。这一步只检查工程是否正常，不改设计。
- 启动前查看各卡空闲显存，每卡至少留 10 GB 余量，每卡不超过 3 条，间隔约 15 秒错开启动，每条作业单独绑 6 核，同任务同种子的 G 与 F 放在同一张卡上。放置记录写入 `receipts/launch_formal.md`。
