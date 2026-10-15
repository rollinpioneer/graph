# GREBRAC-L1-V1：G-ReBRAC 配对验证

授权：2026-10-15 10:33 UTC，用户在决策卡上选择"先做 L1"，即：实现 G-ReBRAC，先测 2k 步耗时，再跑 16 条配对训练。
方案来源：`/mnt/project-files/gfp_simplify_review/GFP_SIMPLIFICATION_REVIEW_20261015.md` 第 4.1–5 节。
远端根目录：`/home/xushijie3/work/grebrac_l1_v1_20261015`（主机 gpu03，账号 xushijie3）。

## 要回答的问题

GFP 在 humanoidmaze-medium task1 上的增益（作者 8 种子：GFP 83.5，ReBRAC 34.1）能否离开"流教师 + 蒸馏"，直接作用在单个确定性 actor 上？同时，这样做会不会伤害 GFP 落后的 task4（GFP 3.0，ReBRAC 19.5）？

## 代码与环境（冻结）

- **源码：** GFP 官方 `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`。远端从 `flow_baselines_preflight_20261013/cache` 的 tarball 解出。38 个源码文件的 sha256 清单与 GitHub 克隆一致（清单哈希 `66aa6e8a…33aa`）。
- **补丁：** `code/grebrac_p1.patch`（sha256 `7b09fe08…60f1`），只改两个文件：
  - `agents/rebrac.py`：新增 `guidance` 开关和 `compute_guidance`。
  - `config/agent/rebrac.yaml`：新增 `guidance: False`、`eta_temperature: 1e-3`、`guidance_normalize: True`。
- **G-ReBRAC 的 actor 损失：**
  - 原版：−λQ(s,π(s)) + α_actor·‖π(s)−a‖²
  - 改为：−λQ(s,π(s)) + α_actor·w̃·‖π(s)−a‖²
  - 其中 g = sigmoid((λ/η)(Q̄(s,a) − Q̄(s,π(s))))，Q̄ 为 target critic，按 q_agg 聚合；w̃ = g / max(mean_batch g, 1e-6)，并 stop-grad。
  - critic、目标网络、采样、优化器均不变。
- **本地检查（CPU，`tests/`）：**
  - `guidance=False` 时，6 次更新后参数与原版 ReBRAC 逐位相同；
  - η→∞（g=0.5，归一化后 w=1）时同样逐位相同；
  - 开启 guidance 时损失有限，fused 更新与验证集日志路径都能运行。
- **环境：** 只读复用 `flow_baselines_preflight_20261013/envs/gfp312`。版本：JAX/jaxlib 0.6.2，Flax 0.10.7，Optax 0.2.5，OGBench 1.1.0，Gymnasium 1.1.1，MuJoCo 3.3.5。
- **数据：** OGBench 1.1.0 `humanoidmaze-medium-navigate-v0(.npz, -val.npz)`，软链接到 `flow_critic_dual_host_prep_v1_20261014/data`。sha256 与 `flow_critic_comparative_train_v1_20261014/data` 中的副本一致（`receipts/data_sha256.txt`）。
- **入口：** `code/entry_spawn.py`，只把 multiprocessing 设为 spawn，再原样运行官方 `main.py`。

## 实验矩阵（冻结）

| 臂 | agent 参数 | 其余配置 |
|---|---|---|
| R：ReBRAC | `agent=rebrac` | 官方 `rebrac_params.csv` 的 humanoidmaze_medium_navigate 行：α_actor 0.01，α_critic 0.01，γ 0.995，q_agg mean，batch 256，1M 步 |
| G：G-ReBRAC | `agent=rebrac agent.guidance=true agent.eta_temperature=0.001` | 同上。η 取自 GFP 官方 CSV 的同环境行；自归一化开启 |

- **任务：** `humanoidmaze-medium-navigate-singletask-task1-v0`、`…-task4-v0`。
- **种子：** 0、1、2、3。共 2 臂 × 2 任务 × 4 种子 = **16 条**，每条 1,000,000 次离线更新，online 0。
- **配对：** 同一种子下两臂的初始化和批次序列相同。guidance 不消耗随机数；本地已验证 `guidance=False` 逐位等同原版。
- **公共参数：** `is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 log_interval=5000 save_interval=100000 save_last_checkpoint=true log_on_wandb=false`。`n_steps_fused=4`，`dataset_on_gpu=true`，均为默认。
- **不做：** 不调参，不加种子，不加任务，不跑原始 g 变体，不跑 cube-double。这些都需要另行授权。

## 主指标与判据（冻结，先于任何正式结果）

- **主指标：** 官方 `final_Actor`，即 800k、900k、1M 三次评价的均值，每次 100 回合。每个种子算配对差 d_s = G_s − R_s，报告 4 个种子的均值和配对 t 95% 区间。每个种子、每个检查点都列出。
- **判定**（单位：百分点）：

| 结果 | 判定 |
|---|---|
| task1 均值 d ≥ +25，且 task4 均值 d ≥ −10 | **简化成功**：guidance 不需要流教师 |
| task1 均值 d < +10 | **不能搬**：guidance 离不开流教师 |
| task1 均值 d ≥ +25，但 task4 均值 d < −10 | **伤绕行**：guidance 本身压制绕行行为（机制发现） |
| task1 均值 d 在 +10 与 +25 之间 | **灰区**：不下结论，报告后由用户决定是否补种子 |
| task1 达标，task4 落在 −10 附近 | 按上表照判，同时报告区间 |

- **参照（不参与判定）：** 作者 CSV 8 种子结果：GFP task1 83.5 / task4 3.0；ReBRAC task1 34.1 / task4 19.5。另有用户自己的 GFP task1 seed0/1。

## 工程停止条件（冻结）

- 任一指标、参数或损失出现 NaN/Inf：停止该条作业，保留状态。
- 非算法原因导致的启动失败（环境、EGL、显存不足）：允许修正后对该条重启一次，并记录在 receipts。
- 低成功率本身**不是**停止或调参理由。
- Q 值发散（|q_mean| > 1000；本任务回报范围约 [−200, 0]）只记录、不停止，在报告中标注。

## 资源

- 服务器为多人共享，使用时不影响他人进程。
- GPU 选择以当时的空闲显存为准，`XLA_PYTHON_CLIENT_PREALLOCATE=false`，每条作业单独绑核。
- 正式启动前先测 2k 步耗时：GFP（官方 task1 配置）、ReBRAC、G-ReBRAC 在同一张卡上依次运行，取第 1000→2000 次更新之间的墙钟时间。结果写入 `receipts/timing_2k.md`。
