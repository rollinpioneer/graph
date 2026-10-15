# GREBRAC-L15-BCREF：用独立 BC 网络作参考的 G-ReBRAC

授权：2026-10-15 15:13 UTC，用户在决策卡"要不要测用独立 BC 网络作参考的 G-ReBRAC？"上选择"测 BC 参考版"。卡上写明的范围："task1+task4 各 4 种子共 8 条新作业，约 1.5 小时，复用本轮 ReBRAC 作对照，判据先冻结再启动。"
前置结果：`/mnt/project-files/gfp_simplify_review/GREBRAC_L1_RESULT_20261015.md`（L1 判为"不能搬"，第 5 节指出未测"独立参考"）。
远端根目录：`/home/xushijie3/work/grebrac_l15_bcref_20261015`（主机 gpu03，账号 xushijie3）。

## 要回答的问题

L1 的 G-ReBRAC 用 actor 自己作 guidance 的参考动作，结果约束在 actor 偏离数据的地方被撤掉，策略失效。这一轮只换掉参考动作的来源：改用一个独立训练、只做行为克隆的确定性网络。

- 如果 task1 的增益回来了：GFP 的 guidance 需要的是"与被约束 actor 分开的参考"，不一定需要流。
- 如果仍然没有增益：单峰的独立 BC 参考不够，增益依赖流教师（多峰参考）或 GFP 的蒸馏结构。

## 代码与环境（冻结）

- **源码：** 与 L1 相同的 GFP 官方 `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`，先打 L1 补丁 p1，再打增量补丁 p2。
  - `code/grebrac_p2.patch`：相对官方原版的完整差异，sha256 `3c7aa489…ef82d`。
  - `code/grebrac_p2_incremental.patch`：相对 p1 的增量，sha256 `a9ae9b0d…e082f`。远端 `receipts/` 中有同一文件。
  - 打完后的文件：`agents/rebrac.py` sha256 `a027c7f0ec30…`，`config/agent/rebrac.yaml` sha256 `3dac1da5b529…`，远端已核对。
- **p2 只做三件事：**
  - 新增配置 `guidance_ref`，默认 `actor`（即 L1 的行为）。
  - `guidance_ref=bc` 时新建模块 `bc_ref`：与 actor 结构相同（4×512，tanh 输出），取 `.mode()` 作确定性动作。它只用未加权 MSE ‖bc_ref(s) − a‖² 拟合数据动作，与 actor 在同一次更新里训练（actor_freq=2，共约 500k 次），同一个 Adam 优化器、同一学习率。
  - guidance 的参考动作改为 stop-grad 的 bc_ref(s)：g = sigmoid((λ/η)(Q̄(s,a) − Q̄(s,bc_ref(s))))。其余与 L1 完全相同：w̃ = g / max(mean g, 1e-6)，actor 损失 −λQ(s,π(s)) + α_actor·w̃·‖π(s)−a‖²。
  - critic、目标网络、采样、actor 结构都不变。优化器是普通 `optax.adam`，没有全局梯度裁剪，所以新增参数不影响其他模块的更新。
- **本地检查（CPU，`tests/test_bcref.py`，输出见 `receipts/local_tests.txt`）：** 全部通过。
  - `guidance=False` 时 6 次更新后参数与原版 ReBRAC、与 p1 均逐位相同。
  - `guidance_ref=actor` 时与 L1 的 G-ReBRAC 逐位相同。
  - 新增 `bc_ref` 模块后，其余模块的初始化不变（Flax 按模块路径派生随机数）。
  - η→∞（g=0.5，w̃=1）且 `guidance_ref=bc` 时，除 `bc_ref` 外所有参数与原版 ReBRAC 逐位相同。这说明 bc_ref 的训练不干扰 actor 和 critic，L1 的 R 臂可以作为配对对照。
  - 40 次更新内参数有限，bc_ref mse 6.80 → 4.67，g 约 0.54；fused 更新与验证集日志路径正常，日志里有 `bc_ref/mse` 和 `actor/guidance`。
  - 参照数组 `orig_off.npy`、`patched_off.npy`、`patched_on_norm.npy` 由 L1 的 `tests/test_grebrac.py` 生成。
- **环境、数据、入口：** 与 L1 相同（`gfp312` 环境只读复用；数据软链接同一份 OGBench 1.1.0 `humanoidmaze-medium-navigate-v0`；`code/entry_spawn.py` 只把 multiprocessing 设为 spawn）。
- **脚本：** `code/launch_formal.sh`（sha256 `9be026fc…0ce53`）、`code/summarize.py`（`66293eee…3b7`）、`code/extract_curves.py`（`ce2ad445…afc3`），远端副本已核对。

## 实验矩阵（冻结）

| 臂 | agent 参数 | 来源 |
|---|---|---|
| B：G-ReBRAC（BC 参考） | `agent=rebrac agent.guidance=true agent.eta_temperature=0.001 agent.guidance_ref=bc` | 本轮新跑 |
| R：ReBRAC | `agent=rebrac` | 复用 L1 的 `runs/formal/R_task{1,4}_s{0..3}`，不重跑 |

- **任务：** `humanoidmaze-medium-navigate-singletask-task1-v0`、`…-task4-v0`。
- **种子：** 0、1、2、3。新作业共 1 臂 × 2 任务 × 4 种子 = **8 条**，每条 1,000,000 次离线更新，online 0。
- **超参数：** 官方 `rebrac_params.csv` 的 humanoidmaze_medium_navigate 行，η = 1e-3 来自 GFP 同环境行，与 L1 完全相同。
- **公共参数：** 与 L1 相同：`is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 log_interval=5000 save_interval=100000 save_last_checkpoint=true log_on_wandb=false`，`n_steps_fused=4`、`dataset_on_gpu=true` 为默认。
- **配对：** 同一种子下 B 与 R 的初始化和批次序列相同（上面的 η→∞ 检查）。R 是更早在别的卡上跑的，GPU 浮点运算不保证逐位可复现，所以配对的含义是同种子，不是逐位相同。
- **不做：** 原始 g（不归一化）变体、其他 η、cube-double、加种子、调参。这些都需要另行授权。

## 主指标与判据（冻结，先于任何 B 臂结果）

- **主指标：** 官方 `final_Actor`（800k、900k、1M 三次评价的均值）。每个种子算 d_s = B_s − R_s，报告 4 个种子的均值和配对 t 95% 区间，并列出每个种子、每个检查点。
- **判定**（单位：百分点，阈值与 L1 相同）：

| 结果 | 判定 |
|---|---|
| task1 均值 d ≥ +25，且 task4 均值 d ≥ −10 | **独立参考即可**：guidance 不需要流，只需要与 actor 分开的参考 |
| task1 均值 d < +10 | **独立单峰参考不够**：增益依赖流教师（多峰参考）或 GFP 的蒸馏结构 |
| task1 均值 d ≥ +25，但 task4 均值 d < −10 | **伤绕行**：guidance 本身压制绕行行为 |
| task1 均值 d 在 +10 与 +25 之间 | **灰区**：不下结论，报告后由用户决定 |
| task1 达标，task4 落在 −10 附近 | 按上表照判，同时报告区间 |

- **R 臂基准（已知，来自 L1）：** task1 均值 25.1，task4 均值 22.8。作者 CSV 8 种子参照（不参与判定）：GFP task1 83.5 / task4 3.0；ReBRAC 34.1 / 19.5。
- **次要诊断（不参与判定）：**
  - bc_ref 的验证集 mse（应明显下降并稳定）。
  - actor 与数据的 mse，与 R 臂（800k–1M 均值约 1.65）和 L1 的 G 臂（8–12）比较。
  - g 均值、g > 0.1、g > 0.75 的比例，与你自己的 GFP task1 日志（800k–1M：g 均值约 0.48，g > 0.75 约 0.17）比较。
  - q_mean，与 R 臂比较。

## 工程停止条件（冻结，与 L1 相同）

- 任一指标、参数或损失出现 NaN/Inf：停止该条作业，保留状态。
- 非算法原因导致的启动失败（环境、EGL、显存不足）：允许修正后对该条重启一次，并记录在 receipts。
- 低成功率本身**不是**停止或调参理由。
- Q 值发散（|q_mean| > 1000）只记录、不停止，在报告中标注。

## 资源

- 服务器多人共享，使用时不查看、不打断他人进程。
- 正式启动前，在一张卡上跑 B 臂 task1 seed 0 的 2k 步（评价关闭），确认单步耗时、g 没有塌到 0、bc_ref mse 在下降，结果写入 `receipts/timing_2k.md`。这一步只检查工程是否正常，不改设计。
- 启动前查看各卡空闲显存，每卡至少留 10 GB 余量，每卡不超过 3 条，间隔约 15 秒错开启动，每条作业单独绑 6 核。放置记录写入 `receipts/launch_formal.md`。
