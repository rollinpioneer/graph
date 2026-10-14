# FLOW-CRITIC-COMPARATIVE-TRAIN-V1 最终审查

状态：ALL_REGISTERED_JOBS_ENDED。八条各 1M，共 8M 正式更新；online=0。全部原生进程及八组只读 matched 评价退出 0，24 个检查点各 100 回合，共 2400 回合。四个组合的 50k T2 已计入各自 1M，完整保存恢复后续至 1M。本轮汇总与散列核验新增更新数为 0。未追加种子、任务、重试或调参。

## 成绩及解释

主要成绩为 800k、900k、1M 三点的等权均值，再对两个训练种子等权平均；不选最高分。表中 ± 是两个种子均值的样本标准差，不是置信区间。G 仅比较 HumanoidMaze-medium task1；D 仅比较 Cube-double task2。

### Matched：主比较

| 模型 | seed0 三点均值 % | seed1 三点均值 % | 两种子均值 ± SD % | Return 均值 |
|---|---|---|---|---|
| G0 | 73.00 | 83.00 | 78.00 ± 7.07 | -1344.89 |
| G1 | 48.00 | 71.67 | 59.83 ± 16.73 | -1579.14 |
| D0 | 72.67 | 20.00 | 46.33 ± 37.24 | -599.82 |
| D1 | 4.33 | 4.00 | 4.17 ± 0.24 | -881.13 |

### Native：官方原生统计

| 模型 | seed0 三点均值 % | seed1 三点均值 % | 两种子均值 ± SD % | Return 均值 |
|---|---|---|---|---|
| G0 | 78.01 | 82.12 | 80.07 ± 2.91 | -1355.29 |
| G1 | 48.41 | 70.87 | 59.64 ± 15.88 | -1595.14 |
| D0 | 68.00 | 25.33 | 46.67 ± 30.17 | -595.09 |
| D1 | 4.00 | 4.00 | 4.00 ± 0.00 | -886.00 |

Matched 下 G1 相对 G0 降低 18.17 个百分点；D1 相对 D0 降低 42.17 个百分点。Return 也变差。本轮两个组合没有表现出改善。D1 两种子均接近失败，其较小种子标准差不能解释为更好的训练稳定性。D0 seed1 的末期退化必须保留，不能只报告 seed0；D1 seed1 在 1M 单点高于 D0 seed1，也不能取代规定的三点均值。

Native 采用固定源评价函数：GFP 指定 100 回合、10 并行环境，Actor 与 VaBC Flow 分开保存；DFQL 指定 50 串行回合，并保留初始评价。GFP 的 evaluate_parallel 在已达到每环境配额后仍把后续 done 写入 stats，直到所有环境达到配额，因此官方 stats.success 和 stats.episode.return 可包含超过 100 次结束。保存的 trajs 恰为 100 回合，stats.acc_reward 由这 100 条轨迹计算。原统计未修正；final_checkpoint_results 保存两种 Return、raw_success、原统计差值。Native 汇总 Return 使用 acc_reward，成功率使用官方 stats.success；二者在 GFP 下不必共享同一分母，禁止当作严格 100 回合统一指标。Matched 恰好 100 回合，原始成功率与 Return 重算一致。

早期 D0 内联 matched 缺少全局 NumPy 控制，标为 UNVERIFIED_GLOBAL_RNG，不用于主成绩。matched_verified 固定 NumPy/Python/reset seed=20261014+episode，策略 seed=420000+episode 并逐动作 split。12 个宿主/种子/检查点配对的初始观测 SHA 列表逐项相等。temperature=0 仍采样 Gaussian 噪声，策略没有被宣称为确定性。

### 每检查点成功率与 Return

| job | 协议 | 组件 | 800k % / Return | 900k % / Return | 1M % / Return |
|---|---|---|---|---|---|
| G0_seed0 | matched_verified | Actor | 74.00 / -1376.49 | 73.00 / -1412.33 | 72.00 / -1380.09 |
| G0_seed0 | native | Actor | 83.64 / -1399.24 | 73.83 / -1447.74 | 76.58 / -1352.80 |
| G0_seed1 | matched_verified | Actor | 85.00 / -1254.32 | 82.00 / -1409.93 | 82.00 / -1236.20 |
| G0_seed1 | native | Actor | 80.36 / -1313.37 | 82.20 / -1380.85 | 83.81 / -1237.73 |
| G1_seed0 | matched_verified | Actor | 59.00 / -1617.99 | 38.00 / -1781.08 | 47.00 / -1683.51 |
| G1_seed0 | native | Actor | 53.10 / -1614.63 | 41.67 / -1726.01 | 50.48 / -1708.58 |
| G1_seed1 | matched_verified | Actor | 75.00 / -1393.79 | 71.00 / -1485.23 | 69.00 / -1513.26 |
| G1_seed1 | native | Actor | 80.99 / -1353.74 | 63.11 / -1627.66 | 68.52 / -1540.24 |
| D0_seed0 | matched_verified | Actor | 78.00 / -382.93 | 67.00 / -520.03 | 73.00 / -413.16 |
| D0_seed0 | native | Actor | 76.00 / -400.08 | 54.00 / -624.42 | 74.00 / -394.30 |
| D0_seed1 | matched_verified | Actor | 29.00 / -626.90 | 29.00 / -707.71 | 2.00 / -948.20 |
| D0_seed1 | native | Actor | 32.00 / -584.48 | 34.00 / -691.22 | 10.00 / -876.04 |
| D1_seed0 | matched_verified | Actor | 1.00 / -682.65 | 0.00 / -997.98 | 12.00 / -849.45 |
| D1_seed0 | native | Actor | 2.00 / -733.96 | 2.00 / -986.76 | 8.00 / -885.32 |
| D1_seed1 | matched_verified | Actor | 1.00 / -901.92 | 0.00 / -991.30 | 11.00 / -863.50 |
| D1_seed1 | native | Actor | 0.00 / -850.16 | 0.00 / -992.52 | 12.00 / -867.30 |

[学习曲线](figures/learning_curves.png)、[Q 与梯度](figures/diagnostics.png)、[Critic 分量](figures/critic_components.png)（另存 PDF）。完整 VaBC Flow 三点统计、100k 原生曲线、5k 损失/Q/全局梯度曲线保存在 final_checkpoint_results、native_learning_curves、loss_learning_curves 的 JSON/CSV；各点原始回合在 runs/<job>/native 和 matched_verified 中。曲线图保留各种子，不把三个相关检查点当作三个独立训练种子。

## 成本与资源

| job | 原生进程 GPU 驻留 h | 独立 matched GPU 驻留 h | 采样 CPU h | 主 PID 显存峰值 MiB | 主 PID RSS 峰值 GiB |
|---|---|---|---|---|---|
| G0_seed0 | 3.303 | 0.383 | 2.708 | 3068 | 7.080 |
| G0_seed1 | 3.253 | 0.371 | 2.892 | 3068 | 7.007 |
| G1_seed0 | 7.739 | 0.279 | 2.485 | 3020 | 7.675 |
| G1_seed1 | 6.544 | 0.225 | 2.541 | 3020 | 7.148 |
| D0_seed0 | 4.381 | 0.208 | 3.365 | 954 | 2.981 |
| D0_seed1 | 4.473 | 0.250 | 3.349 | 952 | 2.989 |
| D1_seed0 | 5.862 | 0.229 | 3.204 | 1212 | 2.857 |
| D1_seed1 | 5.530 | 0.258 | 3.177 | 1212 | 2.853 |

原生作业驻留合计 41.085 GPU·h；独立主 matched 合计 2.204 GPU·h；合计 43.289 作业 GPU·h。首条成功正式作业启动至最后原生退出 8.796 h；至最后主评价退出 9.078 h。

这些是进程驻留时长，含编译、原生评价、保存恢复，包含共享卡重叠，不能视作独占卡或 kernel 活跃小时。两张卡上本账号作业共享，另有其他用户负载；八作业六张 A100 40GB，不能由时长推导算法纯吞吐或在独占资源下的精确加速比。原生的 CPU 计数来自每 15 秒 /proc 主 PID 的 utime/stime/cutime/cstime 累计采样；活着的子进程 CPU 与退出末尾可能未完整纳入。显存/RSS 是主 PID 采样峰值；GFP 十个评价子进程额外占用显存/内存，整棵进程树峰值没有完整连续回执。matched 只有驻留时长，未保留完整 CPU/峰值统计。

D0 早期未核验内联评价成本仍包含在原生时长，无法可靠拆分，未从原生预算减去。GFP 原版首次更新 0 的失败启动、修复验证以及历史 T0 准入成本不在上述成功正式作业合计内，失败日志另存；本轮没有补造这些阶段的完整硬件成本。因此这份成本表可追溯，但不是全部工程投入的精确总账。

## 数值健康、停止条件与异常

所有保留 train.csv 数值有限；运行时对参数、优化器和指标进行有限性检查，核对 Adam count=完成更新数、Agent step=更新数+1。冻结幅值条件为参数绝对值 >1e6 或指标 >1e12 时停止；RSS 主 PID >48 GiB 时保护停止。本批成功进程没有触发这些条件。组合 T2 准入记录 Target 移动、有限梯度、Target 零参数梯度及完整恢复；首次实际 GFP fused4 和 DFQL single 更新验证计入预算。无仅编译而未更新的训练路线。

GFP 两条原版最初在 update0 因 fork/EGL 初始化退出1。spawn 十评价环境修复经零更新随机状态对照验证，用户明确批准每条重启一次，均已使用；原失败回执保留。之后八条正式执行及续接退出0。续接是已批准 50k→1M 阶段，不是失败重试。没有干预其他用户进程。

监控器曾把正常完成的协调器退出或已结束作业的历史卡占用当作警告；已用纯文件夹具修正完成判据和仅活跃本账号 GPU 的资源观察，不改动训练代码或训练参数。最终两个协调器完成回执一致，无本账号新训练/评价/GPU 进程。

## 方法与日志诊断（静态及已有日志）

G0：一步 Actor、VaBC Flow、标量 TD Critic/target；VaBC 权重由价值参照产生，Actor 保留端点蒸馏与归一化 Q 目标。G1：保留这些 Actor/VaBC 目标，把价值学习换成在线/target floq 速度场、8步价值积分和在线蒸馏标量读出，另有 EMA 标量 target 提供 guidance。GFP target_use_next=false 时的当前状态 VaBC 动作、下一状态 Q 评价保留来源实际语义。

D0：多步速度 BC Flow Actor、Euler 采样、终点 Q 动作梯度指导及普通 TD Critic。D1 仅改价值学习来源，下一动作仍由 DFQL 多步策略产生，保留 rollout stop-gradient、终点先 clip 再求 Q 输入梯度、逐步权重、q_agg 与 alpha=2；不调用来源 floq 一步 Actor，不添加无用途的 EMA 标量 guidance target。

公共价值核心：期望型 TD backup，多个初值拟合同一期望目标，Flow loss 在噪声维求和，scalar distillation 拟合 stop-gradient 的在线价值积分均值。Actor 使用标量读出；不通过价值积分反传 Actor。Target Flow（及 G 的 target_critic）按固定源码时序使用更新前在线参数 EMA，保持一步滞后。既有 T0 真实更新/保存恢复记录与本批真实更新准入支持工程接线。

模块可以替换，但价值分布、动作梯度尺度、TD 多样本聚合、EMA 时序及优化器状态相互依赖。Flow loss 与普通 TD MSE 具有不同 reduction/单位，原始 loss 大小不能直接比“精度”。日志 training/grad/norm 为全局参数梯度，不能当作 Actor 动作梯度；GFP 归一化 q_loss=1 不表示 Actor 没有 Q 梯度。组合的全局梯度和价值损失较大且有限，只能作为后续诊断线索。

失败假设：蒸馏标量值准确但动作梯度未校准；target 积分/蒸馏误差影响 bootstrap；1M 组合预算可能不足以达到来源 floq 原配方的 2M 收敛；GFP guidance 对新价值尺度敏感。现有日志不能区分这些因果机制，也没有证明延长或调 alpha 会改善。本轮不追加训练或梯度探测。

研究建议：工程接线与可执行性 GO；当前组合性能放行 NO-GO，保持原宿主作后续受控研究的对照。GFP 原版在本任务两种子结果较一致；DFQL 原版种子波动明显。不能据不同任务给两个宿主整体排名。是否做新的诊断、增加种子或训练长度均需新实验决定。floq 价值 Flow/蒸馏、GFP VaBC/guidance/端点蒸馏、DFQL 终点梯度指导均来自已有方法，本次组合本身不构成原创性证明；没有完成文献穷尽或新算法贡献准入。

## 身份、配置、命令与资产

| 源码 | 完整 commit | 固定快照 |
|---|---|---|
| dfql | 8b6b15803cc1e5605ad1524553a2c95e103b0bad | /home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014/upstream_snapshots/dfql |
| floq | 3d60e638b42bbef018c56cb1199ff37c3470520d | /home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014/upstream_snapshots/floq |
| fql | e8cd16eb490332924dfa2492097219f181765933 | /home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014/upstream_snapshots/fql |
| gfp | 1abbd5f0e287caf9bda70e7bc655d0a499ac8401 | /home/xushijie3/work/flow_critic_dual_host_prep_v1_20261014/upstream_snapshots/gfp |

依赖：JAX/jaxlib0.6.2、Flax0.10.7、Optax0.2.5、OGBench1.1.0、MuJoCo3.3.5；GFP Python3.12/Gymnasium1.1.1，DFQL Python3.10/Gymnasium0.29.1。完整环境清单和依赖锁文件见 receipts/source_*_environment.json 与 source_*_requirements.lock。实际运行配置见 runs/<job>/runtime.json、各输出目录 actual_agent_config.json 和检查点 identity.json；matmul_precision=None、x64=false、默认float32，无加速精度替换。

HumanoidMaze 完整2M转移、obs69/action21、奖励[-1,0]、horizon2000、discount.995；Cube 完整1M转移、obs37/action5、奖励[-2,0]、horizon500、discount.99。OGBench 官方 singletask 奖励重标、普通 next-state 数据；宿主动作clip1e-5，不混 GCRL/D4RL/视觉数据。源数据路径、大小、SHA、字段及批次身份见 source_asset_manifest；本轮重新核验旧资产/数据清单48文件不变，以及92个新检查点文件（含84个阶段/定期保存及8个首次分发准入保存）。weights、data_rng sidecar、config/state 哈希见 checkpoint_index。旧权重未用于热启动；P0 同宿主同种子 Actor/VaBC 初始权重与 Agent RNG 精确一致。

宿主任务参数来源：GFP 固定 CSV/默认源码、DFQL 固定 README Cube 配方/默认源码；公共 Critic 参数来自固定 floq get_config 机械继承，8步/8初值/2ensemble、noise_coverage.1、width512、Hum depth4/Cube depth2是移植卡冻结选择。GFP新增 EMA 标量 guidance target 和多样本 target 是本组合适配设计，不声称来源 floq 原版提供。来源 floq time 编码实际为 cos(t) 一通道，encode_time_dim=64未实际扩展时间特征，遵循源码并记录差异。完整参数及来源分类文件 final_parameter_provenance.json；无未说明的参数搜索。

执行保护：原 scalar Agent 内的 training_authorized/optimizer_steps_allowed 可能仍为历史默认 false/0，scalar 类不读取这些字段；实际外层 job gate 与 checked dispatcher 对原版/组合统一执行授权 true、online false 和1M总上限。它们是控制元数据，不是算法差异。Agent 初始 step=1，实际更新=step−1；最终step1000001不是多训练一步。最终保存的网络参数、Adam和随机状态完整恢复，matched 前后 state/checkpoint 哈希相同。

### 实际完整命令（禁止直接重跑本批）

#### G0_seed0 T1

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/gfp_floq`

```text
taskset -c 64,65,66,67,68,69,70,71 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python main.py wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/G0_seed0.gate.json seed=0 env_name=humanoidmaze-medium-navigate-singletask-task1-v0 save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G0_seed0/output agent.critic_type=scalar offline_steps=1000000 online_steps=0 dataset_on_gpu=true n_steps_fused=4 is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 save_interval=100000 save_last_checkpoint=true log_interval=5000 log_on_wandb=false log_locally=true video_episodes=0 hydra.run.dir=. hydra.output_subdir=null
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "G0_seed0",
  "COMPARATIVE_PHASE": "T1",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "humanoidmaze-medium-navigate-singletask-task1-v0",
  "CUDA_VISIBLE_DEVICES": "6",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### G0_seed1 T1

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/gfp_floq`

```text
taskset -c 80,81,82,83,84,85,86,87 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python main.py wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/G0_seed1.gate.json seed=1 env_name=humanoidmaze-medium-navigate-singletask-task1-v0 save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G0_seed1/output agent.critic_type=scalar offline_steps=1000000 online_steps=0 dataset_on_gpu=true n_steps_fused=4 is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 save_interval=100000 save_last_checkpoint=true log_interval=5000 log_on_wandb=false log_locally=true video_episodes=0 hydra.run.dir=. hydra.output_subdir=null
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "G0_seed1",
  "COMPARATIVE_PHASE": "T1",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "humanoidmaze-medium-navigate-singletask-task1-v0",
  "CUDA_VISIBLE_DEVICES": "7",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### G1_seed0 T2

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/gfp_floq`

```text
taskset -c 32,33,34,35,36,37,38,39 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python main.py wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/G1_seed0.gate.json seed=0 env_name=humanoidmaze-medium-navigate-singletask-task1-v0 save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G1_seed0/output agent.critic_type=floq offline_steps=1000000 online_steps=0 dataset_on_gpu=true n_steps_fused=4 is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 save_interval=100000 save_last_checkpoint=true log_interval=5000 log_on_wandb=false log_locally=true video_episodes=0 hydra.run.dir=. hydra.output_subdir=null
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "G1_seed0",
  "COMPARATIVE_PHASE": "T2",
  "COMPARATIVE_PHASE_STOP": "50000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "humanoidmaze-medium-navigate-singletask-task1-v0",
  "CUDA_VISIBLE_DEVICES": "2",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### G1_seed0 T3

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/gfp_floq`

```text
taskset -c 32,33,34,35,36,37,38,39 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python main.py wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/G1_seed0.gate.json seed=0 env_name=humanoidmaze-medium-navigate-singletask-task1-v0 save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G1_seed0/output agent.critic_type=floq offline_steps=1000000 online_steps=0 dataset_on_gpu=true n_steps_fused=4 is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 save_interval=100000 save_last_checkpoint=true log_interval=5000 log_on_wandb=false log_locally=true video_episodes=0 hydra.run.dir=. hydra.output_subdir=null restore_path=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G1_seed0/output/gfp_OGBench_Env_humanoidmaze_medium_navigate_singletask_task1_v0/date_20261014_162148_seed_0 restore_epoch=50000
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "G1_seed0",
  "COMPARATIVE_PHASE": "T3",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "humanoidmaze-medium-navigate-singletask-task1-v0",
  "CUDA_VISIBLE_DEVICES": "2",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### G1_seed1 T2

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/gfp_floq`

```text
taskset -c 40,41,42,43,44,45,46,47 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python main.py wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/G1_seed1.gate.json seed=1 env_name=humanoidmaze-medium-navigate-singletask-task1-v0 save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G1_seed1/output agent.critic_type=floq offline_steps=1000000 online_steps=0 dataset_on_gpu=true n_steps_fused=4 is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 save_interval=100000 save_last_checkpoint=true log_interval=5000 log_on_wandb=false log_locally=true video_episodes=0 hydra.run.dir=. hydra.output_subdir=null
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "G1_seed1",
  "COMPARATIVE_PHASE": "T2",
  "COMPARATIVE_PHASE_STOP": "50000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "humanoidmaze-medium-navigate-singletask-task1-v0",
  "CUDA_VISIBLE_DEVICES": "3",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### G1_seed1 T3

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/gfp_floq`

```text
taskset -c 40,41,42,43,44,45,46,47 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/gfp/bin/python main.py wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/G1_seed1.gate.json seed=1 env_name=humanoidmaze-medium-navigate-singletask-task1-v0 save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G1_seed1/output agent.critic_type=floq offline_steps=1000000 online_steps=0 dataset_on_gpu=true n_steps_fused=4 is_final=true eval_episodes=100 n_eval_envs=10 eval_interval=100000 save_interval=100000 save_last_checkpoint=true log_interval=5000 log_on_wandb=false log_locally=true video_episodes=0 hydra.run.dir=. hydra.output_subdir=null restore_path=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/G1_seed1/output/gfp_OGBench_Env_humanoidmaze_medium_navigate_singletask_task1_v0/date_20261014_162148_seed_1 restore_epoch=50000
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "G1_seed1",
  "COMPARATIVE_PHASE": "T3",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "humanoidmaze-medium-navigate-singletask-task1-v0",
  "CUDA_VISIBLE_DEVICES": "3",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### D0_seed0 T1

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/dfql_floq`

```text
taskset -c 72,73,74,75,76,77,78,79 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/dfql/bin/python main.py --wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/D0_seed0.gate.json --seed=0 --env_name=cube-double-play-singletask-task2-v0 --save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D0_seed0/output --agent=agents/dfql.py --agent.critic_type=scalar --agent.alpha=2 --agent.discount=0.99 --agent.q_agg=mean --offline_steps=1000000 --online_steps=0 --eval_episodes=50 --eval_interval=100000 --save_interval=100000 --log_interval=5000 --wandb_mode=disabled --video_episodes=0 --run_group=D0_seed0
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "D0_seed0",
  "COMPARATIVE_PHASE": "T1",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "cube-double-play-singletask-task2-v0",
  "CUDA_VISIBLE_DEVICES": "5",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### D0_seed1 T1

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/dfql_floq`

```text
taskset -c 88,89,90,91,92,93,94,95 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/dfql/bin/python main.py --wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/D0_seed1.gate.json --seed=1 --env_name=cube-double-play-singletask-task2-v0 --save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D0_seed1/output --agent=agents/dfql.py --agent.critic_type=scalar --agent.alpha=2 --agent.discount=0.99 --agent.q_agg=mean --offline_steps=1000000 --online_steps=0 --eval_episodes=50 --eval_interval=100000 --save_interval=100000 --log_interval=5000 --wandb_mode=disabled --video_episodes=0 --run_group=D0_seed1
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "D0_seed1",
  "COMPARATIVE_PHASE": "T1",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "cube-double-play-singletask-task2-v0",
  "CUDA_VISIBLE_DEVICES": "4",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### D1_seed0 T2

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/dfql_floq`

```text
taskset -c 96,97,98,99,100,101,102,103 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/dfql/bin/python main.py --wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/D1_seed0.gate.json --seed=0 --env_name=cube-double-play-singletask-task2-v0 --save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D1_seed0/output --agent=agents/dfql.py --agent.critic_type=floq --agent.alpha=2 --agent.discount=0.99 --agent.q_agg=mean --offline_steps=1000000 --online_steps=0 --eval_episodes=50 --eval_interval=100000 --save_interval=100000 --log_interval=5000 --wandb_mode=disabled --video_episodes=0 --run_group=D1_seed0 --agent.block_depth=2 --agent.block_width=512
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "D1_seed0",
  "COMPARATIVE_PHASE": "T2",
  "COMPARATIVE_PHASE_STOP": "50000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "cube-double-play-singletask-task2-v0",
  "CUDA_VISIBLE_DEVICES": "5",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### D1_seed0 T3

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/dfql_floq`

```text
taskset -c 96,97,98,99,100,101,102,103 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/dfql/bin/python main.py --wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/D1_seed0.gate.json --seed=0 --env_name=cube-double-play-singletask-task2-v0 --save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D1_seed0/output --agent=agents/dfql.py --agent.critic_type=floq --agent.alpha=2 --agent.discount=0.99 --agent.q_agg=mean --offline_steps=1000000 --online_steps=0 --eval_episodes=50 --eval_interval=100000 --save_interval=100000 --log_interval=5000 --wandb_mode=disabled --video_episodes=0 --run_group=D1_seed0 --agent.block_depth=2 --agent.block_width=512 --restore_path=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D1_seed0/output/dummy/D1_seed0/sd000_20261014_162149 --restore_epoch=50000
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "D1_seed0",
  "COMPARATIVE_PHASE": "T3",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "cube-double-play-singletask-task2-v0",
  "CUDA_VISIBLE_DEVICES": "5",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### D1_seed1 T2

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/dfql_floq`

```text
taskset -c 104,105,106,107,108,109,110,111 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/dfql/bin/python main.py --wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/D1_seed1.gate.json --seed=1 --env_name=cube-double-play-singletask-task2-v0 --save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D1_seed1/output --agent=agents/dfql.py --agent.critic_type=floq --agent.alpha=2 --agent.discount=0.99 --agent.q_agg=mean --offline_steps=1000000 --online_steps=0 --eval_episodes=50 --eval_interval=100000 --save_interval=100000 --log_interval=5000 --wandb_mode=disabled --video_episodes=0 --run_group=D1_seed1 --agent.block_depth=2 --agent.block_width=512
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "D1_seed1",
  "COMPARATIVE_PHASE": "T2",
  "COMPARATIVE_PHASE_STOP": "50000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "cube-double-play-singletask-task2-v0",
  "CUDA_VISIBLE_DEVICES": "7",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

#### D1_seed1 T3

工作目录：`/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/workspaces/dfql_floq`

```text
taskset -c 104,105,106,107,108,109,110,111 /home/xushijie3/work/flow_critic_comparative_train_v1_20261014/envs/dfql/bin/python main.py --wiring_gate=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/configs/D1_seed1.gate.json --seed=1 --env_name=cube-double-play-singletask-task2-v0 --save_dir=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D1_seed1/output --agent=agents/dfql.py --agent.critic_type=floq --agent.alpha=2 --agent.discount=0.99 --agent.q_agg=mean --offline_steps=1000000 --online_steps=0 --eval_episodes=50 --eval_interval=100000 --save_interval=100000 --log_interval=5000 --wandb_mode=disabled --video_episodes=0 --run_group=D1_seed1 --agent.block_depth=2 --agent.block_width=512 --restore_path=/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/runs/D1_seed1/output/dummy/D1_seed1/sd001_20261014_162149 --restore_epoch=50000
```

环境覆盖：

```json
{
  "COMPARATIVE_JOB": "D1_seed1",
  "COMPARATIVE_PHASE": "T3",
  "COMPARATIVE_PHASE_STOP": "1000000",
  "COMPARATIVE_ROOT": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014",
  "COMPARATIVE_TASK": "cube-double-play-singletask-task2-v0",
  "CUDA_VISIBLE_DEVICES": "7",
  "HOME": "/home/xushijie3/work/flow_critic_comparative_train_v1_20261014/home",
  "MUJOCO_GL": "egl",
  "OMP_NUM_THREADS": "1",
  "OPENBLAS_NUM_THREADS": "1",
  "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.30",
  "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
}
```

## 准入结论与限制

工程运行成功：八条全部达到冻结更新数并退出0；组合完整损失/梯度/Target/真实更新/保存恢复接通。模块职责与状态依赖明确，具备设计后续实验的工程基础。

论文成绩复现：本批仅指定任务1或task2、两个种子，作者对应多任务/种子聚合未独立对齐；作者对应数字标记 NOT_COMPARABLE_AUTHOR_AGGREGATION，不填写猜测值，不声称论文性能复现。GFP Native 统计还具有上面记录的官方并行分母差异。

组合性能：当前固定1M协议下两套组合均低于对应宿主，不能宣传叠加收益。两个种子只构成任务内初步证据；检查点间及共用评价场景相关，未作显著性宣称。SOURCE floq 的2M并不授权自动延长组合。

新算法潜力：没有完成原创性或方法机制证明。后续研究决策须另卡，当前不做调参、不延长、不自动重训。

所有结果/失败/差异保留；branch codex/flow-critic-comparative-train-v1-20261014。服务器权重/数据/完整日志留在本根目录及源资产位置，仅代码、冻结卡、指标、原始轻量评价与回执进入 Git。远端最新提交及本地同步身份以 receipts/git_latest.json 和本地 HANDOFF.md 为准，不覆盖旧分支/旧资产。完成归档后暂停 lab 监控。
