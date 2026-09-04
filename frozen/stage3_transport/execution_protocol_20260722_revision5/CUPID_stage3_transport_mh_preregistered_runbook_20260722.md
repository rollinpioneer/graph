# CUPID Stage 3: Transport-MH 预注册执行文档

版本日期：2026-07-22  
工作根目录：`/home/xushijie/CUPID`  
预检状态：`READY_WITH_RESOURCE_GATE`  
本文件状态：预注册计划已冻结；Stage 3A 已获用户授权并于 2026-07-22 启动

## 执行附录（2026-07-22）

冻结副本 `frozen/stage3_transport_preflight_20260722/` 保持不变。用户随后明确要求继续完整路线，并继续授权使用物理 GPU 1。

- 正式 batch smoke：PASS；loss `1.1256551743`，有梯度参数 `17,181,204`，梯度和更新后参数全部有限，进程峰值 reserved 显存 `294 MiB`。
- 正式基础训练：后台 supervisor 已启动；输出目录为 `repo/data/outputs/train/20260722_cupid_transport_stage3/20260722_cupid_transport_stage3_train_diffusion_unet_lowdim_transport_mh_0`。
- 恢复机制：`training.resume=true`，每 50 epoch 保存 `latest.ckpt`；非零退出由独立 supervisor 等待 30 秒后恢复。
- 独立进程：训练 supervisor、GPU 监控、SMTP 邮件监控、固定 100 rollout 等待器以及 TRAK/稳定性诊断等待器均使用独立 session，PID/状态位于 `pids/stage3_transport/` 与 `status/stage3_transport/`。
- 自动门控：只有基础训练审计 PASS 才收集 seeds 100000--100099；只有 rollout 类别平衡 PASS 才进行正式配置 TRAK smoke；只有 smoke 与资源外推 PASS 才运行原始 CUPID/TRAK；只有离线强门槛 PASS 才允许另行制定过滤重训练。

### 条件通过后的过滤实现约束

作者的 `retrain_policies.sh` 通过 `filter_ratio` 截取 curation 排名列表，实际删除数为 `int(train_mask.sum() * filter_ratio)`；当前工作树也没有该入口预期的 `configs/curation/low_dim/transport_mh/train_config.yaml`。因此不得用近似比例代替本实验的可变大小稳定核心。

若且仅若稳定性诊断得到 `PASS_STABILITY_WEIGHTED_CORE_CANDIDATE`，后续计划必须先：

1. 从完整 100-rollout 的底部 20% Bootstrap 删除概率中冻结 `p >= 0.90` 的最终 Demo ID；
2. 保存有序 ID、数量、来源矩阵/manifest/split/分数哈希；
3. 通过已加入 `get_dataset_masks` 的 `filter_episode_ids` 显式传入冻结 ID 列表；该接口与比例 curation 互斥，并拒绝空列表、重复 ID、非整数 ID、越界 ID 和非训练集 ID；
4. 断言所有删除 ID 都属于原 192 条训练集，validation/holdout mask 完全不变；
5. 先执行同 seed、同 epoch、同评估 seed 的未过滤/过滤成对训练，再决定是否扩展到 seeds 0、1、2。

`filter_episode_ids` 已在正式 Transport 300-demo split 上做掩码级和完整 Hydra 数据集实例化测试：示例删除 5 个训练 ID 后，split 从 192/12/96 严格变为 187/12/96，validation/holdout 逐位不变。该测试只验证机制，不预选或暗示最终过滤 ID。

## 1. 研究目的与边界

Square-MH 的固定过滤和 Bootstrap 序贯分支已经按预注册门槛停止。Stage 3 转向 Transport-MH，目的是检查不同任务上是否存在更强、更稳定的 Demo 影响信号。

本阶段不得把 Square-MH 的低分 Demo、阈值结果或 Bootstrap 结论直接迁移到 Transport-MH。完整 100 条 rollout 仍只能称为有限池参考。低影响分数不能直接称为有害数据。

本文件不授权立即训练。启动 Stage 3A 前必须满足 GPU 资源门槛，并由用户明确继续。

## 2. 已验证输入

- 仓库提交：`2941eba9427a3398d3cbe26b10c01e3ece56bd18`
- 数据集：`repo/data/robomimic/datasets/transport/mh/low_dim_abs.hdf5`
- 数据 SHA-256：`2034f404d1e9dd04514c443f9b2fb2bda99f320b46acd9ffc9983fdbec0f9d95`
- 文件大小：636,980,704 字节
- Demo：300 条
- 原始时序步：195,800
- Episode 长度：最小 392，中位 630，最大 2614，均值 652.67
- 配置：`repo/configs/low_dim/transport_mh/diffusion_policy_cnn/config.yaml`
- 模型参数：17,181,204
- 输入/动作维度：59 / 20
- Rollout 最大步数：700

冻结划分继续使用与 Square 最小复现一致的协议：

```text
seed=0
train_ratio=0.64
val_ratio=0.04
uniform_quality=true
train / validation / holdout = 192 / 12 / 96 demos
train / validation / holdout samples = 125324 / 7758 / 60618
batch_size=256
train batches per epoch=490
```

## 3. 资源门槛

预检机器为 112 个逻辑 CPU、503 GiB RAM、约 1.3 TiB 可用磁盘，GPU 1 为 40 GB A100。预检时 GPU 1 已被其他进程占用 12,333 MiB 且利用率为 83%。

正式启动前必须重新检查：

1. GPU 1 可见；
2. 没有高负载冲突任务，或至少有 28 GiB 连续可用显存；
3. 单 batch 前向、反向和优化器 smoke 无 OOM、NaN 或 Inf；
4. smoke 输出必须与正式输出目录隔离并冻结日志；
5. 不得通过缩小模型、batch size、投影维度或时间步数绕过失败，除非新的预注册文档明确批准。

经验时间估算：1751 epoch 训练约 24-36 小时，100 次 rollout 约 2-4 小时。TRAK 因模型参数约为 Square 的 15.5 倍、样本约为 2.5 倍，可能需要 1-4 天；必须先用完整配置的小样本 smoke 实测外推，估算不能当作完成保证。

## 4. Stage 3A：基础策略

正式训练冻结为：

```text
config-dir=configs/low_dim/transport_mh/diffusion_policy_cnn
config-name=config.yaml
physical GPU=1
training.seed=0
training.num_epochs=1751
training.resume=true
training.checkpoint_every=50
training.rollout_every=50
checkpoint.topk.k=3
task.dataset.seed=0
task.dataset.val_ratio=0.04
task.dataset.dataset_mask_kwargs.train_ratio=0.64
task.dataset.dataset_mask_kwargs.uniform_quality=true
task.env_runner.n_envs=8
dataloader.num_workers=0
val_dataloader.num_workers=0
logging.mode=offline
```

训练必须由 `setsid`/`nohup` 的独立 supervisor 启动，PPID 应为 1。Supervisor 必须记录 PID、尝试次数、退出码、开始/结束时间，并依靠 `latest.ckpt` 自动恢复。结束或终止邮件监控进程也必须独立于 Codex 会话。

训练通过条件：

- 到达 epoch 1750；
- 所有记录的 loss 有限；
- `latest.ckpt` 存在且可加载；
- 最终输出和 checkpoint 哈希冻结；
- 不以训练内周期性 rollout 的单次分数作为方法结论。

## 5. Stage 3B：固定 100 条 Rollout

训练通过后才允许收集：

```text
num_episodes=100
test_start_seed=100000
checkpoint=latest.ckpt
device=cuda:0 (映射到物理 GPU 1)
```

必须保存 100 个 episode 文件、100 个视频、seed、成功标签、决策点数和 SHA-256。类别平衡门槛为成功至少 5 条且失败至少 5 条；未通过则停止 net influence 分支，不补采样、不移动 seed 窗口。

## 6. Stage 3C：TRAK 可行性门槛

仅在固定 rollout 池通过后执行。正式参数保持：

```text
proj_dim=4000
proj_max_batch_size=32
lambda_reg=0.0
use_half_precision=0
loss_fn=square
num_timesteps=64
batch_size=128
seed=0
featurize_holdout=1
finalize_scores=1
```

正式 TRAK 前先运行同一模型、投影维度和 64 时间步的小样本 smoke，记录每个训练样本和 rollout 决策点的实测耗时及峰值显存。只有结果有限且预计总时长、磁盘需求可接受时才启动完全脱离会话的正式作业。

只有 CUDA OOM 时允许把 `proj_max_batch_size` 从 32 改为 16；不得降低 `proj_dim` 或 `num_timesteps`。任何其他错误均停止并诊断。

## 7. Stage 3D：Transport 离线过滤诊断

TRAK 完成后先重建官方 net score，最大误差必须不高于 `2e-3`。随后按顺序执行：

1. 整体排名预算稳定性；
2. 最低 38 条固定过滤稳定性；
3. 50/50 不重叠池压力测试；
4. 固定 5%、10%、20%、30% 边界诊断；
5. 稳定核心与相同数量简单最低分集合对照。

Stage 2B 的强通过门槛保持不变，不根据 Transport 结果调阈值：稳定核心非空率不低于 80%，平均至少 5 条，独立池准确率不低于 70%，相对固定最低 38 条提高至少 0.15，独立池平均排名处于后 25%，50 条 rollout 下至少 5 条稳定 Demo；要证明 Bootstrap 成员选择有效，还需比相同数量简单最低分集合提高至少 0.03。两项准确率增益必须在同一次 50/50 拆分、同一评估方向且稳定核心非空的评价上逐次配对后取均值，不允许用覆盖评价集合不同的总体均值相减。

## 8. 下游训练与多种子门槛

任何过滤后训练均禁止自动启动。只有 Transport 离线诊断达到强通过，才允许另行制定：

- 一次未过滤 Transport 基线重训练；
- 一次冻结过滤集合的重训练；
- 相同训练预算和评估 seed；
- 随后扩展到 seed 0、1、2 的多种子确认。

若离线诊断失败，则停止 Transport 过滤主线，不叠加交叉拟合、主动采样或贝叶斯模块挽救。

## 9. 执行顺序

```text
当前：预检与计划冻结完成
下一授权点：GPU 空闲检查 + 训练显存 smoke
smoke PASS：创建独立 supervisor、邮件监控和恢复机制
基础训练 PASS：固定 100 条 rollout
类别平衡 PASS：TRAK 小样本时间/显存 smoke
TRAK 可行：正式 TRAK 与离线诊断
离线强通过：仅提交下游重训练计划
```

## 10. 当前停止条件

在用户明确允许 Stage 3A 之前停止。当前不得启动 Transport 训练。
