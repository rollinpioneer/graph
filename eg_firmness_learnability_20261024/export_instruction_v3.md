# 导出指令 v3：全部在 A100 上重新推理，不再复现旧缓存

日期：2026-10-24。替代 v1 第 2、3 节和 v2 复现门槛。不训练，不做换角统计，不访问 C4/C5，不改历史目录。继续用分支 `codex/eg-firmness-export-20261024`，新建子目录 `v3/`，已导出的 50 个候选特征作废、不使用。

## 为什么改

第二帧 `cal_0067_0064` 上候选中心都变了，620 个种子视角翻转。原因是：可抓取点的筛选阈值附近只要有一两个点在两种显卡上判得不一样，最远点采样就会整体选出另一批种子。所以在另一张显卡上复现 RTX 4090 的旧缓存原理上不可靠，第一帧对上只是碰巧。

解决办法：**特征和标签必须来自同一次前向。** cal、C3、P2 也像小头训练帧一样，在 A100 上重新推理、重新生成候选、重新做官方评价。同一张 A100 连跑两次已经确认逐位一致。

## 帧

| split | 场景 | 帧 |
|---|---|---|
| probe | 训练 70 + 验证 10（场景表同 v1） | 0, 32, …, 224 |
| cal | 67, 4, 50, 56, 14, 65, 62, 30, 45, 1, 96, 71 | 0, 64, 128, 192 |
| C3 | 100, 106, 115, 124, 132, 139, 150, 157, 160, 169, 176, 186 | 0, 64, 128, 192 |
| P2 | 同 C3 场景 | 32, 96, 160, 224 |

共 784 帧。

## 每帧步骤（所有 split 完全相同）

1. **推理 + 特征：** 照 N0 `infer_frozen.py`（种子 = 20261024 + s×1000 + f，同样的数据集构造、断言、`cd_mask`），同一次前向用 v1 的两个钩子取 `vp`、`ang`，`logit` 取 `grasp_angle_pred` 前 12 类。缓存存 `v3/cache/{split}_{s:04d}_{f:04d}.npz`。
2. **候选：** `owner = direct.owners(raw)`，`_, _, ids = E.postprocess(raw, cd_mask, owner)`，`nom = angle_index[ids]`（与 N0 train/P2 分支相同）。
3. **12 个角度变体：** 原角度一格逐位保持，其余 `R @ Rx((k − nom)·π/12)`。
4. **官方评价：** `evaluate_p2.py` 的评价函数，不改（collision、empty、μ、I1、原因字段）。
5. **EG 碰撞比例 R1（风险项要用）：** 对全部变体，照 N0 `pipeline.py` 的 `scores()` 里的做法：`ModelFreeCollisionDetector(cloud[depth>0], voxel_size=0.005)`（整幅深度点云，不用 GT 分割），`detect(..., approach_dist=0.05, collision_thresh=0.01, return_ious=True)`，取第 0 个比例，即 N0 的 `R1`。
6. **输出：**
   - `v3/eval/{split}_{s:04d}_{f:04d}.npz`：candidate_id、original_angle、object_id、mu、collision、empty、I1、R1（均为 n×12，前两者 n）、unknown_reason、contact_reason、pred；
   - `v3/features/{split}_{s:04d}_{f:04d}.npz`：candidate_id、original_angle、vp、ang、logit（只含 `ids`）。

## 核验（任何一条不满足就停）

- 先在 C3 `100_0000` 上连跑两次第 1 步，`pred`、`vp`、`ang`、`logit` 逐位一致；
- 每帧：特征文件与评价文件的 candidate_id、original_angle 完全一致；`logit[:, ].argmax(1) == original_angle`；
- 每帧：变体原角度格的 collision/empty 与用缓存 pred 直接评价的结果一致（即第 4 步对原姿态的结果）。

## 交付

推送后报告提交号和 `v3/summary.json`，只含：各 split 帧数、候选数、变体数、原角度干净候选数、其中原可行/原弱数量、μ 未知数、I1 缺失数、碰撞或空夹变体比例、两次运行一致性结果、GPU 型号和总耗时。**不报告任何换角结果，不训练小头。**

参考成本：GPU 约 784 帧 × 2 秒 ≈ 30 分钟；官方评价约 37 万个变体，12 进程约 30 分钟。
