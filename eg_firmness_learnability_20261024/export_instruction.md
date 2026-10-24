# 实验指令：为"牢度能否学准"导出特征与标签（不训练）

日期：2026-10-24。对应预注册：`eg_firmness_learnability_prereg_20261024.md`（分支 `claude/project-thread-1bid1z`）。
**本指令只做数据导出：** 冻结 EG 前向、官方评价。不训练任何网络，不做换角选择或统计，不访问 C4/C5，不修改历史目录。新目录 `eg_firmness_export_20261024/`，推到新分支 `codex/eg-firmness-export-20261024`。

## 0 通用约定

- 模型：官方 EG Kinect epoch-10，`economicgrasp_kinect.tar`，SHA256 `a47b717a…5fc5804a`，`is_training=False`，`net.eval()`，TF32 开启（与 N0 `infer_frozen.py` 相同）。
- 特征用前向钩子取，不改模型代码：
  - `vp`：`net.grasp_head` 的输入（forward_pre_hook），形状 (1, 256, 1024)，存为 (1024, 256)；
  - `ang`：`net.grasp_head.global_interaction_module` 的输出（forward_hook），形状 (1024, 4, 64)，取 `[:, 0, :]`，存为 (1024, 64)；
  - `logit`：`end_points['grasp_angle_pred']` 的前 12 类，存为 (1024, 12)。
- 候选编号 `candidate_id` 就是 1024 个种子点的下标（与 N0、C3 现有文件一致）。
- 每个特征文件只保存该帧"候选"的行，顺序与对应评价文件 / per_pose 文件的 `candidate_id` 完全一致。

特征文件格式 `features/{split}_{s:04d}_{f:04d}.npz`，split ∈ {probe, cal, C3, P2}：

| 字段 | 形状 | 说明 |
|---|---|---|
| candidate_id | (n,) | |
| original_angle | (n,) | EG 原角度类别 |
| vp | (n,256) float32 | |
| ang | (n,64) float32 | |
| logit | (n,12) float32 | |

## 1 小头训练帧（新推理 + 新评价）

**场景（已按 rng=20261024 固定，直接使用）：**
- 训练 70 个：0, 3, 5, 7, 8, 10, 11, 13, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 27, 28, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 66, 70, 72, 73, 74, 75, 78, 79, 80, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 97, 98, 99
- 验证 10 个：6, 12, 23, 26, 29, 32, 69, 77, 81, 95
- 每个场景 8 帧：0, 32, 64, 96, 128, 160, 192, 224。共 640 帧。

**步骤（每帧）：**
1. EG 推理：完全照 N0 `infer_frozen.py`（种子 = 20261024 + s×1000 + f，同样的数据集构造、同样的断言和 `cd_mask`），同一次前向里用钩子取特征。缓存存到 `eg_firmness_export_20261024/cache/probe_{s}_{f}.npz`（字段同 N0 缓存）。
2. 候选：照 N0 `pipeline.py` 的 train 分支：`owner = direct.owners(raw)`，`_, _, ids = E.postprocess(raw, cd_mask, owner)`，`nom = angle_index[ids]`。
3. 12 个角度变体：原角度那一格保持原样（逐位相同），其余为 `R @ Rx((k − nom)·π/12)`，与 N0 相同。
4. 官方评价：用 `eg_p2_i1_mu_20261024/evaluate_p2.py` 里的评价函数，不改一行（碰撞、空夹、μ、I1、原因字段）。
5. 输出：
   - `probe_eval/{s:04d}_{f:04d}.npz`：格式同 P2 导出（candidate_id、original_angle、object_id、mu、collision、empty、I1、unknown_reason、contact_reason、pred）；
   - `features/probe_{s:04d}_{f:04d}.npz`：只含 `ids` 这些候选。

## 2 λ 标定帧（N0 的 48 个训练帧）

- 评价：对 `eg_n0_nref_main_20261024/cpu/train_*.npz` 的 `pred`，用同一评价函数算 μ、I1，做法与 P2 导出完全相同（含 C3 回放已通过的那份代码）。逐项核对 candidate_id、original_angle、collision、empty 与 `per_pose/train_*.npz` 一致，不一致就停。输出 `cal_eval/{s:04d}_{f:04d}.npz`。
- 特征：从 N0 缓存 `eg_n0_nref_main_20261024/cache/train_*.npz` 的 `input_points` 重建输入（`coordinates_for_voxel = input_points / 0.005`），重新前向取特征。**复现核对：** 重算的 `pred` 与缓存 `pred` 最大绝对误差 ≤ 1e-5，`angle_index` 完全一致；不满足就停下告诉我。输出 `features/cal_{s:04d}_{f:04d}.npz`，候选与 `per_pose/train_*.npz` 一致。

## 3 测试帧特征（C3 与 P2）

- P2：从 N0 缓存 `cache/P2_*.npz` 的 `input_points` 重建并前向，同样做复现核对；候选与 `per_pose/P2_*.npz` 一致。
- C3：用 `eg_pose_quality_diagnosis_v1_20261022/cache/C3_*.npz`。若其中有 `input_points`，按同样方式重建；若没有，先停下告诉我缓存里有哪些字段。复现核对同上；另外要求导出的 `logit` 与 `eg_i1_angle_physics_20261024/evaluation/*.npz` 的 `angle_logits[:12].T` 最大误差 ≤ 1e-4。候选与 C3 的 I1 文件一致（1,229 个）。
- 测试帧只导出特征，不做任何评价或统计（μ、I1 已有）。

## 4 交付

推送 `codex/eg-firmness-export-20261024`，目录 `eg_firmness_export_20261024/` 下含 `features/`、`probe_eval/`、`cal_eval/`、每帧收据、复现代码，以及 `summary.json`，只报告：
- 各 split 的帧数、候选数、变体数；
- 小头训练帧中碰撞或空夹变体的比例、μ 未知数、I1 缺失数；
- 全部复现核对结果（pred 最大误差、angle_index 一致、C3 logit 最大误差、cal 的碰撞/空夹一致）；
- GPU 型号与总推理时间。

**不要训练小头，不要计算任何换角结果。** 小头训练和判定由冻结代码 `firmness_probe.py` 在另一端运行。

参考成本：GPU 推理约 800 帧，约 20 分钟；官方评价约 32 万个变体，按 C3 的速度约 3 核·时，12 进程并行约 20 分钟。
