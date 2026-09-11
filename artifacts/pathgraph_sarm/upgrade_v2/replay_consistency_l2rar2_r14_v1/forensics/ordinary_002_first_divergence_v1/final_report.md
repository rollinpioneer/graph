# R14 ordinary_002 first-divergence forensic review

## 1. 输入和完整性

本审查只读取已完成的 ordinary_002、旧 cache、授权记录、Git 来源和环境记录。输入清单共 61 个文件，重新计算后 `61/61` 一致；`$EXEC` 与 `$CACHE` 未被修改。

## 2. 授权已消费状态

授权为 R14 ordinary replay 单实例，nonce `e8027aed85c2233b173af332524b288339ddcb2866d0626d`，`authorized_instances=1`，`automatic_retry=false`，消费状态 `CONSUMED_BEFORE_MODEL_CONSTRUCTION`。授权完整性结果为 `PASS_CONSUMED_SINGLE_USE`。

## 3. callback 采样点映射

ordinary state、callback capture 与 cache oracle 均为 37 行，按 sequence/capture_order 一一对应，action、action_index、phase 和时间均在 `1e-12` 门内匹配。

## 4. 首差异的精确位置

首差异不是原报告中的 action-end 才出现，而是在 `lift` 的第一个 `control_tick` callback：capture 12，action index 4，时间 `0.5000000000000002 s`。object/relative L2 为 `1.2249276643646983e-05 m`，gripper L2 为 `0`，weld state 匹配。锁定 geometry tolerance 为 `1e-12 m`。

## 5. 差异随时间的形态

分类为 `PERSISTENT_VARIABLE_OFFSET`。首差异之后有 25 个 callback，object L2 最小/中位数/最大值为 `3.291920575227912e-06` / `1.2249276643650874e-05` / `1.2249352439119271e-05 m`；relative L2 最小/中位数/最大值为 `3.291920575227912e-06` / `1.2249276643650874e-05` / `1.2249352439119271e-05 m`。所有后续值均超过容差；轴符号和 action/phase 边界统计保存在 `divergence_pattern.json`。12.249 微米差异在任务尺度上很小，但在锁定的 exact-replay contract 上是失败。

## 6. attach→lift 状态矩阵

已抽取 close_gripper callback/return、lift before_action、每个 lift control callback、lift action_end callback/return，并保存 qpos、qvel、qacc_warmstart、mocap、eq_active、eq_data、RNG、contacts、ncon、几何和 weld/attached 状态。三个 ordinary 内部边界比较均 PASS。

## 7. attach relpose 一致性

cache 与 ordinary close-gripper world-relative geometry 的 L2 为 `0.0`；ordinary local relative position 与 `model.eq_data` relpose 的 L2 为 `0.0`。ordinary gripper quaternion 为单位四元数。cache 缺少 quaternion、qvel、qacc_warmstart 和 eq_data，因此不能据此排除 hidden-state 差异。

## 8. 源码来源是否可恢复

Round-9 generation lock 指向 `f5e16da903bb7d64a26eac6163d41030a2fbcee7`，但该 commit 不包含 `repair_collection.py` 与 `repaired_simulator.py`。后续 commit 中文件存在不能证明当时未提交的源码相同，因此旧 cache 的精确执行源码树不可恢复。

## 9. 历史环境是否可恢复

R14 result 记录 Python 3.10.19、NumPy 2.2.6、MuJoCo 3.4.0、OpenCV 4.13.0 和平台；但 Round-9 collection 的实际运行时、MUJOCO_GL、renderer backend、warmstart source、RNG source 与 model XML hash 未被证明，状态为 `HISTORICAL_COLLECTION_RUNTIME_NOT_PROVEN`。

## 10. 假设—证据表

H1/H2/H8 标为 SUPPORTED，H3/H6 标为 REFUTED，H4/H5/H7/H9 保持 UNKNOWN；每条记录的 supporting/contradicting/missing evidence 与 JSON key 已写入 `hypothesis_evidence_matrix.csv`。

## 11. A/B/C 路线决定

决定为 `OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE`，采用 Route B：另建可复现 baseline。不是放宽容差，也不是把旧门改判通过。

## 12. 本轮零物理执行

本轮 physical executions 为 0；审查工具未 import MuJoCo、未构造 model/data/renderer、未调用 mj_step/mj_forward、未生成新 RGB。

## 13. 禁止主张

不得主张 LOSS_MECHANISM_CONFIRMED、PASS_BY_SMALL_PHYSICAL_ERROR、PASS_AFTER_TOLERANCE_RELAXATION、R14 instrumented 已授权或 L3 已进入。当前 selected candidate 为 `null`，confirmation 为 `false`。

## 14. 下一次授权是否应准备

只生成了 `DRAFT_NOT_AUTHORIZED` 的 Route B 草案，requested/authorized instances 均为 0。若未来准备新 baseline，必须先取得新的 human authorization、single-use nonce、完整 source/runtime/renderer/model provenance 和新 output root；本轮不能继续使用旧 nonce。
