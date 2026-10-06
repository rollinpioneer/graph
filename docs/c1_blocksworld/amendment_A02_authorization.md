# CP-DISR-C1-BW-IMITATION-A02 (user decision, recorded before any imitation training)

Recorded states:

```text
AMENDMENT_A02_ACCEPTED
PPO_RESULT_STATE=PPO_ID_GATE_FAIL
NEW_TRAINING_PROTOCOL=UNIFORM_PLANNER_IMITATION
AUTHORIZED_TRAINING_RUNS=3
FORMAL_EVAL_CONDITION=ALL_THREE_PASS_IMITATION_ID_GATE
```

Baseline: repository `rollinpioneer/graph`, branch `codex/cp-disr-c1-blocksworld-main-v1`, base commit `72c0fd2cdc4394dcdee3b5bd6bba210060535d57` (PPO result commit, state `PPO_ID_GATE_FAIL`).
Specification: `CP_DISR_C1_Blocksworld_Uniform_Planner_Imitation_Amendment_A02.md` (this directory; it governs).

User decision text (verbatim, 2026-10-06):

> 批准执行：CP-DISR-C1-BW-IMITATION-A02
> 选择方案 A。保留当前 PPO_ID_GATE_FAIL，不覆盖、不退款、不选择 ASNET 的 N=32768 checkpoint。
> 从提交：72c0fd2cdc4394dcdee3b5bd6bba210060535d57
> 在分支：codex/cp-disr-c1-blocksworld-main-v1
> 继续执行三方法统一规划器模仿训练：
> R-C1-BW-IL-B2-0
> R-C1-BW-IL-QMARK-0
> R-C1-BW-IL-ASNET-0
> 三种方法必须从头初始化，使用相同的 planner-labeled D0/D1/D2、完整最优动作集合、相同损失、相同两轮共享状态聚合和固定训练周期。
> 保持原 ID gate：success >= 33/36，decision_perfect >= 33/36。
> 只有三种方法全部通过，才允许一次性执行 A0/A1/A2/B 和规划器基线。
> 任一失败立即停止，不加轮次、不延长训练、不挑 checkpoint、不增加 seed、不修改单一方法、不打开正式评估集。
> 执行以 CP_DISR_C1_Blocksworld_Uniform_Planner_Imitation_Amendment_A02.md 为准。

Implementation files frozen by the prep commit: `src/cp_disr/blocksworld/imitation.py` (trainer, datasets, guards), `src/cp_disr/blocksworld/imitation_launch.py` (registration, ledger, launch),
`scripts/c1_bw_il.py` (prep | register | train | status), `scripts/c1_bw_il_eval.py` (imitation ID gate and, only after all three pass, the one-shot formal evaluation),
`tests/test_c1_bw_imitation.py`, `configs/c1_bw_imitation_a02.json` (frozen training config, D0 hash, hashes of every reused file vs the freeze commit `490a3b11`).
The PPO result root `runs/final_master/c1_route_b/blocksworld_main_v1/20261006T015313Z_490a3b11/` is not modified; imitation results go to `runs/final_master/c1_route_b/blocksworld_main_v1/imitation_a02/`.
