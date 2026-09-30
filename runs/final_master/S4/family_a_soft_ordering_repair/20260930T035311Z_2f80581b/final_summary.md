Family A soft-ordering MVP status: NOT_ESTABLISHED
分支: codex/cp-disr-s4-family-a-soft-ordering-repair
Gate A commit: ab2ca2c57934a0fe3f9e3eec4c5447ae7d1b7556
Gate M commit: PENDING_UNTIL_GATE_M_COMMIT
证据目录: /home/xushijie2/graph_cp_disr_s4_soft_ordering_repair/runs/final_master/S4/family_a_soft_ordering_repair/20260930T035311Z_2f80581b

Configs: SO_TARGET_FIRST, SO_SECOND_FIRST, SO_NEUTRAL
Branch attempts: 12 / 12
Resets: 12 / 12

SO_TARGET_FIRST:
  target-first repeat times: [10.100000, 10.100000]
  second-first repeat times: [10.750000, 10.750000]
  mean ratio: 0.939535
  relative delta: 0.064356
  absolute delta: 0.650000
  criterion met: False
  criterion: both repeats target_first < second_first; mean ratio <= 0.90; mean difference >= 0.75 s

SO_SECOND_FIRST:
  target-first repeat times: [10.850000, 10.850000]
  second-first repeat times: [10.100000, 10.100000]
  mean ratio: 1.074257
  relative delta: 0.074257
  absolute delta: 0.750000
  criterion met: False
  criterion: both repeats second_first < target_first; mean ratio <= 0.90; mean difference >= 0.75 s

SO_NEUTRAL:
  target-first repeat times: []
  second-first repeat times: []
  mean ratio: None
  relative delta: None
  absolute delta: None
  criterion met: None
  criterion: None

Both routes reachable: True
Route lengths equal: True
Skill multisets equal: True
B_PLAN nominal costs equal: True
Canonical first action by config: a:PICK:second_object:v1 (identical across configs; tie-break reads no geometry)

Engineering failures: none
Perception/verifier failures: none
Planner errors: ['10e9bbdf69fceb62', '08270030c1fc0015', '7ec953c0edd5f7d8', 'e247427cee594937']
Controller failures: none

Workers: one worker per GPU, gpus=[1, 3]
Max concurrency: 2
Aggregate throughput: 0.17434963 branches/s over 68.827 s
Single-worker baseline: NOT_MEASURED
Speedup claim: NONE (no matched single-worker baseline; no extra attempts spent)

Mechanism feasibility: NOT_ESTABLISHED
Family role: NOT_ESTABLISHED
Provider authorized: False
Representation authorized: False
S2 authorized: False
Next action: EXPLICIT_RESEARCH_DECISION

New provider calls: 0
New representation forwards: 0
New RL: 0
New optimizer: 0
New elastic: 0
New formal test: 0

Old evidence unchanged: True
Tests: see verify.json
Worktree clean: PENDING_UNTIL_GATE_M_COMMIT
SSH closed: pending final session exit

主要限制: controlled mechanism validation only; single seed pair per config; wall time never used; no provider/representation/RL/optimizer/elastic/S2/S3; not a dataset; cannot alone support the CP-DISR headline.
下一次需要用户明确授权的事项: provider baseline qualification (VLM-only, B_PLAN+R, A_STAT, Full comparisons) and any S2/S3 work.

technical_wave: PASS  remaining_branches_released: True
paired_restore_verified: 6/6
mechanism_gate_utc: 2026-09-30T04:44:17Z
