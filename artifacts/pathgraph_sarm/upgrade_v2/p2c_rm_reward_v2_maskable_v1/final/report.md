# PathGraph P2C-RM V1 report

Git
---
base: ce148ce160df41ec0f8876faf3846a6538954dff
candidate commit: 7a69b8416c41d27afc1dd46740a18d401d721454
result commit: (written before result commit)
branch: research/pathgraph-p2c-rw-maskable-pretrain-v1
main unchanged: True
origin main: 234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2

Execution boundary
------------------
formal training jobs: 0
smoke jobs: 4
smoke timesteps/job: 512
physical executions: 0
model API calls: 0

Reward V2
---------
cost parity states: 913968
cost mismatches: 0
Bellman mismatches: 0
PBRS max error: 1.4432899320127035e-15
cache mismatches: 0
stage-handoff regressions: 0
development V1 hit: 0.8567901234567902
development V2 hit: 1.0
confirmation V1 hit: 0.8427762039660056
confirmation V2 hit: 1.0
per-motif deltas (dev): {"ALTERNATIVE_COST": {"n": 1424, "v1_hit": 0.9943820224719101, "v1_regret": 5.132119367885628e-05, "v2_hit": 1.0, "v2_regret": 0.0}, "INVALIDATION_RECOVERY": {"n": 1664, "v1_hit": 0.5384615384615384, "v1_regret": 0.007691526452271137, "v2_hit": 1.0, "v2_regret": 0.0}, "PRECEDENCE": {"n": 1408, "v1_hit": 1.0, "v1_regret": 0.0, "v2_hit": 1.0, "v2_regret": 0.0}, "SHARED_PREREQUISITE": {"n": 1984, "v1_hit": 0.9233870967741935, "v1_regret": 0.0006848943071931935, "v2_hit": 1.0, "v2_regret": 0.0}}
mean regret change dev: v1=0.0021960820527545033 v2=0.0

Mask
----
{
  "actions_checked": 20439614,
  "all_false_states": 0,
  "evaluation_mask_used": true,
  "invalid_wait_transition_mismatches": 0,
  "legal_mask_mismatches": 0,
  "method_dependence_violations": 0,
  "oracle_optimal_set_fully_masked_states": 0,
  "paired_state_mask_mismatches": 0,
  "passed": true,
  "schema": "P2C_RM_MASK_AUDIT_V1",
  "states_checked": 552422,
  "training_mask_used": true,
  "wait_illegal_states": 0
}

Smoke
-----
{
  "invalid": 0,
  "jobs": 4,
  "nonfinite": 0,
  "parity": true,
  "status": "SMOKE_PASS_NOT_RESEARCH_RESULT"
}

Decision
--------
engineering status: PASS
reward status: PASS
mask status: PASS
scientific route: READY_FOR_P2C_RL_PREREGISTRATION
training_release: false
policy utility evidence: NOT_EVALUATED_IN_THIS_ROUND

Historical state
----------------
P1 unchanged: true
P2A unchanged: true
P2B unchanged: true
D1 unchanged: true
P2C-Q unchanged: true
confirmation_passed: false
