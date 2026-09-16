# P2A weighted-BC utility

BASE: 8e86b246e900e1e12dbb21c88b95046a06c8e4c5
benchmark: ACTION_CONDITIONED_ABSTRACT_SKILL_MDP_NOT_PHYSICS
execution_status: COMPLETE
policy_utility_evidence: UTILITY_NOT_ESTABLISHED_OR_MIXED
robot_policy_gain_claimed: false
physical_cycle_claim: NOT_EVALUATED
confirmation_passed: false
P1 historical confirmation fields unchanged

## Training
train episodes: 1920 smoke_only=False
weight transform: POSITIVE_PROGRESS_FLOOR_V1 rows=50792 max_identity_error=4.440892098500626e-16
jobs recorded: 45
init/batch parity: PASS
checkpoint_rule: FIXED_LAST_STEP

## Method family-macro success
- BC_UNIFORM: 0.9141
- V6_WEIGHT_SHUFFLED: 0.9035
- LINEAR_B_FIRST_R1: 0.8953
- SPARSE_TERMINAL: 0.8859
- LINEAR_A_FIRST_R1: 0.8781
- V6_CAP_POTENTIAL: 0.8672
- UNORDERED_VALID_COUNT: 0.8594
- VALID_COUNT_PLUS_MATCHED_EVENTS_V1: 0.8555
- V6_CAP_COST_ONLY: 0.8516

## Primary paired comparisons (V6 minus baseline)
- vs BC_UNIFORM: mean=-0.046875 low=-0.1234375 high=0.0265625 unit=crossed_policy_seed_and_eval_family
- vs VALID_COUNT_PLUS_MATCHED_EVENTS_V1: mean=0.01171875 low=-0.03515625 high=0.06289306640625014 unit=crossed_policy_seed_and_eval_family

## Exploratory
- vs V6_CAP_COST_ONLY: mean=0.015625 [0.0, 0.039082031250001134]
- vs V6_WEIGHT_SHUFFLED: mean=-0.036328125 [-0.1203125, 0.03828613281250029]
- vs UNORDERED_VALID_COUNT: mean=0.0078125 [-0.0359375, 0.056269531250001135]

Learned policy actually controlled env: true; teacher takeover calls: 0.
This is a reference abstract-skill MDP, not robot policy performance.
