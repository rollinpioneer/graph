# P2B Graph-PPO utility

BASE: 577db67c64343850965fa4e0f6bc7cac0c15674d
algorithm: SB3 PPO 2.7.1
gymnasium: 1.2.2
new_on_policy_training: true
benchmark: ONLINE_RL_ACTION_CONDITIONED_ABSTRACT_SKILL_ENV_NOT_PHYSICS
training_jobs: 48/48
train_environment_steps_per_job: 524288
P2A_demo_reused_for_training: false
teacher_takeover: 0
P1_frozen_sources_modified: false
policy_utility_evidence: RL_REWARD_UTILITY_NOT_ESTABLISHED_OR_MIXED
robot_policy_gain_claimed: false
physical_cycle_claim: NOT_EVALUATED
historical_confirmation_fields_modified: false

This is on-policy PPO. Policies selected actions in training and in deterministic test. Data are new family streams, not P2A demonstrations.
All methods share PPO hyperparameters, 31-d observations, and same-seed actor/critic initialization. Only Phi source changes.
Finite task deadline maps to terminated with effective terminal Phi=0; buffer cuts are not episode ends.

## Test success (episode-macro)
- TASK_ONLY: 0.6422  steps=54.1  invalid=28.9  restore|loss=0.8943378346363421
- COUNT_PBRS: 0.6358  steps=54.9  invalid=26.5  restore|loss=0.9607329842931938
- COUNT_EVENTS_PBRS: 0.5295  steps=61.7  invalid=43.2  restore|loss=0.974466760080959
- GEOM_COUNT_EVENTS_PBRS: 0.8250  steps=43.1  invalid=15.3  restore|loss=0.968872741555381
- GRAPH_COST_PBRS: 0.8461  steps=43.0  invalid=11.5  restore|loss=0.9595125786163522
- GRAPH_FULL_PBRS: 0.8447  steps=40.8  invalid=13.4  restore|loss=0.9740501370936153

## Primary paired comparisons (GRAPH_FULL minus baseline, crossed seed x family, 97.5% CI)
- vs TASK_ONLY: mean=0.2026  [-0.0644, 0.5427]
- vs GEOM_COUNT_EVENTS_PBRS: mean=0.0198  [-0.0894, 0.1302]
Win rule requires mean>=3pp and CI lower bound>0. Seed-level variance keeps the TASK interval crossing 0, so the preregistered flag is mixed even though the point estimate vs sparse task reward is large.

## Leave-one-seed-out (seed-macro)
- leave 211: GRAPH-TASK=0.102  GRAPH-GEOM=-0.005
- leave 223: GRAPH-TASK=0.227  GRAPH-GEOM=0.036
- leave 227: GRAPH-TASK=0.237  GRAPH-GEOM=0.039
- leave 229: GRAPH-TASK=0.228  GRAPH-GEOM=0.020
- leave 233: GRAPH-TASK=0.221  GRAPH-GEOM=0.006
- leave 239: GRAPH-TASK=0.119  GRAPH-GEOM=-0.006
- leave 241: GRAPH-TASK=0.267  GRAPH-GEOM=0.048
- leave 251: GRAPH-TASK=0.220  GRAPH-GEOM=0.020

## Learning
- TASK_ONLY mean normalized AUC: 0.3438
- COUNT_PBRS mean normalized AUC: 0.3545
- COUNT_EVENTS_PBRS mean normalized AUC: 0.2902
- GEOM_COUNT_EVENTS_PBRS mean normalized AUC: 0.6011
- GRAPH_COST_PBRS mean normalized AUC: 0.4591
- GRAPH_FULL_PBRS mean normalized AUC: 0.5611

## Limits
- Abstract skill MDP, not physics/robot/vision.
- One training-family draw; 8 policy seeds. Crossed bootstrap CIs are wide because seed variance is large.
- COUNT_EVENTS_PBRS underperformed TASK_ONLY; dense non-graph GEOM_COUNT_EVENTS is the relevant strong baseline.
- GRAPH_FULL and GRAPH_COST are close; geometric graph term is not a large extra increment here.
- Discounted PBRS identity held on test (summarize abort-on-fail). Label-only LOST->RECOVERING can have nonzero discounted shaping; that is not P1 undiscounted credit.
