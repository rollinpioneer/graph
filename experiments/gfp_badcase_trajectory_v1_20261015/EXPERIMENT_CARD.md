# GFP-BADCASE-TRAJECTORY-V1
Date: 2026-10-15. Status: APPROVED, frozen-weight diagnostic only.
Branch: codex/gfp-badcase-trajectory-v1-20261015
Branch URL: https://github.com/rollinpioneer/graph/tree/codex/gfp-badcase-trajectory-v1-20261015
Parent experiment commit: 96656bacfef91984abddc95de522ca6907cf0ea3
Official GFP: 1abbd5f0e287caf9bda70e7bc655d0a499ac8401
Task: humanoidmaze-medium-navigate-singletask-task1-v0
Source root: /home/xushijie3/work/flow_critic_comparative_train_v1_20261014
Output root: /home/xushijie3/work/gfp_badcase_trajectory_v1_20261015

## Authorized scope
Existing original GFP weights only. Zero optimizer updates, no training or online learning.
At most 48 new complete episodes, at most 96000 environment steps; rendering saved states adds no environment steps.
Selection is frozen in configs/selection.json before rollout: 24 1M failures, 12 1M successes, 6 earlier successful checkpoint controls, 6 alternate policy RNG controls.
First eight: two failures and two successes per training seed. Continue only if initial observation SHA256, success, return, length and terminal flags exactly reproduce original matched_verified.
All later original-seed replays must also match. Any mismatch/nonfinite/corrupted checkpoint stops recording, preserves evidence, no automatic retry.
Earlier controls are individual episodes, not six scenes times two checkpoints. The available chosen set covers 800k and 900k; seed1 earlier selections happen to use 900k.
No extra episodes to manufacture missing evidence. No change to task rules or source assets.

## Selection and interpretation
Outcome-stratified selection over prior 800k/900k success pattern, deterministic SHA256 tie ordering. Success pools prioritize same-scene failures in other model seed.
All identifiers and original reference records in selection.json. Action RNG alternatives add 1000000 to original policy seed; all reset/global RNG seeds unchanged.
This is deliberate case enrichment, not a prevalence sample. Population 1M success remains 72/100 and 82/100. Do not infer failure percentages from 48.
Behavior categories are discovered after capture, not claimed in advance. Similar initial-state successes and earlier checkpoint successes are controls.
First visible path/pose divergence is descriptive, not a verified causal onset. Actor, VaBC, Critic, Guidance and dataset explanations remain hypotheses unless directly measured.
No new internal-module intervention or counterfactual rollout authorized in this card.

## Recording and verification
Per-step observations, qpos/qvel, full MuJoCo integration state, raw/clipped policy actions, policy keys, reward/success/terminal flags, target and robot positions, torso orientation/head height, contact geom pairs and forces.
Observe contacts after each control step; substep-only contacts may be missed. No built-in fall termination in this environment; posture thresholds are descriptive proxies.
Read checkpoint/model/optimizer/Agent RNG state identity before/after. Source checkpoint hashes unchanged. Exact matched seed/reset protocol and temperature=0 still stochastic Gaussian sampling.
Use original installed environment and dependencies; no package upgrades. GPU 0 selected with 38 GiB free and no own-account compute process at preflight; share other users, no waiting for exclusivity.
Resource cost is inference/recording/rendering only, not training. Maximum 8 representative videos, generated from saved simulator states without policy rollout.
Full numerical trajectories/videos remain server-side; source identities, selection, scripts, metrics, plots and receipts are lightweight Git artifacts.

## Deliverables
Frozen selection and asset identity; replay gate receipt; full trajectory/checkpoint index and hashes; case metrics and matched comparisons; approximately 5 failed/3 successful representative videos; concise observational report with counts, onset evidence, repeatability and explicit limits.
After necessary verification, archive only explicit lightweight file list on this branch, push normally and verify SHA; sync local handoff.
Stop after this diagnostic. No new seeds/tasks/training or automatic expanded diagnostic.

