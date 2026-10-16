# GFP-BFC-DEV-SCREEN-V1 — approved first round

Date: 2026-10-16. Authorization: user explicitly requested execution of the first round. This card freezes only stage R of GFP_BFC_METHOD_DESIGN_V1_20261016.md.

Branch: `codex/gfp-bfc-dev-screen-v1-20261016`

Repository: https://github.com/rollinpioneer/graph/tree/codex/gfp-bfc-dev-screen-v1-20261016

Remote root: `/home/xushijie3/work/gfp_bfc_dev_screen_v1_20261016`.

## Scope and identity

Fork the formal `G_task1_s0` 600k and `G_task1_s3` 300k weights in `/home/xushijie3/work/gfp_eta0_ablation_20261015`. These are not the earlier G0 task1 comparison runs. Each fork runs R0–R5 for exactly 100,000 additional offline updates. Total maximum 1,200,000. Online updates zero. No extra seeds, retries, normal-window tests, from-scratch training, task4 training or modules are authorized by this round.

Reuse the existing fixed-source snapshot, its source hash manifest, original environment and dataset loader. The declared upstream GFP commit is `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`; actual file hashes and source checkpoint hashes are recorded in receipts/assets.json. The A–D evidence archive is commit `0a95fc4914876eb011e4707d001f8ab2141c5b3f`.

| Arm | Reference Actor tau | TD increment bound |
|---|---:|---:|
| R0 | no reference, unchanged original agent | original target |
| R1 | .005 | 0 |
| R2 | .0005 | 0 |
| R3 | .00005 | 0 |
| R4 | .0005 | .1 × processed training reward RMS |
| R5 | .0005 | .5 × processed training reward RMS |

`protocol.json` freezes the actual numeric reward RMS/bounds before any branch update. No validation rewards or changing Q magnitudes enter this scale.

## Implementation contract

Only the Actor contribution to the critic TD target changes. Current/reference Actors share the original sample RNG and target critic. Preserve mean ensemble and mean Actor/VaBC aggregation, discount .995, target_use_next=false (VaBC generated at current state and scored at next state), guidance min, temperature .001 and all actor/VaBC losses.

Reference is outside the trainable/Adam parameter tree. Initialize it from the source Actor; after each logical update, EMA toward the newly updated Actor. Four fused updates perform four reference EMAs. Preserve upstream target-critic EMA using the pre-update critic. Do not correct that timing as part of this experiment.

Original TrainState.step starts at 1: source step 600001/300001 corresponds to Adam count 600000/300000. Both counts are checked. Source checkpoints lack historical NumPy/Python sidecars. This is paired fork continuation using restored model, Adam and Agent RNG, not a claim of bitwise replay of the historical run. Sampling is the original GPU/JAX fused4 path.

First four genuine updates of each branch count toward its 100k allocation. Save and reload full state, including reference, Adam, Agent RNG, NumPy/Python states and configuration identity, before continuation. No extra optimizer updates are performed for admission.

## Measurements and evaluation

At fork and every 5k: reuse the A–D 50k-state probe and fixed action noises, online/target values, start-region minus overall value, action-gradient norm/cosine at fork actions, action shift, original losses, BFC delta and clipped fraction. Start region is the four original grid cells (1,1),(2,1),(1,2),(2,2).

A post-fork probe is anomalous when value-gap increase is >=15 AND start action-gradient norm is >=2 times its fork value. Report all 20 points and longest consecutive run. Zero points=avoided; >=2 consecutive=not avoided; other isolated anomalies=borderline. This is a development diagnostic, not a universal numerical instability criterion.

At +50k and +100k save full checkpoints and evaluate the online Actor for exactly 100 episodes, 10 parallel environments, 2000-step episode limit. Total 2400 episodes. Reset seeds 4242000+episode; global NumPy/Python seeded to the same value immediately before each reset. Policy key 4242 folded by ten-episode round and split by global round timestep, identical across arms/windows/checkpoints. Check initial-observation hashes for all matched comparisons. This is a newly frozen matched protocol; legacy B1a evaluation is diagnostic context, not an interchangeable matched result.

Inactive environments are not stepped/reset during the remaining active episodes. Log return, success, length, final start-region membership and sparse xy path. Count external steps, MuJoCo do_simulation control calls (including reset), and internal physics substeps separately. External maximum 4.8M. Control-call budget is 5M (24 evaluation jobs each <=208333); physics substeps are additionally reported, not mislabeled as control calls. No videos.

## Resources and stops

No hour cap; record measured time and GPU placement, CPU time, peak resident memory and resource snapshots. At most six simultaneous jobs, two per GPU, two evaluation jobs; require at least 18,000 MiB free VRAM at dispatch to accommodate the inherited 50k-state probes. Share available cards without waiting for exclusive access. Resource snapshots are shared-device usage, not an attribution of other users' memory to this experiment.

Stop an affected branch on non-finite losses/probes, invalid update count, serialization/hash failure, environment budget or protocol violation. Preserve logs and actual process exit code. No automatic restart. Poor finite performance or a joint-anomaly event is a result, not a reason to stop or tune. Checkpoint evaluation failures also remain failed without silent reruns.

## Interpretation and delivery

Compare original R0, all three EMA settings and both BFC settings in both development windows. Report each point and window, paired episode differences, anomaly duration, clipping activation, online Actor performance and actual cost. BFC is worth considering further only if it suppresses the feedback while retaining capability and improves meaningfully over the best EMA; the design's exploratory criterion is >=10 percentage-point mean advantage without >10-point loss in either window. If EMA suffices, prefer EMA. Two selected forks are not independent broad algorithm validation. Normal-window, from-scratch, longer-horizon and task4 tests need separate authorization.

Archive source additions, card, frozen config, hashes, lightweight diagnostics and receipts only. Weights, processed data and full logs remain remote. Never overwrite old experiments. Submit/push this branch and verify remote SHA; no merge or force push.
