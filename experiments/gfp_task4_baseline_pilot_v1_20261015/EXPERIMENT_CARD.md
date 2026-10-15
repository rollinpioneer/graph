# GFP-TASK4-BASELINE-PILOT-V1

Approved by direct user instruction on 2026-10-15. T1-A plus read-only Z2. No intermediate stage approval required within this scope.
Branch: `codex/gfp-task4-baseline-pilot-v1-20261015` ([link](https://github.com/rollinpioneer/graph/tree/codex/gfp-task4-baseline-pilot-v1-20261015)).
Root: `/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015`.

Exactly GFP and ReBRAC, seeds0/1, humanoidmaze-medium-navigate-singletask-task4-v0,1M offline updates each, online0. No extra seeds/tasks/modules/tuning/retries/extensions. No hour cap. Shared GPUs allowed with measured headroom; no waiting for exclusive GPUs or interference with other users.

Both algorithms use original agents in GFP authors' benchmark commit `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`, OGBench1.1.0. Mechanically inherit Hydra defaults and longest-matching official CSV row. GFP gamma=.995,alpha=.3,eta=.001,q_agg=mean,guidance_agg=min,target_agg=mean. ReBRAC gamma=.995,alpha_actor=.01,alpha_critic=.01,q_agg=mean; preserve delayed Actor update and native fused sampler. Preserve original float32/JAX precision.

## Asset reuse decision before new training

Existing independent ReBRAC task4 seeds0/1 at `/home/xushijie3/work/grebrac_l1_v1_20261015/runs/formal/R_task4_s{seed}` may be adopted read-only, avoiding duplicate training, only after official-config, disabled-guidance original computation, loss/gradient/state equivalence, source/data identity, required checkpoints and actual exit0 pass. Exclude guided variants and seeds2/3. Do not alter its code/processes/artifacts. Scores do not select assets; user fixed seeds0/1. Explicitly retain adoption differences and missing historical health telemetry. Failed admission does not count as official-equivalent; fresh original fallback is restricted to the designated algorithm/seed, no tuning. No automatic restart of failed new jobs.

## Training and evaluation

New jobs use separate copies; native Agent/loss/network sources unchanged. Known EGL fix uses spawn workers with inherited NumPy/Python state, already validated on task1. Instrumentation wraps dispatch, full-state saves and raw native evaluation receipts. First real fused dispatch counts inside1M and must pass finite values, exact Adam/update counters and save/restore identity. Save model/optimizer/Agent RNG/NumPy/Python states; no separate warm-up training.

Batch256,GPU dataset,n_steps_fused4. Log5k; native evaluation/checkpoint100k; retain at least200k/600k/800k/900k/1M. Native100 episodes,10 environments: GFP Actor+VaBC Flow, ReBRAC Actor. Preserve official aggregation separately from raw episode summaries. Diagnostics do not modify gradients or sampling.

Primary matched: Actor800k/900k/1M,100 episodes each; last-three mean per seed, then individual seeds and descriptive mean/SD. Original OGBench1.1 task,horizon2000,reward/success. Global NumPy/Python/reset seed20261015+episode,policy seed420000+episode, fresh split each action. Verify equal initial-observation hashes across algorithms/seeds/checkpoints. GFP stays stochastic at temperature0; ReBRAC temperature0 is deterministic. Save numerical trajectories for Z2. Max matched2.4M environment steps; native evaluations separate, online updates0. Check checkpoint/state hashes before/after read-only evaluation.
## Read-only Z2 and decision
After task4 checkpoints exist, reuse the frozen Z01 train/validation sample indices and whole-batch normalization: seeds0/1 at200k/600k/1M,100 train+100 validation batches of256 each (307200 states), two additional RNG repetitions on10 batches per split (61440 states). Reuse task1 measurements; compute task4 BFS/Euclidean/detour features using its own goal. Preserve task-specific reward identities. No environment creation/steps or optimizer updates for Z2 inference.
Compare neutral mass,Q differences,ensemble ranking disagreement,trajectory-cluster intervals and controlled descriptive associations. Coverage uses grid occupancy,unique trajectories and directed transition/turn counts,not raw-observation nearest-neighbor assertions. Future25/100-step path progress cannot cross trajectory boundaries and is only a collection-policy proxy. Map failures from the already-authorized matched evaluations to topology and coverage; do not generate new interventions. No true action-advantage claim from these proxies.

Stop the affected new job on NaN/Inf,param_abs>1e6,metric_abs>1e12,unexpected update count,config/hash/restore/save error or nonzero exit; preserve evidence,never automatically restart. Poor success alone is a retained result,not a tuning trigger. Match initial state hashes before comparisons. One joint report plus GO/DIAGNOSE/NO-GO table,per-checkpoint/per-seed metrics,cost and asset index. Two seeds are preliminary. GO needs task4 difficulty plus independent harmful action-filter evidence; otherwise DIAGNOSE competing mechanisms or NO-GO current H1. No automatic dynamic-temperature work or seed expansion.
