# FLOW-CRITIC-COMPARATIVE-TRAIN-V1

Approved 2026-10-14 by direct instruction “执行，并行多卡执行，不要等待空闲gpu”. This supersedes the pasted draft/approval boilerplate for this batch. No stage reapproval is needed inside the frozen scope.

Branch: `codex/flow-critic-comparative-train-v1-20261014` ([GitHub](https://github.com/rollinpioneer/graph/tree/codex/flow-critic-comparative-train-v1-20261014)).
Root: `/home/xushijie3/work/flow_critic_comparative_train_v1_20261014`.
Implementation base: `28574c4ecaa9100c762750cf6d10b843c33e318b` (T0 admission); previous assets and branches stay intact.

| Model | Task | Seeds | Updates per seed |
|---|---|---|---|
| G0 official GFP | humanoidmaze-medium-navigate-singletask-task1-v0 | 0,1 | 1,000,000 |
| G1 GFP + floq Critic | same | 0,1 | 1,000,000 |
| D0 official DFQL | cube-double-play-singletask-task2-v0 | 0,1 | 1,000,000 |
| D1 DFQL + floq Critic | same | 0,1 | 1,000,000 |

Exactly 8M formal updates maximum. No online steps, extra seeds/tasks, retries, tuning or extension. Engineering T0's historical four updates are separately identified, never part of these fresh jobs. The first native dispatch admission and combination 50k T2 are included in each fresh job's 1M.

## Fixed sources and environments

GFP `1abbd5f0e287caf9bda70e7bc655d0a499ac8401`; DFQL `8b6b15803cc1e5605ad1524553a2c95e103b0bad`; floq `3d60e638b42bbef018c56cb1199ff37c3470520d`.
Read-only reuse of verified PREP environments, full official OGBench 1.1.0 data/cache and snapshots; new writable host copies only. JAX/jaxlib 0.6.2, Flax 0.10.7, Optax 0.2.5, MuJoCo 3.3.5. GFP Python 3.12/Gymnasium 1.1.1; DFQL Python 3.10/Gymnasium 0.29.1. No dependency upgrades. Inherit native matrix precision and x64 defaults; record runtime values, no precision override. Native default float32 arrays.

Official GFP task CSV remains matched through agent_name=gfp: gamma .995, alpha .3, eta .001, q_agg mean, target_agg mean, guidance_agg min. DFQL pinned README Cube recipe: gamma .99, alpha 2, q_agg mean. Host Actor, VaBC, clipping, target-state semantics, discounts and losses are unchanged. Floq values retain expected TD targets, noise-SUM reduction, pre-optimizer EMA, 8 noise samples / 8 Euler steps / 2 ensembles, width512 and depths4/2. No FA/FAN/Q-Flow. Complete prepared and actual expanded configs are retained.

## Fairness and differences recorded before training

Baseline Agent classes and shared floq core are byte/AST checked against fixed sources. Added modules consume initialization keys differently: copy only the host Actor/VaBC initialization from the original create() into each fresh combination; optimizer remains zero and Agent RNG remains the same pinned split. P0 verifies exact Actor weights, initial actions and Agent RNG for seeds0/1. This explicit initialization control is an engineering addition to the T0 adapter, not a new Actor objective.

GFP uses the official GPU dataset sampler and n_steps_fused=4. DFQL uses its official CPU dataset single-update route. Each route's first actual dispatch must pass finite-state, Adam/update-count and full-state save/restore admission, counted within the job. No compiled-only batch route is admitted. Logging/health instrumentation does not modify gradients, sampling, optimizer, EMA or algorithm precision. All native child exits are recorded, including failures. Prior T0 GFP missing exit code is not reused as evidence for these jobs.

## Evaluation and artifacts frozen in P0

Native: unchanged source evaluators. GFP final-setting 100 episodes, 10 environments, Actor and VaBC Flow; DFQL default 50 episodes and serial evaluator, Actor. Every100k, retaining DFQL's official initial step1 evaluation. Video0. Raw episode success/return/length and native aggregate statistics retained separately; native GFP parallel aggregation quirks are not silently corrected.

Matched: each800k/900k/1M Actor checkpoint, 100 serial episodes in fixed OGBench task; reset seed20261014+episode and policy seed420000+episode, fresh split per action. Same within each host pair and across seeds/checkpoints. Success is any success in the episode, Return=sum rewards, terminal OR truncated ends the episode. Native OGBench horizon/reward/task are retained. Temperature0 does not remove flow-policy Gaussian randomness. Preserve NumPy/Python training RNG around matched evaluation; Agent is immutable.

Engineering correction before any matched checkpoint exists: fixed OGBench HumanoidMaze also draws start noise from global NumPy, so seed NumPy/Python with each reset seed and retain initial observation SHA for each episode. Official native evaluations are unchanged. Verified matched evaluation is a separate read-only process restoring each saved checkpoint, with zero optimizer updates. Results in matched_verified are primary; any early in-memory inline evaluator result in matched is UNVERIFIED_GLOBAL_RNG and excluded. Do not overwrite those diagnostics. Identical starts must be hash-checked across each pair before calling the scores matched.

Primary pairwise result: mean of800k,900k,1M, with each checkpoint and each seed shown; retain native protocol separately. No selection of highest score. Two-seed evidence is preliminary and not author five-task averages. Cross-task G-vs-D ranking is invalid.

Every100k save full model/Adam/Agent RNG plus NumPy/Python RNG sidecar, hashes, config identity and index; verify restore immediately. Combination50k also saved. Train/validation loss logging every5k; progress every30seconds; CPU/memory/GPU process sampling every15seconds. Preserve all checkpoints/full logs remotely; Git stores explicit lightweight files/indices only.

## Automatic phases and safety

Four baselines first in parallel. After T1 engineering success, two combinations per host execute50k T2. Pure diagnostics require finite losses/gradients, target parameter movement and zero target gradients; then successful native exit and full checkpoint hash/config/optimizer/RNG verification permit one T3 continuation from50k to1M. Algorithm configuration and optimizer budget stay1M through both phases; no discarded trial. A host baseline engineering failure blocks that host's combinations. Low success alone is never a stop or tuning criterion.

Startup incident: both GFP baselines exited1 at update0 because fork inherited an active MuJoCo EGL context. A spawn-worker fix preserving native NumPy/Python inheritance passed a zero-update 10-environment reset/step comparison. No restart is automatic; the user explicitly approved one corrected launch per affected seed. That single authorized attempt is recorded in receipts/authorized_startup_repair_dispatched.json. DFQL baseline jobs proceed independently; GFP combinations are blocked while the baseline incident remains unresolved. Full original failure logs are kept. This is an environment startup failure, not a measured algorithm training failure.

Stop the affected job and preserve state on any NaN/Inf in metrics/model/optimizer, inconsistent Adam/update count, checkpoint/hash/config/RNG recovery mismatch, target no movement at50k, parameter absolute max>1e6 or logged metric absolute max>1e12. These broad engineering thresholds are frozen now, not chosen from final scores. Parameter/optimizer finite checks run compiled on each native single update or fused block of4; numerical contamination in intermediate updates cannot be accepted merely because compilation succeeds. No automatic restart.

Resources: four A10040GB cards6/7(GFP),5/4(DFQL), selected as least-loaded safe cards with no account process. Share other-user GPU loads without inspecting their commands or interrupting them. 8 CPU cores per job, BLAS threads1; GPU allocator fraction.30, preallocationfalse; RSS ceiling48GiB per job, available host-memory floor64GiB before launching; GPU-free floor16GiB before launch. If placement becomes unsafe, choose another currently safe low-load card immediately; if none fit report resource blocking, not wait for an exclusive card. Keep at least4GiB GPU headroom. No hour cap; update cap8M. Do not reduce batch256 or precision. GPU residency includes compilation, evaluation and save; CPU and process peaks recorded; wall time separately. Report estimates after actual progress, not reuse old SAW timings.

## Completion and archival

Autonomous run manager does not repeat jobs. Monitoring keeps normal progress quiet, reports failures/required action and final completion. Finish with paired success/return, seed variability, losses/Q diagnostics, timing/resources and full checkpoint indices, then pause monitoring. No merge/force push. Archive to this new branch using explicit paths, verify remote SHA and sync local reports. Older FQL/GFP/floq checkpoints, results and branches are not modified.
