# FLOW-CRITIC-DUAL-HOST-TRAIN-WIRING-T0-V1

Date: 2026-10-14. Status: T0 authorized by the user's explicit GO decision.
Parent preparation commit: `cfbcc568b73795357ec67eabe53b741b0b044970`.
Implementation parent: `d35fd3ffe61c4dc1c72595c4f06fa31827ee4bfc`.

Root: `/home/xushijie3/work/flow_critic_dual_host_train_wiring_t0_v1_20261014`.
Branch: `codex/flow-critic-dual-host-train-wiring-t0-v1-20261014`.
Branch URL: https://github.com/rollinpioneer/graph/tree/codex/flow-critic-dual-host-train-wiring-t0-v1-20261014

## Authorization and endpoint

Repair controlled optimizer entry points, Target EMA write-back and complete native-main dispatch. Execute only minimal T0 admission on random initializations: GFP+floq on HumanoidMaze-medium task1 and DFQL+floq on Cube-double task2, seed 0, batch 256, two real Adam updates per combination (four total). Two steps are necessary because the source's pre-optimizer EMA leaves an initially identical Target unchanged at step 1; step 2 establishes real following movement. Save and restore full state between the two steps. No evaluation episodes or online learning.

T1 new-task Baseline training, T2 short stability runs and T3 1M/two-seed experiments remain NOT_AUTHORIZED. Their protocol/gates require a new decision. Neither the previous 24+26 pure checks nor this T0 gate establishes Baseline or combination performance. Missing prior research reports do not block T0.

## Frozen protocol

| Host | Task | Source host settings | T0 updates | Flow settings |
|---|---|---|---:|---|
| GFP G1 | humanoidmaze-medium-navigate-singletask-task1-v0 | Official CSV: gamma .995, alpha .3, eta .001, q/target mean, guidance min | 2 | Depth4, width512, 2 ensembles, 8 steps/noises |
| DFQL D1 | cube-double-play-singletask-task2-v0 | Official README/defaults: gamma .99, alpha2, q mean, DFQL multistep | 2 | Depth2, width512, 2 ensembles, 8 steps/noises |

lr .0003, original Adam, actor/critic architectures and losses unchanged. Actual reward ranges and OGBench1.1.0 data remain frozen. Float32/default JAX matrix precision, no precision override. Reuse validated isolated environments read-only. CUDA selection uses available GPUs without interrupting any process. Native T0 reads datasets on CPU and runs compute on GPU; no throughput/stability/performance claim is made.

## Acceptance

Closed/default native gates reject before setup; update APIs enforce finite lifetime quota. Native creation, final config, real updates, loss/gradient logging, official checkpoint serialization and restoring must work. Each real update increments model step and Adam count exactly once, changes expected trainable groups, gives Targets no gradient/Adam momentum, uses pre-optimizer EMA, leaves input state immutable, and advances/restores RNG correctly. Step2 must start from the restored step1 state. Checkpoint sidecars and byte hashes must agree. Default/full training and over-budget calls remain blocked. Batch/fused definitions receive import/JIT compilation checks without extra real optimization.

NaN/Inf, incorrect Targets, unexpected parameter changes or restore failures stop the case and preserve its changed state/ledger. No automatic restart, extra steps, parameter tuning or expansion. Existing preparation workspaces, datasets, checkpoints and branch remain unchanged.

Deliver code/patches, both execution gates, full native configs and commands, per-step receipts, checkpoints on server, T0 admission report, next-stage blockers and a new branch archive. Ordinary commit/push follows the user's standing AGENTS workflow; no PR, merge or force push.
