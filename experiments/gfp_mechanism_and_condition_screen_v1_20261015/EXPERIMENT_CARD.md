# GFP-MECHANISM-AND-CONDITION-SCREEN-V1

Date: 2026-10-15. Authorized by user request to execute both screenshot workstreams.
Branch: codex/gfp-mechanism-and-condition-screen-v1-20261015
Branch URL: https://github.com/rollinpioneer/graph/tree/codex/gfp-mechanism-and-condition-screen-v1-20261015
Parent archive: 83d147e1f976ad8f87848989e3b7dda336c2d724.

## A. Frozen mechanism probe
Original GFP commit 1abbd5f0e287caf9bda70e7bc655d0a499ac8401; original G0 seeds 0/1, checkpoint 1M; HumanoidMaze-medium task1. Restore each original full Agent in its verified environment. Model, Adam, RNG and checkpoint hashes must remain unchanged. Optimizer updates 0; no tuning or new modules.

Frozen state list: configs/states.json, selected from the existing 48 recorded trajectories without new interactions. Per seed: two near-goal failures at first distance <=1.2m; two pre-stall failure states; two approximately matched successful controls. Selection logic in code/select_states.py. Controls are similar, not identical states; report matching distance and avoid causal claims from the matching alone.

At most 12 states, 3 branches each, 300 steps each, total new environment steps <=10800 including restoration-validation steps. Existing horizon 2000 unchanged. Original A branch is also the restoration check, not an extra trajectory. Require initial observation/state agreement and original A actions/rewards/termination plus subsequent observations/qpos/qvel within 1e-8 absolute tolerance (actions 1e-7) throughout up to 300 steps. On unexplained divergence stop the affected workstream before treating counterfactual branches as evidence. No automatic retries or extra rollouts.

Candidates: A original Actor sample; B one independently seeded Actor sample; C one VaBC sample using the same noise as B. Only the first action is replaced; all subsequent actions use the same frozen Actor and common per-step policy RNG suffix. No best-of-Q action selection. Same full MuJoCo integration state, goal, elapsed time, task and RNG for all branches. Record both online and target Critic ensemble scores, action distances, clipping, success, reward, discounted partial return, distance, movement, posture and contacts. Pure inference of 32 Actor and 32 VaBC samples per state may describe proposal distributions; it does not establish their actual performance.

Use existing full integration snapshot and restore the simulator phase expected by its split-step implementation. Preserve all old files. No new video required. Stop on nonfinite values, restore mismatch, changed assets or step cap. A 300-step result is a local intervention under one fixed subsequent noise sequence, not a complete Q estimate or a success-rate estimate.

## B. Public condition screening
Read official GFP, NeoRL-2, DSRL and closest primary papers/code. Deliver at most three condition cards: history/delay, safety, movement versus final correction. For each: concrete data/fields/trajectory/evaluation interfaces, actual method boundary, falsifiable prediction, simple alternatives, closest work, future minimal test and exit conditions. Distinguish code/API verification from actual data inspection and from empirical evidence. Do not train, remove necessary inputs, change dynamics while reusing invalid transitions, or implement new modules. Future tests are proposals only.

## Resource and archive
Use one existing GFP environment and one safely shared GPU with headroom; no interruption of any other process. Record actual wall/CPU/GPU residency and environment steps. No time budget. Single keepalive SSH, normal exit and client verification. Archive only explicit code/config/card/summary/receipt/source-evidence list to this branch; no models, full traces or videos. Verify remote SHA, then sync local handoff.

## Known field clarification
Old recorded raw_actions were already clipped by official sample_actions; they are not pre-clip Actor outputs. New diagnostics will label true network output separately. temperature=0 is unused by this GFP sampler and is not deterministic-policy evaluation.
