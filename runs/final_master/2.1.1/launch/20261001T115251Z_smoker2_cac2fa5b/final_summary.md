# CP-DISR-TB-SMOKE-SELECTOR-REPAIR-R2-1 — final summary

Written 2026-10-02 (UTC) after all three workers finished. Baseline 329652cab865a1b189135dc63f3971816f5f941e; prep/source HEAD cac2fa5b3cafb46f1f74370314189f4bca4878fa on branch `codex/cp-disr-final-tb-engineering-launch`, worktree `/home/xushijie2/graph_cp_disr_final_tb_launch`. HEAD and the tracked tree were unchanged (HEAD == prep, tracked tree clean) from registration to the last check (2026-10-02T00:40Z).

Outcome: smoke `TB_RUNTIME_SMOKE_PASS` (2/2 cases), then all three registered base runs started, ran to Tcap and are COMPLETE: R-TB-E-0, R-TB-K-1, R-TB-DK-1 (stop_reason=Tcap, NaN_n=0, hard_fail=null, fresh-load of the final checkpoint present, selected checkpoint exists).
Provider = 0, T_P = 0, formal test = 0, Family B = 0, RoboCasa = 0, elastic = 0, test ids accessed = 0. New RL attempts used 3 / 3 (the three registered plans; no attempt added, nothing launched into a freed GPU slot).

## 1. Old failure kept permanently
`runs/final_master/2.1.1/launch/20261001_prep083715b0/` (26 files; tree digest sha256 30122be2f589309ccbcfbc136744a47f1c720e93e9f923f4a31ba5c19d6b4072 recorded in `prior_smoke_reconciliation.json`) plus `a0_pre_prep/` and `aborted_register_relative_out_path_refused/` are untouched; git shows no change to their tracked files. The old directory still carries its original `final_summary.md`. The correction below is recorded in the NEW directory (`prior_smoke_reconciliation.json` and this file), not by editing the old one, so the old evidence digest stays valid.

**Correction of the old wording.** The old summary said "Zero skills executed; the failure was raised before any controller call". That is withdrawn. An empty `steps` array and no returned Transition do NOT show the executor was not called: `Collector.step` runs policy forward -> select -> executor -> perceive/verify -> evaluator -> snapshot -> next-V probe -> Transition, so the failing next-V probe can run after a real execution. The old persisted receipt carries no executor counter, so the old actual executor-call count is UNVERIFIED_FROM_PERSISTED_RECEIPT. The old attempt was charged 1 skill call and nothing is refunded.

Root cause (reproduced on CPU with the unmodified baseline Collector, blob 932eae64): the old `ScriptedSelector` forced a one-hot script action on the stale target also during the next-value probe, where that target is masked -> all legal logits `-inf` -> NaN Categorical (`ValueError: Expected parameter logits ... nan`). Repair: the script action is applied only at a true `select`; Policy forward/value/hidden/logits/distribution untouched.

## 2. Smoke actually run this round (physical GPU 2, graphics context only on GPU 2)
| case | verdict | executor entered/returned/raised | policy forwards | transitions returned | sim steps | charged this attempt (episodes/constructions/resets/skills) |
|---|---|---|---|---|---|---|
| T_B_dev_00 | PASS (final task success true) | 5 / 5 / 0 | 9 | 5 | 480 | 1 / 2 / 2 / 5 |
| T_B_dev_04 | PASS (final task success true) | 5 / 5 / 0 | 9 | 5 | 479 | 1 / 2 / 2 / 5 |
Both under the same new prep (cac2fa5b); scripted 5-action path PICK second -> PLACE_BUFFER -> OPEN -> PICK target -> PLACE target; provider requests 0, optimizer steps 0, `event_log_errors` empty. Receipt `smoke_receipt.json` label TB_RUNTIME_SMOKE_PASS.

## 3. Old vs new: actual executor calls vs charges
| | actual executor calls | charged skill calls | episodes | constructions | explicit resets |
|---|---|---|---|---|---|
| OLD smoke (083715b0, case 1 failed) | UNVERIFIED (no counter persisted; CPU reproduction shows the failure occurs after one execution, so >=1 is expected but not measured on the real run) | 1 | 1 | 2 | 2 |
| NEW smoke (cac2fa5b), 2 cases | 10 entered = 10 returned, 0 raised (measured) | 10 | 2 | 4 | 4 |
| Cumulative | — | 11 of 24 | 3 of 3 | 6 of 6 | 6 of 6 |
R2 round caps 2 / 4 / 4 / 23 and cumulative caps 3 / 6 / 6 / 24 were respected; old 1/2/2/1 not refunded; the one unused parent episode, 2 constructions, 2 resets were transferred into R2 (+1 episode, +2, +2 newly authorised, 0 extra skill total). New charges equal measured actual calls (10 = 10).

## 4. Pre-smoke CPU evidence (test scope)
- Targeted suite `tests/test_final_tb_smoke_r2.py tests/test_final_tb_launch.py`: 44 passed, 3 warnings (89 s) at HEAD cac2fa5b.
- `selector_regression_receipt.json`: 8 checks all passed (stale-target next-value with the REAL Collector.step and production B1-K Policy on the real T_B template; 5-action script cursor; masked missing target fails closed; terminal/no-candidate; stale action still legal; non-finite not hidden; output fields unchanged; post-execution failure recorded). Only the physical boundary is a test double.
- Mutation: restoring the OLD selector (blob-identical fixture) makes check 01 fail with the same NaN Categorical error; the repaired selector passes. CPU reproduction: reference run 10 tests / 0 failures; mutant 1 failure + 6 errors (expected).
- Scope limit: CPU regression and 2 scripted smoke cases only. NOT_RUN this card: the full repository test suite (the old summary's 679 passed / 13 failed was for the old prep tree and was not re-measured). The smoke cases do not exercise a trained Policy; trained-Policy behaviour is covered only by the three runs themselves (NaN_n = 0 in all three, real-Policy NaN count 0).

## 5. Algorithm source unchanged?
Yes for the algorithm. `git diff --stat 329652cab HEAD` lists exactly 6 files: `src/cp_disr/final_tb.py` (launch/smoke entry incl. ScriptedSelector, receipts, gates), `tests/fixtures/r2_selector_original.txt`, `tests/helpers/final_tb_checks.py`, `tests/helpers/smoke_r2_checks.py`, `tests/test_final_tb_launch.py`, `tests/test_final_tb_smoke_r2.py`. `adapters.py, collector.py, common.py, facts.py, neural.py, rl.py, torch_rl.py` have identical git blobs to the baseline; Policy, Collector, PPO, controllers and tasks were not changed. The concurrency change 2 -> 3 was an in-process override for K-1 only (`launch_k1_w3.py`, event in `concurrency_override_events.jsonl`, wrapper sha256 9bf36865...); the source constant `MAX_WORKERS = 2` was NOT edited and HEAD did not move.

## 6. Formal three-slot start status
All three conditional slots were released after both smokes PASSED and the storage budget passed (`storage_budget.json`: required 8.60 GB incl. 2 GiB reserve, free 10.10 GB at 2026-10-01T12:01Z). First-1024-transition PPO engineering gate passed for each run before the next release. Original N/T budgets (Ncap 16384, Tcap 68812.8), one GPU per run, no extra attempts, no elastic.
| plan | method / seed | GPU | started (UTC) | finished (UTC) | complete + fragment updates | N | interaction seconds T | overshoot over Tcap | optimizer steps |
|---|---|---|---|---|---|---|---|---|---|
| R-TB-E-0 | B1-K+E / 0 | 0 | 10-01 12:01:59 | 10-01 20:55:11 | 14 + 1 | 14512 | 68813.30 | 0.50 s | 872 |
| R-TB-DK-1 | B2 / 1 | 2 | 10-01 12:40:36 | 10-02 00:33:22 | 13 + 1 | 14170 | 68817.10 | 4.30 s | 876 |
| R-TB-K-1 | B1-K / 1 | 1 | 10-01 13:59:06 | 10-01 21:54:48 | 14 + 1 | 14531 | 68813.45 | 0.65 s | 864 |
The stop rule fires after the step that crosses Tcap (final fragment update, eval_final, fresh-load reload, job_summary); the overshoot is recorded, not corrected. All three: NaN_n 0, hard_fail null, clock-audit remainder 0/0/0, N_logged == N_used_by_PPO. No log contains NaN/OOM/EGL/traceback lines. Per-run `*_completion_check.json` hold the sha256 of job_summary / eval_final / fresh-load files.

## 7. Dev10 results as measured (no interpretation; no formal test)
| run | success rate at eval point N = 0 / 4096 / 8192 / final | train success episodes | selected checkpoint |
|---|---|---|---|
| R-TB-E-0 (B1-K+E, seed 0) | 0.0 / 0.0 / 0.0 / 1.0 (final success_n 10, mean discounted return 0.4893) | 981 | final_n_014512_u14 |
| R-TB-DK-1 (B2, seed 1) | 0.0 / 1.0 / 1.0 / 1.0 (success_n 10 each; mean discounted return 0.4988490585995873 at all three non-zero points) | 1644 | final_n_014170_u13 |
| R-TB-K-1 (B1-K, seed 1) | 0.0 / 0.0 / 0.0 / 0.0 | 194 | n_000000 (step-0; all evals tied at 0, selection rule max_mean3 -> max_worst -> earlier N) |
Caveats: dev10 only (10 cases), one run per method, E-0 uses seed 0 while K-1 and DK-1 use seed 1, so method and seed are confounded between E-0 and K-1; no claim about methods is made here. The identical DK-1 mean return at three eval points is reported as measured; whether the three evaluations took identical trajectories is NOT_VERIFIED.

## 8. Concurrency amendment 2 -> 3 (K-1 as third worker) — measured windows
Amendment `concurrency_amendment.json` (+ `addendum1`, written before any 3-worker window was analysed: phase-matched collection-phase rate is the primary comparison, all-in secondary, because PPO phases write no transitions). Pre-launch re-verification `k1_prelaunch_check.json`: all_passed; K-1 config/source/token identity unchanged (HEAD == prep, config sha256 equal to registered), independent GPU 1 (not shared with E-0 on GPU 0 or DK-1 on GPU 2), free space sufficient incl. >= 2 GiB reserve, no OOM/EGL conflict, no new benchmark attempt. K-1 launched 2026-10-01T13:59:06Z.
| window (30 s samples) | E-0 /min | DK-1 /min | K-1 /min | aggregate |
|---|---|---|---|---|
| pre-launch, 2 workers (13:55:53-13:58:54; only 3 min, low confidence) | 35.8 | 21.6 | — | 57.4 |
| 2-worker collection-phase baseline from the addendum (completed spans) | ~30.8 | ~30.1 | — | ~60.9 |
| 0-15 min after launch (13:59:24-14:13:58), all-in = collection-phase | 36.9 | 20.0 | 27.7 | 84.5 |
| 0-30 min (13:59:24-14:29:02), all-in | 27.9 | 20.0 | 28.2 | 76.2 |
| 0-30 min, collection-phase (primary) | 36.6 | 20.0 | 28.2 | 84.9 |
- Aggregate under three workers was higher than under two (84.9 vs ~60.9 collection-phase; 76.2 vs 57.4 all-in), so the "total throughput lower than two workers" stop condition was not met.
- Per worker: E-0 collection-phase 36.6 (vs ~30.8); DK-1 20.0 vs ~30.1 in the window — DK-1's collection interval N 1024->2048 (51.5 min) was its slowest of the run (19.9/min); its next 3-worker intervals were 27.0, 29.7, 28.3, 30.1, 30.3, 34.8, 33.4 /min, i.e. a single slow interval that recovered while three workers kept running, not a persistent deterioration (DK-1 case mix and policy also changed over time, so this is not a clean causal contrast). Mean collection rate per interval by number of live workers (intervals classified by midpoint; confounded by training progress): E-0 32.4 (2 workers, n=2) vs 32.9 (3, n=11); DK-1 30.9 (2, n=2) / 29.2 (3, n=8) / 34.5 (1, n=3); K-1 35.1 (3, n=12) / 37.3 (2, n=2). Full table: `throughput/epoch_analysis.json`.
- PPO update wall time (minutes, published - PPO start; E-0 / DK-1 / K-1): E-0 5.9, 6.0, 3.8, 3.7, 3.7, 8.0, 3.8, 3.7, 7.1, 10.8, 6.8, 6.9, 7.2, 7.1; DK-1 27.1, 19.4, 35.4, 17.4, 17.9, 18.1, 16.1, 13.3, 13.4, 13.4, 13.5, 13.5, 12.0; K-1 4.2, 4.1, 9.9, 4.1, 4.0, 2.8, 2.8, 3.0, 2.9, 2.8, 2.9, 2.9, 2.9, 2.9 (`throughput/window_report.json`). Spikes on E-0 (N=6144: 8.0), DK-1 (N=3072: 35.4) and K-1 (N=3072: 9.9) around 15:26-16:10Z coincided with external GPU contention on all three GPUs (`throughput/ppo_wall_time_observation_1.md`); causality is not proven (no per-process GPU profiling). E-0's PPO time stayed 6.8-10.8 min from N=9216 on while its GPU 0 showed other tenants (PPO-window GPU 0 util mean 94-99 %, mem 5768-27457 MiB, vs util ~20 %, mem 1240-1360 MiB in the 3.7-min windows; `throughput/gpu_storage_analysis.json`). DK-1's 13.3-13.5 min was unchanged between 3, 2 and 1 live workers; K-1's 2.8-2.9 min unchanged between 3 and 2 live workers. E-0 finished before it could be compared in a 2-worker regime, so the cause of its slower PPO is NOT_MEASURED.
- GPU / I/O in the 0-30 min window: GPU 0 (E-0) util mean 5.5 %, mem 911-2117 MiB; GPU 1 (K-1) util mean 85.5 %, mem 3686-6010 MiB; GPU 2 (DK-1) util mean 90.0 %, mem 5607-7427 MiB; I/O wait mean 0.040 (max 0.088); load1 max 88 of 112 cores. Whole-run (1283 samples, 13:55:53Z - 00:40:26Z): I/O wait mean 0.0435, max 0.117; no OOM, no EGL conflict, no worker stopped.
- Decision per the amendment: K-1 was not stopped; E-0 and DK-1 were never touched (N/T/wall clock continuous); no checkpoint-resume was needed.

## 9. Storage
/home free (shared XFS, other users' writers): 20.5 GB at monitoring start, 9.84 GB at the end. Low points: below 7 GB from ~19:45Z (10-01); below 4 GB from 23:22Z; minimum 1.665 GB at 2026-10-02T00:16:17Z; below 2 GiB for ~2.5-3 min (6 samples 00:14:46-00:17:17Z), i.e. briefly under the >= 2 GiB safety margin required at K-1 launch. Passive `storage_guard.py` (stop threshold 1.0 GB, resume 7.0 GB; a dry-run self-test at 17:50:52Z only logged WOULD_STOP without signals) never paused any worker (`GUARD_EXIT_RESUMED_ALL`, paused []); 81 heartbeats. No history deleted, no save frequency lowered, RoboCasa not installed. Final run directory sizes: E-0 2224 MiB, K-1 1716 MiB, DK-1 1581 MiB (total 5521 MiB); the pre-registered per-run peak estimate was 2051 MiB, so E-0 exceeded it by ~173 MiB without operational effect. Attribution of the external free-space drops to other users is not proven per process; filesystem quota: NOT_MEASURED (quota query returned rc=1).

## 10. Unverified / NOT_MEASURED
- Old failure's actual executor-call count: UNVERIFIED. Old summary's "zero skills" claim: withdrawn.
- Full repository suite: NOT_RUN this card. Trained-Policy numerical behaviour in smoke: not exercised (scripted paths); in training: NaN_n 0 x 3.
- Cause of E-0's PPO slowdown from N=9216 and of the 15:26-16:10Z spikes: NOT_MEASURED.
- Quota: NOT_MEASURED. Per-process disk attribution: NOT_MEASURED.
- No method comparison, no held-out test, no provider, no T_P; the three dev10 numbers above are engineering/progress records.
- Large files (transition/decision/episode logs, checkpoints, persistence generations) are NOT in the git commit; their paths, sizes and sha256 are in `artifact_manifest.jsonl`.

## 11. Evidence
Registration/evidence: `runs/final_master/2.1.1/launch/20261001T115251Z_smoker2_cac2fa5b/` (authorization, amendment authorization, budget reconciliation, prior-smoke reconciliation with old-tree digest, launch_plan/state, smoke cases + events + receipt, selector regression + mutation + CPU-reproduction receipts, storage budget and checks, release tokens, concurrency amendment/addendum/override event, K-1 prelaunch check, throughput samples + window/epoch/GPU/storage analyses, guard log, per-run completion checks, logs, helper scripts, artifact manifest). Run directories: `runs/final_master/2.1.1/T_B/{B1-K+E/seed_0,B2/seed_1,B1-K/seed_1}/R-TB-*-20261001T115252Z-cac2fa5b/` (small files committed; large files hashed).
