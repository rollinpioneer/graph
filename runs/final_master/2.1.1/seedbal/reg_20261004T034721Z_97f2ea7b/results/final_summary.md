# CP-DISR-TB-SEED-BALANCE-01 — final summary (frozen dev10 only)

Scope: exactly three new current-profile RL runs (B2 seed 0, B2 seed 2, B1-K+E seed 2) on the unchanged T_B production path; frozen Tcap 68812.8 s / Ncap 16384; dev10 deterministic argmax at N = 0 / 4096 / 8192 / final. No test30, no holdout, no seed 3+, no smoke (registration-only change, 0 smoke episodes).

## 1. Were the three runs executed on the current profile?
Yes. All three ran from prep commit `97f2ea7b0e204d3523a68274749db5b34da9e864` (parent result commit `4d0c426f67dfa1ce23247f89879c0b4952367d53`), through the unchanged `final_tb.run_train` -> `stage2a_v11.train_job`, from scratch (no warm start). The new B2 seed 0 is a NEW current-profile run, not the historical archived R1 B2 seed 0.

## 2. Source / config equivalence (`seed_balance_preflight.json`, verdict PASS, 17/17 checks)
- No pre-existing tracked file modified; production execution-path files and the `final_tb.py` production functions are identical to those the existing runs trained with (`cac2fa5b3cafb46f1f74370314189f4bca4878fa`, `e4dc34ae5ba7ddc8158424a9afae438cd211498a`).
- Run configs differ from R-TB-DK-1 / R-TB-E-0 / R-TB-E-1 only in plan_id, attempt_id, training_seed, source_commit and paths. Train64 (sha256 44e0ab7d...), dev10, H, d_ref, Tcap, Ncap, PPO, rollout, eval schedule, prior mode (absent), runtime manifest (modulo paths) and parameter count (3,539,651 = existing checkpoints) are equal.
- Seed chain: `train_job` takes the seed from the run context and calls `seed_all(seed)` once (Python/NumPy/Torch RNG); policy initialisation differs per seed and repeats per seed; the training seed is present in resolved_config, manifest, environment snapshot, every generation manifest and every checkpoint metadata of the three new runs (`seed_trace_check.json`, all true). Evaluation uses an isolated RNG (capture/restore).
- Recorded finding (production semantics, unchanged, applies equally to all existing and new runs): the train-case order `task_cases[n % 64]` does not depend on the seed. Seeds differ through Python/NumPy/Torch RNG and simulator noise.

## 3. Per-run results
| run | N | T (s) | complete+fragment updates | optimizer steps | 0 | 4096 | 8192 | final | final return | NaN / hard_fail |
|---|---|---|---|---|---|---|---|---|---|---|
| R-TB-B2-0-CURRENT | 14451 | 68812.90 | 14+1 | 892 | 0/10 | 10/10 (0.5000) | 10/10 (0.4988) | 10/10 | 0.4988 | 0 / none |
| R-TB-B2-2-CURRENT | 15583 | 68814.45 | 15+1 | 976 | 0/10 | 10/10 (0.5000) | 10/10 (0.4988) | 10/10 | 0.5000 | 0 / none |
| R-TB-E-2-CURRENT | 14845 | 68815.65 | 14+1 | 920 | 0/10 | 10/10 (0.4126) | 9/10 (0.2884) | 10/10 | 0.4003 | 0 / none |
All three stopped by Tcap, first-update gates passed, GPUs 0 / 1 / 3 (shared with other users' jobs, recorded), storage guard: 0 pauses, 0 hard-margin breaches. Failure at the +E seed 2 8192 point: one episode NO_CANDIDATE_SAFE_TERMINATION.

## 4. Matched table (first_effective_checkpoint)
| | s0 | s1 | s2 |
|---|---|---|---|
| +E | FINAL_ONLY | FINAL_ONLY | **4096** (new) |
| B2 | **4096** (new) | 4096 | **4096** (new) |
| ABS | FINAL_ONLY | 4096 | 8192 |
| NC | NEVER | FINAL_ONLY | NEVER |
Existing cells were re-derived from the committed tables and equal the frozen values in the card.

## 5. Per-family counts (EARLY / FINAL_ONLY / NEVER, of 3 seeds)
- B2: 3 / 0 / 0   - ABS: 2 / 1 / 0   - +E: 1 / 2 / 0   - NC: 0 / 1 / 2

## 6. Same-seed B2 vs +E
- s0: B2 4096 vs +E FINAL_ONLY (B2 earlier). s1: B2 4096 vs +E FINAL_ONLY (B2 earlier). s2: both 4096 (tie in first effective checkpoint; B2 kept 10/10 at 8192, +E dipped to 9/10). B2 is never later than +E.
- Same seed ABS vs NC: s0 FINAL_ONLY vs NEVER, s1 4096 vs FINAL_ONLY, s2 8192 vs NEVER (ABS never later).

## 7. Conclusion class (frozen rules, evaluated in order): **MIXED_REPRESENTATION_PATTERN**
B2 has 3/3 EARLY (so SEED_SENSITIVITY_HIGH does not apply), but +E has an EARLY seed (s2), which by the frozen rule means "direct effects are consistently late" can no longer be written. MECHANISM_PATTERN_REPLICATED is not claimed: it required +E 0/3 EARLY.

## 8. Effect on the T_B-centred hypothesis: **NARROWED**
- Kept / stronger: the B2 early-learning pattern is not a seed accident (3/3 seeds at 4096); successor-family arms (B2, ABS) are early in 5 of 6 runs versus 1 of 6 for the non-successor arms (+E, NC), and B2 is never later than +E at the same seed.
- Narrowed: the contrast is a strong tendency, not a categorical split. +E can reach 10/10 at 4096 on one seed, so explicit grounding is not necessary for early success in this profile, and the "direct effects consistently late" wording is withdrawn. With 3 seeds, ten dev cases, one task profile and no test30, no statistical significance or general sample-efficiency statement is made.

## 9. Structural generalization
Recommendation only (not started): worth doing next, because the open question is no longer whether B2 learns early (it does, 3/3) but whether the successor advantage reflects grounding that transfers across layouts or memorisation of the T_B layout; +E's one early seed also shows that more seeds alone would only tighten a tendency. Extra seeds for +E / ABS can be a secondary, separately authorised option. Nothing further is started.

## Bookkeeping
- First registration (preflight false positive on its own register process) abandoned before any token or attempt (`ABORTED_BEFORE_ANY_RUN.txt`); second registration is the one used.
- Because every GPU was shared with other users' jobs, the strict idle-GPU rule of the previous card could not be met; three distinct GPUs were assigned manually via the unchanged CLI `train` entry (`gpu_assignment_decision.json`). Wall time only.
- Checkpoints and transition logs stay on the server (paths and SHA256 in each run's `checkpoint_manifest.json`).
