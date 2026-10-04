# CP-DISR-TB-REP-CONTROLS-01 — final summary (frozen dev10 only)

Scope: six new RL runs (B2-ABS and B1-K+NC, seeds 0/1/2), frozen T_B current profile (Tcap 68812.8 s, Ncap 16384, PPO/reward/evaluator unchanged), dev10 deterministic argmax at N = 0 / 4096 / 8192 / final.
No test30, no holdout, no provider, no extra seed, no third method. All six attempts COMPLETE by Tcap, NaN 0, hard_fail none (`verify.json` PASS).

## 1. Implementation as defined
- Absolute Successor (`B2-ABS`): B2's unchanged four_views / nominal_apply / encoder calls; candidate rows = E(F_after_i) instead of E(F_after_i) - E(F). Same module set and parameters as B2.
- Matched Neural Composition (`B1-K+NC`): rows = learned interaction over (F, K(a_i)); no nominal_apply, no F_after, no sidecar (source scan, runtime trap and file-read trap tests; 8/8 mutants detected).
- neural.py and every production file are byte-identical to base 9422b837c; legacy B1-K / B1-K+E / B2 forwards are element-wise identical (fresh init and strict load of the real final checkpoints).

## 2. Parameters and computation (representation_capacity_audit.json)
| arm | effective params | vs B2 |
|---|---|---|
| B2 | 849,987 | - |
| B2-ABS | 849,987 | 0 |
| B1-K+NC | 884,291 | +4.04 % (within the 5 % band) |
| B1-K+E | 2,695,363 | +217 % (existing arm, reported not corrected) |
Computation per decision (K legal candidates): B2 / ABS = 1 + 2K encoder calls and K nominal_apply calls; NC = 1 encoder call + K interaction calls; +E = 1 encoder call + K token-readout calls. NC therefore also has much less computation than the successor arms.

## 3. dev10 success at 0 / 4096 / 8192 / final
| run | 0 | 4096 | 8192 | final (N) |
|---|---|---|---|---|
| +E s0 (existing) | 0 | 0 | 0 | 10 (14512) |
| +E s1 (existing) | 0 | 0 | 0 | 10 (14309) |
| B2 s1 (existing) | 0 | 10 | 10 | 10 (14170) |
| B2-ABS s0 | 0 | 0 | 0 | 10 (14070) |
| B2-ABS s1 | 0 | 10 | 10 | 10 (14048) |
| B2-ABS s2 | 0 | 0 | 10 | 10 (15093) |
| B1-K+NC s0 | 0 | 0 | 0 | 0 (14446) |
| B1-K+NC s1 | 0 | 0 | 0 | 10 (14526) |
| B1-K+NC s2 | 0 | 0 | 0 | 0 (15499) |
Mean discounted return is 0.498849 for every successful ABS checkpoint (identical to B2's) and 0.499977 for NC s1 final (identical to +E s1 final). Full N/T per point and failure-reason families are in `representation_control_results.md` and `learning_checkpoint_table.csv`.

## 4. Seed consistency
- Pre-registered "early" = dev10 10/10 at N=4096 or N=8192. B2-ABS: 2 of 3 seeds early (s1 at 4096, s2 at 8192; s0 first succeeds only at final). B1-K+NC: 0 of 3 early; 2 of 3 seeds never succeeded within the budget (final 0/10). Existing +E: 0 of 2 early (both 10/10 at final). Existing B2: s1 early (historical s0, listed apart, early at 8192).
- ABS is therefore not uniform across seeds (s0 is late); with three seeds and ten dev cases no statistical statement is made.

## 5. Case classification (pre-registered rules)
**CASE A**, with a CASE D flag on the Absolute Successor seed spread: Absolute Successor behaves like B2 (earlier learning in 2/3 seeds, same dev return), Matched Neural Composition behaves like +E at 4096/8192 (and is worse at final: 2/3 seeds never succeeded). This supports **explicit state-conditioned successor grounding** as the main candidate explanation; the **latent difference is not necessary** (ABS has none) and **learned neural composition did not reproduce the early learning** (not CASE B, not CASE C). CASE E does not apply (no unexpected new pattern requiring a third method).

## 6. Effect on the current T_B-centred hypothesis: **STRENGTHENED** (moderately, with caveats)
Caveats that must travel with the result: (a) 3 seeds per control, existing B2 has one current-profile seed, +E two, seeds are not balanced across arms; (b) ABS seed 0 is late, so "earlier" is a seed-majority statement, not a guarantee; (c) NC was deliberately capped at +4 % parameters and has far less computation than the successor arms and a lighter state x effect interaction than a four-layer graph encoder applied twice, so a stronger learned composition is not excluded by this card; (d) NC seeds 0 and 2 failing even at final shows NC is also a weaker learner overall, not only a slower one; (e) dev10 only, one task profile, no test30; (f) time_to_first_confirmed_success was not computed (episode-level success time remains UNVERIFIED in the existing eval rows).

## 7. Next stage
Worth considering, but only after a new explicit authorization: seed balance (B2 seeds 0 and 2, +E seed 2, and ABS/NC extra seeds if wanted) plus a structural-generalization check that separates grounding from memorisation of the T_B layout. This card does not start either.

## Deviations and bookkeeping
- Concurrency raised from the suggested 3 to 6 workers at the user's request (supervisor stopped, three plans started with an in-process override; recorded in `concurrency_amendment.json`); no scientific content changed.
- First registration aborted before any token or attempt (storage start gate path bug, fixed in the second prep commit; `ABORTED_BEFORE_ANY_RUN.txt`).
- Baseline test suite: 31 pre-existing failures on the pristine base (test_final_tb_e1 test_08/test_09) plus one path-dependent failure (test_12b) reproduced on a pristine checkout; no failure was introduced by this card (`regression_test_receipt.json`).
