# Post-run zero-new-sample analysis — T_B final runs (R-TB-E-0, R-TB-DK-1, R-TB-K-1)

Generated 2026-10-02T02:30:18Z. No new episode, environment construction, provider call, physical evaluation, formal test, RL run or training step was executed for this report; the only computation was reading existing files, recomputing file hashes, and loading checkpoints to compare weights.

## 0. Result commit and repository state

- Results commit (full SHA): `7c2f58c9844c7a8db4cd5a28fe1615e5630dfeff` — "research: record T_B launch recovery and planned comparisons".
- Branch `codex/cp-disr-final-tb-engineering-launch`, parent `cac2fa5b3cafb46f1f74370314189f4bca4878fa` (source/prep). **Not pushed**: no remote-tracking branch contains the commit, and nothing here claims the remote is in sync.
- The analysis files in this folder (`zero_sample_analysis/`) are new and untracked; they are not part of `7c2f58c98` and nothing was committed after it.
- No training worker or helper process of this task is running. After the analysis all 694 files of the artifact manifest (every file of the three run directories) were re-hashed: 0 missing, 0 changed (`manifest_reverify.json`).

## 1. Comparison table

### Table A — one row per run (existing evaluations only; no episode was run for this table)

Eval points are the pre-registered 0 / 4096 / 8192 / final (Plan v3 §7.1). Success is n of 10 dev10 cases, deterministic argmax. "Frozen discounted return" = recorded `mean_discounted_return` of the eval file (frozen H = 23.1 s). "Final-skill duration" = the recorded `success_seconds` field (see caveat C1); NA = no successful case.

| run | success n/10 at 0 / 4096 / 8192 / final | mean discounted return at 0 / 4096 / 8192 / final | recorded success_seconds (mean over successes) at 0 / 4096 / 8192 / final |
|---|---|---|---|
| R-TB-E-0 B1-K+E s0 | 0 / 0 / 0 / 10 | 0.0000 / 0.0000 / 0.0000 / 0.4893 | NA / NA / NA / 2.915 |
| R-TB-DK-1 B2 s1 | 0 / 10 / 10 / 10 | 0.0000 / 0.4988 / 0.4988 / 0.4988 | NA / 3.540 / 3.540 / 3.540 |
| R-TB-K-1 B1-K s1 | 0 / 0 / 0 / 0 | 0.0000 / 0.0000 / 0.0000 / 0.0000 | NA / NA / NA / NA |

Tables B (budget and compute) and C (every eval point) and the caveats C1-C4 are in `comparison_tables.md`; per-case rows are in `comparison.json`. Caveat C1 matters for the "success time" column: the recorded `success_seconds` is the duration of the final successful skill, not the episode time to success; the latter is not recorded (UNVERIFIED).

## 2. Final checkpoint identity (all three) and DK-1 repeated returns

| run | final checkpoint | sha256 (recomputed now) | equals manifest | manifest N/T/attempt id == job_summary | fresh-load names the final generation | eval_final evaluated this file | persistence copy sha-equal | weights/Adam equal `update_fragment_1` |
|---|---|---|---|---|---|---|---|---|
| R-TB-E-0 | final_n_014512_u14.pt | `f9db3e4ff6dc877dbd4b1b433ac31789e43f6fd45404c35e5103674e4ed4aaba` | True | True | True | True | True | True |
| R-TB-DK-1 | final_n_014170_u13.pt | `b1e331c98d4937b919242912cb3011bb079ec6971a3f8b7b51c682546b7476ca` | True | True | True | True | True | True |
| R-TB-K-1 | final_n_014531_u14.pt | `fa6ca4cf20689aba50c81b52ae2b202772e97cf0a886d74ae58cc22fcb91f385` | True | True | True | True | True | True |

Limit of the fresh-load evidence: the `fresh_load_final_*.json` file records only the generation name and booleans (adam, model, fresh_process), no hash; it is bound to the file by name, by being written after the `.pt` (mtime), and by the generation directory, not by content hash.

DK-1: the three post-update evals (N = 4096, 8192, 14170) used three different checkpoints (distinct sha256; 66 of 141 float tensors differ, max abs diff 4.6e-02 to 1.2e-01) and returned identical per-case numbers (G, steps, reason, success_seconds). The eval rows store no action sequences, so whether the trajectories are identical is **UNVERIFIED**. Details: `dk1_repeated_eval_check.md`.

## 3. Disk below 2 GiB with the guard not pausing

Recorded as an execution deviation, with budget estimate (2051 MiB per run, admission check), actual pause threshold (guard stop below 1.0 GB, which never fired; minimum sampled 1.665 GB at 2026-10-02T00:16:17Z, six 30-s samples below 2 GiB) and the 2 GiB safety margin kept apart. File integrity: all log lines and JSON files parse, transition lines equal N, all checkpoints load, persistence copies agree, no write-error text in logs. No retrain, no overwrite of old logs. Details: `deviation_record_disk_below_2GiB.md`.

## 4. Registration and elastic accounting

- **NONTRIVIAL_LEARNING registered for R-TB-E-0 (B1-K+E, seed 0)** under Experimental Plan v3 §3.1: trigger = the final post-update dev10 file `eval_final.json` (10/10 TASK_SUCCESS; the 4096 and 8192 post-update points were 0/10; step 0 is pre-update and not counted), bound to `final_n_014512_u14.pt` (sha256 `f9db3e4ff6dc877dbd4b1b433ac31789e43f6fd45404c35e5103674e4ed4aaba`, weights and Adam state equal to `update_fragment_1.pt`), the fresh-load file and job_summary by path and sha256. Evaluator independence is established at the schema level only (adapters.EvaluationInput carries no prior, graph or nominal facts); the Evaluator implementation was not re-audited. File: `registration_NONTRIVIAL_LEARNING_R-TB-E-0.json`.
- **Elastic pool**: 4 attempts (ELASTIC-01..04, Plan v3 §7.2). No allocation record or unified ledger exists in this worktree or the 15 sibling worktrees; every ledger found records elastic used 0 (S0 manifest, five S1 ledgers, this card; the three T_B runs are base runs). Remaining by recorded evidence: **4 of 4**; check against an external Master Ledger: **UNVERIFIED**.
- **Request** (not an authorization; nothing started, no token, no ledger change): one run, R-TB-E-1 = B1-K+E seed 1 with the same profile, taking ELASTIC-01 (remaining would be 3). Found in the code: `PLAN_TABLE` is hard-coded to three plans and `MAX_NEW_RL_ATTEMPTS = 3`, so a 4th run needs a launch-entry change, a new prep commit, new registration and a smoke decision (cumulative smoke budget is used up). Options and open decisions are in `authorization_request_R-TB-E-1_ELASTIC-01.md`.

## 5. Conclusion, limited to what the records support

- On this final dev10 (10 cases, deterministic argmax, one run per method, E-0 seed 0, DK-1 and K-1 seed 1): **+E (R-TB-E-0) 10/10, B2 (R-TB-DK-1) 10/10, B1-K (R-TB-K-1) 0/10.**
- It is **not shown that B2 is better than +E, and not shown that the two are equivalent**: both are at the 10/10 ceiling on 10 cases, with different seeds, and the observed differences in mean discounted return (0.4893 vs 0.4988) and recorded final-skill duration (2.915 vs 3.54 s) come from one run each and are not tested.
- Method 2.1.1 is unchanged; Family B and RoboCasa engineering branches were not resumed.

## 6. UNVERIFIED / NOT_MEASURED

- Episode time to success (not recorded; `success_seconds` is the final skill's duration).
- Whether DK-1's three post-update evals took identical trajectories (no action sequences stored).
- Evaluator independence beyond the input schema.
- Elastic usage against an external Master Ledger.
- True minimum free space and exact duration between 30-s samples; per-process attribution of the free-space drop; filesystem quota.
- Fresh-load evidence has no content hash.
- Full repository test suite (not run in this card).

## 7. File index (`zero_sample_analysis/`)

- `authorization_request_R-TB-E-1_ELASTIC-01.json` — sha256 `09419afe1b6a99e5e1a928ee495b690ed2e8a629ab11a596aa530c1ef7bd0fe2`
- `authorization_request_R-TB-E-1_ELASTIC-01.md` — sha256 `af9599d86cce421d5ca3f34812b59f68f446026ee222837b5382e169c9cbc96a`
- `checkpoint_identity.json` — sha256 `326a3f6d27d59938fc369ae6ddd03eebe98aff41f6cb151bbb546403b7368ed3`
- `comparison.json` — sha256 `95d71286635eac4e62e294dadb023098f5f400103ef61a3f4d92173d35709cb8`
- `comparison_eval_points.csv` — sha256 `f66e263a9f468fab283c3af7bc8391025463a487dff9c6dc63d8bef3ec48490e`
- `comparison_tables.md` — sha256 `a01728b9e7027e4b0d9340f8d109b299c106abe759ad387b8fdd5494485ff13b`
- `deviation_record_disk_below_2GiB.json` — sha256 `96c4ea5b9a95b322d6806704cc0dca26ab660c9a9b5867b2348724f5e04eb8bf`
- `deviation_record_disk_below_2GiB.md` — sha256 `6e31f592a9c2a1f3efce4e17a1ea1c8a91159dd1a05e2bc1cf4d3aeb3be77c02`
- `dk1_repeated_eval_check.json` — sha256 `308c3e02610e485166a0101e907c1a99657928888f0d93317dd914791fbfd80a`
- `dk1_repeated_eval_check.md` — sha256 `e98ca8d9e6ad5e03ba5c428822191624895e0fbbde32abc566c4eddd80002eeb`
- `final_vs_fragment.json` — sha256 `0f2eaa117ce2b8e9b30661938790abf2ed09bbf9686ab3fd01843947c1b30c5c`
- `integrity_lowdisk_check.json` — sha256 `386c596aae036b300278e6acad41a082af3135e299e6961afd9d057ea2190a3c`
- `manifest_reverify.json` — sha256 `85449763755534d0f7b86f8e7e8f1d230337ff3e1ca713c653c7db9366bdb278`
- `registration_NONTRIVIAL_LEARNING_R-TB-E-0.json` — sha256 `c1c6e84a364731b9a98da9eb54bc5f3dc2feed08bc6b104873276fdbbe0d386b`
- `weights_identity.json` — sha256 `d9d67f71c3190924be04c6d0efc2ca0649ba6c234723e8ac591206a3fd10c8c7`