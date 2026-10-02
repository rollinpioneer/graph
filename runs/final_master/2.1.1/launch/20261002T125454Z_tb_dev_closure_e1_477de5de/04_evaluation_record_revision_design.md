# Design: unified evaluation record revision (observation and recording only)

Status: DESIGN ONLY. No source file is changed by this record, no environment is started, no episode is run. Implementation, offline tests and any replay need a separate approval.

## 1. What the existing record can and cannot say (verified in the source and the eval files)

- `eval_episodes` (`src/cp_disr/stage2a_v11.py`, line 1009 onward) writes one row per case with: `case_id, success, G, steps, reason, success_seconds, source_n, prior_mode`.
- `success_seconds` (line 1044) is `t.snapshot.elapsed_seconds + t.duration` only if the snapshot has `elapsed_seconds`; `rl.Snapshot` (`src/cp_disr/rl.py`, line 74) has `clock_seconds` and no `elapsed_seconds`, so the fallback `t.duration` is always taken. The recorded value is therefore the duration of the last (successful) skill only. Example from the existing R-TB-E-1 final evaluation: 5 skill steps per successful case, recorded success_seconds 2.90 to 2.95 s, while the episode contains several skills. Label kept for the old field: FINAL_SKILL_DURATION_ONLY. Episode-level time to success: UNVERIFIED in every existing file.
- The action sequence is not stored: rows keep `steps` (a count). `Transition.selected_candidate_id` exists in memory (`rl.py`) but `eval_episodes` does not write it.
- The evaluated model is identified only by the checkpoint path; the eval payload's `hashes` block holds source-code hashes, not the checkpoint file hash.
- The quantity that is needed already exists in the loop: `Collector.step` (`src/cp_disr/collector.py`, line 187) computes `elapsed = clock_end - bundle.episode_start_seconds` and hands it to the independent evaluator in `EvaluationInput(..., elapsed, start, end)`; `actual.success` is that evaluator's confirmation (the guard at line 195 rejects a reward without independent success). Precedent for the right definition: `phase_a_v13_r3.py` (line 439) and `stage1a_v11_final_eval.py` (line 626) record `bundle.clock.now_seconds() - bundle.episode_start_seconds`.

## 2. Target record (schema `eval_row_v2`, additive)

Per episode row, in addition to every existing field (unchanged, same values):

| field | definition | source (existing objects only) |
|---|---|---|
| `episode_start_clock_s` | simulation clock at episode start | `bundle.episode_start_seconds`, read after `start_case` |
| `time_to_first_confirmed_success_s` | simulated seconds from episode start to the end of the first interval in which the independent evaluator confirmed success; null if none | `t.next_snapshot.clock_seconds - bundle.episode_start_seconds` at the first step with `result.success` (equals Collector's `elapsed` for that step) |
| `time_to_first_confirmed_success_label` | `EPISODE_START_TO_FIRST_INDEPENDENT_CONFIRMED_SUCCESS_SIM_SECONDS` | constant |
| `episode_end_elapsed_s` | simulated seconds from episode start to the last interval end | same, at the last step |
| `actions` | ordered list, one entry per executed decision: `decision_index`, `selected_candidate_id`, `clock_start_s`, `clock_end_s`, `duration_s`, `reward`, `weight`, `exit_reason`, `terminated`, `truncated` | `t.snapshot.decision_id`, `t.selected_candidate_id`, `t.snapshot.clock_seconds`, `t.next_snapshot.clock_seconds`, `t.duration`, `t.reward`, `t.weight`, `t.reason` |
| `success_seconds` | unchanged | old computation |
| `success_seconds_label` | `FINAL_SKILL_DURATION_ONLY` | constant, also written for old-style rows by any reader |
| `no_transition_exit` | reason and elapsed when the loop ends through `t is None` (for example INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE) | the returned result dict |

Payload level: `schema_version = eval_row_v2`; `evaluated_model = {path, sha256, bytes}` with the sha256 computed from the file bytes before loading and checked again after the last episode; `evaluated_generation` with N, T and update from the generation manifest; the existing `hashes` (source) block; `case_order`; `eval_rng_isolation` flag; `label` (dev10 / common_dev20 / test).

## 3. Not changed

Reward, discounted return G and weights, action selection (deterministic argmax), termination and deadline logic (T_B deadline 60 s), safety / verifier / evaluator code, the training loop, checkpoint format, case order and the evaluation RNG isolation. `Collector.step` keeps its signature and return values.

## 4. Where it would live (proposal, not applied)

- Option A (preferred): a new evaluation-only module (for example `src/cp_disr/final_tb_eval_record.py`) that makes the same calls in the same order as `eval_episodes` (`seed_all(0)`, `make_policy`, `load_checkpoint`, `start_case`, `apply_episode_prior`, `reset_episode`, `step(..., deterministic=True)`) and only adds the recording above. `stage2a_v11.py` and `collector.py` stay byte-identical, so the execution-path hashes carried by every training manifest (`stage2a_v11`, `collector`) do not move and nothing in a possible further training run changes identity.
- Option B (not recommended): add a write-only attribute inside `Collector.step` or edit `eval_episodes`. It changes the `collector` / `stage2a_v11` source hashes shared with training.

## 5. Offline test plan (no environment, no episode)

1. Field arithmetic with scripted fake transitions and a fake bundle clock: 5-step success, success on the last allowed step, deadline exit, no-candidate exit, INSUFFICIENT_REMAINING exit.
2. Old-field invariance: `success`, `G`, `steps`, `reason` and the old `success_seconds` computed by the new module equal the old computation on the same scripted input.
3. Cross-check: sum of `duration_s` over actions versus `time_to_first_confirmed_success_s`; any difference is written as `inter_step_clock_gap_s`, never silently assumed zero.
4. The last action's `duration_s` equals the old `success_seconds` on successful rows.
5. Model hash: sha256 before load equals sha256 after the last episode; mismatch aborts the evaluation.
6. Static check (AST / hash): `collector.py`, `rl.py` (gamma, interval_reward) and `stage2a_v11.py` hashes unchanged; the new module imports no optimizer and writes no training artifact.
7. Reader test: rows without the new fields load and are labelled FINAL_SKILL_DURATION_ONLY; they never enter a time-to-success column.

## 6. Order of work (all separate approvals)

(a) implement the module and tests, no episodes; (b) optional consistency replay of the frozen final models on the already-used dev10 to compare success, G, steps and the old success_seconds with the existing rows (this runs episodes and is not part of this record); (c) only then a test release. Existing evaluation files are not rewritten and cannot be completed retroactively: the action sequences were never stored, so their episode-level time stays UNVERIFIED.
