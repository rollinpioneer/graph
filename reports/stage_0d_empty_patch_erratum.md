# Stage 0D empty-patch statistical erratum

Status: historical `empty_patch_rate=1.0` is **UNRELIABLE/NOT_RECONSTRUCTABLE**.

Cause: `sample_action()` returned integer candidate indices in `nonempty`, but `run_episode()` used `cid not in nonempty` with string canonical IDs. That comparison is a tautology and counted every selected action as an empty patch.

This file does **not** overwrite `random_episode_log.jsonl`, `decision_structure.csv`, `stage_0d_result.json`, or original success records. Episodes were **not** rerun.

Correct denominators going forward:
- selected-action empty patch rate: selected actions whose nominal contract patch is empty / decisions
- legal-candidate empty patch rate: legal candidates whose nominal contract patch is empty / legal candidates
- empty prior, empty nominal patch, DP==0, and missing measurement are distinct
- missing source cache is not a legal empty prior

Offline reconstruction: **not possible**. The slim episode log dropped per-decision `n_nonempty_patch` / index identity. `n_empty_patch` equals `n_decisions` on all 200 rows.

Original uniform successes remain the exposure record: T_B 13/100, T_C 35/100.
