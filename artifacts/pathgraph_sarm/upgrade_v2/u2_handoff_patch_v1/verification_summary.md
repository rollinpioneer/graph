# U2 correction and U3-entry verification

- Date: `2026-09-05`
- Scope: the same explicit-state stochastic simulator family only. No physical robot, original-task, or unseen-family generalization is claimed.
- Boundary minimal tests passed: exact matching, cross-episode counterexample, and input-order invariance.
- Reward minimal test passed: incoming-transition segment allocation conserves `phi[T-1] - phi[0]`; formal-cache maximum partition residual is `0.0`.
- Boundary recomputation: `BOUNDARY_CACHE_RECOMPUTED`; 25 cached model/rule sources, 5,400 model-episode evaluations, and 0 missing predictions.
- Locked-source test F1±2: historical `0.5878339965889711`; corrected `0.5935190449118818`. This is a corrected re-evaluation of the locked source, not a new checkpoint selection.
- Reward recomputation: `REWARD_CACHE_RECOMPUTED`, test-only family bootstrap with `5,000` resamples; event direction uses each event's stored incoming potential transition; `closed_full_input_cycle_residual=not_measured`.
- Train-only check passed: 504 episodes, 84 root families, 2,651 segments, and 2,147 adjacent transitions. Val/test contributes 0 prompt rows; root-family split overlap is 0.
- Unknown values are retained; prototype support counts are separately recorded as `support_root_families` and `n_segments`.
- Gold budget: shared calibration is 8 root families / 1,108 frames; extra fallback is 30 unique train clips / 135 unique frames.
- Candidate requests: 9 across three conditions; completed candidate count is 0; no raw response or parsed graph exists; status is `MODEL_EXECUTION_PENDING`.
- CLI smoke tests passed for `recompute-boundaries`, `recompute-reward`, `export-u3-train`, and `freeze-handoff`.
- The single cumulative archive is rebuilt at `downloads/upgrade_v2/U0_U1_complete.zip`; raw NPZ/PT/PTH/Parquet/JSONL/log payloads are excluded or represented by placeholders. The lightweight packaging function was used directly because the older umbrella U2 CLI imports an optional training dependency (`torch`) before dispatching a package-only command.
