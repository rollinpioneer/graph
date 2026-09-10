# L2RA-R2 R11 Loss Observability Audit

- Entry/source commit: `668e581b0de9e60373f64107aa15b7e3b8c92b3a`
- Cache: 32 rollouts, 12 unique K4/K5/K6 loss events; baseline replay is in `artifacts/pathgraph_sarm/upgrade_v2/loss_observability_l2rar2_r11_v1/baseline_replay`.
- Physical diagnostic executions: `8`; no training and no API calls.
- Reference labels: legacy labels remain unchanged; physical loss semantics are not inferred from `contact_lost` alone.
- Primary route: `LOSS_REALIZATION_REPAIR_FIRST`

## Verified facts

- The fixed candidates remain `B_count2` and `C3_vector_rho035`; no candidate was selected.
- The baseline replay preserves K1 recall 1.0, K4/K5/K6 recall 0.0, zero listed negative false-emergency rates, and accuracy 0.625.
- Cache evidence contains RGB paths/hashes, contact proxy, commands, requests, oracle positions and events, but not object-specific contact force pairs, qvel history, or writeback counts.

## Mechanism inference

The route is selected only from the bounded probe when its ordinary and instrumented replays are equivalent. The probe is diagnostic evidence, not a new family, training sample, confirmation run, or online candidate score.
For the probed root, K4/K5/K6 all retained object-to-`finger_left`/`finger_right` contact force after weld-off, with no post-detach writeback increase; they are therefore `not_verified` as task-level losses, despite the legacy `contact_lost` event.

## Remaining gaps

- Existing reference labels were not rewritten and the frozen `0.02 m` / `0.01 m` contract was not tuned.
- Any unresolved physical or sensor ambiguity remains explicitly unresolved; no null was converted to false or zero.

historical_status = L2RAR1_PARTIAL_KEEP_G1
scientific_status = L2RAR2_PARTIAL_KEEP_G1
retained_graph = G1_predicate_bound
selected_candidate_id = null
confirmation_run = false
l3_entry_allowed = false
