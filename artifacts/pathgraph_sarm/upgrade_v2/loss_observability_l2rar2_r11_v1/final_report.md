# L2RA-R2 R11 Loss Observability Audit

- Entry/source commit: `668e581b0de9e60373f64107aa15b7e3b8c92b3a`
- Cache: 32 rollouts, 12 unique K4/K5/K6 loss events; baseline replay is in `artifacts/pathgraph_sarm/upgrade_v2/loss_observability_l2rar2_r11_v1/baseline_replay`.
- Physical diagnostic executions: `8` in the last probe invocation; cumulative executions recorded: `40`; budget `8`; status `BUDGET_EXCEEDED_BLOCKED`.
- Reference labels: legacy labels remain unchanged; physical loss semantics are not inferred from `contact_lost` alone.
- Primary route: `INSUFFICIENT_EVIDENCE_STOP`; audit status: `BLOCKED_DIAGNOSTIC_BUDGET_EXCEEDED`.

## Verified facts

- The fixed candidates remain `B_count2` and `C3_vector_rho035`; no candidate was selected.
- The baseline replay preserves K1 recall 1.0, K4/K5/K6 recall 0.0, zero listed negative false-emergency rates, and accuracy 0.625.
- Cache evidence contains RGB paths/hashes, contact proxy, commands, requests, oracle positions and events, but not object-specific contact force pairs, qvel history, or writeback counts.

## Mechanism inference

No physical mechanism claim is authorized. The ordinary/instrumented probe had matching action, control, and event streams, but all four probe cases failed the cached action-end geometry equivalence check. In addition, the cumulative physical diagnostic count exceeded the hard budget. The probe is therefore retained as an audit trail only; it is not used to relabel legacy events, select a repair route, or claim that K4/K5/K6 were or were not physical losses.

## Remaining gaps

- Existing reference labels were not rewritten and the frozen `0.02 m` / `0.01 m` contract was not tuned.
- Any unresolved physical or sensor ambiguity remains explicitly unresolved; no null was converted to false or zero.
- The 32 cached rollouts and 12 unique K4/K5/K6 events remain the scientific cache evidence. No training, API calls, confirmation run, candidate selection, or L3 entry occurred.
- Physical execution accounting is `40` used against a maximum of `8`; this is a blocked execution record, not a scientific gain.

historical_status = L2RAR1_PARTIAL_KEEP_G1
scientific_status = L2RAR2_PARTIAL_KEEP_G1
retained_graph = G1_predicate_bound
selected_candidate_id = null
confirmation_run = false
l3_entry_allowed = false
