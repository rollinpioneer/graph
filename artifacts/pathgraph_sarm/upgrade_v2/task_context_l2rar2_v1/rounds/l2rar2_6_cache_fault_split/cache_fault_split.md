## Material Passport

- Artifact type: cached-experiment validation report
- Verification status: `ANALYZED` (read-only cache recomputation; no physical re-run)
- Data role: frozen `dev_select` diagnostic cache
- Scientific boundary: `L2RAR1_PARTIAL_KEEP_G1`; L3 remains closed

# L2RA-R2 cache fault split

- Status: `CACHE_DIAGNOSTIC_COMPLETE`; retained graph: `G1_predicate_bound`.
- Scope is read-only cached data. Physical rollouts re-executed: `0`; training jobs: `0`; API calls: `0`; confirmation: not run.
- Target scope: four rollouts from families `05` and `07`, cases `K2` and `K8`; full per-frame trace is in `target_hold_trace.csv`.
- Reference scope: all `19` rows from the original `reference_unresolved.csv` remain in the denominator and are preserved verbatim by status/reason.

## Findings

- Frozen `B_count2` marks the initial transient contact as hold evidence in both K2 and K8. K2 clears that memory at loss and never re-establishes it; K8 shares the same early false-hold prefix, then later re-establishes hold after the second contact while `attempt_active=true`.
- All 19 unresolved rows have intact dense/action-end alignment and online hold evidence. Their oracle relative drift exceeds the locked `0.02 m` limit by `0.000305-0.001825 m`; they are reported as `physical_proxy_present_but_outside_frozen_condition`.
- The result is a two-fault split: K2/K8 expose an online short-contact false-hold boundary, while the 19 unresolved rows expose a reference-proxy/collection-geometry boundary. It does not yet authorize changing either contract, deleting unresolved rows, relaxing the physical reference, disabling recovery, expanding candidates, retraining, or confirmation.

## Files

- `target_hold_summary.csv`: first online evidence/memory and offline event alignment for K2/K8.
- `target_hold_trace.csv`: per-observation image/centroid/confidence/contact/phase/evidence trace.
- `visual_frame_review.csv`: manual review of selected source frames; it is diagnostic evidence only.
- `reference_unresolved_split.csv`: all original unresolved rows with diagnostic-only classification.
- `cache_fault_split.json`: machine-readable scope, counts, decision boundary, and provenance.
