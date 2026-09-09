## Material Passport

- Artifact type: cached-experiment mechanism localization
- Verification status: `ANALYZED` (cache-only; no physical re-run)
- Historical boundary: `L2RAR1_PARTIAL_KEEP_G1`
- Current R2 boundary: `L2RAR2_PARTIAL_KEEP_G1`; retained graph: `G1_predicate_bound`; L3 remains closed

# L2RA-R2 mechanism localization

## Online mechanism

- Compared `4` initial K2/K8 false-hold segments, `2` later K8 hold segments, and `4` reference-labeled brief-hold examples; `3` have a `B_count2` onset and `1` does not.
- Every initial false segment contains a sub-noise interval: `True`. A direction disagreement is present in at least one initial false segment: `False`.
- Same-timestamp control/action-end rows are counted twice by the merged dense stream: `False`. The collector keeps one post-update row per timestamp; action-end participation and duplicate counting are therefore distinct questions.
- Every observed later-K8/brief-hold onset contains at least one directionally consistent effective-motion interval: `True`.
- This does not establish a complete replacement rule: one frozen reference-labeled brief hold has no `B_count2` onset. The cache localizes the early false-positive mechanism, but a future hypothesis must also recover that missed positive without reopening short-touch false holds.

## Reference mechanism

- Preserved all `19` unresolved rows and the frozen `0.02 m` limit. First peak-stage counts: `{"constraint_establishment": 12, "loss_boundary": 1, "sustained_hold": 6}`.
- First peak action contexts: `{"lift": 12, "transport_to_target": 7}`; `16/19` peaks persist at the maximum after first attainment, so a plateau is not reported as a one-frame spike.
- Relative position remains `relative_position_world_m = object_xyz_world_m - gripper_xyz_world_m; drift_m = L2(relative_position_world_m[t] - relative_position_world_m[first_weld_row])`.
- `reference_drift_peaks.csv` records anchor and peak vectors, first/last peak times, peak stages, and offline action context. No row was converted to a reference pass.
- Any future geometry or reference change must use a new version and be reported beside this frozen result; it cannot be counted as online algorithm gain.

## Decision boundary

- No candidate expansion, retraining, API/key access, new sampling, reference relaxation, confirmation, or L3 entry was performed or authorized.
- Online predicate work and reference-interface work remain separate validation tracks.
