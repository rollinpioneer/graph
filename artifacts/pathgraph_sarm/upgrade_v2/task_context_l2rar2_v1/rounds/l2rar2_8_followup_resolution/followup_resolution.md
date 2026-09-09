## Material Passport

- Artifact type: bounded cached evidence and simulator-mechanism diagnostic
- Verification status: `ANALYZED`; not an independent confirmation
- Current boundary: `L2RAR2_PARTIAL_KEEP_G1`; retained graph: `G1_predicate_bound`; L3 closed

# L2RA-R2 follow-up resolution

## K5 evidence chain

- Target: `L2RAR2_SELECT_04_820004:K5_brief_hold_loss`. Physical loss is recorded at `0.65s`; the arrived contact stream first reports loss at `0.75s`.
- `B_count2` reaches a maximum raw stable run of `1` and therefore has no onset before physical loss.
- Existing `C3_vector_rho035` and `C3_vector_rho055` both establish causal hold evidence during the lift before loss. The missing B onset is caused by the two-consecutive magnitude gate, not by absence of all online directional evidence.

## Fixed existing-rule comparison

- `B_count2`: `{"initial_false_hold_segments_total": 4, "initial_false_hold_segments_with_evidence": 4, "later_k8_segments_total": 2, "later_k8_segments_with_evidence": 2, "reference_labeled_k5_total": 4, "reference_labeled_k5_with_evidence": 3}`.
- `C3_vector_rho035`: `{"initial_false_hold_segments_total": 4, "initial_false_hold_segments_with_evidence": 0, "later_k8_segments_total": 2, "later_k8_segments_with_evidence": 2, "reference_labeled_k5_total": 4, "reference_labeled_k5_with_evidence": 4}`.
- `C3_vector_rho055`: `{"initial_false_hold_segments_total": 4, "initial_false_hold_segments_with_evidence": 2, "later_k8_segments_total": 2, "later_k8_segments_with_evidence": 2, "reference_labeled_k5_total": 4, "reference_labeled_k5_with_evidence": 4}`.
- All three use the same R2 cache and the same `M1_requested_effect_gate`. No parameter search or method renaming was performed.
- `C3_vector_rho035` is not a complete candidate: on the unchanged M1 interface its K1 recall is `0.0` because no-contact endpoint evidence is `unknown`, producing `needs_observation` instead of retry.

## Simulator/reference consistency

- MuJoCo version: `3.4.0`. Diagnostic simulator instances: `16`; new dataset rollouts: `0`.
- The XML weld has no explicit relative pose; `_attach()` activates the compiled equality unchanged; `_advance()` writes the object to `gripper + [0,0,-0.13]` before every physics step.
- The current mechanism probe exactly matches the first cached post-anchor relative position in `47/48` weld-bearing records. The sole non-exact K8 row differs by `1.6602647e-05 m` and has the same drift direction; all 19 unresolved rows are exact matches.
- The relpose-at-attach counterfactual stays under the unchanged `0.02 m` limit in `48/48` records. This is diagnostic support, not a relabeling of old rows.

## Single repair decision

- Decision: `SIMULATOR_REFERENCE_REPAIR_FIRST`.
- In a new version only, set the weld target to the actual attach pose before activation, keep the existing scripted writeback and the frozen reference threshold, and recollect independent development roots. Old records and labels remain unchanged.
- Existing `C3_vector_rho035` remains a diagnostic explanation for the specified hold segments, not a complete online candidate: it fails K1 through the unchanged M1 unknown-state path.
- No candidate expansion, training, API/key access, confirmation, reference relaxation, unresolved-row deletion, or L3 entry occurred.
