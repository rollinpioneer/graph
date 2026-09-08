# L2RA Final Report

- Status: `L2RA_DIAGNOSIS_ONLY`
- Historical L2R status: `L2R_PARTIAL_KEEP_COARSE_GRAPH`
- Retained graph: `G1_predicate_bound`
- New candidate: `None`
- API calls: `0`; training jobs: `0`; API key read: `false`

## Scope

The evidence is limited to offline replay and the bounded dynamic tabletop proxy. It does not establish real-robot execution, reward improvement, policy improvement, or task generalization.

The new development set contains 24 root families and 96 rollouts. Repeated deterministic predicate streams collapse to 19 content groups; root family, not rollout or frame, is the statistical unit, and repeated content is not treated as independent evidence.

## Diagnosis

Frozen G2 replay reproduced 12 ambiguous rollouts among 96 legacy confirmation rollouts (0.125), all in `slip_then_recover`. The frozen rules make observed slip a subset of generic grasp failure, while the whole-sequence executor reports both guards even when priority selects recovery.

## Gate

- Diagnosis route: `GUARD_SEMANTICS_REPAIR`
- Development route: `DEVELOPMENT_NOT_READY`
- Five selectable configurations were evaluated; none passed all predeclared development gates.
- Standard confirmation: `NOT_RUN`
- Challenge confirmation: `NOT_RUN`
- L3 entry allowed: `false`
