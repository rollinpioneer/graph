# L2R Final Report

- Decision: `L2R_PARTIAL_KEEP_COARSE_GRAPH`
- L1V visual contribution: preserved from the explicit V1 rerating; not re-estimated here.
- Predicate binding contribution: B3 observable interface was frozen on development data and reused unchanged on fresh families.
- Structural edit contribution: 5 development edits were accepted; `G2_evidence_refined` was selected on development and evaluated on fresh families. Refinement support is `false`, so the retained graph is `G1_predicate_bound`.
- Active second-view contribution: not selected.
- Predicate metrics: goal F1=0.9491525423728813, stable-hold F1=1.0, failure F1=1.0, recovery F1=1.0, unknown=0.08640524346343317.
- Fresh selected metrics: branch accuracy=0.9583333333333334, goal precision=1.0, failure recall=1.0/24, recovery recall=1.0/24, coverage=0.9639136904761911.
- Stop reasons: GO gate not met (ambiguous_edge_rate<=0.10); selected-G0=0.708333, selected-G1=0.583333, noninferior=8/8.
- Dynamic evidence: MuJoCo primitive tabletop only.
- Unsupported: reward gain, policy gain, physical robot success, and new-task generalization.
