# L2R Final Report

- Decision: `STOP_VISUAL_REFINEMENT`
- L1V visual contribution: preserved from the explicit V1 rerating; not re-estimated here.
- Predicate binding contribution: B3 observable interface was frozen on development data and reused unchanged on fresh families.
- Structural edit contribution: 5 development edits were accepted as G2 proposals, but the selected graph was `G1_predicate_bound` and therefore did not use them in confirmation.
- Active second-view contribution: not selected.
- Predicate metrics: goal F1=0.9491525423728813, stable-hold F1=1.0, failure F1=1.0, recovery F1=1.0, unknown=0.08640524346343317.
- Fresh selected metrics: branch accuracy=0.375, goal precision=1.0, failure recall=0.0/24, recovery recall=0.0/24, coverage=0.7700000000000012.
- Stop reasons: failure and recovery recalls are both below 0.40 (0.0, 0.0).
- Dynamic evidence: MuJoCo primitive tabletop only.
- Unsupported: reward gain, policy gain, physical robot success, and new-task generalization.
