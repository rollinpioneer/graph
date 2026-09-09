# L2RA-R2 delivery scope

- Scientific status: `L2RAR2_PARTIAL_KEEP_G1`.
- Retained graph: `G1_predicate_bound`.
- L3 entry: closed.
- Development: failed on the frozen development gates; no candidate was locked.
- Focused confirmation: not run because development failed; confirmation seeds remain unconsumed.
- Standard L2R confirmation: `NOT_RUN_OUT_OF_SCOPE`.
- Old R4: `NOT_RUN_PRESERVED`.
- New physical rollouts: 96 (32 fit, 64 select).
- Training jobs: 0; API calls: 0; API key reads: false.
- Raw rollout payloads, RGB frames, large arrays, checkpoints, and credentials are externalized and are not in this lightweight package.
- The six `*_rev2.zip` files under `round_packages/` are retained historical archive bodies. The current delivery uses six `*_rev3.zip` packages generated from the corrected evaluation and lock state. Their scientific meaning is defined by the manifests and reports, not by archive integrity alone.
- K8 was regenerated with `end_action=verify`; the pre-correction lock is retained outside the current lock as an explicit comparison artifact. The current evaluation keeps all 64 physical rollout opportunities and marks 19 reference-unresolved rollouts as `NOT_ESTIMABLE` rather than dropping them from denominators.
- The implementation and delivery metadata commit is assigned after artifact freeze; the package index records the resulting commit and archive hashes.
