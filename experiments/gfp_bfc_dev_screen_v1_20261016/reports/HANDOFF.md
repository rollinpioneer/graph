# Final handoff — GFP-BFC-DEV-SCREEN-V1

Status: COMPLETE. Read FINAL_REPORT.md and final_validation.json. All 12 training jobs and 24 valid evaluation conditions exited 0. Nine original evaluation-reporting failures remain archived; their authorized repairs all succeeded. No active experiment jobs remain. Do not restart any manager or training.

Decision: NO-GO for BFC V1 advancement. Both BFC settings failed the consecutive-anomaly screen in both windows and did not beat the best tested EMA on average. Pure slow EMA only succeeds in one window; no uniformly stable fix is established. No follow-up training is authorized.

Budget: 1,200,000 additional optimizer updates; 6,123,357 physical control calls including all nine failed evaluations and 80 preparation calls, within the authorized 6,750,000 cap. All 36 checkpoint hashes/full-restore receipts pass; source hashes unchanged. All matched initial observation hashes agree. Five repaired evaluations differ in total rollout length from the original failed execution; exact trajectory replay is unverified and its cause unresolved. Do not hide this limitation or interpret small differences as reliable advantages.

Artifacts: FINAL_REPORT.md, summary.json, checkpoint_index.json, mechanism_diagnostics.json, paired_comparisons.json, costs.json, diagnostics.png. Full weights and trajectory paths remain on server. code/finalize.py is read-only analysis, no model updates or environment interaction. Original experiment card/protocol remain unchanged; ALL_PARALLEL_AMENDMENT.md records the approved scheduling and evaluation-budget amendments.

Branch: codex/gfp-bfc-dev-screen-v1-20261016. Latest independently verified remote SHA: receipts/archive_verified.json. Pause lab after final delivery. This completion does not authorize normal-window, from-scratch, longer or task4 experiments.
