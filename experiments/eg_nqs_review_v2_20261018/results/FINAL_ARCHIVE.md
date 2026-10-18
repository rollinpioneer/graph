# Final archive

The authoritative final report is `final_confirmation_report.md`; interpretation of the development/confirmation reversal is in `generalization_gap_analysis.md`. Earlier development reports and scheduling receipts are historical snapshots, not the final status.

All 24 scene-model jobs (6,144 frame-model evaluations) and three postprocessing steps passed. The queue started at 8 workers, then its limit was raised to 32; 20 unfinished jobs ran concurrently at the transition. No 32-worker peak was reached. Scientific settings and existing workers were preserved.

`final_artifact_manifest.json` hashes the original server result files before publication sanitization. It also indexes retained large per-frame tables, figures and detailed case records. Published resource summaries omit process identifiers; published case paths are relative to the experiment root, so those sanitized bytes may differ from server hashes. `publication_file_hashes.json` separately hashes published payload files (excluding itself).

Large checkpoints, frozen predictions, geometry caches, raw RGB/depth case assets and complete logs remain in the experiment directory. The frozen checkpoint identities are in `confirmation_freeze.json` and `all_head_fit_summary.csv`. No new training was launched during final analysis. Additional same-frame sampling and fixed-pool comparisons are descriptive post-hoc checks, not replacements for registered full-frame results.
