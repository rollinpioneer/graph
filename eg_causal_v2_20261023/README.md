# EG causal diagnosis v2 and R1.4

This archive contains the completed first-stage causal mechanism diagnosis and the authorized R1.4 controlled micro-update experiment.

- First-stage report: [causal_summary.md](causal_summary.md)
- R1.4 report: [r14/R14_summary.html](r14/R14_summary.html)
- Main comparisons: [primary_effects.csv](r14/analysis/primary_effects.csv)
- Execution audit: [final_receipt_audit.json](r14/data/final_receipt_audit.json)
- Implementations: `implementation/` and `r14/code/`
- Exact source-2 and remaining-source code versions: `r14/code_versions/`

R1.4 completed 280 Adam updates across 56 branches, 448 evaluation jobs, and 62,181 direct analytic evaluations (including 8 preflight evaluations), with zero unknown labels. Decision: **HOLD**. The targeted intervention did not show a stable benefit over count-matched and actual-step-norm-matched controls. The 1/5-step experiment does not establish absence of long-term effects.

Numerical replay failures and implementation clarifications are retained explicitly. C3/C4/C5 were not accessed, and 1000-step training was not performed.

`git_archive_manifest.json` records file hashes and omitted artifacts. Model/data/cache binaries, transport archives, transient reservation files, and raw tables exceeding 5 MB remain local; this is not a self-contained dataset or checkpoint distribution. All archived scientific outcomes were produced before this Git submission.
