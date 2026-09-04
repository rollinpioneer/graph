# Stage 8.2 Summary

CORE_PIPELINE_REPRODUCED

- GPU inference jobs: 9 / 9 PASS, all CUDA and checkpoint-SHA verified.
- Ensemble suites: val, test, and stage3 diagnostic; every sample has exactly three seed predictions.
- Comparable main-table cells: 96 / 96 PASS.
- Three frozen reference event-rate cells are explicitly `not_comparable`: the frozen Stage-5 display routine recomputed them with the locked-full engine instead of the listed method options.
- Core ablations were newly recomputed from real test predictions and frozen controlled traces. The R1 mixed-explicit ablation display table is retained only as `not_comparable`, never copied into results.
