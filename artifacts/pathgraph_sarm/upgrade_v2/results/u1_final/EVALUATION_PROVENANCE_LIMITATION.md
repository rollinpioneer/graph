# Validation provenance limitation after checkpoint reconciliation

`u1_selected_checkpoints.csv` is retained as the original v1 selection record.
It differs from the currently present `best.pt` recorded by four formal
recovery jobs.  The checksum-locked replacement record is
`registry/u1_selected_checkpoints_v2_hash_locked.csv`.

Until a Python runtime with PyTorch loads the replacement checkpoints and
reruns `evaluate-u1-mechanism`, do not treat the existing
`mechanism_validation_metrics.csv` rows for those four runs as v2-selected
checkpoint metrics.  This limitation is in addition to the existing
mechanism-only/no-independent-physical-test limitation.
