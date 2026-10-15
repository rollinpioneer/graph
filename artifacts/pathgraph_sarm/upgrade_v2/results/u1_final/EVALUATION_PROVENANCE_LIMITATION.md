# Validation provenance limitation after checkpoint reconciliation

`u1_selected_checkpoints.csv` is retained as the original v1 selection record.
It differs from the currently present `best.pt` recorded by four formal
recovery jobs.  The checksum-locked replacement record is
`registry/u1_selected_checkpoints_v2_hash_locked.csv`.

The four v2 replacement checkpoints have now been loaded and forwarded on the
recovery validation split.  Their additive results are in
`mechanism_validation_metrics_v2.csv`, with exact load/forward provenance in
`checkpoint_forward_reverification.json`.  Do not replace the historical v1
rows in `mechanism_validation_metrics.csv`; use the v2 file for those four
v2-selected checkpoints.  The mechanism-only/no-independent-physical-test
limitation remains.
