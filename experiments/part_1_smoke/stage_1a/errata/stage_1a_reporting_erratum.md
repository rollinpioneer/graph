# Stage 1A reporting erratum

This file is additive. It does not modify Stage 1A raw logs, checkpoints, or PASS.

## Commit identities

- training_source_commit: 271b84050a8c06fd8f06978583a2271d1cc91697
- final_delivery_commit: a9b85a433246d46e1bee8a6164d221ff85453a4a
- base_runtime_commit: d270f373dbeb1eace4bc87112236829bb86cc6e6

`experiments/stage_status/stage_1a.json` field `git_hash` is the training-source
commit of the live runner, not the later delivery commit that archived artifacts.

## Gate table correction

The original `tables/smoke_gate_table.csv` copied conditioned DP/residual rates
from the last train_metrics row. Full update 8 had qualifying_n=0, so those
cells were empty even though updates 1-2 had conditioned rate 1.0.

`corrected_smoke_gate_table.csv` is recomputed from the original
`train_metrics.csv` files (identical to diagnostics.csv).

## Opportunity denominators

B2 vs Full train success episode counts are not a sample-efficiency claim.
See `episode_opportunity_audit.csv`. Stage 1A remains PASS.
Method, prior, and reward are unchanged.
