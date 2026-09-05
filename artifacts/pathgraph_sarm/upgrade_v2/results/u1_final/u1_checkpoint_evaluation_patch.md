# V2 checkpoint evaluation patch

The four checkpoint rows that differed from the historical v1 selection CSV
were loaded with PyTorch 2.8.0 on CPU using strict `ValueModel` state-dict
loading, then forwarded on all 915 recovery validation anchors.  The exact
hashes, payload metadata, input shape, head availability, and label counts are
recorded in `checkpoint_forward_reverification.json`.

`mechanism_validation_metrics_v2.csv` contains the resulting seven descriptive
rows.  It is intentionally additive: the historical
`mechanism_validation_metrics.csv` is retained for provenance rather than
silently overwritten.

The v2 values are: cost-only seed 611 D-MSE 0.007512465/D-MAE 0.063663764;
seed 612 0.007505614/0.063719768; seed 613 0.007684571/0.066255141.
Success-only seed 611 q-Brier is 0.068483353.  These remain mechanism-only
validation metrics and do not establish physical or independent generalization.
