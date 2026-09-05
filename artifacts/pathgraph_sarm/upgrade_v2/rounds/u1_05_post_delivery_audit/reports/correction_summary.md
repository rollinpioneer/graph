# U0/U1 post-delivery correction status

The U1 target rebuild consumes the collection-time `success` and `failed`
fields from the Stage6 episode manifest.  It does not infer either event from
the sign of `terminal_reward`.  All 5,725 existing anchors retained the same
q/D supervision values, so no U1 retraining is warranted solely by this label
provenance repair.

`train-value` and `evaluate-u1-mechanism` now apply q/D masks independently.
Unknown short continuations are right-censored and excluded from a head's
complete-supervision loss and metric.  The regression test supplies a short,
negative-reward record without outcome evidence and verifies that it stays
masked rather than becoming an irrecoverable failure.

The former selection CSV disagrees with current formal `job_result.json` in
four recovery runs.  The v1 CSV is retained as history.  The v2 CSV locks the
18 currently present checkpoint hashes.  A PyTorch 2.8.0 CPU runtime strictly
loaded and forwarded all four replacements on 915 recovery validation anchors;
the seven resulting rows are in `results/u1_final/mechanism_validation_metrics_v2.csv`
and their provenance is in `checkpoint_forward_reverification.json`.

U0 topology scoring now uses non-negative Dijkstra and derives fixed chains
from GraphSpec start-to-success paths, not classifier label order.  The legacy
checkpoint was successfully rescored on CPU over all 380 readable episodes;
the baseline v3 tables and report are in `results/u0_corrected_v3/`.  These
completed repairs freeze the U0/U1 mechanism version.  They do not change the
mechanism-only scope or make U2 eligible.
