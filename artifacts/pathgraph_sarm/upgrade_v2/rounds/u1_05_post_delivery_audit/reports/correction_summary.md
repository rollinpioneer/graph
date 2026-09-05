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
18 currently present checkpoint hashes, but the current `python` cannot import
PyTorch, so loading these four replacement weights and recomputing their
validation metrics remains pending.  Existing validation rows must therefore
not be presented as evaluations of the v2 selection for those four runs.

U0 topology scoring now uses non-negative Dijkstra and derives fixed chains
from GraphSpec start-to-success paths, not classifier label order.  A full
legacy checkpoint rescore is likewise pending a Python runtime with PyTorch.
Neither pending operation changes the mechanism-only scope or makes U2
eligible.
