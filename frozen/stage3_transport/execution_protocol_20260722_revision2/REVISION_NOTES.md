# Stage 3 Transport-MH Execution Protocol Revision 1

Revision time: 2026-07-22, before the final checkpoint, fixed 100-rollout
pool, TRAK scores, and Transport stability results existed.

Parent snapshot:
`execution_protocol_20260722/all_files_sha256.txt`
SHA-256 `daac6b79ce50134aff27cd9e531dcc0665b43045aadb61629151c4c8c6821407`.

This revision preserves all training, rollout, TRAK, seed, model, and gate
parameters. It makes only the following pre-result audit and estimator fixes:

1. `audit_final_training.py` now loads the checkpoint on CPU, verifies the
   internal epoch and global step against the final log, requires model/EMA/
   optimizer state dictionaries, and rejects non-finite checkpoint tensors.
2. `stage3_transport_stability_diagnosis.py` rejects an out-of-order rollout
   manifest instead of silently sorting labels without sorting matrix rows.
3. Precision gains over fixed bottom-38 and matched-size bottom-m baselines
   are computed within the same 50/50 repeat and direction before averaging.
   The pre-registered thresholds remain 0.15 and 0.03, respectively.

The original snapshot remains unchanged. The active automatic pipeline uses
the scripts captured by this revision.
