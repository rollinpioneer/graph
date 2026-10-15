# Stage 3 Transport-MH Execution Protocol Revision 3

Revision time: 2026-07-22, before the final checkpoint, fixed Transport
100-rollout pool, TRAK scores, and Transport stability results existed.

Parent snapshot:
`execution_protocol_20260722_revision2/all_files_sha256.txt`
SHA-256 `b65e39bb27bfa88b5648905f0af769633519e4fe7fec683de1e44719c3f71fc7`.

This revision changes no model, dataset, split, seed, rollout, TRAK, or
stability-gate parameter. It strengthens `audit_rollout_pool.py` to:

1. require exactly the configured consecutive test seeds in `eval_log.json`;
2. require finite binary rollout scores and agreement with the logged mean;
3. cross-check each score against filename, metadata, and DataFrame success;
4. require nonempty episode/video files, finite reward/observation/action
   values, consistent tensor shapes, and strictly increasing timesteps.

The checks passed an end-to-end regression against the prior frozen Square-MH
100-rollout pool (100 episodes, 71 successes, 29 failures, 4,066 decisions).
The active automatic pipeline uses the audit script captured by this revision.
