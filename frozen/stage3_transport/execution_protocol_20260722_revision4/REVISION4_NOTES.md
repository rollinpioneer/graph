# Stage 3 Transport-MH Execution Protocol Revision 4

Revision time: 2026-07-22, during base-policy training and before the final
checkpoint, fixed Transport 100-rollout pool, TRAK scores, and Transport
stability results existed.

Parent snapshot:
`execution_protocol_20260722_revision3/all_files_sha256.txt`
SHA-256 `a5fa1d27e2dfc64ec8a8d89799d190e8cb2c7d76b63a61e7073fea2053106fd0`.

This revision changes no current base training, dataset split, seed, rollout,
TRAK, or stability-gate parameter. It adds a dormant exact-ID filtering path
for a possible post-gate paired retraining experiment:

- `get_dataset_masks(..., filter_episode_ids=[...])` removes exactly the given
  original episode IDs from the training mask.
- Exact-ID filtering is mutually exclusive with the author's ratio-based
  curation path and rejects empty, duplicate, noninteger, out-of-range, or
  non-training IDs.
- Validation and holdout masks are not changed.

The mechanism passed a mask test and full Hydra dataset instantiation on the
frozen Transport split: deleting example training IDs `[0,1,2,3,4]` changed
192/12/96 demos to 187/12/96 and reduced training samples from 125,324 to
122,874. These IDs are test fixtures only; no final filtering set is selected
or implied by this revision. No filtered training is authorized before the
strong offline gate passes.
