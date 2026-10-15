# Stage 3 Transport-MH Execution Protocol Revision 6

Revision time: 2026-07-22, during base-policy training and before the fixed
Transport 100-rollout pool, TRAK scores, and Transport stability results
existed.

Parent snapshot:
`execution_protocol_20260722_revision5/all_files_sha256.txt`
SHA-256 `e9872c70ad92086a1c01c857726a53cae34c40ec4bb25084775a7b69a9721ea2`.

This revision changes no training, dataset, split, seed, rollout, TRAK, or
stability-gate parameter. It adds a gated post-diagnosis ID freezer:

- both the status marker and diagnosis text must identify the pre-registered
  strong-pass decision;
- all diagnosed input hashes and the official-score reconstruction limit are
  rechecked;
- the exact full-100 bottom-20% bootstrap core at `p >= 0.90` is frozen in
  ascending score-rank order with provenance hashes and a Hydra override;
- STOP decisions create only a SKIP status and no filter set;
- filtered retraining is never started by this waiter.

Synthetic PASS and STOP tests, Python syntax checks, and shell syntax checks
passed before launch. No real Transport filter ID was available or selected.
