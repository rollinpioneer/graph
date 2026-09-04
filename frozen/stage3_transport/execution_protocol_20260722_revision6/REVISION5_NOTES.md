# Stage 3 Transport-MH Execution Protocol Revision 5

Revision time: 2026-07-22, after the epoch-150 periodic evaluation but before
the final checkpoint, fixed Transport 100-rollout pool, TRAK scores, and
Transport stability results existed.

Parent snapshot:
`execution_protocol_20260722_revision4/all_files_sha256.txt`
SHA-256 `db35733954549dc984b5730d194bfc6921ea975c72f47d142115cc63b3c3b38e`.

This revision changes no training, dataset, split, seed, rollout, TRAK, or
stability-gate parameter. It removes the unsupported `weights_only` keyword
from the final checkpoint audit's `torch.load` call. The frozen CUPID runtime
uses PyTorch 1.12.1, which rejects that newer keyword. Checkpoints are trusted
local artifacts produced by this training process.

The corrected audit loaded the epoch-150 checkpoint in the exact supervisor
environment, recovered global step 73,989, and found all 768 recursively
visited model, EMA, and optimizer tensors finite.
