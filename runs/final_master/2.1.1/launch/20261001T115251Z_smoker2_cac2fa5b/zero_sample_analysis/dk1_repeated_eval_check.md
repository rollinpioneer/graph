# DK-1 repeated evaluation return — checkpoint identity and per-case records (no new episode)

Observation (already in the registered evals): R-TB-DK-1 reports mean discounted return **0.4988490585995873**, success 10/10 and recorded `success_seconds` mean 3.54 at all three post-update evaluation points (N = 4096, 8192, final 14170). This file checks only what the existing records can establish.

## Checked and found

| item | result |
|---|---|
| evaluated checkpoints are different files | sha256 all distinct: `n_000000.pt 916a08b08c29, n_004096.pt 0c8da7c604d4, n_008192.pt ffb932a0131b, final_n_014170_u13.pt b1e331c98d49` |
| evaluated model weights differ between the points | n_004096 vs n_008192: 66 of 141 float tensors differ, max abs diff 4.645e-02; n_008192 vs final: 66 of 141, 9.943e-02; n_004096 vs final: 66 of 141, 1.170e-01 |
| each eval file names the checkpoint it evaluated | `eval_*.json` field `checkpoint` = the respective `.pt`; final eval checkpoint path == final_n_014170_u13.pt (realpath equal) |
| code / split identity of the evals | the `hashes` block (final_tb, stage2a_v11, collector, torch_rl, neural, persistence, runtime_factory, manifests, dev10 split b16130d2…, git_commit cac2fa5b3) is identical across all 12 eval files of the three runs |
| per-case rows across the three points | success, G (full float precision), steps, reason, success_seconds are identical for 4096 vs 8192 and identical for 8192 vs final (10 cases each) |
| final checkpoint = post-last-update model | final weights and Adam state equal `update_fragment_1.pt` (max diff 0.0, optimizer state equal) and differ from `update_complete_13.pt` (max abs diff 3.241e-02, 66 tensors) |

## Not established (UNVERIFIED)

- Whether the policy takes the same action sequence on each dev case at the three checkpoints: the eval rows hold only `case_id, success, G, steps, reason, success_seconds, source_n, prior_mode`; no selected-candidate sequence, logits or per-step trace is stored for evaluation episodes, and no eval decision log exists in the run directory. Identical G, step count (5) and final-skill duration are consistent with identical trajectories but do not prove it. -> **UNVERIFIED**.
- Why the greedy (deterministic argmax) policy gives identical returns at differing weights: not analysed; no mechanism claim. Recomputing it would need new evaluation episodes, which this card does not run.
- Total elapsed time to success per case: not recorded (see caveat C1 in `comparison_tables.md`).

Conclusion allowed from these records: the three DK-1 post-update evaluations are three distinct evaluations of three distinct checkpoints that returned identical per-case numbers; this is not a duplicated or mis-bound evaluation file. It is not shown that the trajectories are identical.
