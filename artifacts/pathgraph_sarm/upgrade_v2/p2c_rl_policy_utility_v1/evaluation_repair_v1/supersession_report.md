# P2C-RL Evaluation Repair V1: Supersession Report

Recorded at: 2026-09-18T18:17:12Z

## Scope

This repair is based on `1eaa0a48dfaf75563c05a0ba974ed3a7465f739e` and is confined to branch
`research/pathgraph-p2c-rl-evaluation-repair-v1`. It does not amend or merge
`main`, rewrite Attempt-03, retrain a policy, select an intermediate
checkpoint, call `.learn()`, or call `optimizer.step()`.

The 60 frozen inputs are exactly the final `policy_524288.zip` checkpoints in
`checkpoint_freeze.tsv`. Their total size is 302,799,616 bytes. Every SHA256
matched the checkpoint file, `complete.json`, and `.meta.json` before
evaluation, and matched again after evaluation.

## Prior result disposition

The two legacy reported effects of 0.0 are
`INVALID_EMPTY_RECORD_INPUT`. The legacy main, stochastic, and critical
modules emitted metadata only, while campaign statistics received an empty
records list. This report supersedes only that scientific interpretation. The
old commit, old result files, training logs, policies, data, reward, mask, PPO
settings, test contract, comparisons, and statistical gates remain unchanged.

Before any repaired result was read or created, `ci_level=0.975` was locked as
an equal-tailed interval using quantiles 0.025 and 0.975 (central 95%). The
protocol-authoritative 200,000 bootstrap replicates, seed 2026091806, practical
margin 0.03, and draw -> seed within draw -> family within motif order were
retained.

## Complete actual records

| Panel | Rows | Mode |
| --- | ---: | --- |
| Main | 30,720 | deterministic masked |
| Stochastic | 15,360 | fixed seeds 2026091802-2026091805 |
| Critical state | 491,520 | all registered test prefixes, masked policy input |

All 60 per-job receipts passed file SHA256 checks. Empty records, missing or
unexpected keys, duplicate keys, and incomplete method panels are hard
failures. Main evaluation recorded zero invalid actions and zero non-finite
observations.

## Registered primary comparisons

| Comparison | Point | 0.025 quantile | 0.975 quantile | Margin >= 0.03 | CI lower > 0 |
| --- | ---: | ---: | ---: | --- | --- |
| RW2 - GEOM | 0.0123697917 | -0.0227864583 | 0.0436197917 | no | no |
| RW2 - FLAT | 0.0192057292 | -0.0076497396 | 0.0449218750 | no | no |

The frozen statistical gate does not pass. The repaired status is
`POLICY_UTILITY_NOT_ESTABLISHED_OR_MIXED`; `passed=false` and
`global_confirmation_passed=false`. No confirmation claim is made.

## Method-level diagnostics

Main deterministic success means were TASK_ONLY 0.799642, RW2 0.791992, GEOM
0.779622, FLAT 0.772786, and GRAPH1 0.755534. Fixed-seed stochastic success
means were RW2 0.898112, TASK_ONLY 0.893880, GRAPH1 0.889974, GEOM 0.883464,
and FLAT 0.879883.

Critical-state oracle agreement means were TASK_ONLY 0.728495, FLAT 0.724508,
GEOM 0.720734, GRAPH1 0.708720, and RW2 0.699666. The independent oracle was
analysis-only; the policy received only the frozen observation and action mask.

## Integrity and recovery disclosure

Attempt-03 SQLite remained read-only, retained `status=RESUMING`, and had the
same before/after SHA256:
`21b76fa435c5efbce1bb3f743989a348fd900b735f9ac075fd7a0a95dfc3f750`.

The first parallel evaluation launch was interrupted after two complete job
receipts because forked workers shared a lazy NPZ ZIP handle. Twenty incomplete
new-output job directories were preserved under the external evaluation root
at `interruptions/attempt_01_shared_npz_handle/jobs`. No Attempt-03 input was
changed. Commit `fd119af67` changed cache loading to validated eager arrays;
attempt 02 resumed from the two valid receipts and completed all 60 jobs.
See `interruption_recovery.json`.

## Artifacts

The compressed episode records, method summaries, primary comparison, and
output manifest are in `final/`. The output manifest SHA256 is
`669d1fb032afa7227571b48f7724f7eea5b56ef8bd5d2c8929e2780717a86ebd`.
The primary comparison SHA256 is
`7a3e822de03c9543ba1c4ed85ab140b0701be2a35f981bd2a5420dd5843950c4`.
