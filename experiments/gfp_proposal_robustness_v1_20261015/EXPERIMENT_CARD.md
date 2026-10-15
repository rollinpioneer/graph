# GFP-PROPOSAL-ROBUSTNESS-V1

Status: AUTHORIZED / protocol frozen before execution. User requested continuing this exact narrow validation. No training, tuning or new modules.

Branch: `codex/gfp-proposal-robustness-v1-20261015`
[Branch](https://github.com/rollinpioneer/graph/tree/codex/gfp-proposal-robustness-v1-20261015)
Parent: `d2ff3ed3767a4ae584fbd8c2a32525e4da414cd7`.
Server root: `/home/xushijie3/work/gfp_proposal_robustness_v1_20261015`.

Question: Do previously generated alternative first actions repeatedly improve local outcomes at selected frozen GFP failure states under new common continuation random sequences?

Reuse all 12 states, both original GFP 1M checkpoints and the exact 36 saved first actions from the parent experiment. No first action is resampled in this card. New policy RNG seeds: 2026101501 through 2026101507, common across candidates. Native environment termination is retained. At most 252 new branches, 300 steps each, 75,600 environment steps, zero optimizer updates. The original suffix is reused from saved results, not re-executed or counted as new validation.

Primary results use seven new suffixes only: per-state success counts and paired wins/losses/ties for resampled Actor versus original, VaBC versus original, and VaBC versus resampled Actor. Previous suffix results are separate; pooled eight-suffix results are secondary. Four prior rescued state/candidate pairs are fixed replication targets in `configs/frozen_protocol.json`.

Operational continuation rule fixed before execution: a target exhibits repeatable local benefit with at least four paired success wins and at most one paired loss among seven new suffixes. Continue the narrow hypothesis only if at least two of four predeclared targets meet this rule. This small-sample exploratory threshold is not statistical significance. All counts and effects are reported regardless; no automatic expansion follows either outcome.

Secondary observations: initial/final/minimum goal distance, distance progress, displacement, path length, wall contact fraction, and fraction of complete 100-step windows staying within 0.5m of their start. Euclidean progress does not establish correct route choice; low head height does not diagnose a fall. Success controls are similar states, not causal matches; three did not succeed within the previous 300-step window despite later full-episode success.

Restoration gate: each new branch must reproduce the corresponding saved first transition within 1e-8 and exactly match reward/termination. The gate is part of the 252 branches. Check full Agent/model/optimizer/RNG and checkpoint hashes before and after. Stop on mismatch, nonfinite values, changed hashes or budget excess; preserve partial results, no automatic retries.

Use verified original GFP environment and original policy, GPU 4 subject to live headroom. One inference worker, preallocation disabled. Record actual exit code, process CPU/RSS/GPU memory samples and wall time. Existing jobs untouched. Full trajectories remain on server; only code, frozen protocol, metrics, provenance and receipts archived.

Limits: selected local states, fixed three candidates, two trained seeds, seven common suffixes and 300-step windows cannot establish overall policy success, a general defect, true Q ordering or a usable action selector. Simple Actor resampling is an explicit comparison; no new mechanism follows from an oracle selecting a successful branch.
