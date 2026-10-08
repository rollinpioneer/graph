# C1-BW-METHOD-SERIAL-SUITE-V3 — claim boundary

## What the evidence supports (this task family, these checkpoints, one training seed, deterministic argmax, one 128-problem confirmation set)
1. With the weights of the M1-GOAL scorer, a two-step contract look-ahead with the learned state value improves all-steps-optimality over the one-step C3 controller (96 vs 79 of 128; h=4: 34 vs 23) and keeps simple problems; with the goal-count leaf it is much worse (43). The learned value is therefore a useful leaf score. This includes extra search compute and contract expansion (about +70% episode wall time).
2. A relation-restricted goal attention (GOAL_REL) gives 95 all-steps-optimal episodes against 83 for the same module with dense attention and 79 for MG_C3, at one-step cost, with a small success loss (122 vs 124).
3. Relative and absolute calibration are interchangeable for one-step control; relative calibration is slightly better as a look-ahead leaf (+6 optimal on 128 problems, no success difference); calibration does not improve over the uncalibrated MG leaf.
4. Next-event supervision with these labels is learnable but made goal-height-4 execution worse (SG_FACT 26/64, SG_JOINT 40/64 successes vs 60/64 for the matched BASE).
5. The recurrent processor, as trained here, changed logits but not decisions: T4 = T8 and REL = SELF on all 128 problems.

## What must not be claimed
- Any ranking of final systems or any "winner": every number is one checkpoint per condition from one training seed; the 128 deterministic episodes are not 128 independent trials; no significance test or multiplicity correction is reported; discordant counts of 1-5 problems are not differences.
- That the look-ahead gain is a learning gain (it adds contract-based search), or that two-step is the right depth (only depth 2 was tested; the goal-count leaf with fixed-id tie-break is a weak baseline, not the best possible heuristic).
- That pairing information, "events", or "sub-goal organisation" helps: only 2.5% of decisions carry pair-specific labels, the matched FACT/JOINT checkpoints were selected at different epochs (80 vs 20), and both are worse than BASE at h=4. The negative h=4 effect may come from the auxiliary loss itself, the gate, or the event definition (OnTable events are 48% of events; no first-On variant was run, by design).
- That recurrence or relational neighbours do not help: the module saw an action loss of about zero on D_train (new-module gradient ~1e-5), so it is under-supervised; a different training signal or the 8-iteration regime of the original deep-thinking work was not tested; no shared/unshared-parameter control exists, and REL = SELF only says that this implementation shows no neighbour effect.
- That GOAL_REL is a relational-reasoning gain: it hides 44-53% of the non-self goal pairs in new problems (40% in training), so the effect may be suppression of unseen cross-tower pairs; the contract-threat term of the mask is empty in Blocksworld; no third mask condition was run.
- That calibration reduced value error or "decoupled" anything beyond keeping the initial policy (value error was not tabulated); generalisation to other domains, hidden contracts, other seeds, sample efficiency, or any statement about KR 2023 / GSP / AIW-AD / RPN (not run).
- That height-4 gains show long-horizon dependencies are solved: height 4 is goal-height extrapolation mixed with n=8, two-tower and colour shifts; training never contained height 4.

## Disclosures
- The runbook file says execution was authorised only by a later message; the user's message "执行实验" (2026-10-08) with the runbook and package attached is recorded in `plan/registration.json`.
- The configuration defines FACT/JOINT organisation losses exactly as the supplied math reference; the CAL objectives use the endpoint pool of plan 8.3 (state, legal successors, recorded 2/4-step optimal endpoints), Huber weight 0.1, scale c = 4.
- Frozen references MG_C3 and B_G1C3 on Board48 / fresh112 are reused from the earlier cards (same code, problems, weights); they were evaluated afresh on Confirm128.
- Checkpoint selection used Board48 only; fresh112, dev36 and the 240-state bank were evaluated for the selected checkpoint only and could not reselect (several selected checkpoints are early epochs, e.g. CAL at epoch 20, JOINT at epoch 20, where Board48 differences are one or two problems).
- Confirm128 was generated after all selections were locked and contains no problem class seen in training, development, earlier confirmation or the 240-state bank sets (852 classes excluded, 0 overlap).
- Post-hoc, descriptive additions written after the confirmation numbers: `scripts/c1_method_serial_post.py` (development summary, recurrent-effect diagnostic) and this narrative.
