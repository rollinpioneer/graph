# CP-DISR-NEXT-ENGINEERING-FIRST-TB-1 — final summary

Outcome: **TB_LAUNCH_BLOCKED_RUNTIME** at engineering smoke case 1. No training run was released.
Provider = 0, T_P training = 0, formal test = 0, elastic = 0, optimizer steps = 0, new RL attempts = 0 / 3.

## Identity
- Baseline (reviewed): 7776f3a5996acd20bcf77b15a09ec772b4d86a2a
- Prep commit (pushed, registration bound to it): 083715b0379577fc8b46489538b440aa7675ba95, branch codex/cp-disr-final-tb-engineering-launch
- Worktree: /home/xushijie2/graph_cp_disr_final_tb_launch (independent T_B worktree). Tracked tree clean after the prep commit.
- Evidence dir: runs/final_master/2.1.1/launch/20261001_prep083715b0 (use an ABSOLUTE --out; a relative --out was refused by register and left `aborted_register_relative_out_path_refused/`, kept as evidence, not deleted).

## A0 — plan status and the known problems
Plan status (searched this worktree and all 15 sibling graph_cp_disr_* worktrees, known run layouts, plus running processes):
R-TB-E-0 (B1-K+E, seed 0) NOT_STARTED; R-TB-DK-1 (B2, seed 1) NOT_STARTED; R-TB-K-1 (B1-K, seed 1) NOT_STARTED. No completed/running attempt was duplicated.

Known problems (binding_and_source_review.json, before vs after at the baseline ref):
- F2 train_job seed hard-coded 0 — fixed in stage2a_v11 (explicit seed reaches the inner runner).
- F3 +E not in the empty-R set — fixed.
- F8 render device not explicit — fixed (optional explicit render_gpu_device_id, first construction and env_factory).
- F1 old CLI lacks --seed/B1-K+E, F4 manifest bound to the old workspace, F5 R1 adapter fixes seed 0/old run ids, F6 dry-run builds environments, F7 `--phase all` materialises later stages — NOT edited in the old files; the new entry (scripts/final_tb_launch.py) bypasses them (derived manifest/split, no dry-run, no phase-all, no provider/sampler) and tests assert this. The old paths remain unsafe for T_B and must not be used.

## A1 — tests (test_receipt.json)
- New suite tests/test_final_tb_launch.py: 27 passed (10 required behaviours + release gating + budget caps + entry hygiene).
- Mutation detection (control copy passes, mutant fails for the right reason): inner seed forced to 0 / +E missing from empty-R set / GPU or environment bound before authorization — all DETECTED.
- Full suite on the clean prep tree: 679 passed, 13 failed. 12 failures are identical at the 7776f3a baseline (old lineage/whitelist guards test_S04, V3 Gate-0, s1 source binding, and 9 test_tp_so_mvp_gate cases that need a missing budget_ledger.json). 1 additional failure vs baseline: the V3-R1 Gate-0 lineage test, which compares the tree with its own base and flags the intentional additive runtime_factory/stage2a_v11 edits as unexpected_source_changes; it guards the old Family B V3-R1 lineage and exercises no T_B behaviour. Not edited, not weakened.

## A2 — smoke (smoke_case1.json, smoke_receipt.json)
- Case 1 = T_B_dev_00 (S0 b_plan_dev success trajectory), physical GPU 2. GPU binding evidence is correct (graphics context only on GPU 2). The initial snapshot was restored, forward probes for all three methods were finite (legal logits/value; B1-K+E effect tokens present, no successor; B2 successor kept; effective R empty).
- First Collector.step raised `ValueError: Categorical logits ... nan` -> FAIL:ENGINEERING_FAILURE -> TB_LAUNCH_BLOCKED_RUNTIME. Zero skills executed; the failure was raised before any controller call.
- Budget used: episode attempts 1/2, construction attempts 2/4, explicit resets 2/4, skill calls 1/24. Case 2 was not run (first must pass); no retry.
- Diagnosis status: the origin of the NaN is NOT established. Offline reproduction shows the smoke selector produces this exact error iff every logit is -inf (mask false for the scripted target), and the policy forward itself would raise the same message if one legal logit were NaN; the first-step probe was finite. The smoke driver stored only the exception message, not the traceback — a diagnosability defect of the entry that I own. This is a launch-harness/runtime engineering block, not evidence about the methods.

## A3/A4 — training
Not released (smoke gate not passed). R-TB-E-0 / R-TB-DK-1 / R-TB-K-1 remain NOT_STARTED; the first 1024-transition prefix, dev10 protocol and E16 envelope were not started.

## Matching plan requested (needs user approval; not executed)
1. New prep commit (execution-source change, so it needs a new registration): store the full traceback and, before the Categorical, record finiteness of (hidden, base_input, per-candidate logits, mask) for the failing step; add a CPU test that drives ScriptedSelector through the real Collector.step with the T_B template and a synthetic snapshot.
2. Re-register against the new prep SHA.
3. Re-run smoke case 1 using the REMAINING smoke budget only (1 episode, 2 constructions, 2 resets, 23 skill calls); case 2 only if case 1 passes. This is a retry, which the card forbade without approval.
4. If the NaN is in the production policy forward on real first-step inputs (not the selector), stop and propose a method-matched fix before any training.
Storage note: /home has ~11 GB free; a checkpoint generation is ~42 MB model+Adam and is written twice (ckpt + generation) -> ~2 GB per run over 16 updates; verify free space before releasing training.

## B — Family B gripper-space note (family_b_clearance_note.md)
Verdict: INSUFFICIENT_FOR_SWEPT_CLEARANCE. Directly recorded: V3-R1 failed in PICK DESCEND/PRESS (persisting into CLOSE): right finger collision mesh `gripper0_finger2_collision` vs the static `pad_v` rim (peak 480.8 N in DESCEND, 439.4 N PRESS; contacts.jsonl of captures/21b32e647802715c/action_00); the pad edge lies inside the open finger footprint on the finger-opening axis (edge distance 0.0402 m vs finger inner face about 0.032-0.0405 m). V3 failed by reach stall (LIFT_SHOW at x=0.2001), no neighbour collision. V3 and V3-R1 each executed only one skill call, so PLACE_BUFFER/release/retreat/carry-back were never exercised in those geometries; swept clearance cannot be established. Two hypotheses (rotate the pad axis off the opening axis; widen edge distance to about 0.07 m or more) are labelled unverified. No coordinate is claimed safe. Needed budget to settle it: a measured sweep run (finger/palm hull and per-step finger qpos, contact positions) — not run.

## C — RoboCasa (robocasa_two_task_note.md; web docs only, nothing installed/downloaded)
RemoveCuttingBoardItems and PlaceDishesBySink are official RoboCasa composite tasks (v1.0 / package 1.0.1; absent from v0.2), registry id = class name. Four-way separation for both: (1) environment creatable — evidence is only that the class exists plus generic gym.make documentation; (2) demos replay — human pretrain datasets are registered, but size is unknown and the stock replay does not check task success, so replay-to-success is unverified; (3) model closes the loop — no per-task evidence; (4) callable atomic skill — no evidence found (partial reading: GitHub tree pages were not accessible). Integration needs a separate Python 3.11 environment (robosuite >=1.5.2, mujoco 3.3.1, numpy 2.2.5; about 10 GB assets) and a mobile-base+torso+arm adapter. Page content was read through a summarizer; claims marked [summarizer] need re-reading. No eligibility or readiness claim. Smallest budget that answers the unknowns: a fresh venv + asset download + one env-creation smoke + two-task demo replay with a success check.
