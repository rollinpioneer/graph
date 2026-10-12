# PC-INFOATTR-06 reproduction

Branch: codex/pc-infoattr-06-20261012-card
Starting commit: fb0a59b6ee6be39e98148ad30d2f99556bc6d926
Frozen discovery commit: 6c03f59140bc7d8b416ea4c72c4f71cb35eba823
Source card SHA256: 11711ea4242a3dcd412262109c0f575b44b0325d313e83495f3fa14c34750e4d

Use lab-xushijie3 with one persistent SSH shell and keepalive. The server worktree is
/home/xushijie3/work/graph_pc_infoattr06_card_20261012.
Set W to that worktree, R=$W/experiments/pc_infoattr06_run_20261012,
P=/home/xushijie3/work/graph_pc_infochain05_card_20261012/experiments/pc_infochain05_run_20261012,
OLD=/home/xushijie3/work/cpdisr_arm_20261011/stage5_pushcube_20261011,
PY=$OLD/env/bin/python, PATH=$OLD/env/bin:$PATH,
LD_LIBRARY_PATH=$OLD/env/lib:$OLD/isaacgym/python/isaacgym/_bindings/linux-x86_64:$LD_LIBRARY_PATH.
Python3.8/Torch2.1.2+cu118 and IsaacGym; import Gym before Torch. Engine commit6b5e482fe479f647d4ca3790fac70c42f4940ab2 with existing official_compatibility.patch.
No package upgrades, training, encoder modifications or changed task physics.

Asset identities are receipts/frozen_assets_identity.json and current_assets_verification.json.
Existing discovery tensors stay at their original paths, independently hashed in results/source_inputs.json.

preanalysis_freeze.json fixes calibration, health holdout and diagnostic frame lists.
analysis.py computes one appearance median prototype/color and pooled feature scale on calibration
only; inference uses only actual foreground particle position, appearance and object-on fields.
Labels use independently inspectable image/projection proxies during calibration, then only after
predictions during scoring. No final outcome labels enter calibration. It is a fitted diagnostic rule.
Thresholds and all source hashes are preserved. The healthy holdout fails the single-view80%/10px
gate; do not refit or interpret diagnostic errors as DLP information loss. Multiview availability
is a scoring upper bound, not a calibrated triangulated position estimate. 3C blocked and permuted
share a scene: their records do not count as two independent healthy scenes.

normal_freeze.json contains the four scene/policy seeds, two independent GPU workers and full200step
settings. The8step cost pilot and both loading failures are included in accounting.
launch_normal.py executes two sequential scenes/worker, workers concurrently, through guard.py.
Each output path must be fresh; do not overwrite or rerun a failed/complete condition in place.
Guard counts full setup/error wall time multiplied by all task-resident GPU contexts, aggregate
task memory across this round's process groups, with safety stop below12GiB and round7200GPUseconds.
normal sampling uses a seeded uninterrupted RNG stream, without per-call reseeding/noise blocks.
Every step captures actual RGB/current+goal DLP, unnormalizedH5 actions, executed first action,
controller target, actual body/EEF/q/qd, all101draws, and complete before/after RNG states.
Only the first H5 action is executed. All full200step endpoints, including falls, are retained.

CPU command sequence after GPU jobs end:

```sh
"$PY" -u "$R/normalizer_audit.py"
"$PY" -u "$R/chain_details.py"
"$PY" -u "$R/mechanism_review.py"
"$PY" -u "$R/audit.py"
"$PY" -u "$R/figures.py"
"$PY" -u "$R/censoring.py"
"$PY" -u "$R/finalize.py"
```

chain_details.py reruns the fixed analysis and joins original inputs, descriptive reader output,
prior reconstruction evidence and command/response facts. Additional event times are explicitly
posthoc annexes; absent reader frames are not invented. normalizer_audit.py reconstructs action
bounds from valid training replay paths on CPU; inverse normalization represents clipped actions
only. The original preclip normalized diffusion proposal was not saved and is marked unavailable.

P2 aligns input[t], executed action[t], controller request[t], actual response[t+1]. Engine
source sets the OSC target once before the simulation loop and retains it after env.step.
audit.py checks numeric target transform, first saved input, all200 unique noise streams/101draws,
exact unchanged physics, frozen source hashes, finite states and record lengths. A direction cosine
does not prove task usefulness, full tracking or effective contact. Center proximity is not contact.
mechanism_review_freeze.json fixes20step approach/departure/no-progress candidates before opening
new endpoints, while the first two new runs were already generating data. Preserve this timing limit.
Repeated overlapping windows are not independent episodes. Noise-blocked discovery behavior is
not automatically original EC-Diffuser behavior; new normal review is an independent mechanism check.

P1-B, information replacement and conditional bounded-response validation statuses/reasons are in
receipts/conditional_validation_decision.json. They are not silently treated as completed experiments.
The card allows stopping when a prerequisite fails or no useful discriminating intervention exists.
Do not automatically train, change architecture, scan noise block lengths or build a contact interface.

finalize.py copies lightweight job receipts and creates resource ledger, status, report, server-only
artifact hashes and explicit light_files.txt. Full logs, RGB/tensors/noise/RNG and weights/data remain
server-only. Git stage exactly the allowlist plus this card's BRANCH_INFO.json and EXECUTION_ANNEX.md;
review staged contents before commit and push with the existing deployment identity. No force push
or merge. Compare local commit with ls-remote, then create a transfer archive and close the SSH client.
Original card bytes are preserved; historical card-only state is superseded by the execution annex.
