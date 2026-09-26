# CP-DISR v1.2 Phase A initial rollout stall diagnosis

Status: **NEEDS_RERUN**; `formal_rerun_gate.passed=false`.

- Exact phase: after `n_000000`, step-0 dev10 evaluation, first case `T_C_dev_04`, at environment reset/`sim.forward` path. It was not a training rollout.
- Geometry: `INITIAL_INTERPENETRATION` in 84/84 frozen T_C train/dev cases using `OBJECT_HALF=0.020 m`.
- Failed pair classification: `RESOURCE_CONTENTION_WITH_NATIVE_MUJOCO_ACTIVITY`; prior stack samples observed MuJoCo collision functions under the shared loaded GPU/high-thread configuration. This is not classified as an intrinsic isolated native stall.
- Isolated diagnostics: B2/dev04, Full/dev04, B2/train01 and B2/D0 each completed 4/4 decisions; sim time advanced; optimizer/PPO=0.
- Source commit: `a3784153294a46be9e86c4b889f006bf7a1511fd`.

No formal E16 rerun, PPO update, VLM request, test-ID, split edit or cache regeneration was performed.
