# Finite-loop reproduction

## Live incident (not bitwise replay)

Bitwise MuJoCo replay of `T_B_train_36` is **not available**: the process was SIGSTOP then SIGINT during sampling; in-memory env/Fact Store/GRU were not serialized. Disk evidence is the transition/decision logs.

Live pattern:

- After a short prefix of `NORMAL_TERMINATION` skills, lid perception becomes unavailable.
- Every subsequent OPEN raises `RuntimeError("perception_unavailable")` with `steps=0`.
- Pad writes duration 0.05, reason CONTINUE.
- Mask remains `[true, false, false, false, false, false, false]` for OPEN.
- Episode clock stays below 60 s deadline; N/T grow.

B0 open episode at stop: `T_B_train_36` decision_id 2431, 2423 padded interrupts.
B1-K open episode at stop: `T_B_train_53` decision_id 2394, 2384 padded interrupts.

## Synthetic reproduction (production SkillExecutor)

`tests/test_clock_integrity.py::test_clock_7_tb_train_36_fault_reproduced_as_zero_physics_interrupt`

1. Execute PICK with lid perception present -> positive real duration.
2. Clear `last_perception`.
3. Repeat OPEN 5 times -> INTERRUPT + `sim_duration <= 0` + clock unchanged.

`test_clock_5` shows 8 OPEN repeats cannot grow N/T without time.
`test_clock_11b` shows Collector raises `DiagnosticAbort` and does not build a PPO transition.

Guards: FakeEnv `max_steps=400`, wall budget 20 s. Guard fires as `RuntimeError("DIAGNOSTIC_*")` / `DiagnosticAbort`, never as task `DEADLINE`.

Exact live RNG/env replay: **NOT_CLAIMED**.
