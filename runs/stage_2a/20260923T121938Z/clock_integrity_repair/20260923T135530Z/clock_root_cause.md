# Clock root cause

## What broke

Two zero-time fallbacks in the shared baseline (`f34789d6`) invented a positive duration when the frozen MuJoCo clock did not advance:

1. `src/cp_disr/platforms/libero/clock.py` `duration_seconds`: if `end-start <= 0`, return `1/20` seconds.
2. `src/cp_disr/platforms/libero/skill_executor.py` `_finish`: if `elapsed <= 0`, return `max(0.05, steps/20)`.

Collector built PPO Transition / Gamma / N / T from that padded duration. Evaluator remaining and `clock.now_seconds()` stayed on the real sim clock (`sim.data.time - origin`). If physics did not step, episode elapsed never reached the task deadline, while recorded T and discount still moved.

Collector remaining was `deadline - now`. After `start_case` the DurationProvider origin is reset, so `now` is episode-relative and that formula is equivalent to `deadline - (now - episode_start)` **only while the clock actually advances**. The non-advancement bug is the physics/pad split, not a second origin bug. The repair still uses the explicit episode-relative formula.

## T_B_train_36 / T_B_train_53 OPEN loops

These are **not** legal no-ops just because the action name is OPEN.

B0 `T_B_train_36` (ep-37): 2432 transitions, 2423 of them `a:OPEN:container:v1` with:

- `controller_exit = INTERRUPT_CONTROLLER_EXCEPTION:RuntimeError`
- `duration_seconds = 0.05` exactly
- `reason = CONTINUE`
- recorded duration sum 167.3 s vs 60 s deadline
- episode_log never closed the episode

B1-K `T_B_train_53` (ep-54): 2395 transitions, 2384 the same interrupt+0.05 pattern, duration sum 173.7 s.

In executor, `_open` calls `_object_xyz("lid")`. Missing perception raises `RuntimeError("perception_unavailable")` **before any `step_osc`**. `_finish` then padded elapsed 0 to 0.05. Collector accepted a PPO transition. Mask still allowed only OPEN, so the policy repeated OPEN forever. Deadline never fired because sim time did not move.

Every logged `duration==0.05` in these two jobs is this interrupt path. There were **zero** non-interrupt 0.05 rows, so we do not confuse them with a real 20 Hz control tick.

Completed earlier episodes (e.g. `T_B_train_00` last action OPEN `TASK_DEADLINE` duration 3.50 s, dur_sum 60.00 s) are consistent with real physics + frozen deadline.

## First-update PASS

Both jobs had `first_update_ok=true` at complete update 1/2. That check did not include clock non-advancement. Both complete updates ingested padded interrupt transitions (see checkpoint impact matrix). PASS is historical only.

## Repair (runtime_revision=clock_integrity_r1)

- Delete both positive-duration fallbacks.
- `duration_seconds`: nonfinite or negative -> `ClockIntegrityError`; zero returned as 0, never epsilon.
- Technical interrupt/NaN: `DiagnosticAbort`; no PPO transition; not DEADLINE; no silent resample.
- Legal reject may take one real confirm tick via `step_osc` hold, then read actual elapsed. Cannot advance -> abort.
- Collector remaining = `deadline - (clock_now - episode_start)`.
- Stage 2A train/select/eval/report refuse `STOP_CLOCK_INTEGRITY.json`.
- continue.sh hard-exits if that STOP file exists.

No network/PPO/num_envs/H/deadline/Ncap/Tcap change. No repeat-action penalty, OPEN hard-mask, or shaping.
