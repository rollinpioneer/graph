# B2 seed_0 crash: episode-elapsed reward offset

Time: 2026-09-21T13:35:22Z
Process: PID 3744566
Error: `DataIntegrityError: Invalid independently confirmed reward event`

Cause:
- Frozen `interval_reward(duration, events)` requires `0 <= offset <= skill_duration`.
- D0 `TaskEvaluator` stamped the first success as `(elapsed_seconds, 1.0)`.
- `elapsed_seconds` is episode time (`end - episode_start`), not skill-interval offset.
- After several skills, episode elapsed exceeds the current skill duration, so a real independently confirmed success was rejected.

This is an adapter time-base bug, not a change to PPO, network, reward formula, B, or ?Q.
Fix: stamp `offset = interval_end_seconds - interval_start_seconds`.
