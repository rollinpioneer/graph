# Evaluation observability

The failed runner publishes `n_000000` before calling `maybe_eval_at_n`. The first frozen dev10 case is `T_C_dev_04`. The old pair emitted no `eval_episode.jsonl`/`eval_decision.jsonl` heartbeat before termination, so the historical file set cannot prove an episode completed. The diagnostic tool now appends `exact_phase_timeline.jsonl` before and after env construction, reset, pose application, `sim.forward`, observation, each decision/controller call and transition.

The bounded runs produced incremental markers for all four cases; each reached four decisions and simulator time 0.20 s. This is observability evidence, not a formal evaluation result.
