# P2C-RL campaign stop report

Status: `CAMPAIGN_INCOMPLETE_NO_CONFIRMATORY_CLAIM`

Stop code: `SMOKE_FAILED_STOP`

## What ran

- ACTIVE release created after the explicit phrase `执行已冻结的 P2C-RL campaign`.
- Detached EXEC_WT `2335a2e9a52340778da335ffb63e269e4d4c4cca`.
- 5/5 smoke jobs `TRAINING_COMPLETE`, `learn_called=true`, `backend=real`.
- Formal jobs started: 0.
- Main test / stochastic / confirmatory statistics: not run.

## Why it stopped

Frozen runner constructs MaskablePPO with `n_envs=8`, `n_steps=256`.
SB3 therefore collects 2048 environment steps before the first update.

Registered smoke budget is 512 steps. Each smoke `complete.json` records `environment_steps=2048`.
`smoke_gate` requires exactly 512, raised `GateError: smoke SMOKE_TASK steps`, and the scheduler did not claim formal jobs.

This is a frozen-runner / smoke-budget mismatch, not a scientific protocol change.

## Policy followed

- No automatic retry.
- No runner patch on this ACTIVE release.
- No formal jobs.
- No confirmatory claim.
- A-stage `campaign_release.json` remains `NOT_RELEASED`.
- main not merged.

## Next step if a later campaign is desired

A new non-scientific runner amendment would be required so smoke can terminate at 512 steps (or the gate can accept a documented SB3 collect size), then a new ACTIVE release after another explicit execution phrase. The current release must not be reused.
