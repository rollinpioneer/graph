# Stage 2A clock-integrity repair summary

This is **not** Stage 2A final delivery. Status is `NEEDS_RERUN`. `repair_validation_passed=true` only means limited production regression passed. `stage_pass` remains false.

## Done

1. Verified PIDs 3600298 / 3602657 / 3602658 and stopped them without another PPO or dispatch of the remaining six jobs.
2. Preserved logs, checkpoints, resume, process snapshots, startup source snapshot.
3. Per-transition clock audit of both live jobs.
4. Removed zero-time duration fallbacks; abort technical zero-physics paths; episode-relative remaining.
5. Limited pytest regression: 32 passed (clock integrity + deadline + duration + zero-prior + empty-patch + masks).
6. Checkpoint impact: both complete updates on both jobs are contaminated.
7. Restart plan written; **not executed**.

## Not done (requires new authorization)

- New attempts for T_B B0 / B1-K
- Start of the six remaining jobs
- Select / final test-ID / Stage 2A report-as-PASS
- 1B / 2B / 2C / 3A
- git push / PR

## Key paths

- Repair dir: `runs/stage_2a/20260923T121938Z/clock_integrity_repair/20260923T135530Z/`
- STOP: `runs/stage_2a/orchestrator/STOP_CLOCK_INTEGRITY.json`
- Old STOP untouched: `experiments/part_2_exploration/stage_2a/orchestrator/STOP_SUPERSEDED_BY_PLAN_V1_1.json`
- Status: `status/stage_2a.json` (history under `status/history/`)

## Runtime identity

- `runtime_revision=clock_integrity_r1`
- Base commit still `f34789d6a5e6953b9416d06eb43dfaa14b6530e1`
- Working tree dirty/untracked with the repair (clock, executor, collector, common, stage2a_v11, continue.sh, tests)

Waiting for an explicit resume-training authorization.
