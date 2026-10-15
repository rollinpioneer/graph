# Stage 5 Entry Decision

- `STAGE5_ENTRY_DECISION = REFINE_STAGE4_MINIMAL`
- checkpoint verification: `True`
- seed passes: `0/3` (required `2`)
- metrics source: real checkpoint prediction files only
- test predictions are reporting-only and were not used for this decision

## Minimal refinement

Retrain only the remaining-cost/failure-recovery calibration head using the existing Stage 4 configuration, preserving the encoder, graph specification, split, and seed set. Reward parameter search is prohibited until a subsequent real recompute emits `REAL_MODEL_READY`.

- failed checks: `failure_cost_increase_rate, recovery_cost_decrease_rate`
- observed: failure direction `0.571`, recovery direction `0.500`
- observed: failure direction `0.571`, recovery direction `0.000`
- observed: failure direction `0.000`, recovery direction `0.000`
