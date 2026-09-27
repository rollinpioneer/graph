# R1 Cost Ledger

`cost_ledger.csv` is the machine-readable ledger. Training stopped at the prescribed Tcap, so the unmet N=16,384 points are neither missing evaluations nor imputed values.

|Method|Real N|Complete PPO|Fragment PPO|Optimizer steps|Sim interaction hours|Train start-to-final wall hours|Dev10|Dev20|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|B1-K|13,623|13|1|812|19.115|10.072|40|20|
|B2|14,464|14|1|892|19.116|15.912|40|20|

Totals: 28,087 real transitions, 27 complete plus 2 fragment PPO updates, 1,704 optimizer steps, 120 development episodes, zero R1 VLM calls, zero R1 test-ID episodes. Pair sampling elapsed wall was ~15.95 h from the first training start through the last final durable generation; summed GPU allocation time was ~25.98 GPU-h before development evaluation. No monetary price was supplied, so currency cost is not estimated.

Older cost records are preserved but not added to R1 totals because scopes can overlap or use different profiles:

- `runs/stage_1a/final_test_id/20260923T105220Z_55e13e1c/cost_ledger.csv`
- `runs/migration_preflight/20260925T154044Z/cost_ledger.csv`
- `runs/migration_preflight_remediation/20260926T000000Z/cost_ledger.csv`
- `runs/phase_a_final_unblock/20260926T003000Z_fa1c2d3e/cost_ledger.csv`
- `runs/phase_a_stall_diagnostic/20260926T065208Z_815e394f/cost_ledger.csv`
- `runs/tc_nonoverlap_remediation/20260926T092217Z_ee1bca3a/cost_ledger.csv`
- `runs/stage_2a/clock_recovery/20260923T170130Z_00ac373b7ec0/actual_cost_ledger.csv`

B1-K train GPU 5 and B2 train GPU 6 ran concurrently; both one-time common dev20 evaluations used GPU 0 after the checkpoint freeze. Earlier dev10 runs were part of each training job. Wall measurements use logged training start through final durable checkpoint publication and exclude dev20; no separate GPU billing meter was available.
