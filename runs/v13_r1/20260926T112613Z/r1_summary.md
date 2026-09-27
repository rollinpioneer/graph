# CP-DISR v1.3 R1 Summary

Status: **COMPLETE at Tcap**. Both formal T_B seed-0 jobs ended normally under the first-of-N/T rule. Neither reached N=16384; no N=16384 evaluation is claimed.

|Method|Final N|Complete + fragment updates|Final dev10|Selected checkpoint N|Common dev20|Dev20 mean G|
|---|---:|---:|---:|---:|---:|---:|
|B1-K|13,623|13 + 1|0/10|0|0/20|0.000000|
|B2|14,464|14 + 1|10/10|14,464|20/20|0.500340|

The common budget interval ends at N=13,623. B2 also has its own final point at N=14,464. Curves connect raw evaluation points only as visual guides; no intermediate outcomes are measured.

Total training cost: 28,087 real transitions, 27 complete PPO updates, 2 fragment updates, 1,704 optimizer steps, and 137,632.15 simulated interaction seconds. Training wall observations: B1-K 10.07 h, B2 15.91 h (parallel); development evaluation 120 episodes total. VLM requests=0; new test-ID episodes=0.

The checkpoint rule was frozen before common dev20 in `checkpoint_index.json`; `checkpoint_generations.csv` indexes all immutable weight generations by hash. Both common dev20 evaluations were run exactly once, on the same 20 frozen dev cases. Dev20 contains dev10 and is descriptive only.

B1-K has zero effective prior and zero DP opportunities here. B2 has contract intervention with absent prior; these observations do not establish VLM gain or A_CAT behavior. No extra method, task, seed, or test evaluation was run.

Artifacts: `learning_return.png`, `learning_success.png`, `final_dev_comparison.csv`, `checkpoint_index.json`, `checkpoint_generations.csv`, `historical_evidence.md`, `Results_draft.md`, `cost_ledger.csv`.
