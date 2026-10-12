# DATA_AUDIT

| Data | Train trajectories | Raw trajectory rows | Effective transitions | Free-cell coverage | Intercell directed edges | High-goal median horizon |
|---|---:|---:|---:|---:|---:|---:|
| navigate | 1000 | 1001 | 1000000 | 46/46 | 96 | 190.5 |
| stitch | 5000 | 201 | 1000000 | 46/46 | 96 | 38.0 |

Both datasets use exactly the same registered Large map and five evaluation tasks. All five task endpoints are connected in the observed directed cell-transition graph for both datasets. This excludes an obvious missing-cell disconnection; it does not establish full-state controllability, learned policy competence, or a single-factor cause.

State observations are qpos (two dimensions), no normalization. Original qpos/qvel/action/terminal fields remain in hashed NPZ files. Server index preserves original trajectory ID, step ID, observation/action/next observation and validity. Invalid terminal rows are explicitly masked; no boundary-crossing tuple is a real transition. Official compact terminal remapping is preserved, including terminal marking at both the final raw row and its predecessor.

Each dataset has 4,096 fixed training-rule probes and shared (s,g,w) indices for D1–D4. Current/future/random value-goal mixture and future-only high-actor goals pass source-level and measured sampler checks. Navigate/Stitch differ in horizon, occupancy weights and collection policy simultaneously. Coarse BFS geometry is a privileged diagnostic proxy, not discounted-value truth. See `results/DATA_AUDIT.json`, `results/SAMPLER_AUDIT.json`, `results/D1_COVERAGE.json` and per-sample CSVs.
