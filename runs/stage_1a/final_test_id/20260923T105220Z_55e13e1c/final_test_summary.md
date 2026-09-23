# Stage 1A final test-ID evaluation

execution_scope: STAGE_1A_FINAL_TEST_ONLY
training_complete: true (unchanged 8 updates / N=8192)
smoke_passed: true
final_test_complete: true
test-ID registration: post-training, frozen before any test result

## Checkpoints

- B2 n_008192 sha256: 3fd4f05ab4fa593271949e2c4b515d8df103954b04bb7a13c11b51e129bff96c
- Full n_008192 sha256: 5cd169a3ab822faa39ac43eedb2b8a54236af610e0e4703b3f2ed6aa4bfaf407
- weights unchanged during eval: true

## Results

| method | valid/planned | success | mean discounted return | mean success time | nonempty source prior | DP opp | DP rate | Delta rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B2 | 30/30 | 29 | 0.6137862199598949 | 15.137931034484652 | 0 | 0 | N/A | N/A |
| Full | 30/30 | 29 | 0.6137862199598949 | 15.137931034484652 | 6 | 18 | 1.0 | 1.0 |

Low test success or Full not leading is not an implementation failure.
New RL updates: 0. Optimizer steps: 0.
