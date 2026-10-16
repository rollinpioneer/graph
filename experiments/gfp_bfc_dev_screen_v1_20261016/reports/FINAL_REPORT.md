# GFP-BFC first-round final report

Decision: **NO-GO for advancing BFC V1 on this evidence.** Both BFC settings failed the frozen consecutive-anomaly criterion in both windows. Neither outperformed the best tested EMA on average. The slowest EMA preserved seed 0 but collapsed in seed 3; it is a useful comparator, not a generally validated remedy. No further training is launched.

## Completion and identity

12 branches completed exactly 100k updates each, total 1.2M; 24 primary evaluations completed, 100 episodes each. The nine original reporting failures remain recorded; the explicitly authorized repairs completed successfully. All 36 saved checkpoint identities and full-restore receipts pass, both source checkpoint hashes are unchanged, and all 24 initial-observation hash vectors match. No non-finite sampled losses/probes were found.

Source: upstream GFP 1abbd5f0e287caf9bda70e7bc655d0a499ac8401; formal G_task1_s0 600k and G_task1_s3 300k forks; OGBench 1.1.0. Full actual config/source identities are in receipts/assets.json and runs/*/identity.json. The reference Actor only changes the TD Actor contribution; online Actor deployment and all other specified routes remain unchanged.

## Success (%) — all checkpoints, no best-checkpoint selection

| Arm | tau | epsilon | s0 +50k | s0 +100k | s3 +50k | s3 +100k | final two-window mean |
|---|---|---|---|---|---|---|---|
| R0 | None | None | 59.0 | 1.0 | 0.0 | 0.0 | 0.5 |
| R1 | 0.005 | 0.0 | 76.0 | 0.0 | 0.0 | 30.0 | 15.0 |
| R2 | 0.0005 | 0.0 | 79.0 | 5.0 | 0.0 | 9.0 | 7.0 |
| R3 | 5e-05 | 0.0 | 62.0 | 74.0 | 26.0 | 0.0 | 37.0 |
| R4 | 0.0005 | 0.09990192690834347 | 66.0 | 6.0 | 0.0 | 22.0 | 14.0 |
| R5 | 0.0005 | 0.4995096345417173 | 60.0 | 2.0 | 0.0 | 22.0 | 12.0 |

R3 is the best tested EMA by final two-window mean (37%) and four-point descriptive mean (40.5%). BFC R4/R5 final means are 14%/12%, differences -23/-25 pp versus R3. Their four-point means are 23.5%/21%. Thus the exploratory +10 pp average benefit criterion fails. R4 improves over same-tau R2 at +100k by +1/+13 pp (s0/s3), but at +50k by -13/0 pp; this is not stable dominance over even the matched-tau baseline.

## Return and final start-region membership

| seed | arm | offset | return | final in start % |
|---|---|---|---|---|
| 0 | R0 | 50000 | -1575.13 | 1.0 |
| 0 | R0 | 100000 | -1993.6 | 96.0 |
| 0 | R1 | 50000 | -1386.84 | 0.0 |
| 0 | R1 | 100000 | -2000.0 | 98.0 |
| 0 | R2 | 50000 | -1431.47 | 0.0 |
| 0 | R2 | 100000 | -1985.17 | 84.0 |
| 0 | R3 | 50000 | -1485.9 | 1.0 |
| 0 | R3 | 100000 | -1372.58 | 0.0 |
| 0 | R4 | 50000 | -1505.01 | 0.0 |
| 0 | R4 | 100000 | -1978.12 | 76.0 |
| 0 | R5 | 50000 | -1567.91 | 0.0 |
| 0 | R5 | 100000 | -1997.15 | 93.0 |
| 3 | R0 | 50000 | -2000.0 | 99.0 |
| 3 | R0 | 100000 | -2000.0 | 99.0 |
| 3 | R1 | 50000 | -2000.0 | 100.0 |
| 3 | R1 | 100000 | -1849.96 | 3.0 |
| 3 | R2 | 50000 | -2000.0 | 98.0 |
| 3 | R2 | 100000 | -1963.1 | 2.0 |
| 3 | R3 | 50000 | -1897.47 | 2.0 |
| 3 | R3 | 100000 | -2000.0 | 99.0 |
| 3 | R4 | 50000 | -2000.0 | 100.0 |
| 3 | R4 | 100000 | -1898.78 | 1.0 |
| 3 | R5 | 50000 | -2000.0 | 100.0 |
| 3 | R5 | 100000 | -1917.94 | 0.0 |

Final start-region membership is an endpoint measure; it is not a claim that the robot remained there for the entire episode. s3 R1/R2/R4/R5 can finish mostly outside the start yet have low success. Leaving the start is insufficient for task completion.

## Mechanism and temporal behavior

| seed | arm | anomaly points/20 | longest run | first offset | final gap rise | final gradient/fork | final action shift |
|---|---|---|---|---|---|---|---|
| 0 | R0 | 6 | 6 | 75000 | 49.758 | 4.097 | 0.675 |
| 0 | R1 | 7 | 7 | 70000 | 46.037 | 3.504 | 0.653 |
| 0 | R2 | 5 | 5 | 80000 | 53.74 | 7.342 | 0.787 |
| 0 | R3 | 0 | 0 | None | 1.167 | 1.207 | 0.526 |
| 0 | R4 | 5 | 5 | 80000 | 55.254 | 7.065 | 0.754 |
| 0 | R5 | 6 | 6 | 75000 | 57.99 | 5.33 | 0.707 |
| 3 | R0 | 3 | 3 | 35000 | 0.865 | 1.939 | 0.706 |
| 3 | R1 | 3 | 3 | 30000 | -13.063 | 0.353 | 0.609 |
| 3 | R2 | 5 | 5 | 30000 | -7.488 | 0.692 | 0.619 |
| 3 | R3 | 2 | 2 | 95000 | 17.413 | 3.022 | 0.727 |
| 3 | R4 | 5 | 5 | 30000 | -2.235 | 0.956 | 0.634 |
| 3 | R5 | 4 | 4 | 30000 | -10.941 | 0.78 | 0.623 |

An anomaly requires BOTH start-minus-overall value-gap rise >=15 and start action-gradient norm >=2 times fork. Two consecutive 5k probes mean not avoided. All arms except s0 R3 meet that failure criterion. This diagnostic is not NaN, a universal instability theorem, or an automatic training stop. Some seed-3 arms recover partial success after an earlier anomaly; counts and endpoint performance must be interpreted together.

| seed | BFC arm | mean clip fraction before first anomaly | max sampled clip fraction | final clip fraction | final mean abs d | final accepted abs d |
|---|---|---|---|---|---|---|
| 0 | R4 | 0.10104 | 0.16016 | 0.14844 | 0.06038 | 0.03838 |
| 0 | R5 | 0.00335 | 0.02344 | 0.01172 | 0.05391 | 0.05163 |
| 3 | R4 | 0.1125 | 0.30078 | 0.14844 | 0.05405 | 0.03774 |
| 3 | R5 | 0.00547 | 0.04688 | 0.00781 | 0.05963 | 0.05841 |

Clipping entries are sampled training-minibatch diagnostics (every 5k), not a full-run or start-state-specific trigger rate. Their presence before anomalies shows the limiter can engage without preventing the measured failure; it does not establish causal protection. Complete Q/gradient cosine/action shift/loss traces are retained in runs/*/probes.json and mechanism_diagnostics.json.

## Paired comparisons and replay limitation

paired_comparisons.json reports per-window, per-checkpoint paired wins/losses and 10,000 episode-bootstrap percentile intervals versus R0, and for BFC versus R2 and R3. These intervals condition on a trained model and the reset distribution; they do not estimate training-seed uncertainty, correct multiple comparisons, or remove selection bias from choosing the best EMA on these same windows.

Five of the nine repaired evaluations have different total rollout lengths from their failed predecessors despite the same declared checkpoint/protocol/RNG schedule. The failed runs did not save complete episode outputs, so strict replay equality cannot be checked and the cause is unresolved. New primary initial states match, but that does not prove full deterministic trajectories. Do not present repairs as bitwise replays or small score differences as robust advantages. No extra replay was run. Exact old/new lengths are in costs.json.

## Actual cost

| Measure | Value |
|---|---|
| optimizer_updates | 1200000 |
| online_updates | 0 |
| primary_conditions | 24 |
| primary_episodes | 2400 |
| failed_reporting_episodes | 900 |
| successful_eval_external_steps | 4381212 |
| failed_eval_external_steps | 1596865 |
| successful_eval_control_calls | 4486812 |
| failed_eval_control_calls | 1636465 |
| preparation_control_calls | 80 |
| total_control_calls | 6123357 |
| total_physics_substeps | 30616785 |
| control_call_cap | 6750000 |
| train_process_residence_hours_upper | 2.137897 |
| eval_process_residence_hours_upper | 0.761471 |
| failed_eval_process_residence_hours_upper | 0.249723 |
| successful_eval_rollout_hours | 0.368606 |
| training_cpu_hours | 1.417305 |
| execution_wall_hours | 0.54926 |
| gpu_busy_hours | NOT_MEASURED; shared-device process residence is not exclusive GPU compute time |
| eval_cpu_hours | NOT_MEASURED |

| seed | arm | GPU | process residence h (upper proxy) | CPU seconds | peak host RSS GiB | sampled own GPU MiB |
|---|---|---|---|---|---|---|
| 0 | R0 | 2 | 0.2018 | 376.9 | 4.272 | 6746 |
| 0 | R1 | 4 | 0.1827 | 499.38 | 4.333 | 6748 |
| 0 | R2 | 2 | 0.2145 | 440.97 | 4.33 | 6748 |
| 0 | R3 | 4 | 0.1731 | 460.35 | 4.328 | 6748 |
| 0 | R4 | 7 | 0.1667 | 441.18 | 4.335 | 6748 |
| 0 | R5 | 7 | 0.1539 | 385.23 | 4.331 | 6748 |
| 3 | R0 | 0 | 0.1723 | 366.94 | 4.274 | 6746 |
| 3 | R1 | 6 | 0.1779 | 456.2 | 4.333 | 6748 |
| 3 | R2 | 6 | 0.1394 | 406.74 | 4.329 | 6748 |
| 3 | R3 | 6 | 0.1809 | 452.6 | 4.324 | 6748 |
| 3 | R4 | 2 | 0.1876 | 407.71 | 4.329 | 6748 |
| 3 | R5 | 0 | 0.1873 | 408.1 | 4.33 | 6748 |

Process residence includes initialization, compilation, probing and polling lag; GPU allocation need not last the whole interval. Shared-GPU residence sums are not exclusive GPU hours. GPU memory values are observed per-process samples, not guaranteed continuous peaks. Evaluation CPU usage was not recorded. Failed evaluation physics counts are derived as control calls times fixed frame_skip=5, verified against all successful evaluations. Preparation includes two constructor resets (80 control calls), not zero physical work.

## Decision and bounds

| Question | Decision |
|---|---|
| Is BFC V1 ready for broader validation? | NO-GO: fails anomaly prevention and advantage over best EMA. |
| Is a pure EMA already a stable fix? | DIAGNOSE: R3 works in s0, fails in s3. |
| Should this run add seeds, tune epsilon or train from scratch? | No; outside authorization and unsupported by this first-round screen. |

These are two selected warm-start development windows, not independent broad validation. Original source checkpoints lack historical NumPy/Python sidecars: branches restore model/Adam/Agent RNG and pair their new sampling, but do not claim exact replay of the historical training. B1a is historical context under a different evaluation schedule and is not used as a matched numerical comparator. No normal-window, long-horizon, from-scratch or task4 effectiveness claim is supported.

Full weights, data and per-step trajectories remain on the server. Final lightweight artifacts: summary.json, checkpoint_index.json, paired_comparisons.json, mechanism_diagnostics.json, costs.json, final_validation.json and diagnostics.png.
