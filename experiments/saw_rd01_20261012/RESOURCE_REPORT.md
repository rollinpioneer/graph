# RESOURCE_REPORT

Actual conservative GPU residency: 14069.568725 seconds = 3.908214 GPU hours, authorized 12.
S0: 83.538117/7200 seconds; S1: 13986.030608/28800 seconds; S2–S4 GPU: 0/7200 seconds (CPU-only).
Aggregate task memory peak: 2460 MiB = 2.402344 GiB, cap 12 GiB.

| Job | Updates | Whole GPU-child wall seconds | Exit | Stop |
|---|---:|---:|---:|---|
| pilot_navigate | 1011 | 41.691577 | 0 | None |
| pilot_stitch | 1011 | 41.846539 | 0 | None |
| formal_navigate | 1000000 | 6825.210190 | 0 | None |
| formal_stitch | 1000000 | 7160.820419 | 0 | None |

122/128 local diagnostic policy segments; 128/128 candidate state-goal groups, ≤16 candidates each. Two independent seed-0 baselines, no extra seed, QRL, or actor-only training.

All complete GPU-child wall time charged, including import/load and CPU evaluation while GPU model resident; CPU-only final/diagnosis processes explicitly hide CUDA. Formal CPU ticks are separate lower-bound snapshots. Initial CPU install/audit timing was not fully instrumented; logs retained, no invented exact CPU cost.

CPU final-evaluation and diagnosis cost is tabulated in `results/RESOURCE_TOTAL.json`; formal process CPU ticks in `receipts/formal_CPU_ACCOUNTING.json`. Resource monitor overhead and initial CPU setup are not represented as exact per-module CPU costs. Shared GPU load changed after the pilots; measured completed job times are the outcome, pilot projections are estimates.
