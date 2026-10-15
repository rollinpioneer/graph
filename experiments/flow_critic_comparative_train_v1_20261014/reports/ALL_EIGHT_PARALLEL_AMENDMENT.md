# Parallel scheduling amendment — 2026-10-14

The user explicitly instructed: 八条全部开始，不要等待. This supersedes the baseline-completion barrier and the preference against sharing own-account GPUs. It changes scheduling and CPU affinity only; tasks, seeds0/1, source commits, model/loss hyperparameters, precision, batch256, evaluation and the8M formal cap remain frozen.

All four baseline native and worker PIDs were adopted unchanged. All four previously unstarted combinations were dispatched once at16:21:42 Asia/Taipei: G1_seed0 GPU2, G1_seed1 GPU3, D1_seed0 GPU5(shared with D0_seed0), D1_seed1 GPU7(shared with G0_seed1). Native child PID receipts and first real-update/full-state restore admissions exist for all four.

Current training entrypoint: code/all_eight_manager.py; code/run_manager.py remains the native worker implementation. The active manager has no T1 completion dependency for the combinations. T2 remains50k included within1M, then T3 resumes verified complete state once. No optimizer warmup/discarded trials. No failed-job retries.

CPU groups: baseline64–95 unchanged, G1 seeds0/1 cores32–39/40–47, D1 seeds0/1 cores96–103/104–111.8 cores/job, BLAS threads1. Current available RAM exceeds330GiB. Safe free-memory admission accounts for JAX allocator cap plus MuJoCo render workers and4GiB headroom; no batch/precision reduction.

Read-only evaluation scheduling permits safe same-account sharing rather than falsely failing on a busy but sufficiently free card. Its active child was absent when the evaluation controller was adopted. No training/native evaluation process was stopped.

Prior startup attempts and prior scheduling receipts remain intact; these are an amended execution of the same card/branch, not new models or tasks.
