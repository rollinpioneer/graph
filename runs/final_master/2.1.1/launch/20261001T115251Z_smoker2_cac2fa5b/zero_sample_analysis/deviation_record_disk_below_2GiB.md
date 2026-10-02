# Execution deviation — /home free space below 2 GiB, guard did not pause

Recorded 2026-10-02T02:27:29Z. Classification: operational execution deviation, no integrity defect found, no change to any run, result or protocol. No retrain was started and no existing log or file was overwritten or deleted.

## Three different quantities (kept apart)

| quantity | value | what it is |
|---|---|---|
| budget estimate | 2151430888 B (2052 MiB) per run x 3 x 1.25 margin + 2 GiB reserve = 8601776312 B required; free 10099720192 B at 2026-10-01T12:01:40Z; passed | launch-time admission estimate (`storage_budget.json`). Actual run dirs: E-0 2224 MiB, K-1 1716 MiB, DK-1 1581 MiB; E-0 exceeded the 2051 MiB estimate by ~173 MiB |
| actual pause threshold | SIGSTOP below 1.0e9 B, resume at 7.0e9 B, polled every 2 s (`storage_guard.py`, live from 2026-10-01T17:51:06Z) | what the guard could act on. Minimum observed 1.665 GB > 1.0 GB, so it never fired (no SIGSTOP; exit log `GUARD_EXIT_RESUMED_ALL`, paused []) |
| operational safety margin | 2 GiB (2,147,483,648 B) | the reserve in the storage gate and the >= 2 GiB condition of the concurrency amendment; the amendment applies it as a launch-time check for K-1 — it is not a pause rule |

The deviation: free space was below the 2 GiB margin for 6 consecutive 30-s samples (2026-10-02T00:14:46Z-00:17:17Z, about 3 min of samples; minimum sampled 1.665 GB at 00:16:17Z), and the guard I configured stops only below 1.0 GB, i.e. its threshold was lower than the margin, so the margin could not have been enforced by it. The guard worked as configured; the configuration was mine. The true minimum between 30-s samples is NOT_MEASURED.

## Timeline from `throughput/samples.jsonl`

- below 4 GB: 2026-10-01T23:22:27Z-2026-10-02T00:18:48Z (113 samples); below 3 GB: 00:14:46Z-00:18:48Z; below 1 GB: never.
- 3.019 GB (00:14:16Z) -> 2.010 -> 1.668 -> 1.665 (00:16:17Z-00:16:47Z) -> 1.731 (00:17:17Z); guard heartbeat 5.05 GB at 00:21:27Z; 9.84 GB at the last monitor sample (00:40:26Z).
- Only R-TB-DK-1 was running (E-0 and K-1 were COMPLETE). It appended about 88 transition lines (13959 -> 14047, roughly 0.2 MB) in that window; the free space fell by about 1.35 GB in at most 60 s. That is consistent with other users writing to the shared filesystem; per-process attribution is NOT_MEASURED and not claimed.

## File integrity (zero new samples; `integrity_lowdisk_check.json`, `checkpoint_identity.json`)

| run | JSONL lines unparsable | transition lines == N | JSON files parsed / unparsable | checkpoints loaded / failed | checkpoint vs persistence copy sha compares / mismatches | lowest free at a checkpoint publish | write-error patterns in train log |
|---|---|---|---|---|---|---|---|
| R-TB-E-0 | 0 | True | 129 / 0 | 19 / 0 | 76 / 0 | 5.79 GB | none |
| R-TB-DK-1 | 0 | True | 123 / 0 | 18 / 0 | 72 / 0 | 3.13 GB | none |
| R-TB-K-1 | 0 | True | 129 / 0 | 19 / 0 | 76 / 0 | 5.10 GB | none |

All checkpoint publishes happened at >= 3.1 GB free (DK-1 update 13 at 23:43:53Z was the lowest), i.e. none inside the below-2-GiB samples; the DK-1 final generation was published at 00:31-00:32Z with > 9 GB free. Final checkpoints of the three runs: sha256 recomputed now equals the manifest, manifest N/T/attempt id equal job_summary, the fresh-load file names the final generation (adam, model, fresh_process all true). These checks show structural and internal consistency; they cannot show bit-exactness against a reference that does not exist.

## Treatment
- No automatic retrain, no rollback, no log overwrite or deletion; the three runs stand as completed and are reported as such.
- Not enacted, only proposed for any future launch: set the guard stop threshold at or above the 2 GiB margin and check free space before each checkpoint publish. Needs your decision with any new launch.
