# FLOQ monitoring handoff

Receipt snapshot (server time): 2026-10-13T23:37:37.006347.

| Job | Updates | Phase | Matched checkpoints done |
|---|---:|---|---:|
| FQL-0 | 1000000 | MATCHED_EVALUATING | 9 |
| FQL-1 | 1000000 | MATCHED_EVALUATING | 8 |
| GFP-0 | 1000000 | COMPLETE | 10 |
| GFP-1 | 1000000 | COMPLETE | 10 |
| floq-P1 | 5000 | P1_PASSED | 0 |
| floq-0 | 140000 | NATIVE_EVALUATING | 0 |
| floq-1 | 149000 | TRAINING_RUNNING | 0 |

No numerical failure, process failure, invalid completed evaluation or indexed checkpoint error found. The manager is alive. No unregistered account training/evaluation/GPU jobs were found.

FLOQ native evaluation normally pauses updates. First formal checkpoint is scheduled at 1M; none is due at the current steps. Early native curves are intermediate observations, not the frozen 2M primary result.

FLOQ GPU 3/6: own peak memory 954 MiB per run; process-tree RAM peak below 2 GiB per run. Whole-card utilization includes other users.

The 30-minute heartbeat remains ACTIVE. Q-Flow remains PAPER_REPRODUCTION_BLOCKED. No new training, restarts or parameter changes. Prior FQL/GFP scores are excluded from this new-card Git archive.

Full trace: ROOT/receipts/monitor/latest.json; lightweight trace: patrols/20261013_2330.json.
