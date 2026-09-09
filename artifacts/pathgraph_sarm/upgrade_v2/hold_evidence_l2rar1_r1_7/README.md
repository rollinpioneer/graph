# L2RA-R1.7 Execution Summary

This compact audit package records the complete local R1.7 execution without
including raw rollout observations or derived feature caches.

- Environment: `/home/xushijie/.venvs/point-bridge/bin/python`, Python 3.11.16
- Development execution: 16 fit families / 64 rollouts and 16 select families / 64 rollouts
- Development route: `DEVELOPMENT_NOT_READY`
- Eligible candidates: `0/8`
- Confirmation: `NOT_RUN_DEVELOPMENT_FAILED`
- Final status: `L2RAR1_PARTIAL_KEEP_G1`
- Retained graph: `G1_predicate_bound`
- L3 entry, R4 consumption, and training: closed / unauthorized
- Attempt-boundary diagnostic: 16 `dev_select` cases; online state did not separate missed grasp from short touch at first contact loss

Raw paired observations and feature caches remain at the local run root
`/tmp/l2ra_r1_7_full` and are intentionally excluded from Git.
