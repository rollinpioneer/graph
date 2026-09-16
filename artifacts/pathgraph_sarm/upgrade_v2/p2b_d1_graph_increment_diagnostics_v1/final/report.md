# P2B-D1 Graph Increment Diagnostics

## Git

- base commit: `0f26660cde901aa32b0cfc032fc75ec01b31687b`
- branch: `research/pathgraph-p2b-d1-graph-increment-diagnostics-v1`
- result commit: filled after commit
- main unchanged: yes

## Execution boundary

- new training jobs: 0
- optimizer updates: 0
- gradient steps: 0
- checkpoint files modified: 0
- diagnostic deterministic episodes: 0
- diagnostic stochastic episodes: 0
- physical simulation: not run
- robot/vision evaluation: not run

## Input integrity

- final checkpoints matched: 48/48
- milestone checkpoints present: 288/288
- test episodes verified: 98304/98304
- fixed subset: 48/48 policies x 512 episodes PASS
- missing artifacts: 0

## Zero-seed diagnosis

- TASK_ONLY:211: `ZERO_SEED_PARTIAL_TASK_STALL_SUPPORTED` (training successes=0, validation=NEVER_DETERMINISTIC_VALIDATION_SUCCESS)
- TASK_ONLY:239: `ZERO_SEED_TRAIN_SUCCESS_NOT_RETAINED` (training successes=60, validation=NEVER_DETERMINISTIC_VALIDATION_SUCCESS)
- COUNT_EVENTS_PBRS:227: `ZERO_SEED_PARTIAL_TASK_STALL_SUPPORTED` (training successes=0, validation=NEVER_DETERMINISTIC_VALIDATION_SUCCESS)
- COUNT_EVENTS_PBRS:233: `ZERO_SEED_PARTIAL_TASK_STALL_SUPPORTED` (training successes=0, validation=NEVER_DETERMINISTIC_VALIDATION_SUCCESS)
- COUNT_EVENTS_PBRS:239: `ZERO_SEED_TRAIN_SUCCESS_NOT_RETAINED` (training successes=1, validation=NEVER_DETERMINISTIC_VALIDATION_SUCCESS)

## Graph vs GEOM

- graph diagnostic: `GRAPH_SIGNAL_LARGELY_REDUNDANT_IN_CURRENT_BENCHMARK`
- GRAPH_ONLY_NONZERO + OPPOSITE_SIGN rate: 0.000186
- GEOM alias groups: 2981
- one-step top-1 agreement: 0.986000
- one-step states: 500

This is a frozen development diagnostic. It does not replace the original P2B result and does not establish a policy increment for PathGraph.

## Historical state

- P1 representation claim: unchanged
- P1 reward-accounting claim: unchanged
- P2A decision: unchanged
- P2B decision: unchanged
- robot policy claim: false
- physical cycle claim: NOT_EVALUATED
- confirmation_passed: false
