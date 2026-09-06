# U4-R1 final handoff

- decision: `STOP_AUTO_GRAPH_KEEP_MANUAL`
- historical U4 status: `U4_COMPLETE_NO_EDIT_GAIN` (unchanged)
- evaluator role_condition executed: `True`
- horizon: `censored_unknown` and excluded from failure-terminal claims
- semantic separability: `CONDITIONAL_SEMANTICS_JUSTIFIED`
- selected graph: `G3_guard_rule_multigraph`
- checkpoint inference: `cupid` CPU, torch `2.8.0+cu126`; CUDA unavailable
- API key read: `false`; API calls: `0`; training jobs: `0`
- U2 formal train occurrences: `11,944` from `504` train episodes; labels diagnostic-only
- fresh confirmation: `36` families, `144` rollouts, `1,540` occurrence rows
- terminal/failure/recovery denominators: `24` / `33` / `33` / `30`; censored horizon: `1`
- scenario non-inferior strata: `0/6`
- no physical robot or new-task generalization claim
