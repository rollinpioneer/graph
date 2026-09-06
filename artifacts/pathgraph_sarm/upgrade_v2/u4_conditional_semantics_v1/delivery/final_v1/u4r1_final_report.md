# U4R1 final handoff

- exit: `STOP_AUTO_GRAPH_KEEP_MANUAL`
- selected graph: `G2_conditional_multigraph`
- D-GATE: `CONTINUE_U4`
- automatic boundary: `computed_and_locked`
- checkpoint: `offline_teacher_to_causal_s623` (`fe0464076a3590de19b31d88cd668d4c0e8cf92ee2a80ec413e8191fea34c94e`)
- runtime: `cupid` CPU inference, torch `2.8.0+cu126`; CUDA was unavailable
- confirmation: 36 families, 144 rollouts, 1,540 occurrence rows; lock gate `PASS`
- G2 minus G1 paired effects: zero for transition, typed coverage, failure recall and recovery recall; 36 paired families
- C04/C10: conditional role guard installed; horizon remains censored and nonterminal occurrences do not activate failure role
- separability: real majority/rule/logistic/tree-depth-4 baselines; observable-only feature audit found 0 forbidden-field violations
- API calls: `0`; API key reads: `0`; training jobs: `0`
- historical U4 B+ fallback result was preserved; this U4R1 confirmation used the recovered PyTorch path
