# PathGraph-SARM L1V Final Report

- Decision: `L1V_READY_FOR_REFINEMENT_MULTIVIEW`
- Selected coarse graph condition: `V2`
- Source commit: `b1d33f3c83e99c914068f1fa3f4ab57fb473250b`
- Input: 24 simulator RGB cases, 12 root families, 48 same-state views
- Development: 6 cases / 18 candidates; confirmation: 18 cases / 54 candidates
- Model: `qwen3.7-plus`; DeepSeek calls: 0; training jobs: 0
- External attempts: 74/80; HTTP success: 74/74; structure valid: 74/74
- Prompt tokens: 119166; completion tokens: 106900; estimated cost: CNY 1.093532

## Confirmation

- T scene-ready: 9/18; paired families: 0/9
- V1 scene-ready: 14/18; paired families: 5/9
- V2 scene-ready: 17/18; paired families: 8/9
- V1-T family effect: 0.277778 [0.111111, 0.444444]
- V2-V1 family effect: 0.166667 [0.000000, 0.333333]
- Unsupported visual assertions: T=0/18, V1=1/18, V2=0/18
- Review: one traceable Codex-agent semantic reviewer; no inter-rater coefficient

## Interpretation

V1 met the absolute scene-ready and gain requirements but reached only 5/9 paired-ready families, below the predeclared 6/9 threshold. V2 reached 17/18 scene-ready and 8/9 paired-ready families, with a 0.444444 family-level effect over T, so only the predefined multiview refinement route is opened.

This result is limited to explicit MuJoCo primitive tabletop scenes. It does not establish physical executability, grasp stability, collision avoidance, reward learning, policy improvement, real-camera performance, or robot generalization. Historical U3/U4 conclusions were not modified.
