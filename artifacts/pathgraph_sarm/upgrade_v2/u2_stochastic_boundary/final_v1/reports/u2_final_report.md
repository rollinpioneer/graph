# U2 stochastic-boundary prototype — final report

## 已支持

- Explicit-state stochastic simulator scope: 720 episodes / 120 root families; 40/40 snapshot restoration anchors passed.
- Weak posterior, unknown state, rule baselines, causal models, segment summaries, fixed oracle-clip budget comparisons, and independent continuation q/D references were executed.
- Boundary source locked on validation only: `offline_teacher_to_causal_s623` (causal_model).
- Observation/action transition alignment audit: `U2_OBSERVATION_ACTION_ALIGNMENT_PASS`; max absolute error = `0.0`. The frozen contract is `observations[t]=x_t; actions[t]=a_{t-1} enters x_t; observations[t][15:17]=actions[t]`.

## 部分支持

- Decision: `GO_U3_WITH_BOUNDARY_FALLBACK`. U3 may consume unknown-aware simulator segment candidates; U4 must retain boundary fallback.
- Best causal variant by validation: `offline_teacher_to_causal`; test boundary F1±2 = 0.4771 ± 0.0792; it does not meet the 0.75 automatic-Gate threshold.
- Weak recovery-start recall = 1.0000; active query supported = False (oracle budget, not human time).

## 未支持

- Fully automatic boundary promotion without fallback.
- Physical robot, original task, or unseen-family generalization.

## 奖励归因

- Gold failure-negative rate: 0.4026; locked automatic/rule: 0.0990.
- Gold recovery-positive rate: 0.0000; locked automatic/rule: 0.9595.
- Event-segment results are simulator-only attribution analyses; episode-level potential return remains telescoping.

## 范围限制

- All claims are restricted to the explicit stochastic simulator family and the frozen 120 root-family split.
- The single ZIP policy supersedes per-round archive delivery: round-level manifests, checksums, and omission records are included inside the one cumulative archive.

## Entering U3/U4

Use `u3_u4_handoff.json`, `segment_event_summary.jsonl`, `cluster_prototypes.json`, and the observed transition table. Preserve `unknown` and simulator-boundary fallback.
