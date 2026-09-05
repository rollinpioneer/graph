# U3 execution summary

## Decision

`U3_INCONCLUSIVE`. No candidate is handed to U4.

## Execution

- Primary model: `qwen3.7-plus`; cross-check: `deepseek-v4-flash`; text only.
- Formal candidate calls: 12 Qwen (9 original plus the one permitted three-request fallback refinement) and 3 DeepSeek, exactly the fixed candidate cap of 15.
- Each DeepSeek primary response used its one permitted schema-repair call successfully. Repairs are corrective calls rather than new candidates, so external model invocations total 18. No more model calls are permitted in this U3 round.
- Every prompt was split-pure train-only; val/test prompt leakage and API-key leakage were both 0.

## Hard validation

- Qwen: 2 hard-valid candidates, both instruction-only.
- Qwen auto-segment: 0 hard-valid candidates.
- Qwen budgeted-fallback: 0 hard-valid candidates after the one permitted prompt refinement.
- DeepSeek: 0 hard-valid candidates. It remains a structural cross-check only, never a substitute for Qwen selection.

The failed candidates are retained as sanitized artifacts and failed their deterministic checks for malformed/truncated JSON, unsupported evidence IDs, unsupported transition pairs, or instruction-only evidence citations. They were not manually repaired or promoted.

## Scope

All graphs remain hypothesized and are limited to the same stochastic simulator family. The U2 boundary fallback remains required; nothing here supports a physical robot, original-task, or unseen-family generalization.
