# One-time Qwen fallback-condition prompt refinement

- Trigger: all three `instruction_plus_budgeted_train_fallback` Qwen outputs ended with `finish_reason=length` and failed JSON parsing.
- Scope: only the three failed fallback-condition requests.
- Unchanged: model (`qwen3.7-plus`), task contract, predicate vocabulary, schema, train-only evidence, ordering, token limit, and input modality.
- Changed once: an output-format instruction requires the smallest schema-compliant graph to avoid truncation.
- Candidate call accounting: original Qwen 9 + refine Qwen 3 + DeepSeek 3 = 15, the fixed cap.
- No further Qwen prompt refinement or candidate generation is allowed in this round.
