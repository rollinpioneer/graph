# Provider run verification

- provider: `qwen`
- expected: `9`
- status_rows: `9`
- http_success: `9`
- content_nonempty: `9`
- schema_valid: `4`
- parsed_files: `4`
- status: `FAIL`

## Failures

- instruction_plus_auto_train_segments_r01: missing parsed candidate
- instruction_plus_auto_train_segments_r03: missing parsed candidate
- instruction_plus_budgeted_train_fallback_r01: missing parsed candidate
- instruction_plus_budgeted_train_fallback_r02: missing parsed candidate
- instruction_plus_budgeted_train_fallback_r03: missing parsed candidate