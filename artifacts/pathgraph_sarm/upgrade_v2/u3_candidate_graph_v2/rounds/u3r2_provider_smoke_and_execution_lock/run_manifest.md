# U3 run manifest

- round: `u3r2_provider_smoke_and_execution_lock`
- start_time: `2026-09-05T23:10:33+08:00`
- end_time: `2026-09-05T23:20:15+08:00`
- source_commit: `3db13d277d36701535847a274547b5fccb4785c0`
- branch: `upgrade/u3-qwen-deepseek-candidates`
- qwen_model: `qwen3.7-plus`
- deepseek_model: `deepseek-v4-flash`
- input_modality: `text_only`
- request_count_planned: `0`
- request_count_completed: `0`
- network_retry_count: `0`
- schema_repair_count: `0`
- test_gold_in_prompt: `false`
- API key logged: `false`
- status: `AUTHENTICATION_FAILED`
- note: `Both minimal schema-only provider smoke calls returned HTTP 401. After the user updated the two-line source, the private mapping was refreshed and both schema-only smokes were retried; both still returned HTTP 401. The external key mapping and source format were verified without printing values; no formal task evidence was sent.`
- output ZIP: `not separately published; user single-ZIP policy`
