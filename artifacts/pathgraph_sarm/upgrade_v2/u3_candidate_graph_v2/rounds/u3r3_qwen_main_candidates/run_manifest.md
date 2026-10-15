# U3 run manifest

- round: `u3r3_qwen_main_candidates`
- source_commit: `3db13d277d36701535847a274547b5fccb4785c0`
- branch: `upgrade/u3-qwen-deepseek-candidates`
- qwen_model: `qwen3.7-plus`
- input_modality: `text_only`
- request_count_planned: `9`
- request_count_completed: `9`
- response_mode: `json_object_local_validation`
- network_retry_count: `recorded per request in usage/qwen_usage.csv`
- schema_repair_count: `0`
- test_gold_in_prompt: `false`
- API key logged: `false`
- status: `QWEN_MAIN_COMPLETED_WITH_FORMAT_FAILURES`
- local schema valid: `4/9`; the remaining five ended at length and were preserved as failed sanitized responses.
- output ZIP: `not separately published; user single-ZIP policy`
