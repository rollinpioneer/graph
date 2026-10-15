# U3 run manifest

- round: `u3r4_deepseek_crosscheck_candidates`
- source_commit: `3db13d277d36701535847a274547b5fccb4785c0`
- branch: `upgrade/u3-qwen-deepseek-candidates`
- deepseek_model: `deepseek-v4-flash`
- input_modality: `text_only`
- request_count_planned: `3`
- request_count_completed: `3`
- response_mode: `json_object_local_validation`
- schema_repair_count: `3` (one successful corrective repair for each DeepSeek primary response)
- external_provider_invocations: `6` (3 formal candidate requests + 3 schema-repair calls; repairs are not new candidates)
- test_gold_in_prompt: `false`
- API key logged: `false`
- status: `DEEPSEEK_CROSSCHECK_COMPLETE`
- local schema valid: `3/3`
- output ZIP: `not separately published; user single-ZIP policy`
