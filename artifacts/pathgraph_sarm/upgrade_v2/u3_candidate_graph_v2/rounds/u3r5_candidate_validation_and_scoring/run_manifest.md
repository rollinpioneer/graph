# U3 run manifest

- round: `u3r5_candidate_validation_and_scoring`
- source_commit: `3db13d277d36701535847a274547b5fccb4785c0`
- input_modality: `text_only`
- original Qwen calls: `9`
- one permitted fallback-condition refinement calls: `3`
- DeepSeek cross-check calls: `3`
- total formal candidate calls: `15` (fixed cap reached; 9 Qwen original + 3 Qwen refinement + 3 DeepSeek primary)
- DeepSeek schema repair calls: `3` (one successful corrective repair per primary response; repairs are not new candidates)
- total external model invocations: `18` (15 formal candidate calls + 3 repair calls)
- test_gold_in_prompt: `false`
- API key logged: `false`
- hard-valid Qwen: `2/6 parsed outputs`; hard-valid DeepSeek: `0/3`
- hard-valid conditions: `instruction_only=2`, `instruction_plus_auto_train_segments=0`, `instruction_plus_budgeted_train_fallback=0`
- status: `U3_INCONCLUSIVE`
- output ZIP: `not separately published; user single-ZIP policy`
