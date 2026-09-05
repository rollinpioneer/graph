# U3 candidate-graph input run manifest

- round_id: `u3_candidate_graph_inputs_v1`
- purpose: construct train-only prompt inputs and pending candidate requests
- scope: same explicit-state stochastic simulator family only
- baseline_commit: `4fd1daa373f7f7cdf99604476c18af2644682a63`
- execution_date: `2026-09-05`
- export_command: `python -m upgrade_v2.u2_handoff_patch.cli export-u3-train --split train --include-unknown true --fallback-max-clips 30`
- freeze_command: `python -m upgrade_v2.u2_handoff_patch.cli freeze-handoff`
- train_input: 504 episodes, 84 root families, unknown retained
- excluded_from_prompts: val/test episodes and test gold
- llm_execution: `MODEL_EXECUTION_PENDING`; 9 requests planned, 0 candidates completed
- gpu: not used; no local model or authorized API was available
- outputs: `prompt_input_manifest.json`, `fallback_policy.json`, `candidates/`, and `u3_proposal_handoff.json`

