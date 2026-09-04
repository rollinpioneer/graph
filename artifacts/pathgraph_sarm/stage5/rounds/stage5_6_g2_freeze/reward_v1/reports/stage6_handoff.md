# Stage 6 Handoff

- entry: G2=GO_STAGE6
- reward_config: configs/reward_config_v1.yaml
- reward_engine: code/reward_engine.py
- model_bundle: configs/model_bundle.json
- weight_schema: configs/stage6_weight_schema.json
- statistics_unit: content_group_id
- reward_selection_locked_before_test: true
- checkpoint_files: referenced_only_not_packaged
