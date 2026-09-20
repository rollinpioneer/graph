# Stage 0A 绑定报告

**BLOCKED**。已绑定 host、repo、分支、真实硬件/OS、软件环境；核心运行资源仍未绑定。

- 软件安装/验证：推荐 profile 虽已解析，但安装/导入/数值检查未全部通过；详见 host_logs 与 software_checks.json。
- 运行平台：simulator_or_robot、environment_version、task_assets。
- 技能与事实：controller_manifest、skill_contracts、verifier_thresholds；真实前置/ADD/DEL、三态证据与标定。
- 视觉：camera、calibration_manifest、perception_checkpoint。
- 时间与安全：skill_timeouts、task_deadlines、actual_interaction_time_unit、reference_skill_seconds_by_task、safety_authorization。
- 独立评价：task_evaluator_version；真实终端成功和结束规则。
- 实现入口：runtime.entrypoints；train_cli/eval_cli/tests_cli/cache_cli/report_cli；模型源码 hash 与参数量。
- 任务与 split：D0/T_A/T_C 的 asset_ref、controller_refs、goal_atoms、skill_contract_refs、camera_verifier_refs、initial_state_generator、deadline_seconds、reference_skill_seconds、split_ref；64/50/50 实例池。
- VLM：region、base_http_api_url、api_account_authorized、fewshot_manifest；seed 参数实际支持能力另待核实。
- 资源配置：实际 num_envs、candidate view 分块与运行时钟来源；GPU 清单不是专属训练资源预约。

完整逐字段证据见 binding_table.csv 与 unbound_must_bind.csv；完整阶段报告见 stage_0a_summary.md。
