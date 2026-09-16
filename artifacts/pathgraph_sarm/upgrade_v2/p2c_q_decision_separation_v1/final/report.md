# P2C-Q 完成报告

## 三句话结论

任务是否真正产生决策分离：是。开发集 2048 槽全部进入分母，有效决策分离 1620 对（每类 352–496，root 均为 32，同合法 mask 子集同步满足），环境门 `ENVIRONMENT_DECISION_SEPARATION_PASS`。确认集 status=`ENVIRONMENT_DECISION_SEPARATION_PASS`，分离对 `1620`。

Graph相对GEOM与Flat增加什么：在开发集关键非终端状态上，Graph conservative oracle hit=0.5920，GEOM=0.1074，Flat=0.1074（Graph−GEOM≈0.485）。GEOM 与 Flat 几乎相同，增量不能用“合同扁平规则已足够”解释；但 Graph 总体未达预登记 90% hit，strict top-set disagreement=0.0901（低于 15% 目标），并列无效质量 Graph 远低于 GEOM。

下一阶段建议与仍未验证的主张：科学路由 `READY_FOR_PPO_UTILITY_PREREGISTRATION`。本轮 0 训练，`training_release=false`，`confirmation_passed=false`，`policy_utility_evidence=NOT_EVALUATED_IN_P2C_Q`。一步塑形排序不是 PPO 策略效用。历史 P1/P2A/P2B/D1 结论未改写。

## Git与来源

- base: `0857173dcd1ddbbb781217c962007d2fa702fe12`
- implementation: `7609dfa56d1f3f85046c096f08d39b06457628a8`
- result commit: 记录于推送后 handoff，不写入本 manifest 以防自指
- 执行树 HEAD=7609dfa56d1f3f85046c096f08d39b06457628a8 clean=True
- 共享 clone 保持 `research/l2ra-completion-audit-v2`，未改 main
- origin/main 现场记录见 `acceptance_evidence/source_provenance.json`
- protocol SHA256: `0f98c4854e5be8aedfd2b2d175b6266a23a4347f500372acd51bdd8af0f62b60`

## 范围

new_training_jobs=0；optimizer_updates=0；physical_executions=0；model_api_calls=0。

正式账目：P2CQ_DEV_EXPORT_001、P2CQ_DEV_QUALIFY_001、P2CQ_CONFIRM_EXPORT_001、P2CQ_CONFIRM_QUALIFY_001，均为 RESERVED_NO_REFUND。

## 环境资格

开发：planned=2048，separating=1620，omitted=0，clip=0，unreachable=0，n_mdp_states=456984。
按类：PRECEDENCE 352、SHARED_PREREQUISITE 496、ALTERNATIVE_COST 356、INVALIDATION_RECOVERY 416；每类 families=32，shared_legal 与 separating 相同。

确认：planned=2048，separating=1620，status=ENVIRONMENT_DECISION_SEPARATION_PASS。

## 候选信号

开发关键态 n=3240。
Graph conservative hit=0.5919753086419753；GEOM=0.10740740740740741；Flat=0.10740740740740741。
strict_top_set_disagreement=0.09012345679012346；top_set_equality=0.14444444444444443。
graph_invalid_tie_mass=0.016790123456790124；geom_invalid_tie_mass=0.6680013346680014。
PRECEDENCE 上 Graph hit=1.0，GEOM/Flat≈0.045。INVALIDATION 上 Graph hit≈0.345，为最弱类。
命中率与策略效用不可混同。

## 工程接受

10 组服务器检查均为 PASS，证据在 `acceptance_evidence/` 与本目录审计 JSON。
研究模块 pytest 22 passed；包 `verify_package.py` exit 0。
`validate_server_acceptance.py` 只验证记录与哈希，不替代科学检查。

## 科学决定

environment_qualification：ENVIRONMENT_DECISION_SEPARATION_PASS
candidate_signal_status：STRUCTURE_SIGNAL_PRESENT_BELOW_90_HIT_TARGET
benchmark_confirmation_status：ENVIRONMENT_DECISION_SEPARATION_PASS
policy_utility_evidence=NOT_EVALUATED_IN_P2C_Q
confirmation_passed=false
training_release=false
历史P1/P2A/P2B/D1状态未更改。
