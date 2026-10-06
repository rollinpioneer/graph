# CP-DISR-C1-BW-HOP-CLOSEOUT-A01 (user decision, recorded before any training)

Recorded states:

```text
AMENDMENT_A01_ACCEPTED
HOP_GT4_STATUS=STRUCTURALLY_EMPTY
RECEPTIVE_FIELD_STATUS=NOT_TESTABLE_STRUCTURALLY_EMPTY
RESUME_FROM=E0
AUTHORIZED_TRAINING_RUNS=3
```

Baseline: repository `rollinpioneer/graph`, base branch `codex/cp-disr-c1-mechanism-confirmation-v1`, base commit `1f7b4f6cabd4c95ce0eff86f3b51e8aa6217312e`, target branch `codex/cp-disr-c1-blocksworld-main-v1`.

User decision text (verbatim, 2026-10-06):

> 批准采用方案 A，并执行：CP-DISR-C1-BW-HOP-CLOSEOUT-A01
> 你之前按照 Runbook §9.3 停止是正确的。本次明确决定：
> 1. 保持生产合同图和 candidate_goal_hops 定义不变；
> 2. 不删除 HandEmpty 或任何生产消息边；
> 3. 不人工指定“候选所服务的目标”；
> 4. 将 hop > 4 记为结构空集：HOP_GT4_STRUCTURALLY_EMPTY；
> 5. 将 receptive-field 状态记为：NOT_TESTABLE_STRUCTURALLY_EMPTY；
> 6. 删除 hop 配额、H5 实验和 LIMIT_RECEPTIVE_FIELD 分类；
> 7. B 切片继续用于 n=6–8 的规模、计划长度、图大小、目标干扰和计算成本分析；
> 8. 接受 n=3 train 的 24 个同构类各重复两次，dev 12 类与 train 完全不重叠，并完整披露 multiplicity；
> 9. 复用已完成且输入未改变的合同、规划器、B2-CACHED、QMARK 和 split 生成检查；
> 10. 运行修订后的 prep verifier，写入正式 split 并冻结；
> 11. 随后执行且仅执行三条训练：R-C1-BW-B2-0 / R-C1-BW-QMARK-0 / R-C1-BW-ASNET-0；
> 12. 三条通过 ID Gate 后，一次性执行 A0/A1/A2/B 正式评估；
> 13. 不增加 seed，不自动切换 imitation，不开发新算法。
> 训练要并行；如果磁盘空间实在不够用了，可以 ssh lab-xushijie3 使用空间。

Evidence that triggered the stop (Runbook 9.3): on the unchanged production message graph every candidate action is 1 or 3 hops from some unmet goal proposition (about 5,700 candidate decisions at n = 5, 6, 8; the HandEmpty proposition is a hub of almost every action); removing HandEmpty in a counterfactual still left 5 hops essentially unobserved (24 of about 5,700). The hop > 4 quota of the B slice (at least 30 optimal decisions with hop > 4) is therefore unconstructible; per A01 nothing in the production graph or the hop definition was changed.

Disclosed consequences: the A1/A2 per-case hop coverage requirement is trivially satisfied (all decisions are hop <= 4, `D_core` is the plain A1/A2 decision-perfect rate); `LIMIT_RECEPTIVE_FIELD` and the H5 experiment do not exist; `hop_strata.csv` is reported descriptively and contains no hop > 4 stratum; the n = 3 train split uses 24 problem-isomorphism classes twice each (dev: 12 distinct classes with initial-state classes disjoint from train).
