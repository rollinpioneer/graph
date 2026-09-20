# Stage 0B — Unit & Invariant Tests

**最终状态：PASS。Stage 0A 保持 BLOCKED；Stage 0C 未启动。**

本报告只验收生产组件公式、权限与数据链，不含任务性能结果。项目 `/home/__compress_data/xushijie/graph_cp_disr_v2_1`，分支 `codex/cp-disr-v2.1-stage-0a`，基准提交 `3669e5a11e59b4b327db714f092b2d9ca14d78d3`；精确测试代码 SHA-256 见 code_hashes.json，最终提交见交付 receipt。

## 最终运行结果

- T01–T25：25/25 PASS。完整 CPU 回归：37 passed、0 failed、0 skipped（其中 T24 与若干增强回归有多个测试函数，另含 runtime gate）。
- CPU FP64 手算/数值补充：7 passed、0 failed、0 skipped。
- GPU FP32 数值/梯度子集：22 passed、0 failed、0 skipped；15 个纯逻辑/CPU 项按选择器未在 GPU 重复执行，属于不适用而非缺失 GPU 验证。
- CPU/GPU 输出及梯度对照：1 passed；比较 260 个张量，最大绝对误差 1.05798244476e-06，最大相对误差 1.56319401867；超容差位置 0。采用逐元素 `abs(error) <= 1e-6 + 1e-5*abs(CPU)`，接近零的参考值会放大相对误差；完整逐张量记录保存在 attempts/008_parity_artifact_fallback/numeric_comparison.json。
- 原始套件首次三组为 9/9/8 passed；增强回归曾发现 2 个错误，原日志与 traceback 保留。修复后受影响 4 项通过，再完整 CPU 回归及适用 GPU 补测全部通过。历史失败没有删除或伪装为首次通过。

## 不变量结论

没有发现真实 Fact Store 污染；nominal overlay 写入拒绝、真实快照 hash 不变。空 prior 的 DP、uP、Delta 严格为零，使用同参数合同分布对照；未以容差掩盖零性质。candidate ID/mask 重算一致，并主动拒绝长度、集合、值变化。Structural Q 只 gather executed action，未选输出直接梯度严格为零；Q/V target 完全 detach。

prior 只在 episode 开始抽样一次，固定 80/20，episode 内 hash 不变；生产 PPO 在四个 epoch 重算 encoder/差分，参数版本变化，禁止网络及 prior 重抽。terminated 不 bootstrap，truncated/buffer 边界使用最后有效状态，不跨 reset。GRU probe 不提交状态，prefix 使用当前参数，环境与 episode 隔离。Full/A_DD/A_Q/A_B 分别验证规定输入、Q 系数和 tanh 开关，未改变其他参数。

## 修复的生产 bug

1. `rl.remap_stored` 在 zip 前未校验 mask 长度，多余 mask 元素被截断后可能被接受。已添加长度一致性拒绝。
2. `torch_rl.RecurrentState` 未检查 snapshot 的 env/episode/prior 身份。已在 commit/probe 前检查环境、决策连续性、episode 与 prior；错误输入在状态写入前拒绝。

这两项仅修复实现权限/数据完整性，不改变冻结 Method、公式、schema、fixture 语义或超参数。全部修复前日志、根因与 diff 位于 attempts/。

## Stage 0C / 1A readiness

Stage 0B 的 schema、ID/effect/redundancy/cache-key 单元前置已通过。但完整 Stage 0C 执行代码与真实输入还不齐：联网 provider transport 仍未绑定；D0/T_A/T_C 各 8 张真实 dev 图像及对象/合同绑定、冻结 3 个 few-shot、请求/原始输出/拒绝/缓存审计尚需完成。API 已由用户指定北京 endpoint `https://dashscope.aliyuncs.com/api/v1`；`qwen3.8-max-0902` 权限及图像输入、非思考、JSON object、temperature=0 等真实行为为 MUST_VERIFY，仅可在后续授权 Stage 0C 验证；不可访问即停止，不自动换模型或 endpoint。

Stage 1A 仍缺真实任务和 split、控制器、相机/标定、冻结感知、Verifier 阈值、独立 evaluator、安全许可、clock/d_ref/timeout/deadline、runtime factory 与 resolved run config、有效缓存和 Stage 0C 验收。train/evaluate 在未绑定时仍 fail closed；本轮只测试拒绝入口，未启动任务动作。

已更新 stage_0b.json；Stage 0A 文件逐字节未变。0 科研训练、0 API 请求、0 机器人/仿真任务动作、0 性能 episode。未推送 GitHub；停止于 Stage 0B。

## T01–T25 状态

| ID | 状态 | 生产组件 |
|---|---|---|
| T01 | PASS | neural.differences;graph.four_views |
| T02 | PASS | neural.AnchoredPrior;Policy |
| T03 | PASS | contracts.nominal_overlay;facts.FactStore |
| T04 | PASS | graph.four_views |
| T05 | PASS | neural.GoalReadout;CandidateReadout |
| T06 | PASS | facts;contracts.precondition_value;nominal_overlay |
| T07 | PASS | contracts.Effects;nominal_overlay |
| T08 | PASS | vlm.validate_relations |
| T09 | PASS | graph.RELATIONS;vlm.validate_relations |
| T10 | PASS | rl.remap_stored;neural.Policy |
| T11 | PASS | torch_rl.q_loss |
| T12 | PASS | torch_rl.q_targets;value_targets |
| T13 | PASS | neural.GraphEncoder;differences;Policy |
| T14 | PASS | rl.Transition;scalar_targets |
| T15 | PASS | neural.Policy;PolicyOutput.select |
| T16 | PASS | vlm.cache_key |
| T17 | PASS | prior.PriorSampler |
| T18 | PASS | torch_rl.PPO.update;recompute_transition |
| T19 | PASS | rl.gamma;interval_reward;cumulative_weights |
| T20 | PASS | adapters.EvaluationInput;skills.candidate_mask;neural.Policy |
| T21 | PASS | neural.AnchoredPrior |
| T22 | PASS | torch_rl.RecurrentState;prefix_hidden |
| T23 | PASS | neural.differences;GraphEncoder |
| T24 | PASS | contracts;neural.Policy |
| T25 | PASS | neural.Policy;torch_rl.ppo_losses |
