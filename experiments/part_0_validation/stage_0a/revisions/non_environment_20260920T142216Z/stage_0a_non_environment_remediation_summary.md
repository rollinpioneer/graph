# Stage 0A 非软件环境、非 API 凭据补全报告

**状态：BLOCKED；本轮代码补全与纯逻辑/静态验收已完成，未执行正式 Stage 0B。**

- 项目 `/home/__compress_data/xushijie/graph_cp_disr_v2_1`；分支 `codex/cp-disr-v2.1-stage-0a`。
- 基准 commit `9e90dfb42dfae8c258ea42f73f9a3947e909b403`；新 commit 见交付 receipt。
- 新 revision `/home/__compress_data/xushijie/graph_cp_disr_v2_1/experiments/part_0_validation/stage_0a/revisions/non_environment_20260920T142216Z`。原 Stage 0A summary 未覆盖，旧 manifests/status 已归档。
- 环境下载/安装、凭据读取、API 请求、仿真/机器人动作、科研训练、性能评价均为 0。

## 1. 真正解决的原 BLOCKED 项

新增正式 `src/cp_disr` package：不可变 typed contracts / grounding / conflict validation；三值 Fact Store 与只读 nominal overlay；Action–Proposition 图、固定节点/goal_refs；本地 relation schema/ID/effect/redundancy/dedup/cache-key；固定 episode 80/20 prior；标准共享 RGCN、DK/DH/DP、零 prior 复用、零锚定有界 Actor；七配置；独立 V/Q、executed-only detached Q targets；SMDP GAE、候选身份/mask/prior 保存、GRU prefix 与当前参数重算 PPO；外部 Protocol、真实 collector 和 fail-closed 入口。

任务 D0/T_A/T_C 结构模板、OPEN/PICK/PLACE/PLACE_BUFFER/MOVE 合同模板、5 个固定 synthetic fixtures、T01–T25 测试代码及 8 个 CLI 已落盘。MOVE 未被声明为真实 runtime 技能。

## 2. 验证结果

- `compileall src tests`：PASS。
- 纯逻辑 pytest：PASS；详见 pure_logic_tests.stdout.log / JUnit。
- YAML/JSON、schema、import-free dependency graph、manifest/entrypoint references、方法源 hash、唯一 prior 协议、未绑定运行拒绝检查：见 static_validation.json，全部通过：True。
- 数值/梯度测试：**NOT_RUN_ENVIRONMENT_BLOCKED**，只编写、编译、静态检查；不能声称 torch/PyG 运行正确性已通过。
- 所有 synthetic fixture 都有 `synthetic_unit_fixture=true`、`paper_performance_eligible=false`；生产 collector/evaluation 拒绝 synthetic snapshot。

## 3. 仅完成接口/模板，尚未真实绑定

EnvironmentAdapter、SkillExecutor、ObservationProvider、PerceptionAdapter、FactVerifier、TaskEvaluator、SafetyManager、DurationProvider、SnapshotBuilder 已有明确接口，但没有编造 concrete 实现。
真实 D0/T_A/T_C 资产、OPEN/PICK/PLACE 控制器、camera/calibration/perception checkpoint、Verifier 阈值、独立 success/termination、时限/clock/d_ref/safety、64/50/50 初始化实例仍未绑定；只有 schema 和模板。API 账户及环境继续由用户处理。逐字段 A/B/C/D 分类见 binding_table.csv 和 unbound_must_bind.csv。

## 4. 服务器已有可复用来源

只读盘点 5 个源项目、65 个代码候选、3229 个资产候选。LIBERO ControlEnv/相机参数/BDDL/XML/独立 check_success 可适配；P1 历史控制器和 P2C rollout/checkpoint 可参考。均有路径/commit/dirty/hash/签名与证据。LIBERO 的默认双视角、无 depth、低层控制步/旧 done 与本版接口有差异；真值 predicates 不能冒充视觉事实。旧 reward shaping/PPO 不能直接继承。没有生成缺乏证据的 resolved task。

## 5. 环境修复后 Stage 0B 还缺什么

生产核心、fixtures、25 项测试和命令已存在。用户完成目标环境后，还需先重新验收 0A 的依赖导入、RGCN 前后向与 checkpoint round-trip；随后在用户明确授权下执行正式 0B CPU T01–T25，GPU 按可用性补测。数值测试运行前不存在已证明通过的承诺，运行发现 bug 时仍需修复。按手册 0B 的前置定义，真实机器人/API 未绑定不妨碍独立 synthetic 单元验收，但不能自动把完整 0A 升为 PASS。

建议未来命令（本轮未执行）：`PYTHONPATH=src .venv/bin/python -m cp_disr run-unit-tests --scope full`。正式 0B 仍需按手册收集 L_TEST/JUnit 和状态报告；CLI 不自动修改阶段状态。

## 6. Stage 0C 与 Stage 1A 仍缺什么

0C：用户 API 配置与权限、实际 snapshot 支持验证、本轮刻意未实现的联网 provider transport、真实 dev 图像/ID/合同、冻结三个 few-shot、完整请求/缓存审计和 24 场景检查；还需 0B 验收。
1A：0B/0C 的真实前置验收，D0 场景与 split、控制器/视觉/Verifier/独立评价/安全/时间绑定、runtime factory 与 resolved run config、真实缓存、资源配置和训练启动前审核；当前 train/evaluate 明确拒绝未绑定运行。

## 7. 是否达到“完成 PyTorch 环境后可重新验收 0A 并执行 0B”

**就核心代码入口、fixtures、测试及静态前置而言：已具备。** 仍必须重新验收软件并由用户授权正式 0B；不能承诺首次数值测试必然全部通过。**若指完整 Stage 0A PASS 后无条件进入 0B：尚未达到**，真实平台/API 缺项仍使完整 0A 为 BLOCKED。上述两种 readiness 已在 stage_0a.json 分开记录。

本轮不安装、不碰凭据、不调用 VLM、不进入正式 Stage 0B；提交当前分支，不推送 GitHub，然后停止。
