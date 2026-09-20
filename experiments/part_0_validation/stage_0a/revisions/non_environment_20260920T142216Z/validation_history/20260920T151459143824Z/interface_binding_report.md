# 接口绑定报告

**接口完成 ≠ 真实运行资源绑定。**

- EnvironmentAdapter、SkillExecutor、ObservationProvider、PerceptionAdapter、FactVerifier、TaskEvaluator、SafetyManager、DurationProvider、SnapshotBuilder 与 RuntimeFactory 的输入/输出 dataclass、Protocol 已定义于 `src/cp_disr/adapters.py`。
- 真实测量、执行结束、三态事实、task_success、奖励事件时间各有独立 schema。TaskEvaluator 输入没有 prior/DK/DP；名义 overlay 没有真实时钟、reward 或执行写接口。
- 合同与 task 模板已通过 schema/逻辑校验；timeout、controller、verifier 无证据时仍 MUST_BIND。MOVE 未绑定为真实能力，模板允许 PICK+PLACE_BUFFER 等价结构。
- 环境/API 为 user_environment_owned，不由本次修复、读取或探测。
- 真实 adapter、校准、视觉 checkpoint、时间/安全许可与任务池仍 external_runtime_missing（在本轮明确搜索范围内未找到可认证绑定，不等于证明整台服务器不存在）。
- discovered_and_bound 仅指已有来源路径/commit/hash 的资料登记；没有据文件名宣称 controller 或任务已绑定。
