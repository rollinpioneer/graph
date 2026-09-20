执行 Stage 0C — VLM Cache Sanity Check。

本次允许先完成Stage 0C缺失的生产实现和输入绑定；只有全部正式条件满足后，才执行真实VLM请求和24个task-scene的cache sanity检查。

严格依据：
1. CP_DISR_M1_Research_Specification_v2.1.md
2. CP_DISR_M1_Implementation_Interfaces_v2.1
3. CP_DISR_v2.1_Minimal_Experimental_Execution_Plan.md
4. CP_DISR_v2.1_Experiment_Manifest_v0.yaml
5. Experimental Agent Kit中的Stage 0C执行卡
6. 当前仓库冻结schema、prompt、task templates和contracts

当前仓库：
- path: /home/__compress_data/xushijie/graph_cp_disr_v2_1
- branch: codex/cp-disr-v2.1-stage-0a
- starting commit: 502d33cbca9a83eccd1c32840a0d0b50189c3ad1

当前状态：
- Stage 0A: BLOCKED
- Stage 0B: PASS
- Stage 0C: NOT_STARTED
- Python/Torch/PyG/GPU环境已验收
- 北京地域和指定endpoint已绑定
- API key只允许由SDK通过受保护环境读取
- 模型权限和真实API行为仍为MUST_VERIFY

不得修改Stage 0A和Stage 0B的既有状态证据。

==================================================
一、本轮唯一目标
==================================================

完成CP-DISR v2.1的正式VLM关系生产链并执行小规模sanity check，回答：

1. 冻结模型和endpoint能否真实调用；
2. m1_soft_relations_v2输出能否稳定解析；
3. relation中的ID、端点类型和effect_fact_ref是否合法；
4. 输出是否大量重复Skill Contract已有信息；
5. 是否几乎总为空prior；
6. 是否存在少量合法、非合同重复、具有场景依据的soft relation；
7. raw response、parsed、rejected和处理日志能否完整缓存和追溯。

本Stage不评价RL性能，不证明VLM关系一定有益。

==================================================
二、禁止事项
==================================================

1. 不执行RL训练、PPO更新或性能评价。
2. 不执行机器人或仿真动作。
3. 不自动进入Stage 1A。
4. 不修改CP-DISR v2.1冻结Method。
5. 不添加SOFT_ORDER。
6. 不恢复m1_soft_relations_v1。
7. 不增加static prior、VLM action ranking或新关系类型。
8. 不训练、微调或蒸馏VLM。
9. 不进行网页搜索、工具调用或在线重规划。
10. 不自动更换模型snapshot、region或endpoint。
11. 不反复调用VLM直到得到更“好看”的relation。
12. 不人工修改原始响应，使其通过schema或语义检查。
13. 不使用测试结果选择prompt或few-shot。
14. 不读取、打印、复制、hash或提交API key。
15. 不推送GitHub。

==================================================
三、先完成Stage 0C生产链
==================================================

检查并补齐以下生产组件。所有测试和正式调用必须使用同一套代码。

A. Provider实现

实现冻结provider adapter，至少支持：

- 指定模型snapshot；
- 北京region；
- 已绑定endpoint；
- multimodal image input；
- JSON object输出；
- temperature=0或服务端等价greedy设置；
- max_output_tokens=2048；
- timeout；
- request ID记录；
- transport/error分类；
- 最多一次格式/传输重试；
- 禁止内容语义重试。

API key：
- 只能由SDK从环境变量或权限受控文件获得；
- 日志只能记录key_present=true/false；
- 不得记录值、部分值、长度、hash或文件内容。

如provider SDK或服务实际参数与配置不一致：
- 记录实际接口证据；
- 不静默猜测；
- 不自动换模型；
- Stage 0C记BLOCKED或NEEDS_RERUN。

B. 冻结relation协议

只能接受：

1. SOFT_SUPPORTS
   - Action -> Action

2. SOFT_RELEVANT_TO_GOAL
   - Action -> signed goal Proposition

每条relation必须具有：

- relation_id
- type
- source_ref
- target_ref
- effect_fact_ref

effect_fact_ref必须：

- 是已注册Proposition；
- 属于source action的已注册ADD或DEL nominal effect；
- 在合同图中存在对应source→fact效果邻接；
- 只作校验和审计元数据；
- 不产生新的合同效果或事实真值。

明确拒绝：

- SOFT_ORDER；
- 新Action；
- 新Fact；
- 新Goal；
- 真实未来状态；
- reward；
- action答案；
- confidence伪装成事实概率；
- 不存在或类型错误的引用。

C. 本地处理链

必须实现并实际调用：

raw response
-> JSON parser
-> m1_soft_relations_v2 schema validation
-> ID validation
-> endpoint type validation
-> effect_fact_ref validation
-> fixed contract redundancy check
-> exact deduplication
-> explicit conflict isolation
-> final E_prior
-> immutable cache write

不得由人工编辑步骤插入中间链路。

D. Cache与审计

每个task-scene必须生成独立cache key。key至少覆盖：

- task specification hash；
- 原始图像内容hash；
- 图像预处理hash；
- object binding；
- allowed IDs；
- Skill Contract版本；
- predicate registry版本；
- model snapshot；
- region和endpoint版本；
- prompt hash；
- few-shot hash；
- relation schema hash；
- temperature；
- max tokens；
- max relations；
- provider SDK版本。

每个cache目录至少保存：

manifest.json
prompt.txt
input_refs.json
request_payload_redacted.json
raw_response.json
parsed_relations.json
rejected_relations.json
processing_log.json

request payload必须删除secret，但保留其余可复核内容。

==================================================
四、绑定正式dev task-scene输入
==================================================

正式矩阵固定为：

- D0: 8个dev初始场景
- T_A: 8个dev初始场景
- T_C: 8个dev初始场景

总计：

24个task-scene。

每个场景必须具有：

- scene_id；
- task_id；
- split=dev；
- 原始真实RGB图像；
- 图像来源和使用权限；
- image SHA-256；
- preprocessing version；
- typed object table；
- object binding状态；
- signed goals；
- allowed grounded Action IDs；
- allowed Proposition IDs；
- Skill Contract摘要；
-初始TRUE/FALSE/UNKNOWN事实；
- task template和真实asset之间的绑定记录。

不得用以下内容代替正式输入：

- 纯文本场景描述；
- 人工生成的空白图；
- toy graph截图；
- 无法确认来源的网络图片；
- 同一张图片复制8次；
- 测试集图片；
- 为了让VLM更容易输出关系而人工摆拍后未记录的图片。

先以只读方式搜索服务器现有项目、数据目录和历史任务资产。

若找到候选图像：
- 记录绝对路径、来源仓库/数据集、版本、split和许可；
- 检查是否与D0/T_A/T_C任务语义匹配；
- 不因文件名相似就声明绑定完成。

若不足24张有效真实dev图像：
- 完成provider、cache、schema和manifest实现；
- 输出缺失场景的逐项采集清单；
- Stage 0C保持BLOCKED；
- 不使用synthetic images补齐正式数量。

==================================================
五、冻结三个few-shot
==================================================

优先读取v2.1接口包中已经定义的prompt和few-shot材料，不得从v2.0自行恢复旧schema。

三个few-shot必须来自独立开发材料，不得来自本次24个正式sanity场景，也不得来自test split。

它们必须覆盖：

1. 合同已完整表达、应输出空relations的例子；
2. 有场景证据且具有合法effect_fact_ref的SOFT_SUPPORTS例子；
3. 无充分证据或独立顺序、应保持空输出，或者接口包中已冻结的SOFT_RELEVANT_TO_GOAL合法例子。

以v2.1接口包为最高优先级，不得为了“覆盖关系类型”改写已经冻结的few-shot。

每个few-shot保存：

- 输入图像和hash；
- task；
- object/action/fact IDs；
-合同摘要；
- 预期JSON；
- 选择理由；
- provenance；
- prompt version。

如现有材料不能形成3个真实、可授权且独立的few-shot：
- 不自行生成虚假图片；
- 输出few_shot_missing_manifest.csv；
- Stage 0C保持BLOCKED。

==================================================
六、API权限与预检
==================================================

只有以下条件全部成立时，才允许发出请求：

- provider生产实现完成；
- model、region和endpoint与manifest完全一致；
- API secret可由SDK安全获得；
- 三个few-shot已冻结；
- 24个正式dev场景已绑定；
- request payload本地schema预检通过；
- cache目录可写；
- secret redaction测试通过。

先执行一个正式矩阵中的预检请求，不另造第25个场景。

预检成功后检查：

- model snapshot是否确实可用；
- 图像是否被服务接受；
- JSON object模式是否按预期工作；
- request ID是否可记录；
- raw response是否可保存；
- 本地parser是否可处理。

若权限拒绝、模型不可用、endpoint不兼容或图像请求失败：
- 保存去敏错误响应；
- 不自动换模型或endpoint；
- 不连续重试；
- Stage 0C写BLOCKED；
- 给出需要用户处理的精确错误码和字段。

预检场景成功后，它计入24个场景，不重复调用。

==================================================
七、正式调用协议
==================================================

对24个task-scene各执行一次冻结请求。

只有以下情况允许最多一次重试：

- transport failure；
- timeout；
- 明确的JSON语法/格式失败。

第二次请求必须：

- 使用相同模型；
- 相同图像；
- 相同prompt/few-shot；
- 相同schema；
- 相同解码配置；
- 使用固定格式错误反馈；
- 保存第一次和第二次完整raw response。

以下情况禁止重试：

- relations为空；
- relation被判合同重复；
- relation语义看起来不理想；
- effect_fact_ref无效；
- relation数量少；
- 输出不利于CP-DISR；
- 人工希望得到另一种relation。

不得使用RL结果或后续性能选择接受哪次响应。

==================================================
八、自动统计
==================================================

对全部24个场景统计：

- request_success_rate；
- first_attempt_success_rate；
- retry_count；
- JSON_valid_rate；
- schema_valid_rate；
- ID_valid_rate；
- endpoint_type_valid_rate；
- effect_fact_ref_valid_rate；
- contract_redundancy_rate；
- exact_duplicate_rate；
- rejected_relation_count；
- final_empty_prior_rate；
- mean_final_relation_count；
- median_final_relation_count；
- max_final_relation_count；
- SOFT_SUPPORTS count；
- SOFT_RELEVANT_TO_GOAL count；
- raw relation count；
- final accepted relation count；
- error type distribution；
- latency distribution；
- token usage（若provider返回）；
- task-wise统计；
- scene-wise统计。

所有分母必须明确。

例如：
- ID_valid_rate的分母是parsed relation数；
- final_empty_prior_rate的分母是成功处理的scene数；
- API failure不能被计作有效EMPTY_PRIOR。

==================================================
九、人工快速语义审阅
==================================================

对24个场景全部进行轻量人工审阅，但不得修改缓存。

每个最终relation标注：

- clearly_scene_supported；
- plausible_but_uncertain；
- obvious_semantic_issue；
- contract_redundant；
- unverifiable_from_allowed_input。

人工审阅只用于报告质量，不决定是否重新请求VLM。

至少回答：

1. 输出是否几乎全部非法；
2. 是否几乎永远EMPTY_PRIOR；
3. 是否存在合法且非合同重复关系；
4. effect_fact_ref是否真的对应source action注册效果；
5. SOFT_RELEVANT_TO_GOAL是否表达“名义后果可能影响目标”，而不是静态动作相关性；
6. 是否出现任何SOFT_ORDER或直接动作命令；
7. 是否存在明显由图像无法支持的语义关系。

人工审阅者、时间和规则版本必须记录。

==================================================
十、PASS / BLOCKED / NEEDS_RERUN
==================================================

使用Stage 0C执行卡中已经冻结的精确标准；不得为了通过而临时修改阈值。

若执行卡没有数值阈值，则按以下原则判定：

PASS：
- 24个正式dev场景全部完成处理；
- provider和cache审计链完整；
- 输出并非几乎全部格式或引用非法；
- 输出并非几乎永远EMPTY_PRIOR；
- 至少存在少量合法、非合同重复的soft relation；
- 没有schema v1、SOFT_ORDER或新ID越权；
- 所有原始和处理结果可追溯。

PASS_WITH_NOTES：
- 生产链完整，24场景完成；
- 存在部分空prior、部分无效或部分语义问题；
- 但仍有可用非冗余relation，且问题已如实统计。

NEEDS_RERUN：
- 发现可修复的provider、parser、cache或validator实现bug；
- 修复后尚未完成相同冻结输入上的完整重跑。

BLOCKED：
- 模型权限或endpoint不可用；
- API secret不可安全获得；
- 少于24个正式真实dev场景；
- 三个冻结few-shot不完整；
- provider生产实现缺失；
- 无法安全保存完整审计证据。

空prior本身不构成失败。
个别语义错误也不自动构成失败。
不得要求所有场景都有relation。

==================================================
十一、输出目录和文件
==================================================

使用：

experiments/part_0_validation/stage_0c/

至少生成：

stage_0c_summary.md
stage_0c_metrics.json
stage_0c_metrics.csv
scene_results.csv
relation_audit.csv
manual_semantic_review.csv
provider_validation.md
provider_environment.json
request_audit.md
cache_index.csv
task_scene_manifest.json
few_shot_manifest.json
model_access_report.md
errors/
logs/
attempts/

正式cache写入：

experiments/vlm_cache/dev/<cache_key>/

更新：

experiments/stage_status/stage_0c.json

不得覆盖Stage 0A和Stage 0B报告。

==================================================
十二、Git与停止规则
==================================================

完成后：

1. 提交provider、cache、binding、报告和状态文件；
2. 输出最终commit hash；
3. 确认工作区干净；
4. 不推送GitHub；
5. 不修改旧仓库；
6. 不执行Stage 1A；
7. 关闭SSH会话并停止。

即使Stage 0C PASS，也只能报告Stage 1A readiness，不能自动开始训练。

==================================================
十三、最终报告必须回答
==================================================

1. Stage 0C最终状态是什么？
2. 24个场景是否全部是真实、可追溯的dev输入？
3. 三个few-shot是否已冻结并与正式矩阵独立？
4. 模型snapshot是否获得真实权限验证？
5. 实际使用的region、endpoint、SDK和响应模式是什么？
6. 是否发生任何自动模型或endpoint替换？
7. 共发出多少次请求、多少次重试？
8. JSON/schema/ID/effect_fact_ref合法率分别是多少？
9. 最终空prior率是多少？
10. 最终合法relation总数和平均数是多少？
11. 两类relation分别有多少？
12. 合同冗余和明显语义错误分别有多少？
13. 是否出现SOFT_ORDER、新技能、新事实或直接动作答案？
14. raw/parsed/rejected/cache是否全部可追溯？
15. 本轮修复了哪些生产代码问题？
16. 是否具备进入Stage 1A的VLM/cache条件？
17. Stage 1A仍缺哪些真实runtime绑定？

完成后立即停止。