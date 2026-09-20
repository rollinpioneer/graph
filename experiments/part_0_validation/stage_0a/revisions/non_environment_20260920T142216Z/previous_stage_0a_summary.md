# CP-DISR v2.1 Stage 0A 完整报告

**最终状态：BLOCKED。Stage 0B 条件尚未具备；Stage 0B 未执行，保持 NOT_STARTED。**

完成时间（UTC）：2026-09-20T13:38:54.111566+00:00。Method：CP-DISR v2.1；计划：EEP-v1.0。

## 1. 授权范围与目录

本轮仅执行 Stage 0A Environment & Manifest Freeze。文档中的后续阶段、旧 prompt 和示例命令只用于理解约束，不扩展本轮授权。

- 服务器项目：`/home/__compress_data/xushijie/graph_cp_disr_v2_1`
- EXPERIMENT_ROOT：`/home/__compress_data/xushijie/graph_cp_disr_v2_1/experiments`
- 分支：`codex/cp-disr-v2.1-stage-0a`
- 取证基准 commit：`7f7a22ef330abcdaefc4ac4e6999dc99c7c345f5`；最终交付 commit 在 delivery receipt 中单独记录。
- origin：`git@github.com:rollinpioneer/graph.git`
- 新仓库采用独立根提交历史，未把过时分支或主仓库文件复制为当前方法实现。旧仓库仅被读取。
- 本轮未推送 GitHub；未切换或修改旧仓库分支。

## 2. 完成了什么

保存全部用户附件与冻结规范；核对源 hash；建立独立仓库、分支和虚拟环境；记录真实 host/GPU/driver/OS；解析推荐依赖；执行以下基础设施检查；定点审查旧项目资源；生成 manifests、绑定表、缺项清单、实现清单、完整报告和状态历史。

Kit 自带的聊天沙箱预检已归档为 supplied_kit 历史，不能当作本服务器执行证据。`inputs/` 和 `experiments/sources/` 原件保持不变。

## 3. 检查结果

| 检查项 | 结果 | 证据（相对阶段目录） |
|---|---|---|
| source_hashes | PASS | `source_inventory.json` |
| target_host_and_new_repository | PASS | `environment_probe.json` |
| dependency_resolution | PASS | `host_logs/resolver.log; ../../../../uv.lock` |
| dependency_install_and_pip_check | BLOCKED | `host_logs/install_attempt_02.log; host_logs/install_attempt_02.exit; pip_check if installation completed` |
| imports | BLOCKED | `software_checks.json` |
| RGCN_forward_backward | BLOCKED | `software_checks.json` |
| checkpoint_roundtrip | BLOCKED | `software_checks.json` |
| CP_DISR_production_entrypoints | BLOCKED | `implementation_inventory.md` |
| D0_T_A_T_C_real_asset_bindings | BLOCKED | `task_manifest.yaml; legacy_resource_review.json` |
| controller_camera_verifier_evaluator_time_safety | BLOCKED | `runtime_manifest.yaml; binding_table.csv` |
| VLM_account_region_endpoint_fewshot | BLOCKED | `../../manifests/vlm_manifest.yaml; environment_probe.json` |

实际主机为 gpu03 / Rocky Linux 10.2 / x86_64；8 张 NVIDIA A100-SXM4-40GB，driver 610.57.04。完整 GPU 清单及探测时显存占用见 environment_probe.json；不承诺资源独占或性能。实际系统 Python 未被覆盖。

软件 profile：**PROPOSED_NOT_VALIDATED**。目标 Python `3.11.13`；完整已解析版本、导入结果和错误见 dependency_table.csv / software_checks.json。独立环境位于 `.venv`，uv 0.8.22 位于 `.bootstrap`。真实 resolver 和安装日志保存在 host_logs；pip check 与 freeze 因安装未完成而未执行，也未生成对应日志，wheel 来源和 uv.lock hash 写入 software_manifest.yaml。

安装 attempt 01 遇到 torch wheel 网络超时，原始失败日志和首次缺包检查保存在 host_logs/attempt_01。attempt 02 保持版本与来源不变，只提高 UV_HTTP_TIMEOUT 至 300 秒、下载并发降为 2；实际结果：attempt 02 达到 900 秒安装时限，退出码 124；没有完成安装。attempt 01 的网络超时退出码为 1。详见 install_attempt_02.log / .exit。安装失败不记作训练失败。软件检查在安装恢复后才可升级 VALIDATED。

本次由于 torch 未安装，RGCN 前后向与 checkpoint round-trip 的计算部分均未执行；不能写为数值测试通过或失败。探针程序设计使用合成张量，仅检查库前后向；若 round-trip 成功，文件是未训练库模型的 state_dict，不是 CP-DISR policy checkpoint。两者不证明方法语义通过，不替代 T01–T25。

## 4. 实际运行与结果边界

- 科研训练完成 / 失败 / 复用：**0 / 0 / 0**；无 training run_id。
- 真实 skill transitions：**0**；PPO updates / optimizer steps：**0 / 0**。
- 性能评价 episodes：**0**；控制器/机器人动作：**0**；VLM 请求：**0**。
- 执行的是一次 host 盘点和一次软件基础设施检查批次，详细检查计数见 check_results.json。
- 没有成功率、学习曲线、机制收益或泛化结果；没有用历史训练结果充当本版结果。

## 5. 仍未绑定的 MUST_BIND 项

- **软件安装/验证**：推荐 profile 虽已解析，但安装/导入/数值检查未全部通过；详见 host_logs 与 software_checks.json。
- **运行平台**：simulator_or_robot、environment_version、task_assets。
- **技能与事实**：controller_manifest、skill_contracts、verifier_thresholds；真实前置/ADD/DEL、三态证据与标定。
- **视觉**：camera、calibration_manifest、perception_checkpoint。
- **时间与安全**：skill_timeouts、task_deadlines、actual_interaction_time_unit、reference_skill_seconds_by_task、safety_authorization。
- **独立评价**：task_evaluator_version；真实终端成功和结束规则。
- **实现入口**：runtime.entrypoints；train_cli/eval_cli/tests_cli/cache_cli/report_cli；模型源码 hash 与参数量。
- **任务与 split**：D0/T_A/T_C 的 asset_ref、controller_refs、goal_atoms、skill_contract_refs、camera_verifier_refs、initial_state_generator、deadline_seconds、reference_skill_seconds、split_ref；64/50/50 实例池。
- **VLM**：region、base_http_api_url、api_account_authorized、fewshot_manifest；seed 参数实际支持能力另待核实。
- **资源配置**：实际 num_envs、candidate view 分块与运行时钟来源；GPU 清单不是专属训练资源预约。

可选任务 T_B/T_D/T_E 同样未绑定，但它们不是当前 0A 的独立阻塞原因。主任务 D0/T_A/T_C 缺项本身已足以阻塞。完整逐文件逐字段占位项共 **141** 行（包含镜像字段和 MUST_VERIFY 类项，不是 141 个独立问题），见 `unbound_must_bind.csv` / `.json`；`binding_table.csv` 同时列出已绑定字段。

API 检查只记录当前非 login SSH 进程中常用环境变量是否设置，未读取或打印密钥值；未发现不等于账户在其他凭据存储中绝对不存在。真实 region、endpoint、账户权限和 3 个开发 few-shot 均未确认。

## 6. 任务与实现判定

D0/T_A/T_C 尚未形成可验证的资产→技能合同→相机/Verifier→独立评价器映射，真实 case_definitions 为空。64/50/50 只是计划数，不伪造初始图像、对象绑定或合法动作序列。CUPID 和 LIBERO 路径真实存在，LIBERO commit 为 8f1084e3132a39270c3a13ebe37270a43ece2a01，发现 BDDL 任务文件；尚无本次模板与固定技能、视觉权限、Verifier、安全合同的一致映射。详见 implementation_inventory.md、legacy_resource_review.json 和 platform_discovery.json。

## 7. 相对冻结配置的变更

冻结 Method、关系 schema、prompt 原件及七方法定义未修改；无性能调参。目录、Git 分支和软件环境由真实 host 绑定。OS 从推荐的 Ubuntu 兼容候选记录为实际 Rocky Linux 10.2；这属于允许的环境适配，软件通过与否以探针为准。torchvision 为可选依赖，尚无绑定视觉前端需要它，故未安装。软件推荐版本未暗中升级或替换。

## 8. 状态与下一步

PASS 要求核心代码和真实运行资产完成绑定；软件检查通过不能抵消这些核心缺项，因此选择 **BLOCKED**，不能标 PASS 或 PASS_WITH_NOTES。

**目前不具备完整执行 Stage 0B 的条件。** 后续应先补做 Stage 0A：明确真实平台及 D0/T_A/T_C 资产，冻结控制器/视觉/Verifier/评价器/时间/安全合同，完成生产模块和 CLI，生成真实 split，并绑定 API 账户地区/端点/few-shot，再重新判定 0A。Stage 0B 即使可用纯程序 fixture，当前生产实现缺失也不能视为可验收。

本轮报告写入后停止。0B 及其他未授权阶段保持 NOT_STARTED；不会自动进入 0B，不调用模型或启动训练。
