# Stage 0A 环境复用与前置复核报告

**完整 Stage 0A：BLOCKED。软件基础设施：PASS_WITH_NOTES。正式 Stage 0B：NOT_STARTED。**

项目：`/home/__compress_data/xushijie/graph_cp_disr_v2_1`；分支：`codex/cp-disr-v2.1-stage-0a`；基准提交：`1a00beb8f9a06753325fb2bdd7c191c3ec731a8e`。最终提交见交付 receipt。

## 本轮完成

- 检查已有 conda/venv 环境，使用已有 Python 3.10.19 和 PyTorch 2.7.1+cu126/CUDA 12.6 文件，在 `.venv-stage0a` 建立隔离环境；仅补齐缺失依赖，未修改旧项目或源环境。
- 首选 Python 3.11.13 改为兼容配置 3.10.19，属于手册允许的软件兼容调整；冻结 Method、Torch/PyG 核心版本和方法参数未改。
- 依赖解析、pip check、import、生产核心模块 import、最小 RGCN 前/反向与未训练 state_dict round-trip 结果见 `software_checks.json` / `infrastructure_command_checks.json`。
- API 文件仅 stat 检查存在与非空，未读取内容、未上传、未写入 Git、未发送 API 请求。文件存在不能证明账户授权、region、endpoint 或模型权限。
- API 地域按用户声明绑定为华北2（北京，中国大陆）`cn-beijing`，endpoint 为 `https://dashscope.aliyuncs.com/api/v1`。`qwen3.8-max-0902` 权限记为 `MUST_VERIFY`，按用户要求在后续获准的 Stage 0C 验证；不可访问即停止，不自动更换模型或 endpoint。
- 全部旧报告、状态及 manifest 保存；本轮独立 revision，已更新 active stage_status 和软件/入口 manifest。

## 真实结果与数量

运行 1 次 Stage 0A 软件 probe；已记录 2 次安装准备中止（遗留队列下载取消、初次复用安装超时）、1 次离线解析失败及后续成功修复，详见 bootstrap_attempts.json。RGCN 前/反向与 round-trip 的实际结果详见 JSON。0 个训练 transition，0 次 PPO update，0 个性能 episode，0 次机器人动作，0 次 VLM 请求。未运行正式 T01–T25。以前的 13 项纯逻辑通过记录仅作为历史证据，不冒充本轮新运行。

## 剩余 MUST_BIND

逐字段清单：`unbound_must_bind.csv`。主要为 D0/T_A/T_C 真实对象与场景、控制器、相机及标定、冻结感知 checkpoint、Verifier 阈值、独立 evaluator、安全授权、skill timeout/deadline/clock/d_ref、64/50/50 初始化池，以及模型访问权限和 few-shot。地域/endpoint 已按用户声明绑定，模型权限明确延后到 Stage 0C 实测。

之前发现的 LIBERO/历史 graph 资源只是可适配来源。本轮未取得能把结构模板认定为真实可执行绑定的新证据，不虚构 resolved task、时长或实验结果。

## 是否具备 Stage 0B 条件

已满足手册所列 core_code + 可运行软件 + schema/fixtures 的独立单元验收前置。仍须用户明确授权才执行正式 Stage 0B；数值方法测试尚未通过验收。完整 Stage 0A 仍因真实 runtime 绑定缺失为 BLOCKED，不能写 PASS。模型权限按用户指令延后至 Stage 0C 验证，不声称已通过。

## 复用限制与后续

复用的 Torch/CUDA 文件仍依赖原环境路径；源环境被删除或更新后必须重新验收。具体链接及来源见 `reused_package_links.json`，实际版本快照见 `installed_requirements.txt`。Stage 0C 还需 API 配置/provider、真实 dev 输入及缓存审核；Stage 1A 还需真实 runtime 全部绑定和前置阶段验收。

本轮停止于 Stage 0A，不推送 GitHub，不自动进入 Stage 0B。
