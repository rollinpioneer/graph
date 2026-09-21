# Stage 0C-R1 平台选择审计

## 结论

本轮唯一推荐平台为 **LIBERO + robosuite + MuJoCo**，但当前只能记为 `CANDIDATE_BLOCKED`，没有任何平台达到 CP-DISR 正式绑定的最低条件。Stage 0C 继续 `BLOCKED`，不生成正式 24 场景或 few-shot 图像。

## 审计对象

| 项目 | 绝对路径 | git commit | dirty | 结果 |
|---|---|---|---|---|
| LIBERO | `/home/__compress_data/xushijie/LIBERO` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | dirty（多处源码/配置修改） | 唯一推荐候选 |
| Metaworld | `/home/__compress_data/xushijie/Metaworld` | `a98086ababc81560772e27e7f63fe5d120c4cc50` | 未在本轮作为目标平台绑定 | 不匹配容器/合同结构 |
| CP-DISR仓库 | `/home/__compress_data/xushijie/graph_cp_disr_v2_1` | `5baf90a4cb45ca2d8b0bd7f51ff2c83b89813cd6` | clean at audit start | 冻结协议和待绑定模板 |

## 已确认能力

LIBERO 仓库包含本地 BDDL、对象/场景 MuJoCo XML 和 MIT LICENSE；固定 `agentview` 相机接口可产生 RGB 和 depth。使用 `/home/__compress_data/xushijie/.conda/envs/lerobotpi0/bin/python`、本地 LIBERO 资产和固定 BDDL 做了只读 `reset()` 探针：KITCHEN_SCENE3 产生 `96x96x3` RGB 与 `96x96x1` depth。该探针不是正式场景采集，没有写入正式输入目录，也没有 VLM/API 请求。LIBERO 资产缓存路径 `/home/__compress_data/xushijie/.cache/libero/assets` 已存在；未下载网络图片或生成图片。

## 未满足的最低条件

- LIBERO/robosuite 现有接口是低层 `OSC_POSE` 7-DoF action，不是 CP-DISR `OPEN`, `PICK`, `PLACE`, `PLACE_BUFFER` 或 `MOVE` 高层技能。
- 没有已绑定的高层 controller、技能后置条件 verifier、独立 success verifier、TaskEvaluator、安全许可、skill timeout/deadline 或参考技能秒数。
- 现有 D0/T_A/T_C 仍是 `STRUCTURE_TEMPLATE_NOT_RUNTIME_ASSET`，其 `controller_ref`, `verifier_ref`, `evaluator_ref`, asset 和 split 均为 MUST_BIND。
- LIBERO 工作树 dirty；不能把未审计修改后的平台状态作为冻结实验平台版本。
- LIBERO LICENSE 证明代码许可，不自动证明所有 bundled asset 的 VLM 上传权限；每个正式 asset 仍需逐项权限证据。

## 精确实施清单

1. 在最终选定且干净的 LIBERO commit 上冻结版本和资产清单，或提供另一个已具备高层技能的目标平台。
2. 实现并审计 CP-DISR `OPEN/PICK/PLACE/PLACE_BUFFER`（或真实 `MOVE`）controller adapter；每个 Action ID 写入 controller manifest。
3. 实现 postcondition verifier、independent success verifier、TaskEvaluator、安全授权、clock unit、timeout/deadline 和 reference skill seconds。
4. 为 D0/T_A/T_C 写真实 task BDDL/scene reset 配置，记录 asset hashes、对象位姿、buffer 区域和 camera config。
5. 逐项确认 bundled asset 的使用与上传许可，建立独立 dev_fewshot pool 和 test split 隔离。
6. 只在上述绑定完成后运行无 API 的 27 输入预检；随后用户单独处理凭证和模型权限。

本轮新增候选 resolved 文件：`configs/tasks/resolved/D0.yaml`, `T_A.yaml`, `T_C.yaml`。它们明确标记 `BLOCKED_PLATFORM_SKILL_BINDING`，不被当作 READY 任务，也不能生成图像。
