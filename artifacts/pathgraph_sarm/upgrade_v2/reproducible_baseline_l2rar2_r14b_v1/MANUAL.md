# L2RA-R2 / R14-B：新可复现物理基线 A → B → C
## Agent 操作手册 V1.0

> **适用对象**：第一次接手本阶段、只具备一般 Git、Python、MuJoCo 和实验工程能力的 Agent。  
> **执行原则**：先完整阅读本手册，再执行任何命令。不得将“准备授权”理解为“已经获准执行”。  
> **当前状态**：所有新物理授权均为 `0`。在用户分别签署 A、B、C 三份单实例授权之前，只允许完成静态实现、纯测试、协议锁、申请材料和结果验证工具。

---

# 0. 一页摘要

旧缓存的精确重建已经正式关闭：

```text
decision:
OLD_CACHE_EXACT_RECONSTRUCTION_NOT_RECOVERABLE

route:
B_NEW_REPRODUCIBLE_BASELINE_REQUIRED
```

原因不是旧缓存“完全没有价值”，而是旧缓存生成时的完整源码树、隐藏物理状态和精确运行环境无法恢复；继续用 `1e-12 m` 精确门追逐旧缓存没有可审计依据。

本轮不再以旧缓存作为通过门，而是建立新的三步链：

```text
A：ordinary baseline A
   当前冻结代码与环境下，生成一条完整的新普通基线
                         ↓ 人工审查
B：ordinary repeat B
   新进程、同代码、同环境、同seed，独立重复
                         ↓
   比较 A == B
                         ↓ 人工审查
C：instrumented replay C
   相同物理代码，仅增加预先审查的只读逐步记录
                         ↓
   比较 B == C
```

只有：

\[
A=B
\quad\text{且}\quad
B=C
\]

在冻结的全部主门上通过，才生成：

```text
R14B_REPRODUCIBLE_BASELINE_CHAIN_PASS
```

该证书只表示**当前物理与观测生成链可重复、仪表不改变轨迹**。它不表示：

- K4/K5/K6 的loss已确认；
- R16强制掉落已执行；
- 某个在线候选通过；
- confirmation完成；
- 可以进入L3。

---

# 1. 固定入口与科学边界

## 1.1 Git入口

```text
repository:
rollinpioneer/graph

forensic source branch:
maintenance/l2rar2-r14-ordinary002-forensics-v1

forensic source commit:
ab024c71d315100f0e375a844856807114bedaab

formal main:
234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2

new implementation branch:
research/l2rar2-r14b-reproducible-baseline-v1
```

R14-B从 `ab024c71...` 新建分支。不得覆盖：

```text
research/l2rar2-controlled-forced-drop-v1
maintenance/l2rar2-r14-ordinary002-forensics-v1
main
```

## 1.2 冻结科学状态

整个R14-B必须保持：

```text
scientific_status      = L2RAR2_PARTIAL_KEEP_G1
retained_graph         = G1_predicate_bound
selected_candidate_id  = null
confirmation_run       = false
l3_entry_allowed       = false
```

## 1.3 历史事实

必须保留：

```text
R11 physical executions = 40 / 8
R11 status              = BUDGET_EXCEEDED_BLOCKED

R14 ordinary_002        = 1 instance consumed
ordinary_002 result     = STOP_AFTER_EXECUTION_1
old-cache exact route   = closed negative result

R14 instrumented        = 0
R16 calibration         = 0
R16 development         = 0
```

不得删除或重置这些账目。

---

# 2. R14-B物理预算与授权结构

## 2.1 三次物理实例必须分别授权

| 阶段 | Stage ID | 请求实例 | 当前授权 | 是否允许自动进入下一阶段 |
|---|---|---:|---:|---:|
| A | `R14B_ORDINARY_BASELINE_A` | 1 | 0 | 否 |
| B | `R14B_ORDINARY_REPEAT_B` | 1 | 0 | 否 |
| C | `R14B_INSTRUMENTED_C` | 1 | 0 | 否 |

**禁止一次授权3个实例。**

每个阶段完成后，runner必须退出。人工检查结果后，用户再决定是否签下一份授权。

## 2.2 授权文件必须绑定

每份授权至少绑定：

```text
schema
status = AUTHORIZED
protocol_id
protocol_sha256
stage
runner_commit
generation_runner_file_hashes
source_lock_sha256
environment_contract_sha256
root_family_id
case_id
family_seed
rollout_seed
program_sha256
physical_spec_sha256
model_xml_expected_sha256（可在A前静态生成时绑定）
authorized_instances = 1
automatic_retry = false
output_root
single_use_nonce
reviewer_id
approved_at_utc
expires_at_utc
```

B授权额外绑定：

```text
baseline_A_artifact_manifest_sha256
baseline_A_result_sha256
baseline_A_review_status = PASS
```

C授权额外绑定：

```text
ordinary_A_vs_B_comparison_sha256
ordinary_A_vs_B_status = PASS
repeat_B_artifact_manifest_sha256
```

Agent不得自行填写：

```text
reviewer_id
approved_at_utc
status = AUTHORIZED
authorized_instances = 1
single_use_nonce
expires_at_utc
```

---

# 3. 新基线固定case、family与期望计数

## 3.1 新命名空间

新基线不复用旧cache的root ID。

```text
root_family_id = L2RAR2_REPRO_BASE_00_870000
family_index   = 0
family_seed    = 870000
rollout_seed   = 87100002
case_id        = K3_normal_hold_pause_resume
scenario       = normal_pick_place
control_variant= v0_short
```

该family仅用于A/B/C重放链，不进入R16算法性能分母。

## 3.2 固定程序

```json
[
  "observe_scene",
  "approach_object",
  "close_gripper",
  "lift",
  "verify",
  "verify",
  "transport_to_target",
  "verify"
]
```

## 3.3 固定计数

在 `v0_short` 下：

```text
high-level actions          = 8
control callbacks           = 29
action-end callbacks        = 8
render callbacks            = 37
physics steps               = 145
ordinary checkpoint rows    = 54
instrumented before/after rows = 290
```

计算依据：

```text
observe_scene       3 controls
approach_object     4
close_gripper       2
lift                3
verify              3
verify              3
transport_to_target 8
verify              3
--------------------------------
total               29 controls
```

每个control固定5个physics step：

\[
29\times5=145.
\]

如果程序或计数不一致，协议失效。不得现场修改expected count。

---

# 4. 建议目录

```bash
export REPO=/home/__compress_data/xushijie/graph_github_upload

export DEV_WT=/home/__compress_data/xushijie/graph_l2ra_r2_r14b_baseline_v1_worktree
export EXEC_WT=/home/__compress_data/xushijie/graph_l2ra_r2_r14b_exec_v1_worktree

export BASE=ab024c71d315100f0e375a844856807114bedaab
export MAIN=234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2
export BRANCH=research/l2rar2-r14b-reproducible-baseline-v1

export PY=/home/xushijie/.conda/envs/lerobot/bin/python

# 原始物理、RGB和完整状态输出，保持外置
export DATA=/home/__compress_data/xushijie/graph_l2ra_r2_r14b_baseline_v1_data
export CONTROL=$DATA/control_v1
export RUN_A=$DATA/executions/baseline_A
export RUN_B=$DATA/executions/repeat_B
export RUN_C=$DATA/executions/instrumented_C

# 轻量报告和申请材料
export ART=$DEV_WT/artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v1
export STATIC=$ART/static_v1
export APPLICATIONS=$ART/applications_v1
export COMPARISONS=$ART/comparisons_v1
export FINAL=$ART/final_v1

export DOWNLOADS=$DEV_WT/downloads/l2ra_r2_v1
```

`RUN_A/RUN_B/RUN_C`只是父目录示意。实际授权必须绑定包含nonce的唯一目录，例如：

```text
baseline_A_<nonce-prefix>
repeat_B_<nonce-prefix>
instrumented_C_<nonce-prefix>
```

不得使用 `/tmp` 保存正式执行或结果。

---

# 5. 新Agent首先需要寻找的资源

必须阅读：

```text
artifacts/pathgraph_sarm/upgrade_v2/replay_consistency_l2rar2_r14_v1/
  forensics/ordinary_002_first_divergence_v1/
    final_report.md
    review_decision.json
    source_provenance.json
    next_stage_handoff.json

upgrade_v2/l2r_replay_consistency/
upgrade_v2/l2r_forced_drop/
upgrade_v2/l2r_task_context/collector.py
upgrade_v2/l2r_task_context/repair_collection.py
upgrade_v2/l2r_hold_evidence/probe_adapter.py
upgrade_v2/visual_refine_l2/dynamic_simulator.py
upgrade_v2/visual_refine_l2/repaired_simulator.py
upgrade_v2/visual_refine_l2/renderer.py
upgrade_v2/visual_refine_l2/vision.py
```

旧K3 cache可用于历史旁路说明，但不得作为A/B/C主门：

```text
/home/__compress_data/xushijie/graph_l2ra_r2_worktree/
artifacts/pathgraph_sarm/upgrade_v2/task_context_l2rar2_v1/
data_attach_relpose_repair_v1/rollouts/
L2RAR2_REPAIR_00_840000/K3_normal_hold_pause_resume
```

---

# 6. P0：创建开发工作树并预检

## 6.1 同步远端

```bash
git -C "$REPO" fetch origin --prune

test "$(git -C "$REPO" rev-parse origin/main)" = "$MAIN"

test "$(git -C "$REPO" rev-parse \
  origin/maintenance/l2rar2-r14-ordinary002-forensics-v1)" = "$BASE"
```

任何不一致：停止并报告实际SHA。

## 6.2 创建开发工作树

```bash
git -C "$REPO" worktree list --porcelain

git -C "$REPO" worktree add \
  -b "$BRANCH" \
  "$DEV_WT" \
  "$BASE"

test "$(git -C "$DEV_WT" rev-parse HEAD)" = "$BASE"
test -z "$(git -C "$DEV_WT" status --porcelain)"
```

如果分支或目录已存在：

1. 不删除；
2. 检查HEAD；
3. 检查未提交文件；
4. 确认BASE是祖先；
5. 无法确认时停止。

## 6.3 包内静态预检

```bash
mkdir -p "$STATIC"

"$PY" -B /path/to/R14B_PACKAGE/tools/preflight.py \
  --repo "$DEV_WT" \
  --output "$STATIC/preflight.json"
```

期望：

```text
status = PASS_STATIC_ONLY
forensic_route = B_NEW_REPRODUCIBLE_BASELINE_REQUIRED
new_physical_authorization = 0
```

---

# 7. P1：新增版本化模块

新增：

```text
upgrade_v2/l2r_reproducible_baseline/
```

建议结构：

```text
__init__.py
protocol.py
environment.py
source_lock.py
authorization.py
fingerprint.py
simulator.py
capture.py
ordinary.py
instrumented.py
comparison.py
cli.py
package_results.py
tests/
```

不得直接修改历史：

```text
dynamic_simulator.py
repaired_simulator.py
ordinary_002执行目录
旧cache
R12 denial journal
```

建议常量：

```python
PROTOCOL_ID = "L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V1"
SIM_VERSION = "l2rar2_reproducible_baseline_sim_v1"
CAPTURE_VERSION = "l2rar2_reproducible_baseline_capture_v1"
COMPARISON_VERSION = "l2rar2_reproducible_baseline_compare_v1"
```

---

# 8. P2：实现统一的A/B/C模拟器代码路径

## 8.1 基类

新类继承：

```python
AttachRelposeDynamicTabletop
```

例如：

```python
class ReproducibleBaselineTabletop(AttachRelposeDynamicTabletop):
    ...
```

A、B、C必须使用同一个类、同一个 `_advance()` 和同一个 `_step_once()`。

## 8.2 不允许A/B使用旧类、C使用另一个物理类

禁止：

```text
A/B = AttachRelposeDynamicTabletop
C   = ControlledForcedDropTabletop
```

因为这样无法隔离“仪表影响”和“类实现差异”。

必须：

```text
A/B/C = ReproducibleBaselineTabletop
```

差异仅为：

```text
A/B: physics_observer = None
C:   physics_observer = ReadOnlyPhysicsRecorder
```

## 8.3 统一step封装

推荐：

```python
def _step_once(self, context):
    if self.physics_observer is not None:
        self.physics_observer.before_step(self, context)
    self.mujoco.mj_step(self.model, self.data)
    if self.physics_observer is not None:
        self.physics_observer.after_step(self, context)
```

`_advance()`必须保持现有：

```text
每个control 5 physics steps
attached时在每step前执行既有scripted writeback
control callback在5步完成后执行
```

不要在本阶段改变脚本回写、weld、solver、control frequency或动作程序。

## 8.4 仪表只读运行时防护

C中每次调用recorder前后，计算可变物理状态指纹：

```text
qpos
qvel
qacc
qacc_warmstart
mocap_pos
mocap_quat
eq_active
xfrc_applied
data.time
model.eq_data
RNG state
events
attempt lifecycle
simulator flags
```

要求：

```text
pre_recorder_state_sha256
==
post_recorder_state_sha256
```

任何不一致：

```text
R14B_INSTRUMENTATION_MUTATION_DETECTED
```

立即停止，C失败。

---

# 9. P3：固定环境合同

## 9.1 启动前环境

每次A/B/C必须使用新进程，并在Python启动前设置：

```bash
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export MUJOCO_GL=egl

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

不得在运行时静默安装、升级或降级软件包。

## 9.2 固定版本

根据当前已成功启动的ordinary环境，协议先锁定：

```text
Python = 3.10.19
NumPy  = 2.2.6
MuJoCo = 3.4.0
OpenCV = 4.13.0
```

任一不一致时：

```text
ENVIRONMENT_CONTRACT_MISMATCH
```

停止，不自动修改协议。

## 9.3 A负责建立运行时基线指纹

A必须保存：

```text
python version/executable
platform/system/release/machine
CPU model hash
critical environment allowlist
sys.path canonical hash

NumPy package path/version/native extension hashes
MuJoCo package path/version/native extension hashes
OpenCV package path/version/native extension hashes

MUJOCO_GL
EGL/GL vendor、renderer、version（能读取时）
renderer dimensions
JPEG encoder version/options
```

不记录：

```text
完整环境变量
token
SSH配置
用户名密码
```

A生成：

```text
runtime_environment_fingerprint.json
runtime_environment_fingerprint_sha256
```

B与C必须精确匹配A的运行时指纹中所有标记为`MAIN_GATE`的字段。

## 9.4 若GL vendor无法读取

允许：

```text
status = UNAVAILABLE_NOT_INSTALLED
```

但必须至少绑定：

```text
MUJOCO_GL=egl
MuJoCo native library hashes
OpenCV native library hashes
platform/machine
raw RGB hash comparison
```

不能伪造GL字符串。

---

# 10. P4：模型与源码指纹

## 10.1 生成XML哈希

在model构造前，用固定spec生成XML字符串并保存：

```text
generated_model.xml
generated_model_xml_sha256
```

A/B/C必须相同。

## 10.2 model构造后的指纹

至少记录：

```text
nq / nv / na / nbody / ngeom / neq
option timestep / gravity / integrator
body_pos/body_quat hash
geom_type/size/pos/quat/friction/mass hash
jnt_type/qposadr/dofadr hash
eq_type/obj1id/obj2id/eq_data hash
camera arrays hash
model names hash
```

生成：

```text
model_fingerprint.json
model_fingerprint_sha256
```

A/B/C完全一致。

## 10.3 生成runner文件清单

`generation_runner_file_hashes`至少包含：

```text
upgrade_v2/l2r_reproducible_baseline/*.py
upgrade_v2/l2r_task_context/collector.py
upgrade_v2/l2r_hold_evidence/probe_adapter.py
upgrade_v2/visual_refine_l2/dynamic_simulator.py
upgrade_v2/visual_refine_l2/repaired_simulator.py
upgrade_v2/visual_refine_l2/renderer.py
upgrade_v2/visual_refine_l2/vision.py
```

测试、报告模板不必作为generation runner，但应在单独analysis manifest中记录。

---

# 11. P5：checkpoint与图像记录合同

## 11.1 ordinary A/B checkpoint

保存：

```text
initial_after_forward
before_action
control_tick_callback
action_end_callback
after_perform_return
```

期望总数：

\[
1+8+29+8+8=54.
\]

## 11.2 每个checkpoint保存

```text
sequence
sampling_point
action
action_index
control_index
time
time_hex

qpos
qvel
qacc
qacc_warmstart
mocap_pos
mocap_quat
eq_active
model_eq_data
xfrc_applied

object_xyz
gripper_xyz
object_qpos
object_qvel

RNG state
events snapshot/hash
attempt lifecycle
simulator flags
contact pair summary

每个数组的shape/dtype
每个数组的little-endian raw-byte SHA256
完整checkpoint_state_sha256
```

JSON中的浮点数用于人工查看；**主门使用array raw-byte hash和canonical state hash**。

## 11.3 capture记录

每次control/action-end callback：

1. render得到原始`uint8`图像；
2. 在JPEG编码前计算：
   ```text
   raw_rgb_sha256
   shape
   dtype
   ```
3. 使用固定JPEG参数保存；
4. 计算：
   ```text
   jpeg_sha256
   ```
5. 保存状态checkpoint；
6. 记录callback exit。

固定callback顺序：

```text
callback_enter
render_start
render_end
raw_hash_computed
jpeg_saved
state_saved
callback_exit
```

## 11.4 视觉检测

全部37张图像在物理程序结束后，以固定顺序运行 `detect_frame()`。

保存：

```text
capture_order
object_centroid
gripper_centroid
object_confidence
gripper_confidence
width
height
detection_canonical_sha256
```

检测不得在物理step之间运行。

---

# 12. P6：A阶段——生成ordinary baseline A

## 12.1 A不是旧cache重放门

A只回答：

> 当前冻结代码、当前冻结环境能否生成一条完整、健康、来源齐全的新基线？

旧cache可在报告中作为历史说明，但不得出现在A主门中。

## 12.2 A前静态准备

完成代码和纯测试后：

```bash
git -C "$DEV_WT" status --short
git -C "$DEV_WT" diff --check

git -C "$DEV_WT" add \
  upgrade_v2/l2r_reproducible_baseline \
  artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v1/static_v1

git -C "$DEV_WT" commit \
  -m "research: prepare new reproducible R14B baseline runner"

git -C "$DEV_WT" push -u origin "$BRANCH"

export RUNNER_COMMIT="$(git -C "$DEV_WT" rev-parse HEAD)"
```

## 12.3 创建固定执行工作树

```bash
git -C "$REPO" worktree add --detach "$EXEC_WT" "$RUNNER_COMMIT"

test "$(git -C "$EXEC_WT" rev-parse HEAD)" = "$RUNNER_COMMIT"
test -z "$(git -C "$EXEC_WT" status --porcelain)"
```

A/B/C都从该detached worktree运行。

在三阶段完成前不得修改该工作树。

## 12.4 生成外置协议锁

```bash
mkdir -p "$CONTROL"

"$PY" -B -m upgrade_v2.l2r_reproducible_baseline.cli plan-protocol \
  --repo "$EXEC_WT" \
  --root-family-id L2RAR2_REPRO_BASE_00_870000 \
  --family-index 0 \
  --family-seed 870000 \
  --rollout-seed 87100002 \
  --case-id K3_normal_hold_pause_resume \
  --output "$CONTROL/protocol_lock.json"
```

记录：

```bash
sha256sum "$CONTROL/protocol_lock.json" \
  > "$CONTROL/protocol_lock.json.sha256"
```

协议必须仍显示：

```text
A authorized=0
B authorized=0
C authorized=0
```

## 12.5 准备A申请

Agent生成：

```text
$APPLICATIONS/baseline_A_application.json
$APPLICATIONS/baseline_A_authorization.template.json
```

模板中：

```text
status = NOT_AUTHORIZED
authorized_instances = 0
```

用户批准后，授权文件应放在外置控制目录：

```text
$CONTROL/runtime_authorizations/
r14b_baseline_A_<nonce-prefix>.json
```

## 12.6 A执行命令

只有有效授权后：

```bash
export A_AUTH=/exact/path/to/r14b_baseline_A_<nonce>.json
export A_OUT=$DATA/executions/baseline_A_<nonce-prefix>

test ! -e "$A_OUT"

env \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$PY" -B -m upgrade_v2.l2r_reproducible_baseline.cli run-ordinary \
    --repo "$EXEC_WT" \
    --stage R14B_ORDINARY_BASELINE_A \
    --protocol "$CONTROL/protocol_lock.json" \
    --authorization "$A_AUTH" \
    --output-root "$A_OUT"
```

runner必须在：

```text
import mujoco
model construction
renderer construction
```

之前消费授权和预留实例。

## 12.7 A质量门

必须全部通过：

```text
authorization consumed once
runner commit/hash match
worktree clean
environment version pass
model XML/model fingerprint complete
8 actions
29 controls
8 action-end callbacks
37 render callbacks
54 checkpoints
145 physics steps
numeric health pass
callback→return exact
weld relpose self-consistent
raw/JPEG/detection outputs complete
artifact manifest complete
no old-cache gate
```

A通过状态：

```text
R14B_BASELINE_A_COMPLETE_WAITING_HUMAN_REVIEW
```

A失败：

```text
STOP_AFTER_BASELINE_A
```

不得运行B。

---

# 13. P7：A人工复核

人工至少核查：

```text
result.json
baseline_quality_gates.json
runtime_environment_fingerprint.json
source_lock.json
model_fingerprint.json
checkpoint_counts.json
numeric_health.json
renderer_capture_manifest.csv
artifact_manifest.json
authorization_consumption.json
```

若通过，由用户签署B授权。

Agent不能以“A看起来正常”为由自动进入B。

---

# 14. P8：B阶段——独立ordinary repeat B

## 14.1 B必须是全新进程和全新输出目录

禁止：

```text
restore A snapshot
复制A输出作为B
同Python进程连续运行A和B
复用A renderer/context
复用A nonce
```

## 14.2 B执行前绑定A

B授权必须绑定：

```text
A artifact_manifest SHA256
A result SHA256
A environment fingerprint SHA256
A model fingerprint SHA256
A human review status PASS
```

## 14.3 B执行命令

```bash
export B_AUTH=/exact/path/to/r14b_repeat_B_<nonce>.json
export B_OUT=$DATA/executions/repeat_B_<nonce-prefix>

test ! -e "$B_OUT"

env \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$PY" -B -m upgrade_v2.l2r_reproducible_baseline.cli run-ordinary \
    --repo "$EXEC_WT" \
    --stage R14B_ORDINARY_REPEAT_B \
    --protocol "$CONTROL/protocol_lock.json" \
    --authorization "$B_AUTH" \
    --baseline-A "$A_OUT" \
    --output-root "$B_OUT"
```

## 14.4 A vs B比较

物理运行结束后，由纯静态比较器执行：

```bash
mkdir -p "$COMPARISONS/ordinary_A_vs_B_v1"

"$PY" -B -m upgrade_v2.l2r_reproducible_baseline.cli compare \
  --left "$A_OUT" \
  --right "$B_OUT" \
  --comparison ordinary_A_vs_B \
  --output-root "$COMPARISONS/ordinary_A_vs_B_v1"
```

---

# 15. A vs B主门合同

## 15.1 来源与环境

必须精确相同：

```text
runner commit
generation runner file hashes
protocol SHA
source lock SHA
runtime package versions
native extension hashes
critical env
model XML SHA
model fingerprint SHA
family/case/seed/program
```

## 15.2 离散序列

必须精确相同：

```text
actions
start/end sim time
low-level controls
events
callback order
capture order
attempt lifecycle
contact proxy
gripper command
render callback count
detection row count
```

## 15.3 数值状态

以下每个checkpoint要求raw-byte SHA256完全相同：

```text
qpos
qvel
qacc
qacc_warmstart
mocap_pos
mocap_quat
eq_active
model.eq_data
xfrc_applied
object/gripper状态
RNG
contact summary
```

也就是：

```text
rtol = 0
atol = 0
byte hash exact
```

若不等，报告：

```text
first mismatch checkpoint
field
left/right array shape
left/right byte SHA
max_abs_diff
L2 diff
```

不得在结果出来后放宽容差。

## 15.4 视觉状态

主门要求：

```text
37/37 raw_rgb_sha256 exact
37/37 jpeg_sha256 exact
37/37 detection_canonical_sha256 exact
```

如果物理完全一致但视觉不同：

```text
R14B_PHYSICS_PASS_RENDER_OR_VISION_FAIL
```

仍停止，不进入C或R16。

## 15.5 B结果

全部通过：

```text
R14B_ORDINARY_A_B_REPRODUCIBILITY_PASS
```

任一失败：

```text
STOP_AFTER_ORDINARY_REPEAT_B
```

C授权保持0。

---

# 16. P9：B人工复核

人工至少查看：

```text
comparison_summary.json
first_mismatch.json
state_hash_comparison.csv
discrete_sequence_comparison.json
render_comparison.csv
detection_comparison.csv
environment_comparison.json
model_comparison.json
```

只有：

```text
all_main_gates_passed = true
```

才允许用户签署C授权。

---

# 17. P10：C阶段——instrumented replay

## 17.1 C的唯一新增行为

C只增加：

```text
每个physics step前后的只读state snapshot
```

不得改变：

```text
动作
控制
weld
scripted writeback
solver
渲染
视觉检测
RNG
callback顺序
```

## 17.2 C授权绑定

C授权必须绑定：

```text
A/B comparison status PASS
A/B comparison SHA256
B artifact manifest SHA256
B result SHA256
same runner commit
same protocol
```

## 17.3 C执行命令

```bash
export C_AUTH=/exact/path/to/r14b_instrumented_C_<nonce>.json
export C_OUT=$DATA/executions/instrumented_C_<nonce-prefix>

test ! -e "$C_OUT"

env \
  PYTHONHASHSEED=0 \
  PYTHONNOUSERSITE=1 \
  MUJOCO_GL=egl \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  "$PY" -B -m upgrade_v2.l2r_reproducible_baseline.cli run-instrumented \
    --repo "$EXEC_WT" \
    --stage R14B_INSTRUMENTED_C \
    --protocol "$CONTROL/protocol_lock.json" \
    --authorization "$C_AUTH" \
    --ordinary-B "$B_OUT" \
    --ordinary-A-B-comparison \
      "$COMPARISONS/ordinary_A_vs_B_v1/comparison_summary.json" \
    --output-root "$C_OUT"
```

## 17.4 C仪表完整性

必须：

```text
physics steps = 145
before-step rows = 145
after-step rows  = 145
total step rows  = 290
recorder mutation checks = 290/290 PASS
```

instrumented trace建议外置，不提交Git。

---

# 18. B vs C主门

运行：

```bash
mkdir -p "$COMPARISONS/ordinary_B_vs_C_v1"

"$PY" -B -m upgrade_v2.l2r_reproducible_baseline.cli compare \
  --left "$B_OUT" \
  --right "$C_OUT" \
  --comparison ordinary_B_vs_instrumented_C \
  --output-root "$COMPARISONS/ordinary_B_vs_C_v1"
```

主门与A/B相同：

```text
source/environment/model exact
actions/controls/events/callback exact
54 checkpoint state hashes exact
37 raw RGB exact
37 JPEG exact
37 detection exact
RNG/contact exact
```

额外要求：

```text
instrumentation mutation = 0
step trace complete = 290
```

通过：

```text
R14B_INSTRUMENTED_PARITY_PASS
```

失败：

```text
R14B_INSTRUMENTATION_MISMATCH
```

R16仍阻断。

---

# 19. P11：生成R14-B证书

只有A、B、C全部通过，生成：

```text
$FINAL/r14b_reproducibility_certificate.json
```

必须包含：

```text
status = R14B_REPRODUCIBLE_BASELINE_CHAIN_PASS
protocol SHA
runner commit
runner file hashes
environment fingerprint SHA
model fingerprint SHA

A result/manifest SHA
B result/manifest SHA
A-vs-B comparison SHA
C result/manifest SHA
B-vs-C comparison SHA

A physical instances = 1
B physical instances = 1
C physical instances = 1
total R14-B instances = 3

R11 history = 40/8 preserved
ordinary_002 = failed and preserved

scientific_status = L2RAR2_PARTIAL_KEEP_G1
selected_candidate = null
confirmation = false
L3 = false
```

证书不得声称：

```text
loss confirmed
forced drop works
recovery candidate passes
```

---

# 20. R16解锁条件

R16 calibration只有在新的授权文件同时绑定：

```text
r14b_reproducibility_certificate SHA256
R14B_REPRODUCIBLE_BASELINE_CHAIN_PASS
R16 runner commit/hash
R16 calibration protocol SHA
```

后才可执行。

R14-B通过并不自动创建R16授权。

---

# 21. 必须实现的纯测试

物理授权前至少完成以下测试。

## 21.1 协议和case

1. `test_fixed_case_is_k3`
2. `test_new_root_namespace`
3. `test_fixed_family_and_rollout_seed`
4. `test_fixed_program_hash`
5. `test_expected_action_count_8`
6. `test_expected_control_count_29`
7. `test_expected_render_count_37`
8. `test_expected_physics_steps_145`

## 21.2 授权

9. `test_missing_authorization_denied_before_import`
10. `test_wrong_stage_denied`
11. `test_wrong_runner_commit_denied`
12. `test_wrong_runner_hash_denied`
13. `test_wrong_protocol_hash_denied`
14. `test_wrong_output_root_denied`
15. `test_expired_authorization_denied`
16. `test_nonce_reuse_denied`
17. `test_instance_reserved_before_factory`
18. `test_crash_consumes_instance`
19. `test_A_authorization_does_not_authorize_B`
20. `test_B_authorization_does_not_authorize_C`

## 21.3 环境

21. `test_pythonhashseed_required`
22. `test_mujoco_gl_egl_required`
23. `test_thread_env_fixed`
24. `test_package_version_mismatch_denied`
25. `test_environment_fingerprint_canonical`
26. `test_native_library_hash_recorded_or_explicit_unknown`

## 21.4 指纹

27. `test_float_array_hash_is_dtype_shape_sensitive`
28. `test_float_array_hash_is_byte_exact`
29. `test_checkpoint_hash_excludes_wall_clock`
30. `test_checkpoint_hash_includes_rng`
31. `test_checkpoint_hash_includes_eq_data`
32. `test_model_fingerprint_canonical`
33. `test_xml_hash_changes_on_model_change`

## 21.5 runner计数

34. `test_ordinary_expected_checkpoints_54`
35. `test_ordinary_expected_captures_37`
36. `test_physics_steps_145`
37. `test_detection_runs_after_physics_program`
38. `test_no_old_cache_main_gate`
39. `test_output_root_must_not_exist`

## 21.6 A/B比较

40. `test_identical_runs_pass`
41. `test_qpos_one_bit_difference_fails`
42. `test_warmstart_difference_fails`
43. `test_rng_difference_fails`
44. `test_event_difference_fails`
45. `test_raw_rgb_difference_fails`
46. `test_detection_difference_fails`
47. `test_environment_difference_fails`
48. `test_model_difference_fails`

## 21.7 instrumentation

49. `test_observer_none_has_no_trace`
50. `test_observer_before_after_count_290`
51. `test_recorder_mutation_detected`
52. `test_instrumented_checkpoint_mismatch_fails`
53. `test_C_requires_AB_pass`
54. `test_instrumented_cannot_run_twice`

## 21.8 结果与状态

55. `test_certificate_requires_all_three_pass`
56. `test_certificate_does_not_select_candidate`
57. `test_certificate_does_not_open_L3`
58. `test_R16_requires_certificate_hash`
59. `test_old_ordinary002_remains_negative`
60. `test_R11_accounting_preserved`

至少60项纯测试。可增加，不可减少。

---

# 22. 静态测试命令

```bash
cd "$DEV_WT"

PYTHONNOUSERSITE=1 "$PY" -B -m unittest discover \
  -s upgrade_v2/l2r_reproducible_baseline/tests \
  -p 'test_*.py' \
  -v \
  | tee "$STATIC/unit_tests.log"

"$PY" -B -m compileall -q \
  upgrade_v2/l2r_reproducible_baseline

git diff --check
```

测试不得构造真实MuJoCo model/data/renderer。使用fake factory、fake arrays和fixture。

---

# 23. 执行账目和共享门禁

新增控制目录：

```text
<git-common-dir>/research_controls/
L2RAR2_R14B_NEW_REPRODUCIBLE_BASELINE_V1/
```

包含：

```text
registry.lock
authorization_journal.jsonl
consumption_journal.jsonl
```

要求：

- `fcntl`文件锁；
- `O_NOFOLLOW`；
- append-only哈希链；
- 写入后`fsync`；
- model import/factory前预留；
- 崩溃仍计数；
- stage不可互换；
- output root不可覆盖；
- nonce全局唯一；
- A/B/C不得并发。

不得修改R12永久零物理策略来增加allow path。

---

# 24. 允许修复什么，何时必须重启链

## 24.1 生成代码改变

如果A之后修改了以下任何文件：

```text
simulator
ordinary runner
instrumented runner
capture
environment
fingerprint
collector/probe/simulator/renderer/vision依赖
```

则：

```text
A无效作为新协议基线
必须新runner commit
必须新protocol
必须重新从A开始
```

已消费的实例仍保留。

## 24.2 纯比较器错误

如果物理输出完整，只是静态comparison代码有bug：

- 保留原比较失败目录；
- 可修复纯比较器；
- 用新analysis commit重新比较已有A/B/C；
- 不修改物理输出；
- 不改变阈值；
- 报告前后差异。

这不自动要求重跑物理。

---

# 25. 失败处理

## 25.1 A失败

```text
STOP_AFTER_BASELINE_A
B/C授权 = 0
R16 = blocked
```

若需要改runner，创建新协议并重新申请A。

## 25.2 A/B不一致

```text
STOP_AFTER_ORDINARY_REPEAT_B
C授权 = 0
R16 = blocked
```

不得运行第三个ordinary作为“重试”。

## 25.3 B/C不一致

```text
R14B_INSTRUMENTATION_MISMATCH
R16 = blocked
```

## 25.4 视觉不一致但物理一致

仍然失败：

```text
R14B_PHYSICS_PASS_RENDER_OR_VISION_FAIL
```

因为R16后续需要在线视觉数据。

## 25.5 全部通过

只生成工程证书，等待R16 calibration的独立申请。

---

# 26. 交付物

## 26.1 Static

```text
static_v1/
  preflight.json
  protocol_draft.json
  source_lock.template.json
  environment_contract.json
  fixed_case_spec.json
  expected_counts.json
  test_results.json
  actual_commands.txt
  run_manifest.json
```

## 26.2 Applications

```text
applications_v1/
  baseline_A_application.json
  baseline_A_authorization.template.json
  repeat_B_application.json
  repeat_B_authorization.template.json
  instrumented_C_application.json
  instrumented_C_authorization.template.json
```

模板始终authorized=0。

## 26.3 Comparison

```text
comparisons_v1/ordinary_A_vs_B_v1/
  comparison_summary.json
  first_mismatch.json
  state_hash_comparison.csv
  discrete_sequence_comparison.json
  environment_comparison.json
  model_comparison.json
  render_comparison.csv
  detection_comparison.csv
  run_manifest.json

comparisons_v1/ordinary_B_vs_C_v1/
  同上
  instrumentation_integrity.json
```

## 26.4 Final

```text
final_v1/
  decision.json
  final_report.md
  next_stage_handoff.json
  r14b_reproducibility_certificate.json（只有全通过才存在）
  physical_execution_accounting.json
  external_artifacts.tsv
  package_manifest.json
  actual_commands.txt
  run_manifest.json
```

---

# 27. 原始执行目录最低文件

A/B：

```text
authorization_consumption.json
protocol_lock.json
source_lock.json
runtime_environment_fingerprint.json
generated_model.xml
model_fingerprint.json

checkpoint_state_trace.jsonl
callback_capture_trace.jsonl
actions.jsonl
low_level_controls.jsonl
events.jsonl
vision_detections.jsonl

rgb/front/*.jpg
renderer_capture_manifest.csv
checkpoint_counts.json
numeric_health.json
baseline_quality_gates.json

artifact_manifest.json
result.json
```

C额外：

```text
physics_step_trace.jsonl或无损NPZ
instrumentation_integrity.json
```

---

# 28. 打包

建议结果包：

```text
downloads/l2ra_r2_v1/
L2RAR2_R14B_New_Reproducible_Baseline_results_v1.zip
```

允许打包：

- 轻量协议、报告、比较表、证书、测试；
- 源码patch；
- manifest和sidecar。

禁止打包：

- 37张RGB；
- 完整checkpoint trace；
- 完整100 Hz trace；
- 授权文件中的不必要身份字段；
- 原始旧cache；
- `.git`；
- `__pycache__`、`.pyc`；
- token/private key；
- 嵌套ZIP。

外置文件写入：

```text
external_artifacts.tsv
```

包含：

```text
role
absolute_path
size_bytes
sha256或tree_sha256
included_in_zip
recovery_instructions
```

---

# 29. Git提交与推送

静态runner提交后，A/B/C使用固定detached worktree。

最终结果可在开发工作树提交：

```bash
git -C "$DEV_WT" add \
  upgrade_v2/l2r_reproducible_baseline \
  artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v1 \
  downloads/l2ra_r2_v1/L2RAR2_R14B_New_Reproducible_Baseline_results_v1.zip.sha256

git -C "$DEV_WT" commit \
  -m "research: establish R14B reproducible baseline chain"

git -C "$DEV_WT" push -u origin "$BRANCH"
```

不得：

```text
merge main
push main
force-push
删除ordinary_002
删除forensics
删除R11/R12账目
```

---

# 30. Agent最终报告模板

```text
R14-B implementation branch:
runner commit:
result commit:
origin/main:

forensic route:
old-cache exact reconstruction:
ordinary_002 status:

protocol ID:
protocol SHA:
source lock SHA:
environment contract SHA:

baseline family:
case:
family seed:
rollout seed:
program SHA:
physical spec SHA:
model XML SHA:

A authorization:
A execution:
A quality gates:
A manifest SHA:

B authorization:
B execution:
A-vs-B source gate:
A-vs-B physics gate:
A-vs-B render gate:
A-vs-B detection gate:
A-vs-B overall:
first mismatch if any:

C authorization:
C execution:
instrumentation rows:
mutation checks:
B-vs-C source gate:
B-vs-C physics gate:
B-vs-C render gate:
B-vs-C detection gate:
B-vs-C overall:

R14-B total physical instances:
R11 historical accounting:
R14 old execution accounting:

reproducibility certificate:
R16 prerequisite status:

scientific status:
retained graph:
selected candidate:
confirmation:
L3:

training jobs:
model API calls:
key reads:

tests:
compileall:
secret scan:
git diff check:

result ZIP:
result ZIP SHA256:
external data roots:
known limitations:
```

---

# 31. 立即停止矩阵

| 条件 | 动作 |
|---|---|
| main或BASE变化 | 停止 |
| forensics不是Route B | 停止 |
| runner工作树不干净 | 不生成授权申请 |
| protocol/source/env hash变化 | 旧授权无效 |
| 无用户签署授权 | 不import MuJoCo |
| output root已存在 | 拒绝 |
| nonce已消费 | 拒绝 |
| A失败 | 不运行B |
| B授权未绑定A | 拒绝 |
| A/B任一主门失败 | 不运行C |
| C授权未绑定A/B PASS | 拒绝 |
| recorder改变状态 | C失败 |
| raw RGB不同 | 全链失败 |
| 需要调容差 | 不调 |
| 生成代码在A后改变 | 重启新协议 |
| R16试图跳过证书 | 拒绝 |
| 工程全通过 | 仍不选候选、不进L3 |

---

# 32. 本阶段最终原则

1. 旧cache保留历史价值，但不再充当exact replay真值。
2. A建立当前基线；B证明当前ordinary可重复；C证明仪表不改变轨迹。
3. 三次运行必须分别授权，分别停止，分别人工复核。
4. A/B/C使用同一runner commit和同一模拟器代码路径。
5. 数值状态使用raw-byte hash精确比较，不在结果后改容差。
6. 物理和视觉重现都必须通过，R16才可继续。
7. A/B/C数据不进入算法性能分母。
8. R14-B证书只是执行基础设施证书，不是科学候选通过。
9. R16 calibration仍需单独授权。
10. G1继续保留，L3继续关闭。

## 核心总结

R14-B不是“再试一次旧缓存”。它从当前可审计代码与环境出发，建立一条新的、完整记录的基线：

```text
ordinary A
→ independent ordinary B
→ read-only instrumented C
```

只有三者在来源、环境、模型、物理状态、渲染和视觉检测上均满足冻结主门，才允许把这条链作为R16受控强制掉落实验的执行基础。
