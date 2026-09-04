# PathGraph-SARM 阶段 6 Agent 详细操作命令 V1.0

- 入口：`G2=GO_STAGE6`。
- 阶段名称：冻结图奖励驱动的 RA-BC 策略训练与闭环验证。
- 第一直接门：真实动作 graph demonstration 可用于 BC。
- 文档形式：入口结论 + 通用规范 + 7 个独立小阶段。
- 固定执行：GPU 提权查看、可并行则多 GPU 并行、每轮 ZIP、checkpoint/大文件默认不打包。
- 主方法名称：`pathgraph_reward_v1_locked`。
- 冻结奖励参数：`lambda=0.5, eta=0.0, beta=0.0`。

> Agent 必须按文档顺序执行。阶段 6.1 未输出 `POLICY_DATA_REAL_ACTION_READY` 时，不得启动完整策略训练；阶段 6.3 未锁定 policy protocol 时，不得启动 24-job 正式矩阵；阶段 6.4 的 checkpoint 未按 validation 锁定时，不得运行 test。


---

<!-- BEGIN FILE: README.md -->

# PathGraph-SARM 阶段 6 Agent 操作文档包 V1.0

## 执行顺序

Agent 必须按以下顺序执行，每个小阶段单独完成、单独验收、单独打 ZIP：

```text
00_阶段5验收与阶段6入口结论.md
01_阶段6通用执行规范.md
阶段6.1_策略可用图数据与持久输入冻结.md
阶段6.2_全训练集奖励权重与加权数据管线.md
阶段6.3_公平训练协议与并行Smoke.md
阶段6.4_多方法多任务多Seed完整策略训练.md
阶段6.5_冻结闭环策略评估.md
阶段6.6_机制统计与失败案例分析.md
阶段6.7_G3决策与M4冻结.md
```

## 阶段 6 总体目标

将阶段 5 冻结的 `reward_v1` 转换为策略训练样本权重，在完全相同的策略架构、训练数据、训练预算与 seed 条件下，对比：

```text
bc_all
linear_sarm_equiv
sequential_transition
pathgraph_reward_v1_locked
```

主要评估：

```text
task success
recovery success
long-horizon completion
A→B / B→A order success
worst-order success
fixed-order non-regression
```

## 小阶段与每轮 ZIP

| 小阶段 | 总体上要完成的工作 | 每轮 ZIP |
|---|---|---|
| 6.1 | 持久化 reward bundle，定位策略入口，采集真正带执行动作的 graph-task demonstrations，冻结策略数据集 | `stage6_1_policy_ready_data_and_input_freeze.zip` |
| 6.2 | 对完整 train/val 数据运行三 seed reward 推理，生成 chunk 权重并接入 weighted BC 数据/损失管线 | `stage6_2_weighting_pipeline.zip` |
| 6.3 | 锁定公平训练协议，运行 2 任务 × 4 方法的并行 smoke，确认训练入口可用 | `stage6_3_protocol_lock_and_smoke.zip` |
| 6.4 | 完成 2 任务 × 4 方法 × 3 seed 的完整策略训练与 validation-only checkpoint 选择 | `stage6_4_full_policy_training.zip` |
| 6.5 | 参数与 checkpoint 锁定后，按条件并行运行闭环 test rollouts | `stage6_5_frozen_policy_evaluation.zip` |
| 6.6 | 进行层级 bootstrap、机制统计、权重覆盖和代表性失败案例分析 | `stage6_6_mechanism_and_statistics.zip` |
| 6.7 | 作出 G3 决策，冻结 M4，并生成 Stage 7 handoff | `stage6_7_g3_m4_freeze.zip` |

阶段全部完成后额外生成：

```text
stage6_complete.zip
```

## 固定执行要求

- GPU 必须先尝试提权查询；
- 相互独立的 seed、method、task、evaluation shard 可以多 GPU 并行时，默认并行；
- 每轮结束立即提供 ZIP 的绝对路径与 SHA256；
- checkpoint、模型权重、原始 demonstration、视频、缓存和大数组默认不进入 ZIP；
- 大文件只写入 `checkpoint_manifest.tsv` 与 `large_file_manifest.tsv`；
- 不安排与策略加权验证无关的全仓库审计、架构搜索或额外泛化测试。

## 阶段出口

```text
artifacts/pathgraph_sarm/stage6/m4_policy_results_v1/g3_decision.md
artifacts/pathgraph_sarm/stage6/m4_policy_results_v1/stage7_handoff.md
artifacts/pathgraph_sarm/stage6/downloads/stage6_complete.zip
```

**核心点：阶段 6 用真实动作数据和冻结图奖励完成公平、并行、可复现的 RA-BC 下游验证。**

<!-- END FILE: README.md -->


---

<!-- BEGIN FILE: 00_阶段5验收与阶段6入口结论.md -->

# PathGraph-SARM 阶段 5 验收与阶段 6 入口结论

## 1. 最终结论

阶段 5 可以正式关闭，阶段 6 可以启动。正式状态记录为：

```text
STAGE5_PACKAGE_INTEGRITY = PASS
STAGE5_G2_DECISION = GO_STAGE6
M3 = GRAPH_REWARD_READY
STAGE6_ENTRY = ALLOWED
STAGE6_FIRST_GATE = POLICY_DATA_REAL_ACTION_READY
```

`STAGE6_FIRST_GATE` 不是要求退回重做阶段 5，而是阶段 6 的第一项直接推进工作：将已经冻结的 `reward_v1` 接入真正带动作的策略训练数据。

## 2. 已复核的阶段 5 交付

上传文件：

```text
stage5_complete.zip
```

SHA256：

```text
fb1b2b8ae5dc16ed71024ec62063b2db95bd540ff60c13a870e984eb42d6cc83
```

复核结果：

- `unzip -t` 完整性检查通过；
- `reward_v1/STAGE5_REWARD_SHA256SUMS.txt` 校验通过；
- `real_predictions_v1/REAL_PREDICTIONS_SHA256SUMS.txt` 校验通过；
- `reward_v1/metrics/g2_gate.json` 的决策为 `GO_STAGE6`；
- `reward_v1/configs/reward_selection_lock.json` 已锁定，参数选择来源为 validation 与 Oracle graph traces；
- 冻结奖励参数为 `lambda=0.5`、`eta=0.0`、`beta=0.0`；
- recovery debt cap 已启用；
- 三 seed reward-model bundle 已形成，阶段 5 的模型门为 2/3 seed 通过，ensemble 已满足 G2；
- Stage 6 weight schema、reward engine、model bundle 和 handoff 均存在。

阶段 5 核心指标：

| 指标 | 数值 |
|---|---:|
| model gate | 2/3 |
| legal path gap | 0 |
| failure negative rate | 1.000 |
| recovery positive rate | 0.750 |
| recovery-cycle nonpositive rate | 1.000 |
| positive loop rate | 0 |
| success-return Spearman | 0.565 |
| success-failure margin | 0.265 |
| fixed-order drop | 0 |

这些结果足以进入图奖励驱动的策略加权阶段。

## 3. 阶段 6 必须先解决的直接数据问题

阶段 2 的 graph-valid 数据可用于节点、边、路径与恢复奖励诊断，但不能直接作为 BC/RA-BC 的动作监督数据。现有阶段 2 交付明确标记：

```text
controller_source = scripted_oracle
source_format = synthetic_lowdim_json
```

进一步检查阶段 2 的 synthetic episode 后，动作字段为：

```json
"action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

因此，这批轨迹不能被静默当作策略训练 demonstration。阶段 6.1 必须直接采集或生成“动作真实执行并被保存”的 graph-task episode，要求保存的 `action_applied` 与传入 `env.step(action)` 的动作一致。

这项工作与策略训练直接相关，不扩展为额外的数据审计，也不推翻阶段 2—5 的奖励模型证据。

## 4. 奖励方法的命名边界

冻结配置中：

```text
eta = 0.0
beta = 0.0
```

因此阶段 6 主方法统一命名为：

```text
pathgraph_reward_v1_locked
```

不要在阶段 6 的下游策略结论中声称：

- 非零 repeated-edge penalty 带来了策略提升；
- 非零 uncertainty LCB 带来了策略提升。

阶段 6 可以继续记录 `reward_std`，但实际冻结权重中 `beta=0`。非零 `eta/beta` 的策略敏感性实验放到后续扩展阶段，不在阶段 6 为追求结果而重新调参。

## 5. 阶段 6 的直接目标

阶段 6 只回答一个主要问题：

> 在相同策略架构、相同动作数据、相同训练预算和相同随机种子的条件下，使用冻结的 PathGraph reward 对 BC 样本进行加权，是否能提升分支/恢复任务的闭环成功率、恢复成功率和最差合法顺序成功率，同时不损害固定顺序表现？

阶段 6 不重新训练 reward model，不使用 policy test rollout 调整 `lambda/eta/beta`，不做自动图发现。

## 6. 入口执行顺序

Agent 必须按以下顺序推进：

```text
6.1 真实动作图任务数据与持久输入冻结
6.2 全训练集 reward 推理、chunk 权重与加权数据管线
6.3 公平训练协议锁定与并行 smoke
6.4 多方法 × 多任务 × 多 seed 完整策略训练
6.5 冻结闭环策略评估
6.6 机制统计、置信区间与失败案例
6.7 G3 决策、M4 冻结与 Stage 7 交接
```

**核心结论：阶段 5 已通过 G2，可以进入阶段 6；阶段 6.1 先补齐真实动作 demonstration，随后立即开展冻结 reward 驱动的并行 RA-BC 策略实验。**

<!-- END FILE: 00_阶段5验收与阶段6入口结论.md -->


---

<!-- BEGIN FILE: 01_阶段6通用执行规范.md -->

# PathGraph-SARM 阶段 6：通用执行规范

> 阶段名称：冻结图奖励驱动的 RA-BC 策略训练与闭环验证。  
> 入口：`G2=GO_STAGE6`。  
> 第一直接门：`POLICY_DATA_REAL_ACTION_READY`。  
> 里程碑：`M4=POLICY_EVIDENCE_READY`。  
> 决策门：`G3`。

---

## 1. 给 Agent 的总命令

在现有 CUPID 仓库内继续推进，不重跑阶段 1—5，不改变冻结的 reward-model checkpoint、GraphSpec、`lambda/eta/beta` 或 Stage 5 selection lock。

先持久化 Stage 5 reward bundle，并构建真正带动作监督的 graph-task policy dataset；随后对全训练集推理并生成 sample/chunk weights；再用同一策略架构、同一初始化、同一 batch 顺序与同一训练预算运行四种方法；checkpoint 仅由 validation 选择；锁定后再运行 test rollouts。

不要把阶段 2 的零动作 synthetic trajectory 作为 BC demonstration。不要根据 Stage 6 test 结果回头调整 reward 参数。不要以 weighted sampling 替代主实验的 weighted loss；主实验保持 batch 数据流一致，只改变 loss 权重。

推进实验优先。除输入 hash、动作可用性、NaN/Inf、权重有效性、checkpoint 可读性、selection lock 和 ZIP 完整性之外，不扩展无关审计。

---

## 2. 统一环境变量

每个新终端先执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/__compress_data/xushijie/CUPID}"
if [ ! -d "$REPO_ROOT" ] && [ -d /home/xushijie/CUPID ]; then
  export REPO_ROOT=/home/xushijie/CUPID
fi
test -d "$REPO_ROOT"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export REPO_CODE_ROOT="${REPO_CODE_ROOT:-$REPO_ROOT/repo}"

export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE5_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage5"
export STAGE5_REWARD="$STAGE5_ROOT/reward_v1"
export STAGE5_PREDICTIONS="$STAGE5_ROOT/real_predictions_v1"

export STAGE6_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage6"
export STAGE6_TOOLS="$REPO_ROOT/tools/stage6"
export STAGE6_CONFIG_ROOT="$REPO_ROOT/configs/stage6"
export STAGE6_ROUNDS="$STAGE6_ROOT/rounds"
export STAGE6_INPUTS="$STAGE6_ROOT/frozen_inputs_v1"
export STAGE6_DATA="$STAGE6_ROOT/policy_dataset_v1"
export STAGE6_WEIGHTS="$STAGE6_ROOT/policy_weights_v1"
export STAGE6_PROTOCOL="$STAGE6_ROOT/policy_protocol_v1"
export STAGE6_TRAIN="$STAGE6_ROOT/policy_training_v1"
export STAGE6_EVAL="$STAGE6_ROOT/policy_evaluation_v1"
export STAGE6_FINAL="$STAGE6_ROOT/m4_policy_results_v1"
export STAGE6_DOWNLOADS="$STAGE6_ROOT/downloads"

export POLICY_SEEDS="${POLICY_SEEDS:-20260909,20260910,20260911}"
export REWARD_SEEDS="${REWARD_SEEDS:-20260906,20260907,20260908}"
export POLICY_CHUNK_HORIZON="${POLICY_CHUNK_HORIZON:-16}"
export POLICY_OBS_HORIZON="${POLICY_OBS_HORIZON:-2}"
export REWARD_HISTORY_STEPS="${REWARD_HISTORY_STEPS:-32}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-7000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$STAGE6_TOOLS/lib" \
  "$STAGE6_CONFIG_ROOT" \
  "$STAGE6_ROUNDS" \
  "$STAGE6_INPUTS" \
  "$STAGE6_DATA" \
  "$STAGE6_WEIGHTS" \
  "$STAGE6_PROTOCOL" \
  "$STAGE6_TRAIN" \
  "$STAGE6_EVAL" \
  "$STAGE6_FINAL" \
  "$STAGE6_DOWNLOADS" \
  "$STAGE6_ROOT/_runtime"
```

将以上变量保存为：

```bash
cat > "$STAGE6_ROOT/stage6_env.sh" <<'SH'
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/__compress_data/xushijie/CUPID}"
if [ ! -d "$REPO_ROOT" ] && [ -d /home/xushijie/CUPID ]; then export REPO_ROOT=/home/xushijie/CUPID; fi
export PYTHON_BIN="${PYTHON_BIN:-python}"
export REPO_CODE_ROOT="${REPO_CODE_ROOT:-$REPO_ROOT/repo}"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE5_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage5"
export STAGE5_REWARD="$STAGE5_ROOT/reward_v1"
export STAGE5_PREDICTIONS="$STAGE5_ROOT/real_predictions_v1"
export STAGE6_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage6"
export STAGE6_TOOLS="$REPO_ROOT/tools/stage6"
export STAGE6_CONFIG_ROOT="$REPO_ROOT/configs/stage6"
export STAGE6_ROUNDS="$STAGE6_ROOT/rounds"
export STAGE6_INPUTS="$STAGE6_ROOT/frozen_inputs_v1"
export STAGE6_DATA="$STAGE6_ROOT/policy_dataset_v1"
export STAGE6_WEIGHTS="$STAGE6_ROOT/policy_weights_v1"
export STAGE6_PROTOCOL="$STAGE6_ROOT/policy_protocol_v1"
export STAGE6_TRAIN="$STAGE6_ROOT/policy_training_v1"
export STAGE6_EVAL="$STAGE6_ROOT/policy_evaluation_v1"
export STAGE6_FINAL="$STAGE6_ROOT/m4_policy_results_v1"
export STAGE6_DOWNLOADS="$STAGE6_ROOT/downloads"
export POLICY_SEEDS="${POLICY_SEEDS:-20260909,20260910,20260911}"
export REWARD_SEEDS="${REWARD_SEEDS:-20260906,20260907,20260908}"
export POLICY_CHUNK_HORIZON="${POLICY_CHUNK_HORIZON:-16}"
export POLICY_OBS_HORIZON="${POLICY_OBS_HORIZON:-2}"
export REWARD_HISTORY_STEPS="${REWARD_HISTORY_STEPS:-32}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-7000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
SH

source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"
```

---

## 3. 阶段 5 入口文件定位

先执行：

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

test -f "$STAGE5_REWARD/metrics/g2_gate.json"
test -f "$STAGE5_REWARD/configs/reward_selection_lock.json"
test -f "$STAGE5_REWARD/configs/reward_config_v1.yaml"
test -f "$STAGE5_REWARD/configs/model_bundle.json"
test -f "$STAGE5_REWARD/configs/stage6_weight_schema.json"
test -f "$STAGE5_REWARD/code/reward_engine.py"
test -f "$STAGE5_REWARD/code/reward_types.py"

"$PYTHON_BIN" - <<'PYCODE'
import json, os
root=os.environ["STAGE5_REWARD"]
g=json.load(open(os.path.join(root,"metrics/g2_gate.json")))
lock=json.load(open(os.path.join(root,"configs/reward_selection_lock.json")))
assert g["decision"]=="GO_STAGE6", g
assert lock["locked"] is True, lock
s=lock["selected"]
assert float(s["lambda"])==0.5
assert float(s["eta"])==0.0
assert float(s["beta"])==0.0
print("STAGE5_G2_AND_LOCK_OK")
PYCODE
```

如果 Stage 5 交付只存在于解压目录而不在仓库标准路径，Agent 将其复制到：

```text
$STAGE5_ROOT/reward_v1
$STAGE5_ROOT/real_predictions_v1
```

复制后必须执行包内 SHA256 校验，不允许手动改写冻结内容。

---

## 4. GPU 必须提权查看

创建脚本：

```bash
cat > "$STAGE6_TOOLS/query_gpus.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

OUT="${1:-/tmp/stage6_gpu_snapshot.txt}"
mkdir -p "$(dirname "$OUT")"

{
  echo "timestamp=$(date -Is)"
  echo "hostname=$(hostname)"
  echo "user=$(id -un)"
  echo "uid=$(id -u)"

  if command -v sudo >/dev/null 2>&1; then
    if sudo -n nvidia-smi \
      --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits; then
      echo "gpu_query_mode=sudo_noninteractive"
      exit 0
    fi
  fi

  if nvidia-smi \
    --query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader,nounits; then
    echo "gpu_query_mode=direct_fallback"
    echo "note=sudo_noninteractive_unavailable"
    exit 0
  fi

  echo "gpu_query_mode=failed"
  echo "note=do_not_conclude_no_gpu_until_interactive_sudo_is_tried"
  exit 2
} | tee "$OUT"
SH
chmod +x "$STAGE6_TOOLS/query_gpus.sh"
```

每个 GPU 小阶段开始时执行：

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_before.txt" || {
  echo "自动查询失败。若当前终端允许交互提权，立即执行：sudo nvidia-smi"
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/gpu_interactive.txt"
}
```

阶段结束后再次执行：

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_after.txt" || true
```

规则：

- 普通权限失败时，不得直接写“无 GPU”；
- 先尝试 `sudo -n nvidia-smi`；
- 可交互时再执行 `sudo nvidia-smi`；
- 若环境明确无 sudo 密码能力，但直接 `nvidia-smi` 可见 GPU，则记录这一事实并继续；
- 每个 CUDA job 在日志首行记录 `CUDA_VISIBLE_DEVICES`、`torch.cuda.is_available()` 与设备名。

---

## 5. 多 GPU 并行原则

只要不产生共享写冲突，以下维度默认并行：

```text
reward seed
policy method
policy seed
task
evaluation condition
evaluation rollout shard
```

优先采用实验级并行：

```text
一个独立 job 占一张 GPU
```

不为了形式上的并行强行改成 DDP。只有单个 policy job 已原生支持 DDP，且单 job 本身明显成为瓶颈时，才使用：

```bash
torchrun --standalone --nproc_per_node=<GPU_COUNT> ...
```

每个 job 必须使用独立目录：

```text
jobs/<job_id>/
  command.sh
  stdout.log
  stderr.log
  status.json
  resolved_config.yaml
  checkpoints/
  metrics/
```

严禁多个 job 写同一个 checkpoint、日志或 Hydra output 目录。

创建通用 GPU 选择辅助命令：

```bash
cat > "$STAGE6_TOOLS/free_gpu_ids.py" <<'PYCODE'
import argparse, subprocess

p=argparse.ArgumentParser()
p.add_argument("--min-free-mb",type=int,default=7000)
a=p.parse_args()

cmd=[
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
]
out=subprocess.check_output(cmd,text=True)
ids=[]
for line in out.strip().splitlines():
    idx,free=[x.strip() for x in line.split(",")]
    if int(free)>=a.min_free_mb:
        ids.append(idx)
print(",".join(ids))
PYCODE
```

调用：

```bash
FREE_GPUS="$("$PYTHON_BIN" "$STAGE6_TOOLS/free_gpu_ids.py" --min-free-mb "$GPU_MIN_FREE_MB")"
echo "FREE_GPUS=$FREE_GPUS"
```

若无空闲 GPU，Agent 记录等待原因并在资源释放后继续；不要把可并行 job 永久改为串行计划。

---

## 6. 阶段 6 冻结配置

创建：

```bash
cat > "$STAGE6_CONFIG_ROOT/stage6.yaml" <<'YAML'
stage:
  name: pathgraph_sarm_stage6
  milestone: M4_POLICY_EVIDENCE_READY
  entry_gate: GO_STAGE6
  first_gate: POLICY_DATA_REAL_ACTION_READY

tasks:
  primary:
    - transport_recovery
    - transport_dual_order
  fixed_order_controls:
    - natural_no_intervention
    - order_A_then_B
    - order_B_then_A

methods:
  required:
    - bc_all
    - linear_sarm_equiv
    - sequential_transition
    - pathgraph_reward_v1_locked
  optional_if_already_runnable:
    - original_sarm
    - arm

reward:
  frozen_name: pathgraph_reward_v1_locked
  lambda: 0.5
  eta: 0.0
  beta: 0.0
  recovery_debt_cap: true
  selection_locked: true
  allow_stage6_retune: false

data:
  action_chunk_horizon: 16
  observation_horizon: 2
  reward_history_steps: 32
  split_group_key: initial_state_group_id
  target:
    transport_dual_order:
      order_A_then_B: 80
      order_B_then_A: 80
    transport_recovery:
      natural_success: 60
      drop_and_regrasp_success: 60
      gripper_reopen_success: 60
      controlled_failure: 40
  minimum_gate:
    train_episodes_per_task: 60
    val_episodes_per_task: 20
    each_dual_order_train: 20
    successful_recovery_train: 20
    nonzero_action_step_ratio: 0.50
    action_std_min: 0.0001

weights:
  source: frozen_reward_v1
  chunk_return: sum
  positive_clip: true
  normalize_per_task: true
  q_clip: 0.99
  gamma_candidates: [1.0, 0.75, 0.5]
  max_weight: 5.0
  min_ess_ratio: 0.25
  max_zero_weight_ratio: 0.85
  min_recovery_positive_coverage: 0.50
  weighted_sampling: false
  weighted_loss: true

policy:
  seeds: [20260909, 20260910, 20260911]
  selection_split: val
  test_used_for_selection: false
  same_architecture_across_methods: true
  same_steps_across_methods: true
  same_initialization_within_task_seed: true
  smoke_steps: 2000

evaluation:
  rollout_count_per_condition: 50
  uniform_min_if_resource_limited: 30
  paired_eval_seeds: true
  conditions:
    transport_recovery:
      - natural_no_intervention
      - drop_and_regrasp
      - gripper_reopen
    transport_dual_order:
      - order_A_then_B
      - order_B_then_A
  bootstrap_resamples: 5000

g3:
  min_graph_task_success_gain: 0.05
  min_improved_policy_seeds: 2
  max_fixed_order_drop: 0.05
  min_recovery_or_worst_order_gain: 0.08
  alternative_min_long_horizon_gain: 0.05
YAML
```

阶段 6 开始后，此配置可以根据实际文件路径做路径字段补充，但不能更改 Stage 5 冻结 reward 参数。策略训练预算可根据现有成功配置锁定一次；一旦阶段 6.3 生成 protocol lock，就不得按方法单独修改。

---

## 7. 统一轮次目录

每个小阶段使用以下目录：

```bash
ROUND_NAME="stage6_1_policy_ready_data_and_input_freeze"  # 按轮次修改
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums}
```

每轮创建 `run_manifest.md`：

```bash
cat > "$ROUND_DIR/run_manifest.md" <<EOF
# $ROUND_NAME run manifest

- start_time: $(date -Is)
- hostname: $(hostname)
- user: $(id -un)
- repo_root: $REPO_ROOT
- git_commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unavailable)
- stage5_reward: $STAGE5_REWARD
- stage6_config: $STAGE6_CONFIG_ROOT/stage6.yaml
- policy_seeds: $POLICY_SEEDS
- reward_seeds: $REWARD_SEEDS
- gpu_query: privileged_first
- parallel_policy: independent_jobs_per_gpu
- checkpoint_packaging: omitted_by_default
- large_file_packaging: manifest_only
EOF
```

实际执行命令必须保存：

```bash
printf '%q ' <实际命令及参数> > "$ROUND_DIR/commands/<job_id>.sh"
printf '\n' >> "$ROUND_DIR/commands/<job_id>.sh"
chmod +x "$ROUND_DIR/commands/<job_id>.sh"
```

---

## 8. 每轮 ZIP：checkpoint 与大文件默认不打包

在已有 Stage 5 `package_round.py` 可用时，优先复制并复用；否则实现：

```text
tools/stage6/package_round.py
```

CLI：

```bash
python tools/stage6/package_round.py \
  --round-dir <ROUND_DIR> \
  --output-zip <OUTPUT_ZIP> \
  --max-file-mb 200
```

脚本必须执行：

1. 遍历 `round-dir`；
2. 排除后缀：
   ```text
   .pt .pth .ckpt .safetensors .bin .onnx
   ```
3. 默认排除目录：
   ```text
   checkpoints/ raw_episodes/ videos/ cache/ wandb/
   ```
4. 其他单文件超过 `max-file-mb` 时排除；
5. 将 checkpoint 写入 `checkpoint_manifest.tsv`；
6. 将其他排除大文件写入 `large_file_manifest.tsv`；
7. manifest 至少含：
   ```text
   path size_bytes sha256_or_not_computed reason related_job
   ```
8. 对实际进入 ZIP 的文件生成 `SHA256SUMS.txt`；
9. 生成 ZIP；
10. 执行：
    ```bash
    unzip -t <ZIP>
    sha256sum <ZIP>
    ```
11. 将 ZIP 完整性结果写入：
    ```text
    zip_integrity.txt
    ```
12. 不因为 checkpoint 未进入 ZIP 而判定失败。

每轮标准调用：

```bash
ZIP_NAME="<本轮ZIP文件名>"
python "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

每轮完成回复必须明确给出：

```text
ROUND_STATUS
ZIP_ABSOLUTE_PATH
ZIP_SHA256
NEXT_STAGE
```

不能只写“已打包”。

---

## 9. 必要的作业状态格式

每个 job 写 `status.json`：

```json
{
  "job_id": "task__method__seed",
  "state": "RUNNING|SUCCEEDED|FAILED",
  "start_time": "",
  "end_time": "",
  "hostname": "",
  "cuda_visible_devices": "",
  "exit_code": 0,
  "command_file": "",
  "stdout_log": "",
  "stderr_log": "",
  "checkpoint_path": "",
  "best_val_metric": null
}
```

失败 job 的处理：

- 显存不足：只降低 batch size，并使用 gradient accumulation 保持有效 batch；
- 临时 CUDA/进程错误：原配置重跑一次；
- 确认代码错误：修复同一问题后重跑受影响 job；
- 不因某一方法失败而给其他方法额外训练预算；
- 不静默跳过失败 job；
- 不为修复训练稳定性改变 Stage 5 reward 参数。

---

## 10. 阶段 6 明确不做的事项

本阶段不要做：

- 自动 task graph discovery；
- LLM node proposal；
- 重新标注 Stage 2 GT；
- 重新训练 Stage 4 reward model；
- 根据 policy test success 调整 `lambda/eta/beta`；
- 大范围 policy architecture search；
- 为 PathGraph 方法单独增加训练步数；
- 使用 weighted sampler 作为主结果；
- 将零动作 synthetic trajectories 混入 BC action supervision；
- 把 checkpoint 强制塞入 ZIP；
- 删除前面各轮 ZIP。

**核心点：阶段 6 固定 reward、固定策略协议、统一数据流，只比较样本权重；独立实验尽可能多 GPU 并行，每一轮都立即交付轻量 ZIP。**

<!-- END FILE: 01_阶段6通用执行规范.md -->


---

<!-- BEGIN FILE: 阶段6.1_策略可用图数据与持久输入冻结.md -->

# 阶段 6.1：策略可用图数据与持久输入冻结

## 一、总体上要干什么

本小阶段完成三件直接影响策略训练的工作：

1. 将 Stage 5 的 reward engine、selection lock、模型 bundle 和 checkpoint 从临时位置持久化为 Stage 6 只读输入；
2. 定位现有 CUPID 策略训练、数据集和 rollout 入口，冻结一个最小策略适配层；
3. 采集真正保存了执行动作的 `transport_recovery` 与 `transport_dual_order` demonstrations，构建无 group 泄漏的 policy dataset v1。

阶段 2 的 synthetic graph traces继续保留为 reward diagnostic，不进入 BC action supervision。

本轮出口：

```text
POLICY_DATA_REAL_ACTION_READY
```

本轮 ZIP：

```text
stage6_1_policy_ready_data_and_input_freeze.zip
```

---

## 二、建立轮次目录

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_1_policy_ready_data_and_input_freeze"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_INPUTS"/{reward_v1,model_checkpoints,graph_specs,policy_adapter} \
  "$STAGE6_DATA"/{raw_episodes,manifests,splits,statistics}

"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_before.txt" || {
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/gpu_interactive.txt"
}
```

创建 `run_manifest.md`，然后记录：

```text
stage5 reward input
stage5 selection lock SHA256
stage5 model bundle SHA256
repository commit
policy code root
collection start time
```

---

## 三、小阶段 6.1-A：持久化 Stage 5 reward bundle

### 6.1-A.1 复制冻结的小文件

```bash
rsync -a --delete \
  "$STAGE5_REWARD/configs/" \
  "$STAGE6_INPUTS/reward_v1/configs/"

rsync -a --delete \
  "$STAGE5_REWARD/code/" \
  "$STAGE6_INPUTS/reward_v1/code/"

rsync -a \
  "$STAGE5_REWARD/FROZEN.md" \
  "$STAGE5_REWARD/STAGE5_REWARD_SHA256SUMS.txt" \
  "$STAGE6_INPUTS/reward_v1/"

cp -a \
  "$STAGE5_REWARD/metrics/g2_gate.json" \
  "$STAGE5_REWARD/metrics/frozen_reward_metrics.json" \
  "$STAGE6_INPUTS/reward_v1/"
```

注意：不得编辑复制后的 reward 参数、engine 逻辑和 selection lock。

### 6.1-A.2 实现 checkpoint 持久化脚本

创建：

```text
tools/stage6/persist_reward_bundle.py
```

CLI：

```bash
python tools/stage6/persist_reward_bundle.py \
  --model-bundle "$STAGE5_REWARD/configs/model_bundle.json" \
  --destination "$STAGE6_INPUTS/model_checkpoints" \
  --output-bundle "$STAGE6_INPUTS/reward_v1/configs/model_bundle.stage6.json" \
  --search-root "$REPO_ROOT" \
  --search-root /tmp \
  --manifest "$ROUND_DIR/manifests/checkpoint_manifest.tsv"
```

脚本必须逐 checkpoint 执行：

1. 读取 `seed`、`path`、`sha256`、`history_steps`；
2. 若原路径存在，计算 SHA256；
3. 若原路径不存在或 hash 不符，只按记录的 SHA256 在以下位置搜索：
   ```text
   $REPO_ROOT
   /tmp
   Stage 4 jobs
   Stage 5 refine jobs
   ```
4. 找到同 SHA 文件后，复制或 reflink 到：
   ```text
   $STAGE6_INPUTS/model_checkpoints/seed_<seed>/best.pt
   ```
5. 优先：
   ```bash
   cp --reflink=auto --preserve=timestamps
   ```
6. 复制后再次计算 SHA256；
7. 生成新 bundle，除 `path` 改为持久路径外，其余模型元数据保持不变；
8. 输出 manifest：
   ```text
   seed old_path persistent_path size_bytes expected_sha256 actual_sha256 status
   ```
9. 任一 checkpoint 无法按 SHA 定位时，输出：
   ```text
   CHECKPOINT_PERSISTENCE_BLOCKED
   ```
   并停止本轮后续工作；
10. checkpoint 不进入本轮 ZIP，只在 manifest 中记录。

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/persist_reward_bundle.py" \
  --model-bundle "$STAGE5_REWARD/configs/model_bundle.json" \
  --destination "$STAGE6_INPUTS/model_checkpoints" \
  --output-bundle "$STAGE6_INPUTS/reward_v1/configs/model_bundle.stage6.json" \
  --search-root "$REPO_ROOT" \
  --search-root /tmp \
  --manifest "$ROUND_DIR/manifests/checkpoint_manifest.tsv" \
  2>&1 | tee "$ROUND_DIR/logs/persist_reward_bundle.log"
```

必要完成检查：

```bash
"$PYTHON_BIN" - <<'PYCODE'
import json, os, hashlib
p=os.path.join(os.environ["STAGE6_INPUTS"],"reward_v1/configs/model_bundle.stage6.json")
b=json.load(open(p))
assert len(b["checkpoints"])==3
for x in b["checkpoints"]:
    path=x["path"]
    assert os.path.isfile(path), path
    h=hashlib.sha256(open(path,"rb").read()).hexdigest()
    assert h==x["sha256"], (path,h,x["sha256"])
print("PERSISTENT_MODEL_BUNDLE_OK")
PYCODE
```

### 6.1-A.3 冻结 input checksum

```bash
(
  cd "$STAGE6_INPUTS"
  find reward_v1 -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$ROUND_DIR/checksums/frozen_reward_input_sha256.txt"
```

checkpoint 本体不写入上述小文件 checksum；其 hash 已存在 `checkpoint_manifest.tsv`。

---

## 四、小阶段 6.1-B：定位策略训练与 rollout 入口

### 6.1-B.1 快速定位，不做全仓库审计

执行：

```bash
find "$REPO_CODE_ROOT" -maxdepth 4 -type f \
  \( -name 'train.py' -o -name 'eval_save_episodes.py' -o -name '*workspace*.py' \
     -o -name '*dataset*.py' -o -name '*policy*.py' \) \
  | sort \
  | tee "$ROUND_DIR/tables/policy_entrypoint_candidates.txt"

find "$REPO_CODE_ROOT" -maxdepth 6 -type f \
  \( -name '*transport*.yaml' -o -name '*square*.yaml' \) \
  | sort \
  | tee "$ROUND_DIR/tables/policy_config_candidates.txt"

grep -RIn --include='*.py' \
  -E 'compute_loss|optimizer.step|DataLoader|env.step|save.*episode|action_chunk|horizon' \
  "$REPO_CODE_ROOT" \
  | head -n 400 \
  > "$ROUND_DIR/tables/policy_hook_candidates.txt"
```

优先入口：

```text
$REPO_CODE_ROOT/train.py
$REPO_CODE_ROOT/eval_save_episodes.py
$REPO_CODE_ROOT/diffusion_policy/config/task/transport_lowdim_abs.yaml
```

若这些路径存在，直接使用；若名称不同，只在候选表中选实际对应入口，不重构整个项目。

### 6.1-B.2 创建 policy adapter 描述

创建：

```text
$STAGE6_INPUTS/policy_adapter/policy_adapter.json
```

字段：

```json
{
  "train_entrypoint": "",
  "rollout_entrypoint": "",
  "base_task_config": "",
  "workspace_class": "",
  "dataset_class": "",
  "policy_class": "",
  "loss_hook_file": "",
  "loss_hook_function": "",
  "env_step_file": "",
  "episode_writer_file": "",
  "action_key_in_dataset": "",
  "observation_keys": [],
  "action_dim": 0,
  "action_chunk_horizon": 16,
  "observation_horizon": 2,
  "base_successful_run_config": "",
  "base_successful_checkpoint": ""
}
```

Agent 填写时必须实际打开候选文件并确认。不要把猜测路径写入 adapter。

使用命令记录代码位置：

```bash
nl -ba <train_entrypoint> \
  | sed -n '<关键行范围>p' \
  > "$ROUND_DIR/reports/train_entrypoint_excerpt.txt"

nl -ba <loss_hook_file> \
  | sed -n '<关键行范围>p' \
  > "$ROUND_DIR/reports/loss_hook_excerpt.txt"

nl -ba <episode_writer_file> \
  | sed -n '<关键行范围>p' \
  > "$ROUND_DIR/reports/episode_writer_excerpt.txt"
```

### 6.1-B.3 锁定现有成功策略配置

从已有成功的 transport 训练/rollout 目录选择一个基础配置，仅作为架构和预算基准。优先查找：

```bash
find "$REPO_CODE_ROOT/data/outputs" "$REPO_ROOT/data/outputs" \
  -type f \( -name '.hydra/config.yaml' -o -name 'config.yaml' \) \
  2>/dev/null \
  | grep -E 'transport|diffusion' \
  | sort \
  | tail -n 50 \
  > "$ROUND_DIR/tables/base_policy_run_candidates.txt"
```

选择规则：

1. 与当前动作维度和 observation 类型兼容；
2. 有成功 rollout 或明确 validation 指标；
3. action chunk 与现有系统兼容，优先 16；
4. 不按 Stage 6 test 结果选择。

复制解析后的基础配置到：

```text
$STAGE6_PROTOCOL/base_policy_config_source.yaml
```

并将来源路径写入 adapter。

---

## 五、小阶段 6.1-C：确认现有 graph traces 不可作为动作监督

实现：

```text
tools/stage6/check_action_usability.py
```

CLI：

```bash
python tools/stage6/check_action_usability.py \
  --manifest <episode_manifest.jsonl> \
  --output "$ROUND_DIR/metrics/stage2_action_usability.json" \
  --table "$ROUND_DIR/tables/stage2_action_statistics.csv"
```

脚本对每个 episode 计算：

```text
episode_id
task_id
scenario
num_steps
action_dim
nonzero_action_steps
nonzero_action_step_ratio
action_abs_mean
action_abs_max
action_std
finite_action_rate
controller_source
usable_for_bc
```

判定：

```python
usable_for_bc = (
    finite_action_rate == 1.0
    and nonzero_action_step_ratio >= 0.50
    and action_std >= 1e-4
)
```

执行阶段 2 graph manifest：

```bash
STAGE2_MANIFEST="$(find "$STAGE2_ROOT" -type f \
  -name 'stage2_episode_manifest_v0.2.jsonl' | head -n 1)"

test -n "$STAGE2_MANIFEST"

"$PYTHON_BIN" "$STAGE6_TOOLS/check_action_usability.py" \
  --manifest "$STAGE2_MANIFEST" \
  --output "$ROUND_DIR/metrics/stage2_action_usability.json" \
  --table "$ROUND_DIR/tables/stage2_action_statistics.csv" \
  2>&1 | tee "$ROUND_DIR/logs/check_stage2_actions.log"
```

预期结论写入：

```text
reward_diagnostic_usable = true
bc_action_supervision_usable = false
```

不要删除阶段 2 数据，也不要将其改写为非零动作。

---

## 六、小阶段 6.1-D：实现真实动作 graph rollout collector

### 6.1-D.1 Collector 的唯一关键要求

创建：

```text
tools/stage6/collect_action_graph_rollouts.py
```

每一步必须执行类似：

```python
action = controller.act(obs, scenario_state)
next_obs, env_reward, terminated, truncated, info = env.step(action)
writer.append(
    observation=obs,
    action_commanded=action,
    action_applied=info.get("action_applied", action),
    next_observation=next_obs,
    info=info,
)
```

严禁：

- 先插值 state，再把 `action=[0,...,0]` 填入；
- 只保存 observation，不保存实际动作；
- 保存 controller 的目标位姿却声称是 `env.step` 动作；
- 将 reward diagnostic synthetic episode 混入 policy dataset。

### 6.1-D.2 Episode 存储契约

每个 episode 至少包含：

```json
{
  "episode_id": "",
  "task_id": "transport_recovery|transport_dual_order",
  "scenario": "",
  "seed": 0,
  "initial_state_group_id": "",
  "controller_source": "scripted_env_controller|teleop|expert_policy",
  "source_format": "stage6_action_episode_v1",
  "success": false,
  "outcome": "success|failure|partial",
  "path_signature": ["A", "B"],
  "recovery": false,
  "failure_count": 0,
  "recovery_count": 0,
  "num_steps": 0,
  "steps": [
    {
      "t": 0,
      "observation": {},
      "action_commanded": [],
      "action_applied": [],
      "next_observation": {},
      "env_reward": 0.0,
      "terminated": false,
      "truncated": false,
      "info": {}
    }
  ]
}
```

大型 observation 数组可写入 `.npz`/HDF5/Zarr，manifest 中保留路径和 schema；无需进入 ZIP。

### 6.1-D.3 Graph 场景定义

`transport_dual_order`：

```text
order_A_then_B
order_B_then_A
```

`transport_recovery`：

```text
natural_success
drop_and_regrasp_success
gripper_reopen_success
controlled_failure
```

干预必须由环境 wrapper 在确定语义触发点执行并记录：

```text
intervention_type
intervention_trigger
intervention_step
pre_intervention_state
post_intervention_state
recovery_started_step
recovery_completed_step
```

示例：

- `drop_and_regrasp_success`：物体已被提升后，在固定阈值触发一次受控 drop，随后 expert/controller 重新抓取并完成；
- `gripper_reopen_success`：夹持成立后强制一次短时 reopen，再允许 controller 恢复；
- `controlled_failure`：触发 drop 或错误释放，但不提供恢复动作，作为低质量/失败 demonstration。

场景触发规则必须按 seed 固定，所有后续策略评估沿用同一类规则。

### 6.1-D.4 Collector CLI

```bash
python tools/stage6/collect_action_graph_rollouts.py \
  --task transport_dual_order \
  --scenario order_A_then_B \
  --num-episodes 20 \
  --seed-start 260000 \
  --output-dir "$STAGE6_DATA/raw_episodes/transport_dual_order/order_A_then_B/shard_00" \
  --manifest "$ROUND_DIR/jobs/dual_ab_00/episode_manifest.jsonl" \
  --policy-adapter "$STAGE6_INPUTS/policy_adapter/policy_adapter.json" \
  --device cuda:0
```

必须支持参数：

```text
--task
--scenario
--num-episodes
--seed-start
--output-dir
--manifest
--policy-adapter
--controller-checkpoint       # 若使用 expert policy
--device
--render-video false
--save-observation lowdim|rgb|both
--max-steps
```

默认 `render-video=false`，避免数据采集被视频编码拖慢。只对少量代表性 episode 另行渲染。

---

## 七、小阶段 6.1-E：并行采集

### 6.1-E.1 目标数量

目标数量：

| 任务 | 场景 | 目标 episode |
|---|---|---:|
| transport_dual_order | order_A_then_B | 80 |
| transport_dual_order | order_B_then_A | 80 |
| transport_recovery | natural_success | 60 |
| transport_recovery | drop_and_regrasp_success | 60 |
| transport_recovery | gripper_reopen_success | 60 |
| transport_recovery | controlled_failure | 40 |

这些是目标量，不要求单进程串行完成。

### 6.1-E.2 建立采集 job 表

创建：

```text
$ROUND_DIR/tables/collection_jobs.tsv
```

字段：

```text
job_id task scenario shard num_episodes seed_start device output_dir status
```

建议每个 shard 10—20 个 episode。例如：

```text
dual_ab_00 transport_dual_order order_A_then_B 0 20 260000 cuda:0 ...
dual_ab_01 transport_dual_order order_A_then_B 1 20 260020 cuda:1 ...
dual_ba_00 transport_dual_order order_B_then_A 0 20 261000 cuda:2 ...
recovery_drop_00 transport_recovery drop_and_regrasp_success 0 15 262000 cuda:3 ...
```

### 6.1-E.3 并行策略

- 若 controller 是 GPU expert policy：一个 collector shard 占一张 GPU；
- 若 controller 是纯 scripted/teleop 且仿真主要用 CPU：按 CPU process 并行，不占用无必要 GPU；
- 多 GPU 可用时，不把不同场景串行排队；
- 每个 job 写自己的 episode 目录和 manifest；
- 采集失败只重跑缺失 seed，不重采全部场景。

记录命令：

```bash
cat > "$ROUND_DIR/commands/<job_id>.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$STAGE6_ROOT/stage6_env.sh"
export CUDA_VISIBLE_DEVICES=<gpu_id>
python "$STAGE6_TOOLS/collect_action_graph_rollouts.py" \
  --task <task> \
  --scenario <scenario> \
  --num-episodes <n> \
  --seed-start <seed> \
  --output-dir <output_dir> \
  --manifest <job_manifest> \
  --policy-adapter "$STAGE6_INPUTS/policy_adapter/policy_adapter.json" \
  --device cuda:0 \
  --render-video false
EOF
```

并行启动时将 PID 和 GPU 写入：

```text
$ROUND_DIR/tables/collection_processes.tsv
```

---

## 八、小阶段 6.1-F：合并、过滤并构建 split

### 6.1-F.1 合并 manifest

实现：

```text
tools/stage6/merge_policy_episode_manifests.py
```

执行：

```bash
find "$ROUND_DIR/jobs" -name 'episode_manifest.jsonl' -print0 \
  | sort -z \
  | xargs -0 cat \
  > "$STAGE6_DATA/manifests/all_collected_episodes.jsonl"
```

然后调用合并脚本：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/merge_policy_episode_manifests.py" \
  --inputs "$STAGE6_DATA/manifests/all_collected_episodes.jsonl" \
  --output "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --duplicates "$ROUND_DIR/tables/duplicate_episode_ids.csv" \
  --summary "$ROUND_DIR/metrics/collection_summary.json"
```

只去除完全重复的 episode ID 或内容 hash；不要因轨迹相似而大规模删除数据。

### 6.1-F.2 动作可用性检查

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/check_action_usability.py" \
  --manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --output "$ROUND_DIR/metrics/policy_action_usability.json" \
  --table "$ROUND_DIR/tables/policy_action_statistics.csv" \
  2>&1 | tee "$ROUND_DIR/logs/check_policy_actions.log"
```

额外核对 `action_applied`：

```text
finite rate = 1
action dimension consistent
nonzero action step ratio >= 0.50
action std >= 1e-4
```

如果 action 是绝对位姿且在部分维度恒定，按整体向量统计，不要求每一维都非零。

### 6.1-F.3 构建 group split

实现：

```text
tools/stage6/build_policy_splits.py
```

按：

```text
initial_state_group_id
```

划分：

```text
train 70%
val   15%
test_demo_holdout 15%
```

`test_demo_holdout` 只用于离线 sanity，不替代闭环 test rollout。

CLI：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_policy_splits.py" \
  --manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --group-key initial_state_group_id \
  --stratify task_id,scenario,outcome \
  --ratios 0.70,0.15,0.15 \
  --seed 20260909 \
  --output "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --summary "$ROUND_DIR/metrics/policy_split_summary.json"
```

必要检查：

```bash
"$PYTHON_BIN" - <<'PYCODE'
import pandas as pd, os
p=os.path.join(os.environ["STAGE6_DATA"],"splits/policy_splits_v1.csv")
d=pd.read_csv(p)
assert not d.groupby("initial_state_group_id")["split"].nunique().gt(1).any()
assert set(d["split"])=={"train","val","test_demo_holdout"}
print(d.groupby(["task_id","scenario","split"]).size())
print("POLICY_SPLIT_NO_GROUP_LEAKAGE")
PYCODE
```

---

## 九、小阶段 6.1-G：数据门决策

实现：

```text
tools/stage6/decide_policy_data_gate.py
```

输出：

```text
$ROUND_DIR/metrics/policy_data_gate.json
$ROUND_DIR/reports/policy_data_gate.md
```

门槛：

```text
每个 graph task:
  train episodes >= 60
  val episodes >= 20

transport_dual_order:
  A→B train >= 20
  B→A train >= 20

transport_recovery:
  successful recovery train >= 20

动作:
  finite_action_rate == 1.0
  nonzero_action_step_ratio >= 0.50
  action_std >= 1e-4
  action_dim 与 policy adapter 一致

split:
  initial_state_group_id 不跨 split

来源:
  policy train episode 不得为 stage2 synthetic_lowdim_json
  controller_source 明确记录
```

决策：

```text
POLICY_DATA_REAL_ACTION_READY
COLLECT_MORE_TARGETED
ACTION_LOGGING_BROKEN
```

处理方式：

- `POLICY_DATA_REAL_ACTION_READY`：立即进入 6.2；
- `COLLECT_MORE_TARGETED`：只补缺失场景/数量，不重采已经合格部分；
- `ACTION_LOGGING_BROKEN`：修复 collector 的动作记录后，只重跑受影响 shard。

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/decide_policy_data_gate.py" \
  --manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --splits "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --action-stats "$ROUND_DIR/tables/policy_action_statistics.csv" \
  --config "$STAGE6_CONFIG_ROOT/stage6.yaml" \
  --output "$ROUND_DIR/metrics/policy_data_gate.json" \
  --report "$ROUND_DIR/reports/policy_data_gate.md"
```

检查：

```bash
grep -q 'POLICY_DATA_REAL_ACTION_READY' \
  "$ROUND_DIR/reports/policy_data_gate.md"
```

---

## 十、冻结 policy dataset v1

门通过后创建：

```text
$STAGE6_DATA/FROZEN.md
$STAGE6_DATA/DATASET_SHA256SUMS.txt
$STAGE6_DATA/dataset_card.md
```

`dataset_card.md` 至少写：

```text
任务和场景数量
数据来源
动作定义
action dimension
observation schema
split group key
每个 split 数量
成功/失败/恢复分布
Stage2 synthetic 是否进入 action supervision：false
```

计算 checksum 时对 manifest、split、schema 和 episode 索引计算；大 episode 数据可以只记录每文件 hash 到 manifest，不进入 ZIP。

---

## 十一、本轮交付结构

本轮 ZIP 至少包含：

```text
run_manifest.md
gpu/gpu_before.txt
gpu/gpu_after.txt
configs/stage6.yaml
manifests/checkpoint_manifest.tsv
manifests/policy_episode_manifest_reference.tsv
tables/policy_entrypoint_candidates.txt
tables/policy_hook_candidates.txt
tables/stage2_action_statistics.csv
tables/policy_action_statistics.csv
tables/collection_jobs.tsv
metrics/stage2_action_usability.json
metrics/policy_action_usability.json
metrics/policy_split_summary.json
metrics/policy_data_gate.json
reports/policy_data_gate.md
reports/collection_summary.md
reports/policy_adapter_summary.md
checksums/frozen_reward_input_sha256.txt
checksums/policy_dataset_manifest.sha256
```

原始 episode、checkpoint 和视频不打包。

---

## 十二、生成本轮 ZIP

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_after.txt" || true

export ZIP_NAME="stage6_1_policy_ready_data_and_input_freeze.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 本轮最终回复格式：

```text
阶段 6.1 状态：POLICY_DATA_REAL_ACTION_READY
真实动作数据：<任务/场景/数量摘要>
持久 checkpoint：3/3，hash verified
split：无 initial_state_group_id 泄漏
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6.2
```

**核心点：本轮不训练策略；只把冻结 reward 变成持久输入，并把 graph-task 数据从“奖励诊断轨迹”升级为“真正可用于 BC 的动作 demonstration”。**

<!-- END FILE: 阶段6.1_策略可用图数据与持久输入冻结.md -->


---

<!-- BEGIN FILE: 阶段6.2_全训练集奖励权重与加权数据管线.md -->

# 阶段 6.2：全训练集奖励权重与加权数据管线

## 一、总体上要干什么

本小阶段使用阶段 5 冻结的三 seed reward-model bundle，对阶段 6.1 已冻结的真实动作 policy dataset 的 `train` 和 `val` split 运行真实推理，生成逐 transition 图奖励，再按策略 action chunk 构造训练权重。

随后在现有策略代码中加入最小 weighted-loss 接口，保证四种方法使用完全相同的数据顺序和训练代码，只更换权重表：

```text
bc_all
linear_sarm_equiv
sequential_transition
pathgraph_reward_v1_locked
```

本轮不训练完整策略，只构建权重、数据索引和 loss 接口。

本轮出口：

```text
WEIGHTING_PIPELINE_READY
```

本轮 ZIP：

```text
stage6_2_weighting_pipeline.zip
```

---

## 二、建立轮次目录与入口检查

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_2_weighting_pipeline"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_WEIGHTS"/{predictions,transition_rewards,chunk_weights,index,statistics}

"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_before.txt" || {
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/gpu_interactive.txt"
}

test -f "$STAGE6_DATA/FROZEN.md"
test -f "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl"
test -f "$STAGE6_DATA/splits/policy_splits_v1.csv"
test -f "$STAGE6_INPUTS/reward_v1/configs/model_bundle.stage6.json"
test -f "$STAGE6_INPUTS/reward_v1/configs/reward_selection_lock.json"
test -f "$STAGE6_INPUTS/reward_v1/configs/reward_config_v1.yaml"
test -f "$STAGE6_INPUTS/reward_v1/code/reward_engine.py"
test -f "$STAGE6_INPUTS/policy_adapter/policy_adapter.json"

grep -q 'POLICY_DATA_REAL_ACTION_READY' \
  "$STAGE6_ROUNDS/stage6_1_policy_ready_data_and_input_freeze/reports/policy_data_gate.md"
```

复制本轮冻结配置：

```bash
cp "$STAGE6_CONFIG_ROOT/stage6.yaml" "$ROUND_DIR/configs/stage6.yaml"
cp "$STAGE6_INPUTS/reward_v1/configs/reward_config_v1.yaml" \
  "$ROUND_DIR/configs/reward_config_v1.yaml"
cp "$STAGE6_INPUTS/reward_v1/configs/reward_selection_lock.json" \
  "$ROUND_DIR/configs/reward_selection_lock.json"
```

---

## 三、小阶段 6.2-A：将 policy episode 转换为 reward-model 输入

### 6.2-A.1 实现特征转换器

创建：

```text
tools/stage6/build_reward_inference_dataset.py
```

CLI：

```bash
python tools/stage6/build_reward_inference_dataset.py \
  --episode-manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --splits "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --include-splits train,val \
  --feature-schema "$STAGE6_INPUTS/reward_v1/configs/feature_schema.json" \
  --label-maps "$STAGE6_INPUTS/reward_v1/configs/label_maps.json" \
  --history-steps "$REWARD_HISTORY_STEPS" \
  --output-dir "$STAGE6_WEIGHTS/index/reward_inference_dataset" \
  --output-manifest "$STAGE6_WEIGHTS/index/reward_inference_manifest.jsonl"
```

若 `feature_schema.json` 和 `label_maps.json` 位于 persistent model bundle 指向路径而未复制，阶段 6.1 应已将其复制到：

```text
$STAGE6_INPUTS/reward_v1/configs/
```

特征转换器必须：

1. 按 policy episode 的实际时间顺序读取 observation 和 action；
2. 构造长度为 32 的 causal history；
3. 时间不足 32 步时使用模型阶段 4 相同的 padding 规则；
4. 禁止读取未来 observation；
5. 保留：
   ```text
   episode_id
   content_group_id
   initial_state_group_id
   task_id
   scenario
   split
   t
   source_episode_path
   ```
6. 输出与 Stage 4 `feature_schema` 完全一致的输入 tensor；
7. 记录 feature dimension、dtype、有限值比例；
8. 大 tensor 文件不进入 ZIP，只在 manifest 中记录。

### 6.2-A.2 必要对齐检查

抽取每个任务 2 个 episode，对前 64 步运行特征转换，输出：

```text
$ROUND_DIR/tables/reward_feature_probe.csv
```

字段：

```text
episode_id
task_id
t
feature_dim
history_length
finite
feature_abs_mean
feature_abs_max
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_reward_inference_dataset.py" \
  --episode-manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --splits "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --include-splits train,val \
  --feature-schema "$STAGE6_INPUTS/reward_v1/configs/feature_schema.json" \
  --label-maps "$STAGE6_INPUTS/reward_v1/configs/label_maps.json" \
  --history-steps "$REWARD_HISTORY_STEPS" \
  --output-dir "$STAGE6_WEIGHTS/index/reward_inference_dataset" \
  --output-manifest "$STAGE6_WEIGHTS/index/reward_inference_manifest.jsonl" \
  --probe-output "$ROUND_DIR/tables/reward_feature_probe.csv" \
  2>&1 | tee "$ROUND_DIR/logs/build_reward_inference_dataset.log"
```

必要检查：

```bash
"$PYTHON_BIN" - <<'PYCODE'
import pandas as pd, os
p=os.path.join(os.environ["STAGE6_ROUNDS"],
               "stage6_2_weighting_pipeline/tables/reward_feature_probe.csv")
d=pd.read_csv(p)
assert d["finite"].all()
assert d["feature_dim"].nunique()==1
assert (d["history_length"]<=int(os.environ["REWARD_HISTORY_STEPS"])).all()
print("REWARD_FEATURE_ALIGNMENT_OK")
PYCODE
```

---

## 四、小阶段 6.2-B：按 reward seed 多 GPU 并行推理

### 6.2-B.1 实现单 seed 推理入口

创建：

```text
tools/stage6/infer_reward_on_policy_data.py
```

CLI：

```bash
python tools/stage6/infer_reward_on_policy_data.py \
  --checkpoint <best.pt> \
  --seed <reward_seed> \
  --model-bundle "$STAGE6_INPUTS/reward_v1/configs/model_bundle.stage6.json" \
  --inference-manifest "$STAGE6_WEIGHTS/index/reward_inference_manifest.jsonl" \
  --splits train,val \
  --batch-size 512 \
  --num-workers 4 \
  --device cuda:0 \
  --output "$STAGE6_WEIGHTS/predictions/predictions_s<seed>.jsonl.gz" \
  --metrics "$ROUND_DIR/jobs/reward_s<seed>/metrics/inference_summary.json"
```

输出每步至少包含：

```text
episode_id
content_group_id
task_id
scenario
split
t
node_probs
edge_type_probs
edge_id_probs
phi_pred
remaining_cost_pred
is_terminal
reward_model_seed
```

必须：

- 真实加载 checkpoint；
- 使用 `model.eval()` 和 `torch.inference_mode()`；
- 输出不含 GT 复制逻辑；
- 记录 checkpoint SHA256；
- 检查 NaN/Inf；
- 保持 episode/time 顺序；
- 不覆盖其他 seed 输出。

### 6.2-B.2 生成三个独立 job

```bash
"$PYTHON_BIN" - <<'PYCODE'
import json, os, csv
bundle=json.load(open(os.path.join(
    os.environ["STAGE6_INPUTS"],
    "reward_v1/configs/model_bundle.stage6.json"
)))
out=os.path.join(
    os.environ["STAGE6_ROUNDS"],
    "stage6_2_weighting_pipeline/tables/reward_inference_jobs.tsv"
)
os.makedirs(os.path.dirname(out),exist_ok=True)
with open(out,"w",newline="") as f:
    w=csv.writer(f,delimiter="\t")
    w.writerow(["job_id","seed","checkpoint","output","status"])
    for x in bundle["checkpoints"]:
        seed=x["seed"]
        w.writerow([
            f"reward_s{seed}",
            seed,
            x["path"],
            os.path.join(os.environ["STAGE6_WEIGHTS"],
                         f"predictions/predictions_s{seed}.jsonl.gz"),
            "PENDING",
        ])
print(out)
PYCODE
```

### 6.2-B.3 并行执行

每个 seed 一个 GPU。创建命令文件，例如：

```bash
cat > "$ROUND_DIR/commands/reward_s20260906.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$STAGE6_ROOT/stage6_env.sh"
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$ROUND_DIR/jobs/reward_s20260906/metrics"
python "$STAGE6_TOOLS/infer_reward_on_policy_data.py" \
  --checkpoint "$STAGE6_INPUTS/model_checkpoints/seed_20260906/best.pt" \
  --seed 20260906 \
  --model-bundle "$STAGE6_INPUTS/reward_v1/configs/model_bundle.stage6.json" \
  --inference-manifest "$STAGE6_WEIGHTS/index/reward_inference_manifest.jsonl" \
  --splits train,val \
  --batch-size 512 \
  --num-workers 4 \
  --device cuda:0 \
  --output "$STAGE6_WEIGHTS/predictions/predictions_s20260906.jsonl.gz" \
  --metrics "$ROUND_DIR/jobs/reward_s20260906/metrics/inference_summary.json"
EOF
chmod +x "$ROUND_DIR/commands/reward_s20260906.sh"
```

按可用 GPU 数量同时启动三个 job；显存不足时只减小 `batch-size`，不改变模型和数据。

每个 job 日志：

```text
jobs/reward_s<seed>/stdout.log
jobs/reward_s<seed>/stderr.log
jobs/reward_s<seed>/status.json
```

必要完成条件：

```text
3/3 seed inference SUCCEEDED
所有 prediction finite
所有 episode 均有连续 t
checkpoint SHA256 匹配 persistent bundle
```

---

## 五、小阶段 6.2-C：聚合 ensemble prediction

实现：

```text
tools/stage6/aggregate_policy_reward_predictions.py
```

CLI：

```bash
python tools/stage6/aggregate_policy_reward_predictions.py \
  --inputs \
    "$STAGE6_WEIGHTS/predictions/predictions_s20260906.jsonl.gz" \
    "$STAGE6_WEIGHTS/predictions/predictions_s20260907.jsonl.gz" \
    "$STAGE6_WEIGHTS/predictions/predictions_s20260908.jsonl.gz" \
  --output "$STAGE6_WEIGHTS/predictions/ensemble_policy_predictions.jsonl.gz" \
  --summary "$ROUND_DIR/metrics/ensemble_policy_prediction_summary.json"
```

按以下主键严格 join：

```text
episode_id + t
```

保留：

```text
per_seed_remaining_cost
per_seed_phi
node_probs_mean
edge_type_probs_mean
edge_id_probs_mean
remaining_cost_mean
remaining_cost_std
phi_mean
phi_std
```

任何 seed 缺少 key 时停止，不允许只用剩余 seed 静默继续。

必要检查：

```text
ensemble row count == each seed row count
per_seed array length == 3
all finite
episode split/task/scenario 一致
```

---

## 六、小阶段 6.2-D：使用冻结 reward engine 生成逐 transition 奖励

### 6.2-D.1 复制并导入冻结 engine

不得复制后修改。设置：

```bash
export PYTHONPATH="$STAGE6_INPUTS:${PYTHONPATH:-}"
```

实现：

```text
tools/stage6/score_policy_episodes_with_frozen_reward.py
```

CLI：

```bash
python tools/stage6/score_policy_episodes_with_frozen_reward.py \
  --predictions "$STAGE6_WEIGHTS/predictions/ensemble_policy_predictions.jsonl.gz" \
  --reward-config "$STAGE6_INPUTS/reward_v1/configs/reward_config_v1.yaml" \
  --reward-engine-root "$STAGE6_INPUTS/reward_v1" \
  --output "$STAGE6_WEIGHTS/transition_rewards/pathgraph_reward_v1_locked.jsonl.gz" \
  --summary "$ROUND_DIR/metrics/pathgraph_transition_reward_summary.json"
```

每个 episode 必须：

1. 调用 `PathGraphRewardEngine.new_episode(...)`；
2. 按 `t` 从小到大依次调用 `step(prev,next,state)`；
3. 不跨 episode 共享 `failure_debt` 和 `edge_history`；
4. 输出 Stage 5 weight schema 字段；
5. 额外保留：
   ```text
   split
   scenario
   transition_index
   source_reward_config_sha256
   source_selection_lock_sha256
   ```
6. 只将 `t` 到 `t+1` 组成 transition；
7. terminal 后不继续评分。

冻结方法名必须写：

```text
pathgraph_reward_v1_locked
```

由于 `beta=0`，应满足：

```text
reward_lcb == reward_mu
```

但仍保留 `reward_std` 供分析。

---

## 七、小阶段 6.2-E：构建三种非图权重基线

所有基线使用同一 episode、同一 chunk、同一正值裁剪、同一归一化程序。只更换 transition reward 来源。

### 6.2-E.1 BC-All

逐 chunk：

```text
raw_weight = 1.0
normalized_weight = 1.0
```

### 6.2-E.2 linear_sarm_equiv

优先读取阶段 3 冻结的 `learned_linear_sarm` prediction。若 Stage 3 已有每 transition 线性 progress delta，则直接映射到阶段 6.1 episode 的对应语义进度。

如果无法对新 policy episode 直接运行 Stage 3 learned checkpoint，则实现兼容线性基线：

\[
r_t^{linear} = \max(0, \hat p_{t+1}-\hat p_t)
\]

其中 `p_t` 是固定单链的归一化 stage + within-stage progress。必须在 `linear_baseline_source.json` 中明确：

```text
source = learned_linear_sarm | canonical_linear_progress_adapter
```

不得在结果中将后者误称为重新训练的原 SARM。

### 6.2-E.3 sequential_transition

使用阶段 3 已冻结的 canonical sequential stage 定义：

```text
same stage forward progress -> positive
valid canonical stage transition -> positive
reverse/noncanonical transition -> zero or negative before positive clip
```

`transport_dual_order` 使用阶段 3 已锁定的 canonical order，不根据 Stage 6 test 选择 A-first 或 B-first。

实现统一脚本：

```text
tools/stage6/build_baseline_transition_rewards.py
```

CLI：

```bash
python tools/stage6/build_baseline_transition_rewards.py \
  --method linear_sarm_equiv \
  --episode-manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --splits "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --stage3-root "$STAGE3_ROOT" \
  --output "$STAGE6_WEIGHTS/transition_rewards/linear_sarm_equiv.jsonl.gz" \
  --source-report "$ROUND_DIR/reports/linear_baseline_source.md"

python tools/stage6/build_baseline_transition_rewards.py \
  --method sequential_transition \
  --episode-manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --splits "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --stage3-root "$STAGE3_ROOT" \
  --output "$STAGE6_WEIGHTS/transition_rewards/sequential_transition.jsonl.gz" \
  --source-report "$ROUND_DIR/reports/sequential_baseline_source.md"
```

---

## 八、小阶段 6.2-F：从 transition reward 生成 action-chunk 权重

### 6.2-F.1 Chunk 对齐

策略样本以 chunk 起点 `t_start` 和 horizon 16 标识：

```text
episode_id
task_id
split
t_start
t_end
sample_index
```

默认：

```text
t_end = min(t_start + 15, episode_last_action_step)
```

若现有 dataset 使用不同的 padding/mask 规则，保持原规则，并将有效 action mask 写入 chunk index。

实现：

```text
tools/stage6/build_policy_chunk_index.py
```

CLI：

```bash
python tools/stage6/build_policy_chunk_index.py \
  --episode-manifest "$STAGE6_DATA/manifests/policy_episode_manifest_v1.jsonl" \
  --splits "$STAGE6_DATA/splits/policy_splits_v1.csv" \
  --action-horizon "$POLICY_CHUNK_HORIZON" \
  --policy-adapter "$STAGE6_INPUTS/policy_adapter/policy_adapter.json" \
  --output "$STAGE6_WEIGHTS/index/policy_chunk_index_v1.parquet" \
  --summary "$ROUND_DIR/metrics/policy_chunk_index_summary.json"
```

### 6.2-F.2 原始 chunk return

对于方法 \(m\)：

\[
R_{i}^{(m)}
=
\sum_{t=t_i}^{t_i+H-1}
r_t^{(m)}
\]

正值权重原始量：

\[
u_i^{(m)} = \max(0, R_i^{(m)})
\]

对 `bc_all`：

\[
u_i^{BC}=1
\]

### 6.2-F.3 只用 train split 拟合归一化

每个 task、每个 method 独立使用 train split 计算：

\[
q_{0.99}^{(m,task)}
\]

然后：

\[
\tilde u_i
=
\min\left(
\frac{u_i}{q_{0.99}+\epsilon},
1
\right)
\]

候选压缩指数：

\[
\gamma\in\{1.0,0.75,0.5\}
\]

最终：

\[
w_i
=
\operatorname{clip}\left(
\frac{\tilde u_i^\gamma}
{\mathbb{E}_{train}[\tilde u^\gamma]+\epsilon},
0,
5
\right)
\]

对 val 使用 train 统计，不重新拟合。

选择 `gamma` 的规则按顺序尝试 `1.0 → 0.75 → 0.5`，选择第一个满足：

```text
ESS / N >= 0.25
zero_weight_ratio <= 0.85
recovery_positive_weight_coverage >= 0.50
all weights finite
mean train weight approximately 1
```

其中：

\[
ESS=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}
\]

`gamma` 选择只使用 train 数据及数据覆盖约束，不使用 policy test success。

### 6.2-F.4 生成权重脚本

创建：

```text
tools/stage6/build_chunk_weights.py
```

CLI：

```bash
python tools/stage6/build_chunk_weights.py \
  --method pathgraph_reward_v1_locked \
  --chunk-index "$STAGE6_WEIGHTS/index/policy_chunk_index_v1.parquet" \
  --transition-rewards "$STAGE6_WEIGHTS/transition_rewards/pathgraph_reward_v1_locked.jsonl.gz" \
  --gamma-candidates 1.0,0.75,0.5 \
  --q-clip 0.99 \
  --max-weight 5.0 \
  --min-ess-ratio 0.25 \
  --max-zero-ratio 0.85 \
  --min-recovery-coverage 0.50 \
  --output "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet" \
  --normalization "$STAGE6_WEIGHTS/statistics/pathgraph_reward_v1_locked_normalization.json" \
  --summary "$ROUND_DIR/metrics/pathgraph_weight_summary.json"
```

对其他方法分别执行。BC-All 可以由同一脚本使用：

```bash
--method bc_all --constant-weight 1.0
```

输出权重表字段：

```text
sample_index
episode_id
initial_state_group_id
task_id
scenario
split
t_start
t_end
method
transition_return
raw_positive_weight
gamma
train_q99
normalized_weight
is_recovery_chunk
is_failure_chunk
edge_type_majority
reward_mu_sum
reward_std_mean
source_reward_sha256
```

---

## 九、小阶段 6.2-G：权重分布检查与冻结

### 6.2-G.1 生成统计表

实现：

```text
tools/stage6/summarize_policy_weights.py
```

输出：

```text
$ROUND_DIR/tables/weight_summary_by_task_method.csv
$ROUND_DIR/tables/weight_summary_by_scenario.csv
$ROUND_DIR/tables/recovery_weight_coverage.csv
$ROUND_DIR/figures/weight_hist_<task>_<method>.png
```

核心字段：

```text
N
mean
std
min
p10
p50
p90
p99
max
zero_ratio
ESS
ESS_ratio
recovery_positive_coverage
failure_positive_coverage
```

必要预期：

- `bc_all` 全部为 1；
- `pathgraph_reward_v1_locked` 的有效 recovery chunk 有非零覆盖；
- failure chunk 不应系统性获得高权重；
- 所有方法均 finite；
- 权重平均值经过 task 内归一后接近 1；
- 不要求图权重与线性权重低相关，相关性只记录。

### 6.2-G.2 冻结 normalization lock

创建：

```text
$STAGE6_WEIGHTS/weight_selection_lock.json
```

内容：

```json
{
  "locked": true,
  "selection_source": "train_weight_distribution_only",
  "policy_test_used": false,
  "chunk_horizon": 16,
  "methods": {
    "bc_all": {},
    "linear_sarm_equiv": {},
    "sequential_transition": {},
    "pathgraph_reward_v1_locked": {}
  }
}
```

每个方法记录 `q99`、`gamma`、mean normalization、max weight 和源文件 SHA256。

---

## 十、小阶段 6.2-H：接入 weighted BC 数据与 loss

### 6.2-H.1 原则

主实验不使用 weighted sampler。所有方法用相同 sampler 和 batch 序列，样本权重仅作用于 loss：

\[
\ell_i
=
\operatorname{mean}_{t,d}
\ell(\hat a_{i,t,d},a_{i,t,d})
\]

\[
\mathcal L
=
\frac{
\sum_i w_i\ell_i
}{
\sum_i w_i+\epsilon
}
\]

若 action mask 存在，先在每个样本内部只对有效 action 位置求均值，再做 batch 权重。

### 6.2-H.2 Dataset 修改

优先新增 wrapper，而不是复制整个 dataset：

```text
repo/.../stage6_weighted_dataset.py
```

`__getitem__` 在原 batch 上增加：

```python
batch["sample_index"]
batch["sample_weight"]
batch["weight_method"]
```

权重 join 键必须是原 dataset 可稳定复现的：

```text
sample_index
```

或：

```text
episode_id + t_start
```

启动时一次性检查 join coverage：

```text
train coverage = 100%
val coverage = 100%
duplicate key = 0
```

### 6.2-H.3 Loss 修改

若原 policy `compute_loss` 返回 scalar，需要最小化修改为：

```python
def compute_loss(self, batch, reduction="mean"):
    # 原有前向与 target 不变
    element_loss = ...  # [B, T, D] 或同等结构
    mask = batch.get("action_mask")
    if mask is not None:
        # 每个样本对有效动作位置求均值
        per_item = masked_mean_over_non_batch_dims(element_loss, mask)
    else:
        per_item = element_loss.reshape(element_loss.shape[0], -1).mean(dim=1)

    if reduction == "none":
        return per_item
    return per_item.mean()
```

workspace 中：

```python
loss_per_item = policy.compute_loss(batch, reduction="none")
weights = batch["sample_weight"].to(loss_per_item.device).float()
loss = (loss_per_item * weights).sum() / weights.sum().clamp_min(1e-8)
```

BC-All 也走相同代码，`weights=1`。

### 6.2-H.4 最小正确性检查

创建：

```text
tools/stage6/check_weighted_loss_integration.py
```

执行两个 batch：

1. 全 1 权重：
   ```text
   abs(original_scalar_loss - weighted_loss) <= 1e-6
   ```
2. 非均匀权重：
   ```text
   weighted loss finite
   gradient finite
   weighted loss 与 uniform loss 不完全相同
   ```

命令：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/check_weighted_loss_integration.py" \
  --policy-adapter "$STAGE6_INPUTS/policy_adapter/policy_adapter.json" \
  --base-config "$STAGE6_PROTOCOL/base_policy_config_source.yaml" \
  --dataset "$STAGE6_DATA" \
  --weights "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet" \
  --output "$ROUND_DIR/metrics/weighted_loss_check.json" \
  2>&1 | tee "$ROUND_DIR/logs/weighted_loss_check.log"
```

只做以上必要测试，不增加完整单元测试套件。

---

## 十一、小阶段 6.2-I：权重管线出口决策

实现：

```text
tools/stage6/decide_weighting_pipeline_gate.py
```

门槛：

```text
3/3 reward seed inference completed
ensemble join complete
all transition rewards finite
all four method weight files present
train/val weight join coverage = 100%
BC-All uniform equivalence <= 1e-6
nonuniform weighted gradient finite
PathGraph ESS/N >= 0.25
PathGraph zero ratio <= 0.85
PathGraph recovery positive coverage >= 0.50
weight selection lock exists
policy test not used
```

输出：

```text
WEIGHTING_PIPELINE_READY
TARGETED_WEIGHT_FIX
REWARD_INFERENCE_BLOCKED
```

只允许的定向修正：

- batch size；
- 特征字段映射；
- chunk join key；
- `gamma` 在预先规定的三个候选中顺序回退；
- loss reduction 形状错误。

不允许在本轮改 `lambda/eta/beta`。

---

## 十二、本轮交付结构

ZIP 至少包含：

```text
run_manifest.md
configs/stage6.yaml
configs/reward_config_v1.yaml
configs/reward_selection_lock.json
gpu/gpu_before.txt
gpu/gpu_after.txt
tables/reward_inference_jobs.tsv
tables/reward_feature_probe.csv
tables/weight_summary_by_task_method.csv
tables/weight_summary_by_scenario.csv
tables/recovery_weight_coverage.csv
metrics/ensemble_policy_prediction_summary.json
metrics/pathgraph_transition_reward_summary.json
metrics/pathgraph_weight_summary.json
metrics/weighted_loss_check.json
metrics/weighting_pipeline_gate.json
reports/linear_baseline_source.md
reports/sequential_baseline_source.md
reports/weighting_pipeline_summary.md
manifests/prediction_manifest.tsv
manifests/large_file_manifest.tsv
checksums/weight_files_sha256.txt
```

大型 prediction、Parquet、原始 episode 和 checkpoint 默认不进入 ZIP。

---

## 十三、生成本轮 ZIP

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_after.txt" || true

export ZIP_NAME="stage6_2_weighting_pipeline.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 本轮最终回复：

```text
阶段 6.2 状态：WEIGHTING_PIPELINE_READY
reward inference：3/3 seeds，CUDA=<true>
方法权重：4/4
PathGraph ESS ratio：<value>
PathGraph zero ratio：<value>
Recovery positive coverage：<value>
weighted-loss uniform equivalence：<value>
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6.3
```

**核心点：本轮把冻结 graph reward 转成与 policy chunk 严格对齐的训练权重，并通过同一 weighted-loss 入口保证所有方法只在权重上不同。**

<!-- END FILE: 阶段6.2_全训练集奖励权重与加权数据管线.md -->


---

<!-- BEGIN FILE: 阶段6.3_公平训练协议与并行Smoke.md -->

# 阶段 6.3：公平训练协议锁定与并行 Smoke

## 一、总体上要干什么

本小阶段不追求最终策略性能。总体目标是将策略训练协议一次性锁定，并用最短的并行 smoke run 确认：

- 四种方法都能读取同一 policy dataset；
- 同一 task/seed 使用相同模型初始化与 batch 顺序；
- 唯一实验变量是 sample/chunk weight；
- CUDA 训练、保存 checkpoint、validation 推理和恢复训练均可用；
- 完整训练矩阵可以安全并行启动。

Smoke 不能用于挑选最佳方法，也不能用于调整 Stage 5 reward 参数。

本轮出口：

```text
POLICY_PROTOCOL_LOCKED
```

本轮 ZIP：

```text
stage6_3_protocol_lock_and_smoke.zip
```

---

## 二、建立轮次目录与入口检查

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_3_protocol_lock_and_smoke"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_PROTOCOL"/{configs,seed_registry,initializations,locks,job_templates}

"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_before.txt" || {
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/gpu_interactive.txt"
}

test -f "$STAGE6_WEIGHTS/weight_selection_lock.json"
test -f "$STAGE6_WEIGHTS/chunk_weights/bc_all.parquet"
test -f "$STAGE6_WEIGHTS/chunk_weights/linear_sarm_equiv.parquet"
test -f "$STAGE6_WEIGHTS/chunk_weights/sequential_transition.parquet"
test -f "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet"
test -f "$STAGE6_INPUTS/policy_adapter/policy_adapter.json"

grep -q 'WEIGHTING_PIPELINE_READY' \
  "$STAGE6_ROUNDS/stage6_2_weighting_pipeline/reports/weighting_pipeline_gate.md"
```

---

## 三、小阶段 6.3-A：冻结基础策略架构与训练预算

### 6.3-A.1 从已有成功配置生成 Stage 6 base config

读取：

```text
$STAGE6_PROTOCOL/base_policy_config_source.yaml
```

创建解析后的：

```text
$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml
```

Agent 只允许补齐数据路径、输出路径、seed、method 和 weight path。以下内容必须在四种方法间完全一致：

```text
policy class
backbone
observation keys
action representation
action dimension
observation horizon
action horizon
normalizer
data augmentation
optimizer
learning-rate schedule
batch size
effective batch size
gradient clipping
EMA setting
total optimizer steps
validation frequency
checkpoint frequency
number of dataloader workers
```

若两个 graph task 的 observation/action schema 不同，可有 task-specific base config，但同一 task 内各方法必须完全一致。

### 6.3-A.2 训练预算选择

预算来源优先级：

1. 现有成功 transport policy 的 optimizer steps；
2. 现有同类 diffusion-policy 默认预算；
3. 若原预算明显过大，可统一缩短，但必须在 smoke 前一次性锁定。

禁止：

- PathGraph 单独增加训练步数；
- BC-All 单独提前停止；
- 按 test success 延长某方法训练；
- 用不同 batch size 造成不同有效优化步数。

显存不足时，所有同任务方法统一使用：

```text
smaller micro batch + gradient accumulation
```

保持 effective batch 一致。

### 6.3-A.3 创建训练协议文件

```bash
cat > "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml" <<'YAML'
protocol_version: stage6-policy-v1
tasks:
  - transport_recovery
  - transport_dual_order
methods:
  - bc_all
  - linear_sarm_equiv
  - sequential_transition
  - pathgraph_reward_v1_locked
policy_seeds:
  - 20260909
  - 20260910
  - 20260911
data:
  dataset_version: policy_dataset_v1
  split_train: train
  split_selection: val
  test_for_selection: false
  sampler: identical_unweighted_sampler
  weighted_sampling: false
  action_horizon: 16
optimization:
  total_optimizer_steps: REPLACE_FROM_BASE_CONFIG
  batch_size: REPLACE_FROM_BASE_CONFIG
  gradient_accumulation_steps: REPLACE_FROM_BASE_CONFIG
  effective_batch_size: REPLACE_FROM_BASE_CONFIG
  validation_every_steps: REPLACE_FROM_BASE_CONFIG
  checkpoint_every_steps: REPLACE_FROM_BASE_CONFIG
  checkpoint_selection_metric: val_action_loss
  checkpoint_selection_mode: min
fairness:
  same_architecture_within_task: true
  same_initialization_within_task_seed: true
  same_data_order_within_task_seed: true
  same_optimizer_steps: true
  only_weight_changes: true
smoke:
  optimizer_steps: 2000
  run_all_tasks_methods: true
YAML
```

将 `REPLACE_FROM_BASE_CONFIG` 替换为实际数值并保存来源说明：

```text
$ROUND_DIR/reports/training_budget_source.md
```

---

## 四、小阶段 6.3-B：冻结 seed registry 与相同初始化

### 6.3-B.1 Seed 设计

使用：

```text
20260909
20260910
20260911
```

每个 policy seed 分解为：

```text
model_seed
data_seed
env_validation_seed
augmentation_seed
```

同一 `(task, policy_seed)` 下四种方法完全相同。

创建：

```text
$STAGE6_PROTOCOL/seed_registry/policy_seed_registry.csv
```

字段：

```text
task_id
policy_seed
model_seed
data_seed
augmentation_seed
validation_seed_start
test_seed_registry_id
```

### 6.3-B.2 保存初始化快照

推荐方式：

1. 对每个 `(task, seed)` 创建一次未训练 policy；
2. 保存 `state_dict` 到：
   ```text
   $STAGE6_PROTOCOL/initializations/<task>/seed_<seed>/init.pt
   ```
3. 四种方法从同一 `init.pt` 加载；
4. 记录 SHA256；
5. `init.pt` 不进入 ZIP，只写 manifest。

实现：

```text
tools/stage6/create_policy_initializations.py
```

CLI：

```bash
python tools/stage6/create_policy_initializations.py \
  --base-config "$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml" \
  --tasks transport_recovery,transport_dual_order \
  --seeds "$POLICY_SEEDS" \
  --output-dir "$STAGE6_PROTOCOL/initializations" \
  --manifest "$ROUND_DIR/manifests/policy_initialization_manifest.tsv"
```

必要检查：

```text
同一 task/seed 四种方法引用同一个 init SHA256
不同 seed 的 init SHA256 应不同
init checkpoint 可加载
```

如果现有 workspace 无法单独保存初始 state，则每个 job 启动后在第 0 step 自动保存并比较 hash；只要四方法 hash 一致即可。

---

## 五、小阶段 6.3-C：建立统一训练 wrapper

创建：

```text
tools/stage6/train_weighted_policy.py
```

这是 Stage 6 所有策略训练的唯一入口。CLI：

```bash
python tools/stage6/train_weighted_policy.py \
  --task transport_recovery \
  --method pathgraph_reward_v1_locked \
  --policy-seed 20260909 \
  --base-config "$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml" \
  --protocol "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml" \
  --dataset-root "$STAGE6_DATA" \
  --weight-file "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet" \
  --init-checkpoint "$STAGE6_PROTOCOL/initializations/transport_recovery/seed_20260909/init.pt" \
  --output-dir "$ROUND_DIR/jobs/transport_recovery__pathgraph_reward_v1_locked__s20260909" \
  --max-optimizer-steps 2000 \
  --device cuda:0
```

Wrapper 必须：

1. 读取 policy adapter，调用真实 workspace/policy；
2. 用相同 train split 和 unweighted sampler；
3. 读取指定 method 的 weight table；
4. 将 sample weight送入 weighted loss；
5. 在日志中打印：
   ```text
   task
   method
   policy_seed
   dataset manifest SHA
   weight file SHA
   init checkpoint SHA
   total optimizer steps
   effective batch size
   CUDA device
   ```
6. 保存：
   ```text
   resolved_config.yaml
   train_metrics.jsonl
   val_metrics.jsonl
   checkpoints/latest.pt
   checkpoints/best_val.pt
   status.json
   ```
7. checkpoint selection 只用 `val_action_loss`；
8. 不读取 test rollout；
9. 支持 `--resume-from latest.pt`；
10. 对 OOM 允许减小 micro batch，并自动增大 accumulation 保持 effective batch；
11. 返回非零 exit code 时更新 `status.json=FAILED`。

### 6.3-C.1 DataLoader 的相同顺序

确保：

```python
generator = torch.Generator()
generator.manual_seed(data_seed)
```

所有方法：

```text
shuffle=true
sampler seed 相同
drop_last 相同
num_workers 相同
persistent_workers 相同
```

不要使用 weight-aware sampler。

如果多 worker 引起批次顺序无法严格复现，可统一将 smoke 的 `num_workers=0` 验证；完整训练仍可用固定 worker seed 的多 worker 配置。

---

## 六、小阶段 6.3-D：生成 Smoke job matrix

Smoke 矩阵：

```text
2 tasks × 4 methods × 1 seed = 8 jobs
```

使用 seed：

```text
20260909
```

创建：

```text
$ROUND_DIR/tables/smoke_jobs.tsv
```

字段：

```text
job_id
task_id
method
policy_seed
weight_file
init_checkpoint
output_dir
command_file
preferred_gpu
status
```

实现：

```text
tools/stage6/build_policy_job_matrix.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_policy_job_matrix.py" \
  --tasks transport_recovery,transport_dual_order \
  --methods bc_all,linear_sarm_equiv,sequential_transition,pathgraph_reward_v1_locked \
  --seeds 20260909 \
  --mode smoke \
  --max-optimizer-steps 2000 \
  --protocol "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml" \
  --dataset-root "$STAGE6_DATA" \
  --weights-root "$STAGE6_WEIGHTS/chunk_weights" \
  --initializations-root "$STAGE6_PROTOCOL/initializations" \
  --output "$ROUND_DIR/tables/smoke_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

---

## 七、小阶段 6.3-E：多 GPU 并行 Smoke

### 6.3-E.1 调度原则

- 有 8 张空闲 GPU：8 job 同时启动；
- 有 4 张：两波；
- 有 2 张：四波；
- 不因 GPU 少而改变 job 配置；
- 每个 job 一张 GPU；
- 每个 job 独立输出目录。

实现：

```text
tools/stage6/launch_gpu_job_matrix.py
```

CLI：

```bash
python tools/stage6/launch_gpu_job_matrix.py \
  --job-table "$ROUND_DIR/tables/smoke_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu 1 \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/smoke_job_status.tsv"
```

调度器必须：

1. 先调用 `nvidia-smi` 获取空闲 GPU；
2. 设置每个 job 的 `CUDA_VISIBLE_DEVICES`；
3. 不让两个 job 写同一目录；
4. 记录 PID、GPU、start/end、exit code；
5. 失败 job 不阻塞其他独立 job；
6. 所有 job 完成后返回失败 job 数量；
7. OOM job允许同配置重启一次，只改变 micro batch/accumulation；
8. 不自动改变方法权重或训练步数。

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/smoke_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/smoke_job_status.tsv" \
  2>&1 | tee "$ROUND_DIR/logs/launch_smoke.log"
```

---

## 八、小阶段 6.3-F：Smoke 验收

实现：

```text
tools/stage6/evaluate_smoke_runs.py
```

对每个 job 检查：

```text
CUDA actually used
optimizer steps >= 2000
train loss finite
val loss finite
checkpoint latest exists
checkpoint best_val exists
best checkpoint loadable
init SHA matches task/seed registry
dataset SHA matches
weight SHA matches
sample-weight join coverage = 100%
```

跨方法检查：

```text
同 task/seed:
  init SHA 相同
  data seed 相同
  total optimizer steps 相同
  effective batch size 相同
  batch count 相同
```

方法差异检查：

- `bc_all` 权重均为 1；
- 其他三种方法权重文件 SHA 不必相同；
- 非均匀方法至少一个 batch 的 `weight_std>0`；
- 所有方法实际进入 weighted-loss code path；
- 不比较 smoke 成功率，不据此选择方法。

输出：

```text
$ROUND_DIR/metrics/smoke_gate.json
$ROUND_DIR/tables/smoke_metrics.csv
$ROUND_DIR/reports/smoke_summary.md
```

门状态：

```text
POLICY_PROTOCOL_LOCKED
FIX_TRAINING_ADAPTER
GPU_EXECUTION_BLOCKED
```

`FIX_TRAINING_ADAPTER` 只修复：

```text
batch key
weight join
loss reduction
Hydra override
checkpoint save/load
显存 micro batch
```

修复后重跑受影响 smoke job，不重新生成数据和 reward 权重。

---

## 九、小阶段 6.3-G：生成 protocol selection lock

Smoke 全部通过后创建：

```text
$STAGE6_PROTOCOL/locks/policy_protocol_lock.json
```

字段：

```json
{
  "locked": true,
  "lock_version": "stage6-policy-protocol-v1",
  "selection_source": "existing_successful_policy_config_plus_smoke_feasibility",
  "policy_test_used": false,
  "tasks": [],
  "methods": [],
  "policy_seeds": [],
  "base_config_sha256": "",
  "training_protocol_sha256": "",
  "dataset_manifest_sha256": "",
  "split_sha256": "",
  "weight_selection_lock_sha256": "",
  "initialization_manifest_sha256": "",
  "total_optimizer_steps": 0,
  "effective_batch_size": 0,
  "validation_metric": "val_action_loss",
  "validation_mode": "min",
  "only_weight_changes": true
}
```

同时生成：

```text
$STAGE6_PROTOCOL/FROZEN.md
$STAGE6_PROTOCOL/POLICY_PROTOCOL_SHA256SUMS.txt
```

Protocol lock 后不得：

- 调方法特定学习率；
- 调方法特定训练步数；
- 重新选择 reward gamma；
- 重新选择 Stage 5 reward 参数；
- 看 test rollout 决定 checkpoint。

---

## 十、本轮交付结构

ZIP 至少包含：

```text
run_manifest.md
configs/base_policy_stage6.yaml
configs/policy_training_protocol.yaml
gpu/gpu_before.txt
gpu/gpu_after.txt
tables/smoke_jobs.tsv
tables/smoke_job_status.tsv
tables/smoke_metrics.csv
metrics/smoke_gate.json
reports/training_budget_source.md
reports/smoke_summary.md
manifests/policy_initialization_manifest.tsv
manifests/checkpoint_manifest.tsv
locks/policy_protocol_lock.json
checksums/POLICY_PROTOCOL_SHA256SUMS.txt
```

Smoke checkpoint 和 init checkpoint 不进入 ZIP。

---

## 十一、生成本轮 ZIP

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_after.txt" || true

export ZIP_NAME="stage6_3_protocol_lock_and_smoke.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 本轮最终回复：

```text
阶段 6.3 状态：POLICY_PROTOCOL_LOCKED
Smoke：8/8 jobs
CUDA：<8/8 或实际数量>
同 task/seed init hash：一致
训练预算：<steps>
effective batch：<value>
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6.4
```

**核心点：本轮只把完整训练协议锁定并跑通；Smoke 不挑结果，正式比较从 6.4 的 24 个公平训练 job 开始。**

<!-- END FILE: 阶段6.3_公平训练协议与并行Smoke.md -->


---

<!-- BEGIN FILE: 阶段6.4_多方法多任务多Seed完整策略训练.md -->

# 阶段 6.4：多方法 × 多任务 × 多 Seed 完整策略训练

## 一、总体上要干什么

本小阶段按照已经冻结的 policy protocol 运行完整策略训练：

\[
2\ \text{tasks}
\times
4\ \text{methods}
\times
3\ \text{policy seeds}
=
24\ \text{independent jobs}
\]

四种方法：

```text
bc_all
linear_sarm_equiv
sequential_transition
pathgraph_reward_v1_locked
```

两项任务：

```text
transport_recovery
transport_dual_order
```

三个策略 seed：

```text
20260909
20260910
20260911
```

每个 job 使用同一 task/seed 的相同初始化、相同数据顺序、相同优化步数、相同架构和相同 validation 规则。唯一主要差异是 chunk weight。

本轮不运行最终 test rollout，不根据训练结果修改 reward 参数。

本轮出口：

```text
FULL_POLICY_TRAINING_COMPLETE
```

本轮 ZIP：

```text
stage6_4_full_policy_training.zip
```

---

## 二、建立轮次目录与入口检查

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_4_full_policy_training"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_TRAIN"/{jobs,selection,manifests,metrics}

"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_before.txt" || {
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/gpu_interactive.txt"
}

test -f "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json"
test -f "$STAGE6_PROTOCOL/FROZEN.md"

"$PYTHON_BIN" - <<'PYCODE'
import json, os
p=os.path.join(os.environ["STAGE6_PROTOCOL"],"locks/policy_protocol_lock.json")
d=json.load(open(p))
assert d["locked"] is True
assert d["policy_test_used"] is False
assert d["only_weight_changes"] is True
print("POLICY_PROTOCOL_LOCK_OK")
PYCODE

grep -q 'POLICY_PROTOCOL_LOCKED' \
  "$STAGE6_ROUNDS/stage6_3_protocol_lock_and_smoke/reports/smoke_summary.md"
```

将 protocol lock 和配置复制到本轮：

```bash
cp "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json" \
  "$ROUND_DIR/configs/policy_protocol_lock.json"
cp "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml" \
  "$ROUND_DIR/configs/policy_training_protocol.yaml"
cp "$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml" \
  "$ROUND_DIR/configs/base_policy_stage6.yaml"
```

---

## 三、小阶段 6.4-A：生成 24-job 矩阵

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_policy_job_matrix.py" \
  --tasks transport_recovery,transport_dual_order \
  --methods bc_all,linear_sarm_equiv,sequential_transition,pathgraph_reward_v1_locked \
  --seeds "$POLICY_SEEDS" \
  --mode full \
  --protocol "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml" \
  --dataset-root "$STAGE6_DATA" \
  --weights-root "$STAGE6_WEIGHTS/chunk_weights" \
  --initializations-root "$STAGE6_PROTOCOL/initializations" \
  --output-root "$STAGE6_TRAIN/jobs" \
  --output "$ROUND_DIR/tables/full_training_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

`full_training_jobs.tsv` 必须有 24 行，字段：

```text
job_id
task_id
method
policy_seed
model_seed
data_seed
weight_file
weight_sha256
init_checkpoint
init_sha256
base_config
protocol_lock
total_optimizer_steps
effective_batch_size
output_dir
command_file
status
```

检查：

```bash
"$PYTHON_BIN" - <<'PYCODE'
import pandas as pd, os
p=os.path.join(os.environ["STAGE6_ROUNDS"],
               "stage6_4_full_policy_training/tables/full_training_jobs.tsv")
d=pd.read_csv(p,sep="\t")
assert len(d)==24, len(d)
assert d["task_id"].nunique()==2
assert d["method"].nunique()==4
assert d["policy_seed"].nunique()==3
assert not d.duplicated(["task_id","method","policy_seed"]).any()

for (task,seed),g in d.groupby(["task_id","policy_seed"]):
    assert g["init_sha256"].nunique()==1, (task,seed)
    assert g["data_seed"].nunique()==1, (task,seed)
    assert g["total_optimizer_steps"].nunique()==1, (task,seed)
    assert g["effective_batch_size"].nunique()==1, (task,seed)

print("FULL_JOB_MATRIX_24_OK")
PYCODE
```

---

## 四、小阶段 6.4-B：每个 job 的标准命令

每个 job 的命令文件必须类似：

```bash
#!/usr/bin/env bash
set -euo pipefail

source "$STAGE6_ROOT/stage6_env.sh"

export CUDA_VISIBLE_DEVICES=<assigned_gpu>
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="$OMP_NUM_THREADS"

JOB_ID="transport_recovery__pathgraph_reward_v1_locked__s20260909"
JOB_DIR="$STAGE6_TRAIN/jobs/$JOB_ID"

mkdir -p "$JOB_DIR"/{checkpoints,metrics,logs,manifests}

python "$STAGE6_TOOLS/train_weighted_policy.py" \
  --task transport_recovery \
  --method pathgraph_reward_v1_locked \
  --policy-seed 20260909 \
  --base-config "$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml" \
  --protocol "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml" \
  --protocol-lock "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json" \
  --dataset-root "$STAGE6_DATA" \
  --weight-file "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet" \
  --weight-lock "$STAGE6_WEIGHTS/weight_selection_lock.json" \
  --init-checkpoint "$STAGE6_PROTOCOL/initializations/transport_recovery/seed_20260909/init.pt" \
  --output-dir "$JOB_DIR" \
  --device cuda:0 \
  --resume auto \
  2>&1 | tee "$JOB_DIR/logs/train_console.log"
```

每个 job 开始时必须输出：

```text
torch.cuda.is_available=true
CUDA device name
task
method
policy seed
init SHA
dataset SHA
weight SHA
protocol lock SHA
optimizer step budget
```

---

## 五、小阶段 6.4-C：多 GPU 并行调度

### 6.4-C.1 默认并行方案

采用实验级并行：

```text
一个独立 job 占一张 GPU
```

示例：

- 8 GPU：每波 8 个，共 3 波；
- 6 GPU：每波 6 个，共 4 波；
- 4 GPU：每波 4 个，共 6 波；
- 2 GPU：每波 2 个，共 12 波。

优先将同一 task/seed 的不同方法分配到不同 GPU 同时启动，减少环境时间漂移。例如第一波：

```text
transport_recovery × seed 20260909 × 4 methods
transport_dual_order × seed 20260909 × 4 methods
```

若只有 4 张 GPU，则先跑同一 task/seed 的四方法，再跑另一任务。

### 6.4-C.2 启动命令

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/full_training_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 30 \
  --status-output "$ROUND_DIR/tables/full_training_job_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_full_training.log"
```

调度期间每 10 分钟写：

```text
timestamp
pending
running
succeeded
failed
GPU utilization
GPU memory
```

到：

```text
$ROUND_DIR/logs/training_progress.log
```

### 6.4-C.3 OOM 与中断处理

OOM：

1. 只修改该 task 的统一 micro batch；
2. 增加 gradient accumulation，使 effective batch 不变；
3. 对该 task 的四种方法和所有 seed使用同一修改；
4. 更新 protocol amendment：
   ```text
   $ROUND_DIR/reports/protocol_amendment_oom.md
   ```
5. amendment 必须在任何受影响 job 完成前锁定；
6. 不改变 optimizer steps。

临时中断：

```bash
--resume auto
```

从 `latest.pt` 继续。恢复后总 optimizer step 不得超过预算。

代码 bug：

- 修复后记录 commit/diff；
- 只重跑受影响 job；
- 确认修复不改变某个方法的数学定义；
- 如果修改 loss 共用路径，四方法受影响 job全部重跑。

---

## 六、小阶段 6.4-D：训练过程记录

每个 job 产生：

```text
resolved_config.yaml
status.json
manifests/input_hashes.json
metrics/train_metrics.jsonl
metrics/val_metrics.jsonl
metrics/best_val.json
checkpoints/latest.pt
checkpoints/best_val.pt
logs/train_console.log
```

`train_metrics.jsonl` 每次记录：

```text
optimizer_step
epoch_or_data_pass
train_loss
unweighted_loss_mean
sample_weight_mean
sample_weight_std
sample_weight_min
sample_weight_max
grad_norm
learning_rate
gpu_memory_allocated_mb
wall_time_seconds
```

`val_metrics.jsonl`：

```text
optimizer_step
val_action_loss_weighted
val_action_loss_unweighted
validation_episode_or_chunk_count
```

checkpoint 选择统一使用：

```text
val_action_loss_unweighted
```

原因：各方法的训练权重不同，选择 checkpoint 时使用相同的未加权 validation action loss，避免某方法通过自己的权重定义获得选择优势。

若原 protocol 已锁定为 `val_action_loss`，在 Stage 6.3 应明确其为未加权版本；否则在所有 full jobs 开始前一次性修正 lock，并记录为 protocol clarification，不得中途改变。

---

## 七、小阶段 6.4-E：Validation-only checkpoint 选择

实现：

```text
tools/stage6/select_policy_checkpoints.py
```

CLI：

```bash
python tools/stage6/select_policy_checkpoints.py \
  --job-root "$STAGE6_TRAIN/jobs" \
  --job-table "$ROUND_DIR/tables/full_training_jobs.tsv" \
  --metric val_action_loss_unweighted \
  --mode min \
  --output "$STAGE6_TRAIN/selection/policy_checkpoint_selection.csv" \
  --lock "$STAGE6_TRAIN/selection/policy_checkpoint_selection_lock.json" \
  --manifest "$ROUND_DIR/manifests/checkpoint_manifest.tsv"
```

输出表字段：

```text
task_id
method
policy_seed
selected_step
selected_val_metric
checkpoint_path
checkpoint_size_bytes
checkpoint_sha256
selection_split
test_used
job_status
```

要求：

```text
selection_split = val
test_used = false
```

选择后不复制 checkpoint 到 ZIP；只记录路径和 hash。

### 6.4-E.1 Checkpoint load check

对 24 个 selected checkpoint 逐个做一次 CPU 或对应 GPU load：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/check_selected_policy_checkpoints.py" \
  --selection "$STAGE6_TRAIN/selection/policy_checkpoint_selection.csv" \
  --output "$ROUND_DIR/metrics/selected_checkpoint_load_check.json"
```

只加载并验证模型键/shape，不运行大量推理。

---

## 八、小阶段 6.4-F：训练完整性与公平性汇总

实现：

```text
tools/stage6/summarize_full_policy_training.py
```

输出：

```text
$ROUND_DIR/tables/full_training_summary.csv
$ROUND_DIR/tables/fairness_matrix.csv
$ROUND_DIR/metrics/full_training_gate.json
$ROUND_DIR/reports/full_training_summary.md
$ROUND_DIR/figures/val_loss_curves_<task>.png
```

检查：

```text
24/24 jobs succeeded
24/24 selected checkpoint exists and loads
每个 job 实际 optimizer steps 相同
同 task/seed init SHA 相同
同 task/seed data seed 相同
同 task 所有方法 effective batch 相同
所有 job CUDA used = true
test rollout count = 0
PathGraph reward parameters unchanged
```

训练 loss高低不是本轮 gate，不能因 PathGraph train loss较高就提前否定。

门状态：

```text
FULL_POLICY_TRAINING_COMPLETE
RETRY_FAILED_JOBS
FAIRNESS_PROTOCOL_VIOLATION
```

处理：

- `RETRY_FAILED_JOBS`：只补跑失败 job；
- `FAIRNESS_PROTOCOL_VIOLATION`：确定受影响 task/seed 范围，重跑同范围内四方法；
- 不重跑已公平完成的其他范围。

---

## 九、可选基线的处理

`original_sarm` 与 `arm` 仅在以下条件同时满足时加入：

```text
已有可运行代码
已有兼容权重或可在一天内生成
不需要改变主训练协议
不延迟 24 个必需 job
```

可选基线使用相同 3 seeds 和预算。其失败不阻止本轮主矩阵完成。

不要将可选基线缺失解释为 Stage 6 失败；主结果必须至少包含四个必需方法。

---

## 十、本轮交付结构

ZIP 至少包含：

```text
run_manifest.md
configs/base_policy_stage6.yaml
configs/policy_training_protocol.yaml
configs/policy_protocol_lock.json
gpu/gpu_before.txt
gpu/gpu_after.txt
tables/full_training_jobs.tsv
tables/full_training_job_status.tsv
tables/full_training_summary.csv
tables/fairness_matrix.csv
metrics/full_training_gate.json
metrics/selected_checkpoint_load_check.json
reports/full_training_summary.md
reports/protocol_amendment_oom.md              # 若发生
figures/val_loss_curves_transport_recovery.png
figures/val_loss_curves_transport_dual_order.png
manifests/checkpoint_manifest.tsv
manifests/large_file_manifest.tsv
selection/policy_checkpoint_selection.csv
selection/policy_checkpoint_selection_lock.json
```

checkpoint、W&B 缓存和大型日志默认不打包。日志可保留末尾摘要或压缩后的文本。

---

## 十一、生成本轮 ZIP

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_after.txt" || true

export ZIP_NAME="stage6_4_full_policy_training.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 本轮最终回复：

```text
阶段 6.4 状态：FULL_POLICY_TRAINING_COMPLETE
完整训练：24/24 jobs
CUDA：24/24
任务：2
方法：4
policy seeds：3
selected checkpoints：24/24，validation-only
公平性检查：通过
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6.5
```

**核心点：本轮完成真正的 24-job 策略训练；所有比较只改变权重，test 在 checkpoint 完全锁定后才开始。**

<!-- END FILE: 阶段6.4_多方法多任务多Seed完整策略训练.md -->


---

<!-- BEGIN FILE: 阶段6.5_冻结闭环策略评估.md -->

# 阶段 6.5：冻结闭环策略评估

## 一、总体上要干什么

本小阶段在训练协议和 24 个策略 checkpoint 均已锁定后，运行闭环 test rollout。所有方法使用相同的环境版本、初始状态 seed、干预 seed、最大步数和成功判定。

本轮主要回答：

1. PathGraph 加权是否提高总体任务成功率；
2. 是否提高受控 failure 后的 recovery success；
3. 是否同时支持 A→B 和 B→A 两种合法顺序；
4. 是否改善最差顺序表现；
5. 是否在自然、固定顺序条件下不退化。

本轮不得根据 test 结果重新选择 checkpoint、重新调权重或重新训练某个方法。

本轮出口：

```text
FROZEN_POLICY_EVALUATION_COMPLETE
```

本轮 ZIP：

```text
stage6_5_frozen_policy_evaluation.zip
```

---

## 二、建立轮次目录与锁检查

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_5_frozen_policy_evaluation"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_EVAL"/{jobs,rollouts,metrics,locks,manifests}

"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_before.txt" || {
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/gpu_interactive.txt"
}

test -f "$STAGE6_TRAIN/selection/policy_checkpoint_selection.csv"
test -f "$STAGE6_TRAIN/selection/policy_checkpoint_selection_lock.json"
test -f "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json"
test -f "$STAGE6_WEIGHTS/weight_selection_lock.json"

"$PYTHON_BIN" - <<'PYCODE'
import json, os
paths=[
 os.path.join(os.environ["STAGE6_TRAIN"],"selection/policy_checkpoint_selection_lock.json"),
 os.path.join(os.environ["STAGE6_PROTOCOL"],"locks/policy_protocol_lock.json"),
 os.path.join(os.environ["STAGE6_WEIGHTS"],"weight_selection_lock.json"),
]
for p in paths:
    d=json.load(open(p))
    assert d.get("locked",False) is True, p
print("ALL_STAGE6_SELECTION_LOCKS_OK")
PYCODE
```

将三个 lock 的 SHA256 写入：

```text
$ROUND_DIR/configs/evaluation_input_locks.json
```

锁文件生成后，评估脚本启动前再次比对；评估期间不得更改。
### 2.1 在查看 test 前锁定主要非图比较对象

创建：

```text
tools/stage6/select_non_graph_comparator.py
```

仅使用阶段 6.4 的 `val_action_loss_unweighted`，在：

```text
bc_all
linear_sarm_equiv
sequential_transition
```

中为每个任务选择均值最优的非图 baseline，并生成：

```text
$STAGE6_EVAL/locks/non_graph_comparator_lock.json
```

CLI：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/select_non_graph_comparator.py" \
  --training-summary "$STAGE6_ROUNDS/stage6_4_full_policy_training/tables/full_training_summary.csv" \
  --methods bc_all,linear_sarm_equiv,sequential_transition \
  --metric val_action_loss_unweighted \
  --mode min \
  --output "$STAGE6_EVAL/locks/non_graph_comparator_lock.json"
```

该 lock 必须在读取任何 Stage 6 test rollout 之前生成。阶段 6.6 的 G3 主差值使用这一比较对象，同时仍报告 PathGraph 对全部三个非图方法的结果。

---

## 三、小阶段 6.5-A：冻结评估条件和 seed registry

### 6.5-A.1 评估条件

`transport_recovery`：

```text
natural_no_intervention
drop_and_regrasp
gripper_reopen
```

`transport_dual_order`：

```text
order_A_then_B
order_B_then_A
```

### 6.5-A.2 Rollout 数量

主计划：

```text
每个 task condition × method × policy seed = 50 rollouts
```

总量：

\[
5\ conditions \times 4\ methods \times 3\ policy\ seeds \times 50
=3000\ rollouts
\]

资源明显受限时，允许所有条件统一降到：

```text
30 rollouts
```

不得只降低某一方法或某一困难条件。

### 6.5-A.3 配对 seed

创建：

```text
$STAGE6_EVAL/locks/evaluation_seed_registry.csv
```

字段：

```text
task_id
condition
rollout_index
env_seed
initial_state_seed
intervention_seed
```

同一 `task + condition + rollout_index` 在所有 method 和 policy seed 中使用相同环境 seed。策略本身可以按 policy seed不同，但评估环境必须配对。

实现：

```text
tools/stage6/build_evaluation_seed_registry.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_evaluation_seed_registry.py" \
  --conditions \
    transport_recovery:natural_no_intervention,drop_and_regrasp,gripper_reopen \
    transport_dual_order:order_A_then_B,order_B_then_A \
  --rollouts-per-condition 50 \
  --base-seed 20261000 \
  --output "$STAGE6_EVAL/locks/evaluation_seed_registry.csv"
```

### 6.5-A.4 干预定义冻结

创建：

```text
$STAGE6_EVAL/locks/intervention_protocol_v1.yaml
```

要求：

```yaml
transport_recovery:
  natural_no_intervention:
    intervention: none
  drop_and_regrasp:
    trigger: first_stable_grasp_or_lift_threshold
    action: controlled_object_drop
    max_interventions: 1
    recovery_success: task_success_after_intervention
  gripper_reopen:
    trigger: first_stable_grasp
    action: force_gripper_reopen_for_fixed_steps
    max_interventions: 1
    recovery_success: task_success_after_intervention

transport_dual_order:
  order_A_then_B:
    goal_order: [A, B]
    intervention: none
  order_B_then_A:
    goal_order: [B, A]
    intervention: none
```

具体阈值沿用阶段 6.1 collector 的场景定义；如环境字段名不同，只做字段映射，不修改语义。

---

## 四、小阶段 6.5-B：实现统一闭环评估入口

创建：

```text
tools/stage6/evaluate_policy_checkpoint.py
```

CLI：

```bash
python tools/stage6/evaluate_policy_checkpoint.py \
  --task transport_recovery \
  --condition drop_and_regrasp \
  --method pathgraph_reward_v1_locked \
  --policy-seed 20260909 \
  --checkpoint <selected_best_val.pt> \
  --seed-registry "$STAGE6_EVAL/locks/evaluation_seed_registry.csv" \
  --intervention-protocol "$STAGE6_EVAL/locks/intervention_protocol_v1.yaml" \
  --rollout-start 0 \
  --rollout-count 10 \
  --max-steps <frozen_max_steps> \
  --device cuda:0 \
  --output-dir "$STAGE6_EVAL/jobs/<job_id>" \
  --save-video false
```

脚本必须：

1. 校验 checkpoint SHA256 与 selection manifest；
2. 加载与训练相同的 observation/action normalizer；
3. `policy.eval()`，使用冻结推理设置；
4. 按 seed registry reset 环境；
5. 在指定 semantic trigger 触发 intervention；
6. 每个 rollout 记录：
   ```text
   task_id
   condition
   method
   policy_seed
   rollout_index
   env_seed
   checkpoint_sha256
   success
   completion_fraction
   first_subgoal_completed
   second_subgoal_completed
   achieved_order
   intervention_triggered
   intervention_step
   recovery_started
   recovery_success
   failure_count
   recovery_count
   episode_length
   termination_reason
   ```
7. 将逐步轨迹保存到大文件目录，可不进 ZIP；
8. 默认不保存视频；
9. 只对后续选出的少量代表性案例补视频；
10. 不能读取 training/val success来改变评估动作。

---

## 五、小阶段 6.5-C：生成评估 shard job 矩阵

建议每个 shard：

```text
10 rollouts
```

50 rollout 时，每个 checkpoint-condition 产生 5 个 shard。

基础组合：

```text
24 selected checkpoints
```

每个 checkpoint只评估其任务的相应条件：

- recovery task：3 conditions；
- dual-order task：2 conditions。

总 checkpoint-condition 数：

\[
(12 \times 3) + (12 \times 2) = 60
\]

每个 5 shard：

\[
60 \times 5 = 300\ jobs
\]

创建：

```text
$ROUND_DIR/tables/evaluation_jobs.tsv
```

字段：

```text
job_id
task_id
condition
method
policy_seed
checkpoint
checkpoint_sha256
rollout_start
rollout_count
output_dir
command_file
preferred_gpu
status
```

实现：

```text
tools/stage6/build_evaluation_job_matrix.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_evaluation_job_matrix.py" \
  --checkpoint-selection "$STAGE6_TRAIN/selection/policy_checkpoint_selection.csv" \
  --seed-registry "$STAGE6_EVAL/locks/evaluation_seed_registry.csv" \
  --intervention-protocol "$STAGE6_EVAL/locks/intervention_protocol_v1.yaml" \
  --shard-size 10 \
  --output "$ROUND_DIR/tables/evaluation_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands" \
  --job-root "$STAGE6_EVAL/jobs"
```

必要检查：

```bash
"$PYTHON_BIN" - <<'PYCODE'
import pandas as pd, os
p=os.path.join(os.environ["STAGE6_ROUNDS"],
               "stage6_5_frozen_policy_evaluation/tables/evaluation_jobs.tsv")
d=pd.read_csv(p,sep="\t")
assert not d.duplicated(["task_id","condition","method","policy_seed","rollout_start"]).any()
assert (d["rollout_count"]>0).all()
print(d.groupby(["task_id","condition"])["rollout_count"].sum())
print("EVALUATION_JOB_MATRIX_OK")
PYCODE
```

---

## 六、小阶段 6.5-D：多 GPU 并行运行评估

独立 shard 默认并行。执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/evaluation_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/evaluation_job_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_evaluation.log"
```

评估并行规则：

- 每张 GPU 一个评估 shard；
- shard 的 rollout seed范围不重叠；
- 相同环境 seed跨方法配对；
- 多个 job 不写同一结果文件；
- 失败 shard只重跑该 shard；
- 环境初始化错误可重跑一次；
- 不能因某方法较慢而减少其 rollout 数；
- 不做 test-time reward shaping；Stage 5 reward只用于训练权重，不作为在线控制器。

若评估主要受 CPU 仿真限制，可将每 GPU 同时挂多个 CPU env，但必须先确认显存和环境进程稳定；默认仍为一个 job/GPU。

---

## 七、小阶段 6.5-E：合并 shard 结果

实现：

```text
tools/stage6/merge_policy_evaluation_shards.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/merge_policy_evaluation_shards.py" \
  --job-table "$ROUND_DIR/tables/evaluation_jobs.tsv" \
  --job-root "$STAGE6_EVAL/jobs" \
  --output "$STAGE6_EVAL/rollouts/all_policy_rollouts.csv" \
  --summary "$ROUND_DIR/metrics/evaluation_merge_summary.json" \
  --duplicates "$ROUND_DIR/tables/evaluation_duplicate_keys.csv"
```

唯一键：

```text
task_id + condition + method + policy_seed + rollout_index
```

必须检查：

```text
duplicate keys = 0
missing rollout keys = 0
checkpoint SHA matches
seed registry matches
all required conditions present
all methods same rollout count
```

---

## 八、小阶段 6.5-F：计算冻结策略指标

实现：

```text
tools/stage6/compute_policy_evaluation_metrics.py
```

输出：

```text
$STAGE6_EVAL/metrics/policy_metrics_by_seed.csv
$STAGE6_EVAL/metrics/policy_metrics_aggregate.csv
$ROUND_DIR/tables/policy_metrics_by_condition.csv
$ROUND_DIR/tables/policy_metrics_by_seed.csv
$ROUND_DIR/figures/success_by_condition.png
$ROUND_DIR/figures/recovery_success.png
$ROUND_DIR/figures/dual_order_success.png
$ROUND_DIR/reports/frozen_policy_evaluation_summary.md
```

### 8.1 基础指标

任务成功率：

\[
\text{SuccessRate}
=
\frac{\#\text{successful rollouts}}{\#\text{rollouts}}
\]

长时序完成度：

\[
\text{Completion}
=
\frac{\#\text{completed semantic goals}}
{\#\text{required semantic goals}}
\]

### 8.2 Recovery 指标

只在 intervention 实际触发的 rollout 上计算：

\[
\text{RecoverySuccess}
=
\frac{
\#(\text{intervention triggered and final success})
}{
\#(\text{intervention triggered})
}
\]

另报告：

```text
intervention_trigger_rate
post_failure_progress_rate
median_steps_to_recovery
```

若某策略从未达到干预触发点，不能将其 recovery success记为 1；应同时报告 overall condition success 和 trigger-conditional recovery。

### 8.3 双顺序指标

分别报告：

```text
success_A_then_B
success_B_then_A
worst_order_success = min(success_A_then_B, success_B_then_A)
order_gap = abs(success_A_then_B - success_B_then_A)
```

### 8.4 固定顺序不退化

固定顺序/自然条件为：

```text
transport_recovery:natural_no_intervention
transport_dual_order:order_A_then_B
transport_dual_order:order_B_then_A
```

报告 PathGraph 相对最强非图 baseline 的差值；最终最强 baseline 在阶段 6.6 用 validation 规则确定，本轮先保留全方法原始结果。

---

## 九、小阶段 6.5-G：冻结评估输出

创建：

```text
$STAGE6_EVAL/locks/policy_evaluation_lock.json
```

字段：

```json
{
  "locked": true,
  "checkpoint_selection_lock_sha256": "",
  "policy_protocol_lock_sha256": "",
  "weight_selection_lock_sha256": "",
  "evaluation_seed_registry_sha256": "",
  "intervention_protocol_sha256": "",
  "rollout_count_per_condition": 50,
  "policy_test_used_for_training": false,
  "reward_retuned_after_test": false,
  "result_table_sha256": ""
}
```

同时创建：

```text
$STAGE6_EVAL/FROZEN.md
$STAGE6_EVAL/POLICY_EVAL_SHA256SUMS.txt
```

评估结果冻结后，阶段 6.6 只能分析，不允许改训练或 checkpoint。

---

## 十、本轮验收

实现：

```text
tools/stage6/decide_policy_evaluation_gate.py
```

门槛：

```text
all required selected checkpoints evaluated
all required conditions evaluated
per condition rollout count >= frozen count
same paired env seeds across methods
no duplicate rollout keys
no missing rollout keys
all success/completion values finite and within [0,1]
checkpoint selection remained validation-only
no reward retuning after test
```

状态：

```text
FROZEN_POLICY_EVALUATION_COMPLETE
RETRY_MISSING_SHARDS
EVALUATION_PROTOCOL_VIOLATION
```

缺失 shard只补跑；协议违规时只重跑受影响的配对范围。

---

## 十一、本轮交付结构

ZIP 至少包含：

```text
run_manifest.md
configs/evaluation_input_locks.json
gpu/gpu_before.txt
gpu/gpu_after.txt
locks/evaluation_seed_registry.csv
locks/intervention_protocol_v1.yaml
locks/policy_evaluation_lock.json
tables/evaluation_jobs.tsv
tables/evaluation_job_status.tsv
tables/policy_metrics_by_condition.csv
tables/policy_metrics_by_seed.csv
metrics/evaluation_merge_summary.json
metrics/policy_metrics_aggregate.csv
reports/frozen_policy_evaluation_summary.md
figures/success_by_condition.png
figures/recovery_success.png
figures/dual_order_success.png
manifests/checkpoint_manifest.tsv
manifests/rollout_large_file_manifest.tsv
checksums/POLICY_EVAL_SHA256SUMS.txt
```

逐步轨迹、原始 rollout、视频和 checkpoint 默认不打包。

---

## 十二、生成本轮 ZIP

```bash
"$STAGE6_TOOLS/query_gpus.sh" "$ROUND_DIR/gpu/gpu_after.txt" || true

export ZIP_NAME="stage6_5_frozen_policy_evaluation.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 本轮最终回复：

```text
阶段 6.5 状态：FROZEN_POLICY_EVALUATION_COMPLETE
selected checkpoints：24
评估条件：5
rollouts：<总数>
paired seed：通过
缺失/重复：0/0
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6.6
```

**核心点：本轮只运行已经锁定的策略和 test seeds；不回头调训练，确保策略成功率、恢复率和双顺序结果可作为最终证据。**

<!-- END FILE: 阶段6.5_冻结闭环策略评估.md -->


---

<!-- BEGIN FILE: 阶段6.6_机制统计与失败案例分析.md -->

# 阶段 6.6：机制统计、置信区间与失败案例分析

## 一、总体上要干什么

本小阶段只分析阶段 6.5 已冻结的闭环结果，不重新训练策略、不改变 checkpoint、不调整 Stage 5 reward 参数。

需要完成：

1. 按阶段 6.5 事先锁定的最强非图 comparator 计算主要效果；
2. 使用 policy seed 与 paired rollout seed 的层级 bootstrap 计算 95% 置信区间；
3. 分析 graph weight 对 recovery、alternative order、failure 和普通 forward chunk 的覆盖；
4. 选取少量代表性成功/失败轨迹解释结果；
5. 形成阶段 6 的最终证据表，供阶段 6.7 作 G3 决策。

本轮出口：

```text
STAGE6_EVIDENCE_READY
```

本轮 ZIP：

```text
stage6_6_mechanism_and_statistics.zip
```

---

## 二、建立轮次目录与冻结输入检查

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_6_mechanism_and_statistics"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_FINAL"/{tables,metrics,figures,reports,locks}

test -f "$STAGE6_EVAL/FROZEN.md"
test -f "$STAGE6_EVAL/locks/policy_evaluation_lock.json"
test -f "$STAGE6_EVAL/locks/non_graph_comparator_lock.json"
test -f "$STAGE6_EVAL/metrics/policy_metrics_by_seed.csv"
test -f "$STAGE6_EVAL/rollouts/all_policy_rollouts.csv"
test -f "$STAGE6_WEIGHTS/weight_selection_lock.json"

"$PYTHON_BIN" - <<'PYCODE'
import json, os
for rel in [
 "locks/policy_evaluation_lock.json",
 "locks/non_graph_comparator_lock.json",
]:
    p=os.path.join(os.environ["STAGE6_EVAL"],rel)
    d=json.load(open(p))
    assert d.get("locked",False) is True, p
print("FROZEN_EVALUATION_INPUT_OK")
PYCODE
```

本轮通常为 CPU 分析，不需要占用 GPU。若要对少量轨迹补模型推理或渲染视频，先按固定规范提权查询 GPU，并将补推理作为独立 job 并行执行。

---

## 三、小阶段 6.6-A：形成全方法结果表

实现：

```text
tools/stage6/build_policy_result_tables.py
```

输入：

```text
$STAGE6_EVAL/rollouts/all_policy_rollouts.csv
$STAGE6_EVAL/locks/non_graph_comparator_lock.json
```

输出：

```text
$ROUND_DIR/tables/result_by_task_condition_method_seed.csv
$ROUND_DIR/tables/result_by_task_method.csv
$ROUND_DIR/tables/pathgraph_vs_each_baseline.csv
$ROUND_DIR/tables/pathgraph_vs_locked_comparator.csv
```

主表必须含：

```text
task_id
condition
method
policy_seed
n_rollouts
success_rate
completion_mean
recovery_trigger_rate
recovery_success_rate
median_steps_to_recovery
success_A_then_B
success_B_then_A
worst_order_success
order_gap
```

所有比例必须同时保留分子和分母，避免仅报告百分比。

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_policy_result_tables.py" \
  --rollouts "$STAGE6_EVAL/rollouts/all_policy_rollouts.csv" \
  --comparator-lock "$STAGE6_EVAL/locks/non_graph_comparator_lock.json" \
  --output-dir "$ROUND_DIR/tables" \
  --summary "$ROUND_DIR/reports/policy_result_summary.md"
```

---

## 四、小阶段 6.6-B：层级配对 Bootstrap

### 6.6-B.1 统计单位

第一层：

```text
policy_seed
```

第二层：

```text
paired rollout_index / env_seed
```

同一 bootstrap replicate：

1. 有放回抽样 policy seed；
2. 在每个抽到的 seed 内，对配对 rollout key有放回抽样；
3. 对 PathGraph 与 comparator 使用相同抽样 key；
4. 计算差值。

不要将 3000 个 rollout 全部当作完全独立样本。

### 6.6-B.2 实现脚本

创建：

```text
tools/stage6/hierarchical_paired_bootstrap.py
```

CLI：

```bash
python tools/stage6/hierarchical_paired_bootstrap.py \
  --rollouts "$STAGE6_EVAL/rollouts/all_policy_rollouts.csv" \
  --method-a pathgraph_reward_v1_locked \
  --comparator-lock "$STAGE6_EVAL/locks/non_graph_comparator_lock.json" \
  --metrics success,completion,recovery_success,worst_order_success,order_gap \
  --resamples 5000 \
  --seed 20261001 \
  --output "$ROUND_DIR/tables/bootstrap_effects.csv" \
  --distribution-dir "$ROUND_DIR/metrics/bootstrap_distributions"
```

输出字段：

```text
task_id
condition_or_aggregate
metric
method_a
method_b
point_estimate_a
point_estimate_b
difference
ci95_low
ci95_high
prob_difference_gt_0
resamples
```

主要效果使用绝对百分点差：

```text
+0.05 = +5 percentage points
```

### 6.6-B.3 预先规定的聚合

计算：

1. `graph_task_success`  
   两个 graph task 全部条件按每条件等权平均；
2. `recovery_success`  
   `drop_and_regrasp` 与 `gripper_reopen` 等权平均；
3. `worst_order_success`  
   每 seed 先取 A→B、B→A 的较小值，再跨 seed 聚合；
4. `fixed_order_success`  
   `natural_no_intervention`、A→B、B→A 等权平均；
5. `long_horizon_completion`  
   所有 graph conditions 等权平均。

不要按 rollout 数量让某一条件因 shard 数更多而获得更高权重。

---

## 五、小阶段 6.6-C：逐 Seed 一致性

生成：

```text
$ROUND_DIR/tables/seed_level_effects.csv
```

字段：

```text
task_id
metric
policy_seed
pathgraph_value
comparator_value
difference
improved
```

G3 使用：

```text
至少 2/3 policy seeds 在 graph_task_success 上改善
```

如果某 seed 完全失败，保留并报告；不得删除异常 seed，除非其 job 明确违反协议且已在阶段 6.5 配对重跑。

---

## 六、小阶段 6.6-D：权重机制分析

目标不是证明因果，只用于确认策略提升是否与预期数据重加权方向一致。

实现：

```text
tools/stage6/analyze_weight_mechanism.py
```

输入：

```text
四个 method 的 chunk weight
policy dataset scenario/path/recovery 标签
Stage 5 transition reward components
```

输出：

```text
$ROUND_DIR/tables/weight_by_semantic_type.csv
$ROUND_DIR/tables/top_weighted_chunks.csv
$ROUND_DIR/tables/downweighted_failure_chunks.csv
$ROUND_DIR/tables/recovery_chunk_retention.csv
$ROUND_DIR/tables/alternative_order_balance.csv
$ROUND_DIR/figures/weight_by_semantic_type.png
$ROUND_DIR/figures/recovery_weight_ecdf.png
$ROUND_DIR/figures/order_weight_balance.png
```

语义类别：

```text
forward_normal
alternative_order_A_then_B
alternative_order_B_then_A
failure_onset
recovery
stagnation
terminal_success
```

至少报告：

```text
chunk count
mean weight
median weight
p90 weight
positive-weight rate
total normalized weight mass
```

关键机制指标：

### Recovery 片段保留率

\[
\text{RecoveryRetention}
=
\frac{
\#\text{recovery chunks with }w>0
}{
\#\text{recovery chunks}
}
\]

### 失败片段抑制率

\[
\text{FailureSuppression}
=
\frac{
\#\text{failure chunks with }w \le q_{25}^{all}
}{
\#\text{failure chunks}
}
\]

### 双路径权重平衡

\[
\text{OrderWeightGap}
=
\frac{
|\bar w_{A\rightarrow B}-\bar w_{B\rightarrow A}|
}{
\max(|\bar w_{A\rightarrow B}|,|\bar w_{B\rightarrow A}|,\epsilon)
}
\]

比较 PathGraph 与 linear/sequential，但不根据这些分析重新调权重。

---

## 七、小阶段 6.6-E：训练行为与权重关系

从每个 selected checkpoint 的训练日志提取：

```text
best validation action loss
convergence step
mean gradient norm
weight mean/std
recovery chunk batch frequency
```

生成：

```text
$ROUND_DIR/tables/training_behavior_by_method.csv
$ROUND_DIR/figures/validation_loss_by_method.png
```

只作描述性分析，不把较低训练 loss当作最终成功证据。

检查是否存在：

```text
PathGraph 权重过稀导致有效 batch 崩溃
某一方法梯度明显爆炸
某一方法实际训练步数不足
```

这些问题如果已经通过阶段 6.4 gate，阶段 6.6 只记录，不回头改训练。

---

## 八、小阶段 6.6-F：代表性轨迹选择

只选择少量可解释案例，不进行大规模人工审计。

每个类别最多 3 个：

```text
PathGraph 成功而 comparator 失败的 recovery 案例
PathGraph 成功而 comparator 失败的 B→A 案例
两者都失败的共同难例
PathGraph 失败而 comparator 成功的反例
```

选择规则固定为：

1. 基于已冻结结果；
2. 在符合类别的 rollout 中按 `rollout_index` 排序；
3. 取最前 3 个；
4. 不人工挑最漂亮的视频。

创建：

```text
$ROUND_DIR/tables/representative_rollouts.csv
$ROUND_DIR/reports/representative_case_notes.md
```

如果需要视频，调用统一渲染脚本：

```bash
python tools/stage6/render_selected_rollouts.py \
  --selection "$ROUND_DIR/tables/representative_rollouts.csv" \
  --rollout-root "$STAGE6_EVAL/rollouts" \
  --output-dir "$ROUND_DIR/figures/representative_videos"
```

视频默认不进入 ZIP，只在 `large_file_manifest.tsv` 中记录。ZIP 可放静态 contact sheet。

---

## 九、小阶段 6.6-G：形成最终证据表

实现：

```text
tools/stage6/build_stage6_evidence_table.py
```

输出：

```text
$STAGE6_FINAL/tables/stage6_primary_results.csv
$STAGE6_FINAL/tables/stage6_seed_results.csv
$STAGE6_FINAL/tables/stage6_mechanism_results.csv
$STAGE6_FINAL/metrics/stage6_evidence.json
$STAGE6_FINAL/reports/stage6_evidence_summary.md
```

`stage6_evidence.json` 至少包含：

```json
{
  "locked_comparator_by_task": {},
  "graph_task_success_gain": 0.0,
  "graph_task_success_ci95": [0.0, 0.0],
  "improved_policy_seed_count": 0,
  "recovery_success_gain": 0.0,
  "recovery_success_ci95": [0.0, 0.0],
  "worst_order_success_gain": 0.0,
  "worst_order_success_ci95": [0.0, 0.0],
  "long_horizon_completion_gain": 0.0,
  "fixed_order_drop": 0.0,
  "order_gap_pathgraph": 0.0,
  "order_gap_comparator": 0.0,
  "policy_protocol_locked": true,
  "reward_retuned_after_test": false
}
```

所有值来自冻结 rollout 与 bootstrap 表，禁止手工填写固定结果。

---

## 十、本轮出口门

实现：

```text
tools/stage6/decide_stage6_evidence_gate.py
```

本轮只检查证据完整性，不作最终 G3：

```text
result tables complete
5000 bootstrap resamples complete
paired keys preserved
3 policy seeds present
all required metrics finite
comparator lock used
mechanism tables complete
no post-test retuning
```

状态：

```text
STAGE6_EVIDENCE_READY
RECOMPUTE_STATISTICS
FROZEN_RESULT_INCOMPLETE
```

如果统计脚本出错，只重算统计，不重跑策略和 rollout。

---

## 十一、本轮交付结构

ZIP 至少包含：

```text
run_manifest.md
configs/evaluation_input_locks.json
tables/result_by_task_condition_method_seed.csv
tables/result_by_task_method.csv
tables/pathgraph_vs_each_baseline.csv
tables/pathgraph_vs_locked_comparator.csv
tables/bootstrap_effects.csv
tables/seed_level_effects.csv
tables/weight_by_semantic_type.csv
tables/recovery_chunk_retention.csv
tables/alternative_order_balance.csv
tables/representative_rollouts.csv
metrics/stage6_evidence.json
reports/policy_result_summary.md
reports/representative_case_notes.md
reports/stage6_evidence_summary.md
figures/success_by_condition.png
figures/weight_by_semantic_type.png
figures/recovery_weight_ecdf.png
figures/order_weight_balance.png
figures/representative_contact_sheet.png
manifests/large_file_manifest.tsv
checksums/statistics_outputs_sha256.txt
```

Bootstrap 全分布如果过大，可不进入 ZIP；保留汇总表、seed 和脚本版本。

---

## 十二、生成本轮 ZIP

```bash
export ZIP_NAME="stage6_6_mechanism_and_statistics.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 本轮最终回复：

```text
阶段 6.6 状态：STAGE6_EVIDENCE_READY
锁定 comparator：<task -> method>
graph-task success gain：<value> [95% CI]
recovery gain：<value> [95% CI]
worst-order gain：<value> [95% CI]
improved seeds：<n>/3
fixed-order drop：<value>
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6.7
```

**核心点：本轮不再改变实验，只把冻结结果转换为可用于 G3 的主效果、逐 seed 一致性、置信区间和机制证据。**

<!-- END FILE: 阶段6.6_机制统计与失败案例分析.md -->


---

<!-- BEGIN FILE: 阶段6.7_G3决策与M4冻结.md -->

# 阶段 6.7：G3 决策、M4 冻结与 Stage 7 交接

## 一、总体上要干什么

本小阶段读取阶段 6.6 已冻结的证据，按预先规定的门槛作出 G3 决策。不得人工挑选有利任务、删除不利 seed 或修改指标。

可能状态：

```text
GO_STAGE7
REFINE_STAGE6
NARROW_TO_REWARD_ONLY
POLICY_DATA_BLOCKED
```

若 `GO_STAGE7`，冻结 M4：策略下游证据就绪，并生成 Stage 7 的消融、规模扩展与自动图探索交接文件。

本轮 ZIP：

```text
stage6_7_g3_m4_freeze.zip
```

阶段总 ZIP：

```text
stage6_complete.zip
```

---

## 二、建立轮次目录与入口检查

```bash
source "$STAGE6_ROOT/stage6_env.sh"
cd "$REPO_ROOT"

export ROUND_NAME="stage6_7_g3_m4_freeze"
export ROUND_DIR="$STAGE6_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6_FINAL"/{configs,tables,metrics,figures,reports,locks,manifests}

test -f "$STAGE6_FINAL/metrics/stage6_evidence.json"
test -f "$STAGE6_FINAL/tables/stage6_primary_results.csv"
test -f "$STAGE6_FINAL/tables/stage6_seed_results.csv"
test -f "$STAGE6_EVAL/locks/policy_evaluation_lock.json"
test -f "$STAGE6_EVAL/locks/non_graph_comparator_lock.json"
test -f "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json"
test -f "$STAGE6_WEIGHTS/weight_selection_lock.json"

grep -q 'STAGE6_EVIDENCE_READY' \
  "$STAGE6_ROUNDS/stage6_6_mechanism_and_statistics/reports/stage6_evidence_summary.md"
```

复制所有决策输入：

```bash
cp "$STAGE6_FINAL/metrics/stage6_evidence.json" \
  "$ROUND_DIR/metrics/stage6_evidence.json"
cp "$STAGE6_EVAL/locks/non_graph_comparator_lock.json" \
  "$ROUND_DIR/configs/non_graph_comparator_lock.json"
cp "$STAGE6_EVAL/locks/policy_evaluation_lock.json" \
  "$ROUND_DIR/configs/policy_evaluation_lock.json"
cp "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json" \
  "$ROUND_DIR/configs/policy_protocol_lock.json"
cp "$STAGE6_WEIGHTS/weight_selection_lock.json" \
  "$ROUND_DIR/configs/weight_selection_lock.json"
```

---

## 三、G3 决策门槛

### 3.1 `GO_STAGE7`

同时满足：

```text
1. 真实动作数据门通过：
   POLICY_DATA_REAL_ACTION_READY

2. 完整训练：
   24/24 required policy jobs completed
   24/24 selected checkpoints validation-only

3. 完整评估：
   all required conditions and paired test seeds completed

4. 图任务总体成功率提升：
   graph_task_success_gain >= 0.05

5. 跨 seed 一致性：
   improved_policy_seed_count >= 2 / 3

6. 固定顺序不退化：
   fixed_order_drop <= 0.05

7. 至少一个图结构专项收益达到：
   recovery_success_gain >= 0.08
   OR
   worst_order_success_gain >= 0.08
   OR
   long_horizon_completion_gain >= 0.05

8. 协议完整：
   reward_retuned_after_test == false
   policy_protocol_locked == true
   paired evaluation == true
```

置信区间用于报告可信度，不设置“CI 下界必须大于 0”的绝对硬门，以免在三 policy seed MVP 中因统计功效不足直接否定明确的跨 seed趋势。但若 95% CI 极宽，Stage 7 必须优先补 seed/rollout 规模，而不是立即扩展方法故事。

### 3.2 `REFINE_STAGE6`

以下情况之一：

```text
总体方向为正，但 graph_task_success_gain 在 [0.02, 0.05)
或只有 1/3 seed 改善
或 recovery/worst-order/long-horizon 专项收益接近门槛
或权重 ESS 过低导致训练信号不足
或某个任务 demonstration 数量明显不足
或一个明确的训练协议实现问题影响全部方法
```

只允许一次定向修正，优先级：

1. 在阶段 6.2 预先规定的 `gamma={1.0,0.75,0.5}` 中选择下一个满足 ESS 的值；
2. 统一增加所有方法的训练预算；
3. 只补缺失的真实动作 demonstration；
4. 修复确定的 weighted-loss 或 evaluation 实现 bug。

限制：

- 不重新调 Stage 5 `lambda/eta/beta`；
- 不根据 test 选择新 gamma；gamma 调整必须由 train ESS/coverage 触发；
- 统一改变训练预算时，四方法全部重跑；
- 补数据时保持原 test seed registry不变；
- 定向修正后生成 `stage6_refine_1.zip`，不无限迭代。

### 3.3 `NARROW_TO_REWARD_ONLY`

满足：

```text
Stage 5 graph reward 机制指标仍有效
但 Stage 6 graph_task_success_gain <= 0
或跨 3 seeds 无稳定改善
或策略提升完全可由 BC-All/非图 baseline 解释
```

此状态不否定 reward-model 研究结果，但表示当前 RA-BC 下游主张不足。后续论文定位收窄为图奖励建模与校准，或切换到备用的 credit/influence 方向。

### 3.4 `POLICY_DATA_BLOCKED`

仅在以下直接阻塞出现时：

```text
无法获得真实 action-bearing graph demonstrations
无法运行兼容的闭环 evaluator
persistent reward checkpoint 无法恢复
```

如果阶段 6.1 已通过，不应在阶段 6.7 再输出该状态。

---

## 四、实现 G3 决策脚本

创建：

```text
tools/stage6/decide_g3.py
```

CLI：

```bash
python tools/stage6/decide_g3.py \
  --evidence "$STAGE6_FINAL/metrics/stage6_evidence.json" \
  --stage6-config "$STAGE6_CONFIG_ROOT/stage6.yaml" \
  --policy-data-gate "$STAGE6_ROUNDS/stage6_1_policy_ready_data_and_input_freeze/metrics/policy_data_gate.json" \
  --weighting-gate "$STAGE6_ROUNDS/stage6_2_weighting_pipeline/metrics/weighting_pipeline_gate.json" \
  --training-gate "$STAGE6_ROUNDS/stage6_4_full_policy_training/metrics/full_training_gate.json" \
  --evaluation-gate "$STAGE6_ROUNDS/stage6_5_frozen_policy_evaluation/metrics/policy_evaluation_gate.json" \
  --output "$ROUND_DIR/metrics/g3_gate.json" \
  --report "$ROUND_DIR/reports/g3_decision.md"
```

`g3_gate.json` 必须包含：

```json
{
  "decision": "",
  "checks": {
    "real_action_policy_data": false,
    "required_training_complete": false,
    "frozen_evaluation_complete": false,
    "graph_task_success_gain": false,
    "seed_consistency": false,
    "fixed_order_non_regression": false,
    "graph_specific_gain": false,
    "reward_not_retuned_after_test": false
  },
  "observed": {
    "graph_task_success_gain": 0.0,
    "improved_policy_seed_count": 0,
    "fixed_order_drop": 0.0,
    "recovery_success_gain": 0.0,
    "worst_order_success_gain": 0.0,
    "long_horizon_completion_gain": 0.0
  },
  "thresholds": {},
  "recommended_next_action": ""
}
```

脚本必须从输入文件读取结果，禁止写固定通过值。

---

## 五、小阶段 6.7-A：冻结 M4 结果包

若 `GO_STAGE7`，创建：

```text
$STAGE6_FINAL/FROZEN.md
$STAGE6_FINAL/g3_decision.md
$STAGE6_FINAL/stage7_handoff.md
$STAGE6_FINAL/M4_SHA256SUMS.txt
```

`FROZEN.md`：

```text
milestone = M4_POLICY_EVIDENCE_READY
g3 = GO_STAGE7
policy_test_used_for_selection = false
reward_retuned_after_test = false
checkpoint_packaging = omitted_by_default
statistics = hierarchical_paired_bootstrap
```

冻结内容至少包含：

```text
configs/
  stage6.yaml
  reward_config_v1.yaml
  policy_training_protocol.yaml
  policy_protocol_lock.json
  weight_selection_lock.json
  policy_checkpoint_selection_lock.json
  policy_evaluation_lock.json
  non_graph_comparator_lock.json

tables/
  stage6_primary_results.csv
  stage6_seed_results.csv
  stage6_mechanism_results.csv
  bootstrap_effects.csv
  checkpoint_selection.csv

metrics/
  stage6_evidence.json
  g3_gate.json

reports/
  stage6_evidence_summary.md
  g3_decision.md
  stage7_handoff.md

manifests/
  policy_dataset_manifest_reference.tsv
  checkpoint_manifest.tsv
  large_file_manifest.tsv
```

checkpoint、原始 demonstration、rollout trajectory 和视频只引用，不复制。

---

## 六、小阶段 6.7-B：Stage 7 交接内容

`stage7_handoff.md` 至少写明：

### 6.1 冻结的 Stage 6 主结果

```text
PathGraph vs locked comparator:
  graph-task success gain
  recovery success gain
  worst-order success gain
  long-horizon completion gain
  fixed-order drop
  improved seeds
  95% CI
```

### 6.2 后续优先实验

若 `GO_STAGE7`，优先顺序：

1. 增加 policy seeds 或 rollout 数，收紧主效果置信区间；
2. 运行策略层面的组件消融：
   ```text
   no_phi
   no_recovery_debt_cap
   cost_only
   ```
3. 成功路径数量扩展；
4. recovery 频率扩展；
5. 人工图 vs 自动图；
6. 非零 `eta/beta` 敏感性只能作为新增实验，不覆盖 Stage 5 frozen v1。

Stage 7 不应先做自动图发现，而应先把最关键的策略效果和消融做实。

### 6.3 命名边界

由于 Stage 5 冻结参数：

```text
eta=0
beta=0
```

Stage 6 主结果名称继续使用：

```text
PathGraph-SARM / pathgraph_reward_v1_locked
```

Stage 7 才可以独立增加：

```text
PathGraph-SARM + nonzero loop penalty
PathGraph-SARM + uncertainty LCB
```

这些必须作为新实验变体，不可回写 Stage 6 主结果。

---

## 七、小阶段 6.7-C：阶段 6 总结报告

创建：

```text
$ROUND_DIR/reports/stage6_summary.md
```

固定结构：

```markdown
# Stage 6 Summary

## Entry
- G2:
- Stage 6.1 policy data gate:

## Data
- task/scenario episode counts:
- action usability:
- split:

## Weighting
- methods:
- PathGraph ESS:
- recovery weight coverage:
- frozen reward config:

## Training
- jobs:
- seeds:
- CUDA:
- protocol:

## Evaluation
- rollout count:
- conditions:
- paired seeds:

## Main results
- graph task success:
- recovery:
- worst order:
- fixed order:

## G3
- decision:
- reason:

## Limitations
- action data source:
- model gate 2/3:
- eta/beta zero:
```

限制必须如实记录，不影响继续推进：

- Stage 5 individual model gate 为 2/3；
- policy data 若以 scripted expert/controller采集，不等于真人 demonstration；
- `eta=0`、`beta=0`；
- 当前任务数为 2。

---

## 八、小阶段 6.7-D：生成本轮 ZIP

```bash
export ZIP_NAME="stage6_7_g3_m4_freeze.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 九、生成阶段总 ZIP

无论 G3 是 `GO_STAGE7`、`REFINE_STAGE6` 还是 `NARROW_TO_REWARD_ONLY`，都生成阶段总 ZIP。

创建总交付暂存目录：

```bash
export COMPLETE_DIR="$STAGE6_ROOT/_runtime/stage6_complete_staging"
rm -rf "$COMPLETE_DIR"
mkdir -p "$COMPLETE_DIR"
```

复制轻量冻结结果：

```bash
rsync -a \
  "$STAGE6_FINAL/" \
  "$COMPLETE_DIR/m4_policy_results_v1/" \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='raw_episodes/' \
  --exclude='rollouts/' \
  --exclude='videos/' \
  --exclude='cache/'

mkdir -p "$COMPLETE_DIR/round_summaries"
for r in \
  stage6_1_policy_ready_data_and_input_freeze \
  stage6_2_weighting_pipeline \
  stage6_3_protocol_lock_and_smoke \
  stage6_4_full_policy_training \
  stage6_5_frozen_policy_evaluation \
  stage6_6_mechanism_and_statistics \
  stage6_7_g3_m4_freeze
do
  mkdir -p "$COMPLETE_DIR/round_summaries/$r"
  for f in run_manifest.md reports metrics tables configs checksums manifests; do
    if [ -e "$STAGE6_ROUNDS/$r/$f" ]; then
      rsync -a "$STAGE6_ROUNDS/$r/$f" \
        "$COMPLETE_DIR/round_summaries/$r/" \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.ckpt' \
        --exclude='*.safetensors'
    fi
  done
done
```

写总 manifest：

```bash
cat > "$COMPLETE_DIR/stage6_complete_manifest.md" <<EOF
# PathGraph-SARM Stage 6 complete

- g2_entry: GO_STAGE6
- stage6_first_gate: $(python -c 'import json; print(json.load(open("'"$STAGE6_ROUNDS"'/stage6_1_policy_ready_data_and_input_freeze/metrics/policy_data_gate.json"))["decision"])')
- g3: $(python -c 'import json; print(json.load(open("'"$ROUND_DIR"'/metrics/g3_gate.json"))["decision"])')
- required_policy_jobs: 24
- policy_seeds: $POLICY_SEEDS
- reward_config: pathgraph_reward_v1_locked
- reward_eta: 0.0
- reward_beta: 0.0
- checkpoint_packaging: omitted_by_default
- raw_episode_packaging: omitted_by_default
- per_round_zips_retained: true
EOF
```

生成总 ZIP：

```bash
export COMPLETE_ZIP="$STAGE6_DOWNLOADS/stage6_complete.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$COMPLETE_DIR" \
  --output-zip "$COMPLETE_ZIP" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$COMPLETE_ZIP" \
  | tee "$ROUND_DIR/checksums/stage6_complete_unzip_test.txt"

sha256sum "$COMPLETE_ZIP" \
  | tee "$ROUND_DIR/checksums/stage6_complete.sha256"
```

不要删除七个小阶段 ZIP。用户明确要求每轮均可下载。

---

## 十、最终 Agent 回复格式

```text
阶段 6 已完成。

G3：<GO_STAGE7 | REFINE_STAGE6 | NARROW_TO_REWARD_ONLY | POLICY_DATA_BLOCKED>

核心结果：
- graph-task success gain:
- improved policy seeds:
- recovery success gain:
- worst-order gain:
- long-horizon completion gain:
- fixed-order drop:
- 95% CI:

七轮 ZIP：
- stage6_1_policy_ready_data_and_input_freeze.zip
- stage6_2_weighting_pipeline.zip
- stage6_3_protocol_lock_and_smoke.zip
- stage6_4_full_policy_training.zip
- stage6_5_frozen_policy_evaluation.zip
- stage6_6_mechanism_and_statistics.zip
- stage6_7_g3_m4_freeze.zip

总 ZIP：
- absolute path:
- SHA256:
- unzip -t:
```

**核心点：G3 只依据冻结的真实策略结果；达到门槛则进入阶段 7，方向不足则诚实收窄，不通过事后调 reward 或删除不利 seed制造通过。**

<!-- END FILE: 阶段6.7_G3决策与M4冻结.md -->
