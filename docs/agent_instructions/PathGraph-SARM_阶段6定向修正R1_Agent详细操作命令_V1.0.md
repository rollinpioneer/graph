# PathGraph-SARM 阶段 6 定向修正 R1：Agent 详细操作命令 V1.0

# PathGraph-SARM 阶段 6 定向修正 R1：Agent 通用执行规范

## 0. 本轮结论与唯一目标

阶段 6 已完整执行，但 G3 为：

```text
REFINE_STAGE6
```

冻结结果中：

```text
graph-task success gain = 0.0666666667
recovery success gain = 0.1666666667
fixed-order drop = 0
improved policy seeds = 1 / 3
```

逐 seed 结果显示：

```text
seed 20260909：PathGraph 相对 locked comparator 的 graph-task success +0.20
seed 20260910：差值 0；两者均为 1.00，属于 ceiling tie
seed 20260911：差值 0；两者均为 1.00，属于 ceiling tie
```

因此，本轮**不修改奖励、不修改权重、不增加训练预算、不补采数据**。只做一次独立的跨 seed 复现实验，解决“只有 1/3 seed 严格改善”的证据不足问题。

本轮唯一允许的实验变化：

```text
新增 policy seeds：20260912、20260913、20260914
```

本轮继续使用：

```text
tasks:
  - transport_recovery
  - transport_dual_order

methods:
  - linear_sarm_equiv
  - pathgraph_reward_v1_locked
```

只运行已锁定的主比较对，不再重复 `bc_all` 和 `sequential_transition`。这是为了直接解决 G3 的 seed consistency 问题，避免重复消耗算力。

---

## 1. 严格冻结项

以下内容不得变化：

```text
Stage 5 reward_v1
lambda / eta / beta
recovery debt cap
Stage 6 chunk weights
gamma = 1.0
policy_data_v1
策略网络架构
action horizon = 16
observation horizon = 2
optimizer
learning-rate schedule
batch size = 32
effective batch size = 32
optimizer steps = 2000
validation cadence = 200
checkpoint selection metric = val_action_loss_unweighted
checkpoint selection split = val
locked comparator = linear_sarm_equiv
test condition definitions
evaluation seed registry
每 condition rollout 数 = 50
bootstrap resamples = 5000
```

禁止事项：

```text
不得根据 Stage 6 test 结果重新调 reward 参数
不得重新选择 comparator
不得更改 gamma
不得为 PathGraph 单独增加训练步数
不得挑选或删除不利 policy seed
不得创建更容易或更难的条件来替换原 G3 test
不得把新 seed 的 validation 或 test 用于调整方法定义
不得进行第二轮方法调参
```

---

## 2. 小阶段与交付 ZIP

```text
6R1.1  修正协议锁定与新 seed 注册
       stage6r1_1_refinement_lock.zip

6R1.2  新 seed 双方法并行训练
       stage6r1_2_new_seed_training.zip

6R1.3  Validation-only checkpoint 选择
       stage6r1_3_validation_selection.zip

6R1.4  冻结配对闭环评估
       stage6r1_4_frozen_paired_evaluation.zip

6R1.5  六 seed 汇总、bootstrap 与 G3-R1 决策
       stage6r1_5_g3r1_decision.zip

总交付：
       stage6_refine1_complete.zip
```

每个小阶段结束立即打 ZIP，不等待总阶段结束。

checkpoint、模型权重、原始 episode、完整 rollout 轨迹、视频和大数组默认不进入 ZIP；只写入 manifest。

---

## 3. 统一环境变量

从仓库根目录执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE6_ROOT="${STAGE6_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6}"
export STAGE6_FINAL="${STAGE6_FINAL:-$STAGE6_ROOT/m4_policy_results_v1}"
export STAGE6_DATA="${STAGE6_DATA:-$STAGE6_ROOT/policy_data_v1}"
export STAGE6_WEIGHTS="${STAGE6_WEIGHTS:-$STAGE6_ROOT/policy_weights_v1}"
export STAGE6_PROTOCOL="${STAGE6_PROTOCOL:-$STAGE6_ROOT/policy_protocol_v1}"
export STAGE6_TRAIN="${STAGE6_TRAIN:-$STAGE6_ROOT/policy_training_v1}"
export STAGE6_EVAL="${STAGE6_EVAL:-$STAGE6_ROOT/policy_evaluation_v1}"
export STAGE6_TOOLS="${STAGE6_TOOLS:-$REPO_ROOT/tools/stage6}"

export STAGE6R1_ROOT="${STAGE6R1_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6_refine1}"
export STAGE6R1_ROUNDS="$STAGE6R1_ROOT/rounds"
export STAGE6R1_PROTOCOL="$STAGE6R1_ROOT/refine_protocol_v1"
export STAGE6R1_TRAIN="$STAGE6R1_ROOT/policy_training_v1"
export STAGE6R1_EVAL="$STAGE6R1_ROOT/policy_evaluation_v1"
export STAGE6R1_FINAL="$STAGE6R1_ROOT/m4_refine1_results_v1"
export STAGE6R1_TOOLS="$REPO_ROOT/tools/stage6_refine1"
export STAGE6R1_DOWNLOADS="${STAGE6R1_DOWNLOADS:-$REPO_ROOT/downloads/stage6_refine1}"

export NEW_POLICY_SEEDS="20260912,20260913,20260914"
export REFINE_METHODS="linear_sarm_equiv,pathgraph_reward_v1_locked"
export REFINE_TASKS="transport_recovery,transport_dual_order"

export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-8000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p \
  "$STAGE6R1_ROOT" \
  "$STAGE6R1_ROUNDS" \
  "$STAGE6R1_PROTOCOL" \
  "$STAGE6R1_TRAIN" \
  "$STAGE6R1_EVAL" \
  "$STAGE6R1_FINAL" \
  "$STAGE6R1_TOOLS" \
  "$STAGE6R1_DOWNLOADS"
```

将其保存为：

```text
artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
```

后续每个小阶段先执行：

```bash
source artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
cd "$REPO_ROOT"
```

---

## 4. GPU 必须提权查看

每个含 GPU 的小阶段开始时执行：

```bash
mkdir -p "$ROUND_DIR/gpu"

if sudo -n nvidia-smi > "$ROUND_DIR/gpu/nvidia_smi_sudo.txt" 2>&1; then
  echo "GPU_QUERY_MODE=sudo_noninteractive" \
    | tee "$ROUND_DIR/gpu/gpu_query_mode.txt"
else
  echo "sudo -n nvidia-smi failed; trying interactive sudo when terminal permits" \
    | tee "$ROUND_DIR/gpu/gpu_query_mode.txt"

  if [ -t 0 ]; then
    sudo nvidia-smi \
      | tee "$ROUND_DIR/gpu/nvidia_smi_sudo_interactive.txt"
    echo "GPU_QUERY_MODE=sudo_interactive" \
      | tee -a "$ROUND_DIR/gpu/gpu_query_mode.txt"
  else
    nvidia-smi \
      | tee "$ROUND_DIR/gpu/nvidia_smi_direct.txt"
    echo "GPU_QUERY_MODE=direct_fallback_noninteractive" \
      | tee -a "$ROUND_DIR/gpu/gpu_query_mode.txt"
  fi
fi

nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader \
  | tee "$ROUND_DIR/gpu/gpu_inventory.csv"

"$PYTHON_BIN" - <<'PY'
import torch
print("torch_version =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("cuda_device_count =", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

普通权限看不到 GPU 时，不允许直接写“无 GPU”。先完成提权查询。

---

## 5. 多 GPU 默认并行

独立训练 job、评估 shard 可以并行时，必须并行。

默认策略：

```text
一个独立 job 占一张 GPU
```

优先使用现有调度器：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/launch_gpu_job_matrix.py" \
  --job-table <job_table.tsv> \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output <job_status.tsv> \
  --resume-failed true
```

若现有调度器不支持新目录，创建轻量 wrapper：

```text
tools/stage6_refine1/launch_gpu_job_matrix.py
```

只允许修改路径解析，不允许改训练或评估逻辑。

---

## 6. 每轮 ZIP 规则

每轮 ZIP 必须至少包含：

```text
run_manifest.md
configs/
commands/
gpu/
logs/
metrics/
tables/
figures/
reports/
manifests/
checksums/
```

默认排除：

```text
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
原始 episode
完整 rollout CSV/JSONL/NPZ
视频
缓存
超过 200 MB 的其他文件
```

未打包大文件写入：

```text
manifests/checkpoint_manifest.tsv
manifests/large_file_manifest.tsv
```

每轮打包：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

如果原 `package_round.py` 不存在或不支持排除 checkpoint，复制阶段 6 已用的版本到：

```text
tools/stage6_refine1/package_round.py
```

不得重新设计复杂打包系统。

---

## 7. 运行记录要求

每个小阶段必须保存：

```text
commands/executed_commands.sh
run_manifest.md
logs/*.log
metrics/*gate*.json
reports/*summary*.md
checksums/*.sha256
```

`run_manifest.md` 至少记录：

```text
stage
round
start_time
end_time
repo_root
git_commit
python
cuda
GPU inventory
input lock SHA256
actual commands
job count
success count
failed count
output ZIP
ZIP SHA256
large files omitted
next action
```

---

## 8. 最终决策边界

本轮只允许一次定向修正。

最终状态：

```text
GO_STAGE7
NARROW_TO_REWARD_ONLY
REPAIR_REFINE1_INFRA
```

含义：

```text
GO_STAGE7
  新增 seed 独立复现通过，可以进入阶段 7。

NARROW_TO_REWARD_ONLY
  新 seed 复现未达到跨 seed 一致性；停止继续调 Stage 6，
  保留 Stage 3–5 的图奖励贡献，将论文主张收窄到 reward modeling。

REPAIR_REFINE1_INFRA
  仅用于 job 失败、文件缺失、配对评估断裂等执行问题；
  只补跑受影响 job，不改变方法。
```

不得再次输出 `REFINE_STAGE6` 来开启无限迭代。

---

# 阶段 6R1.1：修正协议锁定与新 Policy Seed 注册

## 一、总体上要干什么

读取 Stage 6 已冻结结果，将本轮定向修正原因锁定为：

```text
seed consistency replication
```

本小阶段只做三件事：

1. 冻结 Stage 6 的数据、权重、训练协议、比较对象和评估协议；
2. 注册三个新增 policy seed；
3. 在任何新训练开始前冻结 `G3-R1` 判定规则。

本轮不训练模型。

本轮状态：

```text
REFINE1_PROTOCOL_LOCKED
```

本轮 ZIP：

```text
stage6r1_1_refinement_lock.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage6r1_1_refinement_lock"
export ROUND_DIR="$STAGE6R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6R1_PROTOCOL"/{configs,locks,seed_registry,initializations,manifests}
```

本轮不需要占用 GPU，但仍保存当前 GPU 快照，方便后续调度：

```bash
sudo -n nvidia-smi \
  > "$ROUND_DIR/gpu/nvidia_smi_sudo.txt" 2>&1 \
  || nvidia-smi > "$ROUND_DIR/gpu/nvidia_smi_direct.txt" 2>&1 \
  || true
```

---

## 三、定位 Stage 6 冻结输入

依次检查：

```bash
test -f "$STAGE6_FINAL/metrics/g3_decision.json"
test -f "$STAGE6_FINAL/g3_decision.md"
test -f "$STAGE6_WEIGHTS/weight_selection_lock.json"
test -f "$STAGE6_PROTOCOL/locks/policy_protocol_lock.json"
test -f "$STAGE6_TRAIN/selection/policy_checkpoint_selection_lock.json"
test -f "$STAGE6_EVAL/locks/policy_evaluation_lock.json"
test -f "$STAGE6_EVAL/locks/primary_comparator_lock.json"
test -f "$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml"
test -f "$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml"
test -f "$STAGE6_DATA/train_manifest.jsonl"
test -f "$STAGE6_DATA/val_manifest.jsonl"
```

定位原评估 seed registry：

```bash
if [ -f "$STAGE6_EVAL/locks/evaluation_seed_registry.csv" ]; then
  export OLD_EVAL_SEED_REGISTRY="$STAGE6_EVAL/locks/evaluation_seed_registry.csv"
elif [ -f "$STAGE6_EVAL/locks/test_seed_registry.csv" ]; then
  export OLD_EVAL_SEED_REGISTRY="$STAGE6_EVAL/locks/test_seed_registry.csv"
else
  echo "Missing frozen evaluation seed registry."
  echo "Recover the original Stage 6 registry from the Stage 6 workspace."
  exit 2
fi
```

不得重新随机生成另一套 test seed。

---

## 四、确认当前决策确为 REFINE_STAGE6

```bash
"$PYTHON_BIN" - <<'PY'
import json, os
p = os.path.join(os.environ["STAGE6_FINAL"], "metrics/g3_decision.json")
d = json.load(open(p))
assert d["decision"] == "REFINE_STAGE6", d["decision"]
assert d["checks"]["graph_gain_ge_0_05"] is True
assert d["checks"]["improved_seeds_ge_2"] is False
assert d["checks"]["fixed_order_drop_le_0_05"] is True
assert d["checks"]["structure_specific_gain"] is True
assert d["checks"]["no_post_test_reward_retune"] is True
print("STAGE6_REFINE_REASON_CONFIRMED")
PY
```

生成原因文件：

```bash
"$PYTHON_BIN" - <<'PY'
import json, os
src = json.load(open(os.path.join(
    os.environ["STAGE6_FINAL"], "metrics/g3_decision.json"
)))
out = {
  "decision_source": "stage6_frozen_g3",
  "reason": "only_1_of_3_policy_seeds_strictly_improved",
  "graph_task_success_gain": src["evidence"]["graph_task_success_gain"],
  "recovery_success_gain": src["evidence"]["recovery_success_gain"],
  "fixed_order_drop": src["evidence"]["fixed_order_drop"],
  "improved_policy_seed_count": src["evidence"]["improved_policy_seed_count"],
  "refinement_type": "new_policy_seed_replication_only",
  "reward_retune_allowed": False,
  "gamma_change_allowed": False,
  "training_budget_change_allowed": False,
  "new_data_collection_allowed": False
}
p = os.path.join(
    os.environ["STAGE6R1_PROTOCOL"], "locks/refine_reason.json"
)
os.makedirs(os.path.dirname(p), exist_ok=True)
json.dump(out, open(p, "w"), indent=2)
print(json.dumps(out, indent=2))
PY
```

---

## 五、冻结输入文件 SHA256

创建清单：

```bash
cat > "$ROUND_DIR/manifests/refine1_input_files.tsv" <<EOF
name	path
g3_decision	$STAGE6_FINAL/metrics/g3_decision.json
weight_lock	$STAGE6_WEIGHTS/weight_selection_lock.json
policy_protocol_lock	$STAGE6_PROTOCOL/locks/policy_protocol_lock.json
base_policy_config	$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml
policy_training_protocol	$STAGE6_PROTOCOL/configs/policy_training_protocol.yaml
checkpoint_selection_lock	$STAGE6_TRAIN/selection/policy_checkpoint_selection_lock.json
policy_evaluation_lock	$STAGE6_EVAL/locks/policy_evaluation_lock.json
primary_comparator_lock	$STAGE6_EVAL/locks/primary_comparator_lock.json
evaluation_seed_registry	$OLD_EVAL_SEED_REGISTRY
train_manifest	$STAGE6_DATA/train_manifest.jsonl
val_manifest	$STAGE6_DATA/val_manifest.jsonl
pathgraph_weight	$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet
linear_weight	$STAGE6_WEIGHTS/chunk_weights/linear_sarm_equiv.parquet
EOF
```

计算 SHA：

```bash
"$PYTHON_BIN" - <<'PY'
import csv, hashlib, os, json
inp = os.path.join(
    os.environ["ROUND_DIR"], "manifests/refine1_input_files.tsv"
)
out_tsv = os.path.join(
    os.environ["ROUND_DIR"], "manifests/refine1_input_hashes.tsv"
)
rows = []
with open(inp) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        p = r["path"]
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
        h = hashlib.sha256()
        with open(p, "rb") as x:
            for b in iter(lambda: x.read(1024 * 1024), b""):
                h.update(b)
        rows.append({
            "name": r["name"],
            "path": p,
            "size_bytes": os.path.getsize(p),
            "sha256": h.hexdigest()
        })
with open(out_tsv, "w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["name","path","size_bytes","sha256"],
        delimiter="\t"
    )
    w.writeheader()
    w.writerows(rows)

lock = {
    "locked": True,
    "input_count": len(rows),
    "files": rows
}
lock_path = os.path.join(
    os.environ["STAGE6R1_PROTOCOL"], "locks/refine1_input_lock.json"
)
json.dump(lock, open(lock_path, "w"), indent=2)
print("REFINE1_INPUT_LOCK_WRITTEN", lock_path)
PY
```

复制轻量锁文件：

```bash
cp "$STAGE6R1_PROTOCOL/locks/refine_reason.json" \
  "$ROUND_DIR/configs/refine_reason.json"
cp "$STAGE6R1_PROTOCOL/locks/refine1_input_lock.json" \
  "$ROUND_DIR/configs/refine1_input_lock.json"
```

不要复制 checkpoint 和权重文件本体。

---

## 六、创建 R1 训练协议

复制原协议作为基础：

```bash
cp "$STAGE6_PROTOCOL/configs/base_policy_stage6.yaml" \
  "$STAGE6R1_PROTOCOL/configs/base_policy_stage6.yaml"
```

创建：

```text
$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml
```

内容：

```bash
cat > "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" <<'YAML'
protocol_version: stage6-refine1-policy-v1
parent_protocol: stage6-policy-v1

tasks:
  - transport_recovery
  - transport_dual_order

methods:
  - linear_sarm_equiv
  - pathgraph_reward_v1_locked

policy_seeds:
  - 20260912
  - 20260913
  - 20260914

data:
  dataset_version: policy_data_v1
  split_train: train
  split_selection: val
  test_for_selection: false
  sampler: identical_unweighted_sampler
  weighted_sampling: false
  action_horizon: 16
  observation_horizon: 2

optimization:
  total_optimizer_steps: 2000
  batch_size: 32
  effective_batch_size: 32
  validation_every_steps: 200
  checkpoint_selection_metric: val_action_loss_unweighted
  checkpoint_selection_mode: min

fairness:
  same_architecture_within_task: true
  same_initialization_within_task_seed: true
  same_data_order_within_task_seed: true
  same_optimizer_steps: true
  only_weight_changes: true

evaluation:
  reuse_stage6_seed_registry: true
  rollout_count_per_condition: 50
  paired_evaluation: true
  bootstrap_resamples: 5000
YAML
```

比较原协议中的冻结字段：

```bash
"$PYTHON_BIN" - <<'PY'
import yaml, os
old = yaml.safe_load(open(os.path.join(
    os.environ["STAGE6_PROTOCOL"],
    "configs/policy_training_protocol.yaml"
)))
new = yaml.safe_load(open(os.path.join(
    os.environ["STAGE6R1_PROTOCOL"],
    "configs/refine1_policy_training_protocol.yaml"
)))

for key in [
    "total_optimizer_steps",
    "batch_size",
    "effective_batch_size",
    "validation_every_steps",
    "checkpoint_selection_metric",
    "checkpoint_selection_mode",
]:
    assert old["optimization"][key] == new["optimization"][key], key

for key in [
    "sampler",
    "weighted_sampling",
    "action_horizon",
    "observation_horizon",
]:
    assert old["data"][key] == new["data"][key], key

print("FROZEN_TRAINING_FIELDS_MATCH")
PY
```

---

## 七、注册新增 seed

创建：

```text
tools/stage6_refine1/extend_policy_seed_registry.py
```

CLI：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/extend_policy_seed_registry.py" \
  --parent-registry "$STAGE6_PROTOCOL/seed_registry/policy_seed_registry.csv" \
  --tasks "$REFINE_TASKS" \
  --new-policy-seeds "$NEW_POLICY_SEEDS" \
  --output "$STAGE6R1_PROTOCOL/seed_registry/refine1_policy_seed_registry.csv"
```

脚本要求：

1. 保留原 registry 不变；
2. 只输出新 seed 的两任务记录，共 6 行；
3. 每个 `(task_id, policy_seed)` 唯一；
4. 新 seed 不与旧 seed 重复；
5. 同一 policy seed 的随机源按固定函数生成；
6. 生成字段：

```text
task_id
policy_seed
model_seed
data_seed
augmentation_seed
validation_seed_start
test_seed_registry_id
```

推荐固定派生函数：

```python
model_seed = policy_seed
data_seed = policy_seed + 100000
augmentation_seed = policy_seed + 200000
validation_seed_start = policy_seed + 300000
test_seed_registry_id = "stage6_frozen_eval_registry"
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["STAGE6R1_PROTOCOL"],
    "seed_registry/refine1_policy_seed_registry.csv"
)
d = pd.read_csv(p)
assert len(d) == 6, len(d)
assert d["task_id"].nunique() == 2
assert d["policy_seed"].nunique() == 3
assert set(d["policy_seed"]) == {20260912,20260913,20260914}
assert not d.duplicated(["task_id","policy_seed"]).any()
print("REFINE1_SEED_REGISTRY_OK")
PY
```

复制到本轮：

```bash
cp "$STAGE6R1_PROTOCOL/seed_registry/refine1_policy_seed_registry.csv" \
  "$ROUND_DIR/tables/"
```

---

## 八、在实验前冻结 G3-R1 规则

创建：

```text
$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json
```

```bash
cat > "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" <<'JSON'
{
  "locked_before_new_training": true,
  "comparison": {
    "method_a": "pathgraph_reward_v1_locked",
    "method_b": "linear_sarm_equiv",
    "new_policy_seeds": [20260912, 20260913, 20260914],
    "old_policy_seeds": [20260909, 20260910, 20260911]
  },
  "go_stage7": {
    "new_seed_improved_count_min": 2,
    "new_seed_count": 3,
    "combined_graph_task_success_gain_min": 0.05,
    "combined_fixed_order_drop_max": 0.05,
    "structure_specific_any": {
      "combined_recovery_success_gain_min": 0.08,
      "combined_worst_order_success_gain_min": 0.08,
      "combined_long_horizon_completion_gain_min": 0.05
    },
    "reward_retuned_after_test": false,
    "paired_evaluation": true
  },
  "terminal_failure_decision": "NARROW_TO_REWARD_ONLY",
  "infrastructure_failure_decision": "REPAIR_REFINE1_INFRA"
}
JSON
```

说明：

- 新 seed block 必须至少 2/3 严格改善；
- combined gain 使用旧 3 seed + 新 3 seed；
- ceiling tie 不计为改善，但单独报告；
- 本轮失败后不再继续调 Stage 6。

锁定：

```bash
sha256sum \
  "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" \
  "$STAGE6R1_PROTOCOL/locks/refine_reason.json" \
  "$STAGE6R1_PROTOCOL/locks/refine1_input_lock.json" \
  "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" \
  "$STAGE6R1_PROTOCOL/seed_registry/refine1_policy_seed_registry.csv" \
  | tee "$STAGE6R1_PROTOCOL/locks/REFINE1_PROTOCOL_SHA256SUMS.txt"

cp "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" \
  "$ROUND_DIR/configs/"
cp "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" \
  "$ROUND_DIR/configs/"
cp "$STAGE6R1_PROTOCOL/locks/REFINE1_PROTOCOL_SHA256SUMS.txt" \
  "$ROUND_DIR/checksums/"
```

---

## 九、本轮 gate

创建：

```text
tools/stage6_refine1/decide_refine1_protocol_gate.py
```

检查：

```text
Stage 6 decision == REFINE_STAGE6
graph gain >= 0.05
seed consistency was the only failed G3 gate
input hashes complete
locked comparator == linear_sarm_equiv
new seeds == 20260912,20260913,20260914
training fields equal parent protocol
reward/gamma/data/budget change flags == false
G3-R1 rule locked before training
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/decide_refine1_protocol_gate.py" \
  --g3 "$STAGE6_FINAL/metrics/g3_decision.json" \
  --input-lock "$STAGE6R1_PROTOCOL/locks/refine1_input_lock.json" \
  --protocol "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" \
  --seed-registry "$STAGE6R1_PROTOCOL/seed_registry/refine1_policy_seed_registry.csv" \
  --rule "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" \
  --output "$ROUND_DIR/metrics/refine1_protocol_gate.json" \
  --report "$ROUND_DIR/reports/refine1_protocol_summary.md"
```

允许状态：

```text
REFINE1_PROTOCOL_LOCKED
FIX_PROTOCOL_FILES
```

只有 `REFINE1_PROTOCOL_LOCKED` 才能进入 6R1.2。

---

## 十、生成本轮 ZIP

```bash
export ZIP_NAME="stage6r1_1_refinement_lock.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复格式：

```text
阶段 6R1.1 状态：REFINE1_PROTOCOL_LOCKED
新增 policy seeds：20260912, 20260913, 20260914
方法：linear_sarm_equiv, pathgraph_reward_v1_locked
冻结 reward/gamma/data/budget：是
G3-R1 rule SHA256：<hash>
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6R1.2
```

**核心点：先锁定“只加三个 policy seed”的独立复现实验，避免在看到新结果后改变方法或门槛。**

---

# 阶段 6R1.2：新增 Policy Seed 的双方法多 GPU 并行训练

## 一、总体上要干什么

对新增 seed：

```text
20260912
20260913
20260914
```

在两个任务上训练：

```text
linear_sarm_equiv
pathgraph_reward_v1_locked
```

总训练 job：

\[
2\ \text{tasks}
\times
2\ \text{methods}
\times
3\ \text{seeds}
=
12\ \text{jobs}
\]

不重跑旧 seed，不运行最终 test。

本轮状态：

```text
REFINE1_TRAINING_COMPLETE
```

本轮 ZIP：

```text
stage6r1_2_new_seed_training.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage6r1_2_new_seed_training"
export ROUND_DIR="$STAGE6R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6R1_TRAIN"/{jobs,selection,manifests,metrics}
```

先按通用规范提权查看 GPU：

```bash
if sudo -n nvidia-smi > "$ROUND_DIR/gpu/nvidia_smi_sudo.txt" 2>&1; then
  echo "sudo_noninteractive" > "$ROUND_DIR/gpu/gpu_query_mode.txt"
elif [ -t 0 ]; then
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/nvidia_smi_sudo_interactive.txt"
  echo "sudo_interactive" > "$ROUND_DIR/gpu/gpu_query_mode.txt"
else
  nvidia-smi | tee "$ROUND_DIR/gpu/nvidia_smi_direct.txt"
  echo "direct_fallback_noninteractive" > "$ROUND_DIR/gpu/gpu_query_mode.txt"
fi

nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu \
  --format=csv,noheader \
  | tee "$ROUND_DIR/gpu/gpu_inventory.csv"
```

---

## 三、入口检查

```bash
grep -q 'REFINE1_PROTOCOL_LOCKED' \
  "$STAGE6R1_ROUNDS/stage6r1_1_refinement_lock/reports/refine1_protocol_summary.md"

test -f "$STAGE6R1_PROTOCOL/locks/refine1_input_lock.json"
test -f "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json"
test -f "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml"
test -f "$STAGE6R1_PROTOCOL/seed_registry/refine1_policy_seed_registry.csv"
test -f "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet"
test -f "$STAGE6_WEIGHTS/chunk_weights/linear_sarm_equiv.parquet"
```

重算协议 SHA：

```bash
sha256sum -c \
  "$STAGE6R1_PROTOCOL/locks/REFINE1_PROTOCOL_SHA256SUMS.txt" \
  | tee "$ROUND_DIR/checksums/refine1_protocol_recheck.txt"
```

---

## 四、创建 6 个新初始化

任务 × seed：

```text
transport_recovery × 3 seeds
transport_dual_order × 3 seeds
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/create_policy_initializations.py" \
  --base-config "$STAGE6R1_PROTOCOL/configs/base_policy_stage6.yaml" \
  --tasks "$REFINE_TASKS" \
  --seeds "$NEW_POLICY_SEEDS" \
  --output-dir "$STAGE6R1_PROTOCOL/initializations" \
  --manifest "$ROUND_DIR/manifests/policy_initialization_manifest.tsv"
```

若脚本只接受空格分隔 seed，改为：

```bash
--seeds 20260912 20260913 20260914
```

禁止为两个方法分别创建初始化。

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["ROUND_DIR"],
    "manifests/policy_initialization_manifest.tsv"
)
d = pd.read_csv(p, sep="\t")
assert len(d) == 6, len(d)
assert not d.duplicated(["task_id","policy_seed"]).any()
assert d["checkpoint_sha256"].nunique() == 6
assert d["load_ok"].all()
print("REFINE1_INITIALIZATIONS_6_OK")
PY
```

若 manifest 字段名不同，Agent 只调整检查字段映射，不改变生成逻辑。

checkpoint 不进入 ZIP。

---

## 五、生成 12-job 训练矩阵

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_policy_job_matrix.py" \
  --tasks "$REFINE_TASKS" \
  --methods "$REFINE_METHODS" \
  --seeds "$NEW_POLICY_SEEDS" \
  --mode full \
  --protocol "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" \
  --dataset-root "$STAGE6_DATA" \
  --weights-root "$STAGE6_WEIGHTS/chunk_weights" \
  --initializations-root "$STAGE6R1_PROTOCOL/initializations" \
  --output-root "$STAGE6R1_TRAIN/jobs" \
  --output "$ROUND_DIR/tables/refine1_training_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

如果原脚本硬编码 4 方法或旧 seed，创建 wrapper：

```text
tools/stage6_refine1/build_refine1_training_jobs.py
```

wrapper 只负责筛选方法、seed 和输出路径，最终训练命令仍调用：

```text
tools/stage6/train_weighted_policy.py
```

`refine1_training_jobs.tsv` 必须 12 行。

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["ROUND_DIR"],
    "tables/refine1_training_jobs.tsv"
)
d = pd.read_csv(p, sep="\t")
assert len(d) == 12, len(d)
assert set(d["task_id"]) == {
    "transport_recovery",
    "transport_dual_order"
}
assert set(d["method"]) == {
    "linear_sarm_equiv",
    "pathgraph_reward_v1_locked"
}
assert set(d["policy_seed"]) == {
    20260912, 20260913, 20260914
}
assert not d.duplicated(["task_id","method","policy_seed"]).any()

for (task, seed), g in d.groupby(["task_id","policy_seed"]):
    assert len(g) == 2
    assert g["init_sha256"].nunique() == 1, (task, seed)
    assert g["data_seed"].nunique() == 1, (task, seed)
    assert g["total_optimizer_steps"].nunique() == 1, (task, seed)
    assert g["effective_batch_size"].nunique() == 1, (task, seed)

assert (d["total_optimizer_steps"] == 2000).all()
assert (d["effective_batch_size"] == 32).all()
print("REFINE1_JOB_MATRIX_12_OK")
PY
```

---

## 六、每个训练 job 的固定命令

PathGraph 示例：

```bash
export CUDA_VISIBLE_DEVICES=<assigned_gpu>

"$PYTHON_BIN" "$STAGE6_TOOLS/train_weighted_policy.py" \
  --task transport_recovery \
  --method pathgraph_reward_v1_locked \
  --policy-seed 20260912 \
  --base-config "$STAGE6R1_PROTOCOL/configs/base_policy_stage6.yaml" \
  --protocol "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" \
  --dataset-root "$STAGE6_DATA" \
  --weight-file "$STAGE6_WEIGHTS/chunk_weights/pathgraph_reward_v1_locked.parquet" \
  --weight-lock "$STAGE6_WEIGHTS/weight_selection_lock.json" \
  --init-checkpoint "$STAGE6R1_PROTOCOL/initializations/transport_recovery/seed_20260912/init.pt" \
  --output-dir "$STAGE6R1_TRAIN/jobs/transport_recovery__pathgraph_reward_v1_locked__s20260912" \
  --max-optimizer-steps 2000 \
  --device cuda:0 \
  --resume auto
```

Comparator 示例：

```bash
export CUDA_VISIBLE_DEVICES=<assigned_gpu>

"$PYTHON_BIN" "$STAGE6_TOOLS/train_weighted_policy.py" \
  --task transport_recovery \
  --method linear_sarm_equiv \
  --policy-seed 20260912 \
  --base-config "$STAGE6R1_PROTOCOL/configs/base_policy_stage6.yaml" \
  --protocol "$STAGE6R1_PROTOCOL/configs/refine1_policy_training_protocol.yaml" \
  --dataset-root "$STAGE6_DATA" \
  --weight-file "$STAGE6_WEIGHTS/chunk_weights/linear_sarm_equiv.parquet" \
  --weight-lock "$STAGE6_WEIGHTS/weight_selection_lock.json" \
  --init-checkpoint "$STAGE6R1_PROTOCOL/initializations/transport_recovery/seed_20260912/init.pt" \
  --output-dir "$STAGE6R1_TRAIN/jobs/transport_recovery__linear_sarm_equiv__s20260912" \
  --max-optimizer-steps 2000 \
  --device cuda:0 \
  --resume auto
```

每个 job 开始时必须记录：

```text
task
method
policy_seed
model_seed
data_seed
init SHA256
dataset manifest SHA256
weight file SHA256
training protocol SHA256
CUDA used
GPU name
optimizer step budget
```

---

## 七、多 GPU 并行启动

优先让同一 `(task, seed)` 的两个方法同时运行。

例如 6 张 GPU 时，一波运行一个 seed 的全部任务和方法：

```text
GPU 0：recovery / linear / seed 12
GPU 1：recovery / pathgraph / seed 12
GPU 2：dual_order / linear / seed 12
GPU 3：dual_order / pathgraph / seed 12
GPU 4：下一 seed 的 recovery / linear
GPU 5：下一 seed 的 recovery / pathgraph
```

直接执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/refine1_training_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/refine1_training_job_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_refine1_training.log"
```

每 10 分钟记录：

```bash
while pgrep -f "train_weighted_policy.py" >/dev/null; do
  date -Iseconds
  nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
  sleep 600
done >> "$ROUND_DIR/logs/training_progress.log" 2>&1
```

---

## 八、失败处理

### OOM

只允许：

1. 同一 task 的两方法和三个新 seed 使用同一 micro batch；
2. 增加 gradient accumulation；
3. effective batch 保持 32；
4. optimizer steps 保持 2000；
5. 只重跑受影响 task 的 6 个 job。

不得只给 PathGraph 降 batch。

### 中断

使用：

```text
--resume auto
```

只补跑失败 job。

### 共用训练代码 bug

若 bug 影响 weighted loss 共用路径：

- 修复后记录 diff；
- 同一 `(task, seed)` 的两个方法一起重跑；
- 不重跑无影响的其他 job。

---

## 九、训练完成检查

创建：

```text
tools/stage6_refine1/summarize_refine1_training.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/summarize_refine1_training.py" \
  --job-table "$ROUND_DIR/tables/refine1_training_jobs.tsv" \
  --job-root "$STAGE6R1_TRAIN/jobs" \
  --status-table "$ROUND_DIR/tables/refine1_training_job_status.tsv" \
  --output "$ROUND_DIR/metrics/refine1_training_gate.json" \
  --summary "$ROUND_DIR/reports/refine1_training_summary.md" \
  --checkpoint-manifest "$ROUND_DIR/manifests/checkpoint_manifest.tsv"
```

必须满足：

```text
12 / 12 jobs succeeded
12 / 12 CUDA used = true
每个 job optimizer steps = 2000
同 task/seed 两方法 init SHA 相同
同 task/seed 两方法 data seed 相同
同 task/seed 两方法 effective batch 相同
test rollout count = 0
reward parameters unchanged
gamma unchanged
```

允许状态：

```text
REFINE1_TRAINING_COMPLETE
RETRY_FAILED_JOBS
FAIRNESS_PROTOCOL_VIOLATION
```

---

## 十、本轮 ZIP

checkpoint 不打包，只保留 manifest、日志、训练曲线和最终指标。

```bash
export ZIP_NAME="stage6r1_2_new_seed_training.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 6R1.2 状态：REFINE1_TRAINING_COMPLETE
训练：12 / 12
CUDA：12 / 12
新增 seeds：20260912, 20260913, 20260914
任务：2
方法：2
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6R1.3
```

**核心点：只增加三个独立训练 seed，用完全相同的冻结协议复现主比较。**

---

# 阶段 6R1.3：Validation-only Checkpoint 选择与锁定

## 一、总体上要干什么

从 12 个新训练 job 中，只根据冻结的 validation 指标选择 checkpoint：

```text
val_action_loss_unweighted
```

禁止查看或使用 test rollout。

本轮输出 12 个 selected checkpoint 的路径、step 和 SHA256；checkpoint 本体不进入 ZIP。

本轮状态：

```text
REFINE1_SELECTION_LOCKED
```

本轮 ZIP：

```text
stage6r1_3_validation_selection.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage6r1_3_validation_selection"
export ROUND_DIR="$STAGE6R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6R1_TRAIN/selection"
```

本轮 checkpoint load 可用 CPU；若使用 GPU，先按通用规范执行提权 `nvidia-smi`。

---

## 三、入口检查

```bash
grep -q 'REFINE1_TRAINING_COMPLETE' \
  "$STAGE6R1_ROUNDS/stage6r1_2_new_seed_training/reports/refine1_training_summary.md"

test -f \
  "$STAGE6R1_ROUNDS/stage6r1_2_new_seed_training/tables/refine1_training_jobs.tsv"
```

确认没有 test 产物：

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os
root = Path(os.environ["STAGE6R1_TRAIN"]) / "jobs"
bad = []
for p in root.rglob("*"):
    if p.is_file() and any(x in p.name.lower() for x in [
        "test_rollout", "test_success", "frozen_test"
    ]):
        bad.append(str(p))
assert not bad, bad
print("NO_TEST_USED_BEFORE_SELECTION")
PY
```

这只是确保 selection 没有误用 test，不做额外仓库审计。

---

## 四、选择 checkpoint

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/select_policy_checkpoints.py" \
  --job-root "$STAGE6R1_TRAIN/jobs" \
  --job-table "$STAGE6R1_ROUNDS/stage6r1_2_new_seed_training/tables/refine1_training_jobs.tsv" \
  --metric val_action_loss_unweighted \
  --mode min \
  --output "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  --lock "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  --manifest "$ROUND_DIR/manifests/checkpoint_manifest.tsv"
```

若脚本输出字段为 `val_action_loss`，必须确认它对应未加权 validation loss；若旧代码同时保存 weighted/unweighted，明确选择 unweighted 字段。

输出表至少包含：

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

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["STAGE6R1_TRAIN"],
    "selection/refine1_checkpoint_selection.csv"
)
d = pd.read_csv(p)
assert len(d) == 12, len(d)
assert not d.duplicated(["task_id","method","policy_seed"]).any()
assert set(d["policy_seed"]) == {20260912,20260913,20260914}
assert set(d["method"]) == {
    "linear_sarm_equiv",
    "pathgraph_reward_v1_locked"
}
assert (d["selection_split"] == "val").all()
assert (~d["test_used"].astype(bool)).all()
assert d["checkpoint_sha256"].notna().all()
print("REFINE1_SELECTION_12_OK")
PY
```

---

## 五、逐 checkpoint 加载检查

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/check_selected_policy_checkpoints.py" \
  --selection "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  --output "$ROUND_DIR/metrics/refine1_selected_checkpoint_load_check.json"
```

只做：

```text
文件存在
SHA256 匹配
state_dict 可读取
模型键和 shape 可加载
```

不运行 test rollout。

---

## 六、冻结 selection lock

补充输入 SHA：

```bash
sha256sum \
  "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" \
  | tee "$ROUND_DIR/checksums/refine1_selection_sha256.txt"
```

创建 gate：

```text
tools/stage6_refine1/decide_refine1_selection_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/decide_refine1_selection_gate.py" \
  --selection "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  --selection-lock "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  --load-check "$ROUND_DIR/metrics/refine1_selected_checkpoint_load_check.json" \
  --expected-count 12 \
  --output "$ROUND_DIR/metrics/refine1_selection_gate.json" \
  --report "$ROUND_DIR/reports/refine1_selection_summary.md"
```

必须满足：

```text
12 / 12 selected
12 / 12 loadable
selection split = val
test_used = false
same selection metric across methods
reward/gamma/protocol unchanged
```

允许状态：

```text
REFINE1_SELECTION_LOCKED
RESELECT_FROM_VALIDATION
CHECKPOINT_LOAD_FAILURE
```

---

## 七、生成验证曲线图

调用原训练汇总脚本或创建轻量绘图脚本：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/plot_refine1_validation_curves.py" \
  --job-root "$STAGE6R1_TRAIN/jobs" \
  --selection "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  --output-dir "$ROUND_DIR/figures" \
  --summary "$ROUND_DIR/tables/selected_validation_metrics.csv"
```

每个 task 单独一张图，不把多个图塞入 subplot。

---

## 八、本轮 ZIP

```bash
cp "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  "$ROUND_DIR/tables/"
cp "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  "$ROUND_DIR/configs/"

export ZIP_NAME="stage6r1_3_validation_selection.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 6R1.3 状态：REFINE1_SELECTION_LOCKED
选中 checkpoint：12 / 12
selection split：val
test used：false
load check：12 / 12
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6R1.4
```

**核心点：新 seed 继续只按 validation 选 checkpoint，test 在 selection lock 后才允许运行。**

---

# 阶段 6R1.4：新增 Seed 的冻结配对闭环评估

## 一、总体上要干什么

使用 Stage 6 原有且已冻结的 evaluation seed registry，对新增三个 policy seed 运行相同闭环条件。

方法：

```text
linear_sarm_equiv
pathgraph_reward_v1_locked
```

条件：

```text
transport_dual_order:
  A_first
  B_first

transport_recovery:
  natural
  drop_regrasp
  gripper_reopen
```

每个 `(task condition, method, policy seed)`：

```text
50 rollouts
```

总 rollout：

\[
(2 + 3)
\times
2
\times
3
\times
50
=
1500
\]

本轮不重跑旧 seed。

本轮状态：

```text
REFINE1_FROZEN_EVALUATION_COMPLETE
```

本轮 ZIP：

```text
stage6r1_4_frozen_paired_evaluation.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage6r1_4_frozen_paired_evaluation"
export ROUND_DIR="$STAGE6R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6R1_EVAL"/{jobs,rollouts,metrics,locks}
```

提权查看 GPU：

```bash
if sudo -n nvidia-smi > "$ROUND_DIR/gpu/nvidia_smi_sudo.txt" 2>&1; then
  echo "sudo_noninteractive" > "$ROUND_DIR/gpu/gpu_query_mode.txt"
elif [ -t 0 ]; then
  sudo nvidia-smi | tee "$ROUND_DIR/gpu/nvidia_smi_sudo_interactive.txt"
  echo "sudo_interactive" > "$ROUND_DIR/gpu/gpu_query_mode.txt"
else
  nvidia-smi | tee "$ROUND_DIR/gpu/nvidia_smi_direct.txt"
  echo "direct_fallback_noninteractive" > "$ROUND_DIR/gpu/gpu_query_mode.txt"
fi
```

---

## 三、入口检查与评估锁复制

```bash
grep -q 'REFINE1_SELECTION_LOCKED' \
  "$STAGE6R1_ROUNDS/stage6r1_3_validation_selection/reports/refine1_selection_summary.md"

test -f "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv"
test -f "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json"
test -f "$STAGE6_EVAL/locks/policy_evaluation_lock.json"
test -f "$STAGE6_EVAL/locks/primary_comparator_lock.json"
```

定位原 seed registry：

```bash
if [ -f "$STAGE6_EVAL/locks/evaluation_seed_registry.csv" ]; then
  export OLD_EVAL_SEED_REGISTRY="$STAGE6_EVAL/locks/evaluation_seed_registry.csv"
elif [ -f "$STAGE6_EVAL/locks/test_seed_registry.csv" ]; then
  export OLD_EVAL_SEED_REGISTRY="$STAGE6_EVAL/locks/test_seed_registry.csv"
else
  echo "Frozen evaluation seed registry missing."
  exit 2
fi
```

复制锁：

```bash
cp "$STAGE6_EVAL/locks/policy_evaluation_lock.json" \
  "$STAGE6R1_EVAL/locks/parent_policy_evaluation_lock.json"
cp "$STAGE6_EVAL/locks/primary_comparator_lock.json" \
  "$STAGE6R1_EVAL/locks/primary_comparator_lock.json"
cp "$OLD_EVAL_SEED_REGISTRY" \
  "$STAGE6R1_EVAL/locks/evaluation_seed_registry.csv"
```

如果 intervention protocol 文件存在，复制其原版本：

```bash
if [ -f "$STAGE6_EVAL/locks/intervention_protocol_v1.yaml" ]; then
  cp "$STAGE6_EVAL/locks/intervention_protocol_v1.yaml" \
    "$STAGE6R1_EVAL/locks/intervention_protocol_v1.yaml"
fi
```

不得修改 seed registry 或 intervention 参数。

---

## 四、创建评估 job 矩阵

优先复用：

```text
tools/stage6/build_evaluation_job_matrix.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_evaluation_job_matrix.py" \
  --checkpoint-selection "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection.csv" \
  --seed-registry "$STAGE6R1_EVAL/locks/evaluation_seed_registry.csv" \
  --intervention-protocol "$STAGE6R1_EVAL/locks/intervention_protocol_v1.yaml" \
  --shard-size 10 \
  --output "$ROUND_DIR/tables/refine1_evaluation_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands" \
  --job-root "$STAGE6R1_EVAL/jobs"
```

如果原脚本从 checkpoint selection 自动读取方法和 seed，它应自然产生：

```text
5 conditions × 2 methods × 3 policy seeds × 5 shards = 150 jobs
```

每 shard 10 rollout。

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["ROUND_DIR"],
    "tables/refine1_evaluation_jobs.tsv"
)
d = pd.read_csv(p, sep="\t")
assert len(d) == 150, len(d)
assert set(d["method"]) == {
    "linear_sarm_equiv",
    "pathgraph_reward_v1_locked"
}
assert set(d["policy_seed"]) == {
    20260912, 20260913, 20260914
}
assert (d["rollout_count"] == 10).all()
assert not d.duplicated([
    "task_id","condition","method","policy_seed","rollout_start"
]).any()

total = int(d["rollout_count"].sum())
assert total == 1500, total
print("REFINE1_EVAL_MATRIX_1500_OK")
PY
```

如果 intervention protocol 文件在原 Stage 6 使用不同名称，使用原文件路径；禁止重写条件定义。

---

## 五、多 GPU 并行评估

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/refine1_evaluation_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/refine1_evaluation_job_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_refine1_evaluation.log"
```

并行要求：

```text
每个 shard 独立输出
相同 task/condition/rollout_index 跨方法使用相同 env seed
不同 shard 的 rollout 范围不重叠
失败 shard 只补跑该 shard
不得减少某方法的 rollout 数
不得 test-time reward shaping
```

评估期间记录 GPU：

```bash
while pgrep -f "evaluate_policy_checkpoint.py" >/dev/null; do
  date -Iseconds
  nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
  sleep 300
done >> "$ROUND_DIR/logs/evaluation_progress.log" 2>&1
```

---

## 六、合并 1500 条新 rollout

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/merge_policy_evaluation_shards.py" \
  --job-table "$ROUND_DIR/tables/refine1_evaluation_jobs.tsv" \
  --job-root "$STAGE6R1_EVAL/jobs" \
  --output "$STAGE6R1_EVAL/rollouts/refine1_new_seed_rollouts.csv" \
  --summary "$ROUND_DIR/metrics/refine1_evaluation_merge_summary.json" \
  --duplicates "$ROUND_DIR/tables/refine1_evaluation_duplicate_keys.csv"
```

唯一键：

```text
task_id + condition + method + policy_seed + rollout_index
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["STAGE6R1_EVAL"],
    "rollouts/refine1_new_seed_rollouts.csv"
)
d = pd.read_csv(p)
assert len(d) == 1500, len(d)
key = ["task_id","condition","method","policy_seed","rollout_index"]
assert not d.duplicated(key).any()
assert set(d["policy_seed"]) == {20260912,20260913,20260914}
assert set(d["method"]) == {
    "linear_sarm_equiv",
    "pathgraph_reward_v1_locked"
}
assert d["success"].notna().all()
print("REFINE1_NEW_ROLLOUTS_1500_OK")
PY
```

若结果字段不是 `success`，改为实际布尔成功字段；不要改变成功判定。

---

## 七、计算新 seed 冻结指标

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/compute_policy_evaluation_metrics.py" \
  --rollouts "$STAGE6R1_EVAL/rollouts/refine1_new_seed_rollouts.csv" \
  --output-seed "$STAGE6R1_EVAL/metrics/refine1_policy_metrics_by_seed.csv" \
  --output-aggregate "$STAGE6R1_EVAL/metrics/refine1_policy_metrics_aggregate.csv" \
  --output-condition "$ROUND_DIR/tables/refine1_policy_metrics_by_condition.csv" \
  --figures-dir "$ROUND_DIR/figures" \
  --report "$ROUND_DIR/reports/refine1_frozen_policy_evaluation_summary.md"
```

如果现有脚本 CLI 不同，Agent 查看 `--help` 后映射参数，不重写指标定义。

指标至少包括：

```text
success_rate
completion_mean
recovery_success_rate
worst_order_success
fixed_order_success
long_horizon_completion
order_gap
```

---

## 八、冻结本轮评估

创建：

```text
$STAGE6R1_EVAL/locks/refine1_policy_evaluation_lock.json
```

至少写入：

```json
{
  "locked": true,
  "checkpoint_selection_lock_sha256": "",
  "parent_evaluation_seed_registry_sha256": "",
  "new_rollouts_sha256": "",
  "test_used_for_selection": false,
  "paired_evaluation": true,
  "policy_seeds": [20260912, 20260913, 20260914],
  "methods": [
    "linear_sarm_equiv",
    "pathgraph_reward_v1_locked"
  ],
  "rollout_count": 1500
}
```

创建脚本：

```text
tools/stage6_refine1/freeze_refine1_evaluation.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/freeze_refine1_evaluation.py" \
  --selection-lock "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  --seed-registry "$STAGE6R1_EVAL/locks/evaluation_seed_registry.csv" \
  --rollouts "$STAGE6R1_EVAL/rollouts/refine1_new_seed_rollouts.csv" \
  --output "$STAGE6R1_EVAL/locks/refine1_policy_evaluation_lock.json"
```

---

## 九、本轮 gate

实现：

```text
tools/stage6_refine1/decide_refine1_evaluation_gate.py
```

必须满足：

```text
1500 / 1500 rollout 完成
12 selected checkpoints 全部参与评估
paired seed registry 保持不变
duplicate keys = 0
missing keys = 0
all metrics finite
test_used_for_selection = false
reward/gamma unchanged
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/decide_refine1_evaluation_gate.py" \
  --jobs "$ROUND_DIR/tables/refine1_evaluation_jobs.tsv" \
  --job-status "$ROUND_DIR/tables/refine1_evaluation_job_status.tsv" \
  --merge-summary "$ROUND_DIR/metrics/refine1_evaluation_merge_summary.json" \
  --evaluation-lock "$STAGE6R1_EVAL/locks/refine1_policy_evaluation_lock.json" \
  --output "$ROUND_DIR/metrics/refine1_evaluation_gate.json" \
  --report "$ROUND_DIR/reports/refine1_evaluation_gate.md"
```

允许状态：

```text
REFINE1_FROZEN_EVALUATION_COMPLETE
RETRY_MISSING_EVAL_SHARDS
EVALUATION_PAIRING_ERROR
```

---

## 十、本轮 ZIP

完整 rollout 文件默认不打包，写入 manifest：

```bash
printf "path\tsize_bytes\tsha256\treason\n" \
  > "$ROUND_DIR/manifests/large_file_manifest.tsv"

ROLLOUT_PATH="$STAGE6R1_EVAL/rollouts/refine1_new_seed_rollouts.csv"
ROLLOUT_SIZE=$(stat -c '%s' "$ROLLOUT_PATH")
ROLLOUT_SHA=$(sha256sum "$ROLLOUT_PATH" | awk '{print $1}')

printf "%s\t%s\t%s\t%s\n" \
  "$ROLLOUT_PATH" "$ROLLOUT_SIZE" "$ROLLOUT_SHA" \
  "full rollout table omitted from ZIP" \
  >> "$ROUND_DIR/manifests/large_file_manifest.tsv"
```

生成 ZIP：

```bash
export ZIP_NAME="stage6r1_4_frozen_paired_evaluation.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 最终回复：

```text
阶段 6R1.4 状态：REFINE1_FROZEN_EVALUATION_COMPLETE
rollouts：1500 / 1500
policy seeds：3
methods：2
paired evaluation：true
test used for selection：false
ZIP：<绝对路径>
SHA256：<hash>
下一步：阶段 6R1.5
```

**核心点：只用原 Stage 6 的冻结测试条件和环境 seed，评估三个新增 policy seed。**

---

# 阶段 6R1.5：六 Seed 汇总、Bootstrap 与 G3-R1 最终决策

## 一、总体上要干什么

将：

```text
旧 policy seeds：20260909, 20260910, 20260911
新 policy seeds：20260912, 20260913, 20260914
```

合并为 6-seed 证据。

只比较已锁定主比较对：

```text
pathgraph_reward_v1_locked
vs
linear_sarm_equiv
```

先独立检查新 seed block 是否达到 2/3 改善，再计算 6 seed combined gain 和 bootstrap。

本轮作出最终决策：

```text
GO_STAGE7
NARROW_TO_REWARD_ONLY
REPAIR_REFINE1_INFRA
```

本轮 ZIP：

```text
stage6r1_5_g3r1_decision.zip
```

总 ZIP：

```text
stage6_refine1_complete.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage6_refine1/stage6_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage6r1_5_g3r1_decision"
export ROUND_DIR="$STAGE6R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE6R1_FINAL"/{configs,locks,metrics,tables,figures,reports,manifests}
```

本轮以 CPU 统计为主。若需要补渲染，不阻塞 G3-R1；先按通用规范查询 GPU。

---

## 三、定位旧、新 rollout

优先使用旧 Stage 6 的完整冻结 rollout：

```bash
if [ -f "$STAGE6_EVAL/rollouts/all_policy_rollouts.csv" ]; then
  export OLD_ROLLOUTS="$STAGE6_EVAL/rollouts/all_policy_rollouts.csv"
elif [ -f "$STAGE6_EVAL/frozen_test_rollouts.csv" ]; then
  export OLD_ROLLOUTS="$STAGE6_EVAL/frozen_test_rollouts.csv"
else
  echo "Missing original Stage 6 frozen rollouts."
  echo "Recover the large rollout file referenced by the Stage 6 manifest."
  exit 2
fi

export NEW_ROLLOUTS="$STAGE6R1_EVAL/rollouts/refine1_new_seed_rollouts.csv"

test -f "$OLD_ROLLOUTS"
test -f "$NEW_ROLLOUTS"
test -f "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json"
test -f "$STAGE6R1_EVAL/locks/refine1_policy_evaluation_lock.json"
```

不得用旧 `evaluation_summary.csv` 替代原始配对 rollout 做 bootstrap。

---

## 四、合并并过滤主比较对

实现：

```text
tools/stage6_refine1/merge_old_new_seed_rollouts.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/merge_old_new_seed_rollouts.py" \
  --old-rollouts "$OLD_ROLLOUTS" \
  --new-rollouts "$NEW_ROLLOUTS" \
  --methods linear_sarm_equiv,pathgraph_reward_v1_locked \
  --old-seeds 20260909,20260910,20260911 \
  --new-seeds 20260912,20260913,20260914 \
  --output "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv" \
  --summary "$ROUND_DIR/metrics/combined_rollout_summary.json" \
  --duplicates "$ROUND_DIR/tables/combined_duplicate_keys.csv"
```

预期记录数：

```text
旧主比较对：
  5 conditions × 2 methods × 3 seeds × 50 = 1500

新主比较对：
  5 conditions × 2 methods × 3 seeds × 50 = 1500

合计：
  3000
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["STAGE6R1_FINAL"],
    "tables/combined_six_seed_rollouts.csv"
)
d = pd.read_csv(p)
assert len(d) == 3000, len(d)
assert set(d["policy_seed"]) == {
    20260909, 20260910, 20260911,
    20260912, 20260913, 20260914
}
assert set(d["method"]) == {
    "linear_sarm_equiv",
    "pathgraph_reward_v1_locked"
}
key = ["task_id","condition","method","policy_seed","rollout_index"]
assert not d.duplicated(key).any()
print("COMBINED_SIX_SEED_ROLLOUTS_3000_OK")
PY
```

---

## 五、生成六 seed 结果表

复用原脚本：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/build_policy_result_tables.py" \
  --rollouts "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv" \
  --comparator-lock "$STAGE6_EVAL/locks/primary_comparator_lock.json" \
  --output-dir "$ROUND_DIR/tables" \
  --summary "$ROUND_DIR/reports/refine1_policy_result_summary.md"
```

必须生成：

```text
result_by_task_condition_method_seed.csv
result_by_task_method.csv
pathgraph_vs_locked_comparator.csv
seed_level_effects.csv
```

如果原脚本不输出 `seed_level_effects.csv`，调用：

```text
tools/stage6_refine1/build_refine1_seed_effects.py
```

字段：

```text
policy_seed
pathgraph_value
comparator_value
difference
improved
ceiling_tie
degraded
seed_block
```

定义：

```text
improved:
  difference > 0

ceiling_tie:
  difference == 0
  且 pathgraph_value >= 0.95
  且 comparator_value >= 0.95

degraded:
  difference < 0

seed_block:
  old 或 new
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/build_refine1_seed_effects.py" \
  --rollouts "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv" \
  --method-a pathgraph_reward_v1_locked \
  --method-b linear_sarm_equiv \
  --old-seeds 20260909,20260910,20260911 \
  --new-seeds 20260912,20260913,20260914 \
  --output "$ROUND_DIR/tables/refine1_seed_level_effects.csv"
```

---

## 六、计算新 seed 独立复现结果

实现：

```text
tools/stage6_refine1/summarize_new_seed_replication.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/summarize_new_seed_replication.py" \
  --seed-effects "$ROUND_DIR/tables/refine1_seed_level_effects.csv" \
  --new-seeds 20260912,20260913,20260914 \
  --output "$ROUND_DIR/metrics/new_seed_replication.json" \
  --report "$ROUND_DIR/reports/new_seed_replication.md"
```

输出至少包含：

```json
{
  "new_seed_count": 3,
  "new_seed_improved_count": 0,
  "new_seed_ceiling_tie_count": 0,
  "new_seed_degraded_count": 0,
  "per_seed_difference": {}
}
```

不得把 ceiling tie 计为 improved。

---

## 七、六 seed 层级配对 Bootstrap

执行：

```bash
"$PYTHON_BIN" "$STAGE6_TOOLS/hierarchical_paired_bootstrap.py" \
  --rollouts "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv" \
  --method-a pathgraph_reward_v1_locked \
  --comparator-lock "$STAGE6_EVAL/locks/primary_comparator_lock.json" \
  --metrics success,completion,recovery_success,worst_order_success,order_gap \
  --resamples 5000 \
  --seed 20261021 \
  --output "$ROUND_DIR/tables/refine1_bootstrap_effects.csv" \
  --distribution-dir "$ROUND_DIR/metrics/bootstrap_distributions"
```

统计单位：

```text
第一层：policy_seed
第二层：配对 rollout_index / env_seed
```

聚合保持与 Stage 6 一致：

```text
graph_task_success
recovery_success
worst_order_success
fixed_order_success
long_horizon_completion
order_gap
```

不允许把 3000 rollout 当成完全独立样本。

---

## 八、计算 combined evidence

实现：

```text
tools/stage6_refine1/build_g3_refine1_evidence.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/build_g3_refine1_evidence.py" \
  --rollouts "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv" \
  --seed-effects "$ROUND_DIR/tables/refine1_seed_level_effects.csv" \
  --bootstrap "$ROUND_DIR/tables/refine1_bootstrap_effects.csv" \
  --new-seed-replication "$ROUND_DIR/metrics/new_seed_replication.json" \
  --output "$STAGE6R1_FINAL/metrics/g3_refine1_evidence.json" \
  --report "$STAGE6R1_FINAL/reports/g3_refine1_evidence_summary.md"
```

输出至少包含：

```json
{
  "new_seed_improved_count": 0,
  "new_seed_ceiling_tie_count": 0,
  "combined_graph_task_success_gain": 0.0,
  "combined_graph_task_success_ci95": [0.0, 0.0],
  "combined_recovery_success_gain": 0.0,
  "combined_worst_order_success_gain": 0.0,
  "combined_long_horizon_completion_gain": 0.0,
  "combined_fixed_order_drop": 0.0,
  "combined_seed_count": 6,
  "paired_evaluation": true,
  "reward_retuned_after_test": false
}
```

所有值从 rollout 和统计表计算，禁止固定填写。

---

## 九、执行 G3-R1 决策

实现：

```text
tools/stage6_refine1/decide_g3_refine1.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/decide_g3_refine1.py" \
  --evidence "$STAGE6R1_FINAL/metrics/g3_refine1_evidence.json" \
  --rule "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" \
  --protocol-lock "$STAGE6R1_PROTOCOL/locks/refine1_input_lock.json" \
  --selection-lock "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  --evaluation-lock "$STAGE6R1_EVAL/locks/refine1_policy_evaluation_lock.json" \
  --output "$STAGE6R1_FINAL/metrics/g3_refine1_decision.json" \
  --report "$STAGE6R1_FINAL/reports/g3_refine1_decision.md"
```

### `GO_STAGE7`

必须同时满足：

```text
new_seed_improved_count >= 2 / 3
combined_graph_task_success_gain >= 0.05
combined_fixed_order_drop <= 0.05

并且至少一个：
combined_recovery_success_gain >= 0.08
OR
combined_worst_order_success_gain >= 0.08
OR
combined_long_horizon_completion_gain >= 0.05

paired_evaluation == true
reward_retuned_after_test == false
```

### `NARROW_TO_REWARD_ONLY`

实验完整，但任一主要门槛未达到。

此状态表示：

```text
不再继续增加 seed 或调 Stage 6
保留 Stage 3–5 的图奖励中间机制结果
下游 RA-BC 结果作为有限或混合证据
后续论文主张收窄到 graph-structured reward modeling
```

### `REPAIR_REFINE1_INFRA`

仅用于：

```text
训练 job 缺失
checkpoint 无法加载
配对 rollout 缺失
seed registry 不匹配
结果文件损坏
```

只补跑技术上失败的 job，不改变实验设计。

---

## 十、冻结结果

复制结果：

```bash
cp "$STAGE6R1_PROTOCOL/locks/g3_refine1_rule.json" \
  "$STAGE6R1_FINAL/configs/"
cp "$STAGE6R1_PROTOCOL/locks/refine1_input_lock.json" \
  "$STAGE6R1_FINAL/locks/"
cp "$STAGE6R1_TRAIN/selection/refine1_checkpoint_selection_lock.json" \
  "$STAGE6R1_FINAL/locks/"
cp "$STAGE6R1_EVAL/locks/refine1_policy_evaluation_lock.json" \
  "$STAGE6R1_FINAL/locks/"
cp "$ROUND_DIR/tables/refine1_seed_level_effects.csv" \
  "$STAGE6R1_FINAL/tables/"
cp "$ROUND_DIR/tables/refine1_bootstrap_effects.csv" \
  "$STAGE6R1_FINAL/tables/"
cp "$ROUND_DIR/metrics/new_seed_replication.json" \
  "$STAGE6R1_FINAL/metrics/"
```

生成冻结文件：

```bash
cat > "$STAGE6R1_FINAL/FROZEN.md" <<EOF
milestone = M4_REFINE1
decision = $(python - <<'PY'
import json, os
p = os.path.join(
    os.environ["STAGE6R1_FINAL"],
    "metrics/g3_refine1_decision.json"
)
print(json.load(open(p))["decision"])
PY
)
reward_retuned_after_test = false
gamma_changed = false
training_budget_changed = false
new_policy_seeds = 20260912,20260913,20260914
checkpoint_packaging = omitted_by_default
statistics = six_seed_hierarchical_paired_bootstrap
EOF
```

计算 SHA：

```bash
find "$STAGE6R1_FINAL" -type f \
  ! -name '*.pt' \
  ! -name '*.pth' \
  ! -name '*.ckpt' \
  ! -name '*.safetensors' \
  -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$STAGE6R1_FINAL/M4_REFINE1_SHA256SUMS.txt"
```

---

## 十一、本轮 ZIP

完整 rollout 和 checkpoint 不进入 ZIP：

```bash
cat > "$ROUND_DIR/manifests/large_file_manifest.tsv" <<EOF
path	size_bytes	sha256	reason
$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv	$(stat -c '%s' "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv")	$(sha256sum "$STAGE6R1_FINAL/tables/combined_six_seed_rollouts.csv" | awk '{print $1}')	full paired rollout table omitted
EOF
```

生成本轮 ZIP：

```bash
export ZIP_NAME="stage6r1_5_g3r1_decision.zip"

"$PYTHON_BIN" "$STAGE6_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 十二、生成总 ZIP

总 ZIP 收集：

```text
refine protocol locks
new seed registry
12-job training summary
validation selection lock
new frozen evaluation summary
six-seed result tables
bootstrap results
G3-R1 decision
各轮 ZIP 的 SHA256
large-file manifests
```

不收集 checkpoint 和完整 rollout。

执行：

```bash
"$PYTHON_BIN" "$STAGE6R1_TOOLS/package_refine1_complete.py" \
  --root "$STAGE6R1_ROOT" \
  --round-zip-dir "$STAGE6R1_DOWNLOADS" \
  --output "$STAGE6R1_DOWNLOADS/stage6_refine1_complete.zip" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE6R1_DOWNLOADS/stage6_refine1_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage6_refine1_complete_unzip_test.txt"

sha256sum "$STAGE6R1_DOWNLOADS/stage6_refine1_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage6_refine1_complete.sha256"
```

总 ZIP 生成后，不删除五个轮次 ZIP。

---

## 十三、Agent 最终回复格式

### 若通过

```text
阶段 6 定向修正 R1 已完成。
G3-R1：GO_STAGE7

新 seed 改善：<n>/3
新 seed ceiling ties：<n>/3
combined graph-task gain：<value> [95% CI]
combined recovery gain：<value> [95% CI]
combined worst-order gain：<value> [95% CI]
combined long-horizon gain：<value> [95% CI]
combined fixed-order drop：<value>

唯一总交付 ZIP：
<absolute_path>/stage6_refine1_complete.zip

SHA256：
<hash>
```

### 若未通过

```text
阶段 6 定向修正 R1 已完成。
G3-R1：NARROW_TO_REWARD_ONLY

新 seed 改善：<n>/3
combined graph-task gain：<value>
未通过门槛：<list>

停止继续调 Stage 6。
保留 Stage 3–5 的 reward-model 贡献；
Stage 6 下游结果作为有限证据。

唯一总交付 ZIP：
<absolute_path>/stage6_refine1_complete.zip

SHA256：
<hash>
```

**核心点：本轮用三个独立新 seed 对主比较进行一次性复现；通过则进入阶段 7，未通过则停止下游扩张并收窄论文主张。**

