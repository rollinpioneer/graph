# PathGraph-SARM 阶段 7 定向修正 R1：Agent 详细操作命令 V1.0

---

# PathGraph-SARM 阶段 7 定向修正 R1：Agent 通用执行规范

## 0. 为什么不能直接进入阶段 8

`stage7_complete.zip` 的压缩包 SHA256 和 `unzip -t` 均通过，但包内存在与 G4 决策直接冲突的执行证据：

```text
rounds/stage7_4_scaling_and_coverage/tables/coverage_training_status.tsv
  6 / 6 training jobs = FAIL

rounds/stage7_4_scaling_and_coverage/tables/coverage_subset_manifest.csv
  所有 train_episode_count / val_episode_count / test_episode_count = 0

rounds/stage7_4_scaling_and_coverage/tables/coverage_inference_jobs.tsv
  18 / 18 checkpoint_sha256 = MISSING

rounds/stage7_5_ood_and_uncertainty/tables/order_holdout_job_status.tsv
  6 / 6 training jobs = FAIL

rounds/stage7_5_ood_and_uncertainty/logs/run_ood_uncertainty.log
  CSV 写入发生 ValueError
```

但最终报告仍写成：

```text
coverage training jobs = 6 completed
order-holdout training jobs = 6 completed
G4 = GO_STAGE8_REWARD_ONLY
```

因此：

```text
ARCHIVE_INTEGRITY = PASS
EXPERIMENT_CONTENT_GATE = FAIL
STAGE8_ENTRY = BLOCKED_UNTIL_TARGETED_REPAIR
```

这不是要求重做整个阶段 7。只修复两个依赖真实训练 checkpoint 的分支：

```text
7.4 coverage scaling
7.5 order-holdout OOD
```

保持不动：

```text
Stage 7.1 输入冻结
Stage 7.2 核心结构消融
Stage 7.3 历史长度与图粒度
Stage 7.5 已基于冻结主模型完成的 representation perturbation
Stage 7.6 自动图扩展及 KEEP_MANUAL_GRAPH_ONLY 结论
Stage 5 主奖励参数
Stage 6 policy 结论
```

---

## 1. 本轮唯一目标

完成以下闭环：

```text
锁定修正范围
→ 构建非空 coverage / order-holdout 数据
→ 真实 CUDA 训练
→ validation-only 选择 checkpoint
→ 真实 checkpoint 推理
→ 重算 scaling / OOD 指标
→ 更新 G4 与 Stage 8 handoff
```

禁止：

```text
修改 Stage 5 reward_v1
重新训练策略
新增 policy seed
调 lambda / eta / beta
调主模型架构
使用 test 选择 checkpoint
在训练失败时生成公式化占位指标
将 FAIL job 改名为 PASS 而不重跑
```

---

## 2. 小阶段与每轮 ZIP

```text
7R1.1  修正证据冻结与范围锁定
       stage7r1_1_repair_scope_lock.zip

7R1.2  Coverage 数据、训练、推理与指标重算
       stage7r1_2_coverage_real_rerun.zip

7R1.3  Order-holdout 数据、训练、推理与指标重算
       stage7r1_3_order_holdout_real_rerun.zip

7R1.4  Scaling/OOD 表替换与 G4-R1 决策
       stage7r1_4_g4_recompute.zip

7R1.5  修正版 Stage 7 总包与 Stage 8 交接
       stage7r1_5_repackage.zip

总交付：
       stage7_refine1_complete.zip
```

每个小阶段结束立即生成 ZIP。checkpoint、逐帧 prediction、完整数据集等大文件不打包，只写 manifest。

---

## 3. 统一环境变量

从仓库根目录执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"

export STAGE4_ROOT="${STAGE4_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage4}"
export STAGE4_SUPERVISION="${STAGE4_SUPERVISION:-$STAGE4_ROOT/supervision}"
export STAGE4_TOOLS="${STAGE4_TOOLS:-$REPO_ROOT/tools/stage4}"

export STAGE5_ROOT="${STAGE5_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage5}"
export STAGE5_PRED="${STAGE5_PRED:-$STAGE5_ROOT/real_predictions_v1}"
export STAGE5_REWARD="${STAGE5_REWARD:-$STAGE5_ROOT/reward_v1}"
export STAGE5_TOOLS="${STAGE5_TOOLS:-$REPO_ROOT/tools/stage5}"

export STAGE6_ROOT="${STAGE6_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6}"
export STAGE6_PERSISTENT="${STAGE6_PERSISTENT:-$STAGE6_ROOT/stage6_inputs/reward_v1_persistent}"

export STAGE7_ROOT="${STAGE7_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage7_reward_only}"
export STAGE7_ROUNDS="${STAGE7_ROUNDS:-$STAGE7_ROOT/rounds}"
export STAGE7_G4="${STAGE7_G4:-$STAGE7_ROOT/g4_reward_only_v1}"
export STAGE7_TOOLS="${STAGE7_TOOLS:-$REPO_ROOT/tools/stage7}"

export STAGE7R1_ROOT="${STAGE7R1_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage7_refine1}"
export STAGE7R1_ROUNDS="$STAGE7R1_ROOT/rounds"
export STAGE7R1_COVERAGE="$STAGE7R1_ROOT/coverage_real_v1"
export STAGE7R1_OOD="$STAGE7R1_ROOT/order_holdout_real_v1"
export STAGE7R1_G4="$STAGE7R1_ROOT/g4_refine1_v1"
export STAGE7R1_TOOLS="$REPO_ROOT/tools/stage7_refine1"
export STAGE7R1_DOWNLOADS="${STAGE7R1_DOWNLOADS:-$REPO_ROOT/downloads/stage7_refine1}"

export MODEL_SEEDS="20260906,20260907,20260908"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-8000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

mkdir -p \
  "$STAGE7R1_ROOT" \
  "$STAGE7R1_ROUNDS" \
  "$STAGE7R1_COVERAGE" \
  "$STAGE7R1_OOD" \
  "$STAGE7R1_G4" \
  "$STAGE7R1_TOOLS" \
  "$STAGE7R1_DOWNLOADS"
```

保存为：

```text
artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
```

每个小阶段先执行：

```bash
source artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
cd "$REPO_ROOT"
```

---

## 4. GPU 必须提权查看

每个含训练或模型推理的小阶段开始时执行：

```bash
mkdir -p "$ROUND_DIR/gpu"

if sudo -n nvidia-smi > "$ROUND_DIR/gpu/nvidia_smi_sudo.txt" 2>&1; then
  echo "GPU_QUERY_MODE=sudo_noninteractive" \
    | tee "$ROUND_DIR/gpu/gpu_query_mode.txt"
else
  echo "sudo -n nvidia-smi failed" \
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

普通权限查询失败时，不能直接判断没有 GPU。

---

## 5. 多 GPU 默认并行

本轮可并行维度：

```text
coverage fraction × model seed
holdout direction × model seed
checkpoint × split inference
```

默认：

```text
一个独立训练或推理 job 占一张 GPU
```

优先复用已经验证过的调度器：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table <jobs.tsv> \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output <status.tsv> \
  --resume-failed true
```

若该脚本不存在，复用 Stage 6 的 `launch_gpu_job_matrix.py`，只修改路径解析，不重写训练逻辑。

---

## 6. 真实训练 job 的最低证据

每个 training job 必须同时满足：

```text
process exit code = 0
status = PASS
cuda_used = true
optimizer_steps > 0
validation metric 存在且有限
best checkpoint 存在
checkpoint SHA256 非空且不是 MISSING
checkpoint 可以 torch.load
checkpoint 中的 seed / config 与 job 匹配
```

每个 inference job 必须同时满足：

```text
process exit code = 0
status = PASS
loaded_checkpoint_path = selected checkpoint
loaded_checkpoint_sha256 = selection manifest 中的 SHA256
prediction_count > 0
all prediction metrics finite
```

禁止：

```text
训练 FAIL 后生成模拟 checkpoint
训练 FAIL 后沿用 Stage 5 主模型并标记为 coverage/holdout 模型
checkpoint SHA256 写 MISSING
只生成汇总数值而没有逐样本 prediction
```

---

## 7. 每轮 ZIP 规则

ZIP 至少包含：

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
完整逐帧 prediction
视频
大型 npy/npz
缓存
超过 200 MB 的其他文件
```

大文件写入：

```text
manifests/checkpoint_manifest.tsv
manifests/large_file_manifest.tsv
```

打包：

```bash
"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 8. 最终出口

```text
GO_STAGE8_REWARD_ONLY
GO_STAGE8_CORE_REWARD_ONLY
REPAIR_STAGE7R1_INFRA
```

含义：

```text
GO_STAGE8_REWARD_ONLY
  Coverage 与 order-holdout 两个分支均由真实 checkpoint 重算完成，
  G4 的 scaling/OOD 证据保留。

GO_STAGE8_CORE_REWARD_ONLY
  实验执行真实完整，但 scaling 或 order-OOD 指标未达到原描述阈值；
  删除相应扩展主张，仍可凭 Stage 7.2/7.3 的核心 reward 证据进入 Stage 8。

REPAIR_STAGE7R1_INFRA
  仅用于 job 中断、文件缺失、checkpoint 损坏等技术问题；
  只补跑失败 job，不能改方法。
```

不得再次输出一个无边界的 `REFINE_STAGE7`。

---

# 阶段 7R1.1：修正证据冻结与范围锁定

## 一、总体上要干什么

本小阶段不运行新实验。Agent 要从现有 Stage 7 工作区读取原始状态文件，确认修正范围只包括：

```text
coverage scaling 的 6 个训练 + 18 个推理
order-holdout 的 6 个训练 + 12 个推理
依赖上述结果的 scaling / OOD / G4 表
```

本轮状态：

```text
TARGETED_REPAIR_SCOPE_LOCKED
```

本轮 ZIP：

```text
stage7r1_1_repair_scope_lock.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7r1_1_repair_scope_lock"
export ROUND_DIR="$STAGE7R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7R1_ROOT"/{locks,configs,manifests,reports}
```

---

## 三、读取而不是覆盖原 Stage 7

检查：

```bash
test -f "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_training_status.tsv"
test -f "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_subset_manifest.csv"
test -f "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_inference_jobs.tsv"

test -f "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/order_holdout_job_status.tsv"
test -f "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/logs/run_ood_uncertainty.log"

test -f "$STAGE7_G4/metrics/g4_decision.json"
test -f "$STAGE7_G4/tables/coverage_scaling_metrics.csv"
test -f "$STAGE7_G4/tables/ood_reward_metrics.csv"
```

原目录保持只读。所有修正结果写入 `$STAGE7R1_ROOT`。

---

## 四、生成机器可读问题报告

创建：

```text
tools/stage7_refine1/inspect_stage7_blockers.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/inspect_stage7_blockers.py" \
  --coverage-status "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_training_status.tsv" \
  --coverage-subsets "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_subset_manifest.csv" \
  --coverage-inference "$STAGE7_ROUNDS/stage7_4_scaling_and_coverage/tables/coverage_inference_jobs.tsv" \
  --holdout-status "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/order_holdout_job_status.tsv" \
  --ood-log "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/logs/run_ood_uncertainty.log" \
  --g4 "$STAGE7_G4/metrics/g4_decision.json" \
  --output "$ROUND_DIR/metrics/stage7_blockers.json" \
  --report "$ROUND_DIR/reports/stage7_blockers.md"
```

必须检查并写出：

```text
coverage_training_fail_count
coverage_subset_zero_count
coverage_checkpoint_missing_count
holdout_training_fail_count
ood_csv_writer_error_found
g4_declared_go
content_gate_pass
```

预期：

```text
coverage_training_fail_count = 6
coverage_subset_zero_count = 6
coverage_checkpoint_missing_count = 18
holdout_training_fail_count = 6
ood_csv_writer_error_found = true
g4_declared_go = true
content_gate_pass = false
```

若工作区中的状态已经被真实修正，脚本应根据当前文件输出真实值，不固定填写。

---

## 五、冻结允许修改的文件

创建：

```text
$STAGE7R1_ROOT/locks/repair_scope_lock.json
```

内容：

```json
{
  "locked_before_rerun": true,
  "allowed_to_rerun": [
    "stage7_4_coverage_subset_build",
    "stage7_4_coverage_training",
    "stage7_4_coverage_inference",
    "stage7_4_coverage_metrics",
    "stage7_5_order_holdout_build",
    "stage7_5_order_holdout_training",
    "stage7_5_order_holdout_inference",
    "stage7_5_order_holdout_metrics",
    "stage7_g4_recompute"
  ],
  "frozen": [
    "stage5_reward_v1",
    "stage7_2_core_reward_ablations",
    "stage7_3_history_granularity",
    "stage7_5_main_model_perturbation_results",
    "stage7_6_auto_graph_decision",
    "stage6_policy_evidence"
  ],
  "main_reward_retuning_allowed": false,
  "policy_training_allowed": false,
  "test_for_checkpoint_selection_allowed": false,
  "placeholder_metrics_allowed": false
}
```

计算 SHA：

```bash
sha256sum "$STAGE7R1_ROOT/locks/repair_scope_lock.json" \
  | tee "$STAGE7R1_ROOT/locks/repair_scope_lock.sha256"
```

---

## 六、冻结输入 SHA

创建输入表：

```bash
cat > "$ROUND_DIR/manifests/repair_input_files.tsv" <<EOF
name	path
stage4_supervision_frozen	$STAGE4_SUPERVISION/FROZEN.md
stage4_sample_index	$STAGE4_SUPERVISION/tables/sample_index.csv.gz
stage4_episode_manifest	$STAGE4_SUPERVISION/tables/episode_manifest.csv
stage4_content_group_split	$STAGE4_SUPERVISION/tables/content_group_split.csv
stage4_label_maps	$STAGE4_SUPERVISION/configs/label_maps.json
stage4_cost_spec	$STAGE4_SUPERVISION/configs/cost_target_spec.yaml
stage5_reward_config	$STAGE5_REWARD/configs/reward_config_v1.yaml
stage5_reward_lock	$STAGE5_REWARD/configs/reward_selection_lock.json
stage6_model_bundle	$STAGE6_PERSISTENT/configs/model_bundle_persistent.json
stage7_core_ablation_gate	$STAGE7_ROUNDS/stage7_2_core_reward_ablations/metrics/core_ablation_gate.json
stage7_history_gate	$STAGE7_ROUNDS/stage7_3_history_and_granularity/metrics/history_granularity_gate.json
stage7_auto_graph_gate	$STAGE7_ROUNDS/stage7_6_auto_graph_exploration/metrics/auto_graph_gate.json
EOF
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/hash_manifest.py" \
  --input "$ROUND_DIR/manifests/repair_input_files.tsv" \
  --output "$STAGE7R1_ROOT/manifests/repair_input_hashes.tsv" \
  --lock "$STAGE7R1_ROOT/locks/repair_input_lock.json"
```

---

## 七、本轮 Gate

创建：

```text
tools/stage7_refine1/decide_repair_scope_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/decide_repair_scope_gate.py" \
  --blockers "$ROUND_DIR/metrics/stage7_blockers.json" \
  --scope-lock "$STAGE7R1_ROOT/locks/repair_scope_lock.json" \
  --input-lock "$STAGE7R1_ROOT/locks/repair_input_lock.json" \
  --output "$ROUND_DIR/metrics/repair_scope_gate.json" \
  --report "$ROUND_DIR/reports/repair_scope_summary.md"
```

允许：

```text
TARGETED_REPAIR_SCOPE_LOCKED
NO_REPAIR_NEEDED_CURRENT_WORKSPACE
REPAIR_INPUT_MISSING
```

只有 `TARGETED_REPAIR_SCOPE_LOCKED` 才进入 7R1.2。

---

## 八、本轮 ZIP

```bash
cp "$STAGE7R1_ROOT/locks/"* "$ROUND_DIR/configs/"
cp "$STAGE7R1_ROOT/manifests/"* "$ROUND_DIR/manifests/"

export ZIP_NAME="stage7r1_1_repair_scope_lock.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 7R1.1：TARGETED_REPAIR_SCOPE_LOCKED
coverage training FAIL：<n>
coverage zero subsets：<n>
coverage missing checkpoint hash：<n>
order-holdout training FAIL：<n>
只修复 7.4 coverage 与 7.5 order-holdout：是
ZIP：<绝对路径>
SHA256：<hash>
下一步：7R1.2
```

**核心点：只锁定两个真实阻塞分支，不重做已经成立的核心 reward 实验。**

---

# 阶段 7R1.2：Coverage 数据、真实训练、真实推理与指标重算

## 一、总体上要干什么

重新完成 coverage scaling 的学习层实验：

```text
train fraction = 0.25, 0.50
model seeds = 20260906, 20260907, 20260908
```

必须构建非空 content-group 子集，运行 6 个真实 CUDA 训练 job，使用 validation-only 选择 checkpoint，再运行 18 个真实推理 job。

100% 结果继续复用 Stage 5 已冻结主模型，不重训。

本轮状态：

```text
COVERAGE_REAL_RERUN_COMPLETE
```

本轮 ZIP：

```text
stage7r1_2_coverage_real_rerun.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7r1_2_coverage_real_rerun"
export ROUND_DIR="$STAGE7R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7R1_COVERAGE"/{configs,supervision,jobs,selection,predictions,metrics,manifests}
```

按通用规范提权运行 `nvidia-smi`。

入口：

```bash
grep -q 'TARGETED_REPAIR_SCOPE_LOCKED' \
  "$STAGE7R1_ROUNDS/stage7r1_1_repair_scope_lock/reports/repair_scope_summary.md"

sha256sum -c "$STAGE7R1_ROOT/locks/repair_scope_lock.sha256"
```

---

## 三、读取真实监督数据规模

创建：

```text
tools/stage7_refine1/inspect_supervision_counts.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/inspect_supervision_counts.py" \
  --sample-index "$STAGE4_SUPERVISION/tables/sample_index.csv.gz" \
  --episode-manifest "$STAGE4_SUPERVISION/tables/episode_manifest.csv" \
  --content-group-split "$STAGE4_SUPERVISION/tables/content_group_split.csv" \
  --output "$ROUND_DIR/metrics/source_supervision_counts.json" \
  --table "$ROUND_DIR/tables/source_supervision_counts.csv"
```

最低要求：

```text
train episode count > 0
val episode count > 0
test episode count > 0
train content_group count >= 8
至少存在 alternative edge train group
至少存在 recovery edge train group
```

如果真实字段名不同，脚本读取表头后做字段映射，并把映射写入：

```text
configs/supervision_field_map.json
```

不得把无法识别的字段默认为零。

---

## 四、重新构建 coverage 子集

创建：

```text
tools/stage7_refine1/build_real_coverage_subsets.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_real_coverage_subsets.py" \
  --sample-index "$STAGE4_SUPERVISION/tables/sample_index.csv.gz" \
  --episode-manifest "$STAGE4_SUPERVISION/tables/episode_manifest.csv" \
  --content-group-split "$STAGE4_SUPERVISION/tables/content_group_split.csv" \
  --fractions 0.25,0.50 \
  --seeds "$MODEL_SEEDS" \
  --group-key content_group_id \
  --stratify-by task_id,edge_type \
  --required-edge-types alternative,recovery,failure,forward \
  --min-groups-per-required-edge 2 \
  --output-root "$STAGE7R1_COVERAGE/supervision" \
  --manifest "$ROUND_DIR/tables/coverage_subset_manifest_r1.csv" \
  --assignment-table "$ROUND_DIR/tables/coverage_group_assignments.csv" \
  --report "$ROUND_DIR/reports/coverage_subset_build.md"
```

每个子集目录至少包含：

```text
tables/sample_index.csv.gz
tables/episode_manifest.csv
tables/content_group_split.csv
tables/cost_pairs.csv.gz
configs/label_maps.json
configs/feature_schema.json
configs/cost_target_spec.yaml
FROZEN.md
```

val/test 必须从 Stage 4 原样引用或复制，不能重新抽样。

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["ROUND_DIR"],
    "tables/coverage_subset_manifest_r1.csv"
)
d = pd.read_csv(p)
assert len(d) == 6, len(d)
for c in [
    "train_episode_count",
    "val_episode_count",
    "test_episode_count",
    "content_groups_train"
]:
    assert (d[c] > 0).all(), (c, d[c].tolist())
assert set(d["train_fraction"].round(2)) == {0.25, 0.50}
assert not d.duplicated(["train_fraction","seed"]).any()
print("COVERAGE_SUBSETS_NONEMPTY")
PY
```

再检查 split：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/check_group_splits.py" \
  --subset-root "$STAGE7R1_COVERAGE/supervision" \
  --output "$ROUND_DIR/metrics/coverage_split_check.json"
```

必须：

```text
group leakage = 0
val/test hash 与 Stage 4 一致
```

---

## 五、创建真实训练命令模板

优先从 Stage 7.3 已成功训练的 job 中读取实际命令、配置和 checkpoint 格式。

依次查找：

```bash
find "$STAGE7_ROOT/model_variants_v1/history_granularity_v1/jobs" \
  -maxdepth 3 \
  -type f \
  \( -name 'command.sh' \
     -o -name 'run_config.yaml' \
     -o -name 'train_manifest.json' \
     -o -name 'metrics.json' \) \
  | head -n 50
```

若存在成功命令，复制为模板：

```text
tools/stage7_refine1/run_reward_model_training.py
```

只替换：

```text
supervision_root
output_dir
seed
variant_id
```

不得改变：

```text
模型结构
loss weights
optimizer
batch size
train budget
validation metric
history length
default graph
```

若没有命令记录，执行：

```bash
"$PYTHON_BIN" "$STAGE4_TOOLS/train_joint_pathgraph.py" --help \
  | tee "$ROUND_DIR/logs/train_joint_pathgraph_help.txt"
```

根据 Stage 4/7.3 配置生成完整命令，不猜测参数。

---

## 六、生成 6-job 矩阵

创建：

```text
tools/stage7_refine1/build_coverage_training_jobs.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_coverage_training_jobs.py" \
  --subset-root "$STAGE7R1_COVERAGE/supervision" \
  --fractions 0.25,0.50 \
  --seeds "$MODEL_SEEDS" \
  --base-config "$STAGE4_SUPERVISION/configs/resolved_stage4.yaml" \
  --train-wrapper "$STAGE7R1_TOOLS/run_reward_model_training.py" \
  --output-root "$STAGE7R1_COVERAGE/jobs" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs_r1.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

若 `resolved_stage4.yaml` 路径不同，从 Stage 7.3 成功 job 的 config 字段读取真实路径。

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
d = pd.read_csv(
    os.path.join(os.environ["ROUND_DIR"],
                 "tables/coverage_training_jobs_r1.tsv"),
    sep="\t"
)
assert len(d) == 6
assert set(d["train_fraction"].round(2)) == {0.25,0.50}
assert set(d["seed"]) == {20260906,20260907,20260908}
assert not d.duplicated(["train_fraction","seed"]).any()
assert (d["train_episode_count"] > 0).all()
assert (d["content_groups_train"] > 0).all()
print("COVERAGE_TRAIN_JOB_MATRIX_OK")
PY
```

---

## 七、多 GPU 并行训练

执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs_r1.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/coverage_training_status_r1.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_coverage_training_r1.log"
```

8 张 GPU 可用时，6 个 job 同时启动。

每个 job 输出：

```text
command.sh
config_resolved.yaml
train.log
val_metrics.csv
checkpoints/best.pt
checkpoints/final.pt
job_result.json
```

`job_result.json` 至少：

```json
{
  "status": "PASS",
  "exit_code": 0,
  "cuda_used": true,
  "gpu_id": 0,
  "optimizer_steps": 0,
  "best_val_metric": 0.0,
  "best_checkpoint": "",
  "best_checkpoint_sha256": ""
}
```

其中 `optimizer_steps` 必须为真实正整数。

---

## 八、训练完成与 checkpoint 验证

创建：

```text
tools/stage7_refine1/verify_training_jobs.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/verify_training_jobs.py" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs_r1.tsv" \
  --job-root "$STAGE7R1_COVERAGE/jobs" \
  --status-table "$ROUND_DIR/tables/coverage_training_status_r1.tsv" \
  --expected-count 6 \
  --output "$ROUND_DIR/metrics/coverage_training_gate.json" \
  --checkpoint-manifest "$ROUND_DIR/manifests/coverage_checkpoint_manifest.tsv" \
  --report "$ROUND_DIR/reports/coverage_training_summary.md"
```

必须：

```text
6 / 6 PASS
6 / 6 cuda_used=true
6 / 6 optimizer_steps>0
6 / 6 best checkpoint exists
6 / 6 checkpoint SHA256 present
6 / 6 torch.load passes
```

不允许 `MISSING`。

---

## 九、Validation-only checkpoint 选择

创建或复用：

```text
tools/stage7_refine1/select_reward_checkpoints.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/select_reward_checkpoints.py" \
  --job-root "$STAGE7R1_COVERAGE/jobs" \
  --job-table "$ROUND_DIR/tables/coverage_training_jobs_r1.tsv" \
  --selection-split val \
  --test-used false \
  --output "$STAGE7R1_COVERAGE/selection/selected_checkpoints.csv" \
  --lock "$STAGE7R1_COVERAGE/selection/selection_lock.json"
```

必须 6 行，checkpoint SHA 与 manifest 一致。

---

## 十、运行 18 个真实推理 job

创建：

```text
tools/stage7_refine1/build_reward_inference_jobs.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_reward_inference_jobs.py" \
  --selection "$STAGE7R1_COVERAGE/selection/selected_checkpoints.csv" \
  --splits val,test,stage3_diagnostic \
  --default-supervision "$STAGE4_SUPERVISION" \
  --diagnostic-root "$STAGE7_ROOT/../stage3/diagnostic_suite_v1" \
  --output-root "$STAGE7R1_COVERAGE/predictions" \
  --job-table "$ROUND_DIR/tables/coverage_inference_jobs_r1.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

每一行必须带：

```text
checkpoint_path
checkpoint_sha256
split
expected_prediction_count
```

禁止 `checkpoint_sha256=MISSING`。

并行执行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/coverage_inference_jobs_r1.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/coverage_inference_status_r1.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_coverage_inference_r1.log"
```

验证：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/verify_inference_jobs.py" \
  --job-table "$ROUND_DIR/tables/coverage_inference_jobs_r1.tsv" \
  --job-root "$STAGE7R1_COVERAGE/predictions" \
  --status-table "$ROUND_DIR/tables/coverage_inference_status_r1.tsv" \
  --expected-count 18 \
  --selection "$STAGE7R1_COVERAGE/selection/selected_checkpoints.csv" \
  --output "$ROUND_DIR/metrics/coverage_inference_gate.json" \
  --report "$ROUND_DIR/reports/coverage_inference_summary.md"
```

必须 18/18 PASS，loaded checkpoint SHA 全部匹配。

---

## 十一、重算 coverage 指标

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/summarize_real_coverage.py" \
  --prediction-root "$STAGE7R1_COVERAGE/predictions" \
  --selection "$STAGE7R1_COVERAGE/selection/selected_checkpoints.csv" \
  --main-model-predictions "$STAGE5_PRED" \
  --fractions 0.25,0.50,1.00 \
  --statistics-unit content_group_id \
  --output "$STAGE7R1_COVERAGE/metrics/coverage_scaling_metrics_r1.csv" \
  --edge-output "$ROUND_DIR/tables/coverage_by_edge_metrics_r1.csv" \
  --report "$ROUND_DIR/reports/coverage_scaling_r1.md" \
  --figures-dir "$ROUND_DIR/figures"
```

至少包含：

```text
train_fraction
node_macro_f1
alternative_edge_f1
recovery_edge_f1
cost_mae
phi_spearman
path_gap
failure_negative_rate
recovery_positive_rate
cycle_nonpositive_rate
model_seed_mean
model_seed_std
statistics_unit
provenance
```

provenance 必须：

```text
real_trained_coverage_subset
stage5_frozen_main_model
```

---

## 十二、本轮 Gate

创建：

```text
tools/stage7_refine1/decide_coverage_rerun_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/decide_coverage_rerun_gate.py" \
  --subset-manifest "$ROUND_DIR/tables/coverage_subset_manifest_r1.csv" \
  --training-gate "$ROUND_DIR/metrics/coverage_training_gate.json" \
  --inference-gate "$ROUND_DIR/metrics/coverage_inference_gate.json" \
  --metrics "$STAGE7R1_COVERAGE/metrics/coverage_scaling_metrics_r1.csv" \
  --output "$ROUND_DIR/metrics/coverage_rerun_gate.json" \
  --report "$ROUND_DIR/reports/coverage_rerun_gate.md"
```

允许：

```text
COVERAGE_REAL_RERUN_COMPLETE
RETRY_FAILED_COVERAGE_JOBS
COVERAGE_DATA_INSUFFICIENT
```

指标好坏不影响“执行完成”状态；真实执行后即使结果弱，也进入 7R1.4并收窄结论。

---

## 十三、本轮 ZIP

```bash
export ZIP_NAME="stage7r1_2_coverage_real_rerun.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 7R1.2：COVERAGE_REAL_RERUN_COMPLETE
coverage subsets：6 / 6 非空
training：6 / 6 PASS，CUDA
selected checkpoints：6 / 6，SHA256 已记录
inference：18 / 18 PASS
ZIP：<绝对路径>
SHA256：<hash>
下一步：7R1.3
```

**核心点：coverage 曲线必须来自 6 个真实训练 checkpoint，不能再由零样本清单或缺失 checkpoint 生成。**

---

# 阶段 7R1.3：Order-Holdout 数据、真实训练、真实推理与指标重算

## 一、总体上要干什么

重新完成顺序 OOD 分支：

```text
A_first_train → B_first unseen test
B_first_train → A_first unseen test
3 model seeds
```

共 6 个真实 CUDA 训练 job、12 个真实 checkpoint 推理 job。

同时修复原 `run_ood_uncertainty.py` 的 CSV writer 字段错误。只替换 order-holdout 相关行；基于冻结主模型的 representation perturbation 和 uncertainty 结果先做输入契约验证，契约通过则保留，不重复跑 60 个 job。

本轮状态：

```text
ORDER_HOLDOUT_REAL_RERUN_COMPLETE
```

本轮 ZIP：

```text
stage7r1_3_order_holdout_real_rerun.zip
```

---

## 二、建立目录与 GPU 查询

```bash
source artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7r1_3_order_holdout_real_rerun"
export ROUND_DIR="$STAGE7R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7R1_OOD"/{configs,supervision,jobs,selection,predictions,metrics,manifests}
```

按通用规范提权查看 GPU。

---

## 三、修复 CSV writer

复制原脚本到修正目录，不直接改 Stage 7 冻结脚本：

```bash
cp "$STAGE7_TOOLS/run_ood_uncertainty.py" \
  "$STAGE7R1_TOOLS/run_ood_uncertainty_r1.py"
```

在修正版中，所有 CSV 写入必须使用显式字段集合。

推荐函数：

```python
def write_csv(path, rows, fieldnames=None, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({
            key
            for row in rows
            for key in row.keys()
        })
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)
```

要求：

```text
不丢弃 holdout_direction
不丢弃 provenance
不丢弃 recovery/failure/cycle position 字段
若出现未知字段，明确报错并更新 schema
```

运行语法检查：

```bash
"$PYTHON_BIN" -m py_compile \
  "$STAGE7R1_TOOLS/run_ood_uncertainty_r1.py"
```

---

## 四、定位 dual-order fold

优先：

```bash
export DUAL_ORDER_FOLDS="$STAGE4_SUPERVISION/probes/dual_order_folds.json"
```

若不存在：

```bash
find "$REPO_ROOT/artifacts/pathgraph_sarm" \
  -type f \
  -name 'dual_order_folds.json' \
  -print
```

确认内容：

```bash
"$PYTHON_BIN" - <<'PY'
import json, os
p=os.environ["DUAL_ORDER_FOLDS"]
d=json.load(open(p))
print(json.dumps(d, indent=2))
assert d
PY
```

---

## 五、构建非空 order-holdout 监督数据

创建：

```text
tools/stage7_refine1/build_real_order_holdout.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_real_order_holdout.py" \
  --supervision "$STAGE4_SUPERVISION" \
  --folds "$DUAL_ORDER_FOLDS" \
  --directions A_first_train,B_first_train \
  --seeds "$MODEL_SEEDS" \
  --group-key content_group_id \
  --output-root "$STAGE7R1_OOD/supervision" \
  --manifest "$ROUND_DIR/tables/order_holdout_manifest_r1.csv" \
  --assignment-table "$ROUND_DIR/tables/order_holdout_group_assignments.csv" \
  --report "$ROUND_DIR/reports/order_holdout_build.md"
```

每个 seed/direction：

```text
seen-order train group > 0
seen-order val group > 0
seen-order test group > 0
unseen-order test group > 0
recovery task train/val 保持存在
group leakage = 0
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p=os.path.join(os.environ["ROUND_DIR"],
               "tables/order_holdout_manifest_r1.csv")
d=pd.read_csv(p)
assert len(d)==6
for c in [
    "seen_train_groups",
    "seen_val_groups",
    "seen_test_groups",
    "unseen_test_groups"
]:
    assert (d[c] > 0).all(), (c,d[c].tolist())
assert not d.duplicated(["direction","seed"]).any()
print("ORDER_HOLDOUT_SUBSETS_NONEMPTY")
PY
```

---

## 六、生成 6 个训练 job

复用 7R1.2 已验证训练 wrapper：

```bash
test -f "$STAGE7R1_TOOLS/run_reward_model_training.py"
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_order_holdout_training_jobs.py" \
  --supervision-root "$STAGE7R1_OOD/supervision" \
  --directions A_first_train,B_first_train \
  --seeds "$MODEL_SEEDS" \
  --base-config "$STAGE4_SUPERVISION/configs/resolved_stage4.yaml" \
  --train-wrapper "$STAGE7R1_TOOLS/run_reward_model_training.py" \
  --output-root "$STAGE7R1_OOD/jobs" \
  --job-table "$ROUND_DIR/tables/order_holdout_training_jobs_r1.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

检查 6 行、数据非空、test_used_for_selection=false。

---

## 七、多 GPU 真实训练

8 张 GPU 可用时，6 个 job 同时运行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/order_holdout_training_jobs_r1.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 20 \
  --status-output "$ROUND_DIR/tables/order_holdout_training_status_r1.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_order_holdout_training_r1.log"
```

验证：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/verify_training_jobs.py" \
  --job-table "$ROUND_DIR/tables/order_holdout_training_jobs_r1.tsv" \
  --job-root "$STAGE7R1_OOD/jobs" \
  --status-table "$ROUND_DIR/tables/order_holdout_training_status_r1.tsv" \
  --expected-count 6 \
  --output "$ROUND_DIR/metrics/order_holdout_training_gate.json" \
  --checkpoint-manifest "$ROUND_DIR/manifests/order_holdout_checkpoint_manifest.tsv" \
  --report "$ROUND_DIR/reports/order_holdout_training_summary.md"
```

必须 6/6 PASS，CUDA，checkpoint hash 非空。

---

## 八、Validation-only 选择

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/select_reward_checkpoints.py" \
  --job-root "$STAGE7R1_OOD/jobs" \
  --job-table "$ROUND_DIR/tables/order_holdout_training_jobs_r1.tsv" \
  --selection-split val_seen_order \
  --test-used false \
  --output "$STAGE7R1_OOD/selection/selected_checkpoints.csv" \
  --lock "$STAGE7R1_OOD/selection/selection_lock.json"
```

必须：

```text
6 selected
test_used=false
checkpoint hash 与 training manifest 一致
```

---

## 九、构建并运行 12 个真实推理 job

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_order_holdout_inference_jobs.py" \
  --selection "$STAGE7R1_OOD/selection/selected_checkpoints.csv" \
  --supervision-root "$STAGE7R1_OOD/supervision" \
  --splits seen-test,unseen-test \
  --output-root "$STAGE7R1_OOD/predictions" \
  --job-table "$ROUND_DIR/tables/order_holdout_inference_jobs_r1.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

检查：

```text
12 rows
checkpoint_sha256 不为空
每个 checkpoint 有 seen-test 和 unseen-test
```

并行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/order_holdout_inference_jobs_r1.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/order_holdout_inference_status_r1.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_order_holdout_inference_r1.log"
```

验证：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/verify_inference_jobs.py" \
  --job-table "$ROUND_DIR/tables/order_holdout_inference_jobs_r1.tsv" \
  --job-root "$STAGE7R1_OOD/predictions" \
  --status-table "$ROUND_DIR/tables/order_holdout_inference_status_r1.tsv" \
  --expected-count 12 \
  --selection "$STAGE7R1_OOD/selection/selected_checkpoints.csv" \
  --output "$ROUND_DIR/metrics/order_holdout_inference_gate.json" \
  --report "$ROUND_DIR/reports/order_holdout_inference_summary.md"
```

---

## 十、重算 order-holdout 指标

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/evaluate_real_order_holdout.py" \
  --prediction-root "$STAGE7R1_OOD/predictions" \
  --selection "$STAGE7R1_OOD/selection/selected_checkpoints.csv" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --statistics-unit content_group_id \
  --output "$STAGE7R1_OOD/metrics/order_holdout_metrics_r1.csv" \
  --seed-output "$ROUND_DIR/tables/order_holdout_metrics_by_seed.csv" \
  --report "$ROUND_DIR/reports/order_holdout_r1.md" \
  --figures-dir "$ROUND_DIR/figures"
```

必须包含：

```text
direction
split
seen_order_node_f1
unseen_order_node_f1
unseen_order_alternative_edge_f1
unseen_order_path_gap
model_seed_mean
model_seed_std
prediction_count
statistics_unit
provenance=real_trained_order_holdout
```

---

## 十一、验证可保留的 perturbation / uncertainty 分支

原 representation perturbation 使用的是冻结主模型，不依赖失败的 holdout checkpoint。只做一次输入契约验证：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/verify_frozen_perturbation_branch.py" \
  --job-table "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/perturbation_inference_jobs.tsv" \
  --status-table "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/perturbation_inference_status.tsv" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --metrics "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/ood_reward_metrics.csv" \
  --uncertainty "$STAGE7_ROUNDS/stage7_5_ood_and_uncertainty/tables/uncertainty_error_detection.csv" \
  --output "$ROUND_DIR/metrics/frozen_perturbation_branch_check.json"
```

必须：

```text
60/60 PASS
model seeds 与 persistent bundle 一致
job 设置数量为 10 × 3 × 2
metrics finite
```

若通过，直接保留，不重跑。

若不通过，只重跑这 60 个 inference job，不训练新模型。

---

## 十二、本轮 Gate

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/decide_order_holdout_rerun_gate.py" \
  --manifest "$ROUND_DIR/tables/order_holdout_manifest_r1.csv" \
  --training-gate "$ROUND_DIR/metrics/order_holdout_training_gate.json" \
  --inference-gate "$ROUND_DIR/metrics/order_holdout_inference_gate.json" \
  --metrics "$STAGE7R1_OOD/metrics/order_holdout_metrics_r1.csv" \
  --perturbation-check "$ROUND_DIR/metrics/frozen_perturbation_branch_check.json" \
  --output "$ROUND_DIR/metrics/order_holdout_rerun_gate.json" \
  --report "$ROUND_DIR/reports/order_holdout_rerun_gate.md"
```

允许：

```text
ORDER_HOLDOUT_REAL_RERUN_COMPLETE
RETRY_FAILED_HOLDOUT_JOBS
ORDER_HOLDOUT_DATA_INSUFFICIENT
```

---

## 十三、本轮 ZIP

```bash
export ZIP_NAME="stage7r1_3_order_holdout_real_rerun.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 7R1.3：ORDER_HOLDOUT_REAL_RERUN_COMPLETE
holdout subsets：6 / 6 非空
training：6 / 6 PASS，CUDA
selected checkpoints：6 / 6
inference：12 / 12 PASS
perturbation branch：保留 / 已补跑
ZIP：<绝对路径>
SHA256：<hash>
下一步：7R1.4
```

**核心点：顺序 OOD 指标必须来自 A-first/B-first holdout 的真实模型，不得在训练全部失败后继续生成汇总值。**

---

# 阶段 7R1.4：替换无效表并重算 G4-R1

## 一、总体上要干什么

本小阶段不训练。用 7R1.2 和 7R1.3 的真实结果替换：

```text
coverage_scaling_metrics
scaling_boundary
ood_reward_metrics 中的 order_holdout 行
ood_uncertainty_gate 中的 order-holdout 指标
```

保留：

```text
核心消融
历史/粒度
stress graph
recovery-position
representation perturbation
uncertainty
auto graph
policy secondary evidence
```

然后重新生成 claim matrix、G4 决策和 Stage 8 handoff。

本轮状态：

```text
G4_R1_RECOMPUTED
```

本轮 ZIP：

```text
stage7r1_4_g4_recompute.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7r1_4_g4_recompute"
export ROUND_DIR="$STAGE7R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7R1_G4"/{configs,locks,metrics,tables,figures,reports,manifests}
```

---

## 三、入口检查

```bash
grep -q 'COVERAGE_REAL_RERUN_COMPLETE' \
  "$STAGE7R1_ROUNDS/stage7r1_2_coverage_real_rerun/reports/coverage_rerun_gate.md"

grep -q 'ORDER_HOLDOUT_REAL_RERUN_COMPLETE' \
  "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/reports/order_holdout_rerun_gate.md"
```

若状态文件采用 JSON，优先读取 JSON，不依赖 grep。

---

## 四、复制冻结有效表

```bash
cp "$STAGE7_G4/tables/reward_main_table.csv" \
  "$STAGE7R1_G4/tables/"
cp "$STAGE7_G4/tables/core_ablation_effects.csv" \
  "$STAGE7R1_G4/tables/"
cp "$STAGE7_G4/tables/history_granularity_summary.csv" \
  "$STAGE7R1_G4/tables/"
cp "$STAGE7_G4/tables/graph_stress_metrics.csv" \
  "$STAGE7R1_G4/tables/"
cp "$STAGE7_G4/tables/uncertainty_error_detection.csv" \
  "$STAGE7R1_G4/tables/"
cp "$STAGE7_G4/tables/auto_graph_test_metrics.csv" \
  "$STAGE7R1_G4/tables/"
cp "$STAGE7_G4/tables/policy_secondary_evidence.csv" \
  "$STAGE7R1_G4/tables/"
```

复制时保存原 SHA：

```bash
sha256sum \
  "$STAGE7_G4/tables/reward_main_table.csv" \
  "$STAGE7_G4/tables/core_ablation_effects.csv" \
  "$STAGE7_G4/tables/history_granularity_summary.csv" \
  "$STAGE7_G4/tables/graph_stress_metrics.csv" \
  "$STAGE7_G4/tables/uncertainty_error_detection.csv" \
  "$STAGE7_G4/tables/auto_graph_test_metrics.csv" \
  "$STAGE7_G4/tables/policy_secondary_evidence.csv" \
  > "$ROUND_DIR/checksums/frozen_preserved_tables.sha256"
```

---

## 五、替换 coverage 表并重算 scaling boundary

复制：

```bash
cp "$STAGE7R1_COVERAGE/metrics/coverage_scaling_metrics_r1.csv" \
  "$STAGE7R1_G4/tables/coverage_scaling_metrics.csv"
```

重算：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/recompute_scaling_boundary.py" \
  --coverage "$STAGE7R1_G4/tables/coverage_scaling_metrics.csv" \
  --stress "$STAGE7R1_G4/tables/graph_stress_metrics.csv" \
  --output "$STAGE7R1_G4/metrics/scaling_boundary_r1.json" \
  --report "$STAGE7R1_G4/reports/scaling_boundary_r1.md"
```

报告区分：

```text
real coverage learning curve
controlled symbolic stress boundary
```

---

## 六、替换 order-holdout 行

创建：

```text
tools/stage7_refine1/merge_ood_metrics_r1.py
```

输入：

```text
原 Stage 7 ood_reward_metrics.csv
真实 order_holdout_metrics_r1.csv
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/merge_ood_metrics_r1.py" \
  --original "$STAGE7_G4/tables/ood_reward_metrics.csv" \
  --replacement-order "$STAGE7R1_OOD/metrics/order_holdout_metrics_r1.csv" \
  --output "$STAGE7R1_G4/tables/ood_reward_metrics.csv" \
  --replacement-log "$ROUND_DIR/tables/ood_row_replacement_log.csv"
```

规则：

```text
删除原 suite=order_holdout 的全部行
插入真实重算行
保留 recovery_position 与 perturbation 行
不得平均或混合原无效 order-holdout 数值
```

---

## 七、重算 OOD Gate

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/recompute_ood_gate.py" \
  --ood-metrics "$STAGE7R1_G4/tables/ood_reward_metrics.csv" \
  --uncertainty "$STAGE7R1_G4/tables/uncertainty_error_detection.csv" \
  --perturbation-check "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/metrics/frozen_perturbation_branch_check.json" \
  --output "$STAGE7R1_G4/metrics/ood_uncertainty_gate_r1.json" \
  --report "$STAGE7R1_G4/reports/ood_uncertainty_r1.md"
```

阈值沿用原 Stage 7 预先定义：

```text
unseen-order path gap <= 0.25
unseen-order alternative edge F1 >= 0.60
```

若不通过，不算执行失败，只删除“未见顺序泛化”支持状态。

---

## 八、重新构建 claim matrix

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/build_claim_matrix_r1.py" \
  --original-claim-matrix "$STAGE7_G4/tables/claim_matrix.csv" \
  --core-ablation "$STAGE7R1_G4/tables/core_ablation_effects.csv" \
  --history "$STAGE7R1_G4/tables/history_granularity_summary.csv" \
  --coverage "$STAGE7R1_G4/tables/coverage_scaling_metrics.csv" \
  --scaling-boundary "$STAGE7R1_G4/metrics/scaling_boundary_r1.json" \
  --ood "$STAGE7R1_G4/tables/ood_reward_metrics.csv" \
  --ood-gate "$STAGE7R1_G4/metrics/ood_uncertainty_gate_r1.json" \
  --auto-graph "$STAGE7R1_G4/tables/auto_graph_test_metrics.csv" \
  --policy "$STAGE7R1_G4/tables/policy_secondary_evidence.csv" \
  --output "$STAGE7R1_G4/tables/claim_matrix_r1.csv" \
  --report "$STAGE7R1_G4/reports/claim_boundary_r1.md"
```

固定不变：

```text
stable policy improvement = not_supported
automatic graph as main = not_supported
manual graph remains main
policy evidence = secondary_mixed
```

---

## 九、G4-R1 决策

创建：

```text
tools/stage7_refine1/decide_g4_r1.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE7R1_TOOLS/decide_g4_r1.py" \
  --coverage-gate "$STAGE7R1_ROUNDS/stage7r1_2_coverage_real_rerun/metrics/coverage_rerun_gate.json" \
  --holdout-gate "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/metrics/order_holdout_rerun_gate.json" \
  --scaling "$STAGE7R1_G4/metrics/scaling_boundary_r1.json" \
  --ood "$STAGE7R1_G4/metrics/ood_uncertainty_gate_r1.json" \
  --core-ablation "$STAGE7_ROUNDS/stage7_2_core_reward_ablations/metrics/core_ablation_gate.json" \
  --history "$STAGE7_ROUNDS/stage7_3_history_and_granularity/metrics/history_granularity_gate.json" \
  --auto-graph "$STAGE7_ROUNDS/stage7_6_auto_graph_exploration/metrics/auto_graph_gate.json" \
  --claim-matrix "$STAGE7R1_G4/tables/claim_matrix_r1.csv" \
  --output "$STAGE7R1_G4/metrics/g4_r1_decision.json" \
  --report "$STAGE7R1_G4/reports/g4_r1_decision.md"
```

### GO_STAGE8_REWARD_ONLY

要求：

```text
coverage rerun execution complete
order-holdout rerun execution complete
alternative structural support = true
recovery structural support = true
scaling boundary 可从真实 coverage + controlled stress 报告
order-holdout 指标达到原阈值
无 test 后主奖励调参
```

### GO_STAGE8_CORE_REWARD_ONLY

要求：

```text
两个 rerun 执行真实完整
核心 alternative/recovery 消融成立
但真实 coverage 或 order-holdout 指标不支持扩展主张
```

此时：

```text
Stage 8 可以继续
删除 scaling 或 unseen-order 泛化主张
保留核心 reward representation 贡献
```

### REPAIR_STAGE7R1_INFRA

仅用于运行缺失或文件损坏。

---

## 十、更新 Stage 8 handoff

创建：

```text
$STAGE7R1_G4/stage8_handoff_r1.md
```

必须列出：

```text
final G4-R1 decision
最终保留的 primary claims
删除或降级的 claims
manual graph remains main
policy evidence secondary/mixed
coverage result provenance
order-holdout result provenance
checkpoint manifests
large files omitted
Stage 8 只做最终复现、统计、图表与论文材料冻结
```

---

## 十一、本轮 ZIP

```bash
export ZIP_NAME="stage7r1_4_g4_recompute.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 7R1.4：G4_R1_RECOMPUTED
G4-R1：<GO_STAGE8_REWARD_ONLY / GO_STAGE8_CORE_REWARD_ONLY>
真实 coverage 结论：<summary>
真实 order-holdout 结论：<summary>
manual graph remains main：true
policy evidence：secondary/mixed
ZIP：<绝对路径>
SHA256：<hash>
下一步：7R1.5
```

**核心点：G4 必须从真实重跑结果重新作出，指标变弱时收窄主张，不再用占位结果维持原结论。**

---

# 阶段 7R1.5：修正版 Stage 7 总包与 Stage 8 交接

## 一、总体上要干什么

把修正后的 Stage 7 证据整理成一个新的唯一总包：

```text
stage7_refine1_complete.zip
```

原 `stage7_complete.zip` 保留为历史版本，不覆盖、不删除。

本轮状态：

```text
STAGE7_REFINE1_COMPLETE
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage7_refine1/stage7_refine1_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage7r1_5_repackage"
export ROUND_DIR="$STAGE7R1_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE7R1_ROOT/final_package"
```

---

## 三、验证 G4-R1

```bash
test -f "$STAGE7R1_G4/metrics/g4_r1_decision.json"
test -f "$STAGE7R1_G4/reports/g4_r1_decision.md"
test -f "$STAGE7R1_G4/stage8_handoff_r1.md"
```

读取：

```bash
"$PYTHON_BIN" - <<'PY'
import json, os
p=os.path.join(
    os.environ["STAGE7R1_G4"],
    "metrics/g4_r1_decision.json"
)
d=json.load(open(p))
assert d["decision"] in {
    "GO_STAGE8_REWARD_ONLY",
    "GO_STAGE8_CORE_REWARD_ONLY"
}, d
print(d["decision"])
PY
```

若为 `REPAIR_STAGE7R1_INFRA`，不得打 final zip，先补跑受影响 job。

---

## 四、构建最终目录

最终包包括：

```text
repair_scope/
coverage_real_v1/
order_holdout_real_v1/
g4_refine1_v1/
round_summaries/
manifests/
```

只复制轻量结果：

```bash
export FINAL_ROOT="$STAGE7R1_ROOT/final_package"

mkdir -p \
  "$FINAL_ROOT"/{repair_scope,coverage_real_v1,order_holdout_real_v1,g4_refine1_v1,round_summaries,manifests}
```

复制范围锁：

```bash
cp "$STAGE7R1_ROOT/locks/"* "$FINAL_ROOT/repair_scope/"
cp "$STAGE7R1_ROOT/manifests/"* "$FINAL_ROOT/repair_scope/"
```

复制 coverage：

```bash
cp "$STAGE7R1_COVERAGE/metrics/coverage_scaling_metrics_r1.csv" \
  "$FINAL_ROOT/coverage_real_v1/"
cp "$STAGE7R1_COVERAGE/selection/selected_checkpoints.csv" \
  "$FINAL_ROOT/coverage_real_v1/"
cp "$STAGE7R1_COVERAGE/selection/selection_lock.json" \
  "$FINAL_ROOT/coverage_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_2_coverage_real_rerun/manifests/coverage_checkpoint_manifest.tsv" \
  "$FINAL_ROOT/coverage_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_2_coverage_real_rerun/tables/coverage_subset_manifest_r1.csv" \
  "$FINAL_ROOT/coverage_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_2_coverage_real_rerun/metrics/coverage_training_gate.json" \
  "$FINAL_ROOT/coverage_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_2_coverage_real_rerun/metrics/coverage_inference_gate.json" \
  "$FINAL_ROOT/coverage_real_v1/"
```

复制 OOD：

```bash
cp "$STAGE7R1_OOD/metrics/order_holdout_metrics_r1.csv" \
  "$FINAL_ROOT/order_holdout_real_v1/"
cp "$STAGE7R1_OOD/selection/selected_checkpoints.csv" \
  "$FINAL_ROOT/order_holdout_real_v1/"
cp "$STAGE7R1_OOD/selection/selection_lock.json" \
  "$FINAL_ROOT/order_holdout_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/manifests/order_holdout_checkpoint_manifest.tsv" \
  "$FINAL_ROOT/order_holdout_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/tables/order_holdout_manifest_r1.csv" \
  "$FINAL_ROOT/order_holdout_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/metrics/order_holdout_training_gate.json" \
  "$FINAL_ROOT/order_holdout_real_v1/"
cp "$STAGE7R1_ROUNDS/stage7r1_3_order_holdout_real_rerun/metrics/order_holdout_inference_gate.json" \
  "$FINAL_ROOT/order_holdout_real_v1/"
```

复制 G4-R1：

```bash
cp -r "$STAGE7R1_G4/"* "$FINAL_ROOT/g4_refine1_v1/"
```

复制每轮 summary/gate：

```bash
for R in \
  stage7r1_1_repair_scope_lock \
  stage7r1_2_coverage_real_rerun \
  stage7r1_3_order_holdout_real_rerun \
  stage7r1_4_g4_recompute
do
  mkdir -p "$FINAL_ROOT/round_summaries/$R"
  find "$STAGE7R1_ROUNDS/$R" \
    -maxdepth 2 \
    -type f \
    \( -name '*gate*.json' \
       -o -name '*summary*.md' \
       -o -name '*.sha256' \
       -o -name '*manifest*.csv' \
       -o -name '*manifest*.tsv' \) \
    -exec cp {} "$FINAL_ROOT/round_summaries/$R/" \;
done
```

---

## 五、建立大文件 manifest

合并 checkpoint manifest：

```bash
{
  head -n 1 "$FINAL_ROOT/coverage_real_v1/coverage_checkpoint_manifest.tsv"
  tail -n +2 "$FINAL_ROOT/coverage_real_v1/coverage_checkpoint_manifest.tsv"
  tail -n +2 "$FINAL_ROOT/order_holdout_real_v1/order_holdout_checkpoint_manifest.tsv"
} > "$FINAL_ROOT/manifests/checkpoint_manifest.tsv"
```

创建：

```text
large_file_manifest.tsv
```

至少记录：

```text
coverage checkpoints
order-holdout checkpoints
coverage predictions
order-holdout predictions
原 Stage 7 大型主模型 prediction
```

字段：

```text
path
size_bytes
sha256
artifact_type
reason_omitted
required_for_full_recompute
```

---

## 六、建立修正版 checksum

只对实际进入 final package 的文件计算：

```bash
(
  cd "$FINAL_ROOT"
  find . -type f \
    ! -name 'STAGE7R1_SHA256SUMS.txt' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$FINAL_ROOT/STAGE7R1_SHA256SUMS.txt"
```

验证：

```bash
(
  cd "$FINAL_ROOT"
  sha256sum -c STAGE7R1_SHA256SUMS.txt
) | tee "$ROUND_DIR/checksums/stage7r1_internal_checksum_test.txt"
```

---

## 七、写冻结结论

创建：

```text
$FINAL_ROOT/FROZEN.md
```

内容模板：

```bash
DECISION=$("$PYTHON_BIN" - <<'PY'
import json, os
p=os.path.join(
    os.environ["STAGE7R1_G4"],
    "metrics/g4_r1_decision.json"
)
print(json.load(open(p))["decision"])
PY
)

cat > "$FINAL_ROOT/FROZEN.md" <<EOF
milestone = M5_REWARD_EVIDENCE_R1
decision = $DECISION
mode = reward_only
coverage_training = real_cuda_6_of_6
coverage_inference = real_checkpoint_18_of_18
order_holdout_training = real_cuda_6_of_6
order_holdout_inference = real_checkpoint_12_of_12
manual_graph_is_main = true
policy_evidence = secondary_mixed
main_reward_retuned = false
policy_training_reopened = false
checkpoint_packaging = omitted_by_default
EOF
```

重新计算 checksum，因为加入了 `FROZEN.md`：

```bash
(
  cd "$FINAL_ROOT"
  find . -type f \
    ! -name 'STAGE7R1_SHA256SUMS.txt' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$FINAL_ROOT/STAGE7R1_SHA256SUMS.txt"
```

---

## 八、生成本轮 ZIP

```bash
cp "$FINAL_ROOT/FROZEN.md" "$ROUND_DIR/reports/"
cp "$STAGE7R1_G4/stage8_handoff_r1.md" "$ROUND_DIR/reports/"
cp "$STAGE7R1_G4/metrics/g4_r1_decision.json" "$ROUND_DIR/metrics/"
cp "$FINAL_ROOT/STAGE7R1_SHA256SUMS.txt" "$ROUND_DIR/checksums/"

export ZIP_NAME="stage7r1_5_repackage.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 九、生成唯一总 ZIP

```bash
cd "$FINAL_ROOT"
zip -qr "$STAGE7R1_DOWNLOADS/stage7_refine1_complete.zip" .
cd "$REPO_ROOT"

unzip -t "$STAGE7R1_DOWNLOADS/stage7_refine1_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage7_refine1_complete_unzip_test.txt"

sha256sum "$STAGE7R1_DOWNLOADS/stage7_refine1_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage7_refine1_complete.sha256"
```

总 ZIP 生成后，不删除五个小阶段 ZIP。

---

## 十、Agent 最终回复

```text
阶段 7 定向修正 R1 已完成。
G4-R1：<GO_STAGE8_REWARD_ONLY / GO_STAGE8_CORE_REWARD_ONLY>

coverage:
- nonempty subsets：6 / 6
- real CUDA training：6 / 6
- real checkpoint inference：18 / 18

order holdout:
- nonempty subsets：6 / 6
- real CUDA training：6 / 6
- real checkpoint inference：12 / 12

唯一总交付 ZIP：
<绝对路径>/stage7_refine1_complete.zip

SHA256：
<hash>

checkpoint：未打包，见 manifest
Stage 8 entry：ALLOWED
```

**核心点：修正版总包必须让“训练 PASS、checkpoint hash、真实推理和最终 G4”形成可追踪的一致链条。**
