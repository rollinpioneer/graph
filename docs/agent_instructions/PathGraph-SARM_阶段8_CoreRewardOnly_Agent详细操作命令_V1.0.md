# PathGraph-SARM 阶段 8（Core Reward Only）Agent 详细操作命令 V1.0

---

# PathGraph-SARM 阶段 8（Core Reward Only）Agent 通用执行规范

## 0. 阶段入口

阶段 7 定向修正 R1 的冻结出口为：

```text
G4-R1 = GO_STAGE8_CORE_REWARD_ONLY
```

因此阶段 8 可以开始，但最终研究边界已经确定：

```text
主线：PathGraph-SARM 的 graph-structured reward representation
非主线：稳定策略提升、coverage scaling、未见顺序泛化、自动图主方法
```

阶段 8 不再扩展研究问题，不再新增 reward-model 变体，不再训练策略。阶段目标是：

```text
冻结最终主张
→ 从 checkpoint 独立复现核心结果
→ 完成最终统计
→ 生成论文图表和结果文字
→ 建立轻量可复现包
→ 冻结 M6 最终研究交付
```

---

## 1. 最终允许保留的结论

### Primary claims

```text
C1  Graph-structured dense reward 能表达多条合法任务路径。
C2  显式 failure/recovery edge 能正确处理失败与恢复。
C3  remaining cost 与 within-node progress 的结合提供稠密过程信号。
C4  recovery debt accounting 能限制 failure-recovery 循环的过度补偿。
```

### Auxiliary / qualified claims

```text
C5  长历史输入在当前 benchmark 上提高图状态识别，但不是普适必要条件。
C6  ensemble uncertainty 可作为 reward-error 的辅助检测信号。
C7  自动图仅作为扩展；最终方法仍采用人工图。
C8  policy 结果为 secondary/mixed：总体和 recovery 有正向趋势，
    但跨 seed 严格一致性不足。
```

### 禁止写成正面主张

```text
稳定提升 RA-BC / policy success
建立了普适 coverage scaling law
泛化到未见合法顺序
自动 graph discovery 优于人工图
非零 beta 或 eta 已被主实验验证
在线规划能力
真实机器人广泛泛化
```

---

## 2. 阶段 8 小阶段与 ZIP

```text
8.1  最终范围、输入和证据源冻结
     stage8_1_final_scope_and_input_lock.zip

8.2  核心模型与奖励流水线独立复现
     stage8_2_core_pipeline_reproduction.zip

8.3  最终统计、置信区间与效应量
     stage8_3_final_statistics.zip

8.4  论文级表格与图形
     stage8_4_publication_tables_and_figures.zip

8.5  论文结果材料与主张映射
     stage8_5_manuscript_evidence_package.zip

8.6  轻量可复现发布包
     stage8_6_reproducibility_bundle.zip

8.7  M6 最终冻结与项目关闭
     stage8_7_m6_final_freeze.zip

阶段总包：
     stage8_complete.zip
```

每个小阶段结束立即生成 ZIP，不等待阶段 8 全部结束。

---

## 3. 统一环境变量

从仓库根目录执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"

export STAGE3_ROOT="${STAGE3_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage3}"
export STAGE3_DIAG="${STAGE3_DIAG:-$STAGE3_ROOT/diagnostic_suite_v1}"
export STAGE3_INPUT="${STAGE3_INPUT:-$STAGE3_ROOT/input_adapter_v1}"

export STAGE4_ROOT="${STAGE4_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage4}"
export STAGE4_SUPERVISION="${STAGE4_SUPERVISION:-$STAGE4_ROOT/supervision}"
export STAGE4_TOOLS="${STAGE4_TOOLS:-$REPO_ROOT/tools/stage4}"

export STAGE5_ROOT="${STAGE5_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage5}"
export STAGE5_PRED="${STAGE5_PRED:-$STAGE5_ROOT/real_predictions_v1}"
export STAGE5_REWARD="${STAGE5_REWARD:-$STAGE5_ROOT/reward_v1}"
export STAGE5_TOOLS="${STAGE5_TOOLS:-$REPO_ROOT/tools/stage5}"

export STAGE6_ROOT="${STAGE6_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6}"
export STAGE6_PERSISTENT="${STAGE6_PERSISTENT:-$STAGE6_ROOT/stage6_inputs/reward_v1_persistent}"
export STAGE6_M4="${STAGE6_M4:-$STAGE6_ROOT/m4_policy_results_v1}"

export STAGE6R1_ROOT="${STAGE6R1_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage6_refine1}"
export STAGE6R1_M4="${STAGE6R1_M4:-$STAGE6R1_ROOT/m4_refine1_results_v1}"

export STAGE7_ROOT="${STAGE7_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage7_reward_only}"
export STAGE7_ROUNDS="${STAGE7_ROUNDS:-$STAGE7_ROOT/rounds}"

export STAGE7R1_ROOT="${STAGE7R1_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage7_refine1}"
export STAGE7R1_G4="${STAGE7R1_G4:-$STAGE7R1_ROOT/g4_refine1_v1}"
export STAGE7R1_FINAL="${STAGE7R1_FINAL:-$STAGE7R1_ROOT/final_package}"

export STAGE8_ROOT="${STAGE8_ROOT:-$REPO_ROOT/artifacts/pathgraph_sarm/stage8_core_reward}"
export STAGE8_INPUTS="$STAGE8_ROOT/final_inputs_v1"
export STAGE8_REPRO="$STAGE8_ROOT/core_reproduction_v1"
export STAGE8_STATS="$STAGE8_ROOT/final_statistics_v1"
export STAGE8_PAPER="$STAGE8_ROOT/paper_artifacts_v1"
export STAGE8_RELEASE="$STAGE8_ROOT/reproducibility_release_v1"
export STAGE8_M6="$STAGE8_ROOT/m6_final_v1"
export STAGE8_ROUNDS="$STAGE8_ROOT/rounds"
export STAGE8_TOOLS="$REPO_ROOT/tools/stage8"
export STAGE8_DOWNLOADS="${STAGE8_DOWNLOADS:-$REPO_ROOT/downloads/stage8}"

export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-8000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export FINAL_BOOTSTRAP_RESAMPLES="${FINAL_BOOTSTRAP_RESAMPLES:-10000}"
export FINAL_BOOTSTRAP_SEED="${FINAL_BOOTSTRAP_SEED:-20261401}"

mkdir -p \
  "$STAGE8_ROOT" \
  "$STAGE8_INPUTS" \
  "$STAGE8_REPRO" \
  "$STAGE8_STATS" \
  "$STAGE8_PAPER" \
  "$STAGE8_RELEASE" \
  "$STAGE8_M6" \
  "$STAGE8_ROUNDS" \
  "$STAGE8_TOOLS" \
  "$STAGE8_DOWNLOADS"
```

保存为：

```text
artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
```

每个小阶段开始时：

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"
```

---

## 4. Stage 7R1 路径解析

优先使用工作区：

```bash
test -f "$STAGE7R1_G4/metrics/g4_r1_decision.json"
```

若目录结构为 `final_package/g4_refine1_v1`：

```bash
if [ ! -f "$STAGE7R1_G4/metrics/g4_r1_decision.json" ] \
   && [ -f "$STAGE7R1_FINAL/g4_refine1_v1/metrics/g4_r1_decision.json" ]; then
  export STAGE7R1_G4="$STAGE7R1_FINAL/g4_refine1_v1"
fi
```

若当前机器只有交付 ZIP，解压到固定只读目录：

```bash
export STAGE7R1_ARCHIVE="${STAGE7R1_ARCHIVE:-$REPO_ROOT/downloads/stage7_refine1_complete.zip}"
export STAGE7R1_EXTRACTED="$STAGE8_INPUTS/stage7_refine1_extracted"

if [ ! -f "$STAGE7R1_G4/metrics/g4_r1_decision.json" ]; then
  mkdir -p "$STAGE7R1_EXTRACTED"
  unzip -q "$STAGE7R1_ARCHIVE" -d "$STAGE7R1_EXTRACTED"
  export STAGE7R1_G4="$STAGE7R1_EXTRACTED/g4_refine1_v1"
  export STAGE7R1_FINAL="$STAGE7R1_EXTRACTED"
fi
```

只有 Stage 7 compact tables 可以从 ZIP 恢复；checkpoint 和逐样本 prediction 仍必须来自原 workspace / manifest 路径。

---

## 5. GPU 必须提权查看

每个含 GPU 推理的小阶段先执行：

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

不能因为 `sudo -n` 需要密码而写成“无 GPU”；必须继续使用可用的直接查询确认。

---

## 6. 多 GPU 默认并行

本阶段主要 GPU 工作为 9 个独立推理 job：

```text
3 model checkpoints × 3 suites
```

8 张 GPU 可用时：

```text
第一波并行 8 个 job
第二波运行剩余 1 个 job
```

默认：

```text
一个独立推理 job 占一张 GPU
```

优先复用：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table <jobs.tsv> \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output <status.tsv> \
  --resume-failed true
```

CPU bootstrap、表格和绘图使用多进程，但不占用 GPU。

---

## 7. 禁止继续做的工作

阶段 8 禁止：

```text
新增任务或数据
重新训练 reward model
重新训练 policy
新增 policy seed
调整 lambda / eta / beta
调整图节点或边定义
根据最终 test 修改主方法
重新选择人工图或 comparator
将 unsupported claim 重新写回主结论
```

只有运行中断、文件路径缺失、脚本错误时允许修复基础设施并补跑受影响 job。

---

## 8. 每轮 ZIP 规则

每轮 ZIP 至少包含：

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

默认不打包：

```text
*.pt
*.pth
*.ckpt
*.safetensors
*.bin
原始 episode
完整逐帧 prediction
bootstrap 全量抽样分布
视频
大型 npy/npz/parquet
缓存
超过 200 MB 的其他文件
```

大文件写入：

```text
manifests/checkpoint_manifest.tsv
manifests/large_file_manifest.tsv
```

每轮打包：

```bash
"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 9. 统一完成状态

阶段 8 最终只允许：

```text
RESEARCH_COMPLETE_CORE_REWARD_ONLY
REPAIR_STAGE8_INFRA
FINAL_REPRODUCTION_MISMATCH
```

含义：

```text
RESEARCH_COMPLETE_CORE_REWARD_ONLY
  核心 reward 流水线独立复现，统计、图表、论文材料和发布包均冻结。

REPAIR_STAGE8_INFRA
  仅用于 checkpoint 丢失、推理 job 中断、脚本或打包错误；
  不允许修改方法或结论。

FINAL_REPRODUCTION_MISMATCH
  相同 checkpoint 和输入不能复现冻结核心指标；
  必须停止发布并定位差异，不能继续写论文结论。
```

---

# 阶段 8.1：最终范围、输入和证据源冻结

## 一、总体上要干什么

本小阶段将 `GO_STAGE8_CORE_REWARD_ONLY` 转换为不可再扩张的最终执行范围，建立：

```text
final_claim_scope.json
final_input_index.yaml
final_evidence_registry.csv
final_checkpoint_manifest.tsv
final_large_file_manifest.tsv
```

本轮不执行模型推理。

本轮状态：

```text
FINAL_SCOPE_AND_INPUTS_LOCKED
```

本轮 ZIP：

```text
stage8_1_final_scope_and_input_lock.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_1_final_scope_and_input_lock"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_INPUTS"/{configs,locks,manifests,reports}
```

---

## 三、确认 G4-R1

```bash
test -f "$STAGE7R1_G4/metrics/g4_r1_decision.json"
test -f "$STAGE7R1_G4/tables/claim_matrix_r1.csv"
test -f "$STAGE7R1_G4/reports/g4_r1_decision.md"
test -f "$STAGE7R1_G4/stage8_handoff_r1.md"
```

执行：

```bash
"$PYTHON_BIN" - <<'PY'
import json, os
p = os.path.join(
    os.environ["STAGE7R1_G4"],
    "metrics/g4_r1_decision.json"
)
d = json.load(open(p))
assert d["decision"] == "GO_STAGE8_CORE_REWARD_ONLY", d
assert d["alternative_structural_support"] is True
assert d["recovery_structural_support"] is True
assert d["scaling_extension_supported"] is False
assert d["order_holdout_extension_supported"] is False
assert d["manual_graph_remains_main"] is True
assert d["policy_evidence"] == "secondary_mixed"
print("G4_R1_ENTRY_CONFIRMED")
PY
```

---

## 四、创建最终 claim scope

创建：

```text
$STAGE8_INPUTS/locks/final_claim_scope.json
```

内容：

```json
{
  "locked_before_stage8_reproduction": true,
  "final_mode": "core_reward_only",
  "primary_supported": [
    "graph-structured dense reward for multiple legal paths",
    "explicit failure and recovery reward semantics",
    "remaining cost plus within-node progress",
    "recovery debt accounting for loop safety"
  ],
  "qualified_auxiliary": [
    "history helps on the current benchmark",
    "uncertainty is an auxiliary error signal",
    "manual graph is the final main method",
    "policy evidence is secondary and mixed"
  ],
  "removed_or_unsupported": [
    "stable policy improvement",
    "coverage scaling generalization",
    "unseen-order generalization",
    "automatic graph discovery as main contribution",
    "nonzero beta or eta as validated main components"
  ],
  "new_training_allowed": false,
  "new_data_allowed": false,
  "main_reward_retuning_allowed": false,
  "test_driven_revision_allowed": false
}
```

锁定：

```bash
sha256sum "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  | tee "$STAGE8_INPUTS/locks/final_claim_scope.sha256"
```

---

## 五、定位最终模型与数据输入

检查：

```bash
test -f "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json"
test -f "$STAGE5_REWARD/configs/reward_config_v1.yaml"
test -f "$STAGE5_REWARD/configs/reward_selection_lock.json"
test -f "$STAGE5_REWARD/code/reward_engine.py"

test -f "$STAGE4_SUPERVISION/tables/sample_index.csv.gz"
test -f "$STAGE4_SUPERVISION/tables/episode_manifest.csv"
test -f "$STAGE4_SUPERVISION/tables/content_group_split.csv"
test -f "$STAGE4_SUPERVISION/configs/label_maps.json"
test -f "$STAGE4_SUPERVISION/configs/feature_schema.json"
test -f "$STAGE4_SUPERVISION/configs/cost_target_spec.yaml"

test -f "$STAGE5_PRED/tables/ensemble_val_predictions.jsonl.gz"
test -f "$STAGE5_PRED/tables/ensemble_test_predictions.jsonl.gz"
test -f "$STAGE5_PRED/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz"

test -f "$STAGE7R1_G4/tables/reward_main_table.csv"
test -f "$STAGE7R1_G4/tables/core_ablation_effects.csv"
test -f "$STAGE7R1_G4/tables/history_granularity_summary.csv"
test -f "$STAGE7R1_G4/tables/uncertainty_error_detection.csv"
```

若单个路径名称不同，只在相同 Stage 根目录内定位同名文件，不做全仓库扫描。

---

## 六、验证三个 persistent checkpoint

创建：

```text
tools/stage8/build_final_checkpoint_manifest.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_final_checkpoint_manifest.py" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --output "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --load-check "$ROUND_DIR/metrics/checkpoint_load_check.json"
```

脚本必须：

1. 读取 bundle 中的三个 checkpoint；
2. 验证文件存在；
3. 重算 SHA256；
4. 与 bundle 记录一致；
5. 使用 CPU 执行 `torch.load`；
6. 检查 seed、state_dict 和模型配置字段；
7. 输出：

```text
model_seed
checkpoint_path
size_bytes
sha256
bundle_sha256_match
torch_load_ok
model_config_id
```

必须：

```text
3 / 3 SHA256 match
3 / 3 torch_load_ok
```

checkpoint 本体不进入 ZIP。

---

## 七、建立最终输入索引

创建：

```text
tools/stage8/build_final_input_index.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_final_input_index.py" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --checkpoint-manifest "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --supervision "$STAGE4_SUPERVISION" \
  --diagnostic "$STAGE3_DIAG" \
  --reference-predictions "$STAGE5_PRED" \
  --reward-v1 "$STAGE5_REWARD" \
  --g4-r1 "$STAGE7R1_G4" \
  --output "$STAGE8_INPUTS/configs/final_input_index.yaml" \
  --hash-output "$STAGE8_INPUTS/manifests/final_input_hashes.tsv"
```

`final_input_index.yaml` 至少记录：

```yaml
mode: core_reward_only
statistics_unit: content_group_id

model:
  ensemble_size: 3
  bundle: ...
  checkpoint_manifest: ...

suites:
  val: ...
  test: ...
  stage3_diagnostic: ...

reward:
  config: ...
  lock: ...
  engine: ...

reference_results:
  reward_main_table: ...
  core_ablation_effects: ...
  history_granularity: ...
  uncertainty: ...

unsupported_extensions:
  coverage_scaling: false
  unseen_order: false
  stable_policy: false
  auto_graph_main: false
```

---

## 八、建立最终证据注册表

创建：

```text
tools/stage8/build_final_evidence_registry.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_final_evidence_registry.py" \
  --claim-matrix "$STAGE7R1_G4/tables/claim_matrix_r1.csv" \
  --input-index "$STAGE8_INPUTS/configs/final_input_index.yaml" \
  --output "$STAGE8_INPUTS/manifests/final_evidence_registry.csv" \
  --report "$STAGE8_INPUTS/reports/final_evidence_scope.md"
```

字段：

```text
claim_id
claim_text
final_status
priority
source_table
source_metric
source_artifact
provenance
allowed_in_abstract
allowed_in_main_results
required_qualifier
```

规则：

```text
supported primary claim 才能进入摘要结论
partially_supported 只能带限定语
not_supported 只能进入 limitations / negative result
```

---

## 九、建立大文件清单

创建：

```text
$STAGE8_INPUTS/manifests/final_large_file_manifest.tsv
```

至少记录：

```text
3 个 reward-model checkpoint
Stage 5 三个完整 prediction 文件
Stage 3 diagnostic raw suite
任何超过 200 MB 的 supervision / trace 文件
```

字段：

```text
path
artifact_type
size_bytes
sha256
required_for
packaged
reason_omitted
```

`packaged=false`。

---

## 十、本轮 Gate

创建：

```text
tools/stage8/decide_final_input_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/decide_final_input_gate.py" \
  --g4 "$STAGE7R1_G4/metrics/g4_r1_decision.json" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --input-index "$STAGE8_INPUTS/configs/final_input_index.yaml" \
  --checkpoint-check "$ROUND_DIR/metrics/checkpoint_load_check.json" \
  --evidence-registry "$STAGE8_INPUTS/manifests/final_evidence_registry.csv" \
  --output "$ROUND_DIR/metrics/final_input_gate.json" \
  --report "$ROUND_DIR/reports/final_input_gate.md"
```

必须：

```text
G4-R1 = GO_STAGE8_CORE_REWARD_ONLY
3/3 checkpoint verified
main reward config/lock present
three evaluation suites present
claim scope locked
unsupported claims marked false
no new training allowed
```

允许：

```text
FINAL_SCOPE_AND_INPUTS_LOCKED
REPAIR_FINAL_INPUT_PATHS
MISSING_FINAL_CHECKPOINT
```

---

## 十一、本轮 ZIP

```bash
cp "$STAGE8_INPUTS/configs/final_input_index.yaml" "$ROUND_DIR/configs/"
cp "$STAGE8_INPUTS/locks/"* "$ROUND_DIR/configs/"
cp "$STAGE8_INPUTS/manifests/"* "$ROUND_DIR/manifests/"
cp "$STAGE8_INPUTS/reports/final_evidence_scope.md" "$ROUND_DIR/reports/"

export ZIP_NAME="stage8_1_final_scope_and_input_lock.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 8.1：FINAL_SCOPE_AND_INPUTS_LOCKED
final mode：core_reward_only
reward checkpoints：3 / 3
unsupported claims locked：4
new training allowed：false
ZIP：<绝对路径>
SHA256：<hash>
下一步：8.2
```

**核心点：先把最终能说什么、不能说什么以及所有复现输入一次性锁定。**

---

# 阶段 8.2：核心模型与奖励流水线独立复现

## 一、总体上要干什么

从三个持久化 checkpoint 重新运行真实推理，并从新生成的逐样本 prediction 重新计算：

```text
Node / edge metrics
within-node progress
remaining cost
PathGraph reward
linear / sequential baselines
核心结构消融
```

本轮不得直接复制 Stage 5 或 Stage 7 的结果表。参考表只用于最终数值一致性比较。

GPU job：

```text
3 model seeds × 3 suites = 9 inference jobs
```

本轮状态：

```text
CORE_PIPELINE_REPRODUCED
```

本轮 ZIP：

```text
stage8_2_core_pipeline_reproduction.zip
```

---

## 二、建立目录并查询 GPU

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_2_core_pipeline_reproduction"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,jobs,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_REPRO"/{configs,jobs,predictions,ensemble,metrics,tables,manifests}
```

按通用规范执行提权 GPU 查询。

入口：

```bash
grep -q 'FINAL_SCOPE_AND_INPUTS_LOCKED' \
  "$STAGE8_ROUNDS/stage8_1_final_scope_and_input_lock/reports/final_input_gate.md"
```

---

## 三、冻结可复现推理配置

创建：

```text
$STAGE8_REPRO/configs/final_inference_protocol.yaml
```

内容：

```yaml
protocol_version: stage8-core-reproduction-v1

model:
  source_bundle: stage6 persistent reward_v1
  ensemble_seeds: [20260906, 20260907, 20260908]
  eval_mode: true
  torch_inference_mode: true
  augmentation: false
  dropout: false
  history_steps: 32

suites:
  - val
  - test
  - stage3_diagnostic

determinism:
  cudnn_benchmark: false
  cudnn_deterministic: true
  allow_tf32: false
  fixed_batch_order: true

output:
  per_sample_prediction: true
  compression: jsonl.gz
  include_checkpoint_sha256: true
  statistics_unit: content_group_id
```

锁定：

```bash
sha256sum "$STAGE8_REPRO/configs/final_inference_protocol.yaml" \
  | tee "$STAGE8_REPRO/configs/final_inference_protocol.sha256"
```

---

## 四、实现或确认真实推理入口

优先复用 Stage 5 已验证的真实 checkpoint 推理脚本。定位：

```bash
find "$REPO_ROOT/tools/stage5" \
  -maxdepth 2 \
  -type f \
  \( -iname '*real*infer*.py' \
     -o -iname '*ensemble*infer*.py' \
     -o -iname '*checkpoint*infer*.py' \) \
  -print
```

选择标准：

```text
真实执行 torch.load
真实构建模型
读取监督数据特征
逐样本前向
输出 checkpoint path 和 SHA256
不含固定指标字典
不复制 ground truth 为 prediction
```

若没有单一入口，创建：

```text
tools/stage8/run_frozen_reward_inference.py
```

CLI：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/run_frozen_reward_inference.py" \
  --checkpoint <path> \
  --checkpoint-sha256 <hash> \
  --model-config <resolved_config> \
  --suite <val|test|stage3_diagnostic> \
  --supervision-root "$STAGE4_SUPERVISION" \
  --diagnostic-root "$STAGE3_DIAG" \
  --inference-protocol "$STAGE8_REPRO/configs/final_inference_protocol.yaml" \
  --output <prediction.jsonl.gz> \
  --metrics-output <per_seed_metrics.json> \
  --device cuda:0
```

逐样本输出至少包括：

```text
sample_id
episode_id
content_group_id
task_id
split
t
gt_node_id
pred_node_logits
pred_node_id
gt_edge_type
pred_edge_type_logits
pred_edge_type
gt_edge_id
pred_edge_id_logits
pred_edge_id
gt_phi
pred_phi
gt_remaining_cost
pred_remaining_cost
model_seed
checkpoint_path
checkpoint_sha256
```

禁止使用：

```text
outcome
success
scenario
controller_source
episode_id
content_group_id
```

作为模型输入；它们仅能保留为评估元数据。

---

## 五、生成 9-job 推理矩阵

创建：

```text
tools/stage8/build_core_reproduction_jobs.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_core_reproduction_jobs.py" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --checkpoint-manifest "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --suites val,test,stage3_diagnostic \
  --protocol "$STAGE8_REPRO/configs/final_inference_protocol.yaml" \
  --runner "$STAGE8_TOOLS/run_frozen_reward_inference.py" \
  --output-root "$STAGE8_REPRO/jobs" \
  --prediction-root "$STAGE8_REPRO/predictions" \
  --job-table "$ROUND_DIR/tables/core_reproduction_jobs.tsv" \
  --commands-dir "$ROUND_DIR/commands"
```

检查：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd, os
p = os.path.join(
    os.environ["ROUND_DIR"],
    "tables/core_reproduction_jobs.tsv"
)
d = pd.read_csv(p, sep="\t")
assert len(d) == 9, len(d)
assert d["model_seed"].nunique() == 3
assert set(d["suite"]) == {"val","test","stage3_diagnostic"}
assert not d.duplicated(["model_seed","suite"]).any()
assert d["checkpoint_sha256"].str.len().eq(64).all()
print("CORE_REPRODUCTION_9_JOBS_OK")
PY
```

---

## 六、多 GPU 并行推理

8 张 GPU 可用时直接运行：

```bash
"$PYTHON_BIN" "$STAGE7_TOOLS/launch_gpu_job_matrix.py" \
  --job-table "$ROUND_DIR/tables/core_reproduction_jobs.tsv" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --poll-seconds 15 \
  --status-output "$ROUND_DIR/tables/core_reproduction_status.tsv" \
  --resume-failed true \
  2>&1 | tee "$ROUND_DIR/logs/launch_core_reproduction.log"
```

记录运行进度：

```bash
while pgrep -f "run_frozen_reward_inference.py" >/dev/null; do
  date -Iseconds
  nvidia-smi \
    --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
  sleep 300
done >> "$ROUND_DIR/logs/gpu_progress.log" 2>&1
```

---

## 七、验证 9 个推理 job

创建：

```text
tools/stage8/verify_core_reproduction_jobs.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/verify_core_reproduction_jobs.py" \
  --job-table "$ROUND_DIR/tables/core_reproduction_jobs.tsv" \
  --status-table "$ROUND_DIR/tables/core_reproduction_status.tsv" \
  --job-root "$STAGE8_REPRO/jobs" \
  --expected-count 9 \
  --checkpoint-manifest "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --output "$ROUND_DIR/metrics/core_inference_gate.json" \
  --report "$ROUND_DIR/reports/core_inference_summary.md"
```

必须：

```text
9 / 9 process exit code = 0
9 / 9 status = PASS
9 / 9 cuda_used = true
9 / 9 checkpoint SHA match
9 / 9 prediction_count > 0
9 / 9 no NaN/Inf
```

checkpoint、逐样本 prediction 不打包。

---

## 八、构建三 seed ensemble

创建：

```text
tools/stage8/merge_final_ensemble_predictions.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/merge_final_ensemble_predictions.py" \
  --prediction-root "$STAGE8_REPRO/predictions" \
  --seeds 20260906,20260907,20260908 \
  --suites val,test,stage3_diagnostic \
  --output-root "$STAGE8_REPRO/ensemble" \
  --manifest "$ROUND_DIR/tables/ensemble_prediction_manifest.csv" \
  --metrics "$ROUND_DIR/metrics/ensemble_merge_gate.json"
```

每个 sample 必须恰好有 3 个模型预测。Ensemble：

```text
node logits：seed 均值后 argmax
edge-type logits：seed 均值后 argmax
edge-id logits：seed 均值后 argmax
phi：seed 均值
remaining cost：seed 均值
uncertainty：seed 标准差 / entropy
```

---

## 九、重新计算模型指标

创建：

```text
tools/stage8/compute_final_model_metrics.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/compute_final_model_metrics.py" \
  --ensemble-root "$STAGE8_REPRO/ensemble" \
  --statistics-unit content_group_id \
  --output "$STAGE8_REPRO/tables/reproduced_model_metrics.csv" \
  --per-seed-output "$STAGE8_REPRO/tables/reproduced_model_metrics_by_seed.csv" \
  --report "$ROUND_DIR/reports/reproduced_model_metrics.md"
```

至少计算：

```text
node_macro_f1
edge_type_macro_f1_non_none
edge_id_macro_f1
alternative_edge_f1
recovery_edge_f1
phi_mae
phi_spearman
phi_monotonic_violation_rate
remaining_cost_mae
remaining_cost_spearman
cost_pair_accuracy
```

---

## 十、重新计算奖励与基线

创建：

```text
tools/stage8/reproduce_reward_main_table.py
```

输入方法：

```text
pathgraph_reward_v1_locked
pathgraph_cost_plus_phi
pathgraph_cost_only
linear_time_fraction
oracle_linear_chain_A_first
oracle_linear_chain_B_first
sequential_transition_oracle
learned_linear_sarm
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/reproduce_reward_main_table.py" \
  --ensemble-test "$STAGE8_REPRO/ensemble/ensemble_test_predictions.jsonl.gz" \
  --ensemble-diagnostic "$STAGE8_REPRO/ensemble/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --reward-lock "$STAGE5_REWARD/configs/reward_selection_lock.json" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --diagnostic-root "$STAGE3_DIAG" \
  --graph-spec-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --statistics-unit content_group_id \
  --output "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --trace-output "$STAGE8_REPRO/metrics/reproduced_trace_returns.csv" \
  --report "$ROUND_DIR/reports/reproduced_reward_results.md"
```

指标：

```text
legal_path_normalized_gap
failure_negative_rate
recovery_positive_rate
recovery_cycle_nonpositive_rate
positive_loop_rate
recovery_overshoot_rate
success_return_spearman
success_failure_margin
fixed_order_drop
within_node_reward_density
```

---

## 十一、重新计算核心消融

使用 Stage 7.2 已冻结变体定义，不新增变体：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/reproduce_core_ablations.py" \
  --ensemble-test "$STAGE8_REPRO/ensemble/ensemble_test_predictions.jsonl.gz" \
  --ensemble-diagnostic "$STAGE8_REPRO/ensemble/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --reference-ablation-table "$STAGE7R1_G4/tables/core_ablation_effects.csv" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --output "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  --report "$ROUND_DIR/reports/reproduced_core_ablations.md"
```

必须包含：

```text
collapse_alternative_to_A_first
collapse_alternative_to_B_first
remove_recovery_edge
no_recovery_debt_cap
no_phi
cost_only
```

`eta`/`beta` probe 可在附表复现，但不能影响主 gate。

---

## 十二、与冻结参考结果进行一致性比较

创建：

```text
tools/stage8/compare_reproduction_to_reference.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/compare_reproduction_to_reference.py" \
  --reproduced-main "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --reference-main "$STAGE7R1_G4/tables/reward_main_table.csv" \
  --reproduced-ablation "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  --reference-ablation "$STAGE7R1_G4/tables/core_ablation_effects.csv" \
  --absolute-tolerance 1e-6 \
  --relative-tolerance 1e-5 \
  --output "$ROUND_DIR/tables/reproduction_differences.csv" \
  --gate "$ROUND_DIR/metrics/core_reproduction_gate.json" \
  --report "$ROUND_DIR/reports/core_reproduction_comparison.md"
```

匹配规则：

```text
相同方法、相同指标、相同 suite、相同统计单位
```

若 reference 表中某一列只是展示层聚合、不能逐样本重算，必须标记：

```text
comparison_status = not_comparable
reason = <明确原因>
```

不得自动填 0。

允许状态：

```text
CORE_PIPELINE_REPRODUCED
FINAL_REPRODUCTION_MISMATCH
REPAIR_INFERENCE_JOB
```

进入 8.3 必须为 `CORE_PIPELINE_REPRODUCED`。

---

## 十三、本轮 ZIP

建立大文件 manifest：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_reproduction_large_file_manifest.py" \
  --checkpoint-manifest "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --prediction-root "$STAGE8_REPRO/predictions" \
  --ensemble-root "$STAGE8_REPRO/ensemble" \
  --output "$ROUND_DIR/manifests/large_file_manifest.tsv"
```

生成 ZIP：

```bash
export ZIP_NAME="stage8_2_core_pipeline_reproduction.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 8.2：CORE_PIPELINE_REPRODUCED
GPU inference jobs：9 / 9
checkpoints verified：3 / 3
ensemble suites：val / test / diagnostic
main reward table：已从新 prediction 重算
core ablations：已重算
reference comparison：PASS
ZIP：<绝对路径>
SHA256：<hash>
下一步：8.3
```

**核心点：Stage 8 的主表必须从 checkpoint 重新推理生成，不能只是重新打包旧 CSV。**

---

# 阶段 8.3：最终统计、置信区间与效应量

## 一、总体上要干什么

使用 8.2 新生成的逐样本 reward 和 prediction，完成最终统计。统计单位固定为：

```text
content_group_id
```

本轮不运行 GPU，不修改模型和奖励。

本轮状态：

```text
FINAL_STATISTICS_LOCKED
```

本轮 ZIP：

```text
stage8_3_final_statistics.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_3_final_statistics"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_STATS"/{configs,distributions,tables,figures,reports,locks}
```

入口：

```bash
grep -q 'CORE_PIPELINE_REPRODUCED' \
  "$STAGE8_ROUNDS/stage8_2_core_pipeline_reproduction/reports/core_reproduction_comparison.md"
```

---

## 三、在统计前冻结 estimand

创建：

```text
$STAGE8_STATS/configs/primary_estimands.yaml
```

内容：

```yaml
statistics_unit: content_group_id
bootstrap_resamples: 10000
bootstrap_seed: 20261401
ci: percentile_95
multiple_comparison: holm

primary_estimands:
  - id: path_consistency
    source: reproduced_reward_main_table
    metric: legal_path_normalized_gap
    direction: lower_is_better

  - id: failure_sign
    source: reproduced_reward_main_table
    metric: failure_negative_rate
    direction: higher_is_better

  - id: recovery_sign
    source: reproduced_reward_main_table
    metric: recovery_positive_rate
    direction: higher_is_better

  - id: loop_safety
    source: reproduced_reward_main_table
    metric: recovery_cycle_nonpositive_rate
    direction: higher_is_better

  - id: success_separation
    source: reproduced_reward_main_table
    metric: success_failure_margin
    direction: higher_is_better

primary_structural_contrasts:
  - id: alternative_A_collapse
    treatment: full_locked
    comparator: collapse_alternative_to_A_first
    metric: alternate_path_return

  - id: alternative_B_collapse
    treatment: full_locked
    comparator: collapse_alternative_to_B_first
    metric: alternate_path_return

  - id: remove_recovery
    treatment: full_locked
    comparator: remove_recovery_edge
    metric: recovery_positive_rate

  - id: remove_debt_cap
    treatment: full_locked
    comparator: no_recovery_debt_cap
    metric: recovery_overshoot_rate

  - id: remove_phi
    treatment: full_locked
    comparator: no_phi
    metric: within_node_reward_density

secondary:
  - node_macro_f1
  - edge_type_macro_f1_non_none
  - phi_mae
  - remaining_cost_mae
  - uncertainty_reward_error_auroc
```

锁定：

```bash
sha256sum "$STAGE8_STATS/configs/primary_estimands.yaml" \
  | tee "$STAGE8_STATS/locks/primary_estimands.sha256"
```

不得在查看 bootstrap 结果后新增 primary estimand。

---

## 四、准备逐 content-group 统计表

创建：

```text
tools/stage8/build_group_level_statistics_table.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_group_level_statistics_table.py" \
  --ensemble-test "$STAGE8_REPRO/ensemble/ensemble_test_predictions.jsonl.gz" \
  --trace-returns "$STAGE8_REPRO/metrics/reproduced_trace_returns.csv" \
  --main-table "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --ablation-table "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  --group-key content_group_id \
  --stratify task_id,provenance \
  --output "$STAGE8_STATS/tables/group_level_observations.parquet" \
  --summary "$ROUND_DIR/tables/group_level_counts.csv"
```

每行：

```text
content_group_id
task_id
provenance
path_order
scenario_type
method
variant
metric_name
metric_value
model_seed
```

检查：

```text
无 content_group 跨 split
无 NaN/Inf primary metric
不同 provenance 保留独立标签
```

---

## 五、运行 10,000 次分层配对 Bootstrap

创建：

```text
tools/stage8/run_final_group_bootstrap.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/run_final_group_bootstrap.py" \
  --observations "$STAGE8_STATS/tables/group_level_observations.parquet" \
  --estimands "$STAGE8_STATS/configs/primary_estimands.yaml" \
  --resamples "$FINAL_BOOTSTRAP_RESAMPLES" \
  --seed "$FINAL_BOOTSTRAP_SEED" \
  --parallel "$(nproc --all)" \
  --output "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --distribution-dir "$STAGE8_STATS/distributions" \
  --report "$ROUND_DIR/reports/final_bootstrap_summary.md"
```

抽样规则：

```text
按 task_id 和 provenance 分层
在层内重采样 content_group_id
配对对比保持同一 content_group 的方法结果成对
模型 seed 先在组内求均值，同时报告 seed 间标准差
```

不得把逐帧样本当作独立统计单位。

---

## 六、二元率的补充区间

对以下 rate 计算 Wilson 95% CI：

```text
failure_negative_rate
recovery_positive_rate
cycle_nonpositive_rate
positive_loop_rate
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/compute_rate_intervals.py" \
  --observations "$STAGE8_STATS/tables/group_level_observations.parquet" \
  --metrics failure_negative_rate,recovery_positive_rate,recovery_cycle_nonpositive_rate,positive_loop_rate \
  --method wilson \
  --output "$STAGE8_STATS/tables/final_rate_intervals.csv"
```

---

## 七、模型 seed 稳定性

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/summarize_reward_model_seed_stability.py" \
  --per-seed-metrics "$STAGE8_REPRO/tables/reproduced_model_metrics_by_seed.csv" \
  --output "$STAGE8_STATS/tables/model_seed_stability.csv" \
  --report "$ROUND_DIR/reports/model_seed_stability.md"
```

报告：

```text
mean
standard deviation
min
max
coefficient of variation
3/3 seed direction agreement
```

只报告 3 seed 描述，不将 3 个 seed 当作大型统计样本。

---

## 八、主假设表与 Holm 校正

创建：

```text
tools/stage8/build_primary_hypothesis_table.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_primary_hypothesis_table.py" \
  --bootstrap "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --estimands "$STAGE8_STATS/configs/primary_estimands.yaml" \
  --correction holm \
  --output "$STAGE8_STATS/tables/primary_hypothesis_results.csv" \
  --report "$ROUND_DIR/reports/primary_hypothesis_results.md"
```

字段：

```text
estimand_id
point_estimate
ci95_low
ci95_high
raw_p_or_tail_probability
adjusted_p
effect_direction
support_status
```

若样本规模不足以给出稳定 p 值，保留 CI 和 effect size，`adjusted_p=NA`，不得伪造数值。

---

## 九、最终统计摘要

创建：

```text
tools/stage8/build_final_statistical_summary.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_final_statistical_summary.py" \
  --main "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --bootstrap "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --rates "$STAGE8_STATS/tables/final_rate_intervals.csv" \
  --hypotheses "$STAGE8_STATS/tables/primary_hypothesis_results.csv" \
  --seed-stability "$STAGE8_STATS/tables/model_seed_stability.csv" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --output "$STAGE8_STATS/tables/final_result_summary.csv" \
  --report "$STAGE8_STATS/reports/final_statistical_summary.md"
```

报告必须分成：

```text
Supported primary results
Qualified auxiliary results
Unsupported / negative results
```

---

## 十、本轮 Gate

创建：

```text
tools/stage8/decide_final_statistics_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/decide_final_statistics_gate.py" \
  --estimands-lock "$STAGE8_STATS/locks/primary_estimands.sha256" \
  --bootstrap "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --hypotheses "$STAGE8_STATS/tables/primary_hypothesis_results.csv" \
  --summary "$STAGE8_STATS/tables/final_result_summary.csv" \
  --output "$ROUND_DIR/metrics/final_statistics_gate.json" \
  --report "$ROUND_DIR/reports/final_statistics_gate.md"
```

必须：

```text
10000 bootstrap 完成
统计单位 = content_group_id
primary estimand 在运行前锁定
所有 primary point estimate 有 CI
unsupported claims 未被重新提升
```

允许：

```text
FINAL_STATISTICS_LOCKED
REPAIR_BOOTSTRAP_EXECUTION
STATISTICAL_INPUT_MISMATCH
```

---

## 十一、本轮 ZIP

bootstrap 全量分布可不打包，只记录 manifest：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_statistics_large_file_manifest.py" \
  --distribution-dir "$STAGE8_STATS/distributions" \
  --observations "$STAGE8_STATS/tables/group_level_observations.parquet" \
  --output "$ROUND_DIR/manifests/large_file_manifest.tsv"
```

打包：

```bash
export ZIP_NAME="stage8_3_final_statistics.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 8.3：FINAL_STATISTICS_LOCKED
bootstrap：10000 / 10000
statistics unit：content_group_id
primary estimands：<n>
all primary estimates have CI：true
ZIP：<绝对路径>
SHA256：<hash>
下一步：8.4
```

**核心点：最终统计按 content group 配对重采样，不能用大量相关帧制造虚假的样本量。**

---

# 阶段 8.4：论文级表格与图形

## 一、总体上要干什么

把 8.2 和 8.3 的最终结果转换为论文可直接使用的：

```text
CSV / Markdown / LaTeX 表格
PDF / SVG / PNG 图形
图注
数值来源索引
```

所有图表必须从最终 CSV 自动生成，禁止在绘图脚本里硬编码实验数值。

本轮状态：

```text
PUBLICATION_ARTIFACTS_READY
```

本轮 ZIP：

```text
stage8_4_publication_tables_and_figures.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_4_publication_tables_and_figures"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_PAPER"/{tables,figures,captions,source_maps,reports}
```

入口：

```bash
grep -q 'FINAL_STATISTICS_LOCKED' \
  "$STAGE8_ROUNDS/stage8_3_final_statistics/reports/final_statistics_gate.md"
```

---

## 三、冻结图表清单

创建：

```text
$STAGE8_PAPER/source_maps/publication_artifact_plan.yaml
```

内容：

```yaml
main_tables:
  - table_1_main_reward_results
  - table_2_structural_ablations
  - table_3_model_components
  - table_4_final_claim_scope

appendix_tables:
  - table_A1_model_seed_results
  - table_A2_history_granularity
  - table_A3_uncertainty
  - table_A4_policy_secondary_mixed
  - table_A5_negative_extensions

main_figures:
  - figure_1_pathgraph_method
  - figure_2_reward_behavior
  - figure_3_structural_ablations
  - figure_4_history_and_uncertainty

appendix_figures:
  - figure_A1_per_seed_metrics
  - figure_A2_policy_secondary
  - figure_A3_auto_graph_extension
  - figure_A4_coverage_and_unseen_order_negative_results
```

说明：

```text
coverage / unseen-order / policy / auto graph 只能放 appendix 或 limitations。
```

---

## 四、生成论文表格

创建：

```text
tools/stage8/make_publication_tables.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/make_publication_tables.py" \
  --main-results "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --model-results "$STAGE8_REPRO/tables/reproduced_model_metrics.csv" \
  --ablations "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  --statistics "$STAGE8_STATS/tables/final_result_summary.csv" \
  --bootstrap "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --history "$STAGE7R1_G4/tables/history_granularity_summary.csv" \
  --uncertainty "$STAGE7R1_G4/tables/uncertainty_error_detection.csv" \
  --policy "$STAGE7R1_G4/tables/policy_secondary_evidence.csv" \
  --coverage "$STAGE7R1_G4/tables/coverage_scaling_metrics.csv" \
  --ood "$STAGE7R1_G4/tables/ood_reward_metrics.csv" \
  --claim-matrix "$STAGE7R1_G4/tables/claim_matrix_r1.csv" \
  --output-dir "$STAGE8_PAPER/tables" \
  --source-map "$STAGE8_PAPER/source_maps/table_value_sources.csv"
```

每张表输出：

```text
.csv
.md
.tex
```

表格规则：

```text
点估计后附 95% CI
unsupported 项标记为 Not supported，不使用破折号掩盖
NA 与 Not estimable 分开
所有小数位统一
主方法加粗，但不使用颜色作为唯一标识
```

---

## 五、生成方法结构图

创建：

```text
tools/stage8/draw_pathgraph_method_figure.py
```

输入：

```text
manual graph spec
reward config
claim scope
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/draw_pathgraph_method_figure.py" \
  --graph-spec-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --output-base "$STAGE8_PAPER/figures/figure_1_pathgraph_method" \
  --formats pdf,svg,png \
  --dpi 300
```

图中必须展示：

```text
alternative paths
failure edge
recovery edge
remaining-cost reduction
within-node progress
recovery debt cap
RA-BC 仅以虚线标记为 secondary downstream use
```

不得把自动图画成主流程。

---

## 六、生成结果图

创建：

```text
tools/stage8/make_publication_figures.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/make_publication_figures.py" \
  --main-results "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --ablations "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  --bootstrap "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --history "$STAGE7R1_G4/tables/history_granularity_summary.csv" \
  --uncertainty "$STAGE7R1_G4/tables/uncertainty_error_detection.csv" \
  --policy "$STAGE7R1_G4/tables/policy_secondary_evidence.csv" \
  --coverage "$STAGE7R1_G4/tables/coverage_scaling_metrics.csv" \
  --ood "$STAGE7R1_G4/tables/ood_reward_metrics.csv" \
  --auto-graph "$STAGE7R1_G4/tables/auto_graph_test_metrics.csv" \
  --output-dir "$STAGE8_PAPER/figures" \
  --formats pdf,svg,png \
  --dpi 300 \
  --source-map "$STAGE8_PAPER/source_maps/figure_value_sources.csv"
```

必须生成：

```text
figure_2_reward_behavior
  合法路径一致性
  failure / recovery 符号
  loop safety
  success-failure separation

figure_3_structural_ablations
  collapse alternative A/B
  remove recovery edge
  remove debt cap
  remove phi

figure_4_history_and_uncertainty
  history / granularity
  uncertainty risk-coverage
```

附录：

```text
figure_A1_per_seed_metrics
figure_A2_policy_secondary
figure_A3_auto_graph_extension
figure_A4_coverage_and_unseen_order_negative_results
```

负结果必须如实画出，不隐藏。

---

## 七、生成图注

创建：

```text
tools/stage8/build_caption_bank.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_caption_bank.py" \
  --artifact-plan "$STAGE8_PAPER/source_maps/publication_artifact_plan.yaml" \
  --table-source-map "$STAGE8_PAPER/source_maps/table_value_sources.csv" \
  --figure-source-map "$STAGE8_PAPER/source_maps/figure_value_sources.csv" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --output "$STAGE8_PAPER/captions/caption_bank.md"
```

每个图注必须说明：

```text
统计单位
误差条类型
样本来源
是否为 controlled symbolic stress
是否为 secondary / negative result
```

---

## 八、图表检查

创建：

```text
tools/stage8/check_publication_artifacts.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/check_publication_artifacts.py" \
  --tables "$STAGE8_PAPER/tables" \
  --figures "$STAGE8_PAPER/figures" \
  --captions "$STAGE8_PAPER/captions/caption_bank.md" \
  --table-source-map "$STAGE8_PAPER/source_maps/table_value_sources.csv" \
  --figure-source-map "$STAGE8_PAPER/source_maps/figure_value_sources.csv" \
  --output "$ROUND_DIR/metrics/publication_artifact_gate.json" \
  --report "$ROUND_DIR/reports/publication_artifact_summary.md"
```

只检查必要项：

```text
计划中的文件全部存在
PDF/SVG 可打开
PNG 分辨率达标
每个数值有来源
无 NaN/Inf 显示
unsupported claim 未进入 main table/figure 标题
```

允许：

```text
PUBLICATION_ARTIFACTS_READY
REPAIR_FIGURE_RENDERING
MISSING_VALUE_SOURCE
```

---

## 九、本轮 ZIP

图表本身进入 ZIP。

```bash
cp -r "$STAGE8_PAPER/tables" "$ROUND_DIR/"
cp -r "$STAGE8_PAPER/figures" "$ROUND_DIR/"
cp -r "$STAGE8_PAPER/captions" "$ROUND_DIR/"
cp -r "$STAGE8_PAPER/source_maps" "$ROUND_DIR/"

export ZIP_NAME="stage8_4_publication_tables_and_figures.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 8.4：PUBLICATION_ARTIFACTS_READY
main tables：4
appendix tables：5
main figures：4
appendix figures：4
all values source-mapped：true
ZIP：<绝对路径>
SHA256：<hash>
下一步：8.5
```

**核心点：所有图表都从最终结果表自动生成，并明确把负结果和次级结果放在正确位置。**

---

# 阶段 8.5：论文结果材料与主张映射

## 一、总体上要干什么

本小阶段生成可直接用于论文写作的证据材料，但不进行新的文献检索，也不凭空补充实验。

输出：

```text
final_contribution_statement.md
method_section.md
experimental_setup.md
results_section.md
ablation_section.md
limitations.md
abstract_result_sentences.md
caption_bank.md
claim_to_evidence_map.csv
reviewer_question_bank.md
```

本轮状态：

```text
MANUSCRIPT_EVIDENCE_PACKAGE_READY
```

本轮 ZIP：

```text
stage8_5_manuscript_evidence_package.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_5_manuscript_evidence_package"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_PAPER/manuscript"
```

入口：

```bash
grep -q 'PUBLICATION_ARTIFACTS_READY' \
  "$STAGE8_ROUNDS/stage8_4_publication_tables_and_figures/reports/publication_artifact_summary.md"
```

---

## 三、生成最终贡献声明

创建：

```text
tools/stage8/write_contribution_statement.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_contribution_statement.py" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --claim-matrix "$STAGE7R1_G4/tables/claim_matrix_r1.csv" \
  --final-results "$STAGE8_STATS/tables/final_result_summary.csv" \
  --output "$STAGE8_PAPER/manuscript/final_contribution_statement.md"
```

贡献声明必须只包括：

```text
图结构 reward representation
多合法路径
failure/recovery
remaining cost + within-node progress
loop-safe recovery accounting
```

不把 policy、scaling、unseen-order 和 auto graph 写成贡献。

---

## 四、生成方法部分

创建：

```text
tools/stage8/write_method_section.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_method_section.py" \
  --graph-spec-root "$STAGE3_INPUT/runtime_graph_specs_v1.0.1" \
  --reward-config "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --output "$STAGE8_PAPER/manuscript/method_section.md" \
  --equations "$STAGE8_PAPER/manuscript/method_equations.tex"
```

必须包含：

\[
r_t =
C_G(z_t,h_t)-C_G(z_{t+1},h_{t+1})
+
\lambda[\phi_{z_t}(o_{t+1})-\phi_{z_t}(o_t)]
-
\eta n_{\mathrm{loop}}(e_t)
\]

以及：

\[
w_t=\max(0,\mathbb E[r_t]-\beta\operatorname{Std}[r_t])
\]

同时明确最终锁定配置中：

```text
eta = 0
beta = 0
```

所以非零 loop penalty 和 uncertainty LCB 不作为主结果贡献。

---

## 五、生成实验设置

创建：

```text
tools/stage8/write_experimental_setup.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_experimental_setup.py" \
  --input-index "$STAGE8_INPUTS/configs/final_input_index.yaml" \
  --inference-protocol "$STAGE8_REPRO/configs/final_inference_protocol.yaml" \
  --estimands "$STAGE8_STATS/configs/primary_estimands.yaml" \
  --checkpoint-manifest "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --output "$STAGE8_PAPER/manuscript/experimental_setup.md"
```

必须写清：

```text
任务
图节点/边
GT 和 split
三模型 seed
content_group 统计单位
baseline
核心消融
bootstrap 方法
controlled symbolic stress 的定位
policy evidence 的 secondary 定位
```

---

## 六、生成 Results 与 Ablation 文本

创建：

```text
tools/stage8/write_results_sections.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_results_sections.py" \
  --main-table "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  --model-table "$STAGE8_REPRO/tables/reproduced_model_metrics.csv" \
  --ablation-table "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  --statistics "$STAGE8_STATS/tables/final_result_summary.csv" \
  --bootstrap "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  --history "$STAGE7R1_G4/tables/history_granularity_summary.csv" \
  --uncertainty "$STAGE7R1_G4/tables/uncertainty_error_detection.csv" \
  --policy "$STAGE7R1_G4/tables/policy_secondary_evidence.csv" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --results-output "$STAGE8_PAPER/manuscript/results_section.md" \
  --ablation-output "$STAGE8_PAPER/manuscript/ablation_section.md"
```

文本规则：

```text
每个数字从 CSV 读取
每个 primary result 给 CI
不写“显著”除非统计表支持
policy 统一使用 trend / mixed evidence
coverage / unseen-order 统一作为 negative result
```

---

## 七、生成 Limitations

创建：

```text
tools/stage8/write_limitations.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_limitations.py" \
  --claim-matrix "$STAGE7R1_G4/tables/claim_matrix_r1.csv" \
  --g4 "$STAGE7R1_G4/metrics/g4_r1_decision.json" \
  --coverage "$STAGE7R1_G4/tables/coverage_scaling_metrics.csv" \
  --ood "$STAGE7R1_G4/tables/ood_reward_metrics.csv" \
  --policy "$STAGE7R1_G4/tables/policy_secondary_evidence.csv" \
  --auto-graph "$STAGE7R1_G4/tables/auto_graph_test_metrics.csv" \
  --output "$STAGE8_PAPER/manuscript/limitations.md"
```

必须明确：

```text
人工图仍是主方法
coverage scaling 不成立
未见顺序泛化不成立
policy 跨 seed 不稳定
数据任务数量有限
controlled stress 不等于真实机器人泛化
```

---

## 八、生成摘要结果句和图注索引

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_abstract_result_sentences.py" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --final-summary "$STAGE8_STATS/tables/final_result_summary.csv" \
  --output "$STAGE8_PAPER/manuscript/abstract_result_sentences.md"

cp "$STAGE8_PAPER/captions/caption_bank.md" \
  "$STAGE8_PAPER/manuscript/caption_bank.md"
```

摘要结果句不得包含 unsupported claim。

---

## 九、建立 claim-to-evidence map

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_claim_to_evidence_map.py" \
  --claim-registry "$STAGE8_INPUTS/manifests/final_evidence_registry.csv" \
  --tables "$STAGE8_PAPER/tables" \
  --figures "$STAGE8_PAPER/figures" \
  --manuscript "$STAGE8_PAPER/manuscript" \
  --output "$STAGE8_PAPER/manuscript/claim_to_evidence_map.csv"
```

字段：

```text
claim_id
sentence_id
section
table_or_figure
metric
source_csv
support_status
qualifier
```

---

## 十、生成审稿问题库

创建：

```text
tools/stage8/build_reviewer_question_bank.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_reviewer_question_bank.py" \
  --claim-map "$STAGE8_PAPER/manuscript/claim_to_evidence_map.csv" \
  --limitations "$STAGE8_PAPER/manuscript/limitations.md" \
  --output "$STAGE8_PAPER/manuscript/reviewer_question_bank.md"
```

至少覆盖：

```text
为什么不用线性 stage chain
为什么人工图
recovery 如何避免刷分
为什么 policy 不是主结论
为什么不声称 unseen-order 泛化
数据规模限制
与 planner / stage-transition reward 的边界
```

回答只能使用现有证据，不新增事实。

---

## 十一、文本 Gate

创建：

```text
tools/stage8/check_manuscript_claims.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/check_manuscript_claims.py" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --manuscript-dir "$STAGE8_PAPER/manuscript" \
  --claim-map "$STAGE8_PAPER/manuscript/claim_to_evidence_map.csv" \
  --output "$ROUND_DIR/metrics/manuscript_evidence_gate.json" \
  --report "$ROUND_DIR/reports/manuscript_evidence_summary.md"
```

检查：

```text
每个数值在 source map 中存在
unsupported claim 未作为正面结论
eta/beta 没有被写成已验证主贡献
policy 明确 secondary/mixed
manual graph 明确为主方法
```

允许：

```text
MANUSCRIPT_EVIDENCE_PACKAGE_READY
REPAIR_CLAIM_WORDING
MISSING_EVIDENCE_LINK
```

---

## 十二、本轮 ZIP

```bash
cp -r "$STAGE8_PAPER/manuscript" "$ROUND_DIR/"
cp "$STAGE8_INPUTS/manifests/final_evidence_registry.csv" "$ROUND_DIR/manifests/"

export ZIP_NAME="stage8_5_manuscript_evidence_package.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

Agent 回复：

```text
阶段 8.5：MANUSCRIPT_EVIDENCE_PACKAGE_READY
primary claims：<n>
unsupported claims kept out of conclusions：true
numeric statements source-mapped：true
policy evidence wording：secondary/mixed
ZIP：<绝对路径>
SHA256：<hash>
下一步：8.6
```

**核心点：论文文字必须由最终结果表驱动，不能在结论中重新扩大已被冻结的研究边界。**

---

# 阶段 8.6：轻量可复现发布包

## 一、总体上要干什么

建立一个不包含 checkpoint 和大数据、但能够让 Agent 或研究者：

```text
验证发布包
从已有 checkpoint 重新推理
从缓存 prediction 重算指标
重算统计
重画图表
定位所有外部大文件
```

的轻量发布包。

本轮状态：

```text
REPRODUCIBILITY_BUNDLE_READY
```

本轮 ZIP：

```text
stage8_6_reproducibility_bundle.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_6_reproducibility_bundle"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_RELEASE"/{configs,code,scripts,manifests,docs,tests,results}
```

---

## 三、复制冻结配置与结果

复制：

```bash
cp "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  "$STAGE8_RELEASE/configs/"
cp "$STAGE8_INPUTS/configs/final_input_index.yaml" \
  "$STAGE8_RELEASE/configs/"
cp "$STAGE8_REPRO/configs/final_inference_protocol.yaml" \
  "$STAGE8_RELEASE/configs/"
cp "$STAGE8_STATS/configs/primary_estimands.yaml" \
  "$STAGE8_RELEASE/configs/"
cp "$STAGE5_REWARD/configs/reward_config_v1.yaml" \
  "$STAGE8_RELEASE/configs/"
cp "$STAGE5_REWARD/configs/reward_selection_lock.json" \
  "$STAGE8_RELEASE/configs/"

cp "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  "$STAGE8_RELEASE/results/"
cp "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  "$STAGE8_RELEASE/results/"
cp "$STAGE8_REPRO/tables/reproduced_model_metrics.csv" \
  "$STAGE8_RELEASE/results/"
cp "$STAGE8_STATS/tables/final_result_summary.csv" \
  "$STAGE8_RELEASE/results/"
cp "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  "$STAGE8_RELEASE/results/"
```

复制论文图表：

```bash
cp -r "$STAGE8_PAPER/tables" "$STAGE8_RELEASE/results/"
cp -r "$STAGE8_PAPER/figures" "$STAGE8_RELEASE/results/"
cp -r "$STAGE8_PAPER/manuscript" "$STAGE8_RELEASE/docs/"
```

---

## 四、复制最小代码快照

只复制本研究实际依赖的代码：

```text
Stage 4 model definition / dataset loader
Stage 5 reward engine
Stage 8 inference / metrics / stats / figure scripts
必要的 graph spec reader
```

创建：

```text
tools/stage8/build_minimal_code_snapshot.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_minimal_code_snapshot.py" \
  --repo-root "$REPO_ROOT" \
  --stage8-tools "$STAGE8_TOOLS" \
  --reward-engine "$STAGE5_REWARD/code/reward_engine.py" \
  --model-bundle "$STAGE6_PERSISTENT/configs/model_bundle_persistent.json" \
  --output "$STAGE8_RELEASE/code" \
  --manifest "$STAGE8_RELEASE/manifests/code_manifest.tsv"
```

不要复制整个仓库。

---

## 五、生成环境记录

执行：

```bash
git rev-parse HEAD \
  | tee "$STAGE8_RELEASE/manifests/git_commit.txt"

git status --short \
  | tee "$STAGE8_RELEASE/manifests/git_status.txt"

"$PYTHON_BIN" --version \
  2>&1 | tee "$STAGE8_RELEASE/manifests/python_version.txt"

"$PYTHON_BIN" -m pip freeze \
  > "$STAGE8_RELEASE/manifests/pip_freeze.txt"

"$PYTHON_BIN" - <<'PY' \
  > "$STAGE8_RELEASE/manifests/torch_cuda_environment.txt"
import torch
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("cudnn", torch.backends.cudnn.version())
PY
```

若使用 conda：

```bash
conda env export --no-builds \
  > "$STAGE8_RELEASE/manifests/environment.yml" || true
```

---

## 六、生成运行脚本

### `scripts/verify_release.sh`

必须：

```text
检查内部 SHA256
检查配置和 compact results
检查 external_artifact_manifest 路径格式
不要求 checkpoint 实际存在
```

### `scripts/reproduce_from_checkpoints.sh`

参数：

```bash
bash scripts/reproduce_from_checkpoints.sh \
  --model-bundle /path/model_bundle_persistent.json \
  --supervision /path/supervision \
  --diagnostic /path/diagnostic_suite \
  --output /path/output \
  --gpus auto
```

执行：

```text
9 个真实推理 job
ensemble
model metrics
reward main table
core ablations
```

### `scripts/reproduce_from_cached_predictions.sh`

参数：

```bash
bash scripts/reproduce_from_cached_predictions.sh \
  --ensemble-test /path/ensemble_test_predictions.jsonl.gz \
  --ensemble-diagnostic /path/ensemble_diagnostic_predictions.jsonl.gz \
  --output /path/output
```

执行：

```text
reward metrics
core ablations
final statistics
```

### `scripts/reproduce_figures.sh`

参数：

```bash
bash scripts/reproduce_figures.sh \
  --results-dir /path/results \
  --output /path/figures
```

---

## 七、生成外部大文件清单

合并：

```text
final_checkpoint_manifest.tsv
final_large_file_manifest.tsv
Stage 8 reproduction large-file manifest
Stage 8 statistics large-file manifest
```

创建：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_release_external_manifest.py" \
  --checkpoint-manifest "$STAGE8_INPUTS/manifests/final_checkpoint_manifest.tsv" \
  --input-large-files "$STAGE8_INPUTS/manifests/final_large_file_manifest.tsv" \
  --reproduction-large-files "$STAGE8_ROUNDS/stage8_2_core_pipeline_reproduction/manifests/large_file_manifest.tsv" \
  --statistics-large-files "$STAGE8_ROUNDS/stage8_3_final_statistics/manifests/large_file_manifest.tsv" \
  --output "$STAGE8_RELEASE/manifests/external_artifact_manifest.tsv"
```

---

## 八、README

创建：

```text
$STAGE8_RELEASE/README.md
```

至少包含：

```text
研究最终边界
目录结构
快速验证
使用 cached prediction 复现
使用 checkpoint 复现
GPU 查询和并行方式
外部文件清单
不包含哪些大文件
结果表对应论文位置
已知限制
```

---

## 九、最小运行检查

不做全量重复实验，只做：

```bash
bash "$STAGE8_RELEASE/scripts/verify_release.sh" \
  | tee "$ROUND_DIR/logs/verify_release.log"

bash "$STAGE8_RELEASE/scripts/reproduce_from_cached_predictions.sh" \
  --help \
  | tee "$ROUND_DIR/logs/cached_reproduce_help.txt"

bash "$STAGE8_RELEASE/scripts/reproduce_from_checkpoints.sh" \
  --help \
  | tee "$ROUND_DIR/logs/checkpoint_reproduce_help.txt"

bash "$STAGE8_RELEASE/scripts/reproduce_figures.sh" \
  --help \
  | tee "$ROUND_DIR/logs/figure_reproduce_help.txt"

"$PYTHON_BIN" -m compileall -q "$STAGE8_RELEASE/code"
```

再用 compact example / first 10 groups 运行一次 dry-run：

```bash
bash "$STAGE8_RELEASE/scripts/reproduce_from_cached_predictions.sh" \
  --ensemble-test "$STAGE8_REPRO/ensemble/ensemble_test_predictions.jsonl.gz" \
  --ensemble-diagnostic "$STAGE8_REPRO/ensemble/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --output "$ROUND_DIR/logs/dry_run_output" \
  --max-content-groups 10 \
  --dry-run
```

---

## 十、生成发布包 checksum

```bash
(
  cd "$STAGE8_RELEASE"
  find . -type f \
    ! -name 'RELEASE_SHA256SUMS.txt' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$STAGE8_RELEASE/RELEASE_SHA256SUMS.txt"

(
  cd "$STAGE8_RELEASE"
  sha256sum -c RELEASE_SHA256SUMS.txt
) | tee "$ROUND_DIR/checksums/release_internal_check.txt"
```

---

## 十一、本轮 Gate

创建：

```text
tools/stage8/decide_reproducibility_bundle_gate.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/decide_reproducibility_bundle_gate.py" \
  --release-root "$STAGE8_RELEASE" \
  --verify-log "$ROUND_DIR/logs/verify_release.log" \
  --dry-run "$ROUND_DIR/logs/dry_run_output" \
  --external-manifest "$STAGE8_RELEASE/manifests/external_artifact_manifest.tsv" \
  --output "$ROUND_DIR/metrics/reproducibility_bundle_gate.json" \
  --report "$ROUND_DIR/reports/reproducibility_bundle_summary.md"
```

必须：

```text
internal checksums pass
scripts --help pass
Python compile pass
cached dry-run pass
checkpoint/data clearly externalized
checkpoint not packaged
```

允许：

```text
REPRODUCIBILITY_BUNDLE_READY
REPAIR_RELEASE_SCRIPT
MISSING_RELEASE_ARTIFACT
```

---

## 十二、本轮 ZIP

将 release 根目录整体打包：

```bash
cd "$STAGE8_RELEASE"
zip -qr "$STAGE8_DOWNLOADS/stage8_6_reproducibility_bundle.zip" .
cd "$REPO_ROOT"

unzip -t "$STAGE8_DOWNLOADS/stage8_6_reproducibility_bundle.zip" \
  | tee "$ROUND_DIR/checksums/stage8_6_reproducibility_bundle_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/stage8_6_reproducibility_bundle.zip" \
  | tee "$ROUND_DIR/checksums/stage8_6_reproducibility_bundle.sha256"
```

Agent 回复：

```text
阶段 8.6：REPRODUCIBILITY_BUNDLE_READY
internal SHA：PASS
cached dry-run：PASS
checkpoint/data：外置 manifest
checkpoint packaged：false
ZIP：<绝对路径>
SHA256：<hash>
下一步：8.7
```

**核心点：发布包保持轻量，但提供从 checkpoint 或缓存 prediction 重建核心结果的完整入口。**

---

# 阶段 8.7：M6 最终冻结与项目关闭

## 一、总体上要干什么

汇总阶段 8 的六轮结果，生成：

```text
M6 final decision
最终结果索引
最终论文材料索引
最终复现索引
项目关闭报告
stage8_complete.zip
```

本轮不再运行实验。

本轮状态：

```text
RESEARCH_COMPLETE_CORE_REWARD_ONLY
```

本轮 ZIP：

```text
stage8_7_m6_final_freeze.zip
```

总 ZIP：

```text
stage8_complete.zip
```

---

## 二、建立目录

```bash
source artifacts/pathgraph_sarm/stage8_core_reward/stage8_env.sh
cd "$REPO_ROOT"

export ROUND_NAME="stage8_7_m6_final_freeze"
export ROUND_DIR="$STAGE8_ROUNDS/$ROUND_NAME"

mkdir -p \
  "$ROUND_DIR"/{configs,commands,gpu,logs,metrics,tables,figures,reports,manifests,checksums} \
  "$STAGE8_M6"/{configs,locks,metrics,tables,figures,reports,manifests,release}
```

---

## 三、检查前六轮 Gate

必须为：

```text
8.1 FINAL_SCOPE_AND_INPUTS_LOCKED
8.2 CORE_PIPELINE_REPRODUCED
8.3 FINAL_STATISTICS_LOCKED
8.4 PUBLICATION_ARTIFACTS_READY
8.5 MANUSCRIPT_EVIDENCE_PACKAGE_READY
8.6 REPRODUCIBILITY_BUNDLE_READY
```

创建：

```text
tools/stage8/collect_stage8_gates.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/collect_stage8_gates.py" \
  --round-root "$STAGE8_ROUNDS" \
  --output "$ROUND_DIR/tables/stage8_gate_summary.csv" \
  --gate "$ROUND_DIR/metrics/stage8_all_gates.json"
```

若任一基础设施 gate 未通过，只修复该轮；不得修改方法。

---

## 四、复制最终紧凑结果

```bash
cp "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  "$STAGE8_M6/locks/"
cp "$STAGE8_INPUTS/configs/final_input_index.yaml" \
  "$STAGE8_M6/configs/"
cp "$STAGE8_REPRO/configs/final_inference_protocol.yaml" \
  "$STAGE8_M6/configs/"
cp "$STAGE8_STATS/configs/primary_estimands.yaml" \
  "$STAGE8_M6/configs/"

cp "$STAGE8_REPRO/tables/reproduced_model_metrics.csv" \
  "$STAGE8_M6/tables/"
cp "$STAGE8_REPRO/tables/reproduced_reward_main_table.csv" \
  "$STAGE8_M6/tables/"
cp "$STAGE8_REPRO/tables/reproduced_core_ablation_effects.csv" \
  "$STAGE8_M6/tables/"
cp "$STAGE8_STATS/tables/final_result_summary.csv" \
  "$STAGE8_M6/tables/"
cp "$STAGE8_STATS/tables/final_bootstrap_effects.csv" \
  "$STAGE8_M6/tables/"
cp "$STAGE8_STATS/tables/primary_hypothesis_results.csv" \
  "$STAGE8_M6/tables/"

cp -r "$STAGE8_PAPER/tables" "$STAGE8_M6/"
cp -r "$STAGE8_PAPER/figures" "$STAGE8_M6/"
cp -r "$STAGE8_PAPER/manuscript" "$STAGE8_M6/"
```

复制复现包的 README、manifest 和 checksum，不重复嵌套整个大 ZIP：

```bash
cp "$STAGE8_RELEASE/README.md" "$STAGE8_M6/release/"
cp "$STAGE8_RELEASE/RELEASE_SHA256SUMS.txt" "$STAGE8_M6/release/"
cp -r "$STAGE8_RELEASE/manifests" "$STAGE8_M6/release/"
```

---

## 五、生成最终 artifact index

创建：

```text
tools/stage8/build_final_artifact_index.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/build_final_artifact_index.py" \
  --m6-root "$STAGE8_M6" \
  --round-zip-dir "$STAGE8_DOWNLOADS" \
  --external-manifest "$STAGE8_RELEASE/manifests/external_artifact_manifest.tsv" \
  --output "$STAGE8_M6/manifests/final_artifact_index.csv" \
  --report "$STAGE8_M6/reports/final_artifact_index.md"
```

字段：

```text
artifact_id
artifact_type
path
sha256
packaged
external
paper_usage
reproduce_command
```

---

## 六、生成最终结论报告

创建：

```text
tools/stage8/write_final_research_report.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/write_final_research_report.py" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --final-results "$STAGE8_STATS/tables/final_result_summary.csv" \
  --claim-map "$STAGE8_PAPER/manuscript/claim_to_evidence_map.csv" \
  --g4-r1 "$STAGE7R1_G4/metrics/g4_r1_decision.json" \
  --policy-secondary "$STAGE7R1_G4/tables/policy_secondary_evidence.csv" \
  --artifact-index "$STAGE8_M6/manifests/final_artifact_index.csv" \
  --output "$STAGE8_M6/reports/final_research_report.md"
```

报告固定分为：

```text
1. Final research question
2. Final method
3. Supported contributions
4. Core quantitative results
5. Negative / unsupported results
6. Downstream policy evidence
7. Reproducibility
8. Remaining limitations
9. Recommended paper positioning
```

---

## 七、M6 决策

创建：

```text
tools/stage8/decide_m6.py
```

执行：

```bash
"$PYTHON_BIN" "$STAGE8_TOOLS/decide_m6.py" \
  --all-gates "$ROUND_DIR/metrics/stage8_all_gates.json" \
  --reproduction-gate "$STAGE8_ROUNDS/stage8_2_core_pipeline_reproduction/metrics/core_reproduction_gate.json" \
  --statistics-gate "$STAGE8_ROUNDS/stage8_3_final_statistics/metrics/final_statistics_gate.json" \
  --publication-gate "$STAGE8_ROUNDS/stage8_4_publication_tables_and_figures/metrics/publication_artifact_gate.json" \
  --manuscript-gate "$STAGE8_ROUNDS/stage8_5_manuscript_evidence_package/metrics/manuscript_evidence_gate.json" \
  --release-gate "$STAGE8_ROUNDS/stage8_6_reproducibility_bundle/metrics/reproducibility_bundle_gate.json" \
  --claim-scope "$STAGE8_INPUTS/locks/final_claim_scope.json" \
  --output "$STAGE8_M6/metrics/m6_decision.json" \
  --report "$STAGE8_M6/reports/m6_decision.md"
```

### 通过条件

```text
核心 checkpoint 推理已复现
主 reward 表已从新 prediction 重算
核心消融已复现
最终统计完成
论文图表来源可追踪
unsupported claim 未进入主结论
轻量发布包可验证
```

输出：

```text
RESEARCH_COMPLETE_CORE_REWARD_ONLY
REPAIR_STAGE8_INFRA
FINAL_REPRODUCTION_MISMATCH
```

---

## 八、冻结 M6

创建：

```text
$STAGE8_M6/FROZEN.md
```

```bash
cat > "$STAGE8_M6/FROZEN.md" <<EOF
milestone = M6_FINAL
decision = RESEARCH_COMPLETE_CORE_REWARD_ONLY
research_mode = core_reward_only
main_method = pathgraph_reward_v1_locked
manual_graph_is_main = true
stable_policy_claim = false
coverage_scaling_claim = false
unseen_order_claim = false
auto_graph_main_claim = false
policy_evidence = secondary_mixed
new_training_after_m6 = false
checkpoint_packaging = omitted_by_default
statistics_unit = content_group_id
EOF
```

生成内部 SHA：

```bash
(
  cd "$STAGE8_M6"
  find . -type f \
    ! -name 'M6_SHA256SUMS.txt' \
    -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$STAGE8_M6/M6_SHA256SUMS.txt"

(
  cd "$STAGE8_M6"
  sha256sum -c M6_SHA256SUMS.txt
) | tee "$ROUND_DIR/checksums/m6_internal_check.txt"
```

---

## 九、本轮 ZIP

```bash
cp "$STAGE8_M6/metrics/m6_decision.json" "$ROUND_DIR/metrics/"
cp "$STAGE8_M6/reports/m6_decision.md" "$ROUND_DIR/reports/"
cp "$STAGE8_M6/reports/final_research_report.md" "$ROUND_DIR/reports/"
cp "$STAGE8_M6/manifests/final_artifact_index.csv" "$ROUND_DIR/manifests/"
cp "$STAGE8_M6/FROZEN.md" "$ROUND_DIR/reports/"
cp "$STAGE8_M6/M6_SHA256SUMS.txt" "$ROUND_DIR/checksums/"

export ZIP_NAME="stage8_7_m6_final_freeze.zip"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-dir "$ROUND_DIR" \
  --output-zip "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/$ZIP_NAME" \
  | tee "$ROUND_DIR/checksums/${ZIP_NAME%.zip}.sha256"
```

---

## 十、生成阶段总 ZIP

`stage8_complete.zip` 包含：

```text
M6 compact results
final configs and locks
final statistics
publication tables and figures
manuscript evidence materials
release manifests and README
seven round ZIP SHA256
final artifact index
```

不包含：

```text
checkpoint
原始数据
逐帧 prediction
bootstrap 全量分布
缓存
```

执行：

```bash
cd "$STAGE8_M6"
zip -qr "$STAGE8_DOWNLOADS/stage8_complete.zip" .
cd "$REPO_ROOT"

unzip -t "$STAGE8_DOWNLOADS/stage8_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage8_complete_unzip_test.txt"

sha256sum "$STAGE8_DOWNLOADS/stage8_complete.zip" \
  | tee "$ROUND_DIR/checksums/stage8_complete.sha256"
```

总 ZIP 生成后，不删除七个小阶段 ZIP。

---

## 十一、Agent 最终回复

```text
阶段 8 已完成。
M6：RESEARCH_COMPLETE_CORE_REWARD_ONLY

核心 checkpoint 推理：9 / 9
模型 checkpoint：3 / 3 SHA256 verified
核心 reward 结果：独立复现 PASS
核心消融：独立复现 PASS
bootstrap：10000 / 10000
论文主表：4
论文主图：4
unsupported claims excluded：true
policy evidence：secondary/mixed
checkpoint packaged：false

唯一总交付 ZIP：
<绝对路径>/stage8_complete.zip

SHA256：
<hash>
```

**核心点：M6 关闭研究实验主线，后续只进入正式论文撰写、投稿格式适配或新增数据的新研究周期。**
