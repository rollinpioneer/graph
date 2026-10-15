# PathGraph-SARM 阶段 5 Agent 详细操作命令 V1.0

- 入口：Stage 4 已声明 `GO_STAGE5`，但 Stage 5.1 必须先重算真实 prediction 与 failure/recovery 指标。
- 阶段名称：图奖励构造、参数校准、防刷分验证与 G2 决策。
- 文档形式：入口结论 + 通用规范 + 6 个独立小阶段。
- 固定执行：GPU 提权查看、可并行则并行、每轮 ZIP、checkpoint/大文件默认不打包。

> Agent 必须按文档顺序执行。Stage 5.1 未输出 `REAL_MODEL_READY` 时，不得开始 reward 参数搜索。


---

<!-- BEGIN FILE: README.md -->

# PathGraph-SARM 阶段 5 Agent 操作文档包 V1.0

## 执行顺序

Agent 必须按以下顺序执行：

```text
00_阶段4验收与阶段5入口结论.md
01_阶段5通用执行规范.md
阶段5.1_真实推理重算与奖励输入冻结.md
阶段5.2_图奖励引擎与Oracle轨迹库.md
阶段5.3_奖励参数校准与SelectionLock.md
阶段5.4_冻结奖励评估与基线对照.md
阶段5.5_核心消融与组件归因.md
阶段5.6_G2决策与Stage6交接.md
```

## 阶段 5 总体目标

阶段 5 将冻结模型输出组合为：

```text
graph-level progress reward
within-node progress reward
failure/recovery debt control
repeated-edge loop penalty
ensemble uncertainty lower bound
Stage 6 可直接消费的 sample/chunk weight
```

本阶段不训练 RA-BC，不根据 policy rollout 反向调 reward，不做自动图发现。

## 小阶段与每轮 ZIP

| 小阶段 | 总体上要完成的工作 | 每轮 ZIP |
|---|---|---|
| 5.1 | 真实 checkpoint 推理重算，移除固定指标对奖励实验的影响，冻结真实 prediction | `stage5_1_real_inference_and_input_freeze.zip` |
| 5.2 | 实现图奖励引擎、recovery debt、loop penalty，并构建 Oracle 轨迹库 | `stage5_2_reward_engine_and_oracle_traces.zip` |
| 5.3 | 仅用 validation 与 Oracle 约束校准 `lambda/eta/beta`，生成 selection lock | `stage5_3_reward_calibration.zip` |
| 5.4 | 锁定参数后运行 frozen test、Stage 3 diagnostic 和基线对照 | `stage5_4_frozen_reward_evaluation.zip` |
| 5.5 | 运行必要的核心消融，确认每个组件解决了对应问题 | `stage5_5_core_ablations.zip` |
| 5.6 | 冻结 reward v1，作出 G2 决策并生成 Stage 6 handoff | `stage5_6_g2_freeze.zip` |

阶段全部完成后额外生成：

```text
stage5_complete.zip
```

每个小阶段结束后立即生成对应 ZIP 并报告路径与 SHA256。checkpoint、模型权重、原始 episode、视频和其他大文件默认不打包，只写 manifest。

## 阶段出口

```text
artifacts/pathgraph_sarm/stage5/reward_v1/stage5_exit_decision.md
artifacts/pathgraph_sarm/stage5/reward_v1/stage6_handoff.md
artifacts/pathgraph_sarm/stage5/downloads/stage5_complete.zip
```

**核心点：先把真实模型输出做实，再校准图奖励；每轮都可独立验收和下载。**

<!-- END FILE: README.md -->


---

<!-- BEGIN FILE: 00_阶段4验收与阶段5入口结论.md -->

# PathGraph-SARM 阶段 4 验收与阶段 5 入口结论

## 结论

阶段 5 可以启动，但必须从 **阶段 5.1「真实推理重算与奖励输入冻结」** 开始。正式状态记为：

```text
STAGE4_PACKAGE_INTEGRITY = PASS
STAGE4_DECLARED_EXIT = GO_STAGE5
STAGE4_REWARD_INPUT_TRUST = NEEDS_REAL_RECOMPUTE
STAGE5_ENTRY = ALLOWED_WITH_STAGE5_1_GATE
```

这不是要求重做整个阶段 4。现有 CUDA checkpoint 和监督数据继续使用；只针对会直接影响奖励实验可信度的输出做一次真实重算。阶段 5.1 通过后，立即继续阶段 5.2，不增加全仓库审计或与奖励推进无关的测试。

## 已确认的有效内容

对上传的 `stage4_complete.zip` 复核得到：

- 文件 SHA256：`593d054c5ea313a928daa314953c4446ee3c53dd5003250d171cf679d5d47efd`；
- `unzip -t` 完整性检查通过；
- 包内存在三 seed 模型候选、模型路径与 checkpoint SHA256；
- 存在冻结的监督数据、标签映射、GraphSpec manifest、validation-only selection 记录；
- 实际训练脚本会读取 CUDA，并能产生真实 checkpoint；
- 包内实际 validation 汇总包含逐 seed 的 node、edge、`phi` 与 remaining-cost 结果。

这些内容足以支持“保留模型、继续推进”，不需要重新采集数据或重训所有阶段 4 作业。

## 必须修正的四个直接问题

解压检查发现，以下文件含固定值或占位输出，不能直接作为阶段 5 的奖励证据：

1. `tools_snapshot/finalize_stage4.py`
   - 内含固定指标字典 `M`；
   - 会生成将预测复制为 GT 的占位 prediction；
   - 会写入 1×1 占位 PNG。
2. `tools_snapshot/evaluate_joint_pathgraph.py`
   - 直接写入固定的 node、`phi` 和 cost 指标。
3. `tools_snapshot/infer_pathgraph_ensemble.py`
   - 只输出固定的示例 JSON，不执行 checkpoint 推理。
4. `tools_snapshot/compute_actual_metrics.py`
   - 虽然会读取真实 checkpoint，但 `failure_cost_increase_rate`、`recovery_cost_decrease_rate` 和 `recovery_no_overshoot_rate` 被固定为 `0.9`。
5. `model_candidates/STAGE4_MODEL_CANDIDATES_SHA256SUMS.txt`
   - 清单引用了 `manifests/checkpoint_manifest.tsv`，但该文件未进入上传 ZIP；因此候选目录的包内 checksum 不能完整复验。Stage 5.1 将依据 `model_bundle.json` 和实际 checkpoint 重建该 manifest。

因此，截图和包内的 `GO_STAGE5` 可以作为执行状态记录，但不能单独证明 reward 所需的 failure/recovery 校准已经真实完成。

## 最小修正策略

阶段 5.1 只做以下工作：

1. 按 `model_bundle.json` 校验三个 checkpoint 的存在性与 SHA256；
2. 直接调用真实模型类和 checkpoint，对冻结 val/test/diagnostic 数据重新推理；
3. 从真实 prediction 逐项计算 node/edge、`phi`、remaining cost、failure/recovery 方向指标；
4. 生成真实 ensemble prediction 与 uncertainty；
5. 将上述 prediction 和指标冻结为阶段 5 唯一输入；
6. 明确禁止后续脚本读取上述四个占位入口生成的指标文件。

若真实重算达到阶段 4 原定主要门槛，则输出：

```text
STAGE5_ENTRY_DECISION = REAL_MODEL_READY
```

然后直接进入阶段 5.2。若只有一个明确指标未达门槛，则输出 `REFINE_STAGE4_MINIMAL`，只重训对应 head 或对应 seed；不得扩展为新架构搜索。

## 阶段 5 的正式目标

阶段 5 不训练 RA-BC。它负责将阶段 4 模型输出组合为可冻结的图奖励，并完成三个核心证据：

1. 合法不同路径的累计奖励一致；
2. failure-recovery 循环净奖励不为正；
3. 有效 recovery 获得正奖励，同时不会通过反复失败刷分。

阶段 5 结束时输出 `G2`：

```text
GO_STAGE6
REFINE_STAGE5
NO_GO_GRAPH_REWARD
```

只有 `GO_STAGE6` 才进入 RA-BC 与下游策略训练。

**核心结论：可以开始阶段 5，但必须先用真实 checkpoint 重算奖励输入；这一轮是推进实验所需的直接修正，不是额外审计。**

<!-- END FILE: 00_阶段4验收与阶段5入口结论.md -->


---

<!-- BEGIN FILE: 01_阶段5通用执行规范.md -->

# PathGraph-SARM 阶段 5：通用执行规范与目录

> 阶段名称：图奖励构造、校准与防刷分验证。  
> 入口：阶段 4 已提供三 seed checkpoint，但阶段 5.1 必须先真实重算 prediction。  
> 里程碑：`M3 = GRAPH_REWARD_READY`。  
> 决策门：`G2`，决定是否进入 Stage 6 RA-BC。

## 给 Agent 的总命令

在现有 CUPID 仓库中继续执行，不重跑阶段 1—3，不重新设计 GraphSpec。先完成真实 checkpoint 推理和实际指标重算；通过后实现图奖励引擎，使用 validation 与 Oracle 结构约束选择奖励参数；锁定后一次性运行 test、diagnostic、基线和核心消融；最后冻结 `reward_v1` 并作出 G2 决策。

推进实验优先。不要安排全仓库审计、无关模型重构、自动图发现、大范围神经网络搜索或提前训练 RA-BC。必要检查只包括：输入 checkpoint/hash、真实 prediction、无 NaN/Inf、selection lock、reward 结构性质与 ZIP 完整性。

## 1. 统一环境变量

先执行：

```bash
set -euo pipefail

# 优先使用阶段 4 bundle 中实际出现的仓库路径。
export REPO_ROOT="${REPO_ROOT:-/home/__compress_data/xushijie/CUPID}"
if [ ! -d "$REPO_ROOT" ] && [ -d /home/xushijie/CUPID ]; then
  export REPO_ROOT=/home/xushijie/CUPID
fi

test -d "$REPO_ROOT"
cd "$REPO_ROOT"

export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_SUPERVISION="$STAGE4_ROOT/supervision_v1"
export STAGE4_CANDIDATES="$STAGE4_ROOT/model_candidates_v1"
export STAGE4_BUNDLE="$STAGE4_CANDIDATES/model_bundle.json"
export STAGE3_DIAG="$STAGE3_ROOT/diagnostic_suite_v1"
export STAGE3_BASELINES="$STAGE3_ROOT/rounds/stage3_3_baseline_runs"
export GRAPH_SPEC_ROOT="$STAGE3_ROOT/input_adapter_v1/runtime_graph_specs_v1.0.1"

export STAGE5_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage5"
export STAGE5_CONFIG="$REPO_ROOT/configs/stage5/stage5.yaml"
export STAGE5_TOOLS="$REPO_ROOT/tools/stage5"
export STAGE5_ROUNDS="$STAGE5_ROOT/rounds"
export STAGE5_PREDICTIONS="$STAGE5_ROOT/real_predictions_v1"
export STAGE5_REWARD="$STAGE5_ROOT/reward_v1"
export STAGE5_DOWNLOADS="$STAGE5_ROOT/downloads"

export STAGE5_SEEDS="${STAGE5_SEEDS:-20260906,20260907,20260908}"
export HISTORY_STEPS="${HISTORY_STEPS:-32}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$REPO_ROOT/configs/stage5" \
  "$STAGE5_TOOLS/lib" \
  "$STAGE5_ROOT/_runtime" \
  "$STAGE5_ROUNDS" \
  "$STAGE5_PREDICTIONS" \
  "$STAGE5_DOWNLOADS"
```

## 2. 必要入口检查

```bash
test -f "$STAGE4_BUNDLE"
test -f "$STAGE4_CANDIDATES/stage4_exit_decision.md"
grep -q 'GO_STAGE5' "$STAGE4_CANDIDATES/stage4_exit_decision.md"
test -f "$STAGE4_SUPERVISION/tables/episode_manifest.csv"
test -f "$STAGE4_SUPERVISION/configs/label_maps.json"
test -f "$STAGE4_SUPERVISION/configs/feature_schema.json"
test -d "$GRAPH_SPEC_ROOT"
test -d "$STAGE3_DIAG"
```

不要把 `model_candidates_v1/metrics/frozen_test_metrics.json` 当作阶段 5 入口证据。阶段 5.1 将重新生成真实指标。

## 3. GPU 必须提权查看；可并行则并行

创建 GPU 查询脚本：

```bash
cat > "$STAGE5_TOOLS/query_gpus.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-gpu_snapshot.txt}"
mkdir -p "$(dirname "$OUT")"
{
  echo "timestamp=$(date -Iseconds)"
  echo "hostname=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-ALL}"

  if sudo -n nvidia-smi \
      --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits > /tmp/stage5_gpu_sudo.txt 2>/tmp/stage5_gpu_sudo.err; then
    echo "query_mode=sudo_noninteractive"
    cat /tmp/stage5_gpu_sudo.txt
  else
    echo "query_mode=direct_nvidia_smi_after_sudo_unavailable"
    echo "sudo_error=$(tr '\n' ' ' </tmp/stage5_gpu_sudo.err || true)"
    nvidia-smi \
      --query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits
  fi
} | tee "$OUT"
SH
chmod +x "$STAGE5_TOOLS/query_gpus.sh"

bash "$STAGE5_TOOLS/query_gpus.sh" "$STAGE5_ROOT/_runtime/gpu_snapshot_initial.txt"
```

若当前终端允许交互式提权，再额外运行一次：

```bash
sudo nvidia-smi
```

但不要让等待密码阻塞自动作业。`sudo -n` 失败时，记录原因并继续使用可用的直接 `nvidia-smi`。

并行原则：

- checkpoint seed 推理彼此独立：一个 seed 一个 GPU 并行；
- split 推理彼此独立：在显存足够时继续并行；
- reward 参数搜索使用已缓存 prediction，优先 CPU 多进程，不占 GPU；
- baseline/ablation 若只做表格运算，CPU 并行；若需要补推理，再按一个 job 一张 GPU；
- 每个 job 使用独立目录、日志和状态文件，禁止共享写入同一结果文件。

## 4. 创建统一 Stage 5 配置

```bash
cat > "$STAGE5_CONFIG" <<'YAML'
stage: 5
name: graph_reward_calibration
statistics_unit: content_group_id
history_steps: 32

entry:
  require_real_prediction_recompute: true
  forbidden_metric_sources:
    - tools/stage4/finalize_stage4.py
    - tools/stage4/evaluate_joint_pathgraph.py
    - tools/stage4/infer_pathgraph_ensemble.py
    - model_candidates_v1/metrics/frozen_test_metrics.json

model_gate:
  required_seed_passes: 2
  node_macro_f1_min: 0.70
  edge_type_macro_f1_non_none_min: 0.55
  phi_mae_max: 0.18
  phi_spearman_min: 0.65
  cost_mae_max: 0.20
  cost_spearman_min: 0.70
  cost_pair_accuracy_min: 0.75
  failure_cost_increase_rate_min: 0.70
  recovery_cost_decrease_rate_min: 0.70
  terminal_success_cost_p90_max: 0.15

reward:
  cost_component: true
  within_node_component: true
  uncertainty_lcb: true
  recovery_debt_cap: true
  repeated_edge_penalty: true
  reward_clip: 1.5
  repeat_window_steps: 64
  node_confidence_min: 0.55
  edge_confidence_min: 0.55

search:
  lambda_values: [0.0, 0.25, 0.5, 1.0]
  eta_values: [0.0, 0.05, 0.10, 0.20]
  beta_values: [0.0, 0.5, 1.0, 1.5]
  confidence_values: [0.55, 0.70]
  workers: 16
  selection_source:
    - transport_recovery_val
    - oracle_graph_trace_bank
  forbidden_selection_source:
    - transport_recovery_test
    - stage3_frozen_diagnostic

reward_gate:
  oracle_path_normalized_gap_max: 0.10
  learned_path_probe_normalized_gap_max: 0.25
  failure_negative_rate_min: 0.70
  recovery_positive_rate_min: 0.65
  recovery_cycle_nonpositive_rate_min: 0.90
  positive_loop_rate_max: 0.05
  loop_return_mean_max: 0.0
  success_return_auroc_min: 0.70
  success_return_spearman_min: 0.35
  fixed_order_score_drop_max: 0.05
  forward_positive_rate_min: 0.55
  recovery_positive_weight_coverage_min: 0.30

packaging:
  max_file_mb: 200
  omit_extensions: [.pt, .pth, .ckpt, .safetensors, .bin]
  omit_raw_episodes: true
  omit_videos: true
  omit_array_caches: true
YAML
```

## 5. 所有真实指标必须可追溯到 prediction

每个指标 JSON 必须包含：

```json
{
  "metric_version": "stage5-real-v1",
  "prediction_files": [],
  "prediction_sha256": {},
  "statistics_unit": "content_group_id",
  "split": "val|test|diagnostic|oracle",
  "generated_by": "实际脚本路径",
  "generated_at": "ISO-8601"
}
```

禁止：

- 在脚本中直接写入预设的 F1、MAE、方向率或 AUROC；
- 用 GT 值复制成 prediction；
- 使用固定示例 JSON 代替 checkpoint 推理；
- 用 test 指标选择 `lambda/eta/beta`；
- 把相同 `content_group_id` 当作多个独立样本。

## 6. 统一轮次目录

每轮创建：

```bash
ROUND_ID="stage5_X_name"
ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
mkdir -p \
  "$ROUND_DIR/configs" \
  "$ROUND_DIR/jobs" \
  "$ROUND_DIR/logs" \
  "$ROUND_DIR/metrics" \
  "$ROUND_DIR/tables" \
  "$ROUND_DIR/plots" \
  "$ROUND_DIR/reports" \
  "$ROUND_DIR/manifests"

cat > "$ROUND_DIR/run_manifest.md" <<EOF
# Run Manifest
- round_id: $ROUND_ID
- started_at: $(date -Iseconds)
- repo_root: $REPO_ROOT
- statistics_unit: content_group_id
- checkpoint_packaging: omitted_by_default
EOF

cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"
```

## 7. 每轮 ZIP：checkpoint 和大文件默认不打包

Agent 创建 `tools/stage5/package_round.py`。脚本必须：

1. 遍历轮次目录；
2. 永久排除 `.pt/.pth/.ckpt/.safetensors/.bin`；
3. 排除原始 episode、视频、cache 和大数组；
4. 其他单文件超过 `ZIP_MAX_FILE_MB` 时排除；
5. 将排除项写入 `large_file_manifest.tsv`；
6. checkpoint 单独写入 `checkpoint_manifest.tsv`；
7. 对实际进入 ZIP 的文件写 `SHA256SUMS.txt`；
8. 生成 ZIP 后执行 `unzip -t`；
9. 输出 ZIP SHA256；
10. 不因为 checkpoint 未打包而判定失败。

每轮执行模板：

```bash
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"

echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"

"$PYTHON_BIN" "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

每轮结束后 Agent 必须在回复中单独给出 ZIP 绝对路径和 SHA256。不得只说“已打包”。

## 8. 阶段 5 明确不做的事项

- 不训练 RA-BC 或 policy；
- 不根据下游 success 反复回调 reward 参数；
- 不修改 Stage 2 GraphSpec 语义，除非发现明确 ID 拼写错误；
- 不启动自动 graph discovery；
- 不做与奖励结构无关的大规模网络架构搜索；
- 不为追求漂亮结果伪造 test、recovery 或 uncertainty 指标。

**核心点：阶段 5 只做真实推理、图奖励校准和冻结评估；可并行的任务立即并行，每轮立即交付轻量 ZIP。**

<!-- END FILE: 01_阶段5通用执行规范.md -->


---

<!-- BEGIN FILE: 阶段5.1_真实推理重算与奖励输入冻结.md -->

# 阶段 5.1：真实推理重算与奖励输入冻结

## 总体上要干什么

使用阶段 4 的真实三 seed checkpoint，对冻结的 validation、test 和 Stage 3 diagnostic 数据重新执行推理，重新计算所有会被阶段 5 使用的实际指标和 uncertainty。该小阶段只修复固定指标与占位推理问题，不重新设计模型，也不重跑无关训练。

本轮出口：

```text
REAL_MODEL_READY
或
REFINE_STAGE4_MINIMAL
```

只有 `REAL_MODEL_READY` 才继续阶段 5.2。

## 给 Agent 的完整任务命令

> 在当前 CUPID 仓库新增 `tools/stage5/`，直接读取 `model_bundle.json` 中的 checkpoint。不要调用 `finalize_stage4.py`、`evaluate_joint_pathgraph.py`、`infer_pathgraph_ensemble.py` 或包内固定的 `frozen_test_metrics.json`。按 seed 分 GPU 并行推理，输出逐帧 prediction；再由 prediction 计算真实 macro F1、MAE、Spearman、failure/recovery 方向率和 ensemble uncertainty。通过原阶段 4 主要门槛后冻结为 `real_predictions_v1`，否则只给出最小修正项。

## 5.1.1 建立轮次目录并查询 GPU

```bash
set -euo pipefail
cd "$REPO_ROOT"

export ROUND_ID=stage5_1_real_inference_and_input_freeze
export ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"/{configs,jobs,logs,metrics,tables,plots,reports,manifests}
cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"

cat > "$ROUND_DIR/run_manifest.md" <<EOF
# Run Manifest
- round_id: $ROUND_ID
- purpose: recompute truthful model predictions and metrics
- started_at: $(date -Iseconds)
- statistics_unit: content_group_id
- source_bundle: $STAGE4_BUNDLE
EOF

bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"
```

## 5.1.2 校验 model bundle 与 checkpoint

实现：

```text
tools/stage5/verify_stage4_bundle.py
```

CLI：

```bash
python tools/stage5/verify_stage4_bundle.py \
  --bundle "$STAGE4_BUNDLE" \
  --feature-schema "$STAGE4_SUPERVISION/configs/feature_schema.json" \
  --label-maps "$STAGE4_SUPERVISION/configs/label_maps.json" \
  --device cuda:0 \
  --output "$ROUND_DIR/metrics/bundle_verification.json"
```

脚本必须逐项完成：

1. 读取 `model_bundle.json`；
2. 要求 checkpoint 数量为 3；
3. 检查每个 path 是否存在；
4. 对每个 checkpoint 重新计算 SHA256，并与 bundle 记录比较；
5. 使用 `tools.stage4.lib.model.load_model` 加载 checkpoint；
6. 从监督数据取至少 32 个不同窗口，执行真实前向；
7. 检查：
   - node/edge probability 每行和接近 1；
   - `phi` 在 `[0,1]`；
   - remaining cost 非负；
   - 输出没有 NaN/Inf；
   - 不同输入的输出方差不是 0；
   - 三个 seed 的输出不应逐元素完全相同；
8. 输出 checkpoint path、大小、SHA256、加载状态、输出范围和方差；
9. 依据实际 checkpoint 重建 `$ROUND_DIR/manifests/checkpoint_manifest_rebuilt.tsv`，不得依赖上传 ZIP 中缺失的候选 manifest。

通过条件：

```bash
python - <<'PY'
import json, os
p=os.environ['ROUND_DIR']+'/metrics/bundle_verification.json'
d=json.load(open(p))
assert d['all_checkpoints_exist']
assert d['all_hashes_match']
assert d['all_loadable']
assert d['all_outputs_finite']
assert d['input_response_nonconstant']
print('bundle verification passed')
PY
```

若 checkpoint 路径因为仓库迁移失效，但可在 Stage 4 job 目录中定位同 SHA 文件，只更新本地 `model_bundle.stage5_runtime.json`，记录旧路径、新路径和 SHA；不得改 checkpoint 内容。

## 5.1.3 构建真实推理脚本

实现：

```text
tools/stage5/run_real_seed_inference.py
tools/stage5/aggregate_ensemble_predictions.py
tools/stage5/compute_real_model_metrics.py
```

### A. `run_real_seed_inference.py`

输入参数：

```text
--checkpoint
--seed
--supervision-dir
--episode-manifest
--split val|test
--task-id transport_recovery|transport_dual_order|all
--history-steps 32
--stride 1
--batch-size 512
--device cuda:0
--output predictions.jsonl.gz
```

逐行输出字段：

```json
{
  "episode_id": "...",
  "content_group_id": "...",
  "task_id": "...",
  "scenario": "...",
  "split": "val",
  "step": 0,
  "seed": 20260906,
  "node_gt": 0,
  "edge_type_gt": 0,
  "edge_id_gt": 0,
  "phi_gt": 0.0,
  "remaining_cost_gt": 1.0,
  "node_probs": [],
  "edge_type_probs": [],
  "edge_id_probs": [],
  "phi_pred": 0.0,
  "remaining_cost_pred": 0.0
}
```

实现要求：

- 因果窗口只包含 `0..t`；不足 32 步左侧补 0；
- 不能读取 outcome、未来帧或 GT 作为模型输入；
- prediction 必须来自 checkpoint forward；
- `node_pred/edge_pred` 由概率 argmax 计算，不直接复制 GT；
- 每 50 个 batch 写进度日志；
- 结束后写 `DONE` 和 prediction SHA256。

### B. `aggregate_ensemble_predictions.py`

按以下 key 对齐三 seed：

```text
task_id, episode_id, content_group_id, split, step
```

输出：

```text
node_probs_mean
edge_type_probs_mean
edge_id_probs_mean
node_predictive_entropy
node_mutual_information
edge_predictive_entropy
edge_mutual_information
phi_mean
phi_std
remaining_cost_mean
remaining_cost_std
per_seed_phi
per_seed_remaining_cost
```

若任何 seed 缺行、重复 key 或 GT 字段不一致，立即报错，不静默丢行。

### C. `compute_real_model_metrics.py`

所有指标必须从 prediction 计算：

- node macro/micro F1 与 accuracy；
- edge-type macro F1，明确排除 `none` 后再计算 non-none 指标；
- edge-id positive macro F1；
- failure/recovery precision、recall、F1；
- `phi` MAE、Spearman、同节点 monotonic violation；
- cost MAE、RMSE、Spearman、pair accuracy；
- failure cost increase rate；
- recovery cost decrease rate；
- recovery no-overshoot rate；
- terminal success cost p90。

failure/recovery 方向率计算规则：

1. 从连续的 GT edge-type segment 提取事件区间；
2. 事件前值取区间前 3 帧 remaining-cost 中位数；
3. 事件后值取区间末 3 帧中位数；
4. failure：`C_after - C_before >= 0.05` 记为方向正确；
5. recovery：`C_before - C_after >= 0.05` 记为方向正确；
6. no-overshoot：recovery 结束时的 cost 不低于其对应 failure 前 cost 减 `0.05`；
7. 不得写死 `0.9` 或其他常数；
8. 指标以 `content_group_id` 聚合后再计算总体均值；
9. 同时保留逐事件表 `failure_recovery_events.csv`。

## 5.1.4 按 seed 多 GPU 并行推理

先从 bundle 导出作业命令：

```bash
python - <<'PY'
import json, os, pathlib, shlex
bundle=json.load(open(os.environ['STAGE4_BUNDLE']))
rd=pathlib.Path(os.environ['ROUND_DIR'])
cmds=[]
for item in bundle['checkpoints']:
    seed=int(item['seed'])
    ckpt=item['path']
    for split in ('val','test'):
        out=rd/'jobs'/f'{split}_s{seed}'
        out.mkdir(parents=True,exist_ok=True)
        pred=out/'predictions.jsonl.gz'
        cmd=(f"{shlex.quote(os.environ.get('PYTHON_BIN','python'))} "
             f"{shlex.quote(os.environ['STAGE5_TOOLS']+'/run_real_seed_inference.py')} "
             f"--checkpoint {shlex.quote(ckpt)} --seed {seed} "
             f"--supervision-dir {shlex.quote(os.environ['STAGE4_SUPERVISION'])} "
             f"--episode-manifest {shlex.quote(os.environ['STAGE4_SUPERVISION']+'/tables/episode_manifest.csv')} "
             f"--split {split} --task-id all --history-steps {os.environ.get('HISTORY_STEPS','32')} "
             f"--stride 1 --batch-size 512 --device cuda:0 --output {shlex.quote(str(pred))} "
             f"> {shlex.quote(str(out/'stdout.log'))} 2> {shlex.quote(str(out/'stderr.log'))}")
        cmds.append({'job_id':f'{split}_s{seed}','command':cmd,'output_dir':str(out)})
(rd/'configs'/'inference_jobs.json').write_text(json.dumps(cmds,indent=2))
print(len(cmds),'jobs written')
PY
```

实现/复用 `tools/stage5/launch_parallel.py`，要求：

- 读取 GPU free memory；
- 每个 job 设置独立 `CUDA_VISIBLE_DEVICES=<physical_gpu_id>`；
- job 内使用 `cuda:0`；
- 一个 GPU 同时最多一个 job；
- GPU 数不足时排队，不改为全串行；
- 输出 `job_status.csv`，字段至少含 job_id、GPU、start/end、exit_code、output_dir。

运行：

```bash
python "$STAGE5_TOOLS/launch_parallel.py" \
  --jobs "$ROUND_DIR/configs/inference_jobs.json" \
  --gpu-snapshot "$ROUND_DIR/logs/gpu_snapshot_before.txt" \
  --min-free-mb "$GPU_MIN_FREE_MB" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU" \
  --status "$ROUND_DIR/metrics/job_status.csv"
```

全部 job 必须完成：

```bash
python - <<'PY'
import csv, os
p=os.environ['ROUND_DIR']+'/metrics/job_status.csv'
rows=list(csv.DictReader(open(p)))
assert rows and all(int(r['exit_code'])==0 for r in rows)
print('all inference jobs passed:',len(rows))
PY
```

## 5.1.5 聚合 ensemble 与计算真实指标

```bash
for SPLIT in val test; do
  python "$STAGE5_TOOLS/aggregate_ensemble_predictions.py" \
    --inputs "$ROUND_DIR/jobs/${SPLIT}_s*/predictions.jsonl.gz" \
    --output "$ROUND_DIR/tables/ensemble_${SPLIT}_predictions.jsonl.gz" \
    --summary "$ROUND_DIR/metrics/ensemble_${SPLIT}_summary.json"
done

for SEED in 20260906 20260907 20260908; do
  python "$STAGE5_TOOLS/compute_real_model_metrics.py" \
    --predictions "$ROUND_DIR/jobs/val_s${SEED}/predictions.jsonl.gz" \
    --label-maps "$STAGE4_SUPERVISION/configs/label_maps.json" \
    --split val \
    --output "$ROUND_DIR/metrics/val_metrics_s${SEED}.json" \
    --event-table "$ROUND_DIR/tables/failure_recovery_events_val_s${SEED}.csv"
done

python "$STAGE5_TOOLS/compute_real_model_metrics.py" \
  --predictions "$ROUND_DIR/tables/ensemble_val_predictions.jsonl.gz" \
  --label-maps "$STAGE4_SUPERVISION/configs/label_maps.json" \
  --split val \
  --ensemble-input \
  --output "$ROUND_DIR/metrics/ensemble_val_metrics.json" \
  --event-table "$ROUND_DIR/tables/failure_recovery_events_ensemble_val.csv"

python "$STAGE5_TOOLS/compute_real_model_metrics.py" \
  --predictions "$ROUND_DIR/tables/ensemble_test_predictions.jsonl.gz" \
  --label-maps "$STAGE4_SUPERVISION/configs/label_maps.json" \
  --split test \
  --ensemble-input \
  --output "$ROUND_DIR/metrics/ensemble_test_metrics.json" \
  --event-table "$ROUND_DIR/tables/failure_recovery_events_ensemble_test.csv"
```

Stage 3 diagnostic 若其输入不是 Stage 4 supervision 格式，实现 `run_real_diagnostic_inference.py`，只做必要适配，输出同一 prediction schema：

```bash
python "$STAGE5_TOOLS/run_real_diagnostic_inference.py" \
  --bundle "$STAGE4_BUNDLE" \
  --diagnostic-suite "$STAGE3_DIAG" \
  --graph-spec-root "$GRAPH_SPEC_ROOT" \
  --devices auto \
  --output "$ROUND_DIR/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --metrics "$ROUND_DIR/metrics/ensemble_stage3_diagnostic_metrics.json"
```

## 5.1.6 阶段 5 入口判定

实现：

```text
tools/stage5/decide_stage5_entry.py
```

命令：

```bash
python "$STAGE5_TOOLS/decide_stage5_entry.py" \
  --config "$STAGE5_CONFIG" \
  --seed-metrics "$ROUND_DIR/metrics/val_metrics_s*.json" \
  --bundle-verification "$ROUND_DIR/metrics/bundle_verification.json" \
  --output-json "$ROUND_DIR/metrics/stage5_entry_gate.json" \
  --output-md "$ROUND_DIR/reports/stage5_entry_decision.md"
```

`REAL_MODEL_READY` 条件：

- 三个 checkpoint 均存在、hash 一致、可加载；
- 至少 2/3 seed 达到配置中的主要 model gate；
- failure/recovery 方向率来自真实事件表，不是固定值；
- ensemble prediction 有效且非恒定；
- prediction 与 GT 不存在“所有概率都为 1 且全部直接复制”的占位模式；
- test 结果只报告，不参与模型重新选择。

若只差一个明确 head：

```text
REFINE_STAGE4_MINIMAL
```

报告必须明确：需要重训哪个 head、哪一个 seed、使用什么原配置和唯一修改。不得扩展为全模型搜索。

检查：

```bash
cat "$ROUND_DIR/reports/stage5_entry_decision.md"
grep -Eq 'REAL_MODEL_READY|REFINE_STAGE4_MINIMAL' \
  "$ROUND_DIR/reports/stage5_entry_decision.md"
```

只有：

```bash
grep -q 'REAL_MODEL_READY' "$ROUND_DIR/reports/stage5_entry_decision.md"
```

成功时，才冻结真实 prediction：

```bash
rm -rf "$STAGE5_PREDICTIONS.tmp"
mkdir -p "$STAGE5_PREDICTIONS.tmp"/{metrics,tables,configs,manifests,reports}
cp "$ROUND_DIR"/metrics/*.json "$STAGE5_PREDICTIONS.tmp/metrics/"
cp "$ROUND_DIR"/tables/*.jsonl.gz "$STAGE5_PREDICTIONS.tmp/tables/"
cp "$ROUND_DIR"/tables/failure_recovery_events*.csv "$STAGE5_PREDICTIONS.tmp/tables/"
cp "$STAGE5_CONFIG" "$STAGE5_PREDICTIONS.tmp/configs/stage5.yaml"
cp "$ROUND_DIR/reports/stage5_entry_decision.md" "$STAGE5_PREDICTIONS.tmp/reports/"
cp "$STAGE4_BUNDLE" "$STAGE5_PREDICTIONS.tmp/configs/model_bundle.json"

(
  cd "$STAGE5_PREDICTIONS.tmp"
  find . -type f ! -name 'REAL_PREDICTIONS_SHA256SUMS.txt' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > REAL_PREDICTIONS_SHA256SUMS.txt
  sha256sum -c REAL_PREDICTIONS_SHA256SUMS.txt
)

rm -rf "$STAGE5_PREDICTIONS"
mv "$STAGE5_PREDICTIONS.tmp" "$STAGE5_PREDICTIONS"
```

## 5.1.7 本轮交付物

必须存在：

```text
metrics/bundle_verification.json
manifests/checkpoint_manifest_rebuilt.tsv
metrics/job_status.csv
metrics/val_metrics_s20260906.json
metrics/val_metrics_s20260907.json
metrics/val_metrics_s20260908.json
metrics/ensemble_val_metrics.json
metrics/ensemble_test_metrics.json
metrics/stage5_entry_gate.json
tables/ensemble_val_predictions.jsonl.gz
tables/ensemble_test_predictions.jsonl.gz
tables/failure_recovery_events_*.csv
reports/stage5_entry_decision.md
```

prediction 文件超过 200 MB 时可以不进入 ZIP，但必须写入 `large_file_manifest.tsv`，并在 ZIP 中保留 SHA256、行数、schema 和绝对路径。

## 5.1.8 生成本轮 ZIP

```bash
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"

echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"
echo "- entry_decision: $(grep -E 'REAL_MODEL_READY|REFINE_STAGE4_MINIMAL' "$ROUND_DIR/reports/stage5_entry_decision.md" | head -n1)" >> "$ROUND_DIR/run_manifest.md"

python "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

**停止点：`REAL_MODEL_READY` 后立即进入阶段 5.2；不要继续在阶段 5.1 增加无关模型实验。**

<!-- END FILE: 阶段5.1_真实推理重算与奖励输入冻结.md -->


---

<!-- BEGIN FILE: 阶段5.2_图奖励引擎与Oracle轨迹库.md -->

# 阶段 5.2：图奖励引擎与 Oracle 轨迹库

## 总体上要干什么

实现可部署的 PathGraph reward engine，把三 seed 模型输出组合为图级进展、节点内进展、重复边惩罚和不确定性下界；同时构建不依赖 learned model 的 Oracle 图轨迹库，先验证奖励的结构性质。

本轮不选择最终参数，只建立正确实现和结构探针。

## 给 Agent 的完整任务命令

> 基于 `real_predictions_v1` 和冻结 GraphSpec 实现 stateful reward engine。reward 必须只使用当前/下一步 prediction 与过去累计状态，不读取未来帧或 episode outcome。实现 per-seed reward、ensemble 均值/标准差、LCB、recovery debt cap 和 repeated-edge penalty。生成合法 A→B、B→A、failure→recovery、重复循环和 stagnation 的 Oracle trace bank，验证数学性质后交付。

## 5.2.1 建立轮次目录

```bash
set -euo pipefail
cd "$REPO_ROOT"
grep -q 'REAL_MODEL_READY' \
  "$STAGE5_PREDICTIONS/reports/stage5_entry_decision.md"

export ROUND_ID=stage5_2_reward_engine_and_oracle_traces
export ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"/{configs,jobs,logs,metrics,tables,plots,reports,manifests,oracle_traces}
cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"
```

## 5.2.2 实现 reward engine

创建：

```text
tools/stage5/lib/reward_engine.py
tools/stage5/lib/reward_types.py
tools/stage5/score_prediction_trace.py
```

### 统一奖励公式

对 ensemble 中第 `m` 个模型：

\[
r^{(m)}_{t,\mathrm{core}}
=
C^{(m)}_t-C^{(m)}_{t+1}
+\lambda I^{\mathrm{same-node}}_t
\left(\phi^{(m)}_{t+1}-\phi^{(m)}_t\right)
-\eta n_{\mathrm{loop}}(e_t).
\]

然后：

\[
\mu_t=\frac{1}{M}\sum_{m=1}^M r^{(m)}_t,
\qquad
\sigma_t=\operatorname{Std}_m(r^{(m)}_t),
\]

\[
r^{\mathrm{LCB}}_t=\mu_t-\beta\sigma_t,
\qquad
w_t=\max(0,r^{\mathrm{LCB}}_t).
\]

### same-node 条件

`I_same-node=1` 仅当：

- `argmax(node_probs_t) == argmax(node_probs_t1)`；
- 两步 node 最大概率都不低于 `node_confidence_min`；
- 当前转移未被高置信识别为 failure 或 recovery；
- 当前/下一步均为有限值。

否则只使用 cost component。

### repeated-edge penalty

维护最近 `repeat_window_steps` 内已执行 edge ID。对当前高置信 edge：

```text
repeat_count = 该 edge 在窗口中此前出现次数
n_loop = max(0, repeat_count)
loop_penalty = eta * n_loop
```

第一次执行不惩罚；第二次开始惩罚。低置信 edge 不新增硬计数，但需要在输出中标记 `loop_count_skipped_low_confidence=true`。

### recovery debt cap

每个模型维护独立 `failure_debt[m]`：

1. 高置信 failure edge 且 core reward 为负：
   ```text
   failure_debt[m] += -min(core_reward[m], 0)
   ```
2. 高置信 recovery edge 且 core reward 为正：
   ```text
   credited = min(core_reward[m], failure_debt[m])
   failure_debt[m] -= credited
   reward_after_cap[m] = credited - loop_penalty
   ```
3. recovery core reward 为负时保留负值，不强行改正；
4. 非 recovery 的合法 forward/alternative 正进展不受 debt cap；
5. episode 开始时 debt 为 0；
6. attempt ID 切换时保留或清空 debt，必须由 GraphSpec/annotation protocol 明确；默认同一 episode 内保留，任务完成时清零。

这样 recovery 可以获正奖励，但 failure+recovery 不会产生净正收益。

### RewardEngine API

```python
engine = PathGraphRewardEngine(config)
state = engine.new_episode(task_id, episode_id)
result = engine.step(prev_prediction, next_prediction, state)
```

`result` 至少包含：

```text
reward_mu
reward_std
reward_lcb
weight_positive
cost_delta_mu
phi_delta_mu
loop_penalty
loop_count
failure_debt_before
failure_debt_after
recovery_cap_applied
node_id_prev
node_id_next
edge_type_pred
edge_id_pred
node_confidence
edge_confidence
```

所有字段均写入 per-transition 表，便于 Stage 6 直接生成 chunk/sample weight。

## 5.2.3 构建 Oracle trace bank

实现：

```text
tools/stage5/build_oracle_trace_bank.py
```

命令：

```bash
python "$STAGE5_TOOLS/build_oracle_trace_bank.py" \
  --graph-spec-root "$GRAPH_SPEC_ROOT" \
  --tasks transport_dual_order transport_recovery \
  --output-dir "$ROUND_DIR/oracle_traces" \
  --manifest "$ROUND_DIR/tables/oracle_trace_manifest.csv"
```

必须生成以下 trace 类型：

```text
legal_A_then_B
legal_B_then_A
legal_shortest_success
failure_then_recovery
failure_recovery_loop_x1
failure_recovery_loop_x2
failure_recovery_loop_x3
stagnation_same_node
illegal_backtrack
terminal_success_hold
```

Oracle trace 中每步显式给出：

```text
task_id
trace_id
step
node_id
edge_id
edge_type
phi
remaining_cost
attempt_id
is_terminal
expected_property
```

remaining cost 来自冻结 GraphSpec 的到成功代价，而不是 learned model。若 GraphSpec 未提供 edge cost，默认合法 edge cost=1、failure edge cost=+1、recovery edge cost=-1 到 debt 上限，并在 manifest 中记录该约定。

## 5.2.4 运行结构性质检查

实现：

```text
tools/stage5/check_reward_properties.py
```

先使用中性参数：

```yaml
lambda: 0.5
eta: 0.1
beta: 0.0
node_confidence_min: 0.55
edge_confidence_min: 0.55
reward_clip: 1.5
```

运行：

```bash
python "$STAGE5_TOOLS/check_reward_properties.py" \
  --trace-dir "$ROUND_DIR/oracle_traces" \
  --reward-config "$ROUND_DIR/configs/stage5.yaml" \
  --lambda-value 0.5 \
  --eta-value 0.1 \
  --beta-value 0.0 \
  --output-json "$ROUND_DIR/metrics/oracle_reward_properties.json" \
  --output-csv "$ROUND_DIR/tables/oracle_trace_returns.csv" \
  --transition-output "$ROUND_DIR/tables/oracle_transition_rewards.csv"
```

必须通过的实现级性质：

1. cost 下降时 graph component 为正；
2. cost 上升时 graph component 为负；
3. 同节点 `phi` 增长时 within-node component 为正；
4. A→B 与 B→A 两条合法成功路径的纯 cost return 相等；
5. failure edge 产生负奖励；
6. recovery edge可以产生正奖励；
7. failure+recovery 的净奖励不大于 0；
8. 重复次数增加时 loop return 单调下降；
9. stagnation 不应得到持续正奖励；
10. terminal hold 的后续 reward 接近 0。

这些是 reward engine 的直接性质检查，不扩展为通用软件测试套件。

## 5.2.5 在真实 validation prediction 上试运行

```bash
python "$STAGE5_TOOLS/score_prediction_trace.py" \
  --predictions "$STAGE5_PREDICTIONS/tables/ensemble_val_predictions.jsonl.gz" \
  --reward-config "$STAGE5_CONFIG" \
  --lambda-value 0.5 \
  --eta-value 0.1 \
  --beta-value 0.5 \
  --output "$ROUND_DIR/tables/val_reward_smoke.jsonl.gz" \
  --summary "$ROUND_DIR/metrics/val_reward_smoke_summary.json"
```

检查：

- 没有 NaN/Inf；
- 每个 episode 第一行只初始化 state，不伪造转移 reward；
- reward component 求和与 `reward_mu` 一致；
- `reward_lcb <= reward_mu + 1e-8`；
- `weight_positive >= 0`；
- failure/recovery debt 不会为负；
- 真实 trajectory 输出不是全 0。

## 5.2.6 本轮完成条件

- reward engine API 与字段契约完成；
- Oracle trace bank 覆盖两条合法顺序、恢复和 1/2/3 次循环；
- 10 项结构性质全部通过；
- 真实 validation smoke 无 NaN/Inf 且非全零；
- 没有读取未来 outcome 作为 reward 输入；
- 本轮不选择最终 `lambda/eta/beta`。

## 5.2.7 生成本轮 ZIP

```bash
cat > "$ROUND_DIR/reports/summary.md" <<EOF
# Stage 5.2 Summary
- reward_engine: implemented
- oracle_trace_bank: built
- property_checks: $(python -c "import json; d=json.load(open('$ROUND_DIR/metrics/oracle_reward_properties.json')); print(d.get('all_passed'))")
- final_parameter_selection: not_performed_in_this_round
EOF

bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"
echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"

python "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

**停止点：结构性质通过后进入阶段 5.3，不在本轮反复调参。**

<!-- END FILE: 阶段5.2_图奖励引擎与Oracle轨迹库.md -->


---

<!-- BEGIN FILE: 阶段5.3_奖励参数校准与SelectionLock.md -->

# 阶段 5.3：奖励参数校准与 Selection Lock

## 总体上要干什么

只使用 `transport_recovery` validation prediction 和 Oracle trace bank，对 `lambda`、`eta`、`beta` 与置信门槛进行一次有限网格校准；选择满足结构硬约束的最简单配置并立即锁定。test、Stage 3 frozen diagnostic 不得参与选择。

## 给 Agent 的完整任务命令

> 用 Stage 5.1 冻结的真实 validation prediction 和 Stage 5.2 Oracle trace bank运行有限参数网格。先过滤违反 loop/recovery/path consistency 硬约束的配置，再按 validation reward-success、正向覆盖和固定顺序表现选择。参数锁定后写入 selection lock 和输入 SHA256，后续不得根据 test 结果修改。

## 5.3.1 建立轮次目录

```bash
set -euo pipefail
cd "$REPO_ROOT"

export ROUND_ID=stage5_3_reward_calibration
export ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"/{configs,jobs,logs,metrics,tables,plots,reports,manifests}
cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"

test -f "$STAGE5_PREDICTIONS/tables/ensemble_val_predictions.jsonl.gz"
test -f "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/tables/oracle_trace_manifest.csv"
```

## 5.3.2 构建 validation calibration suite

实现：

```text
tools/stage5/build_reward_calibration_suite.py
```

命令：

```bash
python "$STAGE5_TOOLS/build_reward_calibration_suite.py" \
  --val-predictions "$STAGE5_PREDICTIONS/tables/ensemble_val_predictions.jsonl.gz" \
  --oracle-trace-dir "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces" \
  --episode-manifest "$STAGE4_SUPERVISION/tables/episode_manifest.csv" \
  --output-dir "$ROUND_DIR/tables/calibration_suite" \
  --manifest "$ROUND_DIR/tables/calibration_suite_manifest.csv"
```

只纳入：

- `transport_recovery/val`；
- Oracle legal alternative path；
- Oracle failure/recovery loops；
- Oracle stagnation 与 terminal hold；
- 固定顺序 control 的 validation 子集（若已冻结）。

禁止纳入：

- `transport_recovery/test`；
- Stage 3 frozen diagnostic；
- policy rollout success；
- test 中的任何统计摘要。

manifest 必须记录每个输入文件 SHA256 和 selection role。

## 5.3.3 生成有限参数网格

实现：

```text
tools/stage5/build_reward_grid.py
tools/stage5/sweep_reward_params.py
```

网格来自 `stage5.yaml`：

```text
lambda ∈ {0.0, 0.25, 0.5, 1.0}
eta ∈ {0.0, 0.05, 0.10, 0.20}
beta ∈ {0.0, 0.5, 1.0, 1.5}
confidence ∈ {0.55, 0.70}
```

共 128 个配置。prediction 已预计算，因此使用 CPU 多进程即可；不要占用 GPU 做纯表格计算。

```bash
python "$STAGE5_TOOLS/build_reward_grid.py" \
  --config "$STAGE5_CONFIG" \
  --output "$ROUND_DIR/configs/reward_grid.jsonl"

python "$STAGE5_TOOLS/sweep_reward_params.py" \
  --grid "$ROUND_DIR/configs/reward_grid.jsonl" \
  --calibration-suite "$ROUND_DIR/tables/calibration_suite" \
  --workers 16 \
  --output "$ROUND_DIR/tables/reward_sweep.csv" \
  --detail-dir "$ROUND_DIR/jobs/grid" \
  --status "$ROUND_DIR/metrics/grid_status.json"
```

每个 grid job 只写自己的：

```text
jobs/grid/<config_id>/metrics.json
jobs/grid/<config_id>/trace_returns.csv
jobs/grid/<config_id>/DONE
```

主进程最后汇总，不允许并行 job 直接追加同一个 CSV。

## 5.3.4 每个配置必须计算的指标

### Oracle 结构指标

```text
oracle_path_normalized_gap
oracle_legal_path_min_return
oracle_loop_return_mean
oracle_positive_loop_rate
oracle_loop_return_by_repeat_count
oracle_stagnation_positive_rate
oracle_terminal_hold_abs_mean
```

路径归一差：

\[
\Delta_{path}=
\frac{|R(P_A)-R(P_B)|}
{\max(|R(P_A)|,|R(P_B)|,10^{-8})}.
\]

### 真实 validation 指标

```text
failure_negative_rate
recovery_positive_rate
recovery_cycle_nonpositive_rate
forward_positive_rate
recovery_positive_weight_coverage
success_return_auroc
success_return_spearman
success_minus_failure_return_margin
reward_nonzero_rate
reward_lcb_mean
uncertainty_penalty_mean
fixed_order_score_drop
```

统计单位为 `content_group_id`。二分类 success 样本不足时，AUROC 可标记 `not_estimable`，但必须报告 return margin 和 Spearman/point-biserial 替代指标，不得填固定值。

## 5.3.5 配置筛选规则

先执行硬约束过滤：

```text
oracle_path_normalized_gap <= 0.10
oracle_positive_loop_rate <= 0.05
oracle_loop_return_mean <= 0.0
failure_negative_rate >= 0.70
recovery_positive_rate >= 0.65
recovery_cycle_nonpositive_rate >= 0.90
forward_positive_rate >= 0.55
recovery_positive_weight_coverage >= 0.30
fixed_order_score_drop <= 0.05
reward_nonzero_rate > 0.10
```

通过硬约束后，按以下优先级选配置：

1. 更低的 `oracle_path_normalized_gap`；
2. 更低的 `positive_loop_rate`；
3. 更高的 `recovery_cycle_nonpositive_rate`；
4. 更高的 validation success-return AUROC/相关性；
5. 参数更简单：较小 `eta`、较小 `beta`、较小 `lambda` 作为 tie-breaker，避免过度惩罚或奖励塌缩。

不要用加权总分掩盖硬约束失败。

实现：

```text
tools/stage5/select_reward_config.py
```

命令：

```bash
python "$STAGE5_TOOLS/select_reward_config.py" \
  --config "$STAGE5_CONFIG" \
  --sweep "$ROUND_DIR/tables/reward_sweep.csv" \
  --output-config "$ROUND_DIR/configs/reward_selected.yaml" \
  --output-json "$ROUND_DIR/metrics/reward_selection.json" \
  --report "$ROUND_DIR/reports/reward_selection.md"
```

## 5.3.6 仅允许一次局部修正

若 128 个配置无一通过：

- 先读 `hard_constraint_failure_counts.csv`；
- 只对失败最多的一个结构约束做一次局部网格；
- 例如 loop 失败时只扩展 `eta={0.25,0.30,0.40}`；
- recovery 全被压成 0 时只扩展 `beta={0.25,0.75}` 或调低 confidence；
- path gap 失败时只检查 `phi` component 是否跨节点错误累积；先修实现错误，不进行盲目调参；
- 局部修正完成后必须停止搜索并做决策。

不允许开启大范围 Bayesian optimization 或使用 test 反复调参。

## 5.3.7 创建 Selection Lock

```bash
python "$STAGE5_TOOLS/create_reward_selection_lock.py" \
  --selected-config "$ROUND_DIR/configs/reward_selected.yaml" \
  --selection-metrics "$ROUND_DIR/metrics/reward_selection.json" \
  --val-predictions "$STAGE5_PREDICTIONS/tables/ensemble_val_predictions.jsonl.gz" \
  --oracle-manifest "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/tables/oracle_trace_manifest.csv" \
  --reward-engine "$STAGE5_TOOLS/lib/reward_engine.py" \
  --output "$ROUND_DIR/metrics/reward_selection_lock.json"
```

lock 必须包含：

```text
locked=true
selection_source=[transport_recovery_val, oracle_graph_trace_bank]
forbidden_source_verified=true
selected lambda/eta/beta/confidence/reward_clip
reward_engine_sha256
selected_config_sha256
val_prediction_sha256
oracle_manifest_sha256
created_at
```

生成后：

```bash
sha256sum "$ROUND_DIR/metrics/reward_selection_lock.json" \
  > "$ROUND_DIR/metrics/reward_selection_lock.sha256"
```

阶段 5.4 只读取该 lock，不再读取 sweep 排名做重新选择。

## 5.3.8 本轮完成条件

- 有至少一个配置通过全部硬约束；
- selection 只使用 val + Oracle；
- test/diagnostic 未参与选择；
- `reward_selected.yaml` 与 lock 已生成；
- 选定 reward 在 validation 上不是全零；
- 选定 reward 的 recovery 有正权重覆盖；
- lock SHA256 已记录。

## 5.3.9 生成本轮 ZIP

```bash
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"
echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"
echo "- selection_lock: $ROUND_DIR/metrics/reward_selection_lock.json" >> "$ROUND_DIR/run_manifest.md"

python "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

**停止点：Selection Lock 生成后立即结束参数搜索，进入阶段 5.4。**

<!-- END FILE: 阶段5.3_奖励参数校准与SelectionLock.md -->


---

<!-- BEGIN FILE: 阶段5.4_冻结奖励评估与基线对照.md -->

# 阶段 5.4：冻结奖励评估与基线对照

## 总体上要干什么

在阶段 5.3 参数锁定后，一次性运行 frozen test、Stage 3 diagnostic、合法双路径 probe、failure/recovery、循环和固定顺序对照。比较原线性/顺序基线与 PathGraph 奖励，回答“图奖励是否真正修复结构性误评分”。

本轮不得根据 test 结果修改 `lambda/eta/beta`。

## 给 Agent 的完整任务命令

> 读取 `reward_selection_lock.json`，对 frozen test 和 Stage 3 diagnostic 评分。运行线性时间、固定 A-first/B-first、sequential transition、learned linear SARM、PathGraph cost-only、PathGraph full、PathGraph full+LCB 等基线。按 content_group 汇总，输出路径一致性、recovery、loop、reward-success 与固定顺序控制指标，并生成图表和简洁结论。

## 5.4.1 建立轮次目录并验证 lock

```bash
set -euo pipefail
cd "$REPO_ROOT"

export ROUND_ID=stage5_4_frozen_reward_evaluation
export ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
export LOCK_DIR="$STAGE5_ROUNDS/stage5_3_reward_calibration"
export REWARD_LOCK="$LOCK_DIR/metrics/reward_selection_lock.json"
export REWARD_SELECTED="$LOCK_DIR/configs/reward_selected.yaml"

rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"/{configs,jobs,logs,metrics,tables,plots,reports,manifests}
cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"
cp "$REWARD_SELECTED" "$ROUND_DIR/configs/reward_selected.yaml"
cp "$REWARD_LOCK" "$ROUND_DIR/configs/reward_selection_lock.json"

sha256sum -c "$LOCK_DIR/metrics/reward_selection_lock.sha256"
python - <<'PY'
import json, os
p=os.environ['REWARD_LOCK']
d=json.load(open(p))
assert d['locked'] is True
assert d['forbidden_source_verified'] is True
assert 'test' not in ' '.join(d['selection_source']).lower()
print('reward selection lock verified')
PY

bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"
```

## 5.4.2 冻结评估数据

使用：

```text
$STAGE5_PREDICTIONS/tables/ensemble_test_predictions.jsonl.gz
$STAGE5_PREDICTIONS/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz
$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces/
$STAGE3_BASELINES/
```

若 Stage 3 baseline 表格字段不同，只写一次 adapter：

```text
tools/stage5/adapt_stage3_baselines.py
```

统一字段：

```text
method
task_id
trace_id
episode_id
content_group_id
step
reward
cumulative_reward
outcome
scenario
```

不得重训 Stage 3 baseline，也不得改其已冻结输出。

## 5.4.3 基线矩阵

必须评估：

```text
linear_time_fraction
oracle_linear_chain_A_first
oracle_linear_chain_B_first
sequential_transition_oracle
learned_linear_sarm
pathgraph_cost_only
pathgraph_cost_plus_phi
pathgraph_full_no_lcb
pathgraph_full_lcb
```

定义：

- `pathgraph_cost_only`：`lambda=0, eta=0, beta=0`，保留 cost delta；
- `pathgraph_cost_plus_phi`：使用锁定 `lambda`，`eta=0, beta=0`；
- `pathgraph_full_no_lcb`：使用锁定 `lambda/eta`，`beta=0`，启用 debt cap；
- `pathgraph_full_lcb`：完整锁定配置。

## 5.4.4 并行运行评估

prediction 已缓存，主要是 CPU 运算。按 method × suite 独立并行：

```bash
python "$STAGE5_TOOLS/build_frozen_eval_jobs.py" \
  --selection-lock "$REWARD_LOCK" \
  --test-predictions "$STAGE5_PREDICTIONS/tables/ensemble_test_predictions.jsonl.gz" \
  --diagnostic-predictions "$STAGE5_PREDICTIONS/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --oracle-trace-dir "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces" \
  --stage3-baselines "$STAGE3_BASELINES" \
  --output "$ROUND_DIR/configs/frozen_eval_jobs.json"

python "$STAGE5_TOOLS/launch_cpu_jobs.py" \
  --jobs "$ROUND_DIR/configs/frozen_eval_jobs.json" \
  --workers 16 \
  --status "$ROUND_DIR/metrics/job_status.csv"
```

若 diagnostic prediction 尚未缓存且需要 checkpoint 推理，单独按 seed 多 GPU 并行补推理；完成后再运行 CPU 评估。不要为纯 reward 求和占用 GPU。

## 5.4.5 正式指标

### A. 合法路径一致性

```text
legal_path_return_A_first
legal_path_return_B_first
legal_path_normalized_gap
legal_path_sign_consistency
path_order_bias
```

`transport_dual_order` 只有少量唯一 content group，只作为 mechanism probe；明确报告样本单位，不给出夸大的泛化结论。

### B. failure/recovery 校准

```text
failure_negative_rate
recovery_positive_rate
recovery_positive_weight_coverage
failure_recovery_cycle_nonpositive_rate
failure_recovery_cycle_return_mean
recovery_return_vs_stagnation_margin
recovery_return_vs_failure_margin
```

### C. 循环与 reward hacking

```text
positive_loop_rate
loop_return_mean
loop_return_p95
loop_return_by_repeat_count
stagnation_positive_rate
terminal_hold_abs_reward_mean
reward_hacking_case_count
```

### D. reward 与任务结果

```text
success_return_auroc
success_return_spearman
success_minus_failure_return_margin
partial_success_ordering_accuracy
reward_success_correlation_by_task
```

### E. 固定顺序控制

```text
fixed_order_reward_monotonicity
fixed_order_success_failure_margin
fixed_order_score_drop_vs_best_sequential
```

图奖励在本质单链任务上不应明显变差。

### F. uncertainty 作用

```text
reward_mu_vs_lcb_gap
high_uncertainty_weight_suppression
error_top20_weight_mean
error_bottom20_weight_mean
recovery_weight_retention_after_lcb
```

uncertainty 不作为主贡献，但必须证明 LCB 没有把所有 recovery 权重压成 0。

## 5.4.6 统计与置信区间

- 独立单位：`content_group_id`；
- 使用 group bootstrap 1000 次计算主要指标 95% CI；
- `transport_dual_order` 只有两个唯一组时只报告原始两个路径与 gap，不输出伪精确 CI；
- baseline 差异使用配对 content group；
- 不把 frame 当作独立样本计算显著性。

命令：

```bash
python "$STAGE5_TOOLS/summarize_frozen_reward_eval.py" \
  --job-root "$ROUND_DIR/jobs" \
  --statistics-unit content_group_id \
  --bootstrap 1000 \
  --output-metrics "$ROUND_DIR/metrics/frozen_reward_metrics.json" \
  --comparison-table "$ROUND_DIR/tables/baseline_comparison.csv" \
  --case-table "$ROUND_DIR/tables/reward_hacking_cases.csv" \
  --report "$ROUND_DIR/reports/frozen_reward_evaluation.md"
```

## 5.4.7 必须生成的图

```text
plots/legal_path_returns.png
plots/failure_recovery_reward_traces.png
plots/loop_return_by_repeat_count.png
plots/success_failure_return_distribution.png
plots/baseline_core_metrics.png
plots/reward_mu_vs_lcb.png
```

每张图单独保存，不做难以阅读的多子图拼盘。图中注明 task、split、统计单位和样本数。

## 5.4.8 本轮完成条件

- selection lock 在运行前已验证；
- 所有基线使用同一 frozen suite；
- PathGraph full 的路径一致性、loop 和 recovery 指标已报告；
- test 结果未反向改变 reward 参数；
- 固定顺序 control 未被忽略；
- reward-hacking case 有逐条表格可定位；
- 核心图表完成；
- 大型 prediction 可不打包但有 manifest。

## 5.4.9 生成本轮 ZIP

```bash
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"
echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"
echo "- reward_lock_sha256: $(sha256sum "$REWARD_LOCK" | awk '{print $1}')" >> "$ROUND_DIR/run_manifest.md"

python "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

**停止点：冻结评估完成后不改参数，直接进入阶段 5.5 做预定义消融。**

<!-- END FILE: 阶段5.4_冻结奖励评估与基线对照.md -->


---

<!-- BEGIN FILE: 阶段5.5_核心消融与组件归因.md -->

# 阶段 5.5：核心消融与组件归因

## 总体上要干什么

仅运行能够解释主方法的必要消融，确认 `phi`、loop penalty、recovery debt cap 和 uncertainty LCB 分别解决什么问题。此阶段不是新一轮调参，也不重新选择最终 reward 配置。

## 给 Agent 的完整任务命令

> 固定 Stage 5.3 的选定参数，逐个关闭核心组件，在同一 frozen test/diagnostic suite 上运行。输出每个组件对合法路径、recovery、loop 和正权重覆盖的影响。不要增加无关模型、全参数搜索或下游 policy 训练。

## 5.5.1 建立轮次目录

```bash
set -euo pipefail
cd "$REPO_ROOT"

export ROUND_ID=stage5_5_core_ablations
export ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
export REWARD_LOCK="$STAGE5_ROUNDS/stage5_3_reward_calibration/metrics/reward_selection_lock.json"

rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"/{configs,jobs,logs,metrics,tables,plots,reports,manifests}
cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"
cp "$REWARD_LOCK" "$ROUND_DIR/configs/reward_selection_lock.json"
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"
```

## 5.5.2 固定消融矩阵

只运行以下六项：

| ablation_id | 改动 | 要回答的问题 |
|---|---|---|
| `full_lcb` | 完整锁定配置 | 主方法 |
| `no_phi` | `lambda=0` | 节点内奖励是否提高稠密性而不破坏路径一致性 |
| `no_loop_penalty` | `eta=0` | 重复失败/恢复是否会产生正循环 |
| `no_recovery_cap` | 关闭 debt cap | recovery 是否可能超过先前损失并刷分 |
| `no_uncertainty` | `beta=0` | LCB 是否抑制不可靠高分且保留有效 recovery |
| `cost_only` | `lambda=0, eta=0, beta=0, cap=false` | 仅 cost delta 能解决多少问题 |

不得改变 checkpoint、GraphSpec 或其余参数。

## 5.5.3 并行运行

```bash
python "$STAGE5_TOOLS/build_ablation_jobs.py" \
  --selection-lock "$REWARD_LOCK" \
  --test-predictions "$STAGE5_PREDICTIONS/tables/ensemble_test_predictions.jsonl.gz" \
  --diagnostic-predictions "$STAGE5_PREDICTIONS/tables/ensemble_stage3_diagnostic_predictions.jsonl.gz" \
  --oracle-trace-dir "$STAGE5_ROUNDS/stage5_2_reward_engine_and_oracle_traces/oracle_traces" \
  --ablations full_lcb no_phi no_loop_penalty no_recovery_cap no_uncertainty cost_only \
  --output "$ROUND_DIR/configs/ablation_jobs.json"

python "$STAGE5_TOOLS/launch_cpu_jobs.py" \
  --jobs "$ROUND_DIR/configs/ablation_jobs.json" \
  --workers 12 \
  --status "$ROUND_DIR/metrics/job_status.csv"
```

## 5.5.4 汇总指标

每个消融至少输出：

```text
legal_path_normalized_gap
forward_positive_rate
reward_nonzero_rate
failure_negative_rate
recovery_positive_rate
recovery_positive_weight_coverage
recovery_cycle_nonpositive_rate
positive_loop_rate
loop_return_mean
success_return_auroc
fixed_order_score_drop
```

另外生成 component-level 归因表：

```text
episode_id
content_group_id
transition_index
cost_component
phi_component
loop_penalty
recovery_cap_delta
uncertainty_penalty
final_reward
```

汇总：

```bash
python "$STAGE5_TOOLS/summarize_reward_ablations.py" \
  --job-root "$ROUND_DIR/jobs" \
  --full-id full_lcb \
  --output "$ROUND_DIR/tables/ablation_summary.csv" \
  --component-table "$ROUND_DIR/tables/component_attribution.csv.gz" \
  --metrics "$ROUND_DIR/metrics/ablation_metrics.json" \
  --report "$ROUND_DIR/reports/ablation_report.md"
```

## 5.5.5 预期解释，不作为硬编码结果

Agent 必须根据实际结果写结论，不能预填。但应检查以下机制是否成立：

- `no_phi`：reward 更稀疏，within-node progress 降低；
- `no_loop_penalty`：重复边的净回报上升；
- `no_recovery_cap`：failure+recovery 正循环率上升；
- `no_uncertainty`：高误差、高分片段的权重抑制减弱；
- `cost_only`：可能保持路径 telescoping，但节点内 reward 密度不足。

若某组件没有贡献，报告真实结果；不要通过新增参数搜索强行制造差异。

## 5.5.6 必须生成的图

```text
plots/ablation_core_metrics.png
plots/loop_penalty_effect.png
plots/recovery_cap_effect.png
plots/uncertainty_weight_effect.png
plots/component_contribution_distribution.png
```

## 5.5.7 本轮完成条件

- 六个预定义消融全部完成；
- 所有消融使用同一 reward lock、prediction 和统计单位；
- 没有使用消融结果重新选择参数；
- 至少能定位 full 与各消融的逐 transition 差异；
- 报告明确哪些组件有效、哪些效果有限。

## 5.5.8 生成本轮 ZIP

```bash
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"
echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"

python "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

**停止点：完成预定义消融后直接进入阶段 5.6，不增加新的消融矩阵。**

<!-- END FILE: 阶段5.5_核心消融与组件归因.md -->


---

<!-- BEGIN FILE: 阶段5.6_G2决策与Stage6交接.md -->

# 阶段 5.6：G2 决策与 Stage 6 交接

## 总体上要干什么

汇总真实推理、参数锁定、冻结评估与核心消融，冻结 `reward_v1`，作出是否进入 RA-BC 的 G2 决策，并生成 Stage 6 可直接使用的 reward/weight 数据接口。

## 给 Agent 的完整任务命令

> 读取 Stage 5.1—5.5 的冻结结果，不重跑模型、不重选参数。根据预先定义的路径一致性、循环净回报、recovery calibration、reward-success 和固定顺序控制门槛生成 G2 决策。若通过，冻结 reward engine、config、model bundle 引用、输入 checksum、评估表和 Stage 6 handoff；生成每个 transition/chunk 的 reward LCB 与正权重导出规范。

## 5.6.1 建立轮次目录

```bash
set -euo pipefail
cd "$REPO_ROOT"

export ROUND_ID=stage5_6_g2_freeze
export ROUND_DIR="$STAGE5_ROUNDS/$ROUND_ID"
export CAL_DIR="$STAGE5_ROUNDS/stage5_3_reward_calibration"
export EVAL_DIR="$STAGE5_ROUNDS/stage5_4_frozen_reward_evaluation"
export ABL_DIR="$STAGE5_ROUNDS/stage5_5_core_ablations"

rm -rf "$ROUND_DIR"
mkdir -p "$ROUND_DIR"/{configs,logs,metrics,tables,plots,reports,manifests,reward_v1}
cp "$STAGE5_CONFIG" "$ROUND_DIR/configs/stage5.yaml"
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_before.txt"
```

## 5.6.2 汇总所有固定输入

检查文件存在：

```bash
test -f "$STAGE5_PREDICTIONS/reports/stage5_entry_decision.md"
grep -q 'REAL_MODEL_READY' "$STAGE5_PREDICTIONS/reports/stage5_entry_decision.md"

test -f "$CAL_DIR/metrics/reward_selection_lock.json"
test -f "$CAL_DIR/configs/reward_selected.yaml"
test -f "$EVAL_DIR/metrics/frozen_reward_metrics.json"
test -f "$EVAL_DIR/tables/baseline_comparison.csv"
test -f "$ABL_DIR/metrics/ablation_metrics.json"
```

生成阶段汇总：

```bash
python "$STAGE5_TOOLS/build_stage5_evidence_table.py" \
  --entry-gate "$STAGE5_PREDICTIONS/metrics/stage5_entry_gate.json" \
  --selection "$CAL_DIR/metrics/reward_selection.json" \
  --selection-lock "$CAL_DIR/metrics/reward_selection_lock.json" \
  --frozen-eval "$EVAL_DIR/metrics/frozen_reward_metrics.json" \
  --ablations "$ABL_DIR/metrics/ablation_metrics.json" \
  --output-json "$ROUND_DIR/metrics/stage5_evidence.json" \
  --output-csv "$ROUND_DIR/tables/stage5_gate_metrics.csv"
```

## 5.6.3 G2 门槛

`GO_STAGE6` 需要满足：

### 真实模型入口

```text
stage5_entry_decision == REAL_MODEL_READY
checkpoint_hash_verified == true
real_prediction_verified == true
```

### 合法路径

```text
oracle_path_normalized_gap <= 0.10
learned_path_probe_normalized_gap <= 0.25
两条合法成功路径累计 reward 均为正
```

### failure/recovery

```text
failure_negative_rate >= 0.70
recovery_positive_rate >= 0.65
recovery_positive_weight_coverage >= 0.30
recovery_cycle_nonpositive_rate >= 0.90
```

### loop/reward hacking

```text
positive_loop_rate <= 0.05
loop_return_mean <= 0.0
重复次数增加时 loop return 不上升
```

### reward 与结果

满足下面至少一个，同时 success-failure margin 必须为正：

```text
success_return_auroc >= 0.70
或
success_return_spearman >= 0.35
```

### 固定顺序控制

```text
fixed_order_score_drop <= 0.05
```

### 工程出口

```text
reward 无 NaN/Inf
selection lock 未被 test 修改
reward engine/config/input hash 可定位
Stage 6 weight schema 已生成
```

## 5.6.4 决策规则

实现：

```text
tools/stage5/decide_g2.py
```

命令：

```bash
python "$STAGE5_TOOLS/decide_g2.py" \
  --config "$STAGE5_CONFIG" \
  --evidence "$ROUND_DIR/metrics/stage5_evidence.json" \
  --output-json "$ROUND_DIR/metrics/g2_gate.json" \
  --output-md "$ROUND_DIR/reports/stage5_exit_decision.md"
```

输出三选一：

### `GO_STAGE6`

所有核心门槛通过，可以冻结 reward 并接入 RA-BC。

### `REFINE_STAGE5`

只有一个明确可修正的问题，例如：

- `eta` 较低导致少量正循环；
- `beta` 过高导致 recovery 权重覆盖不足；
- reward engine 某一 edge 计数实现错误。

只允许针对该问题做一轮局部修正，然后重新锁定并重跑受影响的 5.3—5.4；不重做全部阶段。

### `NO_GO_GRAPH_REWARD`

真实模型输出正常，但无法同时满足路径一致性、recovery 和 loop 约束，或图奖励没有优于 sequential baseline。此时不进入 RA-BC。

## 5.6.5 冻结 reward_v1

仅在 `GO_STAGE6` 时执行：

```bash
grep -q 'GO_STAGE6' "$ROUND_DIR/reports/stage5_exit_decision.md"

rm -rf "$STAGE5_REWARD.tmp"
mkdir -p "$STAGE5_REWARD.tmp"/{configs,code,metrics,tables,reports,manifests,examples}

cp "$CAL_DIR/configs/reward_selected.yaml" \
  "$STAGE5_REWARD.tmp/configs/reward_config_v1.yaml"
cp "$CAL_DIR/metrics/reward_selection_lock.json" \
  "$STAGE5_REWARD.tmp/configs/reward_selection_lock.json"
cp "$STAGE4_BUNDLE" \
  "$STAGE5_REWARD.tmp/configs/model_bundle.json"
cp "$STAGE5_TOOLS/lib/reward_engine.py" \
  "$STAGE5_REWARD.tmp/code/reward_engine.py"
cp "$STAGE5_TOOLS/lib/reward_types.py" \
  "$STAGE5_REWARD.tmp/code/reward_types.py"
cp "$ROUND_DIR/metrics/g2_gate.json" \
  "$STAGE5_REWARD.tmp/metrics/"
cp "$EVAL_DIR/metrics/frozen_reward_metrics.json" \
  "$STAGE5_REWARD.tmp/metrics/"
cp "$EVAL_DIR/tables/baseline_comparison.csv" \
  "$STAGE5_REWARD.tmp/tables/"
cp "$ABL_DIR/tables/ablation_summary.csv" \
  "$STAGE5_REWARD.tmp/tables/"
cp "$ROUND_DIR/reports/stage5_exit_decision.md" \
  "$STAGE5_REWARD.tmp/reports/"
```

生成 normalization 和 schema：

```bash
python "$STAGE5_TOOLS/build_stage6_weight_schema.py" \
  --reward-config "$STAGE5_REWARD.tmp/configs/reward_config_v1.yaml" \
  --val-rewards "$CAL_DIR/jobs" \
  --output-schema "$STAGE5_REWARD.tmp/configs/stage6_weight_schema.json" \
  --normalization "$STAGE5_REWARD.tmp/configs/reward_normalization.json" \
  --example "$STAGE5_REWARD.tmp/examples/weight_record_example.json"
```

Stage 6 每条 transition/chunk 至少需要：

```text
episode_id
content_group_id
task_id
t_start
t_end
reward_mu
reward_std
reward_lcb
weight_positive
cost_component
phi_component
loop_penalty
recovery_cap_delta
edge_type_pred
edge_id_pred
node_confidence
edge_confidence
```

冻结 checksum：

```bash
cat > "$STAGE5_REWARD.tmp/reports/stage6_handoff.md" <<EOF
# Stage 6 Handoff

- entry: G2=GO_STAGE6
- reward_config: configs/reward_config_v1.yaml
- reward_engine: code/reward_engine.py
- model_bundle: configs/model_bundle.json
- weight_schema: configs/stage6_weight_schema.json
- statistics_unit: content_group_id
- reward_selection_locked_before_test: true
- checkpoint_files: referenced_only_not_packaged
EOF

cat > "$STAGE5_REWARD.tmp/FROZEN.md" <<EOF
# reward_v1 frozen

Frozen at: $(date -Iseconds)
Selection source: transport_recovery validation + Oracle graph traces.
Test and Stage 3 diagnostic were used only after selection lock.
EOF

(
  cd "$STAGE5_REWARD.tmp"
  find . -type f ! -name 'STAGE5_REWARD_SHA256SUMS.txt' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > STAGE5_REWARD_SHA256SUMS.txt
  sha256sum -c STAGE5_REWARD_SHA256SUMS.txt
)

rm -rf "$STAGE5_REWARD"
mv "$STAGE5_REWARD.tmp" "$STAGE5_REWARD"
```

checkpoint 不复制进 `reward_v1`，只保留 bundle 路径、SHA256 和 manifest。

## 5.6.6 生成阶段 5 完整交付

复制轻量结果：

```bash
cp -a "$STAGE5_REWARD" "$ROUND_DIR/reward_v1/"
cp "$STAGE5_REWARD/reports/stage6_handoff.md" "$ROUND_DIR/reports/"
```

本轮 ZIP：

```bash
bash "$STAGE5_TOOLS/query_gpus.sh" "$ROUND_DIR/logs/gpu_snapshot_after.txt"
echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"

python "$STAGE5_TOOLS/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/${ROUND_ID}.zip"
sha256sum "$STAGE5_DOWNLOADS/${ROUND_ID}.zip" \
  | tee "$STAGE5_DOWNLOADS/${ROUND_ID}.zip.sha256"
```

仅在 `GO_STAGE6` 时生成总 ZIP：

```bash
export COMPLETE_DIR="$STAGE5_ROOT/stage5_complete_bundle"
rm -rf "$COMPLETE_DIR"
mkdir -p "$COMPLETE_DIR"/{reward_v1,round_summaries,tools_snapshot,configs,manifests}

rsync -a "$STAGE5_REWARD/" "$COMPLETE_DIR/reward_v1/"
cp "$STAGE5_CONFIG" "$COMPLETE_DIR/configs/stage5.yaml"
rsync -a "$STAGE5_TOOLS/" "$COMPLETE_DIR/tools_snapshot/"

for RID in \
  stage5_1_real_inference_and_input_freeze \
  stage5_2_reward_engine_and_oracle_traces \
  stage5_3_reward_calibration \
  stage5_4_frozen_reward_evaluation \
  stage5_5_core_ablations \
  stage5_6_g2_freeze; do
  mkdir -p "$COMPLETE_DIR/round_summaries/$RID"
  cp "$STAGE5_ROUNDS/$RID/run_manifest.md" \
    "$COMPLETE_DIR/round_summaries/$RID/" 2>/dev/null || true
  cp "$STAGE5_ROUNDS/$RID/reports/summary.md" \
    "$COMPLETE_DIR/round_summaries/$RID/" 2>/dev/null || true
done

cat > "$COMPLETE_DIR/summary.md" <<EOF
# PathGraph-SARM Stage 5 complete

- exit_state: GO_STAGE6
- milestone: M3=GRAPH_REWARD_READY
- selection_lock: validation + Oracle only
- checkpoint_packaging: omitted_by_default
- statistics_unit: content_group_id
EOF

(
  cd "$COMPLETE_DIR"
  find . -type f ! -name 'SHA256SUMS.txt' -print0 \
    | LC_ALL=C sort -z | xargs -0 sha256sum > SHA256SUMS.txt
  sha256sum -c SHA256SUMS.txt
)

python "$STAGE5_TOOLS/package_round.py" \
  --round-id stage5_complete \
  --round-dir "$COMPLETE_DIR" \
  --downloads-dir "$STAGE5_DOWNLOADS" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE5_DOWNLOADS/stage5_complete.zip"
sha256sum "$STAGE5_DOWNLOADS/stage5_complete.zip" \
  | tee "$STAGE5_DOWNLOADS/stage5_complete.zip.sha256"
```

不要删除前六轮 ZIP；每轮 ZIP 和最终 ZIP 都应保留供下载。若存储空间紧张，可在用户确认已下载后再清理。

## 5.6.7 最终 Agent 回复格式

Agent 最终必须回复：

```text
Stage 5 已完成/未完成
G2: GO_STAGE6 | REFINE_STAGE5 | NO_GO_GRAPH_REWARD
核心指标：
- path gap
- failure negative rate
- recovery positive rate
- recovery cycle nonpositive rate
- positive loop rate
- success return AUROC/Spearman
- fixed-order drop

每轮 ZIP 路径与 SHA256：
- stage5_1_...
- stage5_2_...
- stage5_3_...
- stage5_4_...
- stage5_5_...
- stage5_6_...

总 ZIP：
- stage5_complete.zip（仅 GO_STAGE6）
- SHA256
```

**核心点：阶段 5 的出口不是“脚本跑完”，而是图奖励在真实 prediction 上同时满足路径一致性、recovery 和防循环刷分，并以冻结接口交给 Stage 6。**

<!-- END FILE: 阶段5.6_G2决策与Stage6交接.md -->
