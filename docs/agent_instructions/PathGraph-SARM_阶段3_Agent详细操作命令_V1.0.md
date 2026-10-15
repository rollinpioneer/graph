# PathGraph-SARM 阶段 2 验收与阶段 3 入口结论

## 结论

阶段 2 可以关闭，并进入阶段 3。正式状态为：

```text
M1 = GO_STAGE3
Stage 3 entry = allowed
```

本次对 `stage2_complete.zip` 的实际复核结果：

- ZIP SHA256：`59b1a8acb4e517cac10e041b3ae6336a5951ce27c82d56671937585a2149f63a`；
- `unzip -t` 完整性检查通过；
- `M1_SHA256SUMS.txt` 对冻结文件校验通过；
- 2 个 graph-valid 任务：`transport_recovery`、`transport_dual_order`；
- `transport_dual_order` 含 A→B 与 B→A 两条合法路径；
- 显式 recovery episode 共 28 个；
- GT 关键 edge 最少 8 个实例；
- GT split 的 episode/group 泄漏计数为 0；
- Stage 1 占位事件没有进入 GT；
- 新增数据明确标记为 `controller_source=scripted_oracle`。

因此，阶段 2 已完成“人工图、标注协议与 GT 子集冻结”的职责，阶段 3 可以开始做线性 SARM / sequential progress 的结构性误评分验证。

## 阶段 3 开始前必须在 3.1 处理的两个运行时问题

这些问题不要求回退重做阶段 2，但必须在阶段 3 的第一个小阶段创建版本化的 runtime adapter；禁止直接改写 `m1_freeze_v1`。

### 1. Graph spec 与 GT 的边定义存在可修正的不一致

实际检查发现：

- `transport_recovery_graph_v1.yaml` 中存在重复 edge id，且部分重复项的 edge type 冲突；
- GT 中存在 graph spec 未列出的 edge，例如 `recovery_to_B_done`、`B_done_to_terminal_failure`、`in_transit_to_grasped`；
- `transport_recovery` 的首段 node interval 被标为 `in_transit`，而任务语义与 path template 表示该段应是 `start`；
- `transport_dual_order` 的部分 recovery annotation 中，`restored_node` 与恢复完成时的 node interval 不一致。

阶段 3.1 应创建 `runtime_graph_specs_v1.0.1` 和 `m1_runtime_patch_v1.jsonl`，保留原 M1 冻结包只读，并记录每一项修正的来源与理由。

### 2. `transport_dual_order` 同一 scenario 内轨迹内容完全重复

同一 scenario 的多个 episode 虽有不同 episode id，但数值状态序列完全相同。因此：

- 不能把这些 episode 当作独立统计样本；
- 不能用普通 episode split 的结果声称模型具备跨轨迹泛化；
- 阶段 3 必须计算 `content_group_id`，统计时按唯一内容组或 scenario template 聚合；
- learned baseline 在该任务上的结果仅用于验证线性表示的结构行为，主 G1 证据应以冻结 GT 上的成对结构诊断为主。

## 阶段 3 的证据边界

阶段 3 可以回答：

- 单一线性 stage chain 是否必然偏置某一种合法顺序；
- recovery 是否会被线性 progress 误判为倒退、零进展或延迟进展；
- 时间单调型 progress 是否会错误奖励失败/循环耗时；
- 这些误评分是否在 oracle linearization 与 learned linear progress model 中稳定出现。

阶段 3不能回答：

- PathGraph-SARM 是否已经优于基线；
- RA-BC 策略成功率是否提升；
- scripted-oracle 数据能否代表 learned policy 或真实机器人泛化。

以上问题分别留到阶段 4、5、6。

## 进入阶段 3 的执行口径

```text
先冻结 Stage 3 runtime adapter 和 diagnostic suite，
再运行线性基线，
最后只根据冻结诊断集做 G1 决策。
```

**核心结论：阶段 2 可以结束；阶段 3 可以进入，但必须先完成必要的 runtime 图定义修正和内容级分组，避免把可修正的数据接口问题或重复模板误当成方法结论。**


---

# PathGraph-SARM 阶段 3：Agent 详细操作命令（V1.0）

# PathGraph-SARM 阶段 3 通用执行规范与目录

> 阶段名称：基线复现与问题成立性验证。  
> 阶段入口：阶段 2 已完成，`M1=GO_STAGE3`。  
> 阶段里程碑：`M2`。  
> 决策门：`G1`。  
> 目标：证明固定 stage chain / 单一全局 progress 在合法多顺序和失败恢复轨迹上存在可重复、结构性的误评分。  
> 本阶段不训练 PathGraph node/edge/cost 模型，不训练 RA-BC，不做自动图发现，不做大规模策略 rollout。

## 给 Agent 的总命令

在 `/home/xushijie/CUPID` 中执行本阶段。直接使用阶段 2 的冻结 M1 产物，不重新运行阶段 1 或阶段 2。先完成 3.1 的只读输入适配与 runtime errata，再冻结 3.2 diagnostic suite；diagnostic suite 冻结后才允许运行任何正式基线。可独立的 task、baseline、seed 和推理 job 在不发生共享写冲突时必须多 GPU 并行。GPU 必须通过提权命令查看。每个小阶段结束后立即生成一个轻量 ZIP；checkpoint、模型权重、原始 episode、视频、缓存和其他大文件默认不打包，只写 manifest。

不要安排与 G1 无关的广泛代码审计、超参数大搜索、策略训练或泛化测试。一个可选基线缺失时，记录原因并继续完成必做基线，不要让可选项阻塞阶段推进。

## 阶段 3 小阶段

| 小阶段 | 总体上要完成的工作 | 本轮 ZIP |
|---|---|---|
| 3.1 | 保持 M1 只读，建立 runtime graph/annotation patch、内容哈希分组和统一 episode index | `stage3_1_input_adapter.zip` |
| 3.2 | 构建并冻结 fixed-chain、alternative-order、recovery、failure/cycle 诊断集 | `stage3_2_diagnostic_suite.zip` |
| 3.3 | 实现 oracle/learned 线性基线并按 task、orientation、seed 多 GPU 并行运行 | `stage3_3_baseline_runs.zip` |
| 3.4 | 量化合法边负奖励、路径偏置、恢复误判、循环净回报和结果相关性 | `stage3_4_misscoring_analysis.zip` |
| 3.5 | 执行 G1 判定，冻结 M2 问题陈述并生成 Stage 4 handoff | `stage3_5_g1_decision.zip` |

阶段全部完成后额外生成：

```text
stage3_complete.zip
```

## 阶段 3 总体完成条件

1. `m1_freeze_v1` 的 checksum 校验通过且保持只读；所有运行时修正都写入独立版本，不原地修改 M1。
2. runtime graph 中 edge id 唯一，冻结 GT 使用到的每个 edge 都能映射到唯一 `(src, dst, semantic_type)`。
3. 所有 episode 均具有 `content_group_id`；完全相同轨迹不被当作独立统计样本。
4. diagnostic suite 在正式 baseline 结果产生前冻结并带 SHA256。
5. 必做基线至少包含：`linear_time_fraction`、两个 orientation 的 `oracle_linear_chain`、`sequential_transition_oracle`、`learned_linear_sarm` 三个 seed。
6. 在 canonical chain control 上，基线的正向阶段顺序行为正常；误评分集中出现在 alternative/recovery/cycle 诊断上。
7. 输出逐步 prediction、聚合指标、95% bootstrap CI、代表案例图、`g1_decision.md` 和 `stage4_handoff.md`。
8. 每个小阶段及阶段总包均完成 ZIP、SHA256 和 `unzip -t` 检查。

## 统一目录与环境变量

Agent 先执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE1_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage1"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export M1_ROOT="$STAGE2_ROOT/m1_freeze_v1"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export STAGE3_CONFIG="$REPO_ROOT/configs/stage3/stage3.yaml"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export STAGE3_SEEDS="${STAGE3_SEEDS:-20260903,20260904,20260905}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

cd "$REPO_ROOT"
mkdir -p \
  "$REPO_ROOT/configs/stage3" \
  "$REPO_ROOT/tools/stage3/lib" \
  "$STAGE3_ROOT/_runtime" \
  "$STAGE3_ROOT/rounds" \
  "$STAGE3_ROOT/downloads" \
  "$STAGE3_ROOT/input_adapter_v1" \
  "$STAGE3_ROOT/diagnostic_suite_v1" \
  "$STAGE3_ROOT/m2_freeze_v1"

# 只检查本阶段真正需要的入口文件。
test -f "$M1_ROOT/M1_SHA256SUMS.txt"
test -f "$M1_ROOT/stage3_handoff.md"
test -f "$M1_ROOT/selected_graph_tasks_v1.yaml"
test -f "$M1_ROOT/gt_v1/episode_annotations.jsonl"
test -f "$M1_ROOT/gt_v1/gt_splits.csv"
test -f "$M1_ROOT/gt_v1/node_intervals.csv"
test -f "$M1_ROOT/gt_v1/edge_intervals.csv"
test -d "$M1_ROOT/graph_specs_v1"

sed -n '1,160p' "$M1_ROOT/stage3_handoff.md"
```

若上述文件不存在，但 `stage2_complete.zip` 已经下载到项目目录，则只解压，不重跑阶段 2：

```bash
STAGE2_ZIP="${STAGE2_ZIP:-$REPO_ROOT/stage2_complete.zip}"
test -f "$STAGE2_ZIP"
unzip -q "$STAGE2_ZIP" -d "$REPO_ROOT"
test -f "$M1_ROOT/M1_SHA256SUMS.txt"
```

## 创建阶段 3 总配置

```bash
cat > "$STAGE3_CONFIG" <<'YAML'
project:
  repo_root: /home/xushijie/CUPID
  stage1_root: artifacts/pathgraph_sarm/stage1
  stage2_root: artifacts/pathgraph_sarm/stage2
  stage3_root: artifacts/pathgraph_sarm/stage3

inputs:
  m1_root: artifacts/pathgraph_sarm/stage2/m1_freeze_v1
  stage2_collection_root: artifacts/pathgraph_sarm/stage2/rounds/stage2_2_targeted_collection/jobs/synthetic
  stage1_episode_manifest: artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl
  stage1_splits: artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/splits.csv

frozen_tasks:
  graph_tasks: [transport_recovery, transport_dual_order]
  fixed_chain_controls: [square, transport]
  same_domain_chain_controls:
    - {task_id: transport_recovery, scenario: natural_success}
    - {task_id: transport_dual_order, scenario: order_A_then_B}

runtime_adapter:
  output_version: m1_runtime_adapter_v1
  preserve_m1_read_only: true
  derive_observed_edges_from_gt: true
  canonicalize_duplicate_edge_ids: true
  apply_semantic_errata: true
  content_hash_numeric_trace_only: true
  statistics_unit: content_group_id

features:
  numeric_fields: [eef_pos, object_pos, target_pos, gripper_state, action]
  current_state_flags: [subgoal_A_done, subgoal_B_done]
  forbidden_fields: [success, scenario, controller_source, episode_id, outcome]
  history_steps: 32

linearizations:
  dual_order_A_first: [start, A_done, B_done, success]
  dual_order_B_first: [start, B_done, A_done, success]
  transport_recovery_chain: [start, grasped, in_transit, placed, success]
  off_chain_projection: last_valid_rank_with_local_reset

training:
  seeds: [20260903, 20260904, 20260905]
  device: cuda
  batch_size: 128
  epochs: 80
  learning_rate: 0.0003
  weight_decay: 0.0001
  hidden_dim: 128
  gru_layers: 1
  dropout: 0.1
  grad_clip_norm: 1.0
  fixed_budget_no_test_selection: true

required_baselines:
  - linear_time_fraction
  - oracle_linear_chain
  - sequential_transition_oracle
  - learned_linear_sarm
optional_baselines:
  - existing_repo_sarm
  - existing_repo_stage_transition
  - existing_repo_arm

diagnostics:
  freeze_before_scoring: true
  bootstrap_resamples: 2000
  ci: 0.95
  pair_by: [task_id, outcome, structural_case]
  duplicate_weight: inverse_content_group_size

thresholds:
  control_monotonicity_min: 0.90
  alternative_legal_negative_rate_min: 0.20
  normalized_path_score_gap_min: 0.10
  recovery_positive_rate_max_for_failure_evidence: 0.70
  recovery_rank_error_min: 0.20
  cycle_positive_rate_min: 0.10
  required_independent_failure_signatures: 2

runtime:
  cpu_workers: 16
  gpu_min_free_mb: 6000
  max_jobs_per_gpu: 1
  zip_max_file_mb: 200
YAML

sed -n '1,280p' "$STAGE3_CONFIG"
```

## GPU 必须提权查看

创建统一脚本：

```bash
cat > "$REPO_ROOT/tools/stage3/query_gpus.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:-${STAGE3_ROOT:-artifacts/pathgraph_sarm/stage3}/_runtime}"
MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
mkdir -p "$OUT_DIR"

if sudo -n true 2>/dev/null; then
  SUDO=(sudo -n)
else
  # 机器要求交互提权时允许在此提示密码；不能退回普通权限并据此判断无 GPU。
  SUDO=(sudo)
fi

"${SUDO[@]}" nvidia-smi | tee "$OUT_DIR/nvidia_smi_full.txt"
"${SUDO[@]}" nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits \
  | tee "$OUT_DIR/gpu_status.csv"

awk -F',' -v min_mem="$MIN_FREE_MB" '
  {
    for (i=1;i<=NF;i++) gsub(/^ +| +$/, "", $i);
    if ($6+0 >= min_mem && $7+0 <= 30) print $1;
  }
' "$OUT_DIR/gpu_status.csv" > "$OUT_DIR/available_gpu_ids.txt"

printf 'Privileged-visible available GPU IDs: '
tr '\n' ' ' < "$OUT_DIR/available_gpu_ids.txt"
echo
BASH
chmod +x "$REPO_ROOT/tools/stage3/query_gpus.sh"

"$REPO_ROOT/tools/stage3/query_gpus.sh" "$STAGE3_ROOT/_runtime"
```

解释：

- 普通用户权限下 `nvidia-smi` 失败，不等于机器没有 GPU；必须先执行上面的提权检查。
- 独立 baseline、task、orientation、seed job 默认一张 GPU 一个 job 并行。
- 小模型通常不需要 DDP；优先实验级并行，不为形式上的多 GPU 改写模型。
- 分析、构建诊断集等 CPU 工作不强行占 GPU。

## 创建轮次初始化脚本

```bash
cat > "$REPO_ROOT/tools/stage3/init_round.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 ROUND_ID ROUND_DIR PURPOSE" >&2
  exit 2
fi
ROUND_ID="$1"
ROUND_DIR="$2"
PURPOSE="$3"
mkdir -p "$ROUND_DIR"/{configs,logs,metrics,plots,jobs,manifests}
cat > "$ROUND_DIR/run_manifest.md" <<EOF
# Run Manifest

- round_id: $ROUND_ID
- purpose: $PURPOSE
- started_at: $(date -Iseconds)
- repo_root: ${REPO_ROOT:-$(pwd)}
- git_commit: $(git -C "${REPO_ROOT:-$(pwd)}" rev-parse HEAD 2>/dev/null || echo unknown)
- python: $(${PYTHON_BIN:-python} --version 2>&1)
- stage2_m1_root: ${M1_ROOT:-unknown}
EOF
BASH
chmod +x "$REPO_ROOT/tools/stage3/init_round.sh"
```

## 创建每轮轻量 ZIP 打包脚本

checkpoint、模型权重、原始 episode、视频、缓存和大文件默认不打包，但每轮 ZIP 必须存在。

```bash
cat > "$REPO_ROOT/tools/stage3/package_round.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 ROUND_ID ROUND_DIR" >&2
  exit 2
fi
ROUND_ID="$1"
ROUND_DIR="$(realpath "$2")"
STAGE3_ROOT="${STAGE3_ROOT:-$(realpath artifacts/pathgraph_sarm/stage3)}"
DOWNLOAD_DIR="$STAGE3_ROOT/downloads"
ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
ZIP_PATH="$DOWNLOAD_DIR/${ROUND_ID}.zip"
mkdir -p "$DOWNLOAD_DIR"

test -s "$ROUND_DIR/run_manifest.md"
test -s "$ROUND_DIR/summary.md"

grep -q '^- finished_at:' "$ROUND_DIR/run_manifest.md" || \
  echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"

printf 'path\tsize_bytes\tartifact_type\treason_omitted\n' > "$ROUND_DIR/large_file_manifest.tsv"
printf 'path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n' > "$ROUND_DIR/checkpoint_manifest.tsv"

find "$ROUND_DIR" -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' \) \
  -printf '%p\t%s\tcheckpoint_or_model_weight\tdefault_omit_from_zip\n' \
  >> "$ROUND_DIR/large_file_manifest.tsv" || true

find "$ROUND_DIR" -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' \) \
  -printf '%p\t%s\tunknown\tunknown\tunknown\n' \
  >> "$ROUND_DIR/checkpoint_manifest.tsv" || true

find "$ROUND_DIR" -type f \
  \( -name '*.pkl' -o -name '*.hdf5' -o -name '*.mp4' -o -name '*.avi' -o -name '*.mov' \) \
  -printf '%p\t%s\traw_or_heavy_media\tdefault_omit_from_zip\n' \
  >> "$ROUND_DIR/large_file_manifest.tsv" || true

find "$ROUND_DIR" -type f -size +"${ZIP_MAX_FILE_MB}"M \
  ! -name large_file_manifest.tsv ! -name checkpoint_manifest.tsv \
  -printf '%p\t%s\tlarge_file\tover_size_threshold\n' \
  >> "$ROUND_DIR/large_file_manifest.tsv" || true

rm -f "$ZIP_PATH" "$ZIP_PATH.sha256"
cd "$ROUND_DIR"
find . -type f \
  ! -path '*/__pycache__/*' \
  ! -path '*/.cache/*' \
  ! -path '*/wandb/*' \
  ! -path '*/dataset_cache/*' \
  ! -path '*/checkpoints/*' \
  ! -name '*.ckpt' ! -name '*.pt' ! -name '*.pth' ! -name '*.bin' ! -name '*.safetensors' \
  ! -name '*.pkl' ! -name '*.hdf5' ! -name '*.mp4' ! -name '*.avi' ! -name '*.mov' \
  ! -size +"${ZIP_MAX_FILE_MB}"M \
  -print | LC_ALL=C sort | zip -q "$ZIP_PATH" -@

sha256sum "$ZIP_PATH" > "$ZIP_PATH.sha256"
unzip -t "$ZIP_PATH" > "$DOWNLOAD_DIR/${ROUND_ID}_unzip_test.txt"
{
  echo "- $(date -Iseconds) ${ROUND_ID}.zip"
  echo "  - sha256: $(cut -d' ' -f1 "$ZIP_PATH.sha256")"
  echo "  - source: $ROUND_DIR"
} >> "$DOWNLOAD_DIR/index.md"
ls -lh "$ZIP_PATH" "$ZIP_PATH.sha256"
BASH
chmod +x "$REPO_ROOT/tools/stage3/package_round.sh"
```

## 多 GPU 并行运行规则

正式训练前执行：

```bash
"$REPO_ROOT/tools/stage3/query_gpus.sh" "$STAGE3_ROOT/_runtime"
mapfile -t GPU_IDS < "$STAGE3_ROOT/_runtime/available_gpu_ids.txt"

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
  echo "当前提权可见 GPU 中没有满足空闲阈值的设备。CPU 诊断可继续；GPU baseline 等待资源或按较小 batch 在确认可用的 GPU 上运行。" >&2
else
  printf 'Will use GPUs: %s\n' "${GPU_IDS[*]}"
fi
```

独立 job 的调度要求：

1. job matrix 写入 `jobs.tsv`，字段固定为 `job_id task_id method orientation seed command output_dir required`。
2. 每个 job 使用独立 `output_dir`、日志和 checkpoint 目录。
3. 同时运行的 job 数不超过可用 GPU 数乘以 `MAX_JOBS_PER_GPU`。
4. 默认按 `task × orientation × seed` 并行；不要把三个 seed 串行跑。
5. 一个 job 失败时只重跑失败项，其他成功结果立即保留并打包。

基础并行模板：

```bash
JOBS_TSV="$ROUND_DIR/jobs.tsv"
mapfile -t GPU_IDS < "$STAGE3_ROOT/_runtime/available_gpu_ids.txt"
test "${#GPU_IDS[@]}" -ge 1

mapfile -t JOB_LINES < <(tail -n +2 "$JOBS_TSV")
pids=()
active=0
for i in "${!JOB_LINES[@]}"; do
  IFS=$'\t' read -r job_id task_id method orientation seed command output_dir required <<< "${JOB_LINES[$i]}"
  gpu="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
  mkdir -p "$output_dir" "$ROUND_DIR/logs"

  while [ "$(jobs -pr | wc -l)" -ge "${#GPU_IDS[@]}" ]; do
    wait -n || true
  done

  CUDA_VISIBLE_DEVICES="$gpu" \
  nohup bash -lc "set -euo pipefail; $command" \
    > "$ROUND_DIR/logs/${job_id}.log" 2>&1 &
  echo "$!" > "$output_dir/pid.txt"
  echo "$gpu" > "$output_dir/gpu_id.txt"
done
wait
```

## 每轮必须记录的最小产物

每个 `ROUND_DIR` 至少包含：

```text
run_manifest.md
commands.sh
configs/
logs/
metrics/
plots/
summary.md
large_file_manifest.tsv
checkpoint_manifest.tsv
```

GPU 训练轮额外包含：

```text
jobs.tsv
job_status.tsv
每个 job 的 resolved_config.yaml
每个 job 的 metrics.json
每个 job 的 predictions.jsonl 或 prediction manifest
```

## 阶段 3 统一结果目录

```text
artifacts/pathgraph_sarm/stage3/
├── input_adapter_v1/
├── diagnostic_suite_v1/
├── rounds/
│   ├── stage3_1_input_adapter/
│   ├── stage3_2_diagnostic_suite/
│   ├── stage3_3_baseline_runs/
│   ├── stage3_4_misscoring_analysis/
│   └── stage3_5_g1_decision/
├── m2_freeze_v1/
└── downloads/
```

## 本阶段推进原则

- 先做结构性、成对、可解释的诊断，再做 learned baseline；不要先跑大量模型再挑案例。
- diagnostic suite 一旦冻结，不根据结果更换 episode。
- `scripted_oracle` 只作为机制证据；所有摘要和图表中保留该 provenance。
- 完全重复轨迹按 `content_group_id` 聚合，不能把重复 episode 数当统计样本量。
- 可选基线不存在时不阻塞 G1；必做的 oracle 与 learned linear baseline 必须完成。
- Stage 3 只证明“问题成立”，不提前声称 PathGraph 已经解决问题。


---

# 阶段 3.1：M1 运行时修正、内容分组与输入冻结

## 总体上要干什么

保持阶段 2 的 `m1_freeze_v1` 完全只读，建立一个可供阶段 3 使用的 runtime adapter。该 adapter 只解决运行接口中的确定性问题：重复/缺失 edge、明显的 annotation 语义错位、source path 解析、完全重复轨迹的内容分组。完成后冻结唯一的 `stage3_episode_index.csv`、`runtime_graph_specs_v1.0.1` 和 `m1_runtime_patch_v1.jsonl`。

本小阶段不训练模型，不修改阶段 2 原文件，不增加新的研究假设。

## 本轮目录

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE1_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage1"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export M1_ROOT="$STAGE2_ROOT/m1_freeze_v1"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export ROUND_ID="stage3_1_input_adapter"
export ROUND_DIR="$STAGE3_ROOT/rounds/$ROUND_ID"
export ADAPTER_DIR="$STAGE3_ROOT/input_adapter_v1"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage3/init_round.sh" \
  "$ROUND_ID" "$ROUND_DIR" \
  "Create read-only M1 runtime adapter, semantic errata, and content-hash groups"
mkdir -p "$ADAPTER_DIR"/{runtime_graph_specs_v1.0.1,manifests,reports}
cp "$REPO_ROOT/configs/stage3/stage3.yaml" "$ROUND_DIR/configs/"
```

## 步骤 1：校验并锁定 M1 输入

```bash
cd "$M1_ROOT"
sha256sum -c M1_SHA256SUMS.txt \
  | tee "$ROUND_DIR/logs/m1_checksum_check.log"
cd "$REPO_ROOT"

# 只读锁定；若目录本身由其他流程管理，不改变 owner，只移除写权限。
chmod -R a-w "$M1_ROOT"

find "$M1_ROOT" -type f -printf '%P\t%s\n' \
  | LC_ALL=C sort \
  > "$ADAPTER_DIR/manifests/m1_input_files.tsv"
sha256sum "$M1_ROOT/M1_SHA256SUMS.txt" \
  > "$ADAPTER_DIR/manifests/m1_checksum_file.sha256"
```

如果 checksum 失败，不重新生成阶段 2；先定位是解压不完整还是 M1 被改写。恢复原 ZIP 中的冻结文件后再次检查。

## 步骤 2：创建统一入口脚本 `prepare_stage3_inputs.py`

Agent 创建：

```text
tools/stage3/prepare_stage3_inputs.py
```

必须提供以下 CLI：

```bash
python tools/stage3/prepare_stage3_inputs.py \
  --m1-root PATH \
  --repo-root PATH \
  --stage2-collection-root PATH \
  --stage1-manifest PATH \
  --output-dir PATH \
  --apply-known-errata \
  --derive-edges-from-gt \
  --compute-content-groups
```

脚本按以下顺序执行。

### 2.1 读取并连接冻结数据

读取：

```text
m1_freeze_v1/selected_graph_tasks_v1.yaml
m1_freeze_v1/graph_specs_v1/*.yaml
m1_freeze_v1/gt_v1/episode_annotations.jsonl
m1_freeze_v1/gt_v1/gt_splits.csv
m1_freeze_v1/gt_v1/node_intervals.csv
m1_freeze_v1/gt_v1/edge_intervals.csv
m1_freeze_v1/gt_v1/failure_recovery_events.csv
stage1/1.3_dataset_v0.1/episode_manifest.jsonl
```

对 annotation 的 `source_path`：

1. 先按绝对路径检查；
2. 若为 `artifacts/...` 相对路径，则拼接 `$REPO_ROOT`；
3. 若原路径不存在，再尝试 `$STAGE2_ROOT` 下相对解析；
4. 解析结果写入 `resolved_source_path`，不覆盖原字段；
5. 任何无法解析的 graph-task episode 写入 `unresolved_sources.csv`，并阻止本小阶段完成。

### 2.2 从冻结 GT 推导 observed edge

对每条 `edge_interval`：

1. 在同 episode 的 node intervals 中找到 `start_step` 前最后一个稳定 node，作为 `src`；
2. 找到 `end_step` 后第一个稳定 node，作为 `dst`；
3. 汇总同一 `edge_id` 的 `(src,dst,edge_type)`；
4. 同一 observed `edge_id` 必须能得到唯一 src/dst；若冻结 GT 内部存在冲突，输出 `observed_edge_conflicts.csv` 并停止；
5. edge type 先保存 `frozen_edge_type`，再通过下面的 errata 生成 `semantic_edge_type`。

### 2.3 应用已确认的 runtime errata

原 M1 不动；将修正写入：

```text
input_adapter_v1/m1_runtime_patch_v1.jsonl
input_adapter_v1/reports/runtime_errata.md
```

至少包含以下规则：

1. `transport_recovery`：每个 episode 的第一个 node interval 若是 `in_transit` 且覆盖 step 0，则 runtime node 改为 `start`；对应初始 edge 规范化为 `start_to_grasped`。
2. `transport_recovery`：`in_transit_to_dropped_or_misaligned` 的 semantic type 固定为 `failure`。
3. `transport_recovery`：`dropped_or_misaligned_to_recovery` 与 `recovery_to_grasped` 的 semantic type 固定为 `recovery`。
4. `transport_dual_order`：`B_done_to_terminal_failure` 的 semantic type 固定为 `failure`。
5. `transport_dual_order`：`recovery_to_B_done` 的 semantic type固定为 `recovery`。
6. `transport_dual_order`：若 recovery event 的 `restored_node` 与 recovery completion step 所在 node 不一致，以冻结 node interval 为 runtime restored node，并记录原值与修正值；当前预期为 `A_done → B_done`。
7. 所有修正必须记录：`episode_id/task_id/object_type/object_id/field/old_value/new_value/reason/evidence_source`。

不要把不确定的语义猜测写成 patch；本次只修正能由 path template、node interval 与 failure/recovery event 三者交叉确认的项。

### 2.4 生成唯一 runtime graph spec

输出：

```text
input_adapter_v1/runtime_graph_specs_v1.0.1/transport_recovery_graph_runtime_v1.0.1.yaml
input_adapter_v1/runtime_graph_specs_v1.0.1/transport_dual_order_graph_runtime_v1.0.1.yaml
```

生成规则：

1. 节点集合沿用 M1 graph spec；不添加新的研究节点。
2. observed edge 以 runtime-patched GT 为主，确保每个 `edge_id` 唯一。
3. graph spec 中未在 GT 观察到、但方法后续需要的 `stagnation_loop` 可保留，并增加 `observed_in_gt: false`。
4. 对 M1 graph spec 中的重复 edge id 去重；原始重复记录全部写入 `graph_spec_duplicate_edges.csv`。
5. 每个 runtime edge 同时保存：
   - `id`
   - `src`
   - `dst`
   - `frozen_edge_type`
   - `semantic_edge_type`
   - `observed_count`
   - `observed_in_gt`
   - `source=m1_gt_or_m1_spec`
6. 所有 frozen GT edge 都必须被 runtime graph 覆盖。

输出一个校验表：

```text
runtime_graph_validation.csv
```

字段：

```text
task_id,raw_edge_count,unique_runtime_edge_count,duplicate_edge_count,
gt_edge_count,gt_edges_missing_after_patch,src_dst_conflict_count,
semantic_type_conflict_count,validation_pass
```

### 2.5 计算内容级分组

对每个 graph-task JSON episode：

1. 读取 `states`；
2. 只取数值轨迹字段 `eef_pos/object_pos/target_pos/gripper_state/action`；
3. 不把 `episode_id/scenario/controller_source/outcome/info.success` 放入 hash；
4. 将数值转成固定 dtype 与连续数组后计算 SHA256；
5. `content_group_id = sha256[:16]`；
6. 统计相同 content group 是否跨 split；
7. 不修改冻结 split，只在阶段 3 统计中加入 `analysis_weight=1/group_size`；
8. 每个 content group 选字典序最小 episode 作为 `representative_episode_id`。

输出：

```text
content_hash_groups.csv
content_group_split_overlap.csv
stage3_episode_index.csv
```

`stage3_episode_index.csv` 字段固定为：

```text
episode_id,task_id,scenario,outcome,path_signature,controller_source,
source_path,resolved_source_path,split_original,content_sha256,
content_group_id,content_group_size,representative_episode_id,
analysis_weight,is_representative,has_failure,has_recovery
```

对 `square/transport`：使用 stage1 manifest 中的原始 source file hash 作为 `content_group_id`；本阶段不伪造其 stage GT。

## 步骤 3：运行输入适配

```bash
$PYTHON_BIN tools/stage3/prepare_stage3_inputs.py \
  --m1-root "$M1_ROOT" \
  --repo-root "$REPO_ROOT" \
  --stage2-collection-root \
    "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/jobs/synthetic" \
  --stage1-manifest \
    "$STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" \
  --output-dir "$ADAPTER_DIR" \
  --apply-known-errata \
  --derive-edges-from-gt \
  --compute-content-groups \
  2>&1 | tee "$ROUND_DIR/logs/prepare_stage3_inputs.log"
```

复制轻量结果到本轮目录：

```bash
cp -a "$ADAPTER_DIR"/runtime_graph_specs_v1.0.1 "$ROUND_DIR/"
cp "$ADAPTER_DIR"/m1_runtime_patch_v1.jsonl "$ROUND_DIR/"
cp "$ADAPTER_DIR"/stage3_episode_index.csv "$ROUND_DIR/"
cp "$ADAPTER_DIR"/content_hash_groups.csv "$ROUND_DIR/"
cp -a "$ADAPTER_DIR"/reports "$ROUND_DIR/"
```

## 步骤 4：生成输入冻结 manifest

```bash
find "$ADAPTER_DIR" -type f -printf '%P\t%s\n' \
  | LC_ALL=C sort \
  > "$ADAPTER_DIR/manifests/input_adapter_files.tsv"

find "$ADAPTER_DIR" -type f ! -name 'INPUT_ADAPTER_SHA256SUMS.txt' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > "$ADAPTER_DIR/INPUT_ADAPTER_SHA256SUMS.txt"

$PYTHON_BIN - <<'PY'
from pathlib import Path
import csv, json, os
root=Path(os.environ['ADAPTER_DIR'])
idx=list(csv.DictReader(open(root/'stage3_episode_index.csv')))
groups={r['content_group_id'] for r in idx if r['task_id'] in {'transport_recovery','transport_dual_order'}}
summary={
  'episode_count': len(idx),
  'graph_episode_count': sum(r['task_id'].startswith('transport_') for r in idx),
  'unique_content_group_count': len(groups),
  'duplicate_episode_count': sum(int(r['content_group_size'])>1 for r in idx),
}
(root/'input_adapter_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
```

## 本小阶段验收条件

仅执行以下与后续训练直接相关的检查：

```bash
test -s "$ADAPTER_DIR/stage3_episode_index.csv"
test -s "$ADAPTER_DIR/m1_runtime_patch_v1.jsonl"
test -s "$ADAPTER_DIR/runtime_graph_specs_v1.0.1/transport_recovery_graph_runtime_v1.0.1.yaml"
test -s "$ADAPTER_DIR/runtime_graph_specs_v1.0.1/transport_dual_order_graph_runtime_v1.0.1.yaml"

grep -q ',True$' "$ADAPTER_DIR/runtime_graph_validation.csv"
! grep -q ',False$' "$ADAPTER_DIR/runtime_graph_validation.csv"

test ! -s "$ADAPTER_DIR/unresolved_sources.csv" || \
  [ "$(wc -l < "$ADAPTER_DIR/unresolved_sources.csv")" -le 1 ]
```

Agent 在 `summary.md` 明确写：

- M1 是否保持只读；
- runtime graph edge 是否唯一并覆盖所有 GT edge；
- 应用了多少条确定性 patch；
- graph-task episode 数、唯一 content group 数和重复组数量；
- `transport_dual_order` 的统计单位已改为 content group/template；
- 下一步只能读取 `input_adapter_v1`，不得直接用原始 graph spec 做正式诊断。

## 本轮 ZIP

```bash
cat > "$ROUND_DIR/summary.md" <<EOF
# Stage 3.1 summary

- M1 checksum: PASS
- M1 modified in place: NO
- Runtime graph specs: $ADAPTER_DIR/runtime_graph_specs_v1.0.1
- Runtime patch: $ADAPTER_DIR/m1_runtime_patch_v1.jsonl
- Episode index: $ADAPTER_DIR/stage3_episode_index.csv
- Statistics unit: content_group_id
EOF

"$REPO_ROOT/tools/stage3/package_round.sh" "$ROUND_ID" "$ROUND_DIR"
```

交付路径必须写明：

```text
$STAGE3_ROOT/downloads/stage3_1_input_adapter.zip
```

**本小阶段核心点：不回退阶段 2，也不改写 M1；用一个明确、可追踪的 runtime adapter 消除接口错误，并把完全重复轨迹从“独立样本”降为“同一内容组”。**


---

# 阶段 3.2：Reward Diagnostic Suite 构建与冻结

## 总体上要干什么

在任何正式 baseline 结果产生之前，基于阶段 3.1 的 runtime-patched annotations 构建并冻结诊断集。诊断集必须同时覆盖：正常固定链、合法 alternative order、failure→recovery、terminal failure 和可形成闭环的恢复片段。后续不得根据 baseline 结果替换 episode 或阈值。

## 本轮目录

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export ADAPTER_DIR="$STAGE3_ROOT/input_adapter_v1"
export SUITE_DIR="$STAGE3_ROOT/diagnostic_suite_v1"
export ROUND_ID="stage3_2_diagnostic_suite"
export ROUND_DIR="$STAGE3_ROOT/rounds/$ROUND_ID"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage3/init_round.sh" \
  "$ROUND_ID" "$ROUND_DIR" \
  "Build and freeze structural reward diagnostic suite before baseline scoring"
mkdir -p "$SUITE_DIR"/{tables,segments,configs,reports}
cp "$REPO_ROOT/configs/stage3/stage3.yaml" "$ROUND_DIR/configs/"
test -s "$ADAPTER_DIR/stage3_episode_index.csv"
test -s "$ADAPTER_DIR/m1_runtime_patch_v1.jsonl"
```

## 步骤 1：创建 linearization 配置

```bash
cat > "$SUITE_DIR/configs/linearization_specs.yaml" <<'YAML'
linearizations:
  dual_order_A_first:
    task_id: transport_dual_order
    canonical_nodes: [start, A_done, B_done, success]
    off_chain_projection: last_valid_rank_with_local_reset
  dual_order_B_first:
    task_id: transport_dual_order
    canonical_nodes: [start, B_done, A_done, success]
    off_chain_projection: last_valid_rank_with_local_reset
  transport_recovery_chain:
    task_id: transport_recovery
    canonical_nodes: [start, grasped, in_transit, placed, success]
    off_chain_projection: last_valid_rank_with_local_reset

expected_semantic_sign:
  forward: positive
  alternative: positive
  recovery: positive
  failure: negative
  stagnation: non_positive

progress_formula: "(node_rank + within_node_phi) / number_of_transitions"
time_fraction_formula: "step / max(1, episode_length - 1)"
YAML
```

`last_valid_rank_with_local_reset` 的定义固定为：

- 进入 canonical chain 之外的 failure/recovery node 时，保留最近一个合法 canonical node 的 rank；
- 节点内局部进度重置为 0；
- 返回 canonical node 后按该节点的 rank 与局部进度继续；
- 不使用 future outcome，不使用最终成功信息。

## 步骤 2：创建 `build_diagnostic_suite.py`

Agent 创建：

```text
tools/stage3/build_diagnostic_suite.py
```

CLI：

```bash
python tools/stage3/build_diagnostic_suite.py \
  --adapter-dir PATH \
  --m1-root PATH \
  --linearization-spec PATH \
  --output-dir PATH \
  --seed 20260903
```

脚本必须输出以下表。

### 2.1 `diagnostic_episodes.csv`

每个入选 episode 一行：

```text
diagnostic_id,episode_id,task_id,scenario,outcome,path_signature,
split_original,content_group_id,analysis_weight,is_representative,
case_type,controller_source,source_path
```

`case_type` 仅允许：

```text
canonical_chain
alternative_order
recovery_success
terminal_failure
supplementary_fixed_chain_control
```

正式 G1 诊断优先使用 `val/test`。由于 recovery 和 alternative 的冻结样本有限，允许把 train 中的结构案例作为 `mechanism_only` 补充，但必须单列，不能与 test 汇总成同一个最终指标。

### 2.2 `path_pairs.csv`

用于比较相同 terminal outcome 的不同合法路径：

```text
pair_id,task_id,left_episode_id,right_episode_id,left_path,right_path,
outcome,left_content_group_id,right_content_group_id,pair_weight,evaluation_split
```

规则：

- 只配对 `transport_dual_order` 的成功 A→B 与成功 B→A；
- 优先使用代表 episode；
- 同一 content group 不重复进入多个统计独立 pair；
- 若 pair 数不足，不复制 episode 扩大 n，保留真实 pair 数。

### 2.3 `recovery_segments.csv`

```text
segment_id,episode_id,task_id,failure_onset_step,recovery_start_step,
recovery_complete_step,pre_failure_node,failed_node,restored_node,
segment_start_step,segment_end_step,content_group_id,analysis_weight,evaluation_split
```

规则：

- `segment_start_step=max(0,failure_onset_step-5)`；
- `segment_end_step=min(T-1,recovery_complete_step+5)`；
- `restored_node` 使用 runtime patch 后的值；
- 只保留具有明确 failure onset 和 recovery completion 的 episode。

### 2.4 `cycle_segments.csv`

每个 recovery episode 构造一个实际闭环片段：最近合法 node → failure node → recovery → 恢复到同一合法 node。

字段：

```text
cycle_id,episode_id,task_id,start_step,end_step,start_node,end_node,
cycle_kind,content_group_id,analysis_weight,evaluation_split
```

仅当 runtime start/end node 语义相同或属于明确恢复映射时创建。禁止人工拼接不同 episode 的状态作为主结果；跨 episode 拼接只能作为附录 stress case。

### 2.5 `control_segments.csv`

至少包含：

- `transport_recovery / natural_success`；
- `transport_dual_order / order_A_then_B` 作为 A-first canonical control；
- `transport_dual_order / order_B_then_A` 作为 B-first canonical control；
- `square/transport` 只在能找到原 SARM stage 配置或现成 prediction 时作为 supplementary control，不为它们伪造 stage GT。

字段：

```text
control_id,episode_id,task_id,scenario,orientation,start_step,end_step,
content_group_id,analysis_weight,evaluation_split
```

### 2.6 `terminal_pairs.csv`

在同一任务内配对 success 与 terminal failure，用于检查累计 reward 与 outcome 排序。字段：

```text
pair_id,task_id,success_episode_id,failure_episode_id,
match_stage_or_scenario,pair_weight,evaluation_split
```

## 步骤 3：生成 suite 与摘要

```bash
$PYTHON_BIN tools/stage3/build_diagnostic_suite.py \
  --adapter-dir "$ADAPTER_DIR" \
  --m1-root "$REPO_ROOT/artifacts/pathgraph_sarm/stage2/m1_freeze_v1" \
  --linearization-spec "$SUITE_DIR/configs/linearization_specs.yaml" \
  --output-dir "$SUITE_DIR" \
  --seed 20260903 \
  2>&1 | tee "$ROUND_DIR/logs/build_diagnostic_suite.log"
```

生成 `suite_summary.json`，至少包含：

```text
unique_diagnostic_content_groups
canonical_chain_count
alternative_order_count
path_pair_count
recovery_segment_count
cycle_segment_count
terminal_pair_count
test_case_count
mechanism_only_case_count
```

## 步骤 4：只做必要的冻结前检查

```bash
test -s "$SUITE_DIR/tables/diagnostic_episodes.csv"
test -s "$SUITE_DIR/tables/path_pairs.csv"
test -s "$SUITE_DIR/tables/recovery_segments.csv"
test -s "$SUITE_DIR/tables/cycle_segments.csv"
test -s "$SUITE_DIR/tables/control_segments.csv"
test -s "$SUITE_DIR/tables/terminal_pairs.csv"

$PYTHON_BIN - <<'PY'
from pathlib import Path
import csv, os, sys
root=Path(os.environ['SUITE_DIR'])/'tables'
def n(name):
    rows=list(csv.DictReader(open(root/name)))
    return len(rows)
checks={
 'path_pairs.csv': n('path_pairs.csv'),
 'recovery_segments.csv': n('recovery_segments.csv'),
 'control_segments.csv': n('control_segments.csv'),
 'terminal_pairs.csv': n('terminal_pairs.csv'),
}
print(checks)
if checks['path_pairs.csv'] < 1: sys.exit('No alternative-order pair')
if checks['recovery_segments.csv'] < 1: sys.exit('No recovery segment')
if checks['control_segments.csv'] < 2: sys.exit('Insufficient canonical controls')
PY
```

## 步骤 5：冻结 diagnostic suite

```bash
find "$SUITE_DIR" -type f ! -name 'DIAGNOSTIC_SUITE_SHA256SUMS.txt' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > "$SUITE_DIR/DIAGNOSTIC_SUITE_SHA256SUMS.txt"

cat > "$SUITE_DIR/FROZEN.md" <<EOF
# Diagnostic suite v1 frozen

- frozen_at: $(date -Iseconds)
- input_adapter: $ADAPTER_DIR
- rule: no episode replacement or threshold change after formal baseline scoring begins
- statistics_unit: content_group_id
- provenance: scripted_oracle mechanism evidence; not learned-policy evaluation
EOF

chmod -R a-w "$SUITE_DIR"
```

之后运行 baseline 时，只读取该目录，不修改。

## 本小阶段验收条件

- A→B / B→A 成功路径至少形成 1 个独立内容组 pair；
- 至少 1 个显式 recovery segment 和 1 个 terminal pair；
- canonical control 同时覆盖 A-first、B-first 与自然成功恢复任务；
- 每一行保留 `content_group_id`、`analysis_weight`、split 与 provenance；
- suite 已生成 checksum 并只读冻结；
- baseline 正式结果产生前 suite 已冻结。

## 本轮 ZIP

```bash
cp -a "$SUITE_DIR"/tables "$ROUND_DIR/"
cp -a "$SUITE_DIR"/configs "$ROUND_DIR/"
cp "$SUITE_DIR"/suite_summary.json "$ROUND_DIR/metrics/"
cp "$SUITE_DIR"/FROZEN.md "$ROUND_DIR/"
cp "$SUITE_DIR"/DIAGNOSTIC_SUITE_SHA256SUMS.txt "$ROUND_DIR/"

cat > "$ROUND_DIR/summary.md" <<EOF
# Stage 3.2 summary

- Diagnostic suite: frozen before scoring
- Statistics unit: content_group_id
- Required case families: canonical, alternative, recovery, terminal/cycle
- Source: scripted_oracle mechanism evidence
EOF

"$REPO_ROOT/tools/stage3/package_round.sh" "$ROUND_ID" "$ROUND_DIR"
```

交付路径：

```text
$STAGE3_ROOT/downloads/stage3_2_diagnostic_suite.zip
```

**本小阶段核心点：先把反例和对照固定下来，再运行模型；不允许看到结果后挑 episode。**


---

# 阶段 3.3：线性基线实现、GPU 并行训练与统一打分

## 总体上要干什么

实现一套统一的 reward baseline 接口，并完成所有必做方法在冻结 diagnostic suite 上的逐步预测。oracle 方法用于直接证明线性结构的内在限制；learned linear SARM 用同一个标量进度目标验证该问题并非只存在于手写公式。不同 task、orientation、seed 在有多 GPU 时并行运行。

本小阶段不训练 PathGraph 模型，不做 RA-BC，不用测试集选 checkpoint。

## 必做方法

| 方法 ID | 是否训练 | 作用 |
|---|---:|---|
| `linear_time_fraction` | 否 | 表示“时间越长进度越大”的极简单调标量，检查失败/循环被误奖励的问题 |
| `oracle_linear_chain_A_first` | 否 | A-first 总序上的 stage rank + 节点内进度 |
| `oracle_linear_chain_B_first` | 否 | B-first 总序，验证换 orientation 只会交换被惩罚的合法路径 |
| `oracle_linear_chain_recovery` | 否 | 固定 recovery 任务 stage chain |
| `sequential_transition_oracle` | 否 | 只奖励 canonical forward transition，检查 alternative/recovery edge 被遗漏或误判 |
| `learned_linear_sarm` | 是 | 历史 GRU + 标量 progress head，目标仍是单一线性 progress；3 seeds |

可选方法仅在仓库已有可运行实现时加入：`existing_repo_sarm`、`existing_repo_stage_transition`、`existing_repo_arm`。可选方法缺失不能阻塞 G1。

## 本轮目录与 GPU

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export ADAPTER_DIR="$STAGE3_ROOT/input_adapter_v1"
export SUITE_DIR="$STAGE3_ROOT/diagnostic_suite_v1"
export ROUND_ID="stage3_3_baseline_runs"
export ROUND_DIR="$STAGE3_ROOT/rounds/$ROUND_ID"
export PRED_ROOT="$ROUND_DIR/predictions"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage3/init_round.sh" \
  "$ROUND_ID" "$ROUND_DIR" \
  "Implement and run oracle and learned linear-progress baselines"
mkdir -p "$PRED_ROOT" "$ROUND_DIR/checkpoints"
cp "$REPO_ROOT/configs/stage3/stage3.yaml" "$ROUND_DIR/configs/"

# 确认 suite 在运行前已冻结。
test -f "$SUITE_DIR/FROZEN.md"
cd "$SUITE_DIR"
sha256sum -c DIAGNOSTIC_SUITE_SHA256SUMS.txt \
  | tee "$ROUND_DIR/logs/diagnostic_suite_checksum.log"
cd "$REPO_ROOT"

# GPU 必须提权查看。
"$REPO_ROOT/tools/stage3/query_gpus.sh" "$ROUND_DIR"
mapfile -t GPU_IDS < "$ROUND_DIR/available_gpu_ids.txt"
```

## 步骤 1：创建统一 prediction schema

每个方法对每个 episode、每个 step 输出一行 JSONL：

```json
{
  "episode_id": "...",
  "task_id": "...",
  "scenario": "...",
  "split": "test",
  "content_group_id": "...",
  "method": "...",
  "orientation": "A_first|B_first|recovery_chain|none",
  "seed": 20260903,
  "step": 0,
  "node_id_runtime": "start",
  "edge_id_runtime": null,
  "semantic_edge_type": null,
  "progress": 0.0,
  "reward_delta": 0.0,
  "stage_rank": 0,
  "within_node_phi": 0.0,
  "uncertainty": null,
  "controller_source": "scripted_oracle"
}
```

禁止把以下字段作为 learned baseline 输入：

```text
outcome
info.success
scenario
controller_source
episode_id
未来 node/edge 标签
未来帧
```

允许使用的当前时刻输入：

```text
eef_pos, object_pos, target_pos, gripper_state, action,
subgoal_A_done, subgoal_B_done
```

允许的历史仅为当前 step 之前最多 32 步。

## 步骤 2：实现数据与线性标签模块

创建：

```text
tools/stage3/lib/data.py
tools/stage3/lib/linearization.py
```

### `data.py` 必须实现

```python
load_episode(episode_row) -> dict
build_feature_matrix(episode, forbidden_fields) -> np.ndarray
load_runtime_annotation(episode_id) -> dict
window_sequence(features, history_steps) -> np.ndarray
```

细则：

- JSON episode 按 `resolved_source_path` 读取；
- 所有数组转为 `float32`；
- 缺失 `subgoal_A_done/B_done` 时补 0；
- 训练集均值/方差只由训练样本计算并保存到 `normalizer.json`；
- 不读取 outcome 作为输入；
- feature 维度和字段顺序写入 `feature_schema.json`。

### `linearization.py` 必须实现

```python
build_stepwise_runtime_labels(annotation, runtime_patch) -> rows
compute_within_node_phi(step, node_interval, progress_anchors) -> float
project_to_linear_progress(node_id, phi, chain, state) -> float
compute_reward_delta(progress) -> np.ndarray
```

线性 progress：

\[
p_t=\frac{k(z_t)+\phi_t}{K},
\qquad
r_t=p_{t+1}-p_t
\]

其中 `k(z_t)` 是当前 orientation 下的 canonical node rank，`K` 是 canonical transition 数。

对于 off-chain node 使用 `last_valid_rank_with_local_reset`：保留最近 canonical rank，局部进度归零；返回 canonical node 后恢复正常计算。

`linear_time_fraction`：

\[
p_t=\frac{t}{T-1},\qquad r_t=p_{t+1}-p_t
\]

## 步骤 3：实现 oracle baseline scorer

创建：

```text
tools/stage3/score_oracle_baselines.py
```

CLI：

```bash
python tools/stage3/score_oracle_baselines.py \
  --adapter-dir PATH \
  --suite-dir PATH \
  --linearization-spec PATH \
  --methods linear_time_fraction,oracle_linear_chain,sequential_transition_oracle \
  --output-dir PATH
```

`sequential_transition_oracle` 规则：

- canonical forward edge：`+1/K`；
- canonical backward edge：`-1/K`；
- legal alternative edge 若不在当前 orientation：按总序 rank delta 计分；
- failure：`-1/K`；
- recovery：若恢复到旧 rank，按 rank delta；
- stagnation：0；
- 该方法不访问未来 outcome。

运行：

```bash
$PYTHON_BIN tools/stage3/score_oracle_baselines.py \
  --adapter-dir "$ADAPTER_DIR" \
  --suite-dir "$SUITE_DIR" \
  --linearization-spec "$SUITE_DIR/configs/linearization_specs.yaml" \
  --methods linear_time_fraction,oracle_linear_chain,sequential_transition_oracle \
  --output-dir "$PRED_ROOT/oracle" \
  2>&1 | tee "$ROUND_DIR/logs/oracle_baselines.log"
```

oracle 方法在 CPU 上按 task 并行即可：

```bash
for task in transport_recovery transport_dual_order; do
  $PYTHON_BIN tools/stage3/score_oracle_baselines.py \
    --adapter-dir "$ADAPTER_DIR" \
    --suite-dir "$SUITE_DIR" \
    --task "$task" \
    --linearization-spec "$SUITE_DIR/configs/linearization_specs.yaml" \
    --methods linear_time_fraction,oracle_linear_chain,sequential_transition_oracle \
    --output-dir "$PRED_ROOT/oracle/$task" \
    > "$ROUND_DIR/logs/oracle_${task}.log" 2>&1 &
done
wait
```

## 步骤 4：实现 learned linear SARM

创建：

```text
tools/stage3/train_linear_sarm.py
tools/stage3/predict_linear_sarm.py
```

模型固定为：

```text
Input projection: feature_dim → 128
GRU: hidden=128, layers=1, history window=32
Progress head: 128 → 128 → 1, sigmoid
```

训练目标：

\[
\mathcal L
=
\operatorname{SmoothL1}(\hat p_t,p_t)
+0.1\,\mathcal L_{rank}
\]

其中 `rank loss` 只在 canonical clean-success 轨迹上要求后时刻的预测不低于前时刻。不要给 recovery/failure 训练样本强行制造线性标签。

### 训练数据规则

`dual_order_A_first`：

- 训练：`order_A_then_B` 的 clean success；
- 评估：A→B control、B→A alternative、B→A recovery、terminal failure；
- 同 scenario 的完全重复 episode 不作为独立验证样本；训练时每个 content group 每个 epoch 最多采一次，再用在线数值噪声增强输入，噪声仅作用于 train。

`dual_order_B_first`：

- 训练：`order_B_then_A` 的 clean success；
- 评估：B→A control 与 A→B alternative。

`transport_recovery_chain`：

- 训练：`natural_success`；
- 评估：natural success、recovery success、terminal failure。

固定训练预算，不使用 test 选择 epoch。checkpoint 选择只基于 train/val canonical control；若 dual-order 因 content group 只有一个 template 无有效 val，则固定使用最后 epoch 或预先指定 epoch 80，不能看 alternative/test 指标选 checkpoint。

### CLI

```bash
python tools/stage3/train_linear_sarm.py \
  --config configs/stage3/stage3.yaml \
  --adapter-dir PATH \
  --suite-dir PATH \
  --task TASK \
  --orientation ORIENTATION \
  --seed SEED \
  --output-dir UNIQUE_JOB_DIR
```

每个 job 必须生成：

```text
resolved_config.yaml
feature_schema.json
normalizer.json
train_metrics.csv
metrics.json
checkpoints/best_or_final.pt
DONE
```

checkpoint 默认不打包。

## 步骤 5：生成多 GPU job matrix

```bash
cat > "$ROUND_DIR/jobs.tsv" <<EOF
job_id\ttask_id\tmethod\torientation\tseed\tcommand\toutput_dir\trequired
EOF

IFS=',' read -ra SEEDS <<< "${STAGE3_SEEDS:-20260903,20260904,20260905}"
for seed in "${SEEDS[@]}"; do
  for orientation in A_first B_first; do
    job_id="dual_${orientation}_seed${seed}"
    out="$ROUND_DIR/jobs/$job_id"
    cmd="$PYTHON_BIN tools/stage3/train_linear_sarm.py --config $REPO_ROOT/configs/stage3/stage3.yaml --adapter-dir $ADAPTER_DIR --suite-dir $SUITE_DIR --task transport_dual_order --orientation $orientation --seed $seed --output-dir $out"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$job_id" transport_dual_order learned_linear_sarm "$orientation" "$seed" "$cmd" "$out" true \
      >> "$ROUND_DIR/jobs.tsv"
  done

  job_id="recovery_chain_seed${seed}"
  out="$ROUND_DIR/jobs/$job_id"
  cmd="$PYTHON_BIN tools/stage3/train_linear_sarm.py --config $REPO_ROOT/configs/stage3/stage3.yaml --adapter-dir $ADAPTER_DIR --suite-dir $SUITE_DIR --task transport_recovery --orientation recovery_chain --seed $seed --output-dir $out"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" transport_recovery learned_linear_sarm recovery_chain "$seed" "$cmd" "$out" true \
    >> "$ROUND_DIR/jobs.tsv"
done

column -t -s $'\t' "$ROUND_DIR/jobs.tsv"
```

总计 9 个 learned jobs。若有 4 张空闲 GPU，最多同时跑 4 个；若有 8 张，最多同时跑 8 个。默认每个 job 一张 GPU，不使用 DDP。

## 步骤 6：并行训练

```bash
"$REPO_ROOT/tools/stage3/query_gpus.sh" "$ROUND_DIR"
mapfile -t GPU_IDS < "$ROUND_DIR/available_gpu_ids.txt"
if [ "${#GPU_IDS[@]}" -lt 1 ]; then
  echo "没有提权确认的空闲 GPU；oracle 结果保留，learned jobs 暂不冒险占用忙卡。" >&2
  exit 3
fi

mapfile -t JOB_LINES < <(tail -n +2 "$ROUND_DIR/jobs.tsv")
for i in "${!JOB_LINES[@]}"; do
  IFS=$'\t' read -r job_id task_id method orientation seed command output_dir required <<< "${JOB_LINES[$i]}"
  gpu="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
  mkdir -p "$output_dir"

  while [ "$(jobs -pr | wc -l)" -ge "${#GPU_IDS[@]}" ]; do
    wait -n || true
  done

  CUDA_VISIBLE_DEVICES="$gpu" \
  nohup bash -lc "set -euo pipefail; $command" \
    > "$ROUND_DIR/logs/${job_id}.log" 2>&1 &
  echo "$!" > "$output_dir/pid.txt"
  echo "$gpu" > "$output_dir/gpu_id.txt"
done
wait
```

Agent 随后生成 `job_status.tsv`：

```text
job_id,status,exit_code,gpu_id,output_dir,metrics_path,checkpoint_path
```

缺少 `DONE` 的 required job 只重跑该 job；不要重跑成功项。

## 步骤 7：统一推理

```bash
for job_dir in "$ROUND_DIR"/jobs/*; do
  test -f "$job_dir/DONE" || continue
  job_id="$(basename "$job_dir")"
  $PYTHON_BIN tools/stage3/predict_linear_sarm.py \
    --checkpoint "$job_dir/checkpoints/best_or_final.pt" \
    --resolved-config "$job_dir/resolved_config.yaml" \
    --adapter-dir "$ADAPTER_DIR" \
    --suite-dir "$SUITE_DIR" \
    --output "$PRED_ROOT/${job_id}.jsonl" \
    > "$ROUND_DIR/logs/predict_${job_id}.log" 2>&1 &
done
wait
```

若推理可独立占 GPU，同样按 GPU 数并行；多个推理 job 不写同一个 JSONL。

## 步骤 8：可选仓库基线定位

只做一次定向搜索：

```bash
rg -n --glob '*.py' \
  '(SARM|StageAware|stage.?aware|reward[_-]?model|progress[_-]?model|stage[_-]?transition|ARM)' \
  "$REPO_ROOT" \
  > "$ROUND_DIR/manifests/existing_baseline_search.txt" || true
```

如果找到可运行 entrypoint：

- 写 adapter 输出同一 prediction schema；
- 使用相同 diagnostic suite；
- 独立 task/seed 多 GPU 并行；
- 记录原配置与 checkpoint 路径；
- 不因为其超参数与当前模型不同而重做大规模调参。

如果未找到，写：

```text
existing_repo_baseline_status.md
status: NOT_AVAILABLE_IN_CURRENT_CHECKOUT
impact: optional baseline only; G1 proceeds with required baselines
```

## 本小阶段必要检查

```bash
# 必做 oracle prediction 存在。
find "$PRED_ROOT/oracle" -name '*.jsonl' -size +0 | grep -q .

# 9 个 required learned job 均完成。
required=$(tail -n +2 "$ROUND_DIR/jobs.tsv" | awk -F'\t' '$8=="true"{n++} END{print n+0}')
done_count=$(find "$ROUND_DIR/jobs" -mindepth 2 -maxdepth 2 -name DONE | wc -l)
[ "$done_count" -eq "$required" ]

# 所有正式 prediction 不得含 NaN/Inf。
! grep -R -E 'NaN|Infinity|-Infinity' "$PRED_ROOT"
```

Agent 输出 `baseline_run_summary.csv`：

```text
method,task_id,orientation,seed,status,train_seconds,inference_seconds,
prediction_rows,checkpoint_path,checkpoint_packaged
```

其中 `checkpoint_packaged=false`。

## 本轮 ZIP

```bash
cat > "$ROUND_DIR/summary.md" <<EOF
# Stage 3.3 summary

- Required oracle baselines: completed
- Required learned jobs: 9
- GPU query: privileged
- Parallel policy: one independent job per available GPU
- Checkpoints: retained by path, omitted from ZIP
- Predictions: $PRED_ROOT
EOF

"$REPO_ROOT/tools/stage3/package_round.sh" "$ROUND_ID" "$ROUND_DIR"
```

交付路径：

```text
$STAGE3_ROOT/downloads/stage3_3_baseline_runs.zip
```

**本小阶段核心点：给线性方法最强、最公平的机会；用 oracle 和 learned 两条证据链验证结构问题，同时让 task/orientation/seed 默认多 GPU 并行。**


---

# 阶段 3.4：系统性误评分量化、统计与案例可视化

## 总体上要干什么

将阶段 3.3 的逐步预测映射回冻结 diagnostic suite，量化五类现象：canonical control 是否正常、合法 alternative edge 是否被给负奖励、不同合法路径累计分数是否偏置、recovery 是否被误判、failure/recovery cycle 是否产生不合理净回报。统计单位按 `content_group_id`，完全重复 episode 不能扩大置信度。

## 本轮目录

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export ADAPTER_DIR="$STAGE3_ROOT/input_adapter_v1"
export SUITE_DIR="$STAGE3_ROOT/diagnostic_suite_v1"
export BASELINE_DIR="$STAGE3_ROOT/rounds/stage3_3_baseline_runs"
export ROUND_ID="stage3_4_misscoring_analysis"
export ROUND_DIR="$STAGE3_ROOT/rounds/$ROUND_ID"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage3/init_round.sh" \
  "$ROUND_ID" "$ROUND_DIR" \
  "Quantify structural mis-scoring with content-group-aware statistics"
mkdir -p "$ROUND_DIR"/{tables,cases}
cp "$REPO_ROOT/configs/stage3/stage3.yaml" "$ROUND_DIR/configs/"
test -d "$BASELINE_DIR/predictions"
```

## 步骤 1：创建分析脚本

创建：

```text
tools/stage3/aggregate_misscoring.py
tools/stage3/render_reward_cases.py
tools/stage3/bootstrap_metrics.py
```

### `aggregate_misscoring.py` CLI

```bash
python tools/stage3/aggregate_misscoring.py \
  --adapter-dir PATH \
  --suite-dir PATH \
  --prediction-root PATH \
  --output-dir PATH
```

必须计算以下指标。

### 1.1 Canonical control 指标

`control_monotonicity`：canonical control 中 `reward_delta >= -epsilon` 的 transition 比例。

`control_forward_sign_accuracy`：semantic forward edge 获得正奖励的比例。

`control_terminal_progress`：成功终点的 progress 均值。

目的：确认线性基线在它擅长的固定链上正常，避免把实现 bug 误认为结构问题。

### 1.2 Alternative-order 指标

`alternative_legal_negative_rate`：合法 alternative edge 上 `reward_delta < -epsilon` 的比例。

`path_cumulative_reward`：每条成功路径的累计 reward。

`normalized_path_score_gap`：

\[
\frac{|R_{A\rightarrow B}-R_{B\rightarrow A}|}
{\max(\epsilon, (|R_{A\rightarrow B}|+|R_{B\rightarrow A}|)/2)}
\]

`orientation_swap_consistency`：换成 B-first orientation 后，是否只是把受惩罚路径从 B→A 换成 A→B。该结果能说明问题来自单一总序，而非某个 orientation 选错。

### 1.3 Failure / recovery 指标

对每个 recovery segment：

- `failure_sign_accuracy`：failure onset 附近累计 delta 是否为负；
- `recovery_local_positive`：从 recovery start 到 completion 的累计 delta 是否为正；
- `recovery_positive_rate`：有效 recovery segment 得正净回报的比例；
- `recovery_rank_error`：恢复完成时 progress 是否未回到/超过被恢复合法 node 应有的 progress；
- `recovery_delay_steps`：恢复动作开始后多少步才重新获得正 reward。

### 1.4 Cycle 指标

`cycle_net_reward`：实际 failure→recovery→restored-node 闭环的累计 reward。

`cycle_positive_rate`：闭环净 reward > epsilon 的 content group 比例。

`time_fraction_failure_reward`：`linear_time_fraction` 在 terminal failure episode 中因时间流逝获得的累计正 reward。

注意：阶段 3 的目的不是要求所有线性方法都产生正循环；只要出现“合法 recovery 得不到及时正奖励”或“时间进度奖励失败耗时”之一，就属于结构性缺陷。

### 1.5 Outcome 相关性

- episode cumulative reward 与 success 的 point-biserial correlation；
- episode cumulative reward 与完成度的 Spearman correlation；
- graph task 和 canonical control 分开报告；
- `scripted_oracle` provenance 始终保留。

## 步骤 2：重复 episode 的统计处理

所有聚合必须：

1. 先在 episode 内计算指标；
2. 再按 `content_group_id` 聚合；
3. 同一 content group 的 episode 权重总和为 1；
4. bootstrap 重采样单位为 content group，不是 episode id；
5. `transport_dual_order` 的同 scenario 完全重复样本不得被当成 20 或 36 个独立样本；
6. learned model 的三个 seed 是训练重复，不与 episode sample size 混为一谈。

输出：

```text
tables/episode_metrics.csv
tables/content_group_metrics.csv
tables/method_task_summary.csv
tables/path_pair_metrics.csv
tables/recovery_metrics.csv
tables/cycle_metrics.csv
tables/control_metrics.csv
metrics/bootstrap_ci.csv
```

`bootstrap_ci.csv` 字段：

```text
metric,method,task_id,orientation,seed_group,mean,ci_low,ci_high,
bootstrap_unit,n_units,n_episode_rows
```

## 步骤 3：执行分析

```bash
$PYTHON_BIN tools/stage3/aggregate_misscoring.py \
  --adapter-dir "$ADAPTER_DIR" \
  --suite-dir "$SUITE_DIR" \
  --prediction-root "$BASELINE_DIR/predictions" \
  --output-dir "$ROUND_DIR" \
  2>&1 | tee "$ROUND_DIR/logs/aggregate_misscoring.log"

$PYTHON_BIN tools/stage3/bootstrap_metrics.py \
  --metrics "$ROUND_DIR/tables/content_group_metrics.csv" \
  --group-column content_group_id \
  --resamples 2000 \
  --seed 20260903 \
  --output "$ROUND_DIR/metrics/bootstrap_ci.csv" \
  2>&1 | tee "$ROUND_DIR/logs/bootstrap.log"
```

## 步骤 4：生成必要案例图

`render_reward_cases.py` 为每种现象选择固定 suite 中的代表 content group；不从结果中挑最高效应 episode。

必须生成：

```text
plots/dual_order_Afirst_overlay.png
plots/dual_order_Bfirst_overlay.png
plots/orientation_swap_summary.png
plots/recovery_progress_trace.png
plots/recovery_reward_delta.png
plots/cycle_net_reward_summary.png
plots/control_monotonicity.png
plots/method_metric_matrix.png
cases/case_index.md
```

图中标记：

- node interval；
- failure onset；
- recovery start / complete；
- semantic edge type；
- progress 与 reward delta；
- method/orientation/seed；
- `controller_source=scripted_oracle`。

运行：

```bash
$PYTHON_BIN tools/stage3/render_reward_cases.py \
  --suite-dir "$SUITE_DIR" \
  --metrics-dir "$ROUND_DIR/tables" \
  --prediction-root "$BASELINE_DIR/predictions" \
  --output-dir "$ROUND_DIR/plots" \
  --case-index "$ROUND_DIR/cases/case_index.md" \
  2>&1 | tee "$ROUND_DIR/logs/render_cases.log"
```

## 步骤 5：形成误评分结论表

创建 `metrics/failure_signatures.csv`：

```text
signature_id,phenomenon,method,orientation,task_id,metric,value,
threshold,passes_as_structural_failure,content_group_count,evidence_file
```

独立 failure signature 定义：

1. `ALT_ORDER_NEGATIVE`：合法 alternative edge 的负奖励率达到阈值；
2. `PATH_ORIENTATION_BIAS`：A-first/B-first orientation 交换时，被惩罚路径随总序交换；
3. `RECOVERY_NOT_REWARDED`：有效 recovery 的正奖励率过低或恢复排序错误率过高；
4. `TIME_REWARDS_FAILURE`：time fraction 对 terminal failure 仍产生持续正 reward；
5. `POSITIVE_CYCLE`：failure/recovery 闭环出现正净 reward。

G1 不要求五项全部发生；至少两个独立现象稳定出现即可。

## 本小阶段必要检查

```bash
test -s "$ROUND_DIR/tables/method_task_summary.csv"
test -s "$ROUND_DIR/metrics/bootstrap_ci.csv"
test -s "$ROUND_DIR/metrics/failure_signatures.csv"
test -s "$ROUND_DIR/cases/case_index.md"
find "$ROUND_DIR/plots" -name '*.png' -size +0 | grep -q .

# 检查统计单位。
grep -q 'content_group_id' "$ROUND_DIR/metrics/bootstrap_ci.csv"
```

Agent 在 `summary.md` 按以下顺序写：

1. canonical control 是否正常；
2. alternative-order 是否存在 orientation-dependent negative reward；
3. recovery 是否及时获得正 reward；
4. time/cycle 是否产生不合理净收益；
5. 哪些结论来自 oracle，哪些来自 learned baseline；
6. 结果仅是 scripted-oracle mechanism evidence。

## 本轮 ZIP

```bash
cat > "$ROUND_DIR/summary.md" <<EOF
# Stage 3.4 summary

- Statistics unit: content_group_id
- Bootstrap: 2000 group-level resamples
- Main diagnostics: canonical control, alternative order, recovery, cycle, outcome ranking
- Provenance: scripted_oracle mechanism evidence
EOF

"$REPO_ROOT/tools/stage3/package_round.sh" "$ROUND_ID" "$ROUND_DIR"
```

交付路径：

```text
$STAGE3_ROOT/downloads/stage3_4_misscoring_analysis.zip
```

**本小阶段核心点：先证明固定链在正常链上可用，再证明它在合法分支与恢复上系统失真；重复模板只算一个内容组，不能虚增统计显著性。**


---

# 阶段 3.5：G1 决策、M2 冻结与 Stage 4 交接

## 总体上要干什么

依据阶段 3.4 已冻结的指标和阈值作出 G1 决策，不再新增 baseline 或替换诊断 episode。通过时冻结“问题成立”的 M2 证据包，并明确 Stage 4 要训练的历史条件化 node/edge/remaining-cost 模型输入；未通过时只返回必要的任务/线性定义修正，不直接堆图模型参数。

## 本轮目录

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export ANALYSIS_DIR="$STAGE3_ROOT/rounds/stage3_4_misscoring_analysis"
export ADAPTER_DIR="$STAGE3_ROOT/input_adapter_v1"
export SUITE_DIR="$STAGE3_ROOT/diagnostic_suite_v1"
export M2_DIR="$STAGE3_ROOT/m2_freeze_v1"
export ROUND_ID="stage3_5_g1_decision"
export ROUND_DIR="$STAGE3_ROOT/rounds/$ROUND_ID"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage3/init_round.sh" \
  "$ROUND_ID" "$ROUND_DIR" \
  "Evaluate G1, freeze M2 problem evidence, and hand off to Stage 4"
mkdir -p "$M2_DIR"
cp "$REPO_ROOT/configs/stage3/stage3.yaml" "$ROUND_DIR/configs/"
test -s "$ANALYSIS_DIR/metrics/failure_signatures.csv"
test -s "$ANALYSIS_DIR/tables/control_metrics.csv"
```

## 步骤 1：创建 `evaluate_g1_gate.py`

CLI：

```bash
python tools/stage3/evaluate_g1_gate.py \
  --config configs/stage3/stage3.yaml \
  --analysis-dir PATH \
  --output-dir PATH
```

脚本只读取已生成指标，输出：

```text
metrics/g1_gate.json
g1_decision.md
m2_problem_statement.md
stage4_handoff.md
```

## 步骤 2：G1 判定规则

### 必须先满足的 sanity 条件

1. `input_adapter` 校验通过；
2. diagnostic suite 在 baseline 之前冻结；
3. oracle 与 learned required jobs 完成；
4. canonical control 的 `control_monotonicity >= 0.90`；
5. 正式统计按 `content_group_id`；
6. 测试结果未用于 checkpoint/阈值选择。

### 结构性误评分 signature

下列任一项计 1 个独立 signature：

- `ALT_ORDER_NEGATIVE`：`alternative_legal_negative_rate >= 0.20`；
- `PATH_ORIENTATION_BIAS`：`normalized_path_score_gap >= 0.10`，且交换 A-first/B-first 后受惩罚路径同步交换；
- `RECOVERY_NOT_REWARDED`：`recovery_positive_rate <= 0.70` 或 `recovery_rank_error >= 0.20`；
- `TIME_REWARDS_FAILURE`：terminal failure 中 time-fraction reward 仍持续为正；
- `POSITIVE_CYCLE`：`cycle_positive_rate >= 0.10`。

### 决策

```text
GO_STAGE4
```

条件：sanity 全部通过，并且至少 2 个独立 structural signature 成立；其中至少 1 个必须来自 `alternative-order` 或 `recovery`，不能只靠 time-fraction 这一弱基线。

```text
REFINE_STAGE3
```

条件：结构现象存在，但 canonical control 失败、runtime patch 冲突、required prediction 缺失或统计单位错误。只修正直接阻塞项并重跑对应小轮。

```text
NO_GO_PATHGRAPH
```

条件：在冻结 GT 上，两个 orientation 的线性方法都没有出现稳定的 alternative/recovery 结构误评分，或现象完全可归因于实现 bug。

可选基线缺失不导致 `REFINE_STAGE3`。

## 步骤 3：执行 G1

```bash
$PYTHON_BIN tools/stage3/evaluate_g1_gate.py \
  --config "$REPO_ROOT/configs/stage3/stage3.yaml" \
  --analysis-dir "$ANALYSIS_DIR" \
  --output-dir "$ROUND_DIR" \
  2>&1 | tee "$ROUND_DIR/logs/evaluate_g1_gate.log"

cat "$ROUND_DIR/g1_decision.md"
cat "$ROUND_DIR/m2_problem_statement.md"
```

`g1_gate.json` 至少包含：

```json
{
  "decision": "GO_STAGE4",
  "sanity_pass": true,
  "control_monotonicity": 0.0,
  "structural_signature_count": 0,
  "signatures": [],
  "primary_tasks": ["transport_dual_order", "transport_recovery"],
  "statistics_unit": "content_group_id",
  "controller_source": "scripted_oracle"
}
```

实际值由脚本写入，禁止手工填写通过结果。

## 步骤 4：Stage 4 handoff 内容

若 `GO_STAGE4`，`stage4_handoff.md` 必须写清：

- 冻结的 runtime graph spec 路径；
- 冻结 diagnostic suite 路径及 checksum；
- Stage 4 训练/验证/test episode index；
- runtime patch 路径；
- 线性 baseline prediction 与聚合指标路径；
- 已确认的主要失败模式；
- Stage 4 模型必须输出：node belief、edge belief、within-node progress、remaining cost；
- 历史窗口起始建议为 32；
- Stage 4 模型选择只用 validation，不用 Stage 3 test diagnostics；
- `scripted_oracle` provenance 与 content-group 限制；
- checkpoint 位置只写 manifest，不进入 ZIP。

## 步骤 5：冻结 M2 包

```bash
rm -rf "$M2_DIR"
mkdir -p "$M2_DIR"/{metrics,tables,plots,manifests}

cp "$ROUND_DIR/g1_decision.md" "$M2_DIR/"
cp "$ROUND_DIR/m2_problem_statement.md" "$M2_DIR/"
cp "$ROUND_DIR/stage4_handoff.md" "$M2_DIR/"
cp "$ROUND_DIR/metrics/g1_gate.json" "$M2_DIR/metrics/"
cp "$ANALYSIS_DIR/metrics/failure_signatures.csv" "$M2_DIR/metrics/"
cp "$ANALYSIS_DIR/metrics/bootstrap_ci.csv" "$M2_DIR/metrics/"
cp "$ANALYSIS_DIR/tables/method_task_summary.csv" "$M2_DIR/tables/"
cp "$ANALYSIS_DIR/tables/path_pair_metrics.csv" "$M2_DIR/tables/"
cp "$ANALYSIS_DIR/tables/recovery_metrics.csv" "$M2_DIR/tables/"
cp "$ANALYSIS_DIR/tables/control_metrics.csv" "$M2_DIR/tables/"
cp -a "$ANALYSIS_DIR/plots/." "$M2_DIR/plots/"
cp "$ADAPTER_DIR/INPUT_ADAPTER_SHA256SUMS.txt" "$M2_DIR/manifests/"
cp "$SUITE_DIR/DIAGNOSTIC_SUITE_SHA256SUMS.txt" "$M2_DIR/manifests/"

find "$M2_DIR" -type f ! -name 'M2_SHA256SUMS.txt' -print0 \
  | LC_ALL=C sort -z \
  | xargs -0 sha256sum \
  > "$M2_DIR/M2_SHA256SUMS.txt"
```

checkpoint 不复制到 M2；在 `checkpoint_manifest.tsv` 中记录阶段 3.3 的路径即可。

## 步骤 6：生成本轮 ZIP

```bash
cp "$ROUND_DIR/g1_decision.md" "$ROUND_DIR/summary.md"
"$REPO_ROOT/tools/stage3/package_round.sh" "$ROUND_ID" "$ROUND_DIR"
```

交付路径：

```text
$STAGE3_ROOT/downloads/stage3_5_g1_decision.zip
```

## 步骤 7：生成 `stage3_complete.zip`

总包只包含轻量证据、配置、脚本、日志摘要、指标和图表；不打包 checkpoint、原始 episode 或缓存。

```bash
FINAL_STAGING="$STAGE3_ROOT/_stage3_complete_staging"
rm -rf "$FINAL_STAGING"
mkdir -p "$FINAL_STAGING/stage3"

cp -a "$STAGE3_ROOT/input_adapter_v1" "$FINAL_STAGING/stage3/"
cp -a "$STAGE3_ROOT/diagnostic_suite_v1" "$FINAL_STAGING/stage3/"
cp -a "$STAGE3_ROOT/m2_freeze_v1" "$FINAL_STAGING/stage3/"
cp -a "$STAGE3_ROOT/rounds" "$FINAL_STAGING/stage3/"
cp -a "$REPO_ROOT/configs/stage3" "$FINAL_STAGING/configs_stage3"
cp -a "$REPO_ROOT/tools/stage3" "$FINAL_STAGING/tools_stage3"

find "$FINAL_STAGING" -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' \
     -o -name '*.pkl' -o -name '*.hdf5' -o -name '*.mp4' -o -name '*.avi' -o -name '*.mov' \) \
  -delete
find "$FINAL_STAGING" -type d \
  \( -name checkpoints -o -name dataset_cache -o -name wandb -o -name __pycache__ \) \
  -prune -exec rm -rf {} +

FINAL_ZIP="$STAGE3_ROOT/downloads/stage3_complete.zip"
rm -f "$FINAL_ZIP" "$FINAL_ZIP.sha256"
cd "$FINAL_STAGING"
find . -type f ! -size +"${ZIP_MAX_FILE_MB:-200}"M -print \
  | LC_ALL=C sort | zip -q "$FINAL_ZIP" -@
sha256sum "$FINAL_ZIP" > "$FINAL_ZIP.sha256"
unzip -t "$FINAL_ZIP" > "$STAGE3_ROOT/downloads/stage3_complete_unzip_test.txt"
ls -lh "$FINAL_ZIP" "$FINAL_ZIP.sha256"
```

## 本小阶段验收条件

- `g1_gate.json` 由脚本根据固定阈值生成；
- `g1_decision.md` 明确给出 `GO_STAGE4 / REFINE_STAGE3 / NO_GO_PATHGRAPH`；
- `m2_problem_statement.md` 将结论限定为 linear progress 的结构问题；
- `stage4_handoff.md` 可直接作为下一阶段 Agent 输入；
- M2 checksum 通过；
- `stage3_5_g1_decision.zip` 与 `stage3_complete.zip` 均通过 `unzip -t`；
- checkpoint 等大文件不在 ZIP 中，但 manifest 路径完整。

**本小阶段核心点：G1 只决定“线性表示的问题是否成立”；通过后才进入 Stage 4 训练图状态与剩余代价模型。**
