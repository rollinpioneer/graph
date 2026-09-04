# 阶段 1 关闭结论与阶段 2 入口

## 结论

阶段 1 可以结束，结束状态为 **M0 已完成、G0=`SWITCH`**，而不是 PathGraph 模型训练的 `GO`。

已交付内容足以完成阶段 1 的职责：

- 200 个完整 episode，`square` 与 `transport` 各 100 个；
- 资产清单、轨迹覆盖、`dataset_v0.1`、固定 split、候选任务评分与 G0 决策均已生成；
- `group_leakage_count=0`；
- ZIP 完整性检查通过；
- 现有证据只有 forward/failure outcome，没有可验证的 alternative-order 或 episode 内 recovery；
- 已生成定向补采计划与 Stage 2 handoff。

## 必须保留的限制

阶段 1 的 `episode_events.jsonl` 中，部分事件是由任务语义和最终 success/failure 启发式生成的占位值，例如成功子目标按时长比例放置、失败 onset 放在 episode 中点。这些值可以辅助定位文件，但**不能作为 node/edge ground truth，也不能直接训练模型**。阶段 2 必须从原始低维状态、视频帧或新补采时的干预记录中重建真实事件。

## 阶段 2 的实际定位

由于 G0 是 `SWITCH`，阶段 2 不直接进入 reward model 训练，而是依次完成：

1. 从现有原始 rollout 中挖掘真实恢复事件；
2. 不足部分定向补采，并最终确定两个图结构任务；
3. 定义人工 Graph spec v1；
4. 建立标注协议和工具；
5. 形成 node/edge GT 子集；
6. 冻结 M1 并交给阶段 3 做线性 SARM 误评分验证。

**关闭口径：阶段 1 已完成；PathGraph 主线仍处于“补足可验证数据后继续”的状态。**


# PathGraph-SARM 阶段 2：Agent 详细操作命令（V1.0）

> 阶段名称：证据补齐、人工任务图与 Ground Truth 冻结。  
> 已知入口：阶段 1 已完成，`G0=SWITCH`；现有 200 个完整 rollout 有成功/失败 outcome，但没有可验证的 alternative-order 与 episode 内 recovery 标注。  
> 阶段目标：把至少两个任务升级为真正可验证的图结构任务，冻结 `Graph spec v1 + annotation protocol v1 + GT subset v1`，形成 M1 交付包。  
> 本阶段不训练 node/edge reward model，不训练 RA-BC，不做自动图发现；这些工作留到后续阶段。

## 给 Agent 的总命令

直接在 `/home/xushijie/CUPID` 内执行本文件。先读取阶段 1 的最终产物，不重新跑阶段 1。优先从现有 200 个原始 rollout 中提取真实状态事件；确实不足时再定向补采。不要把阶段 1 的启发式 `episode_events.jsonl` 当作 GT。可独立的 rollout、seed、任务和视觉推理必须并行。每个小阶段完成后立即生成对应 ZIP；checkpoint、原始 episode、视频等大文件不打包，只写清单和路径。

## 阶段 2 总体完成条件

阶段 2 只有在以下产物形成后才结束：

1. 至少 2 个 graph-valid 任务；每个任务至少满足“多合法路径”或“显式失败恢复”之一。
2. 整体任务集中至少 1 个任务有两条合法成功路径，且每条至少 8 个 GT episode；至少 1 个任务有 10 个显式 recovery episode。
3. 每个任务有 5-10 个可判别语义节点和完整 edge 字典。
4. node interval、edge interval、attempt、failure onset、recovery complete、within-node progress 均有明确标注规则。
5. 每个关键 edge 至少有 8 个 GT 实例；原始数据和 GT split 按 episode/group 隔离。
6. `m1_decision.md` 为 `GO_STAGE3`，并生成 `stage2_complete.zip`。

## 阶段 2 小阶段

- 2.1：原始轨迹解码、真实事件挖掘与 G0.1 证据刷新。
- 2.2：定向补采、图任务选择与任务集最终确定。
- 2.3：任务边界与人工 Graph spec v1。
- 2.4：标注协议、自动提议和审阅工具。
- 2.5：GT 子集构建与最小一致性复核。
- 2.6：图、数据、标注冻结与 M1 / Stage 3 交接。



## 通用目录和环境变量

已知阶段 1 的实际项目根目录为 `/home/xushijie/CUPID`。Agent 先执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE1_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage1"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export STAGE2_CONFIG="$REPO_ROOT/configs/stage2/stage2.yaml"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"

cd "$REPO_ROOT"
mkdir -p \
  "$REPO_ROOT/configs/stage2" \
  "$REPO_ROOT/tools/stage2" \
  "$REPO_ROOT/tools/stage2/lib" \
  "$STAGE2_ROOT/_runtime" \
  "$STAGE2_ROOT/rounds" \
  "$STAGE2_ROOT/downloads"

# 阶段 1 直接输入。若任一文件不存在，先定位 stage1_complete.zip 并解压回 REPO_ROOT，
# 不要重新运行阶段 1。
test -f "$STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl"
test -f "$STAGE1_ROOT/1.2_trajectory_coverage/trajectory_tags.csv"
test -f "$STAGE1_ROOT/1.4_g0_decision/g0_decision.md"
test -f "$STAGE1_ROOT/1.4_g0_decision/selected_tasks.yaml"

sed -n '1,160p' "$STAGE1_ROOT/1.4_g0_decision/g0_decision.md"
```

创建阶段 2 总配置：

```bash
cat > "$STAGE2_CONFIG" <<'YAML'
project:
  repo_root: /home/xushijie/CUPID
  stage1_root: artifacts/pathgraph_sarm/stage1
  stage2_root: artifacts/pathgraph_sarm/stage2

inputs:
  episode_manifest: artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl
  splits: artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/splits.csv
  trajectory_tags: artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/trajectory_tags.csv
  manual_review_queue: artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/manual_review_queue.csv
  selected_tasks: artifacts/pathgraph_sarm/stage1/1.4_g0_decision/selected_tasks.yaml
  targeted_collection_plan: artifacts/pathgraph_sarm/stage1/1.4_g0_decision/targeted_collection_plan.csv

runtime:
  seed: 20260831
  cpu_workers: 16
  zip_max_file_mb: 200

stage1_status:
  g0: SWITCH
  reason: no_verifiable_alternative_order_or_episode_internal_recovery
  # 阶段 1 的 episode_events.jsonl 含启发式占位事件，不作为 GT 输入。
  reject_stage1_placeholder_events_as_gt: true

evidence_targets:
  selected_graph_tasks: 2
  require_at_least_one_alternative_order_task: true
  require_at_least_one_recovery_task: true
  min_success_per_legal_path: 10
  min_recovery_episodes_per_recovery_task: 10
  min_terminal_failure_episodes_per_task: 8
  min_forward_success_episodes_per_task: 12

annotation_targets:
  min_gt_episodes_per_task: 40
  min_examples_per_critical_edge: 8
  min_examples_per_legal_path: 8
  ambiguous_or_random_review_fraction: 0.10
  history_window_steps: 32
  progress_anchors: [0.0, 0.25, 0.5, 0.75, 1.0]

split:
  seed: 20260831
  ratios: {train: 0.60, val: 0.20, test: 0.20}
  group_keys: [task_instance_id, scene_id, seed, episode_id]
YAML

sed -n '1,260p' "$STAGE2_CONFIG"
```

### 创建统一打包脚本

Agent 在第一次执行阶段 2 时创建以下脚本，后续每个小阶段直接复用：

```bash
cat > "$REPO_ROOT/tools/stage2/package_round.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 ROUND_ID ROUND_DIR" >&2
  exit 2
fi

ROUND_ID="$1"
ROUND_DIR="$(realpath "$2")"
STAGE2_ROOT="${STAGE2_ROOT:-$(realpath artifacts/pathgraph_sarm/stage2)}"
DOWNLOAD_DIR="$STAGE2_ROOT/downloads"
ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
ZIP_PATH="$DOWNLOAD_DIR/${ROUND_ID}.zip"

mkdir -p "$DOWNLOAD_DIR" "$ROUND_DIR"
if [ ! -s "$ROUND_DIR/run_manifest.md" ]; then
  echo "Missing run_manifest.md in $ROUND_DIR" >&2
  exit 3
fi
if [ ! -s "$ROUND_DIR/summary.md" ]; then
  echo "Missing summary.md in $ROUND_DIR" >&2
  exit 4
fi
grep -q "^- finished_at:" "$ROUND_DIR/run_manifest.md" || echo "- finished_at: $(date -Iseconds)" >> "$ROUND_DIR/run_manifest.md"
printf 'path\tsize_bytes\tartifact_type\treason_omitted\n' > "$ROUND_DIR/large_file_manifest.tsv"
printf 'path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n' > "$ROUND_DIR/checkpoint_manifest.tsv"

find "$ROUND_DIR" -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' \) \
  -printf '%p\t%s\tcheckpoint_or_model_weight\tdefault_omit_from_zip\n' \
  >> "$ROUND_DIR/large_file_manifest.tsv" || true

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
  ! -path './__pycache__/*' \
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
chmod +x "$REPO_ROOT/tools/stage2/package_round.sh"
```

### 创建统一轮次初始化脚本

```bash
cat > "$REPO_ROOT/tools/stage2/init_round.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ]; then
  echo "Usage: $0 ROUND_ID ROUND_DIR PURPOSE" >&2
  exit 2
fi
ROUND_ID="$1"
ROUND_DIR="$2"
PURPOSE="$3"
mkdir -p "$ROUND_DIR"/{configs,logs,metrics,plots,jobs}
cat > "$ROUND_DIR/run_manifest.md" <<EOF
# Run Manifest

- round_id: $ROUND_ID
- purpose: $PURPOSE
- started_at: $(date -Iseconds)
- repo_root: ${REPO_ROOT:-$(pwd)}
- git_commit: $(git -C "${REPO_ROOT:-$(pwd)}" rev-parse HEAD 2>/dev/null || echo unknown)
- python: $(${PYTHON_BIN:-python} --version 2>&1)
EOF
BASH
chmod +x "$REPO_ROOT/tools/stage2/init_round.sh"
```

### 创建提权 GPU 查询脚本

```bash
cat > "$REPO_ROOT/tools/stage2/query_gpus.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:?Usage: $0 OUT_DIR [MIN_FREE_MB]}"
MIN_FREE_MB="${2:-12000}"
mkdir -p "$OUT_DIR"

SUDO_CMD=(sudo -n)
if ! sudo -n nvidia-smi >/dev/null 2>&1; then
  echo "sudo -n 无法使用，改为请求交互式 sudo 查看 GPU。" >&2
  sudo nvidia-smi >/dev/null
  SUDO_CMD=(sudo)
fi

"${SUDO_CMD[@]}" nvidia-smi | tee "$OUT_DIR/nvidia_smi_full.txt"
"${SUDO_CMD[@]}" nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits \
  | tee "$OUT_DIR/gpu_status.csv"

"${SUDO_CMD[@]}" nvidia-smi \
  --query-gpu=index,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  | awk -F',' -v min_mem="$MIN_FREE_MB" '
      {
        gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3);
        if ($2 >= min_mem && $3 <= 30) print $1
      }' > "$OUT_DIR/available_gpu_ids.txt"

if [ ! -s "$OUT_DIR/available_gpu_ids.txt" ]; then
  echo "提权后未找到满足阈值的空闲 GPU。" >&2
  exit 3
fi
cat "$OUT_DIR/available_gpu_ids.txt"
BASH
chmod +x "$REPO_ROOT/tools/stage2/query_gpus.sh"
```




## 固定执行规则（本阶段所有小阶段都必须遵守）

### A. GPU 必须提权查看

任何 GPU 训练、策略 rollout、视觉模型推理、批量特征提取开始前，先执行：

```bash
mkdir -p "$STAGE2_ROOT/_runtime"

# 优先使用无交互提权。
sudo -n nvidia-smi | tee "$STAGE2_ROOT/_runtime/nvidia_smi_full.txt"
sudo -n nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits \
  | tee "$STAGE2_ROOT/_runtime/gpu_status.csv"
```

若 `sudo -n` 因需要授权而失败，使用：

```bash
sudo nvidia-smi
sudo nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits \
  | tee "$STAGE2_ROOT/_runtime/gpu_status.csv"
```

普通权限下看不到 GPU 时，不得据此写“无 GPU”。本轮实际使用的 GPU ID 必须写入 `run_manifest.md`。

读取可用 GPU 的默认命令：

```bash
MIN_FREE_MB="${MIN_FREE_MB:-12000}"
mapfile -t GPU_IDS < <(
  sudo -n nvidia-smi \
    --query-gpu=index,memory.free,utilization.gpu \
    --format=csv,noheader,nounits \
  | awk -F',' -v min_mem="$MIN_FREE_MB" '
      {
        gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
        if ($2 >= min_mem && $3 <= 30) print $1
      }'
)
printf 'Available GPUs: %s\n' "${GPU_IDS[*]:-none}"
```

### B. 可以多 GPU 并行时默认并行

优先采用**实验级并行**：不同任务、场景、seed、rollout 分片或模型推理分片各占一张 GPU。只有单个程序已经可靠支持 DDP 时，才使用 `torchrun`。每个 job 必须使用独立输出目录、日志和 checkpoint 目录，不允许并行进程写同一结果文件。

阶段 2 中最适合并行的工作是：

- 不同任务或场景的策略 rollout；
- 不同 seed 范围的定向补采；
- 不同 episode 分片的视觉事件提议；
- 需要视觉编码器时的批量特征提取。

纯 CSV/JSON 整理、Graph spec 编写和标注合并使用 CPU 并行即可，不为形式上的多 GPU 强行改造代码。

### C. 每个小阶段结束都必须生成 ZIP

每轮 ZIP 至少包含：配置、实际运行命令、日志、指标/统计、图表或样例、结果摘要、失败 job 日志、未打包大文件清单。**checkpoint、模型权重、原始 episode、缓存、长视频和其他大文件默认不打包。**

未打包文件写入：

- `large_file_manifest.tsv`
- `checkpoint_manifest.tsv`（有 checkpoint 时）

每个小阶段固定生成：

```text
stage2_1_raw_event_mining.zip
stage2_2_targeted_collection.zip
stage2_3_graph_spec_v1.zip
stage2_4_annotation_tooling.zip
stage2_5_gt_subset_v1.zip
stage2_6_m1_freeze.zip
```

阶段 2 全部完成后，再生成：

```text
stage2_complete.zip
```

ZIP 默认排除扩展名：`.ckpt .pt .pth .bin .safetensors .pkl .hdf5 .mp4 .avi .mov`，并排除超过 `ZIP_MAX_FILE_MB` 的其他文件；默认阈值为 200 MB。

---

# 阶段 2.1：原始轨迹解码与真实事件挖掘

> 执行前置：先完成 `00_阶段1关闭结论与阶段2总览.md` 中的通用目录、`stage2.yaml` 和 `tools/stage2/package_round.sh` 初始化。若这些文件已经存在，直接进入本小阶段，不重复初始化。


## 总体上要干什么

读取阶段 1 清单指向的 200 个原始 `.pkl` rollout，识别真实的低维状态字段和事件，而不是依据最终 success/failure 猜测事件位置。先从已有数据中找出真正的抓取、掉落、重新抓取、放置、回访和停滞；只把有状态或帧证据的事件写入候选标签。

本小阶段的目标不是标完 GT，而是最大限度复用已有 rollout，减少不必要补采。

## 直接给 Agent 的执行命令

### 1. 初始化本轮目录

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE1_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage1"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export STAGE2_CONFIG="$REPO_ROOT/configs/stage2/stage2.yaml"
export ROUND_ID="stage2_1_raw_event_mining"
export ROUND_DIR="$STAGE2_ROOT/rounds/$ROUND_ID"
cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"/{configs,logs,metrics,plots,samples,event_candidates}
cp "$STAGE2_CONFIG" "$ROUND_DIR/configs/stage2.yaml"
```

### 2. 初始化本轮 manifest，并明确禁止误用阶段 1 占位事件

```bash
bash tools/stage2/init_round.sh \
  "$ROUND_ID" "$ROUND_DIR" \
  "decode raw rollouts and mine evidence-backed semantic events"
cat >> "$ROUND_DIR/run_manifest.md" <<EOF
- input_manifest: $STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl
- forbidden_gt_input: $STAGE1_ROOT/1.2_trajectory_coverage/episode_events.jsonl
- note: stage1 episode events are heuristic placeholders and are not used as GT
EOF
```

### 3. 实现原始 episode schema 检查器

创建 `tools/stage2/inspect_episode_schema.py`。必须实现以下行为：

1. 从 `episode_manifest.jsonl` 中按 task/outcome 各取 3 个 episode。
2. 加载 `.pkl`；递归打印字典 key、数组 shape、dtype、首尾步索引。
3. 对数值序列输出最小值、最大值、均值和缺失率，但不把完整数组写入结果。
4. 自动寻找候选字段：
   - action：`action/actions`；
   - 末端位姿：包含 `eef/end_effector/gripper_pos`；
   - 夹爪开合：包含 `gripper/qpos/grip`；
   - 物体位姿：包含 `object/cube/nut/item/obj`；
   - 目标位姿：包含 `target/goal/site`；
   - 接触/碰撞：包含 `contact/collision/touch`；
   - reward/done/success。
5. 生成：
   - `schema_inventory.json`；
   - `field_mapping_candidates.csv`；
   - `representative_episode_keys.md`；
   - 无法加载的 episode 写入 `decode_failures.csv`。
6. 不修改原始 `.pkl`。

运行：

```bash
$PYTHON_BIN tools/stage2/inspect_episode_schema.py \
  --manifest "$STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" \
  --per-task-outcome 3 \
  --output-dir "$ROUND_DIR/schema" \
  2>&1 | tee "$ROUND_DIR/logs/inspect_schema.log"
```

### 4. 冻结实际状态字段映射

根据 `field_mapping_candidates.csv` 和代表 episode，创建：

```bash
cat > "$REPO_ROOT/configs/stage2/task_state_fields.yaml" <<'YAML'
tasks:
  square:
    # Agent 必须替换为原始 pkl 中真实存在的 key；找不到的字段填 null。
    action: null
    eef_pos: null
    gripper_state: null
    object_pos: null
    object_quat: null
    target_pos: null
    contact: null
    collision: null
    success: null
  transport:
    action: null
    eef_pos: null
    gripper_state: null
    object_pos: null
    object_quat: null
    source_pos: null
    target_pos: null
    contact: null
    collision: null
    success: null
YAML
```

Agent 逐项替换 `null`，随后运行一个最小字段读取检查：每个 task 各读取 1 个 success 和 1 个 failure，确认字段长度与 `num_steps` 一致。这里只做一次，不扩展成通用测试框架。

### 5. 实现真实事件提议器

创建 `tools/stage2/mine_semantic_events.py`，接口如下：

```text
--manifest PATH
--field-map PATH
--output-dir PATH
--workers N
--task TASK（可选）
--episode-list PATH（可选）
```

脚本必须基于真实状态序列实现：

- `approach_start/complete`：末端到物体距离持续下降并首次进入近距离阈值；
- `grasp_start/complete`：夹爪闭合后，物体与末端相对位姿在连续帧内稳定，或接触字段给出稳定接触；
- `lift_complete`：物体高度相对初始高度超过阈值，并持续若干帧；
- `transport_start/complete`：抓取后物体向目标移动，目标距离显著下降；
- `place_complete`：物体进入目标区域并保持稳定；
- `release_complete`：夹爪打开且物体在目标区域保持稳定；
- `failure_onset`：最早出现“已抓取后掉落、偏离目标、碰撞导致有效状态丢失、超时停滞”等可证实条件的帧；
- `recovery_start`：failure 后重新开始可恢复动作；
- `recovery_complete`：failure 后首次重新进入先前有效 node，并稳定持续 `stability_frames`；
- `retry`：同一语义 edge 在 failure 后再次开始；
- `revisit`：离开某 node 后又回到该 node；
- `stagnation`：一段窗口内关键距离、物体位姿和 action 都没有有效变化。

阈值要求：

1. 优先从 success episode 的分布分位数估计；
2. 所有最终阈值写入 `event_thresholds.yaml`；
3. 每个事件必须保存 `evidence_fields`、`evidence_start_step`、`evidence_end_step`、`confidence`；
4. 无足够字段或视觉证据时，输出 `needs_review=true`，不得把 episode 中点写成 failure onset；
5. Stage 1 的成功路径字符串只能作为任务语义提示，不能提供事件时间。

推荐默认参数：

```yaml
stability_frames: 5
min_event_gap_frames: 3
stagnation_window_frames: 30
relative_motion_tolerance: 0.02
near_object_quantile: 0.10
place_distance_quantile: 0.10
```

### 6. CPU 并行处理 200 个 episode

```bash
WORKERS="${WORKERS:-16}"
$PYTHON_BIN tools/stage2/mine_semantic_events.py \
  --manifest "$STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" \
  --field-map "$REPO_ROOT/configs/stage2/task_state_fields.yaml" \
  --output-dir "$ROUND_DIR/event_candidates" \
  --workers "$WORKERS" \
  2>&1 | tee "$ROUND_DIR/logs/mine_events.log"
```

必须生成：

```text
event_candidates/events.jsonl
event_candidates/episode_structure_candidates.csv
event_candidates/recovery_candidates.csv
event_candidates/retry_revisit_candidates.csv
event_candidates/event_thresholds.yaml
event_candidates/needs_review.csv
```

### 7. 对需要视觉确认的 episode 生成轻量 review 样例

只对以下 episode 生成 contact sheet，不批量打包完整视频：

- recovery 候选的前 20 个；
- failure 候选的前 10 个；
- 每个 task 随机 5 个 success。

创建 `tools/stage2/render_event_contact_sheets.py`，输出带帧号、时间步、候选事件竖线和关键状态数值的 PNG。若解码或视觉编码使用 GPU，先按固定规则提权查看 GPU，并按 episode 分片并行；普通视频解码用 CPU 多进程即可。

```bash
$PYTHON_BIN tools/stage2/render_event_contact_sheets.py \
  --manifest "$STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" \
  --events "$ROUND_DIR/event_candidates/events.jsonl" \
  --selection "$ROUND_DIR/event_candidates/needs_review.csv" \
  --max-recovery 20 --max-failure 10 --random-success-per-task 5 \
  --output-dir "$ROUND_DIR/samples/contact_sheets" \
  2>&1 | tee "$ROUND_DIR/logs/render_contact_sheets.log"
```

### 8. 生成 G0.1 证据刷新结论

实现 `tools/stage2/summarize_event_evidence.py`，按 task 汇总：

- 可加载 episode 数；
- 有证据的 forward/failure/recovery/retry/revisit 数；
- distinct path signature 数；
- 需要人工确认数；
- 是否达到阶段 2.2 补采前的最低证据。

```bash
$PYTHON_BIN tools/stage2/summarize_event_evidence.py \
  --events "$ROUND_DIR/event_candidates/events.jsonl" \
  --episodes "$ROUND_DIR/event_candidates/episode_structure_candidates.csv" \
  --output-csv "$ROUND_DIR/metrics/event_evidence_summary.csv" \
  --output-md "$ROUND_DIR/g0_1_evidence_refresh.md"
```

## 本小阶段最小完成检查

只检查以下内容：

```bash
test -s "$ROUND_DIR/schema/schema_inventory.json"
test -s "$ROUND_DIR/event_candidates/events.jsonl"
test -s "$ROUND_DIR/event_candidates/episode_structure_candidates.csv"
test -s "$ROUND_DIR/metrics/event_evidence_summary.csv"

$PYTHON_BIN - <<'PY'
import pandas as pd
p = 'artifacts/pathgraph_sarm/stage2/rounds/stage2_1_raw_event_mining/event_candidates/episode_structure_candidates.csv'
df = pd.read_csv(p)
assert len(df) == 200, len(df)
assert df['episode_id'].is_unique
print(df.groupby('task_id').size())
PY
```

完成标准：200 个 episode 全部得到“已提取或明确需复核”的状态；任何 failure/recovery 时间点都有真实字段或帧证据；不出现基于 episode 中点的伪 onset。

## 本轮摘要和 ZIP

`summary.md` 必须写清：现有数据中实际发现多少 recovery、哪些任务仍需补采、哪些字段支持事件判断。直接以 G0.1 结论为基础生成：

```bash
cp "$ROUND_DIR/g0_1_evidence_refresh.md" "$ROUND_DIR/summary.md"
{
  echo "- cpu_workers: ${WORKERS:-16}"
  echo "- gpu_ids: ${GPU_IDS[*]:-none}"
} >> "$ROUND_DIR/run_manifest.md"

bash tools/stage2/package_round.sh "$ROUND_ID" "$ROUND_DIR"
```

Agent 最终明确返回：

```text
$STAGE2_ROOT/downloads/stage2_1_raw_event_mining.zip
```

---

# 阶段 2.2：定向补采与图任务集最终确定

> 执行前置：先完成 `00_阶段1关闭结论与阶段2总览.md` 中的通用目录、`stage2.yaml` 和 `tools/stage2/package_round.sh` 初始化。若这些文件已经存在，直接进入本小阶段，不重复初始化。


## 总体上要干什么

依据阶段 2.1 的真实事件结果，补足图结构证据。优先复用已发现的 recovery；不足时用现有 policy 做定向 rollout。`square` 和 `transport` 可作为 recovery/fixed-chain 对照，但不要强行把单物体固定顺序解释成 alternative order。必须在仓库中选择一个现成多子目标任务，或快速构建一个最小双子目标组合任务，使两种完成顺序都合法。

本小阶段结束时，任务集必须被最终确定：至少两个 graph-valid 任务，整体至少覆盖两条合法成功路径和显式 failure→recovery。

## 直接给 Agent 的执行命令

### 1. 初始化本轮

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE1_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage1"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export STAGE2_CONFIG="$REPO_ROOT/configs/stage2/stage2.yaml"
export ROUND_ID="stage2_2_targeted_collection"
export ROUND_DIR="$STAGE2_ROOT/rounds/$ROUND_ID"
cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"/{configs,logs,metrics,plots,jobs,task_registry,collection_manifests}
cp "$STAGE2_CONFIG" "$ROUND_DIR/configs/stage2.yaml"
bash tools/stage2/init_round.sh \
  "$ROUND_ID" "$ROUND_DIR" \
  "targeted rollout collection and final graph-task selection"
```

### 2. 先读取 2.1 证据，不重复扫描

```bash
EVENT_SUMMARY="$STAGE2_ROOT/rounds/stage2_1_raw_event_mining/metrics/event_evidence_summary.csv"
EVENT_EPISODES="$STAGE2_ROOT/rounds/stage2_1_raw_event_mining/event_candidates/episode_structure_candidates.csv"
test -s "$EVENT_SUMMARY"
column -s, -t "$EVENT_SUMMARY" | sed -n '1,80p'
```

### 3. 枚举可直接执行的候选任务

创建 `tools/stage2/list_task_candidates.py`，只扫描项目当前已有的环境注册表、Hydra 配置、robosuite/robomimic task 名称和 rollout 配置，生成：

```text
task_registry/task_candidate_registry.csv
```

字段至少为：

```text
task_id,registry_source,config_path,env_class,num_objects,num_targets,
independent_subgoals,order_can_vary,existing_checkpoint,rollout_entrypoint,notes
```

执行前先定位注册入口：

```bash
grep -RInE \
  "register_env|TASK_REGISTRY|ENV_REGISTRY|env_name|task_name|rollout|save_episodes|eval_save_episodes" \
  "$REPO_ROOT" \
  --include='*.py' --include='*.yaml' --include='*.yml' --include='*.sh' \
  | head -n 400 \
  > "$ROUND_DIR/task_registry/registry_search.txt" || true

$PYTHON_BIN tools/stage2/list_task_candidates.py \
  --repo-root "$REPO_ROOT" \
  --output "$ROUND_DIR/task_registry/task_candidate_registry.csv" \
  2>&1 | tee "$ROUND_DIR/logs/list_task_candidates.log"
```

候选优先级：

1. 已有 checkpoint 和 rollout 命令；
2. 至少两个可独立完成的对象/子目标；
3. `A→B` 与 `B→A` 均满足相同终局成功条件；
4. 每步状态和完整 history 可保存；
5. 不需要重新训练 policy 才能启动。

### 4. 确定 branch 任务；没有现成任务时立即创建最小组合任务

Agent 根据 registry 选择：

- `TASK_RECOVERY_A`：优先从 `square`、`transport` 中选择 recovery 证据更充分者；
- `TASK_GRAPH_B`：现成的双物体/双子目标任务，或者新建最小组合任务。

写入：

```bash
cat > "$ROUND_DIR/configs/selected_task_plan.yaml" <<'YAML'
recovery_task:
  task_id: TO_BE_FILLED
  source: existing
  required_scenarios: [natural_success, natural_failure, recovery]
branch_task:
  task_id: TO_BE_FILLED
  source: existing_or_minimal_composite
  required_scenarios: [order_A_then_B, order_B_then_A, recovery, terminal_failure]
YAML
```

若仓库没有现成 branch task，不继续反复搜索。创建一个最小双子目标 wrapper：

- 复用现有可执行的对象操作环境和观测/action 空间；
- 放置两个对象 A/B 或为同一环境定义两个独立目标；
- 成功条件为 `done(A) AND done(B)`；
- 不对完成顺序加约束；
- observation 中增加 `subgoal_A_done`、`subgoal_B_done`；
- info 中记录 `subgoal_complete_events` 和完成顺序；
- 不修改 policy 架构；能用现有 policy/脚本分步执行则直接使用；若只需目标条件/对象配置变化，不启动新训练阶段。
- 若现有 learned policy 不能直接执行双子目标，优先采用“目标切换 + 现有单目标 policy”或仿真 scripted/oracle controller 采集演示，并在 manifest 中标记 `controller_source`；不要把 scripted 轨迹伪装成 learned-policy 结果。

新任务代码限定在当前任务所需的最小 wrapper，不建设通用任务图平台。

### 5. 恢复现有 rollout 命令和 checkpoint

从阶段 1 episode 源路径找到原运行目录：

```bash
FIRST_EP=$($PYTHON_BIN - <<'PY'
import json
p='artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl'
with open(p) as f:
    print(json.loads(next(f))['source_path'])
PY
)
RUN_DIR="$(dirname "$(dirname "$FIRST_EP")")"
echo "$RUN_DIR" | tee "$ROUND_DIR/task_registry/example_run_dir.txt"

find "$RUN_DIR" -maxdepth 4 -type f \
  \( -name '*.yaml' -o -name '*.yml' -o -name '*.json' -o -name '*.log' -o -name '*.sh' \
     -o -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' \) \
  | sort > "$ROUND_DIR/task_registry/example_run_files.txt"
```

在以下位置恢复精确 eval 命令：

- `$RUN_DIR/.hydra/config.yaml`、`overrides.yaml`；
- 运行日志；
- 项目已有 eval/rollout shell；
- 保存 `eval_save_episodes` 的代码入口。

Agent 创建 `tools/stage2/rollout_adapter.py`，只做参数适配：task、checkpoint、seed、episode 数、scenario、输出目录。不要重新实现整套 evaluator。

先对每个任务/场景跑 1 个 episode 的 smoke rollout，确认原始 episode、info、干预记录可写出。smoke 结果属于本轮，不另建大规模测试。

对新 branch task 的 smoke episode 运行 `inspect_episode_schema.py`，将真实字段补入 `configs/stage2/task_state_fields.yaml`；随后再启动批量补采，避免新任务事件字段缺失。

### 6. 实现定向恢复场景

创建 `tools/stage2/collect_graph_rollouts.py`，支持：

```text
--task TASK
--checkpoint PATH
--scenario natural_success|natural_failure|terminal_failure|drop_and_regrasp|gripper_reopen|
           object_displacement|order_A_then_B|order_B_then_A
--num-episodes N
--seed-start N
--output-dir PATH
--save-full-history
--save-intervention-log
--controller learned|scripted_oracle|hybrid
```

恢复场景优先使用仿真中的轻量干预：

- `drop_and_regrasp`：在稳定抓取后短暂打开夹爪或施加小幅对象位移，随后恢复 policy 控制；
- `gripper_reopen`：在抓取后触发一次打开动作，允许再次抓取；
- `object_displacement`：放置/运输途中将物体轻微移出当前路径，要求重新接近；
- 干预只发生一次，episode 必须继续运行到成功或终止；
- 每次干预记录 step、类型、幅度、干预前后 object/eef/gripper 状态；
- 不能只保存干预后的 clip，必须保存完整 episode。

branch 场景：

- `order_A_then_B`：先完成 A，再完成 B；
- `order_B_then_A`：先完成 B，再完成 A；
- 两条路径使用相同成功定义；
- path signature 由真实 `subgoal_complete_events` 生成。

### 7. 提权查看 GPU，并行启动独立 rollout jobs

```bash
bash tools/stage2/query_gpus.sh "$ROUND_DIR/gpu_query" "${MIN_FREE_MB:-12000}"
mapfile -t GPU_IDS < "$ROUND_DIR/gpu_query/available_gpu_ids.txt"
[ "${#GPU_IDS[@]}" -ge 1 ]
printf 'Using GPUs: %s\n' "${GPU_IDS[*]}"
echo "- gpu_ids: ${GPU_IDS[*]}" >> "$ROUND_DIR/run_manifest.md"
```

根据 2.1 缺口生成 `jobs.tsv`，每个 job 只负责一个 task/scenario/seed 范围。例如：

```text
job_id  task  scenario  num_episodes  seed_start  checkpoint
j00      TASK_RECOVERY_A  natural_success   12  21000  /path/to/checkpoint
j01      TASK_RECOVERY_A  drop_and_regrasp  10  22000  /path/to/checkpoint
j02      TASK_RECOVERY_A  terminal_failure   8  23000  /path/to/checkpoint
j03      TASK_GRAPH_B     order_A_then_B     10  24000  /path/to/checkpoint
j04      TASK_GRAPH_B     order_B_then_A     10  25000  /path/to/checkpoint
j05      TASK_GRAPH_B     drop_and_regrasp   10  26000  /path/to/checkpoint
j06      TASK_GRAPH_B     terminal_failure    8  27000  /path/to/checkpoint
```

并行 launcher 必须从 `jobs.tsv` 读取并分配 GPU；并发数不超过可用 GPU 数：

```bash
job_index=0
while IFS=$'\t' read -r job_id task scenario num_eps seed_start checkpoint; do
  [ "$job_id" = "job_id" ] && continue
  while [ "$(jobs -pr | wc -l)" -ge "${#GPU_IDS[@]}" ]; do
    wait -n || true
  done
  gpu="${GPU_IDS[$((job_index % ${#GPU_IDS[@]}))]}"
  job_dir="$ROUND_DIR/jobs/$job_id"
  mkdir -p "$job_dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" tools/stage2/collect_graph_rollouts.py \
    --task "$task" --checkpoint "$checkpoint" --scenario "$scenario" \
    --num-episodes "$num_eps" --seed-start "$seed_start" \
    --output-dir "$job_dir" --save-full-history --save-intervention-log \
    > "$ROUND_DIR/logs/$job_id.log" 2>&1 &
  job_index=$((job_index+1))
done < "$ROUND_DIR/jobs.tsv"
wait
```

### 8. 合并新旧 episode，并重新跑真实事件提议

实现 `tools/stage2/merge_collection_manifests.py`，生成：

```text
collection_manifests/stage2_episode_manifest_v0.2.jsonl
collection_manifests/stage2_splits_v0.2.csv
collection_manifests/interventions.jsonl
```

新数据 split 必须按 task instance/seed/episode 分组；同一 episode 不跨 split。保留 stage1 原 split，新增 episode 使用固定 seed 追加划分。

随后只对新增数据运行 2.1 的真实事件提议器，并合并结果：

```bash
$PYTHON_BIN tools/stage2/mine_semantic_events.py \
  --manifest "$ROUND_DIR/collection_manifests/stage2_episode_manifest_v0.2.jsonl" \
  --field-map "$REPO_ROOT/configs/stage2/task_state_fields.yaml" \
  --episode-list "$ROUND_DIR/collection_manifests/new_episode_ids.txt" \
  --output-dir "$ROUND_DIR/new_event_candidates" \
  --workers 16 \
  2>&1 | tee "$ROUND_DIR/logs/mine_new_events.log"
```

### 9. 最终任务门控与选择

创建 `tools/stage2/select_stage2_tasks.py`，按以下硬条件选择 2 个任务：

- 每个任务至少有 12 个 forward success；
- 每个任务至少满足其一：
  - 两条合法成功路径，每条至少 10 个 episode；
  - 至少 10 个有明确 onset/complete 的 recovery episode；
- 整体至少一个 task 覆盖 alternative order；
- 整体至少一个 task 覆盖 recovery；
- 完整 episode history 可读；
- 事件证据不是仅由最终 outcome 推断。

输出：

```text
selected_graph_tasks_v1.yaml
task_evidence_table.csv
collection_summary.md
```

`selected_graph_tasks_v1.yaml` 还要指定固定链对照任务。若 `square/transport` 不满足 graph task，可保留为 Stage 3 固定链对照，不强行纳入主任务。

## 本小阶段最小完成检查

```bash
test -s "$ROUND_DIR/selected_graph_tasks_v1.yaml"
test -s "$ROUND_DIR/task_evidence_table.csv"
test -s "$ROUND_DIR/collection_manifests/stage2_episode_manifest_v0.2.jsonl"

$PYTHON_BIN tools/stage2/select_stage2_tasks.py \
  --check-only \
  --selection "$ROUND_DIR/selected_graph_tasks_v1.yaml" \
  --evidence "$ROUND_DIR/task_evidence_table.csv"
```

不要安排额外泛化测试。本轮只回答：任务是否真实覆盖路径分支/恢复，以及数据是否足以进入人工图与标注。

## 本轮 ZIP

`summary.md` 写明新增 episode 数、每个场景成功数、两条路径计数、recovery 计数、最终选定任务、checkpoint 路径清单。checkpoint、原始 `.pkl` 和视频不入 ZIP。

```bash
test -s "$ROUND_DIR/collection_summary.md"
cp "$ROUND_DIR/collection_summary.md" "$ROUND_DIR/summary.md"
bash tools/stage2/package_round.sh "$ROUND_ID" "$ROUND_DIR"
```

返回：

```text
$STAGE2_ROOT/downloads/stage2_2_targeted_collection.zip
```

---

# 阶段 2.3：任务边界与人工 Graph spec v1

> 执行前置：先完成 `00_阶段1关闭结论与阶段2总览.md` 中的通用目录、`stage2.yaml` 和 `tools/stage2/package_round.sh` 初始化。若这些文件已经存在，直接进入本小阶段，不重复初始化。


## 总体上要干什么

把阶段 2.2 最终选定的两个任务转化为可执行、可标注、可映射的人工任务图。每个任务优先保持 5-10 个语义节点，把视觉上难以区分的细节留给节点内进度；明确 forward、alternative、failure、recovery、stagnation 和是否允许重复。

本小阶段不做自动图发现，不训练模型。

## 直接给 Agent 的执行命令

### 1. 初始化

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export ROUND_ID="stage2_3_graph_spec_v1"
export ROUND_DIR="$STAGE2_ROOT/rounds/$ROUND_ID"
export TASK_SELECTION="$STAGE2_ROOT/rounds/stage2_2_targeted_collection/selected_graph_tasks_v1.yaml"
cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"/{configs,graphs,logs,metrics,plots,mappings}
test -s "$TASK_SELECTION"
cp "$TASK_SELECTION" "$ROUND_DIR/configs/selected_graph_tasks_v1.yaml"
bash tools/stage2/init_round.sh \
  "$ROUND_ID" "$ROUND_DIR" \
  "define task semantics and freeze manual Graph spec v1"
```

### 2. 创建 Graph spec schema

创建 `configs/stage2/graph_spec.schema.json`，要求每个 Graph YAML 含：

```text
graph_id, version, task_id, description,
start_node, success_nodes, terminal_failure_nodes,
nodes[], edges[], path_templates[], history_policy, progress_policy
```

node 字段：

```text
id, name, description, terminal, observable_conditions,
entry_condition, exit_condition, history_required,
within_node_progress_signal, allowed_attempt_groups
```

edge 字段：

```text
id, src, dst, type, description, guard_condition,
completion_condition, repeatable, attempt_group,
base_step_cost, max_repeat_before_stagnation
```

`type` 只允许：

```text
forward, alternative, failure, recovery, stagnation
```

### 3. 为每个任务编写 5-10 节点 Graph YAML

读取：

- `selected_graph_tasks_v1.yaml`；
- 2.1/2.2 的真实 events；
- 每个 path/recovery 的 contact sheet；
- 任务环境中的真实 success 条件。

在 `$ROUND_DIR/graphs/<task_id>_graph_v1.yaml` 编写图。节点命名以语义状态为主，避免把每个连续动作拆成独立 node。

#### 恢复型单物体任务的推荐骨架

仅作为骨架，Agent 必须用实际 task/state 字段替换：

```yaml
graph_id: <task>_graph
version: 1.0.0
task_id: <task>
start_node: start
success_nodes: [success]
terminal_failure_nodes: [terminal_failure]
nodes:
  - id: start
    description: object not yet secured
    terminal: false
    history_required: false
    within_node_progress_signal: negative_eef_object_distance
  - id: grasped
    description: stable grasp established
    terminal: false
    history_required: true
    within_node_progress_signal: grasp_stability
  - id: in_transit
    description: object secured and moving toward target
    terminal: false
    history_required: true
    within_node_progress_signal: negative_object_target_distance
  - id: placed
    description: object inside target region before stable release
    terminal: false
    history_required: true
    within_node_progress_signal: placement_stability
  - id: dropped_or_misaligned
    description: previously secured object lost or valid placement invalidated
    terminal: false
    history_required: true
    within_node_progress_signal: negative_eef_object_distance
  - id: success
    terminal: true
    history_required: true
  - id: terminal_failure
    terminal: true
    history_required: true
edges:
  - {id: start_to_grasped, src: start, dst: grasped, type: forward, repeatable: true, attempt_group: grasp}
  - {id: grasped_to_transit, src: grasped, dst: in_transit, type: forward, repeatable: true, attempt_group: transport}
  - {id: transit_to_placed, src: in_transit, dst: placed, type: forward, repeatable: true, attempt_group: place}
  - {id: placed_to_success, src: placed, dst: success, type: forward, repeatable: false, attempt_group: release}
  - {id: grasped_to_dropped, src: grasped, dst: dropped_or_misaligned, type: failure, repeatable: true, attempt_group: grasp}
  - {id: transit_to_dropped, src: in_transit, dst: dropped_or_misaligned, type: failure, repeatable: true, attempt_group: transport}
  - {id: dropped_to_grasped, src: dropped_or_misaligned, dst: grasped, type: recovery, repeatable: true, attempt_group: grasp}
  - {id: repeated_no_progress, src: dropped_or_misaligned, dst: dropped_or_misaligned, type: stagnation, repeatable: true, attempt_group: recovery}
```

#### 双子目标 alternative-order 任务的推荐骨架

图中必须显式允许：

```text
start -> A_done -> B_done -> success
start -> B_done -> A_done -> success
```

不要为两条路径复制完全相同的视觉节点；推荐用 `completed_subgoal_set` 作为历史状态条件，并为正在操作的对象记录 `active_target`。failure/recovery edge 可以从 `A_in_progress/B_in_progress` 返回相应的 recovery node。

### 4. 明确任务边界和判定条件

每个 task 另写 `$ROUND_DIR/graphs/<task_id>_task_semantics.md`，逐项给出：

- 初始状态；
- 成功终点；
- terminal failure；
- partial completion；
- 可恢复 failure；
- 两条或多条合法顺序；
- 不合法顺序或无效循环；
- 每个 node/edge 依赖的原始状态字段；
- 哪些 node 必须查看历史才能区分。

判定条件必须可落到状态字段或视觉证据，不使用“看起来差不多”作为唯一规则。

### 5. 实现图结构检查器

创建 `tools/stage2/validate_graph_spec.py`，只做推进后续所需的最小检查：

1. node/edge ID 唯一；
2. edge 两端 node 存在；
3. start 可到达 success；
4. edge type 合法；
5. graph task 的关键类型在图中存在；
6. `history_required` 与语义说明一致；
7. 至少一条失败→恢复→合法前进路径；
8. branch task 至少有两条合法 path template。

```bash
for graph in "$ROUND_DIR"/graphs/*_graph_v1.yaml; do
  $PYTHON_BIN tools/stage2/validate_graph_spec.py \
    --graph "$graph" \
    --schema "$REPO_ROOT/configs/stage2/graph_spec.schema.json" \
    --report "$ROUND_DIR/metrics/$(basename "${graph%.yaml}")_validation.json"
done
```

### 6. 将真实事件映射为 node/edge 序列

创建 `tools/stage2/map_events_to_graph.py`：

- 输入 2.1/2.2 events 和 Graph YAML；
- 输出每个 episode 的候选 node intervals、edge intervals、path signature；
- 无法映射的区间标记 `unmapped` 并保留原因；
- 不强迫每一帧进入某 node；短暂过渡可以属于 edge interval；
- history-required node 的映射必须使用 episode 历史和已完成子目标集合。

运行：

```bash
$PYTHON_BIN tools/stage2/map_events_to_graph.py \
  --manifest "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/collection_manifests/stage2_episode_manifest_v0.2.jsonl" \
  --events "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/merged_events.jsonl" \
  --graph-dir "$ROUND_DIR/graphs" \
  --output-dir "$ROUND_DIR/mappings" \
  2>&1 | tee "$ROUND_DIR/logs/map_events_to_graph.log"
```

若 2.2 的 events 文件名不同，Agent 使用实际生成路径并在 `run_manifest.md` 记录，不复制一份虚构路径。

### 7. 生成图示

优先用 Graphviz：

```bash
$PYTHON_BIN tools/stage2/render_graphs.py \
  --graph-dir "$ROUND_DIR/graphs" \
  --output-dir "$ROUND_DIR/plots"
```

输出 SVG/PNG，边标签显示 edge type、repeatable 和 attempt group。

## 本小阶段最小完成检查

- 每个选定任务 5-10 个主要语义 node；
- 每个 graph 都能从 start 到 success；
- branch task 有两条合法路径；
- recovery task 有 failure→recovery→forward；
- 关键 evidence episode 至少 95% 可映射到 node/edge 序列；其余进入 annotation queue；
- 不根据 test 结果反复改图，本轮形成 `v1.0.0`。

```bash
$PYTHON_BIN tools/stage2/summarize_graph_mapping.py \
  --mapping-dir "$ROUND_DIR/mappings" \
  --output-csv "$ROUND_DIR/metrics/graph_mapping_summary.csv" \
  --output-md "$ROUND_DIR/graph_spec_summary.md"
```

## 本轮 ZIP

```bash
test -s "$ROUND_DIR/graph_spec_summary.md"
cp "$ROUND_DIR/graph_spec_summary.md" "$ROUND_DIR/summary.md"
bash tools/stage2/package_round.sh "$ROUND_ID" "$ROUND_DIR"
```

返回：

```text
$STAGE2_ROOT/downloads/stage2_3_graph_spec_v1.zip
```

---

# 阶段 2.4：标注协议、自动提议与审阅工具

> 执行前置：先完成 `00_阶段1关闭结论与阶段2总览.md` 中的通用目录、`stage2.yaml` 和 `tools/stage2/package_round.sh` 初始化。若这些文件已经存在，直接进入本小阶段，不重复初始化。


## 总体上要干什么

建立 Agent 可直接执行的 node/edge 标注格式和审阅工具。用 2.3 的图映射作为初始提议，再依据低维状态和可视化确认关键边界。重点标注 failure onset、recovery complete、attempt、revisit 和合法路径顺序；不要求逐帧手工画连续 progress，而使用少量锚点插值。

## 直接给 Agent 的执行命令

### 1. 初始化

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export ROUND_ID="stage2_4_annotation_tooling"
export ROUND_DIR="$STAGE2_ROOT/rounds/$ROUND_ID"
export GRAPH_DIR="$STAGE2_ROOT/rounds/stage2_3_graph_spec_v1/graphs"
cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"/{configs,logs,schemas,templates,examples,review_bundles,metrics}
cp -a "$GRAPH_DIR" "$ROUND_DIR/configs/graphs"
bash tools/stage2/init_round.sh \
  "$ROUND_ID" "$ROUND_DIR" \
  "build annotation protocol, proposals, and review tooling"
```

### 2. 创建 annotation schema

创建 `configs/stage2/annotation.schema.json`。每个 episode annotation 的逻辑结构为：

```json
{
  "episode_id": "...",
  "task_id": "...",
  "graph_id": "...",
  "graph_version": "1.0.0",
  "source_path": "...",
  "path_signature": ["subgoal_A", "subgoal_B"],
  "outcome": "success|partial_success|failure",
  "node_intervals": [
    {
      "node_id": "...",
      "start_step": 0,
      "end_step": 42,
      "history_required": false,
      "evidence": ["state:key", "frame:42"]
    }
  ],
  "edge_intervals": [
    {
      "edge_id": "...",
      "edge_type": "forward|alternative|failure|recovery|stagnation",
      "start_step": 43,
      "end_step": 58,
      "attempt_index": 1,
      "evidence": ["state:key", "frames:43-58"]
    }
  ],
  "progress_anchors": [
    {"node_id": "...", "step": 43, "value": 0.0},
    {"node_id": "...", "step": 58, "value": 1.0}
  ],
  "failure_events": [
    {"failure_onset_step": 80, "failure_type": "drop", "recoverable": true}
  ],
  "recovery_events": [
    {"recovery_start_step": 81, "recovery_complete_step": 112, "restored_node": "grasped"}
  ],
  "review": {
    "status": "proposed|accepted|edited|ambiguous",
    "reviewer": "agent_or_human_id",
    "notes": ""
  }
}
```

### 3. 写标注手册 v1

创建 `$ROUND_DIR/annotation_manual_v1.md`，至少写清以下规则：

#### Node interval

- node 表示持续的语义状态，不是瞬时动作；
- `start_step` 是该状态首次成立且达到稳定帧数的时刻；
- `end_step` 是状态最后成立的时刻；
- 过渡过程优先标为 edge interval，不强行塞入相邻 node；
- 同一视觉状态因已完成子目标不同而语义不同，使用历史和 `completed_subgoal_set` 判断。

#### Edge interval

- edge 从源 node 条件开始失效或动作明确启动时开始；
- 在目标 node 条件稳定成立时结束；
- `alternative` 是合法成功顺序中的分支，不是任意偏离；
- `failure` 是有效任务状态被破坏；
- `recovery` 必须发生在某个 failure 后，并恢复到可继续前进的有效 node；
- `stagnation` 是重复动作/状态持续但没有有效进展。

#### Failure 与 recovery

- `failure_onset_step`：最早可由状态/帧证明失败发生的时刻；
- 禁止用 episode 中点或最终失败标签倒推 onset；
- `recovery_start_step`：failure 后重新建立有效策略动作的时刻；
- `recovery_complete_step`：被破坏的 node 条件重新稳定成立，或进入后续合法 node 的首个稳定时刻；
- failure+recovery 是一对事件；若未恢复，`recovery_events=[]`。

#### Attempt / revisit

- 同一 `attempt_group` 每重新开始一次，`attempt_index += 1`；
- 进入曾经离开的 node，标记 revisit；
- 只因视觉抖动在相邻帧切换，不计新 attempt，使用稳定帧规则去抖。

#### Within-node progress

- 每个 node 只标 2-5 个锚点：`0, 0.25, 0.5, 0.75, 1.0`；
- 锚点必须对应 graph spec 中的 progress signal，例如距离下降、抓取稳定度、放置稳定度；
- 节点内帧由后续脚本在锚点间插值；
- failure 后进入 recovery node 时重新开始该 node 的 progress，不沿用之前的最高值。

### 4. 实现 annotation queue 生成器

创建 `tools/stage2/make_annotation_queue.py`，输入 Graph mapping 和任务证据，按以下优先级选 episode：

1. 两条合法 path 的 success；
2. 有明确 failure→recovery；
3. terminal failure；
4. retry/revisit/stagnation；
5. 普通 forward success。

输出：

```text
annotation_queue.csv
```

字段：

```text
queue_id,task_id,episode_id,split,source_path,category,path_signature,
priority,proposal_path,review_bundle_path,status
```

### 5. 实现自动提议初始化器

创建 `tools/stage2/initialize_annotations.py`：

- 从 2.3 mappings 和 events 生成 `proposed_annotations/<episode_id>.json`；
- 只使用有 evidence 的边界；
- 不确定区间标为 `ambiguous`；
- 写入 graph/version/source hash；
- 不覆盖已人工编辑文件。

```bash
$PYTHON_BIN tools/stage2/make_annotation_queue.py \
  --selection "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/selected_graph_tasks_v1.yaml" \
  --mapping-dir "$STAGE2_ROOT/rounds/stage2_3_graph_spec_v1/mappings" \
  --output "$ROUND_DIR/annotation_queue.csv"

$PYTHON_BIN tools/stage2/initialize_annotations.py \
  --queue "$ROUND_DIR/annotation_queue.csv" \
  --graph-dir "$GRAPH_DIR" \
  --mapping-dir "$STAGE2_ROOT/rounds/stage2_3_graph_spec_v1/mappings" \
  --schema "$REPO_ROOT/configs/stage2/annotation.schema.json" \
  --output-dir "$ROUND_DIR/proposed_annotations" \
  2>&1 | tee "$ROUND_DIR/logs/initialize_annotations.log"
```

### 6. 实现 episode review bundle

创建 `tools/stage2/render_annotation_bundle.py`，每个 episode 输出一个轻量目录：

```text
review_bundles/<episode_id>/
  overview.png
  event_timeline.png
  keyframes/
  state_trace.csv
  proposal.json
  review_notes.md
```

`overview.png` 显示均匀采样帧；`event_timeline.png` 叠加 node/edge/failure/recovery；`state_trace.csv` 只保存关键状态，不复制原始大数组。完整视频可留在本地并写入 `large_file_manifest.tsv`，不进 ZIP。

CPU 解码可用多进程。若使用视觉编码器生成事件候选，先提权查看 GPU，并按 episode 分片并行。

### 7. 提供简单的审阅/编辑命令

创建 `tools/stage2/review_annotation.py`，支持：

```text
--annotation FILE
--set-node NODE START END
--set-edge EDGE_TYPE EDGE_ID START END ATTEMPT
--add-failure TYPE ONSET RECOVERABLE
--add-recovery START COMPLETE RESTORED_NODE
--add-progress NODE STEP VALUE
--status accepted|edited|ambiguous
--note TEXT
```

每次修改保存历史到 `annotation_history.jsonl`。不需要开发 GUI；命令行和 review bundle 足以推进当前阶段。

### 8. 最小标注示例

每个选定 task 至少完成：

- 2 个普通 success；
- 每条 alternative path 各 2 个；
- 2 个 recovery；
- 2 个 terminal failure。

把接受后的示例放入 `$ROUND_DIR/examples/accepted/`，并在 `annotation_examples.md` 解释关键边界。

### 9. 实现最小一致性检查

创建 `tools/stage2/validate_annotations.py`，只检查：

- step 范围合法、区间不反向；
- node/edge ID 存在于对应 graph；
- recovery 前必须有 failure；
- attempt index 非递减；
- progress 值在 `[0,1]`；
- success episode 最终到达 success node；
- path signature 与实际 subgoal completion order 一致。

```bash
$PYTHON_BIN tools/stage2/validate_annotations.py \
  --annotation-dir "$ROUND_DIR/examples/accepted" \
  --graph-dir "$GRAPH_DIR" \
  --schema "$REPO_ROOT/configs/stage2/annotation.schema.json" \
  --report "$ROUND_DIR/metrics/example_validation.json"
```

## 本小阶段完成条件

- schema、手册、queue、自动提议、review bundle、CLI 均可用；
- 每个 task 的关键类别有接受示例；
- 不要求在本小阶段标完全部 GT；
- 不做无关 UI、权限系统、数据库或通用标注平台。

## 本轮 ZIP

创建 `tools/stage2/summarize_annotation_tooling.py`，汇总 queue 数、已接受示例数、各 edge 类型示例数和 ambiguous 数，然后：

```bash
$PYTHON_BIN tools/stage2/summarize_annotation_tooling.py \
  --queue "$ROUND_DIR/annotation_queue.csv" \
  --example-dir "$ROUND_DIR/examples/accepted" \
  --output "$ROUND_DIR/summary.md"
bash tools/stage2/package_round.sh "$ROUND_ID" "$ROUND_DIR"
```

返回：

```text
$STAGE2_ROOT/downloads/stage2_4_annotation_tooling.zip
```

---

# 阶段 2.5：GT 子集构建与最小一致性复核

> 执行前置：先完成 `00_阶段1关闭结论与阶段2总览.md` 中的通用目录、`stage2.yaml` 和 `tools/stage2/package_round.sh` 初始化。若这些文件已经存在，直接进入本小阶段，不重复初始化。


## 总体上要干什么

使用阶段 2.4 的标注工具，完成可用于 node/edge/remaining-cost 监督和后续评测的 Ground Truth 子集。采样要覆盖路径、恢复和失败，不把全部 200+ episode 都标完；优先达到每个关键 edge 的最低实例数。

## 直接给 Agent 的执行命令

### 1. 初始化

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export ROUND_ID="stage2_5_gt_subset_v1"
export ROUND_DIR="$STAGE2_ROOT/rounds/$ROUND_ID"
export GRAPH_DIR="$STAGE2_ROOT/rounds/stage2_3_graph_spec_v1/graphs"
export TOOLING_DIR="$STAGE2_ROOT/rounds/stage2_4_annotation_tooling"
cd "$REPO_ROOT"
mkdir -p "$ROUND_DIR"/{configs,logs,queues,annotations,gt_v1,metrics,plots,review}
cp -a "$GRAPH_DIR" "$ROUND_DIR/configs/graphs"
cp "$TOOLING_DIR/annotation_manual_v1.md" "$ROUND_DIR/configs/"
bash tools/stage2/init_round.sh \
  "$ROUND_ID" "$ROUND_DIR" \
  "construct balanced node-edge ground-truth subset v1"
```

### 2. 生成平衡 GT 队列

创建 `tools/stage2/sample_gt_subset.py`，优先按以下最低配额采样。对每个 graph task：

- 普通 forward success：至少 12 个；
- terminal failure/stagnation：至少 8 个；
- recovery：至少 10 个；
- branch task 的每条合法 success path：至少 8 个；
- 总计目标：每 task 至少 40 个 episode；有重叠时不重复计算 episode。

采样要求：

- 按 episode/group 划分，不能拆 clip；
- 保留 stage1 已有 split，新增 episode 按固定 seed 分配；
- train/val/test 约为 60/20/20；
- 每个 split 尽量覆盖关键 edge；无法覆盖时在 `coverage_gap.csv` 明确记录；
- test 中保留未在 train 出现的 seed，且至少含 recovery 与 alternative path。

```bash
$PYTHON_BIN tools/stage2/sample_gt_subset.py \
  --queue "$TOOLING_DIR/annotation_queue.csv" \
  --selection "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/selected_graph_tasks_v1.yaml" \
  --graph-dir "$GRAPH_DIR" \
  --min-per-task 40 \
  --min-recovery 10 \
  --min-failure 8 \
  --min-forward 12 \
  --min-per-path 8 \
  --split-seed 20260831 \
  --output "$ROUND_DIR/queues/gt_annotation_queue.csv" \
  --coverage-plan "$ROUND_DIR/queues/gt_coverage_plan.csv"
```

### 3. 批量初始化提议和 review bundles

```bash
$PYTHON_BIN tools/stage2/initialize_annotations.py \
  --queue "$ROUND_DIR/queues/gt_annotation_queue.csv" \
  --graph-dir "$GRAPH_DIR" \
  --mapping-dir "$STAGE2_ROOT/rounds/stage2_3_graph_spec_v1/mappings" \
  --schema "$REPO_ROOT/configs/stage2/annotation.schema.json" \
  --output-dir "$ROUND_DIR/annotations/proposed" \
  2>&1 | tee "$ROUND_DIR/logs/init_gt_proposals.log"

$PYTHON_BIN tools/stage2/render_annotation_bundle.py \
  --queue "$ROUND_DIR/queues/gt_annotation_queue.csv" \
  --annotation-dir "$ROUND_DIR/annotations/proposed" \
  --output-dir "$ROUND_DIR/review/bundles" \
  --workers 16 \
  2>&1 | tee "$ROUND_DIR/logs/render_gt_bundles.log"
```

如视觉处理用 GPU，先提权查看 GPU，并把 queue 切成多个 shard，每个 shard 占一张 GPU。只要可并行，不串行处理所有 episode。

### 4. 执行标注

Agent 按 queue 优先级逐项处理：

1. 打开 review bundle；
2. 对照 raw state trace 和关键帧；
3. 接受或修正 node intervals；
4. 接受或修正 edge intervals；
5. 明确 failure onset/recovery complete；
6. 填 attempt/revisit/path signature；
7. 添加 2-5 个 within-node progress anchors；
8. 设置 `status=accepted` 或 `ambiguous`；
9. 保存到 `$ROUND_DIR/annotations/accepted/<episode_id>.json`。

可以把互不依赖的 episode 队列分给多个 Agent/进程并行处理，但同一 episode 只允许一个主文件；复核版本写入独立目录，最后统一 merge。先按执行单元数切分 queue：

```bash
NUM_SHARDS="${NUM_SHARDS:-4}"
$PYTHON_BIN tools/stage2/split_annotation_queue.py \
  --queue "$ROUND_DIR/queues/gt_annotation_queue.csv" \
  --num-shards "$NUM_SHARDS" \
  --output-dir "$ROUND_DIR/queues/shards"
```

多个 Agent 分别领取一个 shard；禁止两个 Agent 同时写同一 `episode_id`。

### 5. 只复核必要样本

为避免拖慢进度，不做全量双标。复核范围：

- 全部 `ambiguous` episode；
- 全部新 edge 类型的首 2 个实例；
- 其余 accepted 中固定随机抽取 10%。

第二份标注写入 `$ROUND_DIR/annotations/reviewed/`。比较：

- node interval overlap；
- edge type 一致性；
- failure/recovery 边界误差（帧数）；
- path signature 一致性。

差异超出容忍度的样本直接合并为 adjudicated 版本，不开展额外标注研究。

推荐容忍度：

```yaml
node_boundary_tolerance_frames: 5
edge_boundary_tolerance_frames: 5
failure_recovery_tolerance_frames: 8
progress_anchor_value_tolerance: 0.25
```

### 6. 合并并生成训练/评测格式

创建：

- `tools/stage2/merge_reviewed_annotations.py`；
- `tools/stage2/build_gt_tables.py`。

输出 `$ROUND_DIR/gt_v1/`：

```text
episode_annotations.jsonl
node_intervals.csv
edge_intervals.csv
progress_anchors.csv
failure_recovery_events.csv
gt_episode_manifest.jsonl
gt_splits.csv
label_stats.csv
coverage_by_node_edge.csv
annotation_provenance.csv
```

`annotation_provenance.csv` 必须记录该标签来自：state rule、visual review、intervention log 或人工修正。任何阶段 1 占位标签不得出现在 provenance 中。

```bash
$PYTHON_BIN tools/stage2/merge_reviewed_annotations.py \
  --primary-dir "$ROUND_DIR/annotations/accepted" \
  --review-dir "$ROUND_DIR/annotations/reviewed" \
  --output-dir "$ROUND_DIR/annotations/final" \
  --report "$ROUND_DIR/metrics/review_summary.json"

$PYTHON_BIN tools/stage2/build_gt_tables.py \
  --annotation-dir "$ROUND_DIR/annotations/final" \
  --graph-dir "$GRAPH_DIR" \
  --output-dir "$ROUND_DIR/gt_v1" \
  2>&1 | tee "$ROUND_DIR/logs/build_gt_tables.log"
```

### 7. 最小验收统计

```bash
$PYTHON_BIN tools/stage2/validate_annotations.py \
  --annotation-dir "$ROUND_DIR/annotations/final" \
  --graph-dir "$GRAPH_DIR" \
  --schema "$REPO_ROOT/configs/stage2/annotation.schema.json" \
  --report "$ROUND_DIR/metrics/final_annotation_validation.json"

$PYTHON_BIN tools/stage2/summarize_gt_coverage.py \
  --gt-dir "$ROUND_DIR/gt_v1" \
  --selection "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/selected_graph_tasks_v1.yaml" \
  --output-csv "$ROUND_DIR/metrics/gt_coverage_summary.csv" \
  --output-md "$ROUND_DIR/gt_v1_summary.md"
```

完成条件：

- 每个 graph task 至少 40 个 GT episode，或虽略少但每个关键 edge 均达到 8 个实例；
- 整体至少两条合法路径，每条 ≥8 个 GT episode；
- 至少 10 个 recovery GT episode；
- failure onset 与 recovery complete 均为真实边界；
- split 不跨 episode/group；
- 不要求在此阶段训练任何模型。

## 本轮 ZIP

完整视频、原始 episode 不入 ZIP；GT JSON/CSV、review summary、少量 PNG 样例可入 ZIP。

```bash
test -s "$ROUND_DIR/gt_v1_summary.md"
cp "$ROUND_DIR/gt_v1_summary.md" "$ROUND_DIR/summary.md"
bash tools/stage2/package_round.sh "$ROUND_ID" "$ROUND_DIR"
```

返回：

```text
$STAGE2_ROOT/downloads/stage2_5_gt_subset_v1.zip
```

---

# 阶段 2.6：图与数据冻结、M1 决策和 Stage 3 交接

> 执行前置：先完成 `00_阶段1关闭结论与阶段2总览.md` 中的通用目录、`stage2.yaml` 和 `tools/stage2/package_round.sh` 初始化。若这些文件已经存在，直接进入本小阶段，不重复初始化。


## 总体上要干什么

冻结阶段 2 的 Graph spec、标注手册、GT 子集、任务选择和数据 manifest，形成唯一的 M1 版本。完成后 Stage 3 可以直接复现线性 SARM/阶段基线，并验证其是否对多路径和恢复轨迹系统性误评分。

## 直接给 Agent 的执行命令

### 1. 初始化冻结目录

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage2"
export ROUND_ID="stage2_6_m1_freeze"
export ROUND_DIR="$STAGE2_ROOT/rounds/$ROUND_ID"
export FREEZE_DIR="$STAGE2_ROOT/m1_freeze_v1"
cd "$REPO_ROOT"
rm -rf "$FREEZE_DIR"
mkdir -p "$ROUND_DIR"/{logs,metrics,configs} "$FREEZE_DIR"
bash tools/stage2/init_round.sh \
  "$ROUND_ID" "$ROUND_DIR" \
  "freeze Graph spec, GT v1, and produce M1/Stage3 handoff"
```

### 2. 汇总唯一版本

复制以下轻量文件：

```bash
cp -a "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/selected_graph_tasks_v1.yaml" "$FREEZE_DIR/"
cp -a "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/task_evidence_table.csv" "$FREEZE_DIR/"
cp -a "$STAGE2_ROOT/rounds/stage2_2_targeted_collection/collection_manifests" "$FREEZE_DIR/"
cp -a "$STAGE2_ROOT/rounds/stage2_3_graph_spec_v1/graphs" "$FREEZE_DIR/graph_specs_v1"
cp -a "$STAGE2_ROOT/rounds/stage2_4_annotation_tooling/annotation_manual_v1.md" "$FREEZE_DIR/"
cp -a "$REPO_ROOT/configs/stage2/annotation.schema.json" "$FREEZE_DIR/"
cp -a "$REPO_ROOT/configs/stage2/graph_spec.schema.json" "$FREEZE_DIR/"
cp -a "$STAGE2_ROOT/rounds/stage2_5_gt_subset_v1/gt_v1" "$FREEZE_DIR/"
cp -a "$STAGE2_ROOT/rounds/stage2_5_gt_subset_v1/metrics" "$FREEZE_DIR/stage2_metrics"
```

原始 `.pkl`、完整视频和 checkpoint 不复制；只保留 manifest 中的绝对/相对路径和 hash。

### 3. 生成冻结 manifest 与 hash

```bash
$PYTHON_BIN - <<'PY'
from pathlib import Path
import json, datetime
root=Path('artifacts/pathgraph_sarm/stage2/m1_freeze_v1')
files=[]
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name != 'M1_SHA256SUMS.txt':
        files.append({'path':str(p.relative_to(root)),'size_bytes':p.stat().st_size})
(root/'m1_freeze_manifest.json').write_text(json.dumps({
    'version':'pathgraph_sarm_m1_v1',
    'generated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'files':files,
},ensure_ascii=False,indent=2),encoding='utf-8')
PY

find "$FREEZE_DIR" -type f ! -name 'M1_SHA256SUMS.txt' -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$FREEZE_DIR/M1_SHA256SUMS.txt"
```

### 4. 生成 M1 决策

创建 `tools/stage2/evaluate_m1_gate.py`，只依据冻结数据判断：

1. graph-valid task 数 ≥2；
2. 每个 task 满足多路径或 recovery 至少一种结构；
3. 整体至少 1 个 alternative-order task；
4. 整体至少 1 个 recovery task；
5. 每条目标合法路径 GT ≥8；
6. recovery GT ≥10；
7. 每个关键 edge GT ≥8，或在缺口表中有明确原因；
8. Graph spec 验证通过；
9. GT 中没有阶段 1 占位事件来源；
10. train/val/test 没有 episode/group 跨 split。

```bash
$PYTHON_BIN tools/stage2/evaluate_m1_gate.py \
  --freeze-dir "$FREEZE_DIR" \
  --output-json "$ROUND_DIR/metrics/m1_gate.json" \
  --output-md "$ROUND_DIR/m1_decision.md"

cat "$ROUND_DIR/m1_decision.md"
```

决策值：

- `GO_STAGE3`：可以进入线性 SARM 误评分验证；
- `COLLECT_MORE`：只针对明确缺失的 path/edge 再补采，回到 2.2 对应 job，不重做已冻结前的所有步骤；
- `REPLACE_TASK`：某任务本质单链且无法产生 recovery，替换该任务，不强行造 graph。

### 5. 写 Stage 3 handoff

创建 `$FREEZE_DIR/stage3_handoff.md`，必须列出：

- 选定 graph tasks 和固定链 control task；
- Graph spec 路径；
- GT manifest/split 路径；
- 线性 SARM、linear stage、stage-transition baseline 需要使用的同一 episode 列表；
- 多路径误评分评测集；
- recovery 误评分评测集；
- Stage 3 禁止修改的冻结文件；
- Stage 3 可新建的输出目录。

Stage 3 的第一项实验应直接比较：同一合法终局下不同 path 的累计分数，以及 failure→recovery 片段在线性进度中的错误负分/排序。

### 6. 生成阶段 2 总摘要

`$ROUND_DIR/stage2_summary.md` 至少包含：

- 阶段 1 的 `SWITCH` 如何被处理；
- 新挖掘和新补采 episode 数；
- 最终任务与 graph 节点/edge 数；
- GT 数量及 path/recovery/failure 覆盖；
- M1 决策；
- 未打包大文件路径；
- Stage 3 入口。

### 7. 打包本轮和阶段总包

```bash
test -s "$ROUND_DIR/stage2_summary.md"
cp "$ROUND_DIR/stage2_summary.md" "$ROUND_DIR/summary.md"
bash tools/stage2/package_round.sh "$ROUND_ID" "$ROUND_DIR"
```

然后创建总交付目录，只复制轻量必要文件：

```bash
STAGE2_COMPLETE_DIR="$STAGE2_ROOT/stage2_complete_light"
rm -rf "$STAGE2_COMPLETE_DIR"
mkdir -p "$STAGE2_COMPLETE_DIR"
cp -a "$FREEZE_DIR" "$STAGE2_COMPLETE_DIR/m1_freeze_v1"
cp -a "$ROUND_DIR/m1_decision.md" "$STAGE2_COMPLETE_DIR/"
cp -a "$ROUND_DIR/stage2_summary.md" "$STAGE2_COMPLETE_DIR/"
cp -a "$STAGE2_ROOT/downloads/index.md" "$STAGE2_COMPLETE_DIR/round_zip_index.md"
cp -a "$REPO_ROOT/tools/stage2" "$STAGE2_COMPLETE_DIR/tools_stage2"
cp -a "$REPO_ROOT/configs/stage2" "$STAGE2_COMPLETE_DIR/configs_stage2"

# package_round.sh 的排除规则会自动跳过 checkpoint、原始数据、视频和大文件。
bash tools/stage2/package_round.sh "stage2_complete" "$STAGE2_COMPLETE_DIR"
```

返回两个路径：

```text
$STAGE2_ROOT/downloads/stage2_6_m1_freeze.zip
$STAGE2_ROOT/downloads/stage2_complete.zip
```

## 阶段 2 最终停止点

只有 `m1_decision.md` 为 `GO_STAGE3` 时，Stage 2 正常结束。此后不要在 Stage 2 内训练 reward model 或 RA-BC；直接进入 Stage 3。
