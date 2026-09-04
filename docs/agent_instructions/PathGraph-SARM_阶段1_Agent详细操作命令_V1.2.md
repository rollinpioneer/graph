# PathGraph-SARM 阶段 1：Agent 详细操作命令（V1.2）

> 适用范围：实验阶段 1「数据可验证性与路线门控」。  
> 目标：用最短路径把现有数据整理为可直接进入阶段 2 的数据资产，并选出至少两个真实具有多路径或失败恢复结构的任务。  
> 执行原则：优先推进数据整理和任务选择；不在本阶段训练模型，不搭建通用数据平台，不做与阶段产出无关的全仓库重构、CI、安全审计或大规模测试。

## 0. 阶段 1 总体产出

阶段 1 完成时，必须得到以下可直接交给阶段 2 使用的文件：

1. `asset_inventory.csv`：episode 级数据资产清单。
2. `task_summary.csv`：任务级数据规模与模态汇总。
3. `trajectory_tags.csv`：成功、部分成功、失败、恢复、不同合法顺序、重试与回访标签。
4. `coverage_matrix.csv`：`task × path × edge type × outcome` 覆盖矩阵。
5. `dataset_v0.1/episode_manifest.jsonl`：保留完整历史的 episode 清单。
6. `dataset_v0.1/splits.csv`：按 episode/任务实例划分的训练、验证、测试集合。
7. `candidate_task_score.csv`：候选任务评分与排序。
8. `g0_decision.md`：阶段 1 路线决定、选定任务和下一阶段输入。

## 1. 全局目录与变量

Agent 开始执行前，先把下列变量替换为真实路径。不要猜测数据路径；优先读取项目已有配置和 dataset loader。

```bash
export REPO_ROOT="/absolute/path/to/project_repo"
export DATA_ROOT="/absolute/path/to/raw_or_existing_dataset"
export PYTHON_BIN="python"
export STAGE1_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage1"
export STAGE1_CONFIG="$REPO_ROOT/configs/stage1/stage1.yaml"

cd "$REPO_ROOT"
mkdir -p \
  "$REPO_ROOT/configs/stage1" \
  "$REPO_ROOT/tools/stage1" \
  "$STAGE1_ROOT/1.1_asset_inventory" \
  "$STAGE1_ROOT/1.2_trajectory_coverage" \
  "$STAGE1_ROOT/1.3_dataset_v0.1" \
  "$STAGE1_ROOT/1.4_g0_decision"
```

创建阶段配置：

```bash
cat > "$STAGE1_CONFIG" <<'YAML'
project:
  repo_root: /absolute/path/to/project_repo
  data_roots:
    - /absolute/path/to/raw_or_existing_dataset
  # 填写现有训练/数据配置；没有则留空。
  existing_dataset_configs: []
  existing_sarm_annotation_roots: []
  existing_checkpoint_roots: []

runtime:
  seed: 20260831
  workers: 8
  output_root: artifacts/pathgraph_sarm/stage1

inventory:
  # 以项目现有 episode 定义为准；禁止把相邻 clip 当成独立 episode。
  episode_id_keys: [episode_id, traj_id, trajectory_id, demo_id]
  task_id_keys: [task_id, task_name, instruction, environment_name]
  success_keys: [success, is_success, task_success, terminal_success]
  timestamp_keys: [timestamp, timestamps, time, frame_time]
  action_chunk_keys: [action_chunk, chunk_size, horizon, action_horizon]

trajectory_taxonomy:
  outcomes: [success, partial_success, failure, unknown]
  structural_flags: [alternative_order, recovery, retry, revisit, stagnation]
  edge_types: [forward, alternative, recovery, failure, stagnation]

split:
  preserve_official_split: true
  ratios: {train: 0.80, val: 0.10, test: 0.10}
  group_key_priority: [task_instance_id, scene_id, session_id, episode_id]
  stratify_tags: [success, failure, recovery, alternative_order]

selection:
  min_usable_episodes_preferred: 20
  min_recovery_episodes_preferred: 5
  min_distinct_success_paths_preferred: 2
  min_full_history_ratio: 0.90
  min_total_score: 6
  num_tasks_to_select: 2
YAML
```

将配置中的绝对路径改为真实路径：

```bash
sed -n '1,220p' "$STAGE1_CONFIG"
```

## 2. Agent 通用执行约束

- 复用现有 dataset loader、任务元数据、SARM 标注和视频查看工具。只为当前真实格式实现适配，不开发“支持所有格式”的通用框架。
- 原始数据只读。所有新文件写入 `artifacts/pathgraph_sarm/stage1/`。
- 一个 episode 必须保留从开始到结束的历史；原 SARM 的 clip 或干净片段只能作为 episode 内索引，不能替代完整 episode。
- 只做三类必要检查：输出是否生成、episode/group 是否跨 split 泄漏、选定任务是否确实包含结构性多路径或恢复。不要扩展成大规模测试工程。
- 出现缺失字段时，先从已有 metadata、日志、SARM 标注或路径结构恢复；无法恢复的字段设为 `null/unknown`，不要伪造。
- 阶段 1 不训练 reward model、policy、自动图发现模型，也不复现基线。


---

## 固定执行规范：GPU 提权查看、多 GPU 并行、每轮结果 ZIP 交付

以下规则从本阶段开始作为所有实验操作文档的默认执行规范。Agent 不得因为文档某个小步骤未再次重复说明而忽略这些规则。

### 1. GPU 资源必须提权查看

任何将要使用 GPU 的训练、推理、批量特征提取、reward 计算或策略评估开始前，先用提权命令查看 GPU。不要因为普通权限下 `nvidia-smi` 不可见或报错，就判断机器没有 GPU。

先执行：

```bash
mkdir -p "$STAGE1_ROOT/_runtime"

sudo -n nvidia-smi | tee "$STAGE1_ROOT/_runtime/nvidia_smi_full.txt"

sudo -n nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits \
  | tee "$STAGE1_ROOT/_runtime/gpu_status.csv"
```

若当前环境的 `sudo` 需要交互式授权，则改用：

```bash
sudo nvidia-smi
sudo nvidia-smi \
  --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
  --format=csv,noheader,nounits
```

然后根据空闲显存和利用率确定可用 GPU。默认不要占用明显正在执行其他任务的 GPU。记录本轮实际使用的 GPU ID 到该轮的 `run_manifest.md`。

可用 GPU ID 可按以下方式读取；显存阈值根据本实验实际需求修改：

```bash
MIN_FREE_MB=12000
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
printf 'Available GPUs: %s\n' "${GPU_IDS[*]}"
```

如果 `sudo -n` 在机器上不可用，但已知必须提权才能看到 GPU，不要回退到普通权限探测作为最终结论；使用当前环境允许的提权方式后再决定资源分配。

### 2. 能多 GPU 并行时必须并行，优先缩短墙钟时间

当多个实验之间不存在参数更新依赖、共享写入冲突或必须串行的前置关系时，默认并行执行，不要把独立的 seed、baseline、ablation、任务、超参数组合逐个串行跑完。

并行优先级：

1. **优先做实验级并行**：不同 seed、baseline、ablation、task 或超参数组合各占一张 GPU。这通常最直接，也最不需要修改训练代码。
2. **其次才做单实验多 GPU**：当单个训练本身耗时很长、代码已正确支持 DDP/`torchrun`，且扩大单实验 GPU 数确实能缩短时间时使用。
3. 每个并行 job 必须使用独立的 `output_dir`、日志文件和 checkpoint 目录，禁止多个进程写同一个结果目录。
4. 并发 job 数不超过当前确认可用的 GPU 数；若显存足以在同一 GPU 放多个轻量任务，也只有在代码已知不会互相 OOM 时才这样做。
5. 若一个实验阶段只做 CPU 数据整理，不为了“满足多 GPU”而强行使用 GPU；一旦进入可 GPU 并行的训练/推理环节，立即采用上述并行策略。

独立实验并行模板：

```bash
ROUND_ID="r01_example"
ROUND_DIR="$STAGE1_ROOT/rounds/$ROUND_ID"
mkdir -p "$ROUND_DIR/jobs" "$ROUND_DIR/logs"

# 示例 JOBS 中每行是一条独立实验参数。实际执行时替换为本阶段真实命令。
JOBS=(
  "seed=0 variant=baseline"
  "seed=1 variant=baseline"
  "seed=0 variant=pathgraph"
  "seed=1 variant=pathgraph"
)

if [ "${#GPU_IDS[@]}" -lt 1 ]; then
  echo "No privileged-visible free GPU available; do not start GPU experiments." >&2
  exit 2
fi

pids=()
for i in "${!JOBS[@]}"; do
  gpu="${GPU_IDS[$((i % ${#GPU_IDS[@]}))]}"
  job_id=$(printf 'job_%02d' "$i")
  job_dir="$ROUND_DIR/jobs/$job_id"
  mkdir -p "$job_dir"

  # 当已经占满全部可用 GPU 时，等待任意一个 job 完成再继续派发。
  while [ "$(jobs -pr | wc -l)" -ge "${#GPU_IDS[@]}" ]; do
    wait -n
  done

  CUDA_VISIBLE_DEVICES="$gpu" \
  nohup bash -lc "
    set -euo pipefail
    echo '${JOBS[$i]}' > '$job_dir/job_args.txt'
    # 将下一行替换为本小阶段真实训练/推理命令，并把输出目录设为 '$job_dir'。
    python YOUR_SCRIPT.py --output-dir '$job_dir'
  " > "$ROUND_DIR/logs/$job_id.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
```

单实验 DDP 模板，仅在训练脚本已经支持分布式训练时使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node=2 \
  train.py \
  --config path/to/config.yaml \
  --output-dir path/to/unique_output_dir
```

不要为了形式上的并行改写大量模型代码；默认先用“多个独立实验各占一张 GPU”的方式提速。

### 3. 每一轮实验完成后都必须生成一个可下载 ZIP

“每一轮”指一批有共同目的、一起启动或一起比较的实验，例如：某个 baseline 的多 seed、某组 ablation、某次超参数筛选、某个任务组的评估。每轮结束后，不等待整个大阶段结束，立即整理并生成 ZIP。

每个 ZIP 至少包含：

- 本轮 `run_manifest.md`：实验目的、开始/结束时间、GPU ID、运行命令、代码 commit、配置文件路径；
- 本轮所有实际使用的 config；
- 每个 job 的日志；
- 指标文件，如 `metrics.json`、`metrics.csv`、evaluation summary；
- 图表和可视化结果；
- 本轮结果摘要 `summary.md`；
- 失败 job 的错误日志（如果有），但不要因为个别非关键 job 失败而阻止其余已完成结果打包；
- **checkpoint、模型权重、原始数据、缓存、长视频以及其他大文件默认可以不打包。** ZIP 的目标是快速交付可复现实验结论，而不是搬运全部重型产物；
- 对未打包的大文件，在 ZIP 内提供 `large_file_manifest.tsv`（checkpoint 可同时记录到 `checkpoint_manifest.tsv`），至少写明：`path`、`size_bytes`、`job_id`、`artifact_type`、`reason_omitted`，若是 checkpoint 再补充对应 epoch/step 和关键指标。

默认不要把原始大规模 dataset、缓存目录、checkpoint/模型权重、长视频、重复中间产物、`wandb` cache 等大文件塞进 ZIP。除非该文件体积较小且对复现实验结论确有必要，否则只在 manifest 中记录路径与元数据即可。

建议每轮采用以下目录：

```text
artifacts/pathgraph_sarm/<stage>/
├── rounds/
│   └── <ROUND_ID>/
│       ├── run_manifest.md
│       ├── configs/
│       ├── jobs/
│       ├── logs/
│       ├── metrics/
│       ├── plots/
│       └── summary.md
└── downloads/
    ├── <ROUND_ID>.zip
    └── index.md
```

打包命令：

```bash
ROUND_ID="r01_example"
ROUND_DIR="$STAGE1_ROOT/rounds/$ROUND_ID"
DOWNLOAD_DIR="$STAGE1_ROOT/downloads"
mkdir -p "$DOWNLOAD_DIR"

# 先写最小运行清单。
{
  echo "# Run Manifest"
  echo
  echo "- round_id: $ROUND_ID"
  echo "- generated_at: $(date -Iseconds)"
  echo "- repo_root: $REPO_ROOT"
  echo "- git_commit: $(git -C \"$REPO_ROOT\" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "- python: $($PYTHON_BIN --version 2>&1)"
  echo "- gpu_ids: ${GPU_IDS[*]:-none}"
} > "$ROUND_DIR/run_manifest.md"

# 生成“大文件/重型产物未打包清单”。默认阈值 200 MB，可按项目实际情况调整。
ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
LARGE_MANIFEST="$ROUND_DIR/large_file_manifest.tsv"
CHECKPOINT_MANIFEST="$ROUND_DIR/checkpoint_manifest.tsv"

printf 'path\tsize_bytes\tartifact_type\treason_omitted\n' > "$LARGE_MANIFEST"
printf 'path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n' > "$CHECKPOINT_MANIFEST"

# 记录 checkpoint / 模型权重；默认不放入 ZIP。
find "$ROUND_DIR" -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' -o -path '*/checkpoints/*' \) \
  -printf '%p\t%s\tcheckpoint_or_model_weight\tdefault_omit_from_zip\n' \
  >> "$LARGE_MANIFEST" || true

find "$ROUND_DIR" -type f \
  \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.bin' -o -name '*.safetensors' -o -path '*/checkpoints/*' \) \
  -printf '%p\t%s\tunknown\tunknown\tunknown\n' \
  >> "$CHECKPOINT_MANIFEST" || true

# 记录其他超过阈值的大文件；也默认不放入 ZIP。
find "$ROUND_DIR" -type f -size +"${ZIP_MAX_FILE_MB}"M \
  ! -name 'large_file_manifest.tsv' ! -name 'checkpoint_manifest.tsv' \
  -printf '%p\t%s\tlarge_file\tover_size_threshold\n' \
  >> "$LARGE_MANIFEST" || true

# 只打包轻量结果文件：配置、日志、指标、图表、summary、manifest 等。
# checkpoint/模型权重和超过阈值的大文件默认排除。
cd "$(dirname "$ROUND_DIR")"
ROUND_BASENAME="$(basename "$ROUND_DIR")"
rm -f "$DOWNLOAD_DIR/${ROUND_ID}.zip"

find "$ROUND_BASENAME" -type f \
  ! -path '*/__pycache__/*' \
  ! -path '*/.cache/*' \
  ! -path '*/wandb/*' \
  ! -path '*/dataset_cache/*' \
  ! -path '*/checkpoints/*' \
  ! -name '*.ckpt' ! -name '*.pt' ! -name '*.pth' ! -name '*.bin' ! -name '*.safetensors' \
  ! -size +"${ZIP_MAX_FILE_MB}"M \
  -print | zip -q "$DOWNLOAD_DIR/${ROUND_ID}.zip" -@

sha256sum "$DOWNLOAD_DIR/${ROUND_ID}.zip" \
  > "$DOWNLOAD_DIR/${ROUND_ID}.zip.sha256"

{
  echo "- $(date -Iseconds)  ${ROUND_ID}.zip"
  echo "  - sha256: $(cut -d' ' -f1 "$DOWNLOAD_DIR/${ROUND_ID}.zip.sha256")"
} >> "$DOWNLOAD_DIR/index.md"

ls -lh "$DOWNLOAD_DIR/${ROUND_ID}.zip" "$DOWNLOAD_DIR/${ROUND_ID}.zip.sha256"
```

Agent 在每轮结束时必须在回复/交付信息里明确写出该 ZIP 的实际路径，供上层系统转成下载链接。不要只说“结果已保存”。

### 4. 本阶段对上述规范的具体应用

阶段 1 主要是数据整理，通常不会有重型 GPU 训练，因此不要求为了阶段 1 强行占 GPU。但如果资产扫描、视觉特征推理、轨迹标签模型或任何加速步骤实际使用 GPU，必须先提权查看 GPU，并对可以独立运行的任务/数据分片并行执行。

阶段 1 的四个小阶段也视为四轮可交付工作；每完成一个小阶段，都生成对应 ZIP：

```text
stage1_1_asset_inventory.zip
stage1_2_trajectory_coverage.zip
stage1_3_dataset_v0_1.zip
stage1_4_g0_decision.zip
```

阶段 1 全部完成后，再额外生成：

```text
stage1_complete.zip
```

该总 ZIP 汇总四个小阶段的最终产出、`g0_decision.md` 和阶段 2 的输入清单。


# 小阶段 1.1：数据资产盘点——Agent 操作命令

## A. 总体上要干什么

把项目中真正可用的任务、episode、观测模态、动作、时间戳、成功标签、SARM 标注、action-chunk 信息和 checkpoint 整理成机器可读清单。重点不是写报告，而是确定后续脚本能够从哪里读取完整 episode。

## B. 本小阶段必须生成

```text
artifacts/pathgraph_sarm/stage1/1.1_asset_inventory/
├── asset_inventory.csv
├── task_summary.csv
├── data_format_notes.md
└── code_checkpoint_inventory.md
```

## C. 可直接交给 Agent 的任务命令

```text
你现在执行 PathGraph-SARM 阶段 1.1「数据资产盘点」。

工作目录：读取 configs/stage1/stage1.yaml 中的 project.repo_root、data_roots 和 output_root。

执行目标：
1. 找到项目当前实际使用的 dataset loader、数据配置、SARM 标注读取逻辑和训练入口。
2. 以完整 episode 为行，生成 asset_inventory.csv。
3. 按 task_id 聚合生成 task_summary.csv。
4. 记录真实数据格式、关键字段映射和已存在 checkpoint。

严格执行以下步骤：

步骤 1：定位数据入口。
- 在仓库中搜索 Dataset/DataLoader、episode、trajectory、success、action horizon、SARM annotation 等关键词。
- 优先打开训练入口引用的数据配置和 loader；不要只根据文件名猜数据格式。
- 确认 loader 返回的最小单位是完整 episode、clip 还是单帧。
- 若 loader 返回 clip，继续追踪其 parent episode 字段或源文件，使 inventory 最终按完整 episode 建行。

步骤 2：确认实际数据格式。
- 只记录项目实际存在的格式，例如 HDF5、Zarr、NPZ、JSONL、RLDS、目录式视频+metadata。
- 对主格式记录：episode_id、task_id、步数、时间戳、观测键、动作键、成功键、subtask/stage 标注键、action chunk 键。
- 不要实现与本项目无关的格式适配器。

步骤 3：实现 tools/stage1/scan_assets.py。
- CLI 至少支持 --config 和 --output-dir。
- 复用项目已有 loader 或其底层索引；不要逐帧解码全部视频，只读取索引和 metadata。
- 每个 episode 输出一行，字段严格按照本文 D 节。
- source_path 使用绝对路径或可由 repo_root/data_root 稳定解析的路径。
- has_full_episode_history 只有在能够从 episode 开始一直读取到终止时才为 true。
- 若某字段不存在，写空值并在 notes 中说明，不要中断整个扫描。

步骤 4：生成 task_summary.csv。
- 按 task_id 聚合 episode 数、成功/失败已知数量、总步数、完整历史比例、action-chunk 覆盖率、SARM 标注覆盖率和模态列表。

步骤 5：生成说明文件。
- data_format_notes.md：写明实际格式、字段映射、loader 入口、episode 边界定义和当前已知缺口。
- code_checkpoint_inventory.md：列出原 SARM、BC 或相关模型的训练脚本、配置、checkpoint 路径；没有则明确写“未发现”。

步骤 6：运行扫描并修正阻塞问题。
- 首先在少量 episode 上运行，确认字段正确。
- 然后运行全量 metadata 扫描。
- 只修正会导致 episode 缺失、任务映射错误或完整历史无法定位的问题；不要在本阶段重构训练代码。

完成条件：四个输出文件存在；asset_inventory.csv 至少包含一条 episode；task_summary.csv 中每个 task_id 都能回指 inventory；能够明确指出完整 episode 的读取入口。
```

## D. `asset_inventory.csv` 字段规范

脚本至少输出以下列；项目存在额外字段时可追加，不要删除这些核心列。

| 字段 | 含义 |
|---|---|
| `task_id` | 规范化任务名或任务 ID |
| `task_instance_id` | 场景、物体配置或具体任务实例；无则空 |
| `episode_id` | 全局唯一 episode ID |
| `source_path` | 原始 episode 或其索引路径 |
| `source_format` | 实际数据格式 |
| `num_steps` | episode 步数/帧数 |
| `duration_sec` | 可计算时填写 |
| `fps` | 视频/帧率；无则空 |
| `observation_modalities` | RGB、depth、proprio、tactile 等，以 `;` 分隔 |
| `camera_names` | 摄像头名称列表 |
| `action_dim` | 动作维度 |
| `has_gripper` | 是否包含夹爪信号 |
| `has_proprio` | 是否包含本体状态 |
| `has_timestamps` | 是否存在时间戳 |
| `success_label` | `true/false/unknown` |
| `original_outcome` | 原数据中的 outcome 文本 |
| `has_subtask_or_stage_labels` | 是否有 subtask/stage 边界或标签 |
| `sarm_annotation_path` | 原 SARM 标注路径；无则空 |
| `has_action_chunks` | 是否保留 action chunk 信息 |
| `action_chunk_size` | chunk/horizon；无则空 |
| `has_full_episode_history` | 能否恢复完整 episode |
| `original_split` | 原 train/val/test；无则空 |
| `metadata_path` | metadata 路径 |
| `notes` | 缺失字段或特殊情况 |

## E. 实施命令

先定位已有数据代码：

```bash
cd "$REPO_ROOT"
rg -n --hidden \
  -g '!artifacts/**' -g '!outputs/**' -g '!checkpoints/**' \
  "class .*Dataset|DataLoader|episode_id|trajectory_id|task_success|action_horizon|stage.*label|SARM" \
  . | head -n 300
```

实现脚本后先试跑：

```bash
"$PYTHON_BIN" tools/stage1/scan_assets.py \
  --config "$STAGE1_CONFIG" \
  --output-dir "$STAGE1_ROOT/1.1_asset_inventory" \
  --max-episodes 20
```

查看前 20 行，确认 episode 和 task 映射正确：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd
p = "artifacts/pathgraph_sarm/stage1/1.1_asset_inventory/asset_inventory.csv"
df = pd.read_csv(p)
print(df.head(20).to_string(index=False))
print("episodes:", len(df))
print("tasks:", df["task_id"].nunique())
print("full_history_ratio:", df["has_full_episode_history"].fillna(False).mean())
PY
```

全量运行：

```bash
"$PYTHON_BIN" tools/stage1/scan_assets.py \
  --config "$STAGE1_CONFIG" \
  --output-dir "$STAGE1_ROOT/1.1_asset_inventory"
```

## F. 完成检查

只执行以下最小检查：

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("artifacts/pathgraph_sarm/stage1/1.1_asset_inventory")
required = [
    root / "asset_inventory.csv",
    root / "task_summary.csv",
    root / "data_format_notes.md",
    root / "code_checkpoint_inventory.md",
]
missing = [str(p) for p in required if not p.exists()]
assert not missing, f"missing outputs: {missing}"
df = pd.read_csv(root / "asset_inventory.csv")
assert len(df) > 0, "inventory is empty"
assert df["episode_id"].notna().all(), "episode_id has null"
assert not df["episode_id"].duplicated().any(), "episode_id is not unique"
print("Stage 1.1 complete:", len(df), "episodes,", df["task_id"].nunique(), "tasks")
PY
```

## G. 停止点

当四个输出生成且完整 episode 入口明确后，立即进入小阶段 1.2。不要在 1.1 中开始标注图节点、训练 reward model 或重写数据框架。


---

# 小阶段 1.2：轨迹类型与结构覆盖——Agent 操作命令

## A. 总体上要干什么

把每条完整 episode 归类为成功、部分成功、失败或未知，并识别是否包含不同合法顺序、失败后恢复、重试、回访和停滞。最终产出能够直接回答“哪些任务真实具有图结构、关键 edge 有多少样本”的覆盖矩阵。

## B. 本小阶段必须生成

```text
artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage/
├── trajectory_tags.csv
├── episode_events.jsonl
├── path_signatures.jsonl
├── coverage_matrix.csv
├── task_structure_summary.csv
├── manual_review_queue.csv
└── coverage_summary.md
```

## C. 轨迹标签的操作定义

使用项目任务语义和现有日志进行判断，不要求此时已有正式任务图。

| 标签 | 操作定义 |
|---|---|
| `success` | episode 达到任务成功终点，优先采用环境/数据集正式 success 字段 |
| `partial_success` | 至少一个语义子目标完成，但 episode 未达到最终成功 |
| `failure` | episode 结束时未成功；若有已完成子目标，outcome 可标 `partial_success`，同时保留 failure event |
| `alternative_order` | 两条成功 episode 完成相同必要子目标，但子目标首次完成顺序不同，且两种顺序均符合任务规则 |
| `recovery` | 明确失败/失效事件后，机器人重新进入合法执行状态，并继续完成后续子目标或最终成功 |
| `retry` | 对同一目标或同一动作语义发生第二次及以上尝试 |
| `revisit` | episode 返回此前已经访问过的语义状态或子任务位置 |
| `stagnation` | 一段明显持续时间内没有新子目标完成、没有有效状态转移，或重复无效动作 |

`path_signature` 采用“必要子目标首次完成顺序”，例如：

```text
pick_A>place_A>pick_B>place_B
pick_B>place_B>pick_A>place_A
```

失败尝试和恢复不混入成功顺序字符串，另用 `recovery_count`、`retry_count` 和事件流记录。这样能够直接比较合法完成顺序，同时保留恢复信息。

## D. 可直接交给 Agent 的任务命令

```text
你现在执行 PathGraph-SARM 阶段 1.2「轨迹类型与结构覆盖」。

输入：
- 1.1_asset_inventory/asset_inventory.csv
- 项目现有 episode loader、任务 success 规则、SARM stage/subtask 标注、环境日志或事件 metadata

执行目标：
1. 为每个 episode 建立轻量事件流 episode_events.jsonl。
2. 生成 trajectory_tags.csv 和 path_signatures.jsonl。
3. 按 task × path × edge type × outcome 生成 coverage_matrix.csv。
4. 形成任务级结构汇总，供 1.4 直接选择任务。

严格执行以下步骤：

步骤 1：确定任务语义事件来源。
- 优先级依次为：环境正式事件/成功条件 > 数据集 subtask 标签 > 原 SARM stage 边界 > 已有人工注释 > 文件名或日志中的可靠事件。
- 不训练视觉事件检测器。
- 对每个 task_id 写出必要子目标列表和完成条件；存入配置文件的 task_semantics 段或独立 YAML。

步骤 2：实现 tools/stage1/analyze_trajectory_structure.py。
- CLI 支持 --config、--inventory、--output-dir、--max-episodes。
- 对每个 episode 读取 metadata 和必要的少量时序字段，生成标准事件：
  {episode_id, task_id, step, timestamp, event_type, target, source, confidence}。
- event_type 至少支持：subgoal_start、subgoal_complete、failure_onset、recovery_complete、attempt_start、attempt_end、episode_success、episode_end。
- 只在必须判断事件时读取视频/帧；不要全量做昂贵视觉处理。

步骤 3：生成 trajectory_tags.csv。
- 每个 episode 一行。
- outcome 为 success、partial_success、failure 或 unknown。
- structural flag 为布尔值。
- path_signature 使用必要子目标首次完成顺序。
- evidence_source 写明标签来自 metadata、SARM annotation、event log 或 manual。
- evidence_ranges 使用 step 区间，例如 120-180;260-300。
- 无法自动判断的 episode 加入 manual_review_queue.csv，而不是强行赋值。

步骤 4：处理 manual_review_queue。
- 只审阅会影响候选任务判断的 episode：成功但顺序未知、疑似 recovery、疑似 retry/revisit。
- 每个 task 先审阅最多 20 条高价值候选；若已经足以确认路径和恢复覆盖，停止继续审阅。
- 使用项目已有 viewer；如果数据只有帧目录且无查看工具，再实现最小 preview 导出脚本。
- 人工结果回写 trajectory_tags.csv，并把 evidence_source 设为 manual。

步骤 5：生成 path_signatures.jsonl。
- 每行包括 task_id、episode_id、outcome、path_signature、completed_subgoals、failure_count、recovery_count、retry_count、revisit_count。
- 对 success episode 统计不同合法 path_signature 的 episode 数。

步骤 6：生成 coverage_matrix.csv。
- 使用长表结构：task_id、path_signature、edge_type、outcome、episode_count、total_steps、full_history_count、action_chunk_count。
- edge_type 从 coarse events 推导为 forward、alternative、recovery、failure、stagnation。
- alternative 表示该成功 path_signature 与任务中另一合法成功顺序不同；不是普通动作差异。

步骤 7：生成 task_structure_summary.csv 和 coverage_summary.md。
- 每个任务汇总 usable_episodes、success_episodes、failure_episodes、partial_episodes、distinct_success_paths、alternative_order_episodes、recovery_episodes、retry_episodes、revisit_episodes、full_history_ratio、action_chunk_ratio。
- coverage_summary.md 按任务列出最重要的结构证据和缺口，不写泛泛而谈的审计说明。

完成条件：能够按任务回答“有几条成功路径、多少恢复 episode、完整历史是否可用”；至少将结构最强的候选任务排出来。
```

## E. `trajectory_tags.csv` 字段规范

```text
task_id
episode_id
outcome
success
partial_success
failure
alternative_order
recovery
retry
revisit
stagnation
path_signature
completed_subgoals
failure_count
recovery_count
retry_count
revisit_count
evidence_source
evidence_ranges
label_confidence
notes
```

其中 `completed_subgoals` 使用 JSON 字符串或 `;` 分隔列表，必须保持顺序。

## F. 实施命令

先小规模运行：

```bash
"$PYTHON_BIN" tools/stage1/analyze_trajectory_structure.py \
  --config "$STAGE1_CONFIG" \
  --inventory "$STAGE1_ROOT/1.1_asset_inventory/asset_inventory.csv" \
  --output-dir "$STAGE1_ROOT/1.2_trajectory_coverage" \
  --max-episodes 50
```

查看候选和未知标签：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd
root = "artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage"
tags = pd.read_csv(f"{root}/trajectory_tags.csv")
print(tags.groupby(["task_id", "outcome"]).size().unstack(fill_value=0).to_string())
print("\nstructural counts")
cols = ["alternative_order", "recovery", "retry", "revisit", "stagnation"]
print(tags.groupby("task_id")[cols].sum().sort_values(["recovery", "alternative_order"], ascending=False).to_string())
unknown = tags[tags["outcome"].eq("unknown")]
print("\nunknown episodes:", len(unknown))
PY
```

全量运行：

```bash
"$PYTHON_BIN" tools/stage1/analyze_trajectory_structure.py \
  --config "$STAGE1_CONFIG" \
  --inventory "$STAGE1_ROOT/1.1_asset_inventory/asset_inventory.csv" \
  --output-dir "$STAGE1_ROOT/1.2_trajectory_coverage"
```

若有人工复核队列，Agent 只处理结构候选：

```bash
column -s, -t < "$STAGE1_ROOT/1.2_trajectory_coverage/manual_review_queue.csv" | head -n 80
```

人工结果回写后，重新运行汇总模式；脚本应支持读取已有人工标签并保留：

```bash
"$PYTHON_BIN" tools/stage1/analyze_trajectory_structure.py \
  --config "$STAGE1_CONFIG" \
  --inventory "$STAGE1_ROOT/1.1_asset_inventory/asset_inventory.csv" \
  --output-dir "$STAGE1_ROOT/1.2_trajectory_coverage" \
  --reuse-manual-labels
```

## G. 最小完成检查

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import pandas as pd
root = Path("artifacts/pathgraph_sarm/stage1/1.2_trajectory_coverage")
for name in [
    "trajectory_tags.csv", "episode_events.jsonl", "path_signatures.jsonl",
    "coverage_matrix.csv", "task_structure_summary.csv", "manual_review_queue.csv",
    "coverage_summary.md"
]:
    assert (root / name).exists(), f"missing {name}"
tags = pd.read_csv(root / "trajectory_tags.csv")
assert tags["episode_id"].is_unique, "episode_id duplicated"
assert set(tags["outcome"].dropna()).issubset({"success", "partial_success", "failure", "unknown"})
summary = pd.read_csv(root / "task_structure_summary.csv")
print(summary.sort_values(["distinct_success_paths", "recovery_episodes"], ascending=False).head(20).to_string(index=False))
PY
```

## H. 停止点

覆盖矩阵和任务级结构汇总生成后进入 1.3。不要在此阶段定义 5–10 个正式图节点；正式 graph spec 属于阶段 2。


---

# 小阶段 1.3：完整 episode 数据版本与划分——Agent 操作命令

## A. 总体上要干什么

把阶段 1.1 和 1.2 的结果合并成 `dataset_v0.1`。数据版本必须指向完整 episode，保留历史、轨迹类型、路径签名和 action-chunk 信息，并按 episode/任务实例划分 train、val、test，避免同一演示的相邻片段进入不同集合。

## B. 本小阶段必须生成

```text
artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/
├── episode_manifest.jsonl
├── splits.csv
├── split_summary.csv
├── dataset_card.md
├── source_fingerprint.csv
├── manifest.sha256
└── split_checks.json
```

不复制大体量原始视频或图像；manifest 指向原始只读数据。只有当前训练框架要求特定索引格式时，额外生成轻量索引。

## C. 可直接交给 Agent 的任务命令

```text
你现在执行 PathGraph-SARM 阶段 1.3「完整 episode 数据版本与划分」。

输入：
- 1.1_asset_inventory/asset_inventory.csv
- 1.2_trajectory_coverage/trajectory_tags.csv
- 1.2_trajectory_coverage/path_signatures.jsonl
- configs/stage1/stage1.yaml

执行目标：
1. 生成完整 episode manifest。
2. 生成可重复的 train/val/test 划分。
3. 保留图结构相关标签和原始数据定位信息。
4. 只做防止数据泄漏和缺失 episode 的必要检查。

严格执行以下步骤：

步骤 1：实现 tools/stage1/build_dataset_v01.py。
- CLI 支持 --config、--inventory、--tags、--path-signatures、--output-dir。
- 按 episode_id 合并 inventory 与 tags；任何无法匹配的行输出到控制台并写入 dataset_card.md 的 missing mapping 段。
- manifest 每行代表完整 episode，不代表 clip。
- 若现有 SARM 训练使用 clip，manifest 中保存 clip 索引或 annotation_path，但必须同时保存 parent episode 和完整历史范围。

步骤 2：确定 group_id。
- 若有官方 split，且其单位是 episode 或更高层任务实例，优先保留。
- 无官方 split 时，按以下优先级选择 group_id：task_instance_id > scene_id > session_id > episode_id。
- 同一 group_id 的所有 episode 必须进入同一 split。
- 同一 episode 的所有 clip 必须跟随 parent episode split。

步骤 3：执行确定性划分。
- 使用配置中的 seed 和 ratios。
- 以 task_id 为基本分层单位；在不破坏 group 边界的前提下，尽量让 success、failure、recovery、alternative_order 分布在 train/val/test。
- 稀有 recovery 或 alternative 样本过少时，优先保证 train 有足够学习样本，并在 val 或 test 至少保留一个独立 group 用于机制评估。
- 不为追求精确比例拆分 group。

步骤 4：生成 episode_manifest.jsonl。
- 每行至少包含本文 D 节字段。
- source_path 指向原始数据；不要复制原始数据。
- history_start_step 固定为 0，history_end_step 为 episode 最后一步。
- has_full_episode_history=false 的 episode 可以保留在 manifest，但必须标记 usable_for_pathgraph=false；后续任务选择默认不使用。

步骤 5：生成 source_fingerprint.csv 和 manifest.sha256。
- source_fingerprint 记录 source_path、文件大小、mtime、episode_id；若单个容器文件包含多个 episode，记录容器文件指纹和内部 key。
- 不对超大数据逐字节全量哈希；对 manifest 本身做 SHA256 即可。

步骤 6：生成 dataset_card.md。
- 写明数据来源、episode 单位、标签来源、split 策略、任务数量、episode 数、图结构标签数量、已知缺口。
- 明确数据版本名为 pathgraph_stage1_dataset_v0.1。

步骤 7：运行最小 split 检查。
- episode_id 唯一。
- group_id 不跨 split。
- 所有 manifest 行都有 split。
- selected candidate 尚未确定，因此只报告各任务的结构标签分布，不做额外统计测试。

完成条件：后续阶段只需读取 episode_manifest.jsonl 和 splits.csv 即可稳定定位完整 episode、标签和 split。
```

## D. `episode_manifest.jsonl` 每行字段

```json
{
  "dataset_version": "pathgraph_stage1_dataset_v0.1",
  "task_id": "...",
  "task_instance_id": "...",
  "episode_id": "...",
  "group_id": "...",
  "split": "train|val|test",
  "source_path": "...",
  "source_format": "...",
  "num_steps": 0,
  "history_start_step": 0,
  "history_end_step": 0,
  "has_full_episode_history": true,
  "usable_for_pathgraph": true,
  "outcome": "success|partial_success|failure|unknown",
  "path_signature": "...",
  "alternative_order": false,
  "recovery": false,
  "retry": false,
  "revisit": false,
  "stagnation": false,
  "failure_count": 0,
  "recovery_count": 0,
  "action_chunk_size": null,
  "sarm_annotation_path": null,
  "metadata_path": null
}
```

## E. 实施命令

```bash
"$PYTHON_BIN" tools/stage1/build_dataset_v01.py \
  --config "$STAGE1_CONFIG" \
  --inventory "$STAGE1_ROOT/1.1_asset_inventory/asset_inventory.csv" \
  --tags "$STAGE1_ROOT/1.2_trajectory_coverage/trajectory_tags.csv" \
  --path-signatures "$STAGE1_ROOT/1.2_trajectory_coverage/path_signatures.jsonl" \
  --output-dir "$STAGE1_ROOT/1.3_dataset_v0.1"
```

查看 split 分布：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd
p = "artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/splits.csv"
df = pd.read_csv(p)
print(df.groupby(["task_id", "split"]).size().unstack(fill_value=0).to_string())
for tag in ["success", "failure", "recovery", "alternative_order"]:
    if tag in df.columns:
        print(f"\n{tag}")
        print(df.groupby("split")[tag].sum().to_string())
PY
```

## F. 最小完成检查

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import json, pandas as pd
root = Path("artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1")
required = [
    "episode_manifest.jsonl", "splits.csv", "split_summary.csv", "dataset_card.md",
    "source_fingerprint.csv", "manifest.sha256", "split_checks.json"
]
for name in required:
    assert (root / name).exists(), f"missing {name}"
sp = pd.read_csv(root / "splits.csv")
assert sp["episode_id"].is_unique, "episode_id duplicated"
assert sp["split"].isin(["train", "val", "test"]).all(), "invalid split"
leak = sp.groupby("group_id")["split"].nunique()
assert int((leak > 1).sum()) == 0, "group leakage across splits"
with open(root / "split_checks.json", "r", encoding="utf-8") as f:
    print(json.load(f))
print("Stage 1.3 complete:", len(sp), "episodes")
PY
```

## G. 停止点

数据版本和 split 生成后进入 1.4。不要在 1.3 中为了追求完全均衡而反复改变 split；只要 group 无泄漏、关键结构在训练和至少一个 held-out split 中可评估即可。


---

# 小阶段 1.4：候选任务选择与 G0 路线决定——Agent 操作命令

## A. 总体上要干什么

基于现有数据证据，选出至少两个真正存在合法不同顺序或失败恢复的任务，确定进入阶段 2 的任务、episode 范围和数据版本。若结构真实但样本不足，直接生成定向补采清单；只有在结构不存在或完整历史无法恢复时才切换路线。

## B. 本小阶段必须生成

```text
artifacts/pathgraph_sarm/stage1/1.4_g0_decision/
├── candidate_task_score.csv
├── selected_tasks.yaml
├── candidate_task_evidence.md
├── targeted_collection_plan.csv
├── g0_decision.md
└── stage2_handoff.md
```

`targeted_collection_plan.csv` 在无需补采时也保留表头，并写 0 行。

## C. 默认候选评分

这些是执行默认值，可在 `configs/stage1/stage1.yaml` 中调整。评分的目的只是透明排序，不替代任务语义判断。

| 项目 | 默认分值 |
|---|---:|
| 至少 2 条不同成功 `path_signature` | 3 |
| recovery episode ≥ 5 | 3 |
| recovery episode 为 2–4 | 2 |
| recovery episode 为 1 | 1 |
| 完整历史比例 ≥ 0.90 | 2 |
| 可用 episode ≥ 20 | 1 |
| action-chunk 覆盖率 ≥ 0.80 | 1 |

候选任务必须满足：

1. `alternative_order` 或 `recovery` 至少一项有真实证据；
2. 能读取失败前后或顺序变化前后的完整历史；
3. 任务语义上确实允许多顺序、回访或恢复，不是相同阶段内的普通动作差异。

## D. 可直接交给 Agent 的任务命令

```text
你现在执行 PathGraph-SARM 阶段 1.4「候选任务选择与 G0 路线决定」。

输入：
- 1.2_trajectory_coverage/task_structure_summary.csv
- 1.2_trajectory_coverage/coverage_matrix.csv
- 1.2_trajectory_coverage/trajectory_tags.csv
- 1.3_dataset_v0.1/episode_manifest.jsonl
- 1.3_dataset_v0.1/splits.csv

执行目标：
1. 对所有任务计算透明评分并排序。
2. 对排名最高的候选进行少量但直接的 episode 证据确认。
3. 选定至少两个阶段 2 任务，或生成明确的定向补采计划。
4. 输出 G0 决定和阶段 2 交接文件。

严格执行以下步骤：

步骤 1：实现 tools/stage1/select_graph_tasks.py。
- CLI 支持 --config、--task-summary、--coverage、--tags、--manifest、--output-dir。
- 读取 selection 阈值并计算 candidate_task_score.csv。
- 每个分值都保留独立列；不要只输出总分。
- 额外输出 structural_reason，明确是 multiple_success_paths、recovery、两者兼有，还是无结构证据。

步骤 2：确认前 3–5 个候选任务的结构证据。
- 每个候选查看最少数量的代表 episode：
  a. 每种成功 path_signature 各 1–2 条；
  b. recovery episode 2 条；
  c. 如存在重复失败循环，再查看 1 条。
- 只需要确认任务语义和标签没有明显误判；不做大规模人工复核。
- 在 candidate_task_evidence.md 中记录 episode_id、path_signature、关键 step 区间和一句证据说明。

步骤 3：选择任务。
- 首选总分最高且结构互补的两个任务，例如一个以 alternative order 为主、一个以 failure/recovery 为主。
- 若同一任务同时覆盖两种结构，可作为第一任务；第二任务优先选择不同物体配置或不同恢复类型，以减少结论依赖单一任务。
- selected_tasks.yaml 写入 task_id、选择原因、主要结构、可用 episode、train/val/test 数量、关键 path_signature、recovery 数量、manifest 路径。

步骤 4：决定 G0 状态。
- GO：至少两个任务具有真实结构，完整历史可用，当前样本足以开始阶段 2 的人工图和 GT 子集。
- GO_WITH_TARGETED_COLLECTION：至少一个或两个任务结构真实，但关键 path/edge 样本不足。仍完成任务和图语义准备，同时按 targeted_collection_plan.csv 定向补采。
- SWITCH：任务本质均为单链，或失败前后完整历史无法恢复且无法补采。此时停止 PathGraph-SARM 主攻路线。

步骤 5：生成定向补采计划。
- 对每个缺口写一行：task_id、scenario、current_count、target_count、gap、collection_instruction、required_metadata。
- collection_instruction 要直接描述要采什么，例如：
  “在任务 X 中分别执行 A→B 与 B→A 两种合法完成顺序，各补 10 条完整 episode”；
  “在抓取失败后重新对准并成功放置，补 10 条单次恢复 episode”；
  “补 5 条连续两次失败后仍未恢复的循环负例”。
- 只补关键 edge，不用额外采大量普通成功轨迹。

步骤 6：生成 g0_decision.md。
- 写明状态、选定任务、结构证据、数据版本、主要缺口、下一步。
- 不写长篇风险审计；结论必须明确可执行。

步骤 7：生成 stage2_handoff.md。
- 列出阶段 2 直接读取的文件。
- 对每个选定任务列出建议的 5–10 个语义节点候选、必要成功终点和已知 recovery/failure 事件，但不要在阶段 1 正式冻结图。
- 给出阶段 2 第一条执行命令：建立 Graph spec v1 和标注手册。

完成条件：产生明确 G0 状态；若 GO/GO_WITH_TARGETED_COLLECTION，selected_tasks.yaml 至少包含两个目标任务，或清楚说明第二任务待补采的具体数量和场景。
```

## E. 实施命令

```bash
"$PYTHON_BIN" tools/stage1/select_graph_tasks.py \
  --config "$STAGE1_CONFIG" \
  --task-summary "$STAGE1_ROOT/1.2_trajectory_coverage/task_structure_summary.csv" \
  --coverage "$STAGE1_ROOT/1.2_trajectory_coverage/coverage_matrix.csv" \
  --tags "$STAGE1_ROOT/1.2_trajectory_coverage/trajectory_tags.csv" \
  --manifest "$STAGE1_ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" \
  --output-dir "$STAGE1_ROOT/1.4_g0_decision"
```

查看评分：

```bash
"$PYTHON_BIN" - <<'PY'
import pandas as pd
p = "artifacts/pathgraph_sarm/stage1/1.4_g0_decision/candidate_task_score.csv"
df = pd.read_csv(p)
cols = [c for c in [
    "task_id", "total_score", "distinct_success_paths", "recovery_episodes",
    "full_history_ratio", "usable_episodes", "action_chunk_ratio", "structural_reason"
] if c in df.columns]
print(df.sort_values("total_score", ascending=False)[cols].head(20).to_string(index=False))
PY
```

## F. `selected_tasks.yaml` 模板

```yaml
dataset_version: pathgraph_stage1_dataset_v0.1
manifest: artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl
g0_status: GO
selected_tasks:
  - task_id: task_name_1
    primary_structure: alternative_order
    reason: "存在两条以上合法成功顺序，完整历史可恢复"
    usable_episodes: 0
    split_counts: {train: 0, val: 0, test: 0}
    success_path_signatures: []
    recovery_episodes: 0
    candidate_node_concepts: []
  - task_id: task_name_2
    primary_structure: recovery
    reason: "包含失败后重新进入合法路径并成功的完整 episode"
    usable_episodes: 0
    split_counts: {train: 0, val: 0, test: 0}
    success_path_signatures: []
    recovery_episodes: 0
    candidate_node_concepts: []
```

## G. `g0_decision.md` 必须包含的结论格式

```markdown
# G0 路线决定

- 状态：GO / GO_WITH_TARGETED_COLLECTION / SWITCH
- 数据版本：pathgraph_stage1_dataset_v0.1
- 选定任务：...
- 结构证据：...
- 完整历史：...
- 当前关键 edge 数量：...
- 需要补采：无 / 见 targeted_collection_plan.csv
- 阶段 2 入口：读取 selected_tasks.yaml，开始人工 Graph spec v1 与 GT 标注协议。
```

## H. 最小完成检查

```bash
"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml
root = Path("artifacts/pathgraph_sarm/stage1/1.4_g0_decision")
required = [
    "candidate_task_score.csv", "selected_tasks.yaml", "candidate_task_evidence.md",
    "targeted_collection_plan.csv", "g0_decision.md", "stage2_handoff.md"
]
for name in required:
    assert (root / name).exists(), f"missing {name}"
with open(root / "selected_tasks.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
status = cfg.get("g0_status")
assert status in {"GO", "GO_WITH_TARGETED_COLLECTION", "SWITCH"}, status
if status in {"GO", "GO_WITH_TARGETED_COLLECTION"}:
    tasks = cfg.get("selected_tasks", [])
    assert len(tasks) >= 2, "need two selected/targeted tasks"
print("Stage 1.4 complete; G0 status:", status)
PY
```

## I. 停止点

`g0_decision.md` 和 `stage2_handoff.md` 写完后，阶段 1 结束。GO 时直接进入阶段 2；GO_WITH_TARGETED_COLLECTION 时并行补采关键 edge 和编写人工图；SWITCH 时不继续堆叠图模型。


---

# 阶段 1 一键执行入口（四个脚本完成后添加）

为了让后续重跑不依赖手工拼命令，创建一个轻量顺序脚本。它只串联四个小阶段，不引入工作流平台。

```bash
cat > "$REPO_ROOT/tools/stage1/run_stage1.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage1/stage1.yaml}"
ROOT="artifacts/pathgraph_sarm/stage1"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p \
  "$ROOT/1.1_asset_inventory" \
  "$ROOT/1.2_trajectory_coverage" \
  "$ROOT/1.3_dataset_v0.1" \
  "$ROOT/1.4_g0_decision"

"$PYTHON_BIN" tools/stage1/scan_assets.py \
  --config "$CONFIG" \
  --output-dir "$ROOT/1.1_asset_inventory"

"$PYTHON_BIN" tools/stage1/analyze_trajectory_structure.py \
  --config "$CONFIG" \
  --inventory "$ROOT/1.1_asset_inventory/asset_inventory.csv" \
  --output-dir "$ROOT/1.2_trajectory_coverage" \
  --reuse-manual-labels

"$PYTHON_BIN" tools/stage1/build_dataset_v01.py \
  --config "$CONFIG" \
  --inventory "$ROOT/1.1_asset_inventory/asset_inventory.csv" \
  --tags "$ROOT/1.2_trajectory_coverage/trajectory_tags.csv" \
  --path-signatures "$ROOT/1.2_trajectory_coverage/path_signatures.jsonl" \
  --output-dir "$ROOT/1.3_dataset_v0.1"

"$PYTHON_BIN" tools/stage1/select_graph_tasks.py \
  --config "$CONFIG" \
  --task-summary "$ROOT/1.2_trajectory_coverage/task_structure_summary.csv" \
  --coverage "$ROOT/1.2_trajectory_coverage/coverage_matrix.csv" \
  --tags "$ROOT/1.2_trajectory_coverage/trajectory_tags.csv" \
  --manifest "$ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" \
  --output-dir "$ROOT/1.4_g0_decision"

echo "Stage 1 finished. Read: $ROOT/1.4_g0_decision/g0_decision.md"
BASH

chmod +x "$REPO_ROOT/tools/stage1/run_stage1.sh"
```

重跑命令：

```bash
cd "$REPO_ROOT"
PYTHON_BIN="$PYTHON_BIN" bash tools/stage1/run_stage1.sh "$STAGE1_CONFIG"
```

# 阶段 1 最终目录

```text
artifacts/pathgraph_sarm/stage1/
├── 1.1_asset_inventory/
├── 1.2_trajectory_coverage/
├── 1.3_dataset_v0.1/
└── 1.4_g0_decision/
```

# 阶段 1 完成判定

满足以下四项即可进入阶段 2，不追加无关工作：

1. 能从 manifest 稳定读取完整 episode 历史。
2. 已生成任务结构覆盖矩阵，能够定位合法不同顺序和恢复 episode。
3. 已确定两个目标任务，或给出可立即执行的关键 edge 定向补采表。
4. `g0_decision.md` 明确给出 GO、GO_WITH_TARGETED_COLLECTION 或 SWITCH。

**核心点：阶段 1 的结果不是“做完一轮审计”，而是形成一个可训练的数据版本、两项明确任务和阶段 2 可直接读取的交接文件。**
