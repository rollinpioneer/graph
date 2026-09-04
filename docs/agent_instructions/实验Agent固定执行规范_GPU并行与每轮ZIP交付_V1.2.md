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
