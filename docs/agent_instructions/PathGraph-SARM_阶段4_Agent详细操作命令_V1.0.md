# PathGraph-SARM 阶段 4 Agent 详细操作命令 V1.0

- 入口状态：`G1=GO_STAGE4`
- 阶段名称：图状态、节点内进度与剩余代价模型开发
- 文档形式：总体规范 + 6 个独立小阶段
- 固定执行：GPU 提权查看、可并行则多 GPU 并行、每轮生成 ZIP、checkpoint/大文件可不打包

> Agent 必须按本文顺序执行。每个小阶段完成并产出对应 ZIP 后再进入下一小阶段；不得跳过 validation-only selection lock，也不得提前使用 test 调参。



---

<!-- BEGIN FILE: README.md -->

# PathGraph-SARM 阶段 4 Agent 操作文档包 V1.0

## 执行顺序

Agent 必须按下列顺序读取并执行：

```text
00_阶段3验收与阶段4入口结论.md
01_阶段4通用执行规范.md
阶段4.1_冻结监督数据与历史输入.md
阶段4.2_历史编码器与NodeEdgeHeads.md
阶段4.3_节点内进度Phi.md
阶段4.4_RemainingCost.md
阶段4.5_联合训练与模型选择.md
阶段4.6_不确定性与冻结.md
```

`01_阶段4通用执行规范.md` 中的环境变量、GPU 查询、多 GPU 调度器、轻量 ZIP 打包器、总配置和输出契约只需创建一次；后续六个小阶段直接复用。

## 阶段目标

阶段 4 只开发并冻结以下模型输出：

```text
node belief
edge-type / edge-id belief
within-node progress phi
remaining cost C_G
ensemble uncertainty
```

不在本阶段组合最终 graph reward，不调 loop penalty，不训练 RA-BC。

## 必须遵守的运行规则

1. 每个训练小阶段开始时，先运行 `sudo -n nvidia-smi` 路径的 GPU 查询脚本；免密 sudo 不可用时记录并使用直接 `nvidia-smi`，不得据此判断没有 GPU。
2. seed、history、loss 变体、test 推理和 probe 可独立运行时，采用一个 job 一张 GPU 的并行方式。
3. 每个小阶段完成后立即生成一个轻量 ZIP；checkpoint、模型权重、episode、视频、大数组和其他大文件可不打包，但必须写 manifest。
4. 只执行与推进当前阶段直接相关的检查；不要加入全仓库审计、自动图发现、无关泛化测试或大规模调参。
5. validation 用于模型选择；test/diagnostic 只能在 `selection_lock.json` 生成后运行。
6. `transport_dual_order` 当前按两个唯一 content group 作为 mechanism probe，不作为大样本泛化结论。

## 每轮交付

```text
stage4_1_supervision_and_encoder_input.zip
stage4_2_node_edge_heads.zip
stage4_3_within_node_progress.zip
stage4_4_remaining_cost.zip
stage4_5_joint_model_selection.zip
stage4_6_uncertainty_and_freeze.zip
stage4_complete.zip
```

## 阶段出口

阶段 4.6 生成：

```text
artifacts/pathgraph_sarm/stage4/model_candidates_v1/stage4_exit_decision.md
artifacts/pathgraph_sarm/stage4/model_candidates_v1/stage5_handoff.md
artifacts/pathgraph_sarm/stage4/downloads/stage4_complete.zip
```

只有 `stage4_exit_decision.md` 为 `GO_STAGE5` 时，阶段 4 才正式完成。

<!-- END FILE: README.md -->


---

<!-- BEGIN FILE: 00_阶段3验收与阶段4入口结论.md -->

# PathGraph-SARM 阶段 3 验收与阶段 4 入口结论

## 结论

阶段 3 可以正式关闭，允许进入阶段 4。正式状态为：

```text
M2 = PROBLEM_CONFIRMED
G1 = GO_STAGE4
Stage 4 entry = ALLOWED
```

本次对 `stage3_complete.zip` 的实际复核结果：

- 文件 SHA256：`a3b65d58207cc43fc58633b6f59a5f72346e9327ab2d8625102934e1b874b5a3`；
- `unzip -t` 完整性检查通过；
- `stage3/m2_freeze_v1/g1_decision.md` 明确写入 `GO_STAGE4`；
- `g1_gate.json` 中 `sanity_pass=true`；
- learned canonical-control monotonicity 为 `0.9558823529`，超过阶段 3 的 `0.90` 门槛；
- 已确认 4 个结构性误评分签名：`ALT_ORDER_NEGATIVE`、`POSITIVE_CYCLE`、`RECOVERY_NOT_REWARDED`、`TIME_REWARDS_FAILURE`；
- 9 个 learned baseline job 均完成 CUDA 训练；
- M1、runtime adapter、diagnostic suite 和 M2 checksum 均已冻结；
- 所有统计继续以 `content_group_id` 为独立单位，重复的 scripted-oracle episode 不作为独立证据；
- checkpoint 已按约定从 ZIP 中排除并保留 manifest。

因此，阶段 3 已完成其职责：证明固定 stage chain / 单一全局 progress 对合法 alternative order、failure/recovery 和循环存在可复现的结构性误评分。阶段 4 可以开始训练历史条件化的图状态、节点内进度和 remaining-cost 模型。

## 阶段 4 必须继承的证据边界

1. `transport_recovery` 具有可用于 train/val/test 的唯一内容组，是阶段 4 learned model 选择与泛化评估的主任务。
2. `transport_dual_order` 当前只有 2 个唯一数值内容组，分别代表 A-first 和 B-first；其结果只能作为结构机制探针，不能据此声称大规模跨轨迹泛化。
3. 所有 checkpoint 选择只能使用冻结验证集；Stage 3 test diagnostic 和 Stage 4 test 只能在候选配置冻结后运行。
4. 阶段 4 不组合最终 reward，不调整 loop penalty，不训练 RA-BC；这些属于阶段 5 和阶段 6。
5. 阶段 4 必须输出：node belief、edge belief、within-node progress、remaining cost，以及能够在阶段 5 计算均值/方差的不确定性接口。

## 阶段 4 入口输入

```text
artifacts/pathgraph_sarm/stage3/m2_freeze_v1/
artifacts/pathgraph_sarm/stage3/input_adapter_v1/
artifacts/pathgraph_sarm/stage3/diagnostic_suite_v1/
artifacts/pathgraph_sarm/stage3/rounds/stage3_3_baseline_runs/
artifacts/pathgraph_sarm/stage3/rounds/stage3_4_misscoring_analysis/
```

**核心结论：阶段 3 已通过 G1，可以进入阶段 4；阶段 4 直接推进图状态与剩余代价模型，不再重复阶段 3 的问题成立性实验。**

<!-- END FILE: 00_阶段3验收与阶段4入口结论.md -->


---

<!-- BEGIN FILE: 01_阶段4通用执行规范.md -->

# PathGraph-SARM 阶段 4：通用执行规范与目录

> 阶段名称：图状态与剩余代价模型开发。  
> 阶段入口：阶段 3 已完成，`G1=GO_STAGE4`。  
> 阶段产物里程碑：`M2.5 = MODEL_CANDIDATES_READY`。  
> 下一正式决策门：阶段 5 的 `G2`；阶段 4 不冒用 `G2`。  
> 总体目标：用同一个历史条件化编码器，训练并输出 node belief、edge belief、节点内进度 `φ`、到成功的 remaining cost `C_G`，再用多 seed ensemble 提供不确定性。

## 给 Agent 的总命令

在 `/home/xushijie/CUPID` 中继续执行，不重新运行阶段 1—3。先校验 M2 冻结输入并构建泄漏安全的监督数据；随后依次训练 node/edge、within-node progress、remaining-cost 和联合模型；最后用三个独立 seed 组成 ensemble，冻结阶段 5 可直接调用的推理接口。

实验推进优先。只做以下必要检查：冻结输入 checksum、内容组划分、标签有限性、训练是否产生 NaN/Inf、验证指标与候选 checkpoint 可读取。不要安排全仓库审计、无关重构、大范围超参数搜索、自动图发现、RA-BC 或与阶段 4 无关的策略 rollout。

只要不同 seed、history 配置、loss 变体或任务评估之间没有共享写冲突，就必须多 GPU 并行。GPU 状态先尝试提权查看；没有免密 sudo 时，记录情况并使用可用的直接 `nvidia-smi` 快照，不得因为 `sudo -n` 失败就判断没有 GPU。

每个小阶段结束后立即生成轻量 ZIP。checkpoint、模型权重、原始 episode、视频、缓存、`.npz` 大数组和超过阈值的其他文件默认不打包，只写入 manifest。阶段 4 最后额外生成 `stage4_complete.zip`。

## 阶段 4 小阶段与每轮 ZIP

| 小阶段 | 总体上要完成的工作 | 本轮 ZIP |
|---|---|---|
| 4.1 | 冻结 Stage 4 输入，构建 frame/window 监督数据、标签映射、cost target 和跨顺序 probe | `stage4_1_supervision_and_encoder_input.zip` |
| 4.2 | 训练历史编码器与 node/edge heads，比较 history=1 与 history=32 | `stage4_2_node_edge_heads.zip` |
| 4.3 | 训练节点内进度 `φ`，验证节点内排序和单调性 | `stage4_3_within_node_progress.zip` |
| 4.4 | 训练 `C_G` remaining-cost head，比较普通回归与结构约束版本 | `stage4_4_remaining_cost.zip` |
| 4.5 | 联合微调、只用验证集选模型，并在冻结 test/probe 上一次性评估 | `stage4_5_joint_model_selection.zip` |
| 4.6 | 组成 ensemble，输出不确定性接口，冻结模型候选集和 Stage 5 handoff | `stage4_6_uncertainty_and_freeze.zip` |

阶段全部完成后生成：

```text
stage4_complete.zip
```

## 阶段 4 总体完成条件

1. 监督数据使用 `content_group_id` 管理重复内容；`transport_recovery` 保持原 train/val/test，`transport_dual_order` 明确标为 mechanism probe。
2. 模型输入是因果历史窗口，不读取 `outcome`、`success`、`scenario`、`controller_source`、`episode_id` 或未来帧。
3. 必须输出 task-conditioned node probability、edge-type probability、edge-id probability、`φ∈[0,1]` 和非负 `C_G`。
4. node/edge 模型至少跑 3 个 seed；history=1 与 history=32 使用相同预算并行比较。
5. `C_G` 在成功终点附近最低；failure 后上升、有效 recovery 后下降；验证集排序指标稳定。
6. 联合模型选择只读取 validation 指标；test 与 Stage 3 diagnostic 仅在候选配置冻结后执行一次。
7. 三个独立 seed 组成 ensemble，导出分类概率均值、预测熵/互信息，以及 `φ`、`C_G` 的均值和标准差。
8. 每轮 ZIP、SHA256 和完整性检查均通过；checkpoint 等大文件保留路径清单但不强制打包。

## 统一目录与环境变量

Agent 先执行：

```bash
set -euo pipefail

export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export M2_ROOT="$STAGE3_ROOT/m2_freeze_v1"
export ADAPTER_ROOT="$STAGE3_ROOT/input_adapter_v1"
export DIAG_ROOT="$STAGE3_ROOT/diagnostic_suite_v1"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export STAGE4_SEEDS="${STAGE4_SEEDS:-20260906,20260907,20260908}"
export GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_ROOT"
mkdir -p \
  "$REPO_ROOT/configs/stage4" \
  "$REPO_ROOT/tools/stage4/lib" \
  "$STAGE4_ROOT/_runtime" \
  "$STAGE4_ROOT/rounds" \
  "$STAGE4_ROOT/downloads" \
  "$STAGE4_ROOT/supervision_v1" \
  "$STAGE4_ROOT/model_candidates_v1"
```

## 仅在 Stage 3 产物未落盘时恢复文件

正常情况下直接使用现有目录。若 `$M2_ROOT` 不存在，但项目目录中已有 `stage3_complete.zip`，只解压产物，不重跑阶段 3：

```bash
if [ ! -f "$M2_ROOT/g1_decision.md" ]; then
  export STAGE3_ZIP="${STAGE3_ZIP:-$REPO_ROOT/stage3_complete.zip}"
  test -f "$STAGE3_ZIP"
  TMP_STAGE3="$(mktemp -d)"
  unzip -q "$STAGE3_ZIP" -d "$TMP_STAGE3"
  test -f "$TMP_STAGE3/stage3/m2_freeze_v1/g1_decision.md"
  mkdir -p "$STAGE3_ROOT"
  rsync -a "$TMP_STAGE3/stage3/" "$STAGE3_ROOT/"
  if [ -d "$TMP_STAGE3/tools_stage3" ]; then
    mkdir -p "$REPO_ROOT/tools/stage3"
    rsync -a "$TMP_STAGE3/tools_stage3/" "$REPO_ROOT/tools/stage3/"
  fi
  rm -rf "$TMP_STAGE3"
fi
```

## 必要入口检查

```bash
test -f "$M2_ROOT/g1_decision.md"
grep -q 'GO_STAGE4' "$M2_ROOT/g1_decision.md"
test -f "$M2_ROOT/M2_SHA256SUMS.txt"
test -f "$M2_ROOT/stage4_handoff.md"
test -f "$ADAPTER_ROOT/INPUT_ADAPTER_SHA256SUMS.txt"
test -f "$ADAPTER_ROOT/runtime_episode_annotations.jsonl"
test -f "$ADAPTER_ROOT/stage3_episode_index.csv"
test -d "$ADAPTER_ROOT/runtime_graph_specs_v1.0.1"
test -f "$DIAG_ROOT/DIAGNOSTIC_SUITE_SHA256SUMS.txt"

(
  cd "$M2_ROOT"
  sha256sum -c M2_SHA256SUMS.txt
)
(
  cd "$ADAPTER_ROOT"
  sha256sum -c INPUT_ADAPTER_SHA256SUMS.txt
)
(
  cd "$DIAG_ROOT"
  sha256sum -c DIAGNOSTIC_SUITE_SHA256SUMS.txt
)

sed -n '1,200p' "$M2_ROOT/stage4_handoff.md"
```

若 checksum 失败，只修复缺失或被误移动的阶段 3 文件；不要重新调阶段 3 模型，也不要改写冻结文件。

## GPU 必须先尝试提权查看

先创建以下脚本：

```bash
cat > "$REPO_ROOT/tools/stage4/query_gpus.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:?Usage: query_gpus.sh OUT_DIR}"
MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
mkdir -p "$OUT_DIR"
QUERY_FIELDS='index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu'

if sudo -n nvidia-smi >/dev/null 2>&1; then
  echo 'mode=sudo_noninteractive' | tee "$OUT_DIR/query_mode.txt"
  sudo -n nvidia-smi | tee "$OUT_DIR/nvidia_smi_full.txt"
  sudo -n nvidia-smi --query-gpu="$QUERY_FIELDS" --format=csv,noheader,nounits | tee "$OUT_DIR/gpu_status.csv"
elif nvidia-smi >/dev/null 2>&1; then
  {
    echo 'mode=direct_fallback'
    echo 'note=sudo -n unavailable; if an interactive privileged shell is available, run sudo nvidia-smi once and retain its output. Direct visibility is sufficient to continue CUDA jobs.'
  } | tee "$OUT_DIR/query_mode.txt"
  nvidia-smi | tee "$OUT_DIR/nvidia_smi_full.txt"
  nvidia-smi --query-gpu="$QUERY_FIELDS" --format=csv,noheader,nounits | tee "$OUT_DIR/gpu_status.csv"
else
  echo 'mode=unavailable' | tee "$OUT_DIR/query_mode.txt"
  : > "$OUT_DIR/gpu_status.csv"
  : > "$OUT_DIR/available_gpu_ids.txt"
  exit 3
fi

awk -F',' -v min="$MIN_FREE_MB" '{
  gsub(/ /,"",$1); gsub(/ /,"",$6);
  if (($6+0) >= min) print $1
}' "$OUT_DIR/gpu_status.csv" > "$OUT_DIR/available_gpu_ids.txt"

echo "GPU_MIN_FREE_MB=$MIN_FREE_MB" >> "$OUT_DIR/query_mode.txt"
echo "available_gpu_ids=$(paste -sd, "$OUT_DIR/available_gpu_ids.txt")" >> "$OUT_DIR/query_mode.txt"
SH
chmod +x "$REPO_ROOT/tools/stage4/query_gpus.sh"
```

每个训练小阶段开始时都运行：

```bash
GPU_SNAPSHOT_DIR="$STAGE4_ROOT/_runtime/gpu_$(date +%Y%m%d_%H%M%S)"
"$REPO_ROOT/tools/stage4/query_gpus.sh" "$GPU_SNAPSHOT_DIR"
cat "$GPU_SNAPSHOT_DIR/query_mode.txt"
cat "$GPU_SNAPSHOT_DIR/gpu_status.csv"
cat "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt"
```

若 Agent 处于可交互终端且持有 sudo 凭据，可额外执行：

```bash
sudo nvidia-smi | tee "$GPU_SNAPSHOT_DIR/nvidia_smi_interactive_sudo.txt"
```

## 创建轮次初始化脚本

```bash
cat > "$REPO_ROOT/tools/stage4/init_round.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
[ "$#" -ge 3 ] || { echo "Usage: $0 ROUND_ID ROUND_DIR PURPOSE" >&2; exit 2; }
ROUND_ID="$1"
ROUND_DIR="$2"
PURPOSE="$3"
mkdir -p "$ROUND_DIR"/{configs,logs,metrics,plots,jobs,manifests,tables,predictions}
GIT_COMMIT="$(git -C "${REPO_ROOT:-$(pwd)}" rev-parse HEAD 2>/dev/null || echo unavailable)"
TORCH_INFO="$(${PYTHON_BIN:-python} - <<'PY' 2>/dev/null || true
try:
 import torch
 print(f'torch={torch.__version__}; cuda={torch.cuda.is_available()}; cuda_version={torch.version.cuda}; devices={torch.cuda.device_count()}')
except Exception as e:
 print(f'torch_query_failed={e}')
PY
)"
cat > "$ROUND_DIR/run_manifest.md" <<EOF
# Run Manifest

- round_id: $ROUND_ID
- purpose: $PURPOSE
- started_at: $(date -Iseconds)
- repo_root: ${REPO_ROOT:-$(pwd)}
- git_commit: $GIT_COMMIT
- python: $(${PYTHON_BIN:-python} --version 2>&1)
- runtime: $TORCH_INFO
- statistics_unit: content_group_id
- checkpoint_packaging: omitted_by_default
EOF
SH
chmod +x "$REPO_ROOT/tools/stage4/init_round.sh"
```

## 创建轻量 ZIP 打包脚本

该脚本按扩展名和文件大小排除 checkpoint、原始数据与其他大文件；无论是否排除大文件，每轮都必须生成 ZIP。

```bash
cat > "$REPO_ROOT/tools/stage4/package_round.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, zipfile
from pathlib import Path

HEAVY_SUFFIXES = {
    '.ckpt', '.pt', '.pth', '.bin', '.safetensors', '.pkl', '.pickle',
    '.hdf5', '.h5', '.mp4', '.mov', '.avi', '.mkv', '.npz', '.npy'
}
CHECKPOINT_SUFFIXES = {'.ckpt', '.pt', '.pth', '.bin', '.safetensors'}
SKIP_DIR_NAMES = {'__pycache__', '.git', '.pytest_cache'}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--round-id', required=True)
    ap.add_argument('--round-dir', type=Path, required=True)
    ap.add_argument('--downloads-dir', type=Path, required=True)
    ap.add_argument('--max-file-mb', type=float, default=200.0)
    args = ap.parse_args()

    round_dir = args.round_dir.resolve()
    downloads = args.downloads_dir.resolve()
    downloads.mkdir(parents=True, exist_ok=True)
    for req in ('run_manifest.md', 'summary.md'):
        p = round_dir / req
        if not p.is_file() or p.stat().st_size == 0:
            raise SystemExit(f'missing required file: {p}')

    max_bytes = int(args.max_file_mb * 1024 * 1024)
    included, large_rows, checkpoint_rows = [], [], []
    for path in sorted(round_dir.rglob('*')):
        if not path.is_file() or any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        rel = path.relative_to(round_dir)
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if suffix in CHECKPOINT_SUFFIXES:
            checkpoint_rows.append((str(rel), size, '', '', '', 'checkpoint_omitted'))
            continue
        if suffix in HEAVY_SUFFIXES or size > max_bytes:
            reason = 'heavy_extension' if suffix in HEAVY_SUFFIXES else f'larger_than_{args.max_file_mb:g}MB'
            large_rows.append((str(rel), size, suffix or 'none', reason))
            continue
        included.append(path)

    ckpt_manifest = round_dir / 'checkpoint_manifest.tsv'
    if not ckpt_manifest.exists():
        ckpt_manifest.write_text('path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\tnote\n', encoding='utf-8')
    if checkpoint_rows:
        with ckpt_manifest.open('a', encoding='utf-8') as f:
            for row in checkpoint_rows:
                f.write('\t'.join(map(str, row)) + '\n')

    large_manifest = round_dir / 'large_file_manifest.tsv'
    with large_manifest.open('w', encoding='utf-8') as f:
        f.write('path\tsize_bytes\tsuffix\treason_omitted\n')
        for row in large_rows:
            f.write('\t'.join(map(str, row)) + '\n')

    included = [p for p in included if p not in {ckpt_manifest, large_manifest}]
    included += [ckpt_manifest, large_manifest]

    zip_path = downloads / f'{args.round_id}.zip'
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(set(included)):
            zf.write(path, path.relative_to(round_dir))
    with zipfile.ZipFile(zip_path, 'r') as zf:
        bad = zf.testzip()
        if bad is not None:
            raise SystemExit(f'zip integrity failure: {bad}')

    digest = sha256(zip_path)
    (zip_path.with_suffix(zip_path.suffix + '.sha256')).write_text(
        f'{digest}  {zip_path.name}\n', encoding='utf-8')
    (downloads / f'{args.round_id}_unzip_test.txt').write_text(
        'No errors detected in compressed data.\n', encoding='utf-8')
    print(zip_path)
    print(digest)

if __name__ == '__main__':
    main()
PY
chmod +x "$REPO_ROOT/tools/stage4/package_round.py"
```

统一打包命令：

```bash
"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

## 创建多 GPU 独立 job 调度器

阶段 4 数据规模不需要为单 job 强行改成 DDP。默认采用“一个独立配置占一张 GPU”的实验级并行；只有仓库现有训练器已经支持 `torchrun` 且单 job 显存/吞吐确实成为瓶颈时，才使用 DDP。

```bash
cat > "$REPO_ROOT/tools/stage4/launch_parallel.py" <<'PY'
#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, subprocess, time
from pathlib import Path

def read_jobs(path: Path):
    jobs = []
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            for key in ('job_id', 'command', 'output_dir'):
                if key not in row:
                    raise ValueError(f'{path}: missing {key} in {row}')
            jobs.append(row)
    return jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=Path, required=True)
    ap.add_argument('--gpu-ids-file', type=Path, required=True)
    ap.add_argument('--logs-dir', type=Path, required=True)
    ap.add_argument('--status-csv', type=Path, required=True)
    ap.add_argument('--max-jobs-per-gpu', type=int, default=1)
    ap.add_argument('--poll-seconds', type=float, default=2.0)
    args = ap.parse_args()

    jobs = read_jobs(args.jobs)
    gpu_ids = [x.strip() for x in args.gpu_ids_file.read_text().splitlines() if x.strip()]
    if not gpu_ids:
        raise SystemExit('No GPU id is available. Re-run query_gpus.sh after freeing GPUs.')
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.status_csv.parent.mkdir(parents=True, exist_ok=True)

    slots = []
    for gpu in gpu_ids:
        slots.extend([gpu] * args.max_jobs_per_gpu)
    pending = list(jobs)
    running = []
    rows = []

    while pending or running:
        while pending and len(running) < len(slots):
            used = [r['gpu'] for r in running]
            gpu = next((g for g in slots if used.count(g) < slots.count(g)), None)
            if gpu is None:
                break
            job = pending.pop(0)
            out_dir = Path(job['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.logs_dir / f"{job['job_id']}.log"
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            env['STAGE4_PHYSICAL_GPU_ID'] = str(gpu)
            log = log_path.open('w', encoding='utf-8')
            start = time.time()
            log.write(f"job_id={job['job_id']}\nphysical_gpu_id={gpu}\ncommand={job['command']}\n")
            log.flush()
            proc = subprocess.Popen(['bash', '-lc', job['command']], stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append({'proc': proc, 'job': job, 'gpu': gpu, 'log': log, 'log_path': log_path, 'start': start})

        time.sleep(args.poll_seconds)
        keep = []
        for item in running:
            code = item['proc'].poll()
            if code is None:
                keep.append(item)
                continue
            item['log'].close()
            rows.append({
                'job_id': item['job']['job_id'],
                'physical_gpu_id': item['gpu'],
                'exit_code': code,
                'seconds': round(time.time() - item['start'], 3),
                'output_dir': item['job']['output_dir'],
                'log': str(item['log_path']),
            })
        running = keep

    with args.status_csv.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['job_id','physical_gpu_id','exit_code','seconds','output_dir','log'])
        w.writeheader(); w.writerows(rows)
    failed = [r for r in rows if r['exit_code'] != 0]
    if failed:
        raise SystemExit(f'{len(failed)} jobs failed; see {args.status_csv}')

if __name__ == '__main__':
    main()
PY
chmod +x "$REPO_ROOT/tools/stage4/launch_parallel.py"
```

## 创建阶段 4 总配置

```bash
cat > "$STAGE4_CONFIG" <<'YAML'
project:
  repo_root: /home/xushijie/CUPID
  stage3_root: artifacts/pathgraph_sarm/stage3
  stage4_root: artifacts/pathgraph_sarm/stage4

inputs:
  m2_root: artifacts/pathgraph_sarm/stage3/m2_freeze_v1
  adapter_root: artifacts/pathgraph_sarm/stage3/input_adapter_v1
  diagnostic_suite: artifacts/pathgraph_sarm/stage3/diagnostic_suite_v1
  episode_index: artifacts/pathgraph_sarm/stage3/input_adapter_v1/stage3_episode_index.csv
  annotations: artifacts/pathgraph_sarm/stage3/input_adapter_v1/runtime_episode_annotations.jsonl
  graph_specs: artifacts/pathgraph_sarm/stage3/input_adapter_v1/runtime_graph_specs_v1.0.1

scope:
  graph_tasks: [transport_recovery, transport_dual_order]
  primary_learned_task: transport_recovery
  mechanism_probe_task: transport_dual_order
  statistics_unit: content_group_id
  representatives_only: true
  preserve_controller_source: true
  no_test_selection: true

features:
  mode: lowdim
  numeric_fields: [eef_pos, object_pos, target_pos, gripper_state, action]
  info_flags: [subgoal_A_done, subgoal_B_done]
  optional_contact_fields: [contact, contact_state, gripper_contact]
  forbidden_fields: [outcome, success, info.success, scenario, controller_source, episode_id, content_group_id]
  history_steps_primary: 32
  history_steps_control: 1
  causal_left_padding: zero
  add_absolute_episode_step: false
  add_future_features: false

splits:
  transport_recovery: use_split_original
  transport_dual_order: mechanism_probe_only
  dual_order_folds:
    - {fold_id: holdout_A_first, train_path: B_first, eval_path: A_first}
    - {fold_id: holdout_B_first, train_path: A_first, eval_path: B_first}

labels:
  edge_types: [none, forward, alternative, recovery, failure, stagnation]
  node_vocab: task_conditioned
  edge_id_vocab: task_conditioned
  edge_none_sampling_ratio: 3.0
  phi:
    source_priority: [progress_anchors, node_interval_fraction]
    range: [0.0, 1.0]
    terminal_node_value: 1.0
    exclude_transition_boundary_duplicates: true
  remaining_cost:
    source: observed_suffix_edge_cost
    edge_cost_field: base_step_cost
    within_node_residual_weight: 0.25
    normalize_by: per_task_train_p95
    success_value: 0.0
    terminal_failure_absolute_target: censored
    terminal_failure_pairwise_lower_bound: true
    max_normalized_cost: 2.0

model:
  encoder: gru
  input_projection_dim: 192
  hidden_dim: 192
  gru_layers: 1
  dropout: 0.10
  task_embedding_dim: 16
  layer_norm: true
  node_head: task_masked_linear
  edge_type_head: global_linear
  edge_id_head: task_masked_linear
  phi_head: node_conditioned_sigmoid
  cost_head: softplus

training:
  seeds: [20260906, 20260907, 20260908]
  device: cuda
  amp: true
  num_workers: 4
  batch_size: 256
  grad_clip_norm: 1.0
  optimizer: adamw
  learning_rate: 0.0003
  head_learning_rate: 0.0005
  weight_decay: 0.0001
  early_stop_patience: 12
  node_edge_epochs: 100
  phi_epochs: 60
  cost_epochs: 100
  joint_epochs: 80
  freeze_encoder_epochs_for_new_head: 5
  save_best_by_validation_only: true

loss:
  node_ce: 1.0
  edge_type_focal: 1.0
  edge_id_ce: 0.5
  phi_huber: 1.0
  phi_monotonic_rank: 0.5
  cost_huber: 1.0
  cost_temporal_rank: 0.5
  cost_bellman: 0.5
  failure_increase_rank: 1.0
  recovery_decrease_rank: 1.0
  recovery_no_overshoot: 0.5
  terminal_success_zero: 0.5
  focal_gamma: 2.0
  class_weight_cap: 10.0
  failure_margin: 0.10
  recovery_margin: 0.10
  recovery_overshoot_tolerance: 0.05

selection:
  validation_task: transport_recovery
  composite:
    node_macro_f1: 0.25
    edge_type_macro_f1_non_none: 0.20
    edge_id_macro_f1_positive: 0.10
    one_minus_phi_mae: 0.15
    cost_pair_accuracy: 0.15
    cost_spearman_scaled: 0.15
  tie_breakers: [failure_recall, recovery_recall, cost_mae, checkpoint_size]

thresholds:
  node_macro_f1_min: 0.70
  node_improvement_over_majority_min: 0.15
  edge_type_macro_f1_non_none_min: 0.55
  edge_improvement_over_majority_min: 0.15
  failure_recall_min: 0.60
  recovery_recall_min: 0.60
  phi_mae_max: 0.18
  phi_spearman_min: 0.65
  phi_monotonic_violation_max: 0.12
  cost_mae_max: 0.20
  cost_spearman_min: 0.70
  cost_pair_accuracy_min: 0.75
  failure_cost_increase_rate_min: 0.70
  recovery_cost_decrease_rate_min: 0.70
  terminal_success_cost_p90_max: 0.15
  required_seed_passes: 2

uncertainty:
  method: deep_ensemble
  ensemble_size: 3
  classification_temperature_scaling: true
  regression_interval_levels: [0.80, 0.90, 0.95]
  report_ece: true
  report_error_detection_auroc: true

packaging:
  max_file_mb: 200
  omit_checkpoints: true
  omit_raw_episodes: true
  omit_videos: true
  omit_array_caches: true
YAML
```

## 模型输出统一契约

所有训练和推理脚本必须使用同一输出键：

```text
node_logits             [B, N_global]，应用 task mask 后归一
node_probs              [B, N_global]
edge_type_logits        [B, 6]
edge_type_probs         [B, 6]
edge_id_logits          [B, E_global]，应用 task mask 后归一
edge_id_probs           [B, E_global]
phi                     [B]，范围 [0,1]
remaining_cost          [B]，非负
history_embedding       [B,H]
```

ensemble 推理额外输出：

```text
node_probs_mean
node_predictive_entropy
node_mutual_information
edge_type_probs_mean
edge_predictive_entropy
edge_mutual_information
phi_mean
phi_std
remaining_cost_mean
remaining_cost_std
```

所有 JSONL 预测行至少包含：

```text
episode_id, content_group_id, task_id, scenario, split, controller_source,
step, node_gt, node_pred, node_probs,
edge_type_gt, edge_type_pred, edge_type_probs,
edge_id_gt, edge_id_pred, edge_id_probs,
phi_gt, phi_pred, remaining_cost_gt, remaining_cost_pred,
seed, history_steps, model_id
```

## 大文件与 checkpoint 规则

- checkpoint、权重和优化器状态默认不进 ZIP；
- 每个选中 checkpoint 必须在 `checkpoint_manifest.tsv` 记录绝对路径、大小、seed、最佳 epoch、验证指标和可选 SHA256；
- 原始 episode、视频、数组缓存和超过 `ZIP_MAX_FILE_MB` 的文件默认不进 ZIP；
- 不得因大文件被排除而取消本轮 ZIP；
- ZIP 中必须保留足以重建实验的配置、命令、代码快照/哈希、日志摘要、指标、图表、预测样例和大文件清单。

<!-- END FILE: 01_阶段4通用执行规范.md -->


---

<!-- BEGIN FILE: 阶段4.1_冻结监督数据与历史输入.md -->

# 阶段 4.1：冻结监督数据与历史编码器输入
> **本小阶段固定执行规则**：执行前同时遵守 `01_阶段4通用执行规范.md`。GPU 作业先尝试提权查询；独立 job 在不产生共享写冲突时默认多 GPU 并行；本轮结束立即生成 ZIP；checkpoint、权重和其他大文件可以不打包，但必须写 manifest；只做推进本轮所需的最小检查。


## 总体上要干什么

把阶段 3 的 runtime graph、episode index 和 GT annotations 转成阶段 4 可直接训练的因果监督数据。此轮不训练正式模型，重点是固定输入、标签、cost target、内容组权重和 dual-order probe，防止后续各 head 使用不同数据口径。

本阶段结束时，Agent 必须生成冻结的 `supervision_v1`，供 4.2—4.6 只读使用。

## 本轮目录与初始化

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export M2_ROOT="$STAGE3_ROOT/m2_freeze_v1"
export ADAPTER_ROOT="$STAGE3_ROOT/input_adapter_v1"
export DIAG_ROOT="$STAGE3_ROOT/diagnostic_suite_v1"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export ROUND_ID="stage4_1_supervision_and_encoder_input"
export ROUND_DIR="$STAGE4_ROOT/rounds/$ROUND_ID"
export SUP_ROOT="$STAGE4_ROOT/supervision_v1"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export BUILD_WORKERS="${BUILD_WORKERS:-$($PYTHON_BIN - <<'PYWORKERS'
import os
print(max(1, min(os.cpu_count() or 1, 16)))
PYWORKERS
)}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage4/init_round.sh" "$ROUND_ID" "$ROUND_DIR" \
  "Build and freeze leakage-safe supervision, label maps, cost targets, and history input contract"
rm -rf "$SUP_ROOT.tmp"
mkdir -p "$SUP_ROOT.tmp"/{episodes,tables,configs,reports,manifests,probes}
```

## 4.1.1 实现监督数据构建脚本

创建 `tools/stage4/build_supervision_dataset.py`。Agent 必须按以下顺序实现，不得把未来结果或 metadata 塞进 feature：

1. 读取 `stage3_episode_index.csv`、`runtime_episode_annotations.jsonl` 和两个 runtime GraphSpec。
2. 只对存在 runtime annotation 的 graph task 建监督数据。
3. 以 `content_group_id` 为统计单位：
   - `transport_recovery` 默认只取 `is_representative=true` 的唯一内容组；
   - 保持 `split_original`，预期 unique content groups 为 train=32、val=7、test=13；
   - `transport_dual_order` 只保留两个唯一 representative，标记 `role=mechanism_probe`，不进入 validation model selection。
4. 从每条 raw episode 构建 `float32` 当前帧特征，只使用：`eef_pos`、`object_pos`、`target_pos`、`gripper_state`、`action` 和真实存在的 subgoal flags；若 contact 字段存在则记录到 schema 并使用，不存在则跳过，不阻塞。
5. 严禁读取或派生：`outcome`、`success`、`scenario`、`controller_source`、`episode_id`、`content_group_id`、绝对 episode step、未来帧。
6. 不预生成所有 history window。每个 episode 保存基础序列，训练 Dataset 在 `__getitem__` 中因果截取 history=1 或 32，并左侧补零。
7. 为每个 frame 生成：
   - `node_y`：所在 `node_interval` 的 task-conditioned node id；
   - `edge_type_y`：处于 `edge_interval` 时为语义类型，否则为 `none`；
   - `edge_id_y`：处于 edge interval 时为 task-conditioned edge id，否则为 `none`；
   - `phi_y`：优先按 `progress_anchors` 插值；缺失时使用当前 node interval 内的线性分数；范围裁剪到 `[0,1]`；
   - `cost_y_raw`：对最终成功 episode，从当前时刻到 success 的实际 suffix 中尚未完成的 `base_step_cost` 之和，加 `0.25(1-φ_t)` 的节点残差；
   - `cost_y_norm`：只用该任务 train split 的 `cost_y_raw` p95 做归一化，裁剪到 `[0,2]`；
   - `cost_mask`：成功 episode 的有效 frame 为 1；terminal-failure 的绝对回归 target 设为 censored，即 mask=0，但保留排序 pair。
8. 从 failure/recovery annotations 和 edge boundaries 生成 `cost_pairs.csv.gz`，至少包含：
   - `forward_decrease`：合法向前后 `C_after < C_before`；
   - `failure_increase`：failure 后 `C_after > C_before + margin`；
   - `recovery_decrease`：有效 recovery 后 `C_after < C_failure - margin`；
   - `recovery_no_overshoot`：恢复完成后的 cost 不应比 failure 前低超过 tolerance；
   - `terminal_success_zero`：success 附近 cost 接近 0。
9. 生成 dual-order 两个 probe fold：
   - `holdout_A_first`：训练视图含 recovery train + B-first，评估 A-first；
   - `holdout_B_first`：训练视图含 recovery train + A-first，评估 B-first。
   这两个 fold 不参与超参数选择，只用于 4.5 的结构机制报告。
10. 所有 label vocab 使用 task prefix，防止同名 node/edge 混淆；另外保存 task mask。

脚本 CLI 必须支持：

```text
--config
--episode-index
--annotations
--graph-spec-dir
--output-dir
--representatives-only
--statistics-unit
--num-workers
```

建议输出结构：

```text
supervision_v1/
  FROZEN.md
  SUPERVISION_SHA256SUMS.txt
  configs/resolved_stage4.yaml
  configs/feature_schema.json
  configs/label_maps.json
  configs/cost_target_spec.yaml
  tables/episode_manifest.csv
  tables/sample_index.csv.gz
  tables/cost_pairs.csv.gz
  tables/split_summary.csv
  tables/label_coverage.csv
  tables/content_group_overlap.csv
  probes/dual_order_folds.json
  episodes/<episode_id>.npz
  reports/build_summary.md
  reports/label_examples.jsonl
```

每个 episode `.npz` 至少保存：

```text
x, node_y, edge_type_y, edge_id_y,
phi_y, phi_mask,
cost_y_raw, cost_y_norm, cost_mask,
edge_positive_mask
```


每个 episode 的解码、特征拼接和标签落盘互相独立，`build_supervision_dataset.py` 必须使用进程池按 episode 并行；默认 worker 数取 CPU 核数与 16 的较小值。各 worker 只能写自己的临时 episode 文件，主进程负责最终 manifest、normalizer 和 checksum，避免并发写同一个 CSV。若仓库的 episode 解码器内部不是多进程安全的，则改为按 task 分两路进程执行，仍不要无理由串行处理全部 episode。

## 4.1.2 `cost_y_raw` 的具体构造顺序

对每条成功 episode：

1. 读取当前 task GraphSpec 中每个 edge 的 `base_step_cost`；缺失时使用 1.0。
2. 按 `edge_interval.end_step` 对实际发生的 edge 排序。
3. 对每个 frame `t`，累加所有 `end_step >= t` 的实际后缀 edge cost。
4. 加上当前 node 内残差 `0.25(1-φ_t)`。
5. 在 success node 内逐步降到 0；success interval 的最后一帧必须是 0。
6. 对 terminal failure episode 不伪造成功距离；保留 failure boundary pair 和 terminal lower-bound pair。
7. normalizer 只由 `transport_recovery/train` 和 dual probe 的训练视图分别计算，禁止用 val/test p95。

写入 `configs/cost_target_spec.yaml`：

```yaml
version: 1
source: observed_suffix_edge_cost
edge_cost_field: base_step_cost
fallback_edge_cost: 1.0
within_node_residual_weight: 0.25
normalization: per_task_train_p95
clip_normalized: [0.0, 2.0]
success_terminal_value: 0.0
terminal_failure_mode: censored_plus_pairwise
pair_margins:
  failure_increase: 0.10
  recovery_decrease: 0.10
  recovery_no_overshoot_tolerance: 0.05
```

## 4.1.3 实现最小验证脚本

创建 `tools/stage4/validate_supervision_dataset.py`，只检查直接关系到训练的事项：

- train/val/test 的 `content_group_id` 交集为 0；
- feature schema 中没有 forbidden field；
- 每个 sample 的标签范围合法；
- `phi` 有限且在 `[0,1]`；
- cost 有效位置有限且非负；
- success 终点 cost 为 0 或数值容差内接近 0；
- failure/recovery pair 非空；
- 两个 dual-order probe fold 均存在；
- `.npz` 数量与 episode manifest 一致。

CLI：

```text
--supervision-dir
--strict
--report-json
```

不做全仓库单元测试，不扫描无关任务。

## 4.1.4 构建监督数据

```bash
"$PYTHON_BIN" -m tools.stage4.build_supervision_dataset \
  --config "$STAGE4_CONFIG" \
  --episode-index "$ADAPTER_ROOT/stage3_episode_index.csv" \
  --annotations "$ADAPTER_ROOT/runtime_episode_annotations.jsonl" \
  --graph-spec-dir "$ADAPTER_ROOT/runtime_graph_specs_v1.0.1" \
  --output-dir "$SUP_ROOT.tmp" \
  --representatives-only \
  --statistics-unit content_group_id \
  --num-workers "$BUILD_WORKERS" \
  2>&1 | tee "$ROUND_DIR/logs/build_supervision.log"

"$PYTHON_BIN" -m tools.stage4.validate_supervision_dataset \
  --supervision-dir "$SUP_ROOT.tmp" \
  --strict \
  --report-json "$ROUND_DIR/metrics/supervision_validation.json" \
  2>&1 | tee "$ROUND_DIR/logs/validate_supervision.log"
```

## 4.1.5 生成必要统计与冻结

创建 `tools/stage4/summarize_supervision.py`，输出：

- 按 task/split/scenario 的 episode、content-group、frame 数；
- 各 node、edge type、edge id 的 frame 数；
- failure/recovery pair 数；
- 每任务 cost p95 normalizer；
- history-required node 的样本量；
- dual-order probe 的两个唯一 content group id。

运行：

```bash
"$PYTHON_BIN" -m tools.stage4.summarize_supervision \
  --supervision-dir "$SUP_ROOT.tmp" \
  --output-csv "$ROUND_DIR/tables/supervision_summary.csv" \
  --output-md "$ROUND_DIR/summary.md"

cp "$STAGE4_CONFIG" "$ROUND_DIR/configs/stage4.yaml"
cp "$SUP_ROOT.tmp/configs/feature_schema.json" "$ROUND_DIR/configs/"
cp "$SUP_ROOT.tmp/configs/label_maps.json" "$ROUND_DIR/configs/"
cp "$SUP_ROOT.tmp/configs/cost_target_spec.yaml" "$ROUND_DIR/configs/"
cp "$SUP_ROOT.tmp/tables/split_summary.csv" "$ROUND_DIR/tables/"
cp "$SUP_ROOT.tmp/tables/label_coverage.csv" "$ROUND_DIR/tables/"
cp "$SUP_ROOT.tmp/tables/content_group_overlap.csv" "$ROUND_DIR/tables/"
cp "$SUP_ROOT.tmp/probes/dual_order_folds.json" "$ROUND_DIR/configs/"

cat > "$SUP_ROOT.tmp/FROZEN.md" <<EOF
# Supervision v1 frozen

- frozen_at: $(date -Iseconds)
- source_m2: $M2_ROOT
- statistics_unit: content_group_id
- primary_validation_task: transport_recovery
- dual_order_role: mechanism_probe_only
- future_features: forbidden
- test_selection: forbidden
EOF

(
  cd "$SUP_ROOT.tmp"
  find . -type f ! -name 'SUPERVISION_SHA256SUMS.txt' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SUPERVISION_SHA256SUMS.txt
)
rm -rf "$SUP_ROOT"
mv "$SUP_ROOT.tmp" "$SUP_ROOT"
```

在 `run_manifest.md` 追加完成时间与输入 checksum：

```bash
{
  echo "- finished_at: $(date -Iseconds)"
  echo "- supervision_root: $SUP_ROOT"
  echo "- supervision_checksum: $(sha256sum "$SUP_ROOT/SUPERVISION_SHA256SUMS.txt" | awk '{print $1}')"
} >> "$ROUND_DIR/run_manifest.md"
```

## 本轮完成条件

- `supervision_validation.json` 为通过；
- content-group 跨 split 交集为 0；
- `transport_recovery` 的 train/val/test 均有数据；
- node/edge/phi/cost 标签均非空；
- failure/recovery pair 均非空；
- dual-order 两个 probe fold 已生成；
- `SUPERVISION_SHA256SUMS.txt` 可通过 `sha256sum -c`；
- 不要求把 `.npz` 打入 ZIP。

## 生成本轮 ZIP

```bash
"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

本轮停止点：`supervision_v1` 冻结后立即进入 4.2，不做额外数据审计。

<!-- END FILE: 阶段4.1_冻结监督数据与历史输入.md -->


---

<!-- BEGIN FILE: 阶段4.2_历史编码器与NodeEdgeHeads.md -->

# 阶段 4.2：历史编码器与 Node / Edge Heads
> **本小阶段固定执行规则**：执行前同时遵守 `01_阶段4通用执行规范.md`。GPU 作业先尝试提权查询；独立 job 在不产生共享写冲突时默认多 GPU 并行；本轮结束立即生成 ZIP；checkpoint、权重和其他大文件可以不打包，但必须写 manifest；只做推进本轮所需的最小检查。


## 总体上要干什么

训练共享历史编码器和三个分类头：当前 node、当前 edge type、当前 edge id。用完全相同的预算比较 `history_steps=1` 与 `history_steps=32`，证明模型不是依赖未来或 episode 元数据，并选择后续 `φ` 与 `C_G` 使用的 encoder 初始化。

正式训练矩阵为：

```text
2 个 history 配置 × 3 个 seed = 6 个独立 GPU job
```

只要有多张可用 GPU，六个 job 按 GPU 槽位并行排队，不串行等待。

## 本轮目录与 GPU 快照

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export SUP_ROOT="$STAGE4_ROOT/supervision_v1"
export ROUND_ID="stage4_2_node_edge_heads"
export ROUND_DIR="$STAGE4_ROOT/rounds/$ROUND_ID"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage4/init_round.sh" "$ROUND_ID" "$ROUND_DIR" \
  "Train history-conditioned node, edge-type, and edge-id heads; compare history 1 vs 32"
GPU_SNAPSHOT_DIR="$ROUND_DIR/manifests/gpu_snapshot"
"$REPO_ROOT/tools/stage4/query_gpus.sh" "$GPU_SNAPSHOT_DIR"
test -s "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt"
(
  cd "$SUP_ROOT"
  sha256sum -c SUPERVISION_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/supervision_checksum.log"
```

## 4.2.1 实现 Dataset、模型和训练器

创建以下文件：

```text
tools/stage4/lib/data.py
tools/stage4/lib/model.py
tools/stage4/lib/losses.py
tools/stage4/lib/metrics.py
tools/stage4/train_node_edge.py
tools/stage4/evaluate_node_edge.py
tools/stage4/summarize_node_edge.py
```

### Dataset 必须实现

- 从 `episode_manifest.csv` 和 episode `.npz` 读取基础序列；
- 仅在取样时构建因果窗口；
- history=32 时，窗口为 `[t-31,...,t]`，开头不足部分补零；
- history=1 时只取当前帧；
- 返回 `task_id` 和 task mask，但不返回 scenario/outcome 给模型；
- 训练 batch 由两部分构成：50% uniform frame、50% edge-positive frame；如果 edge-positive 数量不足，则按有放回采样正例，不把大量 `none` 淹没关键边；
- 对同一 content group 的样本总权重归一，不因 episode 复制次数放大；
- validation/test 不随机采样，按完整 frame 顺序推理；
- `transport_dual_order` 在本轮可以进入训练视图，但它不参与 validation early stopping 和超参数选择。

### 模型必须实现

```text
input projection -> LayerNorm -> ReLU -> GRU -> final causal hidden state
                                           ├─ task-masked node head
                                           ├─ global edge-type head
                                           └─ task-masked edge-id head
```

具体要求：

- `node_probs` 只在该 task 合法 node 上归一；
- `edge_id_probs` 只在该 task 合法 edge 和 `none` 上归一；
- edge type 类别固定为：`none, forward, alternative, recovery, failure, stagnation`；
- 不得把 GT node、GT edge、future frame、绝对时间步作为 head 输入；
- task embedding 可以输入所有 head；
- checkpoint 必须保存 `feature_schema`、`label_maps`、normalizer、history length 和 model config hash。

### 分类损失


a. Node head：

\[
\mathcal{L}_{node}=\operatorname{CE}(p_\theta(z_t\mid H_t),z_t^*).
\]

b. Edge-type head：

\[
\mathcal{L}_{edge\_type}=\operatorname{FocalCE}(p_\theta(e_t^{type}\mid H_t),e_t^{type,*}).
\]

c. Edge-id head：只在 `edge_positive_mask=1` 的帧上计算。

总损失：

\[
\mathcal{L}_{cls}
=\mathcal{L}_{node}
+\mathcal{L}_{edge\_type}
+0.5\mathcal{L}_{edge\_id}.
\]

要求：

- node class weight 使用 inverse-sqrt frequency，最大权重 10；
- edge type 使用 focal loss，`gamma=2`；
- edge id 使用 inverse-sqrt weight；
- `none` 类不进入 non-none macro F1；
- AMP、gradient clipping 和 early stopping 必须可配置；
- 每个 epoch 保存 train/val loss；只保留 validation composite 最佳 checkpoint 和最后 checkpoint；checkpoint 默认不进 ZIP。

训练 CLI：

```text
python -m tools.stage4.train_node_edge
  --config ...
  --supervision-dir ...
  --history-steps {1,32}
  --seed ...
  --output-dir ...
  [--max-epochs ...]
  [--limit-train-batches ...]
  [--limit-val-batches ...]
```

评估 CLI：

```text
python -m tools.stage4.evaluate_node_edge
  --config ...
  --supervision-dir ...
  --checkpoint ...
  --split val
  --output-dir ...
```

每个正式 job 输出：

```text
resolved_config.yaml
train_metrics.csv
val_metrics.json
val_predictions.jsonl
node_confusion_matrix.csv
edge_type_confusion_matrix.csv
edge_id_confusion_matrix.csv
per_class_metrics.csv
plots/node_confusion_matrix.png
plots/edge_type_confusion_matrix.png
checkpoints/best.pt
checkpoints/final.pt
DONE
```

## 4.2.2 必须计算的 validation 指标

```text
node_macro_f1
node_micro_f1
node_accuracy
node_history_required_macro_f1
edge_type_macro_f1_non_none
edge_type_accuracy_all
edge_id_macro_f1_positive
failure_precision / recall / f1
recovery_precision / recall / f1
alternative_precision / recall / f1（probe-only，不能作为主要统计结论）
```

同时生成 majority baseline：

- node majority：在 train 中每个 task 的最高频 node；
- edge-type majority：全部预测 `none`，并额外提供“非 none 子集上的最高频 edge type”基线；
- edge-id majority：在正 edge 帧中预测最高频 edge id。

## 4.2.3 先做一次最小 smoke run

仅使用一个 seed、2 epochs，验证接口能跑通；通过后删除 smoke checkpoint，避免混入正式候选：

```bash
SMOKE_DIR="$ROUND_DIR/jobs/_smoke_history32"
"$PYTHON_BIN" -m tools.stage4.train_node_edge \
  --config "$STAGE4_CONFIG" \
  --supervision-dir "$SUP_ROOT" \
  --history-steps 32 \
  --seed 20260906 \
  --max-epochs 2 \
  --limit-train-batches 4 \
  --limit-val-batches 2 \
  --output-dir "$SMOKE_DIR" \
  2>&1 | tee "$ROUND_DIR/logs/smoke.log"
test -f "$SMOKE_DIR/DONE"
"$PYTHON_BIN" - <<PY
import json, math
p=json.load(open('$SMOKE_DIR/val_metrics.json'))
for k,v in p.items():
    if isinstance(v,(int,float)) and not math.isfinite(v):
        raise SystemExit(f'non-finite smoke metric: {k}={v}')
print('smoke metrics finite')
PY
rm -rf "$SMOKE_DIR/checkpoints"
```

不要运行仓库全量测试。

## 4.2.4 生成六个正式 job

```bash
JOBS_JSONL="$ROUND_DIR/configs/jobs.jsonl"
: > "$JOBS_JSONL"
for history in 1 32; do
  for seed in 20260906 20260907 20260908; do
    job_id="node_edge_h${history}_s${seed}"
    out="$ROUND_DIR/jobs/$job_id"
    cmd="$PYTHON_BIN -m tools.stage4.train_node_edge --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --history-steps '$history' --seed '$seed' --output-dir '$out'"
    "$PYTHON_BIN" - <<PY >> "$JOBS_JSONL"
import json
print(json.dumps({"job_id": "$job_id", "command": "$cmd", "output_dir": "$out"}))
PY
  done
done
cat "$JOBS_JSONL"
```

## 4.2.5 多 GPU 并行运行

```bash
"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$JOBS_JSONL" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/jobs" \
  --status-csv "$ROUND_DIR/metrics/job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

若某一个 job 失败，只重跑该 job；不要重跑已完成 job：

```bash
awk -F, 'NR>1 && $3 != 0 {print $1}' "$ROUND_DIR/metrics/job_status.csv"
```

重跑单 job 示例：

```bash
FAILED_JOB="node_edge_h32_s20260907"
FAILED_CMD="$(python - <<PY
import json
for line in open('$JOBS_JSONL'):
    row=json.loads(line)
    if row['job_id']=='$FAILED_JOB':
        print(row['command']); break
PY
)"
CUDA_VISIBLE_DEVICES="$(head -n1 "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt")" \
  bash -lc "$FAILED_CMD" 2>&1 | tee "$ROUND_DIR/logs/jobs/${FAILED_JOB}_rerun.log"
```

## 4.2.6 汇总与选择 encoder 初始化

```bash
"$PYTHON_BIN" -m tools.stage4.summarize_node_edge \
  --jobs-root "$ROUND_DIR/jobs" \
  --supervision-dir "$SUP_ROOT" \
  --output-csv "$ROUND_DIR/tables/node_edge_seed_summary.csv" \
  --per-class-csv "$ROUND_DIR/tables/node_edge_per_class.csv" \
  --output-md "$ROUND_DIR/summary.md" \
  --plots-dir "$ROUND_DIR/plots" \
  --selection-json "$ROUND_DIR/metrics/node_edge_selection.json" \
  --checkpoint-manifest "$ROUND_DIR/checkpoint_manifest.tsv"
```

汇总必须报告：

- node macro/micro F1；
- edge type non-none macro F1；
- edge id positive macro F1；
- `failure`、`recovery` 单类 precision/recall/F1；
- history-required node 子集 F1；
- majority baseline；
- history=32 相对 history=1 的差值；
- 三个 seed 的均值、标准差和中位数；
- confusion matrix；
- 每个 checkpoint 的实际路径和大小。

选择规则：

1. 只看 `transport_recovery` validation；不读取 test 或 Stage 3 diagnostic。
2. 若 history=32 的中位 composite 不比 history=1 低超过 0.02，默认选择 history=32，遵循 Stage 3 handoff。
3. 若 history=32 明显更差，先只检查窗口时间对齐、left padding 和 hidden state 取值；修复后只重跑 history=32，不扩展到大量窗口搜索。
4. `transport_dual_order` 的两个内容组只作为训练覆盖/结构 probe，不用于 early stopping。
5. `node_edge_selection.json` 至少包含：

```json
{
  "selected_history_steps": 32,
  "selection_source": "transport_recovery_val_only",
  "selected_checkpoints": {
    "20260906": "/abs/path/to/best.pt",
    "20260907": "/abs/path/to/best.pt",
    "20260908": "/abs/path/to/best.pt"
  },
  "median_metrics": {},
  "history_comparison": {}
}
```

## 本轮完成条件

- 6 个正式 job 全部完成；
- 至少 2/3 个 history=32 seed 满足 node/edge 门槛，或汇总明确给出仅需在 4.5 联合训练修正的单一薄弱 head；
- node F1 高于 majority 至少 0.15，且中位数至少 0.70；
- edge type non-none F1 高于 non-none majority 至少 0.15，且中位数至少 0.55；
- failure 与 recovery 召回的中位数至少 0.60；
- history=1 与 history=32 对照已保存；
- 选择结果只基于 validation；
- checkpoint manifest 已写入，checkpoint 不要求打包。

## 生成本轮 ZIP

```bash
{
  echo "- finished_at: $(date -Iseconds)"
  echo "- selected_history_steps: $(python - <<PY
import json
p=json.load(open('$ROUND_DIR/metrics/node_edge_selection.json'))
print(p['selected_history_steps'])
PY
)"
} >> "$ROUND_DIR/run_manifest.md"

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

本轮停止点：完成 encoder/node/edge 候选后进入 4.3，不在本轮增加视觉 backbone、自动图或额外任务。

<!-- END FILE: 阶段4.2_历史编码器与NodeEdgeHeads.md -->


---

<!-- BEGIN FILE: 阶段4.3_节点内进度Phi.md -->

# 阶段 4.3：训练节点内进度 \(\phi\)
> **本小阶段固定执行规则**：执行前同时遵守 `01_阶段4通用执行规范.md`。GPU 作业先尝试提权查询；独立 job 在不产生共享写冲突时默认多 GPU 并行；本轮结束立即生成 ZIP；checkpoint、权重和其他大文件可以不打包，但必须写 manifest；只做推进本轮所需的最小检查。


## 总体上要干什么

在阶段 4.2 已选定的历史编码器与 node/edge 分类模型上增加节点内进度头，学习当前状态在**当前语义节点内部**的完成程度：

\[
\phi_t=\phi_\theta(H_t,\hat z_t)\in[0,1].
\]

本轮只解决“同一 node 内是否在向该 node 的出口推进”。不要在本轮构造最终 graph reward，也不要开始 RA-BC。训练时允许使用 GT node 作为辅助 teacher-forcing 信号，但正式 validation 指标必须使用模型预测的 node belief；否则会高估部署性能。

正式训练矩阵为：

```text
selected history 配置 × 3 个 seed = 3 个独立 GPU job
```

三个 job 无共享写目录，必须在可用多 GPU 上并行。

## 本轮目录与入口检查

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export SUP_ROOT="$STAGE4_ROOT/supervision_v1"
export NODE_EDGE_ROUND="$STAGE4_ROOT/rounds/stage4_2_node_edge_heads"
export ROUND_ID="stage4_3_within_node_progress"
export ROUND_DIR="$STAGE4_ROOT/rounds/$ROUND_ID"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage4/init_round.sh" "$ROUND_ID" "$ROUND_DIR" \
  "Train prediction-conditioned within-node progress phi from the selected history encoder"

(
  cd "$SUP_ROOT"
  sha256sum -c SUPERVISION_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/supervision_checksum.log"

test -f "$NODE_EDGE_ROUND/metrics/node_edge_selection.json"
cp "$NODE_EDGE_ROUND/metrics/node_edge_selection.json" \
  "$ROUND_DIR/configs/node_edge_selection.json"

SELECTED_HISTORY="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$NODE_EDGE_ROUND/metrics/node_edge_selection.json'))
print(int(p['selected_history_steps']))
PY
)"
echo "selected_history_steps=$SELECTED_HISTORY" | tee -a "$ROUND_DIR/run_manifest.md"

GPU_SNAPSHOT_DIR="$ROUND_DIR/manifests/gpu_snapshot"
"$REPO_ROOT/tools/stage4/query_gpus.sh" "$GPU_SNAPSHOT_DIR"
test -s "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt"
```

## 4.3.1 扩展模型：实现 prediction-conditioned \(\phi\) head

修改或扩展 `tools/stage4/lib/model.py`，不得另建一个与 4.2 不兼容的 encoder。模型须保留 4.2 的全部输出，再增加以下模块：

```text
history_embedding h_t
predicted node belief p(z_t | H_t)
node embedding table E_node
soft node context = p(z_t | H_t) @ E_node
concat[h_t, task_embedding, soft node context]
    -> MLP(192 -> 128 -> 1)
    -> sigmoid
    -> phi_pred
```

实现要求：

1. `node_probs` 必须经过 task mask 后再构造 soft node context。
2. 正式推理的 `phi` 只依赖预测的 `node_probs`；不能读取 `node_y`。
3. 训练时额外计算 `phi_oracle_node`：用 `node_y` 的 one-hot embedding 替换预测 belief，只作为辅助 loss 和诊断指标。
4. `phi` 和 `phi_oracle_node` 均用 sigmoid 限制在 `[0,1]`。
5. `phi` head 不接收绝对 step、episode 长度、outcome 或 future frame。
6. checkpoint 继续保存 feature schema、label maps、normalizer、history length、初始化 checkpoint 路径与 hash。
7. 4.3 的模型读取 4.2 checkpoint 时允许缺失 `phi_head.*`，除此之外不得静默忽略 key mismatch。

推荐 forward 接口：

```python
outputs = model(
    x=history_x,
    history_mask=history_mask,
    task_id=task_id,
    oracle_node_id=node_y_or_none,
)
# outputs 至少包含：
# node_logits, node_probs, edge_type_logits, edge_type_probs,
# edge_id_logits, edge_id_probs, history_embedding,
# phi, phi_oracle_node
```

## 4.3.2 实现 \(\phi\) 损失与采样

在 `tools/stage4/lib/losses.py` 中增加：

### 1. 预测节点条件下的主回归损失

\[
\mathcal L_{\phi,\mathrm{pred}}
=\operatorname{Huber}(\hat\phi_t,\phi_t^*),
\]

只在 `phi_mask=1` 的 frame 上计算。

### 2. GT 节点条件下的辅助回归损失

\[
\mathcal L_{\phi,\mathrm{oracle}}
=\operatorname{Huber}(\hat\phi_t^{oracle},\phi_t^*).
\]

该项权重固定为 `0.5`，目的是区分“phi head 本身不会”与“node 预测错误传导”。

### 3. 同一 node interval 内的排序损失

从同一 episode、同一连续 node interval 内采样 `(t_i,t_j)`，要求 `t_i<t_j` 且 GT 差值至少 `0.10`：

\[
\mathcal L_{mono}
=\max\left(0,\,m_\phi-(\hat\phi_{t_j}-\hat\phi_{t_i})\right),
\qquad m_\phi=0.05.
\]

不得跨 node boundary 生成 monotonic pair，因为 node 切换后 \(\phi\) 可以重新从低值开始。

### 4. 节点入口/出口弱约束

- 每个 node interval 最前 10% frame：目标接近 0；
- 每个非 terminal node interval 最后 10% frame：目标接近 1；
- terminal node 由监督数据中的 `phi_y` 决定，不另加跨节点单调约束。

使用 Huber 计算 endpoint loss，权重 `0.25`。

### 5. 分类保持项

encoder 在前 5 epoch 冻结。第 6 epoch 起解冻 encoder，但保留小权重 node/edge 分类 loss，避免只优化 \(\phi\) 后破坏阶段 4.2 的表征：

\[
\mathcal L_{anchor}=0.25\mathcal L_{node}
+0.15\mathcal L_{edge\_type}
+0.10\mathcal L_{edge\_id}.
\]

总损失：

\[
\mathcal L_{4.3}
=\mathcal L_{\phi,\mathrm{pred}}
+0.5\mathcal L_{\phi,\mathrm{oracle}}
+0.5\mathcal L_{mono}
+0.25\mathcal L_{endpoint}
+\mathcal L_{anchor}.
\]

采样器每个 batch 按下列比例构建：

```text
40% uniform valid-phi frames
30% node-boundary-near frames
20% history-required node frames
10% recovery/failure-adjacent frames
```

同一 `content_group_id` 的总抽样权重归一。validation/test 使用全量顺序帧，不随机重采样。

## 4.3.3 创建训练、评估和汇总脚本

创建：

```text
tools/stage4/train_phi.py
tools/stage4/evaluate_phi.py
tools/stage4/summarize_phi.py
tools/stage4/render_phi_traces.py
```

### `train_phi.py` CLI

```text
python -m tools.stage4.train_phi
  --config CONFIG
  --supervision-dir SUP_ROOT
  --init-checkpoint NODE_EDGE_BEST_PT
  --history-steps N
  --seed SEED
  --output-dir OUT
  [--max-epochs N]
  [--limit-train-batches N]
  [--limit-val-batches N]
```

训练顺序必须实现：

1. 加载 4.2 同 seed 的 selected checkpoint；若同 seed 不存在，立即报错，不从其他 seed 复制。
2. 初始化新 `phi_head`。
3. epoch 1—5：冻结 encoder 和分类 heads，只训练 `phi_head`。
4. epoch 6 起：解冻 encoder，encoder 学习率使用主学习率的 `0.25`；分类 heads继续参与 anchor loss。
5. early stopping 只看 `transport_recovery/val` 的：

\[
S_\phi=0.45(1-\mathrm{MAE})
+0.35\,\mathrm{SpearmanScaled}
+0.20(1-\mathrm{ViolationRate}).
\]

其中 `SpearmanScaled=(Spearman+1)/2`。
6. 只保留 best 和 final checkpoint；默认不打包。

每个正式 job 至少输出：

```text
resolved_config.yaml
train_metrics.csv
val_metrics.json
val_predictions.jsonl
per_node_metrics.csv
monotonic_pair_metrics.csv
boundary_metrics.csv
plots/phi_trace_examples.png
plots/phi_pred_vs_gt.png
plots/phi_error_by_node.png
checkpoints/best.pt
checkpoints/final.pt
DONE
```

### validation 必须计算

```text
phi_mae_predicted_node
phi_mae_oracle_node
phi_spearman_predicted_node
phi_spearman_oracle_node
phi_monotonic_violation_rate
phi_pair_accuracy
phi_boundary_start_mae
phi_boundary_end_mae
phi_mae_by_node
phi_mae_history_required_nodes
node_macro_f1_after_phi_training
edge_type_macro_f1_after_phi_training
```

定义：

- `phi_monotonic_violation_rate`：同一 node interval 的有效 pair 中，`phi_pred(t_j) < phi_pred(t_i)-0.05` 的比例；
- `phi_pair_accuracy`：预测顺序与 GT 顺序一致的 pair 比例；
- 正式门槛使用 `predicted_node` 指标，oracle 指标只用于定位误差来源。

## 4.3.4 先做一个最小 smoke run

```bash
FIRST_SEED=20260906
INIT_CKPT="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$NODE_EDGE_ROUND/metrics/node_edge_selection.json'))
print(p['selected_checkpoints'][str($FIRST_SEED)])
PY
)"
test -f "$INIT_CKPT"

SMOKE_DIR="$ROUND_DIR/jobs/_smoke_phi"
"$PYTHON_BIN" -m tools.stage4.train_phi \
  --config "$STAGE4_CONFIG" \
  --supervision-dir "$SUP_ROOT" \
  --init-checkpoint "$INIT_CKPT" \
  --history-steps "$SELECTED_HISTORY" \
  --seed "$FIRST_SEED" \
  --max-epochs 2 \
  --limit-train-batches 4 \
  --limit-val-batches 2 \
  --output-dir "$SMOKE_DIR" \
  2>&1 | tee "$ROUND_DIR/logs/smoke_phi.log"

test -f "$SMOKE_DIR/DONE"
"$PYTHON_BIN" - <<PY
import json, math
m=json.load(open('$SMOKE_DIR/val_metrics.json'))
for k,v in m.items():
    if isinstance(v,(int,float)) and not math.isfinite(v):
        raise SystemExit(f'non-finite smoke metric: {k}={v}')
print('phi smoke run passed')
PY
rm -rf "$SMOKE_DIR/checkpoints"
```

smoke 通过后立即开始正式 job，不运行全仓库测试。

## 4.3.5 生成三个正式 job

```bash
JOBS_JSONL="$ROUND_DIR/configs/jobs.jsonl"
: > "$JOBS_JSONL"

for seed in 20260906 20260907 20260908; do
  init_ckpt="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$NODE_EDGE_ROUND/metrics/node_edge_selection.json'))
print(p['selected_checkpoints'][str($seed)])
PY
)"
  test -f "$init_ckpt"
  job_id="phi_h${SELECTED_HISTORY}_s${seed}"
  out="$ROUND_DIR/jobs/$job_id"
  cmd="$PYTHON_BIN -m tools.stage4.train_phi --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --init-checkpoint '$init_ckpt' --history-steps '$SELECTED_HISTORY' --seed '$seed' --output-dir '$out'"
  "$PYTHON_BIN" - <<PY >> "$JOBS_JSONL"
import json
print(json.dumps({
  "job_id": "$job_id",
  "command": "$cmd",
  "output_dir": "$out",
  "seed": $seed,
  "init_checkpoint": "$init_ckpt"
}))
PY
done
cat "$JOBS_JSONL"
```

## 4.3.6 多 GPU 并行训练

```bash
"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$JOBS_JSONL" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/jobs" \
  --status-csv "$ROUND_DIR/metrics/job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

确认三个 job 完成：

```bash
"$PYTHON_BIN" - <<PY
import csv, pathlib
p=pathlib.Path('$ROUND_DIR/metrics/job_status.csv')
rows=list(csv.DictReader(p.open()))
assert len(rows)==3, len(rows)
failed=[r for r in rows if int(r['exit_code'])!=0]
assert not failed, failed
for r in rows:
    assert (pathlib.Path(r['output_dir'])/'DONE').is_file(), r
print('all three phi jobs completed')
PY
```

如单个 job 失败，只按 4.2 的相同方式重跑该 job，不重跑其他 seed。

## 4.3.7 汇总、画轨迹并选择 \(\phi\) 候选

```bash
"$PYTHON_BIN" -m tools.stage4.summarize_phi \
  --jobs-root "$ROUND_DIR/jobs" \
  --supervision-dir "$SUP_ROOT" \
  --output-csv "$ROUND_DIR/tables/phi_seed_summary.csv" \
  --per-node-csv "$ROUND_DIR/tables/phi_per_node.csv" \
  --selection-json "$ROUND_DIR/metrics/phi_selection.json" \
  --checkpoint-manifest "$ROUND_DIR/checkpoint_manifest.tsv" \
  --output-md "$ROUND_DIR/summary.md"

"$PYTHON_BIN" -m tools.stage4.render_phi_traces \
  --jobs-root "$ROUND_DIR/jobs" \
  --supervision-dir "$SUP_ROOT" \
  --split val \
  --max-episodes-per-scenario 3 \
  --output-dir "$ROUND_DIR/plots"
```

`phi_selection.json` 至少包含：

```json
{
  "selected_history_steps": 32,
  "selection_source": "transport_recovery_val_only",
  "selected_checkpoints": {
    "20260906": "/abs/path/to/best.pt",
    "20260907": "/abs/path/to/best.pt",
    "20260908": "/abs/path/to/best.pt"
  },
  "seed_pass": {},
  "median_metrics": {},
  "node_edge_drift": {}
}
```

候选选择必须遵循：

1. 只使用 `transport_recovery/val`。
2. 每个 seed 选择该 seed 的 best validation checkpoint。
3. 正式门槛使用 prediction-conditioned `phi`。
4. 记录 oracle-node 与 predicted-node 差距；不以 oracle 指标代替正式指标。
5. 比较 4.2 初始化 checkpoint 与 4.3 checkpoint 的 node/edge 指标，node macro F1 或 edge macro F1 下降超过 `0.03` 时标为 drift；先减小 encoder LR 并只重跑发生 drift 的 seed一次，不扩展大规模搜索。

## 本轮完成条件

- 三个 seed 全部结束且指标有限；
- 至少 2/3 seed 满足：
  - `phi_mae_predicted_node <= 0.18`；
  - `phi_spearman_predicted_node >= 0.65`；
  - `phi_monotonic_violation_rate <= 0.12`；
- prediction-conditioned 指标和 oracle-node 指标均已报告；
- 每个主要 node 均有独立 MAE/排序结果；
- node/edge 表征没有超过 `0.03` 的未处理退化；
- checkpoint manifest 已记录实际路径，checkpoint 不要求进入 ZIP。

若只有 1/3 seed 未过门槛，保留结果并进入 4.4；4.5 联合训练时修正。只有 0/3 通过时，检查一次 `phi_y` 边界、pair 生成和 node context 实现，修正后重跑本轮，不增加无关模型。

## 生成本轮 ZIP

```bash
{
  echo "- finished_at: $(date -Iseconds)"
  echo "- selected_history_steps: $SELECTED_HISTORY"
  echo "- selection_file: $ROUND_DIR/metrics/phi_selection.json"
} >> "$ROUND_DIR/run_manifest.md"

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

本轮停止点：完成 \(\phi\) 候选后立即进入 4.4，不在本轮调最终 reward 系数或 loop penalty。

<!-- END FILE: 阶段4.3_节点内进度Phi.md -->


---

<!-- BEGIN FILE: 阶段4.4_RemainingCost.md -->

# 阶段 4.4：训练图上剩余代价 \(C_G\)
> **本小阶段固定执行规则**：执行前同时遵守 `01_阶段4通用执行规范.md`。GPU 作业先尝试提权查询；独立 job 在不产生共享写冲突时默认多 GPU 并行；本轮结束立即生成 ZIP；checkpoint、权重和其他大文件可以不打包，但必须写 manifest；只做推进本轮所需的最小检查。


## 总体上要干什么

在阶段 4.3 的历史编码器、node/edge heads 和节点内进度 \(\phi\) 基础上，训练从当前历史状态到任一成功节点的非负剩余代价：

\[
C_G(H_t)\ge 0.
\]

本轮必须做一个清晰对照：

```text
abs_only：仅拟合成功轨迹的绝对 remaining-cost target
structured：绝对回归 + 时序排序 + 合法边 Bellman 一致性 + failure/recovery 结构约束
```

正式训练矩阵：

```text
2 个 loss 变体 × 3 个 seed = 6 个独立 GPU job
```

所有 job 使用独立输出目录，GPU 足够时并行执行。本轮仍不计算最终 reward，不调 `lambda/eta/beta`，也不接入 RA-BC。

## 本轮目录与入口检查

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export SUP_ROOT="$STAGE4_ROOT/supervision_v1"
export NODE_EDGE_ROUND="$STAGE4_ROOT/rounds/stage4_2_node_edge_heads"
export PHI_ROUND="$STAGE4_ROOT/rounds/stage4_3_within_node_progress"
export ROUND_ID="stage4_4_remaining_cost"
export ROUND_DIR="$STAGE4_ROOT/rounds/$ROUND_ID"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage4/init_round.sh" "$ROUND_ID" "$ROUND_DIR" \
  "Train absolute and graph-structured remaining-cost heads and select by validation only"

(
  cd "$SUP_ROOT"
  sha256sum -c SUPERVISION_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/supervision_checksum.log"

test -f "$PHI_ROUND/metrics/phi_selection.json"
cp "$PHI_ROUND/metrics/phi_selection.json" "$ROUND_DIR/configs/phi_selection.json"
cp "$SUP_ROOT/configs/cost_target_spec.yaml" "$ROUND_DIR/configs/"

SELECTED_HISTORY="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$PHI_ROUND/metrics/phi_selection.json'))
print(int(p['selected_history_steps']))
PY
)"
echo "selected_history_steps=$SELECTED_HISTORY" | tee -a "$ROUND_DIR/run_manifest.md"

GPU_SNAPSHOT_DIR="$ROUND_DIR/manifests/gpu_snapshot"
"$REPO_ROOT/tools/stage4/query_gpus.sh" "$GPU_SNAPSHOT_DIR"
test -s "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt"
```

## 4.4.1 扩展模型：实现 prediction-conditioned cost head

在 `tools/stage4/lib/model.py` 中增加 remaining-cost head，继续复用同一历史编码器。输入必须全部来自当前和过去：

```text
history_embedding
+ task embedding
+ soft node context（由预测 node_probs 计算）
+ edge-type probability vector
+ predicted phi
    -> MLP(192+... -> 128 -> 64 -> 1)
    -> softplus
    -> remaining_cost
```

具体要求：

1. 输出键严格使用 `remaining_cost`，形状 `[B]`，值非负。
2. cost head 的正式输入使用预测的 node/edge belief 和预测 \(\phi\)，不能读取 GT node、GT edge 或 `phi_y`。
3. 为误差诊断可另外计算 `remaining_cost_oracle_context`，它可使用 GT node、edge 和 `phi_y`，但不得参与正式门槛，也不得在部署接口中替代 `remaining_cost`。
4. 训练前 5 epoch 冻结 encoder、分类 heads 和 phi head，只训练 cost head；第 6 epoch 起解冻 encoder，但使用小学习率并保留上游 anchor losses。
5. 加载阶段 4.3 checkpoint 时只允许 `cost_head.*` 为缺失 key。
6. checkpoint 中保存 cost normalization、GraphSpec hash、loss variant、pair margin 和初始化 checkpoint 信息。

## 4.4.2 实现 cost frame dataset 与 pair dataset

在 `tools/stage4/lib/data.py` 中增加两个数据视图。

### A. `CostFrameDataset`

返回：

```text
sample_id, episode_id, content_group_id, task_id, split,
history_x, history_mask,
node_y, edge_type_y, edge_id_y, phi_y,
cost_y_norm, cost_mask,
is_success_terminal, sample_weight
```

要求：

- 绝对 Huber 只在 `cost_mask=1` 上计算；
- terminal-failure frame 的 `cost_mask=0`，不得伪造绝对 success distance；
- `transport_dual_order` 仍是 mechanism probe，不参与 validation model selection；
- content group 总权重归一。

### B. `CostPairDataset`

读取 `tables/cost_pairs.csv.gz`。每一行必须能定位两个因果窗口，字段至少包括：

```text
pair_id
pair_type
left_sample_id
right_sample_id
pre_failure_sample_id（仅 no_overshoot 使用，可为空）
task_id
split
content_group_id
margin
edge_id
edge_type
edge_cost_norm
pair_weight
```

统一关系语义如下：

| `pair_type` | left/right 定义 | 应满足 |
|---|---|---|
| `forward_decrease` | left=转移前，right=转移后 | \(C_l>C_r\) |
| `temporal_decrease` | left=较早帧，right=较晚帧 | \(C_l>C_r\) |
| `failure_increase` | left=failure 后，right=failure 前 | \(C_l>C_r\) |
| `recovery_decrease` | left=failure 状态，right=recovery 完成后 | \(C_l>C_r\) |
| `recovery_no_overshoot` | left=recovery 后，pre=失败前 | \(C_{recovered}\ge C_{pre}-tol\) |
| `terminal_success_zero` | left=success frame | \(C_l\approx0\) |

每个 pair 的两个窗口必须来自同一个 `content_group_id`；禁止跨 episode 乱配 failure/recovery。

训练时每一步读取一个 frame batch 和一个 pair batch。pair batch 按类型近似均衡；样本不足的类型可有放回采样，但必须在 summary 中报告真实唯一 pair 数。

## 4.4.3 实现两种损失

在 `tools/stage4/lib/losses.py` 中增加以下函数。

### 绝对代价回归

\[
\mathcal L_{abs}
=\operatorname{Huber}(\hat C_t,C_t^*)
\quad\text{on } cost\_mask=1.
\]

### 时序/边排序

对于要求 `left > right` 的 pair：

\[
\mathcal L_{rank}(l,r;m)
=\max\left(0,\,m-(\hat C_l-\hat C_r)\right).
\]

分别对 `forward_decrease`、`temporal_decrease`、`failure_increase` 和 `recovery_decrease` 计算。

### 合法边 Bellman 一致性

仅对 `forward`、`alternative`、`recovery` 的实际完成边计算；不对 failure/stagnation 强行套最短路 Bellman：

\[
\mathcal L_{bellman}
=\left|\hat C_{before}-\left(c_e^{norm}+\hat C_{after}\right)\right|.
\]

### Recovery 不超额奖励约束

设 \(C_{pre}\) 为 failure 前状态，\(C_{rec}\) 为 recovery 完成状态：

\[
\mathcal L_{no\_overshoot}
=\max\left(0,\,C_{pre}-\epsilon- C_{rec}\right),
\qquad \epsilon=0.05.
\]

该项防止 recovery 后的估计代价低于 failure 前太多，从源头减少后续“故意失败再恢复刷正奖励”的可能性。

### Success 归零

\[
\mathcal L_{success}=|C_{success}|.
\]

### 上游保持项

第 6 epoch 起允许 encoder 微调时，加入：

\[
\mathcal L_{upstream}
=0.20\mathcal L_{node}
+0.15\mathcal L_{edge\_type}
+0.10\mathcal L_{edge\_id}
+0.20\mathcal L_{\phi}.
\]

### 两个正式变体

`abs_only`：

\[
\mathcal L_{abs\_only}=\mathcal L_{abs}+\mathcal L_{upstream}.
\]

`structured`：

\[
\begin{aligned}
\mathcal L_{structured}={}&
\mathcal L_{abs}
+0.5\mathcal L_{temporal\_rank}
+0.5\mathcal L_{bellman}\\
&+1.0\mathcal L_{failure\_increase}
+1.0\mathcal L_{recovery\_decrease}\\
&+0.5\mathcal L_{no\_overshoot}
+0.5\mathcal L_{success}
+\mathcal L_{upstream}.
\end{aligned}
\]

所有 loss 都按实际有效样本数归一。某 batch 某类 pair 为空时该项记 0，不产生 NaN。

## 4.4.4 创建训练、评估、汇总和轨迹渲染脚本

创建：

```text
tools/stage4/train_remaining_cost.py
tools/stage4/evaluate_remaining_cost.py
tools/stage4/summarize_remaining_cost.py
tools/stage4/render_cost_traces.py
```

### `train_remaining_cost.py` CLI

```text
python -m tools.stage4.train_remaining_cost
  --config CONFIG
  --supervision-dir SUP_ROOT
  --init-checkpoint PHI_BEST_PT
  --history-steps N
  --loss-variant {abs_only,structured}
  --seed SEED
  --output-dir OUT
  [--max-epochs N]
  [--limit-train-batches N]
  [--limit-val-batches N]
```

训练要求：

1. 初始化必须使用同 seed 的 4.3 selected checkpoint。
2. early stopping 只读取 `transport_recovery/val`。
3. `abs_only` 和 `structured` 使用相同 encoder 初始化、batch size、epoch 上限和 seed。
4. structured 的 pair sampler 只用 train pairs；validation pair metrics 只用于评估和 early stopping。
5. best checkpoint 选择分数：

\[
S_C=0.30(1-\mathrm{MAE})
+0.20\,\mathrm{SpearmanScaled}
+0.20\,\mathrm{PairAcc}
+0.15\,\mathrm{FailureIncreaseRate}
+0.15\,\mathrm{RecoveryDecreaseRate}.
\]

6. test 和 Stage 3 diagnostic 在本轮禁止读取。

每个正式 job 输出：

```text
resolved_config.yaml
train_metrics.csv
val_metrics.json
val_predictions.jsonl
val_pair_predictions.csv
per_pair_type_metrics.csv
per_scenario_metrics.csv
plots/cost_pred_vs_gt.png
plots/cost_trace_examples.png
plots/failure_recovery_cost_trace.png
checkpoints/best.pt
checkpoints/final.pt
DONE
```

## 4.4.5 必须计算的 validation 指标

绝对 target 指标只在 `cost_mask=1` 的成功 frame 上：

```text
cost_mae
cost_rmse
cost_spearman
cost_kendall_tau
terminal_success_cost_p50
terminal_success_cost_p90
```

结构 pair 指标：

```text
cost_pair_accuracy_all
forward_decrease_accuracy
failure_cost_increase_rate
recovery_cost_decrease_rate
recovery_no_overshoot_rate
bellman_mae_legal_edges
```

上游稳定性：

```text
node_macro_f1_after_cost_training
edge_type_macro_f1_after_cost_training
phi_mae_after_cost_training
```

同时按 `natural_success`、`terminal_failure`、`drop_and_regrasp`、`gripper_reopen` 分开报告；没有样本的 scenario 写明 `n=0`，不得以 0 填充后参与平均。

## 4.4.6 先做一个 structured smoke run

```bash
FIRST_SEED=20260906
INIT_CKPT="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$PHI_ROUND/metrics/phi_selection.json'))
print(p['selected_checkpoints'][str($FIRST_SEED)])
PY
)"
test -f "$INIT_CKPT"

SMOKE_DIR="$ROUND_DIR/jobs/_smoke_cost_structured"
"$PYTHON_BIN" -m tools.stage4.train_remaining_cost \
  --config "$STAGE4_CONFIG" \
  --supervision-dir "$SUP_ROOT" \
  --init-checkpoint "$INIT_CKPT" \
  --history-steps "$SELECTED_HISTORY" \
  --loss-variant structured \
  --seed "$FIRST_SEED" \
  --max-epochs 2 \
  --limit-train-batches 4 \
  --limit-val-batches 2 \
  --output-dir "$SMOKE_DIR" \
  2>&1 | tee "$ROUND_DIR/logs/smoke_cost.log"

test -f "$SMOKE_DIR/DONE"
"$PYTHON_BIN" - <<PY
import json, math
m=json.load(open('$SMOKE_DIR/val_metrics.json'))
for k,v in m.items():
    if isinstance(v,(int,float)) and not math.isfinite(v):
        raise SystemExit(f'non-finite smoke metric: {k}={v}')
print('remaining-cost smoke run passed')
PY
rm -rf "$SMOKE_DIR/checkpoints"
```

## 4.4.7 生成六个正式 job

```bash
JOBS_JSONL="$ROUND_DIR/configs/jobs.jsonl"
: > "$JOBS_JSONL"

for variant in abs_only structured; do
  for seed in 20260906 20260907 20260908; do
    init_ckpt="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$PHI_ROUND/metrics/phi_selection.json'))
print(p['selected_checkpoints'][str($seed)])
PY
)"
    test -f "$init_ckpt"
    job_id="cost_${variant}_h${SELECTED_HISTORY}_s${seed}"
    out="$ROUND_DIR/jobs/$job_id"
    cmd="$PYTHON_BIN -m tools.stage4.train_remaining_cost --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --init-checkpoint '$init_ckpt' --history-steps '$SELECTED_HISTORY' --loss-variant '$variant' --seed '$seed' --output-dir '$out'"
    "$PYTHON_BIN" - <<PY >> "$JOBS_JSONL"
import json
print(json.dumps({
  "job_id": "$job_id",
  "command": "$cmd",
  "output_dir": "$out",
  "seed": $seed,
  "loss_variant": "$variant",
  "init_checkpoint": "$init_ckpt"
}))
PY
  done
done
cat "$JOBS_JSONL"
```

## 4.4.8 多 GPU 并行运行

```bash
"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$JOBS_JSONL" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/jobs" \
  --status-csv "$ROUND_DIR/metrics/job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

只重跑失败 job，不重跑成功 job。完成后执行：

```bash
"$PYTHON_BIN" - <<PY
import csv, pathlib
rows=list(csv.DictReader(open('$ROUND_DIR/metrics/job_status.csv')))
assert len(rows)==6, len(rows)
assert all(int(r['exit_code'])==0 for r in rows), rows
assert all((pathlib.Path(r['output_dir'])/'DONE').is_file() for r in rows)
print('all six cost jobs completed')
PY
```

## 4.4.9 汇总并选择 cost 变体

```bash
"$PYTHON_BIN" -m tools.stage4.summarize_remaining_cost \
  --jobs-root "$ROUND_DIR/jobs" \
  --supervision-dir "$SUP_ROOT" \
  --output-csv "$ROUND_DIR/tables/cost_seed_summary.csv" \
  --pair-type-csv "$ROUND_DIR/tables/cost_pair_type_summary.csv" \
  --scenario-csv "$ROUND_DIR/tables/cost_scenario_summary.csv" \
  --comparison-csv "$ROUND_DIR/tables/abs_vs_structured.csv" \
  --selection-json "$ROUND_DIR/metrics/cost_selection.json" \
  --checkpoint-manifest "$ROUND_DIR/checkpoint_manifest.tsv" \
  --output-md "$ROUND_DIR/summary.md"

"$PYTHON_BIN" -m tools.stage4.render_cost_traces \
  --jobs-root "$ROUND_DIR/jobs" \
  --supervision-dir "$SUP_ROOT" \
  --split val \
  --max-episodes-per-scenario 3 \
  --output-dir "$ROUND_DIR/plots"
```

`cost_selection.json` 至少包含：

```json
{
  "selected_loss_variant": "structured",
  "selection_source": "transport_recovery_val_only",
  "selected_history_steps": 32,
  "selected_checkpoints": {
    "20260906": "/abs/path/to/best.pt",
    "20260907": "/abs/path/to/best.pt",
    "20260908": "/abs/path/to/best.pt"
  },
  "seed_pass": {},
  "variant_median_metrics": {},
  "upstream_drift": {}
}
```

选择规则：

1. 只读取 validation，不读取 test/diagnostic/probe。
2. 优先选择 `structured`，条件是其 median composite 不低于 `abs_only` 超过 `0.01`，且 failure/recovery 两个结构指标至少一项更好。
3. 若 `structured` 显著更差，先检查 pair 左右方向、margin 和 cost normalizer；只允许修正明确实现错误后重跑 structured，一律不进行大规模 loss-weight 搜索。
4. 若修正后仍明显更差，选择 val composite 更高的变体进入 4.5，并在 `summary.md` 明确记录结构约束未带来收益。
5. 上游 node/edge F1 下降超过 `0.03` 或 phi MAE 恶化超过 `0.03` 时，先减小 encoder LR 并仅重跑受影响 seed。

## 本轮完成条件

至少 2/3 个 selected variant seed 满足：

- `cost_mae <= 0.20`；
- `cost_spearman >= 0.70`；
- `cost_pair_accuracy_all >= 0.75`；
- `failure_cost_increase_rate >= 0.70`；
- `recovery_cost_decrease_rate >= 0.70`；
- `terminal_success_cost_p90 <= 0.15`。

并且：

- `abs_only` 与 `structured` 使用相同初始化和预算完成比较；
- legal edge Bellman MAE、no-overshoot rate、各 scenario 指标已报告；
- test 和 Stage 3 diagnostic 仍未用于选择；
- checkpoint manifest 已生成，checkpoint 可不打包。

若只有一个结构指标略低于门槛但绝对回归与其余指标稳定，保留候选进入 4.5 联合训练；不要停下来增加大量无关试验。

## 生成本轮 ZIP

```bash
{
  echo "- finished_at: $(date -Iseconds)"
  echo "- selected_history_steps: $SELECTED_HISTORY"
  echo "- selection_file: $ROUND_DIR/metrics/cost_selection.json"
} >> "$ROUND_DIR/run_manifest.md"

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

本轮停止点：selected cost checkpoint 生成后立即进入 4.5，不在本轮调整最终 reward、loop penalty 或 uncertainty \(\beta\)。

<!-- END FILE: 阶段4.4_RemainingCost.md -->


---

<!-- BEGIN FILE: 阶段4.5_联合训练与模型选择.md -->

# 阶段 4.5：联合训练、验证集选型与一次性冻结评估
> **本小阶段固定执行规则**：执行前同时遵守 `01_阶段4通用执行规范.md`。GPU 作业先尝试提权查询；独立 job 在不产生共享写冲突时默认多 GPU 并行；本轮结束立即生成 ZIP；checkpoint、权重和其他大文件可以不打包，但必须写 manifest；只做推进本轮所需的最小检查。


## 总体上要干什么

把阶段 4.2—4.4 的四类输出统一到一个历史条件化多头模型中：

```text
node belief
edge-type belief
edge-id belief
within-node progress phi
remaining cost C_G
```

先在 train 上联合微调，只用 `transport_recovery/val` 进行 early stopping 和模型选择；模型配置及三个 seed checkpoint 锁定后，才允许一次性读取 test、Stage 3 diagnostic suite 和 dual-order mechanism probe。

本轮不计算最终 graph reward，不调 reward 系数，不接入 RA-BC。目标是得到可供阶段 5 调用的**模型候选集**及一次冻结评估结果。

正式 GPU 工作包括：

```text
主联合训练：3 seed
冻结 test 推理：3 seed
Stage 3 diagnostic 推理：3 seed
dual-order holdout probe：2 fold × 3 seed（独立、较短训练）
```

同一组内以及各组之间无共享写冲突时均应多 GPU 并行；禁止把多个 job 写入同一个输出目录。

## 本轮目录与入口检查

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export DIAG_ROOT="$STAGE3_ROOT/diagnostic_suite_v1"
export M2_ROOT="$STAGE3_ROOT/m2_freeze_v1"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export SUP_ROOT="$STAGE4_ROOT/supervision_v1"
export NODE_EDGE_ROUND="$STAGE4_ROOT/rounds/stage4_2_node_edge_heads"
export PHI_ROUND="$STAGE4_ROOT/rounds/stage4_3_within_node_progress"
export COST_ROUND="$STAGE4_ROOT/rounds/stage4_4_remaining_cost"
export ROUND_ID="stage4_5_joint_model_selection"
export ROUND_DIR="$STAGE4_ROOT/rounds/$ROUND_ID"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage4/init_round.sh" "$ROUND_ID" "$ROUND_DIR" \
  "Jointly fine-tune all graph-state heads, lock selection on validation, then run one-shot frozen evaluations"

(
  cd "$SUP_ROOT"
  sha256sum -c SUPERVISION_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/supervision_checksum.log"
(
  cd "$DIAG_ROOT"
  sha256sum -c DIAGNOSTIC_SUITE_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/diagnostic_checksum.log"

test -f "$NODE_EDGE_ROUND/metrics/node_edge_selection.json"
test -f "$PHI_ROUND/metrics/phi_selection.json"
test -f "$COST_ROUND/metrics/cost_selection.json"
cp "$NODE_EDGE_ROUND/metrics/node_edge_selection.json" "$ROUND_DIR/configs/"
cp "$PHI_ROUND/metrics/phi_selection.json" "$ROUND_DIR/configs/"
cp "$COST_ROUND/metrics/cost_selection.json" "$ROUND_DIR/configs/"
cp "$SUP_ROOT/probes/dual_order_folds.json" "$ROUND_DIR/configs/"

SELECTED_HISTORY="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$COST_ROUND/metrics/cost_selection.json'))
print(int(p['selected_history_steps']))
PY
)"
SELECTED_COST_VARIANT="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$COST_ROUND/metrics/cost_selection.json'))
print(p['selected_loss_variant'])
PY
)"
{
  echo "- selected_history_steps: $SELECTED_HISTORY"
  echo "- selected_cost_variant: $SELECTED_COST_VARIANT"
} | tee -a "$ROUND_DIR/run_manifest.md"

GPU_SNAPSHOT_DIR="$ROUND_DIR/manifests/gpu_snapshot"
"$REPO_ROOT/tools/stage4/query_gpus.sh" "$GPU_SNAPSHOT_DIR"
test -s "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt"
```

## 4.5.1 实现联合训练器

创建：

```text
tools/stage4/train_joint_pathgraph.py
tools/stage4/evaluate_joint_pathgraph.py
tools/stage4/summarize_joint_validation.py
tools/stage4/lock_model_selection.py
tools/stage4/evaluate_stage3_diagnostics.py
tools/stage4/train_dual_order_probe.py
tools/stage4/summarize_frozen_evaluation.py
```

### 主模型初始化

每个 seed 必须加载相同 seed 的阶段 4.4 selected checkpoint。该 checkpoint 已含 node、edge、phi 和 cost heads。不得跨 seed 复制初始化。

### 主训练数据

主联合训练按以下批次比例：

```text
70% transport_recovery/train
15% transport_dual_order/A-first representative
15% transport_dual_order/B-first representative
```

说明：dual-order 的两个内容组用于让 task-conditioned heads 覆盖该图，但其重复 scripted-oracle episode 不得扩大权重；每个方向只作为一个 content group。validation model selection 仍只使用 `transport_recovery/val`。

frame batch 和 cost-pair batch均继续采用 content-group 归一。history length 固定为阶段 4.2 选择结果。

### 联合损失

联合训练使用：

\[
\begin{aligned}
\mathcal L_{joint}={}&
1.0\mathcal L_{node}
+1.0\mathcal L_{edge\_type}
+0.5\mathcal L_{edge\_id}\\
&+1.0\mathcal L_{\phi,huber}
+0.5\mathcal L_{\phi,mono}\\
&+1.0\mathcal L_{cost,huber}
+0.5\mathcal L_{cost,temporal}\\
&+0.5\mathcal L_{bellman}
+1.0\mathcal L_{failure\_increase}\\
&+1.0\mathcal L_{recovery\_decrease}
+0.5\mathcal L_{no\_overshoot}
+0.5\mathcal L_{success\_zero}.
\end{aligned}
\]

如果 4.4 选择了 `abs_only`，联合训练仍保留上式结构项，但在 `resolved_config.yaml` 中明确记录“结构约束由 joint 阶段首次启用”。

每个 loss 先按有效样本数归一，再乘权重。禁止因 edge-positive/pair 稀少而让 loss 直接按 batch 总帧数缩小。

### 优化规则

- 从 selected cost checkpoint 初始化；
- 全模型训练，不再冻结 head；
- encoder learning rate = `3e-4`；各 head learning rate = `5e-4`；
- AMP 开启；gradient clip=1.0；
- 最多 80 epoch，patience=12；
- 每个 epoch 保存 train 与 validation 各 head 指标；
- 只保留 validation composite 最优 checkpoint 与 final checkpoint；
- test/diagnostic loader 不得在训练进程中构造。

### Validation composite

在 `transport_recovery/val` 上：

\[
\begin{aligned}
S_{joint}={}&0.25F1_{node}
+0.20F1_{edge\ type}\
&+0.10F1_{edge\ id}
+0.15(1-MAE_{\phi})\\
&+0.15Acc_{cost\ pair}
+0.15SpearmanScaled_{cost}.
\end{aligned}
\]

其中各项均裁剪到 `[0,1]`。tie-breaker 顺序：

```text
recovery_recall
failure_recall
cost_mae（更小优先）
phi_mae（更小优先）
checkpoint_size（更小优先）
```

### `train_joint_pathgraph.py` CLI

```text
python -m tools.stage4.train_joint_pathgraph
  --config CONFIG
  --supervision-dir SUP_ROOT
  --init-checkpoint COST_BEST_PT
  --history-steps N
  --seed SEED
  --output-dir OUT
  [--max-epochs N]
  [--limit-train-batches N]
  [--limit-val-batches N]
```

每个 job 至少输出：

```text
resolved_config.yaml
train_metrics.csv
val_metrics.json
val_predictions.jsonl
val_pair_predictions.csv
per_class_metrics.csv
per_node_phi_metrics.csv
per_pair_type_cost_metrics.csv
loss_balance.csv
plots/validation_metric_history.png
plots/validation_confusions.png
plots/validation_phi_cost_traces.png
checkpoints/best.pt
checkpoints/final.pt
DONE
```

## 4.5.2 实现 test 锁与评估防误用

`evaluate_joint_pathgraph.py` 必须要求：

```text
--selection-lock PATH
```

当 `--split test`、`--diagnostic-suite` 或 `--probe-fold` 被使用时：

1. `selection_lock.json` 必须存在；
2. lock 中 checkpoint path、checkpoint SHA256、config SHA256 与当前输入一致；
3. lock 的 `selection_source` 必须等于 `transport_recovery_val_only`；
4. 不满足时立即退出，禁止静默评估。

`selection_lock.json` 不在训练前创建。必须先完成三个 seed 的 validation 汇总，再由 `lock_model_selection.py` 生成。

## 4.5.3 主联合训练 smoke run

```bash
FIRST_SEED=20260906
INIT_CKPT="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$COST_ROUND/metrics/cost_selection.json'))
print(p['selected_checkpoints'][str($FIRST_SEED)])
PY
)"
test -f "$INIT_CKPT"

SMOKE_DIR="$ROUND_DIR/jobs/_smoke_joint"
"$PYTHON_BIN" -m tools.stage4.train_joint_pathgraph \
  --config "$STAGE4_CONFIG" \
  --supervision-dir "$SUP_ROOT" \
  --init-checkpoint "$INIT_CKPT" \
  --history-steps "$SELECTED_HISTORY" \
  --seed "$FIRST_SEED" \
  --max-epochs 2 \
  --limit-train-batches 4 \
  --limit-val-batches 2 \
  --output-dir "$SMOKE_DIR" \
  2>&1 | tee "$ROUND_DIR/logs/smoke_joint.log"

test -f "$SMOKE_DIR/DONE"
"$PYTHON_BIN" - <<PY
import json, math
m=json.load(open('$SMOKE_DIR/val_metrics.json'))
for k,v in m.items():
    if isinstance(v,(int,float)) and not math.isfinite(v):
        raise SystemExit(f'non-finite smoke metric: {k}={v}')
print('joint smoke run passed')
PY
rm -rf "$SMOKE_DIR/checkpoints"
```

## 4.5.4 生成并行主训练 job

```bash
MAIN_JOBS="$ROUND_DIR/configs/main_joint_jobs.jsonl"
: > "$MAIN_JOBS"

for seed in 20260906 20260907 20260908; do
  init_ckpt="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$COST_ROUND/metrics/cost_selection.json'))
print(p['selected_checkpoints'][str($seed)])
PY
)"
  test -f "$init_ckpt"
  job_id="joint_h${SELECTED_HISTORY}_s${seed}"
  out="$ROUND_DIR/jobs/main/$job_id"
  cmd="$PYTHON_BIN -m tools.stage4.train_joint_pathgraph --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --init-checkpoint '$init_ckpt' --history-steps '$SELECTED_HISTORY' --seed '$seed' --output-dir '$out'"
  "$PYTHON_BIN" - <<PY >> "$MAIN_JOBS"
import json
print(json.dumps({
  "job_id": "$job_id",
  "command": "$cmd",
  "output_dir": "$out",
  "seed": $seed,
  "init_checkpoint": "$init_ckpt"
}))
PY
done
cat "$MAIN_JOBS"

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$MAIN_JOBS" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/main_joint" \
  --status-csv "$ROUND_DIR/metrics/main_joint_job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

仅重跑失败 job。成功后确认三个 `DONE` 文件存在。

## 4.5.5 汇总 validation 并冻结选择

```bash
"$PYTHON_BIN" -m tools.stage4.summarize_joint_validation \
  --jobs-root "$ROUND_DIR/jobs/main" \
  --node-edge-selection "$NODE_EDGE_ROUND/metrics/node_edge_selection.json" \
  --phi-selection "$PHI_ROUND/metrics/phi_selection.json" \
  --cost-selection "$COST_ROUND/metrics/cost_selection.json" \
  --output-csv "$ROUND_DIR/tables/joint_validation_seed_summary.csv" \
  --comparison-csv "$ROUND_DIR/tables/staged_vs_joint.csv" \
  --candidate-json "$ROUND_DIR/metrics/joint_candidates.json" \
  --checkpoint-manifest "$ROUND_DIR/checkpoint_manifest.tsv" \
  --output-md "$ROUND_DIR/validation_summary.md"
```

`joint_candidates.json` 必须包含三个 seed 的：

```text
checkpoint path
checkpoint SHA256
best epoch
resolved config path + SHA256
validation composite
各 head validation metrics
threshold pass/fail
```

随后创建 selection lock：

```bash
"$PYTHON_BIN" -m tools.stage4.lock_model_selection \
  --candidate-json "$ROUND_DIR/metrics/joint_candidates.json" \
  --stage4-config "$STAGE4_CONFIG" \
  --supervision-checksums "$SUP_ROOT/SUPERVISION_SHA256SUMS.txt" \
  --selection-source transport_recovery_val_only \
  --output "$ROUND_DIR/metrics/selection_lock.json" \
  --output-md "$ROUND_DIR/selection_lock.md"

cat "$ROUND_DIR/metrics/selection_lock.json"
sha256sum "$ROUND_DIR/metrics/selection_lock.json" \
  | tee "$ROUND_DIR/metrics/selection_lock.sha256"
```

`selection_lock.json` 一旦生成，本轮后续不得根据 test/diagnostic 结果替换 checkpoint、epoch 或 loss 权重。若发现实现 bug，应删除整个 lock、写明 bug、修正并重新执行 4.5 主训练；不得在 test 上挑 seed。

## 4.5.6 一次性运行 primary test

生成三个 seed 的 test job：

```bash
TEST_JOBS="$ROUND_DIR/configs/frozen_test_jobs.jsonl"
: > "$TEST_JOBS"

for seed in 20260906 20260907 20260908; do
  ckpt="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$ROUND_DIR/metrics/selection_lock.json'))
print(p['checkpoints'][str($seed)]['path'])
PY
)"
  test -f "$ckpt"
  job_id="frozen_test_s${seed}"
  out="$ROUND_DIR/jobs/test/$job_id"
  cmd="$PYTHON_BIN -m tools.stage4.evaluate_joint_pathgraph --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --checkpoint '$ckpt' --selection-lock '$ROUND_DIR/metrics/selection_lock.json' --split test --task transport_recovery --seed '$seed' --output-dir '$out'"
  "$PYTHON_BIN" - <<PY >> "$TEST_JOBS"
import json
print(json.dumps({"job_id":"$job_id","command":"$cmd","output_dir":"$out","seed":$seed}))
PY
done

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$TEST_JOBS" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/frozen_test" \
  --status-csv "$ROUND_DIR/metrics/frozen_test_job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

每个 test job 输出：

```text
test_metrics.json
test_predictions.jsonl
test_pair_predictions.csv
per_class_metrics.csv
per_scenario_metrics.csv
plots/test_confusions.png
plots/test_phi_cost_traces.png
DONE
```

## 4.5.7 在 Stage 3 diagnostic suite 上运行模型输出分析

本步骤只分析 node/edge/phi/cost，不构造最终 reward。每个 seed 一个 job：

```bash
DIAG_JOBS="$ROUND_DIR/configs/diagnostic_jobs.jsonl"
: > "$DIAG_JOBS"

for seed in 20260906 20260907 20260908; do
  ckpt="$($PYTHON_BIN - <<PY
import json
p=json.load(open('$ROUND_DIR/metrics/selection_lock.json'))
print(p['checkpoints'][str($seed)]['path'])
PY
)"
  job_id="stage3_diag_s${seed}"
  out="$ROUND_DIR/jobs/diagnostic/$job_id"
  cmd="$PYTHON_BIN -m tools.stage4.evaluate_stage3_diagnostics --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --diagnostic-suite '$DIAG_ROOT' --checkpoint '$ckpt' --selection-lock '$ROUND_DIR/metrics/selection_lock.json' --seed '$seed' --output-dir '$out'"
  "$PYTHON_BIN" - <<PY >> "$DIAG_JOBS"
import json
print(json.dumps({"job_id":"$job_id","command":"$cmd","output_dir":"$out","seed":$seed}))
PY
done

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$DIAG_JOBS" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/diagnostic" \
  --status-csv "$ROUND_DIR/metrics/diagnostic_job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

必须输出以下**模型级**诊断，不替代阶段 5 reward 指标：

```text
alternative-order node/edge sequence validity
两条合法路径的 terminal remaining_cost
failure boundary 前后 remaining_cost 差值
recovery boundary 前后 remaining_cost 差值
failure 前与 recovery 后 remaining_cost 差值
cycle/stagnation segment 内 cost oscillation
history-required node subset accuracy
```

## 4.5.8 运行 dual-order holdout mechanism probes

由于 `transport_dual_order` 实际只有两个唯一内容组，本实验只证明“模型结构是否能表达未见顺序”，不报告跨数据集泛化或置信区间。

为避免主模型已经同时看过两条路径造成泄漏，probe job 必须：

1. 新建随机初始化模型；
2. 训练数据只包含 `transport_recovery/train + fold 指定的一条 dual-order path`；
3. 对另一条 path 评估；
4. 固定阶段 4 的 history、架构和 loss，不以 probe 结果调参；
5. 每 fold × seed 独立输出。

生成六个 probe job：

```bash
PROBE_JOBS="$ROUND_DIR/configs/dual_order_probe_jobs.jsonl"
: > "$PROBE_JOBS"

for fold in holdout_A_first holdout_B_first; do
  for seed in 20260906 20260907 20260908; do
    job_id="dual_probe_${fold}_s${seed}"
    out="$ROUND_DIR/jobs/dual_probe/$job_id"
    cmd="$PYTHON_BIN -m tools.stage4.train_dual_order_probe --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --fold '$fold' --history-steps '$SELECTED_HISTORY' --seed '$seed' --max-epochs 50 --selection-lock '$ROUND_DIR/metrics/selection_lock.json' --output-dir '$out'"
    "$PYTHON_BIN" - <<PY >> "$PROBE_JOBS"
import json
print(json.dumps({"job_id":"$job_id","command":"$cmd","output_dir":"$out","seed":$seed,"fold":"$fold"}))
PY
  done
done

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$PROBE_JOBS" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/dual_probe" \
  --status-csv "$ROUND_DIR/metrics/dual_probe_job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

probe 必须输出：

```text
heldout_path_metrics.json
heldout_path_predictions.jsonl
predicted_node_edge_sequence.json
terminal_cost.json
plots/heldout_path_trace.png
checkpoints/best.pt（默认不打包）
DONE
```

主要 probe 指标：

```text
heldout_node_macro_f1
heldout_edge_type_macro_f1_non_none
heldout_edge_sequence_validity
heldout_terminal_cost
heldout_phi_spearman
heldout_cost_spearman
```

不要把 3 seed × 完全相同的 heldout content 当成 3 个独立数据样本；seed 只表示优化稳定性。

## 4.5.9 汇总冻结评估

```bash
"$PYTHON_BIN" -m tools.stage4.summarize_frozen_evaluation \
  --selection-lock "$ROUND_DIR/metrics/selection_lock.json" \
  --validation-jobs "$ROUND_DIR/jobs/main" \
  --test-jobs "$ROUND_DIR/jobs/test" \
  --diagnostic-jobs "$ROUND_DIR/jobs/diagnostic" \
  --dual-probe-jobs "$ROUND_DIR/jobs/dual_probe" \
  --output-dir "$ROUND_DIR"
```

脚本至少生成：

```text
summary.md
tables/joint_validation_seed_summary.csv
tables/joint_test_seed_summary.csv
tables/joint_test_per_scenario.csv
tables/diagnostic_model_signatures.csv
tables/dual_order_probe_summary.csv
tables/staged_vs_joint.csv
metrics/stage4_model_metrics.json
plots/validation_vs_test.png
plots/failure_recovery_cost_deltas.png
plots/dual_order_probe_overlay.png
plots/all_head_metric_matrix.png
```

`summary.md` 的结果顺序固定为：

1. validation 选型依据；
2. frozen test 结果；
3. node/edge 结果；
4. \(\phi\) 结果；
5. remaining-cost 结果；
6. Stage 3 diagnostic 模型级结果；
7. dual-order mechanism probe；
8. 失败项和下一轮直接处理建议。

## 本轮完成条件

- 三个主联合训练 seed 均完成；
- `selection_lock.json` 在读取 test 前生成且 hash 已保存；
- test、diagnostic、probe 均在 lock 后运行；
- 至少 2/3 主模型 seed 在 validation 满足：
  - node macro F1 `>=0.70`；
  - edge-type non-none macro F1 `>=0.55`；
  - failure/recovery recall `>=0.60`；
  - phi MAE `<=0.18`；
  - cost MAE `<=0.20`；
  - cost Spearman `>=0.70`；
  - cost pair accuracy `>=0.75`；
- frozen test 指标有限，且相对 validation 没有无法解释的整体崩溃；
- failure 后 cost 上升和 recovery 后 cost 下降均已在 diagnostic 中报告；
- dual-order 两个 holdout fold 均输出结果，并明确标为 mechanism probe；
- checkpoint manifest 包含主模型与 probe checkpoint 路径，权重不要求打包。

如果某一个 head 在联合训练中比 staged checkpoint 下降超过 `0.03`，只检查 loss 归一和 learning rate；修正后重跑 3 个主 seed一次。不要添加大规模架构搜索。

## 生成本轮 ZIP

```bash
{
  echo "- finished_at: $(date -Iseconds)"
  echo "- selection_lock: $ROUND_DIR/metrics/selection_lock.json"
  echo "- frozen_test_completed: true"
  echo "- diagnostic_completed: true"
  echo "- dual_order_probe_completed: true"
} >> "$ROUND_DIR/run_manifest.md"

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

本轮停止点：完成冻结评估后进入 4.6。不要根据 test 或 probe 结果返回更换 checkpoint，除非发现可复现的实现错误并明确作废 selection lock。

<!-- END FILE: 阶段4.5_联合训练与模型选择.md -->


---

<!-- BEGIN FILE: 阶段4.6_不确定性与冻结.md -->

# 阶段 4.6：Deep Ensemble、不确定性接口与阶段 4 冻结
> **本小阶段固定执行规则**：执行前同时遵守 `01_阶段4通用执行规范.md`。GPU 作业先尝试提权查询；独立 job 在不产生共享写冲突时默认多 GPU 并行；本轮结束立即生成 ZIP；checkpoint、权重和其他大文件可以不打包，但必须写 manifest；只做推进本轮所需的最小检查。


## 总体上要干什么

把阶段 4.5 锁定的三个独立 seed 模型组成 deep ensemble，完成分类温度校准、回归不确定性统计和统一推理接口，然后冻结 `model_candidates_v1`，向阶段 5 交付可直接计算 graph reward 的模型输出。

本轮不再改变 encoder、head、loss 或 checkpoint。所有 calibration 只使用 validation；test 与 diagnostic 仅用于冻结后的结果报告。阶段 4 的出口状态命名为：

```text
GO_STAGE5
REFINE_STAGE4
MODEL_DATA_BLOCKED
```

这不是阶段 5 的正式 `G2`。`G2` 仍由阶段 5 的路径一致性、循环净回报和 recovery reward calibration 决定。

## 本轮目录与入口检查

```bash
set -euo pipefail
export REPO_ROOT="${REPO_ROOT:-/home/xushijie/CUPID}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage3"
export DIAG_ROOT="$STAGE3_ROOT/diagnostic_suite_v1"
export STAGE4_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/stage4"
export STAGE4_CONFIG="$REPO_ROOT/configs/stage4/stage4.yaml"
export SUP_ROOT="$STAGE4_ROOT/supervision_v1"
export JOINT_ROUND="$STAGE4_ROOT/rounds/stage4_5_joint_model_selection"
export ROUND_ID="stage4_6_uncertainty_and_freeze"
export ROUND_DIR="$STAGE4_ROOT/rounds/$ROUND_ID"
export CANDIDATE_ROOT="$STAGE4_ROOT/model_candidates_v1"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

cd "$REPO_ROOT"
"$REPO_ROOT/tools/stage4/init_round.sh" "$ROUND_ID" "$ROUND_DIR" \
  "Calibrate and assemble a three-seed PathGraph ensemble, freeze inference contract, and prepare Stage 5 handoff"

(
  cd "$SUP_ROOT"
  sha256sum -c SUPERVISION_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/supervision_checksum.log"
(
  cd "$DIAG_ROOT"
  sha256sum -c DIAGNOSTIC_SUITE_SHA256SUMS.txt
) | tee "$ROUND_DIR/logs/diagnostic_checksum.log"

test -f "$JOINT_ROUND/metrics/selection_lock.json"
test -f "$JOINT_ROUND/metrics/stage4_model_metrics.json"
cp "$JOINT_ROUND/metrics/selection_lock.json" "$ROUND_DIR/configs/"
cp "$JOINT_ROUND/metrics/stage4_model_metrics.json" "$ROUND_DIR/metrics/"

GPU_SNAPSHOT_DIR="$ROUND_DIR/manifests/gpu_snapshot"
"$REPO_ROOT/tools/stage4/query_gpus.sh" "$GPU_SNAPSHOT_DIR"
test -s "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt"
```

## 4.6.1 实现分类温度校准

创建：

```text
tools/stage4/calibrate_classification.py
tools/stage4/summarize_calibration.py
```

`calibrate_classification.py` 每次处理一个 seed，步骤固定为：

1. 从 `selection_lock.json` 读取该 seed checkpoint；
2. 在 `transport_recovery/val` 上重新推理并保存 node、edge-type、edge-id logits；
3. 分别拟合三个正标量温度：
   - `T_node`；
   - `T_edge_type`；
   - `T_edge_id`；
4. 使用 `softplus(raw_T)+1e-3` 保证温度为正；
5. 最小化 validation NLL；edge-id 只在 positive edge frame 上拟合；
6. 最多 200 个 LBFGS step；不更新 checkpoint 权重；
7. 若校准后 NLL 更差，则对应 head 回退 `T=1.0`，并记录 `fallback_identity=true`；
8. 计算校准前后 NLL、ECE、Brier score；
9. 只保存温度 JSON 和轻量预测指标，不保存大 logits 数组到 ZIP。

CLI：

```text
python -m tools.stage4.calibrate_classification
  --config CONFIG
  --supervision-dir SUP_ROOT
  --selection-lock LOCK
  --seed SEED
  --split val
  --output-dir OUT
```

每个 seed 输出：

```text
temperatures.json
calibration_metrics.json
reliability_bins.csv
plots/node_reliability.png
plots/edge_type_reliability.png
plots/edge_id_reliability.png
DONE
```

`temperatures.json` 示例：

```json
{
  "seed": 20260906,
  "source_split": "transport_recovery/val",
  "node_temperature": 1.13,
  "edge_type_temperature": 0.92,
  "edge_id_temperature": 1.05,
  "fallback_identity": {
    "node": false,
    "edge_type": false,
    "edge_id": false
  }
}
```

### 并行运行三个 calibration job

```bash
CAL_JOBS="$ROUND_DIR/configs/calibration_jobs.jsonl"
: > "$CAL_JOBS"

for seed in 20260906 20260907 20260908; do
  job_id="calibration_s${seed}"
  out="$ROUND_DIR/jobs/calibration/$job_id"
  cmd="$PYTHON_BIN -m tools.stage4.calibrate_classification --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --selection-lock '$JOINT_ROUND/metrics/selection_lock.json' --seed '$seed' --split val --output-dir '$out'"
  "$PYTHON_BIN" - <<PY >> "$CAL_JOBS"
import json
print(json.dumps({"job_id":"$job_id","command":"$cmd","output_dir":"$out","seed":$seed}))
PY
done

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$CAL_JOBS" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/calibration" \
  --status-csv "$ROUND_DIR/metrics/calibration_job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"

"$PYTHON_BIN" -m tools.stage4.summarize_calibration \
  --jobs-root "$ROUND_DIR/jobs/calibration" \
  --output-csv "$ROUND_DIR/tables/calibration_seed_summary.csv" \
  --output-json "$ROUND_DIR/metrics/calibration_summary.json" \
  --output-md "$ROUND_DIR/calibration_summary.md" \
  --plots-dir "$ROUND_DIR/plots/calibration"
```

## 4.6.2 实现 deep ensemble 输出

创建：

```text
tools/stage4/lib/ensemble.py
tools/stage4/build_ensemble_manifest.py
tools/stage4/run_ensemble_inference.py
```

### Ensemble 数学定义

对三个 seed 的分类概率 \(p_m(y\mid x)\)：

\[
\bar p(y\mid x)=\frac{1}{M}\sum_{m=1}^M p_m(y\mid x),\quad M=3.
\]

预测熵：

\[
H[\bar p]= -\sum_y \bar p_y\log(\bar p_y+\epsilon).
\]

互信息：

\[
MI=H[\bar p]-\frac1M\sum_m H[p_m].
\]

对 \(\phi\) 和 \(C_G\)：

\[
\mu=\frac1M\sum_m \hat y_m,
\qquad
\sigma=\sqrt{\frac1{M-1}\sum_m(\hat y_m-\mu)^2}.
\]

实现要求：

1. 每个模型先应用该 seed/head 的温度，再取概率平均；
2. node/edge-id 始终先应用 task mask；
3. `phi_mean` 裁剪到 `[0,1]`，但各 seed 原始输出和 std 不得因裁剪被静默覆盖；
4. `remaining_cost_mean/std` 非负；
5. 同一个 batch 的三个模型顺序执行或分 GPU 执行均可，优先选择不会导致 OOM 的方式；
6. ensemble 不复制 checkpoint，只引用 lock 中绝对路径并记录 SHA256；
7. 输出概率和必须在 `1e-5` 容差内为 1。

### 生成 ensemble manifest

```bash
"$PYTHON_BIN" -m tools.stage4.build_ensemble_manifest \
  --selection-lock "$JOINT_ROUND/metrics/selection_lock.json" \
  --calibration-root "$ROUND_DIR/jobs/calibration" \
  --stage4-config "$STAGE4_CONFIG" \
  --feature-schema "$SUP_ROOT/configs/feature_schema.json" \
  --label-maps "$SUP_ROOT/configs/label_maps.json" \
  --cost-target-spec "$SUP_ROOT/configs/cost_target_spec.yaml" \
  --output "$ROUND_DIR/configs/ensemble_manifest.json"

cat "$ROUND_DIR/configs/ensemble_manifest.json"
```

manifest 至少包含：

```text
bundle_version
history_steps
feature schema + SHA256
label maps + SHA256
cost normalizer/spec + SHA256
三个 seed 的 checkpoint path + SHA256
每个 seed 的三个 temperature
model config hash
graph spec hashes
statistics_unit=content_group_id
```

## 4.6.3 实现回归区间校准与不确定性评估

创建：

```text
tools/stage4/calibrate_regression_intervals.py
tools/stage4/evaluate_ensemble_uncertainty.py
tools/stage4/render_uncertainty_cases.py
```

### 回归区间校准

使用 `transport_recovery/val`，分别为 \(\phi\) 和 \(C_G\) 计算非负尺度系数。对于目标 coverage \(q\in\{0.80,0.90,0.95\}\)：

\[
s_q=\operatorname{Quantile}_q\left(
\frac{|y-\mu|}{\max(\sigma,10^{-6})}
\right).
\]

区间为：

\[
[\mu-s_q\sigma,\mu+s_q\sigma].
\]

- \(\phi\) 区间最终裁剪到 `[0,1]`；
- cost 下界裁剪到 0；
- scale 只用 validation 拟合；
- test 只报告实际 coverage 和平均宽度，不再改 scale。

运行：

```bash
"$PYTHON_BIN" -m tools.stage4.calibrate_regression_intervals \
  --config "$STAGE4_CONFIG" \
  --supervision-dir "$SUP_ROOT" \
  --ensemble-manifest "$ROUND_DIR/configs/ensemble_manifest.json" \
  --split val \
  --levels 0.80 0.90 0.95 \
  --output "$ROUND_DIR/configs/regression_interval_calibration.json" \
  --predictions "$ROUND_DIR/predictions/ensemble_val_predictions.jsonl" \
  2>&1 | tee "$ROUND_DIR/logs/regression_calibration.log"
```

### 冻结评估

依次在 `val`、`test` 和 Stage 3 diagnostic 上运行。val 与 test 可按 split 形成独立 GPU job并行；diagnostic 单独一个 job，避免三个进程同时写同一目录。

```bash
UNCERTAINTY_JOBS="$ROUND_DIR/configs/uncertainty_eval_jobs.jsonl"
: > "$UNCERTAINTY_JOBS"

for split in val test; do
  job_id="ensemble_${split}"
  out="$ROUND_DIR/jobs/uncertainty/$job_id"
  cmd="$PYTHON_BIN -m tools.stage4.evaluate_ensemble_uncertainty --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --ensemble-manifest '$ROUND_DIR/configs/ensemble_manifest.json' --regression-calibration '$ROUND_DIR/configs/regression_interval_calibration.json' --split '$split' --task transport_recovery --output-dir '$out'"
  "$PYTHON_BIN" - <<PY >> "$UNCERTAINTY_JOBS"
import json
print(json.dumps({"job_id":"$job_id","command":"$cmd","output_dir":"$out"}))
PY
done

job_id="ensemble_stage3_diagnostic"
out="$ROUND_DIR/jobs/uncertainty/$job_id"
cmd="$PYTHON_BIN -m tools.stage4.evaluate_ensemble_uncertainty --config '$STAGE4_CONFIG' --supervision-dir '$SUP_ROOT' --ensemble-manifest '$ROUND_DIR/configs/ensemble_manifest.json' --regression-calibration '$ROUND_DIR/configs/regression_interval_calibration.json' --diagnostic-suite '$DIAG_ROOT' --output-dir '$out'"
"$PYTHON_BIN" - <<PY >> "$UNCERTAINTY_JOBS"
import json
print(json.dumps({"job_id":"$job_id","command":"$cmd","output_dir":"$out"}))
PY

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/launch_parallel.py" \
  --jobs "$UNCERTAINTY_JOBS" \
  --gpu-ids-file "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt" \
  --logs-dir "$ROUND_DIR/logs/uncertainty" \
  --status-csv "$ROUND_DIR/metrics/uncertainty_job_status.csv" \
  --max-jobs-per-gpu "$MAX_JOBS_PER_GPU"
```

### 必须报告的分类校准/不确定性指标

对 node、edge-type、edge-id 分别报告：

```text
NLL before/after temperature
ECE before/after temperature
Brier score before/after temperature
predictive entropy mean/p90
mutual information mean/p90
error-detection AUROC using predictive entropy
error-detection AUROC using mutual information
```

edge-id 只在 positive edge frame 上计算主要分类指标；另报告 all-frame none accuracy。

### 必须报告的回归不确定性指标

对 \(\phi\) 和 \(C_G\) 分别报告：

```text
ensemble MAE/RMSE
std mean/p90
Spearman(std, absolute_error)
high-error detection AUROC using std
80/90/95% calibrated interval coverage
80/90/95% mean interval width
```

数据量不足时 AUROC 可为 `not_estimable`，但不能填成 0；这不会单独阻塞阶段 5。

渲染高不确定性样例：

```bash
"$PYTHON_BIN" -m tools.stage4.render_uncertainty_cases \
  --val-dir "$ROUND_DIR/jobs/uncertainty/ensemble_val" \
  --test-dir "$ROUND_DIR/jobs/uncertainty/ensemble_test" \
  --diagnostic-dir "$ROUND_DIR/jobs/uncertainty/ensemble_stage3_diagnostic" \
  --top-k 20 \
  --output-dir "$ROUND_DIR/plots/uncertainty_cases"
```

只渲染已有低维轨迹和事件标记；不因缺失视频而阻塞。

## 4.6.4 实现统一推理 API

创建：

```text
tools/stage4/infer_pathgraph_ensemble.py
tools/stage4/lib/streaming_inference.py
tools/stage4/verify_inference_contract.py
```

### Python API

必须能被阶段 5 直接导入：

```python
from tools.stage4.lib.ensemble import PathGraphEnsemble

model = PathGraphEnsemble.from_bundle(
    bundle_path=".../model_candidates_v1/model_bundle.json",
    device="cuda:0",
)
state = model.new_stream(task_id="transport_recovery")
output = model.step(state, current_feature_dict)
```

`model.step` 内部维护长度 32 的因果 history buffer，并输出：

```text
node_probs_mean
node_predictive_entropy
node_mutual_information
edge_type_probs_mean
edge_predictive_entropy
edge_mutual_information
edge_id_probs_mean
phi_mean
phi_std
remaining_cost_mean
remaining_cost_std
per_model_phi
per_model_remaining_cost
```

同时提供离线 CLI：

```text
python -m tools.stage4.infer_pathgraph_ensemble
  --bundle MODEL_BUNDLE_JSON
  --episode-npz EPISODE_NPZ
  --task-id TASK_ID
  --device cuda:0
  --output-jsonl OUTPUT
```

### 推理契约 smoke test

从 validation、test、recovery、terminal-failure 各取一个 episode；若某类不存在则跳过并记录。对每个 episode 执行：

- 离线 batch 推理；
- streaming step-by-step 推理；
- 比较两者结果，最大绝对差 `<=1e-5`；
- 检查概率和、\(\phi\) 范围、cost 非负、std 非负、无 NaN/Inf。

运行：

```bash
"$PYTHON_BIN" -m tools.stage4.verify_inference_contract \
  --supervision-dir "$SUP_ROOT" \
  --ensemble-manifest "$ROUND_DIR/configs/ensemble_manifest.json" \
  --regression-calibration "$ROUND_DIR/configs/regression_interval_calibration.json" \
  --device "cuda:$(head -n1 "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt")" \
  --report-json "$ROUND_DIR/metrics/inference_contract_report.json" \
  --examples-jsonl "$ROUND_DIR/predictions/inference_examples.jsonl" \
  2>&1 | tee "$ROUND_DIR/logs/inference_contract.log"
```

注意：当设置 `CUDA_VISIBLE_DEVICES` 时，进程内设备通常应使用 `cuda:0`。若上述命令直接使用物理 GPU id 会失败，则改为：

```bash
PHYSICAL_GPU_ID="$(head -n1 "$GPU_SNAPSHOT_DIR/available_gpu_ids.txt")"
CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU_ID" \
"$PYTHON_BIN" -m tools.stage4.verify_inference_contract \
  --supervision-dir "$SUP_ROOT" \
  --ensemble-manifest "$ROUND_DIR/configs/ensemble_manifest.json" \
  --regression-calibration "$ROUND_DIR/configs/regression_interval_calibration.json" \
  --device cuda:0 \
  --report-json "$ROUND_DIR/metrics/inference_contract_report.json" \
  --examples-jsonl "$ROUND_DIR/predictions/inference_examples.jsonl" \
  2>&1 | tee "$ROUND_DIR/logs/inference_contract.log"
```

## 4.6.5 冻结 `model_candidates_v1`

创建：

```text
tools/stage4/freeze_model_candidates.py
```

脚本必须执行：

1. 读取 `selection_lock.json`；
2. 确认三个 checkpoint 存在且 SHA256 匹配；
3. 读取 temperature 和 regression interval calibration；
4. 读取 `stage4_model_metrics.json` 和 ensemble uncertainty metrics；
5. 生成只引用、不复制 checkpoint 的 `model_bundle.json`；
6. 复制 feature schema、label maps、cost spec、GraphSpec/runtime hash、stage4 config；
7. 生成 checkpoint manifest；
8. 生成 `stage5_handoff.md`；
9. 作出内部出口决定；
10. 生成候选目录 checksum 并冻结。

运行：

```bash
rm -rf "$CANDIDATE_ROOT.tmp"
mkdir -p "$CANDIDATE_ROOT.tmp"

"$PYTHON_BIN" -m tools.stage4.freeze_model_candidates \
  --config "$STAGE4_CONFIG" \
  --supervision-dir "$SUP_ROOT" \
  --selection-lock "$JOINT_ROUND/metrics/selection_lock.json" \
  --joint-metrics "$JOINT_ROUND/metrics/stage4_model_metrics.json" \
  --ensemble-manifest "$ROUND_DIR/configs/ensemble_manifest.json" \
  --calibration-summary "$ROUND_DIR/metrics/calibration_summary.json" \
  --regression-calibration "$ROUND_DIR/configs/regression_interval_calibration.json" \
  --uncertainty-val "$ROUND_DIR/jobs/uncertainty/ensemble_val/metrics.json" \
  --uncertainty-test "$ROUND_DIR/jobs/uncertainty/ensemble_test/metrics.json" \
  --inference-contract "$ROUND_DIR/metrics/inference_contract_report.json" \
  --output-dir "$CANDIDATE_ROOT.tmp"
```

期望结构：

```text
model_candidates_v1/
  FROZEN.md
  stage4_exit_decision.md
  stage5_handoff.md
  model_bundle.json
  STAGE4_MODEL_CANDIDATES_SHA256SUMS.txt
  configs/
    stage4.yaml
    feature_schema.json
    label_maps.json
    cost_target_spec.yaml
    regression_interval_calibration.json
  calibration/
    seed_20260906_temperatures.json
    seed_20260907_temperatures.json
    seed_20260908_temperatures.json
  manifests/
    checkpoint_manifest.tsv
    graph_spec_manifest.tsv
    code_manifest.tsv
  metrics/
    validation_metrics.json
    frozen_test_metrics.json
    uncertainty_metrics.json
  reports/
    stage4_model_summary.md
```

### 阶段 4 出口判定

`GO_STAGE5`：

- 三个 selected checkpoint 都可读取并通过 hash；
- 至少 2/3 seed 达到阶段 4.5 的主要模型门槛；
- ensemble 输出无 NaN/Inf；
- node/edge 概率合法、\(\phi\in[0,1]\)、\(C_G\ge0\)；
- streaming 与 batch 推理一致；
- uncertainty 字段已生成；
- Stage 5 handoff 可定位所有模型与配置。

`REFINE_STAGE4`：

- 模型可运行，但仅一个 head 未达到门槛或校准/推理接口有明确可修复问题；
- 文档必须给出只针对该问题的一轮最小修正，不开启新架构搜索。

`MODEL_DATA_BLOCKED`：

- 0/3 seed 能学习核心 node/edge/cost；或冻结数据本身无法形成必要标签；
- 此结论必须由当前阶段结果支持，不得仅凭一次训练失败作出。

温度校准是否明显改善 ECE、uncertainty AUROC 是否很高，不单独作为 GO_STAGE5 的硬门槛；它们主要用于阶段 5 的保守加权分析。

### 冻结目录

```bash
cat "$CANDIDATE_ROOT.tmp/stage4_exit_decision.md"
grep -Eq 'GO_STAGE5|REFINE_STAGE4|MODEL_DATA_BLOCKED' \
  "$CANDIDATE_ROOT.tmp/stage4_exit_decision.md"

(
  cd "$CANDIDATE_ROOT.tmp"
  find . -type f ! -name 'STAGE4_MODEL_CANDIDATES_SHA256SUMS.txt' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > STAGE4_MODEL_CANDIDATES_SHA256SUMS.txt
  sha256sum -c STAGE4_MODEL_CANDIDATES_SHA256SUMS.txt
)

rm -rf "$CANDIDATE_ROOT"
mv "$CANDIDATE_ROOT.tmp" "$CANDIDATE_ROOT"
```

如果出口决定不是 `GO_STAGE5`，仍然生成本轮 ZIP，但不要伪称阶段 4 已完成；按 `stage4_exit_decision.md` 的最小修正重新运行对应小阶段。

## 4.6.6 生成本轮摘要与 ZIP

把关键结果复制到本轮目录：

```bash
cp "$CANDIDATE_ROOT/stage4_exit_decision.md" "$ROUND_DIR/"
cp "$CANDIDATE_ROOT/stage5_handoff.md" "$ROUND_DIR/"
cp "$CANDIDATE_ROOT/model_bundle.json" "$ROUND_DIR/configs/"
cp "$CANDIDATE_ROOT/manifests/checkpoint_manifest.tsv" "$ROUND_DIR/checkpoint_manifest.tsv"
cp "$CANDIDATE_ROOT/metrics/uncertainty_metrics.json" "$ROUND_DIR/metrics/" 2>/dev/null || true

cat > "$ROUND_DIR/summary.md" <<EOF
# Stage 4.6 summary

- completed_at: $(date -Iseconds)
- selection_lock: $JOINT_ROUND/metrics/selection_lock.json
- ensemble_size: 3
- calibration_source: transport_recovery/val only
- model_bundle: $CANDIDATE_ROOT/model_bundle.json
- checkpoint_packaging: omitted; see checkpoint_manifest.tsv

## Exit decision

$(cat "$CANDIDATE_ROOT/stage4_exit_decision.md")
EOF

{
  echo "- finished_at: $(date -Iseconds)"
  echo "- candidate_root: $CANDIDATE_ROOT"
  echo "- exit_decision_file: $CANDIDATE_ROOT/stage4_exit_decision.md"
} >> "$ROUND_DIR/run_manifest.md"

"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id "$ROUND_ID" \
  --round-dir "$ROUND_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"
```

## 4.6.7 仅在 `GO_STAGE5` 时生成 `stage4_complete.zip`

先确认六轮 ZIP 都存在并通过完整性检查：

```bash
EXPECTED_ROUNDS=(
  stage4_1_supervision_and_encoder_input
  stage4_2_node_edge_heads
  stage4_3_within_node_progress
  stage4_4_remaining_cost
  stage4_5_joint_model_selection
  stage4_6_uncertainty_and_freeze
)

for rid in "${EXPECTED_ROUNDS[@]}"; do
  zip_path="$STAGE4_ROOT/downloads/${rid}.zip"
  sha_path="${zip_path}.sha256"
  test -f "$zip_path"
  test -f "$sha_path"
  unzip -t "$zip_path" > "$STAGE4_ROOT/downloads/${rid}_unzip_test_final.txt"
  (cd "$STAGE4_ROOT/downloads" && sha256sum -c "${rid}.zip.sha256")
done

grep -q 'GO_STAGE5' "$CANDIDATE_ROOT/stage4_exit_decision.md"
```

创建轻量总交付目录。checkpoint、episode `.npz`、原始视频、大数组和缓存不复制：

```bash
export COMPLETE_DIR="$STAGE4_ROOT/stage4_complete_bundle"
rm -rf "$COMPLETE_DIR"
mkdir -p "$COMPLETE_DIR"/{configs,supervision,round_summaries,model_candidates,manifests,tools_snapshot}

cp "$STAGE4_CONFIG" "$COMPLETE_DIR/configs/stage4.yaml"
cp "$SUP_ROOT/FROZEN.md" "$COMPLETE_DIR/supervision/"
cp "$SUP_ROOT/SUPERVISION_SHA256SUMS.txt" "$COMPLETE_DIR/supervision/"
rsync -a "$SUP_ROOT/configs/" "$COMPLETE_DIR/supervision/configs/"
rsync -a "$SUP_ROOT/tables/" "$COMPLETE_DIR/supervision/tables/"
rsync -a "$SUP_ROOT/reports/" "$COMPLETE_DIR/supervision/reports/"
rsync -a "$SUP_ROOT/probes/" "$COMPLETE_DIR/supervision/probes/"
rsync -a "$CANDIDATE_ROOT/" "$COMPLETE_DIR/model_candidates/"

for rid in "${EXPECTED_ROUNDS[@]}"; do
  src="$STAGE4_ROOT/rounds/$rid"
  dst="$COMPLETE_DIR/round_summaries/$rid"
  mkdir -p "$dst"
  for item in run_manifest.md summary.md validation_summary.md selection_lock.md \
              stage4_exit_decision.md stage5_handoff.md checkpoint_manifest.tsv \
              large_file_manifest.tsv; do
    [ -f "$src/$item" ] && cp "$src/$item" "$dst/"
  done
  [ -d "$src/metrics" ] && rsync -a \
    --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' --exclude='*.pth' \
    "$src/metrics/" "$dst/metrics/"
  [ -d "$src/tables" ] && rsync -a "$src/tables/" "$dst/tables/"
  [ -d "$src/plots" ] && rsync -a "$src/plots/" "$dst/plots/"
done

rsync -a \
  --include='*/' --include='*.py' --include='*.sh' --exclude='*' \
  "$REPO_ROOT/tools/stage4/" "$COMPLETE_DIR/tools_snapshot/"
```

生成每轮 ZIP 清单：

```bash
{
  printf 'round_id\tzip_path\tsha256\tsize_bytes\n'
  for rid in "${EXPECTED_ROUNDS[@]}"; do
    z="$STAGE4_ROOT/downloads/${rid}.zip"
    printf '%s\t%s\t%s\t%s\n' \
      "$rid" "$z" "$(sha256sum "$z" | awk '{print $1}')" "$(stat -c%s "$z")"
  done
} > "$COMPLETE_DIR/manifests/round_zip_manifest.tsv"

cp "$CANDIDATE_ROOT/manifests/checkpoint_manifest.tsv" \
  "$COMPLETE_DIR/manifests/checkpoint_manifest.tsv"
```

写总摘要和 manifest：

```bash
cat > "$COMPLETE_DIR/run_manifest.md" <<EOF
# Stage 4 complete manifest

- completed_at: $(date -Iseconds)
- entry_gate: G1=GO_STAGE4
- exit_state: GO_STAGE5
- rounds: 6
- ensemble_size: 3
- statistics_unit: content_group_id
- primary_task: transport_recovery
- mechanism_probe: transport_dual_order
- checkpoint_packaging: omitted; see manifests/checkpoint_manifest.tsv
EOF

cat > "$COMPLETE_DIR/summary.md" <<EOF
# PathGraph-SARM Stage 4 complete

$(cat "$CANDIDATE_ROOT/reports/stage4_model_summary.md" 2>/dev/null || true)

## Exit decision

$(cat "$CANDIDATE_ROOT/stage4_exit_decision.md")

## Stage 5 handoff

$(cat "$CANDIDATE_ROOT/stage5_handoff.md")
EOF
```

使用统一轻量打包器生成总 ZIP：

```bash
"$PYTHON_BIN" "$REPO_ROOT/tools/stage4/package_round.py" \
  --round-id stage4_complete \
  --round-dir "$COMPLETE_DIR" \
  --downloads-dir "$STAGE4_ROOT/downloads" \
  --max-file-mb "$ZIP_MAX_FILE_MB"

unzip -t "$STAGE4_ROOT/downloads/stage4_complete.zip" \
  | tee "$STAGE4_ROOT/downloads/stage4_complete_unzip_test_final.txt"
sha256sum "$STAGE4_ROOT/downloads/stage4_complete.zip" \
  | tee "$STAGE4_ROOT/downloads/stage4_complete.zip.sha256"
```

Agent 最终必须在回复中提供：

```text
stage4_complete.zip 的绝对路径
SHA256
阶段 4 出口状态
六个逐轮 ZIP 的路径
未打包 checkpoint 的 manifest 路径
Stage 5 handoff 路径
```

## 本轮及阶段 4 完成条件

- 三个 seed 已组成 ensemble；
- 分类温度只在 validation 上拟合；
- regression interval scale 只在 validation 上拟合；
- validation/test/diagnostic 不确定性指标均已输出；
- streaming 与 batch 推理契约通过；
- `model_candidates_v1` 已冻结并通过 checksum；
- `stage4_exit_decision.md` 为 `GO_STAGE5`；
- 六个小阶段 ZIP 均存在并通过 `unzip -t`；
- `stage4_complete.zip` 存在、可解压并附 SHA256；
- checkpoint 等大文件未强制打包，但 manifest 完整。

本阶段停止点：`GO_STAGE5` 后结束阶段 4。下一阶段开始构造 graph reward、路径归一、failure-recovery loop 净回报和 uncertainty lower-bound，不在阶段 4 继续改变模型。

<!-- END FILE: 阶段4.6_不确定性与冻结.md -->
