#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:?output directory}"; mkdir -p "$OUT_DIR"
FIELDS='index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu'
if sudo -n nvidia-smi >/dev/null 2>&1; then MODE=sudo_noninteractive; CMD='sudo -n nvidia-smi'; NOTE=''; else MODE=direct_fallback; CMD='nvidia-smi'; NOTE='note=sudo -n unavailable; direct nvidia-smi visibility used'; fi
printf 'mode=%s\n%s\n' "$MODE" "$NOTE" > "$OUT_DIR/query_mode.txt"
$CMD | tee "$OUT_DIR/nvidia_smi_full.txt"
$CMD --query-gpu="$FIELDS" --format=csv,noheader,nounits | tee "$OUT_DIR/gpu_status.csv"
awk -F',' -v min="${GPU_MIN_FREE_MB:-6000}" '{gsub(/ /,"",$1);gsub(/ /,"",$6);if(($6+0)>=min)print $1}' "$OUT_DIR/gpu_status.csv" > "$OUT_DIR/available_gpu_ids.txt"
echo "available_gpu_ids=$(paste -sd, "$OUT_DIR/available_gpu_ids.txt")" >> "$OUT_DIR/query_mode.txt"
