#!/usr/bin/env bash
set -euo pipefail
OUT_DIR="${1:?Usage: $0 OUT_DIR [MIN_FREE_MB]}"; mkdir -p "$OUT_DIR"
if sudo -n nvidia-smi >/dev/null 2>&1; then
  sudo -n nvidia-smi | tee "$OUT_DIR/nvidia_smi_full.txt"
  sudo -n nvidia-smi --query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | tee "$OUT_DIR/gpu_status.csv"
  sudo -n nvidia-smi --query-gpu=index --format=csv,noheader,nounits > "$OUT_DIR/available_gpu_ids.txt"
else
  echo 'GPU query unavailable; CPU-only stage2 run' > "$OUT_DIR/gpu_status.csv"
  : > "$OUT_DIR/available_gpu_ids.txt"
fi
