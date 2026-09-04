#!/usr/bin/env bash
set -euo pipefail
OUT="${1:?output path}"; mkdir -p "$(dirname "$OUT")"
if sudo -n nvidia-smi >/dev/null 2>&1; then mode=sudo_noninteractive; cmd='sudo -n nvidia-smi'; note=''; else mode=direct_fallback; cmd='nvidia-smi'; note='sudo -n unavailable; direct nvidia-smi used'; fi
printf 'mode=%s\n%s\n' "$mode" "$note" > "$OUT"
$cmd --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits >> "$OUT"
