#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:?provide a new checks output directory}
PY=${PY:-python}
test ! -e "$OUT"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
"$PY" -B -m unittest discover -s tests -p 'test_arithmetic.py' -v >"$OUT/arithmetic.log" 2>&1
if [[ -n "${P2B_SOURCE_REPO:-}" && -n "${P2B_V6_TOOLS:-}" ]]; then
  "$PY" -B -m unittest discover -s tests -p 'test_frozen_adapter.py' -v >"$OUT/frozen_adapter.log" 2>&1
else
  echo 'SOURCE_INTEGRATION_NOT_RUN: set P2B_SOURCE_REPO and P2B_V6_TOOLS' >"$OUT/source_status.txt"
fi
"$PY" -B tools/verify_package.py --root "$ROOT" >"$OUT/package.log"
printf '%s\n' 'Tool checks completed. Upstream PPO smoke and formal experiments are separate.'
