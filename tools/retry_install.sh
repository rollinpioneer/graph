#!/bin/bash
cd /home/__compress_data/xushijie/graph_cp_disr_v2_1
export UV_CACHE_DIR="$PWD/.cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.cache/python" UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=2
out=experiments/part_0_validation/stage_0a/host_logs
timeout 900 .bootstrap/bin/uv sync --frozen --python .venv/bin/python > "$out/install_attempt_02.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out/install_attempt_02.exit"
if test "$rc" -eq 0; then
 .bootstrap/bin/uv pip check --python .venv/bin/python > "$out/pip_check.log" 2>&1
 printf '%s\n' "$?" > "$out/pip_check.exit"
 .bootstrap/bin/uv pip freeze --python .venv/bin/python > "$out/pip_freeze.txt"
 .venv/bin/python tools/stage0a_software_probe.py > "$out/software_probe.log" 2>&1
 printf '%s\n' "$?" > "$out/software_probe.exit"
fi
printf 'DONE\n' > "$out/retry.done"
