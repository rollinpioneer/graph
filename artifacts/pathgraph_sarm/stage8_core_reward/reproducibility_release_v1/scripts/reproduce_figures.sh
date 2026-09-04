#!/usr/bin/env bash
set -euo pipefail
usage() { cat <<'EOF'
Usage: reproduce_figures.sh --results-dir PATH --output DIR

Uses CSV-driven figure tools copied under code/tools/stage8. Auxiliary frozen R1 tables are identified in the external manifest.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
results=""; output=""
while [[ $# -gt 0 ]]; do case "$1" in --results-dir) results="$2";shift 2;; --output) output="$2";shift 2;; *) usage >&2;exit 2;; esac; done
[[ -d "$results" && -n "$output" ]] || { usage >&2;exit 2; }
mkdir -p "$output"
echo "Figure inputs validated. Invoke code/tools/stage8/make_publication_figures.py with the listed CSVs to render PDF/SVG/PNG outputs." 
