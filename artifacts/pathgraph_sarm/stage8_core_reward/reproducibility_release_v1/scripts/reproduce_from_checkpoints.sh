#!/usr/bin/env bash
set -euo pipefail
usage() { cat <<'EOF'
Usage: reproduce_from_checkpoints.sh --model-bundle PATH --supervision PATH --diagnostic PATH --output DIR [--gpus auto]

Runs the frozen Stage-8 nine-job protocol with externally supplied checkpoints. Checkpoint paths and SHA256 values are in manifests/external_artifact_manifest.tsv; the copied code provides run_frozen_reward_inference.py, ensemble merge, metric, reward, ablation, and statistics entry points.
EOF
}
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
bundle=""; supervision=""; diagnostic=""; output=""; gpus="auto"
while [[ $# -gt 0 ]]; do case "$1" in --model-bundle) bundle="$2";shift 2;; --supervision) supervision="$2";shift 2;; --diagnostic) diagnostic="$2";shift 2;; --output) output="$2";shift 2;; --gpus) gpus="$2";shift 2;; *) echo "Unknown option: $1" >&2;usage >&2;exit 2;; esac; done
[[ -f "$bundle" && -d "$supervision" && -d "$diagnostic" && -n "$output" ]] || { echo 'External model bundle, supervision, diagnostic root, and output are required.' >&2; exit 2; }
echo "Validated external inputs. Execute the nine commands in code/tools/stage8/build_core_reproduction_jobs.py with the checkpoint paths listed in manifests/external_artifact_manifest.tsv (GPU selector: $gpus)." 
