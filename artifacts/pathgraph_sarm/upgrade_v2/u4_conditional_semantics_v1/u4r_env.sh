#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
U2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary"
U2_CHECKPOINT="$U2_ROOT/boundary_models_v1/formal/offline_teacher_to_causal_s623/best.pt"
U4B_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u4_bplus_v1"
U4B_TORCH="$U4B_ROOT/torch_recovery_v2"
U4B_DELIVERY="$U4B_ROOT/delivery"
U4R_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u4_conditional_semantics_v1"
U4R_PROTOCOL="$U4R_ROOT/protocol_v1"
U4R_EVAL="$U4R_ROOT/evaluator_v2"
U4R_RESCORE="$U4R_ROOT/frozen_rescore_v1"
U4R_SEPARABILITY="$U4R_ROOT/semantic_separability_v1"
U4R_GRAPHS="$U4R_ROOT/conditional_graphs_v1"
U4R_CONFIRM="$U4R_ROOT/fresh_confirmation_v1"
U4R_FINAL="$U4R_ROOT/final_v1"
U4R_ROUNDS="$U4R_ROOT/rounds"
U4R_TOOLS="$REPO_ROOT/upgrade_v2/u4_conditional"
U4R_DOWNLOADS="${U4R_DOWNLOADS:-$REPO_ROOT/downloads/u4r1}"

PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/cupid/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python)"; fi
CPU_WORKERS="${CPU_WORKERS:-$(nproc --all)}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-6000}"
ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"

export REPO_ROOT U2_ROOT U2_CHECKPOINT U4B_ROOT U4B_TORCH U4B_DELIVERY
export U4R_ROOT U4R_PROTOCOL U4R_EVAL U4R_RESCORE U4R_SEPARABILITY U4R_GRAPHS U4R_CONFIRM U4R_FINAL U4R_ROUNDS U4R_TOOLS U4R_DOWNLOADS
export PYTHON_BIN CPU_WORKERS MAX_JOBS_PER_GPU GPU_MIN_FREE_MB ZIP_MAX_FILE_MB

mkdir -p "$U4R_ROOT" "$U4R_PROTOCOL" "$U4R_EVAL" "$U4R_RESCORE" "$U4R_SEPARABILITY" "$U4R_GRAPHS" "$U4R_CONFIRM" "$U4R_FINAL" "$U4R_ROUNDS" "$U4R_DOWNLOADS"
