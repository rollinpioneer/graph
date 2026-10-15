#!/usr/bin/env bash
set -euo pipefail

export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export REQUIRED_COMMIT="78da21364e09d2a0b4f8cc3b47fde147ef09b5f9"
export U2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary"
export U2_PATCH="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_handoff_patch_v1"
export U3_V1="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph_v2"
export U3_V1_RESPONSES="$U3_V1/responses_v1"
export U3_V1_VALIDATION="$U3_V1/candidate_validation_v1"
export U3_V1_FINAL="$U3_V1/final_v1"
export U3_V1_INPUT="$U3_V1/pre_u3_repair_v1"
export U3G_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u3_grounding_bridge_v1"
export U3G_PROTOCOL="$U3G_ROOT/protocol_v1"
export U3G_EVIDENCE="$U3G_ROOT/evidence_catalog_v1"
export U3G_DATA_GRAPH="$U3G_ROOT/data_only_graph_v1"
export U3G_GROUNDED="$U3G_ROOT/grounded_candidates_v1"
export U3G_EVAL="$U3G_ROOT/evaluation_v1"
export U3G_FINAL="$U3G_ROOT/final_v1"
export U3G_ROUNDS="$U3G_ROOT/rounds"
export U3G_TOOLS="$REPO_ROOT/upgrade_v2/u3_grounding"
export U3G_DOWNLOADS="${U3G_DOWNLOADS:-/tmp/pathgraph_u3g_stage_zips}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"
export CPU_WORKERS="${CPU_WORKERS:-$(nproc --all)}"

mkdir -p "$U3G_ROOT" "$U3G_PROTOCOL" "$U3G_EVIDENCE" "$U3G_DATA_GRAPH" \
  "$U3G_GROUNDED" "$U3G_EVAL" "$U3G_FINAL" "$U3G_ROUNDS" "$U3G_DOWNLOADS"
