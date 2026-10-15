#!/usr/bin/env bash
# U3 runtime paths only. API secrets remain in a repository-external private
# environment file and are never sourced here.
set -euo pipefail

export REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export U2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary"
export U2_PATCH="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_handoff_patch_v1"
export U3_INPUT_V1="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph/inputs_v1"
export U3_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph_v2"
export U3_REPAIR="$U3_ROOT/pre_u3_repair_v1"
export U3_PROTOCOL="$U3_ROOT/model_protocol_v1"
export U3_REQUESTS="$U3_ROOT/requests_v2"
export U3_RESPONSES="$U3_ROOT/responses_v1"
export U3_VALIDATION="$U3_ROOT/candidate_validation_v1"
export U3_SELECTION="$U3_ROOT/candidate_selection_v1"
export U3_FINAL="$U3_ROOT/final_v1"
export U3_ROUNDS="$U3_ROOT/rounds"
export U3_TOOLS="$REPO_ROOT/upgrade_v2/u3"

export QWEN_MODEL="qwen3.7-plus"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export QWEN_CONCURRENCY="${QWEN_CONCURRENCY:-3}"
export DEEPSEEK_CONCURRENCY="${DEEPSEEK_CONCURRENCY:-3}"
export MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-5000}"
export MAX_PROMPT_CHARS="${MAX_PROMPT_CHARS:-120000}"
export API_TIMEOUT_SECONDS="${API_TIMEOUT_SECONDS:-300}"
export API_NETWORK_RETRIES="${API_NETWORK_RETRIES:-4}"
export DEEPSEEK_SCHEMA_REPAIR_LIMIT="${DEEPSEEK_SCHEMA_REPAIR_LIMIT:-1}"
export ZIP_MAX_FILE_MB="${ZIP_MAX_FILE_MB:-200}"

mkdir -p "$U3_ROOT" "$U3_REPAIR" "$U3_PROTOCOL" "$U3_REQUESTS" \
  "$U3_RESPONSES" "$U3_VALIDATION" "$U3_SELECTION" "$U3_FINAL" "$U3_ROUNDS"
