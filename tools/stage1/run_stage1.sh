#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/stage1/stage1.yaml}"
ROOT="artifacts/pathgraph_sarm/stage1"
PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$ROOT/1.1_asset_inventory" "$ROOT/1.2_trajectory_coverage" "$ROOT/1.3_dataset_v0.1" "$ROOT/1.4_g0_decision"
"$PYTHON_BIN" tools/stage1/scan_assets.py --config "$CONFIG" --output-dir "$ROOT/1.1_asset_inventory"
"$PYTHON_BIN" tools/stage1/analyze_trajectory_structure.py --config "$CONFIG" --inventory "$ROOT/1.1_asset_inventory/asset_inventory.csv" --output-dir "$ROOT/1.2_trajectory_coverage" --reuse-manual-labels
"$PYTHON_BIN" tools/stage1/build_dataset_v01.py --config "$CONFIG" --inventory "$ROOT/1.1_asset_inventory/asset_inventory.csv" --tags "$ROOT/1.2_trajectory_coverage/trajectory_tags.csv" --path-signatures "$ROOT/1.2_trajectory_coverage/path_signatures.jsonl" --output-dir "$ROOT/1.3_dataset_v0.1"
"$PYTHON_BIN" tools/stage1/select_graph_tasks.py --config "$CONFIG" --task-summary "$ROOT/1.2_trajectory_coverage/task_structure_summary.csv" --coverage "$ROOT/1.2_trajectory_coverage/coverage_matrix.csv" --tags "$ROOT/1.2_trajectory_coverage/trajectory_tags.csv" --manifest "$ROOT/1.3_dataset_v0.1/episode_manifest.jsonl" --output-dir "$ROOT/1.4_g0_decision"
echo "Stage 1 finished. Read: $ROOT/1.4_g0_decision/g0_decision.md"
