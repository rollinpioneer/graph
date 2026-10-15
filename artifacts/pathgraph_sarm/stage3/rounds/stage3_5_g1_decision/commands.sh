#!/usr/bin/env bash
set -euo pipefail
python tools/stage3/evaluate_g1_gate.py --config configs/stage3/stage3.yaml --analysis-dir artifacts/pathgraph_sarm/stage3/rounds/stage3_4_misscoring_analysis --output-dir artifacts/pathgraph_sarm/stage3/rounds/stage3_5_g1_decision
