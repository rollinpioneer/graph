#!/usr/bin/env bash
set -euo pipefail
python tools/stage3/build_diagnostic_suite.py --adapter-dir artifacts/pathgraph_sarm/stage3/input_adapter_v1 --m1-root artifacts/pathgraph_sarm/stage2/m1_freeze_v1 --linearization-spec artifacts/pathgraph_sarm/stage3/diagnostic_suite_v1/configs/linearization_specs.yaml --output-dir artifacts/pathgraph_sarm/stage3/diagnostic_suite_v1 --seed 20260903
