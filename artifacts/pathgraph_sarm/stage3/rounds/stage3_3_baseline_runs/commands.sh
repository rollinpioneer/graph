#!/usr/bin/env bash
set -euo pipefail
echo 'CUDA-visible learned jobs are recorded in jobs.tsv and job_status.tsv.'
python tools/stage3/score_oracle_baselines.py --adapter-dir artifacts/pathgraph_sarm/stage3/input_adapter_v1 --suite-dir artifacts/pathgraph_sarm/stage3/diagnostic_suite_v1 --linearization-spec artifacts/pathgraph_sarm/stage3/diagnostic_suite_v1/configs/linearization_specs.yaml --methods linear_time_fraction,oracle_linear_chain,sequential_transition_oracle --output-dir artifacts/pathgraph_sarm/stage3/rounds/stage3_3_baseline_runs/predictions/oracle
