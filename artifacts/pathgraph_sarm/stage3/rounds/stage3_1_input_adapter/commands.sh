#!/usr/bin/env bash
set -euo pipefail
python tools/stage3/prepare_stage3_inputs.py --m1-root artifacts/pathgraph_sarm/stage2/m1_freeze_v1 --repo-root /home/xushijie/CUPID --stage2-collection-root artifacts/pathgraph_sarm/stage2/rounds/stage2_2_targeted_collection/jobs/synthetic --stage1-manifest artifacts/pathgraph_sarm/stage1/1.3_dataset_v0.1/episode_manifest.jsonl --output-dir artifacts/pathgraph_sarm/stage3/input_adapter_v1 --apply-known-errata --derive-edges-from-gt --compute-content-groups
