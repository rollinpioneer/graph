#!/usr/bin/env bash
# Authoritative environment for the simulator-scoped U2 stochastic-boundary protocol.
export REPO_ROOT="$(git rev-parse --show-toplevel)"
export PYTHON_BIN="/home/__compress_data/xushijie/.conda/envs/cupid/bin/python"
export BASE_COMMIT="07422057a1ec2142e015dbf69671e029e77fbbc3"
export UPGRADE_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2"
export U1_BRIDGE="$UPGRADE_ROOT/results/u1_data_bridge"
export U1_FINAL="$UPGRADE_ROOT/results/u1_final"
export U2_HANDOFF="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge/u2_handoff_v2.json"
export U2_ROOT="$REPO_ROOT/artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary"
export U2_DATA="$U2_ROOT/data_v1"
export U2_WEAK="$U2_ROOT/weak_events_v1"
export U2_BASELINES="$U2_ROOT/segmentation_baselines_v1"
export U2_MODELS="$U2_ROOT/boundary_models_v1"
export U2_SEGMENTS="$U2_ROOT/segment_representation_v1"
export U2_BUDGET="$U2_ROOT/budgeted_correction_v1"
export U2_REWARD="$U2_ROOT/reward_impact_v1"
export U2_VALUE="$U2_REWARD/value_models"
export U2_FINAL="$U2_ROOT/final_v1"
export U2_ROUNDS="$U2_ROOT/rounds"
export U2_TOOLS="$REPO_ROOT/upgrade_v2/u2"
export U2_DOWNLOADS="$REPO_ROOT/downloads/upgrade_v2"
export GPU_MIN_FREE_MB=12000
export MAX_JOBS_PER_GPU=1
export ZIP_MAX_FILE_MB=20
export OMP_NUM_THREADS=4
# User instruction overrides per-round archives: always refresh only this one ZIP.
export U2_SINGLE_ZIP="$U2_DOWNLOADS/U0_U1_complete.zip"
