# P2C-Q actual commands (server gpu03)

implementation_commit=7609dfa56d1f3f85046c096f08d39b06457628a8
PYTHONPATH uses EXEC worktree; reports written on WT.

Paths:
- PKG=/home/__compress_data/xushijie/PathGraph_P2C_Q_Agent_Package_V1.0
- DATA=/home/__compress_data/xushijie/graph_pathgraph_p2c_q_data/run_v1
- EXEC=/home/__compress_data/xushijie/graph_pathgraph_p2c_q_v1_exec
- WT=/home/__compress_data/xushijie/graph_pathgraph_p2c_q_v1_worktree
- ART=$WT/artifacts/pathgraph_sarm/upgrade_v2/p2c_q_decision_separation_v1

## CLI help (exit 0)

```
export PYTHONPATH=$EXEC/experiments/pathgraph_p2c_q_v1
python -B -m p2cq_research.cli --help
python -B -m p2cq_research.cli export --help
python -B $PKG/tools/qualify_export.py --help
```

## Formal ledger runs (RESERVED_NO_REFUND, no retry)

### P2CQ_DEV_EXPORT_001 -> $DATA/development/export

```
python -B $PKG/tools/reserve_run.py \
  --ledger $DATA/control/campaign_ledger.json \
  --protocol $PKG/contracts/protocol.json \
  --run-id P2CQ_DEV_EXPORT_001 --stage DEV_EXPORT \
  --output-root $DATA/development/export
export PYTHONPATH=$EXEC/experiments/pathgraph_p2c_q_v1
python -B -m p2cq_research.cli export --split development --out $DATA/development/export
```

Result: n_pairs=2048 n_mdps=256 n_mdp_states=456984 omitted=0 clip=0 unreachable=0; export_bytes=2099559566; EXPORT_MANIFEST files_verified=514 status=PASS.

### P2CQ_DEV_QUALIFY_001 -> $DATA/development/analysis

```
python -B $PKG/tools/build_export_manifest.py --root $DATA/development/export
python -B $PKG/tools/reserve_run.py \
  --ledger $DATA/control/campaign_ledger.json \
  --protocol $PKG/contracts/protocol.json \
  --run-id P2CQ_DEV_QUALIFY_001 --stage DEV_QUALIFY \
  --output-root $DATA/development/analysis
python -B $PKG/tools/qualify_export.py \
  --input $DATA/development/export \
  --contract $PKG/contracts/qualification.json \
  --out $DATA/development/analysis
```

Result: ENVIRONMENT_DECISION_SEPARATION_PASS; separating_pairs=1620; QUALIFY_DONE in $DATA/control/dev_chain.log.

### P2CQ_CONFIRM_EXPORT_001 -> $DATA/confirmation/export

```
python -B $PKG/tools/reserve_run.py \
  --ledger $DATA/control/campaign_ledger.json \
  --protocol $PKG/contracts/protocol.json \
  --run-id P2CQ_CONFIRM_EXPORT_001 --stage CONFIRM_EXPORT \
  --output-root $DATA/confirmation/export
export PYTHONPATH=$EXEC/experiments/pathgraph_p2c_q_v1
python -B -m p2cq_research.cli export --split confirmation --out $DATA/confirmation/export
```

Result: n_pairs=2048 n_mdps=256 n_mdp_states=456984 omitted=0 clip=0 unreachable=0; export_bytes=2102164790; families 1302000-1302331; EXPORT_MANIFEST files_verified=514 status=PASS. Logged in $DATA/control/confirm_export.log.

### P2CQ_CONFIRM_QUALIFY_001 -> $DATA/confirmation/analysis

```
python -B $PKG/tools/build_export_manifest.py --root $DATA/confirmation/export
python -B $PKG/tools/reserve_run.py \
  --ledger $DATA/control/campaign_ledger.json \
  --protocol $PKG/contracts/protocol.json \
  --run-id P2CQ_CONFIRM_QUALIFY_001 --stage CONFIRM_QUALIFY \
  --output-root $DATA/confirmation/analysis
python -B $PKG/tools/qualify_export.py \
  --input $DATA/confirmation/export \
  --contract $PKG/contracts/qualification.json \
  --out $DATA/confirmation/analysis
```

Result: ENVIRONMENT_DECISION_SEPARATION_PASS; separating_pairs=1620; all_environment_gates_passed=true; QUALIFY_DONE 2026-09-17T02:24:18+08:00. Qualify PID and watcher exited.

## Finalize and engineering checks

```
python -B $DATA/control/finalize_p2cq.py
# FINALIZE_EXIT:0 2026-09-17T02:24:43+08:00
python -B $PKG/tools/validate_server_acceptance.py \
  --input $ART/server_acceptance.json --evidence-root $ART
python -B -m pytest -q tests/test_semantics.py
python -B $PKG/tools/verify_package.py
```

Tests: pytest 22 passed, exit 0. Package verify_package.py exit 0.
Ledger events remain RESERVED_NO_REFUND; no refund/retry/directory reset.
