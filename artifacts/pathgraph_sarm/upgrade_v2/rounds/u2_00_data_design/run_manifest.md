# Run Manifest — U2.0 stochastic boundary-data design

- round_id: `u2_00_data_design`
- purpose: synchronize the authoritative U2 handoff and create leakage-controlled simulator data for automatic boundary modeling.
- code base before this round: `07422057a1ec2142e015dbf69671e029e77fbbc3`
- Python: `/home/__compress_data/xushijie/.conda/envs/cupid/bin/python`
- runtime: CPU stochastic simulation; no GPU is required for this data-construction round.

Executed command:

```bash
/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m upgrade_v2.cli build-u2-stochastic-boundary-data \
  --historical-handoff artifacts/pathgraph_sarm/upgrade_v2/results/u1_final/u2_handoff.json \
  --handoff artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge/u2_handoff_v2.json \
  --output-dir artifacts/pathgraph_sarm/upgrade_v2/results/u2_boundary_v1 \
  --seed 20260905 \
  --per-stratum 12
```

The original `results/u1_final/u2_handoff.json` remains an unmodified D1–D3-predecessor record. `results/u1_data_bridge/u2_handoff_v2.json` is the authority for simulator-scoped U2. Transition features exclude generation strata and intervention schedules; those fields are available only in the audit manifest.
