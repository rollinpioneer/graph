# Run Manifest — U1 data bridge D2/D3

- round_id: `u1_07_d2_d3_data_bridge`
- purpose: form goal-distance-matched free/collision states, execute independent continuations, and compare five fixed model forms.
- code base before D2/D3: `e6bf023327916a12377080dfa0718a4d63d5e0dd`
- Python: `/home/__compress_data/xushijie/.conda/envs/cupid/bin/python`
- runtime: CPU stochastic simulation; no GPU was required. The privileged GPU verification from D1 remains recorded separately.

Executed commands:

```bash
/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m upgrade_v2.cli build-d2-stochastic-pairs \
  --output-dir artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge \
  --seed 20260905 \
  --pairs 30

/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m upgrade_v2.cli run-d3-stochastic-comparison \
  --pairs artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge/d2_pair_states.jsonl \
  --output-dir artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge \
  --seed 20260905 \
  --replicates 12

/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m unittest tests/test_upgrade_v2_corrections.py
```

The D-only target treats failed continuations as right-censored at the fixed horizon, rather than interpreting an early collision as a fast success. All repetition seeds for a pair remain inside its root-family split. The result permits only a stochastic-simulator-scoped U2 boundary prototype; it does not support a physical or original-task generalization claim.
