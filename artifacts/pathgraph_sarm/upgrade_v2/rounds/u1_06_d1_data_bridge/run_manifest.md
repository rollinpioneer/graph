# Run Manifest — U1 data bridge D1

- round_id: `u1_06_d1_data_bridge`
- purpose: verify complete anchor capture, restoration, and continuation in a state-changing stochastic simulator.
- code base before D1: `6743d23167c27d94c722238d4cc18c47666f321a`
- Python: `/home/__compress_data/xushijie/.conda/envs/cupid/bin/python`
- D1 simulator run: CPU; it does not require GPU acceleration.
- privileged GPU runtime check: passed on CUDA device `4` (A100 40 GB); see `results/u1_data_bridge/d1_gpu_runtime_verification.json`.

Executed commands:

```bash
/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m upgrade_v2.cli verify-gpu-runtime \
  --output artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge/d1_gpu_runtime_verification.json \
  --device-index 4

/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m upgrade_v2.cli run-d1-pusht-restore \
  --sim-root repo \
  --output-dir artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge \
  --tolerance 1e-10

/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m upgrade_v2.cli run-d1-stochastic-restore \
  --output-dir artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge \
  --seed 20260905 \
  --tolerance 1e-12

/home/__compress_data/xushijie/.conda/envs/cupid/bin/python -m unittest tests/test_upgrade_v2_corrections.py
```

The Pymunk result is a non-qualifying diagnostic: its public state surface does not expose the contact-solver cache needed for collision-exact replay. D1 therefore uses the explicit-state stochastic simulator, whose snapshot includes the RNG state. D2 and D3 remain mandatory; U2 is not eligible.
