# Z0 reproduction

This is the CPU-only Z0 stage. It does not authorize or execute the T stage.

Place this directory next to the preserved result directories:

```text
parent/
  eg_z0_prior_locality_20261024/
  eg_i1_angle_physics_20261024/evaluation/   # 48 NPZ + JSON receipts
  eg_i1_collision_controls_20261024/results/ # 48 NPZ + JSON receipts
```

Use Python with NumPy, SciPy, scikit-learn and matplotlib:

```text
python run_z0.py
python report.py
```

Run from a fresh copy if preserving the delivered files byte-for-byte. The input directories are read-only. Each source NPZ is checked against its historical SHA receipt. The original caches contain geometry which is unused by Z0; no additional dataset, checkpoint, GPU or server is required.

`selected_angles.npz` retains all deterministic choices and all 1,000 random choices, indexed by scene/frame/candidate_id. `random_draw_counts.csv` contains fixed-panel Monte Carlo summaries. These ranges are distinct from the paired scene-bootstrap intervals in `paired_intervals.csv`. The main comparison is fixed at k=3, no collision filtering; secondary comparisons must not replace it.

The original preregistration's LOCALITY and NO-PRIOR wording overlaps when P3 is slightly worse. The report explicitly discloses this; failure of the PRIOR criterion is unambiguous.
