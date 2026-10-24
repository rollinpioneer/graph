# I1 + existing collision/PCR controls

See [the Chinese report](report_zh.md) and [fixed protocol](protocol.json). Decision: **HOLD**. No training, parameter search, new force-closure evaluation or C4/C5 access.

O–F reuse 48 frames, 1,229 clean baseline candidates and 14,748 already evaluated angle poses. EG point-cloud filtering reduces I1-induced collisions, but residual collisions and harm remain. PCR's extra local constraints provide limited additional benefit under the fixed unweighted rule.

All I1/PCR angle-selection results use GT information offline. They are not trained-model or deployable-policy results. One newly promoted F Top-1 has no existing physical label and stays unknown.

## Reproduction

- On the research servers, stage this directory at `~/work/grasp/eg_i1_collision_controls_20261024`. `run_controls.py` uses the existing sibling `eg_i1_angle_physics_20261024/evaluation`, `eg_pose_quality_diagnosis_v1_20261022`, `eg_pcr_r0_20261024/sdf` and original EG/GraspNet runtime. These historical inputs are read-only.
- `reproduce.sh SHARD` records the existing environment activation and runs one of three disjoint shards (SHARD 0/1/2). `python run_controls.py pilot` checks one frame; `python run_controls.py SHARD 4` runs the primary controls. No model inference or parameter update occurs.
- `mechanism_check.py SHARD` only attributes the predeclared residual D collisions and feasible false filters to existing geometry/observation differences. It does not select new angles or change thresholds.
- `analyze.py` consumes the new `results/` and the prior angle-evaluation tables; `analyze_mechanism.py` summarizes the bounded geometric audit. Local analysis requires NumPy, pandas, scipy and matplotlib.
- The prior light result directory is available in the parent branch's `eg_physical_diagnostics_20261024/eg_i1_angle_physics_20261024/`. For local re-analysis, copy that directory to a sibling of this round in a separate analysis workspace, preserving the expected names. Full geometry, original raw caches and model dependencies remain external assets identified by the provenance receipts.

`git_archive_manifest.json` defines the uploaded subset; `delivery_manifest.json` describes the complete local round and may list omitted transient logs. Scientific files retain original bytes for SHA-256 verification. No checkpoint or dataset is included.
