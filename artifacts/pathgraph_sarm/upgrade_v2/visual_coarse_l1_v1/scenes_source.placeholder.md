# Externalized Original Simulator RGB Files

- Original directory: `artifacts/pathgraph_sarm/upgrade_v2/visual_coarse_l1_v1/scenes_source/`
- File count: 48 JPEG images.
- Original filenames, relative paths, camera metadata, and SHA256 values: `scenes.jsonl`.
- Purpose: original MuJoCo RGB render outputs before the L1V normalization pass.
- Git status: intentionally not uploaded because `prepared/images/` contains the normalized review copies.
- Restore: rerun `tools/render_simulator_scenes.py` with the committed capture plan and MuJoCo environment, or restore the exact local directory.
