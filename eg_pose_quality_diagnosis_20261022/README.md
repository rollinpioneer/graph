# EG pose–quality mechanism diagnosis — HOLD

Complete bounded zero-training study, 2026-10-22. Official full EG Kinect checkpoint; 48 C3 frames and 16 pre-fixed training frames. No model training, no EG/G source modification, no C4/C5 access.

Open `EG_pose_quality_final_diagnosis.html` for the self-contained Chinese report. `decision.md` is the one-page decision; statistics include all denominators and scene-bootstrap intervals. The report embeds four figures and requires no network to render.

`EG_pose_quality_evidence.csv.gz` is the complete lightweight evidence table (19,164 rows). Decompress with standard gzip tools to `EG_pose_quality_evidence.csv`; the uncompressed CSV is delivered locally. It includes every training raw candidate and the evaluated C3 final-pool union. Unassessed raw C3 candidates are not labelled failures. Model weights, point clouds and full caches are deliberately excluded from this public archive.

The `code` folder contains the independent diagnostic scripts. Remote scripts expect the existing study assets beneath `~/work/grasp`; no credentials are embedded. Local `analyze.py` and `report.py` accept `EG_POSE_OUTPUT_DIR` to the delivered output folder with `data/cache` and `data/evaluation`. `historical_EG_B_48frames.json` supplies the historical frame comparison. The published copies only replace the local absolute output path and baseline-file location; executed-source hashes remain in the asset manifest. Raw-label generation logic, model inference and training logic were not changed.

Reproduction order with authorized assets: inference (`infer.py`), raw-label/evaluator preflight, direct A/B evaluation (`evaluate.py`), physical-frame supplementary controls (`aligned_controls.py`), analysis, report. Do not silently skip the FP32 read-only label-probe correction described in the report. Training is not part of this workflow.

Physical P1/P4 controls were added after recognizing the coordinate-basis ambiguity, on the same 8,732 valid pairs only. Original class-index P1–P5 results are retained. GT counterfactual gains are not deployable-method gains. HOLD concerns independent mechanism/novelty, not an uncompleted diagnostic.

Primary finding: recoverable pose–quality mismatches occur across scenes, but ordinary physical pose errors and train/inference reference frames remain strong alternative explanations; quality scores correlate more with actual than reference quality. The only next question is the incremental value of final-pose quality alignment under a coordinate-consistent, same-budget ordinary-pose control. No further broad diagnostic or training was launched.
