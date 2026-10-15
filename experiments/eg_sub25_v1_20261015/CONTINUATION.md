# EG-SUB25-V1 completed handoff

The current user-authorized experiment is complete. Do not restart training or evaluation. Read final_reproduction_report.md and bad_case_report.md first. One seed20261015,6400training frames,10epochs,16000updates. Development AP51.40379252%;fixed Seen subset54.70872391/official64.10353196%;fixed Novel subset40.12994793/official47.08921875%. All18native scene evaluations completed all256frames;paired inputs match and24case replays match. Twelve JPEG figures are archived and five representative images covering four scenes were visually reviewed.

Important:Novel188contains110unique observations with109..255repeated;Novel189contains171with170..255repeated. Current files freshly match official ZIP CRC indexes. Keep native256frame AP;do not deduplicate or change fixed scenes to improve metrics. The three selected188frames are the same view with different point sampling. These are exploratory cases,not full Novel AP or causal module evidence.

No next method,seed,75scene expansion or retraining is authorized by this completion. A future research decision should first control point sampling and candidate lists when diagnosing ranking/collision/pose effects. Training sufficiency remains a confound.

Server root:/home/xushijie3/work/eg_sub25_v1_20261015
Weights:runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar
Evaluation:runs/EG_SUB25_V1/evaluation
Raw observations/intermediates/full PNG:runs/EG_SUB25_V1/bad_cases
Archive worktree:/home/xushijie3/work/eg_full_repro_v1_20261015/archive_worktree
Branch:https://github.com/rollinpioneer/graph/tree/codex/eg-sub25-v1-20261015

Training hashes:TRAINING_COMPLETE.json;current raw evaluation identities:EVALUATION_ARTIFACT_INDEX.json and compressed CSV. Do not delete the old eg_full_repro root:shared data,supervision,environment and official pretrained weights are still used. All owned experiment jobs ended. FINAL_VALIDATION.json is a pre-SSH-exit snapshot;actual client exit is recorded locally after disconnect. No scheduled monitoring is enabled.
