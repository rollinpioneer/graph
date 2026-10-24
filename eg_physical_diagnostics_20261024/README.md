# EG physical reliability diagnostics — 2026-10-24

Four completed rounds, with implementation, configurations, lightweight candidate-level results, figures and reports. No new training was performed in the R0/decode/I1 rounds; no held-out C4/C5 data were accessed.

| Round | Report | Outcome |
|---|---|---|
| EG bad cases | [Summary](eg_bad_cases_20261023/EG_bad_case_summary.md) | Failure-category inventory |
| PCR R0 | [Report](eg_pcr_r0_20261024/R0_report_zh.md) | Useful physical information; HOLD on unchanged transplantation |
| Differentiable decode | [Report](eg_pcr_decode_zero_20261024/report_zh.md) | Original predictions preserved; angle signal supported, joint depth route HOLD |
| I1 angle physics | [Report](eg_i1_angle_physics_20261024/report_zh.md) | 14,748 actual-pose evaluations; friction improvements accompanied by collisions; HOLD, T0/T1/T2 not started |

I1 Oracle and GT-assisted logit interventions are **not trained or deployable model improvements**. Candidate IDs and original scores were fixed. See reports for scene-level uncertainty, harm, missing-contact coverage and experimental limits.

`git_archive_manifest.json` lists the uploaded subset and omitted assets. Original per-round `delivery_manifest.json` files describe the full local working directories and may reference excluded files. Model weights, download bundles and session logs are omitted; small `.npz` evaluation tables are included. Full geometry and historical caches remain in the existing research asset locations documented in the code and provenance files.

The decode round's `initial_group_axis_check/` is a preserved, **invalid initial attempt**, excluded from the final analysis; its correction is documented in `implementation_note.json`.

Reports retain their historical “not submitted to Git” statements describing their original delivery state. This archive is the subsequent upload; original reports are not rewritten.

Third-party source snapshots retain upstream comments and are for reproduction. Upstream identities and exact revisions are recorded in the R0 execution configuration and source metadata; no claim of authorship is made for upstream implementations.

Future rounds must be committed on a new branch and pushed to `rollinpioneer/graph`, with the remote commit verified, as recorded in the repository's `AGENTS.md`.
