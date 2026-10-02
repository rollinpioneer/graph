# Sub-card request: T_B independent evaluation (CP-DISR-TB-INDEP-EVAL-01)

Status: **REQUEST - NOT EXECUTED**. This request reads no test result, runs no evaluation, loads no model and generates no scene. Everything below is taken from files and from Plan v3 section 12.

## 1. Purpose and limits

One unified independent evaluation of the frozen final T_B models on the T_B test30 scenes, with the revised evaluation record of `04_evaluation_record_revision_design.md`. It does not tune anything, does not select a model on test, does not train, and does not release or read test material for any task other than T_B.

## 2. Model list (frozen in `03_frozen_model_list.md`)

| slot | run | method | seed | list | final checkpoint | N | sha256 | open gaps |
|---|---|---|---|---|---|---|---|---|
| TB-M1 | R2-B0-s0 | B0 | 0 | BASE_6x30 | final_n_014705_u14.pt | 14705 | f50a857ce9b589e65b9b7b4d9e983b7e3babb1b9e93d54875f3e383e99b837f0 | UC1, UC2, UC4, UC5, UC3, UC7 |
| TB-M2 | R1-B1K-s0 | B1-K | 0 | BASE_6x30 | final_n_013623_u13.pt | 13623 | 2c93d96f3999b5192f5ceb7c817e9a00a94937dfa1a22db9e76ac02cf7d0c99a | UC1, UC2, UC4, UC5 |
| TB-M3 | R-TB-K-1 | B1-K | 1 | BASE_6x30 | final_n_014531_u14.pt | 14531 | fa6ca4cf20689aba50c81b52ae2b202772e97cf0a886d74ae58cc22fcb91f385 | none |
| TB-M4 | R1-B2-s0 | B2 | 0 | BASE_6x30 | final_n_014464_u14.pt | 14464 | 976f70102ee993a639edd4572eebe2053ad8ac6d7b450bd43dc3576a367b413e | UC1, UC2, UC4, UC5 |
| TB-M5 | R-TB-DK-1 | B2 | 1 | BASE_6x30 | final_n_014170_u13.pt | 14170 | b1e331c98d4937b919242912cb3011bb079ec6971a3f8b7b51c682546b7476ca | none |
| TB-M6 | R-TB-E-0 | B1-K+E | 0 | BASE_6x30 | final_n_014512_u14.pt | 14512 | f9db3e4ff6dc877dbd4b1b433ac31789e43f6fd45404c35e5103674e4ed4aaba | none |
| TB-C1 | R-TB-E-1 | B1-K+E | 1 | CONDITIONAL_OUTSIDE_628 | final_n_014309_u13.pt | 14309 | 62099c437dabd361412ae2fe3b5b6fadcf785bc88661e32ac7d35564f79522e7 | none |

Main rule: last valid post-update checkpoint of each run. Selected / best-dev columns are separate; no failing method is evaluated at N = 0 in place of its final.

## 3. Data source

- Scenes: `/home/xushijie2/graph_cp_disr_v2_1/experiments/stage_2a_inputs/T_B/test` (T_B_test_00 .. T_B_test_29; each holds CAPTURE_COMPLETE, depth.npy, reset_config.json, rgb.png); scene-set sha256 `db01aa572d9f852935bc52da8ce8bbaf8b35bb6c82c126a703b6ca4a6a183d9e`.
- Split definition: `configs/splits/T_B_stage_2a_test30.json` (sha256 8eed9da517c6d37664c90f70c9cadcd78b97b8e96a568ae9d321161f5694e0bf), `T_B_stage_2a_test_ids.json` (sha256 03264a3d7511eb7e9d6f7fee6398f90acc280297d9f6154920e58cc251dd3beb), `T_B_stage_2a_v11.json` (sha256 b14f0098b6331f527a3e7cd69c834d1da7679c702a4830f4bf0b45d37776abe9). 30 active cases in a fixed order with fixed reset seeds shared by the methods; 20 reserved pool entries (T_B_test_30..49) stay unused.
- Prior / relation material: these methods run with empty R. The no-prior run context carries no cache pointers and `require_case_cache` returns without reading; the 30 `vlm_cache/test` entries therefore are neither needed nor to be opened.
- Provenance evidence (file names, hashes, config declarations only): `06_test_source_provenance_check.json`. In the accessible worktrees no T_B test result file exists (tasks with test-result file names: D0), the final-master T_B train/dev splits carry an empty `test` list, and no small run file of the seven models mentions a test scene. The only test-result files found belong to D0 (cited by identity per Plan v3 12.2, not opened).

## 4. Episode budget

| item | episodes |
|---|---|
| base: 6 models, 30 test cases each | 180 (inside the Plan v3 cap of 628) |
| conditional: B1-K+E seed 1 (R-TB-E-1), 30 cases | 30 (outside the 628; needs its own approval) |
| maximum if the conditional model is approved | 210 |

One pass per model on the fixed scene order, deterministic argmax, evaluation RNG isolated, no repeat to satisfaction. T_B episode deadline 60 s (simulated).

## 5. Test prerequisites (Plan v3 12.1) and their present state

| id | prerequisite | state |
|---|---|---|
| P1 | S4 freeze of method, inputs, optimisation and endpoints, and S5 completion of the prescribed seeds, for T_B (Plan v3 12.1 item 1) | NOT CONFIRMED by this record - needs an explicit statement |
| P2 | Unified main-evaluation model list frozen (final generation per run) with hashes | DONE as a candidate list in `03_frozen_model_list.md`; historical load check open (P8) |
| P3 | Evaluation-record revision implemented, offline-tested and committed before any test run | NOT DONE - design only (`04_...`) |
| P4 | `test_release_manifest`: model and Adam index, source hashes, input hashes, scene order, evaluator and RNG settings, invalid-episode rule | NOT BUILT - built at release from the frozen list |
| P5 | Source qualification of the T_B test30 set (a T_B test used in development cannot be called blind test) | UNRESOLVED - evidence in `06_...`; decision U1 |
| P6 | A runner that takes the frozen list: the existing `evaluate_final_tests` (`stage2a_v11.py`, line 1589) reads an 8-checkpoint `selection_manifest` of the old stage and raises unless exactly 8 are selected | NOT DONE - cannot be reused as is |
| P7 | Derived no-prior test split for the 30 active ids (records without cache pointers, as for the dev10 derivation), built without opening any cache or relation truth | NOT DONE |
| P8 | Load-only compatibility check of the three historical final weights under the current code (no episode), to decide UC1 | NOT RUN - approval requested (D4) |
| P9 | Compute, disk and GPU plan for 180 (+30) episodes and the storage guard | NOT ASSESSED |
| P10 | Approval of the conditional +E seed 1 model (30 episodes outside the 628) | PENDING (D3) |

## 6. Unresolved items

| id | item | decision by |
|---|---|---|
| U1 | T_B test30 qualification: scenes and 30 prior-cache entries were materialized earlier by the old stage_2a flow, before the S6 release step; no T_B test result or development use was found in the accessible worktrees, but use outside them cannot be shown from files. Accept with the old materialization documented, or draw from the reserved pool under a new freeze. | user |
| U2 | Historical models (UC1-UC7): loadability under the current code, evaluator-source differences, R2 B0 original location, commit pointers, the R2 B0 dev20 file holding 10 cases. Carried as gaps; no sidecar, no retraining. | user (after P8) |
| U3 | Invalid-episode rule (infrastructure failure versus policy outcome) must be fixed in the release manifest before the first episode; no repeat to satisfaction. | user / sub-card |
| U4 | Whether the conditional B1-K+E seed 1 enters the same release or a later one. | user |
| U5 | S4 / S5 status for T_B (P1). | user |
| U6 | Whether a dev10 consistency replay of the frozen finals with the revised recorder is wanted before test (it runs episodes, separate card). | user |

## 7. Decisions requested

- D1 Confirm the model list of `03_frozen_model_list.md` (6 base + 1 conditional) and the rule that the final generation is the main-evaluation model.
- D2 Confirm the episode budget: 180 base, maximum 210 with the conditional model.
- D3 Decide whether the conditional B1-K+E seed 1 model is released with the base set.
- D4 Approve the load-only compatibility check (no episode) for the three historical final weights, or state that they stay as recorded gaps.
- D5 Approve a separate implementation card for the evaluation-record revision (source change plus offline tests, no episodes).
- D6 Decide the source-qualification path for T_B test30 (U1) and state the S4 / S5 status (U5).

## 8. What this request did not do

- No test score, test cache, relation truth or test result file was read or opened. The test scene capture files (rgb.png, depth.npy, reset_config.json, CAPTURE_COMPLETE) were byte-hashed only, to fix scene identity; they were not parsed or rendered.
- No evaluation or episode was run; no model was loaded; no scene was generated or regenerated.
- No RL, environment, provider, optimizer or elastic slot was added or used; ELASTIC-02..04 are not allocated; the formal test was not started; Family B and RoboCasa were not resumed.
- No method, training configuration, profile, reward, action or termination setting was changed; no source file was edited.
