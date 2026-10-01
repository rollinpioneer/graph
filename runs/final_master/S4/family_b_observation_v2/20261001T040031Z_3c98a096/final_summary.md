# Family B observation v2 technical validation

- Card: CP-DISR-S4-FAMILY-B-OBS-V2-VALIDATION-1; observation profile `FAMILY_B_PUBLIC_OBS_V2` (sha256 3c98a09649a73a3d), frozen before any v2 reset: yes (committed with the freeze).
- old v1 attempts = 4 (permanently preserved); new v2 validation attempts = 2 (cap 4).
- old remaining 20 = NOT_RELEASED; provider = 0; RL = 0; optimizer = 0; S2/S3 = false; formal test = false.
- Technical gate: **FAIL** (OBS_V2_TECHNICAL_GATE_FAIL).

| branch | layout/context | candidate | status | remaining target supported | selected view |
|---|---|---|---|---|---|
| ee7ae7c863a834dc | layout_0/B_PENDING | pad_u | EXCEPTION | False (OnTable=None, PICK mask=None) | None |
| 223b42a03b10bf36 | layout_0/B_PENDING | pad_v | None | None (OnTable=None, PICK mask=None) | None |
| e9b7593e48b4d9e2 | layout_1/C_PENDING | pad_u | EXCEPTION | False (OnTable=None, PICK mask=None) | None |
| 5f5a6ee2ee55dcef | layout_1/C_PENDING | pad_v | None | None (OnTable=None, PICK mask=None) | None |

- Paired restore: FAIL.
- Gate problems: ee7ae7c863a834dc:setup incomplete; ee7ae7c863a834dc:not TASK_SUCCESS with expected sequence:EXCEPTION; ee7ae7c863a834dc:candidate action missing; ee7ae7c863a834dc:hidden-truth flags not clean; ee7ae7c863a834dc:recorder errors or reset/construction count; 223b42a03b10bf36:no terminal result; e9b7593e48b4d9e2:setup incomplete; e9b7593e48b4d9e2:not TASK_SUCCESS with expected sequence:EXCEPTION; e9b7593e48b4d9e2:candidate action missing; e9b7593e48b4d9e2:hidden-truth flags not clean; e9b7593e48b4d9e2:recorder errors or reset/construction count; 5f5a6ee2ee55dcef:no terminal result; paired restore equivalence not established
- U vs V cost is not compared by this card.

## Root cause and stop decision
- Both dispatched workers (pad_u of layout_0 and layout_1) raised `TypeError` in `verify_frozen_cameras` during env construction: the code indexed `profile["cameras"]` (a list of camera names) by name; the per-camera frozen poses are under `profile["camera_manifest"]["cameras"]`. This is an implementation defect in the freeze-time check, hit before any reset, observation, or skill action (0 actions, 0 frames, 0 live skill calls).
- Because the construction counters were charged, 2 of the 4 v2 attempts are consumed with no observation evidence. Only 2 attempts remain, so the 4/4 requirement of the technical gate can no longer be met. The two pad_v branches (223b42a03b10bf36, 5f5a6ee2ee55dcef) were never dispatched.
- Per the card: gate = FAIL, results kept, no fifth attempt, no camera/profile/config change, remaining 20 not released. The post-pad_v target-observability question is UNANSWERED (not resolved, not refuted).
- Note: the recorder's internal reset/construction counters read -1 in the failed workers because the env never finished constructing; the ledger charges (2 constructions, 2 resets, 2 attempts) are the authoritative counts.
- A follow-up would need a new explicit authorization (fresh v2 attempt budget) after fixing the one-line lookup and adding a unit test that constructs `verify_frozen_cameras` against the real frozen profile (the offline tests used a fake env and did not exercise this path).
