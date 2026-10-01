# Family B observation v2-r2 technical validation

- Card: CP-DISR-S4-FAMILY-B-OBS-V2-R2-VALIDATION-1; frozen profile `FAMILY_B_PUBLIC_OBS_V2` sha256 3c98a09649a73a3d9cee1db7e7a9f67063e515bfe5c3e689cabf4cf09900d917 (unchanged, cameras agentview+sideview).
- v1 consumed attempts: 4 (permanent); v2-r1 consumed attempts: 2 (+2 unused slots RETIRED_NOT_TRANSFERABLE).
- v2-r2 attempts used: 4 (cap 4); cumulative attempts used: 10.
- Successful constructors (actual): 4; constructor attempts: 4; explicit reset calls (actual): 4; internal constructor resets: 8; live skill calls (actual): 24.
- original remaining 20: NOT_RELEASED; provider/RL/optimizer = 0; S2/S3/formal test/TP training = false.
- Canary: **CANARY_PASS** (layout_0/B_PENDING/repeat0/pad_v); remainder phase released by the canary PASS (3 remainder attempts used; the ledger field `remainder_slots_status` still reads PENDING_CANARY because the coordinator only rewrites it on canary failure).
- Full R2 technical gate: **PASS** (OBS_V2_R2_TECHNICAL_GATE_PASS_READY_FOR_REMAINDER_REVIEW).

| branch | layout/context | candidate | task end | remaining target (per-view mask/depth support) | selected view | OnTable | PICK mask |
|---|---|---|---|---|---|---|---|
| 74a1c3df3d169e6b | layout_0/B_PENDING | pad_u | TASK_SUCCESS | agentview: 36/36; sideview: 8/8 | agentview | TRUE | True |
| cf26947cfdf5d8f7 | layout_0/B_PENDING | pad_v | TASK_SUCCESS | agentview: 0/0; sideview: 8/8 | sideview | TRUE | True |
| af6911ea7e4b4f8e | layout_1/C_PENDING | pad_u | TASK_SUCCESS | agentview: 36/36; sideview: 8/8 | agentview | TRUE | True |
| 7ca418154890cccf | layout_1/C_PENDING | pad_v | TASK_SUCCESS | agentview: 0/0; sideview: 8/8 | sideview | TRUE | True |

- Post-pad_v public observability of the remaining target: layout_0=True, layout_1=True
- sideview actually selected (fusion selections across all frames/objects): 20
- Paired restore: PASS; camera live-vs-frozen all pass: True.
- Gate problems: none
- U vs V cost is not compared by this card; speedup NOT_MEASURED.
- Next step requires explicit user authorization (remainder review / any release of the original 20).

## Reading notes (added after the run; no code, config or result was changed)
- The two pad_v routes lose the remaining target in agentview (0 px, hand occlusion, unchanged from v1) but sideview keeps it visible: 8 mask px / 8 depth-support px, ratio 1.0, selected_view = sideview, OnTable(TRUE), Held(FALSE), PICK mask true, then PICK/PLACE and TASK_SUCCESS. The target is publicly decidable after pad_v through the second fixed view.
- Margin caveat: the sideview support (8 px) equals both its own initial reference support (8 px) and the frozen min_pixels (8), so the sideview evidence sits exactly at threshold; it has no pixel margin.
- Pairs: public facts, candidate IDs/mask, qpos/qvel (max diff 0.0), selected views and fused xyz (diff 0.0) are identical for both U/V pairs. Pixels are EXACT for layout_0 agentview and NON_BITWISE_WITHIN_CONTRACT elsewhere (1-5 pixels, max 1 grey level, depth identical); in-process self-repeat renders also differ by that amount, i.e. renderer nondeterminism within the frozen contract. Initial agentview images differ between U and V members in both layouts (informational only, as in v1).
- Camera live-vs-frozen: all four branches PASS; max |pos diff| 0.0, max |quat diff| 2.2e-16 (sideview), fovy diff 0.0, tolerance 1e-9, 128x128, fixed in world.
- Wall time: canary 20.1 s, remainder phase 34.2 s on GPUs 3 and 4 (2 workers); speedup NOT_MEASURED.
- This is a technical validation only: no U-vs-V cost comparison, no release of the original 20, no provider/representation/S2/S3/TP/formal test.
