# Family B v2 physical qualification (staging continuation)

- Card CP-DISR-S4-FAMILY-B-STAGING-V2-CONTINUATION-1; frozen profile `FAMILY_B_PUBLIC_OBS_V2` 3c98a09649a73a3d9cee1db7e7a9f67063e515bfe5c3e689cabf4cf09900d917 (unchanged).
- Gate M: **GATE_M_COMPLETE_REPLAY_REPRODUCES_R2** (offline, zero new samples). Tested pad_v remaining-target sideview support: layout_0/obj_b=8px (margin 0); layout_1/obj_c=8px (margin 0).
- Complementary gate: **COMPLEMENTARY_CONTEXT_TECHNICAL_PASS**.
- Continuation attempts used 20/20; constructions 20; constructor-successful 20; explicit resets 20; internal resets 40; live skill calls 120/120.
- Cumulative Family B physical attempts: 30 (engineering count; not the Experimental Plan v3 '17+4').
- Provider / representation / RL / optimizer / elastic / formal test = 0; S2/S3/TP training = false.
- Physical mechanism status: **NOT_ESTABLISHED**; next action: `S4_RESEARCH_DECISION`.

| layout | context | repeat | C(pad_u) | C(pad_v) | delta u-v | comparable |
|---|---|---|---|---|---|---|
| layout_0 | B_PENDING | 0 | 13.44999999999503 | 11.099999999996333 | 2.3499999999986976 | True |
| layout_0 | B_PENDING | 1 | 13.44999999999503 | 11.099999999996333 | 2.3499999999986976 | True |
| layout_0 | C_PENDING | 0 | 14.449999999995232 | 9.949999999997726 | 4.499999999997506 | True |
| layout_0 | C_PENDING | 1 | 14.449999999995232 | 9.949999999997726 | 4.499999999997506 | True |
| layout_0 | BOTH_PENDING | 0 | 20.899999999996854 | 18.749999999998046 | 2.1499999999988084 | True |
| layout_0 | BOTH_PENDING | 1 | 20.899999999996854 | 18.749999999998046 | 2.1499999999988084 | True |
| layout_1 | B_PENDING | 0 | 14.14999999999522 | 9.749999999997659 | 4.399999999997561 | True |
| layout_1 | B_PENDING | 1 | 14.14999999999522 | 9.749999999997659 | 4.399999999997561 | True |
| layout_1 | C_PENDING | 0 | 13.799999999995059 | 11.299999999996444 | 2.4999999999986144 | True |
| layout_1 | C_PENDING | 1 | 13.799999999995059 | 11.299999999996444 | 2.4999999999986144 | True |
| layout_1 | BOTH_PENDING | 0 | 21.8999999999963 | 17.699999999998628 | 4.199999999997672 | True |
| layout_1 | BOTH_PENDING | 1 | 21.8999999999963 | 17.699999999998628 | 4.199999999997672 | True |

Conditions: 10_plan_length_skill_multiset_and_tail_match=True, 11_no_missing_precondition_no_plan_timeout_controller_or_unknown=True, 12_both_pending_complete_and_reported=True, 13_both_not_required_neutral=True, 14_no_significance_or_generalization_claim=True, 1_24_of_24_task_success=True, 2_24_of_24_complete_records=True, 3_all_pair_restores_in_contract=True, 4_B_and_C_two_repeats_per_layout=True, 5_sign_reversal_within_layout=False, 6_each_abs_delta_above_threshold=True, 7_repeats_agree_in_direction=True, 8_same_candidate_not_always_winner=False, 9_both_candidates_contract_legal=True
- 24-branch task success: 24/24; complete records: 24/24.
- No significance or generalization claim; speedup NOT_MEASURED.
