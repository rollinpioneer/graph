# V6RC1 return controller engineering

Status: RETURN_CONTROLLER_ENGINEERING_NOT_READY
confirmation_passed: false
Selected: RC1_OBJECT_COMPENSATED_DIRECT
V6 reward changed: no

Forensics: {"n": 30, "taxonomy": {"F4_EEF_REACHED_OBJECT_NOT_RETURNED": 17, "F1_REGRASP_FAILED": 4, "F3_EEF_TARGET_NOT_REACHED": 9}, "eef_reached_object_not_returned": 17, "median_eef_error": 0.0, "median_object_error": 0.03429730647320949}
Holdout: {"confirmation_passed": false, "status": "RETURN_CONTROLLER_ENGINEERING_NOT_READY", "metrics": {"controller_id": "RC1_OBJECT_COMPENSATED_DIRECT", "hard_ok": true, "metrics": {"E1_LOSS_RETURN_P40": [3, 4], "E2_LOSS_RETURN_P80": [2, 4], "E3_THREE_RETURNS_P40": [2, 4], "E4_THREE_RETURNS_P80": [2, 4], "E5_REGRASP_OFFSET_LEFT_P40": [3, 4], "E6_REGRASP_OFFSET_RIGHT_P40": [3, 4], "E7_NO_LOSS_OUT_AND_BACK": [4, 4], "E8_COMMANDED_RELEASE_CONTROL": [4, 4]}, "median_return_s": 0.6400000000000002, "p90_return_s": 1.2200000000000006, "n": 32, "failures": 9}, "gate_ok": false}

Limitations: TRANSLATION_COMPENSATED_FIXED_ORIENTATION; constraint-assisted MuJoCo; not a reward confirmation.
