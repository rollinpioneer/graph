from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ConfirmationCase:
    case_id: str
    requested_effect: str
    kind: str
    prehold_s: float

CASES = (
    ConfirmationCase("C1_hold_request_ends_without_hold", "HOLD_OBJECT", "missed_hold", 0.0),
    ConfirmationCase("C2_touch_request_completes_without_hold", "TOUCH_OBJECT", "touch", 0.0),
    ConfirmationCase("C3_stable_hold_clean", "HOLD_OBJECT", "stable", 0.50),
    ConfirmationCase("C4_stable_hold_camera_jitter", "HOLD_OBJECT", "jitter", 0.50),
    ConfirmationCase("C5_weld_off_no_force_outcome_control", "HOLD_OBJECT", "weld_off", 0.50),
    ConfirmationCase("C6_weak_intervention_outcome", "HOLD_OBJECT", "weak", 0.50),
    ConfirmationCase("C7_medium_intervention_outcome", "HOLD_OBJECT", "medium", 0.50),
    ConfirmationCase("C8_strong_loss_clean", "HOLD_OBJECT", "strong", 0.50),
    ConfirmationCase("C9_strong_loss_frame_dropout", "HOLD_OBJECT", "strong_dropout", 0.50),
    ConfirmationCase("C10_brief_hold_strong_loss", "HOLD_OBJECT", "strong", 0.10),
    ConfirmationCase("C11_transport_strong_loss", "HOLD_OBJECT", "transport_strong", 0.50),
    ConfirmationCase("C12_commanded_release", "RELEASE_OBJECT", "release", 0.50),
)
CALIBRATION_FAMILIES = tuple((f"L2RAR2_RGB_CAL_{i:02d}_{880000+i}", 880000+i, 88100000+i*100) for i in range(4))
CONFIRMATION_FAMILIES = tuple((f"L2RAR2_RGB_CONF_{i:02d}_{882000+i}", 882000+i, 88300000+i*100) for i in range(6))

def unique_keys():
    return tuple((family, case.case_id, base+i) for family, _, base in CONFIRMATION_FAMILIES for i, case in enumerate(CASES))

def jitter_deg(time_s: float, start_s: float) -> float:
    import math
    return 1.5 * math.sin(2.0 * math.pi * (time_s-start_s) / 0.40)

def dropout_capture_times(force_start_s: float) -> tuple[float, float, float]:
    return tuple(force_start_s + value for value in (0.10, 0.15, 0.20))
