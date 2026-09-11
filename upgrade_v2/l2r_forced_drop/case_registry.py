from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForcedDropCase:
    case_id: str
    intervention: str
    requested_effect: str
    reference_action: str
    observation_s: float | None = None


CASES = (
    ForcedDropCase("F1_hold_request_ends_without_hold", "none", "HOLD_OBJECT", "retry_grasp"),
    ForcedDropCase("F2_touch_request_completes_without_hold", "none", "TOUCH_OBJECT", "none"),
    ForcedDropCase("F3_stable_hold_no_intervention", "weld_remains_on", "HOLD_OBJECT", "none"),
    ForcedDropCase("F4_weld_off_no_force_outcome_control", "weld_off_no_force", "HOLD_OBJECT", "outcome_conditioned", 1.5),
    ForcedDropCase("F5_forced_drop_regular", "weld_off_plus_selected_force", "HOLD_OBJECT", "recover_if_physical_loss_confirmed", 0.75),
    ForcedDropCase("F6_forced_drop_brief_hold", "weld_off_plus_selected_force_after_minimum_prehold", "HOLD_OBJECT", "recover_if_physical_loss_confirmed", 0.75),
    ForcedDropCase("F7_commanded_release", "commanded_open_no_force", "RELEASE_OBJECT", "none"),
    ForcedDropCase("F8_forced_drop_during_transport_long_observe", "weld_off_plus_selected_force_during_transport", "HOLD_OBJECT", "recover_if_physical_loss_confirmed", 1.5),
)


def case_by_id(case_id: str) -> ForcedDropCase:
    for case in CASES:
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def development_case_ids() -> tuple[str, ...]:
    return tuple(case.case_id for case in CASES)
