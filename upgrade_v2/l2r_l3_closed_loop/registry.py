from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class L3Case:
    case_id: str
    scenario: str
    expected_action: str
    force_loss: bool = False
    contact_dropout: bool = False
    commanded_release: bool = False


CASES = (
    L3Case("L3C1_missed_grasp_retry", "missed_grasp_then_retry", "retry_grasp"),
    L3Case("L3C2_transport_loss_recover", "normal_pick_place", "recover_object", force_loss=True),
    L3Case("L3C3_stable_transport", "normal_pick_place", "none"),
    L3Case("L3C4_single_frame_contact_fault", "normal_pick_place", "none", contact_dropout=True),
    L3Case("L3C5_commanded_release", "normal_pick_place", "none", commanded_release=True),
)

FAMILIES = tuple(
    (f"L2RAR2_R23R1_L3_{index:02d}_{894000 + index}", 894000 + index, 89500000 + index * 100)
    for index in range(4)
)


def registry() -> dict:
    return {
        "schema": "l2rar2_r23_l3_registry_v1",
        "families": [
            {"family_id": family, "family_seed": family_seed, "rollout_seed_base": base}
            for family, family_seed, base in FAMILIES
        ],
        "cases": [case.__dict__ for case in CASES],
        "methods": ["O_C3_CLP3_CANONICAL_TIME", "O_C3_RAW", "RECOVERY_DISABLED"],
        "paired_groups": len(FAMILIES) * len(CASES),
        "physical_rollouts": len(FAMILIES) * len(CASES) * 3,
        "seed_replacement_allowed": False,
        "parameter_search_allowed": False,
    }
