from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FactorArm:
    arm_id: str
    fusion_enabled: bool
    supervisor_v2_enabled: bool


@dataclass(frozen=True)
class Case:
    case_id: str
    fault: str
    recovery_required: bool = True
    commanded_release: bool = False


ARMS = (
    FactorArm("F0_S0_CLP3_BASELINE", False, False),
    FactorArm("F1_S0_FUSION_ONLY", True, False),
    FactorArm("F0_S1_SUPERVISOR_ONLY", False, True),
    FactorArm("F1_S1_FUSION_SUPERVISOR", True, True),
)

CASES = (
    Case("R25C1_contact_absence_edge_missing", "contact_absence"),
    Case("R25C2_rgb_detach_contact_masked", "rgb_detach"),
    Case("R25C3_transient_relocation_error", "transient_relocation"),
    Case("R25C4_goal_verification_retry", "goal_error"),
    Case("R25C5_secondary_loss_rearm", "secondary_loss"),
    Case("R25C6_partial_slip_control", "partial_slip", recovery_required=False),
    Case("R25C7_commanded_release_control", "release", recovery_required=False, commanded_release=True),
)

FAMILIES = tuple(
    (f"L2RAR2_R25_FACTORIAL_{index:02d}_{898000 + index}", 898000 + index, 89900000 + index * 100)
    for index in range(4)
)


def registry() -> dict:
    return {
        "schema": "l2rar2_r25_factorial_registry_v1",
        "families": [{"family_id": family, "family_seed": seed, "rollout_seed_base": base}
                     for family, seed, base in FAMILIES],
        "arms": [asdict(arm) for arm in ARMS],
        "cases": [asdict(case) for case in CASES],
        "paired_groups": len(FAMILIES) * len(CASES),
        "physical_rollouts": len(FAMILIES) * len(CASES) * len(ARMS),
        "new_families": True,
        "seed_replacement": False,
        "parameter_search": False,
    }
