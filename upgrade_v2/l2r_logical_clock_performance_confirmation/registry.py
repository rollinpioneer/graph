from __future__ import annotations

from upgrade_v2.l2r_logical_clock_confirmation.registry import CASES as R20_CASES, ConfirmationCase

CASE_INDEXES = (0, 1, 2, 3, 6, 8, 10, 11)
CASES: tuple[ConfirmationCase, ...] = tuple(R20_CASES[index] for index in CASE_INDEXES)
FAMILIES = tuple((f"L2RAR2_R21_PERF_{index:02d}_{888000 + index}", 888000 + index,
                  88900000 + index * 100) for index in range(4))


def registry() -> dict:
    return {"schema": "l2rar2_r21_confirmation_registry_v1", "families": [
        {"family_id": family, "family_seed": family_seed, "rollout_seed_base": seed_base}
        for family, family_seed, seed_base in FAMILIES],
        "case_order": [case.case_id for case in CASES], "rollouts": 32,
        "seed_replacement_allowed": False, "automatic_rerun_allowed": False}
