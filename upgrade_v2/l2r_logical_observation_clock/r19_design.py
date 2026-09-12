from __future__ import annotations

from pathlib import Path

from .io_utils import read_json, write_json

FAMILIES = tuple({"family_id": f"L2RAR2_R19_CONF_{index:02d}_{886000 + index}",
                  "family_seed": 886000 + index, "rollout_seed_base": 88700000 + index * 100}
                 for index in range(6))
CASES = (
    "T1_transport_stable_clean", "T2_transport_camera_jitter",
    "T3_transport_single_contact_proxy_dropout",
    "T4_transport_same_time_contact_false_then_true",
    "T5_transport_weak_outcome", "T6_transport_medium_outcome",
    "T7_transport_strong_phase_00ms", "T8_transport_strong_phase_10ms",
    "T9_transport_strong_phase_20ms", "T10_transport_strong_phase_30ms",
    "T11_transport_strong_phase_40ms_rgb_dropout", "T12_transport_commanded_release",
)


def build_design(development_root: Path, output_root: Path) -> dict:
    gate = read_json(development_root / "development_gate.json")
    if not gate.get("r19_confirmation_design_allowed"):
        raise RuntimeError("R19_DEVELOPMENT_GATE_REQUIRED")
    output_root.mkdir(parents=True, exist_ok=False)
    registry = {"schema": "l2rar2_r19_confirmation_design_v1", "status": "DESIGN_ONLY",
                "physical_execution_started": False, "families": list(FAMILIES),
                "case_order": list(CASES), "rollouts": len(FAMILIES) * len(CASES),
                "new_family_seed_range": [886000, 886005],
                "new_rollout_seed_bases": [family["rollout_seed_base"] for family in FAMILIES],
                "reuses_R17_or_R18_family_seed": False,
                "candidate_frozen_before_confirmation": "O_C3_CLP2_LOGICAL_CLOCK",
                "same_time_clock_stress_case": CASES[3],
                "parameter_search_on_confirmation_allowed": False}
    write_json(output_root / "confirmation_registry.json", registry)
    return registry

