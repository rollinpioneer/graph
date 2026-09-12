from __future__ import annotations

from . import BASE_COMMIT, CANDIDATE_ID, FORMAL_MAIN, PROTOCOL_ID

FROZEN_BLOBS = {
    "guard": "dea50fa274c9062b43e03f5b445f1cbe43c9c552",
    "detector": "efe789eb3fe7f3b5d851ccd317aa40ca20bdead5",
    "renderer": "a026c3723b469b1eaf1ce8fc34a9c210e625de45",
    "temporal_scoring": "e9182eff3e277017ebee8ad05a5bf3e9120871df",
    "r17_difficulty_ladder": "ff688f4166b6a7f4053e27633c2b8e1252edcd5d",
    "r19_protocol": "ed2a2c954bc20ef65397162f7d11d5a7e887fd89",
    "r19_registry": "e2d3772c5a017cfd4fb8b34792aef46401877a0e",
}
LEVELS = {
    "weak": {"scale": 0.50, "delta_v_local_mps": (20.0, 20.0, -4.0)},
    "medium": {"scale": 0.625, "delta_v_local_mps": (25.0, 25.0, -5.0)},
    "strong": {"scale": 0.75, "delta_v_local_mps": (30.0, 30.0, -6.0)},
}


def protocol_lock() -> dict:
    return {"schema": "l2rar2_r20_protocol_lock_v1", "protocol_id": PROTOCOL_ID,
            "base_branch": "research/l2rar2-r19-logical-observation-clock-development-v1",
            "base_commit": BASE_COMMIT, "formal_main_commit": FORMAL_MAIN,
            "new_branch": "research/l2rar2-r20-logical-clock-physical-confirmation-v1",
            "parent_candidate": CANDIDATE_ID, "frozen_blobs": FROZEN_BLOBS,
            "candidate_contract": {"required_consecutive_valid_observations": 2,
                                   "minimum_physical_gap_s": 0.0,
                                   "maximum_physical_gap_s": 0.10,
                                   "capture_order_strictly_increasing": True,
                                   "physical_time_nondecreasing": True,
                                   "same_time_confirmation_allowed": True,
                                   "backdating_allowed": False,
                                   "parameter_search_allowed": False},
            "confirmation": {"families": 6, "cases": 12, "rollouts": 72, "workers_max": 2},
            "status": "DRAFT_ZERO_EXECUTION", "confirmation_run": False,
            "selected_candidate_id": None, "l3_entry_allowed": False}
