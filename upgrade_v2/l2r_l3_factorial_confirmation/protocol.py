from __future__ import annotations

from upgrade_v2.l2r_loss_observability_fusion.fusion import (
    CONTACT_MAX_GAP_NS,
    CONTACT_REQUIRED_OBSERVATIONS,
    RGB_MAX_GAP_NS,
    RGB_RELATIVE_DISPLACEMENT_PX,
    RGB_REQUIRED_OBSERVATIONS,
)

from . import BASE_COMMIT, CLP3_ID


def protocol_lock() -> dict:
    return {
        "schema": "l2rar2_r25_factorial_protocol_v1",
        "base_commit": BASE_COMMIT,
        "r22_r23_r24_read_only": True,
        "clp3_candidate_id": CLP3_ID,
        "clp3_modified": False,
        "development_source": "R24_READ_ONLY",
        "confirmation_parameter_search": False,
        "factors": {
            "loss_observability_fusion": [False, True],
            "recovery_supervisor_v2": [False, True],
        },
        "fallback_parameters": {
            "contact_required_observations": CONTACT_REQUIRED_OBSERVATIONS,
            "contact_max_gap_ns": CONTACT_MAX_GAP_NS,
            "rgb_required_observations": RGB_REQUIRED_OBSERVATIONS,
            "rgb_max_gap_ns": RGB_MAX_GAP_NS,
            "rgb_relative_displacement_px": RGB_RELATIVE_DISPLACEMENT_PX,
        },
        "supervisor_maximum_outer_cycles": 3,
        "special_outcome": "SIGNAL_NOT_OBSERVED",
        "signal_not_observed_is_candidate_error": False,
    }
