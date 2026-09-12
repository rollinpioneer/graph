from __future__ import annotations

from . import PROTOCOL_ID, RENDERER_BLOB, DETECTOR_BLOB

BASE_COMMIT = "70c93dd8b8e9d95bc3faa67a64f410a37a657e08"
FORMAL_MAIN = "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2"
WIDTH, HEIGHT, JPEG_QUALITY = 192, 144, 86
PHYSICS_HZ, CAPTURE_EVERY = 100, 5
BASE_DELTA_V = (40.0, 40.0, -8.0)
PULSE_DURATION_S = 0.20
SCALES = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25)
METHODS = ("O_B2", "O_C3", "S_G_H", "S_G_R", "S_G_HR")
O_CANDIDATES = {"O_B2": "B_count2", "O_C3": "C3_vector_rho035"}
RECOVERY_DEADLINE_S = 0.75
RETRY_DEADLINE_S = 0.50
PARAMETERS = {
    "direction_cosine_min": 0.8, "evidence_duration_s": 0.05,
    "height_off_m": 0.01, "height_on_m": 0.02, "max_gap_s": 0.25,
    "min_motion_3d_m": 0.004, "relative_hold_drift_m": 0.01,
    "relative_loss_drift_m": 0.02, "relative_rho_max": 0.35,
}

def protocol_lock() -> dict:
    return {
        "schema": "l2rar2_r17_true_rgb_temporal_confirmation_protocol_v1",
        "protocol_id": PROTOCOL_ID, "base_commit": BASE_COMMIT,
        "formal_main_commit": FORMAL_MAIN, "parameter_search_allowed": False,
        "candidate_methods": ["O_B2", "O_C3"],
        "diagnostic_methods": ["S_G_H", "S_G_R", "S_G_HR"],
        "rgb": {"width": WIDTH, "height": HEIGHT, "jpeg_quality": JPEG_QUALITY,
                "capture_every_physics_steps": CAPTURE_EVERY,
                "renderer_blob_sha": RENDERER_BLOB, "detector_blob_sha": DETECTOR_BLOB,
                "oracle_fallback_allowed": False},
        "status": "FROZEN_FOR_CONFIRMATION",
        "development_set_role": "FROZEN_DEVELOPMENT_ONLY",
        "temporal_windows": {
            "recovery_deadline_from_physical_onset_s": RECOVERY_DEADLINE_S,
            "retry_deadline_s": RETRY_DEADLINE_S,
        },
        "difficulty_calibration": {
            "base_delta_v_local_mps": list(BASE_DELTA_V),
            "duration_s": PULSE_DURATION_S,
            "scale_sequence": list(SCALES),
            "families": 4,
            "midpoint_extension_max": 1,
        },
        "confirmation": {"families": 6, "cases": 12, "rollouts": 72,
                         "physics_hz": PHYSICS_HZ, "online_capture_hz": PHYSICS_HZ // CAPTURE_EVERY},
        "selected_candidate_id": None, "confirmation_run": True,
        "l3_entry_allowed": False,
    }
