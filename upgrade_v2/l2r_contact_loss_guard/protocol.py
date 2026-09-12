from __future__ import annotations

from . import CANDIDATE_ID, PROBLEM_ROLLOUT_ID, PROTOCOL_ID

BASE_COMMIT = "d660167c7d7acb4d9f89c36fa5b8ff7d47d16a63"
R17_RUNNER_COMMIT = "0def2be448f287e9beac6d0f69f90790a6437332"
FORMAL_MAIN = "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2"
REQUIRED_SAMPLES = 2
MAXIMUM_PENDING_GAP_S = 0.10
ADJUDICATION_OUTCOMES = (
    "TIMESTAMP_SEMANTICS_BUG",
    "REFERENCE_ONSET_RECOMPUTATION_REQUIRED",
    "GENUINE_NON_RELEASE_CONTACT_PRECURSOR",
)
LEVELS = {
    "weak": {"scale": 0.5, "delta_v_local_mps": (20.0, 20.0, -4.0)},
    "medium": {"scale": 0.625, "delta_v_local_mps": (25.0, 25.0, -5.0)},
    "strong": {"scale": 0.75, "delta_v_local_mps": (30.0, 30.0, -6.0)},
}


def protocol_lock() -> dict:
    return {
        "schema": "l2rar2_r18_contact_loss_persistence_protocol_v1",
        "protocol_id": PROTOCOL_ID, "base_commit": BASE_COMMIT,
        "base_branch": "research/l2rar2-r17-true-rgb-temporal-confirmation-v1",
        "new_branch": "research/l2rar2-r18-contact-loss-persistence-confirmation-v1",
        "r17_runner_commit": R17_RUNNER_COMMIT, "formal_main_commit": FORMAL_MAIN,
        "problem_rollout_id": PROBLEM_ROLLOUT_ID,
        "adjudication_outcomes": list(ADJUDICATION_OUTCOMES),
        "candidate_id": CANDIDATE_ID,
        "guard": {"required_consecutive_valid_samples": REQUIRED_SAMPLES,
                  "maximum_pending_gap_s": MAXIMUM_PENDING_GAP_S,
                  "applies_to_reason_code": "historical_hold_established_then_observed_non_release_contact_loss",
                  "emit_on_first_sample": "none", "emit_on_second_sample": "recover_object",
                  "backdating_allowed": False, "parameter_search_allowed": False},
        "r17_development_gate": {"resolved": "72/72", "temporal_correct": "72/72",
                                 "early_actions": 0, "false_actions": 0,
                                 "missed_actions": 0, "unknown": 0},
        "r18_confirmation": {"families": 6, "cases": 12, "rollouts": 72,
                             "phase_offsets_ms": [0, 10, 20, 30, 40]},
        "confirmation_gate": {"false_actions_T1_T2_T3_T12": 0, "early_actions": 0,
                              "unknown_rate_max": 0.05, "strong_correct_in_window_min": "35/36",
                              "T4_T5_outcome_correct": "12/12", "overall_temporal_correct_min": "71/72",
                              "p90_latency_from_physical_onset_max_s": 0.20},
        "status": "DRAFT_ZERO_EXECUTION",
        "confirmation_run": False, "selected_candidate_id": None,
        "l3_entry_allowed": False,
    }
