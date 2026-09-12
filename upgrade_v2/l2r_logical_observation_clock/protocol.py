from __future__ import annotations

from . import CANDIDATE_ID, PARENT_COMMIT, PROTOCOL_ID


def protocol_lock() -> dict:
    return {"schema": "l2rar2_r19_logical_observation_clock_protocol_v1",
            "protocol_id": PROTOCOL_ID, "parent_r18_commit": PARENT_COMMIT,
            "candidate_id": CANDIDATE_ID, "base_candidate": "O_C3",
            "clock_contract": {"causal_order_field": "capture_order",
                               "physical_interval_field": "time",
                               "capture_order_must_strictly_increase": True,
                               "physical_time_must_be_nondecreasing": True,
                               "minimum_physical_gap_s": 0.0,
                               "maximum_physical_gap_s": 0.10,
                               "same_physical_time_confirmation_allowed": True,
                               "backdating_allowed": False,
                               "action_time_source": "second_confirming_observation"},
            "development_gate": {"raw_B2_parity": "72/72", "raw_C3_parity": "72/72",
                                 "candidate_temporal_correct": "72/72", "early_actions": 0,
                                 "false_actions": 0, "missed_actions": 0, "unknown": 0,
                                 "prefix_causality": "PASS"},
            "parameter_search_allowed": False, "physical_execution_allowed_before_gate": False,
            "selected_candidate_id": None, "confirmation_run": False, "l3_entry_allowed": False}

