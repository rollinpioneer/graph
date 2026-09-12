from __future__ import annotations

from . import CANDIDATE_ID, PARENT_COMMIT, PROTOCOL_ID

FROZEN_BLOBS = {
    "guard": "dea50fa274c9062b43e03f5b445f1cbe43c9c552",
    "temporal_scoring": "e9182eff3e277017ebee8ad05a5bf3e9120871df",
    "physical_reference": "a0e01362537bad716ebfffd88609f0ff4f59e982",
    "r20_collector": "79dd7e578ce739f72a0ed361e468e5605c10d077",
    "o_tier_runner": "f26153784e0375b4ca455b15325930e103d9c455",
    "online_interface": "0d6b2ff93892b18bf358517bbcf473c9458f5d4c",
    "hold_predicates": "949b96eadb7f83fd80b383b5e17d79fb10d44dff",
    "hold_features": "c5ebe94e7999641dfe734ce185f5ae6692bb0e56",
    "physical_levels": "ff688f4166b6a7f4053e27633c2b8e1252edcd5d",
}


def protocol_lock() -> dict:
    return {
        "schema": "l2rar2_r21_protocol_lock_v1",
        "protocol_id": PROTOCOL_ID,
        "parent_commit": PARENT_COMMIT,
        "candidate_id": CANDIDATE_ID,
        "frozen_blobs": FROZEN_BLOBS,
        "physical_confirmation": {"families": 4, "cases": 8, "rollouts": 32, "workers_max": 2},
        "performance_gates": {
            "overall_accuracy_min": 71 / 72,
            "strong_accuracy_min": 29 / 30,
            "early_actions": 0,
            "false_actions": 0,
            "unknown_rate_max": 0.05,
            "physical_onset_latency_p90_max_s": 0.20,
        },
        "logical_clock": {"required_observations": 2, "minimum_gap_s": 0.0,
                          "maximum_gap_s": 0.10, "parameter_search_allowed": False},
        "reference_changed": False,
        "time_scoring_changed": False,
        "l3_entry_allowed": False,
    }
