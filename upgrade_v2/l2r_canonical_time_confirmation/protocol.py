from __future__ import annotations

from . import CANDIDATE_ID, PARENT_COMMIT, PROTOCOL_ID


def protocol_lock()->dict:
    return {"schema":"l2rar2_r22_protocol_lock_v1","protocol_id":PROTOCOL_ID,"parent_commit":PARENT_COMMIT,"candidate_id":CANDIDATE_ID,
            "clock":{"field":"physical_time_ns","source":"capture_physics_step_index","unit":"ns","required_distinct_physical_times":2,"minimum_gap_ns":1,"maximum_gap_ns":100_000_000},
            "historical_replay":{"R17":72,"R20":72,"R21":32,"required_correct":176},
            "physical_confirmation":{"families":4,"cases":6,"rollouts":24,"workers_max":2},
            "frozen":{"o_c3":True,"thresholds":True,"temporal_scoring":True,"physical_reference":True,"parameter_search":False},
            "on_pass":{"selected_candidate_id":CANDIDATE_ID,"l3_entry_ready":True,"l3_entry_allowed":False}}
