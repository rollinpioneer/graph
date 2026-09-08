import unittest

from upgrade_v2.l2r_hold_evidence.event_adapter import run_event_interface
from upgrade_v2.l2r_hold_evidence.hold_features import adjacent_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate


class HoldEvidenceTests(unittest.TestCase):
    def test_zero_motion_does_not_create_new_vector_evidence(self):
        previous = {"object_centroid": (10, 10), "gripper_centroid": (20, 20), "time": 0.0}
        current = {"object_centroid": (10, 10), "gripper_centroid": (20, 20), "time": 0.05}
        geometry = adjacent_features(previous, current, diagonal=100.0)
        self.assertIsNone(geometry["direction_cosine"])
        self.assertNotEqual(evaluate_candidate([{"time": 0.05, "predicates": {"contact_present": "true", "gripper_command_closed": "true"}}], [geometry], "C3_vector", {})[0]["hold_evidence"], "true")

    def test_reverse_motion_is_not_same_direction(self):
        geometry = adjacent_features({"object_centroid": (10, 10), "gripper_centroid": (20, 20), "time": 0.0}, {"object_centroid": (20, 10), "gripper_centroid": (10, 20), "time": 0.05}, diagonal=100.0)
        self.assertLess(geometry["direction_cosine"], 0.0)

    def test_same_timestamp_cannot_accumulate_time(self):
        observations = [{"time": 0.0, "predicates": {"contact_present": "true", "gripper_command_closed": "true"}}, {"time": 0.0, "predicates": {"contact_present": "true", "gripper_command_closed": "true"}}]
        geometry = [{"time": 0.0, "effective_motion_interval": False, "identity_ok": True}, {"time": 0.0, "effective_motion_interval": False, "identity_ok": True}]
        rows = evaluate_candidate(observations, geometry, "C4_time", {"supported_time_min": 0.1, "supported_displacement_min": 0.004})
        self.assertTrue(all(row["supported_time"] == 0.0 for row in rows))

    def test_candidate_unknown_does_not_fallback_to_emergency(self):
        observations = [{"time": 0.0, "predicates": {"contact_present": "unknown", "gripper_command_closed": "unknown", "gripper_command_open": "unknown"}}]
        evidence = [{"time": 0.0, "hold_evidence": "unknown", "hold_memory": "unknown"}]
        rows = run_event_interface(observations, evidence, history_complete=False)
        self.assertEqual(rows[0]["selected_action"], "needs_observation")


if __name__ == "__main__":
    unittest.main()
