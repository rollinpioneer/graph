import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_hold_evidence.diagnose_attempt_boundary import _command_at, _online_signature


class AttemptBoundaryDiagnosticTests(unittest.TestCase):
    def test_online_signature_does_not_use_future_rows(self):
        rows = [
            {"time": 0.4, "gripper_command": "closed", "contact_present": True},
            {"time": 0.45, "gripper_command": "closed", "contact_present": False},
            {"time": 0.65, "gripper_command": "closed", "contact_present": True},
        ]
        result = _online_signature(rows, 0.45)
        self.assertTrue(result["closed"])
        self.assertFalse(result["contact_present"])
        self.assertTrue(result["previous_contact_present"])
        self.assertTrue(result["prior_contact_seen"])

    def test_command_at_uses_arrived_target_only(self):
        controls = [
            {"start_time": 0.35, "end_time": 0.4, "gripper_command": "closed", "mocap_position": [0.0, 0.0, 0.0], "physics_steps": 5},
            {"start_time": 0.4, "end_time": 0.45, "gripper_command": "closed", "mocap_position": [0.0, 0.0, 0.0], "physics_steps": 5},
            {"start_time": 0.45, "end_time": 0.5, "gripper_command": "closed", "mocap_position": [0.0, 0.1, 0.0], "physics_steps": 5},
        ]
        result = _command_at(controls, 0.45)
        self.assertEqual(result["command_target_delta_norm"], 0.0)
        self.assertEqual(result["command_gripper"], "closed")


if __name__ == "__main__":
    unittest.main()
