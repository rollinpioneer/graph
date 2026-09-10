from __future__ import annotations

import unittest

from upgrade_v2.l2r_loss_observability.physics_probe import _compare, _mechanism_summary


class R11PureTests(unittest.TestCase):
    def test_probe_equivalence_requires_actions_events_and_states(self) -> None:
        row = {"actions": [], "events": [], "final_qpos": [], "final_qvel": []}
        self.assertTrue(_compare(row, row)["equivalent"])
        altered = {"actions": [], "events": [{"event": "changed"}], "final_qpos": [], "final_qvel": []}
        self.assertFalse(_compare(row, altered)["equivalent"])

    def test_missing_loss_event_is_not_applicable(self) -> None:
        result = _mechanism_summary([{"time": 1.0}], None)
        self.assertEqual(result["physical_loss_status"], "not_applicable")

    def test_persistent_hand_support_does_not_become_loss(self) -> None:
        trace = [{
            "time": 1.01,
            "weld_active": False,
            "object_qvel": [0.1, 0.0, 0.0],
            "post_detach_writeback_count": 3,
            "contacts": [{"is_hand_support": True, "is_external_support": False}],
        }]
        result = _mechanism_summary(trace, 1.0)
        self.assertEqual(result["physical_loss_status"], "not_verified")


if __name__ == "__main__":
    unittest.main()
