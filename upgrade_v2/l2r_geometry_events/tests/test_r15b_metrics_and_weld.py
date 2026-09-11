from __future__ import annotations

import unittest

from upgrade_v2.l2r_geometry_events.metrics import (
    CORRECT,
    MISSED_REQUIRED_ACTION,
    REFERENCE_UNRESOLVED,
    WRONG_ACTION,
    classify_outcome,
    summarize_method,
)
from upgrade_v2.l2r_geometry_events.weld_audit import (
    ORDER_NO_TRANSITION,
    ORDER_SAME_TIME_ORDERED,
    ORDER_SAME_TIME_UNKNOWN,
    ORDER_STRICTLY_LATER,
    REL_INSIDE,
    REL_NOT_COMPARABLE,
    event_transition_relation,
    weld_transition,
)


def oracle_row(order: int, time: float, weld: str):
    return {"capture_order": str(order), "time": str(time), "weld_state": weld, "frame_index": str(order)}


class OutcomeMetricTests(unittest.TestCase):
    def test_none_equals_none_is_correct(self):
        self.assertEqual(classify_outcome("none", "none", True), CORRECT)

    def test_required_action_missing_is_missed(self):
        self.assertEqual(classify_outcome("recover_object", "none", True), MISSED_REQUIRED_ACTION)
        self.assertEqual(classify_outcome("retry_grasp", "none", True), MISSED_REQUIRED_ACTION)

    def test_wrong_action_is_wrong(self):
        self.assertEqual(classify_outcome("none", "recover_object", True), WRONG_ACTION)
        self.assertEqual(classify_outcome("retry_grasp", "recover_object", True), WRONG_ACTION)

    def test_negative_window_false_recovery(self):
        rows = [
            {"case_id": "K2_touch_request_completes_without_hold", "outcome": WRONG_ACTION, "selected_action": "recover_object", "input_tier": "S"},
            {"case_id": "K3_normal_hold_pause_resume", "outcome": CORRECT, "selected_action": "none", "input_tier": "S"},
        ]
        summary = summarize_method(rows)
        self.assertEqual(summary["negative_window_false_recovery"], 1)
        self.assertEqual(summary["negative_window_false_retry"], 0)

    def test_premature_recovery_is_separate(self):
        rows = [
            {
                "case_id": "K4_regular_hold_loss",
                "outcome": CORRECT,
                "selected_action": "recover_object",
                "premature_emergency": True,
                "premature_recovery": True,
                "premature_retry": False,
                "post_window_emergency": False,
                "input_tier": "S",
            }
        ]
        summary = summarize_method(rows)
        self.assertEqual(summary["premature_recovery"], 1)
        self.assertEqual(summary["post_window_emergency"], 0)

    def test_post_window_action_is_informational(self):
        rows = [
            {
                "case_id": "K1_hold_request_ends_without_hold",
                "outcome": MISSED_REQUIRED_ACTION,
                "selected_action": "none",
                "post_window_emergency": True,
                "input_tier": "O",
            }
        ]
        summary = summarize_method(rows)
        self.assertEqual(summary["post_window_emergency"], 1)
        self.assertEqual(summary["premature_emergency"], 0)

    def test_physical_loss_recall_not_estimable(self):
        rows = [
            {
                "case_id": "K5_brief_hold_loss",
                "outcome": MISSED_REQUIRED_ACTION,
                "selected_action": "none",
                "input_tier": "S",
            }
        ]
        summary = summarize_method(rows)
        self.assertEqual(summary["physical_loss_recall"], "NOT_ESTIMABLE")
        self.assertEqual(summary["independent_physical_loss_status"], "UNRESOLVED")

    def test_reference_unresolved_is_distinct(self):
        self.assertEqual(classify_outcome(None, "none", False), REFERENCE_UNRESOLVED)


class WeldTransitionTests(unittest.TestCase):
    def test_weld_transition_one_to_zero(self):
        rows = [oracle_row(0, 0.05, "1"), oracle_row(1, 0.10, "1"), oracle_row(2, 0.15, "0")]
        result = weld_transition(rows)
        self.assertTrue(result["transition_found"])
        self.assertEqual(result["last_on_time"], 0.10)
        self.assertEqual(result["first_off_time"], 0.15)
        self.assertEqual(result["transition_order_status"], ORDER_STRICTLY_LATER)

    def test_all_weld_on_has_no_release(self):
        rows = [oracle_row(0, 0.05, "1"), oracle_row(1, 0.10, "1")]
        result = weld_transition(rows)
        self.assertFalse(result["transition_found"])
        self.assertIsNone(result["last_on_time"])
        self.assertEqual(result["transition_order_status"], ORDER_NO_TRANSITION)

    def test_same_time_ordered_by_capture_order(self):
        rows = [oracle_row(3, 0.20, "1"), oracle_row(4, 0.20, "0")]
        result = weld_transition(rows)
        self.assertEqual(result["transition_order_status"], ORDER_SAME_TIME_ORDERED)

    def test_same_time_without_order_is_unknown(self):
        rows = [oracle_row(4, 0.20, "1"), oracle_row(4, 0.20, "0")]
        result = weld_transition(rows)
        self.assertEqual(result["transition_order_status"], ORDER_SAME_TIME_UNKNOWN)

    def test_event_and_weld_transition_are_separate(self):
        rows = [oracle_row(0, 0.05, "1"), oracle_row(1, 0.80, "0")]
        transition = weld_transition(rows)
        self.assertEqual(event_transition_relation(0.80, transition), REL_INSIDE)
        self.assertEqual(event_transition_relation(None, transition), REL_NOT_COMPARABLE)
        self.assertEqual(event_transition_relation(0.02, transition), "EVENT_BEFORE_OBSERVED_TRANSITION")


if __name__ == "__main__":
    unittest.main()
