from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_task_context.contract import ControllerRequest, RequestedEffect
from upgrade_v2.l2r_task_context.evaluate import _event_decision
from upgrade_v2.l2r_task_context.evaluate import lock_candidate
from upgrade_v2.l2r_task_context.event_interface import run_m1


def observation(index: int, *, contact: bool = False, closed: bool = True, active: bool = True, end: bool = False, reason: str | None = None, attempt_id: int = 1, stable: str = "false") -> dict:
    return {
        "frame_index": index,
        "capture_order": index,
        "time": index * 0.1,
        "attempt_id": attempt_id,
        "attempt_phase": "ended" if end else "acquiring" if active else "inactive",
        "attempt_active": active,
        "attempt_end": end,
        "attempt_end_reason": reason,
        "attempt_end_sequence": 1,
        "predicates": {
            "contact_present": "true" if contact else "false",
            "gripper_command_closed": "true" if closed else "false",
            "gripper_command_open": "false" if closed else "true",
            "contact_recently_lost": "false",
            "stable_hold_observed": stable,
        },
    }


def evidence(stable: str = "false") -> dict:
    return {"hold_evidence": stable, "hold_memory": "true" if stable == "true" else "false", "closed": "true"}


def request(effect: RequestedEffect = RequestedEffect.HOLD_OBJECT, *, attempt_id: int = 1, received: float = 0.0) -> dict:
    return ControllerRequest("request-1", attempt_id, "track-1", effect, 0.0, -1, received).as_dict()


class EventInterfaceTests(unittest.TestCase):
    def test_hold_and_touch_end_differ_with_same_physics(self):
        rows = [observation(0), observation(1, active=False, end=True, reason="segment_complete")]
        ev = [evidence(), evidence()]
        self.assertEqual(run_m1(rows, ev, [request(RequestedEffect.HOLD_OBJECT)])[-1]["selected_action"], "retry_grasp")
        self.assertEqual(run_m1(rows, ev, [request(RequestedEffect.TOUCH_OBJECT)])[-1]["selected_action"], "none")

    def test_active_transient_contact_does_not_retry(self):
        rows = [observation(0, contact=True), observation(1, contact=False, active=True)]
        out = run_m1(rows, [evidence(), evidence()], [request()])
        self.assertFalse(any(row["effective_guards"]["retry_grasp"] == "true" for row in out))

    def test_true_end_without_hold_retries(self):
        rows = [observation(0), observation(1, active=False, end=True, reason="segment_complete")]
        self.assertEqual(run_m1(rows, [evidence(), evidence()], [request()])[-1]["reason_code"], "hold_request_normally_ended_without_hold")

    def test_missing_history_and_context_block_retry(self):
        rows = [observation(0), observation(1, active=False, end=True, reason="segment_complete")]
        ev = [evidence(), evidence()]
        self.assertEqual(run_m1(rows, ev, [request()], history_complete=False)[-1]["selected_action"], "needs_observation")
        self.assertEqual(run_m1(rows, ev, [], history_complete=True)[-1]["selected_action"], "needs_observation")

    def test_cross_attempt_and_late_context_block_retry(self):
        rows = [observation(0), observation(1, active=False, end=True, reason="segment_complete")]
        ev = [evidence(), evidence()]
        self.assertEqual(run_m1(rows, ev, [request(attempt_id=2)])[-1]["selected_action"], "needs_observation")
        self.assertEqual(run_m1(rows, ev, [request(received=1.0)])[-1]["selected_action"], "needs_observation")

    def test_held_loss_not_masked_by_touch_context(self):
        rows = [observation(0, contact=True, stable="true"), observation(1, contact=False)]
        out = run_m1(rows, [evidence("true"), evidence()], [request(RequestedEffect.TOUCH_OBJECT)])
        self.assertEqual(out[-1]["selected_action"], "recover_object")

    def test_release_and_cancel_end_do_not_retry(self):
        for reason in ("release", "cancelled"):
            rows = [observation(0), observation(1, active=False, end=True, reason=reason)]
            out = run_m1(rows, [evidence(), evidence()], [request()])
            self.assertNotIn(out[-1]["selected_action"], {"retry_grasp", "recover_object"})

    def test_duplicate_end_edge_consumed_once(self):
        rows = [observation(0), observation(1, active=False, end=True, reason="segment_complete"), observation(2, active=False, end=True, reason="segment_complete")]
        out = run_m1(rows, [evidence()] * 3, [request()])
        self.assertTrue(out[1]["end_edge_fresh"])
        self.assertFalse(out[2]["end_edge_fresh"])
        self.assertEqual(out[1]["pending_event_id"], out[2]["pending_event_id"])

    def test_new_attempt_without_observed_start_is_not_complete(self):
        rows = [observation(0, active=False, end=True, reason="segment_complete", attempt_id=2)]
        out = run_m1(rows, [evidence()], [request(attempt_id=2)])
        self.assertEqual(out[-1]["selected_action"], "needs_observation")

    def test_prefix_causality(self):
        prefix = [observation(0, contact=True), observation(1, contact=False)]
        ev = [evidence(), evidence()]
        initial = run_m1(prefix, ev, [request()])
        extended = run_m1(prefix + [observation(2, active=False, end=True, reason="segment_complete")], ev + [evidence()], [request()])
        self.assertEqual(initial, extended[:2])

    def test_none_or_unknown_has_no_legacy_fallback(self):
        rows = [observation(0)]
        out = run_m1(rows, [evidence()], [request(RequestedEffect.OBSERVE)])
        self.assertEqual(out[-1]["selected_action"], "none")

    def test_request_purpose_is_not_outcome(self):
        success = request(RequestedEffect.HOLD_OBJECT)
        failure = request(RequestedEffect.HOLD_OBJECT)
        self.assertEqual(success["requested_effect"], failure["requested_effect"])
        self.assertNotIn("outcome", success)


class EvaluationTests(unittest.TestCase):
    def test_negative_any_time_emergency_is_failure(self):
        meta = {"rollout_id": "r", "root_family_id": "f", "case_id": "K2_touch_request_completes_without_hold"}
        rows = [
            {"time": 0.0, "effective_guards": {"retry_grasp": "true", "recover_object": "false"}},
            {"time": 0.1, "effective_guards": {"retry_grasp": "false", "recover_object": "false"}},
        ]
        ev = [{"time": 0.0, "hold_memory": "false", "closed": "true"}]
        ref = {"event_id": "e", "reference_type": "negative", "expected_action": "none", "decision_window_start": 0.0, "decision_window_end": 0.2}
        result = _event_decision(meta, "M1_requested_effect_gate", rows, ev, ref)
        self.assertTrue(result["false_emergency"])
        self.assertFalse(result["correct"])

    def test_positive_first_wrong_action_is_failure(self):
        meta = {"rollout_id": "r", "root_family_id": "f", "case_id": "K1_hold_request_ends_without_hold"}
        rows = [
            {"time": 0.0, "effective_guards": {"retry_grasp": "false", "recover_object": "true"}},
            {"time": 0.1, "effective_guards": {"retry_grasp": "true", "recover_object": "false"}},
        ]
        ev = [{"time": 0.0, "hold_memory": "false", "closed": "true"}]
        ref = {"event_id": "e", "reference_type": "positive", "expected_action": "retry_grasp", "decision_window_start": 0.0, "decision_window_end": 0.2}
        result = _event_decision(meta, "M1_requested_effect_gate", rows, ev, ref)
        self.assertEqual(result["selected_action"], "recover_object")
        self.assertTrue(result["missed"])

    def test_failed_development_cannot_create_selection_lock(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            development = root / "development"
            development.mkdir()
            (development / "development_gates.json").write_text(json.dumps({"status": "DEVELOPMENT_FAILED", "all_pass": False}))
            (root / "protocol.json").write_text("{}")
            (root / "generation.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                lock_candidate(development, root / "protocol.json", root / "generation.json", root / "selection.json")
            self.assertFalse((root / "selection.json").exists())


if __name__ == "__main__":
    unittest.main()
