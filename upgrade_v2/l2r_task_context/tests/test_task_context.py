from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_task_context.contract import ControllerRequest, RequestedEffect
from upgrade_v2.l2r_task_context.collector import CASES, _lifecycle
from upgrade_v2.l2r_task_context.evaluate import _event_decision, _metrics, _unresolved_decision
from upgrade_v2.l2r_task_context.evaluate import lock_candidate
from upgrade_v2.l2r_task_context.event_interface import run_m1
from upgrade_v2.l2r_task_context.online_interface_repair import run_repaired_interface
from upgrade_v2.l2r_task_context.cache_fault_split import _audit_action_end_alignment, _oracle_intervals
from upgrade_v2.l2r_task_context.mechanism_localization import (
    _classify_interval,
    _evidence_onsets,
    _oracle_peak_intervals,
)
from upgrade_v2.l2r_task_context.followup_resolution import (
    _b_gate_reason,
    _c3_gate_reason,
    _candidate_segment_summary,
)
from upgrade_v2.l2r_task_context.repair_validation import LOCKED_CANDIDATES
from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_task_context.evaluate import _online_observation, _predicates


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
    def test_k8_keeps_attempt_active_after_lift_until_final_verify(self):
        case = CASES["K8_acquisition_touch_then_continue"]
        self.assertEqual(case["end_action"], "verify")
        lift_end = _lifecycle("lift", case["end_action"], "segment_complete", "action_end")
        verify_end = _lifecycle("verify", case["end_action"], "segment_complete", "action_end")
        self.assertFalse(lift_end["attempt_end"])
        self.assertTrue(lift_end["attempt_active"])
        self.assertTrue(verify_end["attempt_end"])
        self.assertFalse(verify_end["attempt_active"])

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


class OnlineInterfaceRepairTests(unittest.TestCase):
    def test_repairs_hold_end_without_treating_unknown_evidence_as_bad_sensor_data(self):
        rows = [observation(0), observation(1, active=False, end=True, reason="segment_complete")]
        out = run_repaired_interface(rows, [evidence("unknown"), evidence("unknown")], [request()])
        self.assertEqual(out[-1]["data_status"], "valid")
        self.assertEqual(out[-1]["hold_state"], "valid_no_current_hold_evidence")
        self.assertEqual(out[-1]["selected_action"], "retry_grasp")

    def test_active_touch_remains_no_action_and_reports_current_state(self):
        rows = [observation(0, contact=True), observation(1, contact=False, active=True)]
        out = run_repaired_interface(rows, [evidence("unknown"), evidence("unknown")], [request()])
        self.assertEqual(out[-1]["hold_state"], "valid_no_current_hold_evidence")
        self.assertEqual(out[-1]["selected_action"], "none")
        self.assertFalse(any(row["selected_action"] == "retry_grasp" for row in out))

    def test_historical_hold_requires_an_observed_loss_for_recovery(self):
        rows = [observation(0, contact=True, stable="true"), observation(1, contact=True)]
        out = run_repaired_interface(rows, [evidence("true"), evidence("unknown")], [request()])
        self.assertTrue(out[-1]["historical_hold_established"])
        self.assertEqual(out[-1]["hold_state"], "historical_hold_established")
        self.assertEqual(out[-1]["selected_action"], "none")
        self.assertEqual(out[-1]["reason_code"], "historical_hold_established_no_current_loss_observed")

    def test_observed_non_release_loss_recovers_and_release_does_not(self):
        rows = [observation(0, contact=True, stable="true"), observation(1, contact=False)]
        out = run_repaired_interface(rows, [evidence("true"), evidence("false")], [request()])
        self.assertEqual(out[-1]["selected_action"], "recover_object")
        release_rows = [observation(0, contact=True, stable="true"), observation(1, contact=False, closed=False, active=False, end=True, reason="release")]
        release = run_repaired_interface(release_rows, [evidence("true"), evidence("false")], [request(RequestedEffect.RELEASE_OBJECT)])
        self.assertEqual(release[-1]["selected_action"], "none")

    def test_invalid_observation_is_needs_observation_not_retry(self):
        rows = [observation(0), {**observation(1, active=False, end=True, reason="segment_complete"), "predicates": {"contact_present": "unknown", "gripper_command_closed": "true", "gripper_command_open": "false"}}]
        out = run_repaired_interface(rows, [evidence(), evidence()], [request()])
        self.assertEqual(out[-1]["data_status"], "data_missing_or_invalid")
        self.assertEqual(out[-1]["selected_action"], "needs_observation")


class EvaluationTests(unittest.TestCase):
    def test_unresolved_reference_stays_in_event_denominator(self):
        meta = {"rollout_id": "r", "root_family_id": "f", "case_id": "K1_hold_request_ends_without_hold"}
        rows = [{
            "time": 0.1,
            "effective_guards": {"retry_grasp": "true", "recover_object": "false"},
            "rollout_conflict": False,
        }]
        unresolved = _unresolved_decision(meta, "M1_requested_effect_gate", rows, {"reason": "missing frozen proxy"})
        metrics = _metrics("M1_requested_effect_gate", [unresolved], [])
        self.assertEqual(metrics["events"], 1)
        self.assertEqual(metrics["positive_events"], 1)
        self.assertEqual(metrics["reference_unresolved"], 1)
        self.assertIsNone(metrics["K1_recall"])

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


class CacheFaultSplitTests(unittest.TestCase):
    def test_action_end_alignment_requires_same_capture_order(self):
        dense = [
            {"time": 0.1, "capture_order": 2, "contact_present": False, "gripper_command": "open"},
        ]
        action_end = [
            {"time": 0.1, "capture_order": 3, "contact_present": False, "gripper_command": "open"},
        ]
        self.assertIn("action_end_row_0_missing_dense_match", _audit_action_end_alignment(dense, action_end))

    def test_oracle_interval_reports_drift_boundary_without_relabeling(self):
        rows = [
            {"time": "0.0", "capture_order": "0", "weld_state": "1", "object_xyz": "[0,0,0]", "gripper_xyz": "[0,0,0]"},
            {"time": "0.1", "capture_order": "1", "weld_state": "1", "object_xyz": "[0.1,0,0]", "gripper_xyz": "[0.0,0.03,0]"},
            {"time": "0.2", "capture_order": "2", "weld_state": "0", "object_xyz": "[0.1,0,0]", "gripper_xyz": "[0.0,0.03,0]"},
        ]
        intervals = _oracle_intervals(rows, 0.02)
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["status"], "reference_unresolved")
        self.assertFalse(intervals[0]["relative_drift_pass"])

    def test_future_and_reference_fields_do_not_change_online_evidence(self):
        base = [
            {"time": 0.0, "capture_order": 0, "contact_present": True, "gripper_command": "closed", "object_centroid": [0, 0], "gripper_centroid": [0, 0], "width": 100, "height": 100},
            {"time": 0.1, "capture_order": 1, "contact_present": True, "gripper_command": "closed", "object_centroid": [2, 0], "gripper_centroid": [2, 0], "width": 100, "height": 100},
            {"time": 0.2, "capture_order": 2, "contact_present": True, "gripper_command": "closed", "object_centroid": [4, 0], "gripper_centroid": [4, 0], "width": 100, "height": 100},
        ]

        def signature(rows):
            online = [_online_observation(row) for row in rows]
            geometry = build_features(online)
            predictions = []
            previous = None
            for row, features in zip(online, geometry):
                predictions.append({**row, "predicates": _predicates(row, features, previous)})
                previous = row
            return [
                (row["hold_evidence"], row["hold_memory"])
                for row in evaluate_candidate(predictions, geometry, "B_count2")
            ]

        contaminated = [
            {**row, "scenario": "future_case", "weld_state": True, "future_outcome": "success", "expected_action": "recover_object"}
            for row in base
        ]
        self.assertEqual(signature(base), signature(contaminated))


class MechanismLocalizationTests(unittest.TestCase):
    def test_static_interval_is_identified_without_changing_predicate(self):
        geometry = {
            "object_displacement_norm": 0.0,
            "gripper_displacement_norm": 0.0,
            "effective_motion_interval": True,
            "direction_cosine": None,
            "relative_vector_error": None,
        }
        self.assertEqual(
            _classify_interval(geometry, exact_repeat=True),
            "exact_static_repeat_accepted_by_magnitude_score",
        )

    def test_bcount2_onset_requires_transition_into_true(self):
        rows = [
            {"hold_evidence": "unknown"},
            {"hold_evidence": "true"},
            {"hold_evidence": "true"},
            {"hold_evidence": "false"},
            {"hold_evidence": "true"},
        ]
        self.assertEqual(_evidence_onsets(rows), [1, 4])

    def test_reference_peak_uses_first_weld_relative_vector(self):
        rows = [
            {"time": "0.0", "capture_order": "0", "phase": "action_end", "weld_state": "1", "object_xyz": "[1,2,3]", "gripper_xyz": "[0,0,0]"},
            {"time": "0.1", "capture_order": "1", "phase": "control_tick", "weld_state": "1", "object_xyz": "[1.03,2,3]", "gripper_xyz": "[0,0,0]"},
            {"time": "0.2", "capture_order": "2", "phase": "control_tick", "weld_state": "1", "object_xyz": "[1.03,2,3]", "gripper_xyz": "[0,0,0]"},
            {"time": "0.3", "capture_order": "3", "phase": "control_tick", "weld_state": "0", "object_xyz": "[1.03,2,3]", "gripper_xyz": "[0,0,0]"},
        ]
        intervals = _oracle_peak_intervals(rows)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0]["maximum_drift"], 0.03)
        self.assertEqual(intervals[0]["peaks"][0]["stage"], "constraint_establishment")
        self.assertEqual(intervals[0]["peaks"][-1]["stage"], "loss_boundary")


class FollowupResolutionTests(unittest.TestCase):
    def test_k5_interval_reasons_separate_magnitude_and_direction(self):
        prediction = {"predicates": {"gripper_command_closed": "true", "contact_present": "true"}}
        previous = {"predicates": {"gripper_command_closed": "true", "contact_present": "true"}}
        geometry = {
            "effective_motion_interval": True,
            "object_displacement_norm": 0.005,
            "gripper_displacement_norm": 0.010,
            "direction_cosine": 0.99,
            "relative_vector_error": 0.20,
        }
        self.assertEqual(_b_gate_reason(prediction, geometry), "magnitude_co_motion_below_0.8")
        self.assertEqual(_c3_gate_reason(prediction, previous, geometry, 0.35), "directional_interval_pass")

    def test_fixed_candidate_summary_does_not_rename_c3(self):
        rows = [
            {"candidate_id": candidate, "segment_kind": kind, "hold_evidence_observed": observed}
            for candidate, kind, observed in (
                ("B_count2", "initial_transient_contact", True),
                ("C3_vector_rho035", "initial_transient_contact", False),
                ("C3_vector_rho035", "same_k8_later_hold", True),
                ("C3_vector_rho035", "reference_labeled_k5", True),
                ("C3_vector_rho055", "initial_transient_contact", True),
            )
        ]
        summary = _candidate_segment_summary(rows)
        self.assertEqual(set(summary), {"B_count2", "C3_vector_rho035", "C3_vector_rho055"})
        self.assertEqual(summary["C3_vector_rho035"]["initial_false_hold_segments_with_evidence"], 0)
        self.assertEqual(summary["C3_vector_rho035"]["later_k8_segments_with_evidence"], 1)

    def test_reference_fields_do_not_change_c3_gate_reason(self):
        prediction = {"predicates": {"gripper_command_closed": "true", "contact_present": "true"}}
        previous = {"predicates": {"gripper_command_closed": "true", "contact_present": "true"}}
        geometry = {
            "effective_motion_interval": True,
            "object_displacement_norm": 0.01,
            "gripper_displacement_norm": 0.01,
            "direction_cosine": 1.0,
            "relative_vector_error": 0.0,
        }
        contaminated = {**prediction, "scenario": "future", "weld_state": True, "expected_action": "recover_object"}
        self.assertEqual(
            _c3_gate_reason(prediction, previous, geometry, 0.35),
            _c3_gate_reason(contaminated, previous, geometry, 0.35),
        )

    def test_attach_relpose_validation_keeps_only_frozen_candidates(self):
        self.assertEqual(LOCKED_CANDIDATES, ("B_count2", "C3_vector_rho035"))


if __name__ == "__main__":
    unittest.main()
