import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_hold_evidence.confirmation import confirm
from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _first_action, _online_observation
from upgrade_v2.l2r_hold_evidence.event_adapter import infer_history_complete, run_event_interface
from upgrade_v2.l2r_hold_evidence.hold_features import adjacent_features, build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.l2r_hold_evidence.inputs import write_json
from upgrade_v2.l2r_hold_evidence.recorder import merge_final_action_observations


def observation(index, *, time=None, closed=True, contact=True, stable="unknown", masked=False):
    return {
        "frame_index": index,
        "time": index * 0.05 if time is None else time,
        "width": 100,
        "height": 100,
        "object_centroid": (10 + index, 10),
        "gripper_centroid": (20 + index, 20),
        "contact_present": contact,
        "gripper_command": "closed" if closed else "open",
        "observation_masked": masked,
        "predicates": {
            "contact_present": "true" if contact else "false",
            "gripper_command_closed": "true" if closed else "false",
            "gripper_command_open": "false" if closed else "true",
            "stable_hold_observed": stable,
            "contact_recently_lost": "false",
        },
    }


class HoldGeometryTests(unittest.TestCase):
    def test_zero_motion_does_not_create_new_vector_evidence(self):
        previous = {"object_centroid": (10, 10), "gripper_centroid": (20, 20), "time": 0.0, "width": 100, "height": 100}
        current = {"object_centroid": (10, 10), "gripper_centroid": (20, 20), "time": 0.05, "width": 100, "height": 100}
        geometry = adjacent_features(previous, current, diagonal=100.0)
        self.assertIsNone(geometry["direction_cosine"])
        rows = evaluate_candidate(
            [observation(0), observation(1)],
            [{"effective_motion_interval": False}, geometry],
            "C3_vector",
        )
        self.assertNotEqual(rows[-1]["hold_evidence"], "true")

    def test_reverse_equal_motion_is_not_co_motion(self):
        geometry = adjacent_features(
            {"object_centroid": (10, 10), "gripper_centroid": (20, 20), "time": 0.0, "width": 100, "height": 100},
            {"object_centroid": (20, 10), "gripper_centroid": (10, 20), "time": 0.05, "width": 100, "height": 100},
            diagonal=100.0,
        )
        self.assertLess(geometry["direction_cosine"], 0.0)

    def test_valid_following_produces_vector_evidence(self):
        observations = [observation(0), observation(1)]
        rows = evaluate_candidate(observations, build_features(observations), "C3_vector")
        self.assertEqual(rows[-1]["hold_evidence"], "true")

    def test_motion_requires_contact_and_closed_at_both_ends(self):
        observations = [observation(0, contact=False), observation(1, contact=True)]
        rows = evaluate_candidate(observations, build_features(observations), "C3_vector")
        self.assertEqual(rows[-1]["hold_evidence"], "unknown")

    def test_masked_geometry_is_removed_before_features(self):
        masked = _online_observation(observation(0, masked=True))
        visible = _online_observation(observation(1))
        geometry = build_features([masked, visible])
        self.assertIsNone(masked["object_centroid"])
        self.assertFalse(geometry[1]["effective_motion_interval"])


class HoldTimeTests(unittest.TestCase):
    def test_same_timestamp_cannot_accumulate_time(self):
        observations = [observation(0, time=0.0), observation(1, time=0.0)]
        rows = evaluate_candidate(observations, build_features(observations), "C4_time", {"supported_time_min": 0.1, "supported_displacement_min": 0.004})
        self.assertTrue(all(row["supported_time"] == 0.0 for row in rows))

    def test_real_seconds_and_displacement_build_hold(self):
        observations = [observation(0), observation(1), observation(2)]
        rows = evaluate_candidate(observations, build_features(observations), "C4_time", {"supported_time_min": 0.1, "supported_displacement_min": 0.004})
        self.assertEqual(rows[-1]["hold_evidence"], "true")

    def test_gap_breaks_accumulation_without_interpolation(self):
        observations = [observation(0, time=0.0), observation(1, time=0.05), observation(2, time=0.40)]
        rows = evaluate_candidate(observations, build_features(observations), "C4_time", {"supported_time_min": 0.1, "supported_displacement_min": 0.004})
        self.assertEqual(rows[-1]["supported_time"], 0.0)
        self.assertNotEqual(rows[-1]["hold_evidence"], "true")

    def test_static_pause_preserves_established_hold(self):
        observations = [observation(0), observation(1)]
        observations.append({**observation(2), "object_centroid": (11, 10), "gripper_centroid": (21, 20)})
        rows = evaluate_candidate(observations, build_features(observations), "C3_vector")
        self.assertEqual(rows[1]["hold_memory"], "true")
        self.assertEqual(rows[2]["hold_memory"], "true")

    def test_short_static_contact_does_not_build_hold(self):
        observations = [observation(0), observation(1)]
        observations[1].update({"object_centroid": (10, 10), "gripper_centroid": (20, 20)})
        rows = evaluate_candidate(observations, build_features(observations), "C4_time", {"supported_time_min": 0.05, "supported_displacement_min": 0.001})
        self.assertEqual(rows[-1]["hold_memory"], "false")

    def test_excessive_relative_drift_blocks_time_hold(self):
        observations = [observation(0), observation(1), observation(2)]
        geometry = build_features(observations)
        geometry[2]["relative_position_current"] = (0.05, 0.0)
        rows = evaluate_candidate(
            observations,
            geometry,
            "C4_time",
            {"supported_time_min": 0.1, "supported_displacement_min": 0.004},
        )
        self.assertNotEqual(rows[-1]["hold_evidence"], "true")


class RecorderTests(unittest.TestCase):
    def test_action_end_replaces_same_timestamp_pre_update_capture(self):
        dense = [
            {"time": 0.0, "capture_order": 0, "phase": "control_tick", "contact_present": True},
            {"time": 0.05, "capture_order": 1, "phase": "control_tick", "contact_present": True},
        ]
        action_end = [
            {"time": 0.05, "capture_order": 2, "phase": "action_end", "contact_present": False},
        ]
        merged = merge_final_action_observations(dense, action_end)
        self.assertEqual([row["time"] for row in merged], [0.0, 0.05])
        self.assertFalse(merged[-1]["contact_present"])
        self.assertEqual(merged[-1]["source_phase"], "action_end")


class EventLifecycleTests(unittest.TestCase):
    def test_missed_grasp_emits_retry(self):
        observations = [observation(0, closed=False, contact=False, stable="false"), observation(1, closed=True, contact=False, stable="false")]
        evidence = [{"hold_evidence": "false", "hold_memory": "false"}] * 2
        self.assertEqual(run_event_interface(observations, evidence)[-1]["selected_action"], "retry_grasp")

    def test_held_loss_emits_recover_but_release_does_not(self):
        observations = [observation(0, closed=False, contact=False), observation(1, stable="true"), observation(2, contact=False)]
        observations[2]["predicates"]["contact_recently_lost"] = "true"
        evidence = [
            {"hold_evidence": "unknown", "hold_memory": "false"},
            {"hold_evidence": "true", "hold_memory": "true"},
            {"hold_evidence": "unknown", "hold_memory": "false"},
        ]
        self.assertEqual(run_event_interface(observations, evidence)[-1]["selected_action"], "recover_object")
        release = observation(2, closed=False, contact=False)
        release["predicates"]["contact_recently_lost"] = "true"
        release_row = run_event_interface(observations[:2] + [release], evidence)[-1]
        self.assertNotEqual(release_row["effective_guards"]["retry_grasp"], "true")
        self.assertNotEqual(release_row["effective_guards"]["recover_object"], "true")

    def test_recovery_then_new_loss_is_a_new_recover_decision(self):
        observations = [
            observation(0, closed=False, contact=False),
            observation(1, stable="true"),
            observation(2, contact=False),
            observation(3, contact=True, stable="true"),
            observation(4, contact=False),
        ]
        observations[2]["predicates"]["contact_recently_lost"] = "true"
        observations[4]["predicates"]["contact_recently_lost"] = "true"
        evidence = [
            {"hold_evidence": value, "hold_memory": "true" if value == "true" else "false"}
            for value in ("unknown", "true", "unknown", "true", "unknown")
        ]
        rows = run_event_interface(observations, evidence)
        self.assertEqual(rows[2]["selected_action"], "recover_object")
        self.assertEqual(rows[4]["selected_action"], "recover_object")

    def test_unknown_does_not_fallback_to_emergency(self):
        observations = [{"time": 0.0, "predicates": {"contact_present": "unknown", "gripper_command_closed": "unknown", "gripper_command_open": "unknown"}}]
        evidence = [{"time": 0.0, "hold_evidence": "unknown", "hold_memory": "unknown"}]
        self.assertEqual(run_event_interface(observations, evidence, history_complete=False)[0]["selected_action"], "needs_observation")

    def test_history_completeness_comes_from_stream(self):
        complete = [observation(0, closed=False, contact=False)]
        masked = [observation(0, closed=False, contact=False, masked=True), observation(1)]
        closed_start = [observation(0, closed=True, contact=True)]
        self.assertTrue(infer_history_complete(complete))
        self.assertFalse(infer_history_complete(masked))
        self.assertFalse(infer_history_complete(closed_start))

    def test_future_suffix_does_not_change_past_output(self):
        prefix = [observation(0, closed=False, contact=False), observation(1, stable="true")]
        evidence = [{"hold_evidence": "unknown", "hold_memory": "false"}, {"hold_evidence": "true", "hold_memory": "true"}]
        before = run_event_interface(prefix, evidence)
        after = run_event_interface(prefix + [observation(2, contact=False)], evidence + [{"hold_evidence": "unknown", "hold_memory": "false"}])
        self.assertEqual(before, after[: len(before)])

    def test_first_wrong_decision_is_not_overwritten(self):
        rows = [
            {"time": 1.0, "effective_guards": {"retry_grasp": "true", "recover_object": "false"}},
            {"time": 1.1, "effective_guards": {"retry_grasp": "false", "recover_object": "true"}},
        ]
        self.assertEqual(_first_action(rows)[0], "retry_grasp")


class ConfirmationGateTests(unittest.TestCase):
    def test_ready_lock_with_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dependency = root / "dependency.json"
            dependency.write_text("{}\n", encoding="utf-8")
            lock = root / "selection.json"
            write_json(lock, {
                "ready_for_confirmation": True,
                "selected_candidate_id": "C3_vector_rho035",
                "locked_files": [{"path": str(dependency), "sha256": "0" * 64}],
            })
            result = confirm(lock, {}, {}, root / "confirm")
            self.assertEqual(result["status"], "BLOCKED_LOCK_HASH_MISMATCH")
            self.assertFalse(result["new_data_consumed"])


if __name__ == "__main__":
    unittest.main()
