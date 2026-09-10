from __future__ import annotations

import json
import unittest

from upgrade_v2.l2r_geometry_events.adapters import (
    DATA_BOUNDARY,
    DATA_MISMATCH,
    DATA_VALID,
    FORBIDDEN_CANDIDATE_INPUTS,
    assert_no_forbidden_inputs,
    build_o_samples,
    build_s_samples,
    oracle_lookup,
)


def observation(order: int, time: float, contact=None, command="open"):
    return {
        "time": time,
        "capture_order": order,
        "frame_index": order,
        "object_centroid": [1.0, 2.0],
        "gripper_centroid": [3.0, 4.0],
        "object_confidence": 1.0,
        "gripper_confidence": 0.5,
        "gripper_command": command,
        "contact_present": contact,
        "attempt_id": 1,
        "attempt_phase": "acquiring",
        "attempt_active": True,
        "attempt_end": False,
        "attempt_end_reason": None,
        "attempt_end_sequence": 1,
        "width": 192,
        "height": 144,
    }


def oracle(order: int, time: float, weld: str = "0"):
    return {
        "frame_index": order,
        "time": time,
        "capture_order": order,
        "phase": "control_tick",
        "weld_state": weld,
        "object_xyz": json.dumps([0.0, 0.0, 0.5 + order * 0.01]),
        "gripper_xyz": json.dumps([0.0, 0.0, 0.6]),
        "object_target_distance": 0.5,
        "repair_version": "l2ra",
    }


class OIsolationTests(unittest.TestCase):
    def test_o_samples_exclude_oracle_and_labels(self):
        samples = build_o_samples([observation(0, 0.0)])
        for sample in samples:
            assert_no_forbidden_inputs(sample)
            for key in ("object_xyz", "gripper_xyz", "weld_state", "case_id"):
                self.assertNotIn(key, sample)
        self.assertIn("object_centroid", samples[0])


class SAdapterTests(unittest.TestCase):
    def test_matching_by_capture_order_and_time(self):
        samples = build_s_samples([observation(0, 0.05, True, "closed")], [oracle(0, 0.05)])
        self.assertEqual(samples[0]["data_quality"], DATA_VALID)
        self.assertEqual(len(samples[0]["object_xyz"]), 3)
        assert_no_forbidden_inputs(samples[0])

    def test_time_mismatch_is_flagged_not_silently_matched(self):
        samples = build_s_samples([observation(0, 0.20, True, "closed")], [oracle(0, 0.05)])
        self.assertEqual(samples[0]["data_quality"], DATA_MISMATCH)
        self.assertEqual(samples[0]["data_quality_reason"], "capture_order_time_mismatch")

    def test_missing_oracle_row_is_masked(self):
        samples = build_s_samples([observation(7, 0.35, True, "closed")], [oracle(0, 0.05)])
        self.assertEqual(samples[0]["data_quality"], "MISSING_OR_INVALID")
        self.assertIsNone(samples[0]["object_xyz"])

    def test_duplicate_oracle_capture_order_is_boundary_unknown(self):
        rows = [oracle(0, 0.05), oracle(0, 0.05)]
        self.assertEqual(oracle_lookup(rows)["duplicate_orders"], [0])
        samples = build_s_samples([observation(0, 0.05, True, "closed")], rows)
        self.assertEqual(samples[0]["data_quality"], DATA_BOUNDARY)

    def test_forbidden_key_detection(self):
        with self.assertRaises(AssertionError):
            assert_no_forbidden_inputs({"case_id": "K5_brief_hold_loss"})
        for key in FORBIDDEN_CANDIDATE_INPUTS:
            with self.assertRaises(AssertionError):
                assert_no_forbidden_inputs({key: 1})


if __name__ == "__main__":
    unittest.main()
