from __future__ import annotations

import unittest

from upgrade_v2.l2r_geometry_events.adapters import DATA_MISMATCH, DATA_VALID
from upgrade_v2.l2r_geometry_events.features import (
    co_motion_qualifies,
    height_above_baseline,
    height_baseline,
    motion_features,
    relative_vector,
)

from .helpers import PROTOCOL, hold_sequence, sample


class HeightBaselineTests(unittest.TestCase):
    def test_first_open_no_contact_sample_anchors(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.05, 1, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open"),
            sample(0.10, 2, [0.0, 0.0, 0.54], [0.0, 0.0, 0.6], False, "open"),
        ]
        baseline = height_baseline(samples)
        self.assertEqual(baseline["status"], "AVAILABLE")
        self.assertAlmostEqual(baseline["z0"], 0.5)
        self.assertEqual(baseline["anchor_capture_order"], 1)

    def test_missing_anchor_reports_unavailable(self):
        samples = [sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed")]
        baseline = height_baseline(samples)
        self.assertEqual(baseline["status"], "HEIGHT_BASELINE_UNAVAILABLE")
        self.assertIsNone(height_above_baseline(samples[0], baseline["z0"]))


class MotionFeatureTests(unittest.TestCase):
    def test_dt_zero_is_not_valid(self):
        previous = sample(0.05, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed")
        current = sample(0.05, 1, [0.0, 0.0, 0.52], [0.0, 0.0, 0.62], True, "closed")
        features = motion_features(previous, current)
        self.assertFalse(features["motion_interval_valid"])
        self.assertEqual(features["invalid_reason"], "non_positive_dt")

    def test_gap_above_max_gap_is_rejected(self):
        previous = sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed")
        current = sample(0.3, 1, [0.0, 0.0, 0.52], [0.0, 0.0, 0.62], True, "closed")
        features = motion_features(previous, current, max_gap_s=0.25)
        self.assertFalse(features["motion_interval_valid"])
        self.assertEqual(features["invalid_reason"], "gap_exceeds_max_gap_s")

    def test_masked_sample_is_not_usable(self):
        previous = sample(0.0, 0, None, None, None, "open")
        current = sample(0.05, 1, [0.0, 0.0, 0.52], [0.0, 0.0, 0.62], True, "closed")
        features = motion_features(previous, current)
        self.assertFalse(features["motion_interval_valid"])

    def test_identical_delta_scores_perfectly(self):
        previous = sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed")
        current = sample(0.05, 1, [0.0, 0.0, 0.506], [0.0, 0.0, 0.606], True, "closed")
        features = motion_features(previous, current)
        self.assertTrue(features["motion_interval_valid"])
        self.assertAlmostEqual(features["direction_cosine"], 1.0, places=9)
        self.assertAlmostEqual(features["relative_vector_error"], 0.0, places=9)
        self.assertTrue(co_motion_qualifies(previous, current, features, PROTOCOL["parameters"]))

    def test_tiny_motion_fails_min_motion(self):
        previous = sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed")
        current = sample(0.05, 1, [0.0, 0.0, 0.501], [0.0, 0.0, 0.601], True, "closed")
        features = motion_features(previous, current)
        self.assertFalse(co_motion_qualifies(previous, current, features, PROTOCOL["parameters"]))

    def test_opposite_directions_fail_direction_gate(self):
        previous = sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed")
        current = sample(0.05, 1, [0.0, 0.0, 0.506], [0.0, 0.0, 0.594], True, "closed")
        features = motion_features(previous, current)
        self.assertLess(features["direction_cosine"], 0.8)
        self.assertFalse(co_motion_qualifies(previous, current, features, PROTOCOL["parameters"]))

    def test_relative_vector_and_mismatch_quality(self):
        good = sample(0.0, 0, [0.1, 0.2, 0.5], [0.0, 0.0, 0.6], True, "closed")
        vector = relative_vector(good)
        for actual, expected in zip(vector, [0.1, 0.2, -0.1]):
            self.assertAlmostEqual(actual, expected, places=12)
        bad = dict(good)
        bad["data_quality"] = DATA_MISMATCH
        self.assertIsNone(relative_vector(bad))
        self.assertEqual(good["data_quality"], DATA_VALID)


class ThresholdBoundaryTests(unittest.TestCase):
    def test_height_on_boundary_is_inclusive(self):
        samples = hold_sequence(steps=3, step_z=0.01)
        baseline = height_baseline(samples)
        if baseline["status"] == "AVAILABLE":
            self.assertAlmostEqual(height_above_baseline(samples[2], baseline["z0"]), 0.02, places=9)


if __name__ == "__main__":
    unittest.main()
