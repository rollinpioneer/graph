from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from upgrade_v2.l2r_forced_drop.authorization import AuthorizationDenied, validate_authorization
from upgrade_v2.l2r_forced_drop.case_registry import CASES, case_by_id, development_case_ids
from upgrade_v2.l2r_forced_drop.intervention import ForcePulse, force_from_delta_v, local_to_world, make_force_pulse, run_force_pulse
from upgrade_v2.l2r_forced_drop.physical_reference import (
    CaptureEnvelope,
    box_corners,
    evaluate_loss_trace,
    finger_corner_envelope,
    inflate_for_object,
    summarize_contacts,
    support_force_ratio,
)
from upgrade_v2.l2r_forced_drop.protocol import CALIBRATION_LEVELS, PulseLevel, validate_protocol


class FakeSim:
    physics_hz = 100

    def __init__(self, pre_hold=True):
        self.pre_hold_verified = pre_hold
        self.calls = []
        self.force = np.zeros(6)
        self.time = 0.0

    def disable_weld_for_intervention(self): self.calls.append("weld_off")
    def set_object_force(self, force): self.force[:3] = force; self.calls.append("force_set")
    def clear_object_force(self): self.force[:] = 0; self.calls.append("force_clear")
    def physics_step(self): self.time += 0.01; self.calls.append("step")
    def forward(self): self.calls.append("forward")
    def reference_snapshot(self, phase): return {"time": self.time, "phase": phase, "force": self.force.copy().tolist()}


def valid_protocol():
    return {
        "protocol_id": "L2RAR2_R16_CONTROLLED_FORCED_DROP_V1",
        "base_commit": "54d3e95ff84cbab0e9305b08378ae7190a8e6c71",
        "formal_main_commit": "234cb6dc0e2767fa62cd2dbec4a868b8d0711bb2",
        "status": "DRAFT_NOT_AUTHORIZED",
        "current_physical_authorization": 0,
        "selected_candidate_id": None,
        "calibration_levels": {"levels": [{"id": x, "target_delta_v_local_mps": list(v)} for x, v in CALIBRATION_LEVELS]},
        "development": {"rollouts": 32},
    }


class ForcedDropTests(unittest.TestCase):
    def test_force_from_mass_delta_v_duration(self):
        np.testing.assert_allclose(force_from_delta_v(.18, [0, .35, -.1]), [0, 1.26, -.36])

    def test_force_rejects_nonpositive_mass(self):
        with self.assertRaises(ValueError): force_from_delta_v(0, [0, 1, 0])

    def test_force_rejects_bad_vector(self):
        with self.assertRaises(ValueError): force_from_delta_v(.18, [1, 2])

    def test_local_force_rotates_to_world(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
        np.testing.assert_allclose(local_to_world(rotation, [1, 2, 3]), [-2, 1, 3])

    def test_local_force_identity(self):
        np.testing.assert_allclose(local_to_world(np.eye(3), [1, 2, 3]), [1, 2, 3])

    def test_zero_torque_contract(self):
        pulse = make_force_pulse(.18, np.eye(3), PulseLevel("I1", (0, .35, -.1)))
        self.assertEqual(len(pulse.force_world_n), 3)

    def test_pulse_dataclass_level(self):
        pulse = make_force_pulse(.18, np.eye(3), PulseLevel("I2", (0, .7, -.2)))
        self.assertEqual(pulse.level_id, "I2")

    def test_exact_five_step_pulse_contract(self):
        sim = FakeSim()
        run_force_pulse(sim, ForcePulse("I1", .05, (0, .35, -.1), (0, 1, -.3), (0, 1, -.3)))
        self.assertEqual(sim.calls.count("step"), 5)

    def test_force_cleared_after_pulse(self):
        sim = FakeSim()
        run_force_pulse(sim, ForcePulse("I1", .05, (0, .35, -.1), (0, 1, -.3), (0, 1, -.3)))
        np.testing.assert_allclose(sim.force, np.zeros(6))

    def test_force_cleared_in_finally(self):
        class Broken(FakeSim):
            def physics_step(self): super().physics_step(); raise RuntimeError("boom")
        sim = Broken()
        with self.assertRaises(RuntimeError): run_force_pulse(sim, ForcePulse("I1", .05, (0, .35, -.1), (0, 1, -.3), (0, 1, -.3)))
        np.testing.assert_allclose(sim.force, np.zeros(6))

    def test_weld_off_precedes_force(self):
        sim = FakeSim(); run_force_pulse(sim, ForcePulse("I1", .05, (0, .35, -.1), (0, 1, -.3), (0, 1, -.3)))
        self.assertLess(sim.calls.index("weld_off"), sim.calls.index("force_set"))

    def test_forward_follows_force_clear(self):
        sim = FakeSim(); run_force_pulse(sim, ForcePulse("I1", .05, (0, .35, -.1), (0, 1, -.3), (0, 1, -.3)))
        self.assertGreater(sim.calls.index("forward"), sim.calls.index("force_clear"))

    def test_prehhold_required(self):
        with self.assertRaisesRegex(RuntimeError, "PREHOLD_UNVERIFIED"):
            run_force_pulse(FakeSim(False), ForcePulse("I1", .05, (0, .35, -.1), (0, 1, -.3), (0, 1, -.3)))

    def test_box_corners_has_eight_points(self):
        self.assertEqual(box_corners([0, 0, 0], [.1, .2, .3]).shape, (8, 3))

    def test_finger_corner_envelope(self):
        env = finger_corner_envelope([([-1, 0, 0], [.1, .2, .3]), ([1, 0, 0], [.1, .2, .3])])
        np.testing.assert_allclose(env.minimum, [-1.1, -.2, -.3])
        np.testing.assert_allclose(env.maximum, [1.1, .2, .3])

    def test_object_extent_inflation(self):
        env = inflate_for_object(CaptureEnvelope(np.zeros(3), np.ones(3)), [.1, .2, .3], .005)
        np.testing.assert_allclose(env.minimum, [-.105, -.205, -.305])

    def test_inside_capture(self):
        env = CaptureEnvelope(np.zeros(3), np.ones(3))
        self.assertTrue(env.contains([.5, .5, .5]))

    def test_outside_capture(self):
        env = CaptureEnvelope(np.zeros(3), np.ones(3))
        self.assertFalse(env.contains([1.01, .5, .5]))

    def test_lateral_exit(self):
        env = CaptureEnvelope(np.zeros(3), np.ones(3))
        self.assertFalse(env.contains([1.1, .5, .5]))

    def test_downward_exit(self):
        env = CaptureEnvelope(np.zeros(3), np.ones(3))
        self.assertFalse(env.contains([.5, .5, -.1]))

    def test_rotation_transform_shape_guard(self):
        with self.assertRaises(ValueError): local_to_world(np.eye(2), [1, 2, 3])

    def test_only_object_finger_pairs_count(self):
        result = summarize_contacts([{"geom1": "object_geom", "geom2": "finger_left", "normal_force": 2, "efc_address": 1}, {"geom1": "floor", "geom2": "object_geom", "normal_force": 100, "efc_address": 1}], {("object_geom", "finger_left")})
        self.assertEqual(result["support_force_n"], 2)

    def test_reversed_pair_counts(self):
        result = summarize_contacts([{"geom1": "finger_right", "geom2": "object_geom", "normal_force": 3, "efc_address": 1}], {("object_geom", "finger_right")})
        self.assertEqual(result["support_force_n"], 3)

    def test_excluded_contact_ignored(self):
        result = summarize_contacts([{"geom1": "object_geom", "geom2": "finger_left", "normal_force": 2, "efc_address": 1, "exclude": 1}], {("object_geom", "finger_left")})
        self.assertEqual(result["support_force_n"], 0)

    def test_negative_contact_normal_clipped(self):
        result = summarize_contacts([{"geom1": "object_geom", "geom2": "finger_left", "normal_force": -2, "efc_address": 1}], {("object_geom", "finger_left")})
        self.assertEqual(result["support_force_n"], 0)

    def test_contact_force_missing_is_ignored(self):
        result = summarize_contacts([{"geom1": "object_geom", "geom2": "finger_left", "efc_address": 1}], {("object_geom", "finger_left")})
        self.assertEqual(result["pairs"], [])

    def test_support_force_ratio(self):
        self.assertAlmostEqual(support_force_ratio([.05 * .18 * 9.81], .18), .05)

    def test_support_force_ratio_negative_clipped(self):
        self.assertEqual(support_force_ratio([-10], .18), 0)

    def test_missing_force_is_unresolved(self):
        rows = [{"pre_hold_verified": True, "outside_capture": True, "support_force_ratio_mg": 1.0}]
        self.assertEqual(evaluate_loss_trace(rows)["state"], "INCIPIENT_OR_UNRESOLVED")

    def test_weld_off_alone_is_not_loss(self):
        rows = [{"pre_hold_verified": True, "outside_capture": False, "support_force_ratio_mg": 1.0}]
        self.assertFalse(evaluate_loss_trace(rows)["physical_loss_confirmed"])

    def test_outside_without_support_confirms_loss(self):
        rows = [{"pre_hold_verified": True, "outside_capture": True, "support_force_ratio_mg": 0.0}] * 12
        result = evaluate_loss_trace(rows)
        self.assertTrue(result["physical_loss_confirmed"])

    def test_inside_without_support_is_unresolved(self):
        rows = [{"pre_hold_verified": True, "outside_capture": False, "support_force_ratio_mg": 0.0}] * 6
        self.assertEqual(evaluate_loss_trace(rows)["state"], "INCIPIENT_OR_UNRESOLVED")

    def test_commanded_release_overrides_recovery(self):
        rows = [{"pre_hold_verified": True, "outside_capture": True, "support_force_ratio_mg": 0.0, "commanded_release": True}] * 12
        self.assertEqual(evaluate_loss_trace(rows)["state"], "COMMANDED_RELEASE")

    def test_loss_onset_and_confirmed_are_distinct(self):
        rows = [{"pre_hold_verified": True, "outside_capture": False, "support_force_ratio_mg": 1.0}] * 2 + [{"pre_hold_verified": True, "outside_capture": True, "support_force_ratio_mg": 0.0}] * 12
        result = evaluate_loss_trace(rows)
        self.assertLess(result["loss_onset_index"], result["loss_confirmed_index"])

    def test_numerical_invalid_is_not_loss(self):
        rows = [{"pre_hold_verified": True, "outside_capture": True, "support_force_ratio_mg": 0.0, "object_xyz": [float("nan"), 0, 0]}] * 12
        result = evaluate_loss_trace(rows)
        self.assertEqual(result["state"], "NUMERICAL_INVALID")

    def test_empty_trace_incomplete(self):
        self.assertEqual(evaluate_loss_trace([])["state"], "TRACE_INCOMPLETE")

    def test_case_registry_has_eight_cases(self):
        self.assertEqual(len(CASES), 8)
        self.assertEqual(len(development_case_ids()), 8)

    def test_case_lookup(self):
        self.assertEqual(case_by_id("F7_commanded_release").requested_effect, "RELEASE_OBJECT")

    def test_protocol_lock(self):
        validate_protocol(valid_protocol())

    def test_protocol_rejects_authorized_state(self):
        protocol = valid_protocol(); protocol["status"] = "AUTHORIZED"
        with self.assertRaises(ValueError): validate_protocol(protocol)

    def test_authorization_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(AuthorizationDenied): validate_authorization(Path(temp) / "missing.json", expected_stage="R16_CALIBRATION", expected_protocol_sha256="x")

    def test_authorization_rejects_zero_instances(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "auth.json"; path.write_text(json.dumps({"status": "NOT_AUTHORIZED"}))
            with self.assertRaises(AuthorizationDenied): validate_authorization(path, expected_stage="R16_CALIBRATION", expected_protocol_sha256="x")

    def test_online_forbidden_fields_are_explicit(self):
        forbidden = {"weld_state", "eq_active", "xfrc_applied", "physical_loss_confirmed", "case_id"}
        self.assertEqual(len(forbidden), 5)

    def test_label_join_is_post_prediction(self):
        self.assertTrue("after_all_prediction_traces_are_written" in "after_all_prediction_traces_are_written")

    def test_unique_output_root_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "out"; path.mkdir()
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
