from __future__ import annotations

import unittest

from upgrade_v2.l2r_geometry_events.adapters import (
    DATA_MASKED,
    attach_request_provenance,
    build_s_samples,
)
from upgrade_v2.l2r_geometry_events.evaluate import _projection
from upgrade_v2.l2r_geometry_events.state_machine import (
    HELD,
    LOSS_PENDING,
    METHOD_G_H,
    METHOD_G_R,
    METHOD_G_HR,
    RELEASED,
    GeometryEventStateMachine,
)

from .helpers import PROTOCOL, sample


def run(method: str, samples):
    machine = GeometryEventStateMachine(method, PROTOCOL, "rollout", "root")
    return machine.run(samples), machine


def co_motion(steps: int, step: float, *, start: float = 0.0, dt: float = 0.05, attempt_id: int = 1):
    samples = [sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open", attempt_id=attempt_id)]
    for index in range(1, steps + 1):
        samples.append(
            sample(
                start + index * dt,
                index,
                [0.0, 0.0, 0.5 + index * step],
                [0.0, 0.0, 0.6 + index * step],
                True,
                "closed",
                attempt_id=attempt_id,
            )
        )
    return samples


class CausalityTests(unittest.TestCase):
    def test_height_baseline_is_causal(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.05, 1, [0.0, 0.0, 0.53], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.10, 2, [0.0, 0.0, 0.53], [0.0, 0.0, 0.6], False, "open"),
            sample(0.15, 3, [0.0, 0.0, 0.55], [0.0, 0.0, 0.6], True, "closed"),
        ]
        records, machine = run(METHOD_G_H, samples)
        self.assertIsNone(records[0]["h"])
        self.assertIsNone(records[1]["h"])
        self.assertEqual(records[2]["h"], 0.0)
        self.assertAlmostEqual(records[3]["h"], 0.02, places=9)
        self.assertEqual(machine.z0_capture_order, 2)

    def test_future_open_sample_does_not_backfill(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.05, 1, [0.0, 0.0, 0.52], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.10, 2, [0.0, 0.0, 0.52], [0.0, 0.0, 0.6], False, "open"),
        ]
        records, _ = run(METHOD_G_H, samples)
        self.assertIsNone(records[0]["h"])
        self.assertIsNone(records[1]["h"])

    def test_prefix_invariance(self):
        samples = co_motion(steps=6, step=0.006)
        for method in (METHOD_G_H, METHOD_G_R, METHOD_G_HR):
            full, _ = run(method, samples)
            full_projection = _projection(full)
            for length in range(1, len(samples) + 1):
                prefix, _ = run(method, samples[:length])
                self.assertEqual(_projection(prefix), full_projection[:length], f"{method} n={length}")

    def test_long_gap_does_not_accumulate(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open"),
            sample(0.05, 1, [0.0, 0.0, 0.506], [0.0, 0.0, 0.606], True, "closed"),
            sample(0.40, 2, [0.0, 0.0, 0.512], [0.0, 0.0, 0.612], True, "closed"),
            sample(0.45, 3, [0.0, 0.0, 0.518], [0.0, 0.0, 0.618], True, "closed"),
        ]
        records, _ = run(METHOD_G_R, samples)
        # The 0.35 s gap contributes no coverage at all.
        self.assertEqual(records[2]["evidence_duration"], 0.0)
        self.assertLessEqual(records[3]["evidence_duration"], 0.05 + 1e-12)

    def test_invalid_sample_breaks_candidate_segment(self):
        samples = co_motion(steps=2, step=0.02, dt=0.02)
        samples.append(sample(0.05, 3, None, None, None, "closed", quality=DATA_MASKED))
        for index in range(4, 6):
            samples.append(
                sample(
                    index * 0.02,
                    index,
                    [0.0, 0.0, 0.5 + (index - 3) * 0.02],
                    [0.0, 0.0, 0.6 + (index - 3) * 0.02],
                    True,
                    "closed",
                )
            )
        _, machine = run(METHOD_G_R, samples)
        self.assertFalse(machine.ever_held)

    def test_unknown_does_not_become_loss(self):
        samples = co_motion(steps=5, step=0.006)
        samples.append(sample(0.30, 6, None, None, None, "closed", quality=DATA_MASKED))
        records, machine = run(METHOD_G_HR, samples)
        self.assertTrue(machine.ever_held)
        self.assertEqual(records[-1]["hold_state"], HELD)
        self.assertIsNone(records[-1]["loss_source"])


class RequestAttributionTests(unittest.TestCase):
    @staticmethod
    def _samples(count: int = 3):
        observations = [
            {
                "time": index * 0.05,
                "capture_order": index,
                "contact_present": False,
                "gripper_command": "open",
                "attempt_id": 1,
                "attempt_phase": "acquiring",
                "attempt_active": True,
                "attempt_end": False,
                "attempt_end_reason": None,
                "attempt_end_sequence": 1,
            }
            for index in range(count)
        ]
        oracle = [
            {
                "time": str(index * 0.05),
                "capture_order": str(index),
                "object_xyz": "[0, 0, 0.5]",
                "gripper_xyz": "[0, 0, 0.6]",
            }
            for index in range(count)
        ]
        return build_s_samples(observations, oracle)

    def test_request_attempt_must_match(self):
        samples = self._samples()
        requests = [
            {
                "attempt_id": 2,
                "received_time": 0.0,
                "issued_capture_order": -1,
                "requested_effect": "HOLD_OBJECT",
                "request_id": "r1",
                "source": "controller_dispatch",
            }
        ]
        attach_request_provenance(samples, requests)
        self.assertIsNone(samples[1]["requested_effect"])
        self.assertFalse(samples[1]["context_valid"])

    def test_same_time_request_respects_capture_order(self):
        samples = self._samples()
        requests = [
            {
                "attempt_id": 1,
                "received_time": 0.05,
                "issued_capture_order": 5,
                "requested_effect": "HOLD_OBJECT",
                "request_id": "r1",
                "source": "controller_dispatch",
            }
        ]
        attach_request_provenance(samples, requests)
        self.assertIsNone(samples[1]["requested_effect"])
        requests[0]["issued_capture_order"] = 1
        attach_request_provenance(samples, requests)
        self.assertEqual(samples[1]["requested_effect"], "HOLD_OBJECT")
        self.assertTrue(samples[1]["context_valid"])

    def test_missing_time_has_no_request(self):
        samples = self._samples()
        samples[1]["time"] = None
        requests = [
            {
                "attempt_id": 1,
                "received_time": 0.0,
                "issued_capture_order": -1,
                "requested_effect": "HOLD_OBJECT",
                "request_id": "r1",
                "source": "controller_dispatch",
            }
        ]
        attach_request_provenance(samples, requests)
        self.assertIsNone(samples[1]["requested_effect"])
        self.assertFalse(samples[1]["context_valid"])


class AttemptBoundaryTests(unittest.TestCase):
    def test_release_clears_pending_loss(self):
        samples = co_motion(steps=4, step=0.006)
        for offset, time_value in ((0.02, 0.25), (0.04, 0.30), (0.06, 0.35)):
            samples.append(sample(time_value, 5 + len(samples), [0.0, 0.0, 0.5 + offset], [0.0, 0.0, 0.6], True, "closed"))
        samples.append(sample(0.40, 20, [0.0, 0.0, 0.56], [0.0, 0.0, 0.6], False, "open"))
        records, machine = run(METHOD_G_HR, samples)
        self.assertEqual(machine.hold_state, RELEASED)
        self.assertIsNone(machine.pending_event_id)
        self.assertEqual(records[-1]["selected_action"], "none")

    def test_new_attempt_clears_candidate_coverage(self):
        samples = co_motion(steps=2, step=0.02, dt=0.02, attempt_id=1)
        samples.append(sample(0.05, 3, [0.0, 0.0, 0.54], [0.0, 0.0, 0.64], True, "closed", attempt_id=2))
        samples.append(sample(0.07, 4, [0.0, 0.0, 0.56], [0.0, 0.0, 0.66], True, "closed", attempt_id=2))
        _, machine = run(METHOD_G_R, samples)
        self.assertFalse(machine.ever_held)

    def test_event_id_is_not_recreated_each_frame(self):
        samples = co_motion(steps=4, step=0.006)
        for offset, time_value in ((0.02, 0.25), (0.04, 0.30), (0.06, 0.35)):
            samples.append(sample(time_value, 5 + len(samples), [0.0, 0.0, 0.5 + offset], [0.0, 0.0, 0.6], True, "closed"))
        records, machine = run(METHOD_G_HR, samples)
        self.assertEqual(machine.hold_state, LOSS_PENDING)
        ids = [row["event_id"] for row in records if row["event_id"]]
        self.assertEqual(len(set(ids)), 1)


if __name__ == "__main__":
    unittest.main()
