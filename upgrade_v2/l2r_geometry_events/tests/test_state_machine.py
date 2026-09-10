from __future__ import annotations

import unittest

from upgrade_v2.l2r_geometry_events.state_machine import (
    HELD,
    LOSS_PENDING,
    METHOD_G_H,
    METHOD_G_R,
    METHOD_G_HR,
    RELEASED,
    UNESTABLISHED,
    GeometryEventStateMachine,
)

from .helpers import PROTOCOL, sample


def run(method: str, samples):
    machine = GeometryEventStateMachine(method, PROTOCOL, "rollout", "root")
    return machine.run(samples), machine


def co_motion(steps: int, step: float, *, contact: bool = True, dt: float = 0.05, base: float = 0.5):
    samples = [sample(0.0, 0, [0.0, 0.0, base], [0.0, 0.0, base + 0.1], False, "open")]
    for index in range(1, steps + 1):
        time_value = index * dt
        samples.append(
            sample(
                time_value,
                index,
                [0.0, 0.0, base + index * step],
                [0.0, 0.0, base + 0.1 + index * step],
                contact,
                "closed",
            )
        )
    return samples


class EntryTests(unittest.TestCase):
    def test_static_contact_never_establishes_hold(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open"),
            sample(0.05, 1, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.10, 2, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], True, "closed"),
        ]
        for method in (METHOD_G_R, METHOD_G_HR):
            records, machine = run(method, samples)
            self.assertFalse(machine.ever_held, method)
            self.assertTrue(all(row["selected_action"] == "none" for row in records))

    def test_low_height_short_co_motion_distinguishes_methods(self):
        samples = co_motion(steps=3, step=0.005)
        _, g_h = run(METHOD_G_H, samples)
        _, g_r = run(METHOD_G_R, samples)
        _, g_hr = run(METHOD_G_HR, samples)
        self.assertFalse(g_h.ever_held, "height baseline misses a low short hold")
        self.assertTrue(g_r.ever_held)
        self.assertTrue(g_hr.ever_held)
        self.assertEqual(g_hr.hold_entry_route, "LOW_HEIGHT_CO_MOTION")
        self.assertFalse(g_hr.lift_confirmed)

    def test_slow_lift_below_step_gate_needs_height_route(self):
        samples = co_motion(steps=10, step=0.003)
        _, g_r = run(METHOD_G_R, samples)
        _, g_hr = run(METHOD_G_HR, samples)
        self.assertFalse(g_r.ever_held, "per-step motion stays below min_motion_3d_m")
        self.assertTrue(g_hr.ever_held, "lift-coupled route uses cumulative height")
        self.assertEqual(g_hr.hold_entry_route, "LIFT_COUPLED")
        self.assertTrue(g_hr.lift_confirmed)

    def test_hold_before_threshold_and_pause(self):
        samples = co_motion(steps=4, step=0.006)
        samples.append(sample(0.25, 5, [0.0, 0.0, 0.524], [0.0, 0.0, 0.624], True, "closed"))
        records, machine = run(METHOD_G_R, samples)
        self.assertTrue(machine.ever_held)
        self.assertEqual(records[-1]["hold_state"], HELD)
        self.assertTrue(all(row["selected_action"] == "none" for row in records))
        self.assertIsNone(records[-1]["loss_source"])


class LossExitTests(unittest.TestCase):
    def test_commanded_release_is_not_a_recovery(self):
        samples = co_motion(steps=4, step=0.006)
        samples.append(sample(0.25, 5, [0.0, 0.0, 0.524], [0.0, 0.0, 0.624], False, "open"))
        records, machine = run(METHOD_G_R, samples)
        self.assertEqual(machine.hold_state, RELEASED)
        self.assertTrue(all(row["selected_action"] != "recover_object" for row in records))

    def test_relative_detach_with_contact_true_raises_recovery(self):
        samples = co_motion(steps=4, step=0.006)
        time_value = 0.25
        for offset in (0.01, 0.025, 0.045):
            samples.append(
                sample(
                    round(time_value, 4),
                    5 + len(samples),
                    [0.0, 0.0, 0.524 + offset],
                    [0.0, 0.0, 0.624],
                    True,
                    "closed",
                )
            )
            time_value += 0.05
        records, machine = run(METHOD_G_HR, samples)
        self.assertEqual(machine.hold_state, LOSS_PENDING)
        self.assertTrue(any(row["selected_action"] == "recover_object" for row in records))
        self.assertIsNotNone(machine.pending_event_id)

    def test_contact_drop_after_hold_raises_recovery(self):
        samples = co_motion(steps=4, step=0.006)
        samples.append(sample(0.25, 5, [0.0, 0.0, 0.524], [0.0, 0.0, 0.624], False, "closed"))
        records, machine = run(METHOD_G_R, samples)
        self.assertEqual(machine.hold_state, LOSS_PENDING)
        self.assertEqual(records[-1]["loss_source"], "contact_true_to_false")
        self.assertTrue(any(row["selected_action"] == "recover_object" for row in records))


class QualityTests(unittest.TestCase):
    def test_gap_larger_than_max_gap_breaks_accumulation(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open"),
            sample(0.05, 1, [0.0, 0.0, 0.506], [0.0, 0.0, 0.606], True, "closed"),
            sample(0.40, 2, [0.0, 0.0, 0.512], [0.0, 0.0, 0.612], True, "closed"),
        ]
        _, machine = run(METHOD_G_R, samples)
        self.assertFalse(machine.ever_held)

    def test_missing_evidence_keeps_history_but_not_quality(self):
        samples = co_motion(steps=4, step=0.006)
        samples.append(sample(0.25, 5, None, None, None, "closed", quality="MISSING_OR_INVALID"))
        records, machine = run(METHOD_G_R, samples)
        self.assertTrue(machine.ever_held)
        self.assertEqual(records[-1]["current_quality"], "MISSING_OR_INVALID")
        self.assertEqual(records[-1]["hold_state"], HELD)
        self.assertEqual(records[-1]["history_quality"], "INCOMPLETE")

    def test_prefix_invariance_of_earlier_records(self):
        samples = co_motion(steps=6, step=0.006)
        full, _ = run(METHOD_G_HR, samples)
        prefix, _ = run(METHOD_G_HR, samples[:4])
        self.assertEqual(full[:4], prefix)

    def test_unestablished_attempt_with_hold_request_ends_in_retry(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open", attempt_active=False),
            sample(0.05, 1, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "closed"),
            sample(
                0.10,
                2,
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.6],
                False,
                "closed",
                attempt_end=True,
                attempt_end_reason="segment_complete",
            ),
        ]
        records, _ = run(METHOD_G_R, samples)
        self.assertEqual(records[-1]["selected_action"], "retry_grasp")
        self.assertEqual(records[-1]["reason_code"], "RETRY_ISSUED_AFTER_UNHELD_HOLD_ATTEMPT")

    def test_unestablished_state_has_no_action_without_request(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open", requested_effect=None),
            sample(0.05, 1, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "closed", requested_effect=None),
        ]
        records, machine = run(METHOD_G_R, samples)
        self.assertEqual(machine.hold_state, UNESTABLISHED)
        self.assertTrue(all(row["selected_action"] == "none" for row in records))


if __name__ == "__main__":
    unittest.main()
