from __future__ import annotations

import unittest

from upgrade_v2.l2r_temporal_loss_decision.cli import _horizon_metric
from upgrade_v2.l2r_temporal_loss_decision.episodes import (
    _attempt_at,
    _last_observation_before,
    _logged_recovery_execution,
)
from upgrade_v2.l2r_temporal_loss_decision.replay import replay
from upgrade_v2.l2r_temporal_loss_decision.shadow_replenishment import design
from upgrade_v2.l2r_temporal_loss_decision.interface_shadow import run as run_interface_shadow


class CensoringEvidenceTests(unittest.TestCase):
    def test_proposal_is_not_execution_log(self) -> None:
        self.assertFalse(_logged_recovery_execution({"recovery_executions": []}))
        self.assertTrue(_logged_recovery_execution({"recovery_executions": [{"success": False}]}))

    def test_attempt_at_uses_time_and_capture_order(self) -> None:
        rows = [
            {"physical_time_ns": 100, "capture_order": 1, "attempt_id": 1},
            {"physical_time_ns": 100, "capture_order": 2, "attempt_id": 2},
        ]
        self.assertEqual(_attempt_at(rows, [], (100, 1)), 1)
        self.assertEqual(_attempt_at(rows, [], (100, 2)), 2)

    def test_evidence_stops_strictly_before_control_boundary(self) -> None:
        rows = [
            {"physical_time_ns": 0, "capture_order": 0},
            {"physical_time_ns": 100, "capture_order": 1},
            {"physical_time_ns": 100, "capture_order": 2},
        ]
        self.assertEqual(_last_observation_before(rows, (100, 2)), rows[1])

    def test_replay_honors_same_time_capture_order_cutoff(self) -> None:
        rows = [
            {"physical_time_ns": 0, "capture_order": 0},
            {"physical_time_ns": 100, "capture_order": 1},
            {"physical_time_ns": 100, "capture_order": 2},
        ]
        result = replay({
            "observations": rows,
            "evidence_end_ns": 100,
            "evidence_end_capture_order": 1,
        }, "B1")
        self.assertEqual(len(result["outputs"]), 2)


class ReportingTests(unittest.TestCase):
    def test_horizon_categories_partition_reference_losses(self) -> None:
        rows = [
            {"truth": "LOSS", "physical_loss_onset_ns": 100, "evidence_end_ns": 1000, "first_evidence_ns": 150},
            {"truth": "LOSS", "physical_loss_onset_ns": 100, "evidence_end_ns": 1000, "first_evidence_ns": 50},
            {"truth": "LOSS", "physical_loss_onset_ns": 100, "evidence_end_ns": 1000, "first_evidence_ns": None},
            {"truth": "LOSS", "physical_loss_onset_ns": 100, "evidence_end_ns": 150, "first_evidence_ns": None},
        ]
        result = _horizon_metric(rows, 100)
        self.assertEqual(result["on_time"], 1)
        self.assertEqual(result["early"], 1)
        self.assertEqual(result["missed"], 1)
        self.assertEqual(result["right_censored"], 1)

    def test_replenishment_design_is_shadow_only(self) -> None:
        payload = design()
        self.assertEqual(payload["total_rollouts"], 24)
        self.assertFalse(payload["candidate_controls_recovery"])
        self.assertFalse(payload["candidate_can_terminate"])
        self.assertEqual(payload["post_intervention_observation_s"], 1.5)

    def test_interface_shadow_distinguishes_guard_dispositions(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            payload = run_interface_shadow(Path(directory))
        cases = {case["name"]: case for case in payload["cases"]}
        self.assertEqual(cases["pending_then_confirm_100ms"]["outputs"][0]["guard_state"], "PENDING")
        self.assertEqual(cases["pending_then_confirm_100ms"]["outputs"][1]["guard_state"], "CONFIRMED")
        self.assertEqual(cases["contact_true_passthrough"]["outputs"][0]["guard_state"], "PASS_THROUGH")
        self.assertEqual(cases["contact_null_clear_no_fake_contact"]["outputs"][0]["guard_state"], "CLEAR")


if __name__ == "__main__":
    unittest.main()
