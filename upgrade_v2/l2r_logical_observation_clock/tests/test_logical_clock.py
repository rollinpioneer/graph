from __future__ import annotations

import unittest
from pathlib import Path

from upgrade_v2.l2r_logical_observation_clock import CANDIDATE_ID
from upgrade_v2.l2r_logical_observation_clock.guard import (
    CONFIRMED_REASON, INTERCEPT_REASON, LogicalObservationClockGuard,
)
from upgrade_v2.l2r_logical_observation_clock.r19_design import CASES, FAMILIES
from upgrade_v2.l2r_logical_observation_clock.package_results import summarize

ROOT = Path(__file__).resolve().parents[2]


def obs(time=1.0, order=10, contact=False, **extra):
    return {"time": time, "capture_order": order, "attempt_id": 1, "contact_present": contact,
            "gripper_command": "closed", "requested_effect": "HOLD_OBJECT", "context_valid": True,
            "attempt_end": False, "attempt_end_reason": None, **extra}


def proposal(action="recover_object", reason=INTERCEPT_REASON, time=1.0, order=10):
    return {"selected_action": action, "reason_code": reason, "time": time, "capture_order": order}


class GuardTests(unittest.TestCase):
    def test_candidate_id_distinct_from_r18(self): self.assertEqual(CANDIDATE_ID, "O_C3_CLP2_LOGICAL_CLOCK")
    def test_first_loss_pending(self): self.assertTrue(LogicalObservationClockGuard().step(obs(), proposal())["contact_loss_pending"])
    def test_first_loss_emits_none(self): self.assertEqual(LogicalObservationClockGuard().step(obs(), proposal())["selected_action"], "none")
    def test_same_physical_time_next_order_confirms(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        row=guard.step(obs(order=11), proposal("none","no_loss",order=11))
        self.assertEqual((row["selected_action"],row["reason_code"]),("recover_object",CONFIRMED_REASON))
    def test_action_uses_second_capture_order(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertEqual(guard.step(obs(order=11),proposal("none","x"))["capture_order"],11)
    def test_action_time_not_backdated(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(1.0,10),proposal())
        self.assertEqual(guard.step(obs(1.05,11),proposal("none","x"))["time"],1.05)
    def test_duplicate_capture_order_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(order=10),proposal("none","x"))["guard_state"],"CONFIRMED")
    def test_reversed_capture_order_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(order=9),proposal("none","x"))["guard_state"],"CONFIRMED")
    def test_decreasing_physical_time_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(.99,11),proposal("none","x"))["guard_state"],"CONFIRMED")
    def test_gap_above_point1_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(1.101,11),proposal("none","x"))["guard_state"],"CONFIRMED")
    def test_contact_restore_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(order=11,contact=True),proposal("none","x"))["guard_state"],"CONFIRMED")
    def test_release_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(order=11,gripper_command="open"),proposal())["guard_state"],"CONFIRMED")
    def test_attempt_change_clears(self):
        guard=LogicalObservationClockGuard(); guard.step(obs(),proposal())
        self.assertNotEqual(guard.step(obs(order=11,attempt_id=2),proposal("none","x"))["guard_state"],"CONFIRMED")
    def test_other_paths_pass_through(self): self.assertEqual(LogicalObservationClockGuard().step(obs(contact=True),proposal("retry_grasp","retry"))["selected_action"],"retry_grasp")


class ContractTests(unittest.TestCase):
    def test_no_parameter_search(self): self.assertNotIn("grid_search", (ROOT/"l2r_logical_observation_clock/guard.py").read_text())
    def test_prefix_projection_contains_time_and_order(self):
        source=(ROOT/"l2r_logical_observation_clock/replay.py").read_text(); self.assertIn('row.get("time")',source); self.assertIn('row.get("capture_order")',source)
    def test_raw_parity_includes_reason(self): self.assertIn("historical_reason_code",(ROOT/"l2r_logical_observation_clock/replay.py").read_text())
    def test_r19_has_new_families(self): self.assertEqual(len(FAMILIES),6)
    def test_r19_has_72_design_rollouts(self): self.assertEqual(len(FAMILIES)*len(CASES),72)
    def test_r19_seeds_do_not_reuse_r17_r18(self): self.assertTrue(all(f["family_seed"]>=886000 for f in FAMILIES))
    def test_same_time_stress_case_registered(self): self.assertIn("T4_transport_same_time_contact_false_then_true",CASES)
    def test_design_gate_is_fail_closed(self): self.assertIn("R19_DEVELOPMENT_GATE_REQUIRED",(ROOT/"l2r_logical_observation_clock/r19_design.py").read_text())
    def test_no_physical_collector_in_module(self): self.assertFalse((ROOT/"l2r_logical_observation_clock/collector.py").exists())
    def test_summary_uses_clock_forensics_field(self):
        source=(ROOT/"l2r_logical_observation_clock/package_results.py").read_text()
        self.assertIn('same_physical_time_groups',source); self.assertNotIn('same_timestamp_groups',source)
    def test_source_audit_requires_zero_physics(self): self.assertIn('"physical_executions": 0',(ROOT/"l2r_logical_observation_clock/audits.py").read_text())


if __name__ == "__main__": unittest.main()
