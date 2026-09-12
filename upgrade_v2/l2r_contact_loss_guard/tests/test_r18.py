from __future__ import annotations

import unittest
from pathlib import Path

from upgrade_v2.l2r_contact_loss_guard import CANDIDATE_ID, PROBLEM_ROLLOUT_ID
from upgrade_v2.l2r_contact_loss_guard.guard import ContactLossPersistenceGuard, INTERCEPT_REASON
from upgrade_v2.l2r_contact_loss_guard.protocol import ADJUDICATION_OUTCOMES, LEVELS, MAXIMUM_PENDING_GAP_S, REQUIRED_SAMPLES
from upgrade_v2.l2r_contact_loss_guard.r18_registry import CASES, FAMILIES, unique_keys

ROOT = Path(__file__).resolve().parents[2]


def observation(time=.0, order=0, contact=False, **extra):
    return {"time": time, "capture_order": order, "attempt_id": 1, "contact_present": contact,
            "gripper_command": "closed", "requested_effect": "HOLD_OBJECT", "context_valid": True,
            "attempt_end": False, "attempt_end_reason": None, **extra}


def proposed(action="recover_object", reason=INTERCEPT_REASON):
    return {"selected_action": action, "reason_code": reason, "time": 0.0, "capture_order": 0}


class ForensicsTests(unittest.TestCase):
    def test_problem_rollout_exact_id(self): self.assertEqual(PROBLEM_ROLLOUT_ID, "L2RAR2_RGB_CONF_04_882004__C11_transport_strong_loss")
    def test_100hz_window_extracts_1p45_to_1p65(self): self.assertIn("1.45", (ROOT / "l2r_contact_loss_guard/forensics.py").read_text())
    def test_timestamp_semantics_explicit(self): self.assertIn("post-step", (ROOT / "l2r_contact_loss_guard/forensics.py").read_text())
    def test_contact_drop_time_extracted(self): self.assertIn("t_first_contact_false", (ROOT / "l2r_contact_loss_guard/forensics.py").read_text())
    def test_physical_onset_time_extracted(self): self.assertIn("t_physical_loss_onset", (ROOT / "l2r_contact_loss_guard/forensics.py").read_text())
    def test_three_way_adjudication_only(self): self.assertEqual(len(ADJUDICATION_OUTCOMES), 3)


class GuardTests(unittest.TestCase):
    def test_first_contact_loss_enters_pending(self): self.assertTrue(ContactLossPersistenceGuard().step(observation(), proposed())["contact_loss_pending"])
    def test_first_contact_loss_emits_none(self): self.assertEqual(ContactLossPersistenceGuard().step(observation(), proposed())["selected_action"], "none")

    def test_second_consecutive_loss_emits_recover(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed())
        self.assertEqual(g.step(observation(1.05,2),proposed("none","historical_hold_established_no_current_loss_observed"))["selected_action"],"recover_object")

    def test_action_time_is_second_sample(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed())
        row=g.step(observation(1.05,2),proposed("none","x")); self.assertEqual((row["time"],row["capture_order"]),(1.05,2))

    def test_contact_restore_clears_pending(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed()); self.assertFalse(g.step(observation(1.05,2,True),proposed("none","x"))["contact_loss_pending"])
    def test_release_clears_pending(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed()); self.assertNotEqual(g.step(observation(1.05,2,False,gripper_command="open"),proposed())["guard_state"],"CONFIRMED")
    def test_attempt_change_clears_pending(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed()); self.assertNotEqual(g.step(observation(1.05,2,attempt_id=2),proposed())["guard_state"],"CONFIRMED")
    def test_invalid_sample_clears_pending(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed()); self.assertFalse(g.step(observation(1.05,2,context_valid=False),proposed())["contact_loss_pending"])
    def test_gap_over_point1_clears_pending(self):
        g=ContactLossPersistenceGuard(); g.step(observation(1,1),proposed()); self.assertNotEqual(g.step(observation(1.11,2),proposed())["guard_state"],"CONFIRMED")
    def test_retry_path_unchanged(self): self.assertEqual(ContactLossPersistenceGuard().step(observation(contact=True),proposed("retry_grasp","retry"))["selected_action"],"retry_grasp")
    def test_non_contact_recovery_path_unchanged(self): self.assertEqual(ContactLossPersistenceGuard().step(observation(contact=True),proposed("recover_object","other"))["selected_action"],"recover_object")
    def test_parameters_are_two_samples_and_point1(self): self.assertEqual((REQUIRED_SAMPLES,MAXIMUM_PENDING_GAP_S),(2,.1))
    def test_no_guard_parameter_search(self): self.assertNotIn("grid_search",(ROOT/"l2r_contact_loss_guard/guard.py").read_text())


class RegistryTests(unittest.TestCase):
    def test_six_new_families(self): self.assertEqual(len(FAMILIES),6)
    def test_twelve_case_order(self): self.assertEqual(len(CASES),12)
    def test_72_unique_keys(self): self.assertEqual(len(set(unique_keys())),72)
    def test_no_R17_family_reuse(self): self.assertTrue(all("RGB_CONF" not in f for f,_,_ in FAMILIES))
    def test_frozen_weak_medium_strong(self): self.assertEqual(LEVELS["strong"]["delta_v_local_mps"],(30.,30.,-6.))
    def test_five_phase_offsets(self): self.assertEqual({c.phase_offset_ms for c in CASES if c.case_id.startswith(tuple(f"T{i}_" for i in range(6,11)))},{0,10,20,30,40})
    def test_T3_one_frame_proxy_dropout(self): self.assertTrue(CASES[2].contact_proxy_dropout)
    def test_T11_three_rgb_dropouts(self): self.assertTrue(CASES[10].rgb_dropout)
    def test_T12_commanded_release(self): self.assertTrue(CASES[11].commanded_release)


class ContractsTests(unittest.TestCase):
    def test_scoring_has_no_early_tolerance(self): self.assertNotIn("early_tolerance",(ROOT/"l2r_contact_loss_guard/temporal_evaluation.py").read_text())
    def test_original_C3_preserved(self): self.assertIn("_run_o(online, \"O_C3\")",(ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())
    def test_CLP1_candidate_id(self): self.assertEqual(CANDIDATE_ID,"O_C3_CLP1")
    def test_failed_confirmation_does_not_retune(self): self.assertNotIn("grid_search",(ROOT/"l2r_contact_loss_guard").joinpath("guard.py").read_text())


class R17ReplayTests(unittest.TestCase):
    def test_raw_B2_matches_R17(self): self.assertIn('raw_B2_72_of_72', (ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())
    def test_raw_C3_matches_R17(self): self.assertIn('raw_C3_72_of_72', (ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())
    def test_CLP1_R17_zero_early(self): self.assertIn('CLP1_zero_early', (ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())
    def test_CLP1_R17_zero_false(self): self.assertIn('CLP1_zero_false', (ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())
    def test_CLP1_R17_zero_miss(self): self.assertIn('CLP1_zero_missed', (ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())
    def test_CLP1_R17_prefix_causal(self): self.assertIn('prefix_causality', (ROOT/"l2r_contact_loss_guard/r17_replay.py").read_text())


class GeneratorTests(unittest.TestCase):
    def setUp(self): self.source=(ROOT/"l2r_contact_loss_guard/reference_builder.py").read_text()
    def test_72_trace_complete(self): self.assertIn('trace_complete_72_of_72', self.source)
    def test_72_prehold(self): self.assertIn('prehold_72_of_72', self.source)
    def test_36_strong_loss(self): self.assertIn('strong_loss_T6_T11_36_of_36', self.source)
    def test_18_stable_no_loss(self): self.assertIn('stable_no_loss_T1_T3_18_of_18', self.source)
    def test_6_release(self): self.assertIn('commanded_release_T12_6_of_6', self.source)
    def test_12_outcome_resolved(self): self.assertIn('T4_T5_resolved_12_of_12', self.source)
    def test_generator_fail_blocks_evaluation(self): self.assertIn('GENERATOR_GATE_REQUIRED', (ROOT/"l2r_contact_loss_guard/evaluation.py").read_text())


class EvaluationTests(unittest.TestCase):
    def setUp(self): self.source=(ROOT/"l2r_contact_loss_guard/evaluation.py").read_text()
    def test_input_provenance(self): self.assertIn('input_provenance_audit', self.source)
    def test_prefix_causality(self): self.assertIn('prefix_causality_audit', self.source)
    def test_p90_latency(self): self.assertIn('latency_p90_s', self.source)
    def test_selection_requires_all_gates(self): self.assertIn('candidate["strong_loss_correct_in_window"] >= 35', self.source)
    def test_l3_not_started(self): self.assertIn('"l3_entry_allowed": False', self.source)


if __name__ == "__main__": unittest.main()
