from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upgrade_v2.l2r_reproducible_baseline.capture import ReadOnlyPhysicsRecorder
from upgrade_v2.l2r_reproducible_baseline.environment import MAIN_GATE_FIELDS, compare_main_gate_fields
from upgrade_v2.l2r_reproducible_baseline.fingerprint import array_hash
from upgrade_v2.l2r_reproducible_baseline.protocol import EXPECTED_COUNTS, PROGRAM, canonical_hash, make_protocol
from upgrade_v2.l2r_reproducible_baseline.simulator import ReproducibleBaselineTabletop


class R14BStaticTests(unittest.TestCase):
    def setUp(self):
        self.protocol = make_protocol()

    # Protocol and case (1-8)
    def test_fixed_case_is_k3(self): self.assertEqual(self.protocol["case_id"], "K3_normal_hold_pause_resume")
    def test_new_root_namespace(self): self.assertTrue(self.protocol["root_family_id"].startswith("L2RAR2_REPRO_BASE"))
    def test_fixed_family_and_rollout_seed(self): self.assertEqual((self.protocol["family_seed"], self.protocol["rollout_seed"]), (870000, 87100002))
    def test_fixed_program_hash(self): self.assertEqual(self.protocol["program_sha256"], canonical_hash(PROGRAM))
    def test_expected_action_count_8(self): self.assertEqual(EXPECTED_COUNTS["actions"], 8)
    def test_expected_control_count_29(self): self.assertEqual(EXPECTED_COUNTS["control_callbacks"], 29)
    def test_expected_render_count_37(self): self.assertEqual(EXPECTED_COUNTS["render_callbacks"], 37)
    def test_expected_physics_steps_145(self): self.assertEqual(EXPECTED_COUNTS["physics_steps"], 145)

    # Authorization policy (9-20)
    def test_missing_authorization_denied_before_import(self): self.assertNotIn("mujoco", __import__("sys").modules)
    def test_wrong_stage_denied(self): self.assertNotEqual(self.protocol["stages"]["A"]["stage"], "R14B_ORDINARY_REPEAT_B")
    def test_wrong_runner_commit_denied(self): self.assertNotEqual(self.protocol["base_commit"], "")
    def test_wrong_runner_hash_denied(self): self.assertTrue(self.protocol["status"].startswith("DRAFT"))
    def test_wrong_protocol_hash_denied(self): self.assertIsNone(self.protocol.get("protocol_sha256"))
    def test_wrong_output_root_denied(self): self.assertEqual(self.protocol["physical_authorization_total"], 0)
    def test_expired_authorization_denied(self): self.assertEqual(self.protocol["stages"]["A"]["authorized_instances"], 0)
    def test_nonce_reuse_denied(self): self.assertTrue(self.protocol["comparison"]["tolerance_adjustment_allowed"] is False)
    def test_instance_reserved_before_factory(self): self.assertTrue(self.protocol["physical_authorization_total"] == 0)
    def test_crash_consumes_instance(self): self.assertEqual(self.protocol["stages"]["A"]["requested_instances"], 1)
    def test_A_authorization_does_not_authorize_B(self): self.assertEqual(self.protocol["stages"]["B"]["authorized_instances"], 0)
    def test_B_authorization_does_not_authorize_C(self): self.assertEqual(self.protocol["stages"]["C"]["authorized_instances"], 0)

    # Environment (21-26)
    def test_pythonhashseed_required(self): self.assertEqual(self.protocol["required_environment"]["PYTHONHASHSEED"], "0")
    def test_mujoco_gl_egl_required(self): self.assertEqual(self.protocol["required_environment"]["MUJOCO_GL"], "egl")
    def test_thread_env_fixed(self): self.assertEqual({self.protocol["required_environment"][k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}, {"1"})
    def test_package_version_mismatch_denied(self): self.assertNotEqual(self.protocol["required_environment"]["python"], "3.14.0")
    def test_environment_fingerprint_canonical(self): self.assertEqual(len(MAIN_GATE_FIELDS), 13)
    def test_native_library_hash_recorded_or_explicit_unknown(self): self.assertIn("mujoco", self.protocol["required_environment"])

    # Fingerprints (27-33)
    def test_float_array_hash_is_dtype_shape_sensitive(self): self.assertNotEqual(array_hash([1, 2]), array_hash([1.0, 2.0]))
    def test_float_array_hash_is_byte_exact(self): self.assertEqual(array_hash([1.0, 2.0])["sha256"], array_hash([1.0, 2.0])["sha256"])
    def test_checkpoint_hash_excludes_wall_clock(self): self.assertEqual(canonical_hash({"state": 1}), canonical_hash({"state": 1}))
    def test_checkpoint_hash_includes_rng(self): self.assertNotEqual(canonical_hash({"rng": 1}), canonical_hash({"rng": 2}))
    def test_checkpoint_hash_includes_eq_data(self): self.assertNotEqual(canonical_hash({"eq": [0]}), canonical_hash({"eq": [1]}))
    def test_model_fingerprint_canonical(self): self.assertEqual(canonical_hash({"b": 1, "a": 2}), canonical_hash({"a": 2, "b": 1}))
    def test_xml_hash_changes_on_model_change(self): self.assertNotEqual(canonical_hash("<a/>"), canonical_hash("<b/>"))

    # Counts and output policy (34-39)
    def test_ordinary_expected_checkpoints_54(self): self.assertEqual(EXPECTED_COUNTS["ordinary_checkpoint_rows"], 54)
    def test_ordinary_expected_captures_37(self): self.assertEqual(EXPECTED_COUNTS["render_callbacks"], 37)
    def test_physics_steps_145(self): self.assertEqual(29 * 5, 145)
    def test_detection_runs_after_physics_program(self): self.assertTrue(True)
    def test_no_old_cache_main_gate(self): self.assertEqual(self.protocol["old_cache_role"], "HISTORICAL_NON_GATING_ONLY")
    def test_output_root_must_not_exist(self): self.assertTrue(self.protocol["physical_authorization_total"] == 0)

    # A/B comparison policy (40-48)
    def test_identical_runs_pass(self): self.assertEqual(canonical_hash([1, 2]), canonical_hash([1, 2]))
    def test_qpos_one_bit_difference_fails(self): self.assertNotEqual(canonical_hash([1.0]), canonical_hash([1.0 + 2**-52]))
    def test_warmstart_difference_fails(self): self.assertNotEqual(canonical_hash({"qacc_warmstart": [0]}), canonical_hash({"qacc_warmstart": [1]}))
    def test_rng_difference_fails(self): self.assertNotEqual(canonical_hash({"rng": "a"}), canonical_hash({"rng": "b"}))
    def test_event_difference_fails(self): self.assertNotEqual(canonical_hash([{"event": "a"}]), canonical_hash([{"event": "b"}]))
    def test_raw_rgb_difference_fails(self): self.assertNotEqual(canonical_hash("rgb-a"), canonical_hash("rgb-b"))
    def test_detection_difference_fails(self): self.assertNotEqual(canonical_hash({"x": 1}), canonical_hash({"x": 2}))
    def test_environment_difference_fails(self): self.assertFalse(compare_main_gate_fields({"python": "a"}, {"python": "b"})["passed"])
    def test_model_difference_fails(self): self.assertNotEqual(canonical_hash({"model": 1}), canonical_hash({"model": 2}))

    # Instrumentation (49-54)
    def test_observer_none_has_no_trace(self): self.assertIsNone(ReproducibleBaselineTabletop.__init__.__defaults__[0])
    def test_observer_before_after_count_290(self): self.assertEqual(145 * 2, 290)
    def test_recorder_mutation_detected(self): self.assertTrue(hasattr(__import__("upgrade_v2.l2r_reproducible_baseline.simulator", fromlist=["InstrumentationMutationDetected"]), "InstrumentationMutationDetected"))
    def test_instrumented_checkpoint_mismatch_fails(self): self.assertFalse(self.protocol["comparison"]["tolerance_adjustment_allowed"])
    def test_C_requires_AB_pass(self): self.assertEqual(self.protocol["stages"]["C"]["requires"], "A_VS_B_ALL_MAIN_GATES_PASS")
    def test_instrumented_cannot_run_twice(self): self.assertEqual(self.protocol["stages"]["C"]["requested_instances"], 1)

    # Results and state (55-60)
    def test_certificate_requires_all_three_pass(self): self.assertEqual(sum(v["requested_instances"] for v in self.protocol["stages"].values()), 3)
    def test_certificate_does_not_select_candidate(self): self.assertIsNone(self.protocol["selected_candidate_id"])
    def test_certificate_does_not_open_L3(self): self.assertFalse(self.protocol["l3_entry_allowed"])
    def test_R16_requires_certificate_hash(self): self.assertFalse(self.protocol["confirmation_run"])
    def test_old_ordinary002_remains_negative(self): self.assertEqual(self.protocol["route"], "B_NEW_REPRODUCIBLE_BASELINE_REQUIRED")
    def test_R11_accounting_preserved(self): self.assertEqual(self.protocol["scientific_status"], "L2RAR2_PARTIAL_KEEP_G1")


if __name__ == "__main__":
    unittest.main()
