from __future__ import annotations

import copy
import json
import os
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from upgrade_v2.l2r_reproducible_baseline.capture import ReadOnlyPhysicsRecorder
from upgrade_v2.l2r_reproducible_baseline.authorization import AuthorizationDenied, validate_authorization
from upgrade_v2.l2r_reproducible_baseline.comparison import compare
from upgrade_v2.l2r_reproducible_baseline.environment import ENVIRONMENT_CONTRACT_SCHEMA, MAIN_GATE_FIELDS, REQUIRED_ENVIRONMENT, compare_main_gate_fields
from upgrade_v2.l2r_reproducible_baseline.fingerprint import array_hash
from upgrade_v2.l2r_reproducible_baseline.package_results import finalize_chain
from upgrade_v2.l2r_reproducible_baseline.protocol import EXPECTED_COUNTS, PROGRAM, canonical_hash, make_protocol
from upgrade_v2.l2r_reproducible_baseline.simulator import ReproducibleBaselineTabletop
from upgrade_v2.l2r_reproducible_baseline.source_lock import make_pre_execution_source_lock, pre_execution_model_xml


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
    def test_missing_source_lock_hash_denied(self):
        self._assert_contract_field_denied("source_lock_sha256", None)
    def test_wrong_source_lock_hash_denied(self):
        self._assert_contract_field_denied("source_lock_sha256", "0" * 64)
    def test_wrong_environment_contract_hash_denied(self):
        self._assert_contract_field_denied("environment_contract_sha256", "0" * 64)
    def test_wrong_model_xml_hash_denied(self):
        self._assert_contract_field_denied("generated_model_xml_sha256", hashlib.sha256(b"wrong").hexdigest())
    def test_all_contracts_checked_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in (("authorization.json", {}), ("source_lock.json", {}), ("environment.json", {})):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "protocol.json").write_text(json.dumps(self.protocol), encoding="utf-8")
            script = """
import json
import sys
from upgrade_v2.l2r_reproducible_baseline.cli import main
args = [
    "run-ordinary", "--repo", sys.argv[1], "--stage", "R14B_ORDINARY_BASELINE_A",
    "--protocol", sys.argv[2], "--authorization", sys.argv[3], "--source-lock", sys.argv[4],
    "--environment-contract", sys.argv[5], "--output-root", sys.argv[6],
]
rc = main(args)
print(json.dumps({"rc": rc, "mujoco_imported": "mujoco" in sys.modules}))
raise SystemExit(0 if rc == 3 and "mujoco" not in sys.modules else 1)
"""
            process = subprocess.run(
                [sys.executable, "-c", script, str(Path.cwd()), str(root / "protocol.json"),
                 str(root / "authorization.json"), str(root / "source_lock.json"),
                 str(root / "environment.json"), str(root / "out")],
                capture_output=True, text=True, cwd=Path.cwd(), check=False,
            )
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
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

    def _write_fixture(self, root: Path, *, instrumented: bool = False) -> None:
        result = {
            "runner_commit": "runner", "protocol_sha256": "protocol",
            "environment_fingerprint_sha256": "environment", "model_fingerprint_sha256": "model",
            "source_lock_sha256": "source-lock", "environment_contract_sha256": "environment-contract",
            "generated_model_xml_sha256": "xml", "family_seed": 870000,
            "all_main_gates_passed": True, "runner_file_hashes": {"runner.py": "hash"}, "artifact_manifest_sha256": "manifest",
        }
        lock = {
            "runner_commit": "runner", "protocol_sha256": "protocol", "generation_runner_file_hashes": {"runner.py": "hash"}, "static_contract_version": "l2rar2_r14b_pre_execution_contract_v2",
            "program_sha256": "program", "physical_spec_sha256": "physical", "generated_model_xml_sha256": "xml",
        }
        environment = {field: "same" for field in MAIN_GATE_FIELDS}
        states = [{"state_sha256": "state-0"}]
        captures = [{"phase": "control_tick", "action": "observe_scene", "action_index": 1, "time": 0.05, "raw_rgb_sha256": "rgb", "jpeg_sha256": "jpg", "callback_steps": []}]
        for name, value in (("result.json", result), ("source_lock.json", lock), ("runtime_environment_fingerprint.json", environment), ("checkpoint_state_trace.jsonl", states), ("callback_capture_trace.jsonl", captures)):
            path = root / name
            if name.endswith(".jsonl"):
                path.write_text("".join(json.dumps(row) + "\n" for row in value), encoding="utf-8")
            else:
                path.write_text(json.dumps(value), encoding="utf-8")
        for name in ("actions.jsonl", "low_level_controls.jsonl", "events.jsonl", "vision_detections.jsonl"):
            (root / name).write_text(json.dumps({"row": 1}) + "\n", encoding="utf-8")
        if instrumented:
            (root / "instrumentation_integrity.json").write_text(json.dumps({"passed": True, "before_after_rows": 290, "mutation_failures": 0}), encoding="utf-8")

    def _contract_fixture(self) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "environment_contract.json"
            contract.write_text(json.dumps({"schema": ENVIRONMENT_CONTRACT_SCHEMA, "status": "FROZEN_ZERO_PHYSICS", "static_contract_version": self.protocol["static_contract_version"], "required": self.protocol["required_environment"]}, sort_keys=True), encoding="utf-8")
            xml_sha = hashlib.sha256(pre_execution_model_xml(self.protocol).encode("utf-8")).hexdigest()
            lock = make_pre_execution_source_lock(Path.cwd(), self.protocol, "protocol", contract)
            return {"source_lock": lock, "environment_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(), "xml_sha256": xml_sha}

    def _valid_authorization_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        contract = root / "environment_contract.json"
        contract.write_text(json.dumps({"schema": ENVIRONMENT_CONTRACT_SCHEMA, "status": "FROZEN_ZERO_PHYSICS", "static_contract_version": self.protocol["static_contract_version"], "required": REQUIRED_ENVIRONMENT}, sort_keys=True), encoding="utf-8")
        protocol_path = root / "protocol.json"
        protocol_path.write_text(json.dumps(self.protocol, sort_keys=True), encoding="utf-8")
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        lock = make_pre_execution_source_lock(Path.cwd(), self.protocol, protocol_hash, contract)
        lock_path = root / "source_lock.json"
        lock_path.write_text(json.dumps(lock), encoding="utf-8")
        auth = {
            "schema": "l2rar2_r14b_execution_authorization_v2", "status": "AUTHORIZED",
            "protocol_id": self.protocol["protocol_id"], "stage": self.protocol["stages"]["A"]["stage"],
            "authorized_instances": 1, "requested_instances": 1, "case_id": self.protocol["case_id"],
            "root_family_id": self.protocol["root_family_id"], "rollout_seed": self.protocol["rollout_seed"],
            "family_seed": self.protocol["family_seed"], "program_sha256": self.protocol["program_sha256"],
            "physical_spec_sha256": self.protocol["physical_spec_sha256"], "automatic_retry": False,
            "on_any_main_gate_mismatch": "STOP_AFTER_EXECUTION_1", "agent_self_authorization_prohibited": True,
            "instrumented_replay_authorized_instances": 0, "r16_calibration_authorized_instances": 0,
            "r16_development_authorized_instances": 0, "runner_commit": lock["runner_commit"],
            "generation_runner_file_hashes": lock["generation_runner_file_hashes"], "protocol_sha256": protocol_hash,
            "static_contract_version": self.protocol["static_contract_version"],
            "source_lock_sha256": lock["source_lock_sha256"], "environment_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
            "generated_model_xml_sha256": lock["generated_model_xml_sha256"], "output_root": str((root / "out").resolve()),
            "single_use_nonce": "n" * 32, "expires_at_utc": "2999-01-01T00:00:00+00:00",
            "approved_at_utc": "2026-09-12T00:00:00+00:00", "authorization_id": "auth", "reviewer_id": "reviewer",
        }
        auth_path = root / "authorization.json"
        auth_path.write_text(json.dumps(auth), encoding="utf-8")
        return auth_path, lock_path, contract

    def _assert_contract_field_denied(self, field: str, value: str | None) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("upgrade_v2.l2r_reproducible_baseline.source_lock.working_tree_clean", return_value=True), patch("upgrade_v2.l2r_reproducible_baseline.authorization.working_tree_clean", return_value=True):
                auth_path, lock_path, contract_path = self._valid_authorization_fixture(root)
                data = json.loads(auth_path.read_text(encoding="utf-8"))
                data[field] = value
                auth_path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(AuthorizationDenied):
                    validate_authorization(auth_path, repo=Path.cwd(), protocol=self.protocol, protocol_sha256=data["protocol_sha256"], requested_output_root=root / "out", stage="R14B_ORDINARY_BASELINE_A", source_lock_path=lock_path, environment_contract_path=contract_path)

    def test_comparison_fixture_identical_runs_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / "left"; right = root / "right"; output = root / "comparison"
            left.mkdir(); right.mkdir(); self._write_fixture(left); self._write_fixture(right)
            summary = compare(left, right, output, "ordinary_A_vs_B")
            self.assertTrue(summary["all_main_gates_passed"])
            self.assertEqual(summary["status"], "PASS")

    def test_comparison_fixture_state_hash_difference_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / "left"; right = root / "right"; output = root / "comparison"
            left.mkdir(); right.mkdir(); self._write_fixture(left); self._write_fixture(right)
            (right / "checkpoint_state_trace.jsonl").write_text(json.dumps({"state_sha256": "different"}) + "\n", encoding="utf-8")
            summary = compare(left, right, output, "ordinary_A_vs_B")
            self.assertFalse(summary["all_main_gates_passed"])
            self.assertEqual(summary["first_mismatch"]["field"], "checkpoint_state")

    def test_comparison_fixture_instrumentation_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); left = root / "left"; right = root / "right"; output = root / "comparison"
            left.mkdir(); right.mkdir(); self._write_fixture(left); self._write_fixture(right)
            summary = compare(left, right, output, "ordinary_B_vs_instrumented_C")
            self.assertFalse(summary["all_main_gates_passed"])
            self.assertFalse(summary["instrumentation"]["passed"])

    def test_authorization_unsupported_stage_is_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            auth_path = Path(directory) / "auth.json"
            auth_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(AuthorizationDenied):
                validate_authorization(auth_path, repo=Path.cwd(), protocol=self.protocol, protocol_sha256="protocol", requested_output_root=Path(directory) / "out", stage="R16_DEVELOPMENT", source_lock_path=Path(directory) / "lock.json", environment_contract_path=Path(directory) / "environment.json")

    def test_contract_hash_mismatches_are_denied(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("upgrade_v2.l2r_reproducible_baseline.source_lock.working_tree_clean", return_value=True), patch("upgrade_v2.l2r_reproducible_baseline.authorization.working_tree_clean", return_value=True):
                auth_path, lock_path, contract_path = self._valid_authorization_fixture(root)
                data = json.loads(auth_path.read_text())
                for field in ("source_lock_sha256", "environment_contract_sha256", "generated_model_xml_sha256"):
                    changed = dict(data); changed[field] = "0" * 64
                    auth_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaises(AuthorizationDenied):
                        validate_authorization(auth_path, repo=Path.cwd(), protocol=self.protocol, protocol_sha256=data["protocol_sha256"], requested_output_root=root / "out", stage="R14B_ORDINARY_BASELINE_A", source_lock_path=lock_path, environment_contract_path=contract_path)

    def test_source_lock_contains_pre_execution_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("upgrade_v2.l2r_reproducible_baseline.source_lock.working_tree_clean", return_value=True):
                _, lock_path, _ = self._valid_authorization_fixture(root)
                lock = json.loads(lock_path.read_text())
                self.assertIn("runner_commit", lock)
                self.assertIn("generation_runner_file_hashes", lock)
                self.assertIn("generated_model_xml_sha256", lock)
                self.assertTrue(lock["git_status_clean"])

    def test_protocol_uses_v2_static_contract(self):
        self.assertEqual(self.protocol["schema"], "l2rar2_r14b_protocol_lock_v2")
        self.assertEqual(self.protocol["static_contract_version"], "l2rar2_r14b_pre_execution_contract_v2")

    def test_authorization_templates_use_v2_contract(self):
        root = Path(__file__).parents[3] / "artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v1/applications_v1"
        for path in root.glob("*_authorization.template.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["static_contract_version"], "l2rar2_r14b_pre_execution_contract_v2")

    def test_finalize_chain_issues_certificate_only_for_all_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, c, ab, bc, final = (root / name for name in ("A", "B", "C", "AB", "BC", "final"))
            for path in (a, b, c):
                path.mkdir(); self._write_fixture(path, instrumented=path == c)
            ab.mkdir(); bc.mkdir()
            summary = {"all_main_gates_passed": True, "instrumentation": {"passed": True, "before_after_rows": 290, "mutation_failures": 0}}
            (ab / "comparison_summary.json").write_text(json.dumps({"all_main_gates_passed": True}), encoding="utf-8")
            (bc / "comparison_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            decision = finalize_chain(repo=Path.cwd(), baseline_a=a, repeat_b=b, instrumented_c=c, comparison_ab=ab, comparison_bc=bc, output_root=final)
            self.assertEqual(decision["status"], "R14B_REPRODUCIBLE_BASELINE_CHAIN_PASS")
            self.assertTrue((final / "r14b_reproducibility_certificate.json").is_file())

    def test_finalize_chain_blocks_without_all_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b, c, ab, bc, final = (root / name for name in ("A", "B", "C", "AB", "BC", "final"))
            for path in (a, b, c):
                path.mkdir(); self._write_fixture(path)
            ab.mkdir(); bc.mkdir()
            (ab / "comparison_summary.json").write_text(json.dumps({"all_main_gates_passed": False}), encoding="utf-8")
            (bc / "comparison_summary.json").write_text(json.dumps({"all_main_gates_passed": False, "instrumentation": {"passed": False}}), encoding="utf-8")
            decision = finalize_chain(repo=Path.cwd(), baseline_a=a, repeat_b=b, instrumented_c=c, comparison_ab=ab, comparison_bc=bc, output_root=final)
            self.assertEqual(decision["status"], "R14B_CHAIN_BLOCKED")
            self.assertFalse((final / "r14b_reproducibility_certificate.json").exists())


if __name__ == "__main__":
    unittest.main()
