from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

from tools.validate_A2_authorization_template import validate_template


ROOT = Path(__file__).parents[3]
APP = ROOT / "artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v2/applications_v1/baseline_A2_authorization.template.json"
PROTOCOL = ROOT / "artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v2/static_v1/protocol_draft.json"
LOCK = ROOT / "artifacts/pathgraph_sarm/upgrade_v2/reproducible_baseline_l2rar2_r14b_v2/static_v1/source_lock.json"


class A2AuthorizationTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = json.loads(APP.read_text(encoding="utf-8"))
        cls.protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def errors_for(self, value):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authorization.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return validate_template(path, PROTOCOL, LOCK)

    def test_corrected_template_accepted(self):
        self.assertEqual(validate_template(APP, PROTOCOL, LOCK), [])

    def test_all_static_fields_match_protocol_and_lock(self):
        self.assertEqual(self.template["protocol_id"], self.protocol["protocol_id"])
        self.assertEqual(self.template["program_sha256"], self.protocol["program_sha256"])
        self.assertEqual(self.template["physical_spec_sha256"], self.protocol["physical_spec_sha256"])
        for field in ("protocol_sha256", "source_lock_sha256", "environment_contract_sha256", "generated_model_xml_sha256"):
            self.assertEqual(self.template[field], self.lock[field])

    def test_generated_model_xml_hash_nonempty(self):
        self.assertEqual(len(self.template["generated_model_xml_sha256"]), 64)
        self.assertNotEqual(self.template["generated_model_xml_sha256"], "0" * 64)

    def test_runner_hashes_complete(self):
        self.assertEqual(self.template["generation_runner_file_hashes"], self.lock["generation_runner_file_hashes"])
        self.assertEqual(len(self.template["generation_runner_file_hashes"]), 22)

    def test_B_C_R16_remain_zero(self):
        for field in ("authorized_instances", "instrumented_replay_authorized_instances", "r16_calibration_authorized_instances", "r16_development_authorized_instances"):
            self.assertEqual(self.template[field], 0)

    def test_human_fields_remain_null(self):
        for field in ("authorization_id", "reviewer_id", "approved_at_utc", "expires_at_utc", "single_use_nonce", "output_root"):
            self.assertIsNone(self.template[field])

    def test_old_v1_template_rejected(self):
        value = copy.deepcopy(self.template)
        value["schema"] = "l2rar2_r14b_execution_authorization_v1"
        self.assertIn("mismatch:schema", self.errors_for(value))
        self.assertIn("old_v1_template", self.errors_for(value))

    def test_missing_static_field_rejected(self):
        value = copy.deepcopy(self.template)
        del value["source_lock_sha256"]
        self.assertIn("missing:source_lock_sha256", self.errors_for(value))

    def test_nonzero_authorization_rejected(self):
        value = copy.deepcopy(self.template)
        value["authorized_instances"] = 1
        self.assertIn("mismatch:authorized_instances", self.errors_for(value))

    def test_nonce_and_output_root_rejected_if_filled(self):
        value = copy.deepcopy(self.template)
        value["single_use_nonce"] = "n" * 32
        value["output_root"] = "/tmp/unauthorized"
        errors = self.errors_for(value)
        self.assertIn("must_be_null:single_use_nonce", errors)
        self.assertIn("must_be_null:output_root", errors)

    def test_validator_is_static_only(self):
        self.assertEqual(validate_template(APP, PROTOCOL, LOCK), [])
        self.assertNotIn("mujoco", sys.modules)


if __name__ == "__main__":
    unittest.main()
