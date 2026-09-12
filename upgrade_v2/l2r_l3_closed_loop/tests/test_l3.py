from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_l3_closed_loop import BASE_COMMIT, CLP3_ID, METHODS
from upgrade_v2.l2r_l3_closed_loop.protocol import FROZEN_BLOBS, protocol_lock
from upgrade_v2.l2r_l3_closed_loop.registry import CASES, FAMILIES, registry
from upgrade_v2.l2r_l3_closed_loop.static_lock import validate_frozen

ROOT = Path(__file__).resolve().parents[3]


class L3Tests(unittest.TestCase):
    def test_frozen_base(self): self.assertEqual(BASE_COMMIT, "cb6ed23f88e44336f17197c300bc72c7fe19df49")
    def test_candidate_frozen(self): self.assertEqual(CLP3_ID, "O_C3_CLP3_CANONICAL_TIME")
    def test_three_arms(self): self.assertEqual(METHODS, (CLP3_ID, "O_C3_RAW", "RECOVERY_DISABLED"))
    def test_registry_is_60(self): self.assertEqual(len(CASES) * len(FAMILIES) * len(METHODS), 60)
    def test_new_seeds(self): self.assertEqual([row[1] for row in FAMILIES], [894000, 894001, 894002, 894003])
    def test_paired_seed_design(self): self.assertEqual(registry()["paired_groups"], 20)
    def test_failure_taxonomy(self):
        self.assertEqual(protocol_lock()["failure_stages"], ["TRIGGERING_ERROR", "RELOCATION_ERROR", "REGRASP_ERROR", "TASK_RECOVERY_ERROR"])
    def test_no_parameter_search(self): self.assertFalse(protocol_lock()["controller_contract"]["parameter_search"])
    def test_clp3_blob(self): self.assertEqual(FROZEN_BLOBS["clp3_guard"], "03b428e2f0fdd215ce31c4149674781bf10129a2")
    def test_frozen_parent_audit(self): self.assertTrue(validate_frozen(ROOT)["passed"])


if __name__ == "__main__": unittest.main()
