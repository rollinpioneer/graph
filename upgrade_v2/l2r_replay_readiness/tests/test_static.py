from __future__ import annotations

import importlib
import csv
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


class R13StaticTests(unittest.TestCase):
    def test_module_is_stdlib_only_and_does_not_import_physics(self) -> None:
        before = set(sys.modules)
        importlib.import_module("upgrade_v2.l2r_replay_readiness.cli")
        new = set(sys.modules) - before
        self.assertNotIn("mujoco", new)
        self.assertFalse(any(name.startswith("upgrade_v2.visual_refine_l2") for name in new))

    def test_unknowns_remain_explicit(self) -> None:
        from upgrade_v2.l2r_replay_readiness.cli import build_readiness
        self.assertEqual(build_readiness.__name__, "build_readiness")

    def test_crosswalk_statuses_are_not_false_equivalence(self) -> None:
        allowed = {"VERIFIED_ONE_TO_ONE", "SOURCE_ORDER_ONLY", "SAME_TIMESTAMP_ORDER_UNKNOWN", "NO_SAVED_COUNTERPART", "NOT_COMPARABLE"}
        self.assertIn("NOT_COMPARABLE", allowed)
        self.assertNotEqual("NOT_COMPARABLE", "VERIFIED_ONE_TO_ONE")

    def test_generation_provenance_recovers_round9_hashes(self) -> None:
        from upgrade_v2.l2r_replay_readiness.cli import build_generation_provenance

        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            generation = build_generation_provenance(repo, Path(directory))
        self.assertEqual(generation["generation_lock"]["status"], "VERIFIED_HASHED_ARTIFACT")
        self.assertEqual(generation["round9_validation_manifest"]["status"], "VERIFIED_HASHED_ARTIFACT")
        self.assertEqual(generation["rollout_manifest"]["status"], "VERIFIED_HASHED_ARTIFACT")
        self.assertEqual(
            generation["rollout_manifest"]["sha256"],
            generation["round9_validation_manifest"]["recorded_rollout_manifest_sha256"],
        )
        self.assertEqual(generation["collection_contract"]["repair_version"], "l2rar2_attach_relpose_v1")
        self.assertEqual(generation["collection_contract"]["collection_version"], "l2rar2_attach_relpose_collection_v1")

    def test_claim_review_keeps_human_decision_empty(self) -> None:
        from upgrade_v2.l2r_replay_readiness.cli import build_claim_review

        repo = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "claim_review.csv"
            build_claim_review(repo, output)
            with output.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertIn("agent_review_recommendation", rows[0])
        self.assertIn("human_decision", rows[0])
        self.assertTrue(all(row["agent_review_recommendation"] == "ACCEPT_QUARANTINE" for row in rows))
        self.assertTrue(all(row["human_decision"] == "" for row in rows))
        self.assertTrue(all(row["human_reviewer_id"] == "" for row in rows))
        self.assertTrue(all(row["human_reviewed_at_utc"] == "" for row in rows))

    def test_updated_call_signatures_are_connected(self) -> None:
        module = importlib.import_module("upgrade_v2.l2r_replay_readiness.cli")
        self.assertIn("generation", inspect.signature(module.build_external).parameters)
        self.assertIn("generation", inspect.signature(module.build_report).parameters)
        self.assertIn("generation", inspect.signature(module.build_readiness).parameters)


if __name__ == "__main__":
    unittest.main()
