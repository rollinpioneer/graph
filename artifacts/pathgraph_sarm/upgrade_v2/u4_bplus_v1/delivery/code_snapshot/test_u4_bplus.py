from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from .diagnostics import decide
from .io import sha256_file, verify_locked_inputs, write_json, write_jsonl
from .occurrence import episode_occurrences
from .simulator_adapter import family_specs, validate_snapshot_replay


class U4BContractsTest(unittest.TestCase):
    def test_snapshot_replay(self) -> None:
        result = validate_snapshot_replay(family_specs(1, 840100)[0])
        self.assertEqual(result["status"], "PASS")

    def test_route_observability_limit_uses_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "metrics.json", {
                "informative_family_count": 12, "high_impact_events": 20,
                "observability_insufficient": True,
            })
            write_jsonl(root / "cases.jsonl", [])
            result = decide(root / "metrics.json", root / "cases.jsonl", True, root / "route.json", root / "route.md")
            self.assertEqual(result["route"], "CONTINUE_WITH_FALLBACK")

    def test_route_semantic_ambiguity_requires_authorization(self) -> None:
        metrics = {
            "informative_family_count": 12, "high_impact_events": 20,
            "observability_insufficient": False,
            "unresolved_high_impact_rate": .5,
            "recovery_recall_gain_ref_minus_auto": 0.0,
            "semantic_unresolved_after_reference": .5,
            "concrete_ambiguous_groups": 2,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); write_json(root / "metrics.json", metrics); write_jsonl(root / "cases.jsonl", [{"case_id": "c1"}])
            denied = decide(root / "metrics.json", root / "cases.jsonl", False, root / "denied.json", root / "denied.md")
            allowed = decide(root / "metrics.json", root / "cases.jsonl", True, root / "allowed.json", root / "allowed.md")
            self.assertEqual(denied["route"], "CONTINUE_WITH_FALLBACK")
            self.assertEqual(allowed["route"], "RUN_U3B")

    def test_episode_ids_are_distinct(self) -> None:
        spec = family_specs(1, 840100)[0]
        from .simulator_adapter import collect_episode
        first = collect_episode(spec, 1, False)
        second = collect_episode(spec, 2, False)
        self.assertNotEqual(first["episode_id"], second["episode_id"])

    def test_confirmation_lock_rejects_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "family_lock.json"
            frozen.write_text("locked\n", encoding="utf-8")
            lock = root / "final_lock.json"
            write_json(lock, {"confirmation_locked": True, "input_hashes": {str(frozen): sha256_file(frozen)}})
            self.assertEqual(verify_locked_inputs(lock, frozen)["status"], "PASS")
            frozen.write_text("changed\n", encoding="utf-8")
            self.assertEqual(verify_locked_inputs(lock, frozen)["status"], "BLOCKED")

    def test_contact_loss_without_done_is_not_terminal(self) -> None:
        observation = [0.0] * 17
        episode = {
            "episode_id": "episode-a", "root_family_id": "family-a",
            "family": {"scenario": "analysis-only"}, "observations": [observation, observation],
            "events": [{"event": "contact_off_failure", "event_id": 1,
                        "all_events": ["contact_off_failure"], "terminal_reason": ""}],
            "success": False,
        }
        row = episode_occurrences(episode, "dev_fit")[0]
        self.assertFalse(row["terminated"])
        self.assertIn("failure_event", row["proposed_semantics"])

    def test_development_and_confirmation_families_are_disjoint(self) -> None:
        specs = family_specs(36, 840100)
        development = {item.root_family_id for item in specs[:24]}
        confirmation = {item.root_family_id for item in specs[24:]}
        self.assertFalse(development & confirmation)

    def test_segment_reward_conservation(self) -> None:
        phi = [0.0, 0.2, -0.1, 0.4]
        transitions = [phi[index + 1] - phi[index] for index in range(len(phi) - 1)]
        self.assertAlmostEqual(sum(transitions), phi[-1] - phi[0])


if __name__ == "__main__":
    unittest.main()
