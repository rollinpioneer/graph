"""Regression checks for the post-delivery U0/U1 audit fixes."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from upgrade_v2.cli import (cmd_build_u2_stochastic_boundary_data,
                            cmd_make_outcome_time_targets,
                            cmd_normalize_continuation_records)
from upgrade_v2.rewards.graph_rules import legal_success_chains, oracle_topology_cost
from upgrade_v2.adapters.stochastic_d1 import StochasticGoalSimulator, StochasticPushSimulator
from upgrade_v2.adapters.stochastic_u2 import U2BoundarySimulator


class GraphRuleTests(unittest.TestCase):
    def test_weighted_oracle_uses_shortest_path_not_first_terminal_edge(self) -> None:
        graph = {
            "terminal_nodes": ["goal"],
            "edges": [
                {"source": "start", "target": "goal", "base_step_cost": 10},
                {"source": "start", "target": "middle", "base_step_cost": 1},
                {"source": "middle", "target": "goal", "base_step_cost": 1},
            ],
        }
        self.assertEqual(oracle_topology_cost("start", graph), 2.0)

    def test_legal_chains_come_from_graph_edges(self) -> None:
        graph = {
            "start_node": "start",
            "success_nodes": ["success"],
            "edges": [
                {"src": "start", "dst": "A_done"},
                {"src": "A_done", "dst": "B_done"},
                {"src": "B_done", "dst": "success"},
                {"src": "start", "dst": "B_done"},
                {"src": "B_done", "dst": "A_done"},
                {"src": "A_done", "dst": "success"},
            ],
        }
        chains = legal_success_chains(graph)
        self.assertIn(["start", "A_done", "B_done", "success"], chains)
        self.assertIn(["start", "B_done", "A_done", "success"], chains)
        self.assertNotEqual(chains, [["A_done", "B_done", "start", "success"]])


class OutcomeEvidenceTests(unittest.TestCase):
    def test_unknown_short_negative_rollout_is_right_censored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            observables = root / "observables"; observables.mkdir()
            observable = {
                "episode_uid": "short_negative", "task_id": "transport_recovery", "root_family_id": "family",
                "provenance": "test", "split": "train", "n_steps": 4, "terminal_reward": -0.01,
                "features": [[0.0] * 11 for _ in range(4)],
            }
            (observables / "observable_records.jsonl").write_text(json.dumps(observable) + "\n", encoding="utf-8")
            evidence = root / "episode_evidence.jsonl"
            evidence.write_text(json.dumps({"episode_id": "short_negative", "task_id": "transport_recovery"}) + "\n", encoding="utf-8")
            continuations = root / "continuations.jsonl"
            cmd_normalize_continuation_records(argparse.Namespace(observables=observables, continuations=evidence, output=continuations, protocol=root / "protocol"))
            targets = root / "targets"
            cmd_make_outcome_time_targets(argparse.Namespace(records=continuations, protocol=root / "protocol", splits=["train"], output_root=targets, summary=root / "summary.csv", examples=root / "examples.csv"))
            first = json.loads((targets / "targets.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(first["terminal_reason"], "unknown")
            self.assertEqual(first["label_reason"], "right_censored")
            self.assertEqual((first["q_mask"], first["d_mask"]), (0, 0))


class D1SnapshotTests(unittest.TestCase):
    def test_snapshot_restores_rng_and_continuation_exactly(self) -> None:
        prefix = [[0.95, 0.0]] * 4
        suffix = [[0.80, 0.30]] * 6
        original = StochasticPushSimulator(seed=20260905)
        for action in prefix:
            original.step(action)
        snapshot = original.snapshot()
        expected = []
        for action in suffix:
            original.step(action)
            expected.append(original.state_vector())
        restored = StochasticPushSimulator(seed=1)
        restored.restore(snapshot)
        actual = []
        for action in suffix:
            restored.step(action)
            actual.append(restored.state_vector())
        for left, right in zip(expected, actual):
            self.assertEqual(float(abs(left - right).max()), 0.0)

    def test_distance_matched_free_and_collision_routes_have_distinct_outcomes(self) -> None:
        free_position = np.asarray([0.25, 0.80])
        goal_distance = np.linalg.norm(StochasticGoalSimulator.goal - free_position)
        collision_position = np.asarray([StochasticGoalSimulator.goal[0] - goal_distance, 0.50])
        self.assertEqual(float(np.linalg.norm(StochasticGoalSimulator.goal - collision_position)), float(goal_distance))
        free_result = StochasticGoalSimulator(free_position, seed=20260905).run_goal_controller()
        collision_result = StochasticGoalSimulator(collision_position, seed=20260905).run_goal_controller()
        self.assertTrue(free_result["success"])
        self.assertTrue(collision_result["failed"])


class U2DataContractTests(unittest.TestCase):
    def test_u2_snapshot_restores_stochastic_transition_and_features_are_numeric(self) -> None:
        original = U2BoundarySimulator(np.asarray([0.20, 0.50]), seed=20260905)
        original.step(np.asarray([1.0, 0.0, 1.0]))
        snapshot = original.snapshot()
        expected_features, expected_info = original.step(np.asarray([0.0, 1.0, 1.0]))
        restored = U2BoundarySimulator(np.asarray([0.20, 0.50]), seed=1)
        restored.restore(snapshot)
        actual_features, actual_info = restored.step(np.asarray([0.0, 1.0, 1.0]))
        self.assertEqual(expected_features, actual_features)
        self.assertEqual(expected_info, actual_info)
        self.assertEqual(len(actual_features), 11)
        self.assertTrue(all(isinstance(value, float) for value in actual_features))

    def test_u2_transition_records_exclude_generation_strata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical = root / "u2_handoff.json"
            historical.write_text('{"u2_eligible": false}\n', encoding="utf-8")
            out = root / "u2"
            handoff = root / "u2_handoff_v2.json"
            cmd_build_u2_stochastic_boundary_data(argparse.Namespace(
                historical_handoff=historical, handoff=handoff, output_dir=out,
                seed=20260905, per_stratum=2,
            ))
            first = json.loads((out / "u2_transition_records.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("generation_stratum", first)
            self.assertNotIn("intervention_schedule", first)
            self.assertTrue(all(isinstance(value, float) for value in first["features_before"]))
            authority = json.loads(handoff.read_text(encoding="utf-8"))
            self.assertTrue(authority["historical_file_preserved"])


if __name__ == "__main__":
    unittest.main()
