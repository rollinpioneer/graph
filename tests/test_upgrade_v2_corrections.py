"""Regression checks for the post-delivery U0/U1 audit fixes."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from upgrade_v2.cli import cmd_make_outcome_time_targets, cmd_normalize_continuation_records
from upgrade_v2.rewards.graph_rules import legal_success_chains, oracle_topology_cost


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


if __name__ == "__main__":
    unittest.main()
