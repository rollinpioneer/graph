"""Focused regression tests for deterministic U3 graph evaluation."""

from __future__ import annotations

import unittest

from upgrade_v2.u3_grounding.evaluate_graphs import (
    _graph_metrics,
    _predicate_fit,
    _reachable,
)


class GroundingMetricTests(unittest.TestCase):
    def test_data_only_predicate_fit_uses_raw_cluster_profile(self) -> None:
        graph = {
            "nodes": [
                {
                    "grounding": {"cluster_handle": "C1"},
                    "observable_predicates": ["contact_present", "object_moving"],
                }
            ]
        }
        segments = [{"split": "val", "evaluation_cluster_id": 1}]
        refs = {1: {"predicates": {"contact_present", "object_moving"}}}
        self.assertEqual(_predicate_fit(graph, segments, refs, "val"), 1.0)

    def test_reachability_requires_a_non_contradicted_path(self) -> None:
        base = {
            "start_cluster_handle": "C0",
            "success_cluster_handle": "C2",
            "nodes": [{"id": "C0"}, {"id": "C2"}],
        }
        no_path = {**base, "edges": []}
        self.assertFalse(_reachable(no_path))
        with_path = {
            **base,
            "edges": [
                {"src": "C0", "dst": "C2", "grounding": {"status": "observed"}}
            ],
        }
        self.assertTrue(_reachable(with_path))
        contradicted_path = {
            **base,
            "edges": [
                {
                    "src": "C0",
                    "dst": "C2",
                    "grounding": {"status": "contradicted"},
                }
            ],
        }
        self.assertFalse(_reachable(contradicted_path))

    def test_unresolved_and_contradicted_rates_are_status_based(self) -> None:
        graph = {
            "nodes": [],
            "edges": [
                {"id": "e0", "raw_pair": [1, 2], "status": "observed"},
                {"id": "e1", "raw_pair": [2, 3], "status": "unresolved"},
                {"id": "e2", "raw_pair": [3, 4], "status": "contradicted"},
            ],
        }
        metrics = _graph_metrics(
            graph,
            segments=[],
            transitions=[],
            refs={},
            split="val",
            bootstrap=10,
            seed=1,
        )
        self.assertAlmostEqual(metrics["unresolved_edge_rate"], 1 / 3, places=7)
        self.assertAlmostEqual(metrics["contradicted_edge_rate"], 1 / 3, places=7)
        self.assertAlmostEqual(metrics["unknown_honesty"], 1 / 3, places=7)

    def test_recovery_recall_uses_contact_off_failure_event(self) -> None:
        graph = {
            "nodes": [],
            "edges": [
                {
                    "id": "e0",
                    "raw_pair": [1, 2],
                    "hypothesized_type": "recovery",
                    "status": "observed",
                }
            ],
        }
        transitions = [
            {
                "split": "val",
                "root_family_id": "f0",
                "from_cluster_id": 1,
                "to_cluster_id": 2,
                "events_before": {"contact_off_failure"},
                "events_after": set(),
            }
        ]
        metrics = _graph_metrics(
            graph,
            segments=[],
            transitions=transitions,
            refs={},
            split="val",
            bootstrap=10,
            seed=1,
        )
        self.assertEqual(metrics["recovery_edge_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
