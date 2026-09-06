from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from .evaluator_v2 import evaluate_node_role, matching_edges
from .guard_dsl import GuardError, evaluate_guard, validate_guard
from .confirm import build_lock, evaluate as evaluate_confirmation
from .io import write_json, write_jsonl


class GuardAndEvaluatorTest(unittest.TestCase):
    def test_guard_equality(self):
        guard = {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}
        self.assertTrue(evaluate_guard(guard, {"terminal_failure_event": True}))
        self.assertFalse(evaluate_guard(guard, {"terminal_failure_event": False}))

    def test_guard_any_of(self):
        guard = {"any_of": [{"field": {"name": "horizon", "comparison": "==", "value": True}}, {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}]}
        self.assertTrue(evaluate_guard(guard, {"horizon": True, "terminal_failure_event": False}))

    def test_guard_rejects_eval(self):
        with self.assertRaises(GuardError):
            validate_guard({"field": {"name": "__import__", "comparison": "==", "value": True}})

    def test_guard_rejects_future(self):
        with self.assertRaises(GuardError):
            validate_guard({"field": {"name": "future_success", "comparison": "==", "value": True}})

    def test_horizon_is_censored(self):
        node = {"id": "C04", "role": "failure_terminal", "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}
        row = {"terminal_status": "censored_unknown", "evaluator_semantics": [], "observable_context": {"horizon": True}}
        result = evaluate_node_role(node, row)
        self.assertEqual(result["occurrence_terminal_status"], "censored_unknown")

    def test_nonterminal_does_not_activate_role(self):
        node = {"id": "C10", "role": "failure_terminal", "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}
        row = {"terminal_status": "nonterminal", "evaluator_semantics": [], "observable_context": {"terminal_failure_event": False}}
        result = evaluate_node_role(node, row)
        self.assertEqual(result["occurrence_terminal_status"], "nonterminal")

    def test_multiple_typed_edges(self):
        graph = {"edges": [{"id": "E1", "raw_pair": [4, 4], "semantic_type": "dwell"}, {"id": "E2", "raw_pair": [4, 4], "semantic_type": "terminal_failure", "guard": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}]}
        row = {"src_cluster_id": 4, "dst_cluster_id": 4, "terminal_status": "failure_terminal", "observable_context": {}}
        self.assertEqual({x["edge_id"] for x in matching_edges(graph, row)}, {"E1", "E2"})

    def test_guarded_edge_not_active(self):
        graph = {"edges": [{"id": "E", "raw_pair": [4, 4], "semantic_type": "terminal_failure", "guard": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}]}
        row = {"src_cluster_id": 4, "dst_cluster_id": 4, "terminal_status": "nonterminal", "observable_context": {}}
        self.assertEqual(matching_edges(graph, row), [])

    def test_guard_missing_field_is_false(self):
        guard = {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}
        self.assertFalse(evaluate_guard(guard, {}))

    def test_no_proposed_semantics_gold(self):
        from .evaluator_v2 import evaluate_occurrence
        graph = {"nodes": [], "edges": []}
        row = {"src_cluster_id": 1, "dst_cluster_id": 2, "terminal_status": "nonterminal", "evaluator_semantics": ["dwell"], "proposed_semantics": ["terminal_failure"], "observable_context": {}}
        result = evaluate_occurrence(graph, row)
        self.assertIsNone(result["proposed_semantics"])
        self.assertEqual(result["evaluator_semantics"], ["dwell"])

    def test_confirmation_gate_blocks_missing_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evaluate_confirmation([], root / "missing.jsonl", root / "metrics.csv", root / "paired.csv", root / "family.csv")
            self.assertEqual(result["status"], "BLOCKED")

    def test_confirmation_gate_blocks_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.json"; selection = root / "selection.json"; protocol = root / "protocol.json"; family = root / "family.json"
            write_json(graph, {"graph_id": "G", "nodes": [], "edges": []})
            write_json(selection, {"selected_graph": "G"}); write_json(protocol, {"version": "test"}); write_json(family, {"families": 1})
            lock = root / "lock.json"; build_lock(lock, [graph], selection, protocol, family)
            write_json(family, {"families": 2})
            result = evaluate_confirmation([graph], root / "missing.jsonl", root / "metrics.csv", root / "paired.csv", root / "family.csv", lock, family)
            self.assertEqual(result["status"], "BLOCKED")

    def test_confirmation_gate_passes_with_locked_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph = root / "graph.json"; selection = root / "selection.json"; protocol = root / "protocol.json"; family = root / "family.json"
            write_json(graph, {"graph_id": "G", "nodes": [], "edges": []})
            write_json(selection, {"selected_graph": "G"}); write_json(protocol, {"version": "test"}); write_json(family, {"families": 1})
            lock = root / "lock.json"; build_lock(lock, [graph], selection, protocol, family)
            occurrences = root / "occurrences.jsonl"
            write_jsonl(occurrences, [{"root_family_id": "f", "src_cluster_id": 1, "dst_cluster_id": 2, "terminal_status": "nonterminal", "evaluator_semantics": [], "observable_context": {}}])
            result = evaluate_confirmation([graph], occurrences, root / "metrics.csv", root / "paired.csv", root / "family.csv", lock, family)
            self.assertEqual(result["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
