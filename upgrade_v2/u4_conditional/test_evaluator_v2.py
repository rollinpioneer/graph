from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from .evaluator_v2 import evaluate_node_role, matching_edges, occurrence_context
from .occurrence_table import observable_context
from .guard_dsl import GuardError, evaluate_guard, guard_status, validate_guard
from .confirm import build_lock, evaluate as evaluate_confirmation
from .io import write_json, write_jsonl


class GuardAndEvaluatorTest(unittest.TestCase):
    def test_guard_equality(self):
        guard = {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}
        self.assertTrue(evaluate_guard(guard, {"terminal_failure_event": True}))
        self.assertFalse(evaluate_guard(guard, {"terminal_failure_event": False}))

    def test_mixed_terminal_failure_event_true_uses_explicit_runtime_observable(self):
        node = {"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}], "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}
        row = {"terminal_status": "nonterminal", "observable_context": {"terminal_failure_event": True}, "runtime_observable_fields": ["terminal_failure_event"]}
        self.assertEqual(evaluate_node_role(node, row)["occurrence_terminal_status"], "failure_terminal")

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

    def test_terminal_status_does_not_activate_role(self):
        node = {"id": "C10", "role": "failure_terminal", "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}
        row = {"terminal_status": "failure_terminal", "evaluator_semantics": [], "observable_context": {}}
        result = evaluate_node_role(node, row)
        self.assertEqual(result["occurrence_terminal_status"], "guard_ambiguous")

    def test_success_terminal_uses_explicit_stable_success_observable(self):
        node = {"id": "C02", "role": "success_terminal"}
        success = {"terminal_status": "nonterminal", "observable_context": {"stable_success_event": True}, "runtime_observable_fields": ["stable_success_event"]}
        not_yet = {"terminal_status": "nonterminal", "observable_context": {"stable_success_event": False}, "runtime_observable_fields": ["stable_success_event"]}
        self.assertEqual(evaluate_node_role(node, success)["occurrence_terminal_status"], "success_terminal")
        self.assertEqual(evaluate_node_role(node, not_yet)["occurrence_terminal_status"], "nonterminal")

    def test_diagnostic_event_log_is_not_online_context(self):
        row = {
            "terminal_status": "failure_terminal",
            "evaluator_event_set": ["terminal_failure"],
            "observable_context": {"all_events": ["terminal_failure"], "event": "terminal_failure"},
        }
        context = occurrence_context(row)
        self.assertNotIn("terminal_failure_event", context)
        self.assertNotIn("all_events", context)
        self.assertNotIn("event", context)

    def test_legacy_label_aliases_are_not_online_context(self):
        row = {"observable_context": {"horizon": True, "nonterminal": False, "terminal_failure_event": True, "terminal_success_event": True}}
        context = occurrence_context(row)
        self.assertNotIn("horizon", context)
        self.assertNotIn("nonterminal", context)
        self.assertNotIn("terminal_failure_event", context)
        self.assertNotIn("terminal_success_event", context)

    def test_guarded_edge_does_not_use_terminal_label(self):
        graph = {"edges": [{"id": "E1", "raw_pair": [4, 4], "semantic_type": "dwell"}, {"id": "E2", "raw_pair": [4, 4], "semantic_type": "terminal_failure", "guard": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}]}
        row = {"src_cluster_id": 4, "dst_cluster_id": 4, "terminal_status": "failure_terminal", "observable_context": {}}
        self.assertEqual({x["edge_id"] for x in matching_edges(graph, row)}, {"E1"})

    def test_guarded_edge_not_active(self):
        graph = {"edges": [{"id": "E", "raw_pair": [4, 4], "semantic_type": "terminal_failure", "guard": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}]}
        row = {"src_cluster_id": 4, "dst_cluster_id": 4, "terminal_status": "nonterminal", "observable_context": {}}
        self.assertEqual(matching_edges(graph, row), [])

    def test_guard_missing_field_is_false(self):
        guard = {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}
        self.assertFalse(evaluate_guard(guard, {}))
        self.assertEqual(guard_status(guard, {}), "ambiguous")

    def test_mixed_terminal_failure_event_requires_observable_field(self):
        result = evaluate_node_role({"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}], "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}, {"terminal_status": "failure_terminal", "observable_context": {}})
        self.assertEqual(result["occurrence_terminal_status"], "guard_ambiguous")

    def test_mixed_false_is_nonterminal(self):
        result = evaluate_node_role({"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}], "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}, {"terminal_status": "nonterminal", "observable_context": {}})
        self.assertEqual(result["occurrence_terminal_status"], "guard_ambiguous")

    def test_missing_role_field_is_ambiguous(self):
        result = evaluate_node_role({"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}], "role_condition": {"field": {"name": "contact_after", "comparison": "==", "value": True}}}, {"terminal_status": "nonterminal", "observable_context": {}})
        self.assertEqual(result["occurrence_terminal_status"], "guard_ambiguous")

    def test_removing_role_condition_changes_evaluation(self):
        condition = {"field": {"name": "contact_after", "comparison": "==", "value": True}}
        row = {"terminal_status": "nonterminal", "observable_context": {"contact_after": True}}
        with_condition = evaluate_node_role({"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}], "role_condition": condition}, row)
        without_condition = evaluate_node_role({"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}]}, row)
        self.assertEqual(with_condition["occurrence_terminal_status"], "failure_terminal")
        self.assertEqual(without_condition["occurrence_terminal_status"], "guard_ambiguous")

    def test_terminal_metrics_report_coverage_and_precision(self):
        from .evaluator_v2 import graph_metrics
        graph = {"graph_id": "G", "nodes": [{"id": "C1", "role": "failure_terminal"}], "edges": []}
        result = graph_metrics(graph, [{"root_family_id": "f", "dst_cluster_id": 1, "terminal_status": "failure_terminal", "evaluator_semantics": [], "observable_context": {}}])
        aggregate = result["aggregate"]
        self.assertIn("failure_terminal_precision", aggregate)
        self.assertIn("terminal_claim_coverage", aggregate)

    def test_same_node_can_be_terminal_or_nonterminal(self):
        node = {"id": "C04", "role": "mixed", "conditional_roles": [{"role": "failure_terminal"}], "role_condition": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}}}
        self.assertEqual(evaluate_node_role(node, {"terminal_status": "failure_terminal", "observable_context": {}})["occurrence_terminal_status"], "guard_ambiguous")
        self.assertEqual(evaluate_node_role(node, {"terminal_status": "nonterminal", "observable_context": {}})["occurrence_terminal_status"], "guard_ambiguous")

    def test_observable_context_ignores_event_history(self):
        before = [0.0] * 17; after = [0.0] * 17
        before[12] = 1.0; after[12] = 1.0
        event_history = [{"all_events": ["contact_off_failure", "recovery_start"]}]
        context = observable_context(before, after, [0.0, 0.0], event_history)
        self.assertFalse(context["contact_recently_lost"])
        self.assertFalse(context["recent_recovery_attempt"])

    def test_guard_rejects_scenario(self):
        with self.assertRaises(GuardError):
            validate_guard({"field": {"name": "scenario", "comparison": "==", "value": "nominal_success"}})

    def test_compact_guard_schema(self):
        guard = {"==": {"field": "contact_after", "constant": True}}
        self.assertTrue(evaluate_guard(guard, {"contact_after": True}))

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
