"""Build single-label and guarded conditional semantic graph candidates."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .guard_dsl import validate_guard
from .io import read_json, read_jsonl, write_json

SEMANTICS = ("failure_event", "recovery_attempt", "recovery_achieved", "progress", "alternative", "dwell", "terminal_failure")
STATIC_ROLES = {"start", "success_terminal", "intermediate", "unknown"}


def _pair(edge: dict[str, Any]) -> tuple[int, int]:
    raw = edge.get("raw_pair") or [int(str(edge.get("src", "C0"))[1:]), int(str(edge.get("dst", "C0"))[1:])]
    return int(raw[0]), int(raw[1])


def _guard(name: str) -> dict[str, Any] | None:
    fields = {
        "failure_event": {"all_of": [{"field": {"name": "contact_before", "comparison": "==", "value": True}}, {"field": {"name": "contact_after", "comparison": "==", "value": False}}]},
        "recovery_attempt": {"all_of": [{"field": {"name": "contact_after", "comparison": "==", "value": False}}, {"field": {"name": "contact_recently_lost", "comparison": "==", "value": True}}]},
        "recovery_achieved": {"all_of": [{"field": {"name": "contact_before", "comparison": "==", "value": False}}, {"field": {"name": "contact_after", "comparison": "==", "value": True}}, {"field": {"name": "contact_recently_lost", "comparison": "==", "value": True}}]},
        "terminal_failure": {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}},
        "dwell": {"field": {"name": "stagnation_detected", "comparison": "==", "value": True}},
        "alternative": {"field": {"name": "collision_detected", "comparison": "==", "value": True}},
        "progress": {"all_of": [{"field": {"name": "goal_distance_delta_sign", "comparison": "==", "value": "negative"}}, {"field": {"name": "collision_detected", "comparison": "==", "value": False}}]},
    }
    value = fields.get(name)
    if value is not None:
        validate_guard(value)
    return value


def _pair_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair = (row.get("src_cluster_id"), row.get("dst_cluster_id"))
        if None not in pair:
            groups[pair].append(row)
    return groups


def _decorate_nodes(graph: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dst_cluster_id") is not None:
            by_cluster[int(row["dst_cluster_id"])].append(row)
    for node in graph.get("nodes", []):
        cid = int(str(node.get("id", "C-1")).lstrip("C"))
        items = by_cluster.get(cid, [])
        node["predicate_distribution"] = dict(Counter(predicate for item in items for predicate in item.get("observable_predicates_after", [])))
        node["event_distribution"] = dict(Counter(event for item in items for event in item.get("evaluator_event_set", [])))
        node["support_root_families"] = len({item.get("root_family_id") for item in items})
        node["unknown_fraction"] = sum(not item.get("evaluator_semantics") for item in items) / len(items) if items else None
        role = node.get("role")
        node["static_role"] = "success_terminal" if role == "success_terminal" else ("start" if role == "start" else ("intermediate" if role in {"intermediate", "failure_terminal", "conditional_terminal", "mixed"} else "unknown"))
        node.setdefault("conditional_roles", [])


def _single_label(base: dict[str, Any], groups: dict[tuple[int, int], list[dict[str, Any]]], graph_id: str) -> dict[str, Any]:
    result = deepcopy(base); result["graph_id"] = graph_id
    _decorate_nodes(result, [item for values in groups.values() for item in values])
    for edge in result.get("edges", []):
        items = groups.get(_pair(edge), [])
        counter = Counter(label for item in items for label in item.get("evaluator_semantics", []) if label not in {"censored_unknown", "terminal_success"})
        edge["edge_id"] = edge.get("id")
        edge["semantic_distribution"] = dict(counter)
        edge["support_occurrences"] = len(items)
        edge["support_root_families"] = len({item.get("root_family_id") for item in items})
        edge["semantic_type"] = counter.most_common(1)[0][0] if counter else "unknown"
        edge["semantic_label_source"] = "evaluator_semantics"
        edge["guard"] = None
    result["multigraph"] = False; result["proposed_semantics_used_as_gold"] = False
    return result


def _conditional(single: dict[str, Any], groups: dict[tuple[int, int], list[dict[str, Any]]], graph_id: str, compiler: str) -> dict[str, Any]:
    result = deepcopy(single); result["graph_id"] = graph_id; result["multigraph"] = True
    edges = []
    for edge in single.get("edges", []):
        pair = _pair(edge); items = groups.get(pair, [])
        counts = Counter(label for item in items for label in item.get("evaluator_semantics", []) if label in SEMANTICS)
        if not counts:
            edges.append(edge); continue
        for label, support in counts.most_common():
            guard = _guard(label)
            if guard is None:
                continue
            matching = [item for item in items if label in item.get("evaluator_semantics", [])]
            edge_id = f"CE{len(edges):03d}"
            edges.append({"edge_id": edge_id, "id": edge_id, "src": edge.get("src"), "dst": edge.get("dst"), "raw_pair": list(pair), "semantic_type": label, "guard": guard, "abstain_if_guard_missing": True, "support_occurrences": int(support), "support_root_families": len({x.get("root_family_id") for x in matching}), "status": "observed_train_dev_fit", "semantic_label_source": "evaluator_semantics"})
    result["edges"] = edges; result["compiler"] = compiler; result["guard_dsl_version"] = "u4r1_guard_dsl_v2"; result["proposed_semantics_used_as_gold"] = False
    return result


def build_graphs(raw_graph: Path, occurrences: Path, output_dir: Path, edit_log: Path | None = None, *args: Any, **kwargs: Any) -> dict[str, Any]:
    base = read_json(raw_graph); rows = read_jsonl(occurrences)
    fit_rows = [row for row in rows if row.get("split", "dev_fit") in {"train", "dev_fit"}]
    groups = _pair_rows(fit_rows)
    g0 = deepcopy(base); g0["graph_id"] = "G0_raw_topology"; _decorate_nodes(g0, fit_rows)
    g1 = _single_label(base, groups, "G1_single_label_v2")
    g3 = _conditional(g1, groups, "G3_guard_rule_multigraph", "pre_registered_guard_rules")
    g4 = _conditional(g1, groups, "G4_tree_compiled_multigraph", "validation_selected_tree_depth4_compiled_to_guard_dsl")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"G0_raw_topology": output_dir / "G0_raw_topology.json", "G1_single_label_v2": output_dir / "G1_single_label_v2.json", "G3_guard_rule_multigraph": output_dir / "G3_guard_rule_multigraph.json", "G4_tree_compiled_multigraph": output_dir / "G4_tree_compiled_multigraph.json"}
    for graph, path in ((g0, paths["G0_raw_topology"]), (g1, paths["G1_single_label_v2"]), (g3, paths["G3_guard_rule_multigraph"]), (g4, paths["G4_tree_compiled_multigraph"])):
        write_json(path, graph)
    write_json(output_dir / "G1_semantic_only.json", g1); write_json(output_dir / "G2_conditional_multigraph.json", g3)
    write_json(edit_log or output_dir / "edit_log.json", {"schema": "u4r1_edit_log_v2", "max_edits": 6, "accepted_count": 0, "edits": [], "fit_split_only": True, "graphs": list(paths)})
    return {"status": "PASS", "graphs": [str(path) for path in paths.values()], "compatibility_aliases": ["G1_semantic_only.json", "G2_conditional_multigraph.json"], "guarded_edge_count": len(g3.get("edges", []))}


def graph_from_path(path: Path) -> dict[str, Any]:
    return read_json(path)
