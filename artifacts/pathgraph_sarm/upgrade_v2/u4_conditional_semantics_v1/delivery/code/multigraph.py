"""Build guarded typed edges while preserving the raw topology."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

from .guard_dsl import validate_guard
from .io import read_csv, read_json, read_jsonl, write_csv, write_json


def _pair(edge: dict[str, Any]) -> tuple[int, int]:
    raw = edge.get("raw_pair") or [int(str(edge.get("src", "C0"))[1:]), int(str(edge.get("dst", "C0"))[1:])]
    return int(raw[0]), int(raw[1])


def _base_graph(path: Path) -> dict[str, Any]:
    graph = deepcopy(read_json(path))
    graph["graph_id"] = "G0_raw_topology"
    return graph


def _semantic_counts(rows: list[dict[str, Any]]) -> dict[tuple[int, int], Counter[str]]:
    result: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for row in rows:
        pair = (row.get("src_cluster_id"), row.get("dst_cluster_id"))
        if None in pair:
            continue
        labels = row.get("evaluator_semantics", [])
        for label in labels:
            if label not in {"censored_unknown", "terminal_success"}:
                result[pair][label] += 1
    return result


def build_graphs(raw_graph: Path, occurrences: Path, output_dir: Path, edit_log: Path | None = None) -> dict[str, Any]:
    base = _base_graph(raw_graph)
    rows = read_jsonl(occurrences)
    counts = _semantic_counts(rows)
    g0 = deepcopy(base)
    g1 = deepcopy(base)
    g1["graph_id"] = "G1_semantic_only"
    for edge in g1.get("edges", []):
        counter = counts.get(_pair(edge), Counter())
        edge["evaluator_semantics"] = counter.most_common(1)[0][0] if counter else "unknown"
        edge["semantic_distribution"] = dict(counter)
        edge["semantic_label_source"] = "evaluator_semantics"
        edge.pop("proposed_semantics", None)
    g2 = deepcopy(g1)
    g2["graph_id"] = "G2_conditional_multigraph"
    guarded = []
    # Conditional role edits are carried forward from U4 B+ evidence, but are
    # represented as typed guarded edges, not as unconditional node labels.
    terminal_guard = {"any_of": [
        {"field": {"name": "terminal_failure_event", "comparison": "==", "value": True}},
        {"field": {"name": "horizon", "comparison": "==", "value": True}},
    ]}
    validate_guard(terminal_guard)
    for node in g2.get("nodes", []):
        if node.get("id") in {"C04", "C10"} and node.get("role_condition"):
            node["conditional_role_guard"] = terminal_guard
            node["static_role"] = node.get("role")
            node["role"] = "conditional_terminal"
    for edge in list(g2.get("edges", [])):
        pair = _pair(edge)
        counter = counts.get(pair, Counter())
        if not counter:
            continue
        dominant = counter.most_common(1)[0][0]
        edge["evaluator_semantics"] = dominant
        edge["guard"] = None
        if pair[1] in {4, 10} and any(label in counter for label in {"terminal_failure", "censored_unknown"}):
            conditional = deepcopy(edge)
            conditional["id"] = f"{edge.get('id')}_guarded_terminal"
            conditional["evaluator_semantics"] = "terminal_failure"
            conditional["guard"] = terminal_guard
            conditional["semantic_label_source"] = "evaluator_semantics"
            g2["edges"].append(conditional)
            guarded.append(conditional["id"])
    g2["multigraph"] = True
    g2["semantic_label_source"] = "evaluator_semantics"
    g2["proposed_semantics_used_as_gold"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "G0_raw_topology.json", g0)
    write_json(output_dir / "G1_semantic_only.json", g1)
    write_json(output_dir / "G2_conditional_multigraph.json", g2)
    edits = [{"proposal_id": "P_CONDITIONAL_ROLE_C04_C10", "operation": "guarded_typed_edges", "guard": terminal_guard, "target_nodes": ["C04", "C10"], "accepted": bool(guarded), "accepted_edge_ids": guarded, "support": "U4 B+ concrete terminal-role counterexamples", "counterexamples": "horizon is censored_unknown; nonterminal role occurrences retained"}]
    write_json(edit_log or output_dir / "edit_log.json", {"schema": "u4r1_edit_log_v1", "max_edits": 6, "accepted_count": len([x for x in edits if x["accepted"]]), "edits": edits})
    return {"status": "PASS", "graphs": [str(output_dir / name) for name in ("G0_raw_topology.json", "G1_semantic_only.json", "G2_conditional_multigraph.json")], "guarded_edge_count": len(guarded)}


def graph_from_path(path: Path) -> dict[str, Any]:
    return read_json(path)
